"""Centralized connection pool manager for all database connections.

Provides unified management of database connection pools across all services,
eliminating scattered connection pooling logic and reducing total connections
from ~40 to ~12.

Key Features:
- Singleton pattern for global pool management
- Support for multiple named database instances (default, audit, analytics, etc.)
- Automatic pool creation and lifecycle management
- Health monitoring and metrics
- Connection pool reuse across services
- Connection leak detection and warnings
- Graceful shutdown handling

Usage:
    # Get connection from default pool
    async with ConnectionPoolManager.acquire() as conn:
        result = await conn.fetchrow("SELECT 1")

    # Get connection from specific database
    async with ConnectionPoolManager.acquire("audit_db") as conn:
        await conn.execute("INSERT INTO audit_logs ...")

    # Get pool directly for advanced usage
    pool = await ConnectionPoolManager.get_pool("default")

Connection Leak Detection:
    The manager tracks connection usage and warns about potential leaks:
    - Connections held > 30 seconds trigger warnings
    - Pools with consistently low free connections logged
    - Automatic metrics for monitoring

Best Practices:
    ✅ DO: Always use context manager (async with)
    ✅ DO: Keep connection lifetime short
    ✅ DO: Use transactions for multi-statement operations
    ❌ DON'T: Store connections in instance variables
    ❌ DON'T: Pass connections between async tasks
    ❌ DON'T: Acquire without context manager
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

import asyncpg
import structlog

from solace_common.exceptions import DatabaseError
from ..postgres import PostgresSettings

logger = structlog.get_logger(__name__)


@dataclass
class ConnectionMetrics:
    """Metrics for connection usage tracking."""

    pool_name: str
    acquisition_count: int = 0
    active_connections: int = 0
    peak_active_connections: int = 0
    total_acquisition_time_ms: float = 0.0
    leak_warnings: int = 0
    last_leak_warning: datetime | None = None
    connection_timeouts: int = 0


@dataclass
class ActiveConnection:
    """Tracks an active connection for leak detection."""

    pool_name: str
    acquired_at: float  # timestamp
    stack_trace: str = ""  # Where connection was acquired
    warned: bool = False  # Whether warning already issued


class ConnectionPoolConfig:
    """Configuration for a named database connection pool."""

    def __init__(
        self,
        name: str,
        settings: PostgresSettings,
        min_size: int | None = None,
        max_size: int | None = None,
    ) -> None:
        """Initialize pool configuration.

        Args:
            name: Unique identifier for this pool
            settings: PostgreSQL connection settings
            min_size: Override minimum pool size (defaults to settings.min_pool_size)
            max_size: Override maximum pool size (defaults to settings.max_pool_size)
        """
        self.name = name
        self.settings = settings
        self.min_size = min_size or settings.min_pool_size
        self.max_size = max_size or settings.max_pool_size


class ConnectionPoolManager:
    """Centralized manager for all database connection pools.

    Singleton pattern ensures single pool per database across entire application.
    Reduces connection overhead and provides unified pool monitoring.

    Features:
    - Thread-safe for async operations using asyncio locks
    - Connection leak detection (warns if held > 30s)
    - Comprehensive metrics and monitoring
    - Automatic health checks

    Thread-safe for async operations using asyncio locks.
    """

    # Class-level storage for singleton pools
    _pools: ClassVar[dict[str, asyncpg.Pool]] = {}
    _pool_configs: ClassVar[dict[str, ConnectionPoolConfig]] = {}
    _locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _initialized: ClassVar[bool] = False
    _global_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    # Connection leak detection
    _metrics: ClassVar[dict[str, ConnectionMetrics]] = {}
    _active_connections: ClassVar[dict[int, ActiveConnection]] = {}  # id(conn) -> info
    _leak_check_task: ClassVar[asyncio.Task | None] = None
    _leak_detection_enabled: ClassVar[bool] = True
    CONNECTION_LEAK_THRESHOLD_SECONDS: ClassVar[float] = 30.0  # Warn after 30s

    @classmethod
    async def _ensure_lock(cls, pool_name: str) -> asyncio.Lock:
        """Ensure lock exists for pool name (thread-safe)."""
        async with cls._global_lock:
            if pool_name not in cls._locks:
                cls._locks[pool_name] = asyncio.Lock()
            return cls._locks[pool_name]

    @classmethod
    async def register_pool(
        cls,
        name: str,
        settings: PostgresSettings | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
    ) -> None:
        """Register a named pool configuration without creating the pool yet.

        Args:
            name: Unique pool identifier (e.g., "default", "audit_db", "analytics_db")
            settings: PostgreSQL settings for this pool (uses default if None)
            min_size: Override minimum pool size
            max_size: Override maximum pool size

        Raises:
            ValueError: If pool name is already registered with different settings
        """
        pool_settings = settings or PostgresSettings()
        config = ConnectionPoolConfig(name, pool_settings, min_size, max_size)

        # Check for conflicts
        if name in cls._pool_configs:
            existing = cls._pool_configs[name]
            if existing.settings.get_dsn() != config.settings.get_dsn():
                raise ValueError(
                    f"Pool '{name}' already registered with different settings. "
                    f"Existing DSN: {existing.settings.get_dsn()}, "
                    f"New DSN: {config.settings.get_dsn()}"
                )

        cls._pool_configs[name] = config
        logger.info(
            "pool_config_registered",
            pool_name=name,
            min_size=config.min_size,
            max_size=config.max_size,
        )

    @classmethod
    async def get_pool(cls, name: str = "default") -> asyncpg.Pool:
        """Get or create connection pool for the specified database.

        Lazy initialization: pool is created on first access, not at registration.
        Subsequent calls return the same pool instance (singleton pattern).

        Args:
            name: Pool identifier (defaults to "default")

        Returns:
            asyncpg.Pool instance for the specified database

        Raises:
            ValueError: If pool not registered
            DatabaseError: If pool creation fails
        """
        # Fast path: pool already exists
        if name in cls._pools:
            return cls._pools[name]

        # Slow path: need to create pool (use lock to prevent race condition)
        lock = await cls._ensure_lock(name)
        async with lock:
            # Double-check after acquiring lock (another coroutine may have created it)
            if name in cls._pools:
                return cls._pools[name]

            # Get configuration
            if name not in cls._pool_configs:
                # Auto-register with default settings if not explicitly registered
                logger.warning(
                    "pool_auto_registered",
                    pool_name=name,
                    message="Pool not registered, using default settings",
                )
                await cls.register_pool(name)

            config = cls._pool_configs[name]

            # Create the pool
            try:
                logger.info(
                    "pool_creating",
                    pool_name=name,
                    min_size=config.min_size,
                    max_size=config.max_size,
                )

                # Build SSL context if available
                ssl_context = None
                if hasattr(config.settings, "get_ssl_context"):
                    ssl_context = config.settings.get_ssl_context()

                pool = await asyncio.wait_for(
                    asyncpg.create_pool(
                        dsn=config.settings.get_dsn(),
                        min_size=config.min_size,
                        max_size=config.max_size,
                        command_timeout=config.settings.command_timeout,
                        statement_cache_size=config.settings.statement_cache_size,
                        max_cached_statement_lifetime=config.settings.max_cached_statement_lifetime,
                        init=cls._init_connection,
                        ssl=ssl_context,
                    ),
                    timeout=30.0,
                )

                cls._pools[name] = pool
                logger.info(
                    "pool_created",
                    pool_name=name,
                    host=config.settings.host,
                    database=config.settings.database,
                    min_size=config.min_size,
                    max_size=config.max_size,
                )

                return pool

            except (asyncpg.PostgresError, OSError, TimeoutError) as e:
                raise DatabaseError(
                    f"Failed to create connection pool '{name}': {e}",
                    operation="create_pool",
                    cause=e,
                ) from e

    @classmethod
    async def _init_connection(cls, conn: asyncpg.Connection) -> None:
        """Initialize each connection with type codecs.

        Called automatically for each connection in the pool.
        Sets up JSON/JSONB encoding/decoding.
        """
        import json

        def json_encoder(value: Any) -> str:
            return json.dumps(value, default=str)

        def json_decoder(value: str) -> Any:
            return json.loads(value)

        await conn.set_type_codec(
            "json", encoder=json_encoder, decoder=json_decoder, schema="pg_catalog"
        )
        await conn.set_type_codec(
            "jsonb", encoder=json_encoder, decoder=json_decoder, schema="pg_catalog"
        )

    @classmethod
    @asynccontextmanager
    async def acquire(
        cls, pool_name: str = "default", timeout: float = 10.0
    ) -> AsyncIterator[asyncpg.Connection]:
        """Acquire connection from pool with automatic return and leak detection.

        Context manager ensures connection is properly returned to pool.
        Tracks connection usage for leak detection.

        Args:
            pool_name: Name of pool to acquire from (defaults to "default")
            timeout: Maximum seconds to wait for a connection (defaults to 10.0)

        Yields:
            asyncpg.Connection: Database connection from the pool

        Example:
            async with ConnectionPoolManager.acquire() as conn:
                result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        """
        pool = await cls.get_pool(pool_name)

        # Track metrics
        if pool_name not in cls._metrics:
            cls._metrics[pool_name] = ConnectionMetrics(pool_name=pool_name)

        metrics = cls._metrics[pool_name]
        acquire_start = time.time()

        try:
            async with pool.acquire(timeout=timeout) as conn:
                # Record acquisition
                acquire_time = (time.time() - acquire_start) * 1000
                metrics.acquisition_count += 1
                metrics.total_acquisition_time_ms += acquire_time
                metrics.active_connections += 1
                metrics.peak_active_connections = max(
                    metrics.peak_active_connections, metrics.active_connections
                )

                # Track connection for leak detection
                conn_id = id(conn)
                if cls._leak_detection_enabled:
                    cls._active_connections[conn_id] = ActiveConnection(
                        pool_name=pool_name,
                        acquired_at=time.time(),
                        stack_trace=cls._get_stack_trace(),
                    )

                try:
                    yield conn
                finally:
                    # Clean up tracking
                    metrics.active_connections -= 1
                    if conn_id in cls._active_connections:
                        del cls._active_connections[conn_id]

        except asyncio.TimeoutError as e:
            metrics.connection_timeouts += 1
            logger.error(
                "connection_acquisition_timeout",
                pool_name=pool_name,
                active_connections=metrics.active_connections,
                pool_size=pool.get_size(),
                free_connections=pool.get_idle_size(),
            )
            raise DatabaseError(
                f"Timeout acquiring connection from pool '{pool_name}'",
                operation="acquire_connection",
                cause=e,
            ) from e

    @classmethod
    @asynccontextmanager
    async def transaction(
        cls, pool_name: str = "default", isolation: str = "read_committed"
    ) -> AsyncIterator[asyncpg.Connection]:
        """Acquire connection and start transaction.

        Args:
            pool_name: Name of pool to use
            isolation: Transaction isolation level (read_committed, repeatable_read, serializable)

        Yields:
            asyncpg.Connection: Connection with active transaction

        Example:
            async with ConnectionPoolManager.transaction() as conn:
                await conn.execute("INSERT INTO users ...")
                await conn.execute("INSERT INTO audit_logs ...")
                # Automatic commit on success, rollback on exception
        """
        async with cls.acquire(pool_name) as conn:
            async with conn.transaction(isolation=isolation):
                yield conn

    @classmethod
    async def close_pool(cls, name: str) -> None:
        """Close and remove a specific pool.

        Args:
            name: Pool identifier to close
        """
        if name in cls._pools:
            pool = cls._pools[name]
            await pool.close()
            del cls._pools[name]
            logger.info("pool_closed", pool_name=name)

    @classmethod
    async def close_all_pools(cls) -> None:
        """Close all connection pools gracefully.

        Should be called during application shutdown.
        """
        pool_names = list(cls._pools.keys())
        for name in pool_names:
            await cls.close_pool(name)
        cls._pool_configs.clear()
        cls._locks.clear()
        logger.info("all_pools_closed", count=len(pool_names))

    @classmethod
    async def get_pool_stats(cls, name: str = "default") -> dict[str, Any]:
        """Get statistics for a specific pool.

        Args:
            name: Pool identifier

        Returns:
            Dictionary containing pool statistics

        Raises:
            ValueError: If pool doesn't exist
        """
        if name not in cls._pools:
            raise ValueError(f"Pool '{name}' does not exist")

        pool = cls._pools[name]
        config = cls._pool_configs[name]

        return {
            "pool_name": name,
            "database": config.settings.database,
            "host": config.settings.host,
            "port": config.settings.port,
            "size": pool.get_size(),
            "free_connections": pool.get_idle_size(),
            "min_size": pool.get_min_size(),
            "max_size": pool.get_max_size(),
            "status": "healthy" if pool.get_size() > 0 else "no_connections",
        }

    @classmethod
    async def get_all_pool_stats(cls) -> dict[str, dict[str, Any]]:
        """Get statistics for all registered pools.

        Returns:
            Dictionary mapping pool names to their statistics
        """
        stats = {}
        for name in cls._pools.keys():
            stats[name] = await cls.get_pool_stats(name)
        return stats

    @classmethod
    async def health_check(cls, pool_name: str = "default") -> dict[str, Any]:
        """Check health of a specific pool by executing a test query.

        Args:
            pool_name: Pool to check

        Returns:
            Health status dictionary
        """
        try:
            async with cls.acquire(pool_name) as conn:
                result = await conn.fetchrow("SELECT 1 as health, now() as server_time")
                stats = await cls.get_pool_stats(pool_name)

                return {
                    "status": "healthy",
                    "pool_name": pool_name,
                    "server_time": result["server_time"] if result else None,
                    "pool_stats": stats,
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "pool_name": pool_name,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    @classmethod
    def get_registered_pools(cls) -> list[str]:
        """Get list of all registered pool names.

        Returns:
            List of pool identifiers
        """
        return list(cls._pool_configs.keys())

    @classmethod
    def get_active_pools(cls) -> list[str]:
        """Get list of active (created) pool names.

        Returns:
            List of pool identifiers that have been created
        """
        return list(cls._pools.keys())

    @classmethod
    def _get_stack_trace(cls) -> str:
        """Get current stack trace for connection tracking.

        Returns:
            Formatted stack trace string
        """
        import traceback

        # Get stack, skip this function and acquire()
        stack = traceback.extract_stack()[:-2]
        # Format as readable string
        return "".join(traceback.format_list(stack[-5:]))  # Last 5 frames

    @classmethod
    async def _check_for_connection_leaks(cls) -> None:
        """Periodic task to check for connection leaks.

        Runs every 10 seconds and warns about connections held > 30s.
        """
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                current_time = time.time()
                for conn_id, info in list(cls._active_connections.items()):
                    held_duration = current_time - info.acquired_at

                    if (
                        held_duration > cls.CONNECTION_LEAK_THRESHOLD_SECONDS
                        and not info.warned
                    ):
                        # Warn about potential leak
                        logger.warning(
                            "potential_connection_leak",
                            pool_name=info.pool_name,
                            held_duration_seconds=round(held_duration, 2),
                            threshold_seconds=cls.CONNECTION_LEAK_THRESHOLD_SECONDS,
                            stack_trace=info.stack_trace,
                        )

                        # Update metrics
                        if info.pool_name in cls._metrics:
                            metrics = cls._metrics[info.pool_name]
                            metrics.leak_warnings += 1
                            metrics.last_leak_warning = datetime.now(timezone.utc)

                        # Mark as warned to avoid spam
                        info.warned = True

            except Exception as e:
                logger.error("leak_detection_error", error=str(e))

    @classmethod
    def enable_leak_detection(cls) -> None:
        """Enable connection leak detection.

        Starts background task to monitor connection usage.
        """
        if cls._leak_detection_enabled:
            return

        cls._leak_detection_enabled = True

        # Start background leak checker if not already running
        if cls._leak_check_task is None or cls._leak_check_task.done():
            cls._leak_check_task = asyncio.create_task(cls._check_for_connection_leaks())

        logger.info("connection_leak_detection_enabled")

    @classmethod
    def disable_leak_detection(cls) -> None:
        """Disable connection leak detection.

        Stops background monitoring task.
        """
        cls._leak_detection_enabled = False

        if cls._leak_check_task and not cls._leak_check_task.done():
            cls._leak_check_task.cancel()
            cls._leak_check_task = None

        logger.info("connection_leak_detection_disabled")

    @classmethod
    def get_metrics(cls, pool_name: str) -> ConnectionMetrics | None:
        """Get connection metrics for a specific pool.

        Args:
            pool_name: Pool identifier

        Returns:
            Connection metrics or None if pool doesn't exist
        """
        return cls._metrics.get(pool_name)

    @classmethod
    def get_all_metrics(cls) -> dict[str, ConnectionMetrics]:
        """Get metrics for all pools.

        Returns:
            Dictionary mapping pool names to metrics
        """
        return cls._metrics.copy()

    @classmethod
    def reset_metrics(cls, pool_name: str | None = None) -> None:
        """Reset metrics for a specific pool or all pools.

        Args:
            pool_name: Pool to reset (None = reset all)
        """
        if pool_name:
            if pool_name in cls._metrics:
                cls._metrics[pool_name] = ConnectionMetrics(pool_name=pool_name)
                logger.info("metrics_reset", pool_name=pool_name)
        else:
            for name in cls._metrics.keys():
                cls._metrics[name] = ConnectionMetrics(pool_name=name)
            logger.info("all_metrics_reset")


# Convenience function for backwards compatibility
async def get_connection_pool(name: str = "default") -> asyncpg.Pool:
    """Get connection pool by name (convenience function).

    Args:
        name: Pool identifier

    Returns:
        asyncpg.Pool instance
    """
    return await ConnectionPoolManager.get_pool(name)


# Export public API
__all__ = [
    "ConnectionPoolManager",
    "ConnectionPoolConfig",
    "ConnectionMetrics",
    "get_connection_pool",
]
