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

One default pool per service (IF-03, D-02; MVP-033):
    ``get_pool("default")`` is the service's single pool, built from ``POSTGRES_*``. The legacy repository
    names (``user_db``, ``diagnosis_db``, ``therapy_db``, ``personality_db``, ``safety_db``) are aliases of
    it (:func:`canonical_pool_name`), so they share the one pool object. In staging and production a pool is
    refused (:class:`~solace_infrastructure.postgres.DatabaseConfigError`, before any connection) for an unset
    or local POSTGRES_HOST or a missing or empty POSTGRES_PASSWORD, and a new pool must log in as a plain
    application role, as itself (:func:`verify_app_role`); a refused pool is closed.

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
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Final

import asyncpg
import structlog
from pydantic import ValidationError

from solace_common.exceptions import DatabaseError

from ..postgres import (
    _SET_RLS_USER_SQL,
    DatabaseConfigError,
    PostgresSettings,
    _is_strict_environment,
    _reset_rls_user,
)

logger = structlog.get_logger(__name__)

DEFAULT_POOL: Final = "default"
LEGACY_POOL_ALIASES: Final = frozenset({"user_db", "diagnosis_db", "therapy_db", "personality_db", "safety_db"})
# Predefined roles the PostgreSQL docs call superuser-equivalent: OS commands through COPY ... TO/FROM PROGRAM,
# and reading or writing any file the server can reach, whatever RLS says. The one list in src/; the test suite
# pins it equal to scripts/db/migrate.py's SUPERUSER_EQUIVALENT_ROLES (the migrate job refuses the same roles).
SUPERUSER_EQUIVALENT_ROLES: Final = ("pg_execute_server_program", "pg_read_server_files", "pg_write_server_files")
ROLE_CHECK_ACQUIRE_TIMEOUT_SECONDS: Final = 10.0
POOL_DISCARD_TIMEOUT_SECONDS: Final = 10.0


def canonical_pool_name(name: str) -> str:
    """The pool a name refers to: ``"default"`` for the legacy per-repository names, else ``name`` itself."""
    return DEFAULT_POOL if name in LEGACY_POOL_ALIASES else name


def _json_encoder(value: Any) -> str:
    return json.dumps(value, default=str)


def _json_decoder(value: str) -> Any:
    return json.loads(value)


async def init_connection(conn: asyncpg.Connection) -> None:
    """The pool ``init=`` hook: JSON and JSONB values travel as Python objects (``json.dumps(default=str)`` /
    ``json.loads``). Public so other pools (WS-05.2's ``pg_migrated`` app pool) use the same codecs."""
    await conn.set_type_codec("json", encoder=_json_encoder, decoder=_json_decoder, schema="pg_catalog")
    await conn.set_type_codec("jsonb", encoder=_json_encoder, decoder=_json_decoder, schema="pg_catalog")


# One row about the connection's roles: the session role (the one that logged in, which ``SET ROLE NONE`` and
# ``RESET ROLE`` return to) and the current role (which a connection-time ``ALTER ROLE/DATABASE ... SET role``
# can switch to). Both are tested (``login``): their own attributes, whether either is a member of the owner of
# any table in schema public (owners bypass RLS, and any member can SET ROLE to the owner, so 'MEMBER' rather
# than 'USAGE', which misses NOINHERIT membership), and whether either is a member of a superuser or BYPASSRLS
# role. A role switched at login is refused outright: the plain app role logs in as itself (WS026-SEC-1).
# Membership of any kind (direct, chained, INHERIT FALSE: 'MEMBER') in a superuser-equivalent predefined role
# ($1, SUPERUSER_EQUIVALENT_ROLES) is refused too (WS026F1-CLO-1), and so is REPLICATION (hardening: the app
# login is NOREPLICATION, as scripts/db/migrate.py requires of the platform roles).
_APP_ROLE_SQL: Final = """
WITH login AS (
    SELECT r.oid, r.rolsuper, r.rolbypassrls, r.rolreplication
    FROM pg_catalog.pg_roles r
    WHERE r.rolname IN (session_user, current_user)
)
SELECT session_user::text AS login_role,
       current_user::text AS effective_role,
       (SELECT count(*) FROM login) AS roles_found,
       session_user <> current_user AS role_switched,
       EXISTS (SELECT 1 FROM login WHERE login.rolsuper) AS is_superuser,
       EXISTS (SELECT 1 FROM login WHERE login.rolbypassrls) AS bypasses_rls,
       EXISTS (
           SELECT 1
           FROM login l
           CROSS JOIN pg_catalog.pg_class c
           JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
             AND pg_catalog.pg_has_role(l.oid, c.relowner, 'MEMBER')
       ) AS owns_public_tables,
       EXISTS (
           SELECT 1
           FROM login l
           CROSS JOIN pg_catalog.pg_roles p
           WHERE p.oid <> l.oid AND (p.rolsuper OR p.rolbypassrls)
             AND pg_catalog.pg_has_role(l.oid, p.oid, 'MEMBER')
       ) AS member_of_privileged_role,
       EXISTS (
           SELECT 1
           FROM login l
           CROSS JOIN pg_catalog.pg_roles p
           WHERE p.rolname = ANY($1::text[])
             AND pg_catalog.pg_has_role(l.oid, p.oid, 'MEMBER')
       ) AS member_of_superuser_equivalent_role,
       EXISTS (SELECT 1 FROM login WHERE login.rolreplication) AS can_replicate
"""
_PRIVILEGE_REASONS: Final = (
    ("is_superuser", "is a superuser"),
    ("bypasses_rls", "has BYPASSRLS"),
    ("owns_public_tables", "owns tables in schema public"),
    ("member_of_privileged_role", "is a member of a superuser or BYPASSRLS role"),
    ("member_of_superuser_equivalent_role",
     "is a member of a superuser-equivalent role (" + ", ".join(SUPERUSER_EQUIVALENT_ROLES) + ")"),
    ("can_replicate", "has REPLICATION"),
    ("role_switched", "switched to another role at login"),
)


async def verify_app_role(conn: Any, *, environment: str | None = None) -> None:
    """Strict environments: refuse a login role that row-level security does not apply to (D-02; MVP-041).

    Both the session role (``session_user``, the role that logged in) and the current role are tested. Raises
    :class:`DatabaseConfigError` when either is a superuser, has BYPASSRLS, is a member (inheriting or not) of the
    owner of any table in schema ``public``, is a member of a superuser or BYPASSRLS role, is a member (direct,
    chained or INHERIT FALSE) of a superuser-equivalent predefined role (:data:`SUPERUSER_EQUIVALENT_ROLES`), or
    has REPLICATION, and when the two differ (a role switched at login by ``ALTER ROLE/DATABASE ... SET role``:
    ``SET ROLE NONE`` returns to the session role). The message names POSTGRES_USER, the reasons and the role
    names only, never a password or DSN. A no-op outside staging and production; ``environment`` None reads
    ENVIRONMENT.
    """
    if not _is_strict_environment(environment):
        return
    row = await conn.fetchrow(_APP_ROLE_SQL, list(SUPERUSER_EQUIVALENT_ROLES))
    if row is None or not row["roles_found"]:
        reasons, roles = ["could not be read from pg_roles"], ""
    else:
        reasons = [text for key, text in _PRIVILEGE_REASONS if row[key]]
        roles = f" (session_user {row['login_role']!r}, current_user {row['effective_role']!r})"
    if reasons:
        raise DatabaseConfigError(
            f"The Postgres login role (POSTGRES_USER) {', '.join(reasons)}{roles}: staging and production services "
            "log in as the application role itself (NOSUPERUSER, NOBYPASSRLS, owning no table, no role switch at "
            "login), so row-level security applies to every query (D-02)",
            operation="verify_app_role",
        )


async def _discard_pool(pool: Any) -> None:
    """Close a pool that will not be used; terminate it when a graceful close fails or takes too long."""
    try:
        await asyncio.wait_for(pool.close(), timeout=POOL_DISCARD_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("pool_discard_failed", error=type(exc).__name__)
        terminate = getattr(pool, "terminate", None)
        if callable(terminate):
            terminate()


async def check_new_pool_role(pool: Any, *, environment: str | None = None) -> None:
    """Strict environments: :func:`verify_app_role` on one connection of a newly created ``pool``.

    On any failure (a refused role, or no connection) the pool is closed before the error propagates, so a
    refused pool is never used or leaked. A no-op outside staging and production.
    """
    if not _is_strict_environment(environment):
        return
    try:
        async with pool.acquire(timeout=ROLE_CHECK_ACQUIRE_TIMEOUT_SECONDS) as conn:
            await verify_app_role(conn, environment=environment)
    except BaseException:
        await _discard_pool(pool)
        raise


def _validated_settings(settings: PostgresSettings | None) -> PostgresSettings:
    """``settings``, or ``PostgresSettings()`` from POSTGRES_* (and .env), after ``validate_for_environment``.

    A missing or invalid POSTGRES_* value (pydantic ``ValidationError``, e.g. an unset POSTGRES_PASSWORD) becomes
    :class:`DatabaseConfigError` naming the variables, never their values; the pydantic error, whose text can
    echo other settings, is not chained.
    """
    if settings is None:
        try:
            settings = PostgresSettings()
        except ValidationError as exc:
            names = sorted({f"POSTGRES_{str(error['loc'][0]).upper()}"
                            for error in exc.errors(include_url=False, include_input=False) if error.get("loc")})
            raise DatabaseConfigError(
                f"{', '.join(names) or 'POSTGRES_*'} is not set or not valid: the service's database pool is "
                "built from the POSTGRES_* variables",
                operation="validate_settings",
            ) from None
    settings.validate_for_environment()
    return settings


def _connection_identity(settings: PostgresSettings) -> tuple[Any, ...]:
    """Where and how a pool connects; two registrations of one name must agree on all of it."""
    return (settings.host, settings.port, settings.database, settings.user, settings.password.get_secret_value(),
            settings.ssl_mode, settings.ssl_root_cert)


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
            name: Unique pool identifier (e.g., "default", "audit_db", "analytics_db"); a legacy alias
                (:data:`LEGACY_POOL_ALIASES`) registers ``"default"``
            settings: PostgreSQL settings for this pool (built from POSTGRES_* if None)
            min_size: Override minimum pool size
            max_size: Override maximum pool size

        Raises:
            DatabaseConfigError: A POSTGRES_* value is missing or invalid (e.g. no POSTGRES_PASSWORD), or, in
                staging and production, POSTGRES_HOST is unset or local or POSTGRES_PASSWORD is empty.
            ValueError: If pool name is already registered with different settings (the message names the
                pool, host and database only, never a DSN or password)
        """
        name = canonical_pool_name(name)
        pool_settings = _validated_settings(settings)
        config = ConnectionPoolConfig(name, pool_settings, min_size, max_size)

        # Check for conflicts
        if name in cls._pool_configs:
            existing = cls._pool_configs[name].settings
            if _connection_identity(existing) != _connection_identity(pool_settings):
                raise ValueError(
                    f"Pool '{name}' is already registered with different settings "
                    f"(registered: host {existing.host}, database {existing.database}; "
                    f"new: host {pool_settings.host}, database {pool_settings.database})"
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
        Subsequent calls return the same pool instance (singleton pattern). A legacy alias
        (:data:`LEGACY_POOL_ALIASES`) returns the ``"default"`` pool.

        In staging and production the new pool must pass :func:`verify_app_role` on one of its
        connections; a refused pool is closed before the error is raised.

        Args:
            name: Pool identifier (defaults to "default")

        Returns:
            asyncpg.Pool instance for the specified database

        Raises:
            DatabaseConfigError: Unusable settings (see :meth:`register_pool`), a strict TLS problem, or, in
                staging and production, a privileged login role. Raised before any connection, except the role
                check, which needs one.
            DatabaseError: If pool creation fails
        """
        name = canonical_pool_name(name)
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
                # Auto-register from POSTGRES_*: the designed path for the default pool.
                (logger.info if name == DEFAULT_POOL else logger.warning)(
                    "pool_auto_registered",
                    pool_name=name,
                    message="Pool not registered, using the POSTGRES_* settings",
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
                await check_new_pool_role(pool)  # strict only; closes the pool before raising

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
        """Initialize each connection with type codecs (delegates to :func:`init_connection`).

        Called automatically for each connection in the pool.
        Sets up JSON/JSONB encoding/decoding.
        """
        await init_connection(conn)

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
        pool_name = canonical_pool_name(pool_name)
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

                # C.2 RLS: set the per-request user GUC on this connection so Postgres
                # RLS policies keyed on app.current_user_id enforce per-user row access.
                # Session-level (is_local=false) so it survives the per-statement autocommits
                # within this acquire; reset on release so it never leaks to the next
                # borrower of this pooled connection; a connection whose reset fails is
                # terminated instead (MVP-058). Unset => RLS returns no user rows (safe).
                from solace_common.request_context import get_current_user_id

                _rls_uid = get_current_user_id()
                try:
                    if _rls_uid:
                        await conn.execute(_SET_RLS_USER_SQL, _rls_uid)
                    yield conn
                finally:
                    try:
                        if _rls_uid:
                            await _reset_rls_user(conn)  # re-raises only a cancellation
                    finally:
                        # Clean up tracking (also when the reset was cancelled)
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
        pool_name = canonical_pool_name(pool_name)
        async with cls.acquire(pool_name) as conn:
            async with conn.transaction(isolation=isolation):
                yield conn

    @classmethod
    async def close_pool(cls, name: str) -> None:
        """Close and remove a specific pool.

        Args:
            name: Pool identifier to close (a legacy alias closes ``"default"``)
        """
        name = canonical_pool_name(name)
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
        name = canonical_pool_name(name)
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
        pool_name = canonical_pool_name(pool_name)
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
        return cls._metrics.get(canonical_pool_name(pool_name))

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
            pool_name = canonical_pool_name(pool_name)
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
    "DEFAULT_POOL",
    "LEGACY_POOL_ALIASES",
    "SUPERUSER_EQUIVALENT_ROLES",
    "ConnectionPoolManager",
    "ConnectionPoolConfig",
    "ConnectionMetrics",
    "canonical_pool_name",
    "check_new_pool_role",
    "get_connection_pool",
    "init_connection",
    "verify_app_role",
]
