"""Solace-AI PostgreSQL Client - Async database operations with connection pooling."""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, TypeVar
from uuid import UUID
import asyncpg
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog
from solace_common.exceptions import DatabaseError, ConfigurationError

logger = structlog.get_logger(__name__)
T = TypeVar("T")


class IsolationLevel(str, Enum):
    """PostgreSQL transaction isolation levels."""

    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings from environment."""

    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="solace")
    user: str = Field(default="solace")
    password: SecretStr = Field(default=SecretStr("solace"))
    min_pool_size: int = Field(default=5, ge=1, le=100)
    max_pool_size: int = Field(default=20, ge=1, le=200)
    command_timeout: float = Field(default=60.0, gt=0)
    statement_cache_size: int = Field(default=100, ge=0)
    max_cached_statement_lifetime: int = Field(default=300, ge=0)
    ssl_mode: str = Field(default="prefer")
    db_schema: str = Field(default="public")
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_", env_file=".env", extra="ignore"
    )

    def get_dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class QueryResult(BaseModel):
    """Container for query execution results."""

    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0)
    execution_time_ms: float = Field(default=0.0)

    @property
    def first(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None

    @property
    def scalar(self) -> Any:
        if self.rows and self.rows[0]:
            return next(iter(self.rows[0].values()), None)
        return None


class PostgresClient:
    """Async PostgreSQL client with connection pooling and transaction support."""

    def __init__(self, settings: PostgresSettings | None = None) -> None:
        self._settings = settings or PostgresSettings()
        self._pool: asyncpg.Pool | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._pool is not None

    async def connect(self) -> None:
        """Initialize connection pool."""
        if self._pool is not None:
            return
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._settings.get_dsn(),
                min_size=self._settings.min_pool_size,
                max_size=self._settings.max_pool_size,
                command_timeout=self._settings.command_timeout,
                statement_cache_size=self._settings.statement_cache_size,
                max_cached_statement_lifetime=self._settings.max_cached_statement_lifetime,
                init=self._init_connection,
            )
            self._connected = True
            logger.info(
                "postgres_pool_connected",
                host=self._settings.host,
                database=self._settings.database,
                pool_size=self._settings.max_pool_size,
            )
        except asyncpg.PostgresError as e:
            raise DatabaseError(
                f"Failed to connect to PostgreSQL: {e}", operation="connect", cause=e
            )

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize each connection with type codecs."""
        await conn.set_type_codec(
            "json", encoder=_json_encoder, decoder=_json_decoder, schema="pg_catalog"
        )
        await conn.set_type_codec(
            "jsonb", encoder=_json_encoder, decoder=_json_decoder, schema="pg_catalog"
        )

    async def disconnect(self) -> None:
        """Close connection pool gracefully."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._connected = False
            logger.info("postgres_pool_disconnected")

    def _ensure_connected(self) -> asyncpg.Pool:
        if not self._pool:
            raise DatabaseError(
                "PostgreSQL client not connected", operation="check_connection"
            )
        return self._pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection from the pool."""
        pool = self._ensure_connected()
        async with pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(
        self, isolation: IsolationLevel = IsolationLevel.READ_COMMITTED
    ) -> AsyncIterator[asyncpg.Connection]:
        """Execute operations within a transaction context."""
        async with self.acquire() as conn:
            async with conn.transaction(isolation=isolation.value):
                yield conn

    async def execute(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> str:
        """Execute a query that doesn't return rows."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        async with self.acquire() as conn:
            try:
                result = await conn.execute(query, *args, timeout=timeout)
                elapsed = (loop.time() - start) * 1000
                logger.debug(
                    "postgres_execute", query=_truncate_query(query), elapsed_ms=elapsed
                )
                return result
            except asyncpg.PostgresError as e:
                raise DatabaseError(
                    f"Query execution failed: {e}", operation="execute", cause=e
                )

    async def fetch(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> QueryResult:
        """Execute query and fetch all rows."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        async with self.acquire() as conn:
            try:
                rows = await conn.fetch(query, *args, timeout=timeout)
                elapsed = (loop.time() - start) * 1000
                result = QueryResult(
                    rows=[dict(r) for r in rows],
                    row_count=len(rows),
                    execution_time_ms=elapsed,
                )
                logger.debug(
                    "postgres_fetch",
                    query=_truncate_query(query),
                    row_count=result.row_count,
                    elapsed_ms=elapsed,
                )
                return result
            except asyncpg.PostgresError as e:
                raise DatabaseError(
                    f"Query fetch failed: {e}", operation="fetch", cause=e
                )

    async def fetch_one(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """Execute query and fetch single row."""
        async with self.acquire() as conn:
            try:
                row = await conn.fetchrow(query, *args, timeout=timeout)
                return dict(row) if row else None
            except asyncpg.PostgresError as e:
                raise DatabaseError(
                    f"Query fetch_one failed: {e}", operation="fetch_one", cause=e
                )

    async def fetch_val(
        self, query: str, *args: Any, column: int = 0, timeout: float | None = None
    ) -> Any:
        """Execute query and fetch single value."""
        async with self.acquire() as conn:
            try:
                return await conn.fetchval(query, *args, column=column, timeout=timeout)
            except asyncpg.PostgresError as e:
                raise DatabaseError(
                    f"Query fetch_val failed: {e}", operation="fetch_val", cause=e
                )

    async def execute_many(
        self, query: str, args_list: list[tuple[Any, ...]], timeout: float | None = None
    ) -> None:
        """Execute query with multiple argument sets."""
        async with self.acquire() as conn:
            try:
                await conn.executemany(query, args_list, timeout=timeout)
                logger.debug(
                    "postgres_execute_many",
                    query=_truncate_query(query),
                    batch_size=len(args_list),
                )
            except asyncpg.PostgresError as e:
                raise DatabaseError(
                    f"Batch execution failed: {e}", operation="execute_many", cause=e
                )

    async def copy_records(
        self,
        table_name: str,
        records: list[tuple[Any, ...]],
        columns: list[str],
        timeout: float | None = None,
    ) -> str:
        """Bulk copy records into table using COPY protocol."""
        async with self.acquire() as conn:
            try:
                result = await conn.copy_records_to_table(
                    table_name, records=records, columns=columns, timeout=timeout
                )
                logger.info(
                    "postgres_copy_records", table=table_name, record_count=len(records)
                )
                return result
            except asyncpg.PostgresError as e:
                raise DatabaseError(
                    f"Copy records failed: {e}", operation="copy_records", cause=e
                )

    async def check_health(self) -> dict[str, Any]:
        """Check database connectivity and return health status."""
        try:
            result = await self.fetch_one("SELECT 1 as health, now() as server_time")
            pool = self._ensure_connected()
            return {
                "status": "healthy",
                "server_time": result["server_time"] if result else None,
                "pool_size": pool.get_size(),
                "pool_free": pool.get_idle_size(),
                "pool_min": pool.get_min_size(),
                "pool_max": pool.get_max_size(),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class PostgresRepository:
    """Base repository class for entity persistence."""

    def __init__(
        self, client: PostgresClient, table_name: str, schema: str = "public"
    ) -> None:
        self._client = client
        self._table = table_name
        self._schema = schema

    @property
    def qualified_table(self) -> str:
        return f"{self._schema}.{self._table}"

    async def find_by_id(self, entity_id: UUID | str) -> dict[str, Any] | None:
        """Find entity by primary key."""
        query = f"SELECT * FROM {self.qualified_table} WHERE id = $1"
        return await self._client.fetch_one(query, entity_id)

    async def find_all(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Find all entities with pagination."""
        query = f"SELECT * FROM {self.qualified_table} ORDER BY created_at DESC LIMIT $1 OFFSET $2"
        result = await self._client.fetch(query, limit, offset)
        return result.rows

    async def exists(self, entity_id: UUID | str) -> bool:
        """Check if entity exists."""
        query = f"SELECT EXISTS(SELECT 1 FROM {self.qualified_table} WHERE id = $1)"
        return await self._client.fetch_val(query, entity_id)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities with optional filter.

        Args:
            filters: Dictionary of column names to values for WHERE clause.
                    All conditions are ANDed together with equality checks.
                    Column names are validated to prevent SQL injection.

        Returns:
            Count of matching records.

        Raises:
            ValueError: If any column name contains invalid characters.
        """
        query = f"SELECT COUNT(*) FROM {self.qualified_table}"
        if filters:
            conditions = []
            values = []
            for i, (column, value) in enumerate(filters.items(), start=1):
                # Validate column name to prevent SQL injection
                if not _is_valid_identifier(column):
                    raise ValueError(f"Invalid column name: {column}")
                conditions.append(f"{column} = ${i}")
                values.append(value)
            query += " WHERE " + " AND ".join(conditions)
            return await self._client.fetch_val(query, *values)
        return await self._client.fetch_val(query)

    async def insert(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Insert entity and return created record."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f"${i+1}" for i in range(len(data)))
        query = f"INSERT INTO {self.qualified_table} ({columns}) VALUES ({placeholders}) RETURNING *"
        return await self._client.fetch_one(query, *data.values())

    async def update(
        self, entity_id: UUID | str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update entity and return updated record."""
        data["updated_at"] = datetime.now(timezone.utc)
        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(data.keys()))
        query = (
            f"UPDATE {self.qualified_table} SET {set_clause} WHERE id = $1 RETURNING *"
        )
        return await self._client.fetch_one(query, entity_id, *data.values())

    async def delete(self, entity_id: UUID | str) -> bool:
        """Delete entity by ID."""
        query = f"DELETE FROM {self.qualified_table} WHERE id = $1"
        result = await self._client.execute(query, entity_id)
        return result == "DELETE 1"

    async def soft_delete(self, entity_id: UUID | str) -> bool:
        """Soft delete by setting deleted_at timestamp."""
        query = f"UPDATE {self.qualified_table} SET deleted_at = $2 WHERE id = $1"
        result = await self._client.execute(
            query, entity_id, datetime.now(timezone.utc)
        )
        return "UPDATE 1" in result


def _json_encoder(value: Any) -> str:
    import json

    return json.dumps(value, default=str)


def _json_decoder(value: str) -> Any:
    import json

    return json.loads(value)


def _truncate_query(query: str, max_length: int = 200) -> str:
    """Truncate query for logging purposes."""
    query = " ".join(query.split())
    return query[:max_length] + "..." if len(query) > max_length else query


import re

# Pattern for valid SQL identifiers (letters, digits, underscore, starting with letter/underscore)
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _is_valid_identifier(name: str) -> bool:
    """Validate that a string is a safe SQL identifier.

    Prevents SQL injection by ensuring column/table names only contain
    alphanumeric characters and underscores, and start with a letter or underscore.

    Args:
        name: The identifier to validate.

    Returns:
        True if the identifier is safe to use in SQL queries.
    """
    if not name or len(name) > 128:  # PostgreSQL identifier limit is 63, but be generous
        return False
    return _IDENTIFIER_PATTERN.match(name) is not None


async def create_postgres_client(
    settings: PostgresSettings | None = None,
) -> PostgresClient:
    """Factory function to create and connect a PostgreSQL client."""
    client = PostgresClient(settings)
    await client.connect()
    return client
