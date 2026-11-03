"""Solace-AI PostgreSQL Client - Async database operations with connection pooling."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import socket
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Final, TypeVar
from urllib.parse import quote
from uuid import UUID

import asyncpg
import structlog
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from solace_common.exceptions import DatabaseError
from solace_common.utils import ValidationUtils

logger = structlog.get_logger(__name__)
T = TypeVar("T")


# --- TLS policy (D-15, D-14; MVP-010, MVP-271) -------------------------------------------------------------------
# One policy for every Postgres client: PostgresClient, ConnectionPoolManager (both through PostgresSettings),
# migrations/env.py and scripts/db/migrate.py all call effective_ssl_mode / build_ssl_context.

SSL_MODES: Final[tuple[str, ...]] = ("disable", "allow", "prefer", "require", "verify-ca", "verify-full")
STRICT_SSL_MODE: Final = "verify-full"
_STRICT_ENVIRONMENTS: Final = frozenset({"staging", "production"})
_UPGRADE_REASON: Final = "strict environments verify the server certificate and host name (D-15)"


class DatabaseConfigError(DatabaseError):
    """The database client settings or login role are unusable.

    Settings errors are raised before any connection attempt: the TLS policy (``POSTGRES_SSL_ROOT_CERT``,
    ``POSTGRES_SSL_MODE``), a strict environment's unset or local ``POSTGRES_HOST`` or empty ``POSTGRES_PASSWORD``
    (:meth:`PostgresSettings.validate_for_environment`), and a missing or invalid ``POSTGRES_*`` value when the pool
    manager builds its settings. The strict role refusal
    (:func:`solace_infrastructure.database.connection_manager.verify_app_role`, ``POSTGRES_USER``) is raised after
    one connection, and the pool it opened has been closed. The message names the variable (``POSTGRES_HOST``,
    ``POSTGRES_PASSWORD``, ``POSTGRES_USER``, ``POSTGRES_SSL_*``), never its value; the role refusal also names
    the session and current role names, never a password or DSN.
    """

    error_code = "DATABASE_CONFIG_ERROR"


def _is_strict_environment(environment: str | None = None) -> bool:
    """True for staging and production (the IF-01 ``is_strict_environment`` rule, read locally).

    ``environment`` None reads the bare ``ENVIRONMENT`` variable, defaulting to ``development``; surrounding
    whitespace and case are ignored.
    """
    if environment is None:
        environment = os.environ.get("ENVIRONMENT", "development")
    return environment.strip().lower() in _STRICT_ENVIRONMENTS


def _normalized_mode(configured: str | None) -> str:
    """Surrounding whitespace is dropped and a blank value means unset, i.e. the default ``prefer``."""
    return (configured or "").strip() or "prefer"


def effective_ssl_mode(configured: str, *, environment: str | None = None) -> str:
    """The sslmode a client actually uses.

    Strict environments (staging, production) always use ``verify-full``; a WARNING ``ssl_mode_upgraded`` names
    the configured mode when it differs (an unrecognised value is logged as ``<unrecognised>``, never echoed).
    Elsewhere the configured mode is returned as configured (stripped; blank means ``prefer``).
    """
    configured = _normalized_mode(configured)
    if not _is_strict_environment(environment):
        return configured
    if configured != STRICT_SSL_MODE:
        named = environment if environment is not None else os.environ.get("ENVIRONMENT", "")
        logger.warning(
            "ssl_mode_upgraded",
            original_mode=configured if configured in SSL_MODES else "<unrecognised>",
            effective_mode=STRICT_SSL_MODE,
            environment=named.strip().lower(),
            reason=_UPGRADE_REASON,
        )
    return STRICT_SSL_MODE


def _config_error(message: str, *, operation: str = "ssl_context") -> DatabaseConfigError:
    return DatabaseConfigError(message, operation=operation)


_HOST_ERROR: Final = (
    "POSTGRES_HOST is unset, is not a plain host name or address (an IPv6 address goes in brackets), or names this "
    "machine (a loopback name or address, or a unix socket): staging and production reach the database server "
    "over the network with verified TLS"
)
_PASSWORD_ERROR: Final = (
    "POSTGRES_PASSWORD is empty: staging and production log in to Postgres with the application role's password"
)
# A plain DNS name or IPv4 literal: ASCII letters, digits, '_', '-' and '.', with at most a trailing dot. Nothing
# asyncpg reads differently from the string: it percent-decodes the host, its URL parsing cuts the host at '/', '?'
# and '#' (and splits user from host at the first '@', so an '@' in POSTGRES_HOST becomes part of the name it dials),
# and it takes a host up to the first ':' unless bracketed.
_PLAIN_HOST_NAME: Final = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,251}[A-Za-z0-9_])?\.?")
_LEGACY_IPV4: Final = re.compile(r"[0-9A-Fa-fXx.]+")


def _is_local_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return address.is_loopback or address.is_unspecified


def _legacy_ipv4(name: str) -> ipaddress.IPv4Address | None:
    """The address getaddrinfo reads from an inet_aton spelling such as ``127.1`` or ``2130706433``, else None."""
    if not _LEGACY_IPV4.fullmatch(name):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(name))
    except (OSError, ValueError):
        return None


def _is_local_host_part(part: str) -> bool:
    """False only for a plain remote host: a DNS name, an IPv4 address or a bracketed IPv6 address that is not a
    loopback or unspecified address or a ``localhost`` name. Anything else counts as local (refused in strict
    environments), because asyncpg could dial something other than what the string seems to say: a unix socket
    (no TLS) for ``%2F...``, loopback for ``%6Cocalhost`` or ``localhost/x``."""
    name = part.strip()
    if not name or "%" in name:  # blank, or percent-encoded (asyncpg decodes it)
        return True
    if name.startswith("[") and name.endswith("]"):
        try:
            return _is_local_address(ipaddress.IPv6Address(name[1:-1]))
        except ValueError:
            return True
    if not _PLAIN_HOST_NAME.fullmatch(name):
        return True  # a socket path, an unbracketed IPv6 address, '@', '/', '?', '#', ':', spaces, non-ASCII
    name = name.lower().rstrip(".")
    if name == "localhost" or name.endswith(".localhost"):  # RFC 6761: always loopback
        return True
    try:
        address: ipaddress.IPv4Address | ipaddress.IPv6Address | None = ipaddress.ip_address(name)
    except ValueError:
        address = _legacy_ipv4(name)
    return address is not None and _is_local_address(address)


def _is_local_host(host: str | None) -> bool:
    """True unless every member of ``host`` (one host, or asyncpg's comma-separated list) is a plain remote host."""
    return any(_is_local_host_part(part) for part in (host or "").split(","))


# --- RLS user GUC (MVP-058) ---------------------------------------------------------------------------------------
_SET_RLS_USER_SQL: Final = "SELECT set_config('app.current_user_id', $1, false)"
_RESET_RLS_USER_SQL: Final = "SELECT set_config('app.current_user_id', '', false)"


def _terminate_connection(conn: Any) -> None:
    """Close ``conn`` at once (asyncpg ``terminate()``); the pool replaces a terminated connection on release."""
    try:
        conn.terminate()
    except Exception as exc:  # already closed, or a connection object without terminate()
        logger.warning("rls_connection_terminate_failed", error=type(exc).__name__)


async def _reset_rls_user(conn: Any) -> None:
    """Clear ``app.current_user_id`` before ``conn`` goes back to its pool.

    A connection whose reset fails would carry the last user's id to its next borrower, so it is terminated
    instead of being handed back. An ordinary failure never fails the request (the WARNING names the error class
    only); a cancellation is re-raised after the connection is terminated.
    """
    try:
        await conn.execute(_RESET_RLS_USER_SQL)
    except BaseException as exc:
        _terminate_connection(conn)
        logger.warning("rls_guc_reset_failed", error=type(exc).__name__, action="connection_terminated")
        if not isinstance(exc, Exception):
            raise


def _root_cert_path(ssl_root_cert: str | None) -> str | None:
    path = (ssl_root_cert or "").strip()
    return path or None


def _load_root_cert(ctx: ssl.SSLContext, path: str) -> None:
    """Trust exactly the CA certificate(s) in ``path`` (read once, so the file checked is the file loaded)."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:  # missing, a directory, unreadable
        raise _config_error("POSTGRES_SSL_ROOT_CERT does not name a readable file") from None
    try:
        ctx.load_verify_locations(cadata=data.decode("ascii"))
    except (UnicodeDecodeError, ssl.SSLError, ValueError):
        raise _config_error("POSTGRES_SSL_ROOT_CERT does not hold a PEM CA certificate") from None


def _verifying_context(ssl_root_cert: str | None, *, check_hostname: bool) -> ssl.SSLContext:
    """CERT_REQUIRED against ``ssl_root_cert`` alone; without one, the system store (non-strict callers only)."""
    path = _root_cert_path(ssl_root_cert)
    if path is None:
        ctx = ssl.create_default_context()
    else:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # loads no CA certificates of its own
        _load_root_cert(ctx, path)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = check_hostname
    return ctx


def build_ssl_context(
    ssl_mode: str, ssl_root_cert: str | None, *, environment: str | None = None
) -> ssl.SSLContext | None:
    """The ``ssl=`` argument for asyncpg (``connect_args={"ssl": ...}`` for SQLAlchemy).

    Strict environments: whatever ``ssl_mode`` says, a context that requires a server certificate chained to the
    CA file ``ssl_root_cert`` (POSTGRES_SSL_ROOT_CERT, and nothing from the system store) and a host name that
    matches it (verify-full). An unset or unreadable root certificate raises :class:`DatabaseConfigError`.

    Elsewhere: ``disable``, ``allow`` and ``prefer`` give None (asyncpg negotiates, so a local non-TLS server
    works); ``require`` encrypts without verifying; ``verify-ca`` and ``verify-full`` verify against
    ``ssl_root_cert`` alone when it is set, else against the system store. Any other mode raises
    :class:`DatabaseConfigError`.
    """
    if _is_strict_environment(environment):
        effective_ssl_mode(ssl_mode, environment=environment)  # logs the upgrade
        if _root_cert_path(ssl_root_cert) is None:
            raise _config_error(
                "POSTGRES_SSL_ROOT_CERT is not set: staging and production verify the Postgres server "
                "certificate and host name (verify-full) against that CA file"
            ) from None
        return _verifying_context(ssl_root_cert, check_hostname=True)
    ssl_mode = _normalized_mode(ssl_mode)
    if ssl_mode in ("disable", "allow", "prefer"):
        return None
    if ssl_mode == "require":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # encryption without verification: development only
        return ctx
    if ssl_mode in ("verify-ca", "verify-full"):
        return _verifying_context(ssl_root_cert, check_hostname=ssl_mode == "verify-full")
    raise _config_error(f"POSTGRES_SSL_MODE is not one of {', '.join(SSL_MODES)}") from None


class IsolationLevel(str, Enum):
    """PostgreSQL transaction isolation levels."""

    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings from environment.

    SSL/TLS (D-15; :func:`effective_ssl_mode`, :func:`build_ssl_context`):
    - In staging and production the effective mode is always "verify-full": the server certificate must chain to
      ``ssl_root_cert`` (POSTGRES_SSL_ROOT_CERT, required there) and match the host name.
    - Elsewhere ``ssl_mode`` (POSTGRES_SSL_MODE) applies as configured; the default "prefer" lets a local
      non-TLS Postgres work.

    Host and password: in staging and production ``host`` (POSTGRES_HOST) must name the database server and
    ``password`` (POSTGRES_PASSWORD) must not be empty; an unset (the ``localhost`` default) or local host, or an
    empty password, is refused by :meth:`validate_for_environment`, which :meth:`get_dsn` and
    :meth:`PostgresClient.connect` call before any connection attempt.

    DSN: :meth:`get_dsn` percent-encodes the user, password and database, so a generated password holding
    ``/``, ``@``, ``#``, ``:`` or ``%`` still connects (MVP-273).
    """

    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="solace")
    user: str = Field(default="solace")
    password: SecretStr = Field(description="PostgreSQL password (required via POSTGRES_PASSWORD env var)")
    min_pool_size: int = Field(default=5, ge=1, le=100)
    max_pool_size: int = Field(default=20, ge=1, le=200)
    command_timeout: float = Field(default=60.0, gt=0)
    statement_cache_size: int = Field(default=100, ge=0)
    max_cached_statement_lifetime: int = Field(default=300, ge=0)
    ssl_mode: str = Field(default="prefer")
    ssl_root_cert: str | None = Field(default=None, description="Path to SSL root certificate for verify-ca/verify-full modes")
    db_schema: str = Field(default="public")
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_", env_file=".env", extra="ignore"
    )

    def get_effective_ssl_mode(self) -> str:
        """The sslmode in use: "verify-full" in staging and production, else ``ssl_mode``."""
        return effective_ssl_mode(self.ssl_mode)

    def validate_for_environment(self, environment: str | None = None) -> None:
        """Refuse settings a strict environment must not connect with (a no-op elsewhere).

        In staging and production ``host`` (each member of a comma-separated list) must be a plain DNS name, IPv4
        address or bracketed IPv6 address that does not name this machine. Unset (the ``localhost`` default),
        blank, a loopback or unspecified address (including inet_aton spellings such as ``127.1``), a
        ``localhost`` name, a unix-socket path, or anything else asyncpg would read differently (percent-encoding,
        '/', '?', '#', '@', an unbracketed ':') raises :class:`DatabaseConfigError` naming POSTGRES_HOST, never a
        value. ``password`` must not be empty (or blank): that raises naming POSTGRES_PASSWORD. When both are
        wrong the one error names both. ``environment`` None reads ENVIRONMENT.
        """
        if not _is_strict_environment(environment):
            return
        problems = []
        if _is_local_host(self.host):
            problems.append(_HOST_ERROR)
        if not self.password.get_secret_value().strip():
            problems.append(_PASSWORD_ERROR)
        if problems:
            raise _config_error("; ".join(problems), operation="validate_settings") from None

    def get_dsn(self) -> str:
        """The ``postgresql://`` URL (after :meth:`validate_for_environment`).

        User, password and database are percent-encoded (``quote(..., safe="")``); asyncpg, libpq and SQLAlchemy
        decode them, so reserved characters in a generated password connect as themselves (MVP-273).
        """
        self.validate_for_environment()
        user = quote(self.user, safe="")
        password = quote(self.password.get_secret_value(), safe="")
        database = quote(self.database, safe="")
        return f"postgresql://{user}:{password}@{self.host}:{self.port}/{database}"

    def get_ssl_context(self) -> ssl.SSLContext | None:
        """The asyncpg ``ssl=`` argument (:func:`build_ssl_context`); None lets asyncpg negotiate ("prefer").

        Raises:
            DatabaseConfigError: In staging/production without a readable POSTGRES_SSL_ROOT_CERT, or for an
                unknown POSTGRES_SSL_MODE elsewhere.
        """
        return build_ssl_context(self.ssl_mode, self.ssl_root_cert)


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

    async def connect(self, max_retries: int = 3, base_delay: float = 1.0) -> None:
        """Initialize connection pool with retry on transient errors.

        Strict environments (staging, production): the settings are validated first (no unset or local host, no
        empty password), and the pool's first connection must pass
        :func:`solace_infrastructure.database.connection_manager.verify_app_role`: neither the session role nor the
        current role is a superuser, has BYPASSRLS, is a member of the owner of a table in ``public``, is a
        member of a superuser or BYPASSRLS role, is a member of a superuser-equivalent predefined role
        (``pg_execute_server_program``, ``pg_read_server_files``, ``pg_write_server_files``) or has REPLICATION,
        and the login was not switched to another role. A refusal is a :class:`DatabaseConfigError`, raised at
        once without a retry; the pool it opened is closed first.
        """
        if self._pool is not None:
            return
        self._settings.validate_for_environment()  # strict: refuse an unset or local host before TLS or a connect
        # Imported here: connection_manager imports this module.
        from .database.connection_manager import check_new_pool_role

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                ssl_context = self._settings.get_ssl_context()
                pool = await asyncpg.create_pool(
                    dsn=self._settings.get_dsn(),
                    min_size=self._settings.min_pool_size,
                    max_size=self._settings.max_pool_size,
                    command_timeout=self._settings.command_timeout,
                    statement_cache_size=self._settings.statement_cache_size,
                    max_cached_statement_lifetime=self._settings.max_cached_statement_lifetime,
                    ssl=ssl_context,
                    init=self._init_connection,
                )
                await check_new_pool_role(pool)  # strict only; closes the pool before raising
                self._pool = pool
                self._connected = True
                logger.info(
                    "postgres_pool_connected",
                    host=self._settings.host,
                    database=self._settings.database,
                    pool_size=self._settings.max_pool_size,
                    ssl_mode=self._settings.get_effective_ssl_mode(),
                )
                return
            except (TimeoutError, asyncpg.PostgresError, OSError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "postgres_connect_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
        raise DatabaseError(
            f"Failed to connect to PostgreSQL after {max_retries} attempts: {last_error}",
            operation="connect",
            cause=last_error,
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

    async def __aenter__(self) -> PostgresClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.disconnect()

    def _ensure_connected(self) -> asyncpg.Pool:
        if not self._pool:
            raise DatabaseError(
                "PostgreSQL client not connected", operation="check_connection"
            )
        return self._pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection from the pool.

        C.2 RLS (P2-6): set the per-request user GUC (``app.current_user_id``) on the
        borrowed connection so Postgres RLS policies enforce per-user row access — even
        when the ``use_connection_pool_manager`` feature flag is OFF and queries route
        through this client instead of :class:`ConnectionPoolManager`. Mirrors
        ``ConnectionPoolManager.acquire`` so RLS cannot be disabled by an infra flag.

        The user id is read from :mod:`solace_common.request_context` (set by the auth
        layer). Session-level (``is_local=false``) so it survives the per-statement
        autocommits within this acquire; reset on release so it never leaks to the next
        borrower of this pooled connection. A connection whose reset fails is terminated
        instead of being handed back with the user still set (MVP-058). Unset context
        (service-to-service, background jobs, anonymous) => no GUC set => RLS returns no
        user rows (safe).
        """
        from solace_common.request_context import get_current_user_id

        pool = self._ensure_connected()
        async with pool.acquire() as conn:
            _rls_uid = get_current_user_id()
            try:
                if _rls_uid:
                    await conn.execute(_SET_RLS_USER_SQL, _rls_uid)
                yield conn
            finally:
                if _rls_uid:
                    await _reset_rls_user(conn)

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
                ) from e

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
                ) from e

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
                ) from e

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
                ) from e

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
                ) from e

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
                ) from e

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
        if not _is_valid_identifier(table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        if not _is_valid_identifier(schema):
            raise ValueError(f"Invalid schema name: {schema}")
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
        """Insert entity and return created record.

        Raises:
            ValueError: If any column name contains invalid characters.
        """
        for column in data.keys():
            if not _is_valid_identifier(column):
                raise ValueError(f"Invalid column name: {column}")
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f"${i+1}" for i in range(len(data)))
        query = f"INSERT INTO {self.qualified_table} ({columns}) VALUES ({placeholders}) RETURNING *"
        return await self._client.fetch_one(query, *data.values())

    async def update(
        self, entity_id: UUID | str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update entity and return updated record.

        Raises:
            ValueError: If any column name contains invalid characters.
        """
        data["updated_at"] = datetime.now(UTC)
        for column in data.keys():
            if not _is_valid_identifier(column):
                raise ValueError(f"Invalid column name: {column}")
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
            query, entity_id, datetime.now(UTC)
        )
        return "UPDATE 1" in result


def _json_encoder(value: Any) -> str:
    return json.dumps(value, default=str)


def _json_decoder(value: str) -> Any:
    return json.loads(value)


def _truncate_query(query: str, max_length: int = 200) -> str:
    """Truncate query for logging purposes."""
    query = " ".join(query.split())
    return query[:max_length] + "..." if len(query) > max_length else query


def _is_valid_identifier(name: str) -> bool:
    """Validate that a string is a safe SQL identifier.

    Delegates to the shared ValidationUtils.is_valid_sql_identifier.
    """
    return ValidationUtils.is_valid_sql_identifier(name)


async def create_postgres_client(
    settings: PostgresSettings | None = None,
) -> PostgresClient:
    """Factory function to create and connect a PostgreSQL client."""
    client = PostgresClient(settings)
    await client.connect()
    return client
