"""Datastore start-up for the service runtime: the Postgres pool and the Redis client (IF-01, MVP-332).

``start_service_runtime`` calls ``start_postgres`` when ``spec.uses_postgres`` and then
``start_redis`` when ``spec.uses_redis``. Each start is wrapped once:

- on success the client is returned, a closer is registered for ``stop_service_runtime`` and
  a critical readiness check (``postgres`` or ``redis``) is registered;
- on failure, ``component_start_failed`` decides. In a strict environment (staging or
  production, from either the spec or the bare ``ENVIRONMENT``) it raises
  ``StartupDependencyError(component, reason)``, so the lifespan fails and uvicorn exits
  non-zero. Elsewhere it logs one WARNING ``runtime_component_degraded`` and registers a
  readiness check that fails, so ``/ready`` answers 503 while the process keeps serving.

Neither a log line nor a ``StartupDependencyError`` ever carries the exception text, which can
hold a DSN, a password or a Redis URL: only exception class names (and fixed text naming a
missing variable) are reported.

Postgres is the shared ``default`` pool of ``ConnectionPoolManager`` (D-02: one ``solace``
database, the ``solace_app`` role), built from ``POSTGRES_*``. In strict environments the
database must also carry the schema revision this release expects (``assert_schema_head``).
Redis is built from ``REDIS_URL`` and must answer ``PING``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import redis.asyncio as redis_asyncio
import structlog

from solace_infrastructure.database.connection_manager import ConnectionPoolManager
from solace_infrastructure.runtime.http import spec_is_strict
from solace_infrastructure.runtime.spec import ServiceRuntimeSpec, StartupDependencyError

if TYPE_CHECKING:  # pragma: no cover - static typing only (bootstrap imports this module)
    import asyncpg

    from solace_infrastructure.runtime.bootstrap import ServiceRuntime

POSTGRES_COMPONENT = "postgres"
REDIS_COMPONENT = "redis"
POSTGRES_POOL_NAME = "default"

# A started datastore's last probe result is reused for this long, and the successful start-up
# connection counts as the first probe. It is shorter than the bootstrap's
# READINESS_REFRESH_SECONDS, so every periodic refresh probes the datastore again, and /ready
# traffic cannot probe a datastore more often than this.
DATASTORE_PROBE_INTERVAL_SECONDS = 5.0

# Redis client socket timeouts, and the bound on the start-up PING.
REDIS_SOCKET_TIMEOUT_SECONDS = 5.0
REDIS_STARTUP_TIMEOUT_SECONDS = 10.0

_CONNECTED = "connected"

_log = structlog.get_logger("solace_infrastructure.runtime.datastores")

DatastoreProbe = Callable[[], Awaitable[None]]


class DatastoreSettingMissing(RuntimeError):  # noqa: N818  (reads as the condition it reports)
    """A variable a datastore needs is not set. The message names the variable, never a value."""

    def __init__(self, setting: str) -> None:
        self.setting = setting
        super().__init__(f"{setting} is not set")


class _DatastoreCheck:
    """Readiness check for a started datastore.

    It runs ``probe`` again only when the last result is older than
    ``DATASTORE_PROBE_INTERVAL_SECONDS``; the start-up connection counts as the first probe.
    A failed probe reports the exception class name. A probe cut off by the readiness timeout
    records nothing, so the next evaluation probes again.
    """

    def __init__(self, probe: DatastoreProbe) -> None:
        self._probe = probe
        self._result: tuple[bool, str] = (True, _CONNECTED)
        self._checked_at = time.monotonic()

    async def __call__(self) -> tuple[bool, str]:
        if time.monotonic() - self._checked_at < DATASTORE_PROBE_INTERVAL_SECONDS:
            return self._result
        try:
            await self._probe()
        except Exception as exc:  # the class name only: the text may hold a DSN or secret
            self._result = (False, type(exc).__name__)
        else:
            self._result = (True, _CONNECTED)
        self._checked_at = time.monotonic()
        return self._result


def _failure_fields(exc: BaseException) -> dict[str, str]:
    """Class names and the raise site of a failure, never its text."""
    fields = {"error": type(exc).__name__}
    cause = exc.__cause__
    if cause is None and not exc.__suppress_context__:
        cause = exc.__context__
    if cause is not None:
        fields["cause"] = type(cause).__name__
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        frame = frames[-1]
        fields["origin"] = f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}"
    if isinstance(exc, DatastoreSettingMissing):
        fields["setting"] = exc.setting
    return fields


def component_start_failed(runtime: ServiceRuntime, component: str, exc: BaseException) -> None:
    """Handle a runtime component that failed to start: fatal in strict, degraded elsewhere.

    Strict: logs one ERROR ``runtime_component_failed`` and raises
    ``StartupDependencyError(component, reason)``. ``reason`` is the exception class name, or
    for ``DatastoreSettingMissing`` its fixed text (for example "REDIS_URL is not set"). The
    original exception is not chained, because its text can carry a DSN or password.

    Otherwise: logs one WARNING ``runtime_component_degraded`` and registers a critical
    readiness check named ``component`` that reports ``(False, <exception class name>)``.

    Both lines carry ``component``, ``error`` (class name), ``cause`` (class name, when there is
    one), ``origin`` (file:line:function of the raise) and, for a missing setting, ``setting``.
    """
    fields = _failure_fields(exc)
    if spec_is_strict(runtime.spec):
        _log.error("runtime_component_failed", component=component, **fields)
        reason = str(exc) if isinstance(exc, DatastoreSettingMissing) else fields["error"]
        raise StartupDependencyError(component, reason) from None
    _log.warning("runtime_component_degraded", component=component, **fields)
    error = fields["error"]

    async def _failed() -> tuple[bool, str]:
        return False, error

    runtime.register_readiness_check(component, _failed, critical=True)


async def _schema_head_check(pool: Any) -> None:
    """Raise unless the database carries the schema revision this release expects.

    Called only in strict environments, so any mismatch is fatal: ``assert_schema_head``
    raises ``SchemaVersionMismatch`` itself when the bare ``ENVIRONMENT`` is strict, and the
    returned revision is compared again in case only the spec's environment is strict.
    """
    from solace_infrastructure.database.schema_check import (
        SCHEMA_HEAD,
        SchemaVersionMismatch,
        assert_schema_head,
    )

    found = await assert_schema_head(pool)
    if found != SCHEMA_HEAD:
        raise SchemaVersionMismatch(expected=SCHEMA_HEAD, found=found)


async def _close_postgres_pools() -> None:
    """The ``postgres`` closer: close every pool, then always empty the pool registry.

    ``close_all_pools`` drops a pool from ``ConnectionPoolManager``'s process-wide registry only
    after its ``close()`` returns. When the runtime's step timeout cuts the close off (asyncpg
    waits for a checked-out connection, and a cancelled close terminates the pool), the closed
    pool would stay registered, and the next lifespan in the same process (TestClient suites,
    in-process restarts) would get it back from ``get_pool``. So the registry is emptied in a
    ``finally``, after terminating any pool the cut-off close never reached.
    """
    try:
        await ConnectionPoolManager.close_all_pools()
    finally:
        _forget_postgres_pools()


def _forget_postgres_pools() -> None:
    """Terminate whatever is still registered and empty the registry (pools, configs, locks)."""
    manager = ConnectionPoolManager
    for pool in list(manager._pools.values()):
        terminate = getattr(pool, "terminate", None)
        if callable(terminate):
            # asyncpg's terminate() returns at once for a pool that is already closed.
            with contextlib.suppress(Exception):
                terminate()
    manager._pools.clear()
    manager._pool_configs.clear()
    manager._locks.clear()


async def start_postgres(spec: ServiceRuntimeSpec, runtime: ServiceRuntime) -> asyncpg.Pool | None:
    """Open the shared ``default`` pool; in strict, also check the schema head.

    Registers ``ConnectionPoolManager.close_all_pools`` as the ``postgres`` closer before the
    attempt, so every pool the process opened is closed at shutdown, including after a failed
    start. Returns the pool, or None when the start failed outside strict.

    Raises:
        StartupDependencyError: in strict, when the pool cannot be opened or the schema
            revision differs (component ``postgres``).
    """
    runtime.register_shutdown(POSTGRES_COMPONENT, _close_postgres_pools)
    try:
        pool = await ConnectionPoolManager.get_pool(POSTGRES_POOL_NAME)
        if spec_is_strict(spec):
            await _schema_head_check(pool)
    except Exception as exc:
        failure: Exception = exc
    else:

        async def _probe() -> None:
            await pool.fetchval("SELECT 1")

        runtime.register_readiness_check(POSTGRES_COMPONENT, _DatastoreCheck(_probe), critical=True)
        return pool
    # Outside the except block, so the StartupDependencyError raised in strict has no context.
    component_start_failed(runtime, POSTGRES_COMPONENT, failure)
    return None


async def _ping(client: Any) -> None:
    if not await client.ping():
        raise ConnectionError("redis did not answer PING")


async def start_redis(spec: ServiceRuntimeSpec, runtime: ServiceRuntime) -> Any:
    """Build the Redis client from ``REDIS_URL`` and check that it answers ``PING``.

    The URL is never logged. On success ``client.aclose`` is registered as the ``redis``
    closer. Returns the ``redis.asyncio.Redis`` client, or None when the start failed outside
    strict (a client that was built is closed first).

    Raises:
        StartupDependencyError: in strict, when ``REDIS_URL`` is unset or Redis does not answer
            (component ``redis``).
    """
    url = os.environ.get("REDIS_URL", "").strip()
    client: Any = None
    try:
        if not url:
            raise DatastoreSettingMissing("REDIS_URL")
        client = redis_asyncio.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
        )
        await asyncio.wait_for(_ping(client), timeout=REDIS_STARTUP_TIMEOUT_SECONDS)
    except Exception as exc:
        failure: Exception = exc
    else:
        runtime.register_shutdown(REDIS_COMPONENT, client.aclose)
        runtime.register_readiness_check(
            REDIS_COMPONENT, _DatastoreCheck(lambda: _ping(client)), critical=True
        )
        return client
    if client is not None:
        with contextlib.suppress(Exception):
            await client.aclose()
    component_start_failed(runtime, REDIS_COMPONENT, failure)
    return None


__all__ = [
    "DATASTORE_PROBE_INTERVAL_SECONDS",
    "POSTGRES_COMPONENT",
    "POSTGRES_POOL_NAME",
    "REDIS_COMPONENT",
    "REDIS_SOCKET_TIMEOUT_SECONDS",
    "REDIS_STARTUP_TIMEOUT_SECONDS",
    "DatastoreSettingMissing",
    "component_start_failed",
    "start_postgres",
    "start_redis",
]
