"""WS-14 real-infra fixtures: user-service as the alias package ``user_service_src`` on the IF-19 Postgres.

* ``UserServicePackage.mod(dotted)`` imports ``user_service_src.<dotted>``. The user-service source directory is
  loaded (read only, never edited here) as the alias package ``user_service_src``, so it collides neither with the
  repository-root ``src`` package nor with another service's ``src``.
* ``us`` -> :class:`UserServicePackage`.
* ``superuser_conn`` -> an asyncpg connection to ``pg_migrated.dsn_superuser`` (with the pool's JSON codec). Tests
  read back what the service wrote through it; the service itself never uses it.
* ``cpm_default`` -> ConnectionPoolManager's ``"default"`` pool built from ``POSTGRES_*`` exactly as a service
  builds it, logging in as ``solace_app`` (NOSUPERUSER NOBYPASSRLS), so every repository query runs under RLS.
  ``pg_migrated``'s own ``"default"`` registration is set aside for the test and put back afterwards, so later
  tests in the session still use it.
* ``seed_user`` -> ``async seed_user(**columns) -> UUID``: a users row inserted as the superuser.
* ``build_repo(cls, **kw)`` -> ``cls(**kw)``; a repository that still requires the legacy positional client gets
  one whose ``acquire()`` is ``ConnectionPoolManager.acquire("default")``.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio

from solace_infrastructure.database.connection_manager import ConnectionPoolManager, init_connection

ROOT = Path(__file__).resolve().parents[3]
USER_SERVICE_SRC = ROOT / "services" / "user-service" / "src"
ALIAS = "user_service_src"
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")


class UserServicePackage:
    """user-service's ``src`` directory, loaded once per process as the alias package ``user_service_src``."""

    @staticmethod
    def _load() -> ModuleType:
        package = sys.modules.get(ALIAS)
        if package is not None:
            return package
        spec = importlib.util.spec_from_file_location(
            ALIAS, USER_SERVICE_SRC / "__init__.py", submodule_search_locations=[str(USER_SERVICE_SRC)])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load user-service from {USER_SERVICE_SRC}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[ALIAS] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            sys.modules.pop(ALIAS, None)
            raise
        return package

    @classmethod
    def mod(cls, dotted: str) -> ModuleType:
        """``user_service_src.<dotted>``, e.g. ``mod("infrastructure.postgres_repository")``."""
        cls._load()
        return importlib.import_module(f"{ALIAS}.{dotted}")


@pytest.fixture(scope="session")
def us() -> UserServicePackage:
    return UserServicePackage()


@pytest_asyncio.fixture(loop_scope="session")
async def superuser_conn(pg_migrated: Any) -> AsyncIterator[asyncpg.Connection]:
    conn = await asyncpg.connect(pg_migrated.dsn_superuser, timeout=30)
    try:
        await init_connection(conn)
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture(loop_scope="session")
async def cpm_default(pg_migrated: Any, monkeypatch: pytest.MonkeyPatch,
                      tmp_path: Path) -> AsyncIterator[asyncpg.Pool]:
    parts = urlsplit(pg_migrated.dsn_app)
    monkeypatch.setenv("POSTGRES_HOST", parts.hostname or "")
    monkeypatch.setenv("POSTGRES_PORT", str(parts.port))
    monkeypatch.setenv("POSTGRES_USER", unquote(parts.username or ""))
    monkeypatch.setenv("POSTGRES_PASSWORD", unquote(parts.password or ""))
    monkeypatch.setenv("POSTGRES_DATABASE", unquote(parts.path.lstrip("/")))
    monkeypatch.setenv("POSTGRES_SSL_MODE", "disable")
    monkeypatch.setenv("ENVIRONMENT", "test")
    # pg_migrated's "default" registration is set aside (not closed): close_all_pools() below would otherwise
    # close the session's app_pool under every later test that uses it.
    kept = {store: getattr(ConnectionPoolManager, store).pop("default", None)
            for store in ("_pools", "_pool_configs", "_locks")}
    await ConnectionPoolManager.close_all_pools()
    try:
        with contextlib.chdir(tmp_path):  # POSTGRES_* only: no .env from the working directory
            pool = await ConnectionPoolManager.get_pool("default")
        yield pool
    finally:
        await ConnectionPoolManager.close_all_pools()
        for store, value in kept.items():
            if value is not None:
                getattr(ConnectionPoolManager, store)["default"] = value


@pytest_asyncio.fixture(loop_scope="session")
async def seed_user(superuser_conn: asyncpg.Connection) -> Callable[..., Awaitable[UUID]]:
    async def _seed(**columns: Any) -> UUID:
        now = datetime.now(UTC)
        row: dict[str, Any] = {
            "id": uuid4(), "created_at": now, "updated_at": now, "encryption_key_id": "none",
            "email": f"seed-{uuid4().hex[:12]}@test.invalid", "password_hash": "argon2-synthetic",
            "display_name": "Seed User",
        }
        row.update(columns)
        bad = [name for name in row if not _IDENTIFIER.fullmatch(name)]
        if bad:
            raise ValueError(f"seed_user: not a column name: {bad}")
        names = ", ".join(row)
        params = ", ".join(f"${i}" for i in range(1, len(row) + 1))
        # Column names are checked against _IDENTIFIER above; every value is a bound parameter.
        await superuser_conn.execute(f"INSERT INTO users ({names}) VALUES ({params})", *row.values())  # noqa: S608
        return row["id"]

    return _seed


class _LegacyClient:
    """The positional ``client`` that pre-MVP-034 repositories required, backed by the default pool."""

    def acquire(self) -> Any:
        return ConnectionPoolManager.acquire("default")


def build_repo(cls: type, **kw: Any) -> Any:
    """``cls(**kw)``; on ``TypeError`` (a repository that still requires a client) ``cls(_LegacyClient(), **kw)``."""
    try:
        return cls(**kw)
    except TypeError:
        return cls(_LegacyClient(), **kw)
