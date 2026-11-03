"""Shared fixtures and the Docker opt-in gate for the tests/bootstrap suite (WS-04).

Gate: items in this directory marked ``realinfra`` or ``e2e`` are skipped unless
``SOLACE_RUN_DOCKER_TESTS=1`` or ``SOLACE_REQUIRE_REALINFRA=1`` is set. With either set, a
missing ``docker`` CLI fails those items instead of skipping them.

Fixtures:
- ``make_spec``: factory for a development ``ServiceRuntimeSpec`` for ``safety-service``;
  keyword overrides replace any field. The runtime is imported lazily, so this module loads
  before ``solace_infrastructure.runtime`` exists (WS-04.1 lands before WS-04.2).
- ``strict_secrets``: sets random values for the variables strict environments require, plus
  a non-superuser ``POSTGRES_USER`` and ``POSTGRES_SSL_MODE=verify-full``, removes the retired
  and debug variables, and returns the mapping it set.
- ``calls`` and ``fakes`` (WS-04.4): ``fakes`` replaces the guards, logging, the Postgres pool
  manager, the schema-head check and the Redis client with fakes that append to ``calls``, sets
  ``REDIS_URL``, and returns the fake pool. The runtime modules are imported inside the fixture
  body, so collection never breaks before they exist.
"""
from __future__ import annotations

import os
import secrets
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_BOOTSTRAP_DIR = Path(__file__).resolve().parent
_DOCKER_MARKERS = ("realinfra", "e2e")
_OPT_IN_VARS = ("SOLACE_RUN_DOCKER_TESTS", "SOLACE_REQUIRE_REALINFRA")
_DOCKER_MISSING = pytest.StashKey[str]()

_RETIRED_AND_DEBUG_VARS = (
    "SECRET_KEY",
    "ENCRYPTION_KEY",
    "JWT_SECRET",
    "API_KEY",
    "DEBUG",
    "TESTING",
    "MOCK_SERVICES",
    "USE_IN_MEMORY_DB",
)


def _in_bootstrap_dir(item: pytest.Item) -> bool:
    # pytest_collection_modifyitems sees every item in the session, so the gate is
    # limited to this directory and never changes other suites' realinfra handling.
    try:
        Path(item.path).resolve().relative_to(_BOOTSTRAP_DIR)
    except ValueError:
        return False
    return True


def _docker_opted_in() -> bool:
    return any(os.environ.get(name) == "1" for name in _OPT_IN_VARS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gated = [
        item
        for item in items
        if _in_bootstrap_dir(item)
        and any(item.get_closest_marker(marker) is not None for marker in _DOCKER_MARKERS)
    ]
    if not gated:
        return
    if not _docker_opted_in():
        skip = pytest.mark.skip(
            reason="Docker-backed test: set SOLACE_RUN_DOCKER_TESTS=1 or SOLACE_REQUIRE_REALINFRA=1 to run it"
        )
        for item in gated:
            item.add_marker(skip)
        return
    if shutil.which("docker") is None:
        reason = (
            "SOLACE_RUN_DOCKER_TESTS=1 or SOLACE_REQUIRE_REALINFRA=1 is set "
            "but the docker CLI is not on PATH"
        )
        for item in gated:
            item.stash[_DOCKER_MISSING] = reason


def pytest_runtest_setup(item: pytest.Item) -> None:
    reason = item.stash.get(_DOCKER_MISSING, None)
    if reason is not None:
        pytest.fail(reason, pytrace=False)


@pytest.fixture
def make_spec() -> Callable[..., Any]:
    """Return a factory building ``ServiceRuntimeSpec`` with development defaults."""

    def _make(**overrides: Any) -> Any:
        from solace_infrastructure.runtime import ServiceRuntimeSpec

        fields: dict[str, Any] = {"service_name": "safety-service", "environment": "development"}
        fields.update(overrides)
        return ServiceRuntimeSpec(**fields)

    return _make


@pytest.fixture
def strict_secrets(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set what a strict environment needs (random values) and return the mapping."""
    for gone in _RETIRED_AND_DEBUG_VARS:
        monkeypatch.delenv(gone, raising=False)
    values = {
        "AUTH_SECRET_KEY": secrets.token_urlsafe(48),
        "ENCRYPTION_MASTER_KEY": secrets.token_hex(32),
        "ENCRYPTION_SEARCH_HASH_SALT": secrets.token_urlsafe(32),
        "POSTGRES_PASSWORD": secrets.token_urlsafe(24),
        "POSTGRES_USER": "solace_app",
        "POSTGRES_SSL_MODE": "verify-full",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


@pytest.fixture
def calls():
    return []


@pytest.fixture
def fakes(monkeypatch, calls):
    from solace_infrastructure.runtime import bootstrap as bs, datastores as ds
    class Pool:
        pass
    pool = Pool()
    async def get_pool(name="default"):
        calls.append(f"get_pool:{name}")
        return pool
    async def close_all():
        calls.append("pg.close_all")
    async def head(p):
        calls.append("schema_head")
    class Redis:
        async def ping(self):
            calls.append("redis.ping"); return True
        async def aclose(self):
            calls.append("redis.aclose")
    monkeypatch.setattr(ds.ConnectionPoolManager, "get_pool", get_pool)
    monkeypatch.setattr(ds.ConnectionPoolManager, "close_all_pools", close_all)
    monkeypatch.setattr(ds, "_schema_head_check", head)
    monkeypatch.setattr(ds.redis_asyncio, "from_url", lambda url, **kw: Redis())
    monkeypatch.setenv("REDIS_URL", "redis://:p4ss@redis:6379/0")
    monkeypatch.setattr(bs.ProductionGuards, "assert_startup",
                        classmethod(lambda cls, env, **kw: calls.append(("guards", tuple(sorted(kw["required_vars"]))))))
    monkeypatch.setattr(bs, "configure_service_logging", lambda spec: calls.append("logging"))
    return pool
