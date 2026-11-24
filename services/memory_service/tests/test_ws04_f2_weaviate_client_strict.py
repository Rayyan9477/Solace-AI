"""WS-04.F2 (W0-close review SEC-4): a missing weaviate-client package is fatal in staging as well as production.
Strict = the service's own MEMORY_ENVIRONMENT OR the bare ENVIRONMENT (D-14); the check used to compare the bare
ENVIRONMENT with 'production' alone."""
from __future__ import annotations

import sys

import pytest

from services.memory_service.src.infrastructure.weaviate_repo import WeaviateRepository, WeaviateSettings


def _repo(monkeypatch, *, bare: str, own: str | None) -> WeaviateRepository:
    monkeypatch.setitem(sys.modules, "weaviate", None)   # `import weaviate` raises ImportError
    monkeypatch.setenv("ENVIRONMENT", bare)
    if own is None:
        monkeypatch.delenv("MEMORY_ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("MEMORY_ENVIRONMENT", own)
    return WeaviateRepository(WeaviateSettings(_env_file=None))


@pytest.mark.parametrize(("bare", "own"), [
    ("staging", None), ("production", None), ("development", "staging"), ("test", "production"),
])
async def test_missing_client_is_fatal_in_strict(monkeypatch, bare, own):
    repo = _repo(monkeypatch, bare=bare, own=own)
    with pytest.raises(RuntimeError, match="weaviate-client package required in staging and production"):
        await repo.initialize()


@pytest.mark.parametrize("bare", ["development", "test"])
async def test_missing_client_is_logged_outside_strict(monkeypatch, bare):
    repo = _repo(monkeypatch, bare=bare, own=bare)
    await repo.initialize()
    assert repo._initialized is False
