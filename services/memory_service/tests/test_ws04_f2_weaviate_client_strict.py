"""WS-04.F2 (W0-close review SEC-4): a missing weaviate-client package is fatal in staging as well as production.
Strict = the service's own MEMORY_ENVIRONMENT OR the bare ENVIRONMENT (D-14); the check used to compare the bare
ENVIRONMENT with 'production' alone. The refusal reaches the boot through MemoryService.initialize (LA-SEC-1)."""
from __future__ import annotations

import sys

import pytest

from services.memory_service.src.domain.service import MemoryService
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


# W0-close review LA-SEC-1: the refusal must reach the boot. MemoryService.initialize (the only caller, run by
# main.lifespan) used to catch every Weaviate error, log weaviate_init_failed and finish initialized.
STRICT = [("staging", None), ("production", None), ("development", "staging"), ("test", "production")]


class _FailingRepo:
    """A Weaviate repository whose initialize fails with an error the repository itself chose to raise."""

    async def initialize(self) -> None:
        raise RuntimeError("weaviate unusable")


def _env(monkeypatch, *, bare: str, own: str | None) -> None:
    monkeypatch.setenv("ENVIRONMENT", bare)
    if own is None:
        monkeypatch.delenv("MEMORY_ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("MEMORY_ENVIRONMENT", own)


@pytest.mark.parametrize(("bare", "own"), STRICT)
async def test_missing_client_refuses_the_memory_service_boot_in_strict(monkeypatch, bare, own):
    service = MemoryService(weaviate_repo=_repo(monkeypatch, bare=bare, own=own))
    with pytest.raises(RuntimeError, match="weaviate-client package required in staging and production"):
        await service.initialize()
    assert service._initialized is False


@pytest.mark.parametrize(("bare", "own"), STRICT)
async def test_a_weaviate_error_is_not_swallowed_in_strict(monkeypatch, bare, own):
    _env(monkeypatch, bare=bare, own=own)
    service = MemoryService(weaviate_repo=_FailingRepo())
    with pytest.raises(RuntimeError, match="weaviate unusable"):
        await service.initialize()
    assert service._initialized is False


@pytest.mark.parametrize("bare", ["development", "test"])
async def test_missing_client_still_boots_the_memory_service_outside_strict(monkeypatch, bare):
    service = MemoryService(weaviate_repo=_repo(monkeypatch, bare=bare, own=bare))
    await service.initialize()
    assert service._initialized is True


@pytest.mark.parametrize("bare", ["development", "test"])
async def test_a_weaviate_error_degrades_to_keyword_search_outside_strict(monkeypatch, bare):
    _env(monkeypatch, bare=bare, own=bare)
    service = MemoryService(weaviate_repo=_FailingRepo())
    await service.initialize()
    assert service._initialized is True
    assert service._weaviate_repo is None
