"""REV-32 (integration): the env-consistency guard must actually RUN at service startup.

``ProductionGuards.assert_environment_consistency`` was fully unit-tested but never
invoked at any service startup, so a misnamed/unset ``*_ENVIRONMENT`` var still silently
pinned a service to ``development`` in prod. This drives the memory service's REAL FastAPI
lifespan to prove the guard is wired in (``ProductionGuards.assert_startup`` runs as the
first startup step, before any logging/heavy init) and fails loud on a production mismatch.

The guard is deliberately the first executable statement in the lifespan, right before
``configure_logging(settings)``. We patch ``configure_logging`` to raise a unique sentinel:

- mismatch (ENVIRONMENT=production, service env resolves to development) => the guard raises
  ``ProductionSecurityError`` BEFORE ``configure_logging`` is ever reached (sentinel never
  fires). This is the RED->GREEN proof: with the guard un-wired the sentinel would fire
  instead (no ProductionSecurityError).
- match (both development) => the guard passes and control flows on to ``configure_logging``,
  which raises the sentinel — proving the guard did not block a coherent environment and did
  not itself run heavy startup.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root + src on path (mirrors the other src-dependent suites).
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

from services.memory_service.src import main as memory_main
from solace_infrastructure.runtime import bootstrap as runtime_bootstrap
from solace_security.production_guards import ProductionSecurityError


class _SentinelStartupReached(Exception):
    """Raised by the patched configure_logging to mark 'execution passed the guard'."""


async def _enter_lifespan() -> None:
    """Enter (and immediately exit) the memory service's real lifespan startup."""
    async with memory_main.lifespan(memory_main.app):
        pass


@pytest.mark.asyncio
async def test_memory_startup_raises_on_env_mismatch(monkeypatch):
    """ENVIRONMENT=production but the service env resolves to development => fail loud.

    The guard runs before configure_logging, so the sentinel must NOT fire: a
    ProductionSecurityError is the only acceptable outcome.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("MEMORY_ENVIRONMENT", raising=False)  # -> settings.environment == "development"

    def _boom(_spec):
        raise _SentinelStartupReached

    monkeypatch.setattr(runtime_bootstrap, "configure_service_logging", _boom)

    with pytest.raises(ProductionSecurityError):
        await _enter_lifespan()


@pytest.mark.asyncio
async def test_memory_startup_passes_when_env_matches(monkeypatch):
    """Bare ENVIRONMENT and the service env agree (development) => guard lets startup proceed.

    Proven by observing that control reaches the (patched) configure_logging step just after
    the guard, i.e. the sentinel fires and NO ProductionSecurityError is raised.
    """
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("MEMORY_ENVIRONMENT", "development")

    def _boom(_spec):
        raise _SentinelStartupReached

    monkeypatch.setattr(runtime_bootstrap, "configure_service_logging", _boom)

    with pytest.raises(_SentinelStartupReached):
        await _enter_lifespan()
