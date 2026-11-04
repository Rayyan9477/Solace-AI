"""WS-15.F1 (W0 finding C-1): a strict therapy boot refuses a mock Kafka producer.

A mock producer "sends" therapy events to an in-process list, so they never reach the broker.
Strict = ``THERAPY_ENVIRONMENT`` OR the bare ``ENVIRONMENT`` names staging or production.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from solace_infrastructure.runtime import StartupDependencyError

from services.therapy_service.tests.runtime_fakes import fake_runtime


def _drive(monkeypatch, *, service_env: str, bare_env: str, use_mock: bool = True):
    import services.therapy_service.src.main as main
    from services.therapy_service.src.infrastructure import event_bridge

    monkeypatch.setenv("ENVIRONMENT", bare_env)
    monkeypatch.setenv("THERAPY_ENVIRONMENT", service_env)
    monkeypatch.setenv("THERAPY_KAFKA_USE_MOCK", "true" if use_mock else "false")
    started: list[bool] = []

    @asynccontextmanager
    async def _runtime(app, spec):
        yield fake_runtime(environment=service_env)

    async def _components(runtime, settings):
        return SimpleNamespace(
            orchestrator=object(), llm_client=object(), event_bus=object(), session_store=None
        )

    async def _bridge(**kwargs):
        started.append(kwargs["use_mock"])
        return object()

    monkeypatch.setattr(main, "service_runtime", _runtime)
    monkeypatch.setattr(main, "build_therapy_components", _components)
    monkeypatch.setattr(event_bridge, "initialize_event_bridge", _bridge)
    return main, started


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_env", "bare_env"),
    [
        ("staging", "staging"),
        ("production", "production"),
        ("development", "staging"),  # dev-looking service env under a strict bare ENVIRONMENT
        ("", "production"),  # blank service env under a strict bare ENVIRONMENT
    ],
)
async def test_strict_boot_refuses_mock_kafka(monkeypatch, service_env, bare_env) -> None:
    main, started = _drive(monkeypatch, service_env=service_env, bare_env=bare_env)
    with pytest.raises(StartupDependencyError) as exc:
        async with main.lifespan(FastAPI()):
            pass
    assert "mock" in str(exc.value).lower()
    assert started == [], "the bridge must not start with a mock producer in strict"


@pytest.mark.asyncio
async def test_dev_boot_with_mock_kafka_still_works(monkeypatch) -> None:
    main, started = _drive(monkeypatch, service_env="development", bare_env="test")
    async with main.lifespan(FastAPI()):
        pass
    assert started == [True]
