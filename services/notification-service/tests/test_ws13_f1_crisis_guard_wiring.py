"""WS-13.F1 (W0-close review CR-1): the strict crisis-consumer guard is wired into the real lifespan, and its allow
path lets a strict boot with a broker-backed consumer start it.

The guard's unit tests (test_if01_runtime_adoption.py) call ``_require_crisis_consumer_in_strict`` directly, so a
lost call in the lifespan (review mutation M1) or a guard that refuses the allowed configuration (mutation M3)
passed every test. These drive ``main.lifespan`` itself, with the runtime bracket, the service factory, the
delivery-event bridge and the consumer start replaced by recorders. Strict = ``NOTIFICATION_SERVICE_ENV`` OR the
bare ``ENVIRONMENT`` (D-14).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

import src.main as main

STRICT = [("production", "production"), ("staging", "staging"), ("development", "production"), ("staging", "development")]


class _FakeRuntime:
    def register_background_task(self, name, task) -> None: ...

    def register_shutdown(self, name, closer) -> None: ...

    def register_readiness_check(self, name, check, *, critical=True) -> None: ...


def _drive(monkeypatch, *, service_env: str, bare_env: str, kafka_enabled: bool, use_mock: bool = False) -> tuple:
    """Patch the lifespan's collaborators; returns (the services the consumer start was called with, the service)."""
    monkeypatch.setenv("ENVIRONMENT", bare_env)
    monkeypatch.setattr(main.SETTINGS.service, "env", service_env)
    monkeypatch.setattr(main.SETTINGS.kafka, "enabled", kafka_enabled)
    monkeypatch.setattr(main.SETTINGS.kafka, "use_mock", use_mock)
    service = object()
    started: list = []

    @asynccontextmanager
    async def _runtime(app, spec):
        yield _FakeRuntime()

    async def _bridge(runtime, initialize):
        return None

    async def _consumer(runtime, notification_service):
        started.append(notification_service)
        return None

    monkeypatch.setattr(main, "service_runtime", _runtime)
    monkeypatch.setattr(main, "_create_notification_service", lambda settings: service)
    monkeypatch.setattr(main, "_start_event_bridge", _bridge)
    monkeypatch.setattr(main, "_start_safety_consumer", _consumer)
    return started, service


async def _boot() -> None:
    async with main.lifespan(FastAPI()):
        pass


@pytest.mark.parametrize(("service_env", "bare_env"), STRICT)
async def test_strict_lifespan_without_kafka_fails_loud_before_the_consumer(monkeypatch, service_env, bare_env):
    started, _ = _drive(monkeypatch, service_env=service_env, bare_env=bare_env, kafka_enabled=False)
    with pytest.raises(RuntimeError, match="required in staging and production"):
        await _boot()
    assert started == []


@pytest.mark.parametrize(("service_env", "bare_env"), STRICT)
async def test_strict_lifespan_refuses_a_mock_consumer(monkeypatch, service_env, bare_env):
    started, _ = _drive(monkeypatch, service_env=service_env, bare_env=bare_env, kafka_enabled=True, use_mock=True)
    with pytest.raises(RuntimeError, match="mock safety-event consumer"):
        await _boot()
    assert started == []


@pytest.mark.parametrize(("service_env", "bare_env"), STRICT)
async def test_strict_lifespan_with_a_broker_consumer_starts_it(monkeypatch, service_env, bare_env):
    started, service = _drive(monkeypatch, service_env=service_env, bare_env=bare_env, kafka_enabled=True, use_mock=False)
    await _boot()
    assert started == [service]


@pytest.mark.parametrize("env", ["development", "test"])
async def test_non_strict_lifespan_without_kafka_still_boots(monkeypatch, env):
    started, service = _drive(monkeypatch, service_env=env, bare_env=env, kafka_enabled=False)
    await _boot()
    assert started == [service]


@pytest.mark.parametrize("env", ["staging", "production"])
def test_strict_guard_allows_a_broker_consumer(env):
    settings = SimpleNamespace(service=SimpleNamespace(env=env), kafka=SimpleNamespace(enabled=True, use_mock=False))
    assert main._require_crisis_consumer_in_strict(settings) is None
