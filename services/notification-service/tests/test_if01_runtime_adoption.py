"""IF-01 adoption for notification-service (WS-13.2); MVP-155 lifespan coverage and the strict guard."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.main as main


def test_if01_notification_spec_declares_every_flag_through_w5() -> None:
    spec = main.SPEC
    assert spec.service_name == "notification-service"
    assert (spec.uses_postgres, spec.uses_redis, spec.event_plane) == (True, True, "publish_consume")
    assert (spec.handles_phi, spec.audited, spec.rate_limited, spec.edge) == (False, True, False, False)


def test_if01_main_has_no_local_platform_boilerplate() -> None:
    for name in ("create_app", "get_notification_service", "_notification_service",
                 "_require_crisis_consumer_in_production"):
        assert not hasattr(main, name), name


def test_mvp_155_real_lifespan_serves_the_runtime_routes(monkeypatch) -> None:
    monkeypatch.setenv("KAFKA_ENABLED", "false")
    with TestClient(main.app) as client:
        assert client.get("/live").json() == {"status": "alive"}
        assert "checks" in client.get("/ready").json()
        assert client.get("/api/v1/notifications/health").status_code == 200
        assert main.app.state.notification_service.get_available_channels()


def _settings(env: str, kafka_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(service=SimpleNamespace(env=env), kafka=SimpleNamespace(enabled=kafka_enabled))


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_155_strict_without_the_crisis_consumer_fails_loud(env: str) -> None:
    with pytest.raises(RuntimeError, match="required in staging and production"):
        main._require_crisis_consumer_in_strict(_settings(env, kafka_enabled=False))


@pytest.mark.parametrize("env", ["development", "test"])
def test_non_strict_without_kafka_is_allowed(env: str) -> None:
    main._require_crisis_consumer_in_strict(_settings(env, kafka_enabled=False))


def test_strict_bare_environment_overrides_a_non_strict_service_env(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(RuntimeError, match="required in staging and production"):
        main._require_crisis_consumer_in_strict(_settings("development", kafka_enabled=False))


@pytest.mark.parametrize("env", ["staging", "production"])
def test_strict_refuses_a_mock_crisis_consumer(env: str) -> None:
    settings = _settings(env, kafka_enabled=True)
    settings.kafka.use_mock = True
    with pytest.raises(RuntimeError, match="mock safety-event consumer"):
        main._require_crisis_consumer_in_strict(settings)


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.checks: dict = {}

    def register_background_task(self, name, task) -> None:
        self.calls.append(("task", name))

    def register_shutdown(self, name, closer) -> None:
        self.calls.append(("shutdown", name))

    def register_readiness_check(self, name, check, *, critical=True) -> None:
        self.calls.append(("ready", name))
        self.checks[name] = (check, critical)


def _kafka_settings(monkeypatch, *, enabled: bool, use_mock: bool = True) -> None:
    monkeypatch.setattr(main.SETTINGS.kafka, "enabled", enabled)
    monkeypatch.setattr(main.SETTINGS.kafka, "use_mock", use_mock)


async def test_safety_consumer_disabled_registers_nothing(monkeypatch, mock_notification_service) -> None:
    _kafka_settings(monkeypatch, enabled=False)
    runtime = _FakeRuntime()
    assert await main._start_safety_consumer(runtime, mock_notification_service) is None
    assert runtime.calls == []


async def test_safety_consumer_registers_task_then_graceful_stop_then_readiness(
    monkeypatch, mock_notification_service
) -> None:
    from src import consumers
    monkeypatch.setattr(consumers, "_safety_consumer", None)
    _kafka_settings(monkeypatch, enabled=True)
    runtime = _FakeRuntime()
    consumer = await main._start_safety_consumer(runtime, mock_notification_service)
    try:
        assert consumer is not None and consumer.is_running
        assert runtime.calls == [("task", "safety_consumer"), ("shutdown", "safety_consumer_stop"),
                                 ("ready", "safety_consumer")]
        check, critical = runtime.checks["safety_consumer"]
        assert critical is True and await check() == (True, "running")
    finally:
        await consumers.shutdown_safety_consumer()
    assert await check() == (False, "stopped")


async def _boom(**_kwargs):
    raise ConnectionError("broker at kafka://secret-host:9092 unreachable")


async def test_safety_consumer_start_failure_outside_strict_is_not_ready(
    monkeypatch, mock_notification_service
) -> None:
    from src import consumers
    _kafka_settings(monkeypatch, enabled=True)
    monkeypatch.setattr(consumers, "initialize_safety_consumer", _boom)
    runtime = _FakeRuntime()
    assert await main._start_safety_consumer(runtime, mock_notification_service) is None
    check, critical = runtime.checks["safety_consumer"]
    assert critical is True and await check() == (False, "unavailable")


async def test_safety_consumer_start_failure_in_strict_fails_boot(monkeypatch, mock_notification_service) -> None:
    from src import consumers
    monkeypatch.setenv("ENVIRONMENT", "staging")
    _kafka_settings(monkeypatch, enabled=True)
    monkeypatch.setattr(consumers, "initialize_safety_consumer", _boom)
    with pytest.raises(ConnectionError):
        await main._start_safety_consumer(_FakeRuntime(), mock_notification_service)


async def test_event_bridge_failure_outside_strict_is_not_ready() -> None:
    runtime = _FakeRuntime()
    runtime.pg_pool = None
    assert await main._start_event_bridge(runtime, _boom) is None
    check, critical = runtime.checks["event_bridge"]
    assert critical is True and await check() == (False, "unavailable")


async def test_event_bridge_failure_in_strict_fails_boot(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    runtime = _FakeRuntime()
    runtime.pg_pool = None
    with pytest.raises(ConnectionError):
        await main._start_event_bridge(runtime, _boom)
