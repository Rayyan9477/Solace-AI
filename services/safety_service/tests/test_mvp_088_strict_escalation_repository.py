"""MVP-088: strict environments never keep crisis escalation records in process memory."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from services.safety_service.src.domain.escalation import (
    InMemoryEscalationRepository,
    PostgresEscalationRepository,
)
from solace_infrastructure.runtime import StartupDependencyError


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_088_strict_boot_without_postgres_fails(env, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("SAFETY_ENVIRONMENT", env)
    monkeypatch.setattr("solace_security.production_guards.ProductionGuards.assert_startup",
                        staticmethod(lambda *a, **k: None))

    async def _unreachable(*args, **kwargs):
        raise ConnectionError("postgres unreachable")

    monkeypatch.setattr(
        "solace_infrastructure.database.connection_manager.ConnectionPoolManager.get_pool", _unreachable)
    import services.safety_service.src.main as main
    try:
        main = importlib.reload(main)
        with pytest.raises(StartupDependencyError):
            with TestClient(main.app):
                pass
    finally:
        monkeypatch.undo()
        importlib.reload(main)


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_088_strict_repository_requires_pool(env):
    from services.safety_service.src.domain.factory import build_escalation_repository
    with pytest.raises(StartupDependencyError):
        build_escalation_repository(None, environment=env)


def test_mvp_088_repository_is_postgres_whenever_a_pool_exists():
    from services.safety_service.src.domain.factory import build_escalation_repository
    assert isinstance(build_escalation_repository(object(), environment="staging"), PostgresEscalationRepository)
    assert isinstance(build_escalation_repository(object(), environment="development"), PostgresEscalationRepository)


def test_mvp_088_dev_without_pool_falls_back_in_memory():
    from services.safety_service.src.domain.factory import build_escalation_repository
    assert isinstance(build_escalation_repository(None, environment="test"), InMemoryEscalationRepository)


def test_mvp_088_spec_declares_every_platform_flag():
    import services.safety_service.src.main as main
    assert (main.SPEC.service_name, main.SPEC.uses_postgres, main.SPEC.event_plane) == (
        "safety-service", True, "publish_consume")
    assert main.SPEC.handles_phi is True and main.SPEC.audited is True and main.SPEC.edge is False


# --------------------------------------------------------------------------------------------
# WS-12.1 crisis path (controller requirement): after IF-01 adoption the safety event bridge and
# its outbox poller still start in the lifespan, a crisis check through the adopted app lands in
# the outbox owned by "safety-service" and is drained to the broker, and the bridge, publisher and
# service stops run as runtime shutdown closers.
# --------------------------------------------------------------------------------------------

_CRISIS_EVENT_TYPE = "safety.crisis.detected"
_SAFETY_SOURCE = "safety-service"  # KafkaEventBridge -> create_publisher(source_service=...)


def _wait_until(predicate, timeout: float = 10.0):
    """Poll ``predicate`` from the test thread while the TestClient loop runs in its own thread."""
    import time

    deadline = time.monotonic() + timeout
    value = predicate()
    while not value and time.monotonic() < deadline:
        time.sleep(0.05)
        value = predicate()
    return value


@pytest.fixture
def adopted_dev_app(monkeypatch):
    """The real safety app in a non-strict environment, with no Postgres or Redis reachable.

    The runtime degrades both datastores (``/ready`` 503), so ``runtime.pg_pool`` is None and the
    bridge's outbox is the in-memory store scoped to ``safety-service``. The module-level event
    publisher and bridge singletons are reset so this lifespan builds them on its own event loop.
    """
    monkeypatch.setenv("SAFETY_ENVIRONMENT", "development")
    monkeypatch.setenv("SAFETY_KAFKA_ENABLED", "true")
    monkeypatch.setenv("SAFETY_KAFKA_USE_MOCK", "true")
    monkeypatch.delenv("REDIS_URL", raising=False)

    async def _unreachable(*args, **kwargs):
        raise ConnectionError("postgres unreachable")

    monkeypatch.setattr(
        "solace_infrastructure.database.connection_manager.ConnectionPoolManager.get_pool", _unreachable)
    import services.safety_service.src.events as events
    import services.safety_service.src.infrastructure.event_bridge as event_bridge
    monkeypatch.setattr(events, "_publisher", None)
    monkeypatch.setattr(event_bridge, "_notification_bridge", None)

    import services.safety_service.src.main as main
    main = importlib.reload(main)
    from solace_security.middleware import AuthenticatedService, get_current_service
    main.app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="orchestrator-service", permissions=["safety:read", "safety:write"])
    try:
        yield main
    finally:
        main.app.dependency_overrides.clear()
        monkeypatch.undo()
        importlib.reload(main)


def test_ws_12_1_crisis_check_in_the_adopted_app_reaches_the_safety_outbox(adopted_dev_app):
    from uuid import uuid4

    main = adopted_dev_app
    with TestClient(main.app) as client:
        runtime = main.app.state.runtime
        closers = [name for name, _ in runtime._shutdown_closers]
        for name in ("safety_service", "event_publisher", "notification_bridge"):
            assert name in closers, f"{name} stop is not a runtime shutdown closer: {closers}"
        # Stop order (reverse registration): bridge, publisher, service, then the runtime's own.
        assert closers.index("notification_bridge") > closers.index("event_publisher") > closers.index(
            "safety_service")

        bridge = main.app.state.notification_bridge
        kafka_bridge = bridge._bridge
        assert kafka_bridge._started is True and kafka_bridge._poller is not None, (
            "the lifespan did not start the event bridge and its outbox poller")

        user_id = uuid4()
        response = client.post("/api/v1/safety/check", json={
            "user_id": str(user_id), "content": "I want to kill myself", "check_type": "pre_check"})
        assert response.status_code == 200, response.text
        assert response.json()["is_safe"] is False

        store = kafka_bridge._publisher._outbox_store
        crisis_rows = _wait_until(lambda: [
            r for r in list(store._records.values())
            if r.event_type == _CRISIS_EVENT_TYPE and r.aggregate_id == user_id])
        assert crisis_rows, "a crisis was detected but no safety.crisis.detected outbox row was written"
        assert {r.source_service for r in crisis_rows} == {_SAFETY_SOURCE}

        producer = kafka_bridge._publisher._producer
        drained = _wait_until(lambda: [
            m for m in producer.get_messages() if m[2].get("event_type") == _CRISIS_EVENT_TYPE])
        assert drained, "the outbox poller did not drain the crisis row to the broker"

    # The lifespan exited: the runtime closers stopped the bridge and its poller.
    assert kafka_bridge._started is False and kafka_bridge._poller is None


def _strict_case(bare: str, service: str, case_id: str):
    return pytest.param((bare, service), id=case_id)


# (bare ENVIRONMENT, SAFETY_ENVIRONMENT). The runtime is strict when EITHER is strict, so safety's
# own strictness decisions must be too: a blank service environment under a strict bare one is
# strict (WS121F1-CLO-1).
STRICT_CASES = [
    _strict_case("staging", "staging", "staging"),
    _strict_case("production", "production", "production"),
    _strict_case("production", "", "production-blank-service-env"),
]


@pytest.fixture
def strict_composition(monkeypatch, request):
    """The real lifespan composition in a strict environment, on a stand-in runtime.

    The runtime is replaced by a stand-in that has a pool and records closers, so the real
    lifespan composition runs without Postgres, Redis or Kafka. ``request.param`` is a
    ``(bare ENVIRONMENT, SAFETY_ENVIRONMENT)`` pair (default production for both). Yields a
    namespace with ``main``, ``event_bridge``, the closers that ran (``stopped``, in order)
    and the readiness checks safety registered (``checks``).
    """
    import secrets
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    bare_env, service_env = getattr(request, "param", ("production", "production"))
    monkeypatch.setenv("ENVIRONMENT", bare_env)
    monkeypatch.setenv("SAFETY_ENVIRONMENT", service_env)
    monkeypatch.setenv("SAFETY_KAFKA_ENABLED", "true")
    monkeypatch.delenv("SAFETY_KAFKA_USE_MOCK", raising=False)
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", secrets.token_hex(32))
    monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", secrets.token_urlsafe(32))
    monkeypatch.setattr("solace_infrastructure.database.base_models.configure_phi_encryption",
                        lambda field_encryptor: None)
    import services.safety_service.src.events as events
    import services.safety_service.src.infrastructure.event_bridge as event_bridge
    import services.safety_service.src.main as main
    monkeypatch.setattr(events, "_publisher", None)

    stopped: list[str] = []
    checks: dict[str, bool] = {}

    class _Runtime:
        pg_pool = object()

        def __init__(self) -> None:
            self.closers: list = []

        def register_shutdown(self, name, closer) -> None:
            self.closers.append((name, closer))

        def register_readiness_check(self, name, check, *, critical=True) -> None:
            checks[name] = critical

    @asynccontextmanager
    async def _service_runtime(app, spec):
        runtime = _Runtime()
        try:
            yield runtime
        finally:
            for name, closer in reversed(runtime.closers):
                await closer()
                stopped.append(name)

    monkeypatch.setattr(main, "service_runtime", _service_runtime)
    yield SimpleNamespace(main=main, event_bridge=event_bridge, stopped=stopped, checks=checks)


@pytest.mark.parametrize("strict_composition", STRICT_CASES[1:], indirect=True)
def test_ws_12_1_strict_bridge_failure_stops_the_boot(strict_composition, monkeypatch):
    """Strict: a crisis event bridge that cannot start fails the boot (no blanket except).

    The closers registered before the failure (publisher, service) still run, and no
    readiness check papers over the failure.
    """
    class BrokerDownError(ConnectionError):
        pass

    async def _broken(**kwargs):
        raise BrokerDownError("broker unreachable")

    monkeypatch.setattr(strict_composition.event_bridge, "initialize_notification_bridge", _broken)
    main = strict_composition.main
    with pytest.raises(BrokerDownError):
        with TestClient(main.app):
            pass
    assert strict_composition.stopped == ["event_publisher", "safety_service"]
    assert main.EVENT_BRIDGE_CHECK not in strict_composition.checks


@pytest.mark.parametrize("strict_composition", STRICT_CASES, indirect=True)
def test_ws_12_1_strict_mock_producer_fails_the_boot(strict_composition, monkeypatch):
    """WS121-CRI-1: a strict environment never runs the crisis bridge on a mock producer.

    A mock producer "publishes" crisis rows to an in-process list and marks them PUBLISHED, so
    no human is paged and the rows are never redelivered. The boot must fail before the bridge
    starts; the closers registered so far still run.
    """
    monkeypatch.setenv("SAFETY_KAFKA_USE_MOCK", "true")
    bridge_starts: list[dict] = []

    async def _record(**kwargs):
        bridge_starts.append(kwargs)
        raise AssertionError("the crisis bridge must not start on a mock producer in strict")

    monkeypatch.setattr(strict_composition.event_bridge, "initialize_notification_bridge", _record)
    main = strict_composition.main
    with pytest.raises(StartupDependencyError, match="mock Kafka producer is not allowed"):
        with TestClient(main.app):
            pass
    assert bridge_starts == []
    assert strict_composition.stopped == ["event_publisher", "safety_service"]
    assert main.EVENT_BRIDGE_CHECK not in strict_composition.checks


@pytest.mark.parametrize("strict_composition", STRICT_CASES[1:], indirect=True)
def test_ws_12_1_strict_phi_encryption_failure_stops_the_boot(strict_composition, monkeypatch):
    """Strict: PHI encryption that cannot be activated fails the boot, never a warning."""
    def _no_encryptor(field_encryptor):
        raise RuntimeError("encryptor unavailable")

    monkeypatch.setattr("solace_infrastructure.database.base_models.configure_phi_encryption",
                        _no_encryptor)
    bridge_starts: list[dict] = []

    async def _record(**kwargs):
        bridge_starts.append(kwargs)
        raise AssertionError("the boot must stop at PHI encryption, before the bridge")

    monkeypatch.setattr(strict_composition.event_bridge, "initialize_notification_bridge", _record)
    with pytest.raises(RuntimeError, match="PHI encryption is required"):
        with TestClient(strict_composition.main.app):
            pass
    assert bridge_starts == []
    assert strict_composition.stopped == ["event_publisher", "safety_service"]


def test_mvp_088_blank_service_env_under_strict_bare_env_requires_pool(monkeypatch):
    """WS121F1-CLO-1: the factory is strict when the bare ENVIRONMENT is, like the runtime."""
    from services.safety_service.src.domain.factory import build_escalation_repository
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(StartupDependencyError):
        build_escalation_repository(None, environment="")


def test_ws_12_1_dev_bridge_failure_makes_the_service_unready(adopted_dev_app, monkeypatch):
    import services.safety_service.src.infrastructure.event_bridge as event_bridge

    async def _broken(*args, **kwargs):
        raise ConnectionError("broker unreachable at kafka://user:secret@broker:9092")

    monkeypatch.setattr(event_bridge, "initialize_notification_bridge", _broken)
    main = adopted_dev_app
    with TestClient(main.app) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        check = response.json()["checks"]["event_bridge"]
        assert check == {"ok": False, "critical": True, "detail": "unavailable"}
        assert "secret" not in response.text
        assert client.get("/live").status_code == 200
