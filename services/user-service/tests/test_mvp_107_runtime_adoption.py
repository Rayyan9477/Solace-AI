"""MVP-107 / MVP-328: user-service runs on the IF-01 runtime; strict boots never fall back to in-memory identity stores."""
from __future__ import annotations

import dataclasses
import pathlib
from datetime import datetime, timezone
from uuid import uuid4

import fakeredis.aioredis
import pytest
import structlog
from fastapi import FastAPI

from solace_infrastructure.runtime import StartupDependencyError, create_service_app
import src.main as main
from src.config import ServiceConfig, UserServiceSettings
from src.events import UserDeletedEvent, to_kafka_event
from runtime_fakes import fake_runtime

MAIN_SRC = pathlib.Path(main.__file__).read_text(encoding="utf-8")


def test_mvp_107_spec_declares_every_platform_dependency() -> None:
    spec = main.SPEC
    assert spec.service_name == "user-service" and spec.event_plane == "publish_consume"
    assert (spec.uses_postgres, spec.uses_redis, spec.handles_phi, spec.audited, spec.rate_limited, spec.edge) == (
        True, True, True, True, True, True)
    assert spec.docs_enabled is None


def test_mvp_107_main_builds_no_private_datastore_clients_or_handlers() -> None:
    for needle in ("PostgresClient", "aioredis.from_url", "ProductionGuards", "structlog.configure(",
                   "exception_handler", "USER_DB_"):
        assert needle not in MAIN_SRC, needle


def test_mvp_107_settings_no_longer_carry_private_datastores() -> None:
    assert not {"database", "redis"} & set(UserServiceSettings.model_fields)
    assert ServiceConfig(env="test").env == "test"


@pytest.mark.asyncio
async def test_mvp_107_strict_boot_without_redis_is_fatal() -> None:
    with pytest.raises(StartupDependencyError):
        await main.build_user_service_state(fake_runtime(redis=None, environment="production"), FastAPI())


@pytest.mark.asyncio
async def test_mvp_107_one_runtime_redis_client_backs_blacklist_and_sessions() -> None:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    state = await main.build_user_service_state(fake_runtime(redis=client), FastAPI())
    assert state.jwt_service._blacklist._redis is client
    assert state.session_manager._redis is client


@pytest.mark.asyncio
async def test_mvp_107_dev_without_redis_warns_and_uses_memory() -> None:
    with structlog.testing.capture_logs() as logs:
        state = await main.build_user_service_state(fake_runtime(redis=None, environment="development"), FastAPI())
    assert type(state.session_manager).__name__ == "SessionManager"
    assert any(e["event"] == "identity_redis_unavailable" and e["log_level"] == "warning" for e in logs)


def test_mvp_328_status_is_gone_and_docs_follow_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # An edge service in strict needs an explicit CORS origin list (IF-01) before the app exists.
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.test")
    paths = {getattr(r, "path", None) for r in main.app.routes}
    assert "/status" not in paths
    assert {"/live", "/ready", "/health", "/metrics", "/.well-known/jwks.json"} <= paths
    strict = create_service_app(dataclasses.replace(main.SPEC, environment="production"),
                                lifespan=main.lifespan, routers=main.ROUTERS, title="probe")
    assert not {"/docs", "/redoc", "/openapi.json", "/status"} & {getattr(r, "path", None) for r in strict.routes}


def test_mvp_107_domain_events_map_to_the_canonical_schema() -> None:
    from solace_events.schemas import UserDeletedKafkaEvent

    event = to_kafka_event(UserDeletedEvent(aggregate_id=uuid4(), deleted_at=datetime.now(timezone.utc),
                                            reason="user_request"))
    assert isinstance(event, UserDeletedKafkaEvent)


def _lifespan_on_fake_runtime(monkeypatch: pytest.MonkeyPatch, runtime, bridge_error: Exception) -> None:
    """Run main.lifespan on ``runtime`` with an event bridge whose start raises ``bridge_error``."""
    from contextlib import asynccontextmanager

    import src.infrastructure.event_bridge as event_bridge

    @asynccontextmanager
    async def _fake_service_runtime(app, spec):
        yield runtime

    async def _failing_bridge(**_kwargs):
        raise bridge_error

    monkeypatch.setattr(main, "service_runtime", _fake_service_runtime)
    monkeypatch.setattr(event_bridge, "initialize_event_bridge", _failing_bridge)


@pytest.mark.asyncio
async def test_mvp_107_strict_event_bridge_failure_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(main, "SPEC", dataclasses.replace(main.SPEC, environment="production"))
    _lifespan_on_fake_runtime(monkeypatch, fake_runtime(redis=client, environment="production"),
                              ConnectionError("broker down"))
    with pytest.raises(ConnectionError):
        async with main.lifespan(FastAPI()):
            pytest.fail("a strict boot must not serve without the user event bridge")


@pytest.mark.asyncio
async def test_mvp_107_dev_event_bridge_failure_boots_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = fake_runtime(redis=None, environment="development")
    _lifespan_on_fake_runtime(monkeypatch, runtime, ConnectionError("broker down"))
    async with main.lifespan(FastAPI()):
        check, critical = runtime.readiness_checks["event_bridge"]
        assert critical is True and await check() == (False, "unavailable")
        assert "user_event_bridge" not in runtime.shutdowns
    assert "user_service_state" in runtime.shutdowns
