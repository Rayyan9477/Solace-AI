"""MVP-027 + IF-01 adoption for personality-service (WS-18.1)."""
from __future__ import annotations

from uuid import uuid4

import solace_events.schemas as canonical
from fastapi.testclient import TestClient

from services.personality_service.src import main as personality_main
from services.personality_service.src.config import reload_config
from services.personality_service.src.events import ProfileUpdatedEvent, to_kafka_event

UNUSED = ("PERSONALITY_REDIS_PASSWORD", "PERSONALITY_DB_PASSWORD")


def test_if01_personality_spec_declares_every_flag_through_w5() -> None:
    spec = personality_main.SPEC
    assert spec.service_name == "personality-service"
    assert (spec.uses_postgres, spec.uses_redis, spec.event_plane) == (True, True, "publish_consume")
    assert (spec.handles_phi, spec.audited, spec.rate_limited, spec.edge) == (True, True, False, False)


def test_if01_personality_main_has_no_local_platform_boilerplate() -> None:
    for name in ("configure_logging", "create_application", "_register_middleware", "_register_exception_handlers"):
        assert not hasattr(personality_main, name), name


def test_mvp_027_config_needs_no_unused_redis_or_db_password(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env file can supply the passwords
    for name in UNUSED:
        monkeypatch.delenv(name, raising=False)
    config = reload_config()
    assert not hasattr(config, "redis") and not hasattr(config, "database")


def test_mvp_027_service_environment_follows_the_deployment(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PERSONALITY_ENVIRONMENT", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "staging")
    assert personality_main.PersonalityServiceAppSettings().environment == "staging"


def test_mvp_027_lifespan_boots_without_personality_secrets(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in UNUSED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(personality_main, "SETTINGS",
                        personality_main.SETTINGS.model_copy(update={"events_enabled": False}))
    with TestClient(personality_main.app) as client:
        assert client.get("/live").status_code == 200
        checks = client.get("/ready").json()["checks"]
    assert checks["personality_orchestrator"]["ok"] is True


def test_mvp_270_personality_events_use_the_single_import_root() -> None:
    event = ProfileUpdatedEvent.create(uuid4(), uuid4(), assessment_count=1, version=2)
    assert isinstance(to_kafka_event(event), canonical.PersonalityProfileUpdatedEvent)


def test_mvp_027_strict_environment_refuses_events_disabled(monkeypatch) -> None:
    import asyncio
    from contextlib import asynccontextmanager

    import pytest
    from solace_infrastructure.runtime import StartupDependencyError

    @asynccontextmanager
    async def _stub_runtime(app, spec):
        yield object()  # the guard fires before the runtime is touched

    monkeypatch.setattr(personality_main, "service_runtime", _stub_runtime)
    monkeypatch.setattr(
        personality_main, "SETTINGS",
        personality_main.SETTINGS.model_copy(update={"environment": "staging", "events_enabled": False}),
    )

    async def _enter() -> None:
        async with personality_main.lifespan(personality_main.app):
            pass

    with pytest.raises(StartupDependencyError) as excinfo:
        asyncio.run(_enter())
    assert excinfo.value.component == "events"
