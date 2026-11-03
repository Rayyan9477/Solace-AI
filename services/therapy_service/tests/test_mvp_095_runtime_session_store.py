"""MVP-095 / MVP-208: the therapy session store uses the runtime REDIS_URL client and never builds its own connection."""
from __future__ import annotations
from uuid import uuid4
import fakeredis.aioredis
import pytest
from services.therapy_service.src.infrastructure.redis_session_repository import (
    RedisSessionRepository, RedisSessionRepositorySettings, build_session_store,
)
from services.therapy_service.src.main import SPEC, TherapyServiceAppSettings, build_therapy_components
from services.therapy_service.tests.runtime_fakes import fake_runtime


def test_mvp_095_store_settings_have_no_connection_fields() -> None:
    assert not {"host", "port", "db", "password", "ssl", "socket_timeout"} & set(RedisSessionRepositorySettings.model_fields)


def test_mvp_095_build_session_store_requires_a_client() -> None:
    assert build_session_store(None) is None
    assert build_session_store(fakeredis.aioredis.FakeRedis(decode_responses=True))._initialized is True


@pytest.mark.asyncio
async def test_mvp_095_start_session_persists_through_runtime_redis() -> None:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    components = await build_therapy_components(fake_runtime(redis=client), TherapyServiceAppSettings())
    start = await components.orchestrator.start_session(uuid4(), uuid4(), {})
    assert await client.exists(f"solace:therapy:session:{start.session_id}") == 1


@pytest.mark.asyncio
async def test_mvp_095_dev_without_redis_reports_not_ready() -> None:
    runtime = fake_runtime(redis=None)
    components = await build_therapy_components(runtime, TherapyServiceAppSettings())
    check, critical = runtime.readiness_checks["therapy_session_store"]
    assert components.session_store is None and critical is True
    assert (await check())[0] is False


@pytest.mark.asyncio
async def test_mvp_095_strict_without_redis_fails_startup() -> None:
    from solace_infrastructure.runtime import StartupDependencyError
    with pytest.raises(StartupDependencyError):
        await build_therapy_components(fake_runtime(redis=None, environment="production"),
                                       TherapyServiceAppSettings(environment="production"))


@pytest.mark.asyncio
async def test_mvp_208_store_without_client_cannot_report_an_empty_erase() -> None:
    store = build_session_store(fakeredis.aioredis.FakeRedis(decode_responses=True))
    store._client = None  # simulates a store that never connected
    with pytest.raises(RuntimeError, match="not initialized"):
        await store.delete_user_data(uuid4())


def test_mvp_095_spec_declares_every_platform_dependency() -> None:
    assert SPEC.service_name == "therapy-service"
    assert (SPEC.uses_postgres, SPEC.uses_redis, SPEC.handles_phi, SPEC.audited) == (True, True, True, True)
    assert SPEC.event_plane == "publish_consume" and SPEC.edge is False and SPEC.rate_limited is False

@pytest.mark.asyncio
async def test_ws151_spe1_llm_closer_registered_before_later_steps_can_fail(monkeypatch) -> None:
    import services.therapy_service.src.infrastructure.repository as repo_module

    def _boom(*args, **kwargs):
        raise RuntimeError("unit of work failed")

    monkeypatch.setattr(repo_module, "create_unit_of_work", _boom)
    runtime = fake_runtime(redis=fakeredis.aioredis.FakeRedis(decode_responses=True))
    with pytest.raises(RuntimeError, match="unit of work failed"):
        await build_therapy_components(runtime, TherapyServiceAppSettings())
    assert "therapy_llm_client" in runtime.shutdowns
