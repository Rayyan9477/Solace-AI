"""MVP-263 / MVP-095 (diagnosis half): diagnosis adopts IF-01, builds its store from REDIS_URL and honours one set of knobs."""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import fakeredis.aioredis
import httpx
import pytest
import structlog
from services.diagnosis_service.src.infrastructure.redis_session_repository import (
    DiagnosisSessionRedisSettings, build_session_store,
)
from services.diagnosis_service.src.main import (
    SPEC, DiagnosisServiceAppSettings, build_diagnosis_components, create_application,
)
from services.diagnosis_service.tests.runtime_fakes import fake_runtime

SRC = Path(__file__).resolve().parents[1] / "src"


def _redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def test_mvp_263_app_settings_have_no_dead_reasoning_knobs() -> None:
    assert not {"reasoning_timeout_ms", "enable_anti_sycophancy"} & set(DiagnosisServiceAppSettings.model_fields)
    assert not (SRC / "config.py").exists()
    assert not [p.name for p in SRC.rglob("*.py") if "DIAGNOSIS_REASONING_" in p.read_text(encoding="utf-8")]


@pytest.mark.asyncio
async def test_mvp_263_real_knobs_reach_the_service_and_the_log(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # never read the repository's .env
    monkeypatch.setenv("DIAGNOSIS_SERVICE_MAX_REASONING_TIME_MS", "30000")
    monkeypatch.setenv("DIAGNOSIS_SERVICE_ENABLE_ANTI_SYCOPHANCY", "false")
    with structlog.testing.capture_logs() as logs:
        components = await build_diagnosis_components(fake_runtime(redis=_redis()), DiagnosisServiceAppSettings())
    assert components.service._settings.max_reasoning_time_ms == 30000
    assert components.service._settings.enable_anti_sycophancy is False
    built = [e for e in logs if e["event"] == "diagnosis_components_built"]
    assert built and built[0]["anti_sycophancy"] is False and built[0]["max_reasoning_time_ms"] == 30000


def test_mvp_095_store_settings_have_no_connection_fields() -> None:
    assert not {"host", "port", "db", "password", "ssl", "socket_timeout"} & set(DiagnosisSessionRedisSettings.model_fields)
    assert build_session_store(None) is None and build_session_store(_redis()) is not None


@pytest.mark.asyncio
async def test_mvp_095_start_session_persists_through_runtime_redis() -> None:
    client = _redis()
    components = await build_diagnosis_components(fake_runtime(redis=client), DiagnosisServiceAppSettings())
    start = await components.service.start_session(uuid4(), "assessment", {}, None)
    assert await client.exists(f"solace:diagnosis:session:{start.session_id}") == 1


@pytest.mark.asyncio
async def test_mvp_095_dev_without_redis_reports_not_ready() -> None:
    runtime = fake_runtime(redis=None)
    components = await build_diagnosis_components(runtime, DiagnosisServiceAppSettings())
    check, critical = runtime.readiness_checks["diagnosis_session_store"]
    assert components.session_store is None and critical is True and (await check())[0] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_095_strict_without_redis_or_events_fails_startup(env) -> None:
    from solace_infrastructure.runtime import StartupDependencyError
    with pytest.raises(StartupDependencyError):
        await build_diagnosis_components(fake_runtime(redis=None, environment=env), DiagnosisServiceAppSettings(environment=env))
    with pytest.raises(StartupDependencyError):
        await build_diagnosis_components(fake_runtime(redis=_redis(), environment=env),
                                         DiagnosisServiceAppSettings(environment=env, events_enabled=False))


def test_mvp_095_spec_declares_every_platform_dependency() -> None:
    assert SPEC.service_name == "diagnosis-service"
    assert (SPEC.uses_postgres, SPEC.uses_redis, SPEC.handles_phi, SPEC.audited) == (True, True, True, True)
    assert SPEC.event_plane == "publish_consume" and SPEC.edge is False and SPEC.rate_limited is False


@pytest.mark.asyncio
async def test_mvp_263_error_bodies_never_echo_exception_text() -> None:
    app = create_application()

    class _Boom:
        _initialized = True
        async def extract_symptoms(self, *a, **kw):
            raise ValueError("I want to end my life - synthetic")

    from solace_security.auth import TokenType
    from solace_security.middleware import AuthenticatedUser, get_current_user
    uid = str(uuid4())
    app.state.diagnosis_service = _Boom()
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=uid, token_type=TokenType.ACCESS, roles=["user"])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.post("/api/v1/diagnosis/extract-symptoms", json={"user_id": uid, "session_id": str(uuid4()), "message": "x"})
    assert resp.status_code == 400 and "end my life" not in resp.text
