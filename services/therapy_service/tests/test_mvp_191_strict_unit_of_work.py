"""MVP-191: a strict environment never falls back to in-memory therapy repositories."""
from __future__ import annotations
import sys
import fakeredis.aioredis
import pytest
from services.therapy_service.src.infrastructure.repository import (
    OutcomeMeasureRepository, TechniqueRepository, TherapySessionRepository,
    TreatmentPlanRepository, create_unit_of_work,
)
from services.therapy_service.src.main import TherapyServiceAppSettings, build_therapy_components
from services.therapy_service.tests.runtime_fakes import fake_runtime


@pytest.mark.parametrize("env", ["staging", "production"])
@pytest.mark.parametrize("cls", [TreatmentPlanRepository, TherapySessionRepository,
                                 TechniqueRepository, OutcomeMeasureRepository])
def test_mvp_191_in_memory_repositories_refuse_strict_envs(monkeypatch, env, cls) -> None:
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(RuntimeError, match="In-memory repositories are not allowed"):
        cls()


def test_mvp_191_strict_import_failure_raises_instead_of_memory_fallback(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "solace_infrastructure.database", None)
    with pytest.raises(RuntimeError, match="therapy postgres repositories unavailable"):
        create_unit_of_work(environment="staging")


def test_mvp_191_strict_bare_environment_refuses_memory_fallback(monkeypatch) -> None:
    """A blank service environment under a strict bare ENVIRONMENT is strict too."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setitem(sys.modules, "solace_infrastructure.database", None)
    with pytest.raises(RuntimeError, match="therapy postgres repositories unavailable"):
        create_unit_of_work(environment="development")


def test_mvp_191_strict_explicit_memory_backend_is_refused(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="not allowed in strict"):
        create_unit_of_work("memory", environment="production")


def test_mvp_191_dev_import_failure_is_flagged_memory(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setitem(sys.modules, "solace_infrastructure.database", None)
    assert create_unit_of_work(environment="development").backend == "memory"


def test_mvp_191_test_environment_defaults_to_memory() -> None:
    assert create_unit_of_work(environment="test").backend == "memory"


@pytest.mark.asyncio
async def test_mvp_191_dev_memory_backend_makes_service_not_ready(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setitem(sys.modules, "solace_infrastructure.database", None)
    runtime = fake_runtime(redis=fakeredis.aioredis.FakeRedis(decode_responses=True))
    await build_therapy_components(runtime, TherapyServiceAppSettings(environment="development"))
    check, critical = runtime.readiness_checks["therapy_repository"]
    ok, detail = await check()
    assert critical is True
    assert ok is False and "memory" in detail
