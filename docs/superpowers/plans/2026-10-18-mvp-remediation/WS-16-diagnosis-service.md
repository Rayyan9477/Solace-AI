# WS-16: Diagnosis service: risk signals, phase safety, persistence and clinical logic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnosis boots on the IF-01 runtime, screens every message with the shared lexicon before any LLM call, keeps the CRISIS phase server-side and sticky for the D-06 crisis watch, publishes at most one categories-only `safety.crisis.detected` per turn with the turn id as correlation id, persists sessions and records against the migrated schema through the PHI codec under RLS, and returns differentials that are clinically gated or flagged for review (never user-facing in the pilot).

**Architecture:** `main.py` declares one `ServiceRuntimeSpec` and a testable composition function `build_diagnosis_components(runtime, settings)` that wires the runtime's Redis client, `default` pool, event plane and PHI codec into `DiagnosisService`; nothing platform-level sits in a swallowing try/except. Inside `DiagnosisService.assess` the order is fixed: lexicon pre-screen (`SymptomExtractor.screen_risk`), crisis publish (outbox, once per turn), session load and ownership, then either the crisis reply (no LLM call) or the bounded 4-step chain whose LLM-derived symptoms may only raise the level. Repositories write the Alembic columns of `001`/`020` through `ConnectionPoolManager.acquire/transaction("default")`, so the RLS GUC is always set, encrypt PHI with `solace_security.phi_codec`, and write the completed record and its `diagnosis.completed` outbox row in one transaction.

**Covers:** 29 register entries (blocker 0, high 5, medium 11, low 13). **Waves:** 0, 2, 3, 5. **Depends on:** WS-01 (lock, markers, import-root guard WS-01.7); WS-02 (slot `020_diagnosis` scaffold and 009 NULLIF policies, WS-02.3; `default` pool and the `diagnosis_db` alias, WS-02.6); WS-03 (env registry and compose: `REDIS_URL`, `EVENTS_ENABLED`, `POSTGRES_*`, new `DIAGNOSIS_*` names, retirements); WS-04 (IF-01 runtime WS-04.4, conformance table WS-04.5, event-plane wiring WS-04.6, codec warm-up WS-04.8, audit wiring WS-04.10, `CRISIS_DETECTED` metric object WS-04.3); WS-05 (IF-19 fixtures WS-05.2, IF-20 framework and the diagnosis event contract WS-05.7); WS-06 (IF-05 plane WS-06.8, `publish(conn=)` WS-06.5, late `subscribe` for the erasure consumer also WS-06.8); WS-07 (`acting_as_user` WS-07.1, `require_service_permission`/`find_unguarded_service_routes` WS-07.4, `authorize_subject` WS-07.8); WS-09 (IF-13 codec WS-09.7, IF-15 `LLMUnavailableError`/`readiness` WS-09.3/WS-09.4, IF-14 audit WS-09.11/WS-09.12); WS-11 (orchestrator→diagnosis contract files WS-11.10); WS-12 (IF-07 `solace_clinical` and the phrase set WS-12.2/WS-12.3, D-05 record); WS-14 (IF-16 `register_user_erasure_consumer` WS-14.13); WS-15 (slot 019 runs before slot 020, WS-15.3). **Owned paths:** `services/diagnosis_service/src/{domain,api.py,config.py,events.py,main.py,schemas.py,__init__.py}`, `services/diagnosis_service/src/infrastructure/{postgres_repository.py,redis_session_repository.py,repository.py,__init__.py}`, `services/diagnosis_service/tests`, `services/diagnosis_service/__init__.py`, `migrations/versions/020_diagnosis.py`; new test files under `tests/realinfra/diagnosis-service/`; `tests/integration/test_diagnosis_vignette.py` and the diagnosis class of `tests/integration/test_event_flow.py` as owner of the code under test (shared-files protocol). **Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

**Notation used in this plan.** `Depends on` lists exact task ids inside WS-16 and in written plans; a change requested from a workstream whose plan is not yet written, or a request without a task id, is written `WS-NN (request: ...)` and the plan fixer replaces it with the owner's task id. Skeleton interfaces delivered in an earlier wave, or in phase A of the same wave, are listed under **Interfaces → Consumes** with their owner. Every task except WS-16.2 is phase B (service paths); WS-16.2 is phase A (migrations, W2 schema lane after slot 019). Commands use the standard environment:
- UNIT `<path>` = `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q`
- REAL `<path>` = the same with `$env:SOLACE_REQUIRE_REALINFRA="1";` added before the interpreter, so a missing Docker host fails instead of skipping.

Every command below is written out in full.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/diagnosis_service/src/main.py` | Modify | IF-01 `SPEC`, `DiagnosisComponents`, `build_diagnosis_components`, lifespan, readiness checks, publisher, codec and erasure-consumer wiring | 16.1, 16.4, 16.5, 16.8, 16.9, 16.13 |
| `services/diagnosis_service/src/config.py` | Delete | dead `DIAGNOSIS_DB_*`/`DIAGNOSIS_REDIS_*`/`DIAGNOSIS_REASONING_*` settings | 16.1 |
| `services/diagnosis_service/src/events.py` | Modify | bare `solace_events` import; `build_crisis_detected_event`; categories-only `SafetyFlagRaisedEvent`; fixed `diagnosis.completed` payload; bounded, purgeable dispatcher log | 16.1, 16.5, 16.8, 16.13, 16.19 |
| `migrations/versions/020_diagnosis.py` | Modify | ciphertext-width code columns, `requires_clinical_review`, `confidence_level` | 16.2 |
| `services/diagnosis_service/src/domain/risk.py` | Create | `DiagnosisRiskScreen`, `screen_diagnosis_message`, `raise_for_symptoms`, `SYMPTOM_RISK_CATEGORIES` | 16.3, 16.4 |
| `services/diagnosis_service/src/domain/symptom_extractor.py` | Modify | lexicon risk indicators, `screen_risk`, self-referential self-harm patterns, temporal attachment, log hygiene, stems, severity modifiers, panic/fear/psychomotor, LLM parsing | 16.3, 16.4, 16.10, 16.14, 16.16, 16.17, 16.18, 16.20 |
| `services/diagnosis_service/src/domain/service.py` | Modify | turn ordering, crisis publish, sticky crisis watch, ownership errors, persistence, working-set differential, phase gate, tiers, lifecycle | 16.1, 16.3–16.10, 16.13–16.15, 16.19, 16.21 |
| `services/diagnosis_service/src/domain/models.py` | Modify | result risk fields, session crisis-watch fields, domain exceptions, phase counters, review flag | 16.3, 16.6, 16.7, 16.15, 16.19, 16.21 |
| `services/diagnosis_service/src/domain/differential.py` | Modify | LLM degraded reason, working-set input, empty guard, minimum-criteria flag, criteria penalties, LLM parsing | 16.4, 16.10, 16.15, 16.18, 16.20 |
| `services/diagnosis_service/src/domain/entities.py` | Modify | record review fields | 16.8 |
| `services/diagnosis_service/src/domain/{severity,value_objects,clinical_codes,evidence}.py` | Modify | "library only" labels | 16.22 |
| `services/diagnosis_service/src/schemas.py` | Modify | response risk fields, `HypothesisDTO` gate and tier fields | 16.3, 16.15, 16.19 |
| `services/diagnosis_service/src/api.py` | Modify | `X-Turn-ID`, status mapping, `/status` permission, subject authorization, PHI audit, log hygiene | 16.3, 16.5, 16.7, 16.12, 16.14, 16.21 |
| `services/diagnosis_service/src/infrastructure/redis_session_repository.py` | Modify | store from the runtime client, crisis-watch fields, CAS rebase, encrypted capped envelope, phase counters, review flag | 16.1, 16.6, 16.7, 16.9, 16.15, 16.19 |
| `services/diagnosis_service/src/infrastructure/postgres_repository.py` | Modify | SQL aligned to `001`/`020`, codec, `default` pool, `save_completed_session`, deleter order | 16.8, 16.13 |
| `services/diagnosis_service/src/infrastructure/repository.py` | Modify | `save_completed_session` on the port | 16.8 |
| `services/diagnosis_service/tests/runtime_fakes.py` | Create | `fake_runtime`, `CapturingPublisher`, `CapturingDispatcher` | 16.1, 16.5 |
| `services/diagnosis_service/tests/fixtures.py` | Modify | in-memory `save_completed_session` | 16.8 |
| `services/diagnosis_service/tests/test_mvp_*.py` | Create | one RED-first file per task | all |
| `services/diagnosis_service/tests/{test_api_integration,test_batch_5_3,test_edge_cases,test_phase_c_session_reconcile,test_session_idor,test_end_session_persistence,test_postgres_record_persistence,test_batch_5_1}.py` | Modify | follow the changed contracts (named per task) | 16.1, 16.3, 16.5, 16.7, 16.8, 16.21 |
| `tests/realinfra/diagnosis-service/test_mvp_*.py` | Create | migrated-Postgres and Redis component tests | 16.2, 16.8, 16.13 |
| `tests/integration/test_diagnosis_vignette.py` | Modify | vignette runs the real step 1 and asserts the published event | 16.11 |
| `tests/integration/test_event_flow.py` | Modify (class `TestDiagnosisBridgeSafetyFlags` only) | crisis events are no longer bridged from the dispatcher | 16.5 |

## Tasks

### Task WS-16.1: Adopt the IF-01 runtime in diagnosis, build the session store from the runtime Redis client and keep one set of reasoning knobs
**Covers:** MVP-263
**Severity:** low
**Wave:** 0
**Depends on:** WS-04.4, WS-04.5, WS-01.7, WS-05.5 (`tests/system/test_smoke.py` asserts `GET /live` instead of the local `/health` body, which answers R4)
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-14 (default assumed: strict means `ENVIRONMENT` in {staging, production}; the pilot runs production)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/main.py` (whole module: `DiagnosisServiceAppSettings` keeps `service_name`, `version`, `environment`, `debug`, `host`, `port`, `log_level` and adds `events_enabled: bool` with `validation_alias=AliasChoices("EVENTS_ENABLED", "events_enabled")`, default `True`; drops `reasoning_timeout_ms`, `enable_anti_sycophancy`, `cors_origins`, `request_timeout_ms`, `max_request_size_mb`, `enable_metrics`, `enable_tracing`; module-level `SPEC`; `DiagnosisComponents`; `build_diagnosis_components`; lifespan on `start_service_runtime`/`stop_service_runtime`; `create_application()` returning `create_service_app(...)`; `app = create_application()`; remove `configure_logging`, the ORM `configure_phi_encryption` block, CORS, the local middleware, the three local exception handlers and the local `/ready` `/live` `/health` handlers; keep `GET /`)
- Modify: `services/diagnosis_service/src/infrastructure/redis_session_repository.py` (`DiagnosisSessionRedisSettings` keeps only `session_ttl` and `key_prefix`, `env_prefix="DIAGNOSIS_SESSION_STORE_"`; `DiagnosisSessionStore.__init__(self, settings=None, *, client)` with `client` required; remove `initialize()` and `close()`; add `build_session_store`)
- Modify: `services/diagnosis_service/src/domain/service.py` (`DiagnosisServiceSettings`: `model_config` gains `populate_by_name=True`; no other change)
- Modify: `services/diagnosis_service/src/events.py` (`to_kafka_event`: `from solace_events.schemas import ...` instead of `src.solace_events`; supporting edit for MVP-270, request R-01.9 of WS-01)
- Delete: `services/diagnosis_service/src/config.py` (only `services/diagnosis_service/tests/test_batch_5_3.py::TestConfiguration`, line 606, reads it through the import at line 13)
- Test: `services/diagnosis_service/tests/runtime_fakes.py` (new helper, not collected)
- Test: `services/diagnosis_service/tests/test_mvp_263_runtime_adoption.py` (new)
- Test: `services/diagnosis_service/tests/test_api_integration.py` (modify `TestHealthEndpoints`: `/health` and `/ready` assert the IF-01 body `status in {"ready", "not_ready"}` and `service == "diagnosis-service"`; `/live` unchanged; root unchanged)
- Test: `services/diagnosis_service/tests/test_batch_5_3.py` (delete class `TestConfiguration` and the `config` import; record the test-count drop in the ledger)

**Interfaces:**
- Consumes: IF-01 `solace_infrastructure.runtime.{ServiceRuntimeSpec, ServiceRuntime, create_service_app, start_service_runtime, stop_service_runtime, is_strict_environment, StartupDependencyError}` (WS-04, W0 phase A2); `runtime.redis` built from `REDIS_URL`; `runtime.pg_pool`; `initialize_event_bridge(kafka_settings=None, use_mock=False, postgres_pool=None)` and `KafkaEventBridge.stop()` (WS-06, names stable); `UnifiedLLMClient(LLMClientSettings())` whose `initialize()` never raises outside strict (WS-09, W0 phase A1).
- Produces: `services.diagnosis_service.src.main.SPEC = ServiceRuntimeSpec(service_name="diagnosis-service", environment=<settings.environment>, uses_postgres=True, uses_redis=True, event_plane="publish_consume", handles_phi=True, audited=True, rate_limited=False, edge=False)`; `@dataclass DiagnosisComponents(service: DiagnosisService, event_dispatcher: EventDispatcher, llm_client: UnifiedLLMClient, session_store: DiagnosisSessionStore | None, repository: DiagnosisRepositoryPort | None)`; `async def build_diagnosis_components(runtime: ServiceRuntime, settings: DiagnosisServiceAppSettings) -> DiagnosisComponents` (logs `diagnosis_components_built` with `max_reasoning_time_ms` and `anti_sycophancy` from the effective `DiagnosisServiceSettings`); `def create_application() -> FastAPI`; `build_session_store(client: "redis.asyncio.Redis | None", settings: DiagnosisSessionRedisSettings | None = None) -> DiagnosisSessionStore | None`; readiness checks `diagnosis_service`, `diagnosis_session_store`, `diagnosis_repository` (all critical); `runtime_fakes.fake_runtime(*, redis=None, pg_pool=None, event_plane=None, phi_codec=None, environment="test") -> SimpleNamespace` exposing `readiness_checks: dict[str, tuple[check, critical]]`, `shutdowns: dict[str, closer]`, `register_readiness_check`, `register_shutdown`, `register_background_task`. Supports WS-01.7 (`-k WS-16`) and WS-04.5 (`-k diagnosis-service`).

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_263_runtime_adoption.py::test_mvp_263_app_settings_have_no_dead_reasoning_knobs`
```python
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
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_263_runtime_adoption.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'build_session_store'` (and `SPEC`, `build_diagnosis_components`). With the imports stubbed, `test_mvp_263_app_settings_have_no_dead_reasoning_knobs` fails because `reasoning_timeout_ms` and `enable_anti_sycophancy` exist (read from `DIAGNOSIS_REASONING_TIMEOUT_MS`/`DIAGNOSIS_ENABLE_ANTI_SYCOPHANCY` and never passed on) and `config.py` holds the unused `DIAGNOSIS_REASONING_` class; the store settings still carry `host`/`port` (`DIAGNOSIS_SESSION_REDIS_*`, which no compose file sets); the error-body test gets `{"error": {"message": "I want to end my life - synthetic"}}` from the local `ValueError` handler.
- [ ] **Step 3: Implement** `async def build_diagnosis_components(runtime: ServiceRuntime, settings: DiagnosisServiceAppSettings) -> DiagnosisComponents`. Order: `strict = is_strict_environment(settings.environment)`; strict and not `settings.events_enabled` raises `StartupDependencyError("diagnosis requires EVENTS_ENABLED in strict environments")`; `session_store = build_session_store(runtime.redis)`, `None` in strict raises `StartupDependencyError("diagnosis session store requires REDIS_URL in strict environments")`; `repository = RepositoryFactory.get_default() if runtime.pg_pool is not None else None`, `None` in strict raises; `llm_client = UnifiedLLMClient(LLMClientSettings())` then `await llm_client.initialize()` unguarded; `service_settings = DiagnosisServiceSettings()`; build extractor, generator, `EventDispatcher()` and `DiagnosisService(...)` as `main.py` does today; `await service.initialize()`; register the three readiness checks (store check `await session_store._client.ping()`) and shutdown closers (`service.shutdown`, `llm_client.shutdown`); never close `runtime.redis`; log `diagnosis_components_built`. The lifespan is `runtime = await start_service_runtime(app, SPEC)` first, then the composition, then `app.state.diagnosis_service = components.service` (and the other `app.state` names the API uses), then, only when `settings.events_enabled`, `bridge = await initialize_event_bridge(postgres_pool=runtime.pg_pool)`, `components.event_dispatcher.subscribe_all(bridge.bridge_event)` and `runtime.register_shutdown("diagnosis_event_bridge", bridge.stop)`; `yield`; and `await stop_service_runtime(runtime)` last.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/bootstrap/test_service_conformance.py -k diagnosis-service -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/build/test_import_roots.py -k WS-16 -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q`. Expected: all pass (`test_gdpr_erasure_wiring.py` and `test_api_integration.py` keep importing `app`), except `tests/integration/test_phi_encryption_activation.py::TestPhiEncryptionActivation::test_phi_services_activate_encryption`, which fails for `diagnosis_service` until request R3 lands (no WS-04 task carries CR-8 yet); record that in the ledger and do not edit the file.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.2: Slot 020 gives diagnosis records ciphertext-width code columns and review fields
**Covers:** MVP-128, MVP-130
**Severity:** medium
**Wave:** 2
**Depends on:** WS-15.3, WS-02.3
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** D-01 (default assumed: Alembic is the schema of record; repository SQL changes to match it, and the diagnosis codes are PHI encrypted at rest)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/020_diagnosis.py` (`upgrade()`/`downgrade()` only; `revision`/`down_revision` untouched)
- Test: `tests/realinfra/diagnosis-service/test_mvp_128_slot_020.py` (new)

**Interfaces:**
- Consumes: IF-03 slot file `020_diagnosis` (WS-02.3), roles `solace_migrator`/`solace_app`, the 009 NULLIF policies on `diagnosis_sessions`/`diagnosis_records`; IF-19 `pg_migrated -> PgHandle(dsn_app, dsn_migrator, dsn_superuser, app_pool)` with `superuser_fetch` (WS-05.2).
- Produces: `diagnosis_records.dsm5_code TEXT NULL`, `diagnosis_records.icd11_code TEXT NULL` (ciphertext), `diagnosis_records.requires_clinical_review BOOLEAN NOT NULL DEFAULT false`, `diagnosis_records.confidence_level VARCHAR(20) NULL`. Downgrade restores `VARCHAR(20)` with `USING left(<col>, 20)` and drops the two columns.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/diagnosis-service/test_mvp_128_slot_020.py::test_mvp_128_record_code_columns_hold_ciphertext`
```python
"""MVP-128 / MVP-130: slot 020 lets diagnosis records hold encrypted codes and the review flag, under RLS as solace_app."""
from __future__ import annotations
from uuid import UUID, uuid4
import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
CIPHERTEXT = "enc:v1:test-k1:" + "A" * 120


async def _insert_user(dsn_superuser: str) -> UUID:
    # If slot 012 added NOT NULL users columns without defaults, add them here with synthetic values.
    uid = uuid4()
    conn = await asyncpg.connect(dsn_superuser)
    try:
        await conn.execute("INSERT INTO users (id, created_at, updated_at, encryption_key_id, email, password_hash, display_name) "
                           "VALUES ($1, now(), now(), 'test-key', $2, 'x', 'Test User')", uid, f"{uid}@test.invalid")
    finally:
        await conn.close()
    return uid


@pytest.mark.asyncio
async def test_mvp_128_record_code_columns_hold_ciphertext(pg_migrated) -> None:
    cols = {r["column_name"]: r for r in await pg_migrated.superuser_fetch(
        "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns "
        "WHERE table_name = 'diagnosis_records'")}
    assert cols["dsm5_code"]["data_type"] == "text" and cols["icd11_code"]["data_type"] == "text"
    assert cols["requires_clinical_review"]["is_nullable"] == "NO" and "false" in cols["requires_clinical_review"]["column_default"]
    assert cols["confidence_level"]["data_type"] == "character varying"


@pytest.mark.asyncio
async def test_mvp_128_app_role_writes_session_then_record_for_its_own_user(pg_migrated) -> None:
    user_a, user_b = await _insert_user(pg_migrated.dsn_superuser), await _insert_user(pg_migrated.dsn_superuser)
    sid, rid = uuid4(), uuid4()
    app = await asyncpg.connect(pg_migrated.dsn_app)
    try:
        await app.execute("SELECT set_config('app.current_user_id', $1, false)", str(user_a))
        await app.execute("INSERT INTO diagnosis_sessions (id, created_at, updated_at, user_id, encryption_key_id, session_number, started_at) "
                          "VALUES ($1, now(), now(), $2, 'test-k1', 1, now())", sid, user_a)
        await app.execute("INSERT INTO diagnosis_records (id, created_at, updated_at, user_id, encryption_key_id, session_id, "
                          "primary_diagnosis, dsm5_code, icd11_code, confidence, severity) "
                          "VALUES ($1, now(), now(), $2, 'test-k1', $3, $4, $4, $4, 0.6, 'MODERATE')", rid, user_a, sid, CIPHERTEXT)
        assert await app.fetchval("SELECT requires_clinical_review FROM diagnosis_records WHERE id = $1", rid) is False
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await app.execute("INSERT INTO diagnosis_sessions (id, created_at, updated_at, user_id, encryption_key_id, session_number, started_at) "
                              "VALUES ($1, now(), now(), $2, 'test-k1', 1, now())", uuid4(), user_b)
    finally:
        await app.close()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/diagnosis-service/test_mvp_128_slot_020.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `dsm5_code` reports `character varying` (001 created `VARCHAR(20)`), `KeyError: 'requires_clinical_review'`, and the record insert raises `asyncpg.exceptions.StringDataRightTruncationError: value too long for type character varying(20)` once the codes are ciphertext.
- [ ] **Step 3: Implement** `upgrade()` with two `op.alter_column(..., type_=sa.Text(), existing_type=sa.String(20), existing_nullable=True)` and two `op.add_column(...)` (`server_default=sa.false()` for the flag); `downgrade()` drops the two columns and alters the codes back with `postgresql_using="left(dsm5_code, 20)"` / `"left(icd11_code, 20)"`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/diagnosis-service -o "addopts=" -p no:logging -q`. Expected: all pass (`test_chain_real_pg.py` upgrade/downgrade/upgrade green). WS-02 reviews the slot diff.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.3: Screen every diagnosis message with the shared lexicon before any LLM call and keep the flags when reasoning times out
**Covers:** MVP-046, MVP-047
**Severity:** high
**Wave:** 2
**Depends on:** WS-16.1, WS-12.3 (lexicon v1 and the phrase set, W1; the diagnosis register phrasings it must also hold are request R1)
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: lexicon v1 as shipped is provisional; max-only fusion; diagnosis never lowers a lexicon level; final lexicon and thresholds need the named clinician's sign-off before W5)
**Real infra:** no
**Files:**
- Create: `services/diagnosis_service/src/domain/risk.py` (`DiagnosisRiskScreen`, `screen_diagnosis_message`)
- Modify: `services/diagnosis_service/src/domain/symptom_extractor.py` (`_build_risk_patterns` keeps only the informational `psychotic_features` pattern; `_detect_risk_indicators(text)` returns `sorted(screen.categories)` plus `"psychotic_features"` when that pattern matches, and logs `risk_indicator_detected` with `count` only; new `screen_risk(self, message: str) -> DiagnosisRiskScreen`; the extractor `ExtractionResult` gains `risk_level: CrisisLevel = CrisisLevel.NONE` and `detection_layer: int = 1`)
- Modify: `services/diagnosis_service/src/domain/service.py` (`assess(..., *, turn_id: UUID | None = None)`; pre-screen before the chain; a crisis turn skips the chain; the timeout result keeps the screen; `_determine_next_phase` escalates on `screen.is_crisis` instead of the literal flag names; `_CRISIS_SAFETY_FLAGS` removed; `DiagnosisServiceSettings.max_reasoning_time_ms` default `6000`; `extract_symptoms` maps `risk_level`)
- Modify: `services/diagnosis_service/src/domain/models.py` (`AssessmentResult` gains `risk_level: CrisisLevel = CrisisLevel.NONE`, `risk_categories: list[str]`, `degraded: list[str]`, `detection_layer: int = 1`; service `ExtractionResult` gains `risk_level`, `degraded`)
- Modify: `services/diagnosis_service/src/schemas.py` (`AssessmentResponse` and `SymptomExtractionResponse` gain `risk_level: CrisisLevel = CrisisLevel.NONE`, `risk_categories: list[str] = []`, `degraded: list[str] = []`)
- Modify: `services/diagnosis_service/src/api.py` (map the three fields in `/assess` and `/extract-symptoms`)
- Test: `services/diagnosis_service/tests/test_mvp_046_lexicon_risk_screen.py` (new)
- Test: `services/diagnosis_service/tests/test_edge_cases.py` (modify `test_all_risk_indicators`: each message's `risk_indicators` equals `sorted(screen_diagnosis_message(msg).categories)` plus `psychotic_features` for the voices case)
- Test: `services/diagnosis_service/tests/test_phase_c_session_reconcile.py` (modify line 189: assert `result.risk_level not in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)` instead of the literal flag names)

**Interfaces:**
- Consumes: IF-07 `solace_clinical.{screen_text, ScreenResult, CATEGORIES, LEXICON_VERSION, level_rank}` and `solace_clinical.phrases.check_detector(detect, phrase_set=None) -> list[PhraseFailure]` (WS-12.2/WS-12.3, W1 phase A); `solace_common.enums.CrisisLevel`.
- Produces: `@dataclass(frozen=True) DiagnosisRiskScreen(level: CrisisLevel, categories: frozenset[str], detection_layer: int, lexicon_version: str)` with property `is_crisis -> bool` (`level_rank(level) >= level_rank(HIGH)`); `screen_diagnosis_message(message: str, *, rule_symptom_names: Iterable[str] = ()) -> DiagnosisRiskScreen` (pure, no I/O; in this task `rule_symptom_names` is accepted and ignored); `SymptomExtractor.screen_risk(message: str) -> DiagnosisRiskScreen`; `AssessmentResult.{risk_level, risk_categories, degraded, detection_layer}`; degraded reason `reasoning_timeout` (no longer a safety flag); `safety_flags` hold IF-07 category codes plus the informational `psychotic_features`.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_046_lexicon_risk_screen.py::test_mvp_047_hanging_llm_cannot_drop_a_crisis_flag`
```python
"""MVP-046 / MVP-047 (absorbs MVP-127): the shared lexicon screens every message before any LLM call; a timeout keeps the flags."""
from __future__ import annotations
import asyncio
from uuid import uuid4
import pytest
from solace_clinical import CATEGORIES, level_rank
from solace_clinical.phrases import check_detector
from solace_common.enums import CrisisLevel
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.risk import screen_diagnosis_message
from services.diagnosis_service.src.domain.service import DiagnosisService, DiagnosisServiceSettings
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.schemas import DiagnosisPhase

HIGH = level_rank(CrisisLevel.HIGH)
REGISTER_POSITIVE = ["I've been thinking about suicide", "I feel suicidal", "I want to die", "I want to end my life",
                     "I'd be better off dead", "I'm going to overdose", "I've been self-harming",
                     "I'm not sad at all, but I want to kill myself"]  # last: negation in an earlier clause (focus 1)
REGISTER_NEGATIVE = ["I'm going to end it with my boyfriend", "I'm cutting back on carbs", "I'm going to give up sugar"]


class _HangingLLM:
    is_available = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(30)
        return '{"symptoms": []}'


async def _started(llm=None, timeout_ms=300):
    svc = DiagnosisService(settings=DiagnosisServiceSettings(max_reasoning_time_ms=timeout_ms),
                           symptom_extractor=SymptomExtractor(SymptomExtractorSettings(), llm_client=llm),
                           differential_generator=DifferentialGenerator(DifferentialSettings(), llm_client=llm))
    await svc.initialize()
    uid = uuid4()
    return svc, uid, await svc.start_session(uid, "assessment", {}, None)


async def _assess(svc, uid, sid, message):
    return await svc.assess(user_id=uid, session_id=sid, message=message, conversation_history=[], existing_symptoms=[],
                            current_phase=DiagnosisPhase.RAPPORT, current_differential=None, user_context={})


@pytest.mark.clinical
def test_mvp_046_detector_passes_the_shared_phrase_set() -> None:
    extractor = SymptomExtractor(SymptomExtractorSettings())
    assert check_detector(lambda text: (extractor.screen_risk(text).level, extractor.screen_risk(text).categories)) == []


@pytest.mark.parametrize("message", REGISTER_POSITIVE)
def test_mvp_046_register_disclosures_are_crisis(message) -> None:
    screen = screen_diagnosis_message(message)
    assert level_rank(screen.level) >= HIGH and screen.categories and screen.categories <= CATEGORIES


@pytest.mark.parametrize("message", REGISTER_NEGATIVE)
def test_mvp_046_benign_phrasings_do_not_page(message) -> None:
    assert level_rank(screen_diagnosis_message(message).level) < HIGH


@pytest.mark.asyncio
async def test_mvp_046_extractor_risk_indicators_are_lexicon_categories() -> None:
    result = await SymptomExtractor(SymptomExtractorSettings()).extract("I feel suicidal", [], [])
    assert "suicidal_ideation" in result.risk_indicators
    assert set(result.risk_indicators) - {"psychotic_features"} <= CATEGORIES


@pytest.mark.asyncio
async def test_mvp_047_hanging_llm_cannot_drop_a_crisis_flag() -> None:
    llm = _HangingLLM()
    svc, uid, start = await _started(llm)
    result = await asyncio.wait_for(_assess(svc, uid, start.session_id, "I want to kill myself"), timeout=2.0)
    assert result.phase == DiagnosisPhase.CRISIS and level_rank(result.risk_level) >= HIGH
    assert "suicidal_ideation" in result.safety_flags and "reasoning_timeout" not in result.safety_flags
    assert llm.calls == 0  # a crisis turn never sends the disclosure to the LLM
    session = svc._active_sessions[start.session_id]
    assert session.phase == DiagnosisPhase.CRISIS and "suicidal_ideation" in session.safety_flags


@pytest.mark.asyncio
async def test_mvp_047_timeout_keeps_the_screen_and_reports_degraded() -> None:
    message = "Everything feels hopeless lately"
    svc, uid, start = await _started(_HangingLLM())
    result = await asyncio.wait_for(_assess(svc, uid, start.session_id, message), timeout=2.0)
    screen = screen_diagnosis_message(message)
    assert "reasoning_timeout" in result.degraded and "reasoning_timeout" not in result.safety_flags
    assert result.risk_level == screen.level and set(screen.categories) <= set(result.safety_flags)


def test_mvp_047_reasoning_budget_fits_inside_the_orchestrator_hop() -> None:
    assert DiagnosisServiceSettings().max_reasoning_time_ms <= 6000  # IF-11 per-hop timeout is 8 s
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_046_lexicon_risk_screen.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'services.diagnosis_service.src.domain.risk'`. With `risk.py` stubbed onto the old regex: `I feel suicidal` gives `risk_indicators == []` (`\b(suicid...)\b` never matches `suicidal`), `I'm cutting back on carbs` gives `self_harm`, and the hanging-LLM case returns `phase=RAPPORT`, `safety_flags=['reasoning_timeout']` with `llm.calls == 1` (step 1 awaits the LLM before the regex check inside the `wait_for`); `max_reasoning_time_ms` is 10000.
- [ ] **Step 3: Implement** `screen_diagnosis_message` as `solace_clinical.screen_text(message)` mapped to `DiagnosisRiskScreen(level, frozenset(categories), 1, lexicon_version)` (no diagnosis-local lexicon). In `assess`: `turn_id = turn_id or uuid4()`; `screen = self._symptom_extractor.screen_risk(message) if self._symptom_extractor else screen_diagnosis_message(message)` runs first; `_load_session` and the existing ownership check follow; if `screen.is_crisis` the result is built without the chain (`phase=CRISIS`, `safety_flags=sorted(screen.categories)`, `risk_level=screen.level`, `extracted_symptoms=[]`, `differential=session.differential or DifferentialDTO()`, `response_text` from the CRISIS entries of `_generate_response`/`_generate_next_question`); otherwise the chain runs under `asyncio.wait_for` and its step-1 risk is fused with the screen by `max(level)` (union of categories), never lowered; the timeout result carries `safety_flags=sorted(screen.categories)`, `risk_level=screen.level`, `phase=CRISIS` when `screen.is_crisis` else the current phase, and `degraded=["reasoning_timeout"]`. `crisis_turn` for the existing fail-loud persist becomes `screen.is_crisis`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/clinical -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/test_diagnosis_vignette.py -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.4: Self-harm symptoms and LLM-reported ideation raise the turn's risk, and LLM failures are visible
**Covers:** MVP-049
**Severity:** high
**Wave:** 2
**Depends on:** WS-16.3, WS-09.3, WS-09.4
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: a self-referential self-harm symptom is HIGH; an LLM-reported ideation symptom is an unreliable signal that may raise to HIGH but never lowers; context-free single words never reach HIGH)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/risk.py` (add `SYMPTOM_RISK_CATEGORIES`, `raise_for_symptoms`; `screen_diagnosis_message` fuses `rule_symptom_names` through `raise_for_symptoms(..., source="rule")`)
- Modify: `services/diagnosis_service/src/domain/symptom_extractor.py` (`self_harm` symptom patterns require self-reference: `r"\b(hurt|harm|cut|burn|scratch|hit|punch)\w*\s+myself\b(?!\s+(some\s+)?slack)"`, `r"\bself[\s-]?(harm|injur)\w*"`; bare `cutting`/`burning` removed; `screen_risk` passes the rule-extracted symptom names; extractor `ExtractionResult` gains `llm_symptom_names: list[str]` and `degraded: list[str]`; `_llm_extract_symptoms` failures (`LLMUnavailableError` or any exception) add `llm_unavailable`)
- Modify: `services/diagnosis_service/src/domain/differential.py` (`_llm_generate_differential` failure adds `llm_unavailable` to `DifferentialResult.degraded: list[str]`, new field)
- Modify: `services/diagnosis_service/src/domain/service.py` (after step 1 the turn screen becomes `raise_for_symptoms(raise_for_symptoms(screen, rule_names, source="rule"), llm_names, source="llm")`; an LLM-only raise to HIGH sets `phase=CRISIS` and the CRISIS reply; extractor and differential `degraded` reasons reach `AssessmentResult.degraded`)
- Modify: `services/diagnosis_service/src/main.py` (`runtime.register_readiness_check("llm", llm_client.readiness, critical=False)`; supporting CR-7 of WS-09)
- Test: `services/diagnosis_service/tests/test_mvp_049_symptom_and_llm_risk.py` (new)

**Interfaces:**
- Consumes: `solace_clinical.{RiskSignal, fuse_risk, level_rank, CATEGORIES}` (WS-12.2); IF-15 `UnifiedLLMClient.readiness() -> tuple[bool, str]` and `LLMUnavailableError` (WS-09.3/WS-09.4); WS-16.3 `DiagnosisRiskScreen`, `screen_diagnosis_message`.
- Produces: `SYMPTOM_RISK_CATEGORIES: Mapping[str, str] = {"self_harm": "self_harm", "self_injury": "self_harm", "suicidal_ideation": "suicidal_ideation", "suicidal_thoughts": "suicidal_ideation", "suicide_plan": "suicide_plan", "overdose": "overdose", "homicidal_ideation": "homicidal_ideation"}`; `raise_for_symptoms(screen: DiagnosisRiskScreen, symptom_names: Iterable[str], *, source: Literal["rule", "llm"]) -> DiagnosisRiskScreen` (a mapped symptom raises the level to at least HIGH and adds its category through `fuse_risk` with `RiskSignal(source=f"diagnosis_{source}", level=HIGH, confidence=0.9 if rule else 0.6, reliable=source == "rule")`; `detection_layer` becomes 3 only when the LLM signal is what raised the level; never lowers); degraded reason `llm_unavailable`; readiness check `llm` (non-critical). Answers WS-09.4 (CR-7 of WS-09: `runtime.register_readiness_check("llm", llm_client.readiness, critical=False)` in `main.py`, pinned by `runtime.readiness_checks["llm"][1] is False`, and `LLMUnavailableError` caught at the extractor and differential call sites instead of testing for `""`).

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_049_symptom_and_llm_risk.py::test_mvp_049_recognised_self_harm_symptom_raises_the_turn`
```python
"""MVP-049: a recognised self-harm symptom or LLM-reported ideation raises the turn to HIGH; risk is never lowered."""
from __future__ import annotations
import json
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4
import fakeredis.aioredis
import pytest
from solace_clinical import level_rank
from solace_common.enums import CrisisLevel
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.risk import raise_for_symptoms, screen_diagnosis_message
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.main import DiagnosisServiceAppSettings, build_diagnosis_components
from services.diagnosis_service.src.schemas import DiagnosisPhase, SeverityLevel, SymptomDTO, SymptomType
from services.diagnosis_service.tests.runtime_fakes import fake_runtime

HIGH = level_rank(CrisisLevel.HIGH)
CALM = "I had a long week at work"


class _JsonLLM:
    is_available = True

    def __init__(self, payload) -> None:
        self._payload = payload

    async def generate(self, **kwargs):
        return json.dumps(self._payload)


class _FailingLLM:
    is_available = True

    async def generate(self, **kwargs):
        raise RuntimeError("provider down")


class _SelfHarmSymptomExtractor:
    """Stub: extracts a SEVERE self_harm symptom but reports no risk (the defect path)."""
    def screen_risk(self, message):
        return screen_diagnosis_message(message)

    async def extract(self, message, history, existing):
        symptom = SymptomDTO(symptom_id=uuid4(), name="self_harm", description="d", symptom_type=SymptomType.BEHAVIORAL,
                             severity=SeverityLevel.SEVERE, confidence=Decimal("0.9"))
        return SimpleNamespace(symptoms=[symptom], temporal_info={}, contextual_factors=[], risk_indicators=[],
                               llm_symptom_names=[], degraded=[])


async def _assess(svc, message):
    await svc.initialize()
    uid = uuid4()
    start = await svc.start_session(uid, "assessment", {}, None)
    return await svc.assess(user_id=uid, session_id=start.session_id, message=message, conversation_history=[],
                            existing_symptoms=[], current_phase=DiagnosisPhase.RAPPORT, current_differential=None, user_context={})


@pytest.mark.asyncio
async def test_mvp_049_recognised_self_harm_symptom_raises_the_turn() -> None:
    result = await _assess(DiagnosisService(symptom_extractor=_SelfHarmSymptomExtractor()), CALM)
    assert level_rank(result.risk_level) >= HIGH and "self_harm" in result.risk_categories
    assert result.phase == DiagnosisPhase.CRISIS and "self_harm" in result.safety_flags


@pytest.mark.asyncio
async def test_mvp_049_llm_reported_ideation_raises_the_turn_at_layer_3() -> None:
    llm = _JsonLLM({"symptoms": [{"name": "suicidal_ideation", "type": "COGNITIVE", "severity": "SEVERE", "confidence": 0.9, "evidence": "e"}]})
    svc = DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings(), llm_client=llm),
                           differential_generator=DifferentialGenerator(DifferentialSettings()))
    result = await _assess(svc, CALM)
    assert level_rank(result.risk_level) >= HIGH and "suicidal_ideation" in result.risk_categories
    assert result.detection_layer == 3 and result.phase == DiagnosisPhase.CRISIS


@pytest.mark.parametrize("message", ["I keep hitting myself when I'm upset", "I scratch myself until I bleed",
                                     "Sometimes I think about hurting myself"])
def test_mvp_049_self_referential_self_harm_is_crisis_before_any_llm(message) -> None:
    screen = SymptomExtractor(SymptomExtractorSettings()).screen_risk(message)
    assert level_rank(screen.level) >= HIGH and "self_harm" in screen.categories and screen.detection_layer == 1


@pytest.mark.parametrize("message", ["I'm cutting back on carbs", "I'm cutting myself some slack this week"])
def test_mvp_049_context_free_cutting_is_not_self_harm(message) -> None:
    names = {s.name for s in SymptomExtractor(SymptomExtractorSettings())._extract_symptoms(message, [])}
    assert "self_harm" not in names


def test_mvp_049_raise_for_symptoms_never_lowers() -> None:
    crisis = screen_diagnosis_message("I want to kill myself")
    assert raise_for_symptoms(crisis, ["anxiety"], source="llm").level == crisis.level
    assert raise_for_symptoms(crisis, [], source="rule") == crisis


@pytest.mark.asyncio
async def test_mvp_049_llm_failure_is_degraded_not_silent() -> None:
    svc = DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings(), llm_client=_FailingLLM()),
                           differential_generator=DifferentialGenerator(DifferentialSettings(), llm_client=_FailingLLM()))
    result = await _assess(svc, CALM)
    assert "llm_unavailable" in result.degraded and result.response_text


@pytest.mark.asyncio
async def test_mvp_049_llm_readiness_is_registered_non_critical() -> None:
    runtime = fake_runtime(redis=fakeredis.aioredis.FakeRedis(decode_responses=True))
    await build_diagnosis_components(runtime, DiagnosisServiceAppSettings())
    assert runtime.readiness_checks["llm"][1] is False
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_049_symptom_and_llm_risk.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ImportError: cannot import name 'raise_for_symptoms'`; once stubbed, the stub-extractor and LLM cases return `risk_level=NONE`, `safety_flags=[]` and a non-CRISIS phase (risk comes only from the lexicon screen and the extractor's `risk_indicators`, never from the extracted SEVERE `self_harm` or the LLM `suicidal_ideation` symptom), `I'm cutting back on carbs` still extracts `self_harm` (bare `cutting` pattern), the failing-LLM case has no `degraded` reason, and no `llm` readiness check exists.
- [ ] **Step 3: Implement** `raise_for_symptoms` and the self-referential patterns; in `_run_assessment_chain` take `rule_names = [s.name for s in symptoms if s.name not in llm_symptom_names]` and `llm_names = getattr(extraction, "llm_symptom_names", [])`, raise the turn screen with both, and use the raised screen for `risk_level`, `risk_categories`, `safety_flags`, `detection_layer` and the phase. The negative-phrase check of WS-16.3 (`check_detector`) must stay green with the new patterns.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/clinical -o "addopts=" -p no:logging -q`. Expected: all pass (`test_api_integration.py::test_extract_symptoms_risk_detection` keeps `self_harm` for "Sometimes I think about hurting myself").
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.5: Publish at most one categories-only `safety.crisis.detected` per turn, with the turn id, from `/assess` and `/extract-symptoms`
**Covers:** MVP-121, MVP-260, MVP-262, MVP-253
**Severity:** medium
**Wave:** 2
**Depends on:** WS-16.4, WS-06.8, WS-04.6, WS-05.7, WS-04.3 (metric object `CRISIS_DETECTED`, W0)
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-28 (default assumed: diagnosis publishes `CrisisDetected` through its own outbox with the turn correlation id and notification coalesces the episode); D-06 (ELEVATED shows resources and pages no one)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/events.py` (add `build_crisis_detected_event`; `SafetyFlagRaisedEvent` fields become `level: str`, `categories: tuple[str, ...]`, `detection_layer: int`, `correlation_id: UUID | None` (no `trigger_text`, `severity` or `recommended_action`); `to_kafka_event(SafetyFlagRaisedEvent)` returns `None`; `EventFactory.safety_flag_raised(user_id, session_id, level, categories, detection_layer, correlation_id)`)
- Modify: `services/diagnosis_service/src/domain/service.py` (constructor `event_publisher: Callable[..., Awaitable[UUID]] | None = None`; `_publish_crisis(user_id, session_id, screen, turn_id) -> list[str]`; `assess` publishes right after the pre-screen when it is a crisis, or after the chain when the chain raised the turn to HIGH, at most once per turn; `extract_symptoms(..., *, turn_id: UUID | None = None)` screens, raises with the extracted symptoms and publishes the same way; one `SafetyFlagRaisedEvent` per crisis turn on the in-process dispatcher; `SymptomExtractedEvent` is dispatched with `extracted_from=""`; the `ReasoningStepResultDTO.input_summary` of step 1 becomes `f"Message: {len(message)} chars"`; log `diagnosis_crisis_detected` with `session_id`, `level` and `category_count` only)
- Modify: `services/diagnosis_service/src/api.py` (`/assess` and `/extract-symptoms` take `http_request: Request`, read `X-Turn-ID` into a UUID, mint `uuid4()` when missing or invalid, and pass `turn_id`)
- Modify: `services/diagnosis_service/src/main.py` (`event_publisher=runtime.event_plane.publish if runtime.event_plane is not None else None`)
- Test: `services/diagnosis_service/tests/runtime_fakes.py` (modify: add `CapturingPublisher(fail: Exception | None = None)` with `.events`, `.conns` and `async publish(event, *, conn=None) -> UUID`, and `CapturingDispatcher` with `.events` and `async dispatch(event)`)
- Test: `services/diagnosis_service/tests/test_mvp_121_one_crisis_event_per_turn.py` (new)
- Create: `services/diagnosis_service/tests/test_mvp_357_event_contracts.py` (producer-side event contract test requested by WS-05.10, request 9 of WS-05)
- Test: `services/diagnosis_service/tests/test_batch_5_3.py` (modify the `EventFactory.safety_flag_raised` case at line 600 to the new signature)
- Test: `tests/integration/test_event_flow.py` (modify class `TestDiagnosisBridgeSafetyFlags` only: the four safety-flag cases assert `diagnosis_to_kafka_event(SafetyFlagRaisedEvent(...)) is None` and one new case asserts `build_crisis_detected_event(...)` validates through `EVENT_REGISTRY`; `test_diagnosis_recorded_converts_to_kafka` unchanged)

**Interfaces:**
- Consumes: IF-05 `ServiceEventPlane.publish(event, *, conn=None) -> UUID`, `solace_events.errors.EventPublishError` (WS-06.8); IF-06 event rules (WS-12); `solace_events.schemas.{CrisisDetectedEvent, EventMetadata, EVENT_REGISTRY}`; `solace_infrastructure.runtime.metrics.CRISIS_DETECTED` (series `solace_crisis_detected_total`, labels `source`, `level`; WS-04.3, W0); IF-20 `solace_testing.contracts.{load_event_contracts, verify_event_producer}` and `tests/contracts/events/diagnosis__notification/safety.crisis.detected.json` (WS-05.7).
- Produces: `events.build_crisis_detected_event(*, user_id: UUID, session_id: UUID, level: CrisisLevel, categories: Iterable[str], turn_id: UUID, detection_layer: int) -> CrisisDetectedEvent` (raises `ValueError` below HIGH or when no category is in `CATEGORIES`; `trigger_indicators` = sorted codes in `CATEGORIES`; `confidence=Decimal("0.90")` at layer 1, `Decimal("0.60")` at layer 3; `escalation_action="page_on_call"`; `requires_human_review=True`; `metadata.correlation_id=turn_id`; `metadata.source_service="diagnosis-service"`); `DiagnosisService(event_publisher=...)`; degraded reasons `crisis_event_publish_failed` and `crisis_event_unpublished` (shared vocabulary with NI-15.1); `X-Turn-ID` on `/assess` and `/extract-symptoms`. Answers WS-06.12 (codes, never `trigger_text`) and WS-05.10 (producer contract test).

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_121_one_crisis_event_per_turn.py::test_mvp_260_two_risk_categories_publish_one_event`
```python
"""MVP-121 / MVP-260 / MVP-262 / MVP-253: one categories-only crisis event per turn, correlated by the turn id; failures surface."""
from __future__ import annotations
from uuid import UUID, uuid4
import httpx
import pytest
from prometheus_client import REGISTRY
from solace_clinical import CATEGORIES
from solace_events.errors import EventPublishError
from solace_events.schemas import EVENT_REGISTRY, CrisisDetectedEvent
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.events import SafetyFlagRaisedEvent, to_kafka_event
from services.diagnosis_service.src.schemas import DiagnosisPhase
from services.diagnosis_service.tests.runtime_fakes import CapturingDispatcher, CapturingPublisher

TURN = UUID("00000000-0000-4000-e000-000000000001")
TWO_RISKS = "I have been hurting myself and I don't want to live"


def _svc(publisher, dispatcher=None) -> DiagnosisService:
    return DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
                            differential_generator=DifferentialGenerator(DifferentialSettings()),
                            event_publisher=publisher.publish if publisher else None, event_dispatcher=dispatcher)


async def _assess(svc, message, turn_id=TURN):
    await svc.initialize()
    uid = uuid4()
    start = await svc.start_session(uid, "assessment", {}, None)
    result = await svc.assess(user_id=uid, session_id=start.session_id, message=message, conversation_history=[],
                              existing_symptoms=[], current_phase=DiagnosisPhase.RAPPORT, current_differential=None,
                              user_context={}, turn_id=turn_id)
    return uid, start, result


def _crisis_count() -> float:
    return sum(REGISTRY.get_sample_value("solace_crisis_detected_total", {"source": "diagnosis", "level": lvl}) or 0
               for lvl in ("HIGH", "CRITICAL"))


@pytest.mark.asyncio
async def test_mvp_260_two_risk_categories_publish_one_event() -> None:
    publisher, dispatcher = CapturingPublisher(), CapturingDispatcher()
    before = _crisis_count()
    uid, start, result = await _assess(_svc(publisher, dispatcher), TWO_RISKS)
    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert isinstance(event, CrisisDetectedEvent) and (event.user_id, event.session_id) == (uid, start.session_id)
    assert event.crisis_level in ("HIGH", "CRITICAL") and event.detection_layer == 1
    assert "self_harm" in event.trigger_indicators and set(event.trigger_indicators) <= CATEGORIES
    assert event.metadata.correlation_id == TURN and event.metadata.source_service == "diagnosis-service"
    assert event.escalation_action == "page_on_call" and event.requires_human_review is True
    dumped = event.model_dump_json()
    assert "hurting myself" not in dumped and "want to live" not in dumped
    assert EVENT_REGISTRY["safety.crisis.detected"].model_validate(event.to_dict()).metadata.correlation_id == TURN
    flags = [e for e in dispatcher.events if isinstance(e, SafetyFlagRaisedEvent)]
    assert len(flags) == 1 and not hasattr(flags[0], "trigger_text")
    assert all("hurting myself" not in str(vars(e)) for e in dispatcher.events)
    assert _crisis_count() == before + 1


def test_mvp_121_dispatcher_safety_flags_are_never_bridged() -> None:
    assert to_kafka_event(SafetyFlagRaisedEvent(user_id=uuid4(), session_id=uuid4(), level="HIGH",
                                                categories=("self_harm",), detection_layer=1, correlation_id=TURN)) is None


@pytest.mark.asyncio
async def test_mvp_253_publish_failure_is_degraded_never_silent() -> None:
    _, _, result = await _assess(_svc(CapturingPublisher(fail=EventPublishError("outbox down"))), "I want to kill myself")
    assert result.phase == DiagnosisPhase.CRISIS and "crisis_event_publish_failed" in result.degraded


@pytest.mark.asyncio
async def test_mvp_253_missing_publisher_is_flagged() -> None:
    _, _, result = await _assess(_svc(None), "I want to kill myself")
    assert "crisis_event_unpublished" in result.degraded


@pytest.mark.asyncio
async def test_mvp_260_non_crisis_turn_publishes_nothing() -> None:
    publisher = CapturingPublisher()
    _, _, result = await _assess(_svc(publisher), "Work was stressful this week")
    assert publisher.events == [] and result.risk_level not in ("HIGH", "CRITICAL")


@pytest.mark.asyncio
async def test_mvp_262_extract_symptoms_publishes_for_a_crisis() -> None:
    publisher = CapturingPublisher()
    svc = _svc(publisher)
    await svc.initialize()
    out = await svc.extract_symptoms(uuid4(), uuid4(), "I want to kill myself", [], [], turn_id=TURN)
    assert "suicidal_ideation" in out.risk_indicators
    assert len(publisher.events) == 1 and publisher.events[0].metadata.correlation_id == TURN
```
Add `test_mvp_121_route_uses_x_turn_id` in the same file: build `create_application()`, set `app.state.diagnosis_service` to a started `_svc(publisher)` with a session started for `uid`, override `get_current_user` with `AuthenticatedUser(user_id=str(uid), token_type=TokenType.SERVICE, roles=["user"])`, POST `/api/v1/diagnosis/assess` with header `X-Turn-ID: 00000000-0000-4000-e000-000000000001` through `httpx.ASGITransport`, and assert `publisher.events[0].metadata.correlation_id == TURN` and the JSON body has `risk_level` in `["HIGH", "CRITICAL"]` and `"degraded" in body`.
In `services/diagnosis_service/tests/test_mvp_357_event_contracts.py::test_mvp_357_diagnosis_crisis_event_matches_the_notification_contract`: `contract = load_event_contracts(REPO / "tests/contracts/events/diagnosis__notification")[0]`; `event = build_crisis_detected_event(user_id=uuid4(), session_id=uuid4(), level=CrisisLevel.HIGH, categories={"self_harm"}, turn_id=TURN, detection_layer=1)`; `assert verify_event_producer(contract, event) == []`; and `pytest.raises(ValueError)` for `level=CrisisLevel.ELEVATED`.
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_121_one_crisis_event_per_turn.py services/diagnosis_service/tests/test_mvp_357_event_contracts.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `TypeError: DiagnosisService.__init__() got an unexpected keyword argument 'event_publisher'` and `ImportError: cannot import name 'build_crisis_detected_event'`; the dispatcher path today emits one `SafetyFlagRaisedEvent` per crisis flag carrying `trigger_text=message[:200]`, which the bridge maps to a HIGH `CrisisDetectedEvent` with the raw text in `trigger_indicators`, a random `correlation_id`, and failures swallowed by `EventDispatcher.dispatch`; `/extract-symptoms` publishes nothing.
- [ ] **Step 3: Implement** `build_crisis_detected_event` and `DiagnosisService._publish_crisis(user_id, session_id, screen, turn_id) -> list[str]`: it awaits `asyncio.wait_for(self._event_publisher(event), 2.0)` once per turn, catches any exception into `crisis_event_publish_failed` (log `diagnosis_crisis_event_publish_failed` with the exception type only), returns `crisis_event_unpublished` with a WARNING when no publisher is configured, dispatches one `SafetyFlagRaisedEvent`, and increments `CRISIS_DETECTED.labels(source="diagnosis", level=...)` for every level at or above ELEVATED. A per-call flag guarantees the at-most-once rule when the pre-screen already published and the chain raises further. Do not route the crisis event through the dispatcher-to-bridge path.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/contracts -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.6: The CRISIS phase is server-side and sticky for the crisis watch, and a crisis turn asks the safety question
**Covers:** MVP-122, MVP-123
**Severity:** medium
**Wave:** 2
**Depends on:** WS-16.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-06 (default assumed: after HIGH/CRITICAL the session stays in CRISIS for the next 3 user turns or 30 minutes, whichever ends first; a new HIGH/CRITICAL re-arms the watch; no permanent lock); D-30 (diagnosis text is never shown to users, so the reply carries no resources list)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/models.py` (`SessionState` gains `crisis_watch_turns_left: int = 0`, `crisis_watch_until: datetime | None = None`, `phase_before_crisis: DiagnosisPhase | None = None`)
- Modify: `services/diagnosis_service/src/domain/service.py` (`DiagnosisServiceSettings.crisis_watch_turns: int = 3`, `crisis_watch_seconds: int = 1800`; the effective current phase is `session.phase`, the request's `current_phase` is ignored; `_apply_crisis_watch(session, screen, now) -> DiagnosisPhase`; `_generate_next_question` returns `CRISIS_SAFETY_QUESTION` for CRISIS before any `missing_info`; the reply is built from the final phase, never the input phase; module constant `CRISIS_SAFETY_QUESTION`)
- Modify: `services/diagnosis_service/src/infrastructure/redis_session_repository.py` (serialize and restore the three new fields)
- Test: `services/diagnosis_service/tests/test_mvp_123_sticky_crisis_phase.py` (new)

**Interfaces:**
- Consumes: WS-16.3 `DiagnosisRiskScreen.is_crisis`; WS-16.5 turn ordering.
- Produces: `CRISIS_SAFETY_QUESTION = "I want to make sure you're safe right now. Can you tell me more about how you're feeling?"`; settings `crisis_watch_turns`, `crisis_watch_seconds` (env `DIAGNOSIS_SERVICE_CRISIS_WATCH_TURNS`, `DIAGNOSIS_SERVICE_CRISIS_WATCH_SECONDS`); the session fields above, persisted in Redis.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_123_sticky_crisis_phase.py::test_mvp_123_client_phase_cannot_de_escalate`
```python
"""MVP-122 / MVP-123: a crisis turn asks the safety question; CRISIS stays server-side for the crisis watch."""
from __future__ import annotations
from uuid import uuid4
import fakeredis
import fakeredis.aioredis
import pytest
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.service import CRISIS_SAFETY_QUESTION, DiagnosisService, DiagnosisServiceSettings
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.infrastructure.redis_session_repository import build_session_store
from services.diagnosis_service.src.schemas import DiagnosisPhase

CRISIS, CALM = "I don't want to live anymore", "Can you give me some tips for sleeping better?"


def _svc(settings=None, store=None) -> DiagnosisService:
    return DiagnosisService(settings=settings or DiagnosisServiceSettings(), session_store=store,
                            symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
                            differential_generator=DifferentialGenerator(DifferentialSettings()))


async def _turn(svc, uid, sid, message, phase=DiagnosisPhase.RAPPORT):
    return await svc.assess(user_id=uid, session_id=sid, message=message, conversation_history=[], existing_symptoms=[],
                            current_phase=phase, current_differential=None, user_context={})


async def _started(svc):
    await svc.initialize()
    uid = uuid4()
    return uid, (await svc.start_session(uid, "assessment", {}, None)).session_id


@pytest.mark.asyncio
async def test_mvp_122_crisis_turn_asks_the_safety_question() -> None:
    svc = _svc()
    uid, sid = await _started(svc)
    result = await _turn(svc, uid, sid, CRISIS)
    assert result.phase == DiagnosisPhase.CRISIS and result.next_question == CRISIS_SAFETY_QUESTION
    assert "safe" in result.response_text.lower() and "duration" not in result.response_text.lower()


@pytest.mark.asyncio
async def test_mvp_123_client_phase_cannot_de_escalate() -> None:
    svc = _svc()
    uid, sid = await _started(svc)
    await _turn(svc, uid, sid, CRISIS)
    result = await _turn(svc, uid, sid, CALM, phase=DiagnosisPhase.RAPPORT)
    assert result.phase == DiagnosisPhase.CRISIS and result.next_question == CRISIS_SAFETY_QUESTION
    assert svc._active_sessions[sid].phase == DiagnosisPhase.CRISIS


@pytest.mark.asyncio
async def test_mvp_123_watch_ends_after_three_calm_turns() -> None:
    svc = _svc()
    uid, sid = await _started(svc)
    await _turn(svc, uid, sid, CRISIS)
    phases = [(await _turn(svc, uid, sid, CALM)).phase for _ in range(4)]
    assert phases == [DiagnosisPhase.CRISIS] * 3 + [DiagnosisPhase.RAPPORT]


@pytest.mark.asyncio
async def test_mvp_123_watch_expires_with_the_window() -> None:
    svc = _svc(DiagnosisServiceSettings(crisis_watch_seconds=0))
    uid, sid = await _started(svc)
    await _turn(svc, uid, sid, CRISIS)
    assert (await _turn(svc, uid, sid, CALM)).phase == DiagnosisPhase.RAPPORT


@pytest.mark.asyncio
async def test_mvp_123_new_worker_keeps_the_crisis_phase() -> None:
    server = fakeredis.FakeServer()
    store = lambda: build_session_store(fakeredis.aioredis.FakeRedis(server=server, decode_responses=True))
    first = _svc(store=store())
    uid, sid = await _started(first)
    await _turn(first, uid, sid, CRISIS)
    second = _svc(store=store())
    await second.initialize()
    assert (await _turn(second, uid, sid, CALM, phase=DiagnosisPhase.RAPPORT)).phase == DiagnosisPhase.CRISIS


@pytest.mark.asyncio
async def test_mvp_123_client_cannot_jump_the_phase_forward() -> None:
    svc = _svc()
    uid, sid = await _started(svc)
    result = await _turn(svc, uid, sid, CALM, phase=DiagnosisPhase.CLOSURE)
    assert result.phase in (DiagnosisPhase.RAPPORT, DiagnosisPhase.HISTORY)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_123_sticky_crisis_phase.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ImportError: cannot import name 'CRISIS_SAFETY_QUESTION'`; once defined, the calm turn after a crisis returns `phase=RAPPORT`, overwrites `session.phase` (the phase is taken from the request) and asks "Could you tell me more about duration?" because `missing_info` beats the CRISIS question; the new worker de-escalates the same way; the CLOSURE request is accepted as the current phase. (WS-16.3 already gives the crisis turn itself the CRISIS reply; this task makes it hold for the watch.)
- [ ] **Step 3: Implement** `_apply_crisis_watch`: on a crisis turn, if `session.phase != CRISIS` store `phase_before_crisis = session.phase`; set `crisis_watch_turns_left = settings.crisis_watch_turns` and `crisis_watch_until = now + crisis_watch_seconds`; return CRISIS. On a calm turn while `session.phase == CRISIS`: if `crisis_watch_turns_left > 0` and `now < crisis_watch_until`, decrement and return CRISIS; otherwise clear the watch and return `phase_before_crisis or RAPPORT`. Non-crisis phase advancement starts from `session.phase` (the confidence rule is unchanged until WS-16.15).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/test_diagnosis_vignette.py -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.7: Ownership errors return 403/404, state and challenge read through Redis, and session persistence never fails a crisis turn
**Covers:** MVP-258
**Severity:** low
**Wave:** 2
**Depends on:** WS-16.6
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/models.py` (`class DiagnosisSessionNotFound(LookupError)`, `class DiagnosisSessionForbidden(PermissionError)`)
- Modify: `services/diagnosis_service/src/domain/service.py` (`assess` and `end_session` raise `DiagnosisSessionNotFound`/`DiagnosisSessionForbidden` after the crisis publish; `get_session_state(session_id, user_id: UUID | None = None)` and `challenge_hypothesis` use `_load_session` and raise the same errors; `_persist_session(session) -> bool` catches `SessionPersistConflict`, `redis.exceptions.RedisError` and `OSError`, logs `diagnosis_session_persist_failed` and returns False; every caller adds `session_persist_failed` to `degraded`; `fail_loud` removed; a store read error in `_load_session` falls back to the in-process copy and adds `session_store_unavailable`)
- Modify: `services/diagnosis_service/src/infrastructure/redis_session_repository.py` (`save_session_cas_merge`: when `get_session` returns `None` the durable key is gone, so set `session.version = 0` and retry; supporting MVP-192, owner WS-15)
- Modify: `services/diagnosis_service/src/api.py` (`/assess`, `/session/end`, `/session/{id}/state` and `/challenge/{id}` map `DiagnosisSessionNotFound` to `HTTPException(404, detail={"code": "SESSION_NOT_FOUND", "message": "Diagnosis session not found"})` and `DiagnosisSessionForbidden` to `HTTPException(403, detail={"code": "USER_ID_MISMATCH", "message": "Session belongs to another user"})`)
- Test: `services/diagnosis_service/tests/test_mvp_258_session_ownership_and_cas.py` (new)
- Test: `services/diagnosis_service/tests/test_session_idor.py` (modify: `test_assess_rejects_non_owner` expects `DiagnosisSessionForbidden` and an unchanged owner session; the end and challenge cases expect the exceptions)
- Test: `services/diagnosis_service/tests/test_phase_c_session_reconcile.py` (modify: `test_crisis_turn_persist_raises_on_cas_exhaustion` becomes `test_crisis_turn_persist_failure_is_degraded`, asserting `"session_persist_failed" in result.degraded` and the cached CRISIS phase; ledger Ruling: reverses P2-9 because the crisis event is already durable in the outbox)

**Interfaces:**
- Consumes: WS-16.5 `event_publisher` and `CapturingPublisher`.
- Produces: `DiagnosisSessionNotFound`, `DiagnosisSessionForbidden`; HTTP codes `SESSION_NOT_FOUND` (404) and `USER_ID_MISMATCH` (403) under FastAPI's `{detail}` body (internal service); degraded reasons `session_persist_failed`, `session_store_unavailable`; the CAS rebase contract shared with WS-15.9 (a missing key is re-created, never an endless conflict).

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_258_session_ownership_and_cas.py::test_mvp_258_foreign_session_is_forbidden`
```python
"""MVP-258 / MVP-192 (diagnosis half): foreign and unknown sessions are errors, Redis is read through, persistence never fails a crisis."""
from __future__ import annotations
from uuid import uuid4
import fakeredis
import fakeredis.aioredis
import httpx
import pytest
from services.diagnosis_service.src.domain.models import DiagnosisSessionForbidden, DiagnosisSessionNotFound, SessionState
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.infrastructure.redis_session_repository import (
    DiagnosisSessionStore, SessionPersistConflict, build_session_store,
)
from services.diagnosis_service.src.schemas import DiagnosisPhase
from services.diagnosis_service.tests.runtime_fakes import CapturingPublisher


class _AlwaysConflictStore(DiagnosisSessionStore):
    async def save_session_cas_merge(self, session, max_retries=5):
        raise SessionPersistConflict(session.session_id)


def _svc(store=None, publisher=None) -> DiagnosisService:
    return DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()), session_store=store,
                            event_publisher=publisher.publish if publisher else None)


async def _assess(svc, uid, sid, message="I had a long day"):
    return await svc.assess(user_id=uid, session_id=sid, message=message, conversation_history=[], existing_symptoms=[],
                            current_phase=DiagnosisPhase.RAPPORT, current_differential=None, user_context={})


@pytest.mark.asyncio
async def test_mvp_258_foreign_session_is_forbidden() -> None:
    svc = _svc()
    await svc.initialize()
    owner = uuid4()
    start = await svc.start_session(owner, "assessment", {}, None)
    with pytest.raises(DiagnosisSessionForbidden):
        await _assess(svc, uuid4(), start.session_id)
    with pytest.raises(DiagnosisSessionForbidden):
        await svc.end_session(uuid4(), start.session_id, False)


@pytest.mark.asyncio
async def test_mvp_258_unknown_session_is_not_found_but_a_crisis_still_publishes() -> None:
    publisher = CapturingPublisher()
    svc = _svc(publisher=publisher)
    await svc.initialize()
    with pytest.raises(DiagnosisSessionNotFound):
        await _assess(svc, uuid4(), uuid4(), "I want to kill myself")
    assert len(publisher.events) == 1


@pytest.mark.asyncio
async def test_mvp_258_state_reads_through_redis_after_a_restart() -> None:
    server = fakeredis.FakeServer()
    store = lambda: build_session_store(fakeredis.aioredis.FakeRedis(server=server, decode_responses=True))
    first = _svc(store=store())
    await first.initialize()
    owner = uuid4()
    start = await first.start_session(owner, "assessment", {}, None)
    second = _svc(store=store())
    await second.initialize()
    state = await second.get_session_state(start.session_id, owner)
    assert state is not None and state["user_id"] == str(owner)
    with pytest.raises(DiagnosisSessionForbidden):
        await second.get_session_state(start.session_id, uuid4())


@pytest.mark.asyncio
async def test_mvp_192_cas_rebases_when_the_durable_key_is_gone() -> None:
    store = build_session_store(fakeredis.aioredis.FakeRedis(decode_responses=True))
    session = SessionState(user_id=uuid4(), version=2)
    await store.save_session_cas_merge(session)
    assert await store.get_session(session.session_id) is not None


@pytest.mark.asyncio
async def test_mvp_192_crisis_turn_never_fails_on_persistence() -> None:
    publisher = CapturingPublisher()
    svc = _svc(store=_AlwaysConflictStore(client=fakeredis.aioredis.FakeRedis(decode_responses=True)), publisher=publisher)
    await svc.initialize()
    uid = uuid4()
    start = await svc.start_session(uid, "assessment", {}, None)
    result = await _assess(svc, uid, start.session_id, "I want to end my life")
    assert result.phase == DiagnosisPhase.CRISIS and "session_persist_failed" in result.degraded
    assert len(publisher.events) == 1
```
Add `test_mvp_258_api_maps_ownership_to_status`: an ASGI client over `create_application()` with `get_current_user` overridden to user B and a session started for user A on `app.state.diagnosis_service`; POST `/assess` with B's `user_id` and A's `session_id` gives 403 with `detail.code == "USER_ID_MISMATCH"`; POST `/session/end` with an unknown session id gives 404 `SESSION_NOT_FOUND`; GET `/session/{unknown}/state` gives 404.
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_258_session_ownership_and_cas.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ImportError: cannot import name 'DiagnosisSessionForbidden'`; once defined, a foreign `/assess` returns a result with `response_text=''` and `safety_flags=['unauthorized_session']` (200 over HTTP), an unknown end returns an empty `SessionEndResult` (200), `get_session_state` takes no `user_id` (`TypeError`) and, once it does, the second worker gets `None` (in-memory dict only, no read-through), `save_session_cas_merge` raises `SessionPersistConflict` after five retries against the missing key, and the crisis turn re-raises `SessionPersistConflict`.
- [ ] **Step 3: Implement** the exceptions, the ordering (screen and publish before `_load_session`, ownership after), the `_load_session` fallback, the boolean `_persist_session`, the rebase and the route mapping (route-level `try/except` for the two domain exceptions only; no app-level handler).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/test_diagnosis_vignette.py -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.8: Session rows and records persist against the migrated schema, encrypted, under RLS, with the `diagnosis.completed` outbox row in the same transaction
**Covers:** MVP-128, MVP-253
**Severity:** medium
**Wave:** 2
**Depends on:** WS-16.2, WS-16.7, WS-09.7, WS-04.8, WS-06.5, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: repository SQL changes to match Alembic); D-02 (services connect as `solace_app`)
**Real infra:** yes
**Files:**
- Modify: `services/diagnosis_service/src/infrastructure/postgres_repository.py` (constructor `PostgresDiagnosisRepository(codec: PHICodec, schema: str = "public")`; `POOL_NAME = "default"`; every query through `ConnectionPoolManager.acquire("default")`, writes through `ConnectionPoolManager.transaction("default")`; drop the `FeatureFlags`/legacy-client branch and `_current_encryption_key_id`; `save_session` writes `session_metadata` (encrypted JSON of `{"symptoms": [...], "hypotheses": [...]}`) instead of the nonexistent `symptoms`/`hypotheses`/`metadata` columns and supplies `encryption_key_id=codec.active_key_id`; `save_record` passes JSONB values as Python objects (never `json.dumps`) and writes `requires_clinical_review` and `confidence_level`; `get_symptom_history` reads `session_metadata`; declare `DIAGNOSIS_PHI_FIELDS`; new `save_completed_session`)
- Modify: `services/diagnosis_service/src/infrastructure/repository.py` (`DiagnosisRepositoryPort.save_completed_session` abstract method; `RepositoryFactory` unchanged)
- Modify: `services/diagnosis_service/src/domain/entities.py` (`DiagnosisRecordEntity.requires_clinical_review: bool = False`, `confidence_level: str | None = None`)
- Modify: `services/diagnosis_service/src/domain/service.py` (`end_session` builds a `DiagnosisSessionEntity` from the live `SessionState` and, when a primary hypothesis exists, a `DiagnosisRecordEntity` plus `to_kafka_event(DiagnosisRecordedEvent(record_id=record.id, ...))`, then calls `save_completed_session(entity, record, event=..., publish=self._event_publisher)`; no record and no event when there is no primary (no fabricated 0.5/MILD); `_store_session_history` runs only after persistence succeeds)
- Modify: `services/diagnosis_service/src/events.py` (`to_kafka_event(DiagnosisRecordedEvent)`: `assessment_id=event.record_id`, `Confidence` mapped from the numeric confidence (<0.50 LOW, <0.70 MODERATE, <0.85 HIGH, else VERY_HIGH), `evidence_summary=""`, severity MINIMAL/MILD→MILD, MODERATE/MODERATELY_SEVERE→MODERATE, SEVERE→SEVERE, `stepped_care_level=0` documented as "not instrument-assessed", `requires_clinical_review=event.requires_clinical_review` (new field, default False))
- Modify: `services/diagnosis_service/src/main.py` (repository `PostgresDiagnosisRepository(codec=runtime.phi_codec or get_phi_codec())` when `runtime.pg_pool` exists; remove `event_dispatcher.subscribe_all(bridge.bridge_event)` because both cross-service events now go straight to the plane)
- Test: `tests/realinfra/diagnosis-service/test_mvp_128_records_real_pg.py` (new)
- Test: `services/diagnosis_service/tests/fixtures.py` (modify: `InMemoryDiagnosisRepository.save_completed_session` saves both and awaits `publish(event)` when given)
- Test: `services/diagnosis_service/tests/test_postgres_record_persistence.py` (modify: patch `ConnectionPoolManager.acquire` with the capturing connection and assert the bound `encryption_key_id` equals `codec.active_key_id` and `primary_diagnosis` starts with `enc:v1:`)
- Test: `services/diagnosis_service/tests/test_end_session_persistence.py` (modify: `_FailingRepo.save_completed_session` raises; add `test_end_session_failure_records_no_history`: after the raise `service._user_history` is empty, and a retry on a working repository records exactly one history entry)

**Interfaces:**
- Consumes: IF-13 `solace_security.phi_codec.{PHICodec, get_phi_codec}` with `encrypt(value, *, field, owner_id)`, `decrypt(...)`, `active_key_id` (WS-09.7); IF-03 `ConnectionPoolManager.acquire/transaction("default")` setting the RLS GUC, `solace_common.request_context.acting_as_user(user_id)` (WS-02/WS-07); IF-05 `publish(event, *, conn=)` (WS-06.5) and `solace_events.publisher.create_publisher(..., use_outbox=True, use_mock=True, postgres_pool=..., source_service="diagnosis-service")` (WS-06.1); IF-19 `pg_migrated` (WS-05.2); slot 020 (WS-16.2).
- Produces: `DIAGNOSIS_PHI_FIELDS = {"diagnosis_sessions": ("messages", "safety_flags", "summary", "session_metadata"), "diagnosis_records": ("primary_diagnosis", "dsm5_code", "icd11_code", "symptom_summary", "supporting_evidence", "differential_diagnoses", "clinician_notes")}` (JSONB PHI values stored as one encrypted JSON string; AAD `field=<column>`, `owner_id=user_id`); `async def save_completed_session(self, session: DiagnosisSessionEntity, record: DiagnosisRecordEntity | None, *, event: BaseEvent | None = None, publish: Callable[..., Awaitable[UUID]] | None = None) -> None` (one transaction: upsert the session row, insert the record, `await publish(event, conn=conn)`); the corrected `diagnosis.completed` payload. Answers WS-09.15 (CR-10) for `postgres_repository.py`.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/diagnosis-service/test_mvp_128_records_real_pg.py::test_mvp_128_end_session_writes_session_record_and_outbox_atomically`
```python
"""MVP-128 / MVP-253: end_session writes the session row, the encrypted record and its outbox row in one RLS-scoped transaction."""
from __future__ import annotations
import json
from uuid import UUID, uuid4
import asyncpg
import pytest
from solace_common.request_context import acting_as_user
from solace_events.errors import EventPublishError
from solace_events.publisher import create_publisher
from solace_security.phi_codec import PHICodec
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.infrastructure.postgres_repository import PostgresDiagnosisRepository
from services.diagnosis_service.src.schemas import DiagnosisPhase

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
MESSAGE = "I feel sad and hopeless, I can't sleep and I have no energy"


async def _insert_user(dsn_superuser: str) -> UUID:
    # If slot 012 added NOT NULL users columns without defaults, add them here with synthetic values.
    uid = uuid4()
    conn = await asyncpg.connect(dsn_superuser)
    try:
        await conn.execute("INSERT INTO users (id, created_at, updated_at, encryption_key_id, email, password_hash, display_name) "
                           "VALUES ($1, now(), now(), 'test-key', $2, 'x', 'Test User')", uid, f"{uid}@test.invalid")
    finally:
        await conn.close()
    return uid


@pytest.fixture
def phi_codec() -> PHICodec:
    return PHICodec(active_key_id="test-k1", keys={"test-k1": bytes(range(32))}, search_salt=b"s" * 16, allow_plaintext=False)


def _jsonb(value):
    """Superuser connections may return JSONB as text or as a decoded value."""
    return json.loads(value) if isinstance(value, str) and value[:1] in ('"', "{", "[") else value


def _svc(repo, publish) -> DiagnosisService:
    return DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
                            differential_generator=DifferentialGenerator(DifferentialSettings()),
                            repository=repo, event_publisher=publish)


async def _session(svc, uid):
    start = await svc.start_session(uid, "assessment", {}, None)
    await svc.assess(user_id=uid, session_id=start.session_id, message=MESSAGE, conversation_history=[], existing_symptoms=[],
                     current_phase=DiagnosisPhase.RAPPORT, current_differential=None, user_context={})
    return start.session_id


@pytest.mark.asyncio
async def test_mvp_128_end_session_writes_session_record_and_outbox_atomically(pg_migrated, phi_codec) -> None:
    user_a, user_b = await _insert_user(pg_migrated.dsn_superuser), await _insert_user(pg_migrated.dsn_superuser)
    publisher = create_publisher(use_outbox=True, use_mock=True, postgres_pool=pg_migrated.app_pool, source_service="diagnosis-service")
    repo = PostgresDiagnosisRepository(codec=phi_codec)
    svc = _svc(repo, publisher.publish)
    await svc.initialize()
    with acting_as_user(str(user_a)):
        sid = await _session(svc, user_a)
        end = await svc.end_session(user_a, sid, True)
    srow = (await pg_migrated.superuser_fetch(
        "SELECT summary, session_metadata, encryption_key_id, is_active FROM diagnosis_sessions WHERE id = $1", sid))[0]
    assert srow["summary"].startswith("enc:v1:") and srow["encryption_key_id"] == phi_codec.active_key_id and srow["is_active"] is False
    assert _jsonb(srow["session_metadata"]).startswith("enc:v1:")
    rrows = await pg_migrated.superuser_fetch("SELECT id, primary_diagnosis, dsm5_code FROM diagnosis_records WHERE session_id = $1", sid)
    assert len(rrows) == 1 and rrows[0]["primary_diagnosis"].startswith("enc:v1:") and rrows[0]["dsm5_code"].startswith("enc:v1:")
    outbox = await pg_migrated.superuser_fetch(
        "SELECT event_payload FROM event_outbox WHERE source_service = 'diagnosis-service' "
        "AND event_type = 'diagnosis.completed' AND aggregate_id = $1", user_a)
    assert len(outbox) == 1 and _jsonb(outbox[0]["event_payload"])["assessment_id"] == str(rrows[0]["id"])
    with acting_as_user(str(user_a)):
        records = await repo.list_user_records(user_a)
    assert [r.primary_diagnosis for r in records] == [end.final_differential.primary.name]
    with acting_as_user(str(user_b)):
        assert await repo.list_user_records(user_a) == [] and await repo.get_session(sid) is None


@pytest.mark.asyncio
async def test_mvp_128_outbox_failure_rolls_back_the_record(pg_migrated, phi_codec) -> None:
    user = await _insert_user(pg_migrated.dsn_superuser)

    async def _failing_publish(event, *, conn=None):
        raise EventPublishError("outbox down")

    svc = _svc(PostgresDiagnosisRepository(codec=phi_codec), _failing_publish)
    await svc.initialize()
    with acting_as_user(str(user)):
        sid = await _session(svc, user)
        with pytest.raises(EventPublishError):
            await svc.end_session(user, sid, True)
    assert await pg_migrated.superuser_fetch("SELECT 1 FROM diagnosis_records WHERE session_id = $1", sid) == []
    assert await pg_migrated.superuser_fetch("SELECT 1 FROM diagnosis_sessions WHERE id = $1", sid) == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/diagnosis-service/test_mvp_128_records_real_pg.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `TypeError: PostgresDiagnosisRepository.__init__() got an unexpected keyword argument 'codec'`; with the keyword accepted, `end_session` raises `asyncpg.exceptions.ForeignKeyViolationError: ... diagnosis_records_session_id_fkey` (no `diagnosis_sessions` row is ever written; `save_session` targets the nonexistent `symptoms`/`hypotheses`/`metadata` columns), the record would store plaintext under `encryption_key_id='primary'`, and the event goes through the dispatcher outside the transaction with `assessment_id` unrelated to the record id.
- [ ] **Step 3: Implement** the aligned repository and `save_completed_session` (`async with ConnectionPoolManager.transaction("default") as conn:` session upsert, record insert, `await publish(event, conn=conn)`; any exception rolls back and propagates); encrypt on write and decrypt on read with the column name as `field`; if WS-06's outbox-mode `publish` needs a started publisher, the test calls `await publisher.start()` (mock producer) before the session; `end_session` maps the live session (`messages`, `safety_flags`, `summary`, `recommendations`, symptoms and differential in `session_metadata`, `ended_at`, `is_active=False`). Add `test_mvp_128_delete_user_data_counts_only_own_rows` to the same file following the pattern above (two users, delete A under `acting_as_user(A)`, count equals A's rows, B's rows remain).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/diagnosis-service -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.9: Encrypted, capped diagnosis session blobs in Redis
**Covers:** MVP-252
**Severity:** low
**Wave:** 2
**Depends on:** WS-16.8, WS-09.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/infrastructure/redis_session_repository.py` (`DiagnosisSessionStore(settings=None, *, client, codec: PHICodec | None = None)`; strict environments without a codec raise `RuntimeError("diagnosis session store requires the PHI codec in strict environments")`; the stored value is the envelope `{"user_id", "version", "enc"}` with `enc = codec.encrypt(json.dumps(_session_to_dict(session)), field="diagnosis_session_blob", owner_id=session.user_id)`; CAS and `delete_session` read `version`/`user_id` from the envelope; legacy plaintext blobs are read only outside strict environments; settings `max_messages: int = 50` and `max_reasoning_steps: int = 8` cap what is serialized; `build_session_store(client, settings=None, *, codec=None)`)
- Modify: `services/diagnosis_service/src/domain/service.py` (`_update_session` trims `session.messages` to the last `max_messages` and `session.reasoning_history` to the last `max_reasoning_steps`)
- Modify: `services/diagnosis_service/src/main.py` (`build_session_store(runtime.redis, codec=runtime.phi_codec)`)
- Test: `services/diagnosis_service/tests/test_mvp_252_session_blob_encrypted.py` (new)

**Interfaces:**
- Consumes: IF-13 `PHICodec.encrypt/decrypt`, `get_phi_codec` (WS-09.7); `solace_infrastructure.runtime.is_strict_environment`.
- Produces: the encrypted envelope; `build_session_store(client, settings=None, *, codec=None)`; env `DIAGNOSIS_SESSION_STORE_MAX_MESSAGES`, `DIAGNOSIS_SESSION_STORE_MAX_REASONING_STEPS`. Answers WS-09.15 (CR-10) for `redis_session_repository.py`.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_252_session_blob_encrypted.py::test_mvp_252_blob_is_ciphertext`
```python
"""MVP-252: the diagnosis Redis blob holds no plaintext messages or flags and stops growing with every turn."""
from __future__ import annotations
import json
from uuid import uuid4
import fakeredis.aioredis
import pytest
from solace_security.phi_codec import PHICodec
from services.diagnosis_service.src.domain.models import SessionState
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.infrastructure.redis_session_repository import DiagnosisSessionStore
from services.diagnosis_service.src.schemas import DiagnosisPhase


@pytest.fixture
def phi_codec() -> PHICodec:
    return PHICodec(active_key_id="test-k1", keys={"test-k1": bytes(range(32))}, search_salt=b"s" * 16, allow_plaintext=False)


@pytest.mark.asyncio
async def test_mvp_252_blob_is_ciphertext(phi_codec) -> None:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = DiagnosisSessionStore(client=client, codec=phi_codec)
    session = SessionState(user_id=uuid4(), messages=[{"role": "user", "content": "my secret worry"}], safety_flags=["self_harm"])
    await store.save_session_cas_merge(session)
    raw = await client.get(f"solace:diagnosis:session:{session.session_id}")
    assert "my secret worry" not in raw and "self_harm" not in raw
    envelope = json.loads(raw)
    assert envelope["user_id"] == str(session.user_id) and envelope["enc"].startswith("enc:v1:") and envelope["version"] == 1
    loaded = await store.get_session(session.session_id)
    assert loaded.messages[-1]["content"] == "my secret worry" and loaded.safety_flags == ["self_harm"]


@pytest.mark.asyncio
async def test_mvp_252_blob_stops_growing(phi_codec) -> None:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    svc = DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
                           session_store=DiagnosisSessionStore(client=client, codec=phi_codec))
    await svc.initialize()
    uid = uuid4()
    start = await svc.start_session(uid, "assessment", {}, None)
    sizes = []
    for i in range(40):
        await svc.assess(user_id=uid, session_id=start.session_id, message=f"I feel tired and sad today {i}", conversation_history=[],
                         existing_symptoms=[], current_phase=DiagnosisPhase.RAPPORT, current_differential=None, user_context={})
        sizes.append(len(await client.get(f"solace:diagnosis:session:{start.session_id}")))
    assert max(sizes[30:]) < 1.2 * sizes[29]
    assert len(svc._active_sessions[start.session_id].messages) <= 50


def test_mvp_252_strict_store_requires_a_codec(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(RuntimeError, match="PHI codec"):
        DiagnosisSessionStore(client=fakeredis.aioredis.FakeRedis(decode_responses=True))
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_252_session_blob_encrypted.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `TypeError: DiagnosisSessionStore.__init__() got an unexpected keyword argument 'codec'`; the blob today is plaintext JSON containing `my secret worry` and `self_harm`, and it grows by roughly 14 KB per turn (messages and reasoning history appended without bound).
- [ ] **Step 3: Implement** the envelope, the caps (applied in `_session_to_dict` and in `_update_session`) and the strict-environment codec requirement.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_session_durability.py` and `test_phase_c_session_reconcile.py` construct stores without a codec under `ENVIRONMENT=test`).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.10: The differential uses the session's working symptom set, empty input yields no diagnosis, and temporal details reach the symptoms
**Covers:** MVP-124, MVP-125, MVP-131
**Severity:** medium
**Wave:** 2
**Depends on:** WS-16.9
**Implementer model:** opus
**Review:** standard
**Decision gate:** D-30 (default assumed: hypotheses are signals for clinicians only, so "no hypothesis" is a valid result)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/service.py` (step 2 hypothesizes from `working_symptoms` (prior session symptoms plus this turn's new ones, deduplicated by name); `session.differential` is replaced only when the turn produced a non-empty differential (never by a timeout or empty result); `end_session` writes no record and dispatches no `DiagnosisRecordedEvent` when the session has no symptoms)
- Modify: `services/diagnosis_service/src/domain/differential.py` (`generate([])` returns no hypotheses; a hypothesis whose `criteria_met` is empty is never produced; `_identify_missing_info` reads the working set)
- Modify: `services/diagnosis_service/src/domain/symptom_extractor.py` (`extract` attaches the turn's `temporal_info` to the symptoms it extracted: `duration`, `onset` (from `onset` or `onset_type`) and `frequency`, and the turn's `contextual_factors` as `triggers`; when the turn extracts no new symptom, the service applies the turn's temporal info to working-set symptoms that lack those fields)
- Test: `services/diagnosis_service/tests/test_mvp_124_working_set_differential.py` (new)

**Interfaces:**
- Consumes: WS-16.3 chain ordering; WS-16.8 `end_session` record path.
- Produces: the guarantee that `differential.primary is None` for symptom-free input and that `SymptomDTO.duration/onset/frequency/triggers` are filled from the user's words when stated.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_124_working_set_differential.py::test_mvp_124_small_talk_does_not_replace_the_differential`
```python
"""MVP-124 / MVP-125 / MVP-131: the differential reflects the whole session, empty input is not a diagnosis, stated timing is kept."""
from __future__ import annotations
from uuid import uuid4
import pytest
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.events import DiagnosisRecordedEvent
from services.diagnosis_service.src.schemas import DiagnosisPhase
from services.diagnosis_service.tests.fixtures import InMemoryDiagnosisRepository
from services.diagnosis_service.tests.runtime_fakes import CapturingDispatcher


async def _svc_and_session(repo=None, dispatcher=None):
    svc = DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
                           differential_generator=DifferentialGenerator(DifferentialSettings()),
                           repository=repo, event_dispatcher=dispatcher)
    await svc.initialize()
    uid = uuid4()
    return svc, uid, (await svc.start_session(uid, "assessment", {}, None)).session_id


async def _turn(svc, uid, sid, message):
    return await svc.assess(user_id=uid, session_id=sid, message=message, conversation_history=[], existing_symptoms=[],
                            current_phase=DiagnosisPhase.RAPPORT, current_differential=None, user_context={})


@pytest.mark.asyncio
async def test_mvp_124_small_talk_does_not_replace_the_differential() -> None:
    svc, uid, sid = await _svc_and_session()
    await _turn(svc, uid, sid, "I feel sad and hopeless and tired all the time")
    second = await _turn(svc, uid, sid, "I can't sleep at night either")
    assert {"depressed_mood", "sleep_disturbance"} <= set(second.differential.primary.criteria_met)  # working set, not this turn only
    await _turn(svc, uid, sid, "I'm fine, tell me a joke")
    kept = svc._active_sessions[sid].differential.primary
    assert kept is not None and kept.hypothesis_id == second.differential.primary.hypothesis_id


@pytest.mark.asyncio
async def test_mvp_125_greeting_only_session_records_nothing() -> None:
    repo, dispatcher = InMemoryDiagnosisRepository(), CapturingDispatcher()
    svc, uid, sid = await _svc_and_session(repo, dispatcher)
    result = await _turn(svc, uid, sid, "hello there")
    assert result.differential.primary is None
    await svc.end_session(uid, sid, True)
    assert await repo.list_user_records(uid) == []
    assert not [e for e in dispatcher.events if isinstance(e, DiagnosisRecordedEvent)]


@pytest.mark.asyncio
async def test_mvp_125_no_symptoms_no_hypotheses() -> None:
    assert (await DifferentialGenerator(DifferentialSettings()).generate([], {})).hypotheses == []


@pytest.mark.asyncio
async def test_mvp_131_stated_timing_is_attached_and_not_asked_again() -> None:
    svc, uid, sid = await _svc_and_session()
    result = await _turn(svc, uid, sid, "I have felt sad and hopeless for 3 weeks, since my divorce, it started suddenly")
    assert result.extracted_symptoms and all(s.duration == "for 3 weeks" for s in result.extracted_symptoms)
    assert all(s.onset for s in result.extracted_symptoms)
    assert "duration" not in (result.next_question or "").lower()


@pytest.mark.asyncio
async def test_mvp_131_follow_up_timing_reaches_earlier_symptoms() -> None:
    svc, uid, sid = await _svc_and_session()
    await _turn(svc, uid, sid, "I feel sad and hopeless")
    second = await _turn(svc, uid, sid, "It has been like this for 2 months")
    assert all(s.duration == "for 2 months" for s in svc._active_sessions[sid].symptoms)
    assert "duration" not in (second.next_question or "").lower()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_124_working_set_differential.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the second turn's primary meets only `sleep_disturbance` (step 2 sees only that turn's symptoms), and the small-talk turn overwrites `session.differential` with an unrelated hypothesis (Adjustment Disorder 0.1 in the register repro); "hello there" yields primary `Adjustment Disorder` (no required symptoms gives `required_met_ratio=1.0`) and `end_session` records it; `generate([])` returns Adjustment Disorder; the symptoms carry `duration=None` and every reply asks "Could you tell me more about duration?".
- [ ] **Step 3: Implement** the working-set input, the empty and no-criteria guards, the conditional differential update, the no-symptom end-session rule and the temporal attachment.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_rev30_reconcile.py` keeps its working-set calibration assertions).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.11: The diagnosis crisis vignette runs the real step 1 and asserts the published event
**Covers:** MVP-360
**Severity:** medium
**Wave:** 2
**Depends on:** WS-16.10
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `tests/integration/test_diagnosis_vignette.py` (`TestDiagnosisCrisisPhaseRouting` no longer replaces any `_stepN_*` method: module helper `_run_crisis_vignette(service, publisher, dispatcher)` runs the real `assess()` of a service built by `_crisis_service(...)` over a real `SymptomExtractor` and `DifferentialGenerator` with a stub LLM; `test_suicidal_ideation_routes_to_crisis_phase` calls it; a mutation meta-test proves the vignette fails when publishing is silenced; the docstring describes the published `CrisisDetectedEvent`; `TestSessionMessagesPersistence` unchanged)
- Test: `tests/integration/test_diagnosis_vignette.py`

**Interfaces:**
- Consumes: WS-16.5 `event_publisher`, `DiagnosisService._publish_crisis`, `SafetyFlagRaisedEvent`; `services.diagnosis_service.tests.runtime_fakes.{CapturingPublisher, CapturingDispatcher}`.
- Produces: the in-process proof that `assess()` emits the diagnosis crisis event, and that the vignette notices when it does not.

- [ ] **Step 1: Write the failing test** in `tests/integration/test_diagnosis_vignette.py::TestDiagnosisCrisisPhaseRouting::test_mvp_360_vignette_catches_a_silenced_publisher`. First write the helper with only the old result-field checks (exactly what the faked vignette asserted), plus the two tests:
```python
from uuid import UUID
from solace_clinical import CATEGORIES
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.events import SafetyFlagRaisedEvent
from services.diagnosis_service.tests.runtime_fakes import CapturingDispatcher, CapturingPublisher

TURN = UUID("00000000-0000-4000-e000-000000000001")


class _StubLLM:
    is_available = True

    async def generate(self, **kwargs):
        return '{"symptoms": [], "hypotheses": []}'


def _crisis_service(publisher, dispatcher) -> DiagnosisService:
    return DiagnosisService(DiagnosisServiceSettings(),
                            symptom_extractor=SymptomExtractor(SymptomExtractorSettings(), llm_client=_StubLLM()),
                            differential_generator=DifferentialGenerator(DifferentialSettings(), llm_client=_StubLLM()),
                            event_dispatcher=dispatcher, event_publisher=publisher.publish)


async def _run_crisis_vignette(service, publisher, dispatcher) -> None:
    await service.initialize()
    user_id = uuid4()
    start = await service.start_session(user_id, "assessment", {}, None)
    result = await service.assess(user_id=user_id, session_id=start.session_id,
                                  message="I don't want to live anymore, I have a plan", conversation_history=[],
                                  existing_symptoms=[], current_phase=DiagnosisPhase.HISTORY, current_differential=None,
                                  user_context={}, turn_id=TURN)
    assert result.phase == DiagnosisPhase.CRISIS
    assert {"suicidal_ideation", "suicide_plan"} & set(result.safety_flags)
    # Step 3 adds the publisher and dispatcher assertions here.


class TestDiagnosisCrisisPhaseRouting:
    @pytest.mark.asyncio
    async def test_suicidal_ideation_routes_to_crisis_phase(self) -> None:
        publisher, dispatcher = CapturingPublisher(), CapturingDispatcher()
        await _run_crisis_vignette(_crisis_service(publisher, dispatcher), publisher, dispatcher)

    @pytest.mark.asyncio
    async def test_mvp_360_vignette_catches_a_silenced_publisher(self, monkeypatch) -> None:
        """MVP-360: a change that stops assess() from publishing must turn the vignette red."""
        async def _silenced(self, *args, **kwargs):
            return []
        monkeypatch.setattr(DiagnosisService, "_publish_crisis", _silenced)
        publisher, dispatcher = CapturingPublisher(), CapturingDispatcher()
        with pytest.raises(AssertionError):
            await _run_crisis_vignette(_crisis_service(publisher, dispatcher), publisher, dispatcher)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/test_diagnosis_vignette.py -o "addopts=" -p no:logging -q`
Expected: FAIL in `test_mvp_360_vignette_catches_a_silenced_publisher` with `Failed: DID NOT RAISE <class 'AssertionError'>`: checks on `result.phase` and `result.safety_flags` alone stay green when the crisis event is never published, which is the false assurance MVP-360 describes (the old vignette additionally faked `_step1_analyze`, so it never reached the dispatch path at all).
- [ ] **Step 3: Implement** only the test: append to `_run_crisis_vignette` the event assertions `assert len(publisher.events) == 1`; `event = publisher.events[0]`; `assert event.event_type == "safety.crisis.detected" and event.metadata.correlation_id == TURN`; `assert set(event.trigger_indicators) <= CATEGORIES and {"suicidal_ideation", "suicide_plan"} & set(event.trigger_indicators)`; `assert "have a plan" not in event.model_dump_json()`; `assert len([e for e in dispatcher.events if isinstance(e, SafetyFlagRaisedEvent)]) == 1`. No production change in this task.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.12: Diagnosis satisfies the orchestrator contracts and `/status` requires `service:read:diagnosis`
**Covers:** none (supporting MVP-022 for WS-11 and MVP-314 for WS-07)
**Severity:** blocker (supporting MVP-022)
**Wave:** 2
**Depends on:** WS-16.11, WS-11.10 (writes `tests/contracts/orchestrator__diagnosis/{session_start,assess,session_end}.json`, W2), WS-07.4, WS-05.7 (IF-20 `verify_provider`, W1)
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-03 (default assumed: receivers resolve service permissions from `SERVICE_PERMISSIONS`; the orchestrator's obo token is accepted by `get_current_user`)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/api.py` (`GET /status` uses `Depends(require_service_permission(ServicePermission.READ_DIAGNOSIS.value))` instead of bare `get_current_service`)
- Create: `services/diagnosis_service/tests/test_mvp_022_contract_orchestrator_diagnosis.py` (provider verification requested by WS-11.10)
- Test: `services/diagnosis_service/tests/test_mvp_314_status_permission.py` (new; requested by WS-07.9)

**Interfaces:**
- Consumes: IF-20 `solace_testing.contracts.{load_contracts, verify_provider(app, contracts_dir, *, auth_headers)}` (WS-05.7) and WS-11.10's `tests/contracts/orchestrator__diagnosis/{session_start,assess,session_end}.json`; IF-04 `require_service_permission`, `find_unguarded_service_routes(app) -> list[str]` (WS-07.4), `ServicePermission.READ_DIAGNOSIS`. The provider test seeds sessions from the contracts' own `request.body_example` ids, so it passes whether or not WS-11 adopts the fixed ids of request R6.
- Produces: provider verification for WS-11.10 and for WS-05.10 (request 10 of WS-05: the `orchestrator__diagnosis` pair's `provider_verification` hop, found by the strings `verify_provider(` and `orchestrator__diagnosis`); a diagnosis app with no unguarded service route for WS-07.9.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_314_status_permission.py::test_mvp_314_no_unguarded_service_route`
```python
"""MVP-314 (supporting): every service-token route in diagnosis checks a permission."""
from __future__ import annotations
from solace_security.middleware import find_unguarded_service_routes
from services.diagnosis_service.src.main import create_application


def test_mvp_314_no_unguarded_service_route() -> None:
    assert find_unguarded_service_routes(create_application()) == []
```
And in `services/diagnosis_service/tests/test_mvp_022_contract_orchestrator_diagnosis.py::test_mvp_022_provider_satisfies_orchestrator_contracts`:
```python
"""MVP-022 (supporting): the diagnosis provider satisfies the orchestrator's contract files (IF-11, IF-20)."""
from __future__ import annotations
from pathlib import Path
from uuid import UUID
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from solace_testing.contracts import load_contracts, verify_provider
from services.diagnosis_service.src.domain.models import SessionState
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.main import create_application

CONTRACTS = Path(__file__).resolve().parents[3] / "tests" / "contracts" / "orchestrator__diagnosis"
OTHER = "00000000-0000-4000-a000-000000000002"
FOREIGN_SID = UUID("00000000-0000-4000-d000-000000000003")  # used only once R6's 403 interaction exists


def test_mvp_022_provider_satisfies_orchestrator_contracts() -> None:
    contracts = load_contracts(CONTRACTS)
    assert len(contracts) >= 3  # session_start, assess and session_end (WS-11.10)
    bodies = [c.request.body_example or {} for c in contracts]
    user = next(b["user_id"] for b in bodies if "user_id" in b)
    service = DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()))
    service._initialized = True
    seeds = {(UUID(b["session_id"]), b.get("user_id", user)) for b in bodies if "session_id" in b} | {(FOREIGN_SID, OTHER)}
    for sid, owner in seeds:  # every session a contract addresses exists and belongs to its example user
        service._active_sessions[sid] = SessionState(session_id=sid, user_id=UUID(owner))
    app = create_application()
    app.state.diagnosis_service = service
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=user, token_type=TokenType.SERVICE, roles=["user"], metadata={"via_service": "orchestrator-service"})
    verify_provider(app, CONTRACTS, auth_headers=lambda interaction: {
        "Authorization": "Bearer obo", "X-Request-ID": "r-1", "X-Correlation-ID": "c-1",
        "X-Turn-ID": "00000000-0000-4000-e000-000000000001", "Idempotency-Key": "k-1"})
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_314_status_permission.py services/diagnosis_service/tests/test_mvp_022_contract_orchestrator_diagnosis.py -o "addopts=" -p no:logging -q`
Expected: FAIL on `find_unguarded_service_routes(...) == ['GET /api/v1/diagnosis/status']` (any signed service token reads operational internals). The contract verification is the acceptance gate for WS-11.10: on the W2-start tree `load_contracts` finds no `orchestrator__diagnosis` files (nothing in the diagnosis suite checks the orchestrator's calls, MVP-022), and against WS-11.10's files the pre-W2 provider fails on `phase`/`safety_flags` drift and the 200-with-blank-body foreign-session case; on this tree it must pass, and any listed mismatch is fixed in `api.py`/`schemas.py` here, never by editing the contract.
- [ ] **Step 3: Implement** the permission dependency on `/status`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/contracts -o "addopts=" -p no:logging -q`. Expected: all pass (`test_api_integration.py` overrides `get_current_service` with an `AuthenticatedService` whose permissions include `service:read:diagnosis`).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.13: Erase every diagnosis store on `user.deleted`, including in-process copies
**Covers:** MVP-121
**Severity:** medium
**Wave:** 3
**Depends on:** WS-16.12, WS-14.13 (IF-16 `register_user_erasure_consumer`, W3 phase A), WS-06.8 (`ServiceEventPlane.subscribe` may be called before or after `start()`; a late subscription starts its consumer at once, W1), WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: PHI is hard-deleted everywhere; the account is never re-created, so an in-process tombstone is safe)
**Real infra:** yes
**Files:**
- Modify: `services/diagnosis_service/src/domain/service.py` (`async def purge_user_caches(self, user_id: UUID) -> None` clears the user's `_active_sessions`, `_user_history`, `_user_session_counts` and dispatcher log entries and records a bounded tombstone; `start_session`, `assess` and `extract_symptoms` for a tombstoned user raise `DiagnosisSessionNotFound` without persisting; `delete_user_data` runs inside `with acting_as_user(str(user_id)):` and calls `purge_user_caches`)
- Modify: `services/diagnosis_service/src/events.py` (`EventDispatcher._event_log` becomes `collections.deque(maxlen=1000)`; `EventDispatcher.purge_user(user_id: UUID) -> int`)
- Modify: `services/diagnosis_service/src/infrastructure/postgres_repository.py` (`delete_user_data` deletes `diagnosis_records`, `diagnosis_hypotheses`, `diagnosis_symptoms`, then `diagnosis_sessions`, so cascades do not hide counts)
- Modify: `services/diagnosis_service/src/main.py` (`build_diagnosis_components` calls `register_user_erasure_consumer(runtime.event_plane, service_name="diagnosis-service", erasure=build_user_data_erasure(repository, session_store=session_store), purge_caches=service.purge_user_caches)` when the plane and the repository exist; module-level import so tests can patch it; supporting edit for MVP-029, owner WS-14)
- Test: `services/diagnosis_service/tests/test_mvp_029_erasure_purges_process_state.py` (new)
- Test: `tests/realinfra/diagnosis-service/test_mvp_029_diagnosis_erasure_all_stores.py` (new)

**Interfaces:**
- Consumes: IF-16 `solace_infrastructure.gdpr.consumer.register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches=None)`; IF-04 `acting_as_user`; IF-19 `pg_migrated`, `redis_url`; WS-16.8 repository; WS-16.9 encrypted store.
- Produces: `DiagnosisService.purge_user_caches(user_id: UUID) -> Awaitable[None]` (the IF-16 `purge_caches` callable); `EventDispatcher.purge_user(user_id) -> int`; the diagnosis store names `diagnosis_postgres` and `diagnosis_redis_session` in the erasure report.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_029_erasure_purges_process_state.py::test_mvp_029_erase_purges_every_in_process_copy`
```python
"""MVP-121 / MVP-029 (diagnosis part): erasure removes the user's in-process copies and the consumer is registered."""
from __future__ import annotations
from uuid import uuid4
import fakeredis.aioredis
import pytest
from services.diagnosis_service.src import main as diagnosis_main
from services.diagnosis_service.src.domain.models import DiagnosisSessionNotFound
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.events import EventDispatcher
from services.diagnosis_service.src.schemas import DiagnosisPhase
from services.diagnosis_service.tests.fixtures import InMemoryDiagnosisRepository
from services.diagnosis_service.tests.runtime_fakes import fake_runtime


@pytest.mark.asyncio
async def test_mvp_029_erase_purges_every_in_process_copy() -> None:
    dispatcher = EventDispatcher()
    svc = DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
                           repository=InMemoryDiagnosisRepository(), event_dispatcher=dispatcher)
    await svc.initialize()
    uid = uuid4()
    start = await svc.start_session(uid, "assessment", {}, None)
    await svc.assess(user_id=uid, session_id=start.session_id, message="I feel sad and tired", conversation_history=[],
                     existing_symptoms=[], current_phase=DiagnosisPhase.RAPPORT, current_differential=None, user_context={})
    await svc.delete_user_data(uid)
    assert start.session_id not in svc._active_sessions and uid not in svc._user_history
    assert uid not in svc._user_session_counts and not [e for e in dispatcher.get_event_log() if e.user_id == uid]
    with pytest.raises(DiagnosisSessionNotFound):
        await svc.start_session(uid, "assessment", {}, None)


@pytest.mark.asyncio
async def test_mvp_029_erasure_consumer_is_registered(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(diagnosis_main, "register_user_erasure_consumer", lambda plane, **kw: calls.append((plane, kw)))
    from solace_security.phi_codec import PHICodec
    codec = PHICodec(active_key_id="test-k1", keys={"test-k1": bytes(range(32))}, search_salt=b"s" * 16, allow_plaintext=False)
    from types import SimpleNamespace
    from services.diagnosis_service.tests.runtime_fakes import CapturingPublisher
    plane = SimpleNamespace(publish=CapturingPublisher().publish)  # the composition only reads plane.publish
    components = await diagnosis_main.build_diagnosis_components(
        fake_runtime(redis=fakeredis.aioredis.FakeRedis(decode_responses=True), pg_pool=object(), event_plane=plane, phi_codec=codec),
        diagnosis_main.DiagnosisServiceAppSettings())
    assert calls and calls[0][0] is plane and calls[0][1]["service_name"] == "diagnosis-service"
    assert calls[0][1]["purge_caches"] == components.service.purge_user_caches


def test_mvp_121_dispatcher_log_is_bounded() -> None:
    assert EventDispatcher()._event_log.maxlen == 1000
```
In `tests/realinfra/diagnosis-service/test_mvp_029_diagnosis_erasure_all_stores.py::test_mvp_029_diagnosis_erasure_removes_every_store` (helpers `_insert_user` and `phi_codec` as in WS-16.8, plus `redis_url`): for users A and B, end one completed session each through `PostgresDiagnosisRepository(codec=phi_codec)` under `acting_as_user`, and save one live session each through `DiagnosisSessionStore(client=redis.asyncio.from_url(redis_url, decode_responses=True), codec=phi_codec)`; run `build_user_data_erasure(repo, session_store=store).erase(str(user_a))` under `acting_as_user(str(user_a))`; assert `report.complete is True` and `report.deleted == {"diagnosis_postgres": 2, "diagnosis_redis_session": 1}`; as superuser `diagnosis_sessions` and `diagnosis_records` hold zero rows for A and one each for B; Redis holds no key containing A's id and still holds B's session key.
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_029_erasure_purges_process_state.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the dispatcher log still holds the user's events, the post-erasure `start_session` succeeds (no tombstone), `diagnosis_main` has no `register_user_erasure_consumer`, and `_event_log` is a plain list. The real-infra file fails on `deleted["diagnosis_postgres"] == 1`: deleting sessions first cascades the records away, so the count hides them.
- [ ] **Step 3: Implement** `purge_user_caches`, the tombstone (bounded dict of erased user ids), `EventDispatcher.purge_user`, the deleter order and the consumer registration.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command, then `$env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/diagnosis-service/test_mvp_029_diagnosis_erasure_all_stores.py -o "addopts=" -p no:logging -q`. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/diagnosis-service -o "addopts=" -p no:logging -q`. Expected: all pass (`test_gdpr_erasure_wiring.py` keeps the store names).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.14: Clinicians reach only authorized subjects, PHI routes are audited, and diagnosis logs carry no clinical attributes
**Covers:** none (supporting MVP-120/MVP-141 for WS-07 and MVP-091 for WS-09)
**Severity:** blocker (supporting MVP-091)
**Wave:** 3
**Depends on:** WS-16.13, WS-07.8, WS-09.11, WS-09.12, WS-04.10
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-21 (default assumed: a clinician reaches a user's diagnosis data only when assigned or acknowledging a recent episode; admins have no PHI read or delete path through diagnosis routes; the RLS user becomes the data owner only after the check); D-10 (audit rows hold ids and action codes only)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/api.py` (every handler that compared `request.user_id` with `current_user.user_id` and bypassed for `CLINICIAN`/`ADMIN` calls `await authorize_subject(current_user, <subject user id>)` before any domain call; `get_session_state` and `challenge` authorize against the session owner; `_audit_phi(event_type, principal, subject_user_id, resource_type, resource_id)` calls `audit(...)` after authorization: `PHI_WRITE` for assess, extract, differential, start, end and challenge; `PHI_READ` for history and state; `PHI_DELETE` for the user delete; log lines `assessment_requested`, `symptom_extraction_requested`, `differential_requested`, `history_requested` keep only route-level ids (no `phase`, counts or clinical fields next to `user_id`))
- Modify: `services/diagnosis_service/src/domain/service.py` (`diagnosis_update_unknown_session`, `session_started`, `session_ended`, `assessment_timeout` and `user_data_deleted` log ids and counts only; no `safety_flags`, categories, hypothesis names or `user_id` next to a level or phase)
- Modify: `services/diagnosis_service/src/domain/symptom_extractor.py` (no log carries a risk category; `risk_indicator_detected` logs `count`)
- Test: `services/diagnosis_service/tests/test_mvp_091_subject_scope_audit_logs.py` (new)

**Interfaces:**
- Consumes: IF-04 W3 `solace_security.middleware.authorize_subject(principal, subject_user_id, *, checker=None)` and `solace_security.authorization.SubjectAccessChecker` (WS-07.8); IF-14 `solace_security.audit_runtime.audit(...)`, `solace_security.audit.{AuditEventType, AuditOutcome}` with `PHI_READ`, `PHI_WRITE`, `PHI_DELETE` (WS-09.11/WS-09.12).
- Produces: the diagnosis part of the W3 isolation matrix (unassigned clinician denied) and PHI-access audit trail.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_091_subject_scope_audit_logs.py::test_mvp_120_unassigned_clinician_is_denied`
```python
"""MVP-120/141/091 (supporting): diagnosis routes authorize the subject, audit PHI access and log no clinical attributes."""
from __future__ import annotations
import functools
import json
from uuid import uuid4
import httpx
import pytest
import structlog
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, authorize_subject, get_current_user
from services.diagnosis_service.src import api as diagnosis_api
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.main import create_application
from services.diagnosis_service.src.schemas import DiagnosisPhase

FORBIDDEN_TEXT = ("kill myself", "suicidal_ideation", "Depress", "hopeless")


class _DenyAll:
    async def is_assigned(self, clinician_id, subject_user_id):
        return False

    async def acknowledged_recent_episode(self, clinician_id, subject_user_id, *, within_hours=72):
        return False


async def _client(principal, service):
    app = create_application()
    app.state.diagnosis_service = service
    app.dependency_overrides[get_current_user] = lambda: principal
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def _service():
    svc = DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()))
    await svc.initialize()
    return svc


@pytest.mark.asyncio
async def test_mvp_120_unassigned_clinician_is_denied(monkeypatch) -> None:
    monkeypatch.setattr(diagnosis_api, "authorize_subject", functools.partial(authorize_subject, checker=_DenyAll()), raising=False)
    clinician = AuthenticatedUser(user_id=str(uuid4()), token_type=TokenType.ACCESS, roles=["clinician"])
    async with await _client(clinician, await _service()) as client:
        resp = await client.post("/api/v1/diagnosis/extract-symptoms", json={"user_id": str(uuid4()), "session_id": str(uuid4()), "message": "x"})
    assert resp.status_code == 403 and resp.json()["detail"]["code"] == "SUBJECT_FORBIDDEN"


@pytest.mark.asyncio
async def test_mvp_091_phi_routes_write_audit_rows(monkeypatch) -> None:
    calls = []

    async def _recorder(event_type, **kwargs):
        calls.append((getattr(event_type, "name", str(event_type)), kwargs))

    monkeypatch.setattr(diagnosis_api, "audit", _recorder, raising=False)
    uid = uuid4()
    svc = await _service()
    start = await svc.start_session(uid, "assessment", {}, None)
    me = AuthenticatedUser(user_id=str(uid), token_type=TokenType.ACCESS, roles=["user"])
    async with await _client(me, svc) as client:
        await client.post("/api/v1/diagnosis/assess", json={"user_id": str(uid), "session_id": str(start.session_id), "message": "I feel low"})
        await client.get(f"/api/v1/diagnosis/session/{start.session_id}/state")
    names = [c[0] for c in calls]
    assert "PHI_WRITE" in names and "PHI_READ" in names
    assert all(c[1]["subject_user_id"] == str(uid) and "I feel low" not in json.dumps(c[1], default=str) for c in calls)


@pytest.mark.asyncio
async def test_mvp_091_no_clinical_attributes_in_logs() -> None:
    svc = await _service()
    uid = uuid4()
    with structlog.testing.capture_logs() as logs:
        start = await svc.start_session(uid, "assessment", {}, None)
        await svc.assess(user_id=uid, session_id=start.session_id, message="I want to kill myself and feel hopeless",
                         conversation_history=[], existing_symptoms=[], current_phase=DiagnosisPhase.RAPPORT,
                         current_differential=None, user_context={})
        await svc.end_session(uid, start.session_id, True)
    assert logs
    for entry in logs:
        blob = json.dumps(entry, default=str)
        assert not {"safety_flags", "categories", "risk_type", "primary", "diagnosis"} & set(entry), entry["event"]
        assert not any(text in blob for text in FORBIDDEN_TEXT), entry["event"]
        if {"level", "phase"} & set(entry):
            assert "user_id" not in entry, entry["event"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_091_subject_scope_audit_logs.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the unassigned clinician gets 200 (the role-only `Role.CLINICIAN` bypass), no route calls `audit`, and the log capture finds `risk_indicator_detected` with `risk_type='suicidal_ideation'` and `assessment_requested`/`session_started` carrying `user_id` next to `phase`.
- [ ] **Step 3: Implement** the authorization calls, `_audit_phi` (never raises into the route, per IF-14) and the log changes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_api_integration.py`'s `_mock_user` acts on its own `TEST_USER_ID`; cases that posted another user's id as a clinician switch to the caller's id).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.15: Phase advancement follows met diagnostic criteria, one step per two turns, and Adjustment Disorder never drives it
**Covers:** MVP-048
**Severity:** high
**Wave:** 5
**Depends on:** WS-16.14, WS-12.3 (§11 of `docs/CLINICAL-VALIDATION.md`, the D-05 record that request R8 extends with the diagnosis rules, W1)
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: the signed-off rule is "advance one phase only when the primary hypothesis meets its required and minimum criteria and at least `min_turns_per_phase` turns were spent in the current phase"; calibrated confidence is reported but no longer gates phases)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/schemas.py` (`HypothesisDTO.meets_minimum_criteria: bool = False`)
- Modify: `services/diagnosis_service/src/domain/differential.py` (`meets_minimum_criteria = bool(required) and required <= symptom_names and len(met) >= min_criteria`; LLM hypotheses keep `False`)
- Modify: `services/diagnosis_service/src/domain/models.py` (`SessionState.turns_in_phase: int = 0`)
- Modify: `services/diagnosis_service/src/domain/service.py` (`DiagnosisServiceSettings.min_turns_per_phase: int = 2`; `_determine_next_phase(current, primary: HypothesisDTO | None, *, turns_in_phase: int) -> DiagnosisPhase` for non-crisis turns; `turns_in_phase` resets on every phase change; the crisis watch of WS-16.6 is unchanged; step 4 orders hypotheses by `(meets_minimum_criteria, confidence)` descending and drops Adjustment Disorder whenever another hypothesis meets its minimum criteria, because DSM-5 makes it a residual diagnosis)
- Modify: `services/diagnosis_service/src/infrastructure/redis_session_repository.py` (serialize `turns_in_phase`)
- Test: `services/diagnosis_service/tests/test_mvp_048_phase_gate.py` (new)

**Interfaces:**
- Consumes: WS-16.10 working-set differential; WS-16.6 crisis watch.
- Produces: `HypothesisDTO.meets_minimum_criteria`; setting `min_turns_per_phase` (env `DIAGNOSIS_SERVICE_MIN_TURNS_PER_PHASE`).

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_048_phase_gate.py::test_mvp_048_rich_depressive_presentation_leaves_rapport`
```python
"""MVP-048: phases advance on met criteria at a bounded pace; intensifier-driven Adjustment Disorder cannot run to CLOSURE."""
from __future__ import annotations
from decimal import Decimal
from uuid import uuid4
import pytest
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.schemas import DiagnosisPhase, SymptomDTO, SymptomType

RICH_MDD = ("I feel sad and empty, I've lost interest in everything, I can't sleep, I'm exhausted, "
            "I can't concentrate, I feel guilty and I'm not eating")
ADJ_INTENSIFIER = "I am extremely sad, extremely anxious, irritable, I can't concentrate, I can't sleep"
ORDER = [DiagnosisPhase.RAPPORT, DiagnosisPhase.HISTORY, DiagnosisPhase.ASSESSMENT, DiagnosisPhase.DIAGNOSIS, DiagnosisPhase.CLOSURE]


async def _phases(message: str, turns: int) -> list:
    svc = DiagnosisService(symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
                           differential_generator=DifferentialGenerator(DifferentialSettings()))
    await svc.initialize()
    uid = uuid4()
    sid = (await svc.start_session(uid, "assessment", {}, None)).session_id
    out = []
    for _ in range(turns):
        result = await svc.assess(user_id=uid, session_id=sid, message=message, conversation_history=[], existing_symptoms=[],
                                  current_phase=DiagnosisPhase.RAPPORT, current_differential=None, user_context={})
        out.append((result.phase, result.differential.primary))
    return out


@pytest.mark.asyncio
async def test_mvp_048_rich_depressive_presentation_leaves_rapport() -> None:
    phases = await _phases(RICH_MDD, 2)
    assert phases[1][0] == DiagnosisPhase.HISTORY and "Depress" in phases[1][1].name


@pytest.mark.asyncio
async def test_mvp_048_intensifiers_cannot_run_the_session_to_closure() -> None:
    phases = await _phases(ADJ_INTENSIFIER, 4)
    idx = [ORDER.index(p) for p, _ in phases]
    assert DiagnosisPhase.CLOSURE not in [p for p, _ in phases] and max(idx) <= 2
    assert idx[0] == 0 and all(b - a <= 1 for a, b in zip(idx, idx[1:]))
    assert not any(b > a and c > b for a, b, c in zip(idx, idx[1:], idx[2:]))  # never two advances in a row
    for (phase, primary), prev in zip(phases[1:], idx):
        if primary is not None and "Adjustment" in primary.name:
            assert ORDER.index(phase) == prev


@pytest.mark.asyncio
async def test_mvp_048_adjustment_never_outranks_a_disorder_that_meets_criteria() -> None:
    (_, first), = await _phases(RICH_MDD, 1)
    assert first is not None and first.meets_minimum_criteria and "Adjustment" not in first.name


@pytest.mark.asyncio
async def test_mvp_048_no_required_symptoms_never_meets_minimum() -> None:
    names = ["depressed_mood", "anxiety", "irritability", "concentration_difficulty", "sleep_disturbance"]
    symptoms = [SymptomDTO(symptom_id=uuid4(), name=n, description="d", symptom_type=SymptomType.EMOTIONAL, confidence=Decimal("0.8")) for n in names]
    hyps = (await DifferentialGenerator(DifferentialSettings()).generate(symptoms, {})).hypotheses
    adjustment = [h for h in hyps if "Adjustment" in h.name]
    assert adjustment and adjustment[0].meets_minimum_criteria is False
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_048_phase_gate.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the rich MDD session stays in RAPPORT (calibrated MDD about 0.41, below the 0.70 gate, because base rates 0.07/0.03 crush the posterior), the intensifier session reaches CLOSURE by turn 4 one step per turn through Adjustment Disorder at about 0.76, and `HypothesisDTO` has no `meets_minimum_criteria`.
- [ ] **Step 3: Implement** the flag in the differential, the residual ordering in step 4, the turn counter and the gate (advance only when `primary is not None and primary.meets_minimum_criteria and turns_in_phase + 1 >= min_turns_per_phase`). Request the D-05 record before merging (R8).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/alignment -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.16: Symptom patterns match inflected words
**Covers:** MVP-050
**Severity:** high
**Wave:** 5
**Depends on:** WS-16.15
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: extraction changes are reviewed with the clinician vignettes before W5 merge)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/symptom_extractor.py` (every truncated stem inside `\b(...)\b` gains `\w*`: `isolat`, `obsess`, `elat`, `euphor`, `impulsiv`, `compuls`, `hyperactiv`, `inattenti`, `purg`, `laxativ`, `starv`, `crav`, `restrict`, `risky\s+behavio`, `hopeless`, `guilt`; the implementer reads all 40 symptom entries and applies the same fix to any other truncated stem)
- Test: `services/diagnosis_service/tests/test_mvp_050_inflected_stems.py` (new)

**Interfaces:**
- Consumes: none beyond the extractor.
- Produces: extraction of the inflected forms below.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_050_inflected_stems.py::test_mvp_050_inflected_words_extract`
```python
"""MVP-050: truncated stems no longer hide key symptoms, and mania is not labelled Adjustment Disorder."""
from __future__ import annotations
import pytest
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings

CASES = [("I keep isolating myself", "social_withdrawal"), ("I feel isolated", "social_withdrawal"),
         ("The hopelessness never lifts", "hopelessness"), ("I'm obsessed with germs", "intrusive_thoughts"),
         ("I'm compulsive about checking locks", "compulsive_behavior"), ("My son is hyperactive", "hyperactivity"),
         ("I'm inattentive at work", "inattention"), ("The cravings are bad", "craving"),
         ("I've been doing risky behavior", "risk_taking"), ("I've been purging after meals and taking laxatives", "purging"),
         ("I'm starving myself and restricting food", "food_restriction"), ("I feel elated and euphoric", "elevated_mood"),
         ("I've been so impulsive lately", "impulsivity"), ("I feel guilty all the time", "guilt")]


@pytest.mark.parametrize("message,symptom", CASES)
def test_mvp_050_inflected_words_extract(message, symptom) -> None:
    assert symptom in {s.name for s in SymptomExtractor(SymptomExtractorSettings())._extract_symptoms(message, [])}


@pytest.mark.asyncio
async def test_mvp_050_mania_yields_a_bipolar_hypothesis() -> None:
    symptoms = SymptomExtractor(SymptomExtractorSettings())._extract_symptoms(
        "I feel elated and euphoric, I barely sleep, I'm impulsive and on a spending spree", [])
    names = [h.name for h in (await DifferentialGenerator(DifferentialSettings()).generate(symptoms, {})).hypotheses]
    assert any("Bipolar" in n for n in names) and not (names and "Adjustment" in names[0])
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_050_inflected_stems.py -o "addopts=" -p no:logging -q`
Expected: FAIL for every case (the `\b(isolat|...)\b` stems cannot match `isolating`, `hopelessness`, `obsessed`, `compulsive`, `hyperactive`, `inattentive`, `cravings`, `behavior`, `purging`, `laxatives`, `starving`, `elated`, `impulsive`, `guilty`), and the mania message extracts only `risk_taking` with primary Adjustment Disorder 0.6.
- [ ] **Step 3: Implement** the `\w*` suffixes (`isolat\w*`, `hopeless\w*`, ...); keep each pattern's other alternatives unchanged.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/clinical -o "addopts=" -p no:logging -q`. Expected: all pass (the phrase-set check of WS-16.3/16.4 stays green).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.17: Severity comes from modifiers in the symptom's own clause, never downgrades a severe symptom, and negated mood words are skipped
**Covers:** MVP-129
**Severity:** medium
**Wave:** 5
**Depends on:** WS-16.16
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: modifiers apply within the clause; a weakening modifier never lowers a base of MODERATELY_SEVERE or SEVERE; negation skips a symptom only outside the `risk_behaviors` category)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/symptom_extractor.py` (`_detect_severity(text: str, base: SeverityLevel, span: tuple[int, int]) -> SeverityLevel`: clauses split on `[.;,!?]` and the word `but`; the strongest modifier in the matched clause wins; result `max(base, strongest)` when base is MODERATELY_SEVERE or SEVERE; `_extract_symptoms` skips a match preceded within three tokens by `not|never|no|don't|isn't|aren't` unless `dsm_category == "risk_behaviors"`)
- Test: `services/diagnosis_service/tests/test_mvp_129_clause_severity.py` (new)

**Interfaces:**
- Consumes: WS-16.16 patterns.
- Produces: clause-scoped severity; the negation rule above.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_129_clause_severity.py::test_mvp_129_severe_self_harm_is_not_downgraded`
```python
"""MVP-129: severity modifiers are clause-scoped, never downgrade severe symptoms, and negated mood is not a symptom."""
from __future__ import annotations
import pytest
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
from services.diagnosis_service.src.schemas import SeverityLevel


def _sev(message: str) -> dict:
    return {s.name: s.severity for s in SymptomExtractor(SymptomExtractorSettings())._extract_symptoms(message, [])}


def test_mvp_129_severe_self_harm_is_not_downgraded() -> None:
    assert _sev("I've been cutting myself a little and the pain is unbearable")["self_harm"] == SeverityLevel.SEVERE


def test_mvp_129_purging_keeps_its_base() -> None:
    assert _sev("I make myself sick after meals, slightly ashamed")["purging"] == SeverityLevel.SEVERE


def test_mvp_129_modifiers_stay_in_their_clause() -> None:
    sev = _sev("I'm a little sad but the hopelessness is unbearable and overwhelming")
    assert sev["hopelessness"] == SeverityLevel.SEVERE and sev["depressed_mood"] == SeverityLevel.MINIMAL


@pytest.mark.parametrize("message", ["I'm not very sad", "I don't feel sad"])
def test_mvp_129_negated_mood_is_not_extracted(message) -> None:
    assert "depressed_mood" not in _sev(message)


def test_mvp_129_negation_never_hides_self_harm() -> None:
    assert "self_harm" in _sev("I can't stop, I'm not able to stop hurting myself")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_129_clause_severity.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `self_harm` and `purging` come back MINIMAL ("a little"/"slightly" come first in the dict and apply to the whole message), `hopelessness` is MINIMAL, and "I'm not very sad" gives `depressed_mood` MODERATELY_SEVERE.
- [ ] **Step 3: Implement** clause splitting, the span-scoped strongest modifier, the no-downgrade rule and the negation skip.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/clinical -o "addopts=" -p no:logging -q`. Expected: all pass (`test_edge_cases.py::test_multiple_severity_indicators` keeps a SEVERE symptom).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.18: The differential honours minimum criteria, duration and exclusions, and panic, fear and psychomotor criteria are extractable
**Covers:** MVP-126
**Severity:** medium
**Wave:** 5
**Depends on:** WS-16.17
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: below `min_criteria` the raw confidence is scaled by `met / min_criteria`; a present exclusion halves it; a stated duration shorter than the requirement scales it by 0.7; LLM hypotheses need a DSM code matching `^F\d{2}(\.\d{1,2})?$`)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/symptom_extractor.py` (new symptom keys `panic` (`\bpanic(\s+attack\w*)?\b`), `fear` (`\b(fear\w*|scared|afraid|terrified)\b`), `psychomotor` (`\b(slowed\s+down|moving\s+slowly|can't\s+sit\s+still|agitated)\b`); `anxiety` keeps its patterns)
- Modify: `services/diagnosis_service/src/domain/differential.py` (`_calculate_match_confidence` applies the three adjustments; `EXCLUSION_SYMPTOMS: Mapping[str, frozenset[str]]` maps exclusion keys to symptom names (`bipolar_history`/`bipolar` → `elevated_mood`, `substance_induced` → `substance_use`, `psychotic` → none in the MVP); `_duration_days(text) -> int | None` parses "for N day/week/month/year"; `_llm_generate_differential` drops items whose `dsm5_code` fails the pattern and reads `severity`/`icd11_code` from the item when valid)
- Test: `services/diagnosis_service/tests/test_mvp_126_criteria_rules.py` (new)

**Interfaces:**
- Consumes: WS-16.15 `meets_minimum_criteria`; WS-16.10 temporal attachment (`SymptomDTO.duration`).
- Produces: `EXCLUSION_SYMPTOMS`; the three confidence adjustments; symptom keys `panic`, `fear`, `psychomotor`.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_126_criteria_rules.py::test_mvp_126_panic_attacks_support_panic_disorder`
```python
"""MVP-126: panic/fear/psychomotor are extractable; min_criteria, duration and exclusions shape the differential."""
from __future__ import annotations
import json
import pytest
from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings


async def _hyps(message: str, llm=None):
    symptoms = SymptomExtractor(SymptomExtractorSettings())._extract_symptoms(message, [])
    return symptoms, (await DifferentialGenerator(DifferentialSettings(), llm_client=llm).generate(symptoms, {})).hypotheses


@pytest.mark.asyncio
async def test_mvp_126_panic_attacks_support_panic_disorder() -> None:
    symptoms, hyps = await _hyps("I keep having panic attacks, my heart racing, I am scared")
    assert {"panic", "fear", "physical_tension"} <= {s.name for s in symptoms}
    conf = {h.name: h.confidence for h in hyps}
    assert conf["Panic Disorder"] >= conf.get("Generalized Anxiety Disorder", 0)


@pytest.mark.asyncio
async def test_mvp_126_two_of_five_mdd_criteria_is_not_a_strong_mdd() -> None:
    _, hyps = await _hyps("I feel sad and I have lost interest")
    mdd = next(h for h in hyps if "Major Depressive" in h.name)
    assert mdd.confidence < 0.6 and mdd.meets_minimum_criteria is False


@pytest.mark.asyncio
async def test_mvp_126_elevated_mood_excludes_unipolar_depression() -> None:
    _, plain = await _hyps("I feel sad and empty, I lost interest, I can't sleep, I'm exhausted, I feel guilty")
    _, mixed = await _hyps("I feel sad and empty, I lost interest, I can't sleep, I'm exhausted, I feel guilty, sometimes I feel euphoric")
    p = next(h for h in plain if "Major Depressive" in h.name).confidence
    m = next(h for h in mixed if "Major Depressive" in h.name).confidence
    assert m <= p / 2 + 0.01


@pytest.mark.asyncio
async def test_mvp_126_llm_hypothesis_needs_a_valid_code() -> None:
    class _LLM:
        is_available = True
        async def generate(self, **kw):
            return json.dumps({"hypotheses": [{"name": "Made Up", "dsm5_code": "XYZ", "confidence": 0.6},
                                              {"name": "Social Anxiety Disorder", "dsm5_code": "F40.10", "confidence": 0.5}]})
    _, hyps = await _hyps("I feel anxious around people", llm=_LLM())
    assert "Made Up" not in {h.name for h in hyps}
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_126_criteria_rules.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the panic message extracts only `[anxiety, physical_tension]` (no `panic`/`fear` keys), so Panic Disorder is penalized ×0.7 (0.36 vs GAD 0.76); two MDD criteria give raw MDD 0.80; elevated mood leaves MDD unchanged (`exclusions` never read); the free-text `XYZ` hypothesis is accepted.
- [ ] **Step 3: Implement** the new patterns, the three adjustments and the LLM code check.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/alignment -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.19: Low-confidence hypotheses are withheld or flagged for clinician review
**Covers:** MVP-130
**Severity:** medium
**Wave:** 5
**Depends on:** WS-16.18
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: ESCALATE (< 0.30) hypotheses are not returned or recorded and set review on the session; LOW (0.30 to < 0.50) are returned and recorded with `requires_clinical_review=True`; thresholds as in docs/CLINICAL-VALIDATION.md)
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/schemas.py` (`HypothesisDTO.confidence_level: ConfidenceLevel | None = None`)
- Modify: `services/diagnosis_service/src/domain/service.py` (`_calibrate_confidence` keeps `CalibrationResult.confidence_level` on each hypothesis and drops the 0.1 floor; ESCALATE hypotheses are removed from primary and alternatives; `SessionState.requires_clinical_review` (new, default False) is set when a hypothesis was withheld or the primary is LOW; `end_session` writes `requires_clinical_review` and `confidence_level` on the record and on `DiagnosisRecordedEvent`)
- Modify: `services/diagnosis_service/src/domain/models.py` (`SessionState.requires_clinical_review: bool = False`)
- Modify: `services/diagnosis_service/src/events.py` (`DiagnosisRecordedEvent.requires_clinical_review: bool = False`, mapped to `DiagnosisCompletedEvent.requires_clinical_review`)
- Modify: `services/diagnosis_service/src/infrastructure/redis_session_repository.py` (serialize `requires_clinical_review`)
- Test: `services/diagnosis_service/tests/test_mvp_130_confidence_tiers.py` (new)

**Interfaces:**
- Consumes: WS-16.8 record and event path; slot 020 `requires_clinical_review`, `confidence_level` (WS-16.2).
- Produces: `HypothesisDTO.confidence_level`; `SessionState.requires_clinical_review`; `diagnosis.completed.requires_clinical_review`.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_130_confidence_tiers.py::test_mvp_130_escalate_tier_is_withheld`
```python
"""MVP-130: ESCALATE-tier hypotheses are withheld; LOW-tier ones are flagged for clinician review end to end."""
from __future__ import annotations
from decimal import Decimal
from uuid import uuid4
import pytest
from services.diagnosis_service.src.domain.confidence import CalibrationResult
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.events import DiagnosisRecordedEvent, to_kafka_event
from services.diagnosis_service.src.schemas import ConfidenceLevel, HypothesisDTO, SymptomDTO, SymptomType
from services.diagnosis_service.tests.fixtures import InMemoryDiagnosisRepository


def _hyp(conf: str) -> HypothesisDTO:
    return HypothesisDTO(hypothesis_id=uuid4(), name="Persistent Depressive Disorder", confidence=Decimal(conf),
                         criteria_met=["depressed_mood"], criteria_missing=["fatigue", "sleep_disturbance", "appetite_change", "hopelessness"])


class _FixedCalibrator:
    def __init__(self, value: str, level: ConfidenceLevel) -> None:
        self._value, self._level = Decimal(value), level

    async def calibrate(self, hypothesis, symptoms, context):
        return CalibrationResult(calibrated_confidence=self._value, confidence_level=self._level,
                                 confidence_interval=(self._value, self._value))


@pytest.mark.asyncio
async def test_mvp_130_escalate_tier_is_withheld() -> None:
    svc = DiagnosisService(calibrator=_FixedCalibrator("0.12", ConfidenceLevel.ESCALATE))
    symptoms = [SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="d", symptom_type=SymptomType.EMOTIONAL)]
    assert await svc._calibrate_confidence([_hyp("0.40")], {}, working_symptoms=symptoms) == []


@pytest.mark.asyncio
async def test_mvp_130_low_tier_is_kept_with_its_level() -> None:
    svc = DiagnosisService(calibrator=_FixedCalibrator("0.40", ConfidenceLevel.LOW))
    symptoms = [SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="d", symptom_type=SymptomType.EMOTIONAL)]
    (kept,) = await svc._calibrate_confidence([_hyp("0.40")], {}, working_symptoms=symptoms)
    assert kept.confidence_level == ConfidenceLevel.LOW and kept.confidence == Decimal("0.40")


@pytest.mark.asyncio
async def test_mvp_130_low_tier_record_and_event_require_review() -> None:
    repo = InMemoryDiagnosisRepository()
    svc = DiagnosisService(repository=repo)
    await svc.initialize()
    uid = uuid4()
    sid = (await svc.start_session(uid, "assessment", {}, None)).session_id
    session = svc._active_sessions[sid]
    session.symptoms = [SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="d", symptom_type=SymptomType.EMOTIONAL)]
    from services.diagnosis_service.src.schemas import DifferentialDTO
    session.differential = DifferentialDTO(primary=_hyp("0.40").model_copy(update={"confidence_level": ConfidenceLevel.LOW}))
    await svc.end_session(uid, sid, False)
    record = (await repo.list_user_records(uid))[0]
    assert record.requires_clinical_review is True and record.confidence_level == "low"
    kafka = to_kafka_event(DiagnosisRecordedEvent(user_id=uid, session_id=sid, primary_diagnosis="x", requires_clinical_review=True))
    assert kafka.requires_clinical_review is True
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_130_confidence_tiers.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the ESCALATE hypothesis at 0.12 is returned (the 0.1 floor keeps it and the tier is discarded), `HypothesisDTO` has no `confidence_level`, the record has no review flag and `DiagnosisCompletedEvent.requires_clinical_review` stays False.
- [ ] **Step 3: Implement** the tier propagation, the withholding rule and the review flag through record, event and Redis.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/diagnosis-service -o "addopts=" -p no:logging -q`. Expected: all pass (`test_confidence_thresholds.py` keeps the tier boundaries).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.19:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.20: LLM symptom and differential output parses robustly and failures are visible
**Covers:** MVP-259, MVP-261
**Severity:** low
**Wave:** 5
**Depends on:** WS-16.19
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/symptom_extractor.py` (new module function `parse_llm_json(text: str) -> dict | None` strips a ```` ```json ```` fence and leading or trailing prose by taking the outermost `{...}`; symptom type lookup is case-insensitive (`item["type"].lower()`); each item is validated on its own and an invalid one is skipped with a WARNING `llm_symptom_item_skipped` (count only); a parse failure logs WARNING `llm_symptom_parse_failed` without the response text and adds `llm_bad_response` to `degraded`)
- Modify: `services/diagnosis_service/src/domain/differential.py` (same parser and per-item validation for hypotheses; `llm_bad_response` on failure)
- Test: `services/diagnosis_service/tests/test_mvp_259_llm_parsing.py` (new)

**Interfaces:**
- Consumes: WS-16.4 `degraded` plumbing.
- Produces: `parse_llm_json`; degraded reason `llm_bad_response`.

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_259_llm_parsing.py::test_mvp_259_uppercase_type_and_one_bad_item`
```python
"""MVP-259 / MVP-261: fenced or wrapped LLM JSON parses, types map case-insensitively, one bad item does not drop the batch."""
from __future__ import annotations
import json
import pytest
import structlog
from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings, parse_llm_json
from services.diagnosis_service.src.schemas import SymptomType


class _LLM:
    is_available = True

    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(self, **kwargs):
        return self._text


ITEMS = [{"name": "rumination", "type": "COGNITIVE", "severity": "MODERATE", "confidence": 0.8, "evidence": "e"},
         {"name": "bad", "type": "EMOTIONAL", "severity": "MILD", "confidence": 1.5, "evidence": "e"}]


@pytest.mark.asyncio
async def test_mvp_259_uppercase_type_and_one_bad_item() -> None:
    symptoms = await SymptomExtractor(SymptomExtractorSettings(), llm_client=_LLM(json.dumps({"symptoms": ITEMS})))._llm_extract_symptoms("x", [])
    assert [(s.name, s.symptom_type) for s in symptoms] == [("rumination", SymptomType.COGNITIVE)]


@pytest.mark.asyncio
async def test_mvp_261_fenced_output_parses() -> None:
    fenced = "```json\n" + json.dumps({"symptoms": ITEMS[:1]}) + "\n```"
    assert parse_llm_json(fenced) == {"symptoms": ITEMS[:1]}
    assert parse_llm_json("Here you go: " + json.dumps({"symptoms": []}) + " Thanks") == {"symptoms": []}
    symptoms = await SymptomExtractor(SymptomExtractorSettings(), llm_client=_LLM(fenced))._llm_extract_symptoms("x", [])
    assert [s.name for s in symptoms] == ["rumination"]


@pytest.mark.asyncio
async def test_mvp_261_unparseable_output_warns_and_degrades() -> None:
    extractor = SymptomExtractor(SymptomExtractorSettings(), llm_client=_LLM("(LLM unavailable)"))
    with structlog.testing.capture_logs() as logs:
        result = await extractor.extract("I feel odd", [], [])
    assert "llm_bad_response" in result.degraded
    assert any(e["event"] == "llm_symptom_parse_failed" and e["log_level"] == "warning" for e in logs)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_259_llm_parsing.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ImportError: cannot import name 'parse_llm_json'`; with it stubbed, the 1.5-confidence item raises a validation error that drops the whole batch (`[]`), `COGNITIVE` maps to EMOTIONAL (lowercase-keyed map), the fenced text fails `json.loads`, and the failure logs only at DEBUG.
- [ ] **Step 3: Implement** the parser, the case-insensitive map and per-item validation in both modules.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.20:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.21: Session lifecycle cleanup: no global lock across Redis I/O, merged and honest history, bounded caches with durable numbering, truthful greetings
**Covers:** MVP-251, MVP-254, MVP-255, MVP-256
**Severity:** low
**Wave:** 5
**Depends on:** WS-16.20
**Implementer model:** opus
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/service.py` (`_load_session` reads the durable copy outside `_session_lock` and takes the lock only to reconcile the cache; `get_history` merges in-process entries with `list_user_records`, de-duplicated by `session_id`, newest first, `limit` applied after the merge; a repository error raises `DiagnosisHistoryUnavailable`; `_active_sessions` is an LRU bounded by `max_cached_sessions` (default 1024), `_user_history` keeps the last 20 entries per user and at most 1024 users; `start_session` numbers sessions from `session_store.get_session_count(user_id) + 1` when a store exists; the "Welcome back" greeting and `loaded_context=True` only when `previous_session_id` resolves to the caller's persisted session through the repository)
- Modify: `services/diagnosis_service/src/domain/models.py` (`class DiagnosisHistoryUnavailable(RuntimeError)`)
- Modify: `services/diagnosis_service/src/api.py` (`/history` maps `DiagnosisHistoryUnavailable` to `HTTPException(503, detail={"code": "HISTORY_UNAVAILABLE", "message": "Diagnosis history is temporarily unavailable"})`)
- Test: `services/diagnosis_service/tests/test_mvp_251_session_lifecycle.py` (new)
- Test: `services/diagnosis_service/tests/test_batch_5_1.py` (modify `test_start_session_returning_user`: `loaded_context is False` without a repository) and `services/diagnosis_service/tests/test_edge_cases.py` (modify `test_longitudinal_tracking_workflow` the same way)

**Interfaces:**
- Consumes: WS-16.8 repository; WS-16.9 store (`get_session_count`).
- Produces: `DiagnosisHistoryUnavailable` and `HISTORY_UNAVAILABLE` (503); setting `max_cached_sessions` (env `DIAGNOSIS_SERVICE_MAX_CACHED_SESSIONS`).

- [ ] **Step 1: Write the failing test** in `services/diagnosis_service/tests/test_mvp_251_session_lifecycle.py::test_mvp_251_one_slow_read_does_not_block_other_sessions`
```python
"""MVP-251/254/255/256: no global lock across Redis I/O, merged history, durable numbering, truthful greetings."""
from __future__ import annotations
import asyncio
from uuid import uuid4
import fakeredis
import fakeredis.aioredis
import pytest
from services.diagnosis_service.src.domain.entities import DiagnosisRecordEntity
from services.diagnosis_service.src.domain.models import DiagnosisHistoryUnavailable, SessionState
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.infrastructure.redis_session_repository import DiagnosisSessionStore, build_session_store
from services.diagnosis_service.tests.fixtures import InMemoryDiagnosisRepository


class _SlowStore(DiagnosisSessionStore):
    def __init__(self, *a, gate: asyncio.Event, slow_id, **kw) -> None:
        super().__init__(*a, **kw)
        self._gate, self._slow_id = gate, slow_id

    async def get_session(self, session_id):
        if session_id == self._slow_id:
            await self._gate.wait()
        return await super().get_session(session_id)


@pytest.mark.asyncio
async def test_mvp_251_one_slow_read_does_not_block_other_sessions() -> None:
    gate, slow = asyncio.Event(), uuid4()
    svc = DiagnosisService(session_store=_SlowStore(client=fakeredis.aioredis.FakeRedis(decode_responses=True), gate=gate, slow_id=slow))
    await svc.initialize()
    uid = uuid4()
    other = await svc.start_session(uid, "assessment", {}, None)
    blocked = asyncio.create_task(svc._load_session(slow))
    await asyncio.sleep(0)
    assert await asyncio.wait_for(svc.get_session_state(other.session_id, uid), timeout=0.5) is not None
    gate.set()
    await blocked


@pytest.mark.asyncio
async def test_mvp_254_history_merges_and_errors_are_not_empty() -> None:
    repo, uid, old_sid = InMemoryDiagnosisRepository(), uuid4(), uuid4()
    await repo.save_record(DiagnosisRecordEntity(user_id=uid, session_id=old_sid, primary_diagnosis="Persistent Depressive Disorder"))
    restarted = DiagnosisService(repository=repo)  # a fresh process: the persisted record is not in memory
    await restarted.initialize()
    restarted._store_session_history(uid, SessionState(user_id=uid))  # one entry from this process
    sessions = (await restarted.get_history(uid, 10, True, True)).sessions
    assert len(sessions) == 2 and str(old_sid) in {s["session_id"] for s in sessions}

    class _Broken(InMemoryDiagnosisRepository):
        async def list_user_records(self, user_id, limit=10):
            raise ConnectionError("db down")

    broken = DiagnosisService(repository=_Broken())
    await broken.initialize()
    with pytest.raises(DiagnosisHistoryUnavailable):
        await broken.get_history(uid, 10, True, True)


@pytest.mark.asyncio
async def test_mvp_255_numbering_survives_a_restart() -> None:
    server, uid = fakeredis.FakeServer(), uuid4()
    store = lambda: build_session_store(fakeredis.aioredis.FakeRedis(server=server, decode_responses=True))
    first = DiagnosisService(session_store=store())
    await first.initialize()
    await first.start_session(uid, "assessment", {}, None)
    restarted = DiagnosisService(session_store=store())
    await restarted.initialize()
    second = await restarted.start_session(uid, "assessment", {}, None)
    assert second.session_number == 2 and "session 2" in second.greeting


@pytest.mark.asyncio
async def test_mvp_256_unknown_previous_session_is_not_remembered() -> None:
    svc = DiagnosisService(repository=InMemoryDiagnosisRepository())
    await svc.initialize()
    result = await svc.start_session(uuid4(), "assessment", {}, uuid4())
    assert result.loaded_context is False and "remember" not in result.greeting.lower()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_251_session_lifecycle.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `get_session_state` times out behind the global lock held across the slow Redis read; `get_history` returns only the in-process entry and an empty list on a database error; the restarted service numbers the session 1 with the first-time greeting; an unknown `previous_session_id` gets "I remember our previous conversation" and `loaded_context=True`.
- [ ] **Step 3: Implement** the lock narrowing, the merged history and its error, the bounded caches, durable numbering and the repository-backed greeting.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_phase_c_session_reconcile.py` keeps its reconcile-on-hit assertions).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.21:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-16.22: Label the unwired instrument and coding modules as library-only
**Covers:** MVP-257
**Severity:** low
**Wave:** 5
**Depends on:** WS-16.21
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/diagnosis_service/src/domain/severity.py`, `services/diagnosis_service/src/domain/value_objects.py`, `services/diagnosis_service/src/domain/clinical_codes.py`, `services/diagnosis_service/src/domain/evidence.py` (module docstrings start with `Library only (MVP-257): not reachable from any diagnosis route or the assessment chain in the MVP; hypothesis severity comes from criteria counts and DiagnosisRecordEntity.assessment_scores stays empty.`; no behaviour change)
- Test: `services/diagnosis_service/tests/test_mvp_257_unwired_modules.py` (new verification lock)

No other file is edited: the existing PHQ-9/GAD-7/PCL-5 threshold tests in the diagnosis suite and WS-05's alignment suite only run unchanged in Step 5.

**Interfaces:**
- Consumes: none.
- Produces: a lock that a later wiring of these modules must also remove the label and update the docs. The docs half of MVP-257 is request R9 to WS-03 (best fit WS-03.15, the W5 docs task); this task does not wait for it because its test reads only service code.

- [ ] **Step 1: Write the verification test** in `services/diagnosis_service/tests/test_mvp_257_unwired_modules.py::test_mvp_257_unwired_modules_are_labelled_and_unimported`
```python
"""MVP-257: the instrument and coding modules are documented as library-only and nothing in the service imports them."""
from __future__ import annotations
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
UNWIRED = ("severity", "value_objects", "clinical_codes", "evidence")


def test_mvp_257_unwired_modules_are_labelled_and_unimported() -> None:
    for name in UNWIRED:
        doc = ast.get_docstring(ast.parse((SRC / "domain" / f"{name}.py").read_text(encoding="utf-8"))) or ""
        assert doc.startswith("Library only (MVP-257)"), name
    importers = []
    for path in SRC.rglob("*.py"):
        if path.stem in UNWIRED:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] in UNWIRED:
                importers.append(f"{path.name}:{node.module}")
    assert importers == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests/test_mvp_257_unwired_modules.py -o "addopts=" -p no:logging -q`
Expected: FAIL on the label assertion for `severity` (its header advertises PHQ-9/GAD-7/PHQ-15 scoring as a service capability); the import scan is already empty, which is the verification that the modules are dead.
- [ ] **Step 3: Implement** the four docstring labels only.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/alignment -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-16.22:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| # | Owner WS | Exact change | Files | Needed by |
|---|---|---|---|---|
| R1 | WS-12 | Lexicon v1 and the phrase set classify at HIGH or above, with the listed category: `I've been thinking about suicide`, `I feel suicidal`, `I want to die`, `I want to end my life`, `I'd be better off dead` (suicidal_ideation), `I'm going to overdose` (overdose), `I've been self-harming`, `I keep hitting myself when I'm upset`, `I scratch myself until I bleed` (self_harm), `I don't want to live anymore, I have a plan` (suicide_plan), `I'm not sad at all, but I want to kill myself` (suicidal_ideation; negation in an earlier clause); and below HIGH: `I'm going to end it with my boyfriend`, `I'm cutting back on carbs`, `I'm going to give up sugar`, `I'm cutting myself some slack this week`. Overlaps R1 of WS-15; diagnosis adds no local lexicon. Target task: WS-12.3 (lexicon v1 and phrase set, W1). | `src/solace_clinical/crisis_lexicon_v1.yaml`, `tests/clinical/crisis_phrases_v1.yaml` | WS-16.3, WS-16.4 |
| R2 | WS-04 | Confirmation only (already planned in WS-04.3): `solace_infrastructure.runtime.metrics` exports `CRISIS_DETECTED` (`solace_crisis_detected_total{source,level}`), the object WS-16.5 increments (the same one WS-15 uses; WS-12's R-12-5 asks for it under the name `CRISIS_DETECTED_TOTAL`, which WS-04.3 does not define). The conformance table row for `diagnosis-service` is already as WS-16.1's `SPEC`. | `src/solace_infrastructure/runtime/metrics.py` | WS-16.5 |
| R3 | WS-04 | Retire the `configure_phi_encryption` AST check in `tests/integration/test_phi_encryption_activation.py` in W0 (CR-8 of WS-09; no WS-04 task carries it yet), because WS-16.1 removes diagnosis's `configure_phi_encryption` call. Same request as R11 of WS-15. Not blocking: WS-16.1 records the expected failure in the ledger until it lands. | `tests/integration/test_phi_encryption_activation.py` | WS-16.1 |
| R4 | WS-05 | Confirmation only (already planned in WS-05.5, W0): `tests/system/test_smoke.py` asserts `GET /live` for diagnosis instead of the pre-IF-01 `/health == "healthy"` body that WS-16.1 removes; WS-16.1 depends on WS-05.5. | `tests/system/test_smoke.py` | WS-16.1 |
| R5 | WS-03 | Env registry and compose: `diagnosis-service` gets `REDIS_URL`, `EVENTS_ENABLED`, `KAFKA_BOOTSTRAP_SERVERS`, `POSTGRES_*` per IF-02, and loses `DIAGNOSIS_REDIS_HOST/PORT` (docker-compose.yml:320-321). New optional, non-secret entries: W0 `DIAGNOSIS_SERVICE_MAX_REASONING_TIME_MS` (dev default 10000, 6000 from W2), `DIAGNOSIS_SERVICE_ENABLE_ANTI_SYCOPHANCY` (true), `DIAGNOSIS_SESSION_STORE_SESSION_TTL` (86400); W2 `DIAGNOSIS_SERVICE_CRISIS_WATCH_TURNS` (3), `DIAGNOSIS_SERVICE_CRISIS_WATCH_SECONDS` (1800), `DIAGNOSIS_SESSION_STORE_MAX_MESSAGES` (50), `DIAGNOSIS_SESSION_STORE_MAX_REASONING_STEPS` (8); W5 `DIAGNOSIS_SERVICE_MIN_TURNS_PER_PHASE` (2), `DIAGNOSIS_SERVICE_MAX_CACHED_SESSIONS` (1024). Mark retired once WS-16.1 lands: `DIAGNOSIS_SESSION_REDIS_*`, `DIAGNOSIS_REDIS_*`, `DIAGNOSIS_DB_*`, `DIAGNOSIS_AI_*`, `DIAGNOSIS_REASONING_*`, `DIAGNOSIS_SAFETY_*`, `DIAGNOSIS_OBS_*`, `DIAGNOSIS_REASONING_TIMEOUT_MS`, `DIAGNOSIS_ENABLE_ANTI_SYCOPHANCY`, `DIAGNOSIS_CORS_ORIGINS`. | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml` | WS-16.1, WS-16.6, WS-16.9, WS-16.15, WS-16.21 |
| R6 | WS-11 | `tests/contracts/orchestrator__diagnosis/` uses fixed example ids so the provider can seed them: user `00000000-0000-4000-a000-000000000001`; `assess.json` session `00000000-0000-4000-d000-000000000001`; `session_end.json` session `...-d000-000000000002` (a separate session, so interaction order does not matter); a 403 `USER_ID_MISMATCH` interaction on session `...-d000-000000000003` (owned by user `...-a000-000000000002`); a 404 `SESSION_NOT_FOUND` interaction on session `...-d000-0000000000ff`; turn `00000000-0000-4000-e000-000000000001`. `assess.json` lists `Authorization` and `X-Turn-ID` in `headers_required`, and its body schema allows the provider fields `risk_level`, `risk_categories` and `degraded`. Not blocking: WS-16.12 seeds whatever ids WS-11.10's `body_example`s carry; only the 403/404 interactions and the schema fields need WS-11.10 to adopt this. | `tests/contracts/orchestrator__diagnosis/*.json` | WS-16.12 |
| R7 | WS-11 | Consumer side (non-blocking for WS-16): the diagnosis agent sends `X-Turn-ID`, starts a new diagnosis session and retries once on 404 `SESSION_NOT_FOUND`, maps 403 to a degraded result, and turns the response's `risk_level` into an IF-07 `RiskSignal(source="diagnosis", reliable=True)` for fusion. | `services/orchestrator_service/src/agents/diagnosis_agent.py` | none (WS-11.10 consumer) |
| R8 | WS-12 | (No WS-12 task carries lane sign-offs yet; the shared-files protocol for `docs/CLINICAL-VALIDATION.md` says they are added as supporting tasks in the WS-12 lane, extending WS-12.3's §11.) Record for D-05 sign-off: diagnosis uses lexicon v1 unchanged (WS-16.3), the self-referential self-harm and LLM raise-only rules (WS-16.4), the crisis watch (WS-16.6), and the W5 scoring changes with their vignettes (WS-16.15 to WS-16.19). | `docs/CLINICAL-VALIDATION.md` | WS-16.4, WS-16.15 to WS-16.19 |
| R9 | WS-03 | Best fit WS-03.15 (W5 docs task, which already rewrites both files; non-blocking for WS-16.22): `docs/SYSTEM-DESIGN-SUMMARY.md` and `docs/KNOWN-LIMITATIONS.md` stop listing `/api/v1/assessment/phq9` and `/gad7` as live diagnosis endpoints and state that instrument scoring is not wired in the MVP. | `docs/SYSTEM-DESIGN-SUMMARY.md`, `docs/KNOWN-LIMITATIONS.md` | WS-16.22 |
| R10 | WS-14 | Confirmation only (planned in WS-14.13, W3 phase A): IF-16 `register_user_erasure_consumer` lands in W3 phase A and lists `diagnosis-service` among the expected services. | `src/solace_infrastructure/gdpr/consumer.py` | WS-16.13 |
| R11 | WS-06 | Confirmation only (planned in WS-06.8, W1): `ServiceEventPlane.subscribe(...)` may be called before or after `start()`, so the service composition can subscribe after `start_service_runtime` started the plane (same as R7 of WS-15 and R-12-3 of WS-12). | `src/solace_events/service_plane.py`, `src/solace_infrastructure/runtime/bootstrap.py` | WS-16.13 |
| R12 | WS-05 | Acknowledged, no task: WS-05 sanctions both edits in its "Shared-directory stewardship" table and WS-05.20's Produces line and makes no edit to either file. Acknowledge the code-owner edits: WS-16.11 rewrites `tests/integration/test_diagnosis_vignette.py` (named in the shared-files protocol) and WS-16.5 edits only class `TestDiagnosisBridgeSafetyFlags` in `tests/integration/test_event_flow.py`. | `tests/integration/test_event_flow.py` | WS-16.5 |

## New interfaces

- **NI-16.1 Diagnosis risk screen.** `services.diagnosis_service.src.domain.risk`: `DiagnosisRiskScreen(level, categories, detection_layer, lexicon_version)` with `is_crisis`; `screen_diagnosis_message(message, *, rule_symptom_names=())`; `raise_for_symptoms(screen, symptom_names, *, source)`; `SYMPTOM_RISK_CATEGORIES`; `SymptomExtractor.screen_risk(message)`. Detection is the shared lexicon plus raise-only symptom signals; nothing lowers a lexicon level.
- **NI-16.2 Diagnosis provider response extension (IF-11 diagnosis provider).** `AssessmentResponse` and `SymptomExtractionResponse` gain `risk_level: CrisisLevel`, `risk_categories: list[str]` (codes from `solace_clinical.CATEGORIES`) and `degraded: list[str]` (vocabulary shared with NI-15.1: `llm_unavailable`, `llm_bad_response`, `reasoning_timeout`, `session_persist_failed`, `session_store_unavailable`, `crisis_event_publish_failed`, `crisis_event_unpublished`); `safety_flags` carry category codes plus the informational `psychotic_features`; `reasoning_timeout` and `unauthorized_session` are no longer safety flags.
- **NI-16.3 Diagnosis request header.** `/assess` and `/extract-symptoms` read `X-Turn-ID` (IF-06) as `metadata.correlation_id`; a missing or invalid header mints a UUID.
- **NI-16.4 Diagnosis HTTP errors.** 404 `{"detail": {"code": "SESSION_NOT_FOUND"}}`, 403 `{"detail": {"code": "USER_ID_MISMATCH"}}` on assess, end, state and challenge; 503 `HISTORY_UNAVAILABLE` on history; 403 `SUBJECT_FORBIDDEN` from `authorize_subject` (W3).
- **NI-16.5 Composition seam.** `services.diagnosis_service.src.main.{SPEC, DiagnosisServiceAppSettings, DiagnosisComponents, build_diagnosis_components, create_application}`; `DiagnosisService(event_publisher=...)`, `assess(..., *, turn_id=None)`, `extract_symptoms(..., *, turn_id=None)`, `purge_user_caches(user_id)`; test doubles in `services/diagnosis_service/tests/runtime_fakes.py`.
- **NI-16.6 Persistence.** `PostgresDiagnosisRepository(codec, schema="public")`, `DiagnosisRepositoryPort.save_completed_session(session, record, *, event=None, publish=None)`, `DIAGNOSIS_PHI_FIELDS`; Redis envelope `{"user_id", "version", "enc"}` with AAD field `diagnosis_session_blob`; slot 020 columns `requires_clinical_review`, `confidence_level` and TEXT `dsm5_code`/`icd11_code`.
- **NI-16.7 Clinical fields.** `HypothesisDTO.meets_minimum_criteria`, `HypothesisDTO.confidence_level`; `SessionState.{crisis_watch_turns_left, crisis_watch_until, phase_before_crisis, turns_in_phase, requires_clinical_review}`; `DiagnosisRecordedEvent.requires_clinical_review`.
- **NI-16.8 Env names.** `DIAGNOSIS_SESSION_STORE_SESSION_TTL`, `DIAGNOSIS_SESSION_STORE_KEY_PREFIX`, `DIAGNOSIS_SESSION_STORE_MAX_MESSAGES`, `DIAGNOSIS_SESSION_STORE_MAX_REASONING_STEPS` replace `DIAGNOSIS_SESSION_REDIS_*`; `DIAGNOSIS_SERVICE_MAX_REASONING_TIME_MS`, `DIAGNOSIS_SERVICE_ENABLE_ANTI_SYCOPHANCY`, `DIAGNOSIS_SERVICE_CRISIS_WATCH_TURNS`, `DIAGNOSIS_SERVICE_CRISIS_WATCH_SECONDS`, `DIAGNOSIS_SERVICE_MIN_TURNS_PER_PHASE`, `DIAGNOSIS_SERVICE_MAX_CACHED_SESSIONS`; `EVENTS_ENABLED` gates the bridge.

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-046 (absorbs MVP-127) | high | WS-16.3 |
| MVP-047 | high | WS-16.3 (flags and phase on timeout), WS-16.5 (event before the chain) |
| MVP-048 | high | WS-16.15 |
| MVP-049 | high | WS-16.4 (flags and phase), WS-16.5 (event) |
| MVP-050 | high | WS-16.16 |
| MVP-121 | medium | WS-16.5 (no raw text in events, dispatcher, reasoning summary), WS-16.13 (in-process copies erased) |
| MVP-122 | medium | WS-16.6 |
| MVP-123 | medium | WS-16.6 |
| MVP-124 | medium | WS-16.10 |
| MVP-125 | medium | WS-16.10 (and no fabricated record in WS-16.8) |
| MVP-126 | medium | WS-16.18 |
| MVP-128 | medium | WS-16.2, WS-16.8 |
| MVP-129 | medium | WS-16.17 |
| MVP-130 | medium | WS-16.2 (columns), WS-16.19 |
| MVP-131 | medium | WS-16.10 |
| MVP-360 | medium | WS-16.11 |
| MVP-251 | low | WS-16.21 |
| MVP-252 | low | WS-16.9 |
| MVP-253 | low | WS-16.5 (crisis publish failures), WS-16.8 (completed payload, transactional publish) |
| MVP-254 | low | WS-16.21 |
| MVP-255 | low | WS-16.21 |
| MVP-256 | low | WS-16.21 |
| MVP-257 | low | WS-16.22 |
| MVP-258 | low | WS-16.7 |
| MVP-259 | low | WS-16.20 |
| MVP-260 | low | WS-16.5 |
| MVP-261 | low | WS-16.20 |
| MVP-262 | low | WS-16.5 |
| MVP-263 | low | WS-16.1 |

Supporting edits for other workstreams' entries (not in Covers): MVP-095 and MVP-192 diagnosis halves (WS-16.1, WS-16.7; owner WS-15, as WS-15's coverage table expects), MVP-270 import root (WS-16.1, owner WS-01), MVP-022 provider contract and MVP-314 `/status` (WS-16.12, owners WS-11 and WS-07), MVP-029 erasure consumer (WS-16.13, owner WS-14), MVP-120/MVP-141 subject scoping and MVP-091 PHI audit (WS-16.14, owners WS-07 and WS-09), MVP-055 codec adoption (WS-16.8, WS-16.9, owner WS-09).

**Incoming requests.** R-01.9 (WS-01.7): WS-16.1, W0. WS-04.5 adoption: WS-16.1, W0. WS-05.10 event-contract test: WS-16.5, W2. WS-06.12 codes-only `trigger_indicators`: WS-16.5, W2 (before WS-06.12 in W3). WS-07.9 `/status` permission: WS-16.12, W2 (before W4). WS-07 W3 `authorize_subject`: WS-16.14, W3. WS-09.4 (CR-7) LLM readiness and `LLMUnavailableError`: WS-16.4 in W2; CR-7's text allows W1 or W2 and WS-16 has no W1 wave; until then diagnosis's broad `except Exception` around LLM calls keeps W1 working. WS-09.15 (CR-10): WS-16.8 and WS-16.9 in W2. WS-11.10 provider verification: WS-16.12, W2. WS-05.10 REST provider verification (request 10): WS-16.12, W2. All are planned no later than the requesting task's wave.

REVIEW_FOCUS: 1=WS-16.3,WS-16.4,WS-16.6; 2=WS-16.5,WS-16.7,WS-16.8; 3=WS-16.7,WS-16.8,WS-16.13,WS-16.14; 4=WS-16.1,WS-16.5,WS-16.9,WS-16.14; 5=WS-16.1,WS-16.9

**Review focus pins.** 1 (disclosures outside the phrase set): WS-16.3 runs the full phrase set through the diagnosis detector, the register phrasings and a negation-in-an-earlier-clause case; WS-16.4 proves LLM and symptom signals can only raise; WS-16.6 keeps risk across turns through the crisis watch. 2 (partial failures on the alert path): WS-16.5 publishes before the chain, at most once per turn, and surfaces publish failures; WS-16.7 never fails a crisis turn on persistence or a Redis read error; WS-16.8 commits record and outbox atomically and rolls both back on failure. 3 (foreign identifiers and context-less work): WS-16.7 (403/404), WS-16.8 (RLS denies user B), WS-16.13 (erasure under `acting_as_user`), WS-16.14 (unassigned clinician). 4 (PHI sinks): WS-16.1 (no exception text in error bodies), WS-16.5 (event, outbox and dispatcher payloads), WS-16.9 (Redis ciphertext), WS-16.14 (logs and audit details). 5 (strict boots with partial configuration): WS-16.1 (no Redis or events in strict fails; the documented knob names are the ones read), WS-16.9 (no codec in strict fails).

**Interface consistency.** IF-01 names (`ServiceRuntimeSpec`, `start_service_runtime`, `stop_service_runtime`, `create_service_app`, `is_strict_environment`, `StartupDependencyError`, `runtime.redis/pg_pool/event_plane/phi_codec`, `register_readiness_check`, `register_shutdown`) are used verbatim, and the diagnosis `SPEC` matches the WS-04.5 conformance row. IF-05 `publish(event, *, conn=None) -> UUID` and `EventPublishError` match WS-06.5/WS-06.8. IF-06 fields match WS-15.5's therapy builder except `source_service="diagnosis-service"` and `detection_layer` 1 or 3. IF-07 `screen_text`, `CATEGORIES`, `level_rank`, `RiskSignal`, `fuse_risk` and `check_detector` are the only lexicon inputs. IF-13 `PHICodec(active_key_id=, keys=, search_salt=, allow_plaintext=)`, `encrypt(value, *, field, owner_id)` and `active_key_id` match WS-09.7. IF-16 `register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches)` matches WS-15.13. The degraded vocabulary and the `risk_level` field name match NI-15.1. The IF-18 counter handle is WS-04.3's `CRISIS_DETECTED`, the same object WS-15.5 uses. Slot 020 runs after slot 019 (WS-15.3) in the W2 schema lane; WS-15 refers to WS-16.1, WS-16.2 and WS-16.7, which are the adoption, slot and CAS-rebase tasks here.

**Open risks.**
1. WS-16.7 reverses the Phase C P2-9 rule for diagnosis (a crisis turn whose session cannot persist used to return 500). The crisis event is already durable in the outbox and diagnosis text is never user-facing, so the turn now returns with `session_persist_failed`. This needs a ledger Ruling at the W2 barrier (WS-15 records the same reversal for therapy).
2. Diagnosis detection is the shared lexicon plus two raise-only symptom rules. If lexicon v1 lacks a register phrasing, WS-16.3 stays red until request R1 lands. That is intended: detection must not weaken.
3. Crisis turns skip the LLM chain, so no differential update happens on those turns. This is clinically conservative and keeps disclosures away from the LLM, but it should be noted in the D-05 record (R8).
4. WS-16.15 to WS-16.19 change clinical scoring and are blocked without the D-05 sign-off at W5 entry. The phase gate deliberately stops using calibrated confidence, because the calibrator's base-rate posterior structurally caps MDD/GAD below 0.7; recalibrating the Bayesian update itself is left for post-MVP.
5. `diagnosis.completed` still carries the hypothesis `condition_name` to analytics. WS-06.12 redacts `evidence_summary` only; if the W3 PHI-egress review treats diagnosis labels on `solace.assessments` as PHI, WS-18/WS-06 must restrict the consumer or WS-16 sends codes only (ledger decision at W3).
6. WS-16.11's red proof needs the W2-start tree export, because by the time it runs the earlier tasks have fixed the behaviour; the rewrite still removes the faked step-1 path that gave false assurance.
7. Deleting `config.py` and `TestConfiguration` lowers the suite count; the drop is recorded in the ledger at the W0 barrier.
8. `tests/integration/test_phi_encryption_activation.py` fails for `diagnosis_service` from WS-16.1 until request R3 lands, because no WS-04 task carries CR-8 of WS-09 yet; WS-16.1 records it in the ledger and does not edit the file.
9. WS-11.10's contract files do not use R6's fixed ids or carry 403/404 interactions; WS-16.12 seeds the ids it finds, so the provider check works either way, but the 403/404 contract coverage exists only once WS-11 adopts R6 (the behaviour itself is pinned by WS-16.7).

## Review responses

- Error, WS-16.1 modifies `tests/test_batch_5_3.py`, which does not exist: fixed. The parenthetical on the `config.py` delete line used a suite-relative path; the file is `services/diagnosis_service/tests/test_batch_5_3.py` (exists; `TestConfiguration` at line 606, the `config` import at line 13) and is now written in full, matching the Test line that already named it.
- Error, WS-05 request 9 (WS-05.10): diagnosis `verify_event_producer` test: fixed. WS-16.5 already planned `services/diagnosis_service/tests/test_mvp_357_event_contracts.py` and named WS-05.10 in Produces, but listed it as a Test line, which the index records as a test file rather than a created file; it is now a Create line.
- Error, WS-09 CR-7 (WS-09.4): rejected: already planned in WS-16.4, which modifies `services/diagnosis_service/src/main.py` with `runtime.register_readiness_check("llm", llm_client.readiness, critical=False)`, asserts `runtime.readiness_checks["llm"][1] is False`, maps `LLMUnavailableError` at the extractor and differential call sites to `llm_unavailable`, and names WS-09.4 in Produces (wording sharpened). The request lists no files, so the file-match check cannot see it.
- Error, WS-09 CR-10 (WS-09.15): rejected: already planned. Both diagnosis ADOPTERS paths are covered: `postgres_repository.py` by WS-16.8 (codec constructor, `encryption_key_id=codec.active_key_id`, drops `_current_encryption_key_id`, the `get_phi_encryptor` caller at `services/diagnosis_service/src/infrastructure/postgres_repository.py:203-215`; Produces "Answers WS-09.15 (CR-10)") and `redis_session_repository.py` by WS-16.9 (Produces "Answers WS-09.15 (CR-10)"); WS-16.1 removes the `configure_phi_encryption` call at `services/diagnosis_service/src/main.py:95-100`. The request lists no files, so the file-match check cannot see it.
- Error, WS-11 request (WS-11.10): provider verification in `services/diagnosis_service/tests/test_mvp_022_contract_orchestrator_diagnosis.py`: fixed. WS-16.12 already planned it (Produces "provider verification for WS-11.10") but as a Test line; it is now a Create line, and the test seeds the user and session ids from the contracts' `body_example`s because WS-11.10 writes its own ids (it did not adopt R6).
- Warning, WS-16.1 names WS-04 and WS-05: fixed. R4 is satisfied by WS-05.5 (W0; it rewrites `tests/system/test_smoke.py` to assert `/live` for the six underscore services), now a dependency. The R3 retirement has no producing task (the WS-04 plan never mentions `test_phi_encryption_activation`), so it left Depends on and stays request R3; the regression step names the expected failure.
- Warning, WS-16.3 names WS-12: fixed. WS-12.3 was already a dependency; the vague token is gone and R1 names WS-12.3 as its target.
- Warning, WS-16.5 names WS-04: fixed. Depends on WS-04.3, which exports the counter as `CRISIS_DETECTED`; the plan's undefined name `CRISIS_DETECTED_TOTAL` was replaced everywhere and R2 became a confirmation row.
- Warning, WS-16.12 names WS-11: fixed. Depends on WS-11.10 (already listed) and WS-05.7; R6 is non-blocking now that the test reads the ids from the contract files.
- Warning, WS-16.13 names WS-14 and WS-06: fixed. Depends on WS-14.13 (IF-16 helper, W3 phase A) and WS-06.8 (subscribe before or after `start()`); R10 and R11 became confirmation rows.
- Warning, WS-16.15 names WS-12: fixed. Depends on WS-12.3, which writes §11 of `docs/CLINICAL-VALIDATION.md` (the D-05 record); no WS-12 task (WS-12.1 to WS-12.16) records lane sign-offs, so R8 stays a request and the D-05 decision gate at W5 entry still blocks the scoring changes.
- Warning, WS-16.22 names WS-03: fixed. The docs edit is not an input to WS-16.22 (its test reads only service code), so the dependency was removed; R9 now names WS-03.15, the W5 docs task that already rewrites both files.
- Warning, WS-16.11 edits `tests/integration/test_diagnosis_vignette.py` owned by WS-05: rejected: sanctioned. The skeleton's `tests/integration` shared-files protocol reads "an existing file is edited only by the owner of the code it tests (e.g. test_diagnosis_vignette.py by WS-16)"; R12 records the acknowledgement for WS-05.
- Warning, WS-16.22 edits `tests/alignment/test_clinical_alignment.py` owned by WS-05: rejected: WS-16.22 never edits it. The path appeared only in a parenthetical on the Modify line saying those tests keep running, and Step 5 only runs `tests/alignment`; the parenthetical moved to a separate sentence so the parser no longer reads it as an edit.
- Not in the findings, closed in the same pass: WS-05.10's request 10 (providers call `verify_provider`) is answered by WS-16.12, the verification of `orchestrator__diagnosis` (the only REST pair with diagnosis as provider; diagnosis consumes no REST contract), and its Produces line now names WS-05.10; the Self-review now has the REVIEW_FOCUS line; R3 states that no WS-04 task carries CR-8 yet (same as R11 of WS-15).
- Reconciliation 7 (R12 to WS-05, acknowledging the code-owner edits of `tests/integration/test_diagnosis_vignette.py` by WS-16.11 and of class `TestDiagnosisBridgeSafetyFlags` in `tests/integration/test_event_flow.py` by WS-16.5): marked "Acknowledged, no task:". WS-05 owns the `tests/integration` steward entry and sanctions both edits in its "Shared-directory stewardship" table; no WS-05 task edits either file, and WS-16.5 and WS-16.11 keep the edits.
