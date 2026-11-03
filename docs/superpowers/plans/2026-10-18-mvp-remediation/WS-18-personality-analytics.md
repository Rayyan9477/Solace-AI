# WS-18: Personality and analytics services Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Personality boots on the shared runtime, persists codec-encrypted profiles that match the migrated schema under RLS, honours the orchestrator's detection request, erases with a durable tombstone and stops fabricating neutral scores; analytics ingests every event from the event plane without loss or PHI logging, reports health truthfully, serves its read APIs to staff only and computes correct, honestly scoped reports.

**Architecture:** Both `main.py` files adopt IF-01 in W0 behind one composition seam each (`build_personality_orchestrator`, `start_kafka_ingestion` then `wire_analytics_ingestion`). Personality's repository runs on the shared `ConnectionPoolManager` pool with IF-13 field encryption and an upsert keyed on the slot-021 unique index; a per-user lock, an in-process erased set and the slot-021 tombstone table make erasure final. Analytics subscribes to the IF-05 plane with group `solace-group-analytics`, feeds `event.to_dict()` to its in-memory aggregator (D-18: non-critical, no PHI) and raises on failure so the plane retries and dead-letters.

**Covers:** 26 register entries (blocker 1, high 1, medium 6, low 18).

**Waves:** 0 (both services adopt IF-01; MVP-027 boot blocker; truthful analytics readiness; import-root and lint gates), 2 (slot 021; encrypted profile persistence; persist-before-cache; detection sources; validation; route permissions and the orchestrator contract; analytics on the event plane, ingest shape, event-time and time zones), 3 (personality erasure with tombstone; analytics PHI-free logs; staff-only reads), 5 (D-18 residuals: fabricated evidence, cache bounds and aggregates, personality events, analytics retention and coverage, report correctness).

**Depends on:**
- WS-01: the import-root guard and the lint gate (`tests/build/test_import_roots.py`, `tests/build/test_mvp_038_lint_gate.py`, WS-01.7) that W0 must turn green for WS-18; markers `realinfra`, `integration`, `contract`.
- WS-02: slot `021_personality_analytics` scaffold and 009 default privileges and NULLIF policies (WS-02.3); the single `default` pool (WS-02.6).
- WS-03: IF-02 env sync for both services (POSTGRES_* as solace_app, `REDIS_URL`, `KAFKA_BOOTSTRAP_SERVERS`, `EVENTS_ENABLED`, `*_ENVIRONMENT = ${ENVIRONMENT}`) and the retirements requested below.
- WS-04: IF-01 runtime and the conformance table (WS-04.4, WS-04.5); event-plane wiring (WS-04.6); codec warm-up `runtime.phi_codec` (WS-04.8).
- WS-05: IF-19 fixtures `pg_migrated`, `kafka_bootstrap` (WS-05.2); IF-20 `verify_provider` (WS-05.7).
- WS-06: IF-05 `ServiceEventPlane.subscribe`, `start_service_event_plane`, `NonRetryableEventError` (WS-06.8); the personality bridge on the plane (WS-06.9).
- WS-07: `acting_as_user` (WS-07.1); `ServicePermission.WRITE_ANALYTICS` and the orchestrator's `WRITE_PERSONALITY` grant (WS-07.3); obo principals, `AuthenticatedService.on_behalf_of`, override-friendly `require_service_permission`, `find_unguarded_service_routes` (WS-07.4).
- WS-09: LLM client init that never raises outside strict (WS-09.2), `LLMUnavailableError` and `readiness()` (WS-09.3, WS-09.4); IF-13 `PHICodec` (WS-09.7).
- WS-11: the consumer contract files `tests/contracts/orchestrator__personality/{detect,style}.json` (WS-11.11).
- WS-14: IF-16 helper `register_user_erasure_consumer` (W3 phase A).

**Owned paths:** `services/personality_service/src/domain`, `services/personality_service/src/ml`, `services/personality_service/src/{api,config,events,main,schemas,__init__}.py`, `services/personality_service/src/infrastructure/{postgres_repository,repository,__init__}.py`, `services/personality_service/tests`, `services/analytics-service/src`, `services/analytics-service/tests`, `migrations/versions/021_personality_analytics.py`; new component tests under `tests/realinfra/personality-analytics/`. Not owned: `services/personality_service/src/infrastructure/event_bridge.py` (WS-06), both Dockerfiles and requirements files (WS-01), the compose files (WS-03). Owned but frozen: `services/analytics-service/src/repository.py` (ClickHouse path, out of scope).

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/personality_service/src/main.py` | Modify | IF-01 SPEC, settings aliases, lifespan, `build_personality_orchestrator`, LLM readiness, erasure-consumer wiring, event bus hand-off | 18.1, 18.4, 18.6, 18.12, 18.17 |
| `services/personality_service/src/config.py` | Modify | Drop the unused required DB and Redis settings | 18.1 |
| `services/personality_service/src/events.py` | Modify | Single `solace_events` import root | 18.1 |
| `services/personality_service/src/infrastructure/postgres_repository.py` | Modify | Shared pool, codec, schema-aligned upsert, tombstone guard and erasure | 18.4, 18.12 |
| `services/personality_service/src/infrastructure/repository.py` | Modify | Factory without the codec-less singleton | 18.4 |
| `services/personality_service/src/domain/errors.py` | Create | `ProfilePersistenceError`, `ProfileErasedError` | 18.5, 18.12 |
| `services/personality_service/src/domain/service.py` | Modify | Persist-before-cache, source labelling, style confidence, erasure lock and tombstone, cache bounds, aggregates, events | 18.5, 18.6, 18.8, 18.12, 18.15, 18.16, 18.17 |
| `services/personality_service/src/domain/trait_detector.py` | Modify | `assess()`/`DetectionOutcome`, source gating, no fabricated LLM, LIWC or RoBERTa evidence | 18.6, 18.15 |
| `services/personality_service/src/ml/roberta_model.py` | Modify | `is_trained` | 18.15 |
| `services/personality_service/src/schemas.py` | Modify | `sources` min length 1, `sample_count >= 1` | 18.7 |
| `services/personality_service/src/api.py` | Modify | 503/410/400 mapping, receiver-side permissions, obo user match | 18.5, 18.7, 18.8, 18.12 |
| `services/personality_service/tests/conftest.py` | Create | `FakeCodec`, `fake_codec`, `RecordingConnection`, `recording_cpm` | 18.4 |
| `services/personality_service/tests/fixtures.py` | Modify | In-memory repository honours tombstones | 18.12 |
| `services/personality_service/tests/{test_infrastructure,test_persist_profile,test_api,test_service,test_trait_detector}.py` | Modify | Existing tests follow the changed contracts | 18.1, 18.4, 18.5, 18.8, 18.15 |
| `services/personality_service/tests/test_mvp_*.py` | Create | One RED-first file per task | 18.1, 18.4-18.8, 18.12, 18.15-18.17 |
| `services/analytics-service/src/main.py` | Modify | IF-01 SPEC, lifespan, W0 Kafka readiness, W2 plane subscription, retention setting | 18.2, 18.9, 18.18 |
| `services/analytics-service/src/config.py` | Modify | F821 `timedelta` | 18.2 |
| `services/analytics-service/src/consumer.py` | Modify | `handle_event`, event time, PHI-free logs, stop and malformed counting, therapy and escalation routing | 18.9, 18.11, 18.13, 18.18, 18.19 |
| `services/analytics-service/src/aggregations.py` | Modify | Timestamps, drop-not-crash, time retention and compaction, coverage start, therapy/escalation labels | 18.11, 18.18, 18.19 |
| `services/analytics-service/src/api.py` | Modify | Remove the lying health route, ingest permission and shape, UTC normalisation, staff-only reads | 18.2, 18.10, 18.11, 18.14, 18.18 |
| `services/analytics-service/src/reports.py` | Modify | UTC custom ranges, coverage, honest compliance coverage, message metric | 18.11, 18.18, 18.19 |
| `services/analytics-service/tests/{test_main,test_api,test_env_binding_p1_9}.py` | Modify | Existing tests follow IF-01 and the plane | 18.2, 18.9, 18.10 |
| `services/analytics-service/tests/test_mvp_*.py` | Create | One RED-first file per task | 18.2, 18.9-18.11, 18.13, 18.14, 18.18, 18.19 |
| `migrations/versions/021_personality_analytics.py` | Modify | Unique profile per user; RLS-forced erasure tombstone table | 18.3 |
| `tests/realinfra/personality-analytics/conftest.py` | Create | `superuser_conn`, `seed_users`, `cpm_default`, `phi_codec` (WS-17's names and semantics) | 18.3 |
| `tests/realinfra/personality-analytics/test_mvp_*.py` | Create | Slot, repository, Kafka ingestion and erasure proofs | 18.3, 18.4, 18.9, 18.12 |

Shared test text used by several personality tasks (32 words, so LIWC has real evidence):

```python
LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")
```

## Tasks

### Task WS-18.1: Personality adopts the IF-01 runtime and boots without the unused Redis and DB passwords
**Covers:** MVP-027
**Severity:** blocker
**Wave:** 0
**Depends on:** WS-04.4, WS-04.5, WS-01.7, WS-09.2, WS-02.6
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-14 (default assumed: staging is as strict as production; `PERSONALITY_ENVIRONMENT` equals `ENVIRONMENT`)
**Real infra:** no
**Files:**
- Modify: `services/personality_service/src/main.py` (delete `configure_logging`, the ProductionGuards call, the PHI-encryption block, `create_application`, `_register_middleware`, `_register_exception_handlers` and the `/`, `/ready`, `/live`, `/health` routes; `PersonalityServiceAppSettings.environment` reads `PERSONALITY_ENVIRONMENT`, then `ENVIRONMENT` (`AliasChoices`); new `events_enabled: bool` read from `EVENTS_ENABLED`, default `True`; module-level `SETTINGS`, `SPEC`, `build_personality_orchestrator`, a new `lifespan` and `app = create_service_app(...)`; keep `run_server`)
- Modify: `services/personality_service/src/config.py` (delete `DatabaseSettings`, `RedisSettings` and the `database`/`redis` fields of `PersonalityServiceConfig`; nothing reads them)
- Modify: `services/personality_service/src/events.py` (`to_kafka_event` imports `solace_events.schemas`, not `src.solace_events.schemas`)
- Modify: `services/personality_service/tests/test_infrastructure.py` (`TestConfigSettings`: drop the `_required_secrets` fixture, `test_database_settings`, `test_redis_settings` and the two imports; `test_main_config` asserts `config.detection is not None` instead of `config.database`)
- Create: `services/personality_service/tests/test_mvp_027_runtime_adoption.py`
- Test: `services/personality_service/tests/test_mvp_027_runtime_adoption.py`; run-only `tests/bootstrap/test_service_conformance.py -k personality-service` (WS-04)

**Interfaces:**
- Consumes: IF-01 `ServiceRuntimeSpec`, `ServiceRuntime`, `create_service_app(spec, *, lifespan, routers, title)`, `start_service_runtime(app, spec)`, `stop_service_runtime(runtime)`, `is_strict_environment`, `StartupDependencyError`, `runtime.pg_pool`, `runtime.register_readiness_check(name, check, *, critical=True)`, `runtime.register_shutdown(name, closer)`; IF-02 `PERSONALITY_ENVIRONMENT`, `ENVIRONMENT`, `EVENTS_ENABLED`; WS-06 `initialize_event_bridge(event_bus=, kafka_settings=, use_mock=, postgres_pool=)` and `shutdown_event_bridge()`; IF-15 `UnifiedLLMClient(LLMClientSettings()).initialize()` (never raises outside strict, WS-09.2).
- Produces: `main.SETTINGS: PersonalityServiceAppSettings`; `main.SPEC = ServiceRuntimeSpec(service_name="personality-service", environment=SETTINGS.environment, version=SETTINGS.version, log_level=SETTINGS.log_level, debug=SETTINGS.debug, uses_postgres=True, uses_redis=True, event_plane="publish_consume", handles_phi=True, audited=True, rate_limited=False, edge=False)` (the WS-04.5 conformance row; satisfies the WS-04.5 request); `def build_personality_orchestrator(runtime: ServiceRuntime, settings: PersonalityServiceAppSettings, *, llm_client: UnifiedLLMClient | None) -> PersonalityOrchestrator` (the only composition seam; 18.4, 18.12 and 18.17 extend it); readiness check `personality_orchestrator` (critical); `app.state.personality_orchestrator`, `app.state.event_bus`, `app.state.llm_client`. The bare import closes the personality half of WS-01.7's `WS-18` import-root case (R-01.12).

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_027_runtime_adoption.py`
```python
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
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_027_runtime_adoption.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `AttributeError: ... no attribute 'SPEC'`; `configure_logging` still present; `reload_config()` raises `ValidationError ... RedisSettings password Field required` (the MVP-027 crash, reproduced without any `.env`); `environment == "development"` because only `PERSONALITY_ENVIRONMENT` is read; the lifespan test raises the same `ValidationError` from `get_config()` (or, with a password in the env, `/ready` has no `checks`); `to_kafka_event` returns a `src.solace_events.schemas` class or `None`, so `isinstance` is False.
- [ ] **Step 3: Implement** `SPEC`, `SETTINGS`, `build_personality_orchestrator(runtime, settings, *, llm_client)` and the lifespan in `main.py`. The lifespan's first statement is `runtime = await start_service_runtime(app, SPEC)` and its last is `await stop_service_runtime(runtime)`. Between them, with no try/except around platform components: the `get_config()` ensemble-weight warning; `llm_client = UnifiedLLMClient(LLMClientSettings())` and `await llm_client.initialize()`; `orchestrator = build_personality_orchestrator(runtime, SETTINGS, llm_client=llm_client)`; `await orchestrator.initialize()`; readiness check `personality_orchestrator` returning `(orchestrator.is_initialized, "initialized" | "not_initialized")`; shutdown closers for the orchestrator and the LLM client; `event_bus = EventBus()` always, stored on `app.state.event_bus`; when `SETTINGS.events_enabled`, `await initialize_event_bridge(event_bus=event_bus, postgres_pool=runtime.pg_pool)` plus a `personality_event_bridge` shutdown closer; when disabled and `is_strict_environment(SETTINGS.environment)`, raise `StartupDependencyError("personality-service requires EVENTS_ENABLED=true in strict environments")`. The lifespan reads the module global `SETTINGS` at call time. `build_personality_orchestrator` builds `PostgresPersonalityRepository(ConnectionPoolManager)` only when `runtime.pg_pool is not None` (else `None`, dev/test only), `TraitDetector(TraitDetectorSettings(enable_llm_detection=settings.enable_llm_detection and llm_client is not None), llm_client)`, `StyleAdapter()` and the orchestrator exactly as today. `app = create_service_app(SPEC, lifespan=lifespan, routers=[(router, "/api/v1/personality")], title="Solace-AI Personality Service")`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** one process per suite. Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests -o "addopts=" -p no:logging -q`, then the same prefix with `tests/bootstrap/test_service_conformance.py -k personality-service`, then with `tests/integration`.
Expected: all pass. `tests/integration/test_phi_encryption_activation.py` must already have had its `configure_phi_encryption` AST check retired by WS-04 (CR-8 / R-17-03); if it still lists `personality_service`, stop and record a ledger Ruling rather than editing that file. `tests/build/test_import_roots.py -k WS-18` turns green only when WS-18.2 has also landed (record in the ledger).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.2: Analytics adopts the IF-01 runtime; a dead Kafka consumer shows in /ready; the lint gate is clean
**Covers:** MVP-110
**Severity:** medium
**Wave:** 0
**Depends on:** WS-04.4, WS-04.5, WS-01.7
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-18 (default assumed: analytics is a non-critical staff dashboard with no PHI), D-14 (default assumed: staging is strict)
**Real infra:** no
**Files:**
- Modify: `services/analytics-service/src/main.py` (delete `_setup_prometheus` (the runtime serves `/metrics`), `_setup_opentelemetry`, `create_app`, the inline `structlog.configure`, the ProductionGuards call and the `/`, `/health`, `/ready` routes; `ServiceConfig.env: Literal["development", "test", "staging", "production"]` read from `ANALYTICS_SERVICE_ENV`, then `ENVIRONMENT` (`AliasChoices`); `if TYPE_CHECKING:` imports of `AnalyticsAggregator`, `ReportService`, `AnalyticsConsumer` for the forward references (F821); module-level `from solace_events.consumer import create_consumer as create_kafka_consumer` and `from solace_events.config import KafkaSettings`; module-level `SETTINGS`, `SPEC`, `async def start_kafka_ingestion(runtime, consumer, settings) -> None`, a new lifespan, `app = create_service_app(...)`)
- Modify: `services/analytics-service/src/config.py` (`from datetime import timedelta` at module level; `get_retention_timedelta(self) -> timedelta` unquoted)
- Modify: `services/analytics-service/src/api.py` (delete the `/api/v1/analytics/health` route: it reported the in-process queue as `consumer_running`, never Kafka)
- Modify: `services/analytics-service/tests/test_main.py` (delete `TestSetupPrometheus`, `TestSetupOpenTelemetry`, `TestCreateApp`; `TestAnalyticsServiceSettings::test_default_settings` and `TestServiceConfig::test_default_values` delete `ENVIRONMENT` and `ANALYTICS_SERVICE_ENV` first)
- Modify: `services/analytics-service/tests/test_api.py` (delete `TestHealthEndpoint`)
- Modify: `services/analytics-service/tests/test_env_binding_p1_9.py` (`test_default_is_development_when_unset` also deletes `ENVIRONMENT`)
- Create: `services/analytics-service/tests/test_mvp_110_runtime_adoption.py`
- Test: the new file; run-only `tests/bootstrap/test_service_conformance.py -k analytics-service`, `tests/build/test_mvp_038_lint_gate.py`, `tests/build/test_import_roots.py`

**Interfaces:**
- Consumes: IF-01 names (as in WS-18.1) plus `runtime.register_background_task(name, task)`; legacy `create_consumer(group_id=, kafka_settings=)` and `register_default_handler` (until WS-18.9).
- Produces: `main.SETTINGS: AnalyticsServiceSettings`; `main.SPEC = ServiceRuntimeSpec(service_name="analytics-service", environment=SETTINGS.service.env, version="1.0.0", log_level=SETTINGS.service.log_level, uses_postgres=True, uses_redis=True, event_plane="publish_consume", handles_phi=False, audited=False, rate_limited=False, edge=False)` (WS-04.5 row; satisfies the WS-04.5 request); `async def start_kafka_ingestion(runtime: ServiceRuntime, consumer: AnalyticsConsumer, settings: AnalyticsServiceSettings) -> None` with readiness check `kafka_consumer` (critical) and background task `kafka_consumer` (both replaced by the plane in WS-18.9); `app = create_service_app(SPEC, lifespan=lifespan, routers=[(router, "")], title="Solace-AI Analytics Service")`. Closes the analytics half of WS-01.7's `WS-18` import-root and lint cases (R-01.12).

- [ ] **Step 1: Write the failing test** in `services/analytics-service/tests/test_mvp_110_runtime_adoption.py`
```python
"""MVP-110 + IF-01 adoption for analytics-service (WS-18.2)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import main as analytics_main  # noqa: E402
from aggregations import AnalyticsAggregator  # noqa: E402
from consumer import AnalyticsConsumer  # noqa: E402


def _failing_consumer(**kwargs):
    raise ConnectionError("broker unreachable")


def _kafka_settings(env: str | None = None):
    settings = analytics_main.SETTINGS.model_copy(deep=True)
    settings.consumer.kafka_enabled = True
    if env:
        settings.service.env = env
    return settings


def test_if01_analytics_spec_flags() -> None:
    spec = analytics_main.SPEC
    assert spec.service_name == "analytics-service"
    assert (spec.uses_postgres, spec.uses_redis, spec.event_plane) == (True, True, "publish_consume")
    assert (spec.handles_phi, spec.audited, spec.rate_limited, spec.edge) == (False, False, False, False)


def test_if01_analytics_has_no_local_probes_or_instrumentator() -> None:
    for name in ("create_app", "_setup_prometheus", "_setup_opentelemetry"):
        assert not hasattr(analytics_main, name), name
    assert "/api/v1/analytics/health" not in {route.path for route in analytics_main.app.routes}


def test_if01_service_env_accepts_test_and_follows_environment(monkeypatch) -> None:
    monkeypatch.delenv("ANALYTICS_SERVICE_ENV", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")
    assert analytics_main.ServiceConfig().env == "test"


def test_mvp_110_consumer_failure_is_visible_in_ready(monkeypatch) -> None:
    monkeypatch.setattr(analytics_main, "SETTINGS", _kafka_settings())
    monkeypatch.setattr(analytics_main, "create_kafka_consumer", _failing_consumer)
    with TestClient(analytics_main.app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["kafka_consumer"]["ok"] is False


async def test_mvp_110_consumer_failure_is_fatal_in_strict(monkeypatch) -> None:
    from solace_infrastructure.runtime import StartupDependencyError

    monkeypatch.setattr(analytics_main, "create_kafka_consumer", _failing_consumer)
    runtime = SimpleNamespace(register_readiness_check=lambda *a, **k: None,
                              register_background_task=lambda *a, **k: None,
                              register_shutdown=lambda *a, **k: None)
    with pytest.raises(StartupDependencyError):
        await analytics_main.start_kafka_ingestion(
            runtime, AnalyticsConsumer(aggregator=AnalyticsAggregator()), _kafka_settings("staging"))
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests/test_mvp_110_runtime_adoption.py -o "addopts=" -p no:logging -q`
Expected: FAIL. No `SPEC`/`SETTINGS`; `create_app` and `/api/v1/analytics/health` exist; `ServiceConfig(env)` rejects `"test"` with a `ValidationError`; the lifespan swallows the consumer failure (`analytics_kafka_consumer_failed`) and `/ready` answers `200 {"status": "ready"}` with no `checks` (MVP-110); `start_kafka_ingestion` does not exist.
- [ ] **Step 3: Implement** in `main.py`: the lifespan is `runtime = await start_service_runtime(app, SPEC)` first, then `_create_services(SETTINGS)`, `set_dependencies(...)`, `await consumer.start()`, `await start_kafka_ingestion(runtime, consumer, SETTINGS)`, `runtime.register_shutdown("analytics_consumer", consumer.stop)`, `yield`, `await stop_service_runtime(runtime)` last. `start_kafka_ingestion`: when `settings.consumer.kafka_enabled` is False, strict raises `StartupDependencyError("analytics-service requires its Kafka consumer in strict environments")`, otherwise it registers `kafka_consumer` as `(False, "kafka consumer disabled")`. When enabled, it builds the consumer with `create_kafka_consumer(group_id=..., kafka_settings=KafkaSettings(bootstrap_servers=...))`, keeps today's default handler and DLQ wiring (WS-18.9 replaces both) but imports them from `solace_events.dead_letter` and `solace_events.schemas` (no `src.` root anywhere in `main.py`), starts it, registers the consume task with `register_background_task("kafka_consumer", task)` and a check returning `(True, "consuming")`. Any exception: strict raises `StartupDependencyError(...)` from it (the message names no exception text); development/test logs `analytics_kafka_consumer_failed` with `error_type` only and registers `kafka_consumer` as `(False, "kafka consumer not started")`. In `config.py` import `timedelta` at module level.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run, one process each with the Step 2 prefix: `services/analytics-service/tests`; `tests/bootstrap/test_service_conformance.py -k analytics-service`; `tests/build/test_mvp_038_lint_gate.py`; `tests/build/test_import_roots.py`; `tests/security/test_config_secrets_hardening.py`. Expected: all pass (the lint gate's `WS-18` case reports no F821 in `main.py` or `config.py`; the `WS-18` import-root case passes once WS-18.1 is in). The deleted tests are listed in the ledger with the new ones as replacements.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.3: Slot 021: one profile per user and an RLS-isolated erasure tombstone
**Covers:** none (schema prerequisite for MVP-080 and MVP-172)
**Severity:** high (W2 exit: repositories match the migrated schema under RLS)
**Wave:** 2 (phase A, schema lane, after slot 020)
**Depends on:** WS-02.3, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: Alembic is the schema of record; the repository adapts and no profile columns change), D-02 (solace_app, NOBYPASSRLS), D-24 (default assumed: after erasure only the user id and the erasure time remain, as a tombstone)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/021_personality_analytics.py` (`upgrade()`/`downgrade()` bodies only; `revision`/`down_revision` untouched)
- Create: `tests/realinfra/personality-analytics/conftest.py` (fixtures `superuser_conn`, `seed_users`, `cpm_default`, `phi_codec`)
- Create: `tests/realinfra/personality-analytics/test_mvp_021_personality_slot.py`
- Test: the new test file; run-only `tests/migrations/test_chain_real_pg.py` (WS-02)

**Interfaces:**
- Consumes: IF-19 `pg_migrated -> PgHandle(dsn_app, dsn_migrator, dsn_superuser, app_pool)`; IF-03 NULLIF policy form; 009 default privileges (DML for solace_app on new tables).
- Produces: unique index `uq_personality_profiles_user_id` on `personality_profiles(user_id)`; table `personality_erasure_tombstones (user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, erased_at TIMESTAMPTZ NOT NULL DEFAULT now())` with RLS enabled and forced and policy `personality_erasure_tombstones_by_user` (USING and WITH CHECK `user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid`). Realinfra fixtures with WS-17's names and semantics, reused by WS-18.4 and WS-18.12: `superuser_conn` (asyncpg connection to `dsn_superuser`), `seed_users` (async `seed_users(n) -> list[UUID]`, synthetic users inserted as superuser with `id, created_at, updated_at, encryption_key_id, email, password_hash, display_name`; extend the column list if slot 012 adds a NOT NULL column without a default, the asyncpg error names it), `cpm_default` (POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE from `urllib.parse.urlsplit(pg_migrated.dsn_app)` plus `POSTGRES_SSL_MODE=disable`, `ConnectionPoolManager.close_all_pools()` then `get_pool("default")`, all pools closed on teardown), `phi_codec` (synthetic `ENCRYPTION_MASTER_KEY` fixed 64-hex test value, `ENCRYPTION_SEARCH_HASH_SALT`, `ENCRYPTION_KEY_ID=test-k1`, `reset_phi_codec()`, returns `get_phi_codec()`).

- [ ] **Step 1: Write the failing test** in `tests/realinfra/personality-analytics/test_mvp_021_personality_slot.py`
```python
"""Slot 021: one personality profile per user and an RLS-isolated erasure tombstone (WS-18.3)."""
from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
INSERT_PROFILE = ("INSERT INTO personality_profiles (id, user_id, created_at, updated_at, encryption_key_id) "
                  "VALUES ($1, $2, now(), now(), 'test-k1')")
INSERT_TOMBSTONE = "INSERT INTO personality_erasure_tombstones (user_id) VALUES ($1)"


async def test_slot021_allows_one_profile_per_user(superuser_conn, seed_users) -> None:
    (user_id,) = await seed_users(1)
    await superuser_conn.execute(INSERT_PROFILE, uuid4(), user_id)
    with pytest.raises(asyncpg.UniqueViolationError):
        await superuser_conn.execute(INSERT_PROFILE, uuid4(), user_id)


async def test_slot021_tombstone_table_has_forced_rls_with_nullif(superuser_conn) -> None:
    row = await superuser_conn.fetchrow(
        "SELECT c.relrowsecurity, c.relforcerowsecurity, p.qual, p.with_check FROM pg_class c "
        "JOIN pg_policies p ON p.tablename = c.relname WHERE c.relname = 'personality_erasure_tombstones'")
    assert row is not None and row["relrowsecurity"] and row["relforcerowsecurity"]
    assert "NULLIF" in row["qual"].upper() and "NULLIF" in row["with_check"].upper()


async def test_slot021_solace_app_cannot_tombstone_or_see_another_user(pg_migrated, seed_users) -> None:
    owner, intruder = await seed_users(2)
    conn = await asyncpg.connect(pg_migrated.dsn_app)
    try:
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", str(intruder))
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute(INSERT_TOMBSTONE, owner)
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", str(owner))
        await conn.execute(INSERT_TOMBSTONE, owner)
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", str(intruder))
        assert await conn.fetchval("SELECT count(*) FROM personality_erasure_tombstones") == 0
    finally:
        await conn.close()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/personality-analytics/test_mvp_021_personality_slot.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The second profile insert for the same user succeeds (001 indexes `user_id` without uniqueness, so two profiles can exist and `get_profile_by_user` reads an arbitrary one); the tombstone query returns `None`; the third test raises `UndefinedTableError`.
- [ ] **Step 3: Implement** `upgrade()`: `op.create_index("uq_personality_profiles_user_id", "personality_profiles", ["user_id"], unique=True)`; `op.create_table("personality_erasure_tombstones", ...)` as specified; `ALTER TABLE ... ENABLE ROW LEVEL SECURITY`, `... FORCE ROW LEVEL SECURITY`, `DROP POLICY IF EXISTS personality_erasure_tombstones_by_user`, `CREATE POLICY ... USING (<NULLIF>) WITH CHECK (<NULLIF>)`. `downgrade()` drops the policy, the table and the index in reverse order. No other table is touched; 008/009 own the existing personality policies.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations/test_chain_real_pg.py -o "addopts=" -p no:logging -q`. Expected: all pass (upgrade head, downgrade base, upgrade head; `alembic_version` equals `021_personality_analytics`). WS-02 reviews the slot diff.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.4: Personality profiles persist on the migrated schema, encrypted through the PHI codec, under RLS
**Covers:** MVP-080
**Severity:** high
**Wave:** 2
**Depends on:** WS-18.3, WS-18.1, WS-09.7, WS-04.8, WS-07.1, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: repository SQL matches 001 plus slot 021; `trait_assessments` and `profile_snapshots` are not written in the MVP)
**Real infra:** yes
**Files:**
- Modify: `services/personality_service/src/infrastructure/postgres_repository.py` (`from solace_security.phi_codec import PHICodec` (the WS-09.15 ADOPTERS marker); `PERSONALITY_PHI_FIELDS = ("ocean_scores", "communication_style", "assessment_history")`; `__init__(self, *, codec: PHICodec, pool_name: str = "default")`; delete `POOL_NAME`, the FeatureFlags import and the legacy-client `_acquire`; writes run in `ConnectionPoolManager.transaction(self._pool_name)`, reads in `.acquire(self._pool_name)`; `save_profile` upserts `id, user_id, ocean_scores, communication_style, assessment_count, stability_score, assessment_history, created_at, updated_at, version, encryption_key_id` with `ON CONFLICT (user_id) DO UPDATE` (every column except `id`, `user_id`, `created_at`); `_row_to_profile` decrypts; `save_assessment` and `save_snapshot` raise `NotImplementedError` before any SQL)
- Modify: `services/personality_service/src/infrastructure/repository.py` (`RepositoryFactory`: delete `configure`, `get_default`, `reset` and the class-level singleton state; `create_postgres(codec, *, pool_name="default")`; `create_unit_of_work(repository)` requires the repository)
- Modify: `services/personality_service/src/main.py` (`build_personality_orchestrator`: `PostgresPersonalityRepository(codec=runtime.phi_codec)` only when `runtime.pg_pool is not None and runtime.phi_codec is not None`; when `is_strict_environment(settings.environment)` and either is `None`, raise `StartupDependencyError("personality-service requires Postgres and the PHI codec")` instead of falling back to the in-memory repository)
- Modify: `services/personality_service/tests/test_infrastructure.py` (`TestRepositoryFactory`: the two `get_default` tests become `test_create_postgres_binds_the_codec(self, fake_codec)`, asserting `RepositoryFactory.create_postgres(fake_codec)._codec is fake_codec`)
- Create: `services/personality_service/tests/conftest.py` (`FakeCodec`, fixture `fake_codec`, `RecordingConnection`, fixture `recording_cpm`)
- Create: `services/personality_service/tests/test_mvp_080_profile_persistence.py`
- Create: `tests/realinfra/personality-analytics/test_mvp_080_personality_repository.py`
- Test: both new test files

**Interfaces:**
- Consumes: IF-13 `PHICodec.encrypt(value, *, field, owner_id)`, `.decrypt(...)`, `.active_key_id`; IF-01 `runtime.phi_codec`, `is_strict_environment`, `StartupDependencyError` (`from solace_infrastructure.runtime import ...`); IF-03 `ConnectionPoolManager.acquire/transaction` (GUC from the request context); WS-18.3 unique index and fixtures; `acting_as_user`.
- Produces: `PostgresPersonalityRepository(*, codec: PHICodec, pool_name: str = "default")`; `PERSONALITY_PHI_FIELDS` with AAD field names `personality_profiles.ocean_scores`, `personality_profiles.communication_style`, `personality_profiles.assessment_history` and `owner_id=user_id`; each PHI column holds the JSON string `codec.encrypt(json.dumps(value, default=_decimal_encoder))`, passed as a Python `str` so the pool's JSONB codec serializes it (no `json.dumps` of the envelope); `encryption_key_id = codec.active_key_id`. Conftest (WS-17 semantics): `FakeCodec.active_key_id = "fake-k1"`, `encrypt` returns `"enc:v1:fake-k1:" + urlsafe_b64encode(f"{field}|{owner_id}|{value}")`, `decrypt` reverses it and raises `ValueError("AAD mismatch")` when field or owner differ; `recording_cpm` monkeypatches `ConnectionPoolManager.acquire` and `.transaction` with `asynccontextmanager` fakes that append the pool name to `.pool_names` and yield one `RecordingConnection` (`execute(sql, *args)` records into `.executed` and returns `.status`, default `"INSERT 0 1"`; `fetch` returns `.rows`; `fetchrow` returns `rows[0]` or `None`; `fetchval` returns `None`). Satisfies the WS-09.15 (CR-10) request for the ADOPTERS file `services/personality_service/src/infrastructure/postgres_repository.py` (import, `active_key_id`, no retired encryptor names, pinned by `test_cr10_repository_adopts_the_if13_codec_and_no_legacy_encryptor`). A strict build without Postgres or the codec fails startup.

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_080_profile_persistence.py`
```python
"""MVP-080: the profile upsert matches the migrated schema and encrypts PHI (WS-18.4)."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from services.personality_service.src.domain.entities import PersonalityProfile as EntityProfile
from services.personality_service.src.domain.value_objects import OceanScores
from services.personality_service.src.infrastructure.postgres_repository import PostgresPersonalityRepository


def _insert_values(sql: str, args: tuple) -> dict:
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    return dict(zip(columns, args))


async def test_mvp_080_upsert_writes_encryption_key_id_and_ciphertext(recording_cpm, fake_codec) -> None:
    user_id = uuid4()
    entity = EntityProfile(user_id=user_id, ocean_scores=OceanScores.neutral(), assessment_count=1)
    await PostgresPersonalityRepository(codec=fake_codec).save_profile(entity)
    sql, args = recording_cpm.connection.executed[0]
    values = _insert_values(sql, args)
    assert recording_cpm.pool_names == ["default"]
    assert "ON CONFLICT (user_id)" in sql
    assert values["encryption_key_id"] == "fake-k1"
    for column in ("ocean_scores", "communication_style", "assessment_history"):
        assert values[column].startswith("enc:v1:fake-k1:"), column
    plain = fake_codec.decrypt(values["ocean_scores"], field="personality_profiles.ocean_scores", owner_id=user_id)
    assert '"openness": 0.5' in plain


async def test_mvp_080_dead_assessment_and_snapshot_writers_fail_without_sql(recording_cpm, fake_codec) -> None:
    repo = PostgresPersonalityRepository(codec=fake_codec)
    with pytest.raises(NotImplementedError):
        await repo.save_assessment(object())
    with pytest.raises(NotImplementedError):
        await repo.save_snapshot(object())
    assert recording_cpm.connection.executed == []


LEGACY = {"configure_phi_encryption", "get_phi_encryptor", "FieldEncryptor", "create_field_encryptor"}


def test_cr10_repository_adopts_the_if13_codec_and_no_legacy_encryptor() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    repo_tree = ast.parse((src / "infrastructure" / "postgres_repository.py").read_text(encoding="utf-8"))
    assert "solace_security.phi_codec" in {n.module for n in ast.walk(repo_tree) if isinstance(n, ast.ImportFrom)}
    offenders = []
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        names |= {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if names & LEGACY:
            offenders.append(path.name)
    assert offenders == []


def test_mvp_080_strict_build_without_the_codec_fails_instead_of_dropping_durability() -> None:
    from services.personality_service.src import main as personality_main
    from solace_infrastructure.runtime import StartupDependencyError

    runtime = SimpleNamespace(pg_pool=object(), phi_codec=None, redis=None, event_plane=None)
    settings = personality_main.SETTINGS.model_copy(update={"environment": "staging"})
    with pytest.raises(StartupDependencyError):
        personality_main.build_personality_orchestrator(runtime, settings, llm_client=None)
```
and in `tests/realinfra/personality-analytics/test_mvp_080_personality_repository.py`
```python
"""MVP-080 on the migrated schema as solace_app under RLS (WS-18.4)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.personality_service.src import main as personality_main
from services.personality_service.src.domain.entities import PersonalityProfile as EntityProfile
from services.personality_service.src.domain.value_objects import OceanScores
from services.personality_service.src.infrastructure.postgres_repository import PostgresPersonalityRepository
from services.personality_service.src.schemas import DetectPersonalityRequest
from solace_common.request_context import acting_as_user

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")


async def test_mvp_080_profile_round_trips_as_ciphertext(cpm_default, seed_users, phi_codec, superuser_conn) -> None:
    (user_id,) = await seed_users(1)
    repo = PostgresPersonalityRepository(codec=phi_codec)
    with acting_as_user(str(user_id)):
        await repo.save_profile(EntityProfile(user_id=user_id, ocean_scores=OceanScores.neutral(), assessment_count=1))
        await repo.save_profile(EntityProfile(user_id=user_id, ocean_scores=OceanScores.neutral(), assessment_count=2))
        loaded = await repo.get_profile_by_user(user_id)
    assert loaded is not None and loaded.assessment_count == 2
    assert loaded.ocean_scores.openness == OceanScores.neutral().openness
    rows = await superuser_conn.fetch(
        "SELECT ocean_scores::text AS o, assessment_history::text AS h, encryption_key_id "
        "FROM personality_profiles WHERE user_id = $1", user_id)
    assert len(rows) == 1
    assert rows[0]["o"].startswith('"enc:v1:') and rows[0]["h"].startswith('"enc:v1:')
    assert rows[0]["encryption_key_id"] == phi_codec.active_key_id


async def test_mvp_080_rls_hides_another_users_profile(cpm_default, seed_users, phi_codec) -> None:
    owner, intruder = await seed_users(2)
    repo = PostgresPersonalityRepository(codec=phi_codec)
    with acting_as_user(str(owner)):
        await repo.save_profile(EntityProfile(user_id=owner, ocean_scores=OceanScores.neutral()))
    with acting_as_user(str(intruder)):
        assert await repo.get_profile_by_user(owner) is None


async def test_mvp_080_detect_persists_through_the_composition_seam(cpm_default, seed_users, phi_codec,
                                                                    superuser_conn) -> None:
    (user_id,) = await seed_users(1)
    runtime = SimpleNamespace(pg_pool=object(), phi_codec=phi_codec, redis=None, event_plane=None)
    orchestrator = personality_main.build_personality_orchestrator(runtime, personality_main.SETTINGS, llm_client=None)
    await orchestrator.initialize()
    with acting_as_user(str(user_id)):
        await orchestrator.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    assert await superuser_conn.fetchval("SELECT count(*) FROM personality_profiles WHERE user_id = $1", user_id) == 1
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_080_profile_persistence.py -o "addopts=" -p no:logging -q`, then `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/personality-analytics/test_mvp_080_personality_repository.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `PostgresPersonalityRepository(codec=...)` raises `TypeError: ... unexpected keyword argument 'codec'` (the constructor takes a legacy client); `postgres_repository.py` has no `solace_security.phi_codec` import (the CR-10 test fails on its first assertion; its legacy-name half passes as a guard because WS-18.1 deleted the `configure_phi_encryption` block that `main.py:100-105` holds today); the strict build returns an orchestrator on the in-memory repository instead of raising; on the old code path the INSERT column list has no `encryption_key_id`, so real Postgres rejects it with `NotNullViolationError` and the composition-seam test finds `count == 0` because the failure is logged and swallowed outside production (MVP-080).
- [ ] **Step 3: Implement** the repository as listed. The three PHI columns are NOT NULL, so `None` scores or style are written as the encrypted `{}`. `_row_to_profile` decrypts each PHI column when it is a `str` (`json.loads(codec.decrypt(value, field=f"personality_profiles.{column}", owner_id=row["user_id"]))`), treating `{}` as `None` for the scores and style and `[]` as an empty history; a non-`str` value raises `ValueError("unencrypted personality column")` (no legacy rows exist: no profile could ever be written). Nothing is logged except ids and counts. `build_personality_orchestrator` checks `is_strict_environment(settings.environment)` before composing and raises `StartupDependencyError` when `runtime.pg_pool` or `runtime.phi_codec` is `None`.
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run, one process each with the Step 2 prefixes: `services/personality_service/tests`; `tests/realinfra/personality-analytics` (with `$env:SOLACE_REQUIRE_REALINFRA="1"`); `tests/solace_security/test_mvp_055_codec_adoption.py` if WS-09.15 has landed (the personality ADOPTERS row must pass). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.5: Persist first, then cache: a failed durable write changes nothing and is reported
**Covers:** MVP-304
**Severity:** low
**Wave:** 2
**Depends on:** WS-18.4
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Create: `services/personality_service/src/domain/errors.py` (`class ProfilePersistenceError(RuntimeError)`)
- Modify: `services/personality_service/src/domain/service.py` (`_get_cached_profile`: a repository read failure raises `ProfilePersistenceError` instead of returning `None`, so a fresh profile can never overwrite the stored one; `_persist_profile`: any repository failure raises `ProfilePersistenceError` from the original in every environment and logs `profile_persist_failed` with `user_id` and `error_type` only; `detect_personality` and `update_profile_from_scores` apply the assessment to `copy.deepcopy` of the cached profile (or a new one), persist it, and only then save it to the `ProfileStore`)
- Modify: `services/personality_service/src/api.py` (`/detect` and `/profile/update` map `ProfilePersistenceError` to `HTTPException(503, detail="Personality store unavailable")`)
- Modify: `services/personality_service/tests/test_persist_profile.py` (`test_persist_tolerates_failure_in_dev` becomes `test_persist_raises_in_every_environment`, expecting `ProfilePersistenceError` under `ENVIRONMENT=development`)
- Create: `services/personality_service/tests/test_mvp_304_persist_before_cache.py`
- Test: the new file and `test_persist_profile.py`

**Interfaces:**
- Consumes: `PersonalityRepositoryPort.save_profile/get_profile_by_user`.
- Produces: `ProfilePersistenceError` in `services.personality_service.src.domain.errors`; HTTP 503 `{"detail": "Personality store unavailable"}` (IF-11 consumers treat it as degraded `personality_unavailable`).

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_304_persist_before_cache.py`
```python
"""MVP-304: the cache never runs ahead of the database (WS-18.5)."""
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.personality_service.src.api import router
from services.personality_service.src.domain.errors import ProfilePersistenceError
from services.personality_service.src.domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from services.personality_service.src.schemas import DetectPersonalityRequest
from services.personality_service.tests.fixtures import InMemoryPersonalityRepository
from solace_security.middleware import AuthenticatedService, get_current_service

LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")


class _FlakyRepo(InMemoryPersonalityRepository):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def save_profile(self, profile) -> None:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("DB connection lost")
        await super().save_profile(profile)


class _UnreadableRepo(InMemoryPersonalityRepository):
    async def get_profile_by_user(self, user_id):
        raise RuntimeError("DB connection lost")


def _orch(repo) -> PersonalityOrchestrator:
    return PersonalityOrchestrator(settings=PersonalityServiceSettings(enable_llm_detection=False),
                                   llm_client=None, repository=repo)


async def test_mvp_304_failed_persist_leaves_the_cache_untouched() -> None:
    orch = _orch(_FlakyRepo(failures=3))
    user_id = uuid4()
    for _ in range(3):
        with pytest.raises(ProfilePersistenceError):
            await orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    assert orch._profile_store.get(user_id) is None


async def test_mvp_304_a_retry_applies_the_assessment_once() -> None:
    repo = _FlakyRepo(failures=1)
    orch = _orch(repo)
    user_id = uuid4()
    with pytest.raises(ProfilePersistenceError):
        await orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    await orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    assert orch._profile_store.get(user_id).assessment_count == 1
    assert (await repo.get_profile_by_user(user_id)).assessment_count == 1


async def test_mvp_304_unreadable_store_never_overwrites_with_a_fresh_profile() -> None:
    repo = _UnreadableRepo()
    with pytest.raises(ProfilePersistenceError):
        await _orch(repo).detect_personality(DetectPersonalityRequest(user_id=uuid4(), text=LONG_TEXT))
    assert repo._stats["profiles_saved"] == 0


def test_mvp_304_api_maps_a_failed_persist_to_503_without_detail() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/personality")
    app.state.personality_orchestrator = _orch(_FlakyRepo(failures=1))
    app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="orchestrator-service", permissions=["service:write:personality", "service:read:personality"])
    response = TestClient(app).post("/api/v1/personality/detect", json={"user_id": str(uuid4()), "text": LONG_TEXT})
    assert response.status_code == 503
    assert "DB connection lost" not in response.text
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_304_persist_before_cache.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ModuleNotFoundError: ... domain.errors`. With the module stubbed: the save error is swallowed outside production (`DID NOT RAISE`) while the cached profile has advanced; the retry case caches `assessment_count == 2` against a stored `1`; the unreadable store logs a warning and saves a fresh profile (`profiles_saved == 1`); the API returns 200 (MVP-304).
- [ ] **Step 3: Implement** the listed changes. Apply the assessment to a deep copy so the cached object is never mutated before the durable write; with no repository (development/test only) the copy goes straight to the cache.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_history_persist_roundtrip.py` still reloads history from the repository).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.6: /detect runs the LLM when asked, labels the source it used and never mixes in a fabricated LLM score
**Covers:** MVP-171 (and the WS-09.4 CR-7 request)
**Severity:** medium
**Wave:** 2
**Depends on:** WS-18.5, WS-09.3, WS-09.4
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-04 (default assumed: whether the LLM may receive the text is decided by the shared client's PHI-egress gate, exposed as `is_available`; personality never bypasses it)
**Real infra:** no
**Files:**
- Modify: `services/personality_service/src/domain/trait_detector.py` (new `@dataclass(frozen=True) class DetectionOutcome` with `scores: OceanScoresDTO` and `contributing_sources: tuple[AssessmentSource, ...]`; new `async def assess(self, text: str, sources: list[AssessmentSource] | None = None) -> DetectionOutcome`; `detect()` returns `(await self.assess(text, sources)).scores`; LIWC runs when `sources` contains `ENSEMBLE`, `LIWC_FEATURES` or `TEXT_ANALYSIS`; the LLM runs when `sources` contains `ENSEMBLE` or `LLM_ZERO_SHOT`, LLM detection is enabled and `getattr(llm_client, "is_available", True)` is true; `LLMBasedDetector.detect` returns `None` on any client error (including `LLMUnavailableError`) or a reply with no parseable JSON object, and `_default_result` is deleted; a result contributes when its ensemble weight times its confidence is positive)
- Modify: `services/personality_service/src/domain/service.py` (`detect_personality` calls `assess()`; `assessment_source` is `ENSEMBLE` when two or more sources contributed, the single contributing source when one did, `request.sources[0]` when none did)
- Modify: `services/personality_service/src/main.py` (lifespan: `runtime.register_readiness_check("llm", llm_client.readiness, critical=False)`)
- Create: `services/personality_service/tests/test_mvp_171_detect_sources.py`
- Test: the new file; `test_detect_llm_robustness.py` stays green unchanged

**Interfaces:**
- Consumes: IF-15 `LLMUnavailableError`, `UnifiedLLMClient.is_available`, `UnifiedLLMClient.readiness() -> tuple[bool, str]`.
- Produces: `TraitDetector.assess(text, sources=None) -> DetectionOutcome`; readiness check `llm` (non-critical; reports whatever `llm_client.readiness()` returns, e.g. WS-09.4's `(False, "disabled: LLM_PHI_EGRESS_APPROVED is false")` in strict without approval); `LLMUnavailableError`, any other client error and an unparseable or empty reply all yield no LLM result (no test for `""`). Satisfies the WS-09.4 CR-7 request for personality, the only service in WS-18 that builds a `UnifiedLLMClient` (`services/personality_service/src/main.py:118-120`; analytics builds none).

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_171_detect_sources.py`
```python
"""MVP-171: requested sources drive what runs and what is reported (WS-18.6)."""
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from services.personality_service.src import main as personality_main
from services.personality_service.src.domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from services.personality_service.src.domain.trait_detector import TextBasedDetector, TraitDetector, TraitDetectorSettings
from services.personality_service.src.schemas import AssessmentSource as S
from services.personality_service.src.schemas import DetectPersonalityRequest, PersonalityTrait
from services.shared.infrastructure.llm_client import LLMUnavailableError

LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")
REPLY = ('{"openness": 0.9, "conscientiousness": 0.8, "extraversion": 0.7, "agreeableness": 0.6, '
         '"neuroticism": 0.2, "evidence": ["planning"]}')


class _FakeLLM:
    is_available = True

    def __init__(self, error: Exception | None = None, reply: str = REPLY) -> None:
        self.error, self.reply, self.calls = error, reply, 0

    async def generate(self, **kwargs) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply


def _orch(llm) -> PersonalityOrchestrator:
    detector = TraitDetector(TraitDetectorSettings(), llm_client=llm)
    return PersonalityOrchestrator(settings=PersonalityServiceSettings(), trait_detector=detector, llm_client=llm)


@pytest.mark.parametrize("sources,calls,label", [
    ([S.TEXT_ANALYSIS, S.LLM_ZERO_SHOT], 1, S.ENSEMBLE),
    ([S.LLM_ZERO_SHOT], 1, S.LLM_ZERO_SHOT),
    ([S.ENSEMBLE], 1, S.ENSEMBLE),
    ([S.LIWC_FEATURES], 0, S.LIWC_FEATURES),
])
async def test_mvp_171_requested_sources_decide_what_runs(sources, calls, label) -> None:
    llm = _FakeLLM()
    orch = _orch(llm)
    await orch.initialize()
    response = await orch.detect_personality(DetectPersonalityRequest(user_id=uuid4(), text=LONG_TEXT, sources=sources))
    assert (llm.calls, response.assessment_source) == (calls, label)


@pytest.mark.parametrize("error,reply", [
    (RuntimeError("provider down"), REPLY),
    (LLMUnavailableError("provider_error"), REPLY),
    (None, ""),
], ids=["provider-exception", "cr7-llm-unavailable", "cr7-empty-reply"])
async def test_mvp_171_llm_failure_adds_no_placeholder_and_is_labelled_honestly(error, reply) -> None:
    orch = _orch(_FakeLLM(error=error, reply=reply))
    await orch.initialize()
    response = await orch.detect_personality(DetectPersonalityRequest(user_id=uuid4(), text=LONG_TEXT))
    liwc = await TextBasedDetector().detect(LONG_TEXT)
    assert response.assessment_source == S.LIWC_FEATURES
    assert response.ocean_scores.openness == pytest.approx(liwc.scores[PersonalityTrait.OPENNESS])


def test_cr7_llm_readiness_is_registered_and_non_critical(monkeypatch) -> None:
    monkeypatch.setattr(personality_main, "SETTINGS",
                        personality_main.SETTINGS.model_copy(update={"events_enabled": False}))
    with TestClient(personality_main.app) as client:
        checks = client.get("/ready").json()["checks"]
    assert checks["llm"]["critical"] is False
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_171_detect_sources.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The first two parameters show `llm.calls == 0` because the LLM is gated on `ENSEMBLE` alone (trait_detector.py:269), the first also reports `ENSEMBLE` without an LLM (MVP-171); the `[ENSEMBLE]` and `[LIWC_FEATURES]` cases pass (guards); the two error cases (a provider exception and the IF-15 `LLMUnavailableError`, built positionally with its reason as WS-10 and WS-15 do) report `ENSEMBLE` and a score pulled toward the neutral 0.5 placeholder at confidence 0.3 (`trait_detector.py:183-185` catches the error and returns `_default_result`, line 223); the empty reply is worse: `_parse_response("")` returns the neutral defaults (line 192) and they count at confidence 0.65 (line 182); `checks` has no `llm` key.
- [ ] **Step 3: Implement** the listed changes. Keep the ensemble weighting rules as they are (WS-18.15 changes which results count as evidence); compute `contributing_sources` from the results that received positive weight, in the order LIWC, LLM, RoBERTa.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.7: Empty sources and zero sample counts are rejected before any state change; errors carry no exception text
**Covers:** MVP-303, MVP-306
**Severity:** low
**Wave:** 2
**Depends on:** WS-18.6
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/personality_service/src/schemas.py` (`DetectPersonalityRequest.sources`: `min_length=1`; `TraitScoreDTO.sample_count: int = Field(ge=1, default=1)`, matching the domain `TraitScore`)
- Modify: `services/personality_service/src/api.py` (`/detect` maps `ValueError` to `HTTPException(400, detail="Invalid personality request")` and `/adapt` to `"Invalid adaptation request"`; neither echoes `str(e)` nor logs it, logging `error_type` only)
- Create: `services/personality_service/tests/test_mvp_303_request_validation.py`
- Test: the new file

**Interfaces:**
- Consumes: none beyond the existing router.
- Produces: 422 for `sources: []` and for `trait_scores[].sample_count < 1` (IF-11 provider schema tightening; the orchestrator always sends one source, WS-11.11).

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_303_request_validation.py`
```python
"""MVP-303 / MVP-306: invalid requests are rejected at the boundary (WS-18.7)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.personality_service.src.api import router
from services.personality_service.src.domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedService, AuthenticatedUser, get_current_service, get_current_user

USER = "00000000-0000-4000-a000-000000000099"
LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")


def _client(orchestrator) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/personality")
    app.state.personality_orchestrator = orchestrator
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=USER, token_type=TokenType.ACCESS, roles=["user"])
    app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="orchestrator-service", permissions=["service:write:personality", "service:read:personality"])
    return TestClient(app)


def _real() -> PersonalityOrchestrator:
    return PersonalityOrchestrator(settings=PersonalityServiceSettings(enable_llm_detection=False), llm_client=None)


def test_mvp_303_empty_sources_are_422_and_change_nothing() -> None:
    orch = _real()
    response = _client(orch).post("/api/v1/personality/detect", json={"user_id": USER, "text": LONG_TEXT, "sources": []})
    assert response.status_code == 422
    assert orch._profile_store.count() == 0


def test_mvp_306_zero_sample_count_is_422_and_change_nothing() -> None:
    orch = _real()
    trait = {"trait": "openness", "value": 0.6, "confidence_lower": 0.5, "confidence_upper": 0.7, "sample_count": 0}
    scores = {"openness": 0.6, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5,
              "neuroticism": 0.5, "trait_scores": [trait]}
    response = _client(orch).post("/api/v1/personality/profile/update", json={"user_id": USER, "ocean_scores": scores})
    assert response.status_code == 422
    assert orch._profile_store.count() == 0


def test_mvp_303_value_errors_never_echo_exception_text() -> None:
    orch = MagicMock(spec=PersonalityOrchestrator)
    orch.detect_personality = AsyncMock(side_effect=ValueError("synthetic internal detail"))
    response = _client(orch).post("/api/v1/personality/detect", json={"user_id": USER, "text": LONG_TEXT})
    assert response.status_code == 400
    assert "synthetic internal detail" not in response.text
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_303_request_validation.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Empty sources are accepted and the profile is updated and cached (`count == 1`) before any response: 500 from `IndexError` at `request.sources[0]` on pre-WS-18.6 code, 200 after it (MVP-303); `sample_count=0` is accepted with 200 and cached (MVP-306); the 400 body contains `synthetic internal detail`.
- [ ] **Step 3: Implement** the listed schema and error-mapping changes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_schemas.py` builds `TraitScoreDTO` with `sample_count >= 1`).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.8: Personality routes enforce receiver-side permissions, the acting user and the orchestrator contract
**Covers:** none (supports MVP-314, owned by WS-07, and MVP-022, owned by WS-11)
**Severity:** high (supporting two blockers owned elsewhere; closes no register entry by itself)
**Wave:** 2
**Depends on:** WS-18.7, WS-07.3, WS-07.4, WS-05.7, WS-11.11
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: service-only routes use `require_service_permission` with receiver-side permissions; an obo principal acts only for its own user)
**Real infra:** no
**Files:**
- Modify: `services/personality_service/src/api.py` (`/detect` and `/profile/update` depend on `require_service_permission(ServicePermission.WRITE_PERSONALITY.value)`; `/style`, `/adapt` and `/status` on `require_service_permission(ServicePermission.READ_PERSONALITY.value)`; new `_require_acting_user(service, user_id) -> None` raising `HTTPException(403, detail="user_id does not match the acting user")` when `getattr(service, "on_behalf_of", None)` is set and differs from `str(user_id)`, called before any orchestrator call; `/profile/update` drops its user-JWT ownership check (no caller uses it; WS-07's request))
- Modify: `services/personality_service/src/domain/service.py` (`get_style` returns `style_parameters.custom_params["confidence"] = profile_confidence`, `0.0` for the default style, where WS-11's cached-style rule reads it)
- Modify: `services/personality_service/tests/test_api.py` (the `get_current_service` override grants `service:read:personality` and `service:write:personality`)
- Create: `services/personality_service/tests/test_mvp_314_personality_route_permissions.py`
- Create: `services/personality_service/tests/test_mvp_022_contract_orchestrator_personality.py` (the file name WS-11 requested)
- Test: both new files and `test_api.py`, `test_rls_guc.py`

**Interfaces:**
- Consumes: IF-04 `require_service_permission(*perms)`, `AuthenticatedService.on_behalf_of`, `find_unguarded_service_routes(app)` (WS-07.4); `ServicePermission.READ_PERSONALITY`/`WRITE_PERSONALITY` and the orchestrator's `WRITE_PERSONALITY` grant (WS-07.3); IF-20 `verify_provider(app, contracts_dir, *, auth_headers)`; `tests/contracts/orchestrator__personality/{detect,style}.json` (WS-11.11); `ServiceTokenManager.create_service_token(..., on_behalf_of=)`.
- Produces: no unguarded service route in the personality app (input to WS-07.9); 403 on an obo/body user mismatch; `custom_params.confidence` on `/style`. Satisfies the WS-07.9 request (personality half) and the WS-11.11 provider-verification request.

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_314_personality_route_permissions.py`
```python
"""Supports MVP-314: receiver-side permissions and the acting user on personality routes (WS-18.8)."""
from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.personality_service.src.api import router
from services.personality_service.src.domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from solace_security.middleware import AuthenticatedService, find_unguarded_service_routes, get_current_service
from solace_security.service_auth import ServicePermission

READ, WRITE = ServicePermission.READ_PERSONALITY.value, ServicePermission.WRITE_PERSONALITY.value
LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")


def _app(service: AuthenticatedService) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/personality")
    app.state.personality_orchestrator = PersonalityOrchestrator(
        settings=PersonalityServiceSettings(enable_llm_detection=False), llm_client=None)
    app.dependency_overrides[get_current_service] = lambda: service
    return app


def test_mvp_314_no_personality_service_route_is_unguarded() -> None:
    assert find_unguarded_service_routes(_app(AuthenticatedService(service_name="x", permissions=[]))) == []


def test_mvp_314_a_reader_cannot_write() -> None:
    client = TestClient(_app(AuthenticatedService(service_name="therapy-service", permissions=[READ])))
    user_id = str(uuid4())
    assert client.post("/api/v1/personality/detect", json={"user_id": user_id, "text": LONG_TEXT}).status_code == 403
    assert client.post("/api/v1/personality/style", json={"user_id": user_id}).status_code == 200


def test_mvp_314_obo_principal_acts_only_for_its_user() -> None:
    acting, other = str(uuid4()), str(uuid4())
    app = _app(AuthenticatedService(service_name="orchestrator-service", permissions=[READ, WRITE], on_behalf_of=acting))
    client = TestClient(app)
    assert client.post("/api/v1/personality/detect", json={"user_id": other, "text": LONG_TEXT}).status_code == 403
    assert client.post("/api/v1/personality/style", json={"user_id": other}).status_code == 403
    assert app.state.personality_orchestrator._profile_store.count() == 0
    assert client.post("/api/v1/personality/detect", json={"user_id": acting, "text": LONG_TEXT}).status_code == 200
```
and in `services/personality_service/tests/test_mvp_022_contract_orchestrator_personality.py`
```python
"""IF-11/IF-20: personality honours the orchestrator's detect and style contracts (WS-18.8)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI

from services.personality_service.src.api import router
from services.personality_service.src.domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from services.personality_service.src.schemas import DetectPersonalityRequest, GetStyleRequest
from solace_security.auth import AuthSettings
from solace_security.middleware import _get_jwt_manager
from solace_security.service_auth import ServiceTokenManager
from solace_testing.contracts import verify_provider

pytestmark = pytest.mark.contract
REPO = Path(__file__).resolve().parents[3]
CONTRACTS = REPO / "tests" / "contracts" / "orchestrator__personality"
LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")


def _orch() -> PersonalityOrchestrator:
    return PersonalityOrchestrator(settings=PersonalityServiceSettings(enable_llm_detection=False), llm_client=None)


async def test_if11_personality_satisfies_the_orchestrator_contracts(monkeypatch) -> None:
    files = sorted(CONTRACTS.glob("*.json"))
    assert {p.stem for p in files} >= {"detect", "style"}
    monkeypatch.setenv("AUTH_SECRET_KEY", "synthetic-contract-test-secret-0123456789abcdef")
    _get_jwt_manager.cache_clear()
    owners = {}
    for path in files:
        contract = json.loads(path.read_text(encoding="utf-8"))
        owners[contract["interaction"]] = contract["request"]["body_example"]["user_id"]
    tokens = ServiceTokenManager(auth_settings=AuthSettings())

    def auth_headers(interaction: str) -> dict[str, str]:
        credentials = tokens.create_service_token(
            "orchestrator-service", target_service="personality-service", on_behalf_of=owners[interaction])
        return {"Authorization": f"Bearer {credentials.token}"}

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/personality")
    app.state.personality_orchestrator = _orch()
    outcome = verify_provider(app, CONTRACTS, auth_headers=auth_headers)
    if inspect.isawaitable(outcome):
        await outcome


async def test_mvp_022_style_carries_profile_confidence_where_the_orchestrator_reads_it() -> None:
    orch = _orch()
    user_id = uuid4()
    await orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    response = await orch.get_style(GetStyleRequest(user_id=user_id))
    assert response.style_parameters.custom_params["confidence"] == pytest.approx(response.profile_confidence)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_314_personality_route_permissions.py services/personality_service/tests/test_mvp_022_contract_orchestrator_personality.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `find_unguarded_service_routes` lists `GET /api/v1/personality/status`, `POST .../adapt`, `POST .../detect` and `POST .../style` (any service token is accepted); the reader's `/detect` returns 200; the obo principal writes and reads another user's profile (200, `count == 1`); `custom_params` has no `confidence`. The contract-verification test is a provider guard and may already pass after WS-07.4: the RED evidence for this task is the permission tests.
- [ ] **Step 3: Implement** the listed route dependencies, `_require_acting_user` and the style confidence. Keep the per-request `set_current_user_id(str(request.user_id))` and its `finally` reset; `getattr` keeps the direct handler calls in `test_rls_guc.py` (which pass `service=None`) working.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run, one process each with the Step 2 prefix: `services/personality_service/tests`; `tests/contracts` (the `orchestrator__personality` pair now has provider verification). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.9: Analytics consumes the event plane with the JSON-safe event form and surfaces handler failures
**Covers:** MVP-108, MVP-236, MVP-233
**Severity:** medium
**Wave:** 2
**Depends on:** WS-18.2, WS-06.8, WS-04.6, WS-05.2
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-18 (default assumed: analytics is a non-critical staff dashboard; its consumer group is `solace-group-analytics`), D-14 (default assumed: staging is strict)
**Real infra:** yes
**Files:**
- Modify: `services/analytics-service/src/main.py` (delete `start_kafka_ingestion`, `_kafka_to_analytics`, the `create_dead_letter_handler`/`ensure_table` block, the `create_kafka_consumer`/`KafkaSettings` imports and `ConsumerConfig.kafka_enabled`/`kafka_bootstrap_servers`; new `ANALYTICS_CONSUMER_GROUP = "solace-group-analytics"`, `ANALYTICS_SUBSCRIPTIONS: dict[str, tuple[str, ...]]` built from `solace_events.schemas.EVENT_REGISTRY` by prefix (`solace.sessions` from `session.`, `solace.safety` from `safety.`, `solace.assessments` from `diagnosis.`, `solace.therapy` from `therapy.`, `homework.` and `treatment.`, `solace.memory` from `memory.`, `solace.personality` from `personality.`), `def register_analytics_ingestion(plane, consumer: AnalyticsConsumer) -> None`, `def wire_analytics_ingestion(runtime: ServiceRuntime, consumer: AnalyticsConsumer) -> None`; the lifespan calls `wire_analytics_ingestion(runtime, consumer)` instead of `start_kafka_ingestion`)
- Modify: `services/analytics-service/src/consumer.py` (new `async def handle_event(self, event: BaseEvent) -> None` that processes `event.to_dict()`: a normalisation failure (`ValueError`, `TypeError`, `KeyError` from `from_raw`) is counted and raised as `NonRetryableEventError`; an aggregator failure is counted with `record_failure` and re-raised; `process_event` keeps its bool contract for the REST path)
- Modify: `services/analytics-service/tests/test_main.py` (`TestConsumerConfigSettings` no longer asserts `kafka_enabled` or `kafka_bootstrap_servers`)
- Modify: `services/analytics-service/tests/test_mvp_110_runtime_adoption.py` (delete the two `start_kafka_ingestion` tests and their `_failing_consumer`/`_kafka_settings` helpers; the plane-wiring tests in `test_mvp_108_plane_ingestion.py` replace them; ledger note)
- Create: `services/analytics-service/tests/test_mvp_108_plane_ingestion.py`
- Create: `tests/realinfra/personality-analytics/test_mvp_108_analytics_kafka_ingestion.py`
- Test: both new files

**Interfaces:**
- Consumes: IF-05 `ServiceEventPlane.subscribe(topic, event_types, handler, *, group_id, max_attempts=5)` (allowed after `start()`, WS-06.8), `start_service_event_plane(...)`, `shutdown_service_event_plane(...)`, `NonRetryableEventError`, `BaseEvent.to_dict()`; IF-01 `runtime.event_plane`, `is_strict_environment`, `StartupDependencyError`; IF-19 `kafka_bootstrap.produce(topic, value, *, key=None)`, `pg_migrated.app_pool`.
- Produces: `ANALYTICS_CONSUMER_GROUP`, `ANALYTICS_SUBSCRIPTIONS`, `register_analytics_ingestion(plane, consumer)`, `wire_analytics_ingestion(runtime, consumer)` (strict with no plane raises `StartupDependencyError`; development/test without a plane logs a WARNING and returns, and the bootstrap's `event_plane` check is already false), `AnalyticsConsumer.handle_event(event)`. Retry, DLQ, idempotency and readiness come from the plane. Satisfies WS-06.8's request (MVP-052 completeness).

- [ ] **Step 1: Write the failing test** in `services/analytics-service/tests/test_mvp_108_plane_ingestion.py`
```python
"""MVP-108 / MVP-236 / MVP-233: analytics on the event plane (WS-18.9)."""
from __future__ import annotations

import inspect
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import main as analytics_main  # noqa: E402
from solace_events.errors import NonRetryableEventError  # noqa: E402
from solace_events.schemas import CrisisDetectedEvent  # noqa: E402

TOPICS = {"solace.sessions", "solace.safety", "solace.assessments", "solace.therapy", "solace.memory",
          "solace.personality"}


class _FakePlane:
    def __init__(self) -> None:
        self.subscriptions: list[tuple] = []

    def subscribe(self, topic, event_types, handler, *, group_id, max_attempts=5) -> None:
        self.subscriptions.append((topic, tuple(event_types), handler, group_id))


def _crisis() -> CrisisDetectedEvent:
    return CrisisDetectedEvent(user_id=uuid4(), crisis_level="CRITICAL", trigger_indicators=["category:self_harm"],
                               detection_layer=3, confidence=Decimal("0.9"), escalation_action="page")


async def test_mvp_108_plane_delivered_event_is_counted(analytics_consumer, analytics_aggregator) -> None:
    plane = _FakePlane()
    analytics_main.register_analytics_ingestion(plane, analytics_consumer)
    assert {s[0] for s in plane.subscriptions} == TOPICS
    assert {s[3] for s in plane.subscriptions} == {"solace-group-analytics"}
    safety = next(s for s in plane.subscriptions if s[0] == "solace.safety")
    assert "safety.crisis.detected" in safety[1]
    await safety[2](_crisis())
    buckets = analytics_aggregator.store._buckets["safety.crisis_events"]
    assert [b.labels for b in buckets.values()] == [{"risk_level": "CRITICAL", "detection_layer": "3"}]


async def test_mvp_233_aggregator_failure_is_raised_to_the_plane(monkeypatch, analytics_consumer,
                                                                 analytics_aggregator) -> None:
    async def _boom(*args, **kwargs):
        raise RuntimeError("aggregator down")

    monkeypatch.setattr(analytics_aggregator, "track_crisis_event", _boom)
    with pytest.raises(RuntimeError):
        await analytics_consumer.handle_event(_crisis())
    assert analytics_consumer.metrics.events_failed == 1


async def test_mvp_233_malformed_event_is_non_retryable(analytics_consumer) -> None:
    bad = SimpleNamespace(event_type="safety.crisis.detected",
                          to_dict=lambda: {"event_type": "safety.crisis.detected", "metadata": {"timestamp": "not-a-time"}})
    with pytest.raises(NonRetryableEventError):
        await analytics_consumer.handle_event(bad)


def test_mvp_236_no_legacy_consumer_or_in_memory_dlq_path_remains() -> None:
    source = inspect.getsource(analytics_main)
    for needle in ("create_dead_letter_handler", "ensure_table", "create_kafka_consumer", "model_dump()"):
        assert needle not in source, needle


def test_mvp_236_strict_without_an_event_plane_fails_startup(analytics_consumer) -> None:
    from solace_infrastructure.runtime import StartupDependencyError

    runtime = SimpleNamespace(event_plane=None, spec=SimpleNamespace(environment="staging"))
    with pytest.raises(StartupDependencyError):
        analytics_main.wire_analytics_ingestion(runtime, analytics_consumer)
```
and in `tests/realinfra/personality-analytics/test_mvp_108_analytics_kafka_ingestion.py`
```python
"""MVP-108 on a real broker: a crisis event on solace.safety reaches the analytics counters (WS-18.9)."""
from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "services" / "analytics-service" / "src"))
import main as analytics_main  # noqa: E402
from aggregations import AnalyticsAggregator, MetricsStore  # noqa: E402
from consumer import AnalyticsConsumer  # noqa: E402
from solace_events.config import KafkaSettings  # noqa: E402
from solace_events.schemas import CrisisDetectedEvent  # noqa: E402
from solace_events.service_plane import shutdown_service_event_plane, start_service_event_plane  # noqa: E402

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_108_crisis_event_on_kafka_is_counted(pg_migrated, kafka_bootstrap) -> None:
    store = MetricsStore()
    consumer = AnalyticsConsumer(aggregator=AnalyticsAggregator(store=store))
    plane = await start_service_event_plane(
        "analytics-service", pool=pg_migrated.app_pool,
        kafka_settings=KafkaSettings(bootstrap_servers=kafka_bootstrap.bootstrap), strict=False)
    try:
        analytics_main.register_analytics_ingestion(plane, consumer)
        event = CrisisDetectedEvent(user_id=uuid4(), crisis_level="HIGH", trigger_indicators=["category:self_harm"],
                                    detection_layer=1, confidence=Decimal("0.8"), escalation_action="page")
        await kafka_bootstrap.produce("solace.safety", event, key=str(event.user_id))
        for _ in range(120):
            if "safety.crisis_events" in store._buckets:
                break
            await asyncio.sleep(0.5)
        labels = [b.labels for b in store._buckets.get("safety.crisis_events", {}).values()]
        assert labels == [{"risk_level": "HIGH", "detection_layer": "1"}]
        assert consumer.metrics.events_failed == 0
    finally:
        await shutdown_service_event_plane("analytics-service")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests/test_mvp_108_plane_ingestion.py -o "addopts=" -p no:logging -q`, then `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/personality-analytics/test_mvp_108_analytics_kafka_ingestion.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `register_analytics_ingestion`, `wire_analytics_ingestion` and `handle_event` do not exist; the source still contains `create_dead_letter_handler`, `ensure_table` and `process_event(event.model_dump())`, whose datetime timestamp makes `from_raw` raise `TypeError: fromisoformat: argument must be str`, swallowed into `False`, so the offset commits and 100% of Kafka events are dropped (MVP-108, MVP-233).
- [ ] **Step 3: Implement** the listed changes. `register_analytics_ingestion` calls `plane.subscribe(topic, ANALYTICS_SUBSCRIPTIONS[topic], consumer.handle_event, group_id=ANALYTICS_CONSUMER_GROUP)` for each topic. `wire_analytics_ingestion` reads `runtime.event_plane`; strictness comes from `is_strict_environment(runtime.spec.environment)`.
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run, one process each with the Step 2 prefixes: `services/analytics-service/tests`; `tests/realinfra/personality-analytics` (with `$env:SOLACE_REQUIRE_REALINFRA="1"`); `tests/bootstrap/test_service_conformance.py -k analytics-service`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.10: /events/ingest requires WRITE_ANALYTICS and counts the fields it is given
**Covers:** MVP-235
**Severity:** low
**Wave:** 2
**Depends on:** WS-18.9, WS-07.3, WS-07.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: `service:write:analytics` is held by no service, so the route is reachable only with an explicit grant), D-18
**Real infra:** no
**Files:**
- Modify: `services/analytics-service/src/api.py` (`ingest_event` depends on `require_service_permission(ServicePermission.WRITE_ANALYTICS.value)`; the event dict puts `request.payload` keys at top level except the reserved `event_type`, `user_id`, `session_id` and `metadata`; `metadata` is `request.metadata` overwritten by the server-set `event_id`, `timestamp`, `correlation_id` and `source_service = service.service_name`)
- Modify: `services/analytics-service/tests/test_api.py` (the `get_current_service` override also grants `service:write:analytics`)
- Create: `services/analytics-service/tests/test_mvp_235_ingest_payload.py`
- Test: the new file and `test_api.py`

**Interfaces:**
- Consumes: `ServicePermission.WRITE_ANALYTICS` (WS-07.3); `require_service_permission` (WS-07.4).
- Produces: the analytics half of WS-07.9's request (no unguarded service route in analytics).

- [ ] **Step 1: Write the failing test** in `services/analytics-service/tests/test_mvp_235_ingest_payload.py`
```python
"""MVP-235: REST-ingested events are counted from their payload and need WRITE_ANALYTICS (WS-18.10)."""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from api import router, set_dependencies  # noqa: E402
from solace_security.middleware import AuthenticatedService, find_unguarded_service_routes, get_current_service  # noqa: E402
from solace_security.service_auth import ServicePermission  # noqa: E402

CRISIS = {"event_type": "safety.crisis.detected",
          "payload": {"crisis_level": "CRITICAL", "detection_layer": 3, "event_type": "system.spoof"}}


def _app(aggregator, reports, consumer, permissions) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    set_dependencies(aggregator, reports, consumer)
    app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="orchestrator-service", permissions=permissions)
    return app


def test_mvp_235_payload_fields_reach_the_counters(analytics_aggregator, report_service, analytics_consumer) -> None:
    app = _app(analytics_aggregator, report_service, analytics_consumer, [ServicePermission.WRITE_ANALYTICS.value])
    response = TestClient(app).post("/api/v1/analytics/events/ingest", json={**CRISIS, "user_id": str(uuid4())})
    assert response.status_code == 200 and response.json()["accepted"] is True
    buckets = analytics_aggregator.store._buckets["safety.crisis_events"]
    assert [b.labels for b in buckets.values()] == [{"risk_level": "CRITICAL", "detection_layer": "3"}]


def test_mvp_235_ingest_requires_write_analytics(analytics_aggregator, report_service, analytics_consumer) -> None:
    app = _app(analytics_aggregator, report_service, analytics_consumer, [ServicePermission.READ_SAFETY.value])
    response = TestClient(app).post("/api/v1/analytics/events/ingest", json={**CRISIS, "user_id": str(uuid4())})
    assert response.status_code == 403
    assert find_unguarded_service_routes(app) == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests/test_mvp_235_ingest_payload.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The crisis is counted with `{"risk_level": "NONE", "detection_layer": "1"}` because the fields sit under `payload` while the handler reads top-level keys (MVP-235); a service without `WRITE_ANALYTICS` gets 200 and the route is listed as unguarded.
- [ ] **Step 3: Implement** the listed changes in `api.py`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.11: Events are bucketed at their own time, naive query times mean UTC, and back-dated records cannot crash the store
**Covers:** MVP-237, MVP-241
**Severity:** low
**Wave:** 2
**Depends on:** WS-18.10
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-18
**Real infra:** no
**Files:**
- Modify: `services/analytics-service/src/consumer.py` (`AnalyticsEvent.from_raw`: a naive ISO timestamp is read as UTC; every `_handle_*` passes `timestamp=event.timestamp`)
- Modify: `services/analytics-service/src/aggregations.py` (`record_counter`, `record_gauge`, `record_timing` and every `track_*` method take `timestamp: datetime | None = None` and pass it to `record`; `record` keeps a local reference to the bucket it created, and a record whose bucket is pruned at once is dropped and counted in `_stats["records_dropped"]` instead of raising `KeyError`; `get_aggregated` treats naive `start_time`/`end_time` as UTC)
- Modify: `services/analytics-service/src/api.py` (new `_as_utc(dt: datetime | None) -> datetime | None` applied to `MetricQueryRequest` times and to the `/reports/generate` query times)
- Modify: `services/analytics-service/src/reports.py` (`ReportTimeRange.custom` normalises naive datetimes to UTC)
- Create: `services/analytics-service/tests/test_mvp_237_event_time.py`
- Test: the new file

**Interfaces:**
- Produces: `timestamp=` keyword on the aggregator and store recording methods; store statistic `records_dropped`; naive times are UTC at every boundary.

- [ ] **Step 1: Write the failing test** in `services/analytics-service/tests/test_mvp_237_event_time.py`
```python
"""MVP-237 / MVP-241: event-time bucketing and UTC-normalised queries (WS-18.11)."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aggregations import AggregationWindow, MetricsStore  # noqa: E402
from api import router, set_dependencies  # noqa: E402
from solace_security.auth import TokenType  # noqa: E402
from solace_security.middleware import AuthenticatedUser, get_current_user  # noqa: E402


def _crisis(at: datetime) -> dict:
    return {"event_type": "safety.crisis.detected", "user_id": str(uuid4()),
            "metadata": {"event_id": str(uuid4()), "timestamp": at.isoformat(), "correlation_id": str(uuid4()),
                         "source_service": "safety-service"},
            "crisis_level": "HIGH", "detection_layer": 1}


async def test_mvp_237_event_lands_in_the_bucket_of_its_own_time(analytics_consumer, analytics_aggregator) -> None:
    stamped = datetime.now(timezone.utc) - timedelta(days=3)
    assert await analytics_consumer.process_event(_crisis(stamped)) is True
    store, now = analytics_aggregator.store, datetime.now(timezone.utc)
    recent = await store.get_aggregated("safety.crisis_events", window_type=AggregationWindow.HOUR,
                                        start_time=now - timedelta(hours=1), end_time=now)
    then = await store.get_aggregated("safety.crisis_events", window_type=AggregationWindow.HOUR,
                                      start_time=stamped - timedelta(minutes=5), end_time=stamped + timedelta(minutes=5))
    assert recent == [] and sum(m.count for m in then) == 1


async def test_mvp_237_back_dated_record_beyond_the_cap_is_dropped_not_crashed() -> None:
    store = MetricsStore(max_windows_per_metric=100)
    now = datetime.now(timezone.utc)
    for minute in range(100):
        await store.record("m", Decimal(1), timestamp=now - timedelta(minutes=minute))
    await store.record("m", Decimal(1), timestamp=now - timedelta(days=1))
    assert (await store.get_statistics())["records_dropped"] == 1


def test_mvp_241_naive_query_times_are_utc(analytics_aggregator, report_service, analytics_consumer) -> None:
    app = FastAPI()
    app.include_router(router)
    set_dependencies(analytics_aggregator, report_service, analytics_consumer)
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=str(uuid4()), token_type=TokenType.ACCESS, roles=["admin"])
    client = TestClient(app, raise_server_exceptions=False)
    asyncio.run(analytics_consumer.process_event(_crisis(datetime.now(timezone.utc))))
    naive_start = (datetime.now(timezone.utc) - timedelta(hours=2)).replace(tzinfo=None).isoformat()
    query = client.post("/api/v1/analytics/metrics/query",
                        json={"metric_name": "safety.crisis_events", "window": "hour", "start_time": naive_start})
    assert query.status_code == 200 and query.json()["total"] == 1
    report = client.post(f"/api/v1/analytics/reports/generate?report_type=safety_overview&start_time={naive_start}")
    assert report.status_code == 200
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests/test_mvp_237_event_time.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The 3-day-old crisis sits in the current-minute bucket, so `recent` is non-empty and `then` is empty (MVP-237); the back-dated record raises `KeyError` at aggregations.py:288 (`record` reads back a bucket that `_prune_old_windows` just deleted); `get_statistics` has no `records_dropped`; both naive-time requests return 500 with `TypeError: can't compare offset-naive and offset-aware datetimes` (MVP-241).
- [ ] **Step 3: Implement** the listed changes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_window_rollup.py` and `test_consumer_to_report_e2e.py` use current timestamps).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.12: Personality erasure is final: tombstone, per-user lock and the user.deleted consumer
**Covers:** MVP-172 (supports MVP-029, owned by WS-14; the lock part of MVP-301 lands here)
**Severity:** medium
**Wave:** 3
**Depends on:** WS-18.8, WS-18.3, WS-06.8, WS-14.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: every personality row is hard-deleted; the tombstone keeps only the pseudonymous user id and the erasure time, retained like audit rows)
**Real infra:** yes
**Files:**
- Modify: `services/personality_service/src/domain/errors.py` (`class ProfileErasedError(RuntimeError)`)
- Modify: `services/personality_service/src/domain/service.py` (`ProfileStore.user_lock(user_id) -> asyncio.Lock`; `ProfileStore.save`/`delete` no longer take the per-user lock (callers hold `user_lock`; asyncio locks are not re-entrant); `PersonalityOrchestrator._erased: set[UUID]`; `detect_personality` runs detection outside the lock, then under `user_lock` raises `ProfileErasedError` for an erased user, else get-or-create, apply, persist, cache; `update_profile_from_scores` the same; a `ProfileErasedError` from the repository marks the user erased and propagates; `delete_user_data` and the deleter built by `build_erasure()` take the same lock, mark the user erased and drop the cache entry before the repository delete; new `async def purge_user_state(self, user_id: UUID) -> None` (idempotent: marks erased, drops the cache entry and its lock when unlocked); new `def build_erasure(self) -> UserDataErasure` registering `personality_postgres`; the module function `build_user_data_erasure(repository)` stays for its existing test)
- Modify: `services/personality_service/src/infrastructure/postgres_repository.py` (`save_profile` becomes `INSERT ... SELECT ... WHERE NOT EXISTS (SELECT 1 FROM personality_erasure_tombstones WHERE user_id = $2) ON CONFLICT (user_id) DO UPDATE ...`; status `INSERT 0 0` raises `ProfileErasedError`; `delete_user_data` runs in one `ConnectionPoolManager.transaction`: delete snapshots, assessments and the profile, then `INSERT INTO personality_erasure_tombstones (user_id) VALUES ($1) ON CONFLICT DO NOTHING`; it returns the rows deleted)
- Modify: `services/personality_service/tests/fixtures.py` (`InMemoryPersonalityRepository` records a tombstone in `delete_user_data` and raises `ProfileErasedError` from `save_profile` for a tombstoned user)
- Modify: `services/personality_service/src/api.py` (`/detect` and `/profile/update` map `ProfileErasedError` to `HTTPException(410, detail="User data has been erased")`)
- Modify: `services/personality_service/src/main.py` (`def register_personality_erasure(plane, orchestrator: PersonalityOrchestrator) -> None` calls `erasure_consumer.register_user_erasure_consumer(plane, service_name="personality-service", erasure=orchestrator.build_erasure(), purge_caches=orchestrator.purge_user_state)` with `from solace_infrastructure.gdpr import consumer as erasure_consumer`; the lifespan calls it when `runtime.event_plane is not None`, and strict with no plane raises `StartupDependencyError`)
- Create: `services/personality_service/tests/test_mvp_172_erasure_tombstone.py`
- Create: `tests/realinfra/personality-analytics/test_mvp_172_personality_erasure.py`
- Test: both new files; `test_gdpr_erasure.py`, `test_rls_guc.py` and `test_service.py` stay green unchanged

**Interfaces:**
- Consumes: IF-16 `register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches=None)` from `src/solace_infrastructure/gdpr/consumer.py` (WS-14.13, W3 phase A); IF-05 `runtime.event_plane`; slot-021 tombstone table (WS-18.3); `acting_as_user`; `UserDataErasure.register/erase/store_names`.
- Produces: `ProfileErasedError`; HTTP 410 `{"detail": "User data has been erased"}` (IF-11: a 4xx, never retried; the orchestrator reports degraded `personality_unavailable`); `PersonalityOrchestrator.build_erasure()`, `.purge_user_state(user_id)`; `ProfileStore.user_lock(user_id)`; `main.register_personality_erasure(plane, orchestrator)`; erasure store name `personality_postgres` (unchanged).

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_172_erasure_tombstone.py`
```python
"""MVP-172: an erased personality profile cannot come back (WS-18.12)."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from services.personality_service.src import main as personality_main
from services.personality_service.src.domain.errors import ProfileErasedError
from services.personality_service.src.domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from services.personality_service.src.domain.trait_detector import TraitDetector, TraitDetectorSettings
from services.personality_service.src.infrastructure.postgres_repository import PostgresPersonalityRepository
from services.personality_service.src.domain.entities import PersonalityProfile as EntityProfile
from services.personality_service.src.domain.value_objects import OceanScores
from services.personality_service.src.schemas import DetectPersonalityRequest
from services.personality_service.tests.fixtures import InMemoryPersonalityRepository

LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")


class _SlowDetector(TraitDetector):
    async def assess(self, text, sources=None):
        await asyncio.sleep(0.2)
        return await super().assess(text, sources)


def _orch(repo, detector=None) -> PersonalityOrchestrator:
    return PersonalityOrchestrator(settings=PersonalityServiceSettings(enable_llm_detection=False),
                                   trait_detector=detector, llm_client=None, repository=repo)


async def test_mvp_172_in_flight_and_later_detects_cannot_recreate_the_profile() -> None:
    repo = InMemoryPersonalityRepository()
    orch = _orch(repo, _SlowDetector(TraitDetectorSettings(enable_llm_detection=False)))
    user_id = uuid4()
    await orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    in_flight = asyncio.create_task(orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT)))
    await asyncio.sleep(0.05)
    await orch.delete_user_data(user_id)
    with pytest.raises(ProfileErasedError):
        await in_flight
    with pytest.raises(ProfileErasedError):
        await orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    assert orch._profile_store.get(user_id) is None
    assert await repo.get_profile_by_user(user_id) is None


async def test_mvp_172_tombstone_survives_a_restart() -> None:
    repo = InMemoryPersonalityRepository()
    user_id = uuid4()
    await _orch(repo).detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    await _orch(repo).delete_user_data(user_id)
    with pytest.raises(ProfileErasedError):
        await _orch(repo).detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    assert await repo.get_profile_by_user(user_id) is None


async def test_mvp_172_guarded_upsert_reports_a_tombstoned_user(recording_cpm, fake_codec) -> None:
    recording_cpm.connection.status = "INSERT 0 0"
    with pytest.raises(ProfileErasedError):
        await PostgresPersonalityRepository(codec=fake_codec).save_profile(
            EntityProfile(user_id=uuid4(), ocean_scores=OceanScores.neutral()))
    assert "personality_erasure_tombstones" in recording_cpm.connection.executed[0][0]


def test_mvp_172_lifespan_wiring_registers_the_erasure_consumer(monkeypatch) -> None:
    captured = {}

    def _capture(plane, *, service_name, erasure, purge_caches=None):
        captured.update(plane=plane, service_name=service_name, stores=erasure.store_names, purge=purge_caches)

    monkeypatch.setattr("solace_infrastructure.gdpr.consumer.register_user_erasure_consumer", _capture)
    orch = _orch(InMemoryPersonalityRepository())
    plane = object()
    personality_main.register_personality_erasure(plane, orch)
    assert captured == {"plane": plane, "service_name": "personality-service",
                        "stores": ("personality_postgres",), "purge": orch.purge_user_state}
```
and in `tests/realinfra/personality-analytics/test_mvp_172_personality_erasure.py`
```python
"""MVP-172 on the migrated schema: erasure deletes, tombstones and blocks re-creation (WS-18.12)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.personality_service.src import main as personality_main
from services.personality_service.src.domain.errors import ProfileErasedError
from services.personality_service.src.schemas import DetectPersonalityRequest
from solace_common.request_context import acting_as_user

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")
COUNT = "SELECT count(*) FROM {table} WHERE user_id = $1"


def _orchestrator(phi_codec):
    runtime = SimpleNamespace(pg_pool=object(), phi_codec=phi_codec, redis=None, event_plane=None)
    return personality_main.build_personality_orchestrator(runtime, personality_main.SETTINGS, llm_client=None)


async def test_mvp_172_erasure_is_final_across_a_restart(cpm_default, seed_users, phi_codec, superuser_conn) -> None:
    target, other = await seed_users(2)
    orchestrator = _orchestrator(phi_codec)
    await orchestrator.initialize()
    for user_id in (target, other):
        with acting_as_user(str(user_id)):
            await orchestrator.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    with acting_as_user(str(target)):
        report = await orchestrator.build_erasure().erase(str(target))
    assert report.complete is True and report.total_deleted == 1
    assert await superuser_conn.fetchval(COUNT.format(table="personality_profiles"), target) == 0
    assert await superuser_conn.fetchval(COUNT.format(table="personality_erasure_tombstones"), target) == 1
    assert await superuser_conn.fetchval(COUNT.format(table="personality_profiles"), other) == 1
    restarted = _orchestrator(phi_codec)
    await restarted.initialize()
    with acting_as_user(str(target)):
        with pytest.raises(ProfileErasedError):
            await restarted.detect_personality(DetectPersonalityRequest(user_id=target, text=LONG_TEXT))
    assert await superuser_conn.fetchval(COUNT.format(table="personality_profiles"), target) == 0
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_172_erasure_tombstone.py -o "addopts=" -p no:logging -q`, then `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/personality-analytics/test_mvp_172_personality_erasure.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'ProfileErasedError'`. With it stubbed: the in-flight detect completes after the erase and re-creates the profile in the cache and the repository (MVP-172); a later or restarted detect creates a fresh profile; the guarded upsert does not mention the tombstone table; `register_personality_erasure` does not exist; on real Postgres no tombstone row is written and the restarted orchestrator re-inserts the profile.
- [ ] **Step 3: Implement** the listed changes. Detection stays outside the per-user lock so an erase never waits for an LLM call; only the read-modify-write, persist and cache steps run under it. `_persist_profile` re-raises `ProfileErasedError` unwrapped (it is not a store failure, so it never becomes `ProfilePersistenceError`). Erasure logs store names and counts only.
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run, one process each with the Step 2 prefixes: `services/personality_service/tests`; `tests/realinfra/personality-analytics` (with `$env:SOLACE_REQUIRE_REALINFRA="1"`); `tests/contracts` (the WS-05.14 erasure contracts list personality-service). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.13: Analytics logs carry no event payloads, identifier values or exception text
**Covers:** MVP-109
**Severity:** medium
**Wave:** 3
**Depends on:** WS-18.11
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-18 (default assumed: analytics holds and logs no PHI)
**Real infra:** no
**Files:**
- Modify: `services/analytics-service/src/consumer.py` (`event_processing_failed` logs `event_type`, `event_id` and `error_type` only, never `event_data` or `str(e)`; `invalid_uuid_in_event` logs the field name only; `consume_loop_error` and `batch_event_failed` log `error_type` only; `handle_event` adds no payload to any log line)
- Create: `services/analytics-service/tests/test_mvp_109_no_payload_logging.py`
- Test: the new file

**Interfaces:**
- Consumes: none.
- Produces: the analytics log contract: failure lines carry `event_type`, `event_id`, `error_type` (and counts) only.

- [ ] **Step 1: Write the failing test** in `services/analytics-service/tests/test_mvp_109_no_payload_logging.py`
```python
"""MVP-109: analytics never logs payload values (WS-18.13)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import consumer as consumer_module  # noqa: E402
from consumer import AnalyticsEvent  # noqa: E402

PHRASE = "synthetic disclosure phrase alpha"
CONDITION = "synthetic condition beta"


class _RecordingLogger:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, dict]] = []

    def _add(self, level, event, **kwargs):
        self.entries.append((level, event, kwargs))

    def debug(self, event, **kwargs): self._add("debug", event, **kwargs)
    def info(self, event, **kwargs): self._add("info", event, **kwargs)
    def warning(self, event, **kwargs): self._add("warning", event, **kwargs)
    def error(self, event, **kwargs): self._add("error", event, **kwargs)


def _crisis() -> dict:
    return {"event_type": "safety.crisis.detected", "user_id": str(uuid4()),
            "metadata": {"event_id": str(uuid4()), "timestamp": datetime.now(timezone.utc).isoformat()},
            "crisis_level": "CRITICAL", "trigger_indicators": [PHRASE], "detection_layer": 1,
            "primary_hypothesis": {"condition_name": CONDITION}}


async def test_mvp_109_processing_failure_logs_no_payload(monkeypatch, analytics_consumer, analytics_aggregator) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(consumer_module, "logger", recorder)

    async def _boom(*args, **kwargs):
        raise RuntimeError(f"failed while handling {PHRASE}")

    monkeypatch.setattr(analytics_aggregator, "track_crisis_event", _boom)
    assert await analytics_consumer.process_event(_crisis()) is False
    text = repr(recorder.entries)
    assert PHRASE not in text and CONDITION not in text
    failure = next(kw for _, event, kw in recorder.entries if event == "event_processing_failed")
    assert set(failure) <= {"event_type", "event_id", "error_type"} and failure["error_type"] == "RuntimeError"


def test_mvp_109_invalid_identifier_warning_omits_the_value(monkeypatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(consumer_module, "logger", recorder)
    AnalyticsEvent.from_raw({"event_type": "session.started", "user_id": "jane.doe@example.com",
                             "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()}})
    assert "jane.doe" not in repr(recorder.entries)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests/test_mvp_109_no_payload_logging.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `event_processing_failed` carries `event_data` (with the trigger phrase and the condition name) and `error=str(e)` (MVP-109); the UUID warning logs `value="jane.doe@example.com"`.
- [ ] **Step 3: Implement** the listed logging changes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.14: Analytics read APIs are staff-only
**Covers:** MVP-234
**Severity:** low
**Wave:** 3
**Depends on:** WS-18.13, WS-07.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-18 (default assumed: staff means `admin` and `superadmin`; clinicians and users, including obo principals with `roles=["user"]`, are denied), D-21 (admins read no PHI; these counters hold none)
**Real infra:** no
**Files:**
- Modify: `services/analytics-service/src/api.py` (`ANALYTICS_READ_ROLES = ("admin", "superadmin")`; `/dashboard`, `/metrics/query`, `/metrics/names`, `/reports/types`, `/reports/generate`, `/reports/{report_type}`, `/consumer/stats` and `/stats` depend on `require_roles(*ANALYTICS_READ_ROLES)`; delete the `except ImportError` stub auth, whose `require_roles` returned `get_current_user` and so checked no role)
- Create: `services/analytics-service/tests/test_mvp_234_staff_only_reads.py`
- Test: the new file; `test_api.py` (its override has `roles=["user", "admin"]`) stays green unchanged

**Interfaces:**
- Consumes: `solace_security.middleware.require_roles` (stable name).
- Produces: `ANALYTICS_READ_ROLES`; 403 for non-staff principals on every read route.

- [ ] **Step 1: Write the failing test** in `services/analytics-service/tests/test_mvp_234_staff_only_reads.py`
```python
"""MVP-234: analytics read endpoints require a staff role (WS-18.14)."""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from api import router, set_dependencies  # noqa: E402
from solace_security.auth import TokenType  # noqa: E402
from solace_security.middleware import AuthenticatedUser, get_current_user  # noqa: E402

READS = [
    ("GET", "/api/v1/analytics/dashboard", None),
    ("POST", "/api/v1/analytics/metrics/query", {"metric_name": "safety.assessments"}),
    ("GET", "/api/v1/analytics/metrics/names", None),
    ("GET", "/api/v1/analytics/reports/types", None),
    ("POST", "/api/v1/analytics/reports/generate?report_type=safety_overview", None),
    ("GET", "/api/v1/analytics/reports/safety_overview", None),
    ("GET", "/api/v1/analytics/consumer/stats", None),
    ("GET", "/api/v1/analytics/stats", None),
]


def _client(roles, aggregator, reports, consumer) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    set_dependencies(aggregator, reports, consumer)
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=str(uuid4()), token_type=TokenType.ACCESS, roles=roles)
    return TestClient(app)


@pytest.mark.parametrize("method,path,body", READS)
@pytest.mark.parametrize("roles,expected", [(["user"], 403), (["clinician"], 403), (["admin"], 200), (["superadmin"], 200)])
def test_mvp_234_reads_are_staff_only(method, path, body, roles, expected, analytics_aggregator, report_service,
                                      analytics_consumer) -> None:
    client = _client(roles, analytics_aggregator, report_service, analytics_consumer)
    response = client.request(method, path, json=body)
    assert response.status_code == expected
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests/test_mvp_234_staff_only_reads.py -o "addopts=" -p no:logging -q`
Expected: FAIL in the 16 `user` and `clinician` cases: each returns 200 because the routes depend on `get_current_user` alone (MVP-234). The staff cases pass (guards).
- [ ] **Step 3: Implement** the listed changes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run, one process each with the Step 2 prefix: `services/analytics-service/tests`; `tests/security` (WS-07.9 acceptance if present). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.15: No fabricated evidence: the untrained RoBERTa stub and the LIWC placeholder stop counting, and no-evidence detections change no profile
**Covers:** MVP-300, MVP-305
**Severity:** low
**Wave:** 5
**Depends on:** WS-18.12
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none (personality style is not clinical scoring; D-30 allows style adaptation)
**Real infra:** no
**Files:**
- Modify: `services/personality_service/src/ml/roberta_model.py` (`RoBERTaPersonalityDetector.is_trained -> bool`: model and tokenizer present and classification-head weights loaded)
- Modify: `services/personality_service/src/domain/trait_detector.py` (`TraitDetectionResult.has_evidence: bool = True`; `TextBasedDetector.detect` marks the LIWC neutral placeholder (no `trait_scores`) and its exception fallback `has_evidence=False`; RoBERTa runs only when `settings.ensemble_weights_roberta > 0` and `roberta.is_trained`; only results with evidence and positive weight enter the scores, the evidence markers and `overall_confidence`; with none, `_neutral_scores()` and `contributing_sources == ()`)
- Modify: `services/personality_service/src/domain/service.py` (`detect_personality` returns the neutral response without touching or persisting the profile when `outcome.contributing_sources` is empty)
- Modify: `services/personality_service/tests/test_trait_detector.py` (`TestTraitDetector::test_detect_with_sources` uses `LONG_TEXT`)
- Modify: `services/personality_service/tests/test_service.py` (`test_detect_personality_creates_profile`, `test_detect_personality_updates_profile`, `test_get_style` and `test_statistics_update` use `LONG_TEXT`; their short texts carried no evidence)
- Create: `services/personality_service/tests/test_mvp_300_no_fabricated_evidence.py`
- Test: the new file and the two modified files

**Interfaces:**
- Produces: `RoBERTaPersonalityDetector.is_trained`; `TraitDetectionResult.has_evidence`; no-evidence detections leave the profile unchanged (the response is neutral at confidence 0.3, as for short texts today).

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_300_no_fabricated_evidence.py`
```python
"""MVP-300 / MVP-305: only real evidence shapes scores, confidence and profiles (WS-18.15)."""
from __future__ import annotations

from uuid import uuid4

import pytest

from services.personality_service.src.domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from services.personality_service.src.domain.trait_detector import TraitDetector, TraitDetectorSettings
from services.personality_service.src.ml.liwc_features import LIWCProcessor
from services.personality_service.src.schemas import DetectPersonalityRequest

LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")
SHORT_TEXT = "I worry all the time and cannot stop thinking about it"
REPLY = ('{"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, '
         '"neuroticism": 0.95, "evidence": ["worry_language"]}')


class _FakeLLM:
    is_available = True

    async def generate(self, **kwargs) -> str:
        return REPLY


async def _detector() -> TraitDetector:
    detector = TraitDetector(TraitDetectorSettings(), llm_client=_FakeLLM())
    await detector.initialize()
    return detector


async def test_mvp_305_liwc_placeholder_does_not_dilute_the_llm() -> None:
    assert len(SHORT_TEXT) >= 50 and len(SHORT_TEXT.split()) < 20
    scores = await (await _detector()).detect(SHORT_TEXT)
    assert scores.neuroticism == pytest.approx(0.95)


async def test_mvp_300_untrained_roberta_adds_no_confidence_drag_or_evidence() -> None:
    scores = await (await _detector()).detect(LONG_TEXT)
    liwc = await LIWCProcessor().process(LONG_TEXT)
    markers = [m for ts in scores.trait_scores for m in ts.evidence_markers]
    assert not any(m.startswith("roberta_") for m in markers)
    assert scores.overall_confidence == pytest.approx((liwc.overall_confidence + 0.65) / 2)


async def test_mvp_305_no_evidence_leaves_the_profile_untouched() -> None:
    orch = PersonalityOrchestrator(settings=PersonalityServiceSettings(enable_llm_detection=False), llm_client=None)
    user_id = uuid4()
    response = await orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=SHORT_TEXT))
    assert response.confidence == pytest.approx(0.3)
    assert orch._profile_store.get(user_id) is None
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_300_no_fabricated_evidence.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Neuroticism is about 0.844 because the LIWC placeholder (0.5 at confidence 0.3) takes about 23% of the weight (MVP-305); the trait evidence contains `roberta_openness` etc. and `overall_confidence` averages in the stub's 0.25 (MVP-300); the no-evidence detection creates and caches a neutral profile.
- [ ] **Step 3: Implement** the listed changes. Never lower detection confidence below what real evidence supports and never invent evidence: a source with no evidence simply does not contribute.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_roberta_model.py` exercises the detector directly and is unaffected; `test_detect_short_text_returns_neutral` still sees confidence 0.3).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.16: The profile cache is bounded and expires; aggregates keep per-trait scores and their real update time
**Covers:** MVP-301, MVP-307
**Severity:** low
**Wave:** 5
**Depends on:** WS-18.15
**Implementer model:** opus
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/personality_service/src/domain/service.py` (`ProfileStore(max_profiles: int = 2000, ttl_seconds: float | None = None, clock: Callable[[], float] = time.monotonic)`: LRU bound on profiles; `get` returns `None` for an entry older than the TTL; eviction and `delete` drop the user's lock when it is not held; `save` no longer sets `updated_at`; `PersonalityServiceSettings.max_cached_profiles: int = Field(default=2000, ge=1)` (`PERSONALITY_SERVICE_MAX_CACHED_PROFILES`); the orchestrator builds `ProfileStore(max_profiles=settings.max_cached_profiles, ttl_seconds=settings.profile_cache_ttl_seconds if repository else None)`; `_update_profile_with_assessment` sets `updated_at`; `_aggregate_scores` keeps one `TraitScoreDTO` per trait (EMA value, bounds from the aggregate confidence, `sample_count = assessment_count`, up to three evidence markers) and `assessed_at = new.assessed_at`)
- Create: `services/personality_service/tests/test_mvp_301_profile_cache.py`
- Test: the new file; `test_service.py::TestProfileStore` stays green unchanged

**Interfaces:**
- Consumes: `ProfileStore.user_lock` (WS-18.12).
- Produces: `ProfileStore(max_profiles, ttl_seconds, clock)`, `PersonalityServiceSettings.max_cached_profiles` (registry entry requested from WS-03).

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_301_profile_cache.py`
```python
"""MVP-301 / MVP-307: bounded, expiring cache and honest aggregates (WS-18.16)."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from services.personality_service.src.domain.entities import PersonalityProfile as EntityProfile
from services.personality_service.src.domain.service import (
    PersonalityOrchestrator, PersonalityProfile, PersonalityServiceSettings, ProfileStore,
)
from services.personality_service.src.domain.value_objects import OceanScores
from services.personality_service.src.schemas import DetectPersonalityRequest
from services.personality_service.tests.fixtures import InMemoryPersonalityRepository

LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")


async def test_mvp_301_cache_is_bounded_and_prunes_locks() -> None:
    store = ProfileStore(max_profiles=3)
    for _ in range(5):
        user_id = uuid4()
        async with store.user_lock(user_id):
            await store.save(PersonalityProfile(user_id=user_id))
    assert store.count() == 3 and len(store._locks) == 3


async def test_mvp_301_entries_expire_after_the_ttl() -> None:
    now = [1000.0]
    store = ProfileStore(max_profiles=10, ttl_seconds=60, clock=lambda: now[0])
    profile = PersonalityProfile(user_id=uuid4())
    await store.save(profile)
    now[0] += 61
    assert store.get(profile.user_id) is None


def test_mvp_301_settings_drive_the_cache() -> None:
    orch = PersonalityOrchestrator(settings=PersonalityServiceSettings(max_cached_profiles=50, profile_cache_ttl_seconds=120),
                                   llm_client=None, repository=InMemoryPersonalityRepository())
    assert (orch._profile_store._max_profiles, orch._profile_store._ttl_seconds) == (50, 120)


async def test_mvp_307_aggregate_keeps_per_trait_scores() -> None:
    orch = PersonalityOrchestrator(settings=PersonalityServiceSettings(enable_llm_detection=False), llm_client=None)
    user_id = uuid4()
    for _ in range(2):
        await orch.detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    trait_scores = orch._profile_store.get(user_id).ocean_scores.trait_scores
    assert len(trait_scores) == 5 and {t.sample_count for t in trait_scores} == {2}


async def test_mvp_307_loading_from_the_store_keeps_updated_at() -> None:
    repo = InMemoryPersonalityRepository()
    user_id = uuid4()
    await repo.save_profile(EntityProfile(user_id=user_id, ocean_scores=OceanScores.neutral(), assessment_count=1))
    stored = await repo.get_profile_by_user(user_id)
    stored.updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    orch = PersonalityOrchestrator(settings=PersonalityServiceSettings(enable_llm_detection=False),
                                   llm_client=None, repository=repo)
    assert (await orch.get_profile(user_id)).last_updated == datetime(2026, 1, 1, tzinfo=timezone.utc)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_301_profile_cache.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `ProfileStore(max_profiles=...)` raises `TypeError` (the store is an unbounded dict, the TTL settings are dead and locks are never pruned, MVP-301); `max_cached_profiles` is not a setting; the aggregate after the second detection has `trait_scores == []`; `last_updated` is the load time because `ProfileStore.save` overwrites `updated_at` (MVP-307).
- [ ] **Step 3: Implement** the listed changes; an `OrderedDict` gives the LRU order and the clock is injected so tests never patch `time.monotonic` (asyncio uses it).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.17: Personality publishes profile-updated events with ids and counts only
**Covers:** MVP-302
**Severity:** low
**Wave:** 5
**Depends on:** WS-18.16, WS-06.9
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-18 (default assumed: analytics counts personality activity; events carry no trait values)
**Real infra:** no
**Files:**
- Modify: `services/personality_service/src/domain/service.py` (`PersonalityOrchestrator(..., event_bus: EventBus | None = None)`; after a successful persist and cache update in `detect_personality` and `update_profile_from_scores`, publish `ProfileUpdatedEvent.create(profile_id, user_id, assessment_count=..., version=...)`; nothing is published for a failed, erased or no-evidence detection; trait values are never put on the bus)
- Modify: `services/personality_service/src/main.py` (`build_personality_orchestrator` receives the lifespan's `EventBus` and passes it as `event_bus`)
- Create: `services/personality_service/tests/test_mvp_302_profile_events.py`
- Test: the new file

**Interfaces:**
- Consumes: `events.EventBus`, `ProfileUpdatedEvent`, `to_kafka_event` (maps to `PersonalityProfileUpdatedEvent`); the bridge's `subscribe_all` (WS-06.9).
- Produces: `build_personality_orchestrator(runtime, settings, *, llm_client, event_bus=None)`; `personality.profile.updated` events on `solace.personality` (analytics counts them as `events.personality.profile.updated`).

- [ ] **Step 1: Write the failing test** in `services/personality_service/tests/test_mvp_302_profile_events.py`
```python
"""MVP-302: profile updates reach the bus, with ids and counts only (WS-18.17)."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
import solace_events.schemas as canonical

from services.personality_service.src import main as personality_main
from services.personality_service.src.domain.errors import ProfilePersistenceError
from services.personality_service.src.domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from services.personality_service.src.events import EventBus, EventType, to_kafka_event
from services.personality_service.src.schemas import DetectPersonalityRequest
from services.personality_service.tests.fixtures import InMemoryPersonalityRepository

LONG_TEXT = ("I keep a careful plan for every week and I like to finish my work early so that I can "
             "relax with friends and talk about the new ideas we discovered together")


class _FailingRepo(InMemoryPersonalityRepository):
    async def save_profile(self, profile) -> None:
        raise RuntimeError("DB connection lost")


def _orch(bus, repo=None) -> PersonalityOrchestrator:
    return PersonalityOrchestrator(settings=PersonalityServiceSettings(enable_llm_detection=False),
                                   llm_client=None, repository=repo, event_bus=bus)


async def test_mvp_302_detect_publishes_profile_updated_without_scores() -> None:
    bus = EventBus()
    user_id = uuid4()
    await _orch(bus).detect_personality(DetectPersonalityRequest(user_id=user_id, text=LONG_TEXT))
    events = bus.get_events_by_type(EventType.PROFILE_UPDATED)
    assert len(events) == 1 and events[0].user_id == user_id
    assert set(events[0].payload) == {"assessment_count", "version"}
    kafka = to_kafka_event(events[0])
    assert isinstance(kafka, canonical.PersonalityProfileUpdatedEvent) and kafka.assessment_count == 1


async def test_mvp_302_failed_persist_publishes_nothing() -> None:
    bus = EventBus()
    with pytest.raises(ProfilePersistenceError):
        await _orch(bus, _FailingRepo()).detect_personality(DetectPersonalityRequest(user_id=uuid4(), text=LONG_TEXT))
    assert bus.get_published_events() == []


def test_mvp_302_composition_hands_the_bus_to_the_orchestrator() -> None:
    bus = EventBus()
    runtime = SimpleNamespace(pg_pool=None, phi_codec=None, redis=None, event_plane=None)
    orch = personality_main.build_personality_orchestrator(runtime, personality_main.SETTINGS, llm_client=None,
                                                           event_bus=bus)
    assert orch._event_bus is bus
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests/test_mvp_302_profile_events.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `PersonalityOrchestrator(..., event_bus=...)` raises `TypeError: unexpected keyword argument 'event_bus'`: nothing ever publishes, so analytics' personality subscriptions receive no data (MVP-302).
- [ ] **Step 3: Implement** the listed changes; publish after the cache update so a failed persist publishes nothing.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run, one process each with the Step 2 prefix: `services/personality_service/tests` (including WS-06's `test_mvp_014_event_bridge_plane.py`); `tests/integration/test_event_schemas.py`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.18: Analytics keeps a full retention window, says what period it covers, and stops cleanly
**Covers:** MVP-111, MVP-232, MVP-233
**Severity:** medium
**Wave:** 5
**Depends on:** WS-18.14
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-18 (default assumed: counters stay in memory, ClickHouse stays out of scope, and reports state their coverage instead of persisting)
**Real infra:** no
**Files:**
- Modify: `services/analytics-service/src/aggregations.py` (`MetricsStore(max_windows_per_metric: int = 1000, retention_hours: int = 168, minute_resolution_hours: int = 2)`: records older than `retention_hours` are dropped and counted in `records_dropped`; minute buckets older than `minute_resolution_hours` are folded into hour buckets with the same labels, and older records go straight to hour buckets, so each value lives in exactly one bucket; the window cap applies per label set; `MetricBucket` keeps no per-value list; new `started_at: datetime`; `AnalyticsAggregator.get_dashboard_metrics` returns `coverage_start`)
- Modify: `services/analytics-service/src/main.py` (`_create_services` passes `retention_hours=settings.metrics.default_retention_hours`)
- Modify: `services/analytics-service/src/reports.py` (`ReportService.generate_report` adds `coverage_start` (ISO) and `coverage_complete` (`time_range.start >= store.started_at`) to every report summary)
- Modify: `services/analytics-service/src/api.py` (`DashboardResponse.coverage_start: str`)
- Modify: `services/analytics-service/src/consumer.py` (`stop()` cancels and awaits `_consume_task`; `consume_loop` counts a malformed queued event with `record_failure`)
- Create: `services/analytics-service/tests/test_mvp_111_retention_and_coverage.py`
- Test: the new file

**Interfaces:**
- Consumes: the `timestamp=` recording API (WS-18.11).
- Produces: `MetricsStore(max_windows_per_metric, retention_hours, minute_resolution_hours)`, `MetricsStore.started_at`, `MetricsStore.retention_hours`; report summary keys `coverage_start`, `coverage_complete`; dashboard `coverage_start`. MINUTE-window queries cover the last `minute_resolution_hours` only.

- [ ] **Step 1: Write the failing test** in `services/analytics-service/tests/test_mvp_111_retention_and_coverage.py`
```python
"""MVP-111 / MVP-232 / MVP-233: full retention window, honest coverage, clean stop (WS-18.18)."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import main as analytics_main  # noqa: E402
from aggregations import AggregationWindow, MetricsStore  # noqa: E402
from reports import ReportPeriod, ReportTimeRange, ReportType  # noqa: E402

LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")


async def test_mvp_111_a_day_of_minute_events_is_fully_counted() -> None:
    store, now = MetricsStore(), datetime.now(timezone.utc)
    for minute in range(1440):
        labels = {"risk_level": LEVELS[minute % 4], "detection_layer": str(1 + (minute // 4) % 2)}
        await store.record("safety.crisis_events", Decimal(1), timestamp=now - timedelta(minutes=minute), labels=labels)
    day = await store.get_aggregated("safety.crisis_events", window_type=AggregationWindow.HOUR,
                                     start_time=now - timedelta(hours=24, minutes=1), end_time=now + timedelta(minutes=1))
    assert sum(m.count for m in day) == 1440
    assert sum(m.count for m in day if m.labels["risk_level"] == "CRITICAL") == 360


async def test_mvp_111_retention_drops_records_older_than_configured() -> None:
    store, now = MetricsStore(retention_hours=168), datetime.now(timezone.utc)
    await store.record("m", Decimal(1), timestamp=now - timedelta(hours=169))
    assert await store.get_aggregated("m", window_type=AggregationWindow.DAY,
                                      start_time=now - timedelta(days=30), end_time=now) == []
    assert (await store.get_statistics())["records_dropped"] == 1


def test_mvp_111_main_passes_the_retention_setting() -> None:
    settings = analytics_main.SETTINGS.model_copy(deep=True)
    settings.metrics.default_retention_hours = 24
    aggregator, _, _ = analytics_main._create_services(settings)
    assert aggregator.store.retention_hours == 24


async def test_mvp_232_reports_state_their_in_memory_coverage(report_service, analytics_aggregator) -> None:
    week = await report_service.generate_report(ReportType.SAFETY_OVERVIEW, ReportTimeRange.for_last_week(), use_cache=False)
    assert week.summary["coverage_start"] == analytics_aggregator.store.started_at.isoformat()
    assert week.summary["coverage_complete"] is False
    since_start = ReportTimeRange(start=analytics_aggregator.store.started_at, end=datetime.now(timezone.utc),
                                  period=ReportPeriod.CUSTOM)
    covered = await report_service.generate_report(ReportType.SAFETY_OVERVIEW, since_start, use_cache=False)
    assert covered.summary["coverage_complete"] is True


async def test_mvp_233_stop_cancels_the_loop_and_malformed_events_are_counted(analytics_consumer) -> None:
    await analytics_consumer.start()
    await analytics_consumer.enqueue_event({"event_type": "session.started", "metadata": {"timestamp": "not-a-time"}})
    for _ in range(50):
        if analytics_consumer.metrics.events_failed:
            break
        await asyncio.sleep(0.02)
    await analytics_consumer.stop()
    assert analytics_consumer.metrics.events_failed == 1
    assert analytics_consumer._consume_task.done()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests/test_mvp_111_retention_and_coverage.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The day count is 1000, not 1440 (the cap keeps the newest 1000 minute buckets across all label sets, MVP-111); `MetricsStore(retention_hours=...)` raises `TypeError` and `default_retention_hours` is never used; summaries have no `coverage_start` (MVP-232); the malformed queued event is logged as `consume_loop_error` but not counted and `_consume_task.done()` is False after `stop()` (MVP-233).
- [ ] **Step 3: Implement** the listed changes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_main.py::test_create_services_respects_config` still sees `_max_windows == 500`).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-18.19: Analytics reports count what they claim to count
**Covers:** MVP-238, MVP-239, MVP-240
**Severity:** low
**Wave:** 5
**Depends on:** WS-18.18
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-18
**Real infra:** no
**Files:**
- Modify: `services/analytics-service/src/reports.py` (compliance `safety_check_coverage` is `"not_assessed"` (analytics cannot see unscreened messages), with raw `safety_assessments` and `messages_received` counts beside it; the engagement report reads `events.session.message.received`; clinical outcomes report `total_therapy_sessions_started` separately from interventions)
- Modify: `services/analytics-service/src/consumer.py` (`therapy.session.started` goes to `track_therapy_session_started`; only `therapy.intervention.delivered` goes to `track_therapy_event`; `safety.escalation.triggered` passes `crisis_level` only when the payload has one)
- Modify: `services/analytics-service/src/aggregations.py` (new `track_therapy_session_started(timestamp=None)` counting `therapy.sessions_started`; `track_escalation(priority="unknown", crisis_level: str | None = None, timestamp=None)` labels `risk_level` only when `crisis_level` is given)
- Create: `services/analytics-service/tests/test_mvp_238_report_correctness.py`
- Test: the new file

**Interfaces:**
- Produces: metric `therapy.sessions_started`; escalation labels `{"priority": ...}` (plus `risk_level` only when known); compliance `safety_check_coverage = "not_assessed"`.

- [ ] **Step 1: Write the failing test** in `services/analytics-service/tests/test_mvp_238_report_correctness.py`
```python
"""MVP-238 / MVP-239 / MVP-240: reports count what they claim (WS-18.19)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from reports import ReportTimeRange, ReportType  # noqa: E402


def _event(event_type: str, **fields) -> dict:
    return {"event_type": event_type, "user_id": str(uuid4()), "session_id": str(uuid4()),
            "metadata": {"event_id": str(uuid4()), "timestamp": datetime.now(timezone.utc).isoformat()}, **fields}


async def test_mvp_238_compliance_coverage_is_not_assessed(analytics_consumer, report_service) -> None:
    for _ in range(5):
        await analytics_consumer.process_event(_event("safety.assessment.completed", risk_level="LOW", detection_layer=1))
    report = await report_service.generate_report(ReportType.COMPLIANCE_AUDIT, ReportTimeRange.for_last_day(), use_cache=False)
    section = next(s for s in report.sections if s.title == "Regulatory Compliance Status")
    assert section.data["safety_check_coverage"] == "not_assessed"


async def test_mvp_239_engagement_counts_received_messages(analytics_consumer, report_service) -> None:
    for _ in range(3):
        await analytics_consumer.process_event(_event("session.message.received", content_length=10))
    report = await report_service.generate_report(ReportType.ENGAGEMENT_METRICS, ReportTimeRange.for_last_day(), use_cache=False)
    assert report.summary["total_messages"] == 3


async def test_mvp_240_only_interventions_count_as_interventions(analytics_consumer, analytics_aggregator) -> None:
    await analytics_consumer.process_event(_event("therapy.session.started"))
    await analytics_consumer.process_event(_event("therapy.intervention.delivered", modality="CBT",
                                                  technique="cognitive_restructuring"))
    await analytics_consumer.process_event(_event("safety.escalation.triggered", priority="HIGH",
                                                  escalation_reason="synthetic"))
    buckets = analytics_aggregator.store._buckets
    assert [b.labels for b in buckets["therapy.interventions"].values()] == [
        {"modality": "CBT", "technique": "cognitive_restructuring"}]
    assert "therapy.sessions_started" in buckets
    assert [b.labels for b in buckets["safety.escalations"].values()] == [{"priority": "HIGH"}]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests/test_mvp_238_report_correctness.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Coverage reads `"100%"` from checks/crises (MVP-238); `total_messages == 0` because the report queries `events.message.sent`, which nothing writes (MVP-239); the session start is counted as an `unknown` intervention and the escalation is labelled `risk_level=NONE` (MVP-240).
- [ ] **Step 3: Implement** the listed changes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/analytics-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_reports_render_data.py` passes `crisis_level="CRITICAL"` explicitly, so its escalation keeps the label; `test_compliance_honesty.py` checks only the retention and audit keys).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-18.19:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| Id | Owner | Change | Files | Needed by |
|---|---|---|---|---|
| R-18-01 | WS-03 | W0 phase A env sync (already in WS-03's plan; confirm): personality-service and analytics-service get POSTGRES_* as solace_app, `REDIS_URL`, `KAFKA_BOOTSTRAP_SERVERS`, `EVENTS_ENABLED=true`, `PERSONALITY_ENVIRONMENT`/`ANALYTICS_SERVICE_ENV = ${ENVIRONMENT}`, healthchecks on `/live`; analytics now declares `uses_postgres`, so it also needs the migrate dependency | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml` | WS-18.1, WS-18.2 |
| R-18-02 | WS-03 | W1 phase A: mark `PERSONALITY_DB_*` and `PERSONALITY_REDIS_*` retired (personality reads neither after WS-18.1). W3 phase A: mark `CONSUMER_KAFKA_ENABLED` and `CONSUMER_KAFKA_BOOTSTRAP_SERVERS` retired (analytics uses the plane after WS-18.9). W5 phase A: registry entry `PERSONALITY_SERVICE_MAX_CACHED_PROFILES` (not secret, not required in strict, dev default 2000) | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `.env.example` (rendered) | WS-18.1, WS-18.9, WS-18.16 |
| R-18-03 | WS-04 | The conformance rows stay as in WS-04.5 (`personality-service`: PHI flags; `analytics-service`: `handles_phi=False`, `audited=False`); retire the `configure_phi_encryption` AST check in `tests/integration/test_phi_encryption_activation.py` (CR-8), which lists `personality_service` and documents the analytics exemption | `tests/bootstrap/test_service_conformance.py`, `tests/integration/test_phi_encryption_activation.py` | WS-18.1, WS-18.2 |
| R-18-04 | WS-05 | IF-19 `pg_migrated` (`dsn_app`, `dsn_superuser`, `app_pool` with the JSONB codec) and `kafka_bootstrap` (`bootstrap`, `produce(topic, BaseEvent, *, key=)`); IF-20 `verify_provider`; `tests/system/test_smoke.py` uses `/live` for personality | `src/solace_testing/realinfra.py`, `src/solace_testing/contracts.py` | WS-18.3, WS-18.4, WS-18.8, WS-18.9 |
| R-18-05 | WS-06 | `ServiceEventPlane.subscribe` accepted after `start()` (WS-06.8 already states it) and one consumer per `group_id` across several topics; `NonRetryableEventError` sends a message to the DLQ at once | `src/solace_events/service_plane.py`, `src/solace_events/errors.py` | WS-18.9 |
| R-18-06 | WS-07 | `ServicePermission.WRITE_ANALYTICS` held by no service and the orchestrator's `WRITE_PERSONALITY` grant (WS-07.3); `AuthenticatedService.on_behalf_of` and override-friendly `require_service_permission` (WS-07.4). W4: when `require_mfa_for_roles` lands (WS-07.11), list analytics' read routes among its adopters or record a ledger Ruling for a W5 follow-up in WS-18 | `src/solace_security/service_auth.py`, `src/solace_security/middleware.py` | WS-18.8, WS-18.10, WS-18.14 |
| R-18-07 | WS-09 | IF-13 `PHICodec` with `active_key_id` and `reset_phi_codec()` (WS-09.7); `LLMUnavailableError`, `is_available` and `readiness()` on `UnifiedLLMClient` (WS-09.3/09.4). The ADOPTERS path `services/personality_service/src/infrastructure/postgres_repository.py` is confirmed | `src/solace_security/phi_codec.py`, `services/shared/infrastructure/llm_client.py` | WS-18.4, WS-18.6 |
| R-18-08 | WS-11 | For awareness, no change required: `/style` now returns `style_parameters.custom_params.confidence` (the field WS-11.11's cache rule reads); `/detect` and `/profile/update` may return 503 (store unavailable) and 410 (user erased), both to be reported as degraded `personality_unavailable` without retry; the `style.json` contract may assert `custom_params.confidence` | `services/orchestrator_service/src/agents/personality_agent.py`, `tests/contracts/orchestrator__personality/style.json` | WS-18.8, WS-18.12 |
| R-18-09 | WS-14 | W3 phase A (planned as WS-14.13): IF-16 `register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches=None)` running the erasure under `acting_as_user`; `personality-service` in the expected-services list | `src/solace_infrastructure/gdpr/consumer.py` | WS-18.12 |
| R-18-10 | WS-02 | Review the slot-021 diff; 009's default privileges must give solace_app DML on `personality_erasure_tombstones` | `migrations/versions/009_platform_roles_rls.py` | WS-18.3 |

## New interfaces

- Personality: `main.SETTINGS`, `main.SPEC`, `build_personality_orchestrator(runtime, settings, *, llm_client, event_bus=None) -> PersonalityOrchestrator`, `register_personality_erasure(plane, orchestrator) -> None`; `PersonalityServiceAppSettings.environment` (`PERSONALITY_ENVIRONMENT`, then `ENVIRONMENT`) and `.events_enabled` (`EVENTS_ENABLED`).
- `services.personality_service.src.domain.errors`: `ProfilePersistenceError` (HTTP 503 `{"detail": "Personality store unavailable"}`), `ProfileErasedError` (HTTP 410 `{"detail": "User data has been erased"}`); `/detect` and `/adapt` 400 bodies carry fixed messages.
- `PostgresPersonalityRepository(*, codec: PHICodec, pool_name: str = "default")`, `PERSONALITY_PHI_FIELDS` with AAD `personality_profiles.<column>`; `RepositoryFactory.create_postgres(codec, *, pool_name="default")`.
- `TraitDetector.assess(text, sources=None) -> DetectionOutcome(scores, contributing_sources)`; `TraitDetectionResult.has_evidence`; `RoBERTaPersonalityDetector.is_trained`.
- `PersonalityOrchestrator(..., event_bus=None)`, `.build_erasure()`, `.purge_user_state(user_id)`; `ProfileStore(max_profiles=2000, ttl_seconds=None, clock=time.monotonic)`, `.user_lock(user_id)`; `PersonalityServiceSettings.max_cached_profiles` (`PERSONALITY_SERVICE_MAX_CACHED_PROFILES`); `/style` `custom_params.confidence`.
- Slot 021: unique index `uq_personality_profiles_user_id`; table `personality_erasure_tombstones` with policy `personality_erasure_tombstones_by_user`.
- Analytics: `main.SETTINGS`, `main.SPEC`, `start_kafka_ingestion(runtime, consumer, settings)` (W0 only, removed in W2), `ANALYTICS_CONSUMER_GROUP = "solace-group-analytics"`, `ANALYTICS_SUBSCRIPTIONS`, `register_analytics_ingestion(plane, consumer)`, `wire_analytics_ingestion(runtime, consumer)`; `AnalyticsConsumer.handle_event(event)`; `ServiceConfig.env` accepts `test` and falls back to `ENVIRONMENT`.
- Analytics store and reports: `timestamp=` on every recording and tracking method; `MetricsStore(max_windows_per_metric=1000, retention_hours=168, minute_resolution_hours=2)`, `.started_at`, `.retention_hours`, statistic `records_dropped`; `track_therapy_session_started`, metric `therapy.sessions_started`; `track_escalation(priority, crisis_level=None, timestamp=None)`; report summary keys `coverage_start` and `coverage_complete`; `DashboardResponse.coverage_start`; `ANALYTICS_READ_ROLES = ("admin", "superadmin")`.
- Test fixtures: `services/personality_service/tests/conftest.py` (`FakeCodec`, `fake_codec`, `RecordingConnection` with settable `.status`, `recording_cpm`); `tests/realinfra/personality-analytics/conftest.py` (`superuser_conn`, `seed_users`, `cpm_default`, `phi_codec`), with WS-17's names and semantics.

## Self-review

**Coverage**

| Register id | Severity | Task |
|---|---|---|
| MVP-027 | blocker | WS-18.1 |
| MVP-080 | high | WS-18.4 (schema prerequisite WS-18.3) |
| MVP-108 | medium | WS-18.9 |
| MVP-109 | medium | WS-18.13 |
| MVP-110 | medium | WS-18.2 (superseded by the plane's readiness in WS-18.9) |
| MVP-111 | medium | WS-18.18 |
| MVP-171 | medium | WS-18.6 |
| MVP-172 | medium | WS-18.12 (tombstone table WS-18.3) |
| MVP-232 | low | WS-18.18 |
| MVP-233 | low | WS-18.9 (the live-path loss), WS-18.18 (stop and malformed counting) |
| MVP-234 | low | WS-18.14 |
| MVP-235 | low | WS-18.10 |
| MVP-236 | low | WS-18.9 |
| MVP-237 | low | WS-18.11 |
| MVP-238 | low | WS-18.19 |
| MVP-239 | low | WS-18.19 |
| MVP-240 | low | WS-18.19 |
| MVP-241 | low | WS-18.11 |
| MVP-300 | low | WS-18.15 |
| MVP-301 | low | WS-18.16 (the lock over the read-modify-write lands in WS-18.12) |
| MVP-302 | low | WS-18.17 |
| MVP-303 | low | WS-18.7 |
| MVP-304 | low | WS-18.5 |
| MVP-305 | low | WS-18.15 |
| MVP-306 | low | WS-18.7 |
| MVP-307 | low | WS-18.16 |

All 26 ids are covered (blocker 1, high 1, medium 6, low 18). Every id sits in the wave the skeleton notes give it, except MVP-301's lock part, which moved from W5 to W3 because MVP-172 needs it.

**Incoming requests**
- WS-01.7 (R-01.12): WS-18.1 (personality `events.py`) and WS-18.2 (analytics `main.py` import root and the seven F821s), both W0.
- WS-04.5: WS-18.1 and WS-18.2, W0, flags exactly as in its table; analytics drops `prometheus_fastapi_instrumentator`.
- WS-06.8: WS-18.9, W2 (`plane.subscribe(..., group_id="solace-group-analytics")`; no `create_dead_letter_handler`/`ensure_table`).
- WS-07.9: WS-18.8 (personality) and WS-18.10 (analytics ingest), W2. Personality's `/profile/update` becomes service-only as requested.
- WS-09.4 (CR-7): WS-18.6, W2 (non-critical `llm` readiness; `LLMUnavailableError`, other errors and an empty reply yield no placeholder; cases `cr7-llm-unavailable` and `cr7-empty-reply`). Analytics builds no LLM client, so CR-7 has no analytics part.
- WS-09.15 (CR-10): WS-18.4, W2 (the ADOPTERS file imports `solace_security.phi_codec`, writes `active_key_id`; `test_cr10_repository_adopts_the_if13_codec_and_no_legacy_encryptor` pins it).
- WS-11.11: WS-18.8, W2 (`test_mvp_022_contract_orchestrator_personality.py`).

**Review focus (numbered as in the skeleton)**
- 1 (crisis disclosures outside the phrase set) and 2 (alert-path partial failures): not in WS-18 code; personality and analytics neither detect crises nor page anyone.
- 3 (foreign-but-valid identifiers, context-less work under RLS): WS-18.8 `test_mvp_314_obo_principal_acts_only_for_its_user`; WS-18.4 `test_mvp_080_rls_hides_another_users_profile`; WS-18.3 tombstone RLS test; WS-18.12 runs erasure under `acting_as_user` on real Postgres.
- 4 (PHI escaping through unplanned sinks): WS-18.13 (payload, nested values and exception text kept out of analytics logs); WS-18.7 (no `str(e)` in error bodies); WS-18.4 (ciphertext at rest checked in raw SQL); WS-18.17 (events carry ids and counts only).
- 5 (strict boots with partially valid configuration): WS-18.1 (no unused Redis/DB password required; environment follows `ENVIRONMENT`); WS-18.2 (a dead consumer fails strict boot and shows in `/ready`; `test` accepted); WS-18.4 (a strict build without Postgres or the codec fails instead of falling back to the in-memory repository); WS-18.9 (strict with no plane fails startup).

REVIEW_FOCUS: 3=WS-18.3,WS-18.4,WS-18.8,WS-18.12; 4=WS-18.4,WS-18.7,WS-18.13,WS-18.17; 5=WS-18.1,WS-18.2,WS-18.4,WS-18.9

**Interface consistency**
- IF-01 flags match the WS-04.5 table for both services; lifespans are bracketed by the runtime with no platform try/except.
- IF-03: every personality table access goes through `ConnectionPoolManager.acquire/transaction`; consumer work runs under `acting_as_user` (IF-16 helper); slot 021 uses the NULLIF form; JSONB parameters are Python objects.
- IF-04/IF-11: receiver-side permissions; obo principals bound to their user; the provider schema stays canonical (additive `custom_params.confidence`, stricter `sources`).
- IF-05: group `solace-group-analytics`; handlers raise to get retry and DLQ; no in-memory DLQ in strict.
- IF-13: field names and owner AAD live in the repository module; `encryption_key_id = active_key_id`.
- IF-16: store name `personality_postgres` unchanged; `purge_caches = PersonalityOrchestrator.purge_user_state`; registration shape `register_personality_erasure` matches WS-17's `register_memory_erasure`.
- Realinfra fixture names and `FakeCodec` semantics match WS-17.

**Open risks**
- WS-18.12 now depends on WS-14.13 (the IF-16 helper, W3 phase A); the deleter marks the user erased itself, so the helper's purge order does not matter.
- The erased set and the per-user lock are process-local; that is enough for one replica (horizontal scaling is out of scope), and the slot-021 tombstone guards restarts and any second writer at the database.
- `/profile/update` loses its user-JWT path (WS-07's request; it has no caller). If the product wants user self-report later, it needs a user route with its own ownership check.
- Personality is `audited=True` but no register entry or request asks it to adopt `audit_phi_access`; recommended for the W3 bug-finding round.
- Analytics counters stay in memory (D-18); reports now state their coverage, but a restart still empties them. `repository.py` (ClickHouse) stays dead and unedited because ClickHouse paths are frozen.
- Analytics read routes are role-gated in W3 but not MFA-gated until WS-07.11's adopters include them (R-18-06).
- WS-18.8's contract test is a provider guard that may already pass before the change; its RED evidence is the permission tests.
- Test-count deltas need a ledger Ruling: WS-18.1 removes two config tests, WS-18.2 removes the Prometheus, OpenTelemetry, `create_app` and router-health tests, WS-18.4 replaces two factory tests with one, WS-18.9 replaces the two W0 Kafka tests; each is replaced by RED-first tests.
- W5's entry criterion asks for D-05 sign-off on scoring changes; WS-18.15 changes personality style evidence, not clinical scoring, and should be ruled out of that gate in the ledger.

## Review responses

- Automated check, Error: plan a task for WS-09's CR-7 request (needed by WS-09.4: register a non-critical `llm` readiness check and catch `LLMUnavailableError` instead of testing for `""`): fixed. WS-18.6 already registered `runtime.register_readiness_check("llm", llm_client.readiness, critical=False)` and named the request in Covers and Produces; it now also pins the IF-15 error and the empty reply (`cr7-llm-unavailable`, `cr7-empty-reply` cases of `test_mvp_171_llm_failure_adds_no_placeholder_and_is_labelled_honestly`; today `trait_detector.py:183-185` and line 192 turn both into neutral placeholder scores), and its readiness text now defers to WS-09.4's strings instead of inventing `"llm: disabled"`. Personality is WS-18's only `UnifiedLLMClient` builder (`services/personality_service/src/main.py:118-120`); analytics has none. The request's Files column is prose ("each service's main.py and LLM call sites", WS-09-phi-codec-audit-llm.md:1767), so a file-matching check cannot see the match.
- Automated check, Error: plan a task for WS-09's CR-10 request (needed by WS-09.15: IF-13 adoption in the ADOPTERS file): fixed. WS-18.4 already rewrote `services/personality_service/src/infrastructure/postgres_repository.py` (the personality ADOPTERS path, WS-09-phi-codec-audit-llm.md:1710) with the `solace_security.phi_codec` import and `encryption_key_id = codec.active_key_id`, and its Produces line names WS-09.15. It now has a local lock, `test_cr10_repository_adopts_the_if13_codec_and_no_legacy_encryptor` (red today: no codec import; the legacy half guards the `configure_phi_encryption` block at `main.py:100-105` that WS-18.1 deletes), plus `test_mvp_080_strict_build_without_the_codec_fails_instead_of_dropping_durability`. As for CR-7, the request names no concrete file ("the eight ADOPTERS files", WS-09-phi-codec-audit-llm.md:1770).
- Automated check, Warning: WS-18.12 named a workstream (`WS-14 (IF-16 helper ...)`) instead of a task in Depends on: fixed. It now depends on WS-14.13, which creates `src/solace_infrastructure/gdpr/consumer.py` with `register_user_erasure_consumer` in W3 phase A (WS-14-accounts-roster-consent-erasure.md:1875-1891); Interfaces, R-18-09 and the open-risk note were updated to match.
- decision 2026-10-19 D-04: checked WS-18.6's D-04 gate and the rest of this plan. No text names Portkey as the production route. Personality still defers to the shared client's PHI-egress gate (`is_available`, `readiness()`), which WS-09.4 now applies to the Bedrock route that strict environments use (WS-09.18). Personality's `UnifiedLLMClient(LLMClientSettings())` call site needs no change, so no task changed.
