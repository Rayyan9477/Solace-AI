# WS-12: Safety service detection, escalation records and the shared crisis lexicon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The safety service detects crises from one clinician-reviewable shared lexicon with max-only fusion (no dilution, no brittle negation, no benign single-word pages), records every HIGH/CRITICAL decision durably and publishes it only through its transactional outbox, protects its endpoints and erases on request.

**Architecture:** A new pure package `src/solace_clinical` (IF-07) owns text normalization, the YAML crisis lexicon v1, `screen_text`, the max-only `fuse_risk` rule, the per-locale versioned crisis-resource registry (D-29: only en-US is populated; any other or unknown locale gets a generic fallback, never an empty list) and the 200+ phrase regression set used by every detector. Inside safety-service, `CrisisDetector` fuses the lexicon signal with raise-only LLM, history, context and trajectory signals; `SafetyService` writes the escalation row and its outbox rows (CrisisDetected, EscalationTriggered, SafetyAssessment, all carrying the turn id as correlation id) in one Postgres transaction through `ServiceEventPlane.publish(event, conn=)` (IF-05) and never calls notification-service (D-28). Paging, coalescing and acknowledgement belong to WS-13 (IF-09).

**Covers:** 21 register entries (blocker 1, high 10, medium 9, low 1).

**Waves:** 0 (runtime adoption, strict escalation store, no private Postgres pool), 1 (shared lexicon in phase A; detection, optional-ML parity, durable decision records and the /check contract in phase B), 3 (PHI codec on escalation records and erasure fan-out), 5 (dead engines removed, durable risk history, bounded in-memory state).

**Depends on:**
- WS-04: IF-01 runtime bootstrap (W0 phase A2) for the main.py adoption; the IF-18 crisis counter accessor (request R-12-5).
- WS-02: slot scaffold with an empty `014_safety` revision and the migrate job (W0); `ConnectionPoolManager` pools with the JSONB codec (IF-03).
- WS-06: IF-05 `ServiceEventPlane.publish(event, *, conn=None)`, `get_service_event_plane()` and `solace_events.errors` (W1 phase A1); safety bridge refactor (R-12-1); `record_publish_failure` (R-12-2).
- WS-07: IF-04 `require_service_permission` with receiver-side permission resolution (W1 phase A1).
- WS-09: IF-15 `LLMUnavailableError` and non-raising `initialize()` (W1 phase A1); IF-13 `get_phi_codec()` (W2).
- WS-14: IF-16 `register_user_erasure_consumer` (W3 phase A).
- WS-05: IF-19 realinfra fixtures (`pg_migrated`) and IF-20 contract helpers; the safety-to-notification event contracts in `tests/contracts/events/safety__notification/` (task WS-05.7, W1 phase A) consumed by WS-12.8.
- WS-01: one lock carrying PyYAML, hypothesis and aiokafka, and the `clinical`, `realinfra`, `contract` markers (W0).
- WS-10: the orchestrator-to-safety consumer contract `tests/contracts/orchestrator__safety/check.json` (task WS-10.16, W2), verified on the provider side by the test WS-12.10 writes.
- WS-13: none. WS-13.13 verifies the consumer half of the same WS-05.7 event contracts (R-13-7 ruling).

**Owned paths:** `services/safety_service/src/domain`, `services/safety_service/src/ml`, `services/safety_service/src/db`, `services/safety_service/src/observability`, `services/safety_service/src/api.py`, `services/safety_service/src/config.py`, `services/safety_service/src/events.py`, `services/safety_service/src/main.py`, `services/safety_service/src/__init__.py`, `services/safety_service/src/infrastructure/{clinician_registry,database,gdpr,repository,telemetry,__init__}.py`, `services/safety_service/config`, `services/safety_service/db`, `services/safety_service/tests`, `services/safety_service/README.md`, `services/safety_service/__init__.py`, `src/solace_clinical`, `tests/clinical`, `migrations/versions/014_safety.py`, `docs/CLINICAL-VALIDATION.md`; new component tests in `tests/realinfra/safety-detection-lexicon/`.

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

**Conventions used by every task below.**
- Commands run from the repo root `D:\Repo\Contextual-Chatbot` in PowerShell 5.1. Unit command template: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q`. Real-infra commands prepend `$env:SOLACE_REQUIRE_REALINFRA="1";` so a skipped container test fails.
- Safety tests import with the existing style `from services.safety_service.src.<module> import ...`. `pytest-asyncio` runs in auto mode (pytest.ini `asyncio_mode = auto`), so `async def test_...` needs no decorator.
- Real-infra test modules start with `pytestmark = [pytest.mark.realinfra, pytest.mark.integration]` and use only IF-19 fixtures.
- Never put message text, matched phrases or exception text in logs, events, metric labels or test output. Log `error_type=type(exc).__name__`, never `str(exc)`.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/safety_service/src/main.py` | Modify | IF-01 SPEC and lifespan; composition only through `domain/factory.py`; LLM client and readiness; erasure consumer | WS-12.1, WS-12.6, WS-12.9, WS-12.12, WS-12.15 |
| `services/safety_service/src/domain/factory.py` | Create | `build_crisis_detector`, `build_escalation_repository` composition seams | WS-12.1, WS-12.6, WS-12.9 |
| `services/safety_service/src/events.py` | Modify | Import root; synchronous outbox `SafetyEventPublisher`; category-only Kafka mapping | WS-12.1, WS-12.8 |
| `src/solace_clinical/__init__.py` | Create | IF-07 exports | WS-12.2, WS-12.3 |
| `src/solace_clinical/normalize.py` | Create | `normalize_text` | WS-12.2 |
| `src/solace_clinical/fusion.py` | Create | `RiskSignal`, `fuse_risk`, `level_rank` | WS-12.2 |
| `src/solace_clinical/resources.py` | Create | Per-locale versioned resource registry (D-29): `crisis_resources`, `crisis_resource_set`, `RESOURCES_VERSION`, `RESOURCE_VERSIONS`, `supported_locales`, `resolve_locale`, `canonical_locale`, the generic fallback with its log and metric | WS-12.2 |
| `src/solace_clinical/crisis_lexicon_v1.yaml` | Create | Lexicon v1 data (provisional, D-05) | WS-12.3 |
| `src/solace_clinical/lexicon.py` | Create | YAML loader, compiled rules, `LEXICON_VERSION` | WS-12.3 |
| `src/solace_clinical/screen.py` | Create | `screen_text`, `ScreenResult`, clause-bounded negation, self-reference | WS-12.3 |
| `src/solace_clinical/phrases.py` | Create | Phrase-set loader and detector checker shared by all detectors | WS-12.3 |
| `tests/clinical/crisis_phrases_v1.yaml` | Create | 200+ phrase regression set | WS-12.3 |
| `tests/clinical/legacy_terms_v0.yaml` | Create | Snapshot of legacy safety keywords and patterns with first-person examples | WS-12.3 |
| `tests/clinical/test_solace_clinical_core.py` | Create | Normalization, fusion, resources tests | WS-12.2 |
| `tests/clinical/test_crisis_phrases_v1.py` | Create | Phrase set against `screen_text` | WS-12.3 |
| `tests/clinical/test_screen_text_properties.py` | Create | hypothesis properties and the 5 ms budget | WS-12.3 |
| `tests/clinical/test_lexicon_legacy_coverage.py` | Create | Lexicon v1 is not weaker than the legacy lists | WS-12.3 |
| `docs/CLINICAL-VALIDATION.md` | Modify | §11 lexicon v1, fusion, thresholds, crisis watch, episode semantics, sign-off table; §3.3 protective factors | WS-12.3, WS-12.7 |
| `migrations/versions/014_safety.py` | Modify | escalations decision columns (W1); `safety_user_risk_state` table with RLS (W5) | WS-12.4, WS-12.14 |
| `services/safety_service/src/ml/llm_assessor.py` | Modify | Shared LLM client only, verbatim user text, bounded, failures unreliable | WS-12.5 |
| `services/safety_service/src/domain/crisis_detector.py` | Modify | Lexicon layer 1 plus raise-only signals, max-only fusion, level-derived score | WS-12.6 |
| `services/safety_service/src/config.py` | Modify | Remove stale crisis thresholds (0.9) and inline-notification settings | WS-12.6, WS-12.9 |
| `services/safety_service/src/domain/service.py` | Modify | Protective factors informational; transactional decision record + publish; resources; purge; durable risk state; bounds | WS-12.7, WS-12.9, WS-12.10, WS-12.12, WS-12.15, WS-12.16 |
| `services/safety_service/src/domain/escalation.py` | Modify | Decision records only (no delivery, no dedup, no in-memory ack); idempotent per turn; codec; pseudonymize | WS-12.9, WS-12.12 |
| `services/safety_service/src/infrastructure/clinician_registry.py` | Modify (delete) | Removed with D-28 | WS-12.9 |
| `services/safety_service/src/api.py` | Modify | IF-06 /check contract; service permissions | WS-12.10, WS-12.11 |
| `services/safety_service/src/infrastructure/gdpr.py` | Modify | Erasure of real stores only, escalation pseudonymization, risk-state deleter | WS-12.12, WS-12.15 |
| `services/safety_service/src/infrastructure/repository.py` | Modify | `PostgresRiskStateRepository` | WS-12.15 |
| `services/safety_service/src/infrastructure/database.py` | Modify (W0), then delete (W5) | No private asyncpg pool; an injected pool only (MVP-271 bypass, R-02.10) | WS-12.17, WS-12.13 |
| `services/safety_service/src/ml/{sentiment_analyzer,pattern_matcher,keyword_detector}.py` | Modify (W1), then delete (W5) | Optional ML libraries imported only when `SAFETY_ML_EXTRAS_ENABLED=true`, so tests take the image path (R-01.15) | WS-12.18, WS-12.13 |
| `services/safety_service/tests/test_if06_check_contract.py`, `test_mvp_017_contract_orchestrator_check.py`, `test_mvp_357_event_contracts.py` | Create | IF-06 /check behaviour; provider verification of WS-10.16's contract; producer verification of WS-05.7's event contracts | WS-12.10, WS-12.11, WS-12.8 |
| `services/safety_service/src/infrastructure/__init__.py`, `src/ml/__init__.py`, `README.md` | Modify | Exports and docs after dead-engine removal | WS-12.13 |
| `services/safety_service/src/ml/{contraindication,keyword_detector,pattern_matcher,sentiment_analyzer}.py`, `src/db/*`, `src/observability/__init__.py`, `src/infrastructure/database.py`, `db/**`, `config/{keywords,patterns,clinical_lexicon}.json` | Modify (delete) | Unwired engines removed (D-19) | WS-12.13 |
| `services/safety_service/tests/*` and `tests/integration/test_safety_service_e2e.py` | Modify | Existing tests of WS-12 code updated where behaviour changes (the e2e file tests only safety code, so the tests/integration protocol lets WS-12 edit it) | per task |
| `tests/integration/test_phi_encryption_activation.py` | Modify | Safety's entry only: `_PHI_ENCRYPTING` to `_PHI_VIA_RUNTIME` (WS-04.15's request) | WS-12.12 |
| `tests/realinfra/safety-detection-lexicon/*.py` | Create | Schema, transaction, erasure and restart proofs on postgres:16 | WS-12.4, WS-12.9, WS-12.12, WS-12.14, WS-12.15 |

## Tasks

### Task WS-12.1: Safety runtime adoption (IF-01) and a strict-only Postgres escalation store
**Covers:** MVP-088
**Severity:** high
**Wave:** 0
**Depends on:** none
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-14 (default assumed: strict means ENVIRONMENT staging or production through `is_strict_environment`; the pilot runs production)
**Real infra:** no
**Files:**
- Create: `services/safety_service/src/domain/factory.py` (`build_escalation_repository`; `build_crisis_detector` holding today's composition verbatim so WS-12.6 can fix the threshold test-first)
- Modify: `services/safety_service/src/main.py` (module-level `SPEC`; lifespan first statement `runtime = await start_service_runtime(app, SPEC)` and last `await stop_service_runtime(runtime)`; delete `configure_logging`, the CORS middleware, `_register_middleware`, `_register_exception_handlers`, and the local `/`, `/ready`, `/live`, `/health` routes; `create_application()` returns `create_service_app(SPEC, lifespan=lifespan, routers=[(router, "/api/v1/safety")], title="Solace-AI Safety Service")` and `app = create_application()`; composition through `domain/factory.py`; `_require_event_delivery_in_production` becomes `_require_event_delivery_in_strict`)
- Modify: `services/safety_service/src/events.py` (`to_kafka_event`: import `solace_events.schemas`, not `src.solace_events.schemas`)
- Modify: `services/safety_service/tests/test_event_delivery_guard.py` (staging now fails loud)
- Test: `services/safety_service/tests/test_mvp_088_strict_escalation_repository.py`
**Interfaces:**
- Consumes: IF-01 `ServiceRuntimeSpec`, `create_service_app(spec, *, lifespan, routers, title)`, `start_service_runtime(app, spec) -> ServiceRuntime`, `stop_service_runtime(runtime)`, `is_strict_environment(env)`, `StartupDependencyError`, `ServiceRuntime.pg_pool`, `ServiceRuntime.register_readiness_check(name, check, *, critical=True)` (WS-04, W0 phase A2).
- Produces: `SPEC = ServiceRuntimeSpec(service_name="safety-service", environment=<SAFETY_ENVIRONMENT>, version="1.0.0", log_level=<SAFETY_LOG_LEVEL>, debug=<SAFETY_DEBUG>, uses_postgres=True, uses_redis=False, event_plane="publish_consume", handles_phi=True, audited=True, rate_limited=False, edge=False)`; `build_escalation_repository(pg_pool: asyncpg.Pool | None, *, environment: str) -> EscalationRepository`; `build_crisis_detector(config: SafetyConfig | None = None, *, llm_assessor: Any | None = None) -> CrisisDetector`; `_require_event_delivery_in_strict(settings) -> None`.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_088_strict_escalation_repository.py`
```python
"""MVP-088: strict environments never keep crisis escalation records in process memory."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from solace_infrastructure.runtime import StartupDependencyError
from services.safety_service.src.domain.escalation import (
    InMemoryEscalationRepository, PostgresEscalationRepository,
)


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
```
Also edit `test_event_delivery_guard.py`: rename the import to `_require_event_delivery_in_strict`; `test_production_without_kafka_fails_loud` becomes `@pytest.mark.parametrize("env", ["staging", "production"]) test_strict_without_kafka_fails_loud` with `match="required in staging and production"`; the allowed list becomes `["development", "test"]`.
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_088_strict_escalation_repository.py services/safety_service/tests/test_event_delivery_guard.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_088_strict_boot_without_postgres_fails[staging]` fails with `DID NOT RAISE` (today main.py:141-153 logs `escalation_repository_in_memory_fallback` and boots under staging); `[production]` fails with `RuntimeError: Postgres escalation repository is required in production` instead of `StartupDependencyError`; the factory tests fail with `ModuleNotFoundError: ...domain.factory`; `test_mvp_088_spec_declares_every_platform_flag` fails with `AttributeError: ... has no attribute 'SPEC'`; the guard test fails because staging does not raise.
- [ ] **Step 3: Implement** `build_escalation_repository(pg_pool, *, environment)` and `build_crisis_detector(config=None, *, llm_assessor=None)` in `services/safety_service/src/domain/factory.py`, then rewrite the lifespan in `main.py`. `build_escalation_repository` returns `PostgresEscalationRepository(pg_pool)` whenever a pool exists; with no pool it raises `StartupDependencyError("escalation records require Postgres in strict environments")` when `is_strict_environment(environment)`, else logs WARNING `escalation_repository_in_memory` and returns `InMemoryEscalationRepository()`. `build_crisis_detector` moves main.py:134-138 verbatim (it still reads `config.crisis_detection`; WS-12.6 fixes that). Lifespan order after `start_service_runtime`: `config = get_safety_config()`; detector and repository through the factory; `EscalationManager(EscalationSettings(...), repository=repo)`; `await initialize_event_publisher()`; `SafetyService(...)`; `await safety_service.initialize()`; the existing PHI-encryption activation block stays verbatim except that `settings.environment == "production"` becomes `is_strict_environment(settings.environment)` (IF-13 retires it; WS-12.12 removes it in W3); `_require_event_delivery_in_strict(settings)`; when `settings.kafka_enabled`, call `initialize_notification_bridge(kafka_settings=KafkaSettings(bootstrap_servers=settings.kafka_bootstrap_servers), use_mock=settings.kafka_use_mock, postgres_pool=runtime.pg_pool)` with no blanket except in strict; outside strict a bridge failure logs WARNING `notification_bridge_unavailable` (no exception text) and registers `runtime.register_readiness_check("event_bridge", _unavailable, critical=True)` returning `(False, "unavailable")`. Shutdown stops the bridge, the publisher and the service, then `await stop_service_runtime(runtime)`. Keep `run_server()`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the safety suite, then the integration files that import safety code, one process each. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests -o "addopts=" -p no:logging -q` then the same with `tests/integration/test_crisis_flow.py tests/integration/test_safety_service_e2e.py tests/integration/test_event_flow.py tests/integration/test_phi_encryption_activation.py` (WS-04.15 keeps safety in `_PHI_ENCRYPTING` until W3, request R-12-7). Expected: all pass. (`tests/system/test_smoke.py::test_safety_service_health` asserts the pre-IF-01 `/health` body; WS-05 updates it under request R-12-9.)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.2: solace_clinical core: normalization, max-only fusion and versioned crisis resources
**Covers:** MVP-030
**Severity:** blocker
**Wave:** 1
**Depends on:** none
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: max-only fusion; LLM, fallback and stub signals may raise but never lower a level; owner acceptance recorded as a ledger Ruling until clinician sign-off) and D-29 (decided 2026-10-19: several countries, US first. Resources are a per-locale, versioned registry; only en-US is populated, with exactly today's set (988 Suicide and Crisis Lifeline, call or text 988; Crisis Text Line, text HOME to 741741; 911); every other or unknown locale gets a generic fallback (the local emergency number plus an international helpline directory), never an empty list and never an error, logged without PHI and counted by a metric. Each further country is added only with clinician-verified resources before its users join; the fallback copy is provisional until the D-29 clinician sign-off)
**Real infra:** no
**Files:**
- Create: `src/solace_clinical/__init__.py`, `src/solace_clinical/normalize.py`, `src/solace_clinical/fusion.py`, `src/solace_clinical/resources.py`, `src/solace_clinical/escalation_reason.py`
- Test: `tests/clinical/test_solace_clinical_core.py`
**Interfaces:**
- Consumes: `solace_common.enums.CrisisLevel` (frozen values, WS-07).
- Produces (IF-07): `normalize_text(text: str) -> str`; `CATEGORIES: frozenset[str]` = {suicidal_ideation, suicide_plan, means_access, self_harm, overdose, homicidal_ideation, abuse_disclosure, imminent_danger, hopelessness}; `@dataclass(frozen=True) RiskSignal(source: str, level: CrisisLevel, confidence: float, reliable: bool)`; `fuse_risk(signals: Sequence[RiskSignal]) -> CrisisLevel`; `crisis_resources(locale: str | None = "en-US") -> list[dict]` with keys `name, contact, kind` (fresh copies; `crisis_resources("en-US")` and `crisis_resources()` return exactly today's three en-US entries); `RESOURCES_VERSION: str = "1.0.0"`, kept as the en-US version (`RESOURCE_VERSIONS[DEFAULT_LOCALE]`) so every existing en-US assertion, WS-05.6's `resources_version == RESOURCES_VERSION` included, stays true. New: `level_rank(level: CrisisLevel) -> int` (NONE 0 ... CRITICAL 4).
- Produces (D-29 locale registry, all in `solace_clinical.resources` and re-exported; the one locale contract WS-08, WS-10, WS-11, WS-14 and WS-15 use): `DEFAULT_LOCALE: str = "en-US"`; `UNDETERMINED_LOCALE: str = "und"`; `supported_locales() -> frozenset[str]` (today `{"en-US"}`); `RESOURCE_VERSIONS: Mapping[str, str]` (read-only, today `{"en-US": "1.0.0"}`; one version per locale); `FALLBACK_RESOURCES_VERSION: str = "fallback-1.0.0"`; `canonical_locale(tag: object) -> str | None` (exactly WS-08.7's `normalize_locale` grammar `^[a-z]{2,3}([-_][A-Za-z]{2})?$`, case-insensitive, returning `ll` or `ll-CC`; `None` for anything else, non-`str` included; never raises); `resolve_locale(claim: object, requested: object = None) -> str` (a `None` or blank claim is `DEFAULT_LOCALE`, so a missing token claim means en-US; a malformed claim is `UNDETERMINED_LOCALE`; `requested` is honoured only when it is in `supported_locales()` and keeps the claim's region, so a request-supplied locale can refine the language but never leave the supported set or change the country; never raises); `@dataclass(frozen=True) CrisisResourceSet(locale: str, resources: tuple[Mapping[str, str], ...], version: str, fallback: bool)` (`locale` is the registry key served, or `"fallback"`); `crisis_resource_set(locale: object = DEFAULT_LOCALE) -> CrisisResourceSet` (the call that serves, logs and counts; callers that need both the list and the version call it once); `resources_version(locale: object = DEFAULT_LOCALE) -> str` (a pure lookup: the locale's version or `FALLBACK_RESOURCES_VERSION`; no log, no count). Fallback (provisional copy, D-29 sign-off): `[{"name": "Local emergency services", "contact": "Call your local emergency number if you are in immediate danger", "kind": "phone"}, {"name": "Find A Helpline (international directory)", "contact": "https://findahelpline.com", "kind": "url"}]`. Every fallback served logs WARNING `crisis_resources_locale_fallback` with `reason` in {`unsupported_locale` (well-formed, not in the registry), `unknown_locale` (malformed or `und`)} and, only for a well-formed tag other than `und`, `locale_tag` (a language or country code, never user data), and increments the counter `CRISIS_RESOURCES_FALLBACK` (`solace_crisis_resources_fallback_total{reason}`, prometheus_client, registered in this module; the alert rule is R-12-11). No setting is introduced: `DEFAULT_LOCALE` is a clinical constant like the D-06 watch values, not an environment variable.
- Produces (IF-06 `escalation_reason` format, crisis-review F10; the one definition WS-12.8, WS-12.9, WS-05.7, WS-06.12 and WS-13.6 use): `format_escalation_reason(source: Literal["auto", "manual"], level: str, categories: Iterable[str]) -> str` (`"<source>:<LEVEL>"`, plus `":" + ",".join(sorted codes in CATEGORIES)` when any remain; other values are dropped); `parse_escalation_reason(reason: str) -> tuple[str, str, tuple[str, ...]] | None` (`(source, level, sorted codes)` when the whole string matches `^(auto|manual):(NONE|LOW|ELEVATED|MEDIUM|HIGH|CRITICAL)(:[a-z_]+(,[a-z_]+)*)?$` and every code is in `CATEGORIES`, else `None`; total over every `str`, never raises). `"unspecified"` is the only other value an `escalation_reason` may carry.

- [ ] **Step 1: Write the failing test** in `tests/clinical/test_solace_clinical_core.py`
```python
"""IF-07 core. MVP-030: no signal may lower a level; fusion is max-only. D-29: per-locale resources, never empty."""
import pytest
from hypothesis import given, settings, strategies as st
from prometheus_client import REGISTRY
from structlog.testing import capture_logs

from solace_clinical import (CATEGORIES, DEFAULT_LOCALE, FALLBACK_RESOURCES_VERSION, RESOURCE_VERSIONS,
                             RESOURCES_VERSION, RiskSignal, canonical_locale, crisis_resource_set, crisis_resources,
                             format_escalation_reason, fuse_risk, level_rank, normalize_text, parse_escalation_reason,
                             resolve_locale, resources_version, supported_locales)
from solace_common.enums import CrisisLevel as L

pytestmark = pytest.mark.clinical
LEVELS = list(L)


def test_normalize_folds_unicode_apostrophes_quotes_and_case():
    assert normalize_text("I Can\u2019t go on") == "i can't go on"
    assert normalize_text("don\u02bct and it\u2018s and \uff07x") == "don't and it's and 'x"
    assert normalize_text("\u201cEND it\u201d") == '"end it"'
    assert normalize_text("\uff33\uff35\uff29\uff23\uff29\uff24\uff25") == "suicide"


def test_normalize_strips_zero_width_and_collapses_whitespace():
    assert normalize_text("  kill\u200b my\u200dself \n\t now\ufeff ") == "kill myself now"


@settings(max_examples=500, deadline=None)
@given(st.text())
def test_normalize_is_total_and_idempotent(raw):
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_categories_are_exactly_the_nine_codes():
    assert CATEGORIES == frozenset({"suicidal_ideation", "suicide_plan", "means_access", "self_harm", "overdose",
                                    "homicidal_ideation", "abuse_disclosure", "imminent_danger", "hopelessness"})


def test_fuse_risk_takes_the_maximum_and_unreliable_never_lowers():
    assert fuse_risk([]) == L.NONE
    assert fuse_risk([RiskSignal("lexicon", L.HIGH, 1.0, True), RiskSignal("llm", L.NONE, 0.2, False)]) == L.HIGH
    assert fuse_risk([RiskSignal("lexicon", L.NONE, 1.0, True), RiskSignal("llm", L.CRITICAL, 0.3, False)]) == L.CRITICAL


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(LEVELS)), st.sampled_from(LEVELS), st.booleans())
def test_adding_any_signal_never_lowers_and_order_is_irrelevant(levels, extra, reliable):
    signals = [RiskSignal(f"s{i}", lvl, 0.5, True) for i, lvl in enumerate(levels)]
    base = fuse_risk(signals)
    assert level_rank(fuse_risk(signals + [RiskSignal("x", extra, 0.1, reliable)])) >= level_rank(base)
    assert fuse_risk(list(reversed(signals))) == base


EN_US = [
    {"name": "988 Suicide and Crisis Lifeline", "contact": "Call or text 988", "kind": "phone"},
    {"name": "Crisis Text Line", "contact": "Text HOME to 741741", "kind": "text"},
    {"name": "Emergency services", "contact": "Call 911 if you are in immediate danger", "kind": "phone"},
]
FALLBACK = [
    {"name": "Local emergency services", "contact": "Call your local emergency number if you are in immediate danger",
     "kind": "phone"},
    {"name": "Find A Helpline (international directory)", "contact": "https://findahelpline.com", "kind": "url"},
]


def _fallbacks(reason):
    return REGISTRY.get_sample_value("solace_crisis_resources_fallback_total", {"reason": reason}) or 0.0


def test_crisis_resources_en_us_are_versioned_and_copied():
    assert crisis_resources() == EN_US and crisis_resources("en-US") == EN_US and RESOURCES_VERSION == "1.0.0"
    assert DEFAULT_LOCALE == "en-US" and supported_locales() == frozenset({"en-US"})
    assert dict(RESOURCE_VERSIONS) == {"en-US": "1.0.0"} and resources_version("en-US") == RESOURCES_VERSION
    crisis_resources()[0]["contact"] = "mutated"
    assert crisis_resources() == EN_US


def test_d29_missing_locale_means_en_us_and_is_not_a_fallback():
    before = _fallbacks("unsupported_locale") + _fallbacks("unknown_locale")
    assert crisis_resources(None) == EN_US and resolve_locale(None) == "en-US" and resolve_locale("  ") == "en-US"
    assert crisis_resource_set(None).fallback is False and resources_version(None) == RESOURCES_VERSION
    assert _fallbacks("unsupported_locale") + _fallbacks("unknown_locale") == before


@pytest.mark.parametrize("tag", ["en-us", "en_US", "EN-us"])
def test_d29_locale_tags_are_canonicalized(tag):
    assert canonical_locale(tag) == "en-US" and resolve_locale(tag) == "en-US" and crisis_resources(tag) == EN_US


@pytest.mark.parametrize("tag,reason", [("fr-FR", "unsupported_locale"), ("en-GB", "unsupported_locale"),
                                        ("en", "unsupported_locale"), ("und", "unknown_locale"),
                                        ("english<script>", "unknown_locale"), ("en--US", "unknown_locale"),
                                        (42, "unknown_locale")])
def test_d29_unsupported_or_unknown_locale_gets_the_generic_fallback(tag, reason):
    before = _fallbacks(reason)
    with capture_logs() as logs:
        served = crisis_resource_set(tag)
    assert served.fallback is True and served.version == FALLBACK_RESOURCES_VERSION and served.locale == "fallback"
    assert [dict(r) for r in served.resources] == FALLBACK and _fallbacks(reason) == before + 1
    assert crisis_resources(tag) == FALLBACK and resources_version(tag) == FALLBACK_RESOURCES_VERSION
    assert not any("988" in r["contact"] or "911" in r["contact"] for r in FALLBACK)
    (event,) = [e for e in logs if e["event"] == "crisis_resources_locale_fallback"]
    assert event["reason"] == reason and set(event) <= {"event", "log_level", "reason", "locale_tag"}
    assert "script" not in str(event)


def test_d29_a_request_locale_only_refines_within_the_supported_set_and_never_changes_the_country():
    assert resolve_locale(None, requested="en-US") == "en-US"
    assert resolve_locale("en-US", requested="fr-FR") == "en-US"      # never leaves the supported set
    assert resolve_locale("en-US", requested="<bad>") == "en-US"
    assert resolve_locale("fr-FR", requested="en-US") == "fr-FR"      # never moves a user to another country's numbers
    assert resolve_locale("english<script>") == "und" and crisis_resources(resolve_locale("fr-FR")) == FALLBACK


def test_d29_every_supported_locale_has_resources_and_its_own_version():
    for loc in supported_locales():
        served = crisis_resource_set(loc)
        assert served.resources and served.fallback is False and served.version == RESOURCE_VERSIONS[loc] == resources_version(loc)


@settings(max_examples=300, deadline=None)
@given(st.one_of(st.none(), st.text(), st.integers()))
def test_d29_resources_are_total_and_never_empty(raw):
    assert crisis_resources(raw) and resolve_locale(raw) and resources_version(raw)
    assert resolve_locale(raw) in supported_locales() | {"und", canonical_locale(raw)}


def test_crisis_review_f10_escalation_reason_round_trips_codes_and_rejects_text():
    reason = format_escalation_reason("auto", "CRITICAL", ["suicidal_ideation", "means_access", "free text"])
    assert reason == "auto:CRITICAL:means_access,suicidal_ideation"
    assert parse_escalation_reason(reason) == ("auto", "CRITICAL", ("means_access", "suicidal_ideation"))
    assert format_escalation_reason("manual", "HIGH", []) == "manual:HIGH"
    assert parse_escalation_reason("manual:HIGH") == ("manual", "HIGH", ())
    for bad in ("Triggers: kill myself", "auto:CRITICAL:not_a_code", "auto:critical:self_harm", "unspecified", ""):
        assert parse_escalation_reason(bad) is None


@settings(max_examples=300, deadline=None)
@given(st.text())
def test_parse_escalation_reason_is_total(raw):
    parse_escalation_reason(raw)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/clinical/test_solace_clinical_core.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'solace_clinical'`. Today the only fusion is the weighted average in `crisis_detector.py:738-804` that averages a failing LLM's 0.2 stub into the score (MVP-030). The `test_d29_*` cases pin D-29: today the resources are hard-coded US lists in several services and no locale is read anywhere, so a non-US user would be shown 988 and 911.
- [ ] **Step 3: Implement** `normalize_text(text: str) -> str`, `fuse_risk(signals) -> CrisisLevel`, `level_rank(level) -> int`, `crisis_resources(locale="en-US") -> list[dict]` and the D-29 locale names from Interfaces in the files above; `__init__.py` re-exports them plus `CATEGORIES`, `RiskSignal`, `RESOURCES_VERSION`, `RESOURCE_VERSIONS`, `FALLBACK_RESOURCES_VERSION`, `DEFAULT_LOCALE`, `UNDETERMINED_LOCALE`, `CrisisResourceSet`, `crisis_resource_set`, `resources_version`, `supported_locales`, `resolve_locale` and `canonical_locale`. `normalize_text` applies, until the output stops changing (at most 4 passes): remove U+200B, U+200C, U+200D, U+2060 and U+FEFF; NFKC; `str.casefold()`; NFKC; map U+2018, U+2019, U+02BC, U+FF07 to `'` and U+201C, U+201D to `"`; collapse `\s+` to one space and strip. `fuse_risk` is the maximum by `level_rank`, `NONE` for an empty sequence; `reliable` is carried for callers' degraded markers and never used to lower. `format_escalation_reason` and `parse_escalation_reason` live in `escalation_reason.py` with the pattern as a module constant `ESCALATION_REASON_PATTERN`. `resources.py` holds the D-29 registry as module data: `_REGISTRY = MappingProxyType({"en-US": LocaleResources(version="1.0.0", resources=<the three EN_US entries>)})`, and `FALLBACK_RESOURCES` with the two fallback entries; `RESOURCE_VERSIONS`, `RESOURCES_VERSION = RESOURCE_VERSIONS[DEFAULT_LOCALE]` and `supported_locales()` derive from `_REGISTRY`, so adding a country is one clinician-verified entry with its own version (the add-a-locale rule WS-12.3 records in CLINICAL-VALIDATION §11). `canonical_locale` uses the compiled pattern with `re.IGNORECASE` and returns `lang.lower()` plus `"-" + region.upper()`; `crisis_resource_set` resolves `None` or blank to `DEFAULT_LOCALE`, serves the registry entry of a canonical supported tag, and otherwise serves the fallback, logging the WARNING through `structlog.get_logger("solace_clinical")` and incrementing `CRISIS_RESOURCES_FALLBACK.labels(reason=...)` inside `try/except Exception: pass`, so serving resources never raises and never depends on the LLM, a downstream service, Kafka or the database. The package is pure: no I/O at import time (the counter is a prometheus_client object; the registry is static data).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/clinical -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.3: Crisis lexicon v1, screen_text, the 200+ phrase set and the clinical record
**Covers:** MVP-084, MVP-182, MVP-086, MVP-178
**Severity:** high
**Wave:** 1
**Depends on:** WS-12.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: provisional lexicon v1 built from safety's keyword lists, patterns and `clinical_lexicon.json` plus every register gap; owner acceptance recorded; named clinician signs before W5), D-06 (episode semantics recorded for review), D-29
**Real infra:** no
**Files:**
- Create: `src/solace_clinical/crisis_lexicon_v1.yaml`, `src/solace_clinical/lexicon.py`, `src/solace_clinical/screen.py`, `src/solace_clinical/phrases.py`, `tests/clinical/crisis_phrases_v1.yaml`, `tests/clinical/legacy_terms_v0.yaml`
- Modify: `src/solace_clinical/__init__.py` (export `screen_text`, `ScreenResult`, `LEXICON_VERSION`); `docs/CLINICAL-VALIDATION.md` (new §11 and header note)
- Test: `tests/clinical/test_crisis_phrases_v1.py`, `tests/clinical/test_screen_text_properties.py`, `tests/clinical/test_lexicon_legacy_coverage.py`
**Interfaces:**
- Consumes: WS-12.2 `normalize_text`, `CATEGORIES`, `level_rank`.
- Produces (IF-07): `screen_text(text: str) -> ScreenResult`; `@dataclass(frozen=True) ScreenResult(level: CrisisLevel, categories: frozenset[str], rule_ids: tuple[str, ...], negated_rule_ids: tuple[str, ...], lexicon_version: str)`; `LEXICON_VERSION: str = "1.0.0"`; lexicon YAML `metadata {version, approved_by, approved_at, jurisdiction}` and `rules [{id, category, level, patterns, requires_first_person, negation_policy}]`. New: `solace_clinical.phrases.load_phrase_set(path: str | os.PathLike | None = None) -> PhraseSet` (default `<repo>/tests/clinical/crisis_phrases_v1.yaml`, resolved from the module file), `PhraseSet(version: str, positive: tuple[PositivePhrase, ...], negative: tuple[NegativePhrase, ...])`, `PositivePhrase(text, min_level: CrisisLevel, categories: frozenset[str])`, `NegativePhrase(text, max_level: CrisisLevel)`, `PhraseFailure(text, kind, expected, actual)`, `check_detector(detect: Callable[[str], CrisisLevel | tuple[CrisisLevel, Iterable[str]]], phrase_set: PhraseSet | None = None) -> list[PhraseFailure]` and `async check_detector_async(detect, phrase_set=None) -> list[PhraseFailure]` (categories are checked only when `detect` returns a tuple). WS-10, WS-15, WS-16 and WS-17 call `check_detector[_async]` from their own suites.

- [ ] **Step 1: Write the failing test** in `tests/clinical/test_crisis_phrases_v1.py`, `tests/clinical/test_screen_text_properties.py` and `tests/clinical/test_lexicon_legacy_coverage.py`
```python
# tests/clinical/test_crisis_phrases_v1.py
from collections import Counter
import pytest
from solace_clinical import CATEGORIES, LEXICON_VERSION, level_rank, screen_text
from solace_clinical.lexicon import load_lexicon
from solace_clinical.phrases import load_phrase_set
from solace_common.enums import CrisisLevel as L

pytestmark = pytest.mark.clinical
PHRASES = load_phrase_set()
REQUIRED_POSITIVE = {  # register regressions: text -> minimum level
    "I want to kill myself": L.CRITICAL, "I'm suicidal": L.HIGH,
    "I've been thinking about ending my life": L.HIGH, "I'm going to jump off the bridge tonight": L.CRITICAL,
    "I took all my pills": L.CRITICAL, "I have been saving up pills and I know how": L.HIGH,
    "I feel like giving up, I am a burden": L.HIGH, "I wrote a goodbye letter and gave away my things": L.HIGH,
    "It isn't fair. I'd be better off dead": L.CRITICAL, "I don't know. Suicide maybe": L.HIGH,
    "Things aren't getting better. Overdose seems easiest": L.HIGH,
    "everyone would be better off without me": L.HIGH, "I can't take this anymore, nobody would miss me": L.HIGH,
    "I can\u2019t go on": L.HIGH, "I don\u2019t want to be alive anymore": L.HIGH,
    "I\u2019m going to end it tonight": L.CRITICAL, "I want to kill my stepfather": L.HIGH,
    "my partner hits me and I'm scared to go home": L.HIGH, "he has a knife and he is outside my door": L.CRITICAL,
    "I want to die :-} bye": L.CRITICAL, "nobody would miss me }": L.HIGH,
    "I cut myself again last night": L.HIGH, "I feel hopeless": L.ELEVATED,
    "I'm so overwhelmed": L.LOW,  # general distress is hopelessness LOW; the WS-12.6 crisis-watch test relies on it
}
REQUIRED_NEGATIVE = {  # text -> maximum level
    "Hi, how are you?": L.NONE, "Thanks, that helps": L.NONE, "I am cutting vegetables for dinner": L.LOW,
    "I'm cutting back on coffee": L.LOW, "I could kill for a coffee": L.LOW,
    "I need to pick up my pills from the pharmacy": L.LOW, "We tied the boat with a rope": L.LOW,
    "My dad has a gun collection": L.LOW, "I bought a knife and some rope for camping by the bridge": L.LOW,
    "Long week: tired, stressed, worried, upset, frustrated and exhausted": L.ELEVATED,
    "We discussed harm reduction strategies": L.NONE, "I'm not suicidal, I just had a rough day": L.LOW,
    "This traffic is killing me": L.LOW,
}
ADVERSARIAL_POSITIVE = {  # review focus 1: disclosures outside plain wording -> minimum level (provisional, D-05)
    "i want to k1ll mys3lf": L.CRITICAL, "I want to kil myself": L.CRITICAL,          # leetspeak, misspelling
    "im sucidal": L.HIGH, "i feel suicidle lately": L.HIGH,                           # misspellings
    "honestly i just wanna unalive myself": L.CRITICAL, "i'm gonna kms tonight": L.HIGH,  # slang
    "I want to die \U0001F62D\U0001F62D": L.CRITICAL, "\U0001F494 i can't go on anymore \U0001F494": L.HIGH,  # emoji
    "me quiero matar": L.CRITICAL, "ya no quiero vivir": L.HIGH,                      # Spanish (US pilot)
    "honestly I just want to morir": L.HIGH,                                          # code-switched
    "what if I just ended it all": L.HIGH,                                            # hypothetical
    "she asked if I ever think 'I want to kill myself' and I said yes": L.CRITICAL,   # quoted speech
    "I don't know what to do and I've been thinking about ending my life": L.HIGH,     # earlier negation, long clause
    ("I'm not going to pretend I'm fine, work is awful and my family isn't talking to me and "
     "honestly I keep thinking I'd be better off dead"): L.CRITICAL,
}
ADVERSARIAL_NEGATIVE = {"I drove 20 kms to work today": L.LOW, "that joke killed me \U0001F602": L.LOW,
                        "no quiero ir a trabajar hoy": L.LOW}

def test_phrase_set_is_large_and_balanced():
    assert len(PHRASES.positive) + len(PHRASES.negative) >= 200 and len(PHRASES.negative) >= 60
    per_category = Counter(c for p in PHRASES.positive for c in p.categories)
    assert all(per_category[c] >= 8 for c in CATEGORIES)
    assert sum("\u2019" in p.text for p in PHRASES.positive) >= 20

def test_register_regressions_are_pinned_in_the_set():
    pos = {p.text: p.min_level for p in PHRASES.positive}
    neg = {n.text: n.max_level for n in PHRASES.negative}
    assert all(level_rank(pos[t]) >= level_rank(lvl) for t, lvl in REQUIRED_POSITIVE.items())
    assert all(level_rank(neg[t]) <= level_rank(lvl) for t, lvl in REQUIRED_NEGATIVE.items())

@pytest.mark.parametrize("phrase", PHRASES.positive, ids=lambda p: p.text[:40])
def test_positive_phrase_reaches_min_level_and_categories(phrase):
    result = screen_text(phrase.text)
    assert level_rank(result.level) >= level_rank(phrase.min_level)
    assert phrase.categories <= result.categories

@pytest.mark.parametrize("phrase", PHRASES.negative, ids=lambda p: p.text[:40])
def test_negative_phrase_stays_at_or_below_max_level(phrase):
    assert level_rank(screen_text(phrase.text).level) <= level_rank(phrase.max_level)

def test_negation_is_clause_bounded_and_never_hits_intent():
    assert screen_text("It isn't fair. I'd be better off dead").level == L.CRITICAL
    assert screen_text("I'm not going to lie, I want to kill myself").level == L.CRITICAL
    negated = screen_text("I'm not suicidal")
    assert level_rank(negated.level) <= level_rank(L.LOW) and negated.negated_rule_ids

@pytest.mark.parametrize("text,floor", ADVERSARIAL_POSITIVE.items(), ids=lambda v: str(v)[:40])
def test_review_focus_1_adversarial_disclosures_reach_their_floor(text, floor):
    assert level_rank(screen_text(text).level) >= level_rank(floor)

@pytest.mark.parametrize("text,ceiling", ADVERSARIAL_NEGATIVE.items(), ids=lambda v: str(v)[:40])
def test_review_focus_1_adversarial_benign_text_stays_low(text, ceiling):
    assert level_rank(screen_text(text).level) <= level_rank(ceiling)

def test_review_focus_1_adversarial_cases_are_in_the_shared_phrase_set():
    pos = {p.text: p.min_level for p in PHRASES.positive}
    neg = {n.text: n.max_level for n in PHRASES.negative}
    assert all(t in pos and level_rank(pos[t]) >= level_rank(lvl) for t, lvl in ADVERSARIAL_POSITIVE.items())
    assert all(t in neg and level_rank(neg[t]) <= level_rank(lvl) for t, lvl in ADVERSARIAL_NEGATIVE.items())

def test_rule_ids_are_codes_and_metadata_is_present():
    result = screen_text("I want to kill myself tonight")
    assert result.rule_ids and all(" " not in rid for rid in result.rule_ids)
    meta = load_lexicon().metadata
    assert meta["version"] == LEXICON_VERSION == result.lexicon_version and meta["jurisdiction"] == "US"
    assert {"approved_by", "approved_at"} <= meta.keys()
```
```python
# tests/clinical/test_screen_text_properties.py (hypothesis + budget)
#  - @given(st.text()) screen_text never raises; level in CrisisLevel; categories <= CATEGORIES;
#    rule_ids sorted and unique; screen_text(x) == screen_text(normalize_text(x)) on level and categories.
#  - for every positive phrase p: level_rank(screen_text(p.text + ". We had pasta for dinner.").level)
#    >= level_rank(screen_text(p.text).level)   # a later clause can never negate an earlier disclosure
#  - budget: text = ("I had a long day at work and the bus was late again. " * 200)[:10_000];
#    median of 20 runs of screen_text(text) after 3 warm-ups is < 0.005 s (time.perf_counter)
# tests/clinical/test_lexicon_legacy_coverage.py
#  - load tests/clinical/legacy_terms_v0.yaml: entries {term, legacy_tier, floor, example}
#  - every entry's example screens at >= its floor AND appears in crisis_phrases_v1.yaml positives
#  - the snapshot lists every CRITICAL/HIGH term of services/safety_service/config/keywords.json, every
#    critical/high keyword of crisis_detector.KeywordSet and every CRITICAL/HIGH patterns.json explanation
#    (assert the counts: len(entries) >= 40)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/clinical -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'screen_text' from 'solace_clinical'` and `ModuleNotFoundError: No module named 'solace_clinical.phrases'`. The behaviours they pin are broken today: safety's Layer-1 negation suppresses "I'd be better off dead" across a sentence boundary (`crisis_detector.py:193-197, 279-293`, MVP-084) and bare "cutting", "pills", "rope", "gun", "knife", "bridge" score HIGH or CRITICAL (`keywords.json:27,42-48`, MVP-182). No legacy list has a Spanish, slang ("unalive", "kms") or misspelled form (`config/keywords.json` and `config/patterns.json` contain none of `matar`, `morir`, `unalive`, `kms`), which the review-focus-1 cases pin.
- [ ] **Step 3: Implement** `screen_text(text: str) -> ScreenResult` in `screen.py` and `load_lexicon() -> Lexicon` (cached; `import yaml` inside the loader) in `lexicon.py`, then author `crisis_lexicon_v1.yaml`, `crisis_phrases_v1.yaml` and `legacy_terms_v0.yaml`, then write §11 of `docs/CLINICAL-VALIDATION.md`. Algorithm: normalize; replace every symbol character (Unicode categories So and Sk, which covers emoji) with a space; screen two variants and keep the maximum: the text itself and a de-leeted copy in which `0 1 3 4 5 7 @ $` become `o i e a s t a s` inside tokens that also contain a letter (so "20 kms" is unchanged); split each variant into clauses on `[.!?;:\n]`, commas, ` - ` and the words `but`, `however`, `though`, `although`, `yet`; for each rule whose literal anchor (the longest literal fragment of a pattern, precomputed at load) occurs in the variant, run its compiled patterns; a match counts unless `negation_policy == "clause"` and a negation cue (`not`, `no`, `never`, `don't`, `doesn't`, `didn't`, `isn't`, `aren't`, `wasn't`, `won't`, `wouldn't`, `no longer`, `denies`, `without`) sits within the three tokens immediately before the match in the same clause (an earlier, unrelated negation in a long clause never suppresses a later disclosure; hypothetical frames such as `what if` and `if i` are not cues); a rule with `requires_first_person: true` counts only when its clause contains `i`, `i'm`, `i've`, `i'd`, `i'll`, `me`, `my`, `myself` or `mine`. Level is the maximum matched rule level; categories, rule ids and negated rule ids are sorted and deduplicated. Authoring rules: every CRITICAL rule and every first-person intent, plan or means rule uses `negation_policy: never`; context-free words (cutting, pills, rope, bridge, gun, knife, overdose) appear only inside self-directed patterns (for example `cut(ting)? (myself|my (wrists?|arms?|legs?|thighs?|skin))`, `(saving|saved|stockpil\w*|hoard\w*) (up )?(my )?(pills|meds|tablets)`, `(jump|leap)\w* (off|from) (the |a )?bridge`); `hopelessness` rules are LOW or ELEVATED only, and general distress phrases (overwhelmed, can't cope, falling apart, breaking down, losing control) are `hopelessness` LOW; perceived burdensomeness and "better off without me" are `suicidal_ideation` HIGH; preparatory acts (goodbye letter, giving things away, will) are `suicide_plan` HIGH; the core intent rules tolerate common misspellings (`su[iy]?c[iy]?d(e|al|le)`, `kil+ ?my ?sel(f|ves)`) and slang (`unalive (my ?self|me)`, first-person `kms`, `end it all`); a provisional Spanish rule set (`me quiero matar`, `matarme`, `suicidarme`, `quiero morir(me)?`, `(ya )?no quiero vivir`) uses `requires_first_person: false` because the verb carries the person, and an English frame plus a Spanish verb (`want to morir`) matches the same rules; every adversarial case above is authored into `crisis_phrases_v1.yaml`. Metadata: `version: "1.0.0"`, `approved_by: "PROVISIONAL - pending D-05 clinical sign-off"`, `approved_at: null`, `jurisdiction: US`. §11 records: lexicon and phrase-set versions and counts, the category-to-level table, negation and self-reference policy, the max-only fusion rule, the 0.85 CRITICAL threshold used to map LLM scores, the 30-minute crisis watch (D-06), the one-band context and trajectory raises (WS-12.6), protective factors informational only (WS-12.7), the IF-09 episode semantics (900 s coalescing, 900 s re-page up to 3 times, one update after acknowledgement), the D-29 resource registry (en-US v1.0.0 with today's three entries, the generic fallback `fallback-1.0.0` with its provisional copy and directory URL, both in the sign-off table, and the add-a-locale rule: a country is added only as a new registry entry with clinician-verified resources and its own version before its users join), the deliberate change that hopelessness alone shows resources without paging, the known over-detection that a negated first-person intent still pages, the provisional misspelling, slang, emoji, leetspeak and Spanish coverage (other languages rely on the raise-only LLM layer when D-04 allows it), and an empty sign-off table (approver, licence, date, lexicon version, phrase-set version).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the `tests/clinical` command above, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/test_syntax_and_imports.py -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.4: Slot 014 escalation decision columns (one auto decision per turn)
**Covers:** MVP-081, MVP-088, MVP-177, MVP-181
**Severity:** high
**Wave:** 1
**Depends on:** none
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-01 (default assumed: Alembic is the schema of record; owners edit only `upgrade()`/`downgrade()` of their slot)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/014_safety.py` (`upgrade()`/`downgrade()` only; revision ids untouched)
- Test: `tests/realinfra/safety-detection-lexicon/test_mvp_088_slot014_escalations.py`
**Interfaces:**
- Consumes: IF-03 slot `014_safety` on the linear chain; IF-19 `pg_migrated -> PgHandle(dsn_app, dsn_migrator, dsn_superuser, app_pool)`.
- Produces: `escalations` gains `correlation_id UUID NULL`, `source VARCHAR(16) NOT NULL DEFAULT 'auto'` with `CHECK (source IN ('auto','manual'))`, `categories TEXT[] NOT NULL DEFAULT '{}'`, `detection_layer SMALLINT NULL`, `encryption_key_id VARCHAR(64) NULL`, `pseudonymized_at TIMESTAMPTZ NULL`; index `ix_escalations_correlation_id`; partial unique index `ux_escalations_auto_correlation ON escalations (correlation_id) WHERE source = 'auto' AND correlation_id IS NOT NULL`. `escalations` stays non-RLS (IF-03).

- [ ] **Step 1: Write the failing test** in `tests/realinfra/safety-detection-lexicon/test_mvp_088_slot014_escalations.py`
```python
import uuid
from datetime import UTC, datetime
import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
INSERT = ("INSERT INTO escalations (escalation_id, user_id, crisis_level, priority, status, reason, created_at, "
          "correlation_id, source, categories) VALUES ($1,$2,'CRITICAL','CRITICAL','PENDING','auto:CRITICAL',$3,$4,$5,$6)")

async def test_mvp_088_slot014_adds_decision_columns(pg_migrated):
    async with pg_migrated.app_pool.acquire() as conn:
        rows = await conn.fetch("SELECT column_name, data_type FROM information_schema.columns "
                                "WHERE table_name = 'escalations'")
    cols = {r["column_name"]: r["data_type"] for r in rows}
    assert {"correlation_id", "source", "categories", "detection_layer", "encryption_key_id",
            "pseudonymized_at"} <= cols.keys()
    assert cols["categories"] == "ARRAY" and cols["correlation_id"] == "uuid"

async def test_mvp_177_one_auto_decision_per_turn(pg_migrated):
    turn, user = uuid.uuid4(), uuid.uuid4()
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute(INSERT, uuid.uuid4(), user, datetime.now(UTC), turn, "auto", ["suicidal_ideation"])
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(INSERT, uuid.uuid4(), user, datetime.now(UTC), turn, "auto", [])
        await conn.execute(INSERT, uuid.uuid4(), user, datetime.now(UTC), turn, "manual", [])  # allowed
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(INSERT, uuid.uuid4(), user, datetime.now(UTC), uuid.uuid4(), "robot", [])
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:SOLACE_REQUIRE_REALINFRA="1"; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/safety-detection-lexicon/test_mvp_088_slot014_escalations.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The column set assertion fails (slot 014 is the empty WS-02 scaffold; migration 003 has no correlation id, so a retried /check produces a second decision with nothing to deduplicate on, MVP-087/177) and the insert fails with `UndefinedColumnError: column "correlation_id" ...`.
- [ ] **Step 3: Implement** `upgrade()` with `op.add_column` for the six columns, the CHECK constraint `ck_escalations_source`, `op.create_index("ix_escalations_correlation_id", ...)` and `op.create_index("ux_escalations_auto_correlation", "escalations", ["correlation_id"], unique=True, postgresql_where=sa.text("source = 'auto' AND correlation_id IS NOT NULL"))`; `downgrade()` drops the indexes, the constraint and the columns in reverse order.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:SOLACE_REQUIRE_REALINFRA="1"; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations/test_chain_real_pg.py -o "addopts=" -p no:logging -q` (upgrade head, downgrade base, upgrade head; WS-02 reviews the slot diff), then the safety suite command from WS-12.1. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.5: Safety LLM layer through the shared client: verbatim text, bounded, failures unreliable
**Covers:** MVP-030, MVP-183
**Severity:** blocker
**Wave:** 1
**Depends on:** none
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-04 (default assumed: in strict envs `LLM_PHI_EGRESS_APPROVED=false` leaves the client unavailable and the lexicon decides; no prompt caching) and D-05
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/ml/llm_assessor.py` (remove LangChain, ChatAnthropic, the legacy anthropic client, the result cache, `_create_fallback_assessment` and the rule-based keyword stub; add `RiskAssessment.reliable`; bound every call)
- Modify: `services/safety_service/tests/test_llm_assessor.py` (drop tests that assert the keyword stub, e.g. `test_fallback_assessment_crisis`; keep model and parsing tests)
- Test: `services/safety_service/tests/test_mvp_030_183_llm_assessor.py`
**Interfaces:**
- Consumes: IF-15 `UnifiedLLMClient.generate(system_prompt, user_message, *, service_name, task_type, max_tokens, ...) -> str` raising `LLMUnavailableError` (from `services.shared.infrastructure.llm_client`), `is_available` (WS-09, W1 phase A1).
- Produces: `LLMAssessor(config: LLMAssessorConfig | None = None, unified_llm_client: Any | None = None)`; `async assess(text: str, user_id: UUID | None = None, context: dict | None = None) -> RiskAssessment` which never raises and never sends `context`; `RiskAssessment.reliable: bool = True`; `LLMAssessorConfig.timeout_seconds: float = 1.5` (env `LLM_ASSESSOR_TIMEOUT_SECONDS`, request R-12-6) and `min_reliable_confidence: Decimal = Decimal("0.5")`.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_030_183_llm_assessor.py`
```python
import asyncio, json, time
from decimal import Decimal
import pytest
import services.safety_service.src.ml.llm_assessor as llm_mod
from services.safety_service.src.ml.llm_assessor import LLMAssessor, LLMAssessorConfig, RiskLevel

REPLY = json.dumps({"risk_level": "CRITICAL", "risk_score": 0.95, "confidence": 0.9,
                    "risk_factors": [{"dimension": "SUICIDAL_IDEATION", "severity": 0.95, "evidence": "x", "rationale": "y"}],
                    "protective_factors": [], "clinical_summary": "s", "recommended_actions": [], "immediate_risk": True})

class _FakeLLMClient:
    is_available = True
    def __init__(self, reply=None, exc=None, delay=0.0):
        self.reply, self.exc, self.delay, self.calls = reply, exc, delay, []
    async def generate(self, system_prompt, user_message, **kwargs):
        self.calls.append((system_prompt, user_message, kwargs))
        if self.delay: await asyncio.sleep(self.delay)
        if self.exc: raise self.exc
        return self.reply

class _StubChatModel:  # stands in for ChatAnthropic so the legacy path never reaches the network
    def __init__(self, **kwargs): pass

async def test_mvp_183_braces_never_fall_back_to_keyword_score(monkeypatch):
    monkeypatch.setattr(llm_mod, "LANGCHAIN_AVAILABLE", True, raising=False)
    monkeypatch.setattr(llm_mod, "ChatAnthropic", _StubChatModel, raising=False)
    result = await LLMAssessor().assess("nobody would miss me }")
    assert result.reliable is False and result.risk_score == Decimal("0")
    assert "Rule-based" not in result.clinical_summary

async def test_mvp_030_provider_failure_is_unreliable_not_a_stub_score(monkeypatch):
    monkeypatch.setattr(llm_mod, "LANGCHAIN_AVAILABLE", False, raising=False)
    client = _FakeLLMClient(exc=RuntimeError("provider down"))
    result = await LLMAssessor(unified_llm_client=client).assess("I have been saving up pills and I know how")
    assert result.reliable is False and result.risk_level == RiskLevel.NONE and result.risk_score == Decimal("0")

async def test_text_reaches_the_model_verbatim_without_context(monkeypatch):
    monkeypatch.setattr(llm_mod, "LANGCHAIN_AVAILABLE", False, raising=False)
    client = _FakeLLMClient(reply=REPLY)
    result = await LLMAssessor(unified_llm_client=client).assess(
        "I want to die :-} bye", context={"active_technique": "exposure_therapy"})
    assert result.reliable is True and result.risk_level == RiskLevel.CRITICAL
    _, user_message, kwargs = client.calls[0]
    assert user_message == "I want to die :-} bye"
    assert kwargs["task_type"] == "crisis" and kwargs["service_name"] == "safety-service"

async def test_every_call_is_bounded_by_the_timeout(monkeypatch):
    monkeypatch.setattr(llm_mod, "LANGCHAIN_AVAILABLE", False, raising=False)
    assessor = LLMAssessor(LLMAssessorConfig(timeout_seconds=0.2), unified_llm_client=_FakeLLMClient(reply=REPLY, delay=5))
    started = time.perf_counter()
    result = await assessor.assess("I feel awful")
    assert time.perf_counter() - started < 1.0 and result.reliable is False

@pytest.mark.parametrize("reply,level,reliable", [
    ("not json", RiskLevel.NONE, False),
    (json.dumps({**json.loads(REPLY), "risk_level": "HIGH", "risk_score": 0.75, "confidence": 0.3}), RiskLevel.HIGH, False),
])
async def test_unparseable_or_low_confidence_replies_are_unreliable(reply, level, reliable, monkeypatch):
    monkeypatch.setattr(llm_mod, "LANGCHAIN_AVAILABLE", False, raising=False)
    result = await LLMAssessor(unified_llm_client=_FakeLLMClient(reply=reply)).assess("x")
    assert (result.risk_level, result.reliable) == (level, reliable)

async def test_no_client_means_no_signal_and_no_caching(monkeypatch):
    monkeypatch.setattr(llm_mod, "LANGCHAIN_AVAILABLE", False, raising=False)
    assert (await LLMAssessor().assess("x")).reliable is False
    client = _FakeLLMClient(reply=REPLY)
    assessor = LLMAssessor(unified_llm_client=client)
    await assessor.assess("same"); await assessor.assess("same")
    assert len(client.calls) == 2

async def test_review_focus_4_failure_logs_carry_no_message_or_exception_text(monkeypatch):
    import structlog
    monkeypatch.setattr(llm_mod, "LANGCHAIN_AVAILABLE", False, raising=False)
    client = _FakeLLMClient(exc=RuntimeError("provider echoed: saving up pills"))
    with structlog.testing.capture_logs() as logs:
        await LLMAssessor(unified_llm_client=client).assess("I have been saving up pills and I know how")
    assert logs and all("pills" not in repr(entry) for entry in logs)
    assert any(entry.get("reason") == "RuntimeError" for entry in logs)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_030_183_llm_assessor.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The brace test reaches `ChatPromptTemplate.from_messages` (`llm_assessor.py:300-309`), which raises on the lone `}` and falls through to the keyword stub (`:350-386`), returning LOW 0.2 with `clinical_summary` "Rule-based assessment ..." and no `reliable` attribute (`AttributeError`); the provider-failure test gets the same stub; the verbatim test sees "Analyze the following text ... Additional context ..." as `user_message`; the timeout test waits 5 s; the caching test sees one call; the log test finds no entry with `reason == "RuntimeError"` (today's handlers log `error=str(e)`, so the exception text reaches the log).
- [ ] **Step 3: Implement** `async LLMAssessor.assess(text, user_id=None, context=None) -> RiskAssessment` in `llm_assessor.py`: empty text or no available client returns `RiskAssessment(risk_level=NONE, risk_score=0, confidence=0, reliable=False, clinical_summary="llm_unavailable")`; otherwise `await asyncio.wait_for(client.generate(system_prompt=<static prompt>, user_message=text, service_name="safety-service", task_type="crisis", max_tokens=config.max_tokens), timeout=config.timeout_seconds)`; strip one optional fenced code block, `json.loads`, `_parse_llm_response`; `reliable = confidence >= config.min_reliable_confidence`. Any `Exception` (IF-15 `LLMUnavailableError`, `asyncio.TimeoutError`, JSON, key, value and type errors, or a provider bug) returns the unreliable NONE result and logs `llm_assessment_unavailable` with `reason=<exception class name>` only; `asyncio.CancelledError` propagates. The static system prompt uses literal single braces (the `{{`/`}}` escaping existed only for `ChatPromptTemplate`). `context` is accepted for signature compatibility and never sent (D-04 minimum necessary).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.6: CrisisDetector on the shared lexicon with max-only fusion, one 0.85 config and raise-only history
**Covers:** MVP-030, MVP-083, MVP-084, MVP-086, MVP-090, MVP-178, MVP-182
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-12.3, WS-12.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: thresholds LOW 0.3 / ELEVATED 0.5 / HIGH 0.7 / CRITICAL 0.85 used only to map LLM scores; max-only fusion; one-band raises for context and trajectory) and D-06 (default assumed: 30-minute crisis watch)
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/domain/crisis_detector.py` (delete `KeywordSet`, `Layer1InputGate` and the ML keyword/sentiment/pattern imports and auto-construction; `CrisisDetector(settings=None, llm_assessor=None)`; new `detect`; keep `Layer2ProcessingGuard`, `Layer3OutputFilter`, `Layer4ContinuousMonitor`, `_crisis_level_from_score`)
- Modify: `services/safety_service/src/domain/factory.py` (`build_crisis_detector(*, llm_assessor=None, settings=None)` builds `CrisisDetectorSettings()` directly)
- Modify: `services/safety_service/src/config.py` (delete `CrisisDetectionConfig` and `SafetyConfig.crisis_detection`, the stale `critical_threshold=0.9`)
- Modify: `services/safety_service/src/main.py` (shared LLM client; `_wire_llm(runtime, llm_client) -> LLMAssessor | None` registers the non-critical `llm` readiness check and returns the assessor; shutdown)
- Modify: `services/safety_service/tests/test_crisis_detector.py` (remove `TestLayer1InputGate` and weight-based `TestCrisisDetector` cases; keep layer 2/3/4, risk-factor and settings tests), `services/safety_service/tests/test_rev25_reconcile.py` (threshold tests through `build_crisis_detector`; drop the Layer-1 latency test, replaced by the tests/clinical budget), `services/safety_service/tests/test_config.py` (drop the 0.9 assertions); delete `services/safety_service/tests/test_crisis_detector_fusion.py` (weighted fusion no longer exists)
- Create: `services/safety_service/tests/test_mvp_083_084_086_178_182_detector_fusion.py`, `services/safety_service/tests/test_mvp_090_history_signal.py`, `services/safety_service/tests/test_phrase_set_safety_detector.py` (tests)
**Interfaces:**
- Consumes: WS-12.2/12.3 `screen_text`, `RiskSignal`, `fuse_risk`, `level_rank`, `CATEGORIES`, `check_detector_async`; WS-12.5 `LLMAssessor`, `RiskAssessment.reliable`; IF-15 `UnifiedLLMClient(LLMClientSettings())`, `await initialize()` (non-raising), `is_available`, `async readiness() -> tuple[bool, str]` (WS-09.4), `shutdown()`; IF-01 `runtime.register_readiness_check(name, check, *, critical=True)`.
- Produces (CR-7 of WS-09, for WS-09.4): `main._wire_llm(runtime, llm_client) -> LLMAssessor | None` calls `runtime.register_readiness_check("llm", llm_client.readiness, critical=False)` and returns `LLMAssessor(unified_llm_client=llm_client)` only when `llm_client.is_available`, else None (the lexicon decides). The assessor already turns `LLMUnavailableError` into an unreliable result (WS-12.5).
- Produces: `CrisisDetector.detect(content, context=None, conversation_history=None, user_risk_history=None, user_id=None) -> DetectionResult` where `DetectionResult` adds `categories: list[str]`, `rule_ids: list[str]`, `lexicon_version: str`, `unreliable_sources: list[str]`; `trigger_indicators` holds category codes only; `risk_score = LEVEL_SCORE[level]` from the settings thresholds (NONE 0, LOW 0.3, ELEVATED 0.5, HIGH 0.7, CRITICAL 0.85); `CrisisDetectorSettings.crisis_watch_seconds: int = 1800` (env `CRISIS_WATCH_SECONDS`, request R-12-6). `user_risk_history` keys read: `last_crisis_level`, `last_crisis_time` (ISO-8601).

- [ ] **Step 1: Write the failing test** in the three test files
```python
# services/safety_service/tests/test_mvp_083_084_086_178_182_detector_fusion.py
import json
from decimal import Decimal
import pytest
from services.safety_service.src.domain.crisis_detector import CrisisDetectorSettings, _crisis_level_from_score
from services.safety_service.src.domain.factory import build_crisis_detector
from services.safety_service.src.ml.llm_assessor import LLMAssessor
from solace_clinical import CATEGORIES, level_rank
from solace_common.enums import CrisisLevel as L

@pytest.fixture(autouse=True)
def container_equivalent_ml(monkeypatch):
    """The image is lexicon-only (D-16): no flashtext, VADER, transformers or LangChain."""
    import services.safety_service.src.ml.keyword_detector as kd
    import services.safety_service.src.ml.sentiment_analyzer as sa
    import services.safety_service.src.ml.llm_assessor as la
    monkeypatch.setattr(kd, "FLASHTEXT_AVAILABLE", False, raising=False)
    monkeypatch.setattr(sa, "VADER_AVAILABLE", False, raising=False)
    monkeypatch.setattr(sa, "TRANSFORMERS_AVAILABLE", False, raising=False)
    monkeypatch.setattr(la, "LANGCHAIN_AVAILABLE", False, raising=False)

class _Client:  # minimal IF-15 stand-in
    is_available = True
    def __init__(self, reply=None, exc=None): self.reply, self.exc = reply, exc
    async def generate(self, system_prompt, user_message, **kw):
        if self.exc: raise self.exc
        return self.reply

def test_mvp_083_composed_detector_uses_critical_threshold_085():
    assert build_crisis_detector()._settings.critical_threshold == Decimal("0.85")

async def test_mvp_083_llm_score_087_is_critical():
    reply = json.dumps({"risk_level": "HIGH", "risk_score": 0.87, "confidence": 0.9, "risk_factors": [],
                        "protective_factors": [], "clinical_summary": "s", "recommended_actions": [], "immediate_risk": True})
    detector = build_crisis_detector(llm_assessor=LLMAssessor(unified_llm_client=_Client(reply=reply)))
    assert (await detector.detect("I feel like giving up, I am a burden")).crisis_level == L.CRITICAL

@pytest.mark.parametrize("text,floor", [("I have been saving up pills and I know how", L.HIGH),
                                        ("I feel like giving up, I am a burden", L.HIGH),
                                        ("I wrote a goodbye letter and gave away my things", L.ELEVATED)])
async def test_mvp_030_failing_llm_never_lowers_level(text, floor):
    detector = build_crisis_detector(llm_assessor=LLMAssessor(unified_llm_client=_Client(exc=RuntimeError("down"))))
    assert level_rank((await detector.detect(text)).crisis_level) >= level_rank(floor)

@pytest.mark.parametrize("text,floor", [("It isn't fair. I'd be better off dead", L.CRITICAL),
                                        ("I don't know. Suicide maybe", L.HIGH),
                                        ("Things aren't getting better. Overdose seems easiest", L.HIGH)])
async def test_mvp_084_negation_is_clause_bounded(text, floor):
    assert level_rank((await build_crisis_detector().detect(text)).crisis_level) >= level_rank(floor)

@pytest.mark.parametrize("text", ["Hi, how are you?", "Thanks, that helps"])
async def test_mvp_086_neutral_text_scores_none(text):
    result = await build_crisis_detector().detect(text)
    assert result.crisis_level == L.NONE and result.risk_score == Decimal("0")

@pytest.mark.parametrize("text", ["I am cutting vegetables for dinner", "I'm cutting back on coffee",
                                  "I need to pick up my pills from the pharmacy", "We tied the boat with a rope",
                                  "My dad has a gun collection"])
async def test_mvp_182_context_free_words_need_self_reference(text):
    assert level_rank((await build_crisis_detector().detect(text)).crisis_level) <= level_rank(L.LOW)

async def test_mvp_178_keyword_sums_never_escalate():
    detector = build_crisis_detector()
    assert level_rank((await detector.detect(
        "Long week: tired, stressed, worried, upset, frustrated and exhausted")).crisis_level) <= level_rank(L.ELEVATED)
    assert level_rank((await detector.detect(
        "I bought a knife and some rope for camping by the bridge")).crisis_level) <= level_rank(L.LOW)

@pytest.mark.parametrize("text", ["I want to kill myself tonight", "I feel hopeless", "Thanks, that helps"])
async def test_score_is_consistent_with_level_and_indicators_are_codes(text):
    result = await build_crisis_detector().detect(text)
    assert _crisis_level_from_score(result.risk_score, CrisisDetectorSettings()) == result.crisis_level
    assert set(result.trigger_indicators) <= CATEGORIES

async def test_review_focus_1_risk_building_across_turns_raises_one_band():
    history = ["Things keep getting worse", "I feel so alone", "I can't do this, it's terrible"]
    fresh = await build_crisis_detector().detect("I feel hopeless")
    built = await build_crisis_detector().detect("I feel hopeless", conversation_history=history)
    assert fresh.crisis_level == L.ELEVATED and built.crisis_level == L.HIGH
    assert built.risk_score == Decimal("0.7") and "hopelessness" in built.categories

class _Runtime:
    def __init__(self): self.checks = []
    def register_readiness_check(self, name, check, *, critical=True): self.checks.append((name, check, critical))

class _Llm:
    def __init__(self, available): self.is_available = available
    async def readiness(self): return (self.is_available, "ok" if self.is_available else "disabled")

@pytest.mark.parametrize("available", [True, False])
def test_cr7_llm_readiness_is_registered_non_critical(available):
    import services.safety_service.src.main as main
    runtime, client = _Runtime(), _Llm(available)
    assessor = main._wire_llm(runtime, client)
    assert runtime.checks == [("llm", client.readiness, False)]
    assert isinstance(assessor, LLMAssessor) if available else assessor is None
```
```python
# services/safety_service/tests/test_mvp_090_history_signal.py  (reuses the container_equivalent_ml fixture body)
#  hist(minutes) = {"previous_crisis_events": 1, "recent_escalation": True, "last_crisis_level": "CRITICAL",
#                   "last_crisis_time": (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()}
#  test_mvp_090_recent_crisis_never_dilutes: text "everyone would be better off without me";
#      after = detect(text, user_risk_history=hist(5)); fresh = detect(text)
#      assert level_rank(after.crisis_level) >= level_rank(fresh.crisis_level) >= level_rank(L.HIGH)
#  test_mvp_090_watch_raises_low_to_elevated: "I'm so overwhelmed" is LOW fresh and ELEVATED with hist(5)
#  test_mvp_090_history_expires_after_the_watch_window: with hist(120) the level equals the fresh level
# services/safety_service/tests/test_phrase_set_safety_detector.py
#  pytestmark = pytest.mark.clinical; detector = build_crisis_detector(llm_assessor=None)
#  async def detect(text): r = await detector.detect(text); return r.crisis_level, set(r.trigger_indicators)
#  assert await check_detector_async(detect) == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_083_084_086_178_182_detector_fusion.py services/safety_service/tests/test_mvp_090_history_signal.py services/safety_service/tests/test_phrase_set_safety_detector.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The composed detector reports `critical_threshold == Decimal("0.9")` (main.py's composition via `config.py:57`, MVP-083) and maps 0.87 to HIGH; the failing-LLM cases average to LOW/ELEVATED (`crisis_detector.py:799-804`, MVP-030); "It isn't fair. I'd be better off dead" scores NONE (MVP-084); "Hi, how are you?" scores ELEVATED 0.5 (sentiment base risk, MVP-086); "I am cutting vegetables for dinner" scores HIGH or CRITICAL (MVP-182); the six distress words floor to CRITICAL (MVP-178); the post-crisis "better off without me" scores below the fresh result (MVP-090); the phrase set reports many failures; the multi-turn test fails with `AttributeError: 'DetectionResult' object has no attribute 'categories'` (and today's trajectory only appends `TRAJECTORY:deteriorating` to a weighted score); `main._wire_llm` does not exist (`AttributeError`), because main.py builds no shared LLM client and registers no `llm` readiness check.
- [ ] **Step 3: Implement** `async CrisisDetector.detect(...) -> DetectionResult` in `crisis_detector.py`: `screen = screen_text(content)`; signals start with `RiskSignal("lexicon", screen.level, 1.0, True)`; when an assessor is injected and `screen.level != CRITICAL`, `assessment = await assessor.assess(content, user_id=user_id)` adds `RiskSignal("llm", max(CrisisLevel(assessment.risk_level), _crisis_level_from_score(assessment.risk_score, settings)), confidence, assessment.reliable)` and maps the dimensions SUICIDAL_IDEATION, SELF_HARM, HOPELESSNESS to category codes; a `user_risk_history` whose `last_crisis_level` is HIGH or CRITICAL and whose `last_crisis_time` is within `crisis_watch_seconds` adds `RiskSignal("history", ELEVATED, 0.5, False)` when `screen.level >= LOW`; `level0 = fuse_risk(signals)`; a contraindicated `context["active_technique"]` for `level0` (Layer 2 table) adds a one-band raise capped at CRITICAL; a deteriorating Layer-4 trajectory over `conversation_history` adds a one-band raise capped at HIGH; `level = fuse_risk(signals)`; `risk_score = LEVEL_SCORE[level]`; `detection_layer` is 1 for lexicon/history, 2 for context, 4 for llm/trajectory, taken from the strongest signal; unreliable sources are listed. Remove `keyword_weight`, `sentiment_weight`, `pattern_weight`, `history_weight` from `CrisisDetectorSettings`; add `crisis_watch_seconds`. In `main.py`: `llm_client = UnifiedLLMClient(LLMClientSettings()); await llm_client.initialize()`; `assessor = _wire_llm(runtime, llm_client)` (registers `("llm", llm_client.readiness, critical=False)`, returns the assessor only when `llm_client.is_available`); `build_crisis_detector(llm_assessor=assessor)`; `await llm_client.shutdown()` at shutdown. The multi-turn raise uses the existing `Layer4ContinuousMonitor.analyze_trajectory(conversation_history, level0)`; `"deteriorating"` adds one band, capped at HIGH.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1, the `tests/clinical` command from WS-12.2, then `tests/integration/test_crisis_flow.py tests/integration/test_safety_service_e2e.py` in one process. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.7: Protective factors are evidence only and never lower the published score
**Covers:** MVP-176
**Severity:** medium
**Wave:** 1
**Depends on:** WS-12.6
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: protective factors are reported to clinicians but never change level or score)
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/domain/service.py` (`check_safety`: delete the H-06 adjustment block at lines 171-179; `_identify_protective_factors` uses normalized text, word boundaries and same-clause negation)
- Modify: `services/safety_service/tests/test_service.py` (`TestProtectiveFactorRiskAdjustment`: the "reduce moderate risk" expectation becomes "never reduces"), `docs/CLINICAL-VALIDATION.md` (§3.3 rewritten to the new rule; §11 cross-reference)
- Test: `services/safety_service/tests/test_mvp_176_protective_factors.py`
**Interfaces:**
- Consumes: WS-12.6 `build_crisis_detector`, `_crisis_level_from_score`; `solace_clinical.normalize_text`.
- Produces: `SafetyCheckResult.risk_score` always equals the detector's level-derived score; `protective_factors: list[{factor_type, strength, description}]` with factor types `social_support`, `family_connection`, `social_connection`, `positive_outlook`, `treatment_engagement`, `treatment_adherence`, `coping_skills`, `active_treatment`, `care_continuity`.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_176_protective_factors.py`
```python
from decimal import Decimal
from uuid import uuid4
from services.safety_service.src.domain.crisis_detector import CrisisDetectorSettings, _crisis_level_from_score
from services.safety_service.src.domain.escalation import EscalationManager, EscalationSettings
from services.safety_service.src.domain.factory import build_crisis_detector
from services.safety_service.src.domain.service import SafetyService, SafetyServiceSettings
from solace_clinical import level_rank
from solace_common.enums import CrisisLevel as L

def _service():
    return SafetyService(SafetyServiceSettings(), build_crisis_detector(), EscalationManager(EscalationSettings()))

async def test_mvp_176_crisis_words_are_not_protective_and_score_matches_level():
    result = await _service().check_safety(uuid4(), uuid4(),
        "I feel hopeless and worthless, I have no support and I want to hurt myself", "pre_check")
    assert level_rank(result.crisis_level) >= level_rank(L.HIGH)
    assert result.risk_score >= Decimal("0.7")
    assert _crisis_level_from_score(result.risk_score, CrisisDetectorSettings()) == result.crisis_level
    assert result.protective_factors == []

async def test_real_protective_factors_are_reported_without_changing_score():
    service = _service()
    plain = await service.check_safety(uuid4(), None, "I want to hurt myself", "pre_check")
    supported = await service.check_safety(uuid4(), None,
        "I want to hurt myself but my family supports me and I see my therapist", "pre_check")
    assert {f["factor_type"] for f in supported.protective_factors} >= {"family_connection", "treatment_engagement"}
    assert supported.risk_score == plain.risk_score and supported.crisis_level == plain.crisis_level
    burden = await service.check_safety(uuid4(), None, "my family would be better off without me", "pre_check")
    assert "family_connection" not in {f["factor_type"] for f in burden.protective_factors}
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_176_protective_factors.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the published score is lowered after the level is set (`service.py:171-179`) so the HIGH result carries a score in the ELEVATED band, and substring matching reports `social_support` and `positive_outlook` from "no support" and "hopeless" (`service.py:400-418`); the family sentence reports `family_connection`.
- [ ] **Step 3: Implement** `_identify_protective_factors(content: str, context: dict) -> list[dict]` in `service.py`: normalize with `normalize_text`, split clauses as `screen_text` does, match each keyword with `\b` boundaries (`hope` must not match `hopeless`), and skip a match when `no`, `not`, `never`, `without`, `lack of` or `nobody` precedes it in the same clause or when the clause also matches a `suicidal_ideation` rule (`screen_text(clause).categories`); `check_safety` no longer touches `detection_result.risk_score`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.8: Synchronous outbox publisher and category-only safety event payloads
**Covers:** MVP-081
**Severity:** high
**Wave:** 1
**Depends on:** WS-12.6, WS-05.7 (W1 phase A: the event contract files and `verify_event_producer`)
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-28 (default assumed: safety publishes only through its transactional outbox; notification-service pages)
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/events.py` (`SafetyEventPublisher` rewritten; `to_kafka_event` mapping; `SafetyCheckCompletedEvent.detection_layer`; drop `CrisisDetectedEvent.content_preview`)
- Modify: `services/safety_service/tests/test_events.py` (`TestSafetyEventPublisher` and `TestSafetyEventPublishing` move from queue semantics to a fake plane), `services/safety_service/tests/test_crisis_event_flow.py` (publisher plus fake plane instead of the in-process queue and bridge)
- Create: `services/safety_service/tests/test_mvp_081_outbox_publisher.py`, `services/safety_service/tests/test_mvp_357_event_contracts.py` (tests; the second is the producer verification WS-05 requested for WS-05.10 and the file R-13-7 of WS-13 refers to)
**Interfaces:**
- Consumes: IF-05 `get_service_event_plane() -> ServiceEventPlane`, `ServiceEventPlane.publish(event: BaseEvent, *, conn=None) -> UUID`, `solace_events.errors.EventPublishError` (WS-06, W1 phase A1); IF-20 `solace_events.schemas.EVENT_REGISTRY`, `solace_testing.contracts.load_event_contracts(dir) -> list[EventContract]` and `verify_event_producer(contract, produced) -> list[str]` (WS-05.7); WS-05.7's contract files `tests/contracts/events/safety__notification/safety.crisis.detected.json`, `safety.escalation.triggered.json` and `safety.assessment.completed.json`; IF-07 `solace_clinical.parse_escalation_reason` (WS-12.2, crisis-review F10).
- Produces: `SafetyEventPublisher(plane: ServiceEventPlane | None = None)` with `async publish(event: SafetyEvent, *, conn: asyncpg.Connection | None = None) -> UUID | None` (returns after the outbox write or inside the caller's transaction; re-raises `EventPublishError`; returns None only when `to_kafka_event` has no mapping), `async start()`/`async stop()` no-ops, deprecated `register_handler`/`register_callback` that only log `safety_event_handler_ignored` once (compatibility until R-12-1 lands); `get_event_publisher()` and `initialize_event_publisher()` keep their names. Kafka mapping (IF-06): CrisisDetected `trigger_indicators` filtered to `CATEGORIES`, `escalation_action="page_on_call"`, `requires_human_review=True`, `detection_layer` from the event; EscalationTriggered `escalation_reason` passes only when `solace_clinical.parse_escalation_reason(reason)` accepts it (the `auto|manual:<LEVEL>[:codes]` form with every code in `CATEGORIES`; IF-06's single format, crisis-review F10), else `"unspecified"`; SafetyAssessment `detection_layer` from the event; `metadata.correlation_id = event.correlation_id`, `metadata.source_service = "safety-service"`. Answers WS-05.10 (request 9 of WS-05): the safety producer's `verify_event_producer` test in `services/safety_service/tests/test_mvp_357_event_contracts.py`, which WS-05.10's coverage guard finds by the strings `verify_event_producer(` and `safety__`.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_081_outbox_publisher.py` and `test_mvp_357_event_contracts.py`
```python
# test_mvp_081_outbox_publisher.py
from decimal import Decimal
from uuid import uuid4
import pytest
import services.safety_service.src.events as events_mod
from services.safety_service.src.events import (CrisisDetectedEvent, EscalationTriggeredEvent,
                                                 SafetyCheckCompletedEvent, SafetyEventPublisher, to_kafka_event)
from solace_events.errors import EventPublishError

class _FakePlane:
    def __init__(self, fail=False): self.fail, self.published = fail, []
    async def publish(self, event, *, conn=None):
        if self.fail: raise EventPublishError("outbox insert failed")
        self.published.append((event, conn)); return event.metadata.event_id

def _crisis(turn=None):
    return CrisisDetectedEvent(user_id=uuid4(), crisis_level="CRITICAL", risk_score=Decimal("0.85"),
        trigger_indicators=["suicidal_ideation", "CRITICAL_KEYWORD:kill myself"], detection_layers=[1],
        requires_escalation=True, requires_human_review=True, correlation_id=turn or uuid4())

async def test_mvp_081_publish_surfaces_outbox_failure(monkeypatch):
    monkeypatch.setattr(events_mod, "get_service_event_plane", lambda: _FakePlane(fail=True), raising=False)
    with pytest.raises(EventPublishError):
        await SafetyEventPublisher().publish(_crisis())

async def test_mvp_081_publish_is_synchronous_and_queue_free(monkeypatch):
    plane = _FakePlane(); monkeypatch.setattr(events_mod, "get_service_event_plane", lambda: plane, raising=False)
    publisher = SafetyEventPublisher()
    for _ in range(3):
        await publisher.publish(_crisis())
    await publisher.stop()
    assert len(plane.published) == 3 and not hasattr(publisher, "_event_queue")

async def test_mvp_081_publish_joins_the_callers_transaction(monkeypatch):
    plane = _FakePlane(); monkeypatch.setattr(events_mod, "get_service_event_plane", lambda: plane, raising=False)
    sentinel = object()
    await SafetyEventPublisher().publish(_crisis(), conn=sentinel)
    assert plane.published[0][1] is sentinel

def test_if06_crisis_payload_carries_category_codes_only():
    turn = uuid4(); kafka = to_kafka_event(_crisis(turn))
    assert kafka.trigger_indicators == ["suicidal_ideation"]
    assert kafka.escalation_action == "page_on_call" and kafka.requires_human_review is True
    assert kafka.metadata.correlation_id == turn and kafka.metadata.source_service == "safety-service"

def test_if06_escalation_reason_and_assessment_layer():
    base = dict(user_id=uuid4(), escalation_id=uuid4(), priority="CRITICAL", crisis_level="CRITICAL")
    assert to_kafka_event(EscalationTriggeredEvent(**base, reason="auto:CRITICAL:suicidal_ideation")).escalation_reason \
        == "auto:CRITICAL:suicidal_ideation"
    assert to_kafka_event(EscalationTriggeredEvent(**base, reason="Triggers: kill myself")).escalation_reason == "unspecified"
    assert to_kafka_event(EscalationTriggeredEvent(**base, reason="auto:CRITICAL:not_a_code")).escalation_reason == "unspecified"
    check = SafetyCheckCompletedEvent(user_id=uuid4(), check_type="pre_check", is_safe=False, crisis_level="HIGH",
                                      risk_score=Decimal("0.7"), detection_layer=4)
    assert to_kafka_event(check).detection_layer == 4
```
```python
# test_mvp_357_event_contracts.py  (producer half of IF-20 for safety-service; WS-13.13 verifies the consumer half)
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
import pytest
from services.safety_service.src.events import (CrisisDetectedEvent, EscalationTriggeredEvent,
                                                 SafetyCheckCompletedEvent, to_kafka_event)
from solace_testing.contracts import load_event_contracts, verify_event_producer

pytestmark = pytest.mark.contract
CONTRACTS = Path(__file__).resolve().parents[3] / "tests" / "contracts" / "events" / "safety__notification"
REAL_EVENTS = {
    "safety.crisis.detected": lambda: CrisisDetectedEvent(
        user_id=uuid4(), crisis_level="CRITICAL", risk_score=Decimal("0.85"),
        trigger_indicators=["suicidal_ideation", "CRITICAL_KEYWORD:kill myself"], detection_layers=[1],
        requires_escalation=True, requires_human_review=True, correlation_id=uuid4()),
    "safety.escalation.triggered": lambda: EscalationTriggeredEvent(
        user_id=uuid4(), escalation_id=uuid4(), priority="CRITICAL", crisis_level="CRITICAL",
        reason="auto:CRITICAL:suicidal_ideation", correlation_id=uuid4()),
    "safety.assessment.completed": lambda: SafetyCheckCompletedEvent(
        user_id=uuid4(), check_type="pre_check", is_safe=False, crisis_level="HIGH",
        risk_score=Decimal("0.7"), detection_layer=1, correlation_id=uuid4()),
}

def test_mvp_357_safety_contracts_cover_every_event_safety_publishes():
    assert {c.event_type for c in load_event_contracts(CONTRACTS)} == set(REAL_EVENTS)

@pytest.mark.parametrize("contract", load_event_contracts(CONTRACTS), ids=lambda c: c.event_type)
def test_mvp_357_safety_producer_matches_the_safety__notification_contract(contract):
    assert verify_event_producer(contract, to_kafka_event(REAL_EVENTS[contract.event_type]())) == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_081_outbox_publisher.py services/safety_service/tests/test_mvp_357_event_contracts.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `publish()` only enqueues (`events.py:383-387`) so `DID NOT RAISE`; nothing reaches the plane and `_event_queue` exists; `publish(..., conn=)` raises `TypeError: unexpected keyword argument 'conn'`; the raw indicator `CRITICAL_KEYWORD:kill myself` is forwarded and `escalation_action == "escalate"` (`events.py:229-237`); free-text reasons pass through; the assessment layer is hard-coded 1 (`events.py:267-272`). The contract test fails on `safety.crisis.detected` because `verify_event_producer` reports the contract's `categories` rule against the forwarded raw indicator and its `equals:"page_on_call"` rule against `"escalate"` (the other two contracts may already pass; they guard the mapping from now on). `SafetyCheckCompletedEvent` silently ignores the unknown `detection_layer` keyword today (pydantic's default `extra="ignore"`), which is why the unit test above asserts the mapped value 4.
- [ ] **Step 3: Implement** `SafetyEventPublisher.publish(event, *, conn=None) -> UUID | None` in `events.py`: `kafka_event = to_kafka_event(event)`; `None` returns `None` with a debug log; `plane = self._plane or get_service_event_plane()` (module-level import from `solace_events.service_plane`, so tests can monkeypatch it); `return await plane.publish(kafka_event, conn=conn)`. Exceptions propagate unchanged. Update `to_kafka_event` per the Produces list and switch its import to `solace_events.schemas` if WS-12.1 has not.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1, then `tests/integration/test_event_flow.py tests/integration/test_safety_service_e2e.py` in one process. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.9: Escalation decisions recorded and published in one transaction, never delivered inline
**Covers:** MVP-081, MVP-085, MVP-087, MVP-088, MVP-177, MVP-179, MVP-181
**Severity:** high
**Wave:** 1
**Depends on:** WS-12.4, WS-12.7, WS-12.8, WS-06.10 (`record_publish_failure`, crisis-review F11), WS-04.3 (metric object `CRISIS_DETECTED`, W0)
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-28 (default assumed: notification-service is the only pager and the episode system of record; safety records decisions and publishes CrisisDetected with the turn correlation id; a record-and-publish block that fails is retried in the background, never inline and never through a non-transactional path, crisis-review F2)
**Real infra:** yes
**Files:**
- Modify: `services/safety_service/src/domain/escalation.py` (delete `NotificationServiceClient`, the `NotificationService` alias, `ClinicianAssigner`, `EscalationWorkflow`, `CrisisResourceManager`, `_active_escalations`, `_dedup_window`, `acknowledge_escalation`, `resolve_escalation`; `EscalationRecord` gains `correlation_id`, `source`, `categories`, `detection_layer`; repositories gain `transaction()` and `insert_decision(...)`; JSONB parameters are Python objects (IF-03))
- Modify: `services/safety_service/src/domain/service.py` (`check_safety(..., *, turn_id=None)`; one transaction per HIGH/CRITICAL decision; publish order; failure handling and the background record retry; stats)
- Modify: `services/safety_service/src/domain/factory.py` (no clinician registry), `services/safety_service/src/main.py` (`SafetyEventPublisher(plane=runtime.event_plane)`; `initialize_notification_bridge(postgres_pool=runtime.pg_pool)` always; background task `safety_record_retry` (a loop that awaits `safety_service.retry_pending_records()` every 1 s and logs, never raises, on an iteration error) registered with `runtime.register_background_task`; stop reading `SAFETY_KAFKA_*` and delete `_require_event_delivery_in_strict`, because from W1 the IF-01 event plane is fatal in strict envs), `services/safety_service/src/config.py` (delete the inline-notification fields of `EscalationConfig`)
- Modify (delete): `services/safety_service/src/infrastructure/clinician_registry.py`, `services/safety_service/tests/test_event_delivery_guard.py`
- Modify: `services/safety_service/tests/conftest.py` (the autouse `_mock_notification_http` becomes a network guard that patches `httpx.AsyncClient.post` with an `AsyncMock` and yields it), `services/safety_service/tests/test_escalation.py`, `services/safety_service/tests/test_service.py`, `services/safety_service/tests/test_postgres_escalation_repo.py`, `tests/integration/test_safety_service_e2e.py` (drop `clinician_registry=`; capturing publisher accepts `conn=`)
- Test: `services/safety_service/tests/test_mvp_085_087_177_179_181_escalation_records.py`, `tests/realinfra/safety-detection-lexicon/test_mvp_081_088_escalation_outbox_txn.py`
**Interfaces:**
- Consumes: WS-12.4 columns and `ux_escalations_auto_correlation`; WS-12.8 `SafetyEventPublisher.publish(event, *, conn=None)`; IF-05 `start_service_event_plane(service_name, *, pool, kafka_settings=None, strict)`; `solace_events.service_plane.record_publish_failure(service_name: str) -> None` (R-12-2, planned in WS-06.10; crisis-review F11); `solace_infrastructure.runtime.metrics.CRISIS_DETECTED` (series `solace_crisis_detected_total`, labels `source`, `level`; WS-04.3; R-12-5 asked for the undefined name `CRISIS_DETECTED_TOTAL`, crisis-review F11); IF-07 `format_escalation_reason` (WS-12.2).
- Produces: `EscalationRepository.transaction() -> AsyncContextManager[conn | None]` and `insert_decision(record: EscalationRecord, *, conn=None) -> tuple[EscalationRecord, bool]` (`ON CONFLICT (correlation_id) WHERE source = 'auto' AND correlation_id IS NOT NULL DO NOTHING`, then the existing row); `EscalationManager(settings=None, repository=None)` with `transaction()`, `async record_decision(*, user_id, session_id, crisis_level: str, categories: Sequence[str], correlation_id: UUID, source: Literal["auto", "manual"], detection_layer: int = 1, conn=None) -> tuple[EscalationRecord, bool]`, `async escalate(user_id, session_id, crisis_level, reason, context=None, priority_override=None, *, correlation_id=None, categories=()) -> EscalationResult` (record only, status PENDING, the free-text `reason` is not stored), `get_crisis_resources(crisis_level) -> list[dict]` (returns `solace_clinical.crisis_resources()`); `SafetyService.check_safety(user_id, session_id, content, check_type, context=None, *, turn_id: UUID | None = None) -> SafetyCheckResult` (adds `categories`, `turn_id`); `async SafetyService.retry_pending_records() -> int` (re-runs each due failed HIGH/CRITICAL record-and-publish block, returns the number committed, never raises) and `SafetyService.pending_record_retries() -> int` (crisis-review F2: at most one pending entry per correlation id, at most 10,000 entries, backoff 1 s doubling to 60 s per entry, kept until the block commits; idempotent through `ux_escalations_auto_correlation`, so a block that did commit earlier publishes nothing again; on overflow a CRITICAL `safety_crisis_record_retry_full` log names `correlation_id` and `user_id`). Stored `reason` = `format_escalation_reason("auto", level, categories)` (`"auto:<LEVEL>:<sorted categories>"`); `actions_taken` = `["escalation_recorded", "page_requested"]`; `context = {}`.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_085_087_177_179_181_escalation_records.py` and the realinfra file
```python
import asyncio
from uuid import uuid4
from services.safety_service.src.domain.escalation import (EscalationManager, EscalationSettings,
                                                           EscalationStatus, InMemoryEscalationRepository)
from services.safety_service.src.domain.factory import build_crisis_detector
from services.safety_service.src.domain.service import SafetyService, SafetyServiceSettings
from solace_common.enums import CrisisLevel as L
from solace_events.errors import EventPublishError

class _Publisher:
    def __init__(self): self.events = []
    async def publish(self, event, *, conn=None): self.events.append((event, conn))

class _FailingRepo(InMemoryEscalationRepository):
    async def save_escalation(self, record): raise ConnectionError("db down")      # today's write path
    async def insert_decision(self, record, *, conn=None): raise ConnectionError("db down")  # new write path

def _svc(repo=None, pub=None):
    return SafetyService(SafetyServiceSettings(), build_crisis_detector(),
                         EscalationManager(EscalationSettings(), repository=repo or InMemoryEscalationRepository()),
                         event_publisher=pub or _Publisher())

CRISIS = "I want to kill myself tonight"

async def test_mvp_085_179_record_claims_only_what_happened(_mock_notification_http):
    repo = InMemoryEscalationRepository()
    await _svc(repo).check_safety(uuid4(), uuid4(), CRISIS, "pre_check")
    [record] = await repo.get_active_escalations()
    assert record.status == EscalationStatus.PENDING and record.notifications_sent == []
    assert set(record.actions_taken) <= {"escalation_recorded", "page_requested"}
    assert _mock_notification_http.post.await_count == 0

async def test_mvp_087_check_never_waits_on_notification(_mock_notification_http):
    async def _hang(*a, **k): await asyncio.sleep(30)
    _mock_notification_http.post.side_effect = _hang
    result = await asyncio.wait_for(_svc().check_safety(uuid4(), uuid4(), CRISIS, "pre_check"), timeout=2.0)
    assert result.crisis_level == L.CRITICAL

async def test_mvp_177_second_crisis_is_recorded_not_suppressed():
    repo, pub, user = InMemoryEscalationRepository(), _Publisher(), uuid4()
    svc = _svc(repo, pub)
    await svc.check_safety(user, uuid4(), "I want to kill myself", "pre_check")
    await svc.check_safety(user, uuid4(), "I am going to end my life tonight", "pre_check")
    assert len(await repo.get_active_escalations(user_id=user)) == 2
    assert [e.event_type.value for e, _ in pub.events].count("safety.escalation.triggered") == 2

async def test_mvp_181_one_check_is_one_correlated_transaction_crisis_first():
    pub = _Publisher()
    await _svc(pub=pub).check_safety(uuid4(), uuid4(), CRISIS, "pre_check")
    assert [e.event_type.value for e, _ in pub.events] == [
        "safety.crisis.detected", "safety.escalation.triggered", "safety.check.completed"]
    assert len({e.correlation_id for e, _ in pub.events}) == 1
    assert all(conn is not None for _, conn in pub.events)

async def test_turn_id_makes_a_retried_check_idempotent():
    repo, pub, user, turn = InMemoryEscalationRepository(), _Publisher(), uuid4(), uuid4()
    svc = _svc(repo, pub)
    for _ in range(2):
        await svc.check_safety(user, uuid4(), CRISIS, "pre_check", turn_id=turn)
    assert len(await repo.get_active_escalations(user_id=user)) == 1
    assert [e.event_type.value for e, _ in pub.events].count("safety.crisis.detected") == 1
    assert all(e.correlation_id == turn for e, _ in pub.events)

async def test_mvp_088_record_failure_publishes_nothing_and_still_answers():
    pub = _Publisher()
    result = await _svc(_FailingRepo(), pub).check_safety(uuid4(), uuid4(), CRISIS, "pre_check")
    assert result.crisis_level == L.CRITICAL and pub.events == []

class _FlakyRepo(InMemoryEscalationRepository):
    def __init__(self, failures):
        super().__init__(); self.failures = failures
    async def insert_decision(self, record, *, conn=None):
        if self.failures:
            self.failures -= 1; raise ConnectionError("db down")
        return await super().insert_decision(record, conn=conn)

async def test_crisis_review_f2_failed_record_is_retried_until_it_commits_and_publishes_once():
    repo, pub, user, turn = _FlakyRepo(failures=2), _Publisher(), uuid4(), uuid4()
    svc = _svc(repo, pub)
    svc._record_retry_backoff_base = 0.0                     # test seam on the retry schedule
    result = await svc.check_safety(user, uuid4(), CRISIS, "pre_check", turn_id=turn)
    assert result.crisis_level == L.CRITICAL and pub.events == [] and svc.pending_record_retries() == 1
    for _ in range(3):
        await svc.retry_pending_records()
    assert svc.pending_record_retries() == 0
    assert [e.event_type.value for e, _ in pub.events].count("safety.crisis.detected") == 1
    assert len(await repo.get_active_escalations(user_id=user)) == 1 and all(e.correlation_id == turn for e, _ in pub.events)

async def test_crisis_review_f11_counts_use_the_shared_metric_objects():
    from prometheus_client import REGISTRY
    crisis = lambda: REGISTRY.get_sample_value("solace_crisis_detected_total", {"source": "safety", "level": "CRITICAL"}) or 0.0
    failed = lambda: REGISTRY.get_sample_value("solace_event_publish_failures_total", {"service": "safety-service"}) or 0.0
    before_crisis, before_failed, user, turn = crisis(), failed(), uuid4(), uuid4()
    svc = _svc()
    for _ in range(2):
        await svc.check_safety(user, uuid4(), CRISIS, "pre_check", turn_id=turn)
    assert crisis() - before_crisis == 1.0                  # once per created decision, source "safety" like the other producers
    await _svc(_FailingRepo()).check_safety(uuid4(), uuid4(), CRISIS, "pre_check")
    assert failed() - before_failed == 1.0                  # a non-publish failure still feeds CrisisEventPublishFailures

async def test_elevated_publishes_the_assessment_only():
    repo, pub = InMemoryEscalationRepository(), _Publisher()
    await _svc(repo, pub).check_safety(uuid4(), None, "I feel hopeless", "pre_check")
    assert [e.event_type.value for e, _ in pub.events] == ["safety.check.completed"]
    assert await repo.get_active_escalations() == []
```
```python
# tests/realinfra/safety-detection-lexicon/test_mvp_081_088_escalation_outbox_txn.py
#  pytestmark = [realinfra, integration]
#  fixture: plane = await start_service_event_plane("safety-service", pool=pg_migrated.app_pool, strict=False);
#     svc composed with PostgresEscalationRepository(pg_migrated.app_pool) and SafetyEventPublisher(plane=plane)
#  test_mvp_081_row_and_outbox_commit_together: check_safety(user, None, CRISIS, "pre_check", turn_id=turn) ->
#     SELECT count(*) FROM escalations WHERE correlation_id = $turn  == 1
#     event_outbox rows (slot 010) with source_service = 'safety-service' and
#       event_payload->'metadata'->>'correlation_id' = str(turn): event types ==
#       {"safety.crisis.detected", "safety.escalation.triggered", "safety.assessment.completed"}
#     SELECT jsonb_typeof(actions_taken) == 'array' (JSONB stored as objects, not strings)
#     a second identical call leaves both counts unchanged
#  test_mvp_088_outbox_failure_rolls_back_the_row: monkeypatch plane.publish to raise EventPublishError for
#     "safety.escalation.triggered" -> check_safety still returns CRITICAL; zero escalations rows and zero
#     outbox rows carry that turn id; then, with the monkeypatch removed, one retry_pending_records() pass commits
#     exactly 1 escalations row and the 3 outbox rows for that turn, and a second pass adds none (crisis-review F2)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_085_087_177_179_181_escalation_records.py -o "addopts=" -p no:logging -q`, then `$env:SOLACE_REQUIRE_REALINFRA="1";` plus the same command on `tests/realinfra/safety-detection-lexicon/test_mvp_081_088_escalation_outbox_txn.py`
Expected: FAIL: the CRITICAL workflow POSTs to `/api/v1/notifications/send` and records `IN_PROGRESS` with "Crisis resources displayed to user", "All processing paused", "Safety dialogue initiated" (`escalation.py:404-449`, MVP-085/179); a hanging POST times the check out (MVP-087); the second crisis is `DEDUPLICATED` and not stored (`escalation.py:826-849`, MVP-177); events go out EscalationTriggered first with three random correlation ids and no connection (MVP-181); `check_safety(..., turn_id=...)` raises `TypeError`; the failing save is swallowed and CrisisDetected is still published (MVP-088); `pending_record_retries` and `retry_pending_records` do not exist, so a transient database failure loses the turn's page for good (crisis-review F2); `solace_infrastructure.runtime.metrics` has no `CRISIS_DETECTED_TOTAL` and `solace_events.service_plane` no `record_publish_failure` until WS-04.3 and WS-06.10 land, which is why the plan now uses `CRISIS_DETECTED` (crisis-review F11); the realinfra rows do not exist.
- [ ] **Step 3: Implement** `SafetyService.check_safety(..., *, turn_id=None)` in `service.py` and the record path in `escalation.py`. After detection and history updates: `turn = turn_id or uuid4()`; for HIGH/CRITICAL, inside `async with self._escalation_manager.transaction() as conn:` call `record_decision(..., correlation_id=turn, source="auto", conn=conn)` and, only when created, publish CrisisDetected (category codes, detection layer, `requires_escalation=True`, `requires_human_review=True`), EscalationTriggered (`escalation_id`, `priority=level`, `reason=record.reason`) and SafetyCheckCompleted, all with `correlation_id=turn` and `conn=conn`, in that order; increment `CRISIS_DETECTED.labels(source="safety", level=level)` once per created decision (`source` matches the other producers' `orchestrator`, `therapy`, `diagnosis`). Any exception in that block: log CRITICAL `safety_crisis_record_failed` with `error_type`, `correlation_id` and `user_id`, call `record_publish_failure("safety-service")` unless it is an `EventPublishError` (the plane counted it), submit the block's inputs (user, session, level, categories, detection layer, turn) to the background record retry, and return the result unchanged (resources never depend on the database). `retry_pending_records()` re-runs the same block for each due entry with the same correlation id, so the unique auto decision keeps it to one decision and one CrisisDetected per turn; a failure doubles that entry's backoff (seam `_record_retry_backoff_base`, default 1.0, capped at 60 s). Below HIGH, publish SafetyCheckCompleted without `conn`; a failure logs WARNING `safety_assessment_publish_failed`. `InMemoryEscalationRepository.transaction()` yields a fresh `object()` sentinel; `PostgresEscalationRepository.transaction()` is `pool.acquire()` plus `conn.transaction()` (escalations is non-RLS, IF-03). `EscalationResult.notification_sent` is always False.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1; then `tests/integration/test_crisis_flow.py tests/integration/test_safety_service_e2e.py tests/integration/test_event_flow.py` in one process; then `$env:SOLACE_REQUIRE_REALINFRA="1";` plus the unit template on `tests/realinfra/safety-detection-lexicon`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.10: The IF-06 /check contract: turn id, mapped session id, categories and shared resources
**Covers:** MVP-087
**Severity:** high
**Wave:** 1
**Depends on:** WS-12.9
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-22 (default assumed: /check answers inside the orchestrator's 2 s budget, never blocks on delivery, and only raises the fused level) and D-29
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/api.py` (`SafetyCheckRequest.session_id: str | None`, `turn_id: UUID | None`; `SafetyCheckResponse` adds `level`, `categories`, `resources`, `resources_version`, `turn_id`; legacy fields kept; `/resources` and `crisis_resources` use `solace_clinical.crisis_resources()`; `_get_crisis_resources` removed)
- Modify: `services/safety_service/src/domain/service.py` (`filter_output` appends the shared resources for ELEVATED and above)
- Modify: `services/safety_service/tests/test_api.py` (resource assertions use `kind`; the mocked escalation manager no longer needs `CrisisResourceManager`)
- Create: `services/safety_service/tests/test_if06_check_contract.py` (test)
- Create: `services/safety_service/tests/test_mvp_017_contract_orchestrator_check.py` (provider verification requested by WS-10 for WS-10.16; skips with a reason until WS-10.16 writes the consumer file in W2, then runs on every suite run)
**Interfaces:**
- Consumes (D-29): `/check`, `/resources` and `filter_output` serve `crisis_resources(DEFAULT_LOCALE)` with `resources_version == RESOURCES_VERSION`. They are informational for the orchestrator, which renders its own locale-resolved resources from the access-token claim (WS-10.5) and never shows safety's, so in-band resources never depend on this service and `/check` needs no locale field.
- Consumes: WS-12.9 `check_safety(..., turn_id=)`, `SafetyCheckResult.categories`; `solace_clinical.crisis_resources`, `RESOURCES_VERSION`; IF-20 `solace_testing.contracts.verify_provider(app, contracts_dir, *, auth_headers)` (WS-05.7); WS-10.16's W2 consumer contract `tests/contracts/orchestrator__safety/check.json` (request R-12-10).
- Produces: the provider half of the `orchestrator__safety` pair for WS-10.16 and WS-05.10's coverage guard (`verify_provider(` and `orchestrator__safety` in a safety test file). IF-06: request `{user_id, session_id?, content, turn_id?, check_type?, context?}`; response `{level, risk_score, categories, requires_escalation, resources, resources_version, detection_layer, turn_id, session_id, ...legacy}`; `turn_id` = body, else header `X-Turn-ID`, else a new UUID; non-UUID `session_id` becomes `uuid5(NAMESPACE_URL, "solace:session:" + raw)`; `resources` non-empty exactly when `level >= ELEVATED`.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_if06_check_contract.py`
```python
from uuid import NAMESPACE_URL, uuid4, uuid5
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.safety_service.src.api import router
from services.safety_service.src.domain.escalation import EscalationManager, InMemoryEscalationRepository
from services.safety_service.src.domain.factory import build_crisis_detector
from services.safety_service.src.domain.service import SafetyService, SafetyServiceSettings
from solace_clinical import RESOURCES_VERSION, crisis_resources
from solace_security.middleware import AuthenticatedService, get_current_service

class _Publisher:
    async def publish(self, event, *, conn=None): return None

@pytest.fixture
def client():
    app = FastAPI(); app.include_router(router, prefix="/api/v1/safety")
    manager = EscalationManager(repository=InMemoryEscalationRepository())
    app.state.safety_service = SafetyService(SafetyServiceSettings(), build_crisis_detector(), manager, _Publisher())
    app.state.escalation_manager = manager
    app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="orchestrator-service", permissions=["service:read:safety"])
    return TestClient(app)

def test_check_accepts_client_session_ids_and_echoes_turn(client):
    turn = uuid4()
    body = client.post("/api/v1/safety/check", json={"user_id": str(uuid4()), "session_id": "client-session-7",
                       "content": "I want to kill myself tonight", "turn_id": str(turn)}).json()
    assert body["level"] == "CRITICAL" and body["requires_escalation"] is True
    assert "suicidal_ideation" in body["categories"]
    assert body["resources"] == crisis_resources() and body["resources_version"] == RESOURCES_VERSION
    assert body["turn_id"] == str(turn)
    assert body["session_id"] == str(uuid5(NAMESPACE_URL, "solace:session:client-session-7"))

def test_elevated_shows_resources_and_none_does_not(client):
    elevated = client.post("/api/v1/safety/check", json={"user_id": str(uuid4()), "content": "I feel hopeless"}).json()
    assert elevated["level"] == "ELEVATED" and elevated["resources"] == crisis_resources()
    calm = client.post("/api/v1/safety/check", json={"user_id": str(uuid4()), "content": "Hi, how are you?"}).json()
    assert calm["level"] == "NONE" and calm["resources"] == []

def test_turn_id_header_is_used_when_body_omits_it(client):
    turn = uuid4()
    body = client.post("/api/v1/safety/check", headers={"X-Turn-ID": str(turn)},
                       json={"user_id": str(uuid4()), "content": "I want to kill myself"}).json()
    assert body["turn_id"] == str(turn)

def test_resources_endpoint_uses_the_shared_list(client):
    assert client.get("/api/v1/safety/resources?level=HIGH").json() == crisis_resources()
```
```python
# test_mvp_017_contract_orchestrator_check.py
from pathlib import Path
import pytest
from fastapi import FastAPI
from services.safety_service.src.api import router
from services.safety_service.src.domain.escalation import EscalationManager, InMemoryEscalationRepository
from services.safety_service.src.domain.factory import build_crisis_detector
from services.safety_service.src.domain.service import SafetyService, SafetyServiceSettings
from solace_security.middleware import AuthenticatedService, get_current_service
from solace_testing.contracts import verify_provider

pytestmark = pytest.mark.contract
CONTRACTS = Path(__file__).resolve().parents[3] / "tests" / "contracts" / "orchestrator__safety"

class _Publisher:
    async def publish(self, event, *, conn=None): return None

def _provider_app() -> FastAPI:
    app = FastAPI(); app.include_router(router, prefix="/api/v1/safety")
    manager = EscalationManager(repository=InMemoryEscalationRepository())
    app.state.safety_service = SafetyService(SafetyServiceSettings(), build_crisis_detector(), manager, _Publisher())
    app.state.escalation_manager = manager
    app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="orchestrator-service", permissions=["service:read:safety"])
    return app

def test_mvp_017_safety_satisfies_the_orchestrator__safety_contract():
    if not CONTRACTS.is_dir():
        pytest.skip("tests/contracts/orchestrator__safety is written by WS-10.16 in W2")
    verify_provider(_provider_app(), CONTRACTS, auth_headers=lambda consumer: {"Authorization": "Bearer contract-test"})
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_if06_check_contract.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `"client-session-7"` is rejected with 422 because `session_id: UUID` (`api.py:58`); the response has no `level`, `categories`, `resources`, `resources_version` or `turn_id`; ELEVATED gets no resources (`api.py:202`); `/resources` returns the legacy `{name, contact, type, available}` shape. Run `services/safety_service/tests/test_mvp_017_contract_orchestrator_check.py` with the same command: in W1 it reports SKIPPED with the WS-10.16 reason (the consumer file does not exist yet); from W2 it runs, and on today's router it would fail on the 422 for a client session id.
- [ ] **Step 3: Implement** the request/response models and `perform_safety_check` in `api.py`: map the session id with `_coerce_session_id(raw: str | None) -> UUID | None`; resolve `turn_id` from body, `request.headers.get("X-Turn-ID")`, else `uuid4()`; pass `turn_id` to `check_safety`; `resources = crisis_resources() if level_rank(level) >= level_rank(ELEVATED) else []` regardless of `include_resources` (a flag can never remove crisis resources). At the W2 barrier the controller runs the safety suite once WS-10.16 has landed and records in the ledger that `test_mvp_017_safety_satisfies_the_orchestrator__safety_contract` passed rather than skipped.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (the provider file SKIPPED in W1 as above)
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.11: Service permissions on every safety endpoint
**Covers:** MVP-309
**Severity:** low
**Wave:** 1
**Depends on:** WS-12.10
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** D-03 (default assumed: service JWTs with receiver-side permission resolution; the orchestrator holds `service:read:safety`; `service:escalate` is held by no MVP caller)
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/api.py` (module-level `require_read_safety = require_service_permission("service:read:safety")` on `/check`, `/detect-crisis`, `/assess`, `/filter-output`, `/resources`, `/status`; `require_trigger_escalation = require_service_permission("service:escalate")` on `/escalate`; drop the unused `get_current_user` import)
- Modify: `services/safety_service/tests/test_api.py`, `services/safety_service/tests/test_if06_check_contract.py` and `services/safety_service/tests/test_mvp_017_contract_orchestrator_check.py` (override `api.require_read_safety` / `api.require_trigger_escalation` instead of `get_current_service`; all three created or existing before this task, the last two by WS-12.10)
- Test: `services/safety_service/tests/test_mvp_309_endpoint_permissions.py`
**Interfaces:**
- Consumes: IF-04 `require_service_permission(*perms)`, `ServiceTokenManager(auth_settings=AuthSettings())`, `solace_security.middleware._get_jwt_manager.cache_clear()` (WS-07).
- Produces: `api.require_read_safety`, `api.require_trigger_escalation` dependency objects that tests override.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_309_endpoint_permissions.py`
```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from services.safety_service.src.api import router
from solace_security import middleware
from solace_security.auth import AuthSettings
from solace_security.service_auth import ServiceTokenManager

@pytest.fixture
def client_and_token(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key-for-mvp-309-at-least-32-chars")
    middleware._get_jwt_manager.cache_clear()
    manager = ServiceTokenManager(auth_settings=AuthSettings())
    app = FastAPI(); app.include_router(router, prefix="/api/v1/safety")
    app.state.safety_service = MagicMock(get_status=AsyncMock(return_value={"status": "operational"}))
    app.state.escalation_manager = MagicMock()
    yield TestClient(app), (lambda svc: {"Authorization": f"Bearer {manager.create_service_token(svc).token}"})
    middleware._get_jwt_manager.cache_clear()

def test_mvp_309_status_and_resources_require_a_token(client_and_token):
    client, _ = client_and_token
    assert client.get("/api/v1/safety/status").status_code == 401
    assert client.get("/api/v1/safety/resources").status_code == 401

def test_mvp_309_services_without_read_safety_are_forbidden(client_and_token):
    client, token = client_and_token
    assert client.get("/api/v1/safety/status", headers=token("personality-service")).status_code == 403
    assert client.get("/api/v1/safety/status", headers=token("orchestrator-service")).status_code == 200

def test_mvp_309_escalate_requires_service_escalate(client_and_token):
    client, token = client_and_token
    body = {"user_id": "00000000-0000-0000-0000-000000000001", "crisis_level": "HIGH", "reason": "x"}
    assert client.post("/api/v1/safety/escalate", json=body, headers=token("orchestrator-service")).status_code == 403
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_309_endpoint_permissions.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `/status` and `/resources` answer 200 without a token (`api.py:324-339`); a personality-service token reads `/status`; an orchestrator token reaches `/escalate` (bare `get_current_service`, `api.py:245-250`) and gets past authentication instead of 403.
- [ ] **Step 3: Implement** the two module-level dependencies in `api.py` and attach them as `Depends(require_read_safety)` / `Depends(require_trigger_escalation)`; update the existing tests' overrides.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.12: Escalation records encrypted at rest and erased by pseudonymization (W3)
**Covers:** MVP-089
**Severity:** high
**Wave:** 3
**Depends on:** WS-12.9
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: escalation records are pseudonymized with a salted hash and kept for duty of care; other safety PHI is hard-deleted)
**Real infra:** yes
**Files:**
- Modify: `services/safety_service/src/infrastructure/gdpr.py` (`build_user_data_erasure(*, escalation_repo, clinical_tables, purge_in_memory) -> UserDataErasure` registering exactly `escalations`, `safety_assessments`, `safety_plans`, `safety_in_memory_state`; new `SafetyClinicalTablesEraser(acquire=ConnectionPoolManager.acquire)` with `delete_assessments(user_id) -> int` and `delete_plans(user_id) -> int`)
- Modify: `services/safety_service/src/domain/escalation.py` (`PostgresEscalationRepository` encrypts `reason` with the IF-13 codec, writes `encryption_key_id`, decrypts on read; `pseudonymize_user(user_id) -> int`)
- Modify: `services/safety_service/src/domain/service.py` (`async purge_user(user_id: UUID) -> None` and `purge_user_count(user_id: str) -> int` over the risk history, conversation history and assessment cache)
- Modify: `services/safety_service/src/main.py` (remove the whole PHI-encryption activation block, including its `EncryptionSettings`, `Encryptor`, `FieldEncryptor` and `configure_phi_encryption` imports, so no safety module uses the retired ORM-listener path; `register_user_erasure_consumer(runtime.event_plane, service_name="safety-service", erasure=..., purge_caches=safety_service.purge_user)` before the plane starts)
- Modify: `services/safety_service/tests/test_gdpr_erasure.py` (store set and signature)
- Modify: `tests/integration/test_phi_encryption_activation.py` (WS-04's request for WS-04.15: move `safety_service` from `_PHI_ENCRYPTING` to `_PHI_VIA_RUNTIME`, the bucket WS-04.15 creates in W0; safety then passes through its `handles_phi=True` SPEC from WS-12.1. Only the safety entry changes; the file locks safety's `main.py`, which WS-12 owns, so the tests/integration protocol allows the edit)
- Test: `services/safety_service/tests/test_mvp_089_erasure_wiring.py`, `tests/realinfra/safety-detection-lexicon/test_mvp_089_erasure_pseudonymize.py`
**Interfaces:**
- Consumes: IF-13 `get_phi_codec().encrypt(value, *, field, owner_id)`, `.decrypt(...)`, `.search_hash(value, *, purpose)`, `.active_key_id` (WS-09, W2); IF-16 `register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches)` (WS-14, W3 phase A); IF-03 `ConnectionPoolManager.acquire()` and `acting_as_user(user_id)`; R-12-3 subscribe-before-start.
- Produces: safety's IF-13 adoption for WS-09.15 (CR-10 of WS-09): `services/safety_service/src/domain/escalation.py`, the module WS-09.15's ADOPTERS map lists for WS-12, imports `solace_security.phi_codec` and writes `codec.active_key_id` to `encryption_key_id`; no safety module calls `configure_phi_encryption` or uses `FieldEncryptor` afterwards. WS-12 has no W2 wave, so the adoption lands here in W3 phase B and WS-09.15 depends on this task. For WS-04.15: `safety_service` moves to `_PHI_VIA_RUNTIME`. Pseudonym `UUID(hex=get_phi_codec().search_hash(str(user_id), purpose="crisis_pseudonym")[:32])`; pseudonymization sets `user_id` to the pseudonym, `session_id = NULL`, `context = {}`, `pseudonymized_at = now()`, re-encrypts `reason` under the pseudonym owner and keeps `crisis_level`, `priority`, `categories`, timestamps.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_089_erasure_wiring.py` and the realinfra file
```python
# test_mvp_089_erasure_wiring.py
from uuid import uuid4
from services.safety_service.src.domain.escalation import InMemoryEscalationRepository
from services.safety_service.src.infrastructure.gdpr import build_user_data_erasure

class _Tables:
    async def delete_assessments(self, user_id): return 0
    async def delete_plans(self, user_id): return 0

def test_mvp_089_only_stores_that_exist_are_registered():
    async def _purge(user_id): return 1
    erasure = build_user_data_erasure(escalation_repo=InMemoryEscalationRepository(), clinical_tables=_Tables(),
                                      purge_in_memory=_purge)
    assert set(erasure.store_names) == {"escalations", "safety_assessments", "safety_plans", "safety_in_memory_state"}

async def test_mvp_089_in_memory_state_is_purged():
    # SafetyService with one crisis check for user U and one for V; await service.purge_user(U)
    # -> U absent from _user_risk_history, _conversation_history and _assessment_cache; V still present
    ...
```
```python
# tests/realinfra/safety-detection-lexicon/test_mvp_089_erasure_pseudonymize.py
#  seed two CRITICAL decisions for U and one for V through PostgresEscalationRepository(pg_migrated.app_pool)
#  test_mvp_089_reason_is_ciphertext_at_rest: raw SELECT reason FROM escalations WHERE correlation_id = $turn
#       LIKE 'enc:v1:%' and encryption_key_id = get_phi_codec().active_key_id
#  test_mvp_089_erasure_completes_and_pseudonymizes: with acting_as_user(U): report = await erasure.erase(str(U))
#       report.complete is True and report.deleted["escalations"] == 2
#       no escalations row has user_id = U; two rows have pseudonymized_at NOT NULL, session_id NULL,
#       categories unchanged, and the reason decrypts under the pseudonym owner id; V's row is untouched
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_089_erasure_wiring.py -o "addopts=" -p no:logging -q`, then `$env:SOLACE_REQUIRE_REALINFRA="1";` plus the same command on `tests/realinfra/safety-detection-lexicon/test_mvp_089_erasure_pseudonymize.py`
Expected: FAIL: `build_user_data_erasure` still demands incident and risk-profile repositories that target `safety_incidents` and `user_risk_profiles`, tables no migration creates (`gdpr.py:62-67`), so an erasure run raises `ErasureIncomplete` (MVP-089); `purge_user` does not exist; the reason column is plaintext.
- [ ] **Step 3: Implement** the new `build_user_data_erasure`, `SafetyClinicalTablesEraser` (RLS tables through `ConnectionPoolManager.acquire()` under the helper's `acting_as_user`), `PostgresEscalationRepository.pseudonymize_user` (one UPDATE per row inside a transaction because `reason` must be re-encrypted), the codec on insert and read (`field="escalations.reason"`, owner = user id), and `SafetyService.purge_user`; wire the consumer in `main.py`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1; `tests/integration/test_phi_encryption_activation.py tests/integration/test_safety_service_e2e.py` in one process (the activation lock stays green only with safety in `_PHI_VIA_RUNTIME`); `$env:SOLACE_REQUIRE_REALINFRA="1";` plus the unit template on `tests/realinfra/safety-detection-lexicon`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.13: Remove the unwired contraindication engine and legacy detectors (W5)
**Covers:** MVP-174
**Severity:** medium
**Wave:** 5
**Depends on:** WS-12.6, WS-12.17, WS-12.18
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** D-19 (default assumed: unwired engines are removed and docs that claim they run are corrected)
**Real infra:** no
**Files:**
- Modify (delete) engines: `services/safety_service/src/ml/contraindication.py`, `services/safety_service/src/ml/keyword_detector.py`, `services/safety_service/src/ml/pattern_matcher.py`, `services/safety_service/src/ml/sentiment_analyzer.py`, `services/safety_service/src/db/contraindication_db.py`, `services/safety_service/src/db/__init__.py`, `services/safety_service/src/observability/__init__.py`, `services/safety_service/src/infrastructure/database.py`
- Modify (delete) data: `services/safety_service/db/schema/contraindication_rules.sql`, `services/safety_service/db/seeds/contraindication_rules_seed.sql`, `services/safety_service/config/keywords.json`, `services/safety_service/config/patterns.json`, `services/safety_service/config/clinical_lexicon.json`
- Modify (delete) their tests: `services/safety_service/tests/test_contraindication.py`, `services/safety_service/tests/test_keyword_detector.py`, `services/safety_service/tests/test_pattern_matcher.py`, `services/safety_service/tests/test_sentiment_analyzer.py`, `services/safety_service/tests/test_batch_3_3_integration.py`, `services/safety_service/tests/test_mvp_086_ml_extras_parity.py` (WS-12.18; the modules it reloads are gone)
- Modify: `services/safety_service/src/ml/__init__.py` (exports only `LLMAssessor`, `RiskAssessment`, `LLMAssessorConfig`), `services/safety_service/src/infrastructure/__init__.py` (drop database exports), `services/safety_service/README.md` (detection is the shared lexicon plus raise-only LLM; no contraindication engine), `services/safety_service/tests/test_mvp_083_084_086_178_182_detector_fusion.py` and `services/safety_service/tests/test_mvp_090_history_signal.py` (the autouse fixture keeps only the `llm_assessor` patch), `services/safety_service/tests/test_mvp_271_no_private_pool.py` (WS-12.17; drop the repository test, keep the AST scan for `create_pool`)
- Test: `services/safety_service/tests/test_mvp_174_no_dead_engines.py`
**Interfaces:**
- Consumes: none.
- Produces: none (deletions). Legacy coverage stays proven by `tests/clinical/legacy_terms_v0.yaml` (WS-12.3), which does not read the deleted JSON.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_174_no_dead_engines.py`
```python
import ast
from pathlib import Path
SERVICE = Path(__file__).resolve().parents[1]
REMOVED = ["src/ml/contraindication.py", "src/ml/keyword_detector.py", "src/ml/pattern_matcher.py",
           "src/ml/sentiment_analyzer.py", "src/db", "src/observability", "src/infrastructure/database.py",
           "db", "config/keywords.json", "config/patterns.json", "config/clinical_lexicon.json"]
DEAD = ("contraindication", "keyword_detector", "pattern_matcher", "sentiment_analyzer", "contraindication_db")

def test_mvp_174_dead_engines_are_gone():
    assert [p for p in REMOVED if (SERVICE / p).exists()] == []

def test_mvp_174_nothing_imports_them():
    offenders = []
    for py in (SERVICE / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and any(d in node.module for d in DEAD):
                offenders.append(str(py))
    assert offenders == []

def test_mvp_174_readme_stops_claiming_the_engine():
    assert "ContraindicationChecker" not in (SERVICE / "README.md").read_text(encoding="utf-8")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_174_no_dead_engines.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the never-constructed `ContraindicationChecker` (`ml/contraindication.py:130-159`), its two duplicate repositories, the unloaded seed SQL and the detectors unwired by WS-12.6 still exist and `ml/__init__.py` imports them.
- [ ] **Step 3: Implement** the deletions and the three edits; record in the ledger the per-suite test-count drop caused by the deleted test files (expected and explained).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1, the `tests/clinical` command from WS-12.2 and `tests/test_syntax_and_imports.py`, one process each. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.14: Slot 014 amendment: per-user risk state table under RLS (W5)
**Covers:** MVP-175
**Severity:** medium
**Wave:** 5
**Depends on:** WS-12.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: slot bodies are amended in place before the first real deployment)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/014_safety.py` (adds `safety_user_risk_state`)
- Test: `tests/realinfra/safety-detection-lexicon/test_mvp_175_slot014_risk_state.py`
**Interfaces:**
- Consumes: IF-03 roles and the NULLIF policy form; IF-19 `pg_migrated`.
- Produces: `safety_user_risk_state(user_id UUID PRIMARY KEY, last_crisis_level VARCHAR(16) NULL, last_crisis_at TIMESTAMPTZ NULL, crisis_events INTEGER NOT NULL DEFAULT 0, recent_levels VARCHAR(16)[] NOT NULL DEFAULT '{}', updated_at TIMESTAMPTZ NOT NULL DEFAULT now())` with `ENABLE` and `FORCE ROW LEVEL SECURITY` and policy `safety_user_risk_state_owner` using `user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid` for USING and WITH CHECK.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/safety-detection-lexicon/test_mvp_175_slot014_risk_state.py`
```python
# pytestmark = [realinfra, integration]
# every GUC is set transaction-locally, so no RLS user is left on a pooled connection (review focus 3):
#   async with pg_migrated.app_pool.acquire() as conn, conn.transaction():
#       await conn.execute("SELECT set_config('app.current_user_id', $1, true)", str(U))
# test_mvp_175_owner_can_write_and_read_own_row: inside one such transaction INSERT a row for U; SELECT returns it
# test_mvp_175_other_users_rows_are_invisible: insert as U, then in a new transaction with the GUC set to V,
#   SELECT count(*) == 0; INSERT for U while the GUC is V raises asyncpg.InsufficientPrivilegeError (WITH CHECK)
# test_mvp_175_no_context_sees_nothing_and_cannot_write: without any GUC, SELECT count(*) == 0 and INSERT raises
#   asyncpg.InsufficientPrivilegeError (a context-less writer fails loud)
# test_mvp_175_policy_uses_nullif: pg_policies.qual for the table contains "NULLIF"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:SOLACE_REQUIRE_REALINFRA="1"; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/safety-detection-lexicon/test_mvp_175_slot014_risk_state.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `UndefinedTableError: relation "safety_user_risk_state" does not exist` (risk history lives only in process dicts, `service.py:120-122`, MVP-175).
- [ ] **Step 3: Implement** the table, RLS and policy in `upgrade()` after the W1 columns, and drop them first in `downgrade()`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `tests/migrations/test_chain_real_pg.py` with `$env:SOLACE_REQUIRE_REALINFRA="1";` (as in WS-12.4). Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.15: Durable risk history that survives restarts, keeps no message text and is erased (W5)
**Covers:** MVP-175
**Severity:** medium
**Wave:** 5
**Depends on:** WS-12.14, WS-12.12
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: with sign-off recorded, the trajectory signal uses the last five lexicon levels instead of stored message text)
**Real infra:** yes
**Files:**
- Modify: `services/safety_service/src/infrastructure/repository.py` (`RiskState` dataclass and `PostgresRiskStateRepository(acquire=ConnectionPoolManager.acquire)` with `load(user_id) -> RiskState | None`, `save(state) -> None` (upsert), `delete_user_data(user_id) -> int`)
- Modify: `services/safety_service/src/domain/service.py` (risk state read and written through the repository under `acting_as_user(user_id)`; `_conversation_history` replaced by `recent_levels`; history passed to the detector from `RiskState`)
- Modify: `services/safety_service/src/domain/crisis_detector.py` (`detect(..., recent_levels: Sequence[CrisisLevel] | None = None)`; trajectory deteriorates when at least 3 of the last 5 levels are LOW or above and the current level is ELEVATED or above)
- Modify: `services/safety_service/src/main.py` (repository wiring), `services/safety_service/src/infrastructure/gdpr.py` (register store `safety_user_risk_state`)
- Test: `services/safety_service/tests/test_mvp_175_durable_risk_state.py`, `tests/realinfra/safety-detection-lexicon/test_mvp_175_risk_state_restart.py`
**Interfaces:**
- Consumes: WS-12.14 table; IF-03 `acting_as_user`, `ConnectionPoolManager.acquire()`.
- Produces: `RiskState(user_id, last_crisis_level: str | None, last_crisis_at: datetime | None, crisis_events: int, recent_levels: tuple[str, ...])`; `SafetyService(..., risk_state_repository: RiskStateRepository | None = None)`.

- [ ] **Step 1: Write the failing test** in both files
```python
# test_mvp_175_durable_risk_state.py (in-memory RiskStateRepository fake shared by two SafetyService instances)
#  test_mvp_175_history_survives_a_restart: service A checks "I want to kill myself" for U; a NEW service B
#     built on the same repository checks "I'm so overwhelmed" for U -> ELEVATED (watch active), not LOW
#  test_mvp_175_no_message_text_is_kept: after checks, no attribute of service B or the saved RiskState
#     contains the substring "overwhelmed" (scan vars() recursively)
# tests/realinfra/safety-detection-lexicon/test_mvp_175_risk_state_restart.py
#  same restart scenario through PostgresRiskStateRepository on pg_migrated; plus erasure removes U's row
#  test_review_focus_3_foreign_user_never_inherits_history: after U's CRITICAL check, a new service checks
#     "I'm so overwhelmed" for V -> LOW (V cannot read U's row through RLS)
#  test_review_focus_3_no_rls_user_left_on_pooled_connections: after the checks, every connection acquired from
#     pg_migrated.app_pool (acquire min_size + 1 in turn) returns '' or NULL for
#     current_setting('app.current_user_id', true)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_175_durable_risk_state.py -o "addopts=" -p no:logging -q`, then the realinfra file with `$env:SOLACE_REQUIRE_REALINFRA="1";`
Expected: FAIL: service B scores the message as a new user (LOW) because history lives in per-process dicts (`service.py:120-122, 382-398`), `SafetyService` has no `risk_state_repository` parameter (`TypeError`, which also fails the two review-focus-3 tests at setup), and `_conversation_history` stores raw text.
- [ ] **Step 3: Implement** the repository, the service read/write (load before detection, save after, both inside `with acting_as_user(user_id):`; a repository failure logs WARNING `risk_state_unavailable` and the check proceeds with in-memory state, never failing the check) and the level-based trajectory.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1, the `tests/clinical` command from WS-12.2 and `$env:SOLACE_REQUIRE_REALINFRA="1";` plus the unit template on `tests/realinfra/safety-detection-lexicon`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.16: Caps and TTLs for the remaining in-memory safety state (W5)
**Covers:** MVP-184
**Severity:** medium
**Wave:** 5
**Depends on:** WS-12.15
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/domain/service.py` (`SafetyServiceSettings.max_tracked_users: int = 10_000`, `state_ttl_seconds: int = 86_400`; the risk-state cache and assessment cache become `OrderedDict` LRUs with TTL eviction on write)
- Test: `services/safety_service/tests/test_mvp_184_bounded_state.py`
**Interfaces:**
- Consumes: WS-12.15 `RiskState` cache.
- Produces: `SafetyServiceSettings.max_tracked_users`, `state_ttl_seconds` (env `SAFETY_SERVICE_MAX_TRACKED_USERS`, `SAFETY_SERVICE_STATE_TTL_SECONDS`, request R-12-6).

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_184_bounded_state.py`
```python
#  test_mvp_184_tracked_users_are_capped: SafetyService(SafetyServiceSettings(max_tracked_users=3), ...);
#     checks for 4 distinct users -> len(service._user_risk_history) == 3 and the first user is evicted
#  test_mvp_184_entries_expire: injected clock advances 86_401 s -> a write evicts the stale user
#  test_mvp_184_assessment_cache_is_capped: 4 users -> len(service._assessment_cache) <= 3
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_184_bounded_state.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `max_tracked_users` is ignored (unknown setting) and the dicts grow without bound (`service.py:120-122`, MVP-184).
- [ ] **Step 3: Implement** `_remember(store: OrderedDict, key, value)` used by every per-user cache: move-to-end on write, evict the oldest beyond `max_tracked_users`, and drop entries older than `state_ttl_seconds` on write; an injected `clock` (default `lambda: datetime.now(UTC)`) supports the TTL test.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.17: No private Postgres pool in safety: the contraindication repository takes an injected pool (W0)
**Covers:** MVP-271 (safety part only; the entry is WS-02's, task WS-02.5, request R-02.10)
**Severity:** low
**Wave:** 0
**Depends on:** none
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** D-15 (default assumed: every service pool goes through the `ConnectionPoolManager` TLS floor) and D-19 (default assumed: the contraindication engine stays unwired until WS-12.13 removes it in W5)
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/infrastructure/database.py` (`ContraindicationRepository.__init__(config=None, *, pool=None)`; `connect()` adopts the injected pool and never calls `asyncpg.create_pool`; `disconnect()` drops its reference without closing a pool it does not own; the `database_connection_failed` log carries `error_type=type(exc).__name__` instead of `error=str(e)`)
- Create: `services/safety_service/tests/test_mvp_271_no_private_pool.py` (test)
**Interfaces:**
- Consumes: IF-01 `ServiceRuntime.pg_pool` (the only pool a caller may inject; nothing constructs the repository today, per D-19).
- Produces (R-02.10 for WS-02.5): `ContraindicationRepository(config: DatabaseConfig | None = None, *, pool: asyncpg.Pool | None = None)`; `async connect() -> bool` is True only with an injected pool; `is_connected` follows it; no module under `services/safety_service/src` calls `create_pool`.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_271_no_private_pool.py`
```python
"""MVP-271 (safety part, R-02.10): safety never opens a Postgres pool outside the TLS floor."""
import ast
from pathlib import Path
import services.safety_service.src.infrastructure.database as db

SRC = Path(__file__).resolve().parents[1] / "src"

def test_mvp_271_safety_never_calls_create_pool():
    offenders = [f"{py.relative_to(SRC)}:{node.lineno}" for py in SRC.rglob("*.py")
                 for node in ast.walk(ast.parse(py.read_text(encoding="utf-8")))
                 if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "create_pool"]
    assert offenders == []

class _Pool:
    closed = False
    async def close(self): self.closed = True

async def test_mvp_271_repository_adopts_the_injected_pool(monkeypatch):
    async def _forbidden(*args, **kwargs): raise AssertionError("private pool opened")
    monkeypatch.setattr(db.asyncpg, "create_pool", _forbidden, raising=False)
    pool = _Pool()
    repo = db.ContraindicationRepository(pool=pool)
    assert await repo.connect() is True and repo.is_connected
    await repo.disconnect()
    assert pool.closed is False and repo.is_connected is False
    assert await db.ContraindicationRepository().connect() is False
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_271_no_private_pool.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The AST test lists `infrastructure\database.py:95` (`asyncpg.create_pool` with no `ssl`, the MVP-271 bypass); the injection test fails with `TypeError: ... unexpected keyword argument 'pool'`.
- [ ] **Step 3: Implement** the constructor, `connect()` (`self._pool = self._injected_pool`; `self._initialized = self._pool is not None`; return it, logging WARNING `contraindication_repository_without_pool` when None) and `disconnect()` in `database.py`. Remove the `asyncpg.create_pool` call; keep the `DatabaseConfig` class for the existing exports.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.18: Optional ML libraries stay off unless explicitly enabled, so tests take the image's path
**Covers:** MVP-086
**Severity:** high
**Wave:** 1
**Depends on:** none
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** D-16 (default assumed: no torch, transformers, VADER, spaCy or flashtext in images; lexicon-only safety)
**Real infra:** no
**Files:**
- Modify: `services/safety_service/src/ml/sentiment_analyzer.py`, `services/safety_service/src/ml/pattern_matcher.py`, `services/safety_service/src/ml/keyword_detector.py` (module-level `ML_EXTRAS_ENABLED = os.getenv("SAFETY_ML_EXTRAS_ENABLED", "false").strip().lower() == "true"`; the `torch`, `transformers`, `vaderSentiment`, `spacy` and `flashtext` imports run only when it is true; otherwise `CUDA_AVAILABLE`, `TRANSFORMERS_AVAILABLE`, `VADER_AVAILABLE`, `SPACY_AVAILABLE` and `FLASHTEXT_AVAILABLE` are False without importing the library)
- Modify: `services/safety_service/tests/test_sentiment_analyzer.py`, `services/safety_service/tests/test_pattern_matcher.py`, `services/safety_service/tests/test_keyword_detector.py` (a test whose expected value needs a library result gets `pytest.mark.skipif(not <module>.ML_EXTRAS_ENABLED, reason="D-16: library not in the image")`; no assertion is loosened)
- Create: `services/safety_service/tests/test_mvp_086_ml_extras_parity.py` (test)
**Interfaces:**
- Consumes: none.
- Produces (R-01.15 for WS-01.2): env `SAFETY_ML_EXTRAS_ENABLED` (default false, development only; registered through R-12-6) and `ML_EXTRAS_ENABLED` in the three modules. WS-12.13 deletes the modules and this test in W5.

- [ ] **Step 1: Write the failing test** in `services/safety_service/tests/test_mvp_086_ml_extras_parity.py`
```python
"""MVP-086 parity (R-01.15): this interpreter has torch, transformers, VADER, spaCy and flashtext; the image has none
(D-16). Safety modules take the image's path unless SAFETY_ML_EXTRAS_ENABLED=true. A subprocess keeps reloads
out of this test process."""
import json, os, subprocess, sys
import pytest

MODULES = ("sentiment_analyzer", "pattern_matcher", "keyword_detector")
PROBE = "\n".join([
    "import importlib, json, sys",
    "out = {}",
    f"for name in {MODULES!r}:",
    "    m = importlib.import_module('services.safety_service.src.ml.' + name)",
    "    out[name] = {k: bool(v) for k, v in vars(m).items() if k.endswith('_AVAILABLE') and k != 'TELEMETRY_AVAILABLE'}",
    "out['loaded'] = sorted({'vaderSentiment', 'spacy', 'flashtext'} & set(sys.modules))",
    "print(json.dumps(out))",
])

def _probe(**env_overrides):
    env = {k: v for k, v in os.environ.items() if k != "SAFETY_ML_EXTRAS_ENABLED"} | env_overrides
    done = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])

def test_mvp_086_optional_ml_is_off_by_default():
    out = _probe()
    assert out.pop("loaded") == []
    assert {name: flags for name, flags in out.items() if any(flags.values())} == {}

def test_mvp_086_extras_are_opt_in():
    pytest.importorskip("vaderSentiment")
    assert "vaderSentiment" in _probe(SAFETY_ML_EXTRAS_ENABLED="true")["loaded"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/safety_service/tests/test_mvp_086_ml_extras_parity.py -o "addopts=" -p no:logging -q`
Expected: `test_mvp_086_optional_ml_is_off_by_default` FAILS: the probe reports `VADER_AVAILABLE` (`sentiment_analyzer.py:55-62`), `SPACY_AVAILABLE` and `TRANSFORMERS_AVAILABLE` (`pattern_matcher.py:19-30`) and `FLASHTEXT_AVAILABLE` (`keyword_detector.py:18-22`) as true and `loaded` lists the libraries, so local tests score text on code paths the lexicon-only image never runs (the image's neutral-text ELEVATED 0.5 of MVP-086 stays invisible locally). `test_mvp_086_extras_are_opt_in` already passes (the libraries import unconditionally today); it guards the opt-in.
- [ ] **Step 3: Implement** the `ML_EXTRAS_ENABLED` gate at the top of each module and wrap the optional imports in `if ML_EXTRAS_ENABLED:` blocks; then run the three existing module test files and add the `skipif` marker only to tests that fail because a library result is no longer produced, listing each one in the ledger.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: the safety suite command from WS-12.1, then `tests/integration/test_crisis_flow.py tests/integration/test_safety_service_e2e.py` in one process. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-12.F1: Row-level security on the five unprotected safety tables
**Covers:** MVP-041
**Severity:** high
**Wave:** 1
**Depends on:** WS-12.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-02 (solace_app is NOBYPASSRLS) and D-21 (clinician access is scoped). Added by controller Ruling on 2026-11-03 from WS-02.3's fix round. Its RLS audit found that no plan task enables RLS on `safety_assessments`, `safety_plans`, `safety_events`, `risk_factors` or `contraindication_checks`, all of which hold per-user crisis data.
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/014_safety.py` (WS-12's slot): ENABLE and FORCE RLS, plus one permissive FOR ALL policy per table, using `NULLIF(current_setting('app.current_user_id', true), '')::uuid` in USING and WITH CHECK, exactly like slot 009. A table holding rows without a user (for example lexicon reference data) is excluded only with a written reason.
- Modify: the safety repositories that touch these tables, so every query runs inside `acting_as_user(user_id)` entered before the connection is acquired (`src/solace_infrastructure/postgres.py` reads the context at acquire). Cross-user work such as escalation workers sets the data owner per row.
- Modify: `tests/migrations/test_chain_real_pg.py`. Remove the five names from `RLS_DEFERRED`. This is the only edit allowed in that file, under the Ruling below.
- Test: `tests/realinfra/safety/test_mvp_041_safety_rls.py`
**Interfaces:**
- Consumes: slot 009's grants and default privileges, and `acting_as_user` (WS-07.1).
- Produces: the five tables isolated per user under solace_app. The RLS audit passes with them removed from `RLS_DEFERRED`.

- [ ] **Step 1: Write the failing test.** On real PG16 at head, as solace_app: insert a row for user A in each of the five tables (under `acting_as_user(A)`), then assert user B reads 0 rows and cannot update or delete them, and a read with no user set returns 0 rows. Also assert that the RLS audit passes with the five names removed from `RLS_DEFERRED`.
- [ ] **Step 2: Run it and confirm it fails.** Expected: FAIL; user B reads user A's rows.
- [ ] **Step 3: Implement** the slot-014 policies and the repository context changes.
- [ ] **Step 4: Run to pass** with `$env:SOLACE_REQUIRE_REALINFRA="1"`.
- [ ] **Step 5: Regression:** tests\migrations and services\safety_service\tests.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-12.F1:head"`.

## Cross-workstream requests

| Id | Owner | Change | Files | Needed by |
|---|---|---|---|---|
| R-12-1 | WS-06 | In the W1 phase-A bridge refactor, the safety bridge stops importing `get_event_publisher`/`SafetyEventHandler` from `..events` and stops registering on the in-process publisher; `initialize_notification_bridge(kafka_settings=None, use_mock=False, postgres_pool=None)` only ensures the plane's outbox poller. Safety publishes through `ServiceEventPlane.publish` directly. | `services/safety_service/src/infrastructure/event_bridge.py` | WS-12.8 |
| R-12-2 | WS-06 | Planned in WS-06.10 (crisis-review F11; WS-12.9 depends on it): add `solace_events.service_plane.record_publish_failure(service_name: str) -> None` incrementing `solace_event_publish_failures_total{service}` for callers whose `publish(conn=)` transaction fails outside the plane. The insertion-order half needs no task: `PostgresOutboxStore.get_pending` already claims rows `ORDER BY created_at ASC` (`src/solace_events/postgres_stores.py:173`). | `src/solace_events/service_plane.py` | WS-12.9 |
| R-12-3 | WS-06, WS-04 | Allow `ServiceEventPlane.subscribe()` after `start_service_runtime()` and before the lifespan calls `await runtime.event_plane.start()` (idempotent), so `register_user_erasure_consumer` can run in the lifespan. | `src/solace_events/service_plane.py`, `src/solace_infrastructure/runtime/bootstrap.py` | WS-12.12 |
| R-12-4 | WS-01 | Pin PyYAML and hypothesis in the lock for every image and test env that imports `solace_clinical`; images copy `src/solace_clinical/crisis_lexicon_v1.yaml` (package data); register the `clinical` marker. | `requirements.in`, `requirements.lock`, `requirements-dev.in`, `requirements-dev.lock`, `pytest.ini` | WS-12.3 |
| R-12-5 | WS-04 | Acknowledged, no task: WS-04.3 already exports the IF-18 counter `solace_crisis_detected_total{source,level}` as `solace_infrastructure.runtime.metrics.CRISIS_DETECTED` (the name WS-10, WS-15 and WS-16 use). WS-12.9 now increments that object with `source="safety"` and depends on WS-04.3; the name `CRISIS_DETECTED_TOTAL` this row first asked for is withdrawn (crisis-review F11). | `src/solace_infrastructure/runtime/metrics.py` | WS-12.9 |
| R-12-6 | WS-03 | Env registry entries (safety-service, not secret, not required): `CRISIS_WATCH_SECONDS=1800`, `LLM_ASSESSOR_TIMEOUT_SECONDS=1.5`, `SAFETY_SERVICE_MAX_TRACKED_USERS=10000`, `SAFETY_SERVICE_STATE_TTL_SECONDS=86400`, `SAFETY_ML_EXTRAS_ENABLED=false` (development only, never set in compose; retired with WS-12.13 in W5); mark retired after WS-12.9: `SAFETY_KAFKA_ENABLED`, `SAFETY_KAFKA_BOOTSTRAP_SERVERS`, `SAFETY_KAFKA_USE_MOCK`, `ESCALATION_NOTIFICATION_SERVICE_URL`, `ESCALATION_DASHBOARD_BASE_URL`, `CLINICIAN_REGISTRY_*`, `SAFETY_REPO_*`, `SAFETY_EVENTS_*`, `CRISIS_KEYWORD_WEIGHT`, `CRISIS_SENTIMENT_WEIGHT`, `CRISIS_PATTERN_WEIGHT`, `CRISIS_HISTORY_WEIGHT`. | `deploy/env/env-contract.yaml`, `docker-compose.yml` | WS-12.6, WS-12.9, WS-12.16, WS-12.18 |
| R-12-7 | WS-04 | Planned as WS-04.15 (W0, WS-09's CR-8): keep `tests/integration/test_phi_encryption_activation.py` with `safety_service` in `_PHI_ENCRYPTING` until WS-12.12 moves it to `_PHI_VIA_RUNTIME` in W3. `configure_phi_encryption` stays importable in `base_models.py` until the W3 barrier through WS-09's CR-11 to WS-02 (no separate request). | `tests/integration/test_phi_encryption_activation.py` | WS-12.12 |
| R-12-8 | WS-15 | MVP-174 therapy half: replace the hard-coded `contraindications=[]` in technique selection with flags derived from the session's crisis level (`high_suicide_risk` at HIGH/CRITICAL), or delete the strict-check branch and document the risk-level filter as the only guard (D-19). | `services/therapy_service/src/domain/service.py`, `services/therapy_service/src/domain/technique_selector.py` | WS-12.13 |
| R-12-9 | WS-05 | After WS-12.1, `tests/system/test_smoke.py::test_safety_service_health` must assert the IF-01 `/health` body (`status` `ready`/`not_ready`, `service`, `checks`). | `tests/system/test_smoke.py` | WS-12.1 |
| R-12-10 | WS-10 | Already planned as WS-10.16 (W2): write `tests/contracts/orchestrator__safety/check.json` from IF-06 (`turn_id` and a client `session_id` string in `body_example`; response schema with `level`, `categories`, `resources`, `resources_version`, `detection_layer`); the provider test `test_mvp_017_contract_orchestrator_check.py` written by WS-12.10 then runs it. | `tests/contracts/orchestrator__safety/check.json` | WS-12.10 |
| R-12-11 | WS-04 | D-29 alert for an unsupported locale: list `solace_crisis_resources_fallback_total{reason}` (reason `unsupported_locale` or `unknown_locale`; no user or text labels) in IF-18's standard metrics as registered by `solace_clinical.resources.CRISIS_RESOURCES_FALLBACK` (not re-registered in `solace_infrastructure.runtime.metrics`; prometheus_client is already in every service image), and add the rule `CrisisResourcesLocaleFallback` (`increase(solace_crisis_resources_fallback_total[15m]) > 0`, severity warning, routed to `ALERTMANAGER_EMAIL_TO`) to `infrastructure/prometheus/rules/solace.yml` with a promtool test that fires on one increment, no later than the W4 rules batch (before the W5 gate). The alert tells the operator a user outside the populated locales reached the crisis path, the input for the D-31 cohort decision. | `infrastructure/prometheus/rules/solace.yml`, `infrastructure/prometheus/tests/solace_rules_test.yml` | WS-12.2 |
| R-12-12 | WS-03 | `docs/KNOWN-LIMITATIONS.md` gains a D-29 entry: crisis resources are a per-locale registry in `solace_clinical` with only en-US populated; any other or unknown locale is shown the generic fallback (local emergency number plus the Find A Helpline directory) and fires `CrisisResourcesLocaleFallback`; admitting a non-US user is an operator and cohort decision (D-31). Replace any "US-only" or "non-US users are outside the pilot" wording. No env registry entry is needed (`DEFAULT_LOCALE` is a code constant). | `docs/KNOWN-LIMITATIONS.md` | WS-12.2 |

## New interfaces

- `solace_clinical.level_rank(level: CrisisLevel) -> int` (NONE 0, LOW 1, ELEVATED 2, HIGH 3, CRITICAL 4).
- D-29 locale registry (WS-12.2): `DEFAULT_LOCALE`, `UNDETERMINED_LOCALE`, `supported_locales()`, `RESOURCE_VERSIONS`, `FALLBACK_RESOURCES_VERSION`, `canonical_locale(tag)`, `resolve_locale(claim, requested=None)`, `CrisisResourceSet`, `crisis_resource_set(locale)`, `resources_version(locale)`, counter `CRISIS_RESOURCES_FALLBACK` (`solace_crisis_resources_fallback_total{reason}`) and WARNING `crisis_resources_locale_fallback`. The locale contract built on them: the access-token claim `loc` (WS-08.21, from the profile locale WS-14.22 validates; default `en-US`) reaches the orchestrator as `AuthenticatedUser.locale` (WS-07 request in WS-11's plan), `run_turn(locale=)` and `crisis_for_refused_message(locale=)` resolve it with `resolve_locale` (WS-10.5), and therapy receives it in its IF-11 message body (WS-11.21, WS-15.23). No database or network call is needed at screen time.
- `solace_clinical.phrases`: `load_phrase_set(path=None) -> PhraseSet`, `PhraseSet`, `PositivePhrase`, `NegativePhrase`, `PhraseFailure`, `check_detector(detect, phrase_set=None) -> list[PhraseFailure]`, `async check_detector_async(detect, phrase_set=None)`. Detector owners (WS-10, WS-15, WS-16, WS-17) call these from their own suites instead of parsing the YAML.
- `/check` response additions beyond IF-06: `turn_id` (echo) and `resources_version`; legacy fields (`crisis_level`, `is_safe`, `crisis_resources`, ...) are retained.
- Safety composition seams: `services.safety_service.src.domain.factory.build_crisis_detector(*, llm_assessor=None, settings=None)` and `build_escalation_repository(pg_pool, *, environment)`.
- `escalations` columns from slot 014 (`correlation_id`, `source`, `categories`, `detection_layer`, `encryption_key_id`, `pseudonymized_at`, unique auto decision per turn) and the W5 table `safety_user_risk_state` (RLS).
- Requested (R-12-2, R-12-5): `solace_events.service_plane.record_publish_failure(service_name)`, `solace_infrastructure.runtime.metrics.CRISIS_DETECTED_TOTAL`.
- Settings: `CRISIS_WATCH_SECONDS`, `LLM_ASSESSOR_TIMEOUT_SECONDS`, `SAFETY_SERVICE_MAX_TRACKED_USERS`, `SAFETY_SERVICE_STATE_TTL_SECONDS`, `SAFETY_ML_EXTRAS_ENABLED` (WS-12.18, development only, removed in W5).
- `services.safety_service.src.main._wire_llm(runtime, llm_client) -> LLMAssessor | None` (WS-12.6, CR-7) and `ContraindicationRepository(config=None, *, pool=None)` (WS-12.17, removed in W5).
- `screen_text` matching refinements (WS-12.3): symbol stripping, a de-leeted second variant and a three-token negation window; IF-07's signature is unchanged.

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-030 | blocker | WS-12.2, WS-12.5, WS-12.6 |
| MVP-081 | high | WS-12.4, WS-12.8, WS-12.9 |
| MVP-083 | high | WS-12.6 |
| MVP-084 | high | WS-12.3, WS-12.6 |
| MVP-085 | high | WS-12.9 |
| MVP-086 | high | WS-12.3, WS-12.6, WS-12.18 (test/image parity) (notification side: WS-13.8) |
| MVP-087 | high | WS-12.9, WS-12.10 |
| MVP-088 | high | WS-12.1, WS-12.4, WS-12.9 |
| MVP-089 | high | WS-12.12 |
| MVP-090 | high | WS-12.6 |
| MVP-174 | medium | WS-12.13 (therapy half: R-12-8) |
| MVP-175 | medium | WS-12.14, WS-12.15 |
| MVP-176 | medium | WS-12.7 |
| MVP-177 | medium | WS-12.4, WS-12.9 |
| MVP-178 | medium | WS-12.3, WS-12.6 |
| MVP-179 | medium | WS-12.9 |
| MVP-181 | medium | WS-12.4, WS-12.9 (notification side: WS-13.6, WS-13.8) |
| MVP-182 | medium | WS-12.3, WS-12.6 |
| MVP-183 | medium | WS-12.5 |
| MVP-184 | medium | WS-12.16 (other services' stores belong to their owners) |
| MVP-309 | low | WS-12.11 |

Supporting work for other owners: MVP-271 safety bypass (WS-02's id, WS-12.17, R-02.10); MVP-357 producer and provider halves (WS-05's id, WS-12.8 and WS-12.10); MVP-055 safety adoption (WS-09's id, WS-12.12, CR-10).

**Incoming requests**
- WS-01 R-01.15 (WS-01.2): WS-12.18, W1 phase B.
- WS-02 R-02.10 (WS-02.5): WS-12.17, W0 phase B.
- WS-04 (WS-04.5): WS-12.1, W0.
- WS-05 request 8 (WS-05.8): WS-12.3 phrase set. WS-05 request 9 (WS-05.10): WS-12.8 `test_mvp_357_event_contracts.py`, W1. WS-05 request 10 (provider `verify_provider`): WS-12.10 `test_mvp_017_contract_orchestrator_check.py`.
- WS-09 CR-7 (WS-09.4): WS-12.6 `_wire_llm`, W1; WS-12.5 already maps `LLMUnavailableError` to an unreliable result. WS-09 CR-10 (WS-09.15): WS-12.12, W3 (WS-12 has no W2 wave; WS-09.15 runs in W3 phase B after it).
- WS-04 (WS-04.15): WS-12.12 moves `safety_service` to `_PHI_VIA_RUNTIME`, W3.
- WS-10 (WS-10.16): WS-12.10, provider test written in W1, live from W2.
- WS-13 R-13-7 (WS-12.8): WS-12.8 now reads `tests/contracts/events/safety__notification/` and depends on WS-05.7.

**Interface consistency**
- IF-06 (produced): `/check` request and response match the skeleton; every safety event carries `metadata.correlation_id = turn_id` and `source_service = "safety-service"`; `trigger_indicators` and `escalation_reason` carry category codes only; at most one CrisisDetected per turn (unique auto decision per correlation id); CrisisDetected uses `escalation_action="page_on_call"` and `requires_human_review=True`. WS-13 consumes exactly these fields (WS-13.6, WS-13.7); WS-05.7 writes the `safety__notification` event contracts, WS-12.8 verifies the producer half with `verify_event_producer` and WS-13.13 the consumer half.
- IF-07 (produced): names, dataclasses, YAML keys, phrase-set format and marker match the skeleton; `level_rank` and the phrase helpers are additions. D-29 (2026-10-19): `crisis_resources(locale)` keeps its name and en-US output, now also accepts `None` (meaning en-US), and returns the generic fallback for any other locale; `RESOURCES_VERSION` stays the en-US version, so WS-05.6's assertion is unchanged; the registry names above are additions.
- IF-05: publish only through `ServiceEventPlane.publish(event, *, conn=)`; topics from `solace_events.schemas`; no Kafka DLQ topics; erasure subscription through IF-16.
- IF-04: `service:read:safety` for all read endpoints, `service:escalate` for `/escalate`; safety's own `service:write:notification` grant becomes unused (WS-07 removes it per D-28).
- IF-13: escalation `reason` is encrypted with AAD `escalations.reason:<owner>`; `context` is always `{}` so nothing else needs the codec; `safety_assessments` is not written by safety (only erased).
- IF-01: SPEC declares postgres, `publish_consume`, `handles_phi`, `audited`; no local handlers or health routes remain after WS-12.1.

**Open risks**
- Lexicon recall depends on the clinician-reviewed phrase set. WS-12.3 pins a provisional adversarial set (leetspeak, misspellings, slang, emoji, Spanish and code-switched text, hypothetical and quoted speech, long clauses with an earlier negation) and WS-12.6 pins a multi-turn build-up, but unseen spellings, other languages and slower multi-turn build-up remain the main false-negative surface. The history watch and the raise-only LLM mitigate; D-05 sign-off gates W5.
- The three-token negation window and the de-leeted variant over-detect some benign text (for example "I don't really want to die" pages); accepted under D-05 and listed in §11.
- IF-07's "hopelessness at most ELEVATED" stops paging for hopelessness-only messages that the old weighted fusion sometimes paged; recorded in CLINICAL-VALIDATION §11 for explicit sign-off.
- Never negating first-person intent pages on "I'm not going to kill myself"; accepted over-detection under D-05, monitored through page volume.
- The deprecated no-op `register_handler` keeps the not-yet-refactored bridge importable; if R-12-1 lands differently the boot smoke catches it.
- The realinfra outbox assertions assume slot 010 keeps the `event_outbox(event_type, event_payload, source_service)` columns; adjust the queries to WS-06's final slot if it renames them.
- Until WS-12.12 lands in W3, deleting `configure_phi_encryption` from `base_models.py` (WS-09's CR-11 cleanup by WS-02) would break safety's import of it; CR-11 now says "no earlier than the W3 barrier".
- The provider test of WS-12.10 skips until WS-10.16 lands in W2; the W2 barrier records that it ran (WS-05.10's coverage guard also requires the file).
- D-29: only en-US is populated. The fallback copy and the Find A Helpline URL are provisional until the D-29 clinician sign-off. A well-formed but language-only profile locale (`en`) has no country, so it gets the fallback and fires the alert; that is deliberate, because the country decides which numbers are correct.

**Review focus** (skeleton order 1-5)
- 1 ingress false negatives: `test_review_focus_1_adversarial_disclosures_reach_their_floor`, `..._adversarial_benign_text_stays_low`, `..._adversarial_cases_are_in_the_shared_phrase_set` and the later-clause property (WS-12.3); `test_review_focus_1_risk_building_across_turns_raises_one_band` and `test_mvp_084_negation_is_clause_bounded` (WS-12.6).
- 2 alert-path partial failures: `test_mvp_081_row_and_outbox_commit_together` and `test_mvp_088_outbox_failure_rolls_back_the_row` on postgres:16 (WS-12.9). Delivery, reapers and acknowledgement races are WS-13's code.
- 3 foreign identifiers and context-less work: `test_review_focus_3_foreign_user_never_inherits_history` and `test_review_focus_3_no_rls_user_left_on_pooled_connections` (WS-12.15); `test_mvp_175_no_context_sees_nothing_and_cannot_write` (WS-12.14).
- 4 PHI through unplanned sinks: `test_review_focus_4_failure_logs_carry_no_message_or_exception_text` and the no-context, no-cache prompt tests (WS-12.5); `test_if06_crisis_payload_carries_category_codes_only` and the free-text reason test (WS-12.8).
- 5 strict boots with partial configuration: `test_mvp_088_strict_boot_without_postgres_fails[staging|production]` and `test_mvp_088_strict_repository_requires_pool` (WS-12.1); staging is as strict as production. (`test_strict_without_kafka_fails_loud` lives until WS-12.9 hands the Kafka guard to the IF-01 event plane.)

REVIEW_FOCUS: 1=WS-12.3,WS-12.6; 2=WS-12.9; 3=WS-12.15,WS-12.14; 4=WS-12.5,WS-12.8; 5=WS-12.1

## Review responses

- E: WS-12.11 modifies `test_if06_check_contract.py`, not created by a dependency: fixed (WS-12.10, a dependency of WS-12.11, now lists it under Create; it was under Test, which the checker does not count as a creation).
- E: WS-12.13 modifies `src/ml/keyword_detector.py` (missing): fixed (full path `services/safety_service/src/ml/keyword_detector.py`, which exists).
- E: WS-12.13 modifies `src/ml/pattern_matcher.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `src/ml/sentiment_analyzer.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `src/db/contraindication_db.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `src/db/__init__.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `src/observability/__init__.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `src/infrastructure/database.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `tests/test_contraindication.py` (missing): fixed (full path under `services/safety_service/tests/`).
- E: WS-12.13 modifies `tests/test_keyword_detector.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `tests/test_pattern_matcher.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `tests/test_sentiment_analyzer.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `tests/test_batch_3_3_integration.py` (missing): fixed (full service path).
- E: WS-12.13 modifies `test_mvp_083_084_086_178_182_detector_fusion.py`, not created earlier: fixed (WS-12.6, W1, now lists its three test files under Create; `test_mvp_090_history_signal.py` also gets its full path in WS-12.13).
- E: request from WS-01 R-01.15 (ML branches unreachable unless enabled): fixed (new task WS-12.18, W1 phase B, `SAFETY_ML_EXTRAS_ENABLED` gate in sentiment_analyzer, pattern_matcher and keyword_detector; RED verified against this interpreter, which has vaderSentiment, spacy, transformers, torch and flashtext installed).
- E: request from WS-02 R-02.10 (safety `database.py` calls `asyncpg.create_pool` without ssl): fixed (new task WS-12.17, W0 phase B; `database.py:95` is the only `create_pool` call in safety src; the repository is unwired, so it takes an injected pool instead).
- E: request from WS-05 (producer `verify_event_producer` tests, for WS-05.10): fixed (WS-12.8 creates `services/safety_service/tests/test_mvp_357_event_contracts.py`, the file the request names, against `tests/contracts/events/safety__notification/`).
- E: request from WS-09 CR-7 (llm readiness, `LLMUnavailableError`): fixed (WS-12.6 `_wire_llm` registers `("llm", llm_client.readiness, critical=False)` with a unit test; WS-12.5 already returns an unreliable result on any exception). The request row names no file path, so the checker cannot match it to a task.
- E: request from WS-09 CR-10 (IF-13 adoption in the ADOPTERS module): fixed (WS-12.12, W3, because WS-12 has no W2 wave and WS-09.15 runs in W3; the codec lives in `domain/escalation.py`, which WS-09.15's ADOPTERS map now lists for WS-12 instead of `infrastructure/repository.py`, whose Postgres repositories nothing wires; WS-12.12 also removes the whole `configure_phi_encryption`/`FieldEncryptor` block at `main.py:116-121`). The row names no file path, so the checker cannot match it.
- E: request from WS-10 (provider `verify_provider` of `orchestrator__safety/check.json`, for WS-10.16): fixed (WS-12.10 creates `services/safety_service/tests/test_mvp_017_contract_orchestrator_check.py`, which skips until WS-10.16 writes the file in W2; R-12-10 now points at WS-10.16 instead of WS-11, because WS-10.16 creates that file).
- E: request from WS-13 R-13-7 (contract dir `events/safety__notification/`, dependency WS-05.7): fixed (WS-12.8 depends on WS-05.7 and reads that directory; the file is named `test_mvp_357_event_contracts.py` to satisfy WS-05's request, and R-13-7 in WS-13's plan now names it).
- W: WS-12.9 edits `tests/integration/test_safety_service_e2e.py` (WS-05's dir): rejected: the tests/integration protocol lets the owner of the tested code edit an existing file, and that file imports only `services.safety_service.src` modules (lines 25-36) and passes `clinician_registry=` at line 91, which WS-12.9 removes.
- W: WS-12.12 edits `tests/integration/test_phi_encryption_activation.py` (WS-05's dir): rejected: the edit is sanctioned and requested. WS-04.15 (W0, CR-8) keeps the file and its request to WS-12 (WS-04 plan, cross-workstream row for WS-04.15) assigns safety's move from `_PHI_ENCRYPTING` to `_PHI_VIA_RUNTIME` to WS-12.12; the entry locks safety's `main.py`, WS-12's code, which the tests/integration protocol requires. The task now names exactly that one-entry change, and the waves (W0 vs W3) keep the two edits apart.
- E (re-check, new): request from WS-04 for WS-04.15 (add `safety_service` to `_PHI_VIA_RUNTIME` when it leaves `_PHI_ENCRYPTING`): fixed (WS-12.12 Files and Produces; Step 5 runs the activation lock; R-12-7 now points at WS-04.15 and drops the retired-file wording).
- W: WS-12.13 edits `src/ml/keyword_detector.py`, no owner: fixed (full path, owned by WS-12 via `services/safety_service/src/ml`).
- W: WS-12.13 edits `src/db/__init__.py`, no owner: fixed (full path, owned via `services/safety_service/src/db`).
- W: WS-12.13 edits `src/ml/pattern_matcher.py`, no owner: fixed (full path).
- W: WS-12.13 edits `services/safety_service`, no owner: fixed (the bare directory token "(all under services/safety_service/)" is gone).
- W: WS-12.13 edits `src/observability/__init__.py`, no owner: fixed (full path, owned via `services/safety_service/src/observability`).
- W: WS-12.13 edits `src/ml/sentiment_analyzer.py`, no owner: fixed (full path).
- W: WS-12.13 edits `src/infrastructure/database.py`, no owner: fixed (full path, owned via `services/safety_service/src/infrastructure/database.py`).
- W: WS-12.13 edits `src/db/contraindication_db.py`, no owner: fixed (full path).
- Review focus (task instruction): fixed (focus 1 had no pinning test; WS-12.3 now pins leetspeak, misspellings, slang, emoji, Spanish and code-switched text, hypothetical and quoted speech and earlier-clause negation; WS-12.6 pins multi-turn build-up; focus 3 and 4 pins added to WS-12.14, WS-12.15 and WS-12.5; WS-12.14's realinfra GUC is now transaction-local so it no longer leaves an RLS user on a pooled connection).
- Reconciliation 10 (WS-05 request 9: every crisis producer runs `verify_event_producer` against `tests/contracts/events/<producer>__notification/`): checked. WS-12.8 already creates `services/safety_service/tests/test_mvp_357_event_contracts.py` with that check against `safety__notification/` (crisis, escalation, assessment); its Produces line now also names WS-05.10, the coverage guard that looks for it.
- crisis-review F2: fixed (safety half). Verified at WS-12.9 Step 3 and `test_mvp_088_record_failure_publishes_nothing_and_still_answers`: when the record-and-publish transaction failed, `/check` answered and published nothing, and nothing retried. WS-12.9 adds a background record retry (one entry per correlation id, at most 10,000, backoff 1 s doubling to 60 s, idempotent through `ux_escalations_auto_correlation`; background task `safety_record_retry`), keeps D-28's outbox-only rule, logs `user_id` with the failure, and adds a unit test and a real-infra assertion.
- crisis-review F4: fixed (in WS-05.6). WS-12.2 is unchanged: "Call or text 988" is the D-29 user-facing text, and the E2E assertion was the defect.
- crisis-review F10: fixed. The format now has one definition: WS-12.2 adds `format_escalation_reason` and `parse_escalation_reason` to `solace_clinical` (IF-07, with tests); WS-12.8 passes only values `parse_escalation_reason` accepts and WS-12.9 stores `format_escalation_reason("auto", level, categories)`; WS-05.7's contract rule, WS-06.12's redaction and WS-13.6's parser call the same function.
- crisis-review F11: fixed. Verified: WS-04.3 exports `CRISIS_DETECTED`, not `CRISIS_DETECTED_TOTAL` (WS-04 plan, New interfaces; WS-15 and WS-16 already switched), and no WS-06 task planned `record_publish_failure`. WS-12.9 now increments `CRISIS_DETECTED` with `source="safety"` (like `orchestrator`, `therapy`, `diagnosis`), depends on WS-04.3 and WS-06.10 (which now produces `record_publish_failure`), and `test_crisis_review_f11_counts_use_the_shared_metric_objects` pins both counters. R-12-2 and R-12-5 are updated.
- decision 2026-10-19 D-29: crisis resources became a per-locale, versioned registry in `solace_clinical` (WS-12.2: Decision gate, Interfaces, test and Step 3). Only en-US is populated, and `crisis_resources("en-US")` and `crisis_resources()` return exactly today's three entries. `RESOURCES_VERSION` stays the en-US version, so WS-05.6 needs no change. Versions are kept per locale in `RESOURCE_VERSIONS`. An unsupported or unknown locale now gets the generic fallback (`fallback-1.0.0`); it is never empty and never raises, logs WARNING `crisis_resources_locale_fallback` without PHI, and counts `solace_crisis_resources_fallback_total{reason}`. The old assertion `crisis_resources("fr-FR") == en-US` ("pilot is US-only") is gone. New names: `DEFAULT_LOCALE`, `UNDETERMINED_LOCALE`, `supported_locales`, `canonical_locale`, `resolve_locale` (a missing claim means en-US; a request refines only within the supported set and never changes the country), `CrisisResourceSet`, `crisis_resource_set`, `resources_version`. New RED tests: en-US unchanged, missing locale means en-US, unsupported or unknown gets the fallback with log and metric, request refinement, per-locale versions, and a totality property.
- decision 2026-10-19 D-29: WS-12.3's §11 now records the registry, the provisional fallback and the add-a-locale rule for sign-off. WS-12.10 states that safety's `/check` and `/resources` serve the `DEFAULT_LOCALE` set, which is informational only because the orchestrator renders its own locale-resolved resources. New requests: R-12-11 to WS-04 (IF-18 metric listing and the `CrisisResourcesLocaleFallback` alert rule) and R-12-12 to WS-03 (a KNOWN-LIMITATIONS entry). No env setting is needed, so there is no IF-02 request: `DEFAULT_LOCALE` is a clinical constant. The file map, Architecture, New interfaces, Interface consistency and Open risks are updated.