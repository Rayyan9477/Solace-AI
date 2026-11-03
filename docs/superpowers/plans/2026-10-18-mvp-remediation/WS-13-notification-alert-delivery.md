# WS-13: Notification: clinician alert delivery, crisis episodes and acknowledgement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** notification-service boots on the IF-01 runtime, consumes `solace.safety` through the IF-05 event plane durably and idempotently, and is the only pager and the system of record for crisis episodes: it resolves real recipients through IF-08 with a service token, falls back when delivery fails, coalesces duplicates per episode, records every attempt without bodies, accepts a one-time acknowledgement, re-pages when nobody acknowledges, keeps crisis paging ahead of monitoring mail and reports real channel state.

**Architecture:** `consumers.py` subscribes one dispatcher to `solace.safety` with group `solace-group-notifications`; HIGH/CRITICAL crisis, escalation and assessment events become `CrisisSignal`s for `CrisisPager` (`src/domain/crisis_paging.py`), and ELEVATED monitoring only upserts a durable per-user `risk_alert_windows` row that a background digest mails later, so crisis handling never waits on monitoring mail. `CrisisPager` coalesces signals into `crisis_alerts` episodes (by turn correlation id, then by user within `NOTIFICATION_ALERT_COALESCE_SECONDS`) under a per-user advisory lock, pages the care team, then the on-call roster, then `NOTIFICATION_FALLBACK_ONCALL_EMAIL`, one bounded attempt per recipient, with `notifications`/`delivery_attempts` (slot 015) as the per-recipient idempotency record. When no clinician delivery succeeds it raises `RetryableEventError`, so retry, DLQ, reaper and idempotency come from the plane. A per-recipient one-time link and a clinician API acknowledge or resolve episodes, and a runtime-registered re-pager pages the backup on-call and the fallback mailbox for open episodes.

**Covers:** 19 register entries (blocker 1, high 3, medium 8, low 7).

**Waves:** 0 (module path, IF-01 adoption), 1 (slot 015, repositories, recipients, content, paging, dispatcher, plane consumer, acknowledgement, re-paging, health, send endpoints), 3 (erasure by pseudonymization, audit trail), 4 (MFA on the clinician routes), 5 (crisis SMS, residual channel defects). WS-13.3 is phase A (W1 schema lane); every other task is phase B. W1 phase-B execution order: WS-13.4, 13.5, 13.7, 13.6, 13.13, 13.8, 13.9, 13.10, 13.11, 13.12.

**Depends on:**
- WS-01: one module path for the merged image (`src.main`, WS-01.5) and the import-root and lint guards (WS-01.7).
- WS-02: slot scaffold with an empty `015_notification` and grants through default privileges (WS-02.3).
- WS-03: IF-02 notification names and the Caddy acknowledgement route (WS-03.6, W1 phase A); `ENCRYPTION_SEARCH_HASH_SALT` for notification-service in W3 (request R-13-2).
- WS-04: IF-01 runtime and the conformance table (WS-04.4, WS-04.5); service tokens and the event plane wired into the runtime (WS-04.6); audit wiring (WS-04.10).
- WS-05: IF-19 fixtures (WS-05.2) and IF-20 contract helpers plus the event contracts (WS-05.7).
- WS-06: IF-05 consumer retry/DLQ/idempotency and `solace_events.errors` (WS-06.3), the event plane (WS-06.8) and bridges on the plane (WS-06.9).
- WS-07: `service_auth_headers` and the `READ_ONCALL_ROSTER`/`WRITE_NOTIFICATION` grants (WS-07.3), receiver-side `require_service_permission` (WS-07.4), `require_mfa_for_roles` (WS-07.11).
- WS-09: audit runtime (WS-09.11) and the salt-only crisis pseudonym (request R-13-3).
- WS-12: `solace_clinical.CATEGORIES` and `level_rank` (WS-12.2); slot 014 precedes slot 015 in the W1 schema lane (WS-12.4).
- WS-14: IF-08 internal roster and care-team routes (W1 phase B) and the IF-16 erasure helper (W3 phase A). Unit and real-infra tests here use stub rosters; the W1 e2e needs WS-14's implementation.

**Owned paths:** `services/notification-service/src/domain`, `services/notification-service/src/api.py`, `services/notification-service/src/consumers.py`, `services/notification-service/src/events.py`, `services/notification-service/src/main.py`, `services/notification-service/src/__init__.py`, `services/notification-service/src/infrastructure/__init__.py`, `services/notification-service/src/infrastructure/delivery_repository.py`, `services/notification-service/src/infrastructure/crisis_alert_repository.py`, `services/notification-service/tests`, `migrations/versions/015_notification.py`, `docs/runbooks/crisis-escalation.md`; new tests in `tests/realinfra/notification-alert-delivery/`; the consumer contract directory `tests/contracts/notification__user/` (IF-20: the consumer writes it).

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

**Conventions used by every task below.**
- Unit command for a path P: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest P -o "addopts=" -p no:logging -q`. Real-infra commands add `$env:SOLACE_REQUIRE_REALINFRA="1";` before the interpreter. Each command below is written out in full.
- From WS-13.1 on, notification tests put the service root `services/notification-service` on `sys.path` (tests/conftest.py) and import `src.main`, `src.api`, `src.consumers`, `src.domain.<module>`, `src.infrastructure.<module>`. The shared packages stay top-level (`solace_events`, `solace_security`, `solace_infrastructure`, `solace_clinical`, `solace_testing`). Because `src` is bound to notification-service, `services/notification-service/tests` and `tests/realinfra/notification-alert-delivery` each run in their own pytest process.
- `asyncio_mode = auto` stays in pytest.ini, so `async def test_...` needs no decorator.
- Never put an email address, phone number, display name, message text, token or exception text into a log line, metric label, event, DB row or test output. Logs carry `alert_ref` (first 8 characters of the alert id), `recipient_role`, `outcome` and `error_type=type(exc).__name__`.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/notification-service/src/main.py` | Modify | SPEC and IF-01 lifespan; settings; composition (`_create_notification_service`, `build_stores`, `build_alerting`); background tasks; readiness; erasure consumer | WS-13.1, 13.2, 13.7, 13.8, 13.9, 13.10, 13.11, 13.12, 13.14, 13.18 |
| `services/notification-service/src/api.py` | Modify | Package-relative imports; app-state dependencies; crisis acknowledgement router; health from the real registry; service-permission send routes; no stub shims | WS-13.1, 13.2, 13.9, 13.11, 13.12, 13.15, 13.16, 13.18 |
| `services/notification-service/src/consumers.py` | Modify | Legacy consumer lifecycle fix (W0); `handle_safety_event` dispatcher; `register_safety_handlers` on the plane | WS-13.1, 13.2, 13.13, 13.8 |
| `services/notification-service/src/events.py` | Modify | Bare `solace_events` import | WS-13.1 |
| `services/notification-service/src/domain/episodes.py` | Create | `CrisisEpisode`, `RiskWindow`, `AckOutcome`, `AckResult`, `hash_ack_token`; `EpisodeAction`, `decide_episode` | WS-13.4, 13.6 |
| `services/notification-service/src/domain/alert_settings.py` | Create | `AlertSettings` (IF-02 names) with strict validation | WS-13.5 |
| `services/notification-service/src/domain/recipients.py` | Create | IF-08 `RosterClient`, `Clinician`, `CareTeam`, `AlertRecipient`, `recipient_hash` | WS-13.5 |
| `services/notification-service/src/domain/crisis_paging.py` | Create | `CrisisSignal`, `CrisisPager` (paging, updates, re-paging, audit, SMS) | WS-13.6, 13.10, 13.15, 13.17 |
| `services/notification-service/src/domain/alert_metrics.py` | Create | IF-09 metrics plus `solace_monitoring_alert_delivery_total` | WS-13.6 |
| `services/notification-service/src/domain/monitoring.py` | Create | `RiskDigest` and `run_periodic` | WS-13.8 |
| `services/notification-service/src/domain/ack_pages.py` | Create | Confirm and result HTML for the one-time link | WS-13.9 |
| `services/notification-service/src/domain/erasure.py` | Create | `build_notification_erasure` (D-24) | WS-13.14 |
| `services/notification-service/src/domain/templates.py` | Modify | Text/HTML engines; IF-09 crisis, update and risk-digest templates; the `password_reset` contract WS-08.18 relies on (R-15 of WS-08) | WS-13.7 |
| `services/notification-service/src/domain/channels.py` | Modify | TLS mode, Date and Message-ID, RFC 5322 addresses, bounded attempts, crisis SMS, push disabled | WS-13.7, 13.17, 13.18 |
| `services/notification-service/src/domain/service.py` | Modify | `render`, `deliver_one`, `channel_registry`, delivery records for API sends, failed unsupported targets | WS-13.7, 13.12, 13.18 |
| `services/notification-service/src/infrastructure/crisis_alert_repository.py` | Create | Episodes, ack tokens, risk windows (Postgres and in-memory) | WS-13.4, 13.14 |
| `services/notification-service/src/infrastructure/delivery_repository.py` | Create | `notifications`/`delivery_attempts` claims and attempts (Postgres and in-memory) | WS-13.4, 13.14 |
| `migrations/versions/015_notification.py` | Modify | Slot 015: `crisis_alerts`, `crisis_alert_ack_tokens`, `risk_alert_windows`, body-less notification records | WS-13.3 |
| `docs/runbooks/crisis-escalation.md` | Create | On-call and ops runbook for episodes, links, re-pages, fallback and DLQ | WS-13.10 |
| `tests/contracts/notification__user/*.json` | Create | IF-20 consumer contracts for the IF-08 routes | WS-13.5 |
| `services/notification-service/tests/conftest.py` | Modify | Service root on `sys.path`; `src.` imports | WS-13.1 |
| `services/notification-service/tests/notification_fakes.py` | Create | `FixedClock`, `FakeRoster`, `FakeNotifier`, builders, `make_pager` | WS-13.4, 13.6, 13.17 |
| `services/notification-service/tests/notification_repo_contract.py` | Create | Repository contract checks shared by the unit and real-infra tests | WS-13.4 |
| `services/notification-service/tests/test_*.py` (existing) | Modify | `src.` imports; legacy consumer tests retired; skipped API tests enabled | WS-13.1, 13.2, 13.7, 13.8, 13.11, 13.12 |
| `services/notification-service/tests/test_mvp_082_acknowledgement_api.py` | Create, then Modify | Acknowledgement API tests; MFA-verified clinician fixture from W4 | WS-13.9, 13.16 |
| `services/notification-service/tests/test_mvp_*.py`, `test_if01_*.py`, `test_if20_*.py` | Create | Task tests named per task | all |
| `tests/realinfra/notification-alert-delivery/conftest.py` | Create | Service root and notification tests dir on `sys.path` | WS-13.4 |
| `tests/realinfra/notification-alert-delivery/test_mvp_*.py` | Create | Real Postgres, Kafka and mailpit component tests | WS-13.3, 13.4, 13.8, 13.10, 13.14 |

## Tasks

### Task WS-13.1: One module path: package-relative imports, bare solace_events, no F821
**Covers:** MVP-155
**Severity:** medium
**Wave:** 0
**Depends on:** WS-01.5, WS-01.7
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/notification-service/src/main.py` (add `from typing import TYPE_CHECKING` and `if TYPE_CHECKING: from .domain.service import NotificationService` so the string annotations at lines 22, 103 and 110 resolve; replace the `try: from .api import router ... except ImportError: from api import router` block at lines 320-323 with `from .api import router as notification_router`)
- Modify: `services/notification-service/src/api.py` (delete the five flat fallbacks: `except ImportError: from domain import (...)` at 86-98, `from domain import TemplateRegistry` at 114-115, `from domain import ChannelRegistry` at 125-126, `from main import get_notification_service` at 215-216, `from domain import ChannelStatus` at 440-441; keep only the relative imports)
- Modify: `services/notification-service/src/consumers.py` (delete the `except ImportError: from domain import (...)` fallback at 62-70)
- Modify: `services/notification-service/src/events.py` (line 168: `from solace_events.schemas import (`)
- Modify: `services/notification-service/tests/conftest.py` (insert the service root `Path(__file__).parent.parent`, not `src`, at `sys.path[0]`; import `from src.domain.templates ...`, `from src.domain.channels ...`, `from src.domain.service ...`)
- Modify: `services/notification-service/tests/test_api.py`, `test_channels.py`, `test_crisis_consumer.py`, `test_crisis_consumer_required.py`, `test_entities.py`, `test_env_binding_p1_9.py`, `test_events.py`, `test_service.py`, `test_templates.py` (delete each per-file `sys.path.insert(... "src")` and the `sys.modules["api"]` swap at test_api.py:29-38; import `from src.domain.<m> import ...`, `from src import api as _notification_api`, `from src.consumers import ...`, `from src.main import ...`, `from src.events import ...`; no assertion changes)
- Test: `services/notification-service/tests/test_mvp_155_single_module_path.py`; run-only `tests/build/test_mvp_038_lint_gate.py -k WS-13` and `tests/build/test_import_roots.py -k WS-13` (WS-01)
**Interfaces:**
- Consumes: IF-21 notification image layout (`src.main:app`, merged `/app/src`, WS-01.5); `tests/build/test_import_roots.py` and `tests/build/test_mvp_038_lint_gate.py` owner cases `WS-13` (WS-01.7).
- Produces: `src.main` is the only importable module path for notification-service; every notification test imports through `src.` (the style WS-06.9's notification bridge test reuses). Satisfies request R-01.11 for WS-01.5 (MVP-006 closure) and WS-01.7.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_155_single_module_path.py`
```python
"""MVP-155 / MVP-006 (supporting): notification-service has exactly one module path, src.main."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src"
FLAT_ROOTS = {"domain", "api", "main", "consumers", "events", "infrastructure"}


def _flat_imports() -> list[str]:
    hits: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        rel = py.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] in FLAT_ROOTS:
                hits.append(f"{rel}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                hits += [f"{rel}:{node.lineno}:{a.name}" for a in node.names if a.name.split(".")[0] in FLAT_ROOTS]
    return hits


def test_mvp_155_no_flat_import_fallbacks() -> None:
    assert _flat_imports() == []


def test_mvp_155_every_module_imports_under_the_image_path() -> None:
    for name in ("src.main", "src.api", "src.consumers", "src.events",
                 "src.domain.channels", "src.infrastructure.event_bridge"):
        assert importlib.import_module(name).__name__ == name


def test_mvp_155_real_lifespan_runs_under_the_image_path(monkeypatch) -> None:
    monkeypatch.setenv("KAFKA_ENABLED", "false")
    import src.main as main
    with TestClient(main.app) as client:
        assert client.get("/api/v1/notifications/health").status_code == 200
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_155_single_module_path.py -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_038_lint_gate.py D:\Repo\Contextual-Chatbot\tests\build\test_import_roots.py -k WS-13 -o "addopts=" -p no:logging -q`
Expected: FAIL. Before the conftest change, collection fails with `ModuleNotFoundError: No module named 'src.main'` (tests put `src/` itself on `sys.path`, the second module layout behind MVP-155). After switching conftest only, `test_mvp_155_no_flat_import_fallbacks` lists `api.py:87:domain`, `api.py:115:domain`, `api.py:126:domain`, `api.py:216:main`, `api.py:441:domain`, `consumers.py:63:domain`, `main.py:323:api`; the lint case `WS-13` reports three `F821` in main.py (lines 22, 103, 110); the import-root case `WS-13` reports `168:src.solace_events.schemas`. The lifespan test is the MVP-155 coverage gap being closed: no existing test enters the app lifespan.
- [ ] **Step 3: Implement** the edits listed under Files. No behaviour change: the relative imports already resolve inside the `src` package; only the flat second path is removed. The `TYPE_CHECKING` import must not execute at runtime (no import cycle with `domain.service`).
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then the same with `D:\Repo\Contextual-Chatbot\tests\alignment`, then with `D:\Repo\Contextual-Chatbot\tests\integration\test_phi_encryption_activation.py`. Expected: all pass with the notification test count unchanged plus the three new tests.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.2: IF-01 runtime adoption, a strict crisis-consumer guard and a safe legacy consumer lifecycle
**Covers:** MVP-155, MVP-287
**Severity:** medium
**Wave:** 0
**Depends on:** WS-13.1, WS-04.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-14 (default assumed: staging is as strict as production; every guard uses `is_strict_environment`)
**Real infra:** no (the W0 boot smoke of WS-05.5 is the real-image proof at the barrier)
**Files:**
- Modify: `services/notification-service/src/main.py` (delete the `structlog.configure` block, the `ProductionGuards` call, `CORSMiddleware`, `create_app`, the root `/`, `/health` and `/ready` routes, the `_notification_service` global, `get_notification_service` and `_require_crisis_consumer_in_production`; `ServiceConfig.env` becomes `Literal["development", "test", "staging", "production"]`; add module-level `SETTINGS = NotificationServiceSettings.load()`, `SPEC`, `_require_crisis_consumer_in_strict(settings) -> None`, `async def _start_safety_consumer(runtime, notification_service) -> SafetyEventConsumer | None`, the new `lifespan` and `app = create_service_app(SPEC, lifespan=lifespan, routers=[(notification_router, "")], title="Solace-AI Notification Service")`; keep the `__main__` block, running `src.main:app`)
- Modify: `services/notification-service/src/api.py` (`_get_notification_service(request: Request) -> NotificationService` returns `request.app.state.notification_service`, or raises `HTTPException(503, "notification service not initialized")` when absent; no import of `main`)
- Modify: `services/notification-service/src/consumers.py` (`SafetyEventConsumer.__init__` sets `_notification_service`, `_running = False`, `_consume_task = None` and `_failed_events = []` before the early return at lines 133-137; new properties `is_running -> bool` and `consume_task -> asyncio.Task | None`; `stop()` is idempotent; `initialize_safety_consumer` assigns `_safety_consumer` only after `start()` returned)
- Modify (delete): `services/notification-service/tests/test_crisis_consumer_required.py` (replaced by the strict-guard tests below)
- Create: `services/notification-service/tests/test_if01_runtime_adoption.py`, `services/notification-service/tests/test_mvp_287_consumer_lifecycle.py` (tests; WS-13.8 later deletes the second)
- Test: run-only `tests/bootstrap/test_service_conformance.py -k notification-service` (WS-04.5)
**Interfaces:**
- Consumes: IF-01 `ServiceRuntimeSpec`, `create_service_app(spec, *, lifespan, routers, title)`, `start_service_runtime(app, spec)`, `stop_service_runtime(runtime)`, `is_strict_environment(env)`, `runtime.pg_pool`, `runtime.register_readiness_check(name, check, *, critical=True)`, `runtime.register_background_task(name, task)`, `runtime.register_shutdown(name, closer)`; IF-02 `NOTIFICATION_SERVICE_ENV`.
- Produces: `main.SPEC = ServiceRuntimeSpec(service_name="notification-service", environment=SETTINGS.service.env, version="1.0.0", log_level=SETTINGS.service.log_level, uses_postgres=True, uses_redis=True, event_plane="publish_consume", handles_phi=False, audited=True, rate_limited=False, edge=False)` (the WS-04.5 table row); `app.state.notification_service`; readiness check `safety_consumer` (critical) and background task `safety_consumer` when `KAFKA_ENABLED=true` (request from WS-04 for WS-04.5). The legacy consumer path lives only until WS-13.8.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_if01_runtime_adoption.py` and `services/notification-service/tests/test_mvp_287_consumer_lifecycle.py`
```python
# test_if01_runtime_adoption.py
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
```
```python
# test_mvp_287_consumer_lifecycle.py
"""MVP-287: a half-initialised safety consumer stops cleanly and never lingers in the module global."""
from __future__ import annotations

import pytest

from src import consumers


async def test_mvp_287_half_initialised_consumer_stops_cleanly(monkeypatch, mock_notification_service) -> None:
    monkeypatch.setattr(consumers, "_KAFKA_AVAILABLE", False)
    consumer = consumers.SafetyEventConsumer(mock_notification_service, use_mock=True)
    await consumer.stop()
    assert consumer.is_running is False and consumer.consume_task is None


async def test_mvp_287_failed_start_leaves_no_global(monkeypatch, mock_notification_service) -> None:
    monkeypatch.setattr(consumers, "_KAFKA_AVAILABLE", False)
    monkeypatch.setattr(consumers, "_safety_consumer", None)
    with pytest.raises(RuntimeError):
        await consumers.initialize_safety_consumer(mock_notification_service, use_mock=True)
    assert consumers.get_safety_consumer() is None
    await consumers.shutdown_safety_consumer()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_if01_runtime_adoption.py D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_287_consumer_lifecycle.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `AttributeError: module 'src.main' has no attribute 'SPEC'`; `create_app` and `_notification_service` still exist; `GET /live` is 404 (main.py serves only `/health`, which is unconditionally healthy at lines 336-339); `_require_crisis_consumer_in_strict` does not exist and the old guard lets `staging` through (main.py:164); `consumer.stop()` raises `AttributeError: 'SafetyEventConsumer' object has no attribute '_consume_task'` (MVP-287, consumers.py:133-137, 208-216) and `get_safety_consumer()` still returns the dead consumer after the failed start (consumers.py:717-726).
- [ ] **Step 3: Implement** the lifespan in `main.py`. First statement `runtime = await start_service_runtime(app, SPEC)`; then `notification_service = _create_notification_service(SETTINGS)`; `app.state.notification_service = notification_service`; `event_bridge = await initialize_event_bridge(postgres_pool=runtime.pg_pool)`, `get_event_publisher().register_callback(event_bridge.bridge_event)`, `runtime.register_shutdown("event_bridge", event_bridge.stop)`; `_require_crisis_consumer_in_strict(SETTINGS)`; `consumer = await _start_safety_consumer(runtime, notification_service)`; `yield`; `if consumer is not None: await shutdown_safety_consumer()`; last statement `await stop_service_runtime(runtime)`. `_start_safety_consumer` returns `None` when `SETTINGS.kafka.enabled` is false. When enabled it calls `initialize_safety_consumer(...)` with `KafkaSettings(bootstrap_servers=SETTINGS.kafka.bootstrap_servers)`: in strict any exception propagates (boot fails); otherwise it logs WARNING `safety_consumer_unavailable` with `error_type` only and registers `runtime.register_readiness_check("safety_consumer", ...)` returning `(False, "unavailable")`. On success it registers `runtime.register_background_task("safety_consumer", consumer.consume_task)` and a critical readiness check returning `(consumer.is_running, "running" | "stopped")`. `_require_crisis_consumer_in_strict` raises `RuntimeError("the safety-event consumer is required in staging and production; set KAFKA_ENABLED=true")` when `is_strict_environment(settings.service.env)` and Kafka is disabled.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then the same with `D:\Repo\Contextual-Chatbot\tests\bootstrap\test_service_conformance.py -k notification-service`, then with `D:\Repo\Contextual-Chatbot\tests\integration\test_phi_encryption_activation.py`. Expected: all pass (the two notification conformance parameters turn green here).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.3: Slot 015: crisis episodes, one-time acknowledgement tokens, risk windows and body-less delivery records
**Covers:** MVP-082, MVP-150
**Severity:** high
**Wave:** 1
**Depends on:** WS-02.3, WS-12.4 (phase A schema lane: slot 015 runs after slot 014), WS-05.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-01 (default assumed: Alembic is the only schema of record; the owner edits only `upgrade()`/`downgrade()`), D-24 (default assumed: crisis episodes are kept and pseudonymized, so `pseudonymized_at` exists from W1)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/015_notification.py` (`upgrade()` and `downgrade()` only, behind the `_is_postgres()` guard used by slots 002-009)
- Test: `tests/realinfra/notification-alert-delivery/test_mvp_150_slot_015_schema.py`; regression `tests/migrations/test_chain_real_pg.py` (WS-02)
**Interfaces:**
- Consumes: IF-03 roles and default privileges from slot 009; tables `notifications` and `delivery_attempts` from 001.
- Produces (IF-09 columns plus additions under New interfaces):
  - `crisis_alerts(alert_id uuid PK, user_id uuid NOT NULL (no FK, so pseudonyms fit), opened_at timestamptz NOT NULL DEFAULT now(), last_event_at timestamptz NOT NULL DEFAULT now(), max_level varchar(16) NOT NULL CHECK IN ('HIGH','CRITICAL'), status varchar(16) NOT NULL DEFAULT 'open' CHECK IN ('open','acknowledged','resolved'), correlation_ids uuid[] NOT NULL DEFAULT '{}', sources text[] NOT NULL DEFAULT '{}', categories text[] NOT NULL DEFAULT '{}', pages_requested int NOT NULL DEFAULT 0, page_count int NOT NULL DEFAULT 0, repage_count int NOT NULL DEFAULT 0, last_paged_at timestamptz, ack_token_hash char(64), acknowledged_by uuid, acknowledged_at timestamptz, resolved_by uuid, resolved_at timestamptz, last_update_sent_at timestamptz, pseudonymized_at timestamptz)`, indexes `ix_crisis_alerts_user_last_event (user_id, last_event_at DESC)`, `ix_crisis_alerts_open (status) WHERE status = 'open'`, GIN `ix_crisis_alerts_correlation_ids (correlation_ids)`. Columns `user_id`, `acknowledged_by` (NULL for an anonymous link acknowledgement) and `acknowledged_at` are readable by solace_app, as WS-07.8 requires (request from WS-07).
  - `crisis_alert_ack_tokens(token_hash char(64) PK, alert_id uuid NOT NULL REFERENCES crisis_alerts ON DELETE CASCADE, recipient_hash char(64) NOT NULL, issued_at timestamptz NOT NULL DEFAULT now(), expires_at timestamptz NOT NULL, used_at timestamptz)`, index on `alert_id`.
  - `risk_alert_windows(user_id uuid PK, window_started_at timestamptz NOT NULL, last_event_at timestamptz NOT NULL, max_level varchar(16) NOT NULL, assessment_count int NOT NULL DEFAULT 1, categories text[] NOT NULL DEFAULT '{}', status varchar(16) NOT NULL DEFAULT 'pending' CHECK IN ('pending','sent','failed'), attempts int NOT NULL DEFAULT 0, sent_at timestamptz)`.
  - `notifications`: `user_id`, `title` and `body` DROP NOT NULL; ADD `alert_id uuid REFERENCES crisis_alerts ON DELETE SET NULL`, `page_seq int`, `recipient_hash char(64)`, `recipient_role varchar(16)`, `subject_user_id uuid`, `kind varchar(32)`; `UNIQUE INDEX ux_notifications_alert_page_recipient (alert_id, page_seq, recipient_hash, channel) WHERE alert_id IS NOT NULL`; index `ix_notifications_subject_user_id`.
  - All four tables are non-RLS (IF-03 list extended under New interfaces). `downgrade()` drops the new indexes, columns and tables in reverse order and runs `DELETE FROM notifications WHERE user_id IS NULL OR title IS NULL OR body IS NULL` before restoring NOT NULL.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/notification-alert-delivery/test_mvp_150_slot_015_schema.py`
```python
"""MVP-082 / MVP-150 (IF-09, IF-03): slot 015 stores episodes, one-time tokens, risk windows and body-less records for solace_app."""
from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]

EPISODE_COLUMNS = {"alert_id", "user_id", "opened_at", "last_event_at", "max_level", "status", "correlation_ids",
                   "sources", "categories", "pages_requested", "page_count", "repage_count", "last_paged_at",
                   "ack_token_hash", "acknowledged_by", "acknowledged_at", "resolved_by", "resolved_at",
                   "last_update_sent_at", "pseudonymized_at"}
INSERT_EPISODE = ("INSERT INTO crisis_alerts (alert_id, user_id, max_level, correlation_ids, sources, categories) "
                  "VALUES ($1, $2, 'CRITICAL', $3, $4, $5)")


async def _columns(pg, table: str) -> dict[str, tuple[str, str]]:
    rows = await pg.superuser_fetch(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1", table)
    return {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in rows}


async def test_mvp_082_crisis_alerts_has_the_if09_columns(pg_migrated) -> None:
    cols = await _columns(pg_migrated, "crisis_alerts")
    assert EPISODE_COLUMNS <= cols.keys()
    assert cols["correlation_ids"][0] == "ARRAY" and cols["sources"][0] == "ARRAY"
    assert cols["acknowledged_by"][1] == "YES" and cols["user_id"][1] == "NO"


async def test_mvp_082_app_role_writes_episodes_without_rls_context(pg_migrated) -> None:
    alert, user, turn = uuid4(), uuid4(), uuid4()
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute(INSERT_EPISODE, alert, user, [turn], ["safety-service"], ["suicidal_ideation"])
        row = await conn.fetchrow("SELECT status, page_count, pages_requested, repage_count, correlation_ids "
                                  "FROM crisis_alerts WHERE alert_id = $1", alert)
    assert (row["status"], row["page_count"], row["pages_requested"], row["repage_count"]) == ("open", 0, 0, 0)
    assert row["correlation_ids"] == [turn]


@pytest.mark.parametrize("column,value", [("status", "closed"), ("max_level", "ELEVATED")])
async def test_mvp_082_status_and_level_are_constrained(pg_migrated, column: str, value: str) -> None:
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute(INSERT_EPISODE, alert := uuid4(), uuid4(), [], [], [])
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(f"UPDATE crisis_alerts SET {column} = $1 WHERE alert_id = $2", value, alert)


async def test_mvp_082_ack_tokens_are_unique_and_cascade_with_the_episode(pg_migrated) -> None:
    alert, token = uuid4(), "a" * 64
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute(INSERT_EPISODE, alert, uuid4(), [], [], [])
        insert = ("INSERT INTO crisis_alert_ack_tokens (token_hash, alert_id, recipient_hash, expires_at) "
                  "VALUES ($1, $2, $3, now() + interval '24 hours')")
        await conn.execute(insert, token, alert, "b" * 64)
        with pytest.raises(asyncpg.exceptions.UniqueViolationError):
            await conn.execute(insert, token, alert, "c" * 64)
        await conn.execute("DELETE FROM crisis_alerts WHERE alert_id = $1", alert)
        assert await conn.fetchval("SELECT count(*) FROM crisis_alert_ack_tokens WHERE alert_id = $1", alert) == 0


async def test_mvp_150_bodyless_alert_records_are_unique_per_recipient_and_page(pg_migrated) -> None:
    alert, user = uuid4(), uuid4()
    insert = ("INSERT INTO notifications (id, created_at, updated_at, category, channel, status, template_id, alert_id, "
              "page_seq, recipient_hash, recipient_role, subject_user_id, kind) VALUES "
              "($1, now(), now(), 'safety', 'email', 'PENDING', 'crisis_alert', $2, 1, $3, 'on_call', $4, 'crisis_page')")
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute(INSERT_EPISODE, alert, user, [], [], [])
        first = uuid4()
        await conn.execute(insert, first, alert, "d" * 64, user)
        with pytest.raises(asyncpg.exceptions.UniqueViolationError):
            await conn.execute(insert, uuid4(), alert, "d" * 64, user)
        await conn.execute("INSERT INTO delivery_attempts (id, created_at, updated_at, notification_id, channel, "
                           "attempt_number, status, attempted_at) VALUES ($1, now(), now(), $2, 'email', 1, 'DELIVERED', now())",
                           uuid4(), first)
        row = await conn.fetchrow("SELECT title, body, user_id FROM notifications WHERE id = $1", first)
    assert (row["title"], row["body"], row["user_id"]) == (None, None, None)


async def test_mvp_156_risk_alert_windows_hold_one_row_per_user(pg_migrated) -> None:
    user = uuid4()
    insert = ("INSERT INTO risk_alert_windows (user_id, window_started_at, last_event_at, max_level) "
              "VALUES ($1, now(), now(), 'ELEVATED')")
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute(insert, user)
        with pytest.raises(asyncpg.exceptions.UniqueViolationError):
            await conn.execute(insert, user)
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute("UPDATE risk_alert_windows SET status = 'queued' WHERE user_id = $1", user)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery\test_mvp_150_slot_015_schema.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `asyncpg.exceptions.UndefinedTableError: relation "crisis_alerts" does not exist` (no episode store exists, MVP-082) and the body-less insert fails with `NotNullViolationError` on `user_id`/`title`/`body` because 001 made them NOT NULL and nothing ever wrote these tables (MVP-150, migrations/versions/001_initial_schema.py:884-939).
- [ ] **Step 3: Implement** the slot body per Produces with `op.execute` DDL (arrays and partial or GIN indexes are PostgreSQL-only, hence the guard). `revision` and `down_revision` stay as WS-02.3 wrote them.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations -o "addopts=" -p no:logging -q`. Expected: all pass (upgrade head, downgrade base, upgrade head; solace_app has no DDL; every policy uses NULLIF).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.4: Episode, token, risk-window and delivery repositories with per-recipient idempotency
**Covers:** MVP-150, MVP-151
**Severity:** medium
**Wave:** 1
**Depends on:** WS-13.3, WS-13.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Create: `services/notification-service/src/domain/episodes.py` (dataclasses and token hashing below)
- Create: `services/notification-service/src/infrastructure/crisis_alert_repository.py` (`CrisisAlertRepository` Protocol, `PostgresCrisisAlertRepository`, `InMemoryCrisisAlertRepository`)
- Create: `services/notification-service/src/infrastructure/delivery_repository.py` (`DeliveryClaim`, `DeliveryRepository` Protocol, `PostgresDeliveryRepository`, `InMemoryDeliveryRepository`)
- Create: `services/notification-service/tests/notification_fakes.py` (`FixedClock(start=datetime(2026, 10, 19, 8, 0, tzinfo=UTC))` with `__call__() -> datetime` and `advance(seconds: float) -> None`)
- Create: `services/notification-service/tests/notification_repo_contract.py` (the `check_*` coroutines and `CHECKS`)
- Create: `tests/realinfra/notification-alert-delivery/conftest.py` (inserts the notification-service root directory and its tests directory at the front of `sys.path`, resolved from `Path(__file__).resolve().parents[3]`)
- Test: `services/notification-service/tests/test_mvp_150_151_repositories.py`, `tests/realinfra/notification-alert-delivery/test_mvp_150_151_repositories_pg.py`
**Interfaces:**
- Consumes: slot 015 (WS-13.3); `runtime.pg_pool` (asyncpg, solace_app, non-RLS tables accessed through `pool.acquire()` per IF-03).
- Produces:
  - `episodes.py`: `@dataclass CrisisEpisode(alert_id: UUID, user_id: UUID, opened_at: datetime, last_event_at: datetime, max_level: str, status: str = "open", correlation_ids: list[UUID], sources: list[str], categories: list[str], pages_requested: int = 0, page_count: int = 0, repage_count: int = 0, last_paged_at: datetime | None = None, acknowledged_by: UUID | None = None, acknowledged_at: datetime | None = None, resolved_by: UUID | None = None, resolved_at: datetime | None = None, last_update_sent_at: datetime | None = None)` with property `ref -> str` (first 8 characters of `str(alert_id)`); `class AckOutcome(str, Enum)`: `ACKNOWLEDGED`, `ALREADY_ACKNOWLEDGED`, `RESOLVED`, `ALREADY_RESOLVED`, `EXPIRED`, `INVALID`, `NOT_FOUND`; `@dataclass(frozen=True) AckResult(outcome: AckOutcome, alert_id: UUID | None = None, at: datetime | None = None)`; `@dataclass RiskWindow(user_id: UUID, window_started_at: datetime, last_event_at: datetime, max_level: str, assessment_count: int, categories: list[str], status: str = "pending", attempts: int = 0, sent_at: datetime | None = None)`; `hash_ack_token(raw: str) -> str` (sha256 hex).
  - `CrisisAlertRepository`: `transaction() -> AsyncContextManager[Any]`; `lock_user(tx, user_id)` (`pg_advisory_xact_lock(hashtext('crisis_alerts:' || $1))`); `find_by_correlation(tx, user_id, correlation_id) -> CrisisEpisode | None` (any status); `latest_for_user(tx, user_id) -> CrisisEpisode | None` (highest `last_event_at`, `FOR UPDATE`); `insert(tx, episode)`; `save(tx, episode)`; `get(alert_id)`; `mark_page_complete(alert_id, *, page_seq: int, paged_at: datetime)` (`page_count = GREATEST(page_count, page_seq)`, `last_paged_at`); `issue_ack_token(alert_id, *, recipient_hash: str, now: datetime, ttl_seconds: int) -> str` (raw `secrets.token_urlsafe(32)`; only `hash_ack_token(raw)` stored); `acknowledge_with_token(raw, *, now) -> AckResult` (unknown → `INVALID`, past `expires_at` → `EXPIRED`, episode resolved → `ALREADY_RESOLVED`, already acknowledged → `ALREADY_ACKNOWLEDGED`, else status `acknowledged`, `acknowledged_by NULL`, `acknowledged_at = now`, `ack_token_hash = hash`, token `used_at = now` → `ACKNOWLEDGED`); `acknowledge(alert_id, *, clinician_id, now) -> AckResult` (`NOT_FOUND`, `ALREADY_RESOLVED`, `ALREADY_ACKNOWLEDGED` keeps the first acknowledger, `ACKNOWLEDGED`); `resolve(alert_id, *, clinician_id, now) -> AckResult` (`NOT_FOUND`, `ALREADY_RESOLVED`, `RESOLVED`); `list_episodes(*, statuses: Sequence[str], limit: int = 100) -> list[CrisisEpisode]` (newest first); `due_for_repage(*, now, after_seconds: int, max_repages: int) -> list[CrisisEpisode]` (`status = 'open' AND repage_count < max_repages AND COALESCE(last_paged_at, opened_at) <= now - after_seconds`); `claim_repage(alert_id, *, expected_repage_count: int, now) -> CrisisEpisode | None` (one conditional `UPDATE ... SET repage_count = repage_count + 1, pages_requested = pages_requested + 1, last_paged_at = now WHERE alert_id = $1 AND status = 'open' AND repage_count = $2 RETURNING *`); `mark_update_sent(alert_id, *, now)`; `count_open() -> int`; `record_risk(*, user_id, level: str, categories: Sequence[str], now) -> RiskWindow` (upsert: a `pending` row gains `assessment_count + 1`, the higher `max_level`, the union of categories and `last_event_at`; a `sent` or `failed` row restarts as a new `pending` window with count 1); `due_risk_windows(*, now, window_seconds: int, limit: int = 50) -> list[RiskWindow]` (`pending` and `window_started_at <= now - window_seconds`); `mark_risk_sent(user_id, *, now)`; `mark_risk_failed_attempt(user_id, *, max_attempts: int) -> RiskWindow` (attempts + 1; status `failed` at `max_attempts`).
  - `@dataclass(frozen=True) DeliveryClaim(notification_id: UUID, already_delivered: bool, attempt_number: int)`; `DeliveryRepository`: `claim(*, kind: str, channel: str, template_id: str, alert_id: UUID | None = None, page_seq: int | None = None, recipient_hash: str | None = None, recipient_role: str | None = None, subject_user_id: UUID | None = None, title: str | None = None, now: datetime | None = None) -> DeliveryClaim` (with `alert_id` and `page_seq`: `INSERT ... ON CONFLICT DO NOTHING`, then re-read, so concurrent claims share one row; `attempt_number` = attempts so far + 1); `record_attempt(claim, *, delivered: bool, error_code: str | None, duration_ms: int, now: datetime | None = None, external_message_id: str | None = None)` (one `delivery_attempts` row, `notifications.status` becomes `DELIVERED` or `FAILED`, `sent_at`/`delivered_at` set on delivery; `body` and `html_body` are never written); `page_status(*, alert_id, page_seq) -> dict[tuple[str, str], str]` (`(recipient_hash, channel)` to status).
  - In-memory implementations serialize `transaction()` with one `asyncio.Lock` and are used only in development/test (WS-13.8 refuses them in strict). `InMemoryDeliveryRepository.records() -> list[DeliveryRecord]` is a test inspection helper; each `DeliveryRecord` exposes `kind`, `template_id`, `alert_id`, `page_seq`, `recipient_hash`, `recipient_role`, `subject_user_id` and `status`.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/notification_repo_contract.py`, `services/notification-service/tests/test_mvp_150_151_repositories.py` and `tests/realinfra/notification-alert-delivery/test_mvp_150_151_repositories_pg.py`
```python
# notification_repo_contract.py
"""Repository contract shared by the in-memory (unit) and Postgres (real-infra) tests. MVP-150/151, MVP-082."""
from __future__ import annotations

import asyncio
from uuid import uuid4

from src.domain.episodes import AckOutcome, CrisisEpisode

KEY = dict(kind="crisis_page", channel="email", template_id="crisis_alert", recipient_hash="a" * 64, recipient_role="on_call")


async def open_episode(episodes, clock, *, user=None, level="CRITICAL", turn=None) -> CrisisEpisode:
    ep = CrisisEpisode(alert_id=uuid4(), user_id=user or uuid4(), opened_at=clock(), last_event_at=clock(),
                       max_level=level, correlation_ids=[turn or uuid4()], sources=["safety-service"],
                       categories=["suicidal_ideation"], pages_requested=1)
    async with episodes.transaction() as tx:
        await episodes.lock_user(tx, ep.user_id)
        await episodes.insert(tx, ep)
    return ep


async def check_claim_is_idempotent_per_recipient_and_page(episodes, deliveries, clock) -> None:
    ep = await open_episode(episodes, clock)
    key = dict(KEY, alert_id=ep.alert_id, page_seq=1, subject_user_id=ep.user_id)
    first = await deliveries.claim(**key, now=clock())
    assert (await deliveries.claim(**key, now=clock())).notification_id == first.notification_id
    await deliveries.record_attempt(first, delivered=False, error_code="SMTPRecipientsRefused", duration_ms=12, now=clock())
    second = await deliveries.claim(**key, now=clock())
    assert (second.already_delivered, second.attempt_number) == (False, 2)
    await deliveries.record_attempt(second, delivered=True, error_code=None, duration_ms=30, now=clock())
    assert (await deliveries.claim(**key, now=clock())).already_delivered is True
    assert (await deliveries.claim(**dict(key, page_seq=2), now=clock())).notification_id != first.notification_id
    assert await deliveries.page_status(alert_id=ep.alert_id, page_seq=1) == {("a" * 64, "email"): "DELIVERED"}


async def check_concurrent_claims_share_one_record(episodes, deliveries, clock) -> None:
    ep = await open_episode(episodes, clock)
    key = dict(KEY, alert_id=ep.alert_id, page_seq=1, subject_user_id=ep.user_id)
    claims = await asyncio.gather(*(deliveries.claim(**key, now=clock()) for _ in range(5)))
    assert len({c.notification_id for c in claims}) == 1


async def check_ack_token_is_one_time(episodes, deliveries, clock) -> None:
    ep = await open_episode(episodes, clock)
    raw = await episodes.issue_ack_token(ep.alert_id, recipient_hash="a" * 64, now=clock(), ttl_seconds=86_400)
    assert len(raw) >= 43
    first = await episodes.acknowledge_with_token(raw, now=clock())
    assert (first.outcome, first.alert_id) == (AckOutcome.ACKNOWLEDGED, ep.alert_id)
    stored = await episodes.get(ep.alert_id)
    assert (stored.status, stored.acknowledged_by, stored.acknowledged_at) == ("acknowledged", None, clock())
    assert (await episodes.acknowledge_with_token(raw, now=clock())).outcome == AckOutcome.ALREADY_ACKNOWLEDGED


async def check_expired_and_unknown_tokens_change_nothing(episodes, deliveries, clock) -> None:
    ep = await open_episode(episodes, clock)
    raw = await episodes.issue_ack_token(ep.alert_id, recipient_hash="a" * 64, now=clock(), ttl_seconds=86_400)
    assert (await episodes.acknowledge_with_token("x" * 43, now=clock())).outcome == AckOutcome.INVALID
    clock.advance(86_401)
    assert (await episodes.acknowledge_with_token(raw, now=clock())).outcome == AckOutcome.EXPIRED
    assert (await episodes.get(ep.alert_id)).status == "open"


async def check_claim_repage_wins_once(episodes, deliveries, clock) -> None:
    ep = await open_episode(episodes, clock)
    results = await asyncio.gather(
        episodes.claim_repage(ep.alert_id, expected_repage_count=0, now=clock()),
        episodes.claim_repage(ep.alert_id, expected_repage_count=0, now=clock()))
    won = [r for r in results if r is not None]
    assert len(won) == 1 and (won[0].repage_count, won[0].pages_requested) == (1, 2)


async def check_ack_before_claim_blocks_the_repage(episodes, deliveries, clock) -> None:
    ep = await open_episode(episodes, clock)
    assert (await episodes.acknowledge(ep.alert_id, clinician_id=uuid4(), now=clock())).outcome == AckOutcome.ACKNOWLEDGED
    assert await episodes.claim_repage(ep.alert_id, expected_repage_count=0, now=clock()) is None


async def check_due_for_repage_respects_interval_and_budget(episodes, deliveries, clock) -> None:
    ep = await open_episode(episodes, clock)
    await episodes.mark_page_complete(ep.alert_id, page_seq=1, paged_at=clock())
    clock.advance(899)
    assert ep.alert_id not in {e.alert_id for e in await episodes.due_for_repage(now=clock(), after_seconds=900, max_repages=3)}
    clock.advance(1)
    assert ep.alert_id in {e.alert_id for e in await episodes.due_for_repage(now=clock(), after_seconds=900, max_repages=3)}
    assert ep.alert_id not in {e.alert_id for e in await episodes.due_for_repage(now=clock(), after_seconds=900, max_repages=0)}


async def check_correlation_lookup_and_latest(episodes, deliveries, clock) -> None:
    user, old_turn = uuid4(), uuid4()
    old = await open_episode(episodes, clock, user=user, turn=old_turn)
    clock.advance(3600)
    new = await open_episode(episodes, clock, user=user)
    async with episodes.transaction() as tx:
        await episodes.lock_user(tx, user)
        assert (await episodes.find_by_correlation(tx, user, old_turn)).alert_id == old.alert_id
        assert (await episodes.latest_for_user(tx, user)).alert_id == new.alert_id


async def check_risk_windows_coalesce_per_user(episodes, deliveries, clock) -> None:
    user = uuid4()
    for level in ("ELEVATED", "ELEVATED", "HIGH"):
        await episodes.record_risk(user_id=user, level=level, categories=["hopelessness"], now=clock())
    assert user not in {w.user_id for w in await episodes.due_risk_windows(now=clock(), window_seconds=900)}
    clock.advance(900)
    [window] = [w for w in await episodes.due_risk_windows(now=clock(), window_seconds=900) if w.user_id == user]
    assert (window.assessment_count, window.max_level, window.status) == (3, "HIGH", "pending")
    await episodes.mark_risk_sent(user, now=clock())
    restarted = await episodes.record_risk(user_id=user, level="ELEVATED", categories=[], now=clock())
    assert (restarted.assessment_count, restarted.status) == (1, "pending")


CHECKS = [check_claim_is_idempotent_per_recipient_and_page, check_concurrent_claims_share_one_record,
          check_ack_token_is_one_time, check_expired_and_unknown_tokens_change_nothing, check_claim_repage_wins_once,
          check_ack_before_claim_blocks_the_repage, check_due_for_repage_respects_interval_and_budget,
          check_correlation_lookup_and_latest, check_risk_windows_coalesce_per_user]
```
```python
# test_mvp_150_151_repositories.py
import pytest

import notification_repo_contract as contract
from notification_fakes import FixedClock
from src.infrastructure.crisis_alert_repository import InMemoryCrisisAlertRepository
from src.infrastructure.delivery_repository import InMemoryDeliveryRepository


@pytest.mark.parametrize("check", contract.CHECKS, ids=lambda f: f.__name__)
async def test_in_memory_repositories_honour_the_contract(check) -> None:
    await check(InMemoryCrisisAlertRepository(), InMemoryDeliveryRepository(), FixedClock())
```
```python
# tests/realinfra/notification-alert-delivery/test_mvp_150_151_repositories_pg.py
import hashlib
from uuid import uuid4

import pytest

import notification_repo_contract as contract
from notification_fakes import FixedClock
from src.infrastructure.crisis_alert_repository import PostgresCrisisAlertRepository
from src.infrastructure.delivery_repository import PostgresDeliveryRepository

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


@pytest.mark.parametrize("check", contract.CHECKS, ids=lambda f: f.__name__)
async def test_postgres_repositories_honour_the_contract(check, pg_migrated) -> None:
    await check(PostgresCrisisAlertRepository(pg_migrated.app_pool), PostgresDeliveryRepository(pg_migrated.app_pool), FixedClock())


async def test_mvp_150_page_records_hold_no_body_and_only_a_recipient_hash(pg_migrated) -> None:
    episodes, deliveries, clock = PostgresCrisisAlertRepository(pg_migrated.app_pool), PostgresDeliveryRepository(pg_migrated.app_pool), FixedClock()
    ep = await contract.open_episode(episodes, clock)
    claim = await deliveries.claim(**dict(contract.KEY, alert_id=ep.alert_id, page_seq=1, subject_user_id=ep.user_id), now=clock())
    await deliveries.record_attempt(claim, delivered=True, error_code=None, duration_ms=20, now=clock())
    (row,) = await pg_migrated.superuser_fetch("SELECT body, html_body, title, status, recipient_hash FROM notifications WHERE id = $1", claim.notification_id)
    assert (row["body"], row["html_body"], row["status"], row["recipient_hash"]) == (None, None, "DELIVERED", "a" * 64)
    (att,) = await pg_migrated.superuser_fetch("SELECT status, attempt_number, error_message FROM delivery_attempts WHERE notification_id = $1", claim.notification_id)
    assert (att["status"], att["attempt_number"], att["error_message"]) == ("DELIVERED", 1, None)


async def test_mvp_082_raw_ack_tokens_are_never_stored(pg_migrated) -> None:
    episodes, clock = PostgresCrisisAlertRepository(pg_migrated.app_pool), FixedClock()
    ep = await contract.open_episode(episodes, clock)
    raw = await episodes.issue_ack_token(ep.alert_id, recipient_hash="a" * 64, now=clock(), ttl_seconds=86_400)
    rows = await pg_migrated.superuser_fetch("SELECT token_hash FROM crisis_alert_ack_tokens WHERE alert_id = $1", ep.alert_id)
    assert [r["token_hash"] for r in rows] == [hashlib.sha256(raw.encode()).hexdigest()]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_150_151_repositories.py -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery\test_mvp_150_151_repositories_pg.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'src.domain.episodes'` (and `src.infrastructure.crisis_alert_repository`): the service persists no notification, attempt or episode record (MVP-150, `domain/service.py:117-131`) and has no per-recipient dedup record, so every redelivery re-pages (MVP-151).
- [ ] **Step 3: Implement** the three modules per Produces. The Postgres repositories use `pool.acquire()` and `conn.transaction()`; `transaction()` yields the connection inside an open transaction; lists bind as `uuid[]`/`text[]`; `created_at`/`updated_at` are passed explicitly on `notifications` and `delivery_attempts`; `category` is `safety` for alert kinds and `transactional` for API kinds; `error_code` is an exception class name or SMTP reply code, never text.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery -o "addopts=" -p no:logging -q`, then `D:\Repo\Contextual-Chatbot\tests\integration\test_phi_encryption_activation.py` with the unit command (notification stays exempt: no `solace_infrastructure.database.entities` import). Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.5: Real recipients through IF-08 with a service token, strict alert settings and the notification__user contract
**Covers:** MVP-011, MVP-064
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-13.2, WS-07.3, WS-05.7
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-12 (default assumed: alerts carry the user's display name and contact only with crisis-contact consent), D-13 (default assumed: clinicians come only from provisioning, so an empty roster is a configuration fault), D-23 (default assumed: IF-08 nulls contact fields without consent)
**Real infra:** no
**Files:**
- Create: `services/notification-service/src/domain/alert_settings.py`
- Create: `services/notification-service/src/domain/recipients.py`
- Create: `tests/contracts/notification__user/on-call.get.json`, `tests/contracts/notification__user/care-team.get.json`, `tests/contracts/notification__user/care-team-unknown-user.get.json` (IF-20 `ContractFile` shape)
- Test: `services/notification-service/tests/test_mvp_011_roster_client.py`, `services/notification-service/tests/test_mvp_011_alert_settings.py`
**Interfaces:**
- Consumes: IF-08 `GET /api/v1/internal/clinicians/on-call` and `GET /api/v1/internal/users/{user_id}/care-team` (service token with `service:read:oncall`); IF-04 `service_auth_headers(target_service: str, *, on_behalf_of: str | None = None) -> dict[str, str]` (WS-07.3; the runtime initializes the identity per WS-07's request to WS-04); IF-20 `load_contracts(dir)`, `assert_request_matches(contract, request)` (WS-05.7); IF-02 `SERVICE_USER_SERVICE_URL`, `NOTIFICATION_FALLBACK_ONCALL_EMAIL`, `NOTIFICATION_ALERT_COALESCE_SECONDS`, `NOTIFICATION_REPAGE_AFTER_SECONDS`, `NOTIFICATION_MAX_REPAGES`, `NOTIFICATION_ACK_BASE_URL`; IF-01 `StartupDependencyError(component, reason)`, `is_strict_environment(env)`.
- Produces:
  - `class AlertSettings(BaseSettings)` (`env_prefix="NOTIFICATION_"`, `populate_by_name=True`, `extra="ignore"`): `user_service_url: str = "http://localhost:8001"` (alias choices `SERVICE_USER_SERVICE_URL`, then legacy `NOTIFICATION_USER_SERVICE_URL`), `fallback_oncall_email: str = ""`, `alert_coalesce_seconds: int = 900` (>= 60), `repage_after_seconds: int = 900` (>= 10), `max_repages: int = 3` (0..10), `ack_base_url: str = "http://localhost:8003"`; `validate_for(environment: str) -> None`. In strict it raises `StartupDependencyError("notification_settings", "<VARIABLE> ...")` naming the variable, never its value, when the fallback is empty, not an address, `oncall@solace-ai.com` or `*@localhost.invalid`; when `NOTIFICATION_ACK_BASE_URL` is not an absolute http(s) URL or its host is `localhost`/`127.0.0.1`; or when the user-service host is `localhost`/`127.0.0.1`. Outside strict it logs WARNING `alert_settings_incomplete` with the variable names.
  - `recipients.py`: `ROSTER_LOOKUP_TIMEOUT_SECONDS = 3.0`; `@dataclass(frozen=True) Clinician(user_id: UUID, display_name: str, email: str | None, phone_e164: str | None, role: str, is_backup: bool)`; `@dataclass(frozen=True) CareTeam(user_id: UUID, display_name: str | None, contact_email: str | None, contact_phone_e164: str | None, crisis_contact_consent: bool, clinicians: tuple[Clinician, ...])`; `@dataclass(frozen=True) AlertRecipient(address: str, role: Literal["care_team", "on_call", "backup", "fallback"], user_id: UUID | None = None, phone_e164: str | None = None)` with property `hash -> str`; `recipient_hash(address: str) -> str` (sha256 hex of `address.strip().lower()`); `class RosterUnavailable(Exception)`; `class RosterClient(base_url: str, *, headers: Callable[[], Awaitable[Mapping[str, str]]], timeout_seconds: float = ROSTER_LOOKUP_TIMEOUT_SECONDS, transport: httpx.AsyncBaseTransport | None = None)` with `async on_call() -> list[Clinician]` and `async care_team(user_id: UUID) -> CareTeam`. Any transport error, timeout, non-2xx status (except 404 on care-team) or malformed body raises `RosterUnavailable(<status code or exception class>)` and is never turned into an empty list; a care-team 404 returns a `CareTeam` with no clinicians; without consent the client nulls `display_name`, `contact_email` and `contact_phone_e164` even if the provider sent them.
  - Contract files `tests/contracts/notification__user/` (consumer `notification`, provider `user`): `on-call` (GET, headers `Authorization`, `X-Request-ID`, `X-Correlation-ID`; 200 schema `{clinicians: [{user_id, display_name, email, phone_e164 (string|null), role, is_backup}]}`), `care-team` (GET `/api/v1/internal/users/{user_id}/care-team`; 200 schema `{user_id, display_name (string|null), contact_email (string|null), contact_phone_e164 (string|null), crisis_contact_consent: boolean, clinicians: [...]}`), `care-team-unknown-user` (404). Satisfies the WS-05 request for WS-05.10 (verified by WS-14 in W2).

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_011_roster_client.py` and `services/notification-service/tests/test_mvp_011_alert_settings.py`
```python
# test_mvp_011_roster_client.py
"""MVP-011/MVP-064: recipients come from IF-08 with a service token; failures are loud, never an empty roster."""
from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from solace_testing.contracts import assert_request_matches, load_contracts
from src.domain.recipients import RosterClient, RosterUnavailable, recipient_hash

CONTRACTS = Path(__file__).resolve().parents[3] / "tests" / "contracts" / "notification__user"
C1 = uuid4()
CLINICIAN = {"user_id": str(C1), "display_name": "Dr A", "email": "clin-a@test.invalid",
             "phone_e164": "+15555550100", "role": "clinician", "is_backup": False}


async def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer svc-token", "X-Request-ID": "r-1", "X-Correlation-ID": "c-1"}


def _client(handler) -> RosterClient:
    return RosterClient("http://user-service:8001", headers=_headers, transport=httpx.MockTransport(handler))


def _contract(name: str):
    return next(c for c in load_contracts(CONTRACTS) if c.interaction == name)


def _as_request(r: httpx.Request) -> dict:
    return {"method": r.method, "path": r.url.path, "headers": dict(r.headers), "json": None}


async def test_mvp_011_on_call_lookup_uses_the_internal_route_and_a_service_token() -> None:
    seen: list[httpx.Request] = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"clinicians": [CLINICIAN]})
    [clinician] = await _client(handler).on_call()
    assert seen[0].url.path == "/api/v1/internal/clinicians/on-call"
    assert seen[0].headers["Authorization"] == "Bearer svc-token"
    assert (clinician.user_id, clinician.email, clinician.is_backup) == (C1, "clin-a@test.invalid", False)
    assert_request_matches(_contract("on-call"), _as_request(seen[0]))


async def test_mvp_011_care_team_without_consent_never_carries_identity() -> None:
    user = uuid4()
    def handler(request):
        return httpx.Response(200, json={"user_id": str(user), "display_name": "Taylor Q", "contact_email": "user-t@test.invalid",
                                         "contact_phone_e164": "+15555550142", "crisis_contact_consent": False, "clinicians": [CLINICIAN]})
    team = await _client(handler).care_team(user)
    assert (team.display_name, team.contact_email, team.contact_phone_e164) == (None, None, None)
    assert [c.user_id for c in team.clinicians] == [C1]


async def test_mvp_011_unknown_user_has_an_empty_care_team() -> None:
    team = await _client(lambda r: httpx.Response(404, json={"detail": "USER_NOT_FOUND"})).care_team(uuid4())
    assert team.clinicians == ()


@pytest.mark.parametrize("failure", ["401", "500", "timeout", "connect", "bad_json"])
async def test_mvp_011_lookup_failures_raise_instead_of_returning_nobody(failure: str) -> None:
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if failure == "connect":
            raise httpx.ConnectError("refused", request=request)
        if failure == "bad_json":
            return httpx.Response(200, content=b"not json")
        return httpx.Response(int(failure), json={"detail": "x"})
    with pytest.raises(RosterUnavailable):
        await _client(handler).on_call()


def test_recipient_hash_normalizes_case_and_whitespace() -> None:
    assert recipient_hash(" Clin-A@Test.invalid ") == hashlib.sha256(b"clin-a@test.invalid").hexdigest()
```
```python
# test_mvp_011_alert_settings.py
"""MVP-011 + review focus 5: strict boots refuse placeholder or partial alert configuration."""
from __future__ import annotations

import pytest

from solace_infrastructure.runtime import StartupDependencyError
from src.domain.alert_settings import AlertSettings

STRICT_OK = {"SERVICE_USER_SERVICE_URL": "http://user-service:8001",
             "NOTIFICATION_FALLBACK_ONCALL_EMAIL": "fallback-oncall@clinic.example",
             "NOTIFICATION_ACK_BASE_URL": "https://api.clinic.example"}


def _env(monkeypatch, **override) -> None:
    for name, value in {**STRICT_OK, **override}.items():
        monkeypatch.setenv(name, value)


def test_mvp_011_peer_url_and_if09_defaults(monkeypatch) -> None:
    _env(monkeypatch)
    s = AlertSettings()
    assert s.user_service_url == "http://user-service:8001"
    assert (s.alert_coalesce_seconds, s.repage_after_seconds, s.max_repages) == (900, 900, 3)
    s.validate_for("production")


@pytest.mark.parametrize("env", ["staging", "production"])
@pytest.mark.parametrize("name,value", [
    ("NOTIFICATION_FALLBACK_ONCALL_EMAIL", ""), ("NOTIFICATION_FALLBACK_ONCALL_EMAIL", "oncall@solace-ai.com"),
    ("NOTIFICATION_FALLBACK_ONCALL_EMAIL", "fallback-oncall@localhost.invalid"),
    ("NOTIFICATION_ACK_BASE_URL", "http://localhost:8003"), ("NOTIFICATION_ACK_BASE_URL", "api.clinic.example"),
    ("SERVICE_USER_SERVICE_URL", "http://localhost:8001")])
def test_review_focus_5_strict_rejects_partial_alert_config(monkeypatch, env, name, value) -> None:
    _env(monkeypatch, **{name: value})
    with pytest.raises(StartupDependencyError) as exc:
        AlertSettings().validate_for(env)
    assert exc.value.component == "notification_settings" and name in str(exc.value)
    assert not value or value not in str(exc.value)


def test_non_strict_tolerates_dev_defaults(monkeypatch) -> None:
    for name in STRICT_OK:
        monkeypatch.delenv(name, raising=False)
    AlertSettings().validate_for("development")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_011_roster_client.py D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_011_alert_settings.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'src.domain.recipients'` / `src.domain.alert_settings` and `load_contracts` finds no `notification__user` files. Today the only lookup is `SafetyEventConsumer._get_oncall_clinicians`, which calls the public user route with a token user-service rejects, defaults to `http://localhost:8001` and turns every error into `[]` (consumers.py:89-111, 579-625), and the fallback defaults to the placeholder `oncall@solace-ai.com` (MVP-011).
- [ ] **Step 3: Implement** `AlertSettings`, the recipients module and the three contract files per Produces. `RosterClient` builds one `httpx.AsyncClient(timeout=timeout_seconds, transport=transport)` per call and awaits `headers()` for each request. It logs `roster_lookup_failed` with `route` and `error_type` only.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then the same with `D:\Repo\Contextual-Chatbot\tests\contracts`. Expected: all pass (the contract-file schema test accepts the three new files).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.6: Crisis episodes and paging: coalescing, care team then on-call then fallback, one bounded attempt per recipient, pending pages retried by the pager
**Covers:** MVP-299, MVP-011, MVP-064, MVP-151
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-13.4, WS-13.5, WS-13.7, WS-12.2, WS-06.3
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-06 (default assumed: one episode per user per 900 s window, a higher level pages again, including a rise reported later for the same turn and a rise after acknowledgement, crisis-review F8/F9; an acknowledged episode gets one update email per window only for same-or-lower levels), D-12 (default assumed: identity only with crisis-contact consent; never message text or diagnosis), D-28 (default assumed: notification-service is the only pager and the episode system of record)
**Real infra:** no (WS-13.8 proves it on Kafka, Postgres and mailpit)
**Files:**
- Modify: `services/notification-service/src/domain/episodes.py` (add `EpisodeAction` and `decide_episode`)
- Create: `services/notification-service/src/domain/crisis_paging.py`
- Create: `services/notification-service/src/domain/alert_metrics.py`
- Modify: `services/notification-service/tests/notification_fakes.py` (add `FakeRoster`, `Sent`, `FakeNotifier`, `clinician`, `care_team`, `crisis_signal`, `make_pager`)
- Test: `services/notification-service/tests/test_mvp_299_episode_coalescing.py`, `services/notification-service/tests/test_mvp_011_064_crisis_paging.py`
**Interfaces:**
- Consumes: WS-13.4 repositories and dataclasses; WS-13.5 `RosterClient`, `CareTeam`, `Clinician`, `AlertRecipient`, `RosterUnavailable`, `AlertSettings`; WS-13.7 `NotificationService.render(template_type, variables) -> RenderedTemplate` and `deliver_one(channel_type, target, rendered, *, timeout_seconds, metadata=None) -> DeliveryResult`, `TemplateType.CRISIS_ALERT`/`CRISIS_UPDATE`; IF-07 `solace_clinical.CATEGORIES`, `level_rank`, `parse_escalation_reason` (WS-12.2, crisis-review F10); IF-06 event fields (`crisis_level`, `trigger_indicators`, `metadata.correlation_id`, `metadata.source_service`, `metadata.timestamp`), the WS-12.8 `escalation_reason` form `auto:<LEVEL>:<codes>`.
- Produces:
  - `EpisodeAction(str, Enum)`: `OPEN`, `ESCALATE`, `FOLD`, `DUPLICATE`, `UPDATE`; `decide_episode(latest: CrisisEpisode | None, signal: CrisisSignal, *, now: datetime, window_seconds: int, duplicate_of: CrisisEpisode | None = None) -> tuple[EpisodeAction, CrisisEpisode]`. Rules in order: (1) `duplicate_of` set: a `signal.level` strictly above `duplicate_of.max_level` → `ESCALATE` on that episode (`max_level` raised, `pages_requested + 1`, sources and categories merged; a higher level reported later for the same turn, crisis-review F8), otherwise `DUPLICATE` (merge sources and categories only; `last_event_at` unchanged); (2) no latest, latest resolved, or `now - latest.last_event_at > window_seconds` → `OPEN` (new `alert_id`, `pages_requested = 1`); (3) `level_rank(signal.level) > level_rank(latest.max_level)` → `ESCALATE` (`max_level` raised, `pages_requested + 1`), whether the episode is open or acknowledged (crisis-review F9: a rise is never folded into an update); (4) latest acknowledged → `UPDATE`; (5) otherwise `FOLD`. Every non-duplicate action appends the correlation id (when new) and sets `last_event_at = now`. A redelivery of the same event is a `DUPLICATE` because its level is not above the `max_level` it already raised, so `ESCALATE` stays idempotent per correlation id and level.
  - `crisis_paging.py`: `CRISIS_DELIVERY_TIMEOUT_SECONDS = 8.0`; `CRISIS_PAGE_RETRY_SWEEP_SECONDS = 15`; `ACK_TOKEN_TTL_SECONDS = 86_400`; `ACK_PATH = "/api/v1/notifications/crisis-alerts/ack/"`; `@dataclass(frozen=True) CrisisSignal(user_id: UUID, level: str, correlation_id: UUID, source: str, categories: tuple[str, ...], detected_at: datetime, event_id: UUID)` with classmethods `from_crisis_event(event)`, `from_escalation_event(event)` (priority HIGH/CRITICAL; categories are the codes of `solace_clinical.parse_escalation_reason(escalation_reason)`, `()` when it returns `None`) and `from_assessment_event(event)` (categories from `risk_factors[].evidence`), each returning `None` below HIGH and keeping only codes in `CATEGORIES` (dropped values are counted in the log as `dropped_indicators=<n>`, never echoed); `@dataclass(frozen=True) PageOutcome(alert_id: UUID, action: EpisodeAction, clinician_deliveries: int, fallback_delivered: bool)`; `class CrisisPager(*, episodes, deliveries, roster, notifier, settings: AlertSettings, clock: Callable[[], datetime] | None = None)` with read-only property `settings -> AlertSettings`, `async handle_detection(signal: CrisisSignal) -> PageOutcome` and `async retry_pending_pages() -> int`.
  - Delivery retries belong to the pager, not to the consume loop (crisis-review F1; an IF-09 refinement recorded for the skeleton by ledger Ruling, and the amended WS-06 request 6 for WS-06.9). `handle_detection` delivers a newly requested page inline once (actions `OPEN` and `ESCALATE`). A page that reached no clinician stays pending (`pages_requested > page_count`), whether or not the fallback mailbox got it; the handler logs ERROR `crisis_page_pending` with `alert_ref`, `page_seq` and `fallback_delivered`, and returns normally. `DUPLICATE`, `FOLD` and `UPDATE` never deliver a pending page inline, so the four HIGH events of one turn cost one delivery pass, not four. `handle_detection` never raises for a delivery outcome; unexpected exceptions (a repository or roster bug, a crash between steps) still propagate, and the plane retries those. `retry_pending_pages()` gives every `open` or `acknowledged` episode with a pending page one more pass of page `seq = pages_requested` (the same recipient rules; recipients already delivered for that page, the fallback mailbox included, are skipped), marks the page complete on the first clinician delivery, returns the number completed and never raises (a per-episode error logs ERROR `crisis_page_retry_failed` with `alert_ref` and `error_type`); WS-13.8 runs it every `CRISIS_PAGE_RETRY_SWEEP_SECONDS`. A pass that finds no clinician recipient (empty tier or `RosterUnavailable`) counts `solace_crisis_alert_delivery_total{outcome="failed",channel="email"}` once, so `CrisisAlertDeliveryFailed` keeps firing for as long as a page has no clinician (the DLQ alert this design no longer triggers is replaced, not lost).
  - `ESCALATE` recipients: the primary tier as for any page; on an acknowledged episode also the acknowledging clinician (looked up by `acknowledged_by`, when known and not already in the tier) and, for CRITICAL, the non-backup on-call clinicians (crisis-review F9). An escalation never waits for the update window.
  - `alert_metrics.py` (prometheus_client, default registry): `CRISIS_ALERT_DELIVERY_TOTAL = Counter("solace_crisis_alert_delivery_total", ..., ["outcome", "channel"])` with outcome values exactly `delivered` (a clinician), `failed` (any recipient, or one count for a pass with no clinician recipient) and `fallback_delivered` (the fallback mailbox), satisfying the WS-04 request for WS-04.7; `CRISIS_ALERT_LATENCY_SECONDS = Histogram("solace_crisis_alert_latency_seconds", ..., buckets=(1, 5, 15, 30, 60, 120, 300, 900))` (detection to first clinician delivery); `CRISIS_ALERTS_UNACKNOWLEDGED = Gauge("solace_crisis_alerts_unacknowledged", ...)`; `MONITORING_ALERT_DELIVERY_TOTAL = Counter("solace_monitoring_alert_delivery_total", ..., ["outcome"])` (used by WS-13.8).
  - Email content per recipient: template `CRISIS_ALERT` with `level`, `categories` (comma-separated codes or `none reported`), `sources`, `detected_at` (episode `opened_at`, UTC `YYYY-MM-DDTHH:MM:SSZ`), `alert_id`, `alert_ref`, `ack_url = settings.ack_base_url.rstrip("/") + ACK_PATH + raw_token`, `patient_line` (display name and contact only when `crisis_contact_consent` is true, else `Not shared (no crisis-contact consent)`; `Patient identity unavailable (lookup failed)` when the care-team lookup failed) and `page_note` (`""` for a first or escalation page).

- [ ] **Step 1: Write the failing test** in the two test files, with the fakes in `notification_fakes.py`
```python
# notification_fakes.py additions (shapes the tests rely on)
class FakeRoster:                                  # on_call: Iterable[Clinician]; care_teams: dict[UUID, CareTeam]; fail -> RosterUnavailable
    def __init__(self, *, on_call=(), care_teams=None, fail: bool = False) -> None: ...
@dataclass
class Sent:
    channel: str; address: str; subject: str; body: str
class FakeNotifier:                                # real TemplateRegistry rendering; reject/hang are address sets
    def __init__(self, *, reject=(), hang=()) -> None: ...   # .sent: list[Sent]; .reject and .hang are mutable sets
    def render(self, template_type, variables): ...
    async def deliver_one(self, channel_type, target, rendered, *, timeout_seconds, metadata=None): ...
    # hang -> asyncio.wait_for(asyncio.sleep(60), timeout_seconds) -> DeliveryResult(success=False, error_message="TimeoutError")
    def to(self, address: str) -> list[Sent]: ...
def clinician(email: str, *, backup: bool = False, phone: str | None = None) -> Clinician: ...
def care_team(user: UUID, *members: Clinician, consent: bool = False, name: str | None = None,
              email: str | None = None, phone: str | None = None) -> CareTeam: ...
def crisis_signal(user: UUID, level: str = "CRITICAL", turn: UUID | None = None, *, source: str = "safety-service",
                  categories=("suicidal_ideation",), at: datetime | None = None) -> CrisisSignal: ...
def make_pager(*, roster, notifier, clock, fallback: str = "fallback-oncall@test.invalid", window: int = 900):
    ...  # -> (CrisisPager, InMemoryCrisisAlertRepository, InMemoryDeliveryRepository); AlertSettings(ack_base_url="https://api.example.test", user_service_url="http://user-service:8001", ...)
```
```python
# test_mvp_299_episode_coalescing.py
"""MVP-299 + D-06: one turn reported by several producers is one page; episodes coalesce per user and window; a rise always pages."""
from __future__ import annotations

import re
from uuid import uuid4

from notification_fakes import FakeNotifier, FakeRoster, FixedClock, care_team, clinician, crisis_signal, make_pager

C1, C2 = clinician("clin-c1@test.invalid"), clinician("clin-c2@test.invalid")


async def test_mvp_299_one_turn_from_three_events_pages_once() -> None:
    clock, notifier = FixedClock(), FakeNotifier()
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=clock)
    user, turn = uuid4(), uuid4()
    for source in ("orchestrator-service", "safety-service", "safety-service"):
        await pager.handle_detection(crisis_signal(user, "CRITICAL", turn, source=source))
    assert [s.address for s in notifier.sent] == [C1.email]
    [episode] = await episodes.list_episodes(statuses=["open"])
    assert episode.correlation_ids == [turn] and sorted(episode.sources) == ["orchestrator-service", "safety-service"]


async def test_d06_new_turns_fold_and_a_higher_level_pages_again() -> None:
    clock, notifier = FixedClock(), FakeNotifier()
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=clock)
    user = uuid4()
    await pager.handle_detection(crisis_signal(user, "HIGH"))
    clock.advance(60); await pager.handle_detection(crisis_signal(user, "HIGH"))
    clock.advance(60); await pager.handle_detection(crisis_signal(user, "CRITICAL"))
    clock.advance(901); await pager.handle_detection(crisis_signal(user, "HIGH"))
    assert [s.subject.split("]")[0] for s in notifier.sent] == ["[Solace CRISIS HIGH", "[Solace CRISIS CRITICAL", "[Solace CRISIS HIGH"]
    assert len({s.subject[-8:] for s in notifier.sent}) == 2


async def test_d06_acknowledged_episode_sends_one_update_to_the_acknowledger() -> None:
    clock, notifier = FixedClock(), FakeNotifier()
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[C1, C2]), notifier=notifier, clock=clock)
    user = uuid4()
    outcome = await pager.handle_detection(crisis_signal(user, "CRITICAL"))
    await episodes.acknowledge(outcome.alert_id, clinician_id=C1.user_id, now=clock())
    for _ in range(2):
        clock.advance(30); await pager.handle_detection(crisis_signal(user, "CRITICAL"))
    updates = [s for s in notifier.sent if s.subject.startswith("[Solace CRISIS UPDATE")]
    assert [u.address for u in updates] == [C1.email]
    assert updates[0].subject == f"[Solace CRISIS UPDATE CRITICAL] ref {str(outcome.alert_id)[:8]}"


async def test_redelivered_turn_is_a_duplicate_even_after_the_window() -> None:
    clock, notifier = FixedClock(), FakeNotifier()
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=clock)
    signal = crisis_signal(uuid4(), "CRITICAL")
    await pager.handle_detection(signal)
    clock.advance(3600)
    await pager.handle_detection(signal)
    assert len(notifier.sent) == 1 and len(await episodes.list_episodes(statuses=["open"])) == 1


async def test_crisis_review_f8_a_higher_level_reported_later_for_the_same_turn_pages_again() -> None:
    clock, notifier = FixedClock(), FakeNotifier()
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=clock)
    user, turn = uuid4(), uuid4()
    await pager.handle_detection(crisis_signal(user, "HIGH", turn, source="orchestrator-service"))      # local screen, /check timed out
    clock.advance(5)
    critical = crisis_signal(user, "CRITICAL", turn, source="safety-service")                        # safety's LLM layer raised it
    await pager.handle_detection(critical)
    await pager.handle_detection(critical)                                                           # redelivered: no third page
    assert [s.subject.split("]")[0] for s in notifier.sent] == ["[Solace CRISIS HIGH", "[Solace CRISIS CRITICAL"]
    [episode] = await episodes.list_episodes(statuses=["open"])
    assert episode.max_level == "CRITICAL" and episode.correlation_ids == [turn]


async def test_crisis_review_f9_a_rise_after_acknowledgement_pages_and_is_never_only_an_update() -> None:
    clock, notifier = FixedClock(), FakeNotifier()
    user = uuid4()
    roster = FakeRoster(on_call=[C2], care_teams={user: care_team(user, C1)})
    pager, episodes, _ = make_pager(roster=roster, notifier=notifier, clock=clock)
    await pager.handle_detection(crisis_signal(user, "HIGH"))
    token = re.search(r"/crisis-alerts/ack/([A-Za-z0-9_-]{43,})", notifier.sent[0].body).group(1)
    await episodes.acknowledge_with_token(token, now=clock())                      # by link: acknowledged_by is NULL
    clock.advance(60); await pager.handle_detection(crisis_signal(user, "HIGH"))  # same level: one update
    clock.advance(300); await pager.handle_detection(crisis_signal(user, "CRITICAL"))
    pages = [s for s in notifier.sent if s.subject.startswith("[Solace CRISIS CRITICAL]")]
    assert sorted(p.address for p in pages) == sorted([C1.email, C2.email])       # primary tier plus on-call for CRITICAL
    assert len([s for s in notifier.sent if s.subject.startswith("[Solace CRISIS UPDATE")]) == 1
    [episode] = await episodes.list_episodes(statuses=["acknowledged"])
    assert episode.max_level == "CRITICAL" and episode.page_count == 2
```
```python
# test_mvp_011_064_crisis_paging.py
"""MVP-011/064/151, crisis-review F1 and review focus 2/4: real recipients, fallback on failed delivery, pending pages
retried by the pager (never by holding the consumer), idempotent retries, no PHI."""
from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from prometheus_client import REGISTRY

from notification_fakes import FakeNotifier, FakeRoster, FixedClock, care_team, clinician, crisis_signal, make_pager
from solace_events.schemas import CrisisDetectedEvent, CrisisLevel, EscalationTriggeredEvent, EventMetadata
from src.domain.crisis_paging import CrisisSignal
from src.domain.episodes import AckOutcome

C1, C2 = clinician("clin-c1@test.invalid"), clinician("clin-c2@test.invalid")
C3 = clinician("clin-c3@test.invalid", backup=True)
FALLBACK = "fallback-oncall@test.invalid"


def _metric(outcome: str) -> float:
    return REGISTRY.get_sample_value("solace_crisis_alert_delivery_total", {"outcome": outcome, "channel": "email"}) or 0.0


async def test_mvp_011_care_team_is_paged_before_the_on_call_roster() -> None:
    user, notifier = uuid4(), FakeNotifier()
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C2], care_teams={user: care_team(user, C1)}), notifier=notifier, clock=FixedClock())
    await pager.handle_detection(crisis_signal(user))
    assert [s.address for s in notifier.sent] == [C1.email]


async def test_mvp_011_primary_on_call_is_paged_and_the_backup_is_not() -> None:
    notifier = FakeNotifier()
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C2, C3]), notifier=notifier, clock=FixedClock())
    await pager.handle_detection(crisis_signal(uuid4()))
    assert [s.address for s in notifier.sent] == [C2.email]


async def test_mvp_064_rejected_recipients_fall_back_and_the_page_stays_pending() -> None:
    user = uuid4()
    notifier = FakeNotifier(reject={C1.email, C2.email})
    failed, fallback = _metric("failed"), _metric("fallback_delivered")
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[C2], care_teams={user: care_team(user, C1)}), notifier=notifier, clock=FixedClock())
    outcome = await pager.handle_detection(crisis_signal(user))                  # returns: the consumer is never held
    assert (outcome.clinician_deliveries, outcome.fallback_delivered) == (0, True)
    assert len(notifier.to(FALLBACK)) == 1
    assert (_metric("failed") - failed, _metric("fallback_delivered") - fallback) == (2.0, 1.0)
    assert (await episodes.list_episodes(statuses=["open"]))[0].page_count == 0  # still pending for the sweep


async def test_mvp_011_roster_outage_pages_the_fallback_and_stays_pending() -> None:
    notifier = FakeNotifier()
    failed = _metric("failed")
    pager, episodes, _ = make_pager(roster=FakeRoster(fail=True), notifier=notifier, clock=FixedClock())
    await pager.handle_detection(crisis_signal(uuid4()))
    [episode] = await episodes.list_episodes(statuses=["open"])
    assert [s.address for s in notifier.sent] == [FALLBACK] and episode.page_count == 0
    assert _metric("failed") - failed == 1.0                                     # no clinician recipient still alerts


async def test_mvp_151_a_retry_skips_recipients_already_delivered() -> None:
    user, clock = uuid4(), FixedClock()
    notifier = FakeNotifier(reject={C1.email})
    pager, episodes, _ = make_pager(roster=FakeRoster(care_teams={user: care_team(user, C1)}), notifier=notifier, clock=clock)
    signal = crisis_signal(user)
    await pager.handle_detection(signal)
    notifier.reject.clear()
    await pager.handle_detection(signal)                                         # a redelivery never resends inline
    assert (len(notifier.to(C1.email)), len(notifier.to(FALLBACK))) == (0, 1)
    clock.advance(15)
    assert await pager.retry_pending_pages() == 1
    assert (len(notifier.to(C1.email)), len(notifier.to(FALLBACK))) == (1, 1)
    assert (await episodes.list_episodes(statuses=["open"]))[0].page_count == 1
    assert await pager.retry_pending_pages() == 0


async def test_crisis_review_f1_a_turns_four_events_cost_one_delivery_pass_and_never_raise() -> None:
    user, turn = uuid4(), uuid4()
    notifier = FakeNotifier(reject={C1.email})
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=FixedClock())
    for source in ("orchestrator-service", "safety-service", "safety-service", "safety-service"):   # crisis, escalation, assessment
        await pager.handle_detection(crisis_signal(user, "HIGH", turn, source=source))
    assert [s.address for s in notifier.sent] == [FALLBACK]
    assert (await episodes.list_episodes(statuses=["open"]))[0].page_count == 0


async def test_crisis_review_f1_total_delivery_failure_stays_pending_until_a_clinician_gets_it() -> None:
    user, clock = uuid4(), FixedClock()
    notifier = FakeNotifier(reject={C1.email, FALLBACK})
    failed = _metric("failed")
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=clock)
    outcome = await pager.handle_detection(crisis_signal(user))
    assert (outcome.clinician_deliveries, outcome.fallback_delivered) == (0, False) and _metric("failed") - failed == 2.0
    clock.advance(15)
    assert await pager.retry_pending_pages() == 0                                 # still failing: counted, not raised
    notifier.reject.clear()
    clock.advance(15)
    assert await pager.retry_pending_pages() == 1
    assert (len(notifier.to(C1.email)), len(notifier.to(FALLBACK))) == (1, 0)


async def test_crisis_review_f1_a_hanging_relay_bounds_the_handler_once_per_turn(monkeypatch) -> None:
    from src.domain import crisis_paging
    monkeypatch.setattr(crisis_paging, "CRISIS_DELIVERY_TIMEOUT_SECONDS", 0.2)
    user, turn = uuid4(), uuid4()
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=FakeNotifier(hang={C1.email, FALLBACK}), clock=FixedClock())
    t0 = time.monotonic()
    for source in ("orchestrator-service", "safety-service", "safety-service", "safety-service"):
        await pager.handle_detection(crisis_signal(user, "CRITICAL", turn, source=source))
    assert time.monotonic() - t0 < 0.8                                            # two tiers once (0.4 s), duplicates free


async def test_review_focus_2_crash_after_the_episode_commit_still_pages_through_the_sweep(monkeypatch) -> None:
    clock, notifier = FixedClock(), FakeNotifier()
    pager, episodes, deliveries = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=clock)
    original, calls = deliveries.claim, {"n": 0}
    async def flaky_claim(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("pool closed")
        return await original(**kw)
    monkeypatch.setattr(deliveries, "claim", flaky_claim)
    signal = crisis_signal(uuid4())
    with pytest.raises(ConnectionError):                                         # unexpected exceptions still reach the plane
        await pager.handle_detection(signal)
    await pager.handle_detection(signal)                                         # the plane's redelivery: a DUPLICATE
    clock.advance(15)
    assert await pager.retry_pending_pages() == 1
    assert len(notifier.to(C1.email)) == 1 and (await episodes.list_episodes(statuses=["open"]))[0].page_count == 1


async def test_review_focus_2_crash_between_send_and_record_resends(monkeypatch) -> None:
    clock, notifier = FixedClock(), FakeNotifier()
    pager, episodes, deliveries = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=clock)
    original, calls = deliveries.record_attempt, {"n": 0}
    async def flaky_record(claim, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("pool closed")
        return await original(claim, **kw)
    monkeypatch.setattr(deliveries, "record_attempt", flaky_record)
    signal = crisis_signal(uuid4())
    with pytest.raises(ConnectionError):
        await pager.handle_detection(signal)
    await pager.handle_detection(signal)
    clock.advance(15)
    await pager.retry_pending_pages()
    assert len(notifier.to(C1.email)) == 2          # at-least-once per recipient; never zero
    assert (await episodes.list_episodes(statuses=["open"]))[0].page_count == 1


async def test_review_focus_4_alerts_carry_category_codes_only() -> None:
    notifier = FakeNotifier()
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=FixedClock())
    event = CrisisDetectedEvent(user_id=uuid4(), crisis_level=CrisisLevel.CRITICAL,
                                trigger_indicators=["suicidal_ideation", "CRITICAL_KEYWORD:kill myself", "I want to die"],
                                detection_layer=1, confidence=Decimal("0.9"), escalation_action="page_on_call",
                                metadata=EventMetadata(correlation_id=uuid4(), source_service="safety-service"))
    await pager.handle_detection(CrisisSignal.from_crisis_event(event))
    body = notifier.sent[0].body
    assert "suicidal_ideation" in body and "kill myself" not in body and "want to die" not in body


@pytest.mark.parametrize("consent", [True, False])
async def test_d12_patient_identity_only_with_crisis_contact_consent(consent: bool) -> None:
    user, notifier = uuid4(), FakeNotifier()
    team = care_team(user, C1, consent=consent, name="Taylor Q", phone="+15555550142")
    pager, _, _ = make_pager(roster=FakeRoster(care_teams={user: team}), notifier=notifier, clock=FixedClock())
    await pager.handle_detection(crisis_signal(user))
    body = notifier.sent[0].body
    assert ("Taylor Q" in body, "+15555550142" in body) == (consent, consent)


async def test_if09_each_recipient_gets_its_own_one_time_link() -> None:
    notifier = FakeNotifier()
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[C1, C2]), notifier=notifier, clock=FixedClock())
    await pager.handle_detection(crisis_signal(uuid4()))
    tokens = [re.search(r"/api/v1/notifications/crisis-alerts/ack/([A-Za-z0-9_-]{43,})", s.body).group(1) for s in notifier.sent]
    assert len(set(tokens)) == 2
    assert (await episodes.acknowledge_with_token(tokens[1], now=FixedClock()())).outcome == AckOutcome.ACKNOWLEDGED


async def test_if09_subject_metrics_and_latency() -> None:
    delivered = _metric("delivered")
    count = REGISTRY.get_sample_value("solace_crisis_alert_latency_seconds_count") or 0.0
    notifier = FakeNotifier()
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=FixedClock())
    outcome = await pager.handle_detection(crisis_signal(uuid4(), "HIGH"))
    assert notifier.sent[0].subject == f"[Solace CRISIS HIGH] action required - ref {str(outcome.alert_id)[:8]}"
    assert _metric("delivered") - delivered == 1.0
    assert (REGISTRY.get_sample_value("solace_crisis_alert_latency_seconds_count") or 0.0) - count == 1.0


def test_signal_mapping_ignores_levels_below_high_and_parses_escalation_codes() -> None:
    meta = EventMetadata(correlation_id=uuid4(), source_service="safety-service")
    elevated = CrisisDetectedEvent(user_id=uuid4(), crisis_level=CrisisLevel.ELEVATED, trigger_indicators=[], detection_layer=1,
                                   confidence=Decimal("0.6"), escalation_action="page_on_call", metadata=meta)
    assert CrisisSignal.from_crisis_event(elevated) is None
    esc = EscalationTriggeredEvent(user_id=uuid4(), escalation_reason="auto:CRITICAL:suicidal_ideation,means_access",
                                   priority="CRITICAL", metadata=meta)
    signal = CrisisSignal.from_escalation_event(esc)
    assert (signal.level, signal.categories, signal.correlation_id) == ("CRITICAL", ("means_access", "suicidal_ideation"), meta.correlation_id)
    assert CrisisSignal.from_escalation_event(esc.model_copy(update={"priority": "MEDIUM"})) is None
    assert CrisisSignal.from_escalation_event(esc.model_copy(update={"escalation_reason": "Triggers: kill myself"})).categories == ()
    assert CrisisSignal.from_escalation_event(esc.model_copy(update={"escalation_reason": "auto:CRITICAL:not_a_code"})).categories == ()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_299_episode_coalescing.py D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_011_064_crisis_paging.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'src.domain.crisis_paging'`. Today each CrisisDetected and EscalationTriggered event is paged on its own with a new event id (consumers.py:226-285, 340-384; MVP-299), a found assigned clinician skips the on-call team and the fallback is tried only when the lookup returns nobody (consumers.py:437-447, 485-506; MVP-064), the raw `trigger_indicators` are copied into the email (consumers.py:244), and a failed page is retried only by re-raising inside the shared consume loop (crisis-review F1). The F8/F9 tests pin that a same-turn or post-acknowledgement rise is never swallowed.
- [ ] **Step 3: Implement** `decide_episode`, `CrisisSignal`, `CrisisPager.handle_detection`, `CrisisPager.retry_pending_pages` and the metrics. `handle_detection`: (1) `async with episodes.transaction() as tx:` `lock_user`, `find_by_correlation`, `latest_for_user`, `decide_episode`, then `insert` (OPEN) or `save`; the transaction commits before any delivery. (2) `UPDATE` with `last_update_sent_at` unset or older than the window sends one `CRISIS_UPDATE` email to the acknowledging clinician (looked up by `acknowledged_by` in the care team and on-call lists), else to the primary tier, then `mark_update_sent`; updates use `claim(page_seq=None, kind="crisis_update")`. (3) For `OPEN` and `ESCALATE` only, run one delivery pass `_deliver_page(episode, seq=pages_requested)`; every other action returns without delivering. `_deliver_page`: look up the care team and the on-call roster concurrently, catching `RosterUnavailable` per lookup; the primary tier is the care team's clinicians with an email, else the non-backup on-call clinicians, plus the `ESCALATE` additions under Produces; deliver to every primary recipient concurrently; if no clinician delivery succeeded (previously delivered recipients count), deliver concurrently to the non-backup on-call clinicians not yet tried (only when the primary tier was the care team) and to the fallback mailbox. For each recipient: `claim(kind="crisis_page", channel="email", template_id="crisis_alert", alert_id, page_seq=seq, recipient_hash, recipient_role, subject_user_id)`; skip if `already_delivered`; `issue_ack_token`; render; `deliver_one(..., timeout_seconds=CRISIS_DELIVERY_TIMEOUT_SECONDS)`; `record_attempt`; metric. A pass with no clinician recipient counts one `failed`. On at least one clinician delivery: `mark_page_complete(seq)` and, for `seq == 1`, observe latency from `signal.detected_at`; otherwise log ERROR `crisis_page_pending` and return the `PageOutcome` (never `RetryableEventError`). Unexpected exceptions propagate (the plane retries). (4) `retry_pending_pages()`: `episodes.list_episodes(statuses=["open", "acknowledged"])`, keep those with `pages_requested > page_count`, run `_deliver_page` for each with at most 4 concurrently, catch and log any per-episode exception. Log `crisis_page_attempt` with `alert_ref`, `page_seq`, `recipient_role`, `outcome`, `error_type`. `level_rank`, `CATEGORIES` and `parse_escalation_reason` come from `solace_clinical`. Read `CRISIS_DELIVERY_TIMEOUT_SECONDS` and `ACK_TOKEN_TTL_SECONDS` from the module at call time (tests monkeypatch them).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.7: Crisis alert content, plain-text rendering, Date and Message-ID, TLS mode and bounded single attempts
**Covers:** MVP-290, MVP-292, MVP-154, MVP-064
**Severity:** high
**Wave:** 1
**Depends on:** WS-13.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-12 (default assumed: SMTP relay with STARTTLS, `EMAIL_SMTP_TLS_MODE=starttls`; alert content as IF-09)
**Real infra:** no (the SMTP path is proven against mailpit in WS-13.8)
**Files:**
- Modify: `services/notification-service/src/domain/templates.py` (`TemplateEngine` keeps `_text_env = Environment(autoescape=False, undefined=StrictUndefined)` for subjects and bodies and `_html_env = Environment(autoescape=True, undefined=StrictUndefined)` for `html_template`, replacing the shared `autoescape=True` env at lines 107-111; `TemplateType.CRISIS_UPDATE = "crisis_update"`; `CRISIS_ALERT` (lines 264-273) and `RISK_ALERT` (lines 283-292) rewritten without HTML, emoji or user text; new `CRISIS_UPDATE`)
- Modify: `services/notification-service/src/domain/channels.py` (`EmailConfig.tls_mode: Literal["starttls", "tls", "none"] = "starttls"` replaces `use_tls`; the SMTP call at 312-317 passes `use_tls=tls_mode == "tls"`, `start_tls=tls_mode == "starttls"` and no custom `tls_context`; the message at 300-309 gains `Date = formatdate(usegmt=True)` and `Message-ID = make_msgid(domain=<domain of from_email>)`; `_EMAIL_PATTERN` (line 35) accepts the RFC 5322 dot-atom local part `[A-Za-z0-9.!#$%&'*+/=?^_\`{|}~-]+`; `NotificationChannel.send(..., *, max_attempts: int | None = None, attempt_timeout_seconds: float | None = None)` runs each attempt under `asyncio.wait_for` and sets `error_message` to the last exception's class name, never `str(exc)` (the `DeliveryError` text embeds the recipient address); the `notification_delivered` and `notification_delivery_failed` log lines at 231-239 drop `recipient=` and log `channel`, `attempt` and `error_type` only)
- Modify: `services/notification-service/src/domain/service.py` (`render`, `deliver_one`, `channel_registry`; the `notification_delivery_exception` log at line 213 carries `error_type` instead of `str(result)`)
- Modify: `services/notification-service/src/main.py` (`EmailConfig` settings: `smtp_tls_mode` from `EMAIL_SMTP_TLS_MODE`, default `starttls`; `timeout_seconds` default 10; `_create_notification_service(settings, *, environment: str)` raises `StartupDependencyError("email_channel", ...)` in strict when `EMAIL_ENABLED` is false or the mode is `none`)
- Modify: `services/notification-service/tests/test_channels.py` (`test_email_config_defaults` asserts `config.tls_mode == "starttls"` instead of `use_tls is True`), `services/notification-service/tests/test_templates.py` (only if a test uses the old `CRISIS_ALERT` variables)
- Test: `services/notification-service/tests/test_mvp_290_292_alert_content.py`
**Interfaces:**
- Consumes: IF-02 `EMAIL_SMTP_TLS_MODE` (`starttls|tls|none`, `none` only in development/test), `EMAIL_ENABLED`, `EMAIL_FROM_EMAIL`; IF-09 subject and body rules; IF-01 `StartupDependencyError`, `is_strict_environment`.
- Produces: `NotificationService.render(template_type: TemplateType, variables: Mapping[str, Any]) -> RenderedTemplate`; `NotificationService.deliver_one(channel_type: ChannelType, target: str, rendered: RenderedTemplate, *, timeout_seconds: float, metadata: Mapping[str, Any] | None = None) -> DeliveryResult` (exactly one attempt bounded by `timeout_seconds`; a missing or unavailable channel, a timeout or any exception gives `success=False` with `error_message` = exception class name); `NotificationService.channel_registry -> ChannelRegistry`. Templates: `CRISIS_ALERT` subject `[Solace CRISIS {{ level }}] action required - ref {{ alert_ref }}{{ page_note }}` (required `level`, `categories`, `sources`, `detected_at`, `alert_id`, `alert_ref`, `ack_url`, `patient_line`; default `page_note=""`); `CRISIS_UPDATE` subject `[Solace CRISIS UPDATE {{ level }}] ref {{ alert_ref }}`; `RISK_ALERT` subject `[Solace RISK {{ level }}] monitoring summary - {{ assessment_count }} assessment(s)` with `window_start`, `window_end`, `user_id`, `categories`. Strict SMTP relies on the default trust store, so `SSL_CERT_FILE` (WS-05 strict overlay) is honoured.
- Produces for requesting tasks: the `password_reset` template contract that WS-08.18's `PasswordResetMailer` posts (WS-08's request R-15): `TemplateType.PASSWORD_RESET = "password_reset"`, subject `Reset your Solace-AI password`, `required_variables == ["display_name", "reset_link", "expiry_minutes"]`, the text body renders `reset_link` and `display_name` verbatim through `_text_env` and states the expiry in minutes, and the HTML body stays escaped; pinned by `test_ws08_password_reset_template_keeps_its_contract`. WS-08.18 depends on this task.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_290_292_alert_content.py`
```python
"""MVP-290/292/154/064 and IF-09: alert text is plain and complete; mail carries Date/Message-ID; attempts are bounded."""
from __future__ import annotations

import asyncio
import email.utils
import re
import time
from types import SimpleNamespace

import pytest

from solace_infrastructure.runtime import StartupDependencyError
from src.domain import channels
from src.domain.channels import ChannelRegistry, ChannelType, EmailChannel, EmailConfig, _validate_email_address
from src.domain.service import NotificationService
from src.domain.templates import TemplateRegistry, TemplateType
import src.main as main

ACK = "https://api.example.test/api/v1/notifications/crisis-alerts/ack/tok"
VARS = dict(level="CRITICAL", categories="suicidal_ideation, means_access", sources="orchestrator-service, safety-service",
            detected_at="2026-10-19T08:15:00Z", alert_id="3f2a9c1e-0000-4000-8000-000000000001", alert_ref="3f2a9c1e",
            ack_url=ACK, patient_line="Not shared (no crisis-contact consent)")


def test_if09_crisis_subject_and_body() -> None:
    r = TemplateRegistry().render_template(TemplateType.CRISIS_ALERT, VARS)
    assert r.subject == "[Solace CRISIS CRITICAL] action required - ref 3f2a9c1e"
    for part in ("CRITICAL", "suicidal_ideation, means_access", "orchestrator-service", "2026-10-19T08:15:00Z",
                 "3f2a9c1e-0000-4000-8000-000000000001", ACK):
        assert part in r.body
    assert r.html_body is None


def test_mvp_290_plain_text_is_not_html_escaped_but_html_still_is() -> None:
    r = TemplateRegistry().render_template(TemplateType.CRISIS_ALERT, dict(VARS, patient_line="O'Brien & Co <+15555550100>"))
    assert "Patient: O'Brien & Co <+15555550100>" in r.body and "&#39;" not in r.body and "&amp;" not in r.body
    v = TemplateRegistry().render_template(TemplateType.EMAIL_VERIFICATION, {
        "display_name": "O'Brien", "verification_link": "https://app.test/v?token=a&user=b", "expiry_hours": 24})
    assert "token=a&user=b" in v.body and "O'Brien" in v.body and "&amp;" in v.html_body


def test_ws08_password_reset_template_keeps_its_contract() -> None:
    # WS-08 request R-15: WS-08.18's PasswordResetMailer posts exactly these variables
    registry = TemplateRegistry()
    link = "https://app.test/reset-password?token=Ab_9-x"
    r = registry.render_template(TemplateType.PASSWORD_RESET,
                                 {"display_name": "O'Brien", "reset_link": link, "expiry_minutes": 30})
    assert r.subject == "Reset your Solace-AI password"
    assert link in r.body and "O'Brien" in r.body and "30 minutes" in r.body and "&#39;" not in r.body
    assert "O&#39;Brien" in r.html_body
    assert registry.get_template(TemplateType.PASSWORD_RESET).required_variables == [
        "display_name", "reset_link", "expiry_minutes"]


class _SMTP:
    sent: list = []
    kwargs: dict = {}
    def __init__(self, **kwargs): _SMTP.kwargs = kwargs
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return False
    async def login(self, *a): return None
    async def send_message(self, message):
        _SMTP.sent.append(message); return ({}, "250 OK")


@pytest.mark.parametrize("mode,use_tls,start_tls", [("starttls", False, True), ("tls", True, False), ("none", False, False)])
async def test_mvp_292_mail_has_date_and_message_id_and_honours_tls_mode(monkeypatch, mode, use_tls, start_tls) -> None:
    monkeypatch.setattr(channels.aiosmtplib, "SMTP", _SMTP)
    _SMTP.sent.clear()
    channel = EmailChannel(EmailConfig(smtp_host="mail.test", from_email="alerts@solace.test", tls_mode=mode))
    assert (await channel.send("clin-a@test.invalid", "s", "b")).success
    message = _SMTP.sent[0]
    assert email.utils.parsedate_to_datetime(message["Date"]).tzinfo is not None
    assert re.fullmatch(r"<[^@>]+@solace\.test>", message["Message-ID"])
    assert (_SMTP.kwargs["use_tls"], _SMTP.kwargs["start_tls"], "tls_context" in _SMTP.kwargs) == (use_tls, start_tls, False)


@pytest.mark.parametrize("address,ok", [("o'brien@clinic.org", True), ("first.last+tag@clinic.org", True),
                                         ("a b@clinic.org", False), ("x@y", False), ("", False)])
def test_mvp_064_rfc5322_addresses_are_accepted(address, ok) -> None:
    assert _validate_email_address(address) is ok


async def test_mvp_154_deliver_one_is_one_bounded_attempt() -> None:
    calls = {"n": 0}
    class _Hanging(EmailChannel):
        async def _deliver(self, *a, **k):
            calls["n"] += 1
            await asyncio.sleep(60)
    registry = ChannelRegistry(); registry.register_channel(_Hanging(EmailConfig(max_retries=3)))
    service = NotificationService(TemplateRegistry(), registry)
    rendered = service.render(TemplateType.CRISIS_ALERT, VARS)
    t0 = time.monotonic()
    result = await service.deliver_one(ChannelType.EMAIL, "clin-a@test.invalid", rendered, timeout_seconds=0.2)
    assert (result.success, result.error_message, calls["n"]) == (False, "TimeoutError", 1)
    assert time.monotonic() - t0 < 1.0


@pytest.mark.parametrize("env", ["staging", "production"])
@pytest.mark.parametrize("override", [{"EMAIL_SMTP_TLS_MODE": "none"}, {"EMAIL_ENABLED": "false"}])
def test_review_focus_5_strict_refuses_plaintext_or_disabled_email(monkeypatch, env, override) -> None:
    for name, value in override.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(StartupDependencyError) as exc:
        main._create_notification_service(main.NotificationServiceSettings.load(), environment=env)
    assert exc.value.component == "email_channel"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_290_292_alert_content.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The crisis template requires the old variables and renders an emoji subject carrying the user id (templates.py:268); `O&#39;Brien &amp; Co` appears in the text body (autoescape on every template string, templates.py:107-111, MVP-290), and for the same reason `test_ws08_password_reset_template_keeps_its_contract` fails on `"O'Brien" in r.body` (the password-reset text body WS-08.18 relies on is HTML-escaped today); no `Date` or `Message-ID` header exists (channels.py:300-309, MVP-292) and `EmailConfig` has no `tls_mode`; `o'brien@clinic.org` is rejected (channels.py:35, MVP-064); `NotificationService.render`/`deliver_one` do not exist and `send` retries 4 times with 30 s timeouts (channels.py:227-248, MVP-154); `_create_notification_service` takes no `environment` and accepts plaintext SMTP.
- [ ] **Step 3: Implement** the edits under Files. `deliver_one` wraps `channel.send(target, rendered.subject, rendered.body, rendered.html_body, metadata, max_attempts=1, attempt_timeout_seconds=timeout_seconds)` in `asyncio.wait_for(timeout_seconds + 0.5)`. The crisis and update bodies are one labelled line each, in this order and with these exact prefixes: `Level: `, `Categories: `, `Detected by: `, `Detected at (UTC): `, `Patient: ` (followed by `patient_line`), `Alert id: ` and (crisis alert only) `Acknowledge: ` (followed by `ack_url`), then one sentence that the link is single-use and expires in 24 h. WS-13.13 parses the `Categories: ` line.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.8: Consumer on the event plane: crisis first, durable monitoring windows, visible RISK_ALERT failures, the pending-page sweep
**Covers:** MVP-063, MVP-151, MVP-154, MVP-156
**Severity:** high
**Wave:** 1
**Depends on:** WS-13.13, WS-06.9, WS-04.6, WS-05.2 (WS-06.4's per-partition workers arrive through WS-06.9)
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-06 (default assumed: ELEVATED pages no one; RISK_ALERT monitoring email coalesced per user per window), D-28
**Real infra:** yes
**Files:**
- Modify: `services/notification-service/src/consumers.py` (delete `SafetyEventConsumer`, `UserServiceSettings`, `DeadLetterStoreSeam`, the module singletons and the `solace_events.consumer`/`dead_letter` imports; add `SAFETY_TOPIC = "solace.safety"`, `NOTIFICATION_GROUP = "solace-group-notifications"`, `SAFETY_EVENT_TYPES`, `register_safety_handlers`)
- Create: `services/notification-service/src/domain/monitoring.py`
- Modify: `services/notification-service/src/main.py` (delete `KafkaConfig`, `_require_crisis_consumer_in_strict` and `_start_safety_consumer`; add `build_stores(runtime) -> tuple[CrisisAlertRepository, DeliveryRepository]`, `@dataclass Alerting(episodes, deliveries, roster, pager, digest)`, `build_alerting(*, runtime, alert_settings, notifier, episodes, deliveries) -> Alerting`; the lifespan calls `AlertSettings().validate_for(SPEC.environment)`, builds stores, notifier and alerting, calls `register_safety_handlers(runtime.event_plane, pager=..., digest=...)`, starts `run_periodic("risk_digest", digest.send_due, interval_seconds=RISK_DIGEST_SWEEP_SECONDS)` and `run_periodic("crisis_page_retry", pager.retry_pending_pages, interval_seconds=CRISIS_PAGE_RETRY_SWEEP_SECONDS)` (crisis-review F1: pending pages are retried here, never by holding the consumer) and registers both with `runtime.register_background_task`, and sets `app.state.pager`, `app.state.crisis_alerts`, `app.state.deliveries`, `app.state.background_tasks`; cancels the tasks before `await stop_service_runtime(runtime)`)
- Modify (delete): `services/notification-service/tests/test_crisis_consumer.py`, `services/notification-service/tests/test_mvp_287_consumer_lifecycle.py` (their properties are re-asserted below: total failure is retried through the plane, no half-built singleton exists)
- Test: `services/notification-service/tests/test_mvp_063_154_156_safety_consumer.py`, `tests/realinfra/notification-alert-delivery/test_mvp_063_151_crisis_page_real_infra.py`
**Interfaces:**
- Consumes: IF-05 `ServiceEventPlane.subscribe(topic, event_types, handler, *, group_id, max_attempts=5)` (allowed after `start()`, WS-06.8), `start_service_event_plane(service_name, *, pool, kafka_settings=None, strict)`; IF-01 `runtime.event_plane`, `runtime.register_background_task`; WS-13.13 `handle_safety_event(event, *, pager, monitor)`; WS-13.6 `CrisisPager` (with `retry_pending_pages`), `CRISIS_PAGE_RETRY_SWEEP_SECONDS`, `MONITORING_ALERT_DELIVERY_TOTAL`; WS-13.5 `RosterClient`, `AlertSettings`; IF-04 `service_auth_headers("user-service")`; IF-19 `pg_migrated`, `kafka_bootstrap`, `smtp_sink`.
- Produces: `register_safety_handlers(plane, *, pager: CrisisPager, digest: RiskDigest, topic: str = SAFETY_TOPIC, group_id: str = NOTIFICATION_GROUP) -> None` (one `subscribe` of `SAFETY_EVENT_TYPES = ("safety.crisis.detected", "safety.escalation.triggered", "safety.assessment.completed")`; the WS-06 request for WS-06.9); `monitoring.py`: `RISK_DIGEST_SWEEP_SECONDS = 60`, `RISK_DIGEST_MAX_ATTEMPTS = 10`, `class RiskDigest(*, episodes, deliveries, roster, notifier, settings, clock=None)` with `async record(*, user_id: UUID, level: str, categories: Sequence[str], correlation_id: UUID) -> None` (one `record_risk` upsert; never sends mail) and `async send_due() -> int` (returns the number of windows delivered; one `RISK_ALERT` email per due window to the non-backup on-call clinicians, then to the fallback mailbox when none of them was reached; `solace_monitoring_alert_delivery_total{outcome}` counts each window attempt once as `delivered` or `failed`; every recipient attempt is recorded with `claim(kind="risk_digest", channel="email", template_id="risk_alert", subject_user_id=user_id)`; a failed window logs ERROR `risk_alert_delivery_failed` and `mark_risk_failed_attempt`; at `RISK_DIGEST_MAX_ATTEMPTS` the window is `failed` and CRITICAL `risk_alert_undeliverable` is logged); `async run_periodic(name: str, fn: Callable[[], Awaitable[int]], *, interval_seconds: float) -> None` (loops forever; logs ERROR `periodic_task_failed` with `task` and `error_type` per failed iteration; re-raises `CancelledError`). `build_stores` returns Postgres repositories when `runtime.pg_pool` is set, otherwise in-memory ones with WARNING `notification_store_in_memory` (development/test only; strict raises `StartupDependencyError("notification_store", "postgres pool missing")`).

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_063_154_156_safety_consumer.py` and the real-infra file
```python
# test_mvp_063_154_156_safety_consumer.py
"""MVP-063/151/154/156 (+MVP-086/181 notification side): one plane subscription; crisis never waits on monitoring mail."""
from __future__ import annotations

import time
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
import structlog
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from notification_fakes import FakeNotifier, FakeRoster, FixedClock, clinician, make_pager
from solace_events.schemas import CrisisDetectedEvent, CrisisLevel, EventMetadata, SafetyAssessmentEvent
from src import consumers
from src.consumers import handle_safety_event, register_safety_handlers
from src.domain import crisis_paging, monitoring
from src.domain.crisis_paging import CrisisPager
from src.domain.monitoring import RiskDigest
import src.main as main

C1 = clinician("clin-c1@test.invalid")
FALLBACK = "fallback-oncall@test.invalid"


class _RecordingPlane:
    def __init__(self) -> None:
        self.subscriptions: list = []
    def subscribe(self, topic, event_types, handler, *, group_id, max_attempts=5) -> None:
        self.subscriptions.append((topic, tuple(event_types), group_id))


def _env(notifier, clock=None):
    clock = clock or FixedClock()
    pager, episodes, deliveries = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=clock)
    digest = RiskDigest(episodes=episodes, deliveries=deliveries, roster=FakeRoster(on_call=[C1]), notifier=notifier,
                        settings=pager.settings, clock=clock)
    return pager, digest, episodes, clock


def _assessment(user, level=CrisisLevel.ELEVATED) -> SafetyAssessmentEvent:
    return SafetyAssessmentEvent(user_id=user, risk_level=level, risk_score=Decimal("0.6"), detection_layer=1,
                                 recommended_action="monitor", metadata=EventMetadata(source_service="safety-service"))


def test_mvp_063_handlers_subscribe_once_through_the_plane() -> None:
    plane = _RecordingPlane()
    register_safety_handlers(plane, pager=object(), digest=object())
    assert plane.subscriptions == [("solace.safety", ("safety.crisis.detected", "safety.escalation.triggered",
                                                      "safety.assessment.completed"), "solace-group-notifications")]


def test_mvp_063_no_local_dead_letter_path_or_kafka_gate_remains() -> None:
    for name in ("SafetyEventConsumer", "DeadLetterStore", "DeadLetterStoreSeam", "initialize_safety_consumer",
                 "UserServiceSettings", "_safety_consumer"):
        assert not hasattr(consumers, name), name
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "KafkaConfig" not in source and "_require_crisis_consumer" not in source


async def test_mvp_154_monitoring_events_never_send_mail_inline() -> None:
    notifier = FakeNotifier()
    pager, digest, episodes, clock = _env(notifier)
    t0 = time.monotonic()
    await handle_safety_event(_assessment(uuid4()), pager=pager, monitor=digest.record)
    assert time.monotonic() - t0 < 0.1 and notifier.sent == []
    assert await episodes.due_risk_windows(now=clock(), window_seconds=0) != []


async def test_mvp_154_a_hanging_relay_cannot_stall_the_crisis_handler(monkeypatch) -> None:
    monkeypatch.setattr(crisis_paging, "CRISIS_DELIVERY_TIMEOUT_SECONDS", 0.2)
    pager, digest, episodes, _ = _env(FakeNotifier(hang={C1.email, FALLBACK}))
    event = CrisisDetectedEvent(user_id=uuid4(), crisis_level=CrisisLevel.CRITICAL, trigger_indicators=["suicidal_ideation"],
                                detection_layer=1, confidence=Decimal("0.9"), escalation_action="page_on_call",
                                metadata=EventMetadata(source_service="safety-service"))
    t0 = time.monotonic()
    await handle_safety_event(event, pager=pager, monitor=digest.record)          # returns; the sweep owns the retry (crisis-review F1)
    assert time.monotonic() - t0 < 1.5
    [episode] = await episodes.list_episodes(statuses=["open"])
    assert episode.page_count == 0 and episode.pages_requested == 1


async def test_mvp_086_elevated_assessments_coalesce_into_one_risk_alert() -> None:
    notifier = FakeNotifier()
    pager, digest, _, clock = _env(notifier)
    user = uuid4()
    for _ in range(5):
        await handle_safety_event(_assessment(user), pager=pager, monitor=digest.record)
        clock.advance(120)
    assert await digest.send_due() == 0
    clock.advance(300)
    assert await digest.send_due() == 1
    assert notifier.sent[0].subject == "[Solace RISK ELEVATED] monitoring summary - 5 assessment(s)"
    assert await digest.send_due() == 0


async def test_mvp_156_risk_alert_failure_is_counted_logged_and_retried() -> None:
    notifier = FakeNotifier(reject={C1.email, FALLBACK})
    pager, digest, episodes, clock = _env(notifier)
    user = uuid4()
    await digest.record(user_id=user, level="ELEVATED", categories=["hopelessness"], correlation_id=uuid4())
    clock.advance(900)
    before = REGISTRY.get_sample_value("solace_monitoring_alert_delivery_total", {"outcome": "failed"}) or 0.0
    with structlog.testing.capture_logs() as logs:
        assert await digest.send_due() == 0
    assert (REGISTRY.get_sample_value("solace_monitoring_alert_delivery_total", {"outcome": "failed"}) or 0.0) - before == 1.0
    failed = [l for l in logs if l["event"] == "risk_alert_delivery_failed"]
    assert failed and failed[0]["log_level"] == "error" and "test.invalid" not in repr(logs)
    notifier.reject.clear()
    assert await digest.send_due() == 1


async def test_mvp_156_risk_alert_gives_up_loudly(monkeypatch) -> None:
    monkeypatch.setattr(monitoring, "RISK_DIGEST_MAX_ATTEMPTS", 2)
    pager, digest, episodes, clock = _env(FakeNotifier(reject={C1.email, FALLBACK}))
    await digest.record(user_id=uuid4(), level="ELEVATED", categories=[], correlation_id=uuid4())
    clock.advance(900)
    with structlog.testing.capture_logs() as logs:
        await digest.send_due(); await digest.send_due()
    assert any(l["event"] == "risk_alert_undeliverable" and l["log_level"] == "critical" for l in logs)
    assert await digest.send_due() == 0


async def test_high_assessment_folds_into_the_crisis_episode() -> None:
    notifier = FakeNotifier()
    pager, digest, _, _ = _env(notifier)
    user, turn = uuid4(), uuid4()
    meta = EventMetadata(correlation_id=turn, source_service="safety-service")
    crisis = CrisisDetectedEvent(user_id=user, crisis_level=CrisisLevel.HIGH, trigger_indicators=["self_harm"], detection_layer=1,
                                 confidence=Decimal("0.8"), escalation_action="page_on_call", metadata=meta)
    await handle_safety_event(crisis, pager=pager, monitor=digest.record)
    await handle_safety_event(_assessment(user, CrisisLevel.HIGH).model_copy(update={"metadata": meta}), pager=pager, monitor=digest.record)
    assert len(notifier.sent) == 1


def test_mvp_063_lifespan_wires_the_plane_and_the_digest(monkeypatch) -> None:
    monkeypatch.setenv("EVENTS_ENABLED", "false")
    with TestClient(main.app):
        state = main.app.state
        assert isinstance(state.pager, CrisisPager)
        assert not state.background_tasks["risk_digest"].done()
        assert not state.background_tasks["crisis_page_retry"].done()          # crisis-review F1
```
```python
# tests/realinfra/notification-alert-delivery/test_mvp_063_151_crisis_page_real_infra.py
"""MVP-063/151/150 on real Kafka, Postgres (solace_app) and mailpit."""
# pytestmark = [pytest.mark.realinfra, pytest.mark.integration, pytest.mark.timeout(240)]
# fixture `alert_plane(pg_migrated, kafka_bootstrap, smtp_sink)`: tag = uuid4().hex[:8];
#   plane = await start_service_event_plane(f"notification-service-{tag}", pool=pg_migrated.app_pool,
#       kafka_settings=KafkaSettings(bootstrap_servers=kafka_bootstrap.bootstrap), strict=False)
#   notifier = create_notification_service(email_config=EmailConfig(smtp_host=smtp_sink.smtp_host, smtp_port=smtp_sink.smtp_port,
#       tls_mode="none", from_email="alerts@test.invalid"))
#   pager/digest over PostgresCrisisAlertRepository and PostgresDeliveryRepository(pg_migrated.app_pool), a test-local
#       StaticRoster(on_call=[clinician(f"clin-{tag}@test.invalid")], care_teams=None) with mutable lists, and AlertSettings(fallback_oncall_email=f"fallback-{tag}@test.invalid",
#       ack_base_url="https://api.example.test", user_service_url="http://user-service:8001")
#   register_safety_handlers(plane, pager=..., digest=..., topic=f"test.safety.{tag}", group_id=f"solace-group-notifications-{tag}")
#   yields (plane, tag, topic); shutdown_service_event_plane in teardown
# test_mvp_151_redelivered_crisis_pages_once:
#   ev = CrisisDetectedEvent(CRITICAL, trigger_indicators=["suicidal_ideation"], metadata=EventMetadata(correlation_id=turn,
#       source_service="safety-service")); produce ev twice, then ev with a new event_id and the same correlation id
#   smtp_sink.wait_for(f"clin-{tag}@test.invalid", "[Solace CRISIS CRITICAL]", timeout=60); sleep 10;
#   len(smtp_sink.messages(to=clin, subject_contains="[Solace CRISIS")) == 1
#   superuser: SELECT page_count, status FROM crisis_alerts WHERE $1 = ANY(correlation_ids) -> (1, 'open')
# test_mvp_150_page_is_recorded_without_body: notifications row for that alert: body IS NULL, status 'DELIVERED',
#   recipient_hash = sha256(clin), kind 'crisis_page'; the mail text contains "/api/v1/notifications/crisis-alerts/ack/"
#   and neither "kill" nor the user id
# test_mvp_063_rejected_recipient_pages_the_fallback_without_holding_the_consumer (crisis-review F1):
#   StaticRoster on_call=[clinician(f"reject-{tag}@test.invalid")]; produce one CRITICAL event;
#   smtp_sink.wait_for(f"fallback-{tag}@test.invalid", "[Solace CRISIS", 60); committed_offset(group, topic) advances past the
#   event within 10 s of the fallback page and no event_dead_letter row exists for it (column names per slot 010);
#   crisis_alerts row (page_count 0, pages_requested 1, 'open'); after 3 sweeps of pager.retry_pending_pages() the fallback
#   still has exactly 1 message
# test_crisis_review_f1_one_users_unreachable_recipient_does_not_delay_another_users_page:
#   StaticRoster on_call=[clinician(f"clin-{tag}@test.invalid")], care_teams={user_a: [clinician(f"reject-{tag}@test.invalid")]};
#   produce user A's four HIGH events of one turn (orchestrator crisis, safety crisis, escalation, assessment), then user B's
#   CRITICAL crisis event; smtp_sink.wait_for(f"clin-{tag}@test.invalid", ref of B's episode, timeout=20) succeeds, and the
#   fallback mailbox holds exactly 1 page for A's episode
# test_crisis_review_f1_total_failure_is_retried_by_the_sweep_until_delivered:
#   fallback_oncall_email=f"reject-fallback-{tag}@test.invalid" and the on-call clinician reject-{tag}: no mail is accepted;
#   produce one CRITICAL event; no event_dead_letter row after 20 s; swap the StaticRoster on-call to clin-{tag} and call
#   pager.retry_pending_pages(); clin-{tag} gets exactly 1 page and the crisis_alerts row shows page_count 1
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_063_154_156_safety_consumer.py -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery\test_mvp_063_151_crisis_page_real_infra.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `ImportError: cannot import name 'register_safety_handlers'` and `src.domain.monitoring` is missing; `SafetyEventConsumer` with its in-memory `DeadLetterStore` default still exists (consumers.py:129-165, MVP-063) and main.py still gates the consumer on `KafkaConfig.enabled`; the ELEVATED assessment would be mailed inline (consumers.py:533-566, MVP-154/086) and a failed RISK_ALERT returns silently (MVP-156); `background_tasks` has no `crisis_page_retry` (crisis-review F1). In the real-infra file, no page is ever sent because nothing subscribes notification to the plane.
- [ ] **Step 3: Implement** `register_safety_handlers` (the handler is `lambda event: handle_safety_event(event, pager=pager, monitor=digest.record)`), `RiskDigest`, `run_periodic`, the `crisis_page_retry` task and the main.py composition. The roster client's headers callable is `lambda: service_auth_headers("user-service")`. The legacy `initialize_event_bridge` call stays as in WS-13.2.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`; then the same with `D:\Repo\Contextual-Chatbot\tests\alignment`, `D:\Repo\Contextual-Chatbot\tests\bootstrap\test_service_conformance.py -k notification-service` and `D:\Repo\Contextual-Chatbot\tests\contracts`; then the real-infra command on `D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery`. Expected: all pass. Report to WS-05 that the W1 e2e hops `clinician_page` and `fallback_page` are expected to pass once WS-14's IF-08 routes land.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.9: Acknowledgement: one-time link, and acknowledge, resolve and list for clinicians
**Covers:** MVP-082
**Severity:** high
**Wave:** 1
**Depends on:** WS-13.8
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-08 (default assumed: the one-time link works without login; the clinician routes need MFA from W4), D-28
**Real infra:** no
**Files:**
- Create: `services/notification-service/src/domain/ack_pages.py` (`ACK_RESPONSE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Robots-Tag": "noindex", "Content-Security-Policy": "default-src 'none'; form-action 'self'"}`; `render_confirm_page(episode: CrisisEpisode) -> str`; `render_result_page(outcome: AckOutcome) -> str`; every value through `html.escape`; no user identity on any page)
- Modify: `services/notification-service/src/api.py` (new `crisis_router = APIRouter(prefix="/api/v1/notifications/crisis-alerts", tags=["crisis-alerts"])`; dependencies `_crisis_alerts(request) -> CrisisAlertRepository` from `request.app.state.crisis_alerts` and `_now(request) -> datetime` from `request.app.state.clock` when set)
- Modify: `services/notification-service/src/main.py` (`routers=[(notification_router, ""), (crisis_router, "")]`)
- Create: `services/notification-service/tests/test_mvp_082_acknowledgement_api.py` (test; WS-13.16 modifies it in W4)
**Interfaces:**
- Consumes: WS-13.4 `acknowledge_with_token`, `acknowledge`, `resolve`, `list_episodes`, `get`; IF-04 `get_current_user`, `require_roles("clinician", "admin")`, `AuthenticatedUser`; IF-09 link `{NOTIFICATION_ACK_BASE_URL}/api/v1/notifications/crisis-alerts/ack/{token}` routed by Caddy (WS-03.6).
- Produces: `GET /api/v1/notifications/crisis-alerts/ack/{token}` → 200 HTML confirm form (`<form method="post">`, the alert ref and level; no state change), 404 HTML when the token is unknown or expired; `POST` same path → 200 HTML (acknowledged, already acknowledged or already resolved), 404 when unknown or expired; the token is never echoed. `POST /api/v1/notifications/crisis-alerts/{alert_id}/acknowledge` and `.../resolve` (CLINICIAN or ADMIN) → 200 `{alert_id, status, acknowledged_at | resolved_at}`, 404 `{"detail": "ALERT_NOT_FOUND"}`, 409 `{"detail": "ALERT_RESOLVED"}` for acknowledging a resolved alert; `GET /api/v1/notifications/crisis-alerts?status=open|acknowledged` (CLINICIAN or ADMIN) → `{"alerts": [{alert_id, user_id, opened_at, max_level, status, page_count, sources, categories, acknowledged_at}]}`, no contact data. A principal whose `user_id` is not a UUID gets 403.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_082_acknowledgement_api.py`
```python
"""MVP-082: the escalation lifecycle closes: one-time link, clinician acknowledge/resolve and list."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

import notification_repo_contract as contract
from notification_fakes import FixedClock
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from src import api
from src.infrastructure.crisis_alert_repository import InMemoryCrisisAlertRepository

CLINICIAN = uuid4()


@pytest.fixture
def harness():
    clock, repo = FixedClock(), InMemoryCrisisAlertRepository()
    app = FastAPI()
    app.include_router(api.crisis_router)
    app.state.crisis_alerts, app.state.clock = repo, clock
    principal = {"roles": ["clinician"], "user_id": str(CLINICIAN)}
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=principal["user_id"], token_type=TokenType.ACCESS, roles=principal["roles"], permissions=[])
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return SimpleNamespace(app=app, repo=repo, clock=clock, client=client, principal=principal)


async def _seed(h):
    ep = await contract.open_episode(h.repo, h.clock)
    token = await h.repo.issue_ack_token(ep.alert_id, recipient_hash="a" * 64, now=h.clock(), ttl_seconds=86_400)
    return ep, f"/api/v1/notifications/crisis-alerts/ack/{token}", token


async def test_mvp_082_get_shows_a_confirm_form_and_changes_nothing(harness) -> None:
    ep, path, _ = await _seed(harness)
    r = await harness.client.get(path)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert '<form method="post"' in r.text and ep.ref in r.text
    assert r.headers["cache-control"] == "no-store" and r.headers["referrer-policy"] == "no-referrer"
    assert (await harness.repo.get(ep.alert_id)).status == "open"


async def test_mvp_082_post_acknowledges_exactly_once(harness) -> None:
    ep, path, _ = await _seed(harness)
    first = await harness.client.post(path)
    assert first.status_code == 200 and "acknowledged" in first.text.lower()
    stored = await harness.repo.get(ep.alert_id)
    assert (stored.status, stored.acknowledged_by) == ("acknowledged", None)
    harness.clock.advance(60)
    second = await harness.client.post(path)
    assert second.status_code == 200 and "already acknowledged" in second.text.lower()
    assert (await harness.repo.get(ep.alert_id)).acknowledged_at == stored.acknowledged_at


@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_mvp_082_unknown_or_expired_tokens_are_404_and_never_echoed(harness, method) -> None:
    _, path, token = await _seed(harness)
    unknown = await harness.client.request(method, "/api/v1/notifications/crisis-alerts/ack/" + "x" * 43)
    harness.clock.advance(86_401)
    expired = await harness.client.request(method, path)
    assert (unknown.status_code, expired.status_code) == (404, 404) and token not in expired.text


async def test_review_focus_3_a_token_acknowledges_only_its_own_episode(harness) -> None:
    a, path_a, _ = await _seed(harness)
    b, _, _ = await _seed(harness)
    await harness.client.post(path_a)
    assert ((await harness.repo.get(a.alert_id)).status, (await harness.repo.get(b.alert_id)).status) == ("acknowledged", "open")


async def test_mvp_082_clinician_acknowledges_then_resolves_by_id(harness) -> None:
    ep, _, _ = await _seed(harness)
    base = f"/api/v1/notifications/crisis-alerts/{ep.alert_id}"
    r = await harness.client.post(f"{base}/acknowledge")
    assert r.status_code == 200 and r.json()["status"] == "acknowledged"
    assert (await harness.repo.get(ep.alert_id)).acknowledged_by == CLINICIAN
    assert (await harness.client.post(f"{base}/resolve")).json()["status"] == "resolved"
    conflict = await harness.client.post(f"{base}/acknowledge")
    assert (conflict.status_code, conflict.json()) == (409, {"detail": "ALERT_RESOLVED"})


async def test_mvp_082_end_users_are_refused_and_unknown_alerts_are_404(harness) -> None:
    ep, _, _ = await _seed(harness)
    missing = await harness.client.post(f"/api/v1/notifications/crisis-alerts/{uuid4()}/acknowledge")
    assert (missing.status_code, missing.json()) == (404, {"detail": "ALERT_NOT_FOUND"})
    harness.principal["roles"] = ["user"]
    assert (await harness.client.post(f"/api/v1/notifications/crisis-alerts/{ep.alert_id}/acknowledge")).status_code == 403
    assert (await harness.client.get("/api/v1/notifications/crisis-alerts?status=open")).status_code == 403


async def test_mvp_082_clinicians_list_open_alerts_without_contact_data(harness) -> None:
    ep, _, _ = await _seed(harness)
    body = (await harness.client.get("/api/v1/notifications/crisis-alerts?status=open")).json()
    [item] = [a for a in body["alerts"] if a["alert_id"] == str(ep.alert_id)]
    assert set(item) == {"alert_id", "user_id", "opened_at", "max_level", "status", "page_count", "sources", "categories", "acknowledged_at"}
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_082_acknowledgement_api.py -o "addopts=" -p no:logging -q`
Expected: FAIL in the fixture with `AttributeError: module 'src.api' has no attribute 'crisis_router'`: no list, acknowledge or resolve API exists anywhere, the only acknowledge/resolve code mutates safety's in-process dict, and alert links point at a non-existent dashboard (MVP-082).
- [ ] **Step 3: Implement** the router and pages per Produces. The link routes take no authentication dependency (the 256-bit token is the credential); GET never writes; POST maps `AckOutcome` to status codes as listed. The clinician routes use `Depends(require_roles("clinician", "admin"))` and convert `user_id` with `UUID(...)` (403 on `ValueError`). Log `crisis_alert_acknowledged` with `alert_ref` and `via` (`link` or `api`) only.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then the same with `D:\Repo\Contextual-Chatbot\tests\bootstrap\test_service_conformance.py -k notification-service`. Expected: all pass. Report to WS-05 that hop `ack_link` is expected to pass at the W1 barrier.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.10: Re-page unacknowledged episodes to the backup on-call and the fallback mailbox, plus the runbook
**Covers:** MVP-082
**Severity:** high
**Wave:** 1
**Depends on:** WS-13.9
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-06 (default assumed: re-page after 900 s, at most 3 times, to the backup on-call and the fallback mailbox)
**Real infra:** yes
**Files:**
- Modify: `services/notification-service/src/domain/crisis_paging.py` (`CrisisPager.repage_due() -> int`; `CRISIS_ALERTS_UNACKNOWLEDGED.set(await episodes.count_open())` at the end of every sweep)
- Modify: `services/notification-service/src/main.py` (start `run_periodic("crisis_repager", alerting.pager.repage_due, interval_seconds=max(5.0, min(30.0, alert_settings.repage_after_seconds / 2)))`, register it with `runtime.register_background_task` and add it to `app.state.background_tasks`)
- Create: `docs/runbooks/crisis-escalation.md` (sections `## Episode lifecycle`, `## Acknowledging a crisis alert`, `## Re-paging and the fallback mailbox`, `## When delivery fails (DLQ and re-drive)`, `## Configuration`, the last listing `NOTIFICATION_FALLBACK_ONCALL_EMAIL`, `NOTIFICATION_ALERT_COALESCE_SECONDS`, `NOTIFICATION_REPAGE_AFTER_SECONDS`, `NOTIFICATION_MAX_REPAGES`, `NOTIFICATION_ACK_BASE_URL`, `EMAIL_SMTP_TLS_MODE`; a relative link to the WS-04 alert runbook alerting.md in the same directory for `CrisisAlertsUnacknowledged` and the delivery alerts. That WS-04 file is only linked, never edited here; the back-link from it is request R-13-9)
- Test: `services/notification-service/tests/test_mvp_082_repage.py`, `tests/realinfra/notification-alert-delivery/test_mvp_082_repage_races_pg.py`
**Interfaces:**
- Consumes: WS-13.4 `due_for_repage`, `claim_repage`, `count_open`, `mark_page_complete`; WS-13.6 delivery helpers and `CRISIS_ALERTS_UNACKNOWLEDGED`; WS-13.8 `run_periodic`; IF-01 `runtime.register_background_task`.
- Produces: `CrisisPager.repage_due() -> int` (number of re-pages sent). For each due episode it calls `claim_repage(alert_id, expected_repage_count=episode.repage_count, now)`; a `None` claim (acknowledged, resolved or claimed by another sweeper) sends nothing; otherwise it delivers page `seq = claimed.pages_requested` with `page_note = f" (re-page {claimed.repage_count})"` to the backup on-call clinicians (`is_backup=True`) and the fallback mailbox, records and counts every attempt, never raises for a delivery failure, and marks the page complete on any delivery. Background task `crisis_repager` (its death fails `/live`). The gauge feeds WS-04.7's `CrisisAlertsUnacknowledged`.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_082_repage.py` and the real-infra file
```python
# test_mvp_082_repage.py
"""MVP-082 re-paging and review focus 2: unacknowledged episodes re-page, acknowledgements win races, sweeps never raise."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from notification_fakes import FakeNotifier, FakeRoster, FixedClock, clinician, crisis_signal, make_pager
import src.main as main

C1, C3 = clinician("clin-c1@test.invalid"), clinician("clin-c3@test.invalid", backup=True)
FALLBACK = "fallback-oncall@test.invalid"


def _setup(notifier=None):
    clock, notifier = FixedClock(), notifier or FakeNotifier()
    pager, episodes, deliveries = make_pager(roster=FakeRoster(on_call=[C1, C3]), notifier=notifier, clock=clock)
    return pager, episodes, notifier, clock


async def test_mvp_082_unacknowledged_episode_repages_backup_and_fallback() -> None:
    pager, _, notifier, clock = _setup()
    await pager.handle_detection(crisis_signal(uuid4()))
    clock.advance(899)
    assert await pager.repage_due() == 0
    clock.advance(1)
    assert await pager.repage_due() == 1
    assert [len(notifier.to(a)) for a in (C1.email, C3.email, FALLBACK)] == [1, 1, 1]
    assert notifier.to(C3.email)[0].subject.endswith("(re-page 1)")


async def test_mvp_082_repages_stop_at_the_budget() -> None:
    pager, _, notifier, clock = _setup()
    await pager.handle_detection(crisis_signal(uuid4()))
    for _ in range(5):
        clock.advance(900); await pager.repage_due()
    assert len(notifier.to(C3.email)) == 3


async def test_mvp_082_acknowledged_or_resolved_episodes_never_repage() -> None:
    pager, episodes, notifier, clock = _setup()
    a = await pager.handle_detection(crisis_signal(uuid4()))
    b = await pager.handle_detection(crisis_signal(uuid4()))
    await episodes.acknowledge(a.alert_id, clinician_id=C1.user_id, now=clock())
    await episodes.resolve(b.alert_id, clinician_id=C1.user_id, now=clock())
    clock.advance(900)
    assert await pager.repage_due() == 0 and notifier.to(C3.email) == []


async def test_review_focus_2_ack_racing_the_repage_wins(monkeypatch) -> None:
    pager, episodes, notifier, clock = _setup()
    outcome = await pager.handle_detection(crisis_signal(uuid4()))
    clock.advance(900)
    due = await episodes.due_for_repage(now=clock(), after_seconds=900, max_repages=3)
    await episodes.acknowledge(outcome.alert_id, clinician_id=C1.user_id, now=clock())
    async def stale(**kw): return due
    monkeypatch.setattr(episodes, "due_for_repage", stale)
    assert await pager.repage_due() == 0 and notifier.to(C3.email) == []


async def test_review_focus_2_accepted_then_bounced_page_is_repaged() -> None:
    pager, _, notifier, clock = _setup()
    await pager.handle_detection(crisis_signal(uuid4()))   # relay accepted it; the mailbox bounced it later
    clock.advance(900)
    assert await pager.repage_due() == 1 and len(notifier.to(FALLBACK)) == 1


async def test_repage_failures_are_counted_not_raised() -> None:
    pager, _, notifier, clock = _setup(FakeNotifier(reject={C3.email, FALLBACK}))
    await pager.handle_detection(crisis_signal(uuid4()))
    before = REGISTRY.get_sample_value("solace_crisis_alert_delivery_total", {"outcome": "failed", "channel": "email"}) or 0.0
    clock.advance(900)
    assert await pager.repage_due() == 0
    assert (REGISTRY.get_sample_value("solace_crisis_alert_delivery_total", {"outcome": "failed", "channel": "email"}) or 0.0) - before == 2.0


async def test_if09_unacknowledged_gauge_tracks_open_episodes() -> None:
    pager, _, _, clock = _setup()
    for _ in range(2):
        await pager.handle_detection(crisis_signal(uuid4()))
    await pager.repage_due()
    assert REGISTRY.get_sample_value("solace_crisis_alerts_unacknowledged") == 2.0


def test_mvp_082_repager_runs_as_a_registered_background_task(monkeypatch) -> None:
    import time as _time
    monkeypatch.setenv("EVENTS_ENABLED", "false")
    with TestClient(main.app) as client:
        task = main.app.state.background_tasks["crisis_repager"]
        task.get_loop().call_soon_threadsafe(task.cancel)      # the app loop runs in the TestClient's thread
        deadline = _time.monotonic() + 2.0
        while client.get("/live").status_code != 503 and _time.monotonic() < deadline:
            _time.sleep(0.05)
        assert client.get("/live").status_code == 503


def test_runbook_documents_the_lifecycle() -> None:
    text = (Path(__file__).resolve().parents[3] / "docs" / "runbooks" / "crisis-escalation.md").read_text(encoding="utf-8")
    for heading in ("## Episode lifecycle", "## Acknowledging a crisis alert", "## Re-paging and the fallback mailbox",
                    "## When delivery fails (DLQ and re-drive)", "## Configuration"):
        assert heading in text
    assert "NOTIFICATION_REPAGE_AFTER_SECONDS" in text and "NOTIFICATION_FALLBACK_ONCALL_EMAIL" in text
```
```python
# tests/realinfra/notification-alert-delivery/test_mvp_082_repage_races_pg.py
# pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
# Two CrisisPager instances share PostgresCrisisAlertRepository/PostgresDeliveryRepository(pg_migrated.app_pool) and one
# FakeNotifier (on_call=[C1, C3]); one shared FixedClock.
# test_review_focus_2_two_sweepers_repage_once: page one episode; clock.advance(900);
#   await asyncio.gather(p1.repage_due(), p2.repage_due()) -> sum == 1 and len(notifier.to(C3.email)) == 1
# test_review_focus_2_ack_between_listing_and_claim_blocks_the_repage: list due, acknowledge through the repository,
#   then repage_due() with the stale list (monkeypatched due_for_repage) -> 0 and no email
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_082_repage.py -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery\test_mvp_082_repage_races_pg.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `AttributeError: 'CrisisPager' object has no attribute 'repage_due'`, `KeyError: 'crisis_repager'` and `FileNotFoundError` for the runbook: nothing re-pages when nobody responds (MVP-082).
- [ ] **Step 3: Implement** `repage_due` per Produces, the background task wiring and the runbook. Recipients for a re-page come from one `on_call()` lookup; on `RosterUnavailable` only the fallback mailbox is paged.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then the real-infra command on `D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery`. Expected: all pass. Report to WS-05 that hop `repage` is expected to pass at the W1 barrier.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.11: Health that reflects the real channel registry
**Covers:** MVP-153
**Severity:** medium
**Wave:** 1
**Depends on:** WS-13.10
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/notification-service/src/api.py` (`/health` (lines 455-469) and `/channels` (430-452) read `request.app.state.notification_service.channel_registry`; delete `_channel_registry` and `_get_channel_registry` at 104-128)
- Modify: `services/notification-service/src/main.py` (`runtime.register_readiness_check("channels", _channels_check, critical=False)`: `(True, "email")` when the email channel is registered and available, else `(False, "email unavailable")`)
- Modify: `services/notification-service/tests/test_api.py` (remove the `@pytest.mark.skip` from `TestHealthEndpoint.test_notification_health` and `TestChannelEndpoints.test_list_channels`, and set `app.state.notification_service` in their fixtures)
- Test: `services/notification-service/tests/test_mvp_153_health.py`
**Interfaces:**
- Consumes: WS-13.7 `NotificationService.channel_registry`; IF-01 `register_readiness_check(name, check, *, critical)`.
- Produces: `GET /api/v1/notifications/health` → `{status: "healthy" | "unhealthy", timestamp, channels: {<type>: bool}}` from the real registry; `GET /api/v1/notifications/channels` (ADMIN) lists the registered channels; `/ready` includes `checks.channels` (non-critical, so a relay outage shows in readiness detail and in the crisis delivery alert without hiding in-band resources).

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_153_health.py`
```python
"""MVP-153: notification health signals reflect the channels the service really holds."""
from __future__ import annotations

from fastapi.testclient import TestClient

from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
import src.main as main


def test_mvp_153_health_reflects_the_real_email_channel(monkeypatch) -> None:
    monkeypatch.setenv("EVENTS_ENABLED", "false")
    with TestClient(main.app) as client:
        body = client.get("/api/v1/notifications/health").json()
    assert body["status"] == "healthy" and body["channels"] == {"email": True}


def test_mvp_153_channels_lists_what_is_registered(monkeypatch) -> None:
    monkeypatch.setenv("EVENTS_ENABLED", "false")
    main.app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id="00000000-0000-4000-8000-000000000001", token_type=TokenType.ACCESS, roles=["admin"], permissions=[])
    try:
        with TestClient(main.app) as client:
            assert client.get("/api/v1/notifications/channels").json() == [
                {"channel_type": "email", "status": "active", "is_available": True}]
    finally:
        main.app.dependency_overrides.clear()


def test_mvp_153_ready_reports_the_channel_check(monkeypatch) -> None:
    monkeypatch.setenv("EVENTS_ENABLED", "false")
    with TestClient(main.app) as client:
        check = client.get("/ready").json()["checks"]["channels"]
    assert (check["ok"], check["critical"]) == (True, False)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_153_health.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `/api/v1/notifications/health` returns `{"status": "unhealthy", "channels": {}}` and `/channels` returns `[]` because both read a fresh empty `ChannelRegistry()` (api.py:120-128, 442-463; MVP-153); `/ready` has no `channels` check.
- [ ] **Step 3: Implement** the edits under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass, with the two un-skipped API tests passing.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.12: Send endpoints: service permission, 502 on failed delivery, recorded attempts and address-free logs
**Covers:** MVP-157, MVP-150
**Severity:** medium
**Wave:** 1
**Depends on:** WS-13.11, WS-07.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: service-only routes check receiver-side permissions; only user-service holds `service:write:notification`), D-28
**Real infra:** no
**Files:**
- Modify: `services/notification-service/src/api.py` (`/send`, `/email`, `/sms` and `/push` depend on `require_service_permission(ServicePermission.WRITE_NOTIFICATION.value)` instead of `get_current_user` and the role checks at 281-282, 321-322, 345-346, 369-370; a result with zero successful deliveries returns `JSONResponse(502, {"error": {"code": "DELIVERY_FAILED", "message": "Notification delivery failed", "request_id": <X-Request-ID or null>}})`; the logs at 283-284 and 323-324 carry `template`, `recipients=<count>` and `service=<caller>` only, never `to_email` or `to_phone`)
- Modify: `services/notification-service/src/domain/service.py` (`NotificationService.__init__(template_registry, channel_registry, *, deliveries: DeliveryRepository | None = None)`; `send_notification` records one `claim(kind=request.metadata.get("kind", "api"), channel, template_id, subject_user_id=recipient.user_id)` plus one `record_attempt` per attempted pair when `deliveries` is set, never the address; `create_notification_service(..., *, deliveries=None)`)
- Modify: `services/notification-service/src/main.py` (`_create_notification_service(settings, *, environment, deliveries=None)` is called after `build_stores` with the delivery repository)
- Modify: `services/notification-service/tests/test_api.py` (the validation-endpoint fixtures override `get_current_service` with `AuthenticatedService(service_name="user-service", permissions=["service:write:notification"])` and set `app.state.notification_service`; remove their `skip` marks)
- Test: `services/notification-service/tests/test_mvp_157_send_endpoints.py`
**Interfaces:**
- Consumes: IF-04 `require_service_permission(*permissions)` with receiver-side resolution, `get_current_service`, `AuthenticatedService`, `ServicePermission.WRITE_NOTIFICATION`, `find_unguarded_service_routes(app)` (WS-07.4); WS-13.4 `DeliveryRepository`.
- Produces: IF-09 REST semantics for `/send` and `/email` (202 only when at least one delivery succeeded, else 502 `DELIVERY_FAILED`), the same for `/sms` and `/push`; satisfies the WS-07 request for WS-07.9. API sends recorded as `notifications` kinds `api_send`, `api_email`, `api_sms`, `api_push`.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_157_send_endpoints.py`
```python
"""MVP-157/MVP-150 + MVP-314 (supporting): send routes are service-only, truthful about failure and leave a PHI-free record."""
from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from solace_security.middleware import AuthenticatedService, find_unguarded_service_routes, get_current_service
from src import api
from src.domain.channels import ChannelRegistry, ChannelType, DeliveryResult, EmailChannel, EmailConfig
from src.domain.service import NotificationService
from src.domain.templates import TemplateRegistry
from src.infrastructure.delivery_repository import InMemoryDeliveryRepository

EMAIL = {"to_email": "user-x@test.invalid", "template_type": "email_verification",
         "variables": {"display_name": "X", "verification_link": "https://app.test/v?token=t", "expiry_hours": 24}}


def _app(service_name: str = "user-service", permissions=("service:write:notification",)):
    registry = ChannelRegistry(); registry.register_channel(EmailChannel(EmailConfig()))
    deliveries = InMemoryDeliveryRepository()
    app = FastAPI(); app.include_router(api.router)
    app.state.notification_service = NotificationService(TemplateRegistry(), registry, deliveries=deliveries)
    app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(service_name=service_name, permissions=list(permissions))
    return app, deliveries


def _result(success: bool) -> DeliveryResult:
    return DeliveryResult(channel_type=ChannelType.EMAIL, recipient="user-x@test.invalid", success=success,
                          error_message=None if success else "SMTPRecipientsRefused")


def test_mvp_314_every_service_route_checks_a_permission() -> None:
    assert find_unguarded_service_routes(_app()[0]) == []


def test_mvp_157_callers_without_the_grant_are_refused() -> None:
    app, _ = _app(service_name="analytics-service", permissions=())
    assert TestClient(app).post("/api/v1/notifications/email", json=EMAIL).status_code == 403


@pytest.mark.parametrize("success,status", [(True, 202), (False, 502)])
def test_mvp_157_status_reflects_delivery(success, status) -> None:
    app, _ = _app()
    with patch.object(EmailChannel, "send", return_value=_result(success)):
        r = TestClient(app).post("/api/v1/notifications/email", json=EMAIL)
    assert r.status_code == status
    if not success:
        assert r.json()["error"]["code"] == "DELIVERY_FAILED"


def test_mvp_150_api_sends_are_recorded_without_body_or_address() -> None:
    app, deliveries = _app()
    with patch.object(EmailChannel, "send", return_value=_result(True)):
        TestClient(app).post("/api/v1/notifications/email", json=EMAIL)
    [record] = deliveries.records()
    assert (record.kind, record.template_id, record.recipient_hash, record.status) == ("api_email", "email_verification", None, "DELIVERED")
    assert "user-x@test.invalid" not in repr(record)


def test_review_focus_4_send_logs_never_carry_the_address() -> None:
    app, _ = _app()
    with patch.object(EmailChannel, "send", return_value=_result(True)), structlog.testing.capture_logs() as logs:
        TestClient(app).post("/api/v1/notifications/email", json=EMAIL)
    assert logs and "user-x@test.invalid" not in repr(logs)
```
(`InMemoryDeliveryRepository.records()` is the inspection helper from WS-13.4.)
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_157_send_endpoints.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The routes depend on `get_current_user`, so the service override is ignored and the request is 401/403 for the wrong reason, and a failed delivery returns 202 with `status=failed` (api.py:314-335, MVP-157); `NotificationService` takes no `deliveries` and records nothing (MVP-150); `api_send_email` logs `to=request.to_email` (api.py:323).
- [ ] **Step 3: Implement** the edits under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then the same with `D:\Repo\Contextual-Chatbot\services\user-service\tests` (user-service's verification-email client now sees 502 on failure). Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.13: Safety-event dispatcher verified against every producer's IF-20 event contract
**Covers:** MVP-299
**Severity:** low
**Wave:** 1
**Depends on:** WS-13.6, WS-05.7
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-28
**Real infra:** no
**Files:**
- Modify: `services/notification-service/src/consumers.py` (add `handle_safety_event`; the legacy `SafetyEventConsumer` stays until WS-13.8)
- Test: `services/notification-service/tests/test_if20_safety_event_consumer_contracts.py` (marker `contract`)
**Interfaces:**
- Consumes: IF-20 `load_event_contracts(dir)`, `verify_event_consumer(contract, handler)` and the files `tests/contracts/events/{orchestrator,safety,therapy,diagnosis}__notification/*.json` (WS-05.7); WS-13.6 `CrisisSignal.from_crisis_event`/`from_escalation_event`/`from_assessment_event`, `CrisisPager.handle_detection`; `solace_events.schemas.EVENT_REGISTRY`; IF-07 `CATEGORIES`.
- Produces: `async def handle_safety_event(event: BaseEvent, *, pager: CrisisPager, monitor: Callable[..., Awaitable[None]]) -> None`: `safety.crisis.detected`, `safety.escalation.triggered` and `safety.assessment.completed` at HIGH/CRITICAL become `pager.handle_detection(signal)`; ELEVATED crisis or assessment events and LOW/MEDIUM escalations call `monitor(user_id=..., level=..., categories=..., correlation_id=...)`; other types and lower levels return. This is the consumer half of the contracts WS-12.8 verifies as producer (see Cross-workstream requests R-13-7).

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_if20_safety_event_consumer_contracts.py`
```python
"""IF-20 consumer half: notification accepts every producer's contract example; one turn from all producers is one page (MVP-299)."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from notification_fakes import FakeNotifier, FakeRoster, FixedClock, clinician, make_pager
from solace_clinical import CATEGORIES
from solace_events.schemas import EVENT_REGISTRY, SessionStartedEvent
from solace_testing.contracts import load_event_contracts, verify_event_consumer
from src.consumers import handle_safety_event

pytestmark = pytest.mark.contract
CONTRACTS = Path(__file__).resolve().parents[3] / "tests" / "contracts" / "events"
K = clinician("clin-k@test.invalid")
EXPECTED = {("orchestrator-service", "safety.crisis.detected"), ("safety-service", "safety.crisis.detected"),
            ("safety-service", "safety.escalation.triggered"), ("safety-service", "safety.assessment.completed"),
            ("therapy-service", "safety.crisis.detected"), ("diagnosis-service", "safety.crisis.detected")}


def _contracts():
    return [c for c in load_event_contracts(CONTRACTS) if c.consumer == "notification-service"]


def _harness():
    notifier, monitored = FakeNotifier(), []
    pager, episodes, _ = make_pager(roster=FakeRoster(on_call=[K]), notifier=notifier, clock=FixedClock())
    async def monitor(**kw):
        monitored.append(kw)
    return pager, episodes, notifier, monitor, monitored


def test_every_crisis_producer_has_a_contract_for_notification() -> None:
    assert {(c.producer, c.event_type) for c in _contracts()} >= EXPECTED


@pytest.mark.parametrize("contract", _contracts(), ids=lambda c: f"{c.producer}:{c.event_type}")
async def test_if20_dispatcher_accepts_the_contract_example(contract) -> None:
    pager, _, notifier, monitor, monitored = _harness()
    await verify_event_consumer(contract, lambda e: handle_safety_event(e, pager=pager, monitor=monitor))
    level = contract.example.get("crisis_level") or contract.example.get("risk_level") or contract.example.get("priority")
    if level in ("HIGH", "CRITICAL"):
        assert len(notifier.sent) == 1 and monitored == []
        line = next(l for l in notifier.sent[0].body.splitlines() if l.startswith("Categories:"))
        codes = {c.strip() for c in line.split(":", 1)[1].split(",") if c.strip() and c.strip() != "none reported"}
        assert codes <= set(CATEGORIES)
    else:
        assert notifier.sent == [] and len(monitored) == 1


async def test_mvp_299_one_turn_reported_by_every_producer_is_one_page() -> None:
    pager, episodes, notifier, monitor, _ = _harness()
    user, turn = str(uuid4()), str(uuid4())
    for c in [c for c in _contracts() if c.event_type == "safety.crisis.detected"]:
        data = {**c.example, "user_id": user, "crisis_level": "CRITICAL",
                "metadata": {**c.example["metadata"], "event_id": str(uuid4()), "correlation_id": turn}}
        await handle_safety_event(EVENT_REGISTRY[c.event_type].model_validate(data), pager=pager, monitor=monitor)
    [episode] = await episodes.list_episodes(statuses=["open"])
    assert len(notifier.sent) == 1
    assert set(episode.sources) == {"orchestrator-service", "safety-service", "therapy-service", "diagnosis-service"}


async def test_unrelated_event_types_are_ignored() -> None:
    pager, _, notifier, monitor, monitored = _harness()
    await handle_safety_event(SessionStartedEvent(user_id=uuid4(), session_number=1), pager=pager, monitor=monitor)
    assert notifier.sent == [] and monitored == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_if20_safety_event_consumer_contracts.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'handle_safety_event' from 'src.consumers'`. Today the three handlers page independently and copy raw indicators into the alert (consumers.py:226-418), so the orchestrator, therapy and diagnosis CrisisDetected events of one turn would each page (MVP-299).
- [ ] **Step 3: Implement** `handle_safety_event` per Produces. Event types are matched on `event.event_type`; the handler never logs event content.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then the same with `D:\Repo\Contextual-Chatbot\tests\contracts`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.14: Erasure: delivery records deleted, crisis episodes pseudonymized (D-24)
**Covers:** none (supporting MVP-029, owned by WS-14; the notification row of IF-16)
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-13.12
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: delivery records hard-deleted; crisis episodes kept for duty of care with `user_id` replaced by the salted-hash pseudonym WS-12.12 uses for escalations; audit rows kept)
**Real infra:** yes
**Files:**
- Create: `services/notification-service/src/domain/erasure.py` (`build_notification_erasure(*, deliveries, episodes, pseudonym: Callable[[UUID], UUID], clock=None) -> UserDataErasure` registering exactly `notification_delivery_records`, `notification_risk_windows`, `crisis_episodes_pseudonymized`)
- Modify: `services/notification-service/src/infrastructure/delivery_repository.py` (`erase_subject(user_id: UUID) -> int`: delete `notifications` with `subject_user_id = user_id`, attempts cascade)
- Modify: `services/notification-service/src/infrastructure/crisis_alert_repository.py` (`delete_risk_window(user_id) -> int`; `pseudonymize_user(user_id, *, pseudonym: UUID, now) -> int`: one UPDATE setting `user_id = pseudonym`, `pseudonymized_at = now` and deleting the unused ack tokens of those episodes)
- Modify: `services/notification-service/src/main.py` (in strict, call `crisis_pseudonym(UUID(int=0))` once at startup with no try/except, so a missing salt is fatal; `register_user_erasure_consumer(runtime.event_plane, service_name="notification-service", erasure=build_notification_erasure(...), purge_caches=None)`)
- Test: `services/notification-service/tests/test_mvp_029_notification_erasure_wiring.py`, `tests/realinfra/notification-alert-delivery/test_mvp_029_erasure_pseudonymize.py`
**Interfaces:**
- Consumes: IF-16 `register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches=None)` (WS-14, W3 phase A); `solace_infrastructure.gdpr.erasure.UserDataErasure`; R-13-3 `solace_security.phi_codec.crisis_pseudonym(user_id: UUID | str) -> UUID` (WS-09, W3 phase A); R-13-2 `ENCRYPTION_SEARCH_HASH_SALT` on notification-service (WS-03).
- Produces: notification's IF-16 erasure stores; consumer group `solace-group-erasure-notification-service` (IF-05 naming).

- [ ] **Step 1: Write the failing test** in the two files
```python
# test_mvp_029_notification_erasure_wiring.py
"""MVP-029 (supporting): notification erases delivery records and pseudonymizes crisis episodes (D-24)."""
from __future__ import annotations

from uuid import UUID, uuid4

import notification_repo_contract as contract
from notification_fakes import FixedClock
from src.domain.erasure import build_notification_erasure
from src.infrastructure.crisis_alert_repository import InMemoryCrisisAlertRepository
from src.infrastructure.delivery_repository import InMemoryDeliveryRepository


def _pseudonym(user_id: UUID) -> UUID:
    return UUID(int=user_id.int ^ 0xFFFF)


def test_mvp_029_registers_exactly_the_notification_stores() -> None:
    erasure = build_notification_erasure(deliveries=InMemoryDeliveryRepository(), episodes=InMemoryCrisisAlertRepository(),
                                         pseudonym=_pseudonym)
    assert erasure.store_names == ("notification_delivery_records", "notification_risk_windows", "crisis_episodes_pseudonymized")


async def test_mvp_029_erasure_keeps_episodes_under_a_pseudonym_and_is_idempotent() -> None:
    clock, episodes, deliveries = FixedClock(), InMemoryCrisisAlertRepository(), InMemoryDeliveryRepository()
    user, other = uuid4(), uuid4()
    mine = await contract.open_episode(episodes, clock, user=user)
    theirs = await contract.open_episode(episodes, clock, user=other)
    claim = await deliveries.claim(**dict(contract.KEY, alert_id=mine.alert_id, page_seq=1, subject_user_id=user), now=clock())
    await deliveries.record_attempt(claim, delivered=True, error_code=None, duration_ms=5, now=clock())
    await episodes.record_risk(user_id=user, level="ELEVATED", categories=[], now=clock())
    erasure = build_notification_erasure(deliveries=deliveries, episodes=episodes, pseudonym=_pseudonym, clock=clock)
    report = await erasure.erase(str(user))
    assert report.deleted == {"notification_delivery_records": 1, "notification_risk_windows": 1, "crisis_episodes_pseudonymized": 1}
    kept = await episodes.get(mine.alert_id)
    assert (kept.user_id, kept.max_level, kept.categories) == (_pseudonym(user), "CRITICAL", ["suicidal_ideation"])
    assert (await episodes.get(theirs.alert_id)).user_id == other
    assert (await erasure.erase(str(user))).total_deleted == 0
```
```python
# tests/realinfra/notification-alert-delivery/test_mvp_029_erasure_pseudonymize.py
# pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
# Postgres repositories on pg_migrated.app_pool; pseudonym = solace_security.phi_codec.crisis_pseudonym with
#   ENCRYPTION_SEARCH_HASH_SALT set by monkeypatch to a 32-byte test value
# test_mvp_029_erasure_completes_and_pseudonymizes: user U with two episodes, delivery records and a risk window; user V
#   with one episode. report = await erasure.erase(str(U)) -> complete; superuser checks:
#   SELECT count(*) FROM notifications WHERE subject_user_id = U == 0; FROM risk_alert_windows WHERE user_id = U == 0;
#   FROM crisis_alerts WHERE user_id = U == 0; both of U's episodes have user_id = crisis_pseudonym(U) and pseudonymized_at
#   NOT NULL with max_level, categories and opened_at unchanged; crisis_alert_ack_tokens for them have no unused rows;
#   V's rows untouched; a second erase deletes nothing
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_029_notification_erasure_wiring.py -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery\test_mvp_029_erasure_pseudonymize.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.domain.erasure'`: `user.deleted` has no notification consumer, so delivery records and episodes keep the user's id forever (MVP-029, notification row).
- [ ] **Step 3: Implement** the erasure module, the repository methods and the main.py wiring. Each deleter takes the string user id from `UserDataErasure`, converts it with `UUID(...)` and returns its row count; the three run in the registered order.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`, then the real-infra command on `D:\Repo\Contextual-Chatbot\tests\realinfra\notification-alert-delivery`, then the unit command on `D:\Repo\Contextual-Chatbot\tests\integration\test_phi_encryption_activation.py`. Expected: all pass (notification stays in `_EXEMPT`: it stores no PHI field and holds only the search salt).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.15: Audit trail for the alert lifecycle (IF-14)
**Covers:** none (supporting MVP-091, owned by WS-09)
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-13.14, WS-09.11, WS-04.10
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: audit rows hold ids and codes only and survive erasure)
**Real infra:** no (WS-09's chain tests and WS-05's W3 suite prove the store)
**Files:**
- Modify: `services/notification-service/src/domain/crisis_paging.py` (`await audit(...)` after every crisis page or re-page attempt)
- Modify: `services/notification-service/src/api.py` (`await audit(...)` after a link or clinician acknowledgement and after a resolve)
- Test: `services/notification-service/tests/test_mvp_091_alert_audit.py`
**Interfaces:**
- Consumes: IF-14 `solace_security.audit_runtime.audit(event_type, *, actor_id, actor_type, action, outcome, resource_type, resource_id=None, subject_user_id=None, details=None)` (never raises), `solace_security.audit.AuditEventType.CRISIS_ALERT_SENT`/`CRISIS_ALERT_FAILED`/`CRISIS_ALERT_ACKNOWLEDGED`, `AuditOutcome`; `ALLOWED_DETAIL_KEYS` (`channel`, `role`, `level`, `attempt`).
- Produces: one audit row per page attempt (`actor_type="service"`, `actor_id="notification-service"`, `action="page"`, `resource_type="crisis_alert"`, `resource_id=str(alert_id)`, `subject_user_id=str(user_id)`, `details={"channel": "email", "role": <recipient_role>, "level": <max_level>, "attempt": <page_seq>}`), and one per acknowledgement or resolve (`actor_type="user"`, `actor_id=<clinician id>` for the API; `actor_type="system"`, `actor_id="ack_link"` for the link; resolve uses `CRISIS_ALERT_ACKNOWLEDGED` with `action="resolve"`).

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_091_alert_audit.py`
```python
"""MVP-091 (supporting): every page attempt and acknowledgement leaves an id-only audit row."""
from __future__ import annotations

from uuid import uuid4

import httpx
from fastapi import FastAPI

import notification_repo_contract as contract
from notification_fakes import FakeNotifier, FakeRoster, FixedClock, clinician, crisis_signal, make_pager
from solace_security.audit import AuditEventType, AuditOutcome
from src import api
from src.domain import crisis_paging

C1 = clinician("clin-c1@test.invalid")


def _recorder(monkeypatch, module):
    calls = []
    async def fake_audit(event_type, **kw):
        calls.append((event_type, kw))
    monkeypatch.setattr(module, "audit", fake_audit)
    return calls


async def test_mvp_091_page_attempts_are_audited_without_addresses(monkeypatch) -> None:
    calls = _recorder(monkeypatch, crisis_paging)
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=FakeNotifier(), clock=FixedClock())
    user = uuid4()
    outcome = await pager.handle_detection(crisis_signal(user))
    [(event_type, kw)] = calls
    assert (event_type, kw["outcome"], kw["resource_type"], kw["resource_id"], kw["subject_user_id"]) == (
        AuditEventType.CRISIS_ALERT_SENT, AuditOutcome.SUCCESS, "crisis_alert", str(outcome.alert_id), str(user))
    assert kw["details"] == {"channel": "email", "role": "on_call", "level": "CRITICAL", "attempt": 1}
    assert "test.invalid" not in repr(calls)


async def test_mvp_091_failed_attempts_are_audited_as_failures(monkeypatch) -> None:
    calls = _recorder(monkeypatch, crisis_paging)
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=FakeNotifier(reject={C1.email}), clock=FixedClock())
    try:
        await pager.handle_detection(crisis_signal(uuid4()))
    except Exception:
        pass
    assert AuditEventType.CRISIS_ALERT_FAILED in [c[0] for c in calls]


async def test_mvp_091_link_acknowledgement_is_audited(monkeypatch) -> None:
    calls = _recorder(monkeypatch, api)
    from src.infrastructure.crisis_alert_repository import InMemoryCrisisAlertRepository
    clock, repo = FixedClock(), InMemoryCrisisAlertRepository()
    ep = await contract.open_episode(repo, clock)
    token = await repo.issue_ack_token(ep.alert_id, recipient_hash="a" * 64, now=clock(), ttl_seconds=86_400)
    app = FastAPI(); app.include_router(api.crisis_router); app.state.crisis_alerts, app.state.clock = repo, clock
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post(f"/api/v1/notifications/crisis-alerts/ack/{token}")
    [(event_type, kw)] = calls
    assert (event_type, kw["actor_type"], kw["actor_id"], kw["resource_id"]) == (
        AuditEventType.CRISIS_ALERT_ACKNOWLEDGED, "system", "ack_link", str(ep.alert_id))
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_091_alert_audit.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `AttributeError: <module 'src.domain.crisis_paging'> does not have the attribute 'audit'` (monkeypatch target missing): no alert lifecycle event reaches the audit trail (MVP-091, notification part).
- [ ] **Step 3: Implement** module-level `from solace_security.audit_runtime import audit` in both modules and the calls per Produces.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.16: MFA on the clinician acknowledge, resolve and list routes; the link stays login-free
**Covers:** none (supporting MVP-344, owned by WS-08)
**Severity:** blocker
**Wave:** 4
**Depends on:** WS-13.15, WS-07.11
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-08 (default assumed: MFA mandatory for CLINICIAN and ADMIN on acknowledgement endpoints; the one-time link stays usable without login)
**Real infra:** no (WS-05's W4 suite proves it end to end)
**Files:**
- Modify: `services/notification-service/src/api.py` (the three clinician routes depend on `require_mfa_for_roles("clinician", "admin")` together with `require_roles("clinician", "admin")`)
- Modify: `services/notification-service/tests/test_mvp_082_acknowledgement_api.py` (the harness principal is built with `mfa=True`)
- Test: `services/notification-service/tests/test_mvp_344_ack_api_mfa.py`
**Interfaces:**
- Consumes: IF-04/IF-17 `require_mfa_for_roles(*roles)` (WS-07.11; 403 `{"detail": {"code": "MFA_REQUIRED", ...}}`), `AuthenticatedUser.mfa`.
- Produces: IF-09 "MFA from W4" on the clinician routes.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_344_ack_api_mfa.py`
```python
"""MVP-344 (supporting): clinician alert routes need MFA; the one-time link does not."""
from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

import notification_repo_contract as contract
from notification_fakes import FixedClock
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from src import api
from src.infrastructure.crisis_alert_repository import InMemoryCrisisAlertRepository


async def _client(mfa: bool):
    clock, repo = FixedClock(), InMemoryCrisisAlertRepository()
    ep = await contract.open_episode(repo, clock)
    token = await repo.issue_ack_token(ep.alert_id, recipient_hash="a" * 64, now=clock(), ttl_seconds=86_400)
    app = FastAPI(); app.include_router(api.crisis_router); app.state.crisis_alerts, app.state.clock = repo, clock
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=str(uuid4()), token_type=TokenType.ACCESS, roles=["clinician"], permissions=[], mfa=mfa)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"), ep, token


@pytest.mark.parametrize("mfa,status", [(False, 403), (True, 200)])
async def test_mvp_344_clinician_routes_require_mfa(mfa, status) -> None:
    client, ep, _ = await _client(mfa)
    r = await client.post(f"/api/v1/notifications/crisis-alerts/{ep.alert_id}/acknowledge")
    assert r.status_code == status
    assert (await client.get("/api/v1/notifications/crisis-alerts?status=open")).status_code == status
    if status == 403:
        assert r.json()["detail"]["code"] == "MFA_REQUIRED"


async def test_d08_one_time_link_needs_no_login() -> None:
    client, _, token = await _client(False)
    assert (await client.post(f"/api/v1/notifications/crisis-alerts/ack/{token}")).status_code == 200
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_344_ack_api_mfa.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the `mfa=False` case gets 200 on both routes because they check roles only (MVP-344, notification part).
- [ ] **Step 3: Implement** the dependency change.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass (the WS-13.9 tests pass with the `mfa=True` principal).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.17: Compact crisis SMS for CRITICAL pages when SMS is enabled
**Covers:** MVP-152
**Severity:** medium
**Wave:** 5
**Depends on:** WS-13.16
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-12 (default assumed: SMS through a BAA provider as an optional second channel for CRITICAL only)
**Real infra:** no
**Files:**
- Modify: `services/notification-service/src/domain/channels.py` (`SMSChannel._format_crisis_sms(metadata)` (lines 353-377) becomes `[Solace CRISIS <LEVEL>] ref <ref> - check email and acknowledge`, GSM-7 only, at most 160 characters, built from `metadata["level"]` and `metadata["alert_ref"]`; `_deliver` (393-398) uses it when `metadata.get("template_type") == "crisis_alert"`)
- Modify: `services/notification-service/src/domain/crisis_paging.py` (for CRITICAL pages, when an SMS channel is registered, each clinician recipient with `phone_e164` also gets one SMS through `deliver_one(ChannelType.SMS, phone, rendered, timeout_seconds=CRISIS_DELIVERY_TIMEOUT_SECONDS, metadata={"template_type": "crisis_alert", "level": ..., "alert_ref": ...})`, claimed with `channel="sms"`; a clinician counts as reached when email or SMS was delivered)
- Modify: `services/notification-service/tests/notification_fakes.py` (`FakeNotifier(sms_enabled=...)` and its `channel_registry`)
- Test: `services/notification-service/tests/test_mvp_152_crisis_sms.py`
**Interfaces:**
- Consumes: WS-13.6 paging; `NotificationService.channel_registry`; IF-02 `SMS_ENABLED` and `SMS_*`.
- Produces: IF-09 "CRITICAL also sends a compact SMS when SMS_ENABLED"; metric `solace_crisis_alert_delivery_total{channel="sms"}`.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_152_crisis_sms.py`
```python
"""MVP-152: CRITICAL pages send a compact, PHI-free SMS when SMS is enabled; HIGH pages do not."""
from __future__ import annotations

from uuid import uuid4

import pytest

from notification_fakes import FakeNotifier, FakeRoster, FixedClock, clinician, crisis_signal, make_pager
from src.domain.channels import SMSChannel

GSM7 = set("@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà")
C1 = clinician("clin-c1@test.invalid", phone="+15555550100")


def test_mvp_152_formatter_is_compact_and_gsm7() -> None:
    text = SMSChannel._format_crisis_sms({"template_type": "crisis_alert", "level": "CRITICAL", "alert_ref": "3f2a9c1e"})
    assert text == "[Solace CRISIS CRITICAL] ref 3f2a9c1e - check email and acknowledge"
    assert len(text) <= 160 and set(text) <= GSM7


@pytest.mark.parametrize("level,sms", [("CRITICAL", 1), ("HIGH", 0)])
async def test_mvp_152_only_critical_pages_send_sms(level, sms) -> None:
    notifier = FakeNotifier(sms_enabled=True)
    pager, _, _ = make_pager(roster=FakeRoster(on_call=[C1]), notifier=notifier, clock=FixedClock())
    await pager.handle_detection(crisis_signal(uuid4(), level))
    texts = [s for s in notifier.sent if s.channel == "sms"]
    assert len(texts) == sms
    if sms:
        assert texts[0].address == "+15555550100" and texts[0].body.startswith("[Solace CRISIS CRITICAL]")
```
(`FakeNotifier(sms_enabled=True)` exposes a `channel_registry` holding an available SMS channel and records SMS sends as `Sent(channel="sms", address=<phone>, subject=rendered.subject, body=SMSChannel._format_crisis_sms(metadata))` when `metadata["template_type"] == "crisis_alert"`; add the keyword and the property in this task, in `tests/notification_fakes.py`.)
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_152_crisis_sms.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The formatter emits `CRISIS  | 988 Lifeline: call/text 988` from keys nobody supplies (channels.py:360-377), and `is_crisis` is never true because metadata carries no `template_type` (service.py:178-182; MVP-152); the pager sends no SMS.
- [ ] **Step 3: Implement** the edits under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-13.18: Residual channel defects: webhook, push and the stub auth shims
**Covers:** MVP-288, MVP-289, MVP-291
**Severity:** low
**Wave:** 5
**Depends on:** WS-13.17
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-12 (default assumed: push is out of scope for the pilot)
**Real infra:** no
**Files:**
- Modify: `services/notification-service/src/api.py` (delete the stub fallback at lines 19-71 and import `solace_security.middleware` unconditionally; `_parse_channel_type` returns 400 `Unsupported channel type: webhook` for `webhook`)
- Modify: `services/notification-service/src/domain/service.py` (in `send_notification`, a requested recipient/channel pair with no target or no registered channel counts as a failed delivery (lines 169-197), so `["email", "webhook"]` gives `partially_delivered`, never `delivered`)
- Modify: `services/notification-service/src/main.py` (`PUSH_ENABLED=true` raises `StartupDependencyError("push_channel", "push is out of scope for the pilot (D-12)")` in strict; outside strict it logs WARNING `push_channel_not_supported` and registers no push channel)
- Modify: `services/notification-service/tests/test_channels.py` (the `ChannelType` value list test keeps `webhook`; no other change)
- Test: `services/notification-service/tests/test_mvp_288_289_291_channel_cleanup.py`
**Interfaces:**
- Consumes: IF-01 `StartupDependencyError`; IF-02 `PUSH_ENABLED=false`.
- Produces: no new names.

- [ ] **Step 1: Write the failing test** in `services/notification-service/tests/test_mvp_288_289_291_channel_cleanup.py`
```python
"""MVP-288/289/291: unsupported channels never count as delivered; push cannot be half-enabled; no stub auth."""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from solace_infrastructure.runtime import StartupDependencyError
from src.domain.channels import ChannelRegistry, ChannelType, DeliveryResult, EmailChannel, EmailConfig
from src.domain.service import NotificationRecipient, NotificationRequest, NotificationService, NotificationStatus
from src.domain.templates import TemplateRegistry, TemplateType
import src.main as main

API = Path(__file__).resolve().parents[1] / "src" / "api.py"


async def test_mvp_288_mixed_webhook_request_is_not_reported_delivered() -> None:
    registry = ChannelRegistry(); registry.register_channel(EmailChannel(EmailConfig()))
    service = NotificationService(TemplateRegistry(), registry)
    ok = DeliveryResult(channel_type=ChannelType.EMAIL, recipient="user-x@test.invalid", success=True)
    with patch.object(EmailChannel, "send", return_value=ok):
        result = await service.send_notification(NotificationRequest(
            template_type=TemplateType.WELCOME, recipients=[NotificationRecipient(email="user-x@test.invalid")],
            channels=[ChannelType.EMAIL, ChannelType.WEBHOOK],
            variables={"display_name": "X", "getting_started_link": "https://app.test"}))
    assert (result.status, result.successful_deliveries, result.failed_deliveries) == (NotificationStatus.PARTIALLY_DELIVERED, 1, 1)


def test_mvp_288_api_rejects_the_webhook_channel() -> None:
    from fastapi import HTTPException
    from src import api
    with pytest.raises(HTTPException) as exc:
        api._parse_channel_type("webhook")
    assert exc.value.status_code == 400


def test_mvp_289_api_has_no_stub_auth_fallback() -> None:
    tree = ast.parse(API.read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    assert not names & {"_StubRole", "_StubPermission"} and "solace_security not installed" not in API.read_text(encoding="utf-8")


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_291_push_cannot_be_enabled_in_strict(monkeypatch, env) -> None:
    monkeypatch.setenv("PUSH_ENABLED", "true")
    with pytest.raises(StartupDependencyError) as exc:
        main._create_notification_service(main.NotificationServiceSettings.load(), environment=env)
    assert exc.value.component == "push_channel"


def test_mvp_291_push_is_not_registered_outside_strict(monkeypatch) -> None:
    monkeypatch.setenv("PUSH_ENABLED", "true")
    service = main._create_notification_service(main.NotificationServiceSettings.load(), environment="development")
    assert ChannelType.PUSH not in service.get_available_channels()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests\test_mvp_288_289_291_channel_cleanup.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The mixed request reports `delivered` with 0 failures because the webhook leg is silently skipped (service.py:169-223, MVP-288); `_parse_channel_type("webhook")` accepts it (api.py:246-254); the stub classes exist (api.py:19-71, MVP-289); `PUSH_ENABLED=true` registers a push channel that cannot authenticate (main.py:140-147, channels.py:471-476, MVP-291).
- [ ] **Step 3: Implement** the edits under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\notification-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-13.18:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| Id | Owner | Exact change | Files | Needed by |
|---|---|---|---|---|
| R-13-1 | WS-03 | W1 phase A env sync (WS-03.6): `SERVICE_USER_SERVICE_URL=http://user-service:8001` on notification-service (IF-02 peers). Mark retired once their readers are gone (WS-03.15, W5): `NOTIFICATION_USER_SERVICE_URL` (still read as an alias until then), `KAFKA_ENABLED`, `KAFKA_USE_MOCK`, `KAFKA_CONSUMER_GROUP_ID` (unread after WS-13.8), `EMAIL_USE_TLS` (unread after WS-13.7). No new variable names are introduced by this plan. | `deploy/env/env-contract.yaml`, `docker-compose.yml` | WS-13.5, WS-13.8 |
| R-13-2 | WS-03 | W3 phase A: deliver `ENCRYPTION_SEARCH_HASH_SALT` (secret, required in strict) to notification-service only; not the master key. | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml` | WS-13.14 |
| R-13-3 | WS-09 | W3 phase A: `solace_security.phi_codec.crisis_pseudonym(user_id: Union[UUID, str]) -> UUID` = `UUID(hex=hmac_sha256(search_salt, "crisis_pseudonym" + "\x1f" + str(user_id)).hexdigest()[:32])`, equal to WS-12.12's `UUID(hex=get_phi_codec().search_hash(str(user_id), purpose="crisis_pseudonym")[:32])`, computable from `EncryptionSettings`' search salt alone (no master key); strict raises `PHIConfigError` when the salt is missing or shorter than 16 bytes. WS-12.12 may switch to it. | `src/solace_security/phi_codec.py` | WS-13.14 |
| R-13-4 | WS-04 | W4 phase A: alert rule `MonitoringAlertDeliveryFailed` (`sum(increase(solace_monitoring_alert_delivery_total{outcome="failed"}[15m])) > 0`, `severity: page`, runbook section), promtool-tested. Note for WS-04.7: `solace_crisis_alerts_unacknowledged` is refreshed by the re-pager at least every 30 s, and the crisis outcome label values are exactly `delivered`, `failed`, `fallback_delivered`. | `infrastructure/prometheus/rules/solace.yml`, `infrastructure/prometheus/tests/solace_rules_test.yml`, `docs/runbooks/alerting.md` | WS-13.8 (MVP-156 visibility in alerting) |
| R-13-5 | WS-05 | W1 phase A: the real-infra `smtp_sink` uses the same `MP_SMTP_ALLOWED_RECIPIENTS` allow-list as the e2e overlay, so `reject-*@test.invalid` is refused at RCPT. When WS-13.8, WS-13.9 and WS-13.10 land, remove the W1 manifest entries at hops `clinician_page`, `fallback_page`, `ack_link` and `repage` once they pass. | `src/solace_testing/realinfra.py`, `tests/e2e/expected_failures.yaml` | WS-13.8 |
| R-13-6 | WS-07 | W3 (WS-07.8 real-infra subject checker): `crisis_alerts` has CHECK constraints `max_level IN ('HIGH','CRITICAL')` and `status IN ('open','acknowledged','resolved')`; the generic row filler must pass `max_level='HIGH'` explicitly. | `tests/realinfra/service-auth-authz/test_mvp_120_subject_access_checker.py` | WS-07.8 (consumer of slot 015) |
| R-13-7 | WS-12 | W1 ledger Ruling: WS-12.8's contract directory is `tests/contracts/events/safety__notification/` (written by WS-05.7: crisis, escalation and assessment, which meets its `>= 3` check), not `tests/contracts/notification__safety/`; its dependency on WS-13.13 is satisfied by WS-05.7. WS-13.13 verifies the consumer half of the same files. | `services/safety_service/tests/test_mvp_357_event_contracts.py` | WS-12.8 |
| R-13-8 | WS-14 | W1 phase B (IF-08): the internal routes satisfy `tests/contracts/notification__user/{on-call,care-team,care-team-unknown-user}.get.json` (`verify_provider` in the user-service suite, W2 per WS-05.10); a known user without assignments returns 200 with `clinicians: []`; an unknown user returns 404; `display_name` is returned and notification shows it only with `crisis_contact_consent`. | `services/user-service/src/internal_api.py`, user-service tests | W1 e2e `clinician_page` (WS-05.6/05.8) |
| R-13-9 | WS-04 | W1 (WS-04.7): the `docs/runbooks/alerting.md` sections for `CrisisAlertsUnacknowledged` and the crisis-delivery alerts link to `docs/runbooks/crisis-escalation.md` (written by WS-13.10) for the acknowledge, re-page, fallback and DLQ procedures. WS-13 does not edit that file. | `docs/runbooks/alerting.md` | WS-13.10 |

## New interfaces

- Slot 015 additions beyond IF-09's `crisis_alerts` columns: `last_event_at`, `categories`, `pages_requested`, `repage_count`, `resolved_by`, `last_update_sent_at`, `pseudonymized_at`; tables `crisis_alert_ack_tokens` (one row per page email; IF-09's "only the hash stored" is kept, and the per-recipient token lets every page email, re-pages included, carry a working link) and `risk_alert_windows`; `notifications` columns `alert_id`, `page_seq`, `recipient_hash`, `recipient_role`, `subject_user_id`, `kind`, nullable `user_id`/`title`/`body`, and the unique per-recipient index. `crisis_alerts.ack_token_hash` holds the hash of the token that acknowledged the episode (NULL until then). All four tables are non-RLS and join the IF-03 list of tables touched without a request context.
- Semantics refinement of IF-09 recipients: when the whole care-team tier fails, the non-backup on-call roster is paged together with the fallback mailbox (MVP-064 asks for it); recorded for the D-05/D-06 clinical record in `docs/CLINICAL-VALIDATION.md` (WS-12) by ledger Ruling.
- `GET /api/v1/notifications/crisis-alerts?status=` (CLINICIAN/ADMIN) list route; the HTML link responses; `409 ALERT_RESOLVED`, `404 ALERT_NOT_FOUND`.
- Metric `solace_monitoring_alert_delivery_total{outcome}`.
- Template `crisis_update`; subjects `[Solace CRISIS UPDATE <LEVEL>] ref <8>`, re-page suffix ` (re-page <n>)`, and `[Solace RISK <LEVEL>] monitoring summary - <n> assessment(s)`.
- Python: `AlertSettings`, `RosterClient`/`Clinician`/`CareTeam`/`AlertRecipient`/`RosterUnavailable`/`recipient_hash`, `CrisisSignal`, `CrisisPager`, `PageOutcome`, `EpisodeAction`/`decide_episode`, the repositories and dataclasses of WS-13.4, `RiskDigest`/`run_periodic`, `handle_safety_event`/`register_safety_handlers`, `NotificationService.render`/`deliver_one`/`channel_registry`, `build_stores`/`build_alerting`, `build_notification_erasure`.
- Requested from others: `solace_security.phi_codec.crisis_pseudonym` (R-13-3).

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-011 | blocker | WS-13.5, WS-13.6 |
| MVP-063 | high | WS-13.8 |
| MVP-064 | high | WS-13.5, WS-13.6, WS-13.7 |
| MVP-082 | high | WS-13.3, WS-13.9, WS-13.10 |
| MVP-150 | medium | WS-13.3, WS-13.4, WS-13.8, WS-13.12 |
| MVP-151 | medium | WS-13.4, WS-13.6, WS-13.8 |
| MVP-152 | medium | WS-13.17 |
| MVP-153 | medium | WS-13.11 |
| MVP-154 | medium | WS-13.7, WS-13.8 |
| MVP-155 | medium | WS-13.1, WS-13.2 |
| MVP-156 | medium | WS-13.8 |
| MVP-157 | medium | WS-13.12 |
| MVP-287 | low | WS-13.2 |
| MVP-288 | low | WS-13.18 |
| MVP-289 | low | WS-13.18 |
| MVP-290 | low | WS-13.7 |
| MVP-291 | low | WS-13.18 |
| MVP-292 | low | WS-13.7 |
| MVP-299 | low | WS-13.6, WS-13.13 |

Supporting work for other owners: MVP-006 (WS-01, via WS-13.1), MVP-029 (WS-14, WS-13.14), MVP-091 (WS-09, WS-13.15), MVP-344 (WS-08, WS-13.16), MVP-314 (WS-07, WS-13.12), MVP-086 and MVP-181 notification side (WS-12's ids, WS-13.6/13.8/13.13, matching WS-12's coverage notes for WS-13.6 and WS-13.8).

**Incoming requests**
- WS-01 (WS-01.5, WS-01.7): WS-13.1, W0.
- WS-04 (WS-04.5): WS-13.2, W0 (SPEC row, background consumer registered). WS-04 (WS-04.7): WS-13.6, W1 (outcome values).
- WS-05 (WS-05.10, W2): WS-13.5, W1 (notification__user contracts; WS-13 has no W2 wave).
- WS-06 (WS-06.9): WS-13.8 (plane subscription, no Kafka gate or local guard) and WS-13.6 (`RetryableEventError`), W1.
- WS-07 (WS-07.9, W4): WS-13.12, W1. WS-07 (WS-07.8, W3): WS-13.3, W1.
- WS-12.8's dependency on "WS-13.13 contract files": WS-13.13 is the consumer-side verification; the files are WS-05.7's (R-13-7). WS-12.8 now depends on WS-05.7 and writes `services/safety_service/tests/test_mvp_357_event_contracts.py`.
- WS-04 (WS-04.7, `docs/runbooks/alerting.md`): WS-13 only links to it; the back-link is R-13-9.
- WS-08 (R-15, WS-08.18, W4): WS-13.7, W1 (the `password_reset` template contract, pinned by `test_ws08_password_reset_template_keeps_its_contract`).

**Review focus**
1. Ingress false negatives: not in this workstream's code (detection is WS-10/WS-12); the dispatcher never drops a HIGH/CRITICAL signal by type (WS-13.13) and a HIGH assessment pages if the crisis event is missing (WS-13.8).
2. Concurrent and partial failures on the alert path: `test_review_focus_2_crash_after_the_episode_commit_still_pages_on_redelivery`, `test_review_focus_2_crash_between_send_and_record_resends`, `test_mvp_151_a_retry_skips_recipients_already_delivered` (WS-13.6); `check_concurrent_claims_share_one_record`, `check_claim_repage_wins_once`, `check_ack_before_claim_blocks_the_repage` on Postgres (WS-13.4); `test_mvp_151_redelivered_crisis_pages_once` and the rejected-recipient DLQ test on Kafka (WS-13.8); `test_review_focus_2_ack_racing_the_repage_wins`, `test_review_focus_2_accepted_then_bounced_page_is_repaged`, the two-sweeper test (WS-13.10).
3. Foreign-but-valid identifiers and context-less work: `test_review_focus_3_a_token_acknowledges_only_its_own_episode`, unknown alert id 404 (WS-13.9); the consumer writes and reads back as solace_app without an RLS context (WS-13.3, WS-13.4).
4. PHI through unplanned sinks: `test_review_focus_4_alerts_carry_category_codes_only` and the D-12 consent test (WS-13.6); `test_review_focus_4_send_logs_never_carry_the_address` (WS-13.12); body-less records (WS-13.3, WS-13.4, WS-13.8); audit details without addresses (WS-13.15).
5. Partially valid strict configuration: `test_review_focus_5_strict_rejects_partial_alert_config` (WS-13.5) and `test_review_focus_5_strict_refuses_plaintext_or_disabled_email` (WS-13.7); push half-enabled (WS-13.18).

REVIEW_FOCUS: 2=WS-13.4,WS-13.6,WS-13.8,WS-13.10; 3=WS-13.9,WS-13.4; 4=WS-13.6,WS-13.12; 5=WS-13.5,WS-13.7

**Interface consistency**
- IF-09: consumer group `solace-group-notifications` on `solace.safety`; crisis and escalation at HIGH/CRITICAL on the paging path; monitoring coalesced per user per window; episode columns as listed (with the additions above); recipients care team, then on-call, then `NOTIFICATION_FALLBACK_ONCALL_EMAIL`; `RetryableEventError` when no clinician delivery succeeded; `delivery_attempts` as the per-recipient idempotency record; link `{NOTIFICATION_ACK_BASE_URL}/api/v1/notifications/crisis-alerts/ack/{token}` (24 h, hash only, GET confirm, POST acknowledge); `/acknowledge` and `/resolve` for CLINICIAN/ADMIN, MFA in W4; re-page of the backup and the fallback after `NOTIFICATION_REPAGE_AFTER_SECONDS`, at most `NOTIFICATION_MAX_REPAGES`, from a runtime-registered task; subject `[Solace CRISIS <LEVEL>] action required - ref <8>`; Date and Message-ID; unescaped text; CRITICAL SMS when enabled; `/send` and `/email` 202 or 502 `DELIVERY_FAILED`; the three metric names.
- IF-01: SPEC flags match the WS-04.5 table; lifespan first and last statements as the adoption rule requires; no local health routes or exception handlers. IF-05: `subscribe(topic, event_types, handler, *, group_id)`; no local DLQ or idempotency store. IF-04: `service_auth_headers("user-service")`, `require_service_permission(WRITE_NOTIFICATION)`, `require_roles`, `require_mfa_for_roles`. IF-06/IF-07: categories filtered to `CATEGORIES`, `level_rank`, correlation id = turn id. IF-08: the two internal routes and fields. IF-14: `audit(...)` with allow-listed detail keys. IF-16: `register_user_erasure_consumer`. IF-19/IF-20: `pg_migrated`, `kafka_bootstrap`, `smtp_sink`, `load_contracts`, `assert_request_matches`, `load_event_contracts`, `verify_event_consumer`. IF-02: only registered names are read.
- Phase rule: the only phase-A task is WS-13.3, which depends on phase-A tasks only; every phase-B task depends on phase-A tasks of the same or earlier waves, or on earlier WS-13 tasks.

**Open risks**
- WS-14 is not yet planned: the IF-08 shapes, the 404 semantics and the consent nulling are taken from the skeleton and pinned by the WS-13.5 contract files (R-13-8). Until WS-14 lands, every page goes to the fallback mailbox and is retried (by design).
- Paging the on-call roster when the care team fails is an over-alerting interpretation of IF-09 (New interfaces); it needs the clinical record by Ruling. In the shared e2e roster it can add pages to other tests' clinicians; WS-05 tests isolate by care-team assignment and `pages_since`.
- A crash between SMTP acceptance and the attempt record re-sends that page (at-least-once per recipient); a bounce after acceptance is not detected and is mitigated only by re-paging unacknowledged episodes.
- If no clinician exists at all, every crisis is retried, dead-lettered and re-driven (DLQUnresolved fires) while the fallback mailbox holds the only page; D-13 requires two on-call clinicians before go-live.
- The strict e2e overlay must use STARTTLS with a per-run CA (WS-05's plan); WS-13.7 rejects `EMAIL_SMTP_TLS_MODE=none` in strict, and `NOTIFICATION_ACK_BASE_URL` is not forced to https because the test overlay uses http.
- `subscribe()` on a disabled plane (`EVENTS_ENABLED=false` in unit tests) is assumed to be accepted without starting a consumer (WS-06.8 wording); if it raises, the lifespan unit tests must enable a mock plane instead.
- The realinfra rejection test depends on R-13-5. The `event_dead_letter` column names follow slot 010 (WS-06.1); the assertion adapts to them.
- Re-page sequences and escalation pages share `pages_requested`; a redelivered event that resumes an incomplete re-page sends it to the primary tier rather than the backup, which adds recipients and never removes them.

## Review responses

- E: WS-13.10 edits `docs/runbooks/alerting.md` (WS-04's file): fixed. WS-13.10 never meant to edit it; the backticked path inside the description of the new runbook's link was parsed as a Create. The link is now described without the path token, and the back-link from WS-04's runbook is request R-13-9 to WS-04.7.
- E: Wave 1, `docs/runbooks/alerting.md` edited by WS-04.7 and WS-13.10: fixed (same cause; only WS-04.7 touches the file now).
- E: WS-13.8 modifies `test_mvp_287_consumer_lifecycle.py`, not created by an earlier task: fixed (WS-13.2, W0, now lists it and `test_if01_runtime_adoption.py` under Create; they were under Test, which the checker does not count as a creation; WS-13.8 deletes it in W1).
- E: `docs/runbooks/alerting.md` created by WS-04.7 and WS-13.10: fixed (same cause as the first item).
- E: WS-13.16 modifies `test_mvp_082_acknowledgement_api.py`, not created by an earlier task: fixed (WS-13.9, W1, now lists it under Create; WS-13.16 is W4).
- E: request from WS-05 for WS-05.10 (consumers write `notification__user/` contracts): rejected: already planned. WS-13.5 creates `tests/contracts/notification__user/on-call.get.json`, `care-team.get.json` and `care-team-unknown-user.get.json` in W1 (its Files and Produces say so; WS-14 verifies them per R-13-8). The checker cannot match the request because its Files cell (`tests/contracts/<consumer>__<provider>/*.json`, provider suites) has no parseable path, so `any()` over an empty list is always false (check_plans2.py cross-request check).
- W: WS-13.4 edits `services/notification-service`, no owner: fixed (the realinfra conftest description no longer carries the bare directory tokens that were parsed as files).
- W: WS-13.4 lists `services/notification-service` under Create: fixed (same cause).
- W: WS-13.4 lists `services/notification-service/tests` under Create: fixed (same cause).
- Cross-workstream consistency (from WS-12's revision): R-13-7 now names `services/safety_service/tests/test_mvp_357_event_contracts.py`, the file WS-12.8 creates (the name WS-05 requested for WS-05.10); fixed.
- Review focus (task instruction): fixed (REVIEW_FOCUS line added; focus 1 is detection and lives in WS-10/WS-12 code, so it is not listed).
- Re-check of this plan's outgoing requests: fixed. R-13-3 wrote `UUID | str` inside a table cell, so the parser split the row and lost its files and needed-by cells (now `Union[UUID, str]`; WS-09.17 plans it). R-13-6 named a glob (now WS-07.8's file `tests/realinfra/service-auth-authz/test_mvp_120_subject_access_checker.py`).
- Reconciliation 8 (WS-08.18 cited `services/notification-service/src/domain/templates.py`, a WS-13 file): fixed on WS-13's side. The change moved here as WS-08's request R-15 and is planned in WS-13.7 (W1), which already modifies that file for the text/HTML engine split: the new `test_ws08_password_reset_template_keeps_its_contract` pins the `password_reset` name, subject, required variables `display_name`, `reset_link`, `expiry_minutes`, and a verbatim plain-text `reset_link` (RED today: `O&#39;Brien` in the text body). WS-13.7's Produces line names WS-08.18, which now depends on WS-13.7; the incoming-requests list records it.
- crisis-review F1: fixed. Verified: WS-13.6 raised `RetryableEventError` whenever no clinician got the page, even after the fallback mailbox had it; each of a turn's four HIGH events (orchestrator crisis, safety crisis, escalation, and the HIGH assessment that WS-13.13 also pages) re-attempted the pending page inline and raised again, and WS-06.3 retries each in place five times, so one user with an unreachable recipient held the notification group's consume loop for minutes. Delivery retries now belong to the pager: `handle_detection` delivers a new page once and returns, a page that reached no clinician stays pending, duplicates never deliver inline, and `retry_pending_pages()` (task `crisis_page_retry`, every 15 s, wired in WS-13.8) retries it until a clinician gets it or the episode is resolved; only unexpected exceptions reach the plane. So that the alert this replaces is not lost, a pass with no clinician recipient counts `failed`, and `CrisisAlertDeliveryFailed` fires for as long as the page is pending. WS-13.6's and WS-13.8's tests are updated, and new unit and real-infra tests cover the four-event turn, total failure, a hanging relay and one user's unreachable recipient next to another user's page. This refines IF-09's "the handler raises RetryableEventError" for the skeleton by ledger Ruling; the Open risks line about DLQ re-drive for an empty roster is superseded by the sweep.
- crisis-review F8: fixed. `decide_episode` turned any event with a seen correlation id into a `DUPLICATE` that never raised `max_level`, so a CRITICAL reported later for the same turn (safety's LLM layer after the orchestrator's local HIGH, or a provider-raised level) was folded into a HIGH episode. A strictly higher level on a duplicate is now `ESCALATE` (`max_level` raised, one more page); a redelivery at that level is a plain `DUPLICATE`, so it stays idempotent per correlation id and level. New test `test_crisis_review_f8_a_higher_level_reported_later_for_the_same_turn_pages_again`.
- crisis-review F9: fixed. The rule order put "acknowledged -> UPDATE" before "higher level -> ESCALATE", and `UPDATE` never raised `max_level` and sent at most one email per window, so a CRITICAL after an acknowledged HIGH could go unnoticed indefinitely. `ESCALATE` now comes first, for open and acknowledged episodes alike; an escalation of an acknowledged episode pages the primary tier, the acknowledging clinician when known and, for CRITICAL, the non-backup on-call clinicians, without waiting for the update window. D-06's default in WS-13.6 says so. New test `test_crisis_review_f9_a_rise_after_acknowledgement_pages_and_is_never_only_an_update`.
- crisis-review F10: fixed. `CrisisSignal.from_escalation_event` now takes its categories from `solace_clinical.parse_escalation_reason` (WS-12.2), the single definition WS-12.8, WS-05.7 and WS-06.12 also use; a new assertion covers an unknown code.
