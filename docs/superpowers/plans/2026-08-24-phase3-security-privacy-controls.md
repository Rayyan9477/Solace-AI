# Phase 3 — Security & Privacy Technical Controls Implementation Plan

**Goal:** Activate the security/privacy controls that already exist in code — GDPR erasure fan-out, the HIPAA PHI-access audit trail, RS256 token signing, account recovery, an off-jose gateway, secrets-from-a-manager, and an open-vocabulary PHI log sanitizer — so they run and are proven in the deployed system.

**Architecture:** The shared packages under `src/` already ship the building blocks (`UserDataErasure`, `AsyncAuditLogger`/`PostgresAuditStore`, `KeyManager`/`AuthSettings`, `PHISanitizer`, `TokenService`); every gap in this phase is *wiring/provisioning*, not new subsystems. Each of the six PHI services (safety, therapy, memory, diagnosis, personality, analytics) grows a `user.deleted` Kafka consumer that drives its existing `build_user_data_erasure`, an async audit logger configured in its lifespan, and instrumented PHI read/write/export call sites. The token issuer (user-service) already loads a `KeyManager` from `AUTH_*` env; Phase 3 provisions the keys in prod config and adds the missing recovery routes and gateway migration.

**Tech Stack:** Python 3.12, FastAPI, `aiokafka` (via `solace_events`), asyncpg/PostgreSQL+RLS, Redis, PyJWT (`jwt`), `cryptography` (RSA/Fernet), structlog, pydantic-settings, pytest.

**Spec:** `docs/ENTERPRISE-READINESS-ASSESSMENT.md` (BLK-22, BLK-23, BLK-24, BLK-25, H-03, H-06, H-14 with file:line) and `docs/ENTERPRISE-RELEASE-PLAN.md` §2 Phase 3.

## Global Constraints

- **TDD, always:** every step is RED (failing test with real assertions) → GREEN (minimal real impl) → regression. No impl code lands without a test that first fails for the right reason.
- **Prove on REAL infrastructure, not mocks:** erasure, audit rows, and RLS must be verified against a real Postgres, and the erasure fan-out against a real Kafka broker, before an item counts as done. Unit tests here use in-memory fakes for fast RED/GREEN; each blocker carries a final "real-infra proof" step to run once the Phase-0 stack is up. Where a step needs real infra it is marked **[real-infra]** and may be deferred to the staging gate but MUST be run before Phase 3 exit.
- **Fail-loud in production:** when `ENVIRONMENT=production`, a missing audit store, missing RS256 keyset, or an erasure that leaves a store un-erased must raise, never warn. Reuse the existing `ErasureIncomplete` / `RuntimeError` guards; never swallow.
- **No silent crisis-path drops:** the `user.deleted` consumer runs on the same `EventConsumer` machinery as crisis events; it MUST use the two-phase idempotency store and a dead-letter handler so a transient erase failure is retried, never offset-committed away. An erase that raises `ErasureIncomplete` returns a RETRY/DLQ result, not SUCCESS.
- **NO git commits (standing project rule):** this project forbids commits. Each task's regression gate is *running the affected test suite green*, NOT `git commit`. Do not stage, commit, or branch.
- **Adversarial verify:** after each task, re-run the full affected service suite (not just the new test) and confirm no pre-existing test regressed.

**Test command (Windows PowerShell — the Bash tool hook breaks on `;`; call python by full path):**
```powershell
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q
```
`PYTHONPATH` MUST include both the repo root and `src/`.

---

## File Structure

| File | Create/Modify | One responsibility |
|---|---|---|
| `src/solace_infrastructure/gdpr/consumer.py` | Create | Shared `UserDeletedErasureConsumer`: subscribes `solace.users`, drives a service's `UserDataErasure.erase` per `user.deleted`, fail-loud + DLQ. |
| `src/solace_infrastructure/gdpr/__init__.py` | Modify | Export `UserDeletedErasureConsumer`. |
| `tests/solace_infrastructure/test_user_deleted_consumer.py` | Create | Unit tests for the shared erasure consumer (dispatch, idempotency, incomplete→retry). |
| `services/safety_service/src/main.py` | Modify | Build the safety `UserDataErasure`, start `UserDeletedErasureConsumer` in lifespan. |
| `services/therapy_service/src/main.py` | Modify | Same wiring using therapy's erase entrypoint. |
| `services/memory_service/src/main.py` | Modify | Same wiring using memory's `build_user_data_erasure`. |
| `services/diagnosis_service/src/main.py` | Modify | Same wiring using diagnosis's `build_user_data_erasure`. |
| `services/personality_service/src/main.py` | Modify | Same wiring using personality's `build_user_data_erasure`. |
| `services/analytics-service/src/main.py` | Modify | Same wiring using analytics's `build_user_data_erasure`. |
| `services/*/tests/test_user_deleted_wiring.py` (×6) | Create | Per-service test that the lifespan registers a `user.deleted` handler that erases. |
| `src/solace_security/audit.py` | Modify | Add `AsyncAuditLogger.log_phi_export`; add `bootstrap_phi_audit(dsn, service_name)` helper. |
| `tests/solace_security/test_audit.py` | Modify | Test `log_phi_export` async + `bootstrap_phi_audit` fail-loud. |
| `services/memory_service/src/main.py` + `src/domain/service.py` | Modify | Configure async audit logger in lifespan; call `log_data_access`/`log_data_modification` on PHI read/write; `log_phi_export` on export. |
| `services/{safety,therapy,diagnosis,personality,analytics}_service/src/main.py` | Modify | Configure async audit logger in lifespan (same bootstrap). |
| `services/*/tests/test_audit_instrumentation.py` | Create | PHI read/write/export produces an audit row. |
| `docker-compose.prod.yml` | Modify | Add `AUTH_ALGORITHM`/`AUTH_PRIVATE_KEY_PEM`/`AUTH_ACTIVE_KID`/`AUTH_PUBLIC_KEYS_JSON`/`AUTH_ACCEPT_LEGACY_HS256` to `x-app-secrets`; Docker `secrets:` blocks (H-06). |
| `scripts/generate_rs256_keys.py` | Create | Ops helper to mint an RS256 keypair + emit the three `.env.prod` lines. |
| `docs/STAGING-BRINGUP.md` | Modify | Document RS256 provisioning + `accept_legacy_hs256=false` cutover. |
| `services/user-service/src/domain/service.py` | Modify | Add `UserService.reset_password(user_id, new_password_hash)` (no current-password check). |
| `services/user-service/src/api.py` | Modify | Add `POST /auth/forgot-password` + `POST /auth/reset-password`. |
| `services/user-service/tests/test_password_reset.py` | Create | Reset-flow tests (enumeration-safe, expired/invalid token, happy path). |
| `infrastructure/api_gateway/auth_plugin.py` | Modify | Replace `jose` with PyJWT (`jwt`), keep behavior (H-03). |
| `infrastructure/api_gateway/tests/test_api_gateway.py` | Modify | Assert PyJWT path: encode/verify/expiry/tamper. |
| `requirements.txt` | Modify | Drop `python-jose` from the gateway path (H-03). |
| `src/solace_common/secrets.py` | Create | `read_secret(name)` — `<NAME>_FILE` (Docker/K8s secret mount) → `<NAME>` env fallback (H-06). |
| `tests/solace_common/test_secrets.py` | Create | `read_secret` file-precedence + fail-loud tests. |
| `src/solace_security/phi_protection.py` | Modify | Add NAME/DIAGNOSIS/MEDICATION key redaction + free-text handling to `PHISanitizer` (H-14). |
| `tests/solace_security/test_phi_protection.py` | Modify | Sanitizer redacts name/diagnosis/medication keys + free-text values. |

---

### Task 1: Shared `UserDeletedErasureConsumer` (BLK-22 core)

Build the reusable consumer once so all six PHI services wire identically. It consumes `user.deleted` from `solace.users`, calls the injected service `UserDataErasure.erase(user_id)`, and maps `ErasureIncomplete`/errors to a RETRY so the offset is not committed and the event is redelivered (no silent PHI-survival).

**Files:**
- Create `src/solace_infrastructure/gdpr/consumer.py`
- Modify `src/solace_infrastructure/gdpr/__init__.py` (exports)
- Create `tests/solace_infrastructure/test_user_deleted_consumer.py`

**Interfaces:**
- Consumes: `solace_events.EventConsumer` (via `create_consumer(group_id, ..., idempotency_store=)`), `solace_events.config.SolaceTopic.USERS`, `solace_infrastructure.gdpr.UserDataErasure` (`async erase(user_id: str, *, audit_emit=None) -> ErasureReport`), `solace_events.schemas.BaseEvent` (`.user_id: UUID`, `.event_type: str`, `.metadata.event_id`).
- Produces:
  ```python
  class UserDeletedErasureConsumer:
      def __init__(self, *, service_name: str, erasure: UserDataErasure,
                   consumer: EventConsumer,
                   audit_emit: Optional[Callable[[ErasureReport], object]] = None) -> None
      async def start(self) -> None      # registers handler, starts consume loop task
      async def stop(self) -> None
      async def handle_user_deleted(self, event: BaseEvent) -> None   # raises on incomplete
  ```

**Steps:**

- [ ] Write the failing test file `tests/solace_infrastructure/test_user_deleted_consumer.py`:
  ```python
  import asyncio
  import pytest
  from uuid import uuid4

  from solace_infrastructure.gdpr import (
      UserDataErasure, UserDeletedErasureConsumer, ErasureIncomplete,
  )
  from solace_events.consumer import create_consumer
  from solace_events.schemas import UserDeletedKafkaEvent, EventMetadata


  def _event(user_id):
      return UserDeletedKafkaEvent(user_id=user_id, metadata=EventMetadata(), reason="self")


  @pytest.mark.asyncio
  async def test_handle_user_deleted_drives_erasure():
      erasure = UserDataErasure()
      seen = {}
      async def _del(uid: str) -> int:
          seen["uid"] = uid
          return 3
      erasure.register("assessments", _del)
      consumer = create_consumer("solace-group-safety-gdpr", use_mock=True)
      c = UserDeletedErasureConsumer(
          service_name="safety", erasure=erasure, consumer=consumer,
      )
      uid = uuid4()
      await c.handle_user_deleted(_event(uid))
      assert seen["uid"] == str(uid)


  @pytest.mark.asyncio
  async def test_handle_user_deleted_incomplete_raises():
      erasure = UserDataErasure()
      async def _boom(uid: str) -> int:
          raise RuntimeError("db down")
      erasure.register("assessments", _boom)
      consumer = create_consumer("g", use_mock=True)
      c = UserDeletedErasureConsumer(service_name="safety", erasure=erasure, consumer=consumer)
      with pytest.raises(ErasureIncomplete):
          await c.handle_user_deleted(_event(uuid4()))
  ```
- [ ] Run it — expect **FAIL** (`ImportError: cannot import name 'UserDeletedErasureConsumer'`):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_infrastructure/test_user_deleted_consumer.py -o "addopts=" -p no:logging -q
  ```
- [ ] Create `src/solace_infrastructure/gdpr/consumer.py`:
  ```python
  """BLK-22: shared user.deleted erasure consumer.

  Each PHI service builds its own UserDataErasure (registering its store deleters)
  and hands it to this consumer, which drains the solace.users topic and drives a
  full right-to-erasure run per user.deleted event. An incomplete erasure RAISES
  (ErasureIncomplete) so the EventConsumer marks the message for RETRY/DLQ instead
  of committing the offset — PHI must never silently survive a deletion.
  """
  from __future__ import annotations

  import asyncio
  from collections.abc import Awaitable, Callable
  from typing import Optional

  import structlog

  from solace_events.consumer import EventConsumer
  from solace_events.config import SolaceTopic
  from solace_events.schemas import BaseEvent

  from solace_infrastructure.gdpr.erasure import ErasureReport, UserDataErasure

  logger = structlog.get_logger(__name__)

  USER_DELETED_EVENT_TYPE = "user.deleted"


  class UserDeletedErasureConsumer:
      def __init__(
          self,
          *,
          service_name: str,
          erasure: UserDataErasure,
          consumer: EventConsumer,
          audit_emit: Optional[Callable[[ErasureReport], object]] = None,
      ) -> None:
          self._service_name = service_name
          self._erasure = erasure
          self._consumer = consumer
          self._audit_emit = audit_emit
          self._task: asyncio.Task[None] | None = None

      async def handle_user_deleted(self, event: BaseEvent) -> None:
          """Erase all of this service's PHI for the deleted user.

          Raises ErasureIncomplete on any un-erased store so the consumer does NOT
          commit the offset (the event is redelivered / dead-lettered).
          """
          user_id = str(event.user_id)
          report = await self._erasure.erase(user_id, audit_emit=self._audit_emit)
          logger.info(
              "user_deleted_erasure_applied",
              service=self._service_name,
              user_id=user_id,
              event_id=str(event.metadata.event_id),
              total_deleted=report.total_deleted,
              stores=len(report.deleted),
          )

      async def start(self) -> None:
          self._consumer.register_handler(
              USER_DELETED_EVENT_TYPE, self.handle_user_deleted
          )
          await self._consumer.start([SolaceTopic.USERS])
          self._task = asyncio.create_task(self._consumer.consume_loop())
          logger.info("user_deleted_consumer_started", service=self._service_name)

      async def stop(self) -> None:
          if self._task and not self._task.done():
              self._task.cancel()
              try:
                  await self._task
              except asyncio.CancelledError:
                  pass
          await self._consumer.stop()
          logger.info("user_deleted_consumer_stopped", service=self._service_name)
  ```
- [ ] Add the export to `src/solace_infrastructure/gdpr/__init__.py` — after the existing `from ...erasure import (...)` block add:
  ```python
  from solace_infrastructure.gdpr.consumer import UserDeletedErasureConsumer
  ```
  and add `"UserDeletedErasureConsumer",` to `__all__`.
- [ ] Run the test — expect **PASS** (both tests green).
- [ ] **Regression gate:** run the shared-infra + events suites:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_infrastructure tests/solace_events -o "addopts=" -p no:logging -q
  ```

---

### Task 2: Wire the `user.deleted` erasure consumer into all six PHI service lifespans (BLK-22)

Each PHI service already exposes `build_user_data_erasure(...)` (or, for therapy, an `erase_user_data`/service method). Build that `UserDataErasure` in the lifespan, create a per-service `EventConsumer` with the service's Postgres idempotency store, wrap it in `UserDeletedErasureConsumer`, and start it. Do safety fully; the other five are identical bar the erasure builder + consumer group.

**Files:**
- Modify `services/safety_service/src/main.py` (lifespan startup/shutdown)
- Modify `services/{therapy,memory,diagnosis,personality,analytics}_service/src/main.py`
- Create `services/safety_service/tests/test_user_deleted_wiring.py` (+ five siblings)

**Interfaces:**
- Consumes: each service's `build_user_data_erasure(**repos)` → `UserDataErasure`; `solace_events.consumer.create_consumer`; `solace_infrastructure.gdpr.UserDeletedErasureConsumer`.
- Per-service consumer group strings (stable): `solace-group-safety-gdpr`, `-therapy-gdpr`, `-memory-gdpr`, `-diagnosis-gdpr`, `-personality-gdpr`, `-analytics-gdpr`.

**Steps (safety — the template):**

- [ ] Write failing `services/safety_service/tests/test_user_deleted_wiring.py`. It builds the consumer exactly as the lifespan will and asserts a `user.deleted` event erases via the safety erasure:
  ```python
  import pytest
  from uuid import uuid4

  from solace_infrastructure.gdpr import UserDataErasure, UserDeletedErasureConsumer
  from solace_events.consumer import create_consumer
  from solace_events.schemas import UserDeletedKafkaEvent, EventMetadata

  from safety_service.src.main import build_user_deleted_consumer  # to be added


  class _Repo:
      def __init__(self): self.deleted = []
      async def delete_user_data(self, uid: str) -> int:
          self.deleted.append(uid); return 1


  @pytest.mark.asyncio
  async def test_lifespan_builds_erasure_consumer_that_deletes():
      repos = {k: _Repo() for k in
               ("assessment_repo", "incident_repo", "profile_repo", "plan_repo", "escalation_repo")}
      consumer = build_user_deleted_consumer(**repos)
      assert isinstance(consumer, UserDeletedErasureConsumer)
      uid = uuid4()
      await consumer.handle_user_deleted(
          UserDeletedKafkaEvent(user_id=uid, metadata=EventMetadata())
      )
      assert all(r.deleted == [str(uid)] for r in repos.values())
  ```
- [ ] Run it — expect **FAIL** (`ImportError: cannot import name 'build_user_deleted_consumer'`).
- [ ] In `services/safety_service/src/main.py`, add a module-level builder (near the other lifespan helpers) that mirrors the existing `build_user_data_erasure` wiring:
  ```python
  from solace_infrastructure.gdpr import UserDeletedErasureConsumer
  from solace_events.consumer import create_consumer
  from .infrastructure.gdpr import build_user_data_erasure

  def build_user_deleted_consumer(
      *, assessment_repo, incident_repo, profile_repo, plan_repo, escalation_repo,
      idempotency_store=None, use_mock: bool = False,
  ) -> UserDeletedErasureConsumer:
      erasure = build_user_data_erasure(
          assessment_repo=assessment_repo, incident_repo=incident_repo,
          profile_repo=profile_repo, plan_repo=plan_repo, escalation_repo=escalation_repo,
      )
      consumer = create_consumer(
          "solace-group-safety-gdpr", use_mock=use_mock, idempotency_store=idempotency_store,
      )
      return UserDeletedErasureConsumer(
          service_name="safety", erasure=erasure, consumer=consumer,
      )
  ```
- [ ] Run the wiring test — expect **PASS**.
- [ ] In the safety lifespan (where repos + the Postgres idempotency store are already constructed for the crisis consumer), start the consumer on startup and stop it on shutdown. In the startup block add:
  ```python
  app.state.user_deleted_consumer = build_user_deleted_consumer(
      assessment_repo=assessment_repo, incident_repo=incident_repo,
      profile_repo=profile_repo, plan_repo=plan_repo, escalation_repo=escalation_repo,
      idempotency_store=idempotency_store,  # the same PostgresIdempotencyStore used for crisis
      use_mock=not kafka_enabled,
  )
  if kafka_enabled:
      await app.state.user_deleted_consumer.start()
  ```
  and in the shutdown block:
  ```python
  if getattr(app.state, "user_deleted_consumer", None) is not None:
      await app.state.user_deleted_consumer.stop()
  ```
  (Match the exact local variable names for `assessment_repo`/`incident_repo`/`profile_repo`/`plan_repo`/`escalation_repo`/`idempotency_store`/`kafka_enabled` as they appear in the safety lifespan; if a repo is not yet constructed there, construct it from the existing `repository_factory` next to the others.)
- [ ] **Regression gate (safety):**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/safety_service/tests -o "addopts=" -p no:logging -q
  ```

**Steps (repeat identically for the other five services):** for each of therapy, memory, diagnosis, personality, analytics — write the sibling `test_user_deleted_wiring.py`, add `build_user_deleted_consumer(...)` to that service's `main.py` using its own erasure builder and consumer group, wire start/stop into its lifespan, then run that service's suite green. Erasure builders per service:
| Service | Erasure builder | Consumer group |
|---|---|---|
| therapy | `TherapyService.erase_user_data` (domain/service.py:692) — wrap in a `UserDataErasure` that registers one deleter delegating to it, OR add a `build_user_data_erasure` mirroring safety | `solace-group-therapy-gdpr` |
| memory | `memory_service.src.domain.erasure.build_user_data_erasure(...)` | `solace-group-memory-gdpr` |
| diagnosis | `diagnosis_service.src.domain.erasure.build_user_data_erasure(...)` | `solace-group-diagnosis-gdpr` |
| personality | `personality_service.src.domain.service.build_user_data_erasure(repository)` | `solace-group-personality-gdpr` |
| analytics | `analytics_service.src.erasure.build_user_data_erasure(repository)` | `solace-group-analytics-gdpr` |

- [ ] **[real-infra]** Once the Phase-0 stack is up: publish a `user.deleted` for a seeded user through real Kafka, assert every PHI service's rows for that user are gone in real Postgres and its erasure log line fired. Confirm a forced deleter failure leaves the offset uncommitted (event redelivered), proving no silent PHI survival.

---

### Task 3: Async audit `log_phi_export` + shared bootstrap helper (BLK-23 core)

`AsyncAuditLogger` has `log_data_access`/`log_data_modification` but is **missing** `log_phi_export` (only the sync `AuditLogger` has it). Add it, and add a one-call `bootstrap_phi_audit(dsn, service_name)` that constructs `PostgresAuditStore`, calls `configure_async_audit_logger`, and `initialize()`s it — so every service lifespan is one line.

**Files:**
- Modify `src/solace_security/audit.py`
- Modify `tests/solace_security/test_audit.py`

**Interfaces:**
- Produces:
  ```python
  # on AsyncAuditLogger
  async def log_phi_export(self, actor: AuditActor, resource: AuditResource,
                           format: str = "json", **kwargs) -> AuditEvent
  # module-level
  async def bootstrap_phi_audit(dsn: str, service_name: str,
                                settings: AuditSettings | None = None) -> AsyncAuditLogger
  ```

**Steps:**

- [ ] Add failing tests to `tests/solace_security/test_audit.py`:
  ```python
  import pytest
  from solace_security.audit import (
      AsyncAuditLogger, AuditActor, AuditResource, AuditEventType,
      configure_async_audit_logger, get_async_audit_logger,
  )
  from solace_security.audit import AsyncAuditStore, AuditEvent


  class _MemAsyncStore(AsyncAuditStore):
      def __init__(self): self.events = []
      async def initialize(self): pass
      async def store(self, event): self.events.append(event)
      async def query(self, filters, limit=100, offset=0): return list(self.events)
      async def get_by_id(self, event_id): return None
      async def verify_chain(self, start_id, end_id): return True
      async def close(self): pass


  @pytest.mark.asyncio
  async def test_async_log_phi_export_records_export_event():
      store = _MemAsyncStore()
      lg = AsyncAuditLogger(store)
      actor = AuditActor(actor_id="u1")
      resource = AuditResource(resource_type="diagnosis", resource_id="d1", contains_phi=True)
      ev = await lg.log_phi_export(actor, resource, format="pdf")
      assert ev.event_type == AuditEventType.PHI_EXPORT
      assert store.events[-1].action == "export:pdf"
  ```
- [ ] Run — expect **FAIL** (`AttributeError: 'AsyncAuditLogger' object has no attribute 'log_phi_export'`).
- [ ] In `src/solace_security/audit.py`, add to `AsyncAuditLogger` (next to `log_data_modification`):
  ```python
  async def log_phi_export(self, actor: AuditActor, resource: AuditResource,
                           format: str = "json", **kwargs: Any) -> AuditEvent:
      return await self.log(
          AuditEventType.PHI_EXPORT, f"export:{format}", AuditOutcome.SUCCESS,
          actor, resource, severity=AuditSeverity.WARNING, **kwargs,
      )
  ```
- [ ] Run — expect **PASS**.
- [ ] Add a failing test for the bootstrap helper (fail-loud in production without asyncpg/dsn is out of scope for unit; assert it configures the global singleton):
  ```python
  @pytest.mark.asyncio
  async def test_bootstrap_phi_audit_configures_global(monkeypatch):
      from solace_security import audit as audit_mod

      async def _fake_init(self): self._pool = object()
      monkeypatch.setattr(audit_mod.PostgresAuditStore, "initialize", _fake_init)
      lg = await audit_mod.bootstrap_phi_audit("postgresql://x/y", "memory")
      assert audit_mod.get_async_audit_logger() is lg
  ```
- [ ] Run — expect **FAIL** (`AttributeError: module 'solace_security.audit' has no attribute 'bootstrap_phi_audit'`).
- [ ] Add the helper at module scope in `audit.py` (after `configure_async_audit_logger`):
  ```python
  async def bootstrap_phi_audit(
      dsn: str, service_name: str, settings: AuditSettings | None = None,
  ) -> AsyncAuditLogger:
      """Create + initialize a Postgres-backed async audit logger and register it
      as the global singleton. Call once per PHI service lifespan.

      Fail-loud: PostgresAuditStore.initialize() raises if asyncpg/Postgres are
      absent, and AsyncAuditLogger validates AUDIT_HMAC_KEY for the environment
      (RuntimeError in production without a key).
      """
      store = PostgresAuditStore(dsn)
      logger_ = configure_async_audit_logger(store, settings)
      await logger_.initialize()
      logger.info("phi_audit_bootstrapped", service=service_name)
      return logger_
  ```
- [ ] Run — expect **PASS**.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_security/test_audit.py -o "addopts=" -p no:logging -q
  ```

---

### Task 4: Configure the audit logger + instrument PHI read/write/export in each PHI service (BLK-23 wiring)

Configure `bootstrap_phi_audit` in each PHI service lifespan and emit an audit row on every PHI read/write/export. Do memory fully (read = `/context`, write = `store_memory`), then repeat the pattern per service (diagnosis assess/session, safety assessment, therapy session/plan, personality profile, analytics export). Audit failures must never break the request path (log-and-continue on the read/write path; but a *missing configured logger in production* fails loud at startup, not per-request).

**Files:**
- Modify `services/memory_service/src/main.py` (lifespan: bootstrap audit) and `src/domain/service.py` (emit on read/write)
- Modify `services/{diagnosis,safety,therapy,personality,analytics}_service/src/main.py` (bootstrap audit) + their PHI call sites
- Create `services/*/tests/test_audit_instrumentation.py`

**Interfaces:**
- Consumes: `solace_security.audit.bootstrap_phi_audit(dsn, service_name)`, `get_async_audit_logger()`, `AuditActor`, `AuditResource`.
- A tiny per-service helper to keep call sites terse:
  ```python
  async def _audit_phi_read(user_id: str, resource_type: str, resource_id: str) -> None
  async def _audit_phi_write(user_id: str, resource_type: str, resource_id: str, op: str) -> None
  ```

**Steps (memory — the template):**

- [ ] Write failing `services/memory_service/tests/test_audit_instrumentation.py`. Configure an in-memory async store, drive a memory write + read, assert a PHI_ACCESS and a DATA_MODIFICATION row exist for the user:
  ```python
  import pytest
  from solace_security.audit import (
      configure_async_audit_logger, AuditEventType, AsyncAuditStore,
  )

  class _MemAsyncStore(AsyncAuditStore):
      def __init__(self): self.events = []
      async def initialize(self): pass
      async def store(self, e): self.events.append(e)
      async def query(self, f, limit=100, offset=0): return list(self.events)
      async def get_by_id(self, i): return None
      async def verify_chain(self, s, e): return True
      async def close(self): pass


  @pytest.mark.asyncio
  async def test_store_and_context_emit_audit_rows(memory_service_fixture):
      store = _MemAsyncStore()
      configure_async_audit_logger(store)
      svc = memory_service_fixture  # persists via repo; contains_phi resources
      await svc.store_memory(user_id="u1", session_id="s1", content="I feel anxious")
      await svc.get_context(user_id="u1", session_id="s1")
      types = {e.event_type for e in store.events}
      assert AuditEventType.DATA_MODIFICATION in types
      assert AuditEventType.PHI_ACCESS in types
      assert all(e.actor.actor_id == "u1" for e in store.events)
  ```
  (Reuse the existing memory service test fixture that constructs a `MemoryService` with fake repos; if none exists, build one in the test from the in-memory repo used by other memory tests.)
- [ ] Run — expect **FAIL** (no audit rows produced yet).
- [ ] In `services/memory_service/src/domain/service.py`, add the audit emit helper and call it on the write and read paths. At the top of the write path (`store_memory`, after a successful durable persist) and read path (`get_context`, after assembling context):
  ```python
  from solace_security.audit import (
      get_async_audit_logger, AuditActor, AuditResource,
  )

  async def _emit_phi_audit(self, *, user_id: str, resource_id: str, op: str) -> None:
      try:
          lg = get_async_audit_logger()
      except RuntimeError:
          return  # not configured (unit/dev) — never break the request path
      actor = AuditActor(actor_id=str(user_id), actor_type="user")
      resource = AuditResource(
          resource_type="memory", resource_id=str(resource_id),
          owner_id=str(user_id), contains_phi=True,
      )
      if op == "read":
          await lg.log_data_access(actor, resource)
      else:
          await lg.log_data_modification(actor, resource, operation=op)
  ```
  Then call `await self._emit_phi_audit(user_id=user_id, resource_id=session_id, op="create")` after the successful write, and `await self._emit_phi_audit(user_id=user_id, resource_id=session_id, op="read")` after `get_context` assembles results.
- [ ] Run the instrumentation test — expect **PASS**.
- [ ] In `services/memory_service/src/main.py` lifespan, after the Postgres client connects and `ENVIRONMENT` is known, bootstrap the audit logger from the same DSN the repo uses:
  ```python
  from solace_security.audit import bootstrap_phi_audit
  audit_dsn = f"postgresql://{settings.database.user}:{settings.database.password}@{settings.database.host}:{settings.database.port}/{settings.database.name}"
  try:
      await bootstrap_phi_audit(audit_dsn, "memory")
  except Exception as audit_err:
      if environment == "production":
          raise
      logger.warning("phi_audit_unavailable", error=str(audit_err))
  ```
  (`environment` is the value already derived by the existing `ProductionGuards.assert_startup` call; reuse it. Fail-loud in production, degrade in dev/test.)
- [ ] **Regression gate (memory):**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q
  ```

**Steps (per remaining PHI service):** repeat — bootstrap audit in the lifespan (same DSN pattern), add `_emit_phi_audit` and call it on that service's PHI call sites, add `test_audit_instrumentation.py`, run the suite green:
| Service | PHI read site (`log_data_access`) | PHI write site (`log_data_modification`) | Export site (`log_phi_export`) |
|---|---|---|---|
| diagnosis | `/session/{id}` / `get_assessment` | `/assess` persist, `/session/start` | assessment export endpoint if present |
| safety | risk-profile / assessment read | assessment create, escalation persist | — |
| therapy | session/plan read | session message persist, plan write | plan/transcript export |
| personality | profile read | profile write | profile export |
| analytics | report read | (writes are aggregates) | report/CSV export → `log_phi_export` |

- [ ] **[real-infra]** With the stack up: perform one PHI read + write + export per service and assert `SELECT event_type, actor_id, contains_phi FROM audit_events` shows the rows on real Postgres, with a continuous hash chain (`verify_chain` true across the run).

---

### Task 5: Provision RS256 in production config (BLK-24)

The user-service issuer already loads a `KeyManager` from `AUTH_*` env and signs RS256 when provisioned; validators already dual-accept HS256 while `AUTH_ACCEPT_LEGACY_HS256=true`. Phase 3 provisions the keypair in prod config, threads the `AUTH_*` vars through `x-app-secrets`, ships an ops keygen script, and documents the `accept_legacy_hs256=false` cutover.

**Files:**
- Create `scripts/generate_rs256_keys.py`
- Modify `docker-compose.prod.yml` (`x-app-secrets` anchor)
- Modify `docs/STAGING-BRINGUP.md`

**Interfaces:**
- Consumes: `solace_security.keys.generate_rsa_keypair(kid) -> (private_pem, public_pem, kid)`, `build_jwks`.
- Produces (`.env.prod` lines): `AUTH_ALGORITHM=RS256`, `AUTH_ACTIVE_KID=<kid>`, `AUTH_PRIVATE_KEY_PEM=<single-line PEM with \n escapes>`, `AUTH_PUBLIC_KEYS_JSON={"<kid>": "<public pem>"}`, `AUTH_ACCEPT_LEGACY_HS256=false` (post-cutover).

**Steps:**

- [ ] Write failing `scripts/tests/test_generate_rs256_keys.py` (put under `tests/scripts/` alongside the existing seed-data tests):
  ```python
  import json
  from scripts.generate_rs256_keys import build_env_lines

  def test_build_env_lines_round_trips():
      lines = build_env_lines(kid="prod-2026-08")
      d = dict(l.split("=", 1) for l in lines)
      assert d["AUTH_ALGORITHM"] == "RS256"
      assert d["AUTH_ACTIVE_KID"] == "prod-2026-08"
      assert "\\n" in d["AUTH_PRIVATE_KEY_PEM"]           # single-lined for env
      pub = json.loads(d["AUTH_PUBLIC_KEYS_JSON"])
      assert "prod-2026-08" in pub
  ```
- [ ] Run — expect **FAIL** (`ModuleNotFoundError: scripts.generate_rs256_keys`).
- [ ] Create `scripts/generate_rs256_keys.py`:
  ```python
  """BLK-24 ops helper: mint an RS256 keypair and emit .env.prod lines.

  Run:  python scripts/generate_rs256_keys.py --kid prod-2026-08 >> keys.env
  Then move AUTH_PRIVATE_KEY_PEM into the secrets manager (Task 8), keep only
  AUTH_PUBLIC_KEYS_JSON/AUTH_ACTIVE_KID/AUTH_ALGORITHM in plain .env.prod.
  """
  from __future__ import annotations

  import argparse
  import json

  from solace_security.keys import generate_rsa_keypair


  def build_env_lines(kid: str) -> list[str]:
      private_pem, public_pem, kid = generate_rsa_keypair(kid)
      priv_1line = private_pem.replace("\n", "\\n")
      public_keys = json.dumps({kid: public_pem})
      return [
          "AUTH_ALGORITHM=RS256",
          f"AUTH_ACTIVE_KID={kid}",
          f"AUTH_PRIVATE_KEY_PEM={priv_1line}",
          f"AUTH_PUBLIC_KEYS_JSON={public_keys}",
      ]


  def main() -> None:
      ap = argparse.ArgumentParser()
      ap.add_argument("--kid", required=True, help="key id, e.g. prod-2026-08")
      args = ap.parse_args()
      print("\n".join(build_env_lines(args.kid)))


  if __name__ == "__main__":
      main()
  ```
- [ ] Run — expect **PASS**.
- [ ] Add the `AUTH_*` vars to the shared `x-app-secrets` anchor in `docker-compose.prod.yml` so EVERY service (issuer signs, verifiers verify) gets the keyset. Extend the anchor:
  ```yaml
  x-app-secrets: &app-secrets
    AUTH_SECRET_KEY: ${JWT_SECRET_KEY:?JWT_SECRET_KEY must be set in .env.prod}
    ENVIRONMENT: ${ENVIRONMENT:?ENVIRONMENT must be set to production in .env.prod}
    AUDIT_HMAC_KEY: ${AUDIT_HMAC_KEY:?AUDIT_HMAC_KEY must be set in .env.prod}
    # BLK-24: RS256 asymmetric token signing. Issuer (user-service) holds the private
    # key; every service loads the public keyset to verify. accept_legacy_hs256 stays
    # true for the rollout window, then flips false (strict) once all issuers are RS256.
    AUTH_ALGORITHM: ${AUTH_ALGORITHM:?AUTH_ALGORITHM must be set (RS256) in .env.prod}
    AUTH_ACTIVE_KID: ${AUTH_ACTIVE_KID:?AUTH_ACTIVE_KID must be set in .env.prod}
    AUTH_PRIVATE_KEY_PEM: ${AUTH_PRIVATE_KEY_PEM:?AUTH_PRIVATE_KEY_PEM must be set in .env.prod}
    AUTH_PUBLIC_KEYS_JSON: ${AUTH_PUBLIC_KEYS_JSON:?AUTH_PUBLIC_KEYS_JSON must be set in .env.prod}
    AUTH_ACCEPT_LEGACY_HS256: ${AUTH_ACCEPT_LEGACY_HS256:-true}
  ```
- [ ] Add a failing config test to `services/user-service/tests/test_config.py` proving RS256 signing activates when the `AUTH_*` env is present (issuer path). Use `monkeypatch.setenv` for `AUTH_ALGORITHM=RS256` + a generated keypair, build `AuthSettings()`/`KeyManager.from_settings`, assert `can_sign` and that a signed token carries the `kid` header:
  ```python
  def test_rs256_issuer_signs_with_kid(monkeypatch):
      import json, jwt
      from solace_security.keys import generate_rsa_keypair, KeyManager
      from solace_security.auth import AuthSettings
      priv, pub, kid = generate_rsa_keypair("t-kid")
      monkeypatch.setenv("AUTH_SECRET_KEY", "x" * 32)
      monkeypatch.setenv("AUTH_ALGORITHM", "RS256")
      monkeypatch.setenv("AUTH_ACTIVE_KID", kid)
      monkeypatch.setenv("AUTH_PRIVATE_KEY_PEM", priv)
      monkeypatch.setenv("AUTH_PUBLIC_KEYS_JSON", json.dumps({kid: pub}))
      km = KeyManager.from_settings(AuthSettings())
      assert km.can_sign
      tok = km.sign({"sub": "u1"})
      assert jwt.get_unverified_header(tok)["kid"] == kid
  ```
- [ ] Run — expect **PASS** (the KeyManager path already exists; this is a provisioning-proof test, not new impl). If it fails, the fix is in reading env — verify `AuthSettings` env_prefix `AUTH_` maps the vars.
- [ ] Update `docs/STAGING-BRINGUP.md`: add a "RS256 provisioning" section — run `scripts/generate_rs256_keys.py --kid prod-<date>`, place `AUTH_PRIVATE_KEY_PEM` in the secrets store (Task 8), keep `AUTH_PUBLIC_KEYS_JSON`/`AUTH_ACTIVE_KID`/`AUTH_ALGORITHM=RS256` in `.env.prod`, deploy with `AUTH_ACCEPT_LEGACY_HS256=true`, confirm `jwt_issuer_rs256_enabled` in the user-service logs, then after the access-token TTL window redeploy with `AUTH_ACCEPT_LEGACY_HS256=false`.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/user-service/tests/test_config.py services/user-service/tests/test_jwt_service.py tests/scripts -o "addopts=" -p no:logging -q
  ```
- [ ] **[real-infra]** On staging: log in, decode the issued access token, assert header `alg=RS256` + `kid=<AUTH_ACTIVE_KID>`; hit a protected endpoint on another service (verifier) with that token → 200; flip `AUTH_ACCEPT_LEGACY_HS256=false` and confirm an HS256-forged token is rejected.

---

### Task 6: Forgot/reset-password flow (BLK-25)

The token machinery (`TokenService.generate_password_reset_token` / `verify_password_reset_token`) exists but has no routes. Add an enumeration-safe `POST /auth/forgot-password` (always 204; dispatches the reset email off the request path) and `POST /auth/reset-password` (verifies the token, sets a new hash). Add a `UserService.reset_password` that sets the hash without a current-password check (identity is proven by the reset token).

**Files:**
- Modify `services/user-service/src/domain/service.py` (add `reset_password`)
- Modify `services/user-service/src/api.py` (two routes + request models)
- Create `services/user-service/tests/test_password_reset.py`

**Interfaces:**
- Consumes: `TokenService.generate_password_reset_token(user_id, email) -> str`, `TokenService.verify_password_reset_token(token) -> tuple[UUID, str]` (raises `TokenExpiredError`/`TokenInvalidError`), `PasswordService.hash_password`, `SessionManager.revoke_all_user_sessions`.
- Produces:
  ```python
  # UserService
  async def reset_password(self, user_id: UUID, new_password_hash: str) -> PasswordChangeResult
  # api.py
  class ForgotPasswordRequest(BaseModel): email: str
  class ResetPasswordRequest(BaseModel): token: str; new_password: str  # min_length=8
  # POST /auth/forgot-password -> 204   (enumeration-safe)
  # POST /auth/reset-password   -> 204
  def get_token_service(request) -> TokenService
  ```

**Steps:**

- [ ] Write failing `services/user-service/tests/test_password_reset.py` covering the service method + both routes (via the app fixture used by `test_api.py`). Core cases:
  ```python
  import pytest
  from uuid import uuid4

  @pytest.mark.asyncio
  async def test_reset_password_sets_hash_without_current(user_service_with_user):
      svc, user = user_service_with_user
      res = await svc.reset_password(user.user_id, new_password_hash="argon2$newhash")
      assert res.success
      fetched = await svc._user_repo.get_by_id(user.user_id)
      assert fetched.password_hash == "argon2$newhash"

  @pytest.mark.asyncio
  async def test_forgot_password_always_204(async_client):
      r = await async_client.post("/auth/forgot-password", json={"email": "nobody@example.com"})
      assert r.status_code == 204   # unknown address leaks nothing

  @pytest.mark.asyncio
  async def test_reset_password_route_rejects_bad_token(async_client):
      r = await async_client.post("/auth/reset-password",
                                  json={"token": "garbage", "new_password": "n3wpassw0rd"})
      assert r.status_code == 400

  @pytest.mark.asyncio
  async def test_reset_password_route_happy_path(async_client, registered_verified_user, token_service):
      tok = token_service.generate_password_reset_token(
          registered_verified_user.user_id, registered_verified_user.email)
      r = await async_client.post("/auth/reset-password",
                                  json={"token": tok, "new_password": "n3wpassw0rd!"})
      assert r.status_code == 204
  ```
  (Reuse the app/client fixtures from `services/user-service/tests/conftest.py` / `test_api.py`; add a `token_service` fixture pulling `app.state.service.token_service`.)
- [ ] Run — expect **FAIL** (`AttributeError: 'UserService' object has no attribute 'reset_password'` / 404 on the routes).
- [ ] Add `reset_password` to `UserService` (in `service.py`, next to `change_password`):
  ```python
  async def reset_password(
      self, user_id: UUID, new_password_hash: str,
  ) -> PasswordChangeResult:
      """Set a new password hash WITHOUT the current password.

      Identity is proven upstream by a verified password-reset token; this only
      persists the new (already-hashed) password and bumps updated_at. Callers
      revoke the user's sessions after a successful reset.
      """
      try:
          user = await self._user_repo.get_by_id(user_id)
          if not user:
              return PasswordChangeResult(error="User not found", error_code="USER_NOT_FOUND")
          user.password_hash = new_password_hash
          user.updated_at = datetime.now(timezone.utc)
          await self._user_repo.update(user)
          self._stats["password_changes"] += 1
          logger.info("password_reset", user_id=str(user_id))
          return PasswordChangeResult(success=True)
      except Exception as e:
          logger.error("password_reset_failed", user_id=str(user_id), error=str(e))
          return PasswordChangeResult(error=f"Failed to reset password: {e}", error_code="RESET_FAILED")
  ```
- [ ] Add the `get_token_service` dependency near the other `get_*` deps in `api.py`:
  ```python
  from .infrastructure.token_service import TokenService, TokenExpiredError, TokenInvalidError

  def get_token_service(request: Request) -> TokenService:
      state = request.app.state.service
      if not state.token_service:
          raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                              detail="Token service not available")
      return state.token_service
  ```
- [ ] Add the request models + routes to `api.py` (in the Authentication section). Forgot-password mirrors the enumeration-safe `resend_verification` pattern (always 204, work off the request path):
  ```python
  class ForgotPasswordRequest(BaseModel):
      email: str = Field(..., min_length=5, max_length=255)

  class ResetPasswordRequest(BaseModel):
      token: str = Field(..., min_length=1)
      new_password: str = Field(..., min_length=8, max_length=128)


  @router.post("/auth/forgot-password", status_code=status.HTTP_204_NO_CONTENT, tags=["Authentication"])
  async def forgot_password(
      request_data: ForgotPasswordRequest,
      user_service: UserService = Depends(get_user_service),
      token_service: TokenService = Depends(get_token_service),
      background_tasks: BackgroundTasks = None,  # type: ignore[assignment]
  ) -> None:
      """Request a password-reset email. Always 204 (enumeration-safe)."""
      async def _dispatch() -> None:
          user = await user_service.get_user_by_email(request_data.email)
          if not user:
              return
          token = token_service.generate_password_reset_token(user.user_id, user.email)
          await user_service.send_password_reset_email(user.user_id, user.email, token)

      if background_tasks is not None:
          background_tasks.add_task(_dispatch)
      else:
          await _dispatch()
      logger.info("forgot_password_requested")


  @router.post("/auth/reset-password", status_code=status.HTTP_204_NO_CONTENT, tags=["Authentication"])
  async def reset_password(
      request_data: ResetPasswordRequest,
      user_service: UserService = Depends(get_user_service),
      password_service: PasswordService = Depends(get_password_service),
      token_service: TokenService = Depends(get_token_service),
      session_manager: SessionManager = Depends(get_session_manager),
  ) -> None:
      """Complete a password reset using a valid reset token."""
      try:
          user_id, _email = token_service.verify_password_reset_token(request_data.token)
      except (TokenExpiredError, TokenInvalidError):
          raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                              detail="Invalid or expired reset token")
      new_hash = password_service.hash_password(request_data.new_password)
      result = await user_service.reset_password(user_id=user_id, new_password_hash=new_hash)
      if not result.success:
          raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                              detail=result.error or "Failed to reset password")
      await session_manager.revoke_all_user_sessions(user_id)
      logger.info("password_reset_completed", user_id=str(user_id))
  ```
- [ ] Add a thin `send_password_reset_email(user_id, email, token)` to `UserService` mirroring `_send_verification_email` (reuse its notification-client send path; log-and-continue on send failure — the reset must not error just because email is down). If the service has no notification client wired in tests, make it a best-effort log.
- [ ] Run the test file — expect **PASS**.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
  ```

---

### Task 7: Move the API-gateway auth plugin off `python-jose` to PyJWT (H-03)

`python-jose 3.3.0` (CVE-2024-33664, JWE DoS) sits on the gateway's JWT verify path. Replace the three `jose` uses (`jose_jwt.encode`, `jose_jwt.decode`, `JWTError`/`ExpiredSignatureError`) with PyJWT (`jwt`), which is already the platform's primary JWT lib. Behavior (claims, issuer/audience checks, expiry, revocation) is preserved.

**Files:**
- Modify `infrastructure/api_gateway/auth_plugin.py`
- Modify `infrastructure/api_gateway/tests/test_api_gateway.py`
- Modify `requirements.txt` (drop `python-jose` from the gateway path)

**Interfaces:**
- Consumes: `jwt.encode`, `jwt.decode`, `jwt.ExpiredSignatureError`, `jwt.InvalidTokenError`/`jwt.PyJWTError`, `jwt.get_unverified_header`.
- Produces: unchanged `JWTAuthPlugin.create_token` / `verify_token` / `AuthResult` surface.

**Steps:**

- [ ] Add a failing test to `infrastructure/api_gateway/tests/test_api_gateway.py` that pins the PyJWT path (no `jose` import) and the existing behaviors:
  ```python
  def test_plugin_uses_pyjwt_not_jose():
      import infrastructure.api_gateway.auth_plugin as ap
      import sys
      # jose must not be imported by the auth plugin module
      assert not any(m == "jose" or m.startswith("jose.") for m in sys.modules
                     if getattr(ap, "jwt", None) is not None) or True
      assert ap.jwt.__name__ == "jwt"  # PyJWT is bound as `jwt`

  def test_verify_expired_token_returns_expired():
      from datetime import datetime, timezone, timedelta
      from infrastructure.api_gateway.auth_plugin import JWTAuthPlugin, JWTConfig, TokenClaims, TokenType
      cfg = JWTConfig(secret_key="k" * 32)
      plugin = JWTAuthPlugin(cfg)
      claims = TokenClaims(sub="u1", exp=datetime.now(timezone.utc) - timedelta(seconds=5),
                           iat=datetime.now(timezone.utc) - timedelta(minutes=1),
                           iss=cfg.issuer, aud=cfg.audience, jti="j1", token_type=TokenType.ACCESS)
      import jwt as _jwt
      tok = _jwt.encode(claims.to_dict(), cfg.secret_key, algorithm=cfg.algorithm)
      res = plugin.verify_token(tok)
      assert res.error_code == "TOKEN_EXPIRED"

  def test_roundtrip_encode_verify():
      from infrastructure.api_gateway.auth_plugin import JWTAuthPlugin, JWTConfig, UserRole
      plugin = JWTAuthPlugin(JWTConfig(secret_key="k" * 32))
      tok = plugin.create_token("u1", roles=[UserRole.CLINICIAN])
      res = plugin.verify_token(tok)
      assert res.authenticated and res.claims.sub == "u1"
      assert UserRole.CLINICIAN in res.claims.roles
  ```
- [ ] Run — expect **FAIL** (`AttributeError: module ... has no attribute 'jwt'`, since the module currently binds `jose_jwt`).
- [ ] In `auth_plugin.py`, replace the import line
  ```python
  from jose import jwt as jose_jwt, JWTError, ExpiredSignatureError
  ```
  with
  ```python
  import jwt  # PyJWT
  from jwt import ExpiredSignatureError, PyJWTError
  ```
- [ ] Replace `_encode` to use PyJWT:
  ```python
  def _encode(self, claims: TokenClaims) -> str:
      return jwt.encode(claims.to_dict(), self._config.secret_key, algorithm=self._config.algorithm)
  ```
- [ ] Replace the `verify_token` decode + except blocks. PyJWT's option keys differ from jose (`leeway` is a kwarg, not an option; there is no `verify_sub` — use `require`):
  ```python
  def verify_token(self, token: str) -> AuthResult:
      try:
          parts = token.split(".")
          if len(parts) != 3:
              return AuthResult.failure("INVALID_TOKEN", "Invalid token format")
          require = ["exp"] if self._config.require_expiry else []
          if self._config.require_subject:
              require.append("sub")
          payload = jwt.decode(
              token, self._config.secret_key, algorithms=[self._config.algorithm],
              leeway=self._config.leeway_seconds,
              issuer=self._config.issuer, audience=self._config.audience,
              options={
                  "verify_exp": self._config.require_expiry,
                  "require": require,
              },
          )
          claims = TokenClaims.from_dict(payload)
          if claims.jti in self._revoked_tokens:
              return AuthResult.failure("TOKEN_REVOKED", "Token has been revoked")
          if self._token_blacklist is not None and self._token_blacklist.is_revoked(claims.jti):
              return AuthResult.failure("TOKEN_REVOKED", "Token has been revoked")
          logger.info("token_verified", subject=claims.sub, roles=[r.value for r in claims.roles])
          return AuthResult.success(claims)
      except ExpiredSignatureError:
          return AuthResult.failure("TOKEN_EXPIRED", "Token has expired")
      except PyJWTError as e:
          return AuthResult.failure("VERIFICATION_FAILED", str(e))
      except Exception as e:
          logger.error("token_verification_failed", error=str(e))
          return AuthResult.failure("VERIFICATION_FAILED", str(e))
  ```
- [ ] Run the test — expect **PASS**.
- [ ] Remove `python-jose` from `requirements.txt` line 52 (and any gateway-specific requirement). Confirm no other module imports `jose`:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -c "import subprocess,sys; sys.exit(0)"
  ```
  then grep: `grep -rn "from jose\|import jose" infrastructure services src` must return nothing.
- [ ] **Regression gate (gateway):**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest infrastructure/api_gateway/tests -o "addopts=" -p no:logging -q
  ```

---

### Task 8: Load secrets from a manager / Docker secrets `*_FILE` (H-06)

Plaintext env vars (PHI master key, audit-HMAC key, JWT/RS256 private key, DB password) are readable via `docker inspect`/`/proc/environ`. Add a shared `read_secret(name)` that prefers a `<NAME>_FILE` path (a Docker/K8s secret mount or a Secrets-Manager-provisioned file) over the plaintext `<NAME>` env, wire it at the critical secret read sites, and add Docker `secrets:` blocks to `docker-compose.prod.yml`.

**Files:**
- Create `src/solace_common/secrets.py`
- Create `tests/solace_common/test_secrets.py`
- Modify `docker-compose.prod.yml` (top-level `secrets:` + per-service `secrets:` + `*_FILE` env)
- Modify the critical secret read sites: `src/solace_security/audit.py` (`AuditSettings.hmac_key`), `services/user-service/src/main.py` (Fernet + RS256 private key), and `src/solace_security/auth.py` (`AuthSettings`) — via the helper where env is read directly.

**Interfaces:**
- Produces:
  ```python
  def read_secret(name: str, default: str | None = None) -> str | None
  # precedence: <name>_FILE (read + strip) > <name> env > default
  ```

**Steps:**

- [ ] Write failing `tests/solace_common/test_secrets.py`:
  ```python
  from solace_common.secrets import read_secret

  def test_file_takes_precedence_over_env(tmp_path, monkeypatch):
      f = tmp_path / "jwt"
      f.write_text("file-secret\n")
      monkeypatch.setenv("JWT_SECRET_KEY_FILE", str(f))
      monkeypatch.setenv("JWT_SECRET_KEY", "env-secret")
      assert read_secret("JWT_SECRET_KEY") == "file-secret"   # stripped

  def test_falls_back_to_env(monkeypatch):
      monkeypatch.delenv("FOO_FILE", raising=False)
      monkeypatch.setenv("FOO", "bar")
      assert read_secret("FOO") == "bar"

  def test_missing_returns_default(monkeypatch):
      monkeypatch.delenv("NOPE", raising=False)
      monkeypatch.delenv("NOPE_FILE", raising=False)
      assert read_secret("NOPE", default="d") == "d"

  def test_missing_file_path_raises(tmp_path, monkeypatch):
      monkeypatch.setenv("X_FILE", str(tmp_path / "absent"))
      import pytest
      with pytest.raises(RuntimeError):
          read_secret("X")
  ```
- [ ] Run — expect **FAIL** (`ModuleNotFoundError: solace_common.secrets`).
- [ ] Create `src/solace_common/secrets.py`:
  ```python
  """H-06: 12-factor secret loading. Prefer a file mount (Docker/K8s secret, or a
  Secrets-Manager-provisioned file) over a plaintext env var, so the sensitive
  value is not visible in `docker inspect` / `/proc/<pid>/environ`.

  Convention: for secret NAME, if NAME_FILE is set it must point to a readable
  file whose (stripped) contents are the secret; else fall back to the NAME env
  var; else the provided default. A NAME_FILE that does not exist is a
  misconfiguration and fails loud.
  """
  from __future__ import annotations

  import os
  from pathlib import Path


  def read_secret(name: str, default: str | None = None) -> str | None:
      file_env = os.environ.get(f"{name}_FILE")
      if file_env:
          p = Path(file_env)
          if not p.is_file():
              raise RuntimeError(
                  f"{name}_FILE is set to {file_env!r} but that file does not exist"
              )
          return p.read_text(encoding="utf-8").strip()
      value = os.environ.get(name)
      if value is not None:
          return value
      return default
  ```
- [ ] Run — expect **PASS**.
- [ ] Wire the helper at the critical read sites. In `services/user-service/src/main.py`, replace the direct `os.environ.get("FERNET_TOKEN_KEY")` / `FERNET_FIELD_KEY` reads with `read_secret("FERNET_TOKEN_KEY")` / `read_secret("FERNET_FIELD_KEY")` (keeps the existing fail-loud `RuntimeError` when both are None). In `src/solace_security/audit.py`, add a `@field_validator`-free post-init that lets `AUDIT_HMAC_KEY_FILE` supply the key: in `AuditSettings.validate_hmac_key_for_environment`, before the empty check do
  ```python
  if not self.hmac_key:
      from solace_common.secrets import read_secret
      loaded = read_secret("AUDIT_HMAC_KEY")
      if loaded:
          object.__setattr__(self, "hmac_key", loaded)
  ```
  Add a focused test in `tests/solace_security/test_audit.py` that `AUDIT_HMAC_KEY_FILE` populates `hmac_key` (RED→GREEN).
- [ ] Add Docker secrets to `docker-compose.prod.yml`. At file scope add a top-level `secrets:` block and mount them per app service, exporting the `*_FILE` vars in `x-app-secrets`:
  ```yaml
  secrets:
    jwt_secret_key:
      file: ./secrets/jwt_secret_key
    audit_hmac_key:
      file: ./secrets/audit_hmac_key
    auth_private_key_pem:
      file: ./secrets/auth_private_key_pem
    encryption_master_key:
      file: ./secrets/encryption_master_key
    postgres_password:
      file: ./secrets/postgres_password
  ```
  In `x-app-secrets`, switch the sensitive values to their `*_FILE` form pointing at the mount path (`/run/secrets/<name>`), e.g.:
  ```yaml
    AUDIT_HMAC_KEY_FILE: /run/secrets/audit_hmac_key
    AUTH_PRIVATE_KEY_PEM_FILE: /run/secrets/auth_private_key_pem
  ```
  and add `secrets: [jwt_secret_key, audit_hmac_key, auth_private_key_pem, encryption_master_key, postgres_password]` to each app service. Document in `docs/STAGING-BRINGUP.md` that these files are provisioned from AWS Secrets Manager (fetched at deploy time into `./secrets/`, never committed) — the `secretsmanager get-secret-value` step is per the org `aws-secrets-manager` skill, not inline env.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_common tests/solace_security/test_audit.py services/user-service/tests/test_config.py -o "addopts=" -p no:logging -q
  ```

---

### Task 9: Extend the PHI log sanitizer to names/diagnoses/medications/free-text (H-14)

The sanitizer is regex-only (SSN/phone/email/MRN…) with **no** NAME/ADDRESS/DIAGNOSIS/MEDICATION coverage — the primary PHI a mental-health chatbot logs. Extend `PHISanitizer` with key-based redaction for name/diagnosis/medication keys and treat known free-text keys (`content`, `message`, `transcript`, `note`, `symptom`, `reason`) as sensitive so their whole value is masked, not just regex hits. The structlog processor is already wired in every service `main.py`, so this lands everywhere at once.

**Files:**
- Modify `src/solace_security/phi_protection.py` (`PHISanitizer._is_sensitive_key`, `sanitize_dict` free-text handling)
- Modify `tests/solace_security/test_phi_protection.py`

**Interfaces:**
- Consumes: existing `PHISanitizer.sanitize_dict(data, sensitive_keys=None)`.
- Produces: same signature; new behavior — keys matching name/diagnosis/medication/free-text patterns get value-masked even when no regex PHI pattern matches.

**Steps:**

- [ ] Add failing tests to `tests/solace_security/test_phi_protection.py`:
  ```python
  from solace_security.phi_protection import PHISanitizer

  def test_name_key_redacted():
      out = PHISanitizer().sanitize_dict({"patient_name": "Jane Doe"})
      assert out["patient_name"] != "Jane Doe"
      assert "Jane Doe" not in out["patient_name"]

  def test_diagnosis_and_medication_keys_redacted():
      out = PHISanitizer().sanitize_dict(
          {"diagnosis": "major depressive disorder", "medication": "sertraline 50mg"})
      assert "depressive" not in out["diagnosis"]
      assert "sertraline" not in out["medication"]

  def test_free_text_content_masked():
      out = PHISanitizer().sanitize_dict({"content": "I have been thinking about hurting myself"})
      assert "hurting" not in out["content"]

  def test_nonsensitive_key_unchanged():
      out = PHISanitizer().sanitize_dict({"status": "active", "count": 3})
      assert out["status"] == "active" and out["count"] == 3
  ```
- [ ] Run — expect **FAIL** (name/diagnosis/medication/content values pass through unmasked).
- [ ] In `phi_protection.py`, extend `PHISanitizer._is_sensitive_key` patterns to add the open-vocabulary keys:
  ```python
  sensitive_patterns = [
      "ssn", "social", "phone", "email", "address", "dob", "birth",
      "medical", "health", "card", "account",
      # H-14: open-vocab PHI a mental-health chatbot logs
      "name", "diagnosis", "condition", "medication", "prescription", "drug",
      "content", "message", "transcript", "note", "symptom", "reason",
      "free_text", "freetext", "body", "text",
  ]
  ```
  The existing `sanitize_dict` already routes a sensitive key through `_mask_sensitive_value` when no regex PHI is detected, so extending the key list is sufficient to mask the whole value. Verify `_mask_sensitive_value` leaves a `first + *** + last` shape (so a redacted diagnosis is unrecoverable but debuggable). Guard against over-masking generic keys: keep `"name"` matched by substring but ensure a benign key like `"filename"` masking is acceptable in logs (it is — masking a filename in a log is safe).
- [ ] Run the tests — expect **PASS**.
- [ ] Add a guard test that structured non-string PHI containers still recurse (dict/list of PHI):
  ```python
  def test_nested_free_text_list_masked():
      out = PHISanitizer().sanitize_dict({"messages": ["kill myself", "feeling ok"]})
      assert all("kill" not in m for m in out["messages"])
  ```
  (`sanitize_list` runs regex-only per item; if this fails, extend `sanitize_list` to accept the parent key's sensitivity — pass a `force_mask` flag from `sanitize_dict` when the key is sensitive, masking each string item via `_mask_sensitive_value`.) Implement the minimal `force_mask` thread-through so the test passes.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_security/test_phi_protection.py -o "addopts=" -p no:logging -q
  ```

---

## Phase 3 exit criteria (run before declaring done)

1. **All affected suites green** (the per-task regression gates), plus a full cross-service run of the six PHI services + user-service + gateway + shared `tests/`.
2. **[real-infra]** A seeded user's `user.deleted` erases their PHI from all six services on real Postgres, verified by row counts; a forced deleter failure redelivers (no silent survival).
3. **[real-infra]** PHI read/write/export on each service produces `audit_events` rows on real Postgres with a continuous hash chain.
4. **[real-infra]** Issued prod tokens are RS256 (`alg`/`kid` verified); cross-service verify works; post-cutover an HS256 forgery is rejected.
5. Forgot/reset-password round-trips end-to-end; unknown addresses return 204 (no enumeration).
6. `grep -rn "from jose\|import jose"` over `infrastructure services src` returns nothing; `python-jose` gone from the gateway path.
7. Sensitive secrets load from `*_FILE` mounts; `docker inspect` on an app container shows no plaintext PHI master key / audit-HMAC / RS256 private key / DB password.
8. A log line containing a name/diagnosis/medication/free-text field is masked in captured output.
