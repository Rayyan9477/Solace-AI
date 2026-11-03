# Phase 1 — Crisis Pipeline Integrity Implementation Plan

**Goal:** Make a detected crisis reliably reach a clinician end-to-end over real Kafka with zero silent drops — enable the event plane, emit therapy-originated crises, make safety publish durable-and-loud, drain every outbox, stop transient failures from permanently de-duping, reap the DLQ, and expose DLQ/lag metrics.

**Architecture:** Each service raises local domain events; a per-service `KafkaEventBridge` converts them (`to_kafka_event`) and writes them to a durable Postgres **transactional outbox**; an `OutboxPoller` background task drains the outbox to Kafka topics; the notification service's `SafetyEventConsumer` consumes `solace.safety`, dedupes via a two-phase idempotency ledger, and delivers alerts, routing permanent failures to a durable `PostgresDLQStore` that a DLQ reaper re-drives with exponential backoff. This phase wires the pieces that exist but are inert (pollers on 4 bridges, therapy crisis emission, durable safety dispatch, two-phase consumer idempotency, DLQ reaper, metrics) and flips the enable flags.

**Tech Stack:** Python 3.12, FastAPI ×10, LangGraph, aiokafka (transactional outbox), PostgreSQL (asyncpg) + RLS, Redis, structlog, Pydantic v2 / pydantic-settings, Docker Compose. Shared event lib: `src/solace_events/` (publisher, consumer, dead_letter, postgres_stores, config, schemas).

**Spec:** `docs/ENTERPRISE-READINESS-ASSESSMENT.md` §3B (BLK-07…BLK-14) and §4 (H-15); program context in `docs/ENTERPRISE-RELEASE-PLAN.md` Phase 1.

---

## Global Constraints

- **TDD, always:** every task writes a FAILING test first (RED), then the minimal implementation (GREEN), then re-runs the affected suite as the regression gate.
- **Prove on real infrastructure where it matters:** the in-memory stores (`InMemoryOutboxStore`, `InMemoryIdempotencyStore`, `DeadLetterStore`) mirror the Postgres claim/reserve contracts, so unit tests exercise the real state machine; the Postgres SQL branches are asserted at the string/DDL level here and proven against a live Postgres in the Phase-1 E2E gate (staging). Never assert a crisis guarantee against a `MagicMock` that cannot fail.
- **Fail-loud in production:** any Kafka/outbox/DLQ/bridge wiring that is required for crisis delivery MUST `raise` (not `logger.warning`) when `ENVIRONMENT=production` (or the service's resolved `environment == "production"`).
- **No silent crisis drops:** every change on the safety/therapy → outbox → Kafka → consumer → DLQ path preserves at-least-once delivery and a durable, alertable trail. Each crisis-path change ships a **fault-injection test** proving a transient failure does not lose the event.
- **NO git commits — this project forbids commits.** Each task ENDS by running the affected test suite (the exact PowerShell command below) as the regression gate. Do **not** run `git commit`, `git add`, or open a PR.
- **Test command (Windows PowerShell; the Bash tool hook breaks on `;`):**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q
  ```
  `PYTHONPATH` MUST include both the repo root and `src/`. Set `$env:ENVIRONMENT` per test (`"test"`, or `"production"` where a fail-loud path is under test).

---

## File Structure

| File | Create/Modify | One responsibility |
|---|---|---|
| `docker-compose.yml` | Modify | Turn the event plane ON: set `*_KAFKA_ENABLED=true` + bootstrap servers for safety/therapy/notification. |
| `tests/deployment/test_compose_kafka_enabled.py` | Create | Assert compose enables Kafka on the three crisis-path services. |
| `services/therapy_service/src/events.py` | Modify | Add `CrisisDetectedEvent` domain event + `to_kafka_event` crisis mapping branch. |
| `services/therapy_service/src/domain/service.py` | Modify | `_handle_crisis` publishes the crisis domain event to the EventBus. |
| `services/therapy_service/tests/test_crisis_event_emission.py` | Create | Prove a therapy crisis emits a mappable crisis event (fault-injection: no silent drop). |
| `services/safety_service/src/events.py` | Modify | Durable-at-publish (synchronous dispatch for critical events) + non-swallowing `_dispatch_event`. |
| `services/safety_service/src/infrastructure/event_bridge.py` | Modify | Bridge re-raises on outbox-write failure for critical events. |
| `services/safety_service/tests/test_event_delivery_guard.py` | Modify | Fault-injection: a failing bridge handler propagates on the crisis path. |
| `services/notification-service/src/infrastructure/event_bridge.py` | Modify | Start/stop an `OutboxPoller` so notification's outbox drains. |
| `services/orchestrator_service/src/infrastructure/event_bridge.py` | Modify | Start/stop an `OutboxPoller`. |
| `services/personality_service/src/infrastructure/event_bridge.py` | Modify | Start/stop an `OutboxPoller`. |
| `services/user-service/src/infrastructure/event_bridge.py` | Modify | Start/stop an `OutboxPoller`. |
| `services/notification-service/tests/test_bridge_outbox_poller.py` | Create | Prove each of the 4 bridges drains its outbox. |
| `src/solace_events/publisher.py` | Modify | Extend `IdempotencyStore` protocol + `InMemoryIdempotencyStore` with `reserve/finalize/release`; add `ReservationState`. |
| `src/solace_events/postgres_stores.py` | Modify | `PostgresIdempotencyStore.reserve/finalize/release` + `status` column DDL. |
| `src/solace_events/consumer.py` | Modify | Two-phase idempotency: reserve→handlers→finalize(success)/release(failure). |
| `tests/solace_events/test_consumer_two_phase.py` | Create | Fault-injection: a transient handler failure releases the reservation (no permanent de-dupe). |
| `src/solace_events/dead_letter.py` | Modify | Add `DeadLetterReaper` (periodic `process_retriable`) + `compute_backoff_seconds` reuse. |
| `src/solace_events/publisher.py` | Modify | `OutboxPoller` exponential backoff + jitter on poll failure. |
| `tests/solace_events/test_dlq_reaper.py` | Create | Prove the reaper re-drives a retriable record; prove poller backs off on repeated failure. |
| `services/notification-service/src/main.py` | Modify | Inject `dlq_store=PostgresDLQStore(_event_pool)` into `initialize_safety_consumer`. |
| `services/notification-service/tests/test_dlq_store_injection.py` | Create | Prove the durable DLQ store is injected when a pool exists. |
| `services/orchestrator_service/src/main.py` | Modify | Kafka-bridge init fail-loud in production (BLK-14). |
| `services/orchestrator_service/tests/test_bridge_failloud.py` | Create | Prove production bridge-init failure raises; dev warns. |
| `src/solace_events/monitoring.py` | Create | `EventPlaneMetrics` — DLQ backlog + consumer lag snapshot (H-15). |
| `tests/solace_events/test_monitoring.py` | Create | Prove metrics reflect DLQ backlog and consumer lag. |

---

### Task 1: BLK-07 — Turn the event plane ON in compose

The whole pipeline is dead because `kafka_enabled` defaults to `False` on safety/therapy/notification and compose never overrides it. Safety settings bind `env_prefix="SAFETY_"`, therapy `"THERAPY_"`, notification's `KafkaConfig` binds `env_prefix="KAFKA_"`.

**Files:**
- Modify `docker-compose.yml` (safety-service env ~244–252; therapy-service env ~385–395; notification-service env ~277–289).
- Create `tests/deployment/test_compose_kafka_enabled.py`.

**Interfaces:**
- Consumes: `docker-compose.yml` (YAML).
- Produces: env vars `SAFETY_KAFKA_ENABLED=true`, `SAFETY_KAFKA_BOOTSTRAP_SERVERS=kafka:9092`, `THERAPY_KAFKA_ENABLED=true`, `THERAPY_KAFKA_BOOTSTRAP_SERVERS=kafka:9092`, `KAFKA_ENABLED=true`, `KAFKA_BOOTSTRAP_SERVERS=kafka:9092` (notification).

**Steps:**

- [ ] Write the FAILING test `tests/deployment/test_compose_kafka_enabled.py`:
  ```python
  """BLK-07: the crisis event plane must be ENABLED in the shipped compose file."""
  from __future__ import annotations

  from pathlib import Path

  import yaml

  COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


  def _env(service: str) -> dict[str, str]:
      data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
      env = data["services"][service]["environment"]
      # compose env may be a list ("K=V") or a mapping; normalize to a dict of str
      if isinstance(env, list):
          out: dict[str, str] = {}
          for item in env:
              k, _, v = str(item).partition("=")
              out[k] = v
          return out
      return {k: str(v) for k, v in env.items()}


  def test_safety_kafka_enabled():
      env = _env("safety-service")
      assert env.get("SAFETY_KAFKA_ENABLED") == "true"
      assert env.get("SAFETY_KAFKA_BOOTSTRAP_SERVERS") == "kafka:9092"


  def test_therapy_kafka_enabled():
      env = _env("therapy-service")
      assert env.get("THERAPY_KAFKA_ENABLED") == "true"
      assert env.get("THERAPY_KAFKA_BOOTSTRAP_SERVERS") == "kafka:9092"


  def test_notification_kafka_enabled():
      env = _env("notification-service")
      assert env.get("KAFKA_ENABLED") == "true"
      assert env.get("KAFKA_BOOTSTRAP_SERVERS") == "kafka:9092"
  ```
- [ ] Run it — expect FAIL (keys absent):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/deployment/test_compose_kafka_enabled.py -o "addopts=" -p no:logging -q
  ```
- [ ] Add to the `safety-service` `environment:` block in `docker-compose.yml` (after line 252, `AUTH_SECRET_KEY`):
  ```yaml
      SAFETY_KAFKA_ENABLED: "true"
      SAFETY_KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      SAFETY_KAFKA_USE_MOCK: "false"
  ```
- [ ] Add to the `therapy-service` `environment:` block (after line 395, `AUTH_SECRET_KEY`):
  ```yaml
      THERAPY_KAFKA_ENABLED: "true"
      THERAPY_KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      THERAPY_KAFKA_USE_MOCK: "false"
  ```
  Also add a `kafka` dependency so it starts after the broker — extend the therapy `depends_on:` (lines 398–402) with:
  ```yaml
      kafka:
        condition: service_healthy
  ```
- [ ] Add to the `notification-service` `environment:` block (after line 289, `AUTH_SECRET_KEY`):
  ```yaml
      KAFKA_ENABLED: "true"
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      KAFKA_USE_MOCK: "false"
  ```
- [ ] Run it — expect PASS.
- [ ] **Regression gate:** run the affected suite (do NOT commit):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/deployment/test_compose_kafka_enabled.py -o "addopts=" -p no:logging -q
  ```

---

### Task 2: BLK-08 — Therapy detects a crisis but emits no event

`_handle_crisis` sets risk + persists the session but never publishes to the EventBus, and `therapy/src/events.py:to_kafka_event` has no crisis branch — so a therapy-originated crisis never reaches Kafka. Add a `CrisisDetectedEvent` domain event (reusing the existing `RISK_LEVEL_CRITICAL` enum), map it to the canonical `solace_events.schemas.CrisisDetectedEvent`, and publish it in `_handle_crisis`. The therapy `KafkaEventBridge` already `subscribe_all`s the EventBus and already starts an `OutboxPoller`, so a published event flows straight to the durable outbox.

**Files:**
- Modify `services/therapy_service/src/events.py` (add dataclass after `SteppedCareChangedEvent` ~line 176; add mapping branch inside `to_kafka_event` before the final `return None` ~line 241, and extend the import block ~line 186–192).
- Modify `services/therapy_service/src/domain/service.py` (`_handle_crisis`, lines 428–444).
- Create `services/therapy_service/tests/test_crisis_event_emission.py`.

**Interfaces:**
- Consumes: `EventBus.publish(DomainEvent)`; `SessionManager.get_session(session_id) -> SessionState` (has `.user_id`, `.session_id`, `.current_risk`).
- Produces: therapy `CrisisDetectedEvent.create(session_id, user_id, risk_level: str, trigger_indicators: list[str]) -> CrisisDetectedEvent` (event_type `RISK_LEVEL_CRITICAL`); `to_kafka_event(...) -> solace_events.schemas.CrisisDetectedEvent` with `crisis_level`, `trigger_indicators`, `detection_layer=1`, `confidence=Decimal("0.9")`, `escalation_action="escalate"`, `requires_human_review=True`.

**Steps:**

- [ ] Write the FAILING test `services/therapy_service/tests/test_crisis_event_emission.py`:
  ```python
  """BLK-08: a therapy-detected crisis must emit a mappable crisis event (no silent drop)."""
  from __future__ import annotations

  from uuid import uuid4

  import pytest

  from services.therapy_service.src.events import (
      CrisisDetectedEvent,
      EventType,
      to_kafka_event,
  )


  def test_crisis_domain_event_has_critical_type():
      ev = CrisisDetectedEvent.create(
          session_id=uuid4(),
          user_id=uuid4(),
          risk_level="HIGH",
          trigger_indicators=["Crisis keyword detected: suicide"],
      )
      assert ev.event_type == EventType.RISK_LEVEL_CRITICAL
      assert ev.payload["risk_level"] == "HIGH"
      assert ev.payload["trigger_indicators"] == ["Crisis keyword detected: suicide"]


  def test_crisis_event_maps_to_canonical_kafka_crisis():
      from solace_events.schemas import CrisisDetectedEvent as KafkaCrisis

      ev = CrisisDetectedEvent.create(
          session_id=uuid4(),
          user_id=uuid4(),
          risk_level="HIGH",
          trigger_indicators=["Potential harm language detected"],
      )
      kafka_ev = to_kafka_event(ev)
      assert isinstance(kafka_ev, KafkaCrisis)
      assert kafka_ev.event_type == "safety.crisis.detected"
      assert kafka_ev.escalation_action == "escalate"
      assert kafka_ev.requires_human_review is True
      assert "Potential harm language detected" in kafka_ev.trigger_indicators


  def test_crisis_event_requires_user_id():
      # to_kafka_event returns None for a user-less event; a crisis always has one.
      ev = CrisisDetectedEvent.create(
          session_id=uuid4(), user_id=uuid4(), risk_level="CRITICAL", trigger_indicators=["x"],
      )
      assert to_kafka_event(ev) is not None
  ```
- [ ] Run it — expect FAIL (`ImportError: cannot import name 'CrisisDetectedEvent'`):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/therapy_service/tests/test_crisis_event_emission.py -o "addopts=" -p no:logging -q
  ```
- [ ] Add the domain event dataclass in `services/therapy_service/src/events.py` after `SteppedCareChangedEvent` (~line 176):
  ```python
  @dataclass
  class CrisisDetectedEvent(DomainEvent):
      """Event raised when the therapy safety check detects a crisis.

      BLK-08: reuses the RISK_LEVEL_CRITICAL type so therapy crises ride the same
      canonical crisis schema the notification consumer already handles. The
      EventBus-subscribed KafkaEventBridge converts + persists this to the durable
      outbox, so a therapy crisis reaches a clinician like a safety-service crisis.
      """
      event_type: EventType = field(default=EventType.RISK_LEVEL_CRITICAL)
      aggregate_type: str = "therapy_session"

      @classmethod
      def create(
          cls,
          session_id: UUID,
          user_id: UUID,
          risk_level: str,
          trigger_indicators: list[str],
      ) -> CrisisDetectedEvent:
          return cls(
              aggregate_id=session_id,
              user_id=user_id,
              payload={"risk_level": risk_level, "trigger_indicators": list(trigger_indicators)},
          )
  ```
- [ ] Extend the `to_kafka_event` import block (~lines 186–192) to also import the canonical crisis schema + `CrisisLevel`:
  ```python
          from src.solace_events.schemas import (
              EventMetadata as KafkaMetadata,
              TherapySessionStartedEvent as KafkaTherapyStarted,
              InterventionDeliveredEvent as KafkaInterventionDelivered,
              ProgressMilestoneEvent as KafkaMilestone,
              CrisisDetectedEvent as KafkaCrisisDetected,
              CrisisLevel as KafkaCrisisLevel,
              TherapyModality,
          )
  ```
- [ ] Add the mapping branch inside `to_kafka_event`, immediately before the final `return None` (~line 241):
  ```python
      if event.event_type == EventType.RISK_LEVEL_CRITICAL:
          raw_level = str(event.payload.get("risk_level", "HIGH")).upper()
          try:
              crisis_level = KafkaCrisisLevel(raw_level)
          except ValueError:
              crisis_level = KafkaCrisisLevel.HIGH
          return KafkaCrisisDetected(
              **base,
              crisis_level=crisis_level,
              trigger_indicators=list(event.payload.get("trigger_indicators", [])),
              detection_layer=1,
              confidence=Decimal("0.9"),
              escalation_action="escalate",
              requires_human_review=True,
          )
  ```
  (`Decimal` is already imported inside `to_kafka_event` at line 185.)
- [ ] Run the events test — expect PASS.
- [ ] Now the service-level emission. Add the publish to `_handle_crisis` in `services/therapy_service/src/domain/service.py`, right after the `await self._persist_session(...)` call (line 438) and before the `return TherapyMessageResult(...)`:
  ```python
          # BLK-08: emit a crisis domain event so the KafkaEventBridge (subscribed to
          # the EventBus) converts + persists it to the durable outbox and it reaches
          # the notification consumer -> a clinician. Do NOT swallow: a crisis that
          # fails to enqueue must surface.
          if self._event_bus is not None:
              crisis_session = self._session_manager.get_session(session_id)
              if crisis_session is not None:
                  from ..events import CrisisDetectedEvent as _CrisisDetectedEvent
                  risk = crisis_session.current_risk
                  crisis_event = _CrisisDetectedEvent.create(
                      session_id=session_id,
                      user_id=crisis_session.user_id,
                      risk_level=risk.value if hasattr(risk, "value") else str(risk),
                      trigger_indicators=list(safety_result.get("alerts", [])),
                  )
                  await self._event_bus.publish(crisis_event)
  ```
- [ ] Extend the test with a service-level emission proof (append to `test_crisis_event_emission.py`):
  ```python
  @pytest.mark.asyncio
  async def test_handle_crisis_publishes_to_event_bus():
      from services.therapy_service.src.events import EventBus
      from services.therapy_service.src.domain.session_manager import (
          SessionManager,
          SessionManagerSettings,
      )
      from services.therapy_service.src.domain.service import (
          TherapyOrchestrator,
          TherapyOrchestratorSettings,
      )
      from services.therapy_service.src.domain.technique_selector import (
          TechniqueSelector,
          TechniqueSelectorSettings,
      )

      bus = EventBus()
      session_manager = SessionManager(SessionManagerSettings())
      orch = TherapyOrchestrator(
          settings=TherapyOrchestratorSettings(),
          technique_selector=TechniqueSelector(TechniqueSelectorSettings()),
          session_manager=session_manager,
          llm_client=None,
          event_bus=bus,
      )
      await orch.initialize()
      start = await orch.start_session(user_id=uuid4(), context={})
      bus.clear_published_events()
      # A crisis keyword drives _check_safety -> _handle_crisis.
      await orch.process_message(session_id=start.session_id, message="I want to end my life")
      types = [e.event_type for e in bus.get_published_events()]
      from services.therapy_service.src.events import EventType
      assert EventType.RISK_LEVEL_CRITICAL in types, f"no crisis event published: {types}"
  ```
  (If `TherapyOrchestrator.__init__` requires more collaborators in this codebase revision, construct them the way `services/therapy_service/tests/test_false_positive_crisis.py` does — reuse that test's fixture wiring so the constructor call is correct.)
- [ ] Run the full test file — expect PASS.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/therapy_service/tests/test_crisis_event_emission.py services/therapy_service/tests/test_events.py services/therapy_service/tests/test_event_bridge_wiring.py services/therapy_service/tests/test_false_positive_crisis.py -o "addopts=" -p no:logging -q
  ```

---

### Task 3: BLK-09 — Safety crisis events ride a non-durable, swallow-all queue

`SafetyEventPublisher.publish()` puts events on an in-memory `asyncio.Queue` drained by a background task; `_dispatch_event()` catches and logs every handler/callback exception. So a crisis event is (a) lost if the process dies before the background task drains it, and (b) silently dropped if the bridge's durable-outbox write fails. Fix: for **critical** events (crisis/escalation) dispatch **synchronously** in `publish()` (durable-at-publish — the bridge's outbox write completes before `publish()` returns) and make `_dispatch_event` **propagate** failures for critical events. Also stop the safety `KafkaEventBridge.handle()` from swallowing on the critical path so the outbox-write failure actually reaches the propagating dispatch.

**Files:**
- Modify `services/safety_service/src/events.py` (`SafetyEventPublisher`, lines 345–417).
- Modify `services/safety_service/src/infrastructure/event_bridge.py` (`KafkaEventBridge.handle`, lines 113–137).
- Modify `services/safety_service/tests/test_event_delivery_guard.py`.

**Interfaces:**
- Consumes: `SafetyEvent` subclasses (`CrisisDetectedEvent`, `EscalationTriggeredEvent`, …).
- Produces: `SafetyEventPublisher.publish(event)` — synchronous durable dispatch for critical events, propagating on failure; `_dispatch_event(event)` raises the first handler/callback exception for critical events; `KafkaEventBridge.handle(event)` re-raises on publish failure for critical events.
- New module constant: `CRITICAL_EVENT_TYPES: frozenset[EventType]` in `events.py`.

**Steps:**

- [ ] Write the FAILING fault-injection test — append to `services/safety_service/tests/test_event_delivery_guard.py` (or create it if absent):
  ```python
  """BLK-09: a crisis event must be durably dispatched and MUST NOT be swallowed."""
  from __future__ import annotations

  from decimal import Decimal
  from uuid import uuid4

  import pytest

  from services.safety_service.src.events import (
      CrisisDetectedEvent,
      EventType,
      SafetyCheckCompletedEvent,
      SafetyEventHandler,
      SafetyEventPublisher,
  )


  class _BoomHandler(SafetyEventHandler):
      def __init__(self) -> None:
          self.calls = 0

      async def handle(self, event) -> None:
          self.calls += 1
          raise RuntimeError("durable outbox write failed")


  def _crisis() -> CrisisDetectedEvent:
      return CrisisDetectedEvent(
          user_id=uuid4(),
          crisis_level="HIGH",
          risk_score=Decimal("0.9"),
          trigger_indicators=["suicide"],
          requires_escalation=True,
      )


  @pytest.mark.asyncio
  async def test_crisis_publish_propagates_handler_failure():
      pub = SafetyEventPublisher()
      boom = _BoomHandler()
      pub.register_handler(EventType.CRISIS_DETECTED, boom)
      # A failed durable dispatch on the crisis path must NOT be swallowed.
      with pytest.raises(RuntimeError):
          await pub.publish(_crisis())
      assert boom.calls == 1  # dispatched synchronously, inline


  @pytest.mark.asyncio
  async def test_noncritical_publish_still_tolerant():
      pub = SafetyEventPublisher()
      boom = _BoomHandler()
      pub.register_handler(EventType.SAFETY_CHECK_COMPLETED, boom)
      # Non-critical telemetry keeps legacy tolerant behavior (enqueue, background).
      await pub.start()
      await pub.publish(
          SafetyCheckCompletedEvent(user_id=uuid4(), check_type="keyword", is_safe=True)
      )
      await pub.stop()
  ```
- [ ] Run it — expect FAIL (`publish` currently enqueues and never raises):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/safety_service/tests/test_event_delivery_guard.py -o "addopts=" -p no:logging -q
  ```
- [ ] In `services/safety_service/src/events.py`, add a module constant after the `EventType` enum (~line 39):
  ```python
  # BLK-09: events whose loss is a life-safety failure. These are dispatched
  # synchronously (durable-at-publish) and their dispatch errors are propagated,
  # never swallowed.
  CRITICAL_EVENT_TYPES: frozenset[EventType] = frozenset(
      {EventType.CRISIS_DETECTED, EventType.ESCALATION_TRIGGERED}
  )
  ```
- [ ] Replace `SafetyEventPublisher.publish` (lines 383–387) with a critical-aware version:
  ```python
      async def publish(self, event: SafetyEvent) -> None:
          """Publish a safety event.

          BLK-09: critical (crisis/escalation) events are dispatched SYNCHRONOUSLY so
          the registered KafkaEventBridge writes them to the durable Postgres outbox
          before this returns (durable-at-publish); any failure propagates so the
          caller can fail loud. Non-critical telemetry keeps the tolerant background
          queue path.
          """
          if event.event_type in CRITICAL_EVENT_TYPES:
              await self._dispatch_event(event)
              logger.info(
                  "safety_critical_event_dispatched",
                  event_type=event.event_type.value,
                  event_id=str(event.event_id),
                  user_id=str(event.user_id) if event.user_id else None,
              )
              return
          await self._event_queue.put(event)
          logger.info("safety_event_published", event_type=event.event_type.value,
                      event_id=str(event.event_id), user_id=str(event.user_id) if event.user_id else None)
  ```
- [ ] Replace `_dispatch_event` (lines 405–417) so it does not swallow critical failures:
  ```python
      async def _dispatch_event(self, event: SafetyEvent) -> None:
          """Dispatch event to handlers and callbacks.

          BLK-09: for CRITICAL_EVENT_TYPES a handler/callback failure is re-raised
          (after all are attempted) so a durable-write failure on the crisis path is
          never silently dropped. Non-critical events keep the log-and-continue
          behavior so a noisy audit sink cannot fail a safety assessment.
          """
          critical = event.event_type in CRITICAL_EVENT_TYPES
          first_error: Exception | None = None
          handlers = self._handlers.get(event.event_type, [])
          for handler in handlers:
              try:
                  await handler.handle(event)
              except Exception as e:
                  first_error = first_error or e
                  logger.error("event_handler_error", handler=type(handler).__name__, error=str(e))
          for callback in self._async_callbacks:
              try:
                  await callback(event)
              except Exception as e:
                  first_error = first_error or e
                  logger.error("event_callback_error", error=str(e))
          if critical and first_error is not None:
              raise first_error
  ```
- [ ] Make the safety bridge re-raise on the critical path. In `services/safety_service/src/infrastructure/event_bridge.py`, replace the `except Exception as e:` block inside `KafkaEventBridge.handle` (lines 131–137) with:
  ```python
          except Exception as e:
              logger.error(
                  "kafka_publish_failed",
                  event_type=event.event_type.value,
                  event_id=str(event.event_id),
                  error=str(e),
              )
              # BLK-09: a failed durable-outbox write for a crisis/escalation must not
              # be swallowed — propagate so SafetyEventPublisher._dispatch_event raises
              # and the safety domain can fail loud instead of losing the event.
              from ..events import CRITICAL_EVENT_TYPES
              if event.event_type in CRITICAL_EVENT_TYPES:
                  raise
  ```
- [ ] Run the fault-injection test — expect PASS.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/safety_service/tests/test_events.py services/safety_service/tests/test_event_delivery_guard.py services/safety_service/tests/test_crisis_event_flow.py services/safety_service/tests/test_event_bridge_outbox.py -o "addopts=" -p no:logging -q
  ```

---

### Task 4: BLK-10 — 4 of 7 bridges never start an OutboxPoller

notification/orchestrator/personality/user bridges `create_publisher(use_outbox=True, ...)` but never run an `OutboxPoller`, so outbox rows persist and never drain to Kafka. Safety and therapy already start one; mirror that pattern. (Diagnosis is the 7th and already correct.)

**Files:**
- Modify `services/notification-service/src/infrastructure/event_bridge.py`.
- Modify `services/orchestrator_service/src/infrastructure/event_bridge.py`.
- Modify `services/personality_service/src/infrastructure/event_bridge.py`.
- Modify `services/user-service/src/infrastructure/event_bridge.py`.
- Create `services/notification-service/tests/test_bridge_outbox_poller.py`.

**Interfaces:**
- Consumes: `solace_events.publisher.OutboxPoller(publisher).start()/.stop()`.
- Produces: each bridge holds `self._poller: OutboxPoller | None`, started in `start()`, stopped in `stop()`.

**Steps:**

- [ ] Write the FAILING test `services/notification-service/tests/test_bridge_outbox_poller.py` (drives all four bridges through a mock publisher and asserts the outbox is drained):
  ```python
  """BLK-10: notification/orchestrator/personality/user bridges must drain their outbox."""
  from __future__ import annotations

  import asyncio
  from uuid import uuid4

  import pytest

  from solace_events.schemas import SessionStartedEvent


  def _event():
      return SessionStartedEvent(user_id=uuid4(), session_number=1)


  async def _drain_and_assert(bridge, publish_coro):
      await bridge.start()
      try:
          # publish an event through the bridge's publisher, then let the poller run
          await bridge._publisher.publish(_event())
          for _ in range(50):
              await asyncio.sleep(0.02)
              msgs = bridge._publisher._producer.get_messages()
              if msgs:
                  break
          assert bridge._publisher._producer.get_messages(), "outbox never drained to Kafka"
          assert bridge._poller is not None
      finally:
          await bridge.stop()


  @pytest.mark.asyncio
  async def test_notification_bridge_drains_outbox():
      from services.notification_service.src.infrastructure.event_bridge import KafkaEventBridge
      bridge = KafkaEventBridge(use_mock=True)
      await _drain_and_assert(bridge, None)


  @pytest.mark.asyncio
  async def test_user_bridge_drains_outbox():
      from services.user_service.src.infrastructure.event_bridge import KafkaEventBridge
      bridge = KafkaEventBridge(use_mock=True)
      await _drain_and_assert(bridge, None)


  @pytest.mark.asyncio
  async def test_orchestrator_bridge_drains_outbox():
      from services.orchestrator_service.src.infrastructure.event_bridge import KafkaEventBridge
      bridge = KafkaEventBridge(event_bus=None, use_mock=True)
      await _drain_and_assert(bridge, None)


  @pytest.mark.asyncio
  async def test_personality_bridge_drains_outbox():
      from services.personality_service.src.events import EventBus
      from services.personality_service.src.infrastructure.event_bridge import KafkaEventBridge
      bridge = KafkaEventBridge(event_bus=EventBus(), use_mock=True)
      await _drain_and_assert(bridge, None)
  ```
  (Import paths use the dotted package form; the notification/user services live under hyphenated dirs — import via `services.notification_service...` if a package alias exists, otherwise adjust to the import style the existing `services/notification-service/tests/test_events.py` uses. Match the working test's import convention.)
- [ ] Run it — expect FAIL (`bridge._poller` is `None`; `get_messages()` empty because nothing drains):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/notification-service/tests/test_bridge_outbox_poller.py -o "addopts=" -p no:logging -q
  ```
- [ ] **notification** `event_bridge.py`: extend the import (line 16) and `start`/`stop`.
  - Change line 16 to import `OutboxPoller`:
    ```python
      from src.solace_events.publisher import EventPublisher, OutboxPoller, create_publisher
    ```
  - In `__init__`, after `self._started = False` (line 47) add `self._poller = None` (place it before `self._started` assignment so both exist even on the not-available branch — mirror safety: set `self._poller = None` at the top of `__init__`).
  - Replace `start` (lines 50–55):
    ```python
      async def start(self) -> None:
          """Start the Kafka publisher and the outbox drain poller."""
          if self._publisher and not self._started:
              await self._publisher.start()
              # BLK-10: without a poller the publisher writes to the outbox but never
              # sends — notification delivery events never reach Kafka.
              self._poller = OutboxPoller(self._publisher)
              await self._poller.start()
              self._started = True
              logger.info("notification_kafka_bridge_started")
    ```
  - Replace `stop` (lines 57–62):
    ```python
      async def stop(self) -> None:
          """Stop the outbox poller and the Kafka publisher."""
          if self._publisher and self._started:
              if self._poller:
                  await self._poller.stop()
                  self._poller = None
              await self._publisher.stop()
              self._started = False
              logger.info("notification_kafka_bridge_stopped")
    ```
- [ ] **orchestrator** `event_bridge.py`: import `OutboxPoller` (line 19), init `self._poller = None` in `__init__`, and update `start`/`stop` the same way (start the poller after `await self._publisher.start()`, before `self._event_bus.subscribe_all`; stop it before `self._publisher.stop()`). Concretely, `start` becomes:
  ```python
      async def start(self) -> None:
          """Start the Kafka publisher, outbox drain poller, and auto-subscribe."""
          if self._publisher and not self._started:
              await self._publisher.start()
              self._poller = OutboxPoller(self._publisher)
              await self._poller.start()
              self._started = True
              if self._event_bus:
                  self._event_bus.subscribe_all(self._handle_event)
              logger.info("orchestrator_kafka_bridge_started")
  ```
  and `stop`:
  ```python
      async def stop(self) -> None:
          """Stop the outbox poller and the Kafka publisher."""
          if self._publisher and self._started:
              if self._poller:
                  await self._poller.stop()
                  self._poller = None
              await self._publisher.stop()
              self._started = False
              logger.info("orchestrator_kafka_bridge_stopped")
  ```
  (Add `self._poller = None` in `__init__`, e.g. immediately after `self._event_bus = event_bus`.)
- [ ] **personality** `event_bridge.py`: same edit — import `OutboxPoller` (line 15), `self._poller = None` after `self._event_bus = event_bus`, start the poller in `start()` before `subscribe_all`, stop it in `stop()` before `self._publisher.stop()`.
- [ ] **user-service** `event_bridge.py`: same edit — import `OutboxPoller` (line 16), `self._poller = None` at the top of `__init__`, start the poller in `start()`, stop it in `stop()` (this bridge has no EventBus, so it mirrors notification exactly).
- [ ] Run the test — expect PASS.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/notification-service/tests/test_bridge_outbox_poller.py services/notification-service/tests/test_events.py services/personality_service/tests/ services/user-service/tests/test_event_bridge.py -o "addopts=" -p no:logging -q
  ```
  (If a listed path does not exist in this revision, drop it — run whichever of the four services' event/bridge suites exist.)

---

### Task 5: BLK-11 — Consumer idempotency claims BEFORE the handler runs → transient failure permanently de-dupes

`EventConsumer._process_message` calls `idempotency_store.mark_processed()` (a one-shot claim) BEFORE invoking handlers; on a handler exception the message is marked FAILED and the offset advances, and the claim is already recorded — so a legitimate redelivery (from the outbox's at-least-once re-publish) is skipped forever. This is at-most-once on crisis events. Fix: **two-phase** — `reserve` before handlers, `finalize` on success, `release` on failure (so a redelivery is reprocessed, not de-duped). Keep `mark_processed` for backward compatibility (existing tests/callers).

**Files:**
- Modify `src/solace_events/publisher.py` (`IdempotencyStore` protocol + `InMemoryIdempotencyStore`; add `ReservationState`).
- Modify `src/solace_events/postgres_stores.py` (`PostgresIdempotencyStore` + `PROCESSED_EVENTS_TABLE_DDL` `status` column).
- Modify `src/solace_events/consumer.py` (`_process_message` two-phase flow).
- Create `tests/solace_events/test_consumer_two_phase.py`.

**Interfaces:**
- Produces (new): `ReservationState(str, Enum)` = `NEW | IN_FLIGHT | DONE`; `IdempotencyStore.reserve(event_id, consumer_group) -> ReservationState`; `IdempotencyStore.finalize(event_id, consumer_group) -> None`; `IdempotencyStore.release(event_id, consumer_group) -> None`.
- Consumes: `EventConsumer._process_message` uses `reserve/finalize/release` when the store exposes them; falls back to `mark_processed` when it does not.

**Steps:**

- [ ] Write the FAILING fault-injection test `tests/solace_events/test_consumer_two_phase.py`:
  ```python
  """BLK-11: a transient handler failure must NOT permanently de-dupe a redelivery."""
  from __future__ import annotations

  from uuid import uuid4

  import pytest

  from solace_events.publisher import (
      InMemoryIdempotencyStore,
      ReservationState,
  )
  from solace_events.consumer import EventConsumer, MockKafkaConsumerAdapter, ProcessingStatus
  from solace_events.schemas import SessionStartedEvent


  def _msg(event: SessionStartedEvent) -> dict:
      return event.to_dict()


  @pytest.mark.asyncio
  async def test_reserve_release_allows_reprocess():
      store = InMemoryIdempotencyStore()
      eid, grp = uuid4(), "crisis-consumer"
      assert await store.reserve(eid, grp) == ReservationState.NEW
      # second concurrent delivery while in flight -> IN_FLIGHT (skip)
      assert await store.reserve(eid, grp) == ReservationState.IN_FLIGHT
      # a transient failure releases the reservation
      await store.release(eid, grp)
      # a legitimate redelivery is treated as NEW again (NOT permanently de-duped)
      assert await store.reserve(eid, grp) == ReservationState.NEW
      await store.finalize(eid, grp)
      # once finalized, a duplicate is skipped as DONE
      assert await store.reserve(eid, grp) == ReservationState.DONE


  @pytest.mark.asyncio
  async def test_consumer_transient_failure_then_success_reprocesses():
      store = InMemoryIdempotencyStore()
      adapter = MockKafkaConsumerAdapter()
      consumer = EventConsumer(adapter, idempotency_store=store, consumer_group="crisis-consumer")

      calls = {"n": 0}

      async def flaky(event):
          calls["n"] += 1
          if calls["n"] == 1:
              raise RuntimeError("transient downstream outage")

      consumer.register_handler("session.started", flaky)
      ev = SessionStartedEvent(user_id=uuid4(), session_number=1)

      # First delivery: handler raises -> FAILED, reservation released.
      r1 = await consumer._process_message("solace.sessions", 0, 0, _msg(ev))
      assert r1.status == ProcessingStatus.FAILED
      # Redelivery (same event_id, outbox re-publish): must be reprocessed, not skipped.
      r2 = await consumer._process_message("solace.sessions", 0, 1, _msg(ev))
      assert r2.status == ProcessingStatus.SUCCESS
      assert calls["n"] == 2
      # A THIRD duplicate after success is skipped (dedupe on the happy path holds).
      r3 = await consumer._process_message("solace.sessions", 0, 2, _msg(ev))
      assert r3.status == ProcessingStatus.SKIP
  ```
- [ ] Run it — expect FAIL (`ImportError: ReservationState`; store has no `reserve`):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_events/test_consumer_two_phase.py -o "addopts=" -p no:logging -q
  ```
- [ ] In `src/solace_events/publisher.py`, add the `ReservationState` enum (after the `OutboxStatus` enum, ~line 61) and reuse the existing `from enum import Enum` import:
  ```python
  class ReservationState(str, Enum):
      """Two-phase idempotency reservation outcome (BLK-11).

      NEW      -> this delivery reserved the slot; run the handlers.
      IN_FLIGHT-> another delivery holds an un-finalized reservation; skip (avoid
                  concurrent double-fire); it will be finalized or released.
      DONE     -> already processed successfully; skip (dedupe).
      """
      NEW = "NEW"
      IN_FLIGHT = "IN_FLIGHT"
      DONE = "DONE"
  ```
- [ ] Extend the `IdempotencyStore` Protocol (lines 157–176) with the two-phase methods (add after `was_processed`):
  ```python
      async def reserve(self, event_id: UUID, consumer_group: str) -> "ReservationState":
          """Reserve (event_id, consumer_group) before running handlers.

          NEW if this delivery may process; IN_FLIGHT if an un-finalized reservation
          exists (skip); DONE if already finalized (skip).
          """
          ...

      async def finalize(self, event_id: UUID, consumer_group: str) -> None:
          """Mark a reserved slot processed (IN_FLIGHT -> DONE)."""
          ...

      async def release(self, event_id: UUID, consumer_group: str) -> None:
          """Release a reservation after a failure so a redelivery can reprocess."""
          ...
  ```
- [ ] Replace `InMemoryIdempotencyStore` (lines 178–199) so it tracks per-key state while keeping `mark_processed`/`was_processed`:
  ```python
  class InMemoryIdempotencyStore:
      """In-memory idempotency store for tests/dev (mirrors the Postgres seam)."""

      def __init__(self) -> None:
          # key -> "IN_FLIGHT" | "DONE"
          self._state: dict[tuple[str, str], str] = {}
          self._lock = asyncio.Lock()

      async def mark_processed(self, event_id: UUID, consumer_group: str) -> bool:
          """Legacy one-shot claim: True if first-seen (kept for back-compat)."""
          async with self._lock:
              key = (str(event_id), consumer_group)
              if key in self._state:
                  return False
              self._state[key] = "DONE"
              return True

      async def was_processed(self, event_id: UUID, consumer_group: str) -> bool:
          async with self._lock:
              return self._state.get((str(event_id), consumer_group)) == "DONE"

      async def reserve(self, event_id: UUID, consumer_group: str) -> ReservationState:
          async with self._lock:
              key = (str(event_id), consumer_group)
              existing = self._state.get(key)
              if existing == "DONE":
                  return ReservationState.DONE
              if existing == "IN_FLIGHT":
                  return ReservationState.IN_FLIGHT
              self._state[key] = "IN_FLIGHT"
              return ReservationState.NEW

      async def finalize(self, event_id: UUID, consumer_group: str) -> None:
          async with self._lock:
              self._state[(str(event_id), consumer_group)] = "DONE"

      async def release(self, event_id: UUID, consumer_group: str) -> None:
          async with self._lock:
              key = (str(event_id), consumer_group)
              if self._state.get(key) == "IN_FLIGHT":
                  del self._state[key]

      def clear(self) -> None:
          """Clear all records (for testing)."""
          self._state.clear()
  ```
- [ ] In `src/solace_events/postgres_stores.py`, evolve `PROCESSED_EVENTS_TABLE_DDL` (lines 53–60) to carry a `status` column (default `DONE` so a pre-existing `mark_processed` insert is treated as finalized):
  ```python
  PROCESSED_EVENTS_TABLE_DDL = """
  CREATE TABLE IF NOT EXISTS processed_events (
      event_id UUID NOT NULL,
      consumer_group VARCHAR(255) NOT NULL,
      status VARCHAR(16) NOT NULL DEFAULT 'DONE',
      processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (event_id, consumer_group)
  );
  -- BLK-11: an existing ledger predating two-phase reservations needs the column.
  ALTER TABLE processed_events ADD COLUMN IF NOT EXISTS status VARCHAR(16) NOT NULL DEFAULT 'DONE';
  """
  ```
- [ ] In `PostgresIdempotencyStore`, keep `mark_processed` (insert `status='DONE'`) and add the two-phase methods. Update `mark_processed` (lines 279–294) to set status explicitly, then append:
  ```python
      async def mark_processed(self, event_id: UUID, consumer_group: str) -> bool:
          async with self._pool.acquire() as conn:
              row = await conn.fetchrow(
                  """INSERT INTO processed_events (event_id, consumer_group, status)
                     VALUES ($1, $2, 'DONE')
                     ON CONFLICT (event_id, consumer_group) DO NOTHING
                     RETURNING event_id""",
                  event_id, consumer_group,
              )
              return row is not None

      async def reserve(self, event_id: UUID, consumer_group: str):
          """Atomic two-phase reserve keyed on (event_id, consumer_group) (BLK-11)."""
          from .publisher import ReservationState
          async with self._pool.acquire() as conn:
              row = await conn.fetchrow(
                  """INSERT INTO processed_events (event_id, consumer_group, status)
                     VALUES ($1, $2, 'IN_FLIGHT')
                     ON CONFLICT (event_id, consumer_group) DO NOTHING
                     RETURNING status""",
                  event_id, consumer_group,
              )
              if row is not None:
                  return ReservationState.NEW
              existing = await conn.fetchrow(
                  """SELECT status FROM processed_events
                     WHERE event_id = $1 AND consumer_group = $2""",
                  event_id, consumer_group,
              )
              if existing is not None and existing["status"] == "DONE":
                  return ReservationState.DONE
              return ReservationState.IN_FLIGHT

      async def finalize(self, event_id: UUID, consumer_group: str) -> None:
          async with self._pool.acquire() as conn:
              await conn.execute(
                  """UPDATE processed_events SET status = 'DONE', processed_at = NOW()
                     WHERE event_id = $1 AND consumer_group = $2""",
                  event_id, consumer_group,
              )

      async def release(self, event_id: UUID, consumer_group: str) -> None:
          async with self._pool.acquire() as conn:
              await conn.execute(
                  """DELETE FROM processed_events
                     WHERE event_id = $1 AND consumer_group = $2 AND status = 'IN_FLIGHT'""",
                  event_id, consumer_group,
              )
  ```
- [ ] Rewrite the idempotency block in `EventConsumer._process_message` (`src/solace_events/consumer.py`, lines 421–445) to two-phase, and wrap handler invocation so a failure releases the reservation. Replace lines 421–463 (from the `# P2-8` comment through the SUCCESS `return`) with:
  ```python
              # BLK-11: two-phase idempotency. Reserve BEFORE handlers; finalize on
              # success; RELEASE on failure so a transient error does not permanently
              # de-dupe a legit redelivery (the outbox is at-least-once). Falls back to
              # the legacy one-shot claim for stores without the two-phase API.
              reserved = False
              store = self._idempotency_store
              if store is not None and hasattr(store, "reserve"):
                  from .publisher import ReservationState
                  state = await store.reserve(event.metadata.event_id, self._consumer_group)
                  if state != ReservationState.NEW:
                      self._metrics.record_duplicate()
                      logger.info(
                          "duplicate_event_skipped",
                          event_type=event.event_type,
                          event_id=str(event.metadata.event_id),
                          consumer_group=self._consumer_group,
                          reservation_state=state.value,
                          topic=topic, partition=partition, offset=offset,
                      )
                      return ProcessingResult(
                          status=ProcessingStatus.SKIP, event_id=event.metadata.event_id
                      )
                  reserved = True
              elif store is not None:
                  first_seen = await store.mark_processed(
                      event.metadata.event_id, self._consumer_group
                  )
                  if not first_seen:
                      self._metrics.record_duplicate()
                      logger.info(
                          "duplicate_event_skipped",
                          event_type=event.event_type,
                          event_id=str(event.metadata.event_id),
                          consumer_group=self._consumer_group,
                          topic=topic, partition=partition, offset=offset,
                      )
                      return ProcessingResult(
                          status=ProcessingStatus.SKIP, event_id=event.metadata.event_id
                      )
              handlers = (
                  self._handlers.get(event.event_type, []) or self._default_handlers
              )
              try:
                  for handler in handlers:
                      await handler(event)
              except Exception:
                  # Release the reservation so a redelivery reprocesses (no de-dupe),
                  # then fall through to the existing FAILED + DLQ path below.
                  if reserved and store is not None:
                      await store.release(event.metadata.event_id, self._consumer_group)
                  raise
              if reserved and store is not None:
                  await store.finalize(event.metadata.event_id, self._consumer_group)
              processing_time = int(
                  (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
              )
              self._metrics.record_success(processing_time)
              logger.debug(
                  "Event processed",
                  event_type=event.event_type,
                  event_id=str(event.metadata.event_id),
                  time_ms=processing_time,
              )
              return ProcessingResult(
                  status=ProcessingStatus.SUCCESS, event_id=event.metadata.event_id
              )
  ```
  The existing `except Exception as e:` block (the FAILED + dead-letter path, lines 464–491) stays as-is and now catches the re-raised handler failure after the reservation was released.
- [ ] Run the two-phase test — expect PASS.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_events/test_consumer_two_phase.py tests/solace_events/test_consumer.py tests/solace_events/test_outbox_claim.py -o "addopts=" -p no:logging -q
  ```

---

### Task 6: BLK-12 — No DLQ reaper; poller hot-loops on failure

`DeadLetterHandler.process_retriable()` exists but is never called, so a DLQ'd crisis event parks forever; and `OutboxPoller._poll_loop` sleeps a fixed interval even when every flush raises, hot-looping and log-storming during a Kafka outage. Add a `DeadLetterReaper` background task that periodically drives `process_retriable()`, and give `OutboxPoller` exponential backoff + jitter on consecutive failures.

**Files:**
- Modify `src/solace_events/dead_letter.py` (add `DeadLetterReaper`).
- Modify `src/solace_events/publisher.py` (`OutboxPoller._poll_loop` backoff+jitter; reuse `compute_backoff_seconds`).
- Create `tests/solace_events/test_dlq_reaper.py`.

**Interfaces:**
- Produces: `DeadLetterReaper(handler: DeadLetterHandler, poll_interval_ms: int = 5000, topic: str | None = None, batch_size: int = 10)` with `.start()/.stop()`; loops `await handler.process_retriable(topic, batch_size)`.
- Consumes: `compute_backoff_seconds(retry_count, base, cap)` (already in `publisher.py`); `random.uniform` for jitter.

**Steps:**

- [ ] Write the FAILING test `tests/solace_events/test_dlq_reaper.py`:
  ```python
  """BLK-12: a DLQ reaper must re-drive retriable records; the poller must back off."""
  from __future__ import annotations

  import asyncio
  from uuid import uuid4

  import pytest

  from solace_events.dead_letter import (
      DeadLetterHandler,
      DeadLetterRecord,
      DeadLetterReaper,
      DeadLetterStore,
      RetryPolicy,
  )
  from solace_events.schemas import SessionStartedEvent


  def _record() -> DeadLetterRecord:
      ev = SessionStartedEvent(user_id=uuid4(), session_number=1)
      return DeadLetterRecord.from_failed_event(
          ev, original_topic="solace.safety", consumer_group="crisis", error=RuntimeError("boom"),
      )


  @pytest.mark.asyncio
  async def test_reaper_redrives_retriable_record():
      store = DeadLetterStore()
      rec = _record()
      await store.save(rec)
      handler = DeadLetterHandler(store=store, retry_policy=RetryPolicy(max_retries=5), consumer_group="crisis")

      redriven = {"n": 0}

      async def retry_ok(event):
          redriven["n"] += 1
          return True

      handler.register_retry_handler(rec.event_type, retry_ok)
      reaper = DeadLetterReaper(handler, poll_interval_ms=20, topic="solace.safety")
      await reaper.start()
      for _ in range(50):
          await asyncio.sleep(0.02)
          if redriven["n"] > 0:
              break
      await reaper.stop()
      assert redriven["n"] >= 1
      assert await store.count_unresolved() == 0  # resolved after successful retry


  @pytest.mark.asyncio
  async def test_poller_backs_off_on_repeated_failure():
      from solace_events.publisher import OutboxPoller

      class _AlwaysFailPublisher:
          def __init__(self):
              self.calls = 0

          async def flush_outbox(self, batch_size):
              self.calls += 1
              raise RuntimeError("kafka down")

      pub = _AlwaysFailPublisher()
      poller = OutboxPoller(pub, poll_interval_ms=10)  # 10ms base
      await poller.start()
      await asyncio.sleep(0.2)
      await poller.stop()
      # With fixed 10ms sleep, ~20 calls would occur in 200ms; backoff must throttle
      # it well below that.
      assert pub.calls < 10, f"poller did not back off: {pub.calls} calls"
  ```
- [ ] Run it — expect FAIL (`ImportError: DeadLetterReaper`; poller does not back off):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_events/test_dlq_reaper.py -o "addopts=" -p no:logging -q
  ```
- [ ] Add `DeadLetterReaper` to `src/solace_events/dead_letter.py` (append after `DeadLetterHandler`, before `get_dlq_topic`):
  ```python
  class DeadLetterReaper:
      """Background task that periodically re-drives retriable DLQ records (BLK-12).

      Without this, a dead-lettered crisis event parks forever. The reaper calls
      ``DeadLetterHandler.process_retriable`` on an interval so a durable, retriable
      failure (a crisis alert that failed transiently) is re-attempted and either
      resolves or exhausts its retry budget with a permanent, alertable record.
      """

      def __init__(
          self,
          handler: DeadLetterHandler,
          poll_interval_ms: int = 5000,
          topic: str | None = None,
          batch_size: int = 10,
      ) -> None:
          self._handler = handler
          self._poll_interval_ms = poll_interval_ms
          self._topic = topic
          self._batch_size = batch_size
          self._running = False
          self._task: asyncio.Task[None] | None = None

      async def start(self) -> None:
          self._running = True
          self._task = asyncio.create_task(self._reap_loop())
          logger.info("dlq_reaper_started", interval_ms=self._poll_interval_ms, topic=self._topic)

      async def stop(self) -> None:
          self._running = False
          if self._task:
              self._task.cancel()
              try:
                  await self._task
              except asyncio.CancelledError:
                  pass
          logger.info("dlq_reaper_stopped")

      async def _reap_loop(self) -> None:
          while self._running:
              try:
                  count = await self._handler.process_retriable(self._topic, self._batch_size)
                  if count:
                      logger.info("dlq_reaper_redrove", count=count, topic=self._topic)
              except Exception as e:  # never let the reaper die on one bad record
                  logger.error("dlq_reaper_error", error=str(e))
              await asyncio.sleep(self._poll_interval_ms / 1000)
  ```
- [ ] Give `OutboxPoller._poll_loop` backoff+jitter in `src/solace_events/publisher.py` (replace lines 622–631). Reuse the module's `compute_backoff_seconds` and track a consecutive-failure counter:
  ```python
      async def _poll_loop(self) -> None:
          """Main polling loop with exponential backoff + jitter on failure (BLK-12)."""
          import random

          consecutive_failures = 0
          base_seconds = self._poll_interval_ms / 1000
          while self._running:
              try:
                  published = await self._publisher.flush_outbox(self._batch_size)
                  if published > 0:
                      logger.debug("Outbox flush completed", published=published)
                  consecutive_failures = 0
                  await asyncio.sleep(base_seconds)
              except Exception as e:
                  consecutive_failures += 1
                  delay = compute_backoff_seconds(
                      consecutive_failures, base_seconds, DEFAULT_RETRY_MAX_SECONDS
                  )
                  # +/-10% jitter so many pollers do not resynchronize their retries.
                  delay += random.uniform(-0.1 * delay, 0.1 * delay)
                  logger.error(
                      "Outbox poll error",
                      error=str(e),
                      consecutive_failures=consecutive_failures,
                      backoff_seconds=round(delay, 3),
                  )
                  await asyncio.sleep(max(delay, base_seconds))
  ```
  (`compute_backoff_seconds`, `DEFAULT_RETRY_MAX_SECONDS` are already defined at the top of `publisher.py`.)
- [ ] Run the reaper test — expect PASS.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_events/test_dlq_reaper.py tests/solace_events/test_dead_letter.py tests/solace_events/test_publisher.py tests/solace_events/test_outbox_claim.py -o "addopts=" -p no:logging -q
  ```

---

### Task 7: BLK-13 — Notification's durable Postgres DLQ store is never injected

`SafetyEventConsumer` already accepts an injectable `dlq_store` (defaulting to the in-memory `DeadLetterStore`), but `notification/src/main.py` calls `initialize_safety_consumer(...)` without it — so a failed crisis alert is lost on restart. Inject a `PostgresDLQStore(_event_pool)` when a pool exists; also start a `DeadLetterReaper` over it (from Task 6) so those durable failures are re-driven.

**Files:**
- Modify `services/notification-service/src/main.py` (consumer init block, lines 250–277; shutdown ~286–289).
- Create `services/notification-service/tests/test_dlq_store_injection.py`.

**Interfaces:**
- Consumes: `solace_events.postgres_stores.PostgresDLQStore(pool)`; `initialize_safety_consumer(..., dlq_store=...)`; `DeadLetterHandler(store=..., consumer_group="notification-service-safety-consumer")`; `DeadLetterReaper(handler, topic=SolaceTopic.SAFETY.value)`.
- Produces: `SafetyEventConsumer._dlq_store` is a `PostgresDLQStore` when `_event_pool` is available.

**Steps:**

- [ ] Write the FAILING test `services/notification-service/tests/test_dlq_store_injection.py`:
  ```python
  """BLK-13: notification must inject a durable PostgresDLQStore when a pool exists."""
  from __future__ import annotations

  from unittest.mock import AsyncMock, MagicMock

  import pytest

  from services.notification_service.src.consumers import SafetyEventConsumer


  def test_injected_dlq_store_is_used():
      from solace_events.postgres_stores import PostgresDLQStore

      pool = MagicMock()
      dlq = PostgresDLQStore(pool)
      consumer = SafetyEventConsumer(
          notification_service=MagicMock(),
          use_mock=True,
          dlq_store=dlq,
      )
      assert consumer._dlq_store is dlq


  def test_default_dlq_store_is_in_memory():
      from solace_events.dead_letter import DeadLetterStore

      consumer = SafetyEventConsumer(notification_service=MagicMock(), use_mock=True)
      assert isinstance(consumer._dlq_store, DeadLetterStore)
  ```
  (Match the import style used by the existing `services/notification-service/tests/test_crisis_consumer.py`.)
- [ ] Run it — expect PASS for the seam (it already exists) but keep it as the guard; if the import path differs, fix the import to the working convention, then confirm both pass. This test locks the seam so the main-wiring change below cannot regress it.
- [ ] In `services/notification-service/src/main.py`, build the durable DLQ store and inject it. Replace the consumer-init block (lines 254–277) with:
  ```python
      dlq_reaper = None
      if settings.kafka.enabled:
          try:
              from .consumers import initialize_safety_consumer, shutdown_safety_consumer
              # Build Kafka settings if solace_events is available
              kafka_settings = None
              try:
                  from solace_events.config import KafkaSettings
                  kafka_settings = KafkaSettings(
                      bootstrap_servers=settings.kafka.bootstrap_servers,
                  )
              except ImportError:
                  logger.warning("solace_events_not_available", reason="Cannot configure Kafka settings")

              # BLK-13: inject a durable Postgres DLQ store so a failed crisis alert
              # survives a worker restart; default in-memory store loses it.
              dlq_store = None
              if _event_pool is not None:
                  try:
                      from solace_events.postgres_stores import PostgresDLQStore
                      dlq_store = PostgresDLQStore(_event_pool)
                      await dlq_store.ensure_table()
                      logger.info("notification_dlq_store_durable")
                  except Exception as e:
                      if settings.service.env == "production":
                          raise RuntimeError(
                              f"durable DLQ store is required in production: {e}"
                          ) from e
                      logger.warning("notification_dlq_store_in_memory", error=str(e))

              safety_consumer = await initialize_safety_consumer(
                  notification_service=_notification_service,
                  kafka_settings=kafka_settings,
                  use_mock=settings.kafka.use_mock,
                  dlq_store=dlq_store,
              )

              # BLK-12/BLK-13: reap the durable DLQ so a dead-lettered crisis alert is
              # re-driven instead of parking forever.
              if dlq_store is not None:
                  from solace_events.dead_letter import DeadLetterHandler, DeadLetterReaper
                  from solace_events.config import SolaceTopic
                  dlq_handler = DeadLetterHandler(
                      store=dlq_store, consumer_group="notification-service-safety-consumer"
                  )
                  dlq_reaper = DeadLetterReaper(dlq_handler, topic=SolaceTopic.SAFETY.value)
                  await dlq_reaper.start()

              logger.info("safety_event_consumer_started")
          except Exception as e:
              # M1-gate R2 (P0): don't boot "healthy" with a dead crisis consumer in prod.
              logger.error("safety_consumer_start_failed", error=str(e))
              if settings.service.env == "production":
                  raise
  ```
  (`_event_pool` is already fetched above at lines 237–242. Move that fetch earlier if needed so it is in scope; it already is.)
- [ ] Stop the reaper on shutdown. In the shutdown section (after the `if safety_consumer:` block ~line 287), add:
  ```python
      if dlq_reaper:
          try:
              await dlq_reaper.stop()
              logger.info("notification_dlq_reaper_stopped")
          except Exception as e:
              logger.error("notification_dlq_reaper_stop_failed", error=str(e))
  ```
- [ ] Run the injection test — expect PASS.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/notification-service/tests/test_dlq_store_injection.py services/notification-service/tests/test_crisis_consumer.py services/notification-service/tests/test_crisis_consumer_required.py -o "addopts=" -p no:logging -q
  ```

---

### Task 8: BLK-14 — Orchestrator Kafka-bridge init silently swallows failure in production

`orchestrator/src/main.py:182-199` wraps bridge init in a blanket `try/except` that only `logger.warning`s — so if Kafka is down at boot, the orchestrator's event plane is silently disabled in production. Make it fail-loud when `environment == "production"`.

**Files:**
- Modify `services/orchestrator_service/src/main.py` (lines 182–199).
- Create `services/orchestrator_service/tests/test_bridge_failloud.py`.

**Interfaces:**
- Consumes: `settings.environment` (`OrchestratorAppSettings`, `env_prefix="ORCHESTRATOR_"`).
- Produces: bridge-init exception re-raised iff `settings.environment == "production"`.

**Steps:**

- [ ] Write the FAILING test `services/orchestrator_service/tests/test_bridge_failloud.py`:
  ```python
  """BLK-14: orchestrator Kafka-bridge init must fail loud in production."""
  from __future__ import annotations

  import pytest


  @pytest.mark.asyncio
  async def test_bridge_failure_raises_in_production(monkeypatch):
      import services.orchestrator_service.src.infrastructure.event_bridge as eb

      async def _boom(*a, **k):
          raise RuntimeError("kafka unreachable")

      monkeypatch.setattr(eb, "initialize_event_bridge", _boom)

      # Simulate the main.py guard directly (unit-level: the try/except policy).
      from services.orchestrator_service.src.main import _init_event_bridge_or_fail

      with pytest.raises(RuntimeError):
          await _init_event_bridge_or_fail(environment="production", event_bus=None, postgres_pool=None)


  @pytest.mark.asyncio
  async def test_bridge_failure_warns_in_dev(monkeypatch):
      import services.orchestrator_service.src.infrastructure.event_bridge as eb

      async def _boom(*a, **k):
          raise RuntimeError("kafka unreachable")

      monkeypatch.setattr(eb, "initialize_event_bridge", _boom)
      from services.orchestrator_service.src.main import _init_event_bridge_or_fail

      # dev tolerates a missing broker -> returns None, does not raise.
      result = await _init_event_bridge_or_fail(environment="development", event_bus=None, postgres_pool=None)
      assert result is None
  ```
- [ ] Run it — expect FAIL (`_init_event_bridge_or_fail` does not exist):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/orchestrator_service/tests/test_bridge_failloud.py -o "addopts=" -p no:logging -q
  ```
- [ ] Extract the bridge-init policy into a testable helper in `services/orchestrator_service/src/main.py` (add near the top-level functions, before `lifespan`):
  ```python
  async def _init_event_bridge_or_fail(environment: str, event_bus, postgres_pool):
      """Initialize the Kafka event bridge; fail loud in production (BLK-14).

      In production a broker outage at boot must NOT silently disable the event
      plane (which carries crisis events) — raise. In dev/test, degrade to None.
      """
      from .infrastructure.event_bridge import initialize_event_bridge
      try:
          bridge = await initialize_event_bridge(event_bus=event_bus, postgres_pool=postgres_pool)
          logger.info("orchestrator_kafka_bridge_started")
          return bridge
      except Exception as e:
          if environment == "production":
              raise RuntimeError(f"orchestrator event bridge is required in production: {e}") from e
          logger.warning("orchestrator_kafka_bridge_not_configured", error=str(e))
          return None
  ```
- [ ] Replace the inline bridge-init block in `lifespan` (lines 182–199) to call the helper:
  ```python
      # Initialize Kafka event bridge for cross-service event publishing
      event_bridge = None
      from .events import get_event_bus
      _event_pool = None
      try:
          from solace_infrastructure.database.connection_manager import ConnectionPoolManager
          _event_pool = await ConnectionPoolManager.get_pool()
      except Exception:
          logger.debug("event_outbox_pool_not_available", hint="Using in-memory outbox")
      event_bridge = await _init_event_bridge_or_fail(
          environment=settings.environment,
          event_bus=get_event_bus(),
          postgres_pool=_event_pool,
      )
      if event_bridge is not None:
          app.state.event_bridge = event_bridge
  ```
- [ ] Run the test — expect PASS.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/orchestrator_service/tests/test_bridge_failloud.py -o "addopts=" -p no:logging -q
  ```

---

### Task 9: H-15 — Wire DLQ-backlog + consumer-lag metrics

`DeadLetterHandler.get_dlq_stats()` and `EventConsumer.metrics` (incl. `current_lag`) exist but nothing reads them — the exact "lost crisis alert" mode has no metric. Add a small `EventPlaneMetrics` snapshot object that aggregates DLQ backlog (via a `DLQStore.count_unresolved()`/`count_by_topic()`) and a consumer's `ConsumerMetrics`, so the notification service and reaper can surface them (and Phase 4 alerting can page on them).

**Files:**
- Create `src/solace_events/monitoring.py`.
- Create `tests/solace_events/test_monitoring.py`.

**Interfaces:**
- Produces: `EventPlaneMetrics.collect(dlq_store, consumer_metrics) -> dict[str, Any]` with keys `dlq_unresolved: int`, `dlq_by_topic: dict[str,int]`, `consumer_lag: int`, `messages_failed: int`, `messages_skipped_duplicate: int`, `success_rate: float`.
- Consumes: `DLQStore.count_unresolved()`, `DLQStore.count_by_topic()` (both `DeadLetterStore` and `PostgresDLQStore` implement them); `ConsumerMetrics` dataclass fields.

**Steps:**

- [ ] Write the FAILING test `tests/solace_events/test_monitoring.py`:
  ```python
  """H-15: DLQ backlog + consumer lag must be exposed as a metrics snapshot."""
  from __future__ import annotations

  from uuid import uuid4

  import pytest

  from solace_events.consumer import ConsumerMetrics
  from solace_events.dead_letter import DeadLetterRecord, DeadLetterStore
  from solace_events.monitoring import EventPlaneMetrics
  from solace_events.schemas import SessionStartedEvent


  @pytest.mark.asyncio
  async def test_collect_reports_dlq_and_lag():
      store = DeadLetterStore()
      ev = SessionStartedEvent(user_id=uuid4(), session_number=1)
      await store.save(
          DeadLetterRecord.from_failed_event(
              ev, original_topic="solace.safety", consumer_group="crisis", error=RuntimeError("x")
          )
      )
      cm = ConsumerMetrics()
      cm.record_failure()
      cm.record_duplicate()
      cm.current_lag = 42

      snap = await EventPlaneMetrics.collect(store, cm)
      assert snap["dlq_unresolved"] == 1
      assert snap["dlq_by_topic"].get("solace.safety.dlq") == 1
      assert snap["consumer_lag"] == 42
      assert snap["messages_failed"] == 1
      assert snap["messages_skipped_duplicate"] == 1
      assert 0.0 <= snap["success_rate"] <= 100.0
  ```
- [ ] Run it — expect FAIL (`No module named 'solace_events.monitoring'`):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_events/test_monitoring.py -o "addopts=" -p no:logging -q
  ```
- [ ] Create `src/solace_events/monitoring.py`:
  ```python
  """Solace-AI Event-Plane Metrics - DLQ backlog + consumer lag snapshot (H-15).

  The DLQ-backlog and consumer-lag signals already exist (DeadLetterHandler /
  DLQStore counts and ConsumerMetrics.current_lag) but nothing reads them, so the
  'lost crisis alert' failure mode is invisible. This aggregates them into one
  snapshot that a service can log or expose for Phase-4 alerting.
  """
  from __future__ import annotations

  from typing import Any, Protocol

  import structlog

  from .consumer import ConsumerMetrics

  logger = structlog.get_logger(__name__)


  class _CountingStore(Protocol):
      async def count_unresolved(self) -> int: ...
      async def count_by_topic(self) -> dict[str, int]: ...


  class EventPlaneMetrics:
      """Aggregates DLQ backlog + consumer lag into a single metrics snapshot."""

      @staticmethod
      async def collect(
          dlq_store: _CountingStore | None,
          consumer_metrics: ConsumerMetrics | None,
      ) -> dict[str, Any]:
          dlq_unresolved = 0
          dlq_by_topic: dict[str, int] = {}
          if dlq_store is not None:
              try:
                  dlq_unresolved = await dlq_store.count_unresolved()
                  dlq_by_topic = await dlq_store.count_by_topic()
              except Exception as e:  # metrics collection must never crash a service
                  logger.error("event_plane_metrics_dlq_error", error=str(e))
          cm = consumer_metrics or ConsumerMetrics()
          snapshot = {
              "dlq_unresolved": dlq_unresolved,
              "dlq_by_topic": dlq_by_topic,
              "consumer_lag": cm.current_lag,
              "messages_failed": cm.messages_failed,
              "messages_skipped_duplicate": cm.messages_skipped_duplicate,
              "messages_processed": cm.messages_processed,
              "success_rate": cm.success_rate,
          }
          logger.info("event_plane_metrics", **snapshot)
          return snapshot
  ```
- [ ] Run the test — expect PASS.
- [ ] (Wiring hook, no new test) Note for the executor: the notification lifespan (Task 7) can periodically call `await EventPlaneMetrics.collect(dlq_store, safety_consumer._consumer.metrics)` from the reaper loop or a small background task; Phase 4 (`BLK-31`) consumes this snapshot to page on `dlq_unresolved > 0` and `consumer_lag` growth. Leave the periodic emitter to Phase 4 to avoid duplicating the alerting task here.
- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_events/test_monitoring.py tests/solace_events/test_dead_letter.py tests/solace_events/test_consumer.py -o "addopts=" -p no:logging -q
  ```

---

## Phase-1 exit gate (run after all tasks)

Run the full crisis-path regression across the shared event lib and the three crisis-path services (no commits):
```powershell
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/solace_events/ services/safety_service/tests/ services/therapy_service/tests/ services/notification-service/tests/ services/orchestrator_service/tests/ tests/deployment/test_compose_kafka_enabled.py -o "addopts=" -p no:logging -q
```
**Exit criteria (per program plan Phase 1):** with a live broker (staging), an automated E2E drives a synthetic CRITICAL case detection → escalation → clinician-notification on real Kafka; the Task-5 fault-injection test proves a transient handler failure does NOT drop the event (reservation released → redelivery reprocessed → DLQ reaper re-drives); therapy-originated crises reach the pipeline (Task 2). The real-Kafka E2E + real-Postgres DLQ/idempotency proof is the staging gate that closes the phase; the unit/contract suites above are the permanent CI regression guard.
