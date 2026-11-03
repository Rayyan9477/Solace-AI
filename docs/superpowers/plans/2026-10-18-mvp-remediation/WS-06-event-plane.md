# WS-06: Durable event plane: outbox, pollers, consumer retry, DLQ and lag signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every MVP service publishes through a durable, service-scoped Postgres outbox drained by its own poller, and every consumer retries in place, dead-letters durably, records idempotency only after success and is re-driven by a reaper, so no crisis event is lost or silently dropped, while outbox backlog, consumer lag and DLQ depth are exported as metrics.

**Architecture:** `solace_events` keeps its publisher, consumer and DLQ building blocks and gains one composition root, `solace_events.service_plane` (IF-05): a per-process `ServiceEventPlane` that owns the service's outbox store, `EventPublisher`, `OutboxPoller`, consumers, `DLQReaper` and metric sampler. The IF-01 bootstrap (WS-04) starts it, and the seven `event_bridge.py` adapters reuse it while keeping their names and keyword arguments. Event tables are created only by Alembic slot `010_event_plane` and are verified, never created, at runtime, because services connect as `solace_app` without DDL rights. Strictness is passed in (`strict=`), so `solace_events` never imports the runtime package. A parity test pins `solace_events._env.is_strict_env` to `is_strict_environment`.

**Covers:** 21 register entries (blocker 5, high 4, medium 7, low 5).

**Waves:**
- W0 phase A2: schema of record, per-service ownership, bridge import roots.
- W1 phase A1: consumer, outbox, DLQ reaper, service plane, bridges, metrics, then the single topic registry with create-if-missing at plane start (WS-06.14, last in the lane). This is one serial lane on the W1 critical path. It must start at W1 entry, because every W1 service lane consumes IF-05.
- W3 phase A: erasure, retention, payload hygiene.
- W5: offset-tracker pruning and closure of MVP-267.

**Depends on:**
- WS-01: the aiokafka and prometheus-client pins. aiokafka itself is installed by the controller at W0 entry into the dedicated venv `D:\tmp\mvp-exec\venv` (D-20; the anaconda interpreter is never touched), and WS-01.2 re-syncs that venv to the dev lock. Also the import-root guard `tests/build/test_import_roots.py` (WS-01.7), whose `WS-06` case WS-06.2 turns green, and the MVP-269 closure check (WS-01.12, W5) that WS-06.14 satisfies.
- WS-02: the slot scaffold `010_event_plane` in the fixed chain (WS-02.3, which WS-06.1 follows in the W0 phase-A2 schema lane), `solace_app` DML grants through the 009 default privileges, the migrate entrypoint, and the requirement that 007 never creates or alters `event_outbox`.
- WS-05: the IF-19 fixtures `pg_migrated` (with `create_database` and `superuser_fetch`) and `kafka_bootstrap` (with `committed_offset`, `pause` and `unpause`), used by every real-infra test here. Also the manifest plugin, used only at barriers.
- WS-04 (for closure, not for this plan's tests): it wires `start_service_event_plane` into the bootstrap in W1 phase A2 and ships the crisis-path alert rules over this plan's metrics.

**Owned paths:**
- `src/solace_events`
- `src/solace_infrastructure/kafka`
- the seven bridge files, `services/{orchestrator_service,safety_service,therapy_service,diagnosis_service,personality_service,user-service,notification-service}/src/infrastructure/event_bridge.py`
- `migrations/versions/010_event_plane.py`
- `tests/solace_events`
- `tests/realinfra/event-plane/` (component tests, per the IF-19 protocol)
- new `test_mvp_014_event_bridge_plane.py` files in the seven services' `tests/` (each tests only this workstream's bridge file)
- one existing case in `tests/solace_infrastructure/kafka/test_topics.py` (WS-06.14), under the shared-file rule that the owner of the code under test edits it

**Spec:** docs/MVP-ISSUE-REGISTER.md (the entries listed per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

### Conventions used by every task

- Unit command for any path P:
  `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest P -o "addopts=" -p no:logging -q`
- Real-infra command: the same, plus `$env:SOLACE_REQUIRE_REALINFRA="1";` before the interpreter, so a missing Docker daemon fails the run instead of skipping it.
- Real-infra test modules start with `pytestmark = [pytest.mark.realinfra, pytest.mark.integration]`. They use only the IF-19 fixtures: `pg_migrated`, `kafka_bootstrap`, `smtp_sink`, `redis_url`, `fake_llm`.
- Isolation inside the shared session containers:
  - every test uses a unique `source_service` (`f"svc-<id>-{uuid4().hex[:6]}"`), consumer group and Kafka topic (`f"test.mvp<id>.{uuid4().hex[:8]}"`);
  - tests that publish to fixed topics filter by a per-test `user_id`.
- Fake pools in unit tests mirror the existing `_RecordingConn`/`_RecordingPool` pattern in `tests/solace_events/test_outbox_claim.py`.
- Test helpers are defined locally at the top of each new test file. A little duplication is accepted, so no task edits another task's test file:
  - `_crisis(user_id=None, trigger_indicators=None)` builds a `CrisisDetectedEvent` (CRITICAL, `["suicidal_ideation"]`, layer 1, confidence 0.9, `escalation_action="page_on_call"`);
  - `async def _append(lst, item): lst.append(item)`, `_always_fail` (an async handler raising `RuntimeError("smtp relay unreachable")`) and `_ok_handler` (an async no-op);
  - `_no_sleep` (an async no-op taking seconds);
  - `_wait_until(pred, timeout)` polls a sync predicate every 20 ms; `_wait_until_async(fn, equals, timeout)` polls an async function until it returns `equals`;
  - `_run_until(consumer, adapter, pred)` starts the consumer, runs `consume_loop` until `pred()` holds (2 s timeout), then stops it;
  - `_record()` / `_outbox_record()` build an `OutboxRecord`;
  - `_dlq(event, group, dlq_entry_at=None)` builds a retriable `DeadLetterRecord`;
  - `_started_publisher(pg, service)` returns a started `EventPublisher` on `PostgresOutboxStore(pg.app_pool, source_service=service)` with a mock producer.
- No test or log line prints message text, emails or secrets. Synthetic data only.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `migrations/versions/010_event_plane.py` | Modify (upgrade/downgrade bodies only) | `event_outbox`, `processed_events`, `event_dead_letter` DDL | WS-06.1 |
| `src/solace_events/errors.py` | Create, then Modify | `EventStoreSchemaError`, IF-05 errors, plane config errors, `safe_error_message` | WS-06.1, WS-06.3, WS-06.12 |
| `src/solace_events/postgres_stores.py` | Modify | Verify-only schema check; per-service, claim-guarded outbox; DLQ columns, claims and counts; transactional save; erasure and retention queries | WS-06.1, WS-06.3, WS-06.5, WS-06.6, WS-06.7, WS-06.10, WS-06.11 |
| `src/solace_events/publisher.py` | Modify | `OutboxRecord.source_service`, `create_publisher(source_service=)`, `publish(conn=)`, claim tokens, release on cancel, poller backoff, background producer connect, strict guard, payload redaction hook | WS-06.1, WS-06.5, WS-06.6, WS-06.7, WS-06.8, WS-06.10, WS-06.12 |
| `src/solace_events/consumer.py` | Modify | In-place retry, durable DLQ, record-after-success idempotency, raw-bytes decoding, per-message isolation, seek-back, lag, offset pruning | WS-06.3, WS-06.4, WS-06.10, WS-06.13 |
| `src/solace_events/dead_letter.py` | Modify | Optional event fields, raw payload and Kafka coordinates, redrive fields, strict guard in `create_dead_letter_handler` | WS-06.3, WS-06.7, WS-06.8 |
| `src/solace_events/reaper.py` | Create | `DLQReaper` with per-group selection and its own redrive budget | WS-06.7 |
| `src/solace_events/_env.py` | Create | `is_strict_env()` mirror of `is_strict_environment` | WS-06.8 |
| `src/solace_events/service_plane.py` | Create, then Modify | IF-05 `ServiceEventPlane` and module functions; metric sampler; topics task; `erase_user_events`; retention task | WS-06.8, WS-06.10, WS-06.14, WS-06.11 |
| `src/solace_events/config.py` | Modify | D-24 retention on `solace.safety` and `solace.users` in the single registry | WS-06.14 |
| `src/solace_events/topics.py` | Create | `plan_topics`, `ensure_topics`, `TopicSpec` | WS-06.14 |
| `src/solace_infrastructure/kafka/topics.py`, `tests/solace_infrastructure/kafka/test_topics.py` | Modify | Legacy `register_solace_topics` derived from `TOPIC_CONFIGS`, without `*.dlq` topics | WS-06.14 |
| `src/solace_events/metrics.py` | Create | The six IF-05 metric objects plus the redaction counter | WS-06.10, WS-06.12 |
| `src/solace_events/retention.py` | Create | Retention windows and the hourly purger | WS-06.11 |
| `src/solace_events/payload_policy.py` | Create | `redact_free_text(event)` | WS-06.12 |
| `src/solace_events/schemas.py` | Modify | `UserErasureCompletedKafkaEvent` added to `EVENT_REGISTRY` | WS-06.11 |
| `src/solace_events/__init__.py` | Modify | Export the new public names | WS-06.3, WS-06.7, WS-06.8, WS-06.14, WS-06.11 |
| seven `services/*/src/infrastructure/event_bridge.py` | Modify | Bare `solace_events` imports and the service name (W0); delegate to the plane (W1) | WS-06.2, WS-06.9 |
| `tests/solace_events/test_outbox_claim.py` | Modify | Claim-SQL argument assertions follow the new parameters | WS-06.1, WS-06.6 |
| `tests/solace_events/test_consumer.py` | Modify | The fixture disables backoff sleeps; the failure test asserts 5 attempts | WS-06.3 |
| `tests/solace_events/test_mvp_*.py` | Create | Unit tests per task | all |
| `tests/realinfra/event-plane/test_mvp_*.py` | Create | Real Postgres/Kafka tests per task | WS-06.1, 06.4, 06.5, 06.6, 06.7, 06.8, 06.10, 06.11, 06.12, 06.14 |
| `services/*/tests/test_mvp_014_event_bridge_plane.py` (seven files) | Create | Bridge behaviour per service | WS-06.9 |

## Tasks

### Task WS-06.1: Event tables come only from slot 010; stores verify the schema and claim per service
**Covers:** MVP-266, MVP-133
**Severity:** medium
**Wave:** 0
**Depends on:** WS-02.3, WS-05.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-01 (default assumed: Alembic is the only schema of record; no service runs DDL at runtime). D-02 (default assumed: services run as solace_app with DML only).
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/010_event_plane.py` (upgrade()/downgrade() bodies only; `revision`/`down_revision` untouched). WS-02.3 creates this file as an empty slot earlier in the same W0 phase-A2 schema lane, and this task runs after it (the `migrations/versions/*` handoff protocol; WS-02 reviews the slot diff).
- Create: `src/solace_events/errors.py` (`EventStoreSchemaError`)
- Modify: `src/solace_events/postgres_stores.py`. Delete `OUTBOX_TABLE_DDL`, `PROCESSED_EVENTS_TABLE_DDL` and `DLQ_TABLE_DDL`. Add `REQUIRED_COLUMNS` and `verify_event_tables`. All three stores' `ensure_table()` delegate to `verify_event_tables`. `PostgresOutboxStore.__init__(pool, *, source_service)` now writes `source_service` on save and filters the claim on it.
- Modify: `src/solace_events/publisher.py`:
  - `OutboxRecord.source_service: str | None = None`;
  - `InMemoryOutboxStore(source_service: str = "local")`, which stamps unset records on save and claims only matching rows;
  - `create_publisher(..., source_service: str | None = None)`, which raises `ValueError` when `postgres_pool` is given without it.
- Modify: `tests/solace_events/test_outbox_claim.py` (`test_postgres_get_pending_issues_atomic_claim_sql`: construct with `source_service="svc-test"`; assert `args == (50, 120.0, "svc-test")`)
- Test: `tests/solace_events/test_mvp_266_verify_only_ensure_table.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_266_slot_010_schema.py` (new)

**Interfaces:**
- Consumes:
  - IF-03 slot `010_event_plane`, scaffolded by WS-02 with `down_revision = "009_platform_roles_rls"`; `solace_app` DML through `ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator` (009);
  - IF-19 `pg_migrated -> PgHandle(dsn_app, dsn_migrator, dsn_superuser, app_pool)`, `await PgHandle.superuser_fetch(sql, *args) -> list[asyncpg.Record]` and `await PgHandle.create_database(name, *, migrate_to="head") -> PgHandle` (WS-05.2).
- Produces:
  - the tables below;
  - `class EventStoreSchemaError(RuntimeError)` with `missing: dict[str, list[str]]` (table to missing columns; a missing table lists all of its columns);
  - `REQUIRED_COLUMNS: dict[str, frozenset[str]]`;
  - `async def verify_event_tables(conn, tables: Iterable[str] = ("event_outbox", "processed_events", "event_dead_letter")) -> None`;
  - `PostgresOutboxStore(pool, *, source_service: str)`.

**Slot 010 DDL.** It is fixed here and used by every later task:
- **`event_outbox`.** Columns:
  - `id UUID PRIMARY KEY`, `source_service VARCHAR(64) NOT NULL`, `event_id UUID NOT NULL`, `event_type VARCHAR(255) NOT NULL`, `event_payload JSONB NOT NULL`, `aggregate_id UUID NOT NULL` (the user id and message key), `topic VARCHAR(255) NOT NULL`, `partition_key VARCHAR(255) NOT NULL`;
  - `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`, `published_at TIMESTAMPTZ`;
  - `status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING','PUBLISHING','PUBLISHED','FAILED'))`;
  - `retry_count INT NOT NULL DEFAULT 0`, `last_error TEXT`, `claimed_at TIMESTAMPTZ`, `claim_token UUID`, `next_retry_at TIMESTAMPTZ`.

  Indexes:
  - `ix_event_outbox_claimable (source_service, created_at) WHERE status IN ('PENDING','PUBLISHING')`;
  - `ix_event_outbox_published_at (published_at) WHERE status = 'PUBLISHED'`;
  - `ix_event_outbox_aggregate (aggregate_id)`.
- **`processed_events`.** `event_id UUID NOT NULL`, `consumer_group VARCHAR(255) NOT NULL`, `processed_at TIMESTAMPTZ NOT NULL DEFAULT now()`, `PRIMARY KEY (event_id, consumer_group)`; index `ix_processed_events_processed_at (processed_at)`.
- **`event_dead_letter`.** Columns:
  - `id UUID PRIMARY KEY`, `consumer_group VARCHAR(255) NOT NULL` (a consumer group, or `outbox:<service>`);
  - `original_topic VARCHAR(255) NOT NULL`, `dlq_topic VARCHAR(255) NOT NULL` (a compatibility label, never a Kafka topic);
  - `kafka_partition INT`, `kafka_offset BIGINT`;
  - `original_event JSONB` (nullable), `raw_payload BYTEA`, `event_type VARCHAR(255)`, `event_id UUID`, `user_id UUID`;
  - `error_message TEXT NOT NULL`, `error_type VARCHAR(255) NOT NULL`, `stack_trace TEXT`;
  - `retry_count INT NOT NULL DEFAULT 0`, `redrive_count INT NOT NULL DEFAULT 0`;
  - `first_failure_at`, `last_failure_at`, `dlq_entry_at` (each `TIMESTAMPTZ NOT NULL DEFAULT now()`), `next_retry_at TIMESTAMPTZ`;
  - `is_retriable BOOLEAN NOT NULL DEFAULT TRUE`, `resolved BOOLEAN NOT NULL DEFAULT FALSE`, `resolved_at TIMESTAMPTZ`, `resolution_notes TEXT`.

  Indexes:
  - `ix_event_dead_letter_redrive (consumer_group, next_retry_at) WHERE NOT resolved AND is_retriable`;
  - `ix_event_dead_letter_unresolved (consumer_group) WHERE NOT resolved`;
  - `ix_event_dead_letter_user (user_id)`;
  - `ix_event_dead_letter_resolved_at (resolved_at) WHERE resolved`.
- No RLS on these tables (IF-03 lists them as non-RLS). `downgrade()` drops the three tables.

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_266_verify_only_ensure_table.py`:
```python
EXPECTED = {
    "event_outbox": {"id", "source_service", "event_id", "event_type", "event_payload", "aggregate_id", "topic",
                     "partition_key", "created_at", "published_at", "status", "retry_count", "last_error",
                     "claimed_at", "claim_token", "next_retry_at"},
    "processed_events": {"event_id", "consumer_group", "processed_at"},
    "event_dead_letter": {"id", "consumer_group", "original_topic", "dlq_topic", "kafka_partition", "kafka_offset",
                          "original_event", "raw_payload", "event_type", "event_id", "user_id", "error_message",
                          "error_type", "stack_trace", "retry_count", "redrive_count", "first_failure_at",
                          "last_failure_at", "dlq_entry_at", "next_retry_at", "is_retriable", "resolved",
                          "resolved_at", "resolution_notes"},
}
# _SchemaConn.fetch answers information_schema.columns queries from a {table: columns} dict and records every SQL text;
# execute/fetchrow record SQL and return "OK"/None. _SchemaPool.acquire() returns the conn.

async def test_mvp_266_dlq_and_ledger_ensure_table_execute_no_ddl():
    conn = _SchemaConn(EXPECTED)
    await PostgresDLQStore(_SchemaPool(conn)).ensure_table()
    await PostgresIdempotencyStore(_SchemaPool(conn)).ensure_table()
    assert not [s for s in conn.sql if "CREATE" in s.upper() or "ALTER" in s.upper()]

async def test_mvp_266_missing_columns_raise_event_store_schema_error():
    from solace_events.errors import EventStoreSchemaError
    partial = {**EXPECTED, "event_dead_letter": EXPECTED["event_dead_letter"] - {"raw_payload", "redrive_count"}}
    with pytest.raises(EventStoreSchemaError) as exc:
        await PostgresDLQStore(_SchemaPool(_SchemaConn(partial))).ensure_table()
    assert sorted(exc.value.missing["event_dead_letter"]) == ["raw_payload", "redrive_count"]

async def test_mvp_133_postgres_claim_filters_on_source_service():
    conn = _RecordingConn()
    await PostgresOutboxStore(_RecordingPool(conn), source_service="svc-a").get_pending(limit=10, stale_after_seconds=60)
    sql, args = conn.calls[-1]
    assert "source_service = $3" in " ".join(sql.split())
    assert args == (10, 60.0, "svc-a")

def test_mvp_133_create_publisher_requires_service_name_with_pool():
    with pytest.raises(ValueError):
        create_publisher(use_mock=True, postgres_pool=_RecordingPool(_RecordingConn()))
    pub = create_publisher(use_mock=True, postgres_pool=_RecordingPool(_RecordingConn()), source_service="svc-a")
    assert pub._outbox_store._source_service == "svc-a"

async def test_mvp_133_in_memory_store_claims_only_its_service():
    store = InMemoryOutboxStore(source_service="svc-a")
    mine, other = _record(), _record()
    other.source_service = "svc-b"
    await store.save(mine); await store.save(other)
    assert [r.id for r in await store.get_pending()] == [mine.id]
```
In `tests/realinfra/event-plane/test_mvp_266_slot_010_schema.py` (with a local copy of `EXPECTED`):
```python
async def test_mvp_266_slot_010_creates_event_tables(pg_migrated):
    rows = await pg_migrated.superuser_fetch(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = ANY($1::text[])", list(EXPECTED))
    got: dict[str, set[str]] = {}
    for r in rows:
        got.setdefault(r["table_name"], set()).add(r["column_name"])
    assert got == EXPECTED

async def test_mvp_266_solace_app_has_dml_but_no_ddl(pg_migrated):
    async with pg_migrated.app_pool.acquire() as conn:
        assert await conn.fetchval("SELECT current_user") == "solace_app"
        rid = uuid4()
        await conn.execute(
            "INSERT INTO event_outbox (id, source_service, event_id, event_type, event_payload, aggregate_id, topic,"
            " partition_key) VALUES ($1, 'svc-266', $2, 'test.event', $3, $4, 'solace.test', 'k')",
            rid, uuid4(), {"k": "v"}, uuid4())
        assert await conn.execute("UPDATE event_outbox SET status = 'PUBLISHED' WHERE id = $1", rid) == "UPDATE 1"
        assert await conn.execute("DELETE FROM event_outbox WHERE id = $1", rid) == "DELETE 1"
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("ALTER TABLE event_outbox ADD COLUMN x INT")

async def test_mvp_266_stores_verify_as_solace_app(pg_migrated):
    for store in (PostgresOutboxStore(pg_migrated.app_pool, source_service="svc-266"),
                  PostgresDLQStore(pg_migrated.app_pool), PostgresIdempotencyStore(pg_migrated.app_pool)):
        await store.ensure_table()

async def test_mvp_266_store_verification_fails_loud_before_slot_010(pg_migrated):
    db = await pg_migrated.create_database(f"ev266_{uuid4().hex[:8]}", migrate_to="009_platform_roles_rls")
    with pytest.raises(EventStoreSchemaError) as exc:
        await PostgresOutboxStore(db.app_pool, source_service="svc-266").ensure_table()
    assert "event_outbox" in exc.value.missing

async def test_mvp_133_each_service_claims_only_its_own_rows(pg_migrated):
    a = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-a-{uuid4().hex[:6]}")
    b = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-b-{uuid4().hex[:6]}")
    rec_a = [_outbox_record() for _ in range(3)]
    rec_b = [_outbox_record() for _ in range(3)]
    for r in rec_a: await a.save(r)
    for r in rec_b: await b.save(r)
    assert {r.id for r in await a.get_pending(limit=100)} == {r.id for r in rec_a}
    rows = await pg_migrated.superuser_fetch("SELECT status FROM event_outbox WHERE id = ANY($1::uuid[])", [r.id for r in rec_b])
    assert sorted(r["status"] for r in rows) == ["PENDING"] * 3
```
- [ ] **Step 2: Run them and confirm they fail.**
  - Run: unit command with `tests/solace_events/test_mvp_266_verify_only_ensure_table.py`.
    Expected:
    - FAIL: `test_mvp_266_dlq_and_ledger_ensure_table_execute_no_ddl` finds `CREATE TABLE IF NOT EXISTS event_dead_letter` in the recorded SQL. That is the startup DDL race the register describes.
    - FAIL: the `EventStoreSchemaError` import raises `ModuleNotFoundError`.
    - FAIL: the claim SQL has no `source_service` filter (the `TypeError` on the `source_service` keyword shows the ownership gap).
  - Run: real-infra command with `tests/realinfra/event-plane/test_mvp_266_slot_010_schema.py`.
    Expected: FAIL. `got == {}` because slot 010 is the empty WS-02 scaffold, so no event table exists after `upgrade head`.
- [ ] **Step 3: Implement.**
  - Write `upgrade()`/`downgrade()` in `010_event_plane.py` with the DDL above. Use `op.execute` with plain `CREATE TABLE`, not `IF NOT EXISTS`, so a pre-existing table fails loudly.
  - Write `async def verify_event_tables(conn, tables=...) -> None`. It runs one `SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = ANY($1::text[])` query and raises `EventStoreSchemaError(missing)` naming tables and columns only. Each store's `ensure_table()` acquires a connection and calls it.
  - Change `PostgresOutboxStore.save` to insert `source_service`.
  - Change `get_pending` to add `AND source_service = $3` inside the claim subquery, so pending, backoff and stale reclaim all stay per service.
- [ ] **Step 4: Run them and confirm they pass.** Run the same two commands. Expected: PASS.
- [ ] **Step 5: Regression.**
  - Unit command with `tests/solace_events`. Expected: all pass.
  - Real-infra command with `tests/migrations`. Expected: WS-02's upgrade head, downgrade base, upgrade head chain test is green with slot 010 filled.
  - Real-infra command with `tests/realinfra/event-plane`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.2: The seven bridges import bare solace_events and stamp their own service on outbox rows
**Covers:** MVP-133
**Severity:** medium
**Wave:** 0
**Depends on:** WS-06.1, WS-01.7
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify the seven bridges. In each file:
  - replace the `try: from src.solace_events... except ImportError:` block with direct `from solace_events.publisher import ...` / `from solace_events.config import ...` imports, so a missing package is an import error in every environment;
  - pass `source_service="<service>"` to `create_publisher`;
  - keep every class, function and keyword argument name.

  The files and service names:
  - `services/orchestrator_service/src/infrastructure/event_bridge.py` ("orchestrator-service")
  - `services/safety_service/src/infrastructure/event_bridge.py` ("safety-service")
  - `services/therapy_service/src/infrastructure/event_bridge.py` ("therapy-service")
  - `services/diagnosis_service/src/infrastructure/event_bridge.py` ("diagnosis-service")
  - `services/personality_service/src/infrastructure/event_bridge.py` ("personality-service")
  - `services/user-service/src/infrastructure/event_bridge.py` ("user-service")
  - `services/notification-service/src/infrastructure/event_bridge.py` ("notification-service")
- Test: `tests/solace_events/test_mvp_133_bridge_wiring_static.py` (new)

**Interfaces:**
- Consumes: `create_publisher(..., source_service=...)` from WS-06.1; the `ServiceIdentity` values in `solace_security.service_auth`.
- Produces: bridge modules importable through the bare `solace_events` root. This is a supporting edit for MVP-270 (owned by WS-01), whose guard `tests/build/test_import_roots.py` (created by WS-01.7, W0 phase A1) registers these seven files as the `WS-06` case. Satisfies request R-01.4(a) for WS-01.7: `test_mvp_270_no_src_root_imports[WS-06]` is run red in Step 2 and green in Step 4.

- [ ] **Step 1: Write the failing test** in `tests/solace_events/test_mvp_133_bridge_wiring_static.py`:
```python
REPO = Path(__file__).resolve().parents[2]
BRIDGES = {
    "services/orchestrator_service/src/infrastructure/event_bridge.py": "orchestrator-service",
    "services/safety_service/src/infrastructure/event_bridge.py": "safety-service",
    "services/therapy_service/src/infrastructure/event_bridge.py": "therapy-service",
    "services/diagnosis_service/src/infrastructure/event_bridge.py": "diagnosis-service",
    "services/personality_service/src/infrastructure/event_bridge.py": "personality-service",
    "services/user-service/src/infrastructure/event_bridge.py": "user-service",
    "services/notification-service/src/infrastructure/event_bridge.py": "notification-service",
}
_SERVICE_ARG = {"create_publisher": "source_service", "get_or_create_service_event_plane": "service_name"}

def _tree(path): return ast.parse((REPO / path).read_text(encoding="utf-8"))

@pytest.mark.parametrize("path", sorted(BRIDGES))
def test_mvp_133_bridge_imports_bare_solace_events(path):
    mods = {n.module for n in ast.walk(_tree(path)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not {m for m in mods if m.startswith("src.solace_events")}
    assert any(m.startswith("solace_events") for m in mods)

@pytest.mark.parametrize("path,service", sorted(BRIDGES.items()))
def test_mvp_133_bridge_names_its_service(path, service):
    values = []
    for c in ast.walk(_tree(path)):
        if isinstance(c, ast.Call):
            name = getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            if name in _SERVICE_ARG:
                values += [k.value.value for k in c.keywords if k.arg == _SERVICE_ARG[name] and isinstance(k.value, ast.Constant)]
                values += [a.value for a in c.args[:1] if isinstance(a, ast.Constant)]
    assert values and set(values) == {service}
```
The test accepts `get_or_create_service_event_plane` as well, so it stays valid after the W1 refactor in WS-06.9.
- [ ] **Step 2: Run it and confirm it fails.**
  - Run: unit command with `tests/solace_events/test_mvp_133_bridge_wiring_static.py`.
  - Expected: FAIL in 14 cases:
    - all seven bridges import `src.solace_events.publisher`;
    - no bridge passes `source_service`.
  - Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_import_roots.py -k "WS-06" -o "addopts=" -p no:logging -q`
  - Expected: FAIL in `test_mvp_270_no_src_root_imports[WS-06]`, listing the 15 `src.solace_events` imports of the seven bridges (for example `services/safety_service/src/infrastructure/event_bridge.py: 25:src.solace_events.schemas`, `26:src.solace_events.publisher`, `27:src.solace_events.config`).
- [ ] **Step 3: Implement.** Edit the seven files as listed. Keep `_KAFKA_AVAILABLE`-dependent branches only where a class body reads it; the import can no longer fail silently, so set the flag to `True`.
- [ ] **Step 4: Run it and confirm it passes.** Run the same two commands. Expected: PASS, including `test_mvp_270_no_src_root_imports[WS-06]`.
- [ ] **Step 5: Regression.** Run one process per suite with the unit command:
  - `tests/solace_events`
  - `services/orchestrator_service/tests`
  - `services/safety_service/tests`
  - `services/therapy_service/tests`
  - `services/diagnosis_service/tests`
  - `services/personality_service/tests`
  - `services/user-service/tests`
  - `services/notification-service/tests`

  Expected: all pass. The existing bridge-wiring tests, for example `services/diagnosis_service/tests/test_event_bridge_wiring.py`, stay green.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.3: Consumer: bounded in-place retry, durable DLQ, malformed messages to the DLQ, idempotency recorded only after success
**Covers:** MVP-007, MVP-134, MVP-138
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-06.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `src/solace_events/errors.py`. Add:
  - `class EventPlaneError(Exception)`;
  - `RetryableEventError`, `NonRetryableEventError` and `EventPublishError`, all subclasses of `EventPlaneError`;
  - `class EventPlaneConfigError(EventPlaneError, RuntimeError)`;
  - `class EventPlaneNotStartedError(EventPlaneError, RuntimeError)`;
  - `class MalformedEventError(NonRetryableEventError)`;
  - `def safe_error_message(exc: BaseException) -> str`.
- Modify: `src/solace_events/consumer.py`:
  - `EventConsumer.__init__` gains keyword-only `dlq_store: DLQStore | None = None`, `max_attempts: int = 5`, `retry_base_seconds: float = 1.0`, `retry_max_seconds: float = 16.0` and `handler_timeout_seconds: float = 30.0`;
  - add a `self._sleep = asyncio.sleep` seam;
  - rewrite `_process_message(topic, partition, offset, value: dict | bytes) -> ProcessingResult`;
  - `create_consumer(..., dlq_store=None, max_attempts=5, retry_base_seconds: float | None = None)`, where `None` means 0.0 with `use_mock=True` and 1.0 otherwise.
- Modify: `src/solace_events/dead_letter.py`:
  - `DeadLetterRecord.event_type/event_id/user_id` become optional;
  - new fields `raw_payload: bytes | None = None`, `kafka_partition: int | None = None`, `kafka_offset: int | None = None`, `redrive_count: int = 0` and `resolved_at: datetime | None = None`;
  - new `@classmethod from_consumer_failure(cls, *, consumer_group, topic, partition, offset, value, error, attempts, retriable, event=None) -> DeadLetterRecord`.
- Modify: `src/solace_events/postgres_stores.py` (`PostgresDLQStore.save` and `_row_to_record` read and write the new columns)
- Modify: `src/solace_events/__init__.py` (export the errors, `InMemoryIdempotencyStore`, `PostgresIdempotencyStore` and `IdempotencyStore`)
- Modify: `tests/solace_events/test_consumer.py`. The `TestEventConsumer.consumer` fixture becomes `EventConsumer(adapter, retry_base_seconds=0.0)`. `test_process_message_failure` additionally asserts the failing handler ran 5 times.
- Test: `tests/solace_events/test_mvp_007_consumer_retry_dlq.py` (new)
- Test: `tests/solace_events/test_mvp_134_malformed_to_dlq.py` (new)
- Test: `tests/solace_events/test_mvp_138_idempotency_after_success.py` (new)

**Interfaces:**
- Consumes: IF-05 semantics as frozen at W1 entry.
- Produces the `ProcessingStatus` meanings:
  - `SUCCESS`: handled.
  - `SKIP`: no handler, unknown type or duplicate. Committable.
  - `FAILED`: durably dead-lettered. Committable.
  - `RETRY`: the DLQ write failed. Not committable; WS-06.4 seeks back.

  Retry schedule:
  - Backoff after failed attempt *n* is `min(retry_base_seconds * 2 ** (n - 1), retry_max_seconds)`: 1, 2, 4, 8 s for 5 attempts.
  - `RetryableEventError` and any other `Exception` are retried in place. Each attempt is bounded by `asyncio.wait_for(..., handler_timeout_seconds)`.
  - `NonRetryableEventError`, invalid JSON, non-dict JSON, a missing `event_type` and `ValidationError` go to the DLQ immediately, with `is_retriable=False`.
  - Blocking bound (crisis-review F1, a correction of IF-05's "about 15 s of partition blocking at most", which counts only the backoff): one message holds its partition for the backoff sum (1 + 2 + 4 + 8 = 15 s at the defaults) plus up to `max_attempts * handler_timeout_seconds` of handler time (5 x 30 s = 150 s worst case), so about 165 s at most. From WS-06.4 the hold applies to that message's partition only, and WS-13.6 keeps crisis delivery retries out of this path (its handler returns once the episode is committed).

  Idempotency order:
  1. `was_processed` is checked before the handler.
  2. `mark_processed` runs only after success, with 3 quick retries.
  3. A ledger error never stops the handler from running and never causes a commit without a successful run.

  `safe_error_message(exc)`:
  - for a pydantic `ValidationError`: `"ValidationError: <n> error(s) at <loc>[, <loc>]"`, without input values;
  - otherwise: `f"{type(exc).__name__}: {str(exc)[:200]}"`.

  DLQ records never store traceback text (`stack_trace=None`).

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_007_consumer_retry_dlq.py` (`_crisis()` builds a `CrisisDetectedEvent` with `crisis_level=CrisisLevel.CRITICAL`, `trigger_indicators=["suicidal_ideation"]`, `detection_layer=1`, `confidence=Decimal("0.9")`, `escalation_action="page_on_call"`; `_no_sleep` is `async def _no_sleep(_s): return None`):
```python
async def test_mvp_007_failed_handler_is_retried_in_place_until_success():
    calls = []
    async def handler(event):
        calls.append(event.metadata.event_id)
        if len(calls) < 3:
            raise RuntimeError("smtp relay unreachable")
    consumer = EventConsumer(MockKafkaConsumerAdapter())
    consumer._sleep = _no_sleep
    consumer.register_handler("safety.crisis.detected", handler)
    result = await consumer._process_message("solace.safety", 0, 7, _crisis().to_dict())
    assert result.status == ProcessingStatus.SUCCESS
    assert len(calls) == 3

async def test_mvp_007_backoff_doubles_from_one_second_capped_at_sixteen():
    slept = []
    async def record(s): slept.append(s)
    consumer = EventConsumer(MockKafkaConsumerAdapter(), max_attempts=6, dlq_store=DeadLetterStore())
    consumer._sleep = record
    consumer.register_handler("safety.crisis.detected", _always_fail)
    await consumer._process_message("solace.safety", 0, 7, _crisis().to_dict())
    assert slept == [1.0, 2.0, 4.0, 8.0, 16.0]

async def test_mvp_007_exhausted_attempts_dead_letter_durably():
    dlq = DeadLetterStore()
    attempts = []
    async def failing(event):
        attempts.append(1); raise RuntimeError("smtp relay unreachable")
    ev = _crisis()
    consumer = EventConsumer(MockKafkaConsumerAdapter(), dlq_store=dlq, consumer_group="g-007", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", failing)
    result = await consumer._process_message("solace.safety", 0, 7, ev.to_dict())
    assert len(attempts) == 5 and result.status == ProcessingStatus.FAILED
    (rec,) = await dlq.get_by_topic("solace.safety.dlq")
    assert (rec.event_id, rec.consumer_group, rec.retry_count, rec.kafka_offset, rec.is_retriable) == (ev.metadata.event_id, "g-007", 5, 7, True)
    assert rec.error_type == "RuntimeError" and rec.stack_trace is None

async def test_mvp_007_dlq_write_failure_is_not_committable():
    class _DownDLQ:
        async def save(self, record): raise ConnectionError("db down")
    consumer = EventConsumer(MockKafkaConsumerAdapter(), dlq_store=_DownDLQ(), consumer_group="g-007", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", _always_fail)
    assert (await consumer._process_message("solace.safety", 0, 7, _crisis().to_dict())).status == ProcessingStatus.RETRY

async def test_mvp_007_non_retryable_error_dead_letters_without_retry():
    attempts = []
    async def reject(event):
        attempts.append(1); raise NonRetryableEventError("invalid business key")
    dlq = DeadLetterStore()
    consumer = EventConsumer(MockKafkaConsumerAdapter(), dlq_store=dlq, consumer_group="g-007", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", reject)
    assert (await consumer._process_message("solace.safety", 0, 7, _crisis().to_dict())).status == ProcessingStatus.FAILED
    assert len(attempts) == 1
    assert (await dlq.get_by_topic("solace.safety.dlq"))[0].is_retriable is False

async def test_mvp_007_consume_loop_commits_only_after_success():
    adapter = MockKafkaConsumerAdapter()
    adapter.add_message("solace.safety", 0, 0, _crisis().to_dict())
    calls = []
    async def flaky(event):
        calls.append(1)
        if len(calls) < 3: raise RuntimeError("smtp relay unreachable")
    consumer = EventConsumer(adapter)
    consumer._sleep = _no_sleep
    consumer.register_handler("safety.crisis.detected", flaky)
    await consumer.start(["solace.safety"])
    task = asyncio.create_task(consumer.consume_loop())
    await _wait_until(lambda: adapter.get_committed_offsets() == {("solace.safety", 0): 1}, timeout=2.0)
    await consumer.stop(); task.cancel()
    assert len(calls) == 3
```
In `tests/solace_events/test_mvp_134_malformed_to_dlq.py`:
```python
async def test_mvp_134_schema_invalid_event_reaches_the_dead_letter_handler():
    captured = []
    async def capture(event, error): captured.append(error)
    bad = _crisis().to_dict(); bad["detection_layer"] = 5          # register repro: le=4
    consumer = EventConsumer(MockKafkaConsumerAdapter())
    consumer.register_handler("safety.crisis.detected", _ok_handler)
    consumer.set_dead_letter_handler(capture)
    await consumer._process_message("solace.safety", 1, 42, bad)
    assert len(captured) == 1

async def test_mvp_134_schema_invalid_event_is_stored_with_payload_and_no_input_values():
    dlq = DeadLetterStore()
    bad = _crisis().to_dict(); bad["detection_layer"] = 5
    consumer = EventConsumer(MockKafkaConsumerAdapter(), dlq_store=dlq, consumer_group="g-134", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", _ok_handler)
    result = await consumer._process_message("solace.safety", 1, 42, bad)
    (rec,) = await dlq.get_by_topic("solace.safety.dlq")
    assert result.status == ProcessingStatus.FAILED
    assert (rec.original_event, rec.event_type, rec.error_type, rec.kafka_partition, rec.kafka_offset, rec.is_retriable) == \
           (bad, "safety.crisis.detected", "ValidationError", 1, 42, False)
    assert "detection_layer" in rec.error_message and "input_value" not in rec.error_message

@pytest.mark.parametrize("value,expected_raw,expected_error", [
    (b"\xff\x00not-json", b"\xff\x00not-json", "MalformedEventError"),
    (b"[1, 2]", b"[1, 2]", "MalformedEventError"),
])
async def test_mvp_134_undecodable_bytes_are_stored_raw(value, expected_raw, expected_error):
    dlq = DeadLetterStore()
    consumer = EventConsumer(MockKafkaConsumerAdapter(), dlq_store=dlq, consumer_group="g-134", retry_base_seconds=0.0)
    result = await consumer._process_message("solace.safety", 0, 3, value)
    (rec,) = await dlq.get_by_topic("solace.safety.dlq")
    assert result.status == ProcessingStatus.FAILED
    assert (rec.raw_payload, rec.original_event, rec.error_type, rec.event_id) == (expected_raw, None, expected_error, None)

async def test_mvp_134_missing_event_type_is_dead_lettered():
    dlq = DeadLetterStore()
    consumer = EventConsumer(MockKafkaConsumerAdapter(), dlq_store=dlq, consumer_group="g-134", retry_base_seconds=0.0)
    await consumer._process_message("solace.safety", 0, 4, {"user_id": str(uuid4())})
    (rec,) = await dlq.get_by_topic("solace.safety.dlq")
    assert rec.event_type is None and rec.error_type == "MalformedEventError"
```
In `tests/solace_events/test_mvp_138_idempotency_after_success.py`:
```python
class _FlakyLedger(InMemoryIdempotencyStore):
    def __init__(self):
        super().__init__(); self.fail_next = {"was_processed": 1, "mark_processed": 1}
    async def was_processed(self, event_id, group):
        if self.fail_next["was_processed"]: self.fail_next["was_processed"] -= 1; raise ConnectionError("pool exhausted")
        return await super().was_processed(event_id, group)
    async def mark_processed(self, event_id, group):
        if self.fail_next["mark_processed"]: self.fail_next["mark_processed"] -= 1; raise ConnectionError("pool exhausted")
        return await super().mark_processed(event_id, group)

async def test_mvp_138_ledger_errors_never_drop_the_event():
    handled = []
    consumer = EventConsumer(MockKafkaConsumerAdapter(), idempotency_store=_FlakyLedger(), consumer_group="g-138")
    consumer._sleep = _no_sleep
    consumer.register_handler("safety.crisis.detected", lambda e: _append(handled, e))
    result = await consumer._process_message("solace.safety", 0, 1, _crisis().to_dict())
    assert result.status == ProcessingStatus.SUCCESS and len(handled) == 1

async def test_mvp_138_ledger_is_written_only_after_success():
    store, handled, state = InMemoryIdempotencyStore(), [], {"fail": True}
    async def handler(event):
        if state["fail"]: raise RuntimeError("smtp relay unreachable")
        handled.append(event.metadata.event_id)
    ev = _crisis()
    consumer = EventConsumer(MockKafkaConsumerAdapter(), idempotency_store=store, consumer_group="g-138")
    consumer._sleep = _no_sleep
    consumer.register_handler("safety.crisis.detected", handler)
    await consumer._process_message("solace.safety", 0, 1, ev.to_dict())
    assert await store.was_processed(ev.metadata.event_id, "g-138") is False
    state["fail"] = False
    second = await consumer._process_message("solace.safety", 0, 2, ev.to_dict())
    assert second.status == ProcessingStatus.SUCCESS and handled == [ev.metadata.event_id]
    assert await store.was_processed(ev.metadata.event_id, "g-138") is True

async def test_mvp_138_duplicate_after_success_is_skipped():
    store, handled = InMemoryIdempotencyStore(), []
    consumer = EventConsumer(MockKafkaConsumerAdapter(), idempotency_store=store, consumer_group="g-138")
    consumer.register_handler("safety.crisis.detected", lambda e: _append(handled, e))
    payload = _crisis().to_dict()
    assert (await consumer._process_message("solace.safety", 0, 1, payload)).status == ProcessingStatus.SUCCESS
    assert (await consumer._process_message("solace.safety", 0, 2, payload)).status == ProcessingStatus.SKIP
    assert len(handled) == 1
```
- [ ] **Step 2: Run them and confirm they fail.**
  - Run: unit command with `tests/solace_events/test_mvp_007_consumer_retry_dlq.py`. Expected: FAIL.
    - The first test sees `len(calls) == 1` and `FAILED`: a failed crisis handler is consumed once and never retried (MVP-007).
    - `test_mvp_007_consume_loop_commits_only_after_success` sees `len(calls) == 1` with offset 1 committed.
    - The others raise `TypeError` on `dlq_store` / `max_attempts`.
  - Run: unit command with `tests/solace_events/test_mvp_134_malformed_to_dlq.py`. Expected: FAIL. `captured == []`: the dead-letter handler re-deserializes, raises and is swallowed, so nothing is captured (MVP-134).
  - Run: unit command with `tests/solace_events/test_mvp_138_idempotency_after_success.py`. Expected: FAIL.
    - The first test gets `FAILED` with zero handler calls: the ledger error before the handler drops the event (MVP-138).
    - The second test finds the ledger row present after a failure, and the redelivery is `SKIP`ped.
- [ ] **Step 3: Implement.** `_process_message` runs these stages:
  1. **Decode.** `bytes` goes through `json.loads` on UTF-8. A decode failure or a non-dict value is a `MalformedEventError`, sent to the DLQ with `raw_payload`.
  2. **Validate.** A missing or non-str `event_type` is a `MalformedEventError`. A registry type that fails validation (`ValidationError`) goes to the DLQ with `original_event`. An unknown type is `SKIP`.
  3. **Select handlers.** If none is registered, return `SKIP` with no ledger write.
  4. **Ledger check.** `was_processed`, retried in place on error. If the ledger is still down, proceed with a WARNING and count the event as not processed.
  5. **Attempt loop.** Each attempt is bounded by `handler_timeout_seconds`, with backoff between attempts.
  6. **Ledger write.** `mark_processed` after success.
  7. **On exhaustion or `NonRetryable`.** Build `DeadLetterRecord.from_consumer_failure(...)` and save it to the DLQ store (in-memory `DeadLetterStore` when none is given). Then call the legacy `_dead_letter_handler(event_or_None, safe_message)` when one is set, which keeps `services/notification-service/tests/test_crisis_consumer.py` green. Return `FAILED`; if the store save raises, return `RETRY`.

  `_get_fallback_event_id` must accept any value type.
- [ ] **Step 4: Run them and confirm they pass.** Run the same three commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process per suite with the unit command:
  - `tests/solace_events`
  - `services/notification-service/tests`
  - `services/analytics-service/tests`

  Expected: all pass. These are the legacy `create_consumer` callers.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.4: Consume loop: per-message isolation, raw-bytes decoding, seek-back on uncommittable messages, commit-failure recovery, partitions processed concurrently, proven on real Kafka
**Covers:** MVP-007, MVP-135
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-06.3, WS-05.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-20 (decided 2026-10-19: aiokafka at its D-16 lock pin is installed at W0 entry by the controller into the dedicated venv `D:\tmp\mvp-exec\venv`, created with `--system-site-packages` over the anaconda interpreter, which stays untouched; every command here already runs that venv's interpreter. The W0 barrier also runs the suites in the clean lock-only venv `D:\tmp\mvp-exec\venv-clean` (WS-01.2), so this task's unit tests must pass there too.)
**Real infra:** yes
**Files:**
- Modify: `src/solace_events/consumer.py`:
  - `KafkaConsumerAdapter.seek(topic, partition, offset)` is a concrete method that raises `NotImplementedError`, implemented by both adapters;
  - `KafkaConsumerAdapter.pause(topic, partition)` and `resume(topic, partition)` are concrete methods that raise `NotImplementedError`, implemented by both adapters (AIOKafka: `AIOKafkaConsumer.pause`/`resume` on the `TopicPartition`; Mock: a paused partition's messages are not returned by `poll` until it is resumed);
  - `AIOKafkaConsumerAdapter.start` drops `value_deserializer`, so values arrive as raw bytes, and `poll` passes `max_records`;
  - `MockKafkaConsumerAdapter.seek` rewinds its poll index to that message;
  - `EventConsumer.__init__` gains keyword-only `max_inflight_partitions: int = 16`;
  - `consume_loop` is rewritten.
- Test: `tests/solace_events/test_mvp_135_message_isolation.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_007_kafka_redelivery.py` (new)

**Interfaces:**
- Consumes: `ProcessingStatus.RETRY` semantics (WS-06.3); IF-19 `kafka_bootstrap -> KafkaHandle` with `bootstrap`, `await produce(topic, value: BaseEvent | dict | bytes, *, key=None)`, `await committed_offset(group_id, topic, partition=0) -> int | None` (WS-05.2); `pg_migrated`.
- Produces: `consume_loop` guarantees:
  - every message is processed and committed on its own;
  - partitions are processed concurrently (crisis-review F1): each partition with polled messages gets one worker task that handles them in offset order while that partition's fetch is paused, and the loop keeps polling the other partitions, so one partition's in-place retries (WS-06.3's bound, up to about 165 s) never delay a message on another partition; at most `max_inflight_partitions` workers run at once; order within a partition is kept;
  - a `RETRY` outcome seeks the partition back to that offset and skips the rest of that partition's batch;
  - a commit error (`aiokafka.errors.CommitFailedError` or any adapter commit exception) stops that partition's worker, seeks the partition back to its first uncommitted offset and resumes it, so a rebalance causes redelivery, never loss;
  - framework exceptions are contained per message;
  - `stop()` lets in-flight workers finish their current message and commits nothing after it.

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_135_message_isolation.py`:
```python
async def test_mvp_135_list_payload_does_not_abort_the_batch():
    adapter = MockKafkaConsumerAdapter()
    events = [_crisis() for _ in range(3)]
    adapter.add_message("solace.safety", 0, 0, events[0].to_dict())
    adapter.add_message("solace.safety", 0, 1, [1, 2])                       # register case B
    adapter.add_message("solace.safety", 0, 2, events[1].to_dict())
    adapter.add_message("solace.safety", 0, 3, events[2].to_dict())
    handled = []
    consumer = EventConsumer(adapter, dlq_store=DeadLetterStore(), consumer_group="g-135", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", lambda e: _append(handled, e.metadata.event_id))
    await _run_until(consumer, adapter, lambda: adapter.get_committed_offsets() == {("solace.safety", 0): 4})
    assert handled == [e.metadata.event_id for e in events]

async def test_mvp_135_commit_failure_redelivers_without_loss():
    adapter = _CommitFailsOnceAdapter(fail_at_offset=2)   # Mock subclass: commit raises CommitFailedError once, then rewinds to the last good commit
    events = [_crisis() for _ in range(6)]
    for i, e in enumerate(events):
        adapter.add_message("solace.safety", 0, i, e.to_dict())
    handled = []
    consumer = EventConsumer(adapter, idempotency_store=InMemoryIdempotencyStore(), consumer_group="g-135", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", lambda e: _append(handled, e.metadata.event_id))
    await _run_until(consumer, adapter, lambda: adapter.get_committed_offsets() == {("solace.safety", 0): 6})
    assert sorted(handled) == sorted(e.metadata.event_id for e in events)

async def test_mvp_007_uncommittable_message_is_redelivered_not_skipped():
    adapter = MockKafkaConsumerAdapter()
    first, second = _crisis(), _crisis()
    adapter.add_message("solace.safety", 0, 0, first.to_dict())
    adapter.add_message("solace.safety", 0, 1, second.to_dict())
    dlq = _DLQFailsFirstSave()                          # first save raises ConnectionError, later saves stored
    seen = []
    async def handler(event):
        seen.append(event.metadata.event_id)
        if event.metadata.event_id == first.metadata.event_id: raise RuntimeError("smtp relay unreachable")
    consumer = EventConsumer(adapter, dlq_store=dlq, consumer_group="g-007", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", handler)
    await _run_until(consumer, adapter, lambda: adapter.get_committed_offsets() == {("solace.safety", 0): 2})
    assert seen.count(first.metadata.event_id) == 10 and seen[-1] == second.metadata.event_id
    assert len(dlq.saved) == 1

async def test_crisis_review_f1_a_retrying_partition_does_not_delay_another_partition():
    adapter = MockKafkaConsumerAdapter()
    stuck, other = _crisis(), _crisis()
    adapter.add_message("solace.safety", 0, 0, stuck.to_dict())
    adapter.add_message("solace.safety", 1, 0, other.to_dict())
    release, handled = asyncio.Event(), []
    async def handler(event):
        if event.metadata.event_id == stuck.metadata.event_id:
            await release.wait()                        # one user's page still being retried
        handled.append(event.metadata.event_id)
    consumer = EventConsumer(adapter, consumer_group="g-f1", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", handler)
    await consumer.start(["solace.safety"])
    task = asyncio.create_task(consumer.consume_loop())
    try:
        await _wait_until(lambda: adapter.get_committed_offsets().get(("solace.safety", 1)) == 1, timeout=1.0)
        assert handled == [other.metadata.event_id] and ("solace.safety", 0) not in adapter.get_committed_offsets()
        release.set()
        await _wait_until(lambda: adapter.get_committed_offsets().get(("solace.safety", 0)) == 1, timeout=1.0)
    finally:
        release.set(); await consumer.stop(); task.cancel()

async def test_crisis_review_f1_messages_of_one_partition_keep_their_order():
    adapter = MockKafkaConsumerAdapter()
    events = [_crisis() for _ in range(3)]
    for i, e in enumerate(events):
        adapter.add_message("solace.safety", 0, i, e.to_dict())
    adapter.add_message("solace.safety", 1, 0, _crisis().to_dict())
    order = []
    async def handler(event):
        await asyncio.sleep(0.02); order.append(event.metadata.event_id)
    consumer = EventConsumer(adapter, consumer_group="g-f1o", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", handler)
    await _run_until(consumer, adapter, lambda: adapter.get_committed_offsets() == {("solace.safety", 0): 3, ("solace.safety", 1): 1})
    assert [x for x in order if x in {e.metadata.event_id for e in events}] == [e.metadata.event_id for e in events]
```
In `tests/realinfra/event-plane/test_mvp_007_kafka_redelivery.py`:
```python
async def test_mvp_007_failed_handler_is_not_committed_and_is_redelivered_after_restart(kafka_bootstrap, pg_migrated):
    topic, group, ev = f"test.mvp007.{uuid4().hex[:8]}", f"g-007-{uuid4().hex[:8]}", _crisis()
    await kafka_bootstrap.produce(topic, ev)
    first = create_consumer(group, kafka_settings=KafkaSettings(bootstrap_servers=kafka_bootstrap.bootstrap),
                            dlq_store=PostgresDLQStore(pg_migrated.app_pool), retry_base_seconds=1.0)
    attempted = asyncio.Event()
    async def failing(event):
        attempted.set(); raise RuntimeError("smtp relay unreachable")
    first.register_handler("safety.crisis.detected", failing)
    await first.start([topic]); loop_task = asyncio.create_task(first.consume_loop())
    await asyncio.wait_for(attempted.wait(), 30)
    loop_task.cancel(); await first.stop()                                 # crash mid-retry
    assert await kafka_bootstrap.committed_offset(group, topic) is None
    handled = []
    second = create_consumer(group, kafka_settings=KafkaSettings(bootstrap_servers=kafka_bootstrap.bootstrap),
                             dlq_store=PostgresDLQStore(pg_migrated.app_pool))
    second.register_handler("safety.crisis.detected", lambda e: _append(handled, e.metadata.event_id))
    await second.start([topic]); task2 = asyncio.create_task(second.consume_loop())
    await _wait_until_async(lambda: kafka_bootstrap.committed_offset(group, topic), equals=1, timeout=45)
    task2.cancel(); await second.stop()
    assert handled == [ev.metadata.event_id]

async def test_mvp_007_dead_letter_row_is_durable_before_commit(kafka_bootstrap, pg_migrated):
    # handler always raises; max_attempts=5, retry_base_seconds=0.2
    # within 30 s exactly one event_dead_letter row exists (superuser_fetch WHERE consumer_group = $1)
    #   with retry_count = 5 and event_id = ev.metadata.event_id
    # committed_offset == 1 is observed only after that row exists
    ...

async def test_mvp_134_poison_bytes_go_to_the_dlq_and_the_partition_moves_on(kafka_bootstrap, pg_migrated):
    # produce b"\x00\x01garbage" then a valid crisis event
    # DLQ row raw_payload == b"\x00\x01garbage"; the valid event is handled; committed_offset == 2
    ...

async def test_crisis_review_f1_retries_on_one_partition_do_not_block_another_on_kafka(kafka_bootstrap, pg_migrated):
    # the test creates a two-partition topic with AIOKafkaAdminClient.create_topics and produces two crisis events
    # with keys chosen through aiokafka's default partitioner so they land on partitions 0 and 1;
    # the partition-0 event's handler always raises (max_attempts=5, retry_base_seconds=1.0: about 15 s of retries)
    # the partition-1 event is handled and committed (committed_offset(group, topic, partition=1) == 1) within 5 s,
    # while partition 0 is still uncommitted; partition 0 is committed after its DLQ row exists
    ...
```
- [ ] **Step 2: Run them and confirm they fail.**
  - Run: unit command with `tests/solace_events/test_mvp_135_message_isolation.py`. Expected: FAIL.
    - The list payload raises from `_get_fallback_event_id` inside the outer batch try, so only 1 event is handled and the commit stalls at 1. This is register case B.
    - The commit-failure case leapfrogs the commit and handles 2 of 6 (case A).
    - The redelivery test sees offset 1 committed after offset 0's `RETRY`, because the loop moves past it.
    - `test_crisis_review_f1_a_retrying_partition_does_not_delay_another_partition` times out: today's loop handles the batch one message at a time across all partitions, so the held partition-0 handler keeps partition 1 uncommitted (crisis-review F1). The ordering test may already pass; it guards the rewrite.
  - Run: real-infra command with `tests/realinfra/event-plane/test_mvp_007_kafka_redelivery.py`. Expected: FAIL.
    - `AIOKafkaConsumerAdapter` deserializes inside `getmany`, so the poison message raises out of every poll and the partition never moves.
    - The redelivery assertions fail until seek-back exists.
    - The two-partition test sees partition 1 committed only after partition 0's retries end.
- [ ] **Step 3: Implement.**
  - Rewrite `consume_loop` as a poll loop that hands each polled partition's messages to a per-partition worker: the partition is paused (`pause(topic, partition)`) while its worker runs and resumed when the worker ends, and the loop keeps polling the other partitions. At most `max_inflight_partitions` workers run; when that many are busy the loop waits for one to end before polling again.
  - Each worker is the per-message loop: messages in offset order, `offset + 1` committed after each committable outcome.
  - After a `RETRY`, the worker seeks the partition back to that offset once and drops the rest of its batch.
  - On a commit exception the worker logs the event name only (no payload), seeks its partition back to its first uncommitted offset, stops and resumes the partition; the loop keeps polling.
  - Keep `_maybe_commit_periodic`; `stop()` cancels polling, waits for each worker's current message, then stops the adapter.
- [ ] **Step 4: Run them and confirm they pass.** Run the same two commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_events`;
  - unit command with `services/notification-service/tests`;
  - real-infra command with `tests/realinfra/event-plane`.

  Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.5: Transactional outbox publish(conn=) and JSONB parameters passed as objects
**Covers:** MVP-132, MVP-267
**Severity:** medium
**Wave:** 1
**Depends on:** WS-06.3
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Modify: `src/solace_events/publisher.py`:
  - protocol `OutboxStore.save(record, *, conn=None)`;
  - `EventPublisher.publish(event, topic=None, *, conn: asyncpg.Connection | None = None) -> UUID`;
  - `_publish_via_outbox(event, topic, conn)` wraps store errors in `EventPublishError` (cause chained, message without payload);
  - `InMemoryOutboxStore.save(record, *, conn=None)` raises `EventPublishError("transactional publish requires the Postgres outbox")` when `conn` is not None.
- Modify: `src/solace_events/postgres_stores.py`:
  - `PostgresOutboxStore.save(record, *, conn=None)` runs on `conn` when given, without acquiring;
  - `event_payload` (outbox) and `original_event` (DLQ) are bound as Python dicts. The pool codec serializes them (IF-03); `json.dumps` is removed. Readers still accept a `str` for legacy rows.
- Test: `tests/solace_events/test_mvp_132_transactional_publish.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_132_outbox_same_transaction.py` (new)

**Interfaces:**
- Consumes: the IF-03 pool codec (`ConnectionPoolManager` pools serialize JSONB parameters); `pg_migrated.app_pool` built with that codec (WS-05.2).
- Produces: `EventPublisher.publish(event, topic=None, *, conn=None) -> UUID`. With `conn`, the outbox row joins the caller's transaction; WS-12's escalation path uses it (D-28), as do WS-14's `user.deleted` (IF-16) and `ServiceEventPlane.publish(event, *, conn=None)` (WS-06.8).

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_132_transactional_publish.py`. `_SchemaRecordingPool` answers the WS-06.1 schema check with full columns and records every SQL statement:
```python
async def test_mvp_132_publish_writes_on_the_callers_connection():
    pool_conn, caller_conn = _SchemaRecordingConn(), _SchemaRecordingConn()
    store = PostgresOutboxStore(_SchemaRecordingPool(pool_conn), source_service="svc-132")
    pub = EventPublisher(MockKafkaProducerAdapter(), store, use_outbox=True)
    await pub.start()
    await pub.publish(_crisis(), conn=caller_conn)
    assert any(s.lstrip().startswith("INSERT INTO event_outbox") for s, _ in caller_conn.calls)
    assert not any(s.lstrip().startswith("INSERT INTO event_outbox") for s, _ in pool_conn.calls)

async def test_mvp_267_outbox_payload_is_bound_as_a_dict():
    conn = _SchemaRecordingConn()
    store = PostgresOutboxStore(_SchemaRecordingPool(conn), source_service="svc-267")
    await store.save(OutboxRecord.from_event(_crisis()))
    _, args = next((s, a) for s, a in conn.calls if s.lstrip().startswith("INSERT INTO event_outbox"))
    assert any(isinstance(a, dict) and a.get("event_type") == "safety.crisis.detected" for a in args)

async def test_mvp_132_in_memory_outbox_refuses_transactional_publish():
    pub = EventPublisher(MockKafkaProducerAdapter(), InMemoryOutboxStore(), use_outbox=True)
    await pub.start()
    with pytest.raises(EventPublishError):
        await pub.publish(_crisis(), conn=object())
```
In `tests/realinfra/event-plane/test_mvp_132_outbox_same_transaction.py`:
```python
async def test_mvp_132_outbox_row_rolls_back_with_the_domain_row(pg_migrated, scratch_table):
    # scratch_table: superuser creates "test_domain_132(id uuid primary key)" and grants DML to solace_app; dropped after
    pub = await _started_publisher(pg_migrated, "svc-132")
    rid, ev = uuid4(), _crisis()
    async with pg_migrated.app_pool.acquire() as conn:
        with pytest.raises(RuntimeError):
            async with conn.transaction():
                await conn.execute("INSERT INTO test_domain_132 (id) VALUES ($1)", rid)
                await pub.publish(ev, conn=conn)
                raise RuntimeError("abort after both writes")
    assert await _count(pg_migrated, "test_domain_132", "id", rid) == 0
    assert await _count(pg_migrated, "event_outbox", "event_id", ev.metadata.event_id) == 0

async def test_mvp_132_outbox_row_commits_with_the_domain_row(pg_migrated, scratch_table):
    # same transaction without the raise -> both counts == 1
    ...

async def test_mvp_267_event_payloads_are_stored_as_jsonb_objects(pg_migrated):
    pub = await _started_publisher(pg_migrated, "svc-267"); ev = _crisis()
    await pub.publish(ev)
    (row,) = await pg_migrated.superuser_fetch("SELECT jsonb_typeof(event_payload) AS t FROM event_outbox WHERE event_id = $1", ev.metadata.event_id)
    assert row["t"] == "object"
    await PostgresDLQStore(pg_migrated.app_pool).save(_dlq_record_for(ev))
    (row,) = await pg_migrated.superuser_fetch("SELECT jsonb_typeof(original_event) AS t FROM event_dead_letter WHERE event_id = $1", ev.metadata.event_id)
    assert row["t"] == "object"
```
- [ ] **Step 2: Run them and confirm they fail.**
  - Run: unit command with `tests/solace_events/test_mvp_132_transactional_publish.py`.
    Expected: FAIL.
    - `TypeError: publish() got an unexpected keyword argument 'conn'`. The outbox has no transaction seam (MVP-132).
    - The payload argument is a `str` produced by `json.dumps` (MVP-267).
  - Run: real-infra command with `tests/realinfra/event-plane/test_mvp_132_outbox_same_transaction.py`.
    Expected: FAIL.
    - The `TypeError` above.
    - `jsonb_typeof == 'string'`: the payload is double-encoded.
- [ ] **Step 3: Implement.** Thread `conn` through `publish -> _publish_via_outbox -> store.save`. In `PostgresOutboxStore.save`, run on the given connection, otherwise in `async with self._pool.acquire() as conn`. Drop the `json.dumps` pre-dumps in both stores. Keep `$n::jsonb` casts.
- [ ] **Step 4: Run them and confirm they pass.** Run the same two commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_events`;
  - real-infra command with `tests/realinfra/event-plane`.

  Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.6: Outbox claims: claim tokens, guarded transitions, release on cancel and bookkeeping failure, batch circuit-break, poller error backoff, graceful stop
**Covers:** MVP-136, MVP-264, MVP-268
**Severity:** medium
**Wave:** 1
**Depends on:** WS-06.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Modify: `src/solace_events/publisher.py`:
  - `DEFAULT_CLAIM_STALE_SECONDS = 120.0` and `DEFAULT_RETRY_MAX_SECONDS = 30.0`;
  - `OutboxRecord.claim_token: UUID | None = None`;
  - `InMemoryOutboxStore`: `get_pending(limit, stale_after_seconds, *, claim_token=None)` stamps the token; `mark_published`, `reset_to_pending`, `mark_failed` and `increment_retry` take keyword `claim_token` and act only while `status == PUBLISHING` and the token matches; `reset_to_pending` and `mark_failed` return `bool`; `increment_retry` returns `int | None`;
  - `flush_outbox` claims with a fresh token and stops at the first send failure, releasing the rest without a retry increment. A `finally` releases every still-claimed, unfinished record under `asyncio.shield`, including on `CancelledError`;
  - `OutboxPoller.__init__(publisher, poll_interval_ms=100, batch_size=100, error_backoff_max_seconds=30.0, stop_timeout_seconds=10.0)`;
  - `_poll_loop` backs off on errors: the delay doubles from the poll interval up to the cap, ERROR is logged on the first failure and then at most once per 60 s with a suppressed count, and the delay resets after a clean flush;
  - `stop()` waits up to `stop_timeout_seconds` for the in-flight flush before cancelling.
- Modify: `src/solace_events/postgres_stores.py`. `get_pending` sets `claim_token = $4`. All transitions use `WHERE id = $1 AND status = 'PUBLISHING' AND ($n::uuid IS NULL OR claim_token = $n)`.
- Modify: `tests/solace_events/test_outbox_claim.py`. `test_postgres_get_pending_issues_atomic_claim_sql` asserts `args[:3] == (50, 120.0, "svc-test")` and `isinstance(args[3], UUID)`.
- Test: `tests/solace_events/test_mvp_136_claim_release.py` (new)
- Test: `tests/solace_events/test_mvp_264_claim_guard.py` (new)
- Test: `tests/solace_events/test_mvp_268_poller_backoff.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_264_guarded_transitions_pg.py` (new)

**Interfaces:**
- Consumes: WS-06.5's store signatures.
- Produces: claim-token store semantics (keyword `claim_token`, so existing call sites keep working); `OutboxPoller.stop()` finishes the in-flight flush, as IF-05 `.stop()` requires.

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_136_claim_release.py`:
```python
async def test_mvp_136_cancel_mid_flush_releases_every_claimed_row():
    outbox, gate = InMemoryOutboxStore(), asyncio.Event()
    producer = _BlockingProducer(gate)          # Mock subclass: send() awaits gate forever
    pub = EventPublisher(producer, outbox, use_outbox=True); await pub.start()
    for _ in range(5): await pub.publish(_crisis())
    flush = asyncio.create_task(pub.flush_outbox())
    await asyncio.sleep(0.05); flush.cancel()
    with contextlib.suppress(asyncio.CancelledError): await flush
    assert {r.status for r in outbox._records.values()} == {OutboxStatus.PENDING}
    assert all(r.claimed_at is None for r in outbox._records.values())

async def test_mvp_136_first_send_failure_releases_the_rest_of_the_batch_without_retry_cost():
    outbox, producer = InMemoryOutboxStore(), _CountingFailingProducer()
    pub = EventPublisher(producer, outbox, use_outbox=True, retry_base_seconds=0.0); await pub.start()
    for _ in range(5): await pub.publish(_crisis())
    await pub.flush_outbox()
    assert producer.sends == 1
    assert sorted(r.retry_count for r in outbox._records.values()) == [0, 0, 0, 0, 1]
    assert {r.status for r in outbox._records.values()} == {OutboxStatus.PENDING}

async def test_mvp_136_bookkeeping_failure_releases_the_rest():
    outbox = _IncrementRaisesOnceStore()          # InMemoryOutboxStore subclass; increment_retry raises ConnectionError once
    pub = EventPublisher(_CountingFailingProducer(), outbox, use_outbox=True, retry_base_seconds=0.0); await pub.start()
    for _ in range(3): await pub.publish(_crisis())
    with contextlib.suppress(ConnectionError): await pub.flush_outbox()
    assert OutboxStatus.PUBLISHING not in {r.status for r in outbox._records.values()}

async def test_mvp_136_poller_stop_finishes_the_in_flight_flush():
    # producer.send sleeps 0.2 s; poller.start(); publish 2; await asyncio.sleep(0.05); await poller.stop()
    # both records end PUBLISHED, none left PUBLISHING
    ...
```
In `tests/solace_events/test_mvp_264_claim_guard.py`:
```python
async def test_mvp_264_stolen_claim_cannot_reset_a_published_row():
    store = InMemoryOutboxStore(); r = _record(); await store.save(r)
    (a,) = await store.get_pending(claim_token=uuid4())
    store._records[r.id].claimed_at = datetime.now(timezone.utc) - timedelta(seconds=999)
    (b,) = await store.get_pending(claim_token=uuid4(), stale_after_seconds=120)
    assert await store.mark_published(r.id, claim_token=b.claim_token) is True
    assert await store.reset_to_pending(r.id, claim_token=a.claim_token) is False
    assert await store.mark_failed(r.id, "late failure", claim_token=a.claim_token) is False
    assert store._records[r.id].status == OutboxStatus.PUBLISHED

async def test_mvp_264_postgres_transitions_are_claim_guarded():
    conn = _RecordingConn(execute_result="UPDATE 0")
    store = PostgresOutboxStore(_RecordingPool(conn), source_service="svc-264")
    token = uuid4()
    assert await store.reset_to_pending(uuid4(), claim_token=token) is False
    assert await store.mark_failed(uuid4(), "x", claim_token=token) is False
    for sql, args in conn.calls:
        assert "status = 'PUBLISHING'" in " ".join(sql.split()) and token in args
```
In `tests/solace_events/test_mvp_268_poller_backoff.py`:
```python
async def test_mvp_268_poll_errors_back_off_and_log_once():
    pub = _FlushRaises(ConnectionRefusedError("db down"))       # EventPublisher stub: flush_outbox always raises
    poller = OutboxPoller(pub, poll_interval_ms=100)
    with structlog.testing.capture_logs() as logs:
        await poller.start(); await asyncio.sleep(1.0); await poller.stop()
    assert pub.flush_calls <= 4                                   # 0.1 + 0.2 + 0.4 s waits fit in 1 s
    assert sum(1 for e in logs if e["event"] == "Outbox poll error" and e["log_level"] == "error") == 1
```
In `tests/realinfra/event-plane/test_mvp_264_guarded_transitions_pg.py`:
```python
async def test_mvp_264_late_poller_cannot_flip_a_row_reclaimed_by_another(pg_migrated):
    store = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-264-{uuid4().hex[:6]}")
    r = _outbox_record(); await store.save(r)
    (a,) = await store.get_pending(limit=1, claim_token=uuid4())
    await pg_migrated.superuser_fetch("UPDATE event_outbox SET claimed_at = now() - interval '10 minutes' WHERE id = $1", r.id)
    (b,) = await store.get_pending(limit=1, claim_token=uuid4())
    assert await store.mark_published(r.id, claim_token=b.claim_token) is True
    assert await store.reset_to_pending(r.id, claim_token=a.claim_token) is False
    (row,) = await pg_migrated.superuser_fetch("SELECT status FROM event_outbox WHERE id = $1", r.id)
    assert row["status"] == "PUBLISHED"
```
- [ ] **Step 2: Run them and confirm they fail.** Run the unit command once per new unit file, then the real-infra command for the PG file.
  - Expected for `test_mvp_136_claim_release.py`: FAIL. Cancelled rows stay `PUBLISHING` (MVP-136a). All 5 sends are attempted (c). A raising `increment_retry` strands the rest (b).
  - Expected for `test_mvp_264_claim_guard.py`: FAIL. `TypeError` on `claim_token`. The unguarded reset flips a `PUBLISHED` row back to `PENDING` (MVP-264).
  - Expected for `test_mvp_268_poller_backoff.py`: FAIL. About 10 flush calls and about 10 ERROR lines in 1 s (MVP-268).
  - Expected for `test_mvp_264_guarded_transitions_pg.py`: FAIL. The late `reset_to_pending` is unguarded.
- [ ] **Step 3: Implement** the signatures listed under Files. Release unsent records with `reset_to_pending(id, next_retry_at=None, claim_token=token)`. The release in `finally` is wrapped in `asyncio.shield(...)` and swallows its own errors with a WARNING. The stale window drops to 120 s, because a flush can no longer outlive it: there is one send timeout per failed flush.
- [ ] **Step 4: Run them and confirm they pass.** Run the same commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_events`;
  - real-infra command with `tests/realinfra/event-plane`.

  Expected: all pass, including `test_outbox_claim.py` and `test_publisher_comprehensive.py`.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.7: DLQ reaper with per-group selection, no head-of-line blocking and its own redrive budget
**Covers:** MVP-051, MVP-137
**Severity:** high
**Wave:** 1
**Depends on:** WS-06.6
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Create: `src/solace_events/reaper.py`:
  - `class ReaperStats` (dataclass: `redriven`, `resolved`, `failed`, `parked`, `requeued`: int);
  - `class DLQReaper`.
- Modify: `src/solace_events/postgres_stores.py`. `PostgresDLQStore` gains:
  - `claim_redrivable(groups: Sequence[str], limit: int, lease_seconds: float) -> list[DeadLetterRecord]`, a lease claim: `UPDATE ... SET next_retry_at = now() + lease WHERE id IN (SELECT ... WHERE NOT resolved AND is_retriable AND consumer_group = ANY($1) AND (next_retry_at IS NULL OR next_retry_at <= now()) ORDER BY next_retry_at NULLS FIRST, dlq_entry_at LIMIT $2 FOR UPDATE SKIP LOCKED) RETURNING *`;
  - `record_redrive_failure(record_id, error_message, next_retry_at, park: bool) -> None`;
  - `count_unresolved_by_group(groups) -> dict[str, int]`.

  `mark_resolved` sets `resolved_at = now()`.
- Modify: `src/solace_events/dead_letter.py` (in-memory `DeadLetterStore` gains the same three methods)
- Modify: `src/solace_events/publisher.py`. `_route_to_dlq` writes `consumer_group=f"outbox:{source_service}"`, where `source_service` comes from the outbox store.
- Modify: `src/solace_events/__init__.py` (export `DLQReaper` and `ReaperStats`)
- Test: `tests/solace_events/test_mvp_051_dlq_reaper.py` (new)
- Test: `tests/solace_events/test_mvp_137_reaper_budget.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_051_reaper_pg.py` (new)

**Interfaces:**
- Consumes: DLQ fields from WS-06.3; claim-guarded outbox save from WS-06.6.
- Produces:
  ```
  DLQReaper(store, *, handlers: Mapping[str, Mapping[str, EventHandler]], outbox_requeue: Callable[[DeadLetterRecord], Awaitable[None]] | None = None,
            idempotency_store=None, interval_seconds=60.0, max_redrives=10, batch_size=10, lease_seconds=300.0,
            backoff_base_seconds=60.0, backoff_max_seconds=3600.0)
  ```
  Methods: `async run_once() -> ReaperStats`, `async start()`, `async stop()`.
  - `handlers` maps a consumer group to an event-type handler map. Only those groups, plus `outbox:<service>` when `outbox_requeue` is set, are ever selected.
  - A row whose type has no handler in its group is parked: `is_retriable=False`, `resolution_notes="no handler for event type"`.
  - The budget is `redrive_count < max_redrives`. `retry_count` (attempts before dead-lettering) is never used for it.
  - On success: `mark_resolved("redriven")`, and `mark_processed` when an idempotency store is given.
  - An `outbox:*` row is re-enqueued, then resolved `"requeued"`.

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_051_dlq_reaper.py`:
```python
async def test_mvp_051_reaper_redrives_a_dead_lettered_crisis_event():
    store = DeadLetterStore(); ev = _crisis(); handled = []
    await store.save(_dlq(ev, group="solace-group-notifications"))
    async def handler(event): handled.append(event.metadata.event_id)
    reaper = DLQReaper(store, handlers={"solace-group-notifications": {"safety.crisis.detected": handler}})
    stats = await reaper.run_once()
    assert (stats.redriven, stats.resolved, handled) == (1, 1, [ev.metadata.event_id])
    assert await store.count_unresolved() == 0

async def test_mvp_051_outbox_dead_letters_are_requeued():
    store, requeued = DeadLetterStore(), []
    await store.save(_dlq(_crisis(), group="outbox:safety-service"))
    async def requeue(record): requeued.append(record.event_id)
    stats = await DLQReaper(store, handlers={}, outbox_requeue=requeue).run_once()
    assert stats.requeued == 1 and len(requeued) == 1 and await store.count_unresolved() == 0
```
In `tests/solace_events/test_mvp_137_reaper_budget.py`:
```python
async def test_mvp_137_handlerless_rows_do_not_block_the_crisis_row():
    store, old = DeadLetterStore(), datetime.now(timezone.utc) - timedelta(hours=1)
    for _ in range(10):
        await store.save(_dlq(_session_event(), group="solace-group-notifications", dlq_entry_at=old))
    crisis = _crisis(); handled = []
    await store.save(_dlq(crisis, group="solace-group-notifications"))
    reaper = DLQReaper(store, handlers={"solace-group-notifications": {"safety.crisis.detected": lambda e: _append(handled, e)}}, batch_size=10)
    await reaper.run_once(); await reaper.run_once()
    assert len(handled) == 1
    assert sum(1 for r in store._records.values() if r.resolution_notes == "no handler for event type") == 10

async def test_mvp_137_outbox_retry_count_does_not_consume_the_redrive_budget():
    store, crisis = DeadLetterStore(), _crisis()
    rec = _dlq(crisis, group="solace-group-notifications"); rec.retry_count = 10
    await store.save(rec)
    reaper = DLQReaper(store, handlers={"solace-group-notifications": {"safety.crisis.detected": _always_fail}}, backoff_base_seconds=0.0)
    for _ in range(3): await reaper.run_once()
    stored = store._records[rec.id]
    assert (stored.is_retriable, stored.redrive_count) == (True, 3)

async def test_mvp_137_redrive_budget_parks_after_ten_failures():
    # same as above, 10 runs -> is_retriable False, redrive_count 10, resolved False (stays alertable)
    ...
```
In `tests/realinfra/event-plane/test_mvp_051_reaper_pg.py`:
```python
async def test_mvp_051_two_reapers_never_redrive_the_same_row_twice(pg_migrated):
    group = f"g-051-{uuid4().hex[:6]}"; store = PostgresDLQStore(pg_migrated.app_pool)
    rows = [_dlq(_crisis(), group=group) for _ in range(20)]
    for r in rows: await store.save(r)
    seen = []
    async def handler(event): seen.append(event.metadata.event_id); await asyncio.sleep(0.01)
    r1 = DLQReaper(store, handlers={group: {"safety.crisis.detected": handler}}, batch_size=10)
    r2 = DLQReaper(store, handlers={group: {"safety.crisis.detected": handler}}, batch_size=10)
    await asyncio.gather(r1.run_once(), r2.run_once())
    assert sorted(seen) == sorted(r.event_id for r in rows)
    assert (await store.count_unresolved_by_group([group]))[group] == 0
```
- [ ] **Step 2: Run them and confirm they fail.** Run the unit command for each unit file and the real-infra command for the PG file. Expected: FAIL. `ModuleNotFoundError: No module named 'solace_events.reaper'`: no production re-drive path exists, which is the write-only DLQ of MVP-051. With today's only API (`DeadLetterHandler.process_retriable`), the crisis row behind 10 handler-less rows gets 0 attempts and parks after 1 failure (`retry_count=11`), as in the register's repro of MVP-137.
- [ ] **Step 3: Implement** `DLQReaper` and the store methods listed. `run_once` processes rows sequentially, each handler call bounded by a 30 s timeout. Backoff after redrive failure *n* is `min(backoff_base * 2 ** (n - 1), backoff_max)`. `start()` loops `run_once()` every `interval_seconds` until `stop()`. `DeadLetterHandler` is left untouched for its legacy caller (analytics until WS-18 migrates).
- [ ] **Step 4: Run them and confirm they pass.** Run the same commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_events`;
  - real-infra command with `tests/realinfra/event-plane`;
  - unit command with `services/analytics-service/tests`.

  Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.8: Service event plane: one composition root, producer connect with backoff, strict fail-loud, one EVENTS_ENABLED flag
**Covers:** MVP-053, MVP-052, MVP-028
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-06.4, WS-06.7
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-14 (default assumed: staging is as strict as production; every guard uses strictness, never 'production' alone). D-20 (decided 2026-10-19: aiokafka is installed in the dedicated venv `D:\tmp\mvp-exec\venv` at W0 entry, never in anaconda, and the lock-only venv `D:\tmp\mvp-exec\venv-clean` proves it at the W0 barrier).
**Real infra:** yes
**Files:**
- Create: `src/solace_events/_env.py` (`def is_strict_env(env: str | None = None) -> bool`: `env`, or else `os.environ["ENVIRONMENT"]` (default `"development"`), lower-cased, is in `{"staging", "production"}`)
- Create: `src/solace_events/service_plane.py` (IF-05; see Interfaces)
- Modify: `src/solace_events/publisher.py`:
  - `EventPublisher.start()` verifies the outbox schema (fatal), then attempts `producer.start()` once inline. On failure it does not raise; it starts a background connector with backoff 1, 2, 4 ... capped at 60 s, forever;
  - new `is_connected` property;
  - `flush_outbox()` returns 0 without claiming while disconnected;
  - `publish()` in outbox mode works before connect, while direct mode raises `EventPublishError`;
  - `stop()` cancels the connector;
  - `create_publisher` raises `EventPlaneConfigError` when `is_strict_env()` and no `postgres_pool`, and when `use_mock` in strict.
- Modify: `src/solace_events/dead_letter.py` (`create_dead_letter_handler` raises `EventPlaneConfigError` when `is_strict_env()` and no pool or store)
- Modify: `src/solace_events/__init__.py` (export the service-plane names)
- Test: `tests/solace_events/test_mvp_053_producer_connect_backoff.py` (new)
- Test: `tests/solace_events/test_mvp_052_strict_never_in_memory.py` (new)
- Test: `tests/solace_events/test_mvp_028_events_enabled_and_bootstrap.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_053_kafka_down_at_boot.py` (new)

**Interfaces:**
- Consumes: IF-01 `is_strict_environment` (parity test only); WS-06.3 consumer, WS-06.6 poller and WS-06.7 reaper; IF-02 `EVENTS_ENABLED` and `KAFKA_BOOTSTRAP_SERVERS`.
- Produces, IF-05 exactly as written, plus the additions flagged as new:
  - `async def start_service_event_plane(service_name: str, *, pool: asyncpg.Pool | None, kafka_settings: KafkaSettings | None = None, strict: bool, use_mock: bool = False) -> ServiceEventPlane`. This registers and starts the process plane; `use_mock` is new.
  - `def get_service_event_plane(service_name: str | None = None) -> ServiceEventPlane`. Raises `EventPlaneNotStartedError` when none is registered; the optional name is new.
  - New: `def get_or_create_service_event_plane(service_name: str, *, pool, kafka_settings=None, use_mock: bool = False) -> ServiceEventPlane`. Synchronous; constructs with `strict=is_strict_env()` and registers the plane without starting it. The bridges use it.
  - New: `async def shutdown_service_event_plane(service_name: str | None = None) -> None`.
  - `class ServiceEventPlane(service_name, *, pool, kafka_settings=None, strict, use_mock=False, metrics_interval_seconds=15.0, reaper_interval_seconds=60.0)`:
    - the constructor is synchronous and validates the configuration;
    - `async start()` / `async stop()` (stop finishes the in-flight flush and releases claims);
    - `async publish(event, *, conn=None) -> UUID`;
    - `subscribe(topic, event_types, handler, *, group_id, max_attempts=5) -> None`. May be called before or after `start()`; a late subscription starts its consumer at once;
    - `readiness() -> tuple[bool, str]`;
    - new: the properties `publisher` and `poller`, `background_tasks() -> dict[str, asyncio.Task]` and `consumer_groups() -> tuple[str, ...]`.
  - Strict rules raise `EventPlaneConfigError`:
    - `pool is None`;
    - `EVENTS_ENABLED` false;
    - `use_mock`;
    - `KAFKA_BOOTSTRAP_SERVERS` unset when `kafka_settings is None`;
    - any bootstrap host in `{localhost, 127.0.0.1}`.

    In development/test, `EVENTS_ENABLED=false` gives a disabled plane: in-memory outbox, no producer, poller or consumers, and `readiness() == (False, "events disabled")`. No pool gives an in-memory outbox and DLQ with a WARNING and `readiness() == (False, "in-memory outbox (development/test only)")`.
  - Consumers built by `subscribe`: one `EventConsumer` per `group_id` with `PostgresDLQStore` and `PostgresIdempotencyStore` (in-memory outside strict when there is no pool). They start with backoff, and each runs its own consume task. One `DLQReaper` covers every group plus `outbox:<service>`, re-enqueueing through the plane's own outbox.
  - `readiness()` is `(True, "ok")` only when the producer is connected, the poller task is alive and every consumer task is running. The detail string names the first failing part.

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_053_producer_connect_backoff.py`:
```python
class _FlakyStartProducer(MockKafkaProducerAdapter):
    def __init__(self, failures): super().__init__(); self.failures = failures; self.starts = 0
    async def start(self):
        self.starts += 1
        if self.starts <= self.failures: raise ConnectionError("broker not listening")
        await super().start()

async def test_mvp_053_start_survives_broker_down_and_publish_is_captured():
    producer = _FlakyStartProducer(failures=2)
    pub = EventPublisher(producer, InMemoryOutboxStore(), use_outbox=True)
    pub._connect_backoff_base = 0.01                     # test seam on the connector
    await pub.start()
    assert pub.is_connected is False
    event_id = await pub.publish(_crisis())
    assert await pub.flush_outbox() == 0
    await _wait_until(lambda: pub.is_connected, timeout=2.0)
    assert await pub.flush_outbox() == 1 and producer.get_messages()[0][2]["metadata"]["event_id"] == str(event_id)
```
In `tests/solace_events/test_mvp_052_strict_never_in_memory.py`:
```python
@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_052_create_publisher_refuses_in_memory_outbox_in_strict(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(EventPlaneConfigError):
        create_publisher(use_mock=False, postgres_pool=None, source_service="safety-service")

@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_052_create_dead_letter_handler_refuses_in_memory_dlq_in_strict(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(EventPlaneConfigError):
        create_dead_letter_handler("solace-group-notifications", postgres_pool=None)

@pytest.mark.parametrize("env,strict", [("development", False), ("test", False), ("staging", True), ("production", True), ("PRODUCTION", True)])
def test_mvp_052_strict_parity_with_runtime(env, strict):
    from solace_infrastructure.runtime import is_strict_environment
    assert is_strict_env(env) is strict is is_strict_environment(env)
```
In `tests/solace_events/test_mvp_028_events_enabled_and_bootstrap.py`:
```python
def test_mvp_028_strict_plane_requires_events_enabled(monkeypatch):
    monkeypatch.setenv("EVENTS_ENABLED", "false"); monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    with pytest.raises(EventPlaneConfigError, match="EVENTS_ENABLED"):
        ServiceEventPlane("safety-service", pool=object(), strict=True)     # constructor does no I/O

@pytest.mark.parametrize("bootstrap", [None, "localhost:9092", "127.0.0.1:29092"])
def test_mvp_028_strict_plane_refuses_localhost_or_unset_broker(monkeypatch, bootstrap):
    monkeypatch.delenv("EVENTS_ENABLED", raising=False)
    if bootstrap is None: monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    else: monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", bootstrap)
    with pytest.raises(EventPlaneConfigError, match="KAFKA_BOOTSTRAP_SERVERS"):
        ServiceEventPlane("orchestrator-service", pool=object(), strict=True)

def test_mvp_028_strict_plane_refuses_missing_pool_and_mock(monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    with pytest.raises(EventPlaneConfigError): ServiceEventPlane("safety-service", pool=None, strict=True)
    with pytest.raises(EventPlaneConfigError): ServiceEventPlane("safety-service", pool=object(), strict=True, use_mock=True)

async def test_mvp_028_events_disabled_in_test_is_visible_in_readiness(monkeypatch):
    monkeypatch.setenv("EVENTS_ENABLED", "false")
    plane = await start_service_event_plane("safety-service", pool=None, strict=False, use_mock=True)
    try:
        assert plane.readiness() == (False, "events disabled")
        await plane.publish(_crisis())                       # captured in memory, never raises
    finally:
        await shutdown_service_event_plane("safety-service")

async def test_mvp_028_mock_plane_drains_and_consumes(monkeypatch):
    monkeypatch.setenv("EVENTS_ENABLED", "true")
    plane = await start_service_event_plane("safety-service", pool=None, strict=False, use_mock=True)
    try:
        ev = _crisis(); await plane.publish(ev)
        await _wait_until(lambda: plane.publisher._producer.get_messages(), timeout=2.0)
        assert plane.poller is not None and get_service_event_plane() is plane
    finally:
        await shutdown_service_event_plane("safety-service")
```
In `tests/realinfra/event-plane/test_mvp_053_kafka_down_at_boot.py`:
```python
async def test_mvp_053_plane_boots_with_broker_down_then_delivers(kafka_bootstrap, pg_migrated, monkeypatch):
    service = f"svc-053-{uuid4().hex[:6]}"
    kafka_bootstrap.pause()
    try:
        plane = await start_service_event_plane(service, pool=pg_migrated.app_pool,
                                                kafka_settings=KafkaSettings(bootstrap_servers=kafka_bootstrap.bootstrap),
                                                strict=False)
        assert plane.readiness()[0] is False
        ev = _crisis(); await plane.publish(ev)
        (row,) = await pg_migrated.superuser_fetch("SELECT status FROM event_outbox WHERE event_id = $1", ev.metadata.event_id)
        assert row["status"] == "PENDING"
    finally:
        kafka_bootstrap.unpause()
    await _wait_until(lambda: plane.readiness()[0], timeout=90)
    got = await kafka_bootstrap.consume_until("solace.safety", lambda m: m["metadata"]["event_id"] == str(ev.metadata.event_id), timeout=60)
    assert got
    await shutdown_service_event_plane(service)
```
The test passes `strict=False` because the testcontainers broker is published on localhost. Strict broker validation is covered by the unit tests above.
- [ ] **Step 2: Run them and confirm they fail.** Run the unit command for each unit file and the real-infra command for the PG/Kafka file. Expected: FAIL.
  - `EventPublisher.start()` raises `ConnectionError` when the broker is down at boot, so outbox capture never starts (MVP-053).
  - Under `ENVIRONMENT=staging`, `create_publisher(postgres_pool=None)` returns an in-memory outbox with only a warning, because the guard checks `'production'` alone (MVP-052).
  - `ModuleNotFoundError: solace_events.service_plane` and `solace_events._env`: no single flag or strict check exists for the crisis plane (MVP-028).
- [ ] **Step 3: Implement.**
  - Add `ServiceEventPlane` and the module functions per the Interfaces.
  - The registry is a module-level `dict[str, ServiceEventPlane]`. `get_service_event_plane()` without a name returns the only registered plane, or raises `EventPlaneNotStartedError`.
  - Parse `EVENTS_ENABLED` as a true value in `{"1", "true", "yes", "on"}`, default `"true"`.
  - Validation messages name the variable, never its value.
  - The connector seam `_connect_backoff_base` defaults to 1.0.
- [ ] **Step 4: Run them and confirm they pass.** Run the same commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_events`;
  - real-infra command with `tests/realinfra/event-plane`;
  - unit command with `services/safety_service/tests`;
  - unit command with `services/therapy_service/tests`;
  - unit command with `services/diagnosis_service/tests`.

  Expected: all pass. The bridges still use `create_publisher` until WS-06.9.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.9: The seven bridges delegate to the plane: the poller always runs, strict envs fail loud, names stay stable
**Covers:** MVP-014, MVP-015, MVP-028
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-06.8
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-14 (default assumed: staging is strict)
**Real infra:** no
**Files:**
- Modify all seven bridge files. In each:
  - the constructor obtains `self._plane = get_or_create_service_event_plane("<service>", pool=postgres_pool, kafka_settings=kafka_settings, use_mock=use_mock)`. In strict, `EventPlaneConfigError` propagates out of `initialize_*`;
  - `start()` awaits `self._plane.start()` (idempotent) and subscribes to the EventBus as today;
  - `_handle_event` / `handle` / `bridge_event` call `await self._plane.publish(kafka_event)`. `bridge_event` keeps returning `bool`; a failure increments the plane's publish-failure counter and logs the event type only;
  - `stop()` and `shutdown_event_bridge()` call `shutdown_service_event_plane("<service>")` only when the bridge created the plane;
  - read-only properties `_publisher` (the plane's `EventPublisher`) and `_poller` (the plane's `OutboxPoller`) keep the existing wiring tests green;
  - safety, therapy and diagnosis drop their own `OutboxPoller`; the plane owns the poller.

  The files:
  - `services/orchestrator_service/src/infrastructure/event_bridge.py`
  - `services/safety_service/src/infrastructure/event_bridge.py`
  - `services/therapy_service/src/infrastructure/event_bridge.py`
  - `services/diagnosis_service/src/infrastructure/event_bridge.py`
  - `services/personality_service/src/infrastructure/event_bridge.py`
  - `services/user-service/src/infrastructure/event_bridge.py`
  - `services/notification-service/src/infrastructure/event_bridge.py`
- Test (one new file per service): `services/orchestrator_service/tests/test_mvp_014_event_bridge_plane.py`, `services/safety_service/tests/test_mvp_014_event_bridge_plane.py`, `services/therapy_service/tests/test_mvp_014_event_bridge_plane.py`, `services/diagnosis_service/tests/test_mvp_014_event_bridge_plane.py`, `services/personality_service/tests/test_mvp_014_event_bridge_plane.py`, `services/user-service/tests/test_mvp_014_event_bridge_plane.py`, `services/notification-service/tests/test_mvp_014_event_bridge_plane.py`

**Interfaces:**
- Consumes: IF-05 from WS-06.8; each service's own `events.py` (`to_kafka_event`, `EventBus`/`EventFactory`), unchanged.
- Produces:
  - `initialize_event_bridge(event_bus=..., kafka_settings=None, use_mock=False, postgres_pool=None)` and `initialize_notification_bridge(kafka_settings=None, use_mock=False, postgres_pool=None)` keep their names and keywords;
  - when the bootstrap already started the plane, the bridge reuses it and ignores `postgres_pool`/`kafka_settings`;
  - strict envs raise `EventPlaneConfigError` instead of falling back.

- [ ] **Step 1: Write the failing tests.** Each service file imports the bridge the way that service's existing tests do:
  - `from services.<svc>.src.infrastructure.event_bridge import ...` for the five underscore services;
  - `from src.infrastructure.event_bridge import ...` with the service root on `sys.path` for user-service and notification-service. Use the import style of WS-13's W0 normalization test for notification.

  The orchestrator file:
```python
async def _messages(producer, timeout=2.0):
    for _ in range(int(timeout / 0.05)):
        if producer.get_messages(): return producer.get_messages()
        await asyncio.sleep(0.05)
    return producer.get_messages()

async def test_mvp_014_orchestrator_crisis_event_reaches_kafka(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test"); monkeypatch.delenv("EVENTS_ENABLED", raising=False)
    bridge = await initialize_event_bridge(event_bus=EventBus(), use_mock=True, postgres_pool=None)
    try:
        ev = EventFactory.crisis_detected(session_id=uuid4(), user_id=uuid4(), risk_level="HIGH", crisis_type="suicidal_ideation")
        assert await bridge.bridge_event(ev) is True
        msgs = await _messages(bridge._publisher._producer)
        assert [m[2]["event_type"] for m in msgs] == ["safety.crisis.detected"]
        assert bridge._poller is not None
    finally:
        await shutdown_event_bridge()

@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_015_orchestrator_bridge_without_pool_fails_loud_in_strict(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env); monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    with pytest.raises(EventPlaneConfigError):
        await initialize_event_bridge(event_bus=EventBus(), postgres_pool=None)
```
  The other six files follow the same shape with their own entry points:
  - **safety:** `initialize_notification_bridge(use_mock=True, postgres_pool=None)`, then `await bridge._bridge.handle(CrisisDetectedEvent(...))`. The strict test uses `initialize_notification_bridge(postgres_pool=None)`.
  - **therapy and personality:** `initialize_event_bridge(event_bus=EventBus(), use_mock=True, postgres_pool=None)`, then `bridge_event(<a bridged domain event>)`.
  - **diagnosis, user and notification:** `initialize_event_bridge(use_mock=True, postgres_pool=None)`, then `bridge_event(...)` with one bridged event from that service's `events.py`. Notification uses a delivery event; user uses `user.deleted`.

  Every file asserts:
  - the Kafka event type reaches the mock producer within 2 s;
  - `bridge._poller is not None`;
  - `EventPlaneConfigError` under `staging` and `production` without a pool.
- [ ] **Step 2: Run them and confirm they fail.** Run the unit command for each of the seven new files (one process per service suite path). Expected: FAIL.
  - Orchestrator, user, notification and personality get no message within 2 s. They save to the outbox but never start an `OutboxPoller` (MVP-014).
  - All seven strict cases fail with `DID NOT RAISE` under `staging`, because the in-memory outbox fallback is keyed on `'production'` only (MVP-015/MVP-028). Under `production` they raise a plain `RuntimeError`, not `EventPlaneConfigError`.
  - Safety, therapy and diagnosis already pass the delivery assertion. It stays as a non-regression check.
- [ ] **Step 3: Implement** the bridge changes listed under Files. The bridge modules no longer read any `*_KAFKA_ENABLED` setting. Per-service gating in the service `main.py` files is removed by the owners (Cross-workstream requests).
- [ ] **Step 4: Run them and confirm they pass.** Run the same seven commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each with the unit command:
  - `tests/solace_events`
  - `services/orchestrator_service/tests`
  - `services/safety_service/tests`
  - `services/therapy_service/tests`
  - `services/diagnosis_service/tests`
  - `services/personality_service/tests`
  - `services/user-service/tests`
  - `services/notification-service/tests`
  - `tests/integration`

  Expected: all pass, including `test_event_bridge_outbox.py`, `test_crisis_event_flow.py` and both `test_event_bridge_wiring.py` files.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.10: Backlog, lag and DLQ signals for the crisis pipeline
**Covers:** MVP-353
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-06.9
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Create: `src/solace_events/metrics.py`. The prometheus_client objects on the default `REGISTRY`:
  - `solace_outbox_pending{service}` (Gauge)
  - `solace_outbox_oldest_pending_seconds{service}` (Gauge)
  - `solace_consumer_lag{group,topic}` (Gauge)
  - `solace_dlq_unresolved{group}` (Gauge)
  - `solace_event_publish_failures_total{service}` (Counter)
  - `solace_consumer_handler_failures_total{group,event_type}` (Counter)

  Label values are service names, group ids, topics and event types only.
- Modify: `src/solace_events/postgres_stores.py`: `PostgresOutboxStore.pending_stats() -> tuple[int, float]` counts `PENDING`/`PUBLISHING` rows for its own service and returns the age of the oldest in seconds (0.0 when none). `PostgresDLQStore.count_unresolved_by_group` is used as is.
- Modify: `src/solace_events/publisher.py`: `InMemoryOutboxStore.pending_stats()`. `EventPublisher.publish` increments `solace_event_publish_failures_total` before raising `EventPublishError`.
- Modify: `src/solace_events/consumer.py`:
  - every failed handler attempt increments `solace_consumer_handler_failures_total`;
  - `KafkaConsumerAdapter.lag() -> dict[str, int]` is implemented. AIOKafka computes it from broker state for every partition of each subscribed topic (topic metadata, not only this member's assignment), as end offset minus committed offset, a missing commit counting from the partition's earliest offset; so a member that holds no partition after a rebalance still reports the group's lag (crisis-review F13). The mock reports pending messages;
  - the lag value is mirrored into `ConsumerMetrics.current_lag`.
- Modify: `src/solace_events/service_plane.py`:
  - add a sampler task, `metrics_interval_seconds` (default 15 s), that sets the four gauges for the plane's service and groups. A failed sample keeps the last value and logs one WARNING per minute;
  - new constructor keyword `watch_lag: Sequence[tuple[str, str]] = CRISIS_LAG_WATCH`, with `CRISIS_LAG_WATCH = (("solace-group-notifications", "solace.safety"),)`: every plane's sampler also sets `solace_consumer_lag{group,topic}` for each watched pair from broker state through the admin client, whether or not this process consumes it, so the crisis consumer's lag stays visible while notification-service is down, frozen or holds no partition (crisis-review F13); a mock or disabled plane skips the watch;
  - new `def record_publish_failure(service_name: str) -> None`: increments `solace_event_publish_failures_total{service=service_name}` for callers whose `publish(conn=)` transaction fails outside the plane (WS-12's R-12-2, crisis-review F11).
- Test: `tests/solace_events/test_mvp_353_event_metrics.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_353_lag_and_backlog.py` (new)

**Interfaces:**
- Consumes: IF-05 metric names (frozen); prometheus-client pin (WS-01).
- Produces: the six IF-05 series. WS-04 builds alert rules on them: CrisisOutboxBacklog, NotificationConsumerLag, CrisisConsumerStalled (crisis-review F13), DLQUnresolved and CrisisEventPublishFailures. Because several planes export the watched crisis pair, rules over `solace_consumer_lag` aggregate with `max`, never `sum`. WS-05 drives those rules in its W1 alerting E2E. Also `record_publish_failure(service_name)` and `CRISIS_LAG_WATCH`; `record_publish_failure` is consumed by WS-12.9 (R-12-2) and by WS-10.3's publish retrier for timed-out attempts (crisis-review F2, F11).

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_353_event_metrics.py`:
```python
def _value(name, labels): return REGISTRY.get_sample_value(name, labels)

async def test_mvp_353_outbox_backlog_gauges_track_undrained_rows(monkeypatch):
    monkeypatch.setenv("EVENTS_ENABLED", "true")
    plane = ServiceEventPlane("svc-353a", pool=None, strict=False, use_mock=True, metrics_interval_seconds=0.05)
    plane.publisher._producer = _NeverStartsProducer()           # stays disconnected
    await plane.start()
    try:
        for _ in range(3): await plane.publish(_crisis())
        await asyncio.sleep(0.3)
        assert _value("solace_outbox_pending", {"service": "svc-353a"}) == 3
        assert _value("solace_outbox_oldest_pending_seconds", {"service": "svc-353a"}) > 0.0
    finally:
        await plane.stop()

async def test_mvp_353_dlq_unresolved_gauge_per_group():
    # plane subscribes group "g-353" (mock); a DLQ record for "g-353" is saved to plane's DLQ store
    # after one sampler tick: solace_dlq_unresolved{group="g-353"} == 1
    ...

async def test_mvp_353_handler_failures_are_counted_per_attempt():
    before = _value("solace_consumer_handler_failures_total", {"group": "g-353f", "event_type": "safety.crisis.detected"}) or 0.0
    consumer = EventConsumer(MockKafkaConsumerAdapter(), dlq_store=DeadLetterStore(), consumer_group="g-353f", retry_base_seconds=0.0)
    consumer.register_handler("safety.crisis.detected", _always_fail)
    await consumer._process_message("solace.safety", 0, 0, _crisis().to_dict())
    assert _value("solace_consumer_handler_failures_total", {"group": "g-353f", "event_type": "safety.crisis.detected"}) == before + 5

async def test_mvp_353_outbox_write_failures_are_counted():
    # InMemoryOutboxStore subclass whose save raises ConnectionError; publish -> EventPublishError
    # solace_event_publish_failures_total{service="svc-353w"} increases by 1
    ...

def test_crisis_review_f11_record_publish_failure_counts_for_the_service():
    before = _value("solace_event_publish_failures_total", {"service": "svc-f11"}) or 0.0
    record_publish_failure("svc-f11")
    assert _value("solace_event_publish_failures_total", {"service": "svc-f11"}) == before + 1

async def test_crisis_review_f13_lag_counts_partitions_this_member_does_not_hold():
    # AIOKafkaConsumerAdapter with a fake admin/metadata seam: topic "solace.safety" has partitions 0..3;
    # committed offsets {0: 5, 1: 2, 2: None, 3: 0}, earliest {2: 0}, end {0: 5, 1: 3, 2: 1, 3: 0}; assignment empty
    # adapter.lag() == {"solace.safety": 2}
    ...
```
In `tests/realinfra/event-plane/test_mvp_353_lag_and_backlog.py`:
```python
async def test_mvp_353_stuck_consumer_shows_lag(kafka_bootstrap, pg_migrated):
    topic, group, release = f"test.mvp353.{uuid4().hex[:8]}", f"g-353-{uuid4().hex[:6]}", asyncio.Event()
    for _ in range(12): await kafka_bootstrap.produce(topic, _crisis())
    plane = ServiceEventPlane(f"svc-353-{uuid4().hex[:6]}", pool=pg_migrated.app_pool,
                              kafka_settings=KafkaSettings(bootstrap_servers=kafka_bootstrap.bootstrap),
                              strict=False, metrics_interval_seconds=1.0)
    async def stuck(event): await release.wait()
    plane.subscribe(topic, ["safety.crisis.detected"], stuck, group_id=group)
    await plane.start()
    try:
        await _wait_until(lambda: (REGISTRY.get_sample_value("solace_consumer_lag", {"group": group, "topic": topic}) or 0) >= 11, timeout=40)
    finally:
        release.set(); await plane.stop()

async def test_mvp_353_undrained_outbox_ages(kafka_bootstrap, pg_migrated):
    # kafka_bootstrap.pause(); plane started; publish 1; after 6 s:
    # solace_outbox_oldest_pending_seconds{service} >= 5 and solace_outbox_pending{service} == 1; unpause in finally
    ...

async def test_crisis_review_f13_watched_group_lag_is_sampled_by_a_plane_that_does_not_consume_it(kafka_bootstrap, pg_migrated):
    # topic, group = unique names; produce 1 crisis event to topic; no consumer of that group ever runs
    # plane = ServiceEventPlane(f"svc-f13-{tag}", pool=pg_migrated.app_pool, kafka_settings=..., strict=False,
    #                           metrics_interval_seconds=1.0, watch_lag=((group, topic),)); no subscribe; start
    # within 30 s: REGISTRY solace_consumer_lag{group=group, topic=topic} == 1; plane.stop() in finally
    ...
```
- [ ] **Step 2: Run them and confirm they fail.** Run the unit command for the unit file and the real-infra command for the real-infra file. Expected: FAIL. `REGISTRY.get_sample_value` returns `None` for every series: no lag, outbox-age or DLQ-depth signal exists for the crisis path (MVP-353). `record_publish_failure` and the `watch_lag` keyword do not exist: a safety transaction failure is never counted and nobody reports the crisis group's lag while its consumer is down or idle (crisis-review F11, F13).
- [ ] **Step 3: Implement.** Add the metric objects, `record_publish_failure` and the sampler listed above. The AIOKafka `lag()` uses `AIOKafkaAdminClient.list_consumer_group_offsets(group_id)` for committed offsets and the topic's partition metadata with end and beginning offsets against broker state, so a dead consume task or a member without partitions still shows growing lag. It needs a short-lived admin client per sample, closed after use; the watched pairs share that client.
- [ ] **Step 4: Run them and confirm they pass.** Run the same commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_events`;
  - real-infra command with `tests/realinfra/event-plane`.

  Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.11: Event-table erasure, retention purge and the user.erasure.completed schema
**Covers:** MVP-054
**Severity:** high
**Wave:** 3
**Depends on:** WS-06.10
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: a user's outbox and DLQ rows are hard-deleted at erasure except the erasure-protocol rows `user.deleted`/`user.erasure.completed`; published outbox rows are purged after 7 days, resolved DLQ rows after 30 days)
**Real infra:** yes
**Files:**
- Modify: `src/solace_events/schemas.py`. Add `class UserErasureCompletedKafkaEvent(BaseEvent)` with:
  - `event_type: Literal["user.erasure.completed"]`;
  - `service_name: str`, `erasure_request_id: UUID`, `stores: dict[str, int]`, `completed_at: datetime`.

  Register it in `EVENT_REGISTRY`; the topic is `solace.users` through the existing `"user."` prefix. No existing field changes.
- Modify: `src/solace_events/__init__.py` (export it, plus `erase_user_events`)
- Modify: `src/solace_events/postgres_stores.py`:
  - `PostgresOutboxStore.delete_for_user(user_id, *, keep_event_types) -> int`;
  - `PostgresOutboxStore.purge_published(older_than: timedelta) -> int`;
  - `PostgresDLQStore.delete_for_user(user_id, groups, *, keep_event_types) -> int`;
  - `PostgresDLQStore.purge_resolved(groups, older_than) -> int`;
  - `PostgresIdempotencyStore.purge(groups, older_than) -> int`.

  The in-memory stores get equivalent methods.
- Create: `src/solace_events/retention.py`:
  - `OUTBOX_PUBLISHED_RETENTION = timedelta(days=7)`;
  - `DLQ_RESOLVED_RETENTION = timedelta(days=30)`;
  - `PROCESSED_EVENTS_RETENTION = timedelta(days=30)`;
  - `PROTOCOL_EVENT_TYPES = frozenset({"user.deleted", "user.erasure.completed"})`;
  - `class RetentionPurger(plane, *, interval_seconds=3600.0)` with `async run_once() -> dict[str, int]`, `start()` and `stop()`.
- Modify: `src/solace_events/service_plane.py`:
  - `async def erase_user_events(user_id: UUID) -> int` (IF-05/IF-16) deletes the process plane's own outbox rows (`source_service`) and its groups' plus `outbox:<service>` DLQ rows for that user, keeping `PROTOCOL_EVENT_TYPES`. It returns the total deleted;
  - the plane starts a `RetentionPurger` and lists it in `background_tasks()`.
- Test: `tests/solace_events/test_mvp_054_erasure_and_retention.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_054_erasure_retention_pg.py` (new)

**Interfaces:**
- Consumes: IF-16 (WS-14's `register_user_erasure_consumer` calls `erase_user_events` inside each PHI service's erasure handler).
- Produces: `erase_user_events(user_id) -> int`, `UserErasureCompletedKafkaEvent` and the retention constants. WS-05 builds the erasure event contracts (IF-20) on the schema.

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_054_erasure_and_retention.py`:
```python
def test_mvp_054_erasure_completed_event_is_registered():
    from solace_events.schemas import EVENT_REGISTRY, UserErasureCompletedKafkaEvent, get_topic_for_event
    ev = UserErasureCompletedKafkaEvent(user_id=uuid4(), service_name="memory-service", erasure_request_id=uuid4(),
                                        stores={"postgres": 3, "redis": 2}, completed_at=datetime.now(timezone.utc))
    assert EVENT_REGISTRY["user.erasure.completed"] is UserErasureCompletedKafkaEvent
    assert get_topic_for_event(ev) == "solace.users"
    assert deserialize_event(ev.to_dict()) == ev
```
In `tests/realinfra/event-plane/test_mvp_054_erasure_retention_pg.py`:
```python
async def test_mvp_054_erase_user_events_removes_the_users_rows_only(pg_migrated, monkeypatch):
    service = f"svc-054-{uuid4().hex[:6]}"
    plane = await start_service_event_plane(service, pool=pg_migrated.app_pool, strict=False, use_mock=True)
    try:
        target, other = uuid4(), uuid4()
        for uid in (target, other):
            await plane.publish(_crisis(user_id=uid))
            await plane.publish(UserDeletedKafkaEvent(user_id=uid, reason="user_request"))
        await PostgresDLQStore(pg_migrated.app_pool).save(_dlq(_crisis(user_id=target), group=f"outbox:{service}"))
        deleted = await erase_user_events(target)
        assert deleted == 2
        rows = await pg_migrated.superuser_fetch(
            "SELECT event_type FROM event_outbox WHERE aggregate_id = $1 AND source_service = $2", target, service)
        assert [r["event_type"] for r in rows] == ["user.deleted"]
        assert await _dlq_count(pg_migrated, target) == 0
        assert await _outbox_count(pg_migrated, other, service) == 2
    finally:
        await shutdown_service_event_plane(service)

async def test_mvp_054_retention_purges_old_published_and_resolved_rows(pg_migrated):
    # rows (superuser backdated): PUBLISHED published_at now()-8d (purged), PUBLISHED now()-6d (kept),
    # PENDING created_at now()-30d (kept); resolved DLQ resolved_at now()-31d (purged), now()-29d (kept),
    # unresolved DLQ dlq_entry_at now()-100d (kept)
    # RetentionPurger(plane).run_once() -> {"outbox": 1, "dlq": 1, "processed_events": >= 0}; survivors as listed
    ...
```
- [ ] **Step 2: Run them and confirm they fail.** Run the unit command for the unit file and the real-infra command for the PG file. Expected: FAIL.
  - `KeyError: 'user.erasure.completed'` / `ImportError: UserErasureCompletedKafkaEvent`.
  - `ImportError: cannot import name 'erase_user_events'`: nothing deletes a user's event rows.
  - `ModuleNotFoundError: solace_events.retention`: no retention purge exists. Event payloads with PHI stay forever (MVP-054).
- [ ] **Step 3: Implement** the store methods, `erase_user_events` and `RetentionPurger` listed. Every `DELETE` is scoped by `source_service` (outbox) or by `consumer_group = ANY($groups)` (DLQ, processed events), so one service never deletes another's rows. Log only counts.
- [ ] **Step 4: Run them and confirm they pass.** Run the same commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_events`;
  - real-infra command with `tests/realinfra/event-plane`;
  - unit command with `tests/integration`.

  Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.12: No free text in event payloads; dead-letter errors carry no input values
**Covers:** MVP-054
**Severity:** high
**Wave:** 3
**Depends on:** WS-06.11, WS-12.2 (`CATEGORIES` and `parse_escalation_reason`, W1)
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: events carry ids and category codes only). D-05 (default assumed: provisional lexicon v1 category codes from `solace_clinical.CATEGORIES` are the allowed values).
**Real infra:** yes
**Files:**
- Create: `src/solace_events/payload_policy.py` (`def redact_free_text(event: BaseEvent) -> tuple[BaseEvent, list[str]]`)
- Modify: `src/solace_events/metrics.py` (`solace_event_payload_redactions_total{service,event_type}` counter)
- Modify: `src/solace_events/publisher.py`. `EventPublisher.publish` applies `redact_free_text` before building the `OutboxRecord`. For every redaction it increments the counter and logs `event_payload_redacted` with the event type and field paths only. It never rejects an event, so a crisis page is never dropped over a payload defect.
- Modify: `src/solace_events/errors.py` (`safe_error_message` also strips `input_value=` fragments from any exception text and truncates to 200 characters)
- Test: `tests/solace_events/test_mvp_054_payload_redaction.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_054_no_free_text_at_rest.py` (new)

**Interfaces:**
- Consumes: IF-07 `solace_clinical.CATEGORIES` and `parse_escalation_reason` (WS-12.2, W1); IF-06 rule "category codes only, never raw text".
- Produces the redaction table, applied at publish time:

| Event type | Field paths | Rule |
|---|---|---|
| `safety.crisis.detected` | `trigger_indicators[*]` | `"uncategorized"` unless in CATEGORIES |
| `safety.assessment.completed` | `risk_factors[*].evidence`, `protective_factors[*]` | `"uncategorized"` unless in CATEGORIES |
| `safety.escalation.triggered` | `escalation_reason` | kept when it is `"unspecified"` or `solace_clinical.parse_escalation_reason` accepts it (the IF-06 `auto\|manual:<LEVEL>[:codes]` form, every code in CATEGORIES; WS-12.2, crisis-review F10), else `"unspecified"` |
| `user.created` | `email`, `display_name` | `""` |
| `system.error` | `message` / `stack_trace` | `"redacted"` / `None` |
| `safety.incident.created` | `description` | `"redacted"` |
| `safety.incident.resolved`, `safety.escalation.resolved` | `resolution_notes` | `"redacted"` |
| `safety.crisis.resolved` | `resolution_notes` | `None` |
| `diagnosis.completed` | `primary_hypothesis.evidence_summary`, `differential_hypotheses[*].evidence_summary` | `"redacted"` |
| `homework.assigned` | `homework_description` | `"redacted"` |
| `treatment.response` | `notes` | `""` |
| `notification.failed` | `error_message` | `"redacted"` |
| `therapy.progress.milestone` | `milestone_description` | `"redacted"` |
| `safety.plan.activated` | `activation_reason` | `"redacted"` |
| `safety.plan.updated` | `update_reason` | `""` |
| `safety.output.filtered` | `filter_reason` | `"redacted"` |
| `safety.risk.level_changed` | `change_reason` | `None` |

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_054_payload_redaction.py`:
```python
async def test_mvp_054_raw_trigger_text_never_reaches_the_outbox():
    store = InMemoryOutboxStore(source_service="diagnosis-service")
    pub = EventPublisher(MockKafkaProducerAdapter(), store, use_outbox=True); await pub.start()
    ev = _crisis(trigger_indicators=["I want to end my life tonight", "suicidal_ideation"])
    await pub.publish(ev)
    (rec,) = store._records.values()
    assert rec.event_payload["trigger_indicators"] == ["uncategorized", "suicidal_ideation"]
    assert "end my life" not in json.dumps(rec.event_payload)

async def test_mvp_054_user_created_carries_no_contact_details():
    # publish UserCreatedKafkaEvent(email="person@example.invalid", display_name="Synthetic Name", role="user")
    # stored payload: email == "" and display_name == ""; metric
    # solace_event_payload_redactions_total{service="user-service", event_type="user.created"} increased by 1
    ...

async def test_crisis_review_f10_escalation_reason_codes_survive_and_text_does_not():
    store = InMemoryOutboxStore(source_service="safety-service")
    pub = EventPublisher(MockKafkaProducerAdapter(), store, use_outbox=True); await pub.start()
    kept = EscalationTriggeredEvent(user_id=uuid4(), escalation_reason="auto:CRITICAL:suicidal_ideation", priority="CRITICAL")
    text = EscalationTriggeredEvent(user_id=uuid4(), escalation_reason="Triggers: kill myself", priority="CRITICAL")
    await pub.publish(kept); await pub.publish(text)
    reasons = {r.event_payload["escalation_reason"] for r in store._records.values()}
    assert reasons == {"auto:CRITICAL:suicidal_ideation", "unspecified"}

def test_mvp_054_validation_errors_lose_their_input_values():
    try:
        CrisisDetectedEvent.model_validate({**_crisis().to_dict(), "detection_layer": "I want to end my life"})
    except ValidationError as exc:
        msg = safe_error_message(exc)
    assert "end my life" not in msg and "detection_layer" in msg
```
In `tests/realinfra/event-plane/test_mvp_054_no_free_text_at_rest.py`:
```python
async def test_mvp_054_outbox_rows_hold_codes_not_text(pg_migrated):
    pub = await _started_publisher(pg_migrated, f"svc-054r-{uuid4().hex[:6]}")
    ev = _crisis(trigger_indicators=["I want to end my life tonight"])
    await pub.publish(ev)
    (row,) = await pg_migrated.superuser_fetch("SELECT event_payload::text AS p FROM event_outbox WHERE event_id = $1", ev.metadata.event_id)
    assert "end my life" not in row["p"] and '"uncategorized"' in row["p"]
```
- [ ] **Step 2: Run them and confirm they fail.** Run the unit command for the unit file and the real-infra command for the PG file. Expected: FAIL.
  - The stored payload contains `"I want to end my life tonight"`: raw crisis trigger text is kept in `event_outbox` (MVP-054, the diagnosis `trigger_text` path).
  - `user.created` keeps the email.
  - `safe_error_message` keeps `input_value='I want to end my life'`.
  - The free-text escalation reason is stored as written; once a CATEGORIES-only rule exists, it would also rewrite `auto:CRITICAL:suicidal_ideation` to `uncategorized`, which is why the table uses `parse_escalation_reason` (crisis-review F10).
- [ ] **Step 3: Implement** `redact_free_text` from the Interfaces table using `model_copy(update=...)` (events are frozen). Nested models are rebuilt rather than mutated. The function is total: an unknown type returns the event unchanged with `[]`.
- [ ] **Step 4: Run them and confirm they pass.** Run the same commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each with the unit command:
  - `tests/solace_events`
  - `services/notification-service/tests`
  - `services/safety_service/tests`
  - `services/diagnosis_service/tests`
  - `tests/integration`

  Then the real-infra command with `tests/realinfra/event-plane`. Expected: all pass. Producers that already send category codes are unchanged.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.13: Bounded offset-tracker memory; closure check for JSONB double-encoding
**Covers:** MVP-265, MVP-267
**Severity:** low
**Wave:** 5
**Depends on:** WS-06.4
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `src/solace_events/consumer.py`. `OffsetTracker.mark_processed` prunes `_processed[key]` below the new committable offset and keeps `_min_offset` at it, so memory is bounded by in-flight offsets.
- Test: `tests/solace_events/test_mvp_265_offset_tracker_pruning.py` (new)

**Interfaces:**
- Consumes: the JSONB realinfra tests delivered by WS-12 and WS-14 (Cross-workstream requests); `tests/realinfra/event-plane/test_mvp_132_outbox_same_transaction.py::test_mvp_267_event_payloads_are_stored_as_jsonb_objects` (WS-06.5).
- Produces: no new interface.

- [ ] **Step 1: Write the failing test** in `tests/solace_events/test_mvp_265_offset_tracker_pruning.py`:
```python
async def test_mvp_265_processed_offsets_are_pruned():
    tracker = OffsetTracker()
    for off in range(10_000):
        await tracker.track_received("solace.assessments", 0, off)
        await tracker.mark_processed("solace.assessments", 0, off)
    assert len(tracker._processed[("solace.assessments", 0)]) <= 1
    assert await tracker.get_committed("solace.assessments", 0) == 10_000

async def test_mvp_265_out_of_order_commit_still_correct_after_pruning():
    tracker = OffsetTracker()
    for off in (1, 2, 3): await tracker.track_received("t", 0, off)
    assert await tracker.mark_processed("t", 0, 2) is None
    assert await tracker.mark_processed("t", 0, 1) == 3
    assert await tracker.mark_processed("t", 0, 3) == 4
    assert len(tracker._processed[("t", 0)]) <= 1
```
- [ ] **Step 2: Run it and confirm it fails.** Run: unit command with `tests/solace_events/test_mvp_265_offset_tracker_pruning.py`. Expected: FAIL with `assert 10000 <= 1`: the processed set grows by one entry per message forever (MVP-265).
- [ ] **Step 3: Implement** pruning in `mark_processed`. Once a committable offset *c* is known, drop every processed offset below *c* and set `_min_offset[key] = c`.
- [ ] **Step 4: Run it and confirm it passes.** Run the same command. Expected: PASS.
- [ ] **Step 5: Regression and MVP-267 closure.**
  - Run the unit command with `tests/solace_events`. Expected: all pass.
  - Verification for MVP-267: run the real-infra command once for each of these:
    - `tests/realinfra/event-plane/test_mvp_132_outbox_same_transaction.py`;
    - WS-12's escalation JSONB test file;
    - WS-14's preferences JSONB test file.

    Expected: all pass, and every asserted `jsonb_typeof` is `object` or `array`, never `string`. If either supporting file is missing, record a ledger Ruling and leave MVP-267 open. Do not mark it fixed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-06.14: One topic registry with D-24 retention, created if missing when the event plane starts
**Covers:** none (supporting edit for MVP-269, owned by WS-01; request R-01.4(b))
**Severity:** low
**Wave:** 1
**Depends on:** WS-06.10
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-24 (default assumed: Kafka `solace.safety` and `solace.users` retain 7 days)
**Real infra:** yes
**Files:**
- Modify: `src/solace_events/config.py`. `TOPIC_CONFIGS[SolaceTopic.SAFETY]` and `TOPIC_CONFIGS[SolaceTopic.USERS]` get `retention_ms=604800000`. The `TopicConfig` field defaults stay as they are, because `tests/solace_events/test_config.py:32` pins `replication_factor == 3`.
- Create: `src/solace_events/topics.py`:
  - `@dataclass(frozen=True) class TopicSpec: name: str; partitions: int; replication_factor: int; configs: dict[str, str]`;
  - `def plan_topics(configs: Iterable[TopicConfig], *, broker_count: int) -> list[TopicSpec]`. Replication is `min(cfg.replication_factor, broker_count)`, so it fits the single pilot broker, and `min.insync.replicas` is `min(cfg.min_insync_replicas, replication)`. `configs` carries `retention.ms`, `cleanup.policy` and `min.insync.replicas`;
  - `async def ensure_topics(kafka_settings: KafkaSettings, configs: Iterable[TopicConfig] | None = None, *, admin_factory: Callable[..., Any] | None = None) -> list[str]`. It uses one short-lived `AIOKafkaAdminClient` built from `kafka_settings.get_connection_params()` (lazy aiokafka import) and closes it in `finally`. It reads the broker count from cluster metadata, lists the topics, creates only the missing ones (default `TOPIC_CONFIGS.values()`) and returns the created names. A "topic already exists" outcome, whether raised or reported per topic in the response, counts as success. Any other admin error propagates. Existing topics are never altered.
- Modify: `src/solace_events/service_plane.py`. `ServiceEventPlane.start()` starts a `topics` background task, listed in `background_tasks()`. It calls `ensure_topics(self._kafka_settings)` with the connector backoff (1 s, doubling, capped at 60 s; test seam `_topics_backoff_base`) until the first success, then exits. Until then `readiness()` is `(False, "kafka topics: not ensured")` when every other part is ready. Mock and disabled planes skip it. The producer still connects on its own schedule (WS-06.8). Outbox rows stay `PENDING` and the poller retries them, so nothing is lost while a topic is missing.
- Modify: `src/solace_infrastructure/kafka/topics.py`. `TopicManager.register_solace_topics()` registers one `TopicDefinition` per `TOPIC_CONFIGS` entry (name, partitions, replication_factor, retention_ms, `CleanupPolicy(cfg.cleanup_policy)`, min_insync_replicas), imported inside the method. It registers no `*.dlq` definitions, because IF-05 has no Kafka DLQ topics. The hard-coded list at lines 319-344 is deleted.
- Modify: `tests/solace_infrastructure/kafka/test_topics.py`. `TestTopicManager.test_register_solace_topics` now asserts that every registry name is present and that no name ends in `.dlq`. The shared-file rule for `tests/solace_infrastructure` lets the code owner edit it, and WS-06 owns `src/solace_infrastructure/kafka`.
- Modify: `src/solace_events/__init__.py` (export `ensure_topics`, `plan_topics`, `TopicSpec`)
- Test: `tests/solace_events/test_mvp_269_topic_registry_and_creation.py` (new)
- Test: `tests/realinfra/event-plane/test_mvp_269_topics_created_at_plane_start.py` (new)

**Interfaces:**
- Consumes: `KafkaSettings.get_connection_params()`, `TOPIC_CONFIGS`, `SolaceTopic`; `ServiceEventPlane.start()`, `background_tasks()` and `readiness()` (WS-06.8); IF-19 `kafka_bootstrap` and `pg_migrated` (WS-05.2).
- Produces: `TOPIC_CONFIGS` as the single topic registry, with `TopicManager.register_solace_topics` derived from it. Topics are created if missing at plane start, so a strict broker with `KAFKA_AUTO_CREATE_TOPICS_ENABLE=false` still carries crisis events. Satisfies request R-01.4(b) for WS-01.12 (the MVP-269 Kafka-part closure check, W5) and lets WS-03 land the strict-overlay auto-create switch (WS-01's R-01.1) after this task in W1 phase A.

- [ ] **Step 1: Write the failing tests.** In `tests/solace_events/test_mvp_269_topic_registry_and_creation.py`:
```python
SEVEN_DAYS_MS = 7 * 24 * 3600 * 1000

def test_mvp_269_safety_and_users_retain_seven_days():
    assert TOPIC_CONFIGS[SolaceTopic.SAFETY].retention_ms == SEVEN_DAYS_MS
    assert TOPIC_CONFIGS[SolaceTopic.USERS].retention_ms == SEVEN_DAYS_MS

@pytest.mark.parametrize("brokers,rf,isr", [(1, 1, 1), (3, 3, 2)])
def test_mvp_269_plan_fits_replication_to_the_cluster(brokers, rf, isr):
    (spec,) = plan_topics([TOPIC_CONFIGS[SolaceTopic.SAFETY]], broker_count=brokers)
    assert (spec.name, spec.partitions, spec.replication_factor) == ("solace.safety", 4, rf)
    assert spec.configs["min.insync.replicas"] == str(isr) and spec.configs["retention.ms"] == str(SEVEN_DAYS_MS)

# _FakeAdmin(existing: set[str], brokers: int = 1, create_error: Exception | None = None) records created names and
# whether close() ran; list_topics(), describe_cluster() and create_topics(new_topics) mimic the aiokafka admin client.

async def test_mvp_269_ensure_topics_creates_only_missing_topics():
    admin = _FakeAdmin(existing={"solace.safety"})
    created = await ensure_topics(KafkaSettings(bootstrap_servers="kafka:9092"), admin_factory=lambda **_: admin)
    assert set(created) == {c.name for c in TOPIC_CONFIGS.values()} - {"solace.safety"} and admin.closed

async def test_mvp_269_concurrent_creation_by_another_service_is_success():
    admin = _FakeAdmin(existing=set(), create_error=_already_exists())     # the aiokafka "topic already exists" error
    assert await ensure_topics(KafkaSettings(bootstrap_servers="kafka:9092"), admin_factory=lambda **_: admin) == []

def test_mvp_269_legacy_manager_is_derived_and_has_no_dlq_topics():
    manager = TopicManager(); manager.register_solace_topics(); legacy = manager.get_registered_topics()
    assert not [n for n in legacy if n.endswith(".dlq")]
    for cfg in TOPIC_CONFIGS.values():
        assert (legacy[cfg.name].partitions, legacy[cfg.name].retention_ms) == (cfg.partitions, cfg.retention_ms)

async def test_mvp_269_topic_creation_is_retried_until_it_succeeds(monkeypatch):
    # monkeypatch solace_events.service_plane.ensure_topics with a fake that raises ConnectionError twice, then returns [];
    # plane = ServiceEventPlane("svc-269", pool=None, kafka_settings=KafkaSettings(bootstrap_servers="kafka:9092"), strict=False)
    # with a mock producer and plane._topics_backoff_base = 0.01: "topics" is in plane.background_tasks() after start();
    # within 2 s the fake has been called exactly 3 times and that task is done without an exception; stop() in finally
```
In `tests/realinfra/event-plane/test_mvp_269_topics_created_at_plane_start.py`:
```python
async def test_mvp_269_ensure_topics_applies_the_registry_settings(kafka_bootstrap):
    name = f"test.mvp269.{uuid4().hex[:8]}"
    settings, cfg = KafkaSettings(bootstrap_servers=kafka_bootstrap.bootstrap), TopicConfig(name=name, partitions=2, retention_ms=SEVEN_DAYS_MS)
    assert await ensure_topics(settings, [cfg]) == [name]
    assert await ensure_topics(settings, [cfg]) == []
    # an AIOKafkaAdminClient reports 2 partitions, replication 1 and retention.ms == "604800000" for the topic

async def test_mvp_269_plane_start_creates_the_crisis_topics(kafka_bootstrap, pg_migrated):
    # ServiceEventPlane(f"svc-269-{uuid4().hex[:6]}", pool=pg_migrated.app_pool,
    #                   kafka_settings=KafkaSettings(bootstrap_servers=kafka_bootstrap.bootstrap), strict=False).start();
    # within 30 s the broker lists "solace.safety" and "solace.users" and plane.readiness() == (True, "ok"); stop() in finally
```
- [ ] **Step 2: Run them and confirm they fail.**
  - Run: unit command with `tests/solace_events/test_mvp_269_topic_registry_and_creation.py`. Expected: FAIL at collection with `ModuleNotFoundError: No module named 'solace_events.topics'`. With the import removed, the retention test fails on `7776000000` (the 90-day default, `src/solace_events/config.py:45`), and the legacy test fails because `register_solace_topics` registers `solace.sessions.dlq` and gives `solace.safety` `31536000000` (`src/solace_infrastructure/kafka/topics.py:323-324, 341-344`). There are two registries that disagree, and nothing creates topics, so a broker with auto-create disabled would drop crisis events.
  - Run: real-infra command with `tests/realinfra/event-plane/test_mvp_269_topics_created_at_plane_start.py`. Expected: FAIL with the same `ModuleNotFoundError`.
- [ ] **Step 3: Implement** the registry values, `topics.py`, the plane's `topics` task and the derived legacy registry as listed. Check the admin response shapes of `describe_cluster` and `create_topics` against the pinned aiokafka version in the real-infra test.
- [ ] **Step 4: Run them and confirm they pass.** Run the same two commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_events`;
  - unit command with `tests/solace_infrastructure/kafka`;
  - unit command with `tests/test_syntax_and_imports.py` (every `SolaceTopic` keeps a `TOPIC_CONFIGS` entry);
  - real-infra command with `tests/realinfra/event-plane`.

  Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-06.14:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| # | Owner | Exact change | Files | Needed by |
|---|---|---|---|---|
| 1 | WS-02 | Scaffold `010_event_plane` as an empty revision on the chain (`down_revision = "009_platform_roles_rls"`). 009's `ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator` must give `solace_app` SELECT/INSERT/UPDATE/DELETE on tables created by 010. 007 must not create or alter `event_outbox` on a fresh database: guard or no-op it, since 010 owns the table. | `migrations/versions/010_event_plane.py` (scaffold only), `migrations/versions/009_platform_roles_rls.py`, `migrations/versions/007_outbox_claimed_at.py` | WS-06.1 |
| 2 | WS-05 | IF-19 fixture capabilities used here: `PgHandle.superuser_fetch(sql, *args)`; `PgHandle.create_database(name, *, migrate_to)`; `app_pool` built with the production JSONB codec; `KafkaHandle.produce(topic, BaseEvent \| dict \| bytes, *, key=None)`, `committed_offset(group_id, topic, partition=0)`, `consume_until(topic, predicate, timeout)`, `pause()`/`unpause()`. | `src/solace_testing/realinfra.py`, `tests/realinfra/conftest.py` | WS-06.1, WS-06.4 |
| 3 | WS-04 | When `spec.event_plane != 'none'`, the bootstrap calls `start_service_event_plane(spec.service_name, pool=runtime.pg_pool, kafka_settings=None, strict=is_strict_environment(spec.environment))`. In strict it translates `EventPlaneConfigError` to `StartupDependencyError`. It registers readiness check `event_plane` (critical) from `plane.readiness()` and every task from `plane.background_tasks()`, and calls `shutdown_service_event_plane()` from `stop_service_runtime`. Alert rules over this plan's metrics: `CrisisOutboxBacklog` (`solace_outbox_oldest_pending_seconds > 120`), `NotificationConsumerLag` (`max by (group, topic) (solace_consumer_lag{group="solace-group-notifications",topic="solace.safety"}) > 10 for 5m`), `CrisisConsumerStalled` (`max by (group, topic) (solace_consumer_lag{group="solace-group-notifications",topic="solace.safety"}) > 0 for 3m`; crisis-review F13: one stuck crisis turn is 4 messages, below the volume rule; every plane samples this pair, so rules use `max`, never `sum`), `DLQUnresolved` (`solace_dlq_unresolved > 0`), `CrisisEventPublishFailures` (`increase(solace_event_publish_failures_total[5m]) > 0`). The alert rules are planned in WS-04.7. | `src/solace_infrastructure/runtime/bootstrap.py`, `infrastructure/prometheus/rules/solace.yml` | WS-06.9 (closure of MVP-015/028), WS-06.10 (closure of MVP-353) |
| 4 | WS-03 | Registry entry `EVENTS_ENABLED` (services: every event-plane service; `required_in_strict: true`; `dev_default: "true"`). Set `EVENTS_ENABLED=true` and `KAFKA_BOOTSTRAP_SERVERS=kafka:9092` on orchestrator, safety, therapy, diagnosis, personality, user-service, notification and analytics. Mark `SAFETY_KAFKA_ENABLED`, `THERAPY_KAFKA_ENABLED`, notification `KAFKA_ENABLED` and the other per-service flags retired once their owners stop reading them (W1). | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml` | WS-06.9 (MVP-028 closure) |
| 5 | WS-12 | `main.py`: call `initialize_notification_bridge(...)` unconditionally and delete `_require_event_delivery_in_production` and the `settings.kafka_enabled` gating; the plane fails loud in strict. `domain/escalation.py`: bind the `notifications_sent`/`actions_taken`/`context` JSONB parameters as Python objects (no `json.dumps`), with a real-infra test asserting `jsonb_typeof(...) <> 'string'`. | `services/safety_service/src/main.py`, `services/safety_service/src/domain/escalation.py` | WS-06.9, WS-06.13 |
| 6 | WS-13 | `main.py`: drop the `KafkaConfig.enabled` gating and `_require_crisis_consumer_in_production`. `consumers.py`: subscribe through `plane.subscribe("solace.safety", [...], handler, group_id="solace-group-notifications")`. Amended by crisis-review F1 (planned in WS-13.6 and WS-13.8): a page that reached no clinician is retried by the pager's own sweep, not by raising, so delivery retries never hold the shared consume loop; the handler raises only for unexpected exceptions, and those keep the plane's retry, DLQ, idempotency and reaper. | `services/notification-service/src/main.py`, `services/notification-service/src/consumers.py` | WS-06.9 (MVP-028 end to end); MVP-007 is proven in WS-05's W1 fault E2E |
| 7 | WS-15 | `main.py`: initialize the therapy bridge unconditionally; stop reading `THERAPY_KAFKA_ENABLED`. | `services/therapy_service/src/main.py` | WS-06.9 |
| 8 | WS-14 | `postgres_repository.py`: bind `notification_channels` as a Python list (no `json.dumps`), with a real-infra test asserting `jsonb_typeof(notification_channels) = 'array'` (W2). `events.py`: `user.created` carries `email=""` and `display_name=""` (W3), so the plane's redaction never has to fire for it. | `services/user-service/src/infrastructure/postgres_repository.py`, `services/user-service/src/events.py` | WS-06.13 (MVP-267 closure), WS-06.12 |
| 9 | WS-18 | Move analytics Kafka ingestion to `plane.subscribe(..., group_id="solace-group-analytics")` and drop the `create_dead_letter_handler`/`ensure_table` calls (W2), so no legacy in-memory DLQ path remains in strict profiles. | `services/analytics-service/src/main.py` | WS-06.8 (MVP-052 completeness) |
| 10 | WS-16 | `events.py`: `trigger_indicators` carries category codes from `solace_clinical.CATEGORIES`, never `trigger_text` (their W2 task). The plane's redaction is the backstop, not the fix. | `services/diagnosis_service/src/events.py` | WS-06.12 |
| 11 | WS-01 | Pin aiokafka and prometheus-client in the locks. Per D-20 (decided 2026-10-19), aiokafka is installed at W0 entry by the controller into the dedicated venv `D:\tmp\mvp-exec\venv` only (never into anaconda), WS-01.2 re-syncs that venv to the dev lock, and the W0 barrier runs the suites in the lock-only venv `D:\tmp\mvp-exec\venv-clean`. | `requirements.in`, `requirements-dev.in`, `requirements.lock`, `requirements-dev.lock` | WS-06.4, WS-06.10 |

## New interfaces

All are additions; nothing in IF-05 is renamed:
- `solace_events.errors`:
  - `EventPlaneError` as base;
  - `EventStoreSchemaError(RuntimeError)` with `.missing`;
  - `EventPlaneConfigError` and `EventPlaneNotStartedError` (both also `RuntimeError`);
  - `MalformedEventError(NonRetryableEventError)`;
  - `safe_error_message(exc) -> str`.
- Service plane additions:
  - `start_service_event_plane(..., use_mock=False)`;
  - `get_service_event_plane(service_name=None)`;
  - `get_or_create_service_event_plane(service_name, *, pool, kafka_settings=None, use_mock=False)`;
  - `shutdown_service_event_plane(service_name=None)`;
  - `ServiceEventPlane` constructor keywords `metrics_interval_seconds` and `reaper_interval_seconds`;
  - properties `publisher` and `poller`; `background_tasks()`; `consumer_groups()`;
  - `subscribe()` is allowed after `start()`.
- `solace_events._env.is_strict_env(env=None) -> bool`, with a parity test against `is_strict_environment`.
- Store API:
  - `PostgresOutboxStore(pool, *, source_service)`;
  - keyword `claim_token` on `get_pending`, `mark_published`, `reset_to_pending`, `mark_failed` and `increment_retry`;
  - `save(record, *, conn=None)` and `pending_stats()`;
  - `delete_for_user`, `purge_published`;
  - DLQ store: `claim_redrivable`, `record_redrive_failure`, `count_unresolved_by_group`, `delete_for_user`, `purge_resolved`;
  - `PostgresIdempotencyStore.purge`;
  - `REQUIRED_COLUMNS` and `verify_event_tables`.
- `EventPublisher` additions:
  - `publish(..., *, conn=None)`;
  - `is_connected`, which comes with a background connector;
  - `create_publisher(..., source_service=None)`.
- Consumer additions:
  - `EventConsumer(..., *, dlq_store, max_attempts, retry_base_seconds, retry_max_seconds, handler_timeout_seconds)`;
  - `KafkaConsumerAdapter.seek()` and `lag()`;
  - `create_consumer(..., dlq_store=None, max_attempts=5, retry_base_seconds=None)`.
- `DeadLetterRecord` optional `event_type/event_id/user_id`, plus `raw_payload`, `kafka_partition`, `kafka_offset`, `redrive_count`, `resolved_at` and `from_consumer_failure(...)`.
- `solace_events.reaper.DLQReaper` and `ReaperStats`.
- `solace_events.retention` constants and `RetentionPurger`. `solace_events.payload_policy.redact_free_text`.
- Metric `solace_event_payload_redactions_total{service,event_type}`, beyond the IF-05 six.
- DLQ `consumer_group` label `outbox:<service>` for publisher-side dead letters.
- Topic registry (WS-06.14): `solace_events.topics.TopicSpec`, `plan_topics(configs, *, broker_count)` and `ensure_topics(kafka_settings, configs=None, *, admin_factory=None) -> list[str]`; the plane's `topics` background task and readiness detail `kafka topics: not ensured`; `TOPIC_CONFIGS` is the single registry and `TopicManager.register_solace_topics` is derived from it.

## Self-review

**Coverage**

| Register id | Severity | Task(s) |
|---|---|---|
| MVP-007 | blocker | WS-06.3, WS-06.4 |
| MVP-014 | blocker | WS-06.9 |
| MVP-015 | blocker | WS-06.9 (plus WS-04 readiness wiring and the owners' W0 main.py adoption) |
| MVP-028 | blocker | WS-06.8, WS-06.9 (plus requests 3-7) |
| MVP-353 | blocker | WS-06.10 (alert rules via request 3) |
| MVP-051 | high | WS-06.7 |
| MVP-052 | high | WS-06.8 |
| MVP-053 | high | WS-06.8 |
| MVP-054 | high | WS-06.11, WS-06.12 |
| MVP-132 | medium | WS-06.5 |
| MVP-133 | medium | WS-06.1, WS-06.2 |
| MVP-134 | medium | WS-06.3 (real Kafka poison case in WS-06.4) |
| MVP-135 | medium | WS-06.4 |
| MVP-136 | medium | WS-06.6 |
| MVP-137 | medium | WS-06.7 |
| MVP-138 | medium | WS-06.3 |
| MVP-264 | low | WS-06.6 |
| MVP-265 | low | WS-06.13 |
| MVP-266 | low | WS-06.1 |
| MVP-267 | low | WS-06.5 (event stores), WS-06.13 (closure after WS-12/WS-14) |
| MVP-268 | low | WS-06.6 |

All 21 assigned ids are covered: 5 blocker, 4 high, 7 medium, 5 low. WS-06.14 covers no WS-06 id; it is the supporting edit for MVP-269 (WS-01) requested as R-01.4(b).

REVIEW_FOCUS: 2=WS-06.3,WS-06.4,WS-06.5,WS-06.6,WS-06.7; 4=WS-06.3,WS-06.11,WS-06.12; 5=WS-06.1,WS-06.8,WS-06.9

- Focus 2 (alert-path concurrency and partial failure): a crash between the domain write and the outbox row (WS-06.5 `test_mvp_132_outbox_row_rolls_back_with_the_domain_row`), between delivery and idempotency recording (WS-06.3 `test_mvp_138_ledger_is_written_only_after_success`), a rebalance or commit failure mid-batch (WS-06.4 `test_mvp_135_commit_failure_redelivers_without_loss`), two pollers after a restart (WS-06.6 `test_mvp_264_late_poller_cannot_flip_a_row_reclaimed_by_another`) and two reapers (WS-06.7 `test_mvp_051_two_reapers_never_redrive_the_same_row_twice`). SMTP bounce and acknowledgement races are WS-13 code.
- Focus 4 (PHI sinks): DLQ rows without input values (WS-06.3 `test_mvp_134_schema_invalid_event_is_stored_with_payload_and_no_input_values`), outbox payloads with codes, not text (WS-06.12), event rows erased per user (WS-06.11).
- Focus 5 (strict boots with partial config): strict never falls back to in-memory stores and refuses an unset or loopback broker (WS-06.8), the schema check fails loud before slot 010 (WS-06.1), and a bridge without a pool fails loud in strict (WS-06.9).
- Focuses 1 and 3 do not fall in this workstream's code: detection is WS-12's, and the event tables are non-RLS by IF-03.

**Interface consistency**
- IF-05 names are used verbatim: `start_service_event_plane`, `get_service_event_plane`, `ServiceEventPlane.publish(event, *, conn=None) -> UUID`, `.subscribe(topic, event_types, handler, *, group_id, max_attempts=5)`, `.start()`, `.stop()`, `.readiness() -> tuple[bool, str]`, the three `solace_events.errors` classes, the six metric names, `erase_user_events(user_id) -> int` and `UserErasureCompletedKafkaEvent(user_id, service_name, erasure_request_id, stores, completed_at)`.
- Retry follows IF-05: 1 s, doubling, capped at 16 s, 5 attempts. The reaper runs every 60 s with a budget of 10. Message key `str(user_id)`. No `*.dlq` Kafka topic is ever produced; `dlq_topic` is a column label.
- Consumer group names follow IF-05: `solace-group-notifications`, `solace-group-analytics`, `solace-group-erasure-<service>`.
- Bridges keep `initialize_event_bridge` / `initialize_notification_bridge` and their keywords. The `_publisher` and `_poller` attributes stay readable, so services' existing wiring tests are untouched.
- `source_service` values are the `ServiceIdentity` strings.
- The fixture names consumed match WS-05's plan: `pg_migrated`, `PgHandle.superuser_fetch`, `PgHandle.create_database`, `kafka_bootstrap` with `produce`, `committed_offset`, `consume_until`, `pause` and `unpause` (all in WS-05.2, which WS-06.1, WS-06.4 and WS-06.14 depend on directly or through WS-06.1).
- `migrations/versions/010_event_plane.py` appears in WS-02.3 (create) and WS-06.1 (modify) in W0. This is the designed `migrations/versions/*` handoff: one schema lane, WS-02.3 first, which WS-06.1's `Depends on` now states.

**Open risks**
- **Critical-path lane.** The W1 phase-A1 lane (WS-06.3 to WS-06.10, then WS-06.14) is nine sequential tasks on one package. Every W1 service lane waits for WS-06.8/WS-06.9. The controller should start it at W1 entry and keep metrics (WS-06.10) and the topic registry (WS-06.14) last, because phase B does not need them. WS-03's strict-overlay `KAFKA_AUTO_CREATE_TOPICS_ENABLE=false` (WS-01's R-01.1) must land after WS-06.14.
- **AIOKafka behaviour.** The `seek`, admin-client lag and commit-failure behaviour is proven only against cp-kafka 7.5 in testcontainers. A broker-side difference (for example `list_consumer_group_offsets` output) shows up only in WS-06.10's real-infra test.
- **Partition blocking.** In-place retries block a partition for up to about 15 s, plus the handler timeout (30 s per attempt). WS-13's priority path (IF-09) and bounded channel timeouts (MVP-154) are needed so monitoring traffic never delays a crisis page.
- **Redaction masks producer bugs.** `redact_free_text` is a backstop and could hide them. The redaction counter and WS-05's W3 PHI capture scan must stay at zero for well-behaved producers. A non-zero counter is a defect in the producer's lane.
- **MVP-267 closure** depends on WS-12/WS-14 landing their JSONB fixes and tests. WS-06.13 leaves it open with a Ruling if they slip.
- **Registry topics in shared test brokers.** From WS-06.14 every real plane creates the registry topics (for example `solace.safety` with 4 partitions) on the session Kafka container. Another workstream's real-infra test that reads a fixed registry topic at partition 0 only would then miss messages keyed to other partitions. Such tests should use unique topics or read all partitions (`consume_until` does).
- **Dead code.** `src/solace_infrastructure/kafka/monitoring.py` (`KafkaMonitor`) stays unused. Lag comes from the plane. It is left in place: no register entry asks for its removal, and deleting it would also delete its tests in `tests/solace_infrastructure/kafka`.

## Review responses

- Plan check (error): W0 file `migrations/versions/010_event_plane.py` is edited by WS-02.3 and WS-06.1 in the same wave without a dependency. Fixed: WS-06.1 now depends on WS-02.3, and its Files line and the Self-review name the `migrations/versions/*` handoff (WS-02 plan, lines 302 and 1406, records the same lane).
- Plan check (error): request R-01.4 from WS-01 (needed by WS-01.7 and WS-01.12). Fixed: part (a) is in WS-06.2, which now depends on WS-01.7 and runs `tests/build/test_import_roots.py -k "WS-06"` red in Step 2 and green in Step 4. Part (b) is the new task WS-06.14 (W1 phase A1, last in the lane): `TOPIC_CONFIGS` carries the D-24 7-day retention for `solace.safety` and `solace.users`, `ensure_topics` creates missing topics at plane start with replication capped at the broker count, and `TopicManager.register_solace_topics` is derived from the registry without `*.dlq` topics. Its Produces line names WS-01.12.
- Plan check (warning): WS-06.1 modifies `010_event_plane.py`, which an earlier task or dependency does not create (it is created by WS-02.3). Fixed by the same WS-02.3 dependency.
- AGY (error/PATH): `010_event_plane.py` does not exist in the snapshot, and WS-06.1 depends only on WS-05.2. Fixed: WS-02.3 added to WS-06.1's `Depends on`. The file is absent today because WS-02.3 creates slots 010 to 021 (WS-02 plan, Task WS-02.3).
- crisis-review F1: fixed. Verified: IF-05's "about 15 s of partition blocking at most" counts only the 1 + 2 + 4 + 8 s backoff while each attempt may run `handler_timeout_seconds` (30 s), and WS-06.4's loop (like today's `consumer.py:374-400`) handled one message at a time across all partitions of the group's single consume task, so one user's failing crisis page held every other user's page. WS-06.3's Interfaces now state the real bound (about 165 s per message, a correction of the IF-05 text for the ledger Ruling). WS-06.4 processes partitions concurrently: one worker per partition while its fetch is paused, order kept, per-message commit, `RETRY` seek-back and commit-failure recovery per worker, `max_inflight_partitions` (16); a unit test and a two-partition real-Kafka test pin it. Request 6 to WS-13 is amended: crisis delivery retries move to WS-13.6's pager-owned sweep, so a delivery failure never holds the loop at all. WS-06.8 needs no change: its one consume task per group now hosts the per-partition workers.
- crisis-review F10: fixed. WS-06.12's redaction rewrote any `escalation_reason` outside `CATEGORIES` to `uncategorized`, which would erase the IF-06 form `auto:CRITICAL:suicidal_ideation` that notification parses. It now keeps `unspecified` and any value `solace_clinical.parse_escalation_reason` (WS-12.2) accepts, and rewrites anything else to `unspecified`; a new test pins both cases, and WS-06.12 depends on WS-12.2.
- crisis-review F11: fixed. Verified: no WS-06 task or request row planned WS-12's R-12-2 `record_publish_failure`. WS-06.10 now produces `solace_events.service_plane.record_publish_failure(service_name)` (increments `solace_event_publish_failures_total{service}`) with a unit test; WS-12.9 and WS-10.3's publish retrier depend on WS-06.10. R-12-2's ordering half needs no task: `get_pending` already orders by `created_at` (`postgres_stores.py:173`). Consequence for the lane: phase-B tasks WS-10.3 and WS-12.9 now wait for WS-06.10, so the controller should not leave it last in phase A1.
- crisis-review F13: fixed. WS-06.10's AIOKafka lag counted `end_offsets(assigned)` only, so a member left without partitions reported 0, and only notification-service's own plane sampled its group, so a stopped or frozen notification-service exported no lag at all. `lag()` now covers every partition of the subscribed topics from broker state, and every plane samples the crisis pair through `watch_lag=CRISIS_LAG_WATCH`; a unit test (no assignment) and a real-infra test (a plane that does not consume the group) pin it. Request 3 now asks WS-04 for `CrisisConsumerStalled` (`> 0` for 3m) and `max`-aggregated lag rules.
- decision 2026-10-19 D-20: the "installed in the local Python" assumption is gone. aiokafka is installed by the controller at W0 entry into the dedicated venv `D:\tmp\mvp-exec\venv` (system site packages over an untouched anaconda), WS-01.2 re-syncs the venv to the dev lock, and the W0 barrier also runs the suites in the lock-only venv `D:\tmp\mvp-exec\venv-clean`. Changed: the WS-01 dependency line, the decision gates of WS-06.4 and WS-06.8, and cross-workstream request 11. No test, step or interface changed; every command already used the venv interpreter.
