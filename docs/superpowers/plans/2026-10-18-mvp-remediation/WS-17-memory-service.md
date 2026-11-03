# WS-17: Memory service: isolation, durable storage, context assembly and erasure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make memory-service a per-user-isolated, durably persisted, codec-encrypted store whose context assembly returns the newest turns in order, retains crisis content permanently, sends no PHI to embedding vendors and erases every store when an account is deleted.

**Architecture:** `main.py` adopts the IF-01 runtime in W0 and composes the service through one testable seam, `build_memory_service(runtime, settings)`. `PostgresRepository` keeps its SQLAlchemy Core statements but executes them on the shared `ConnectionPoolManager` pool (RLS GUC from the request context) from W0, and in W2 its statements are aligned with the migrated schema plus slot 018 and run every PHI column through the IF-13 codec. Retrieval stays keyword/BM25 (D-04) over the caller's own candidate records only, and erasure runs through the shared `UserDataErasure` registry, fed by the IF-16 `user.deleted` consumer.

**Covers:** 24 register entries (blocker 0, high 4, medium 8, low 12).

**Waves:** 0 (runtime adoption, shared pool, the published `therapy__memory` context examples), 2 (slot 018; ownership and the orchestrator and therapy provider contracts; durable encrypted writes and reads; crisis retention; context assembly), 3 (per-user ranking; erasure; session ownership and clinician scoping; PHI egress; PHI-access audit), 5 (automatic consolidation with crisis-aware summaries; decay).

**Depends on:**
- WS-01: realinfra/integration/contract/clinical markers, the lock (SQLAlchemy 2.0.x, asyncpg 0.30, bm25s), the import-root guard.
- WS-02: slot `018_memory` scaffold (revision `018_memory`, down `017_orchestrator_checkpoints`), 009 roles and the NULLIF rewrite of the existing `memory_records` policy, duplicate `memory_records.access_count` removed from 001.
- WS-03: IF-02 registry entries and the memory-service compose block (POSTGRES_* as solace_app, REDIS_URL, WEAVIATE_ENABLED, MEMORY_EMBEDDINGS_PROVIDER).
- WS-04: IF-01 runtime package and `tests/bootstrap/test_service_conformance.py`.
- WS-05: IF-19 realinfra fixtures (`pg_migrated`, `redis_url`, WS-05.2) and the IF-20 contract framework `verify_provider` (WS-05.7).
- WS-06: IF-05 event plane, needed by the erasure consumer.
- WS-07: `acting_as_user` (W0), obo acceptance in `get_current_user` (W1), `can_access_subject`/`set_subject_context` (W3).
- WS-09: IF-13 PHI codec (W2) and IF-14 `audit_phi_access` (W3).
- WS-11: consumer contract files `tests/contracts/orchestrator__memory/{context,session_message}.json` (WS-11.12, W2).
- WS-12: IF-07 `solace_clinical` lexicon and `tests/clinical/crisis_phrases_v1.yaml` (W1).
- WS-14: IF-16 `register_user_erasure_consumer` helper (WS-14.13, W3 phase A).
- WS-05 and WS-15: the `therapy__memory` consumer contract, created by WS-05.10 (W2 phase A) from WS-17.19's published examples and read by WS-15.10's test (W2) (R-17-04).

**Owned paths:** `services/memory_service/src`, `services/memory_service/tests`, `services/memory_service/__init__.py`, `migrations/versions/018_memory.py`; new component tests under `tests/realinfra/memory-service/`.

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/memory_service/src/main.py` | Modify | IF-01 SPEC, lifespan, `build_memory_service` seam, erasure-consumer wiring, embeddings setting | 17.1, 17.2, 17.5, 17.10, 17.11, 17.13 |
| `services/memory_service/src/api.py` | Modify | Subject authorization, 503/404/500 mapping, query-free logs, PHI-access audit | 17.4, 17.5, 17.11, 17.12, 17.13, 17.14 |
| `services/memory_service/src/schemas.py` | Modify | Canonical `therapy__memory` context examples (17.19); `AddMessageRequest.role` limited to user and assistant (17.12) | 17.12, 17.19 |
| `services/memory_service/src/domain/service.py` | Modify | Persist-first writes, durable reads, crisis retention, durable context, per-user ranking, erasure, session ownership, consolidation, decay | 17.5, 17.6, 17.7, 17.9, 17.10, 17.11, 17.12, 17.13, 17.15, 17.16 |
| `services/memory_service/src/domain/models.py` | Modify | `MemoryRecord.is_safety_critical`, `MemoryPersistenceError`, `SessionOwnershipError`, `enable_decay_archival` | 17.5, 17.12, 17.16 |
| `services/memory_service/src/domain/context_assembler.py` | Modify | Display order, newest-first history, summary and facts sections | 17.8, 17.9 |
| `services/memory_service/src/domain/consolidation.py` | Modify | Crisis-aware summaries, assistant-only technique labels | 17.15 |
| `services/memory_service/src/domain/decay_manager.py` | Modify | Safety categories never decay | 17.16 |
| `services/memory_service/src/domain/erasure.py` | Modify | Postgres deleter counts the four memory tables | 17.11 |
| `services/memory_service/src/infrastructure/postgres_repo.py` | Modify | Shared-pool transport, schema-aligned statements, codec, filtered reads, erasure, decay | 17.2, 17.5, 17.6, 17.9, 17.11, 17.15, 17.16 |
| `services/memory_service/src/infrastructure/redis_cache.py` | Modify | Client injected from the runtime; erase fails loud without a client | 17.1, 17.11 |
| `services/memory_service/src/infrastructure/weaviate_repo.py` | Modify | `WEAVIATE_ENABLED`; loud init failure; lazy re-init on erase | 17.1, 17.11 |
| `services/memory_service/src/infrastructure/hybrid_search.py` | Modify | `rank_bm25` per-call index | 17.10 |
| `services/memory_service/src/events.py` | Modify | Single `solace_events` import root | 17.1 |
| `migrations/versions/018_memory.py` | Modify | RLS on session_summaries, therapeutic_events, memory_user_profiles; no user_facts table | 17.3 |
| `services/memory_service/tests/conftest.py` | Create | `recording_cpm`, `memory_service_inmem`, `memory_api`, `fake_codec` fixtures | 17.2, 17.4, 17.5 |
| `services/memory_service/tests/test_infrastructure.py` | Modify | Drop `PostgresSettings` tests; codec-aware repository construction | 17.2, 17.5 |
| `services/memory_service/tests/test_gdpr_erasure.py` | Modify | Caches purged on failure; Redis erase without client raises | 17.11 |
| `services/memory_service/tests/test_mvp_*.py` | Create | One RED-first file per task; provider-contract locks for `orchestrator__memory` (17.17) and `therapy__memory` (17.18); the published context examples (17.19) | 17.1-17.19 |
| `tests/realinfra/memory-service/conftest.py` | Create | `superuser_conn`, `seed_users`, `cpm_default`, `phi_codec` built on IF-19 fixtures | 17.3 |
| `tests/realinfra/memory-service/test_mvp_*.py` | Create | Real-schema, RLS, ciphertext and erasure proofs | 17.3, 17.5, 17.6, 17.11, 17.15, 17.16 |

## Tasks

### Task WS-17.1: Adopt the IF-01 runtime in memory main.py, keep Weaviate off by default and take Redis from the runtime
**Covers:** none (IF-01/IF-02 adoption and the D-04 default; supports MVP-270, owned by WS-01)
**Severity:** high (W0 boot gate; closes no register entry by itself)
**Wave:** 0
**Depends on:** none
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-04 (default assumed: WEAVIATE_ENABLED=false and MEMORY_EMBEDDINGS_PROVIDER=none; retrieval keyword/BM25 only)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/main.py` (delete `configure_logging`, the ProductionGuards call, the PHI-encryption block, `create_application`, `_register_middleware`, `_register_exception_handlers` and the `/`, `/ready`, `/live`, `/health` routes; add module-level `SETTINGS`, `SPEC`, `build_memory_service`, a new `lifespan` and `app = create_service_app(...)`; keep `MemoryServiceAppSettings` and `run_server`)
- Modify: `services/memory_service/src/infrastructure/redis_cache.py` (`RedisCache.__init__(self, settings=None, *, client=None)`; `initialize()` pings an injected client and sets `_initialized`)
- Modify: `services/memory_service/src/infrastructure/weaviate_repo.py` (`WeaviateSettings.enabled: bool = Field(default=False)`, read from `WEAVIATE_ENABLED`)
- Modify: `services/memory_service/src/events.py` (lines 15-17: import `solace_events.schemas`, `solace_events.publisher`, `solace_events.consumer`)
- Create: `services/memory_service/tests/test_mvp_if01_runtime_adoption.py`
- Test: `services/memory_service/tests/test_mvp_if01_runtime_adoption.py`; run-only `tests/bootstrap/test_service_conformance.py` (WS-04)

**Interfaces:**
- Consumes: IF-01 `ServiceRuntimeSpec`, `ServiceRuntime`, `create_service_app(spec, *, lifespan, routers, title)`, `start_service_runtime(app, spec)`, `stop_service_runtime(runtime)`, `runtime.pg_pool`, `runtime.redis`, `runtime.register_readiness_check(name, check, *, critical=True)`; IF-02 `MEMORY_ENVIRONMENT`, `REDIS_URL`, `WEAVIATE_ENABLED`.
- Produces: `main.SETTINGS: MemoryServiceAppSettings`; `main.SPEC` = `ServiceRuntimeSpec(service_name="memory-service", environment=SETTINGS.environment, version=SETTINGS.version, log_level=SETTINGS.log_level, debug=SETTINGS.debug, uses_postgres=True, uses_redis=True, event_plane="publish_consume", handles_phi=True, audited=True, rate_limited=False, edge=False)`; `def build_memory_service(runtime: ServiceRuntime, settings: MemoryServiceAppSettings) -> MemoryService` (the only composition seam; later tasks extend it); readiness check `memory_service` (critical).

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_if01_runtime_adoption.py`
```python
"""IF-01/IF-02/D-04 adoption for memory-service (WS-17.1)."""
from __future__ import annotations

import solace_events.schemas as canonical
from fastapi.testclient import TestClient

from services.memory_service.src import events as memory_events
from services.memory_service.src import main as memory_main


def test_if01_memory_spec_declares_every_flag_through_w5() -> None:
    spec = memory_main.SPEC
    assert spec.service_name == "memory-service"
    assert (spec.uses_postgres, spec.uses_redis) == (True, True)
    assert spec.event_plane == "publish_consume"
    assert (spec.handles_phi, spec.audited) == (True, True)
    assert (spec.rate_limited, spec.edge) == (False, False)


def test_if01_memory_main_has_no_local_platform_boilerplate() -> None:
    for name in ("configure_logging", "create_application",
                 "_register_middleware", "_register_exception_handlers"):
        assert not hasattr(memory_main, name), name


def test_if01_lifespan_registers_readiness_and_keeps_weaviate_off(monkeypatch) -> None:
    monkeypatch.delenv("WEAVIATE_ENABLED", raising=False)
    with TestClient(memory_main.app) as client:
        assert client.get("/live").status_code == 200
        checks = client.get("/ready").json()["checks"]
        service = memory_main.app.state.memory_service
    assert checks["memory_service"]["ok"] is True
    assert service._weaviate_repo is None  # D-04: constructed only when WEAVIATE_ENABLED=true


def test_mvp_270_memory_events_use_the_single_import_root() -> None:
    assert issubclass(memory_events.MemoryStoredEvent, canonical.BaseEvent)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_if01_runtime_adoption.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `AttributeError: module 'services.memory_service.src.main' has no attribute 'SPEC'`; `configure_logging` still present; `_weaviate_repo` is a `WeaviateRepository` instance (constructed unconditionally today); `issubclass` is False because events.py imports `src.solace_events`.
- [ ] **Step 3: Implement** `SPEC`, `SETTINGS`, `build_memory_service(runtime, settings)` and the lifespan in `services/memory_service/src/main.py`. The lifespan's first statement is `runtime = await start_service_runtime(app, SPEC)` and its last is `await stop_service_runtime(runtime)`; between them: `memory_service = build_memory_service(runtime, SETTINGS)`, `await memory_service.initialize()`, `app.state.memory_service = memory_service`, `app.state.settings = SETTINGS`, `runtime.register_readiness_check("memory_service", _service_ready)` where the check returns `(memory_service._initialized, "initialized" or "not_initialized")`, then after `yield` `await memory_service.shutdown()`. No try/except around runtime components. `build_memory_service` creates today's collaborators (DecayManager, ContextAssembler, ConsolidationPipeline, HybridSearchEngine) plus `RedisCache(RedisSettings(), client=runtime.redis)` only when `runtime.redis is not None`, `WeaviateRepository(ws)` only when `WeaviateSettings().enabled`, and `PostgresRepository()` only when `runtime.pg_pool is not None`. `app = create_service_app(SPEC, lifespan=lifespan, routers=[(router, "/api/v1/memory")], title="Solace-AI Memory Service")`. In `RedisCache`, an injected client is stored as `_client` and `initialize()` awaits `client.ping()`; the legacy host/port path stays for callers that pass no client (main.py never does).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suites, one process each.
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`, then the same with `tests/integration/test_session_lifecycle.py tests/integration/test_event_schemas.py`, then with `tests/bootstrap/test_service_conformance.py -k memory`.
Expected: all pass. `tests/integration/test_phi_encryption_activation.py` must already have had its `configure_phi_encryption` AST check retired by WS-04 (cross-workstream request R-17-03); if it still asserts that call, stop and record a ledger Ruling instead of editing that file.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.2: Run PostgresRepository on the shared ConnectionPoolManager pool (transport only)
**Covers:** none (IF-03 one pool per service with the RLS GUC set by ConnectionPoolManager; D-02 solace_app over verified TLS)
**Severity:** high (W0 exit: strict services connect as solace_app with verify-full TLS)
**Wave:** 0
**Depends on:** WS-17.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-02 (default assumed: one pool per service, solace_app, NOBYPASSRLS)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/infrastructure/postgres_repo.py` (delete `PostgresSettings`, `create_async_engine`, `async_sessionmaker`, `NullPool` and the engine fields; `__init__(self, *, pool_name: str = "default")`; `session()` yields `_CoreSession` inside `ConnectionPoolManager.transaction(self._pool_name)`; new private `_CoreSession`, `_CoreResult`, `_CoreRow`; `initialize()` runs `SELECT 1` through `ConnectionPoolManager.acquire`; `close()` is a no-op; `get_statistics()` reports `pool` instead of `database`/`host`. The `Table` objects and every method body stay exactly as they are.)
- Modify: `services/memory_service/src/main.py` (`build_memory_service` keeps calling `PostgresRepository()`)
- Modify: `services/memory_service/tests/test_infrastructure.py` (drop `TestPostgresSettings` and the `PostgresSettings` import; `TestPostgresRepository` constructs `PostgresRepository()` and asserts `get_statistics()["pool"] == "default"`)
- Create: `services/memory_service/tests/conftest.py` (`RecordingConnection` and fixture `recording_cpm`)
- Create: `services/memory_service/tests/test_mvp_if03_repo_shared_pool.py`
- Test: `services/memory_service/tests/test_mvp_if03_repo_shared_pool.py`, `services/memory_service/tests/test_infrastructure.py`

**Interfaces:**
- Consumes: `solace_infrastructure.database.connection_manager.ConnectionPoolManager.acquire(pool_name)` and `.transaction(pool_name)` (IF-03; they set and reset `app.current_user_id` from `solace_common.request_context`).
- Produces: `PostgresRepository(*, pool_name: str = "default")`; `PostgresRepository.session()` async context manager yielding an object with `async execute(stmt) -> result` (`fetchone()`, `fetchall()`, `scalar()`, `rowcount`; rows expose `_mapping`), so existing statement code and the `session()`-stub tests (`test_rev29_a4_reconcile.py`, `test_decay_unit_consistency.py`) keep working; conftest fixture `recording_cpm` exposing `.connection.executed: list[tuple[str, tuple]]`, `.connection.fetched`, `.connection.rows`, `.pool_names`, `.closed`.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_if03_repo_shared_pool.py` (the fixture lives in the new `conftest.py`: it monkeypatches `ConnectionPoolManager.acquire` and `.transaction` with `asynccontextmanager` fakes that append the pool name to `pool_names` and yield one `RecordingConnection`, and `close_pool`/`close_all_pools` with recorders that append to `closed`; `RecordingConnection.execute(sql, *args)` records and returns `"INSERT 0 1"`, `fetch` returns `rows`, `fetchrow` returns `rows[0]` or None, `fetchval` returns `None`)
```python
"""IF-03: memory's repository uses the one shared pool (WS-17.2)."""
from __future__ import annotations

import inspect
from decimal import Decimal
from uuid import uuid4

from services.memory_service.src.domain.models import MemoryRecord
from services.memory_service.src.infrastructure import postgres_repo
from services.memory_service.src.infrastructure.postgres_repo import PostgresRepository


def test_if03_repository_builds_no_private_engine() -> None:
    source = inspect.getsource(postgres_repo)
    assert "create_async_engine" not in source
    assert "async_sessionmaker" not in source


async def test_if03_writes_run_in_a_connection_pool_manager_transaction(recording_cpm) -> None:
    record = MemoryRecord(user_id=uuid4(), session_id=uuid4(), content="synthetic note",
                          importance_score=Decimal("0.5"))
    await PostgresRepository().store_memory_record(record)
    assert recording_cpm.pool_names == ["default"]
    sql, args = recording_cpm.connection.executed[0]
    assert sql.startswith("INSERT INTO memory_records")
    assert "$1" in sql and record.record_id in args


async def test_if03_initialize_probes_and_close_leaves_the_runtime_pool_open(recording_cpm) -> None:
    repo = PostgresRepository()
    await repo.initialize()
    await repo.close()
    assert recording_cpm.connection.fetched[0][0] == "SELECT 1"
    assert recording_cpm.closed == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_if03_repo_shared_pool.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `create_async_engine` is in the module source; the store test errors with a connection failure from the private SQLAlchemy engine (ConnectionPoolManager is never called); `initialize()` never issues `SELECT 1` through the pool.
- [ ] **Step 3: Implement** `_CoreSession.execute(stmt)` in `postgres_repo.py`: compile with `sqlalchemy.dialects.postgresql.asyncpg.PGDialect_asyncpg()` and `compile_kwargs={"render_postcompile": True}`; build `params = dict(compiled.params)`; for `Insert` statements fill Python-side column defaults where the param is `None` (`d.arg(None) if d.is_callable else d.arg`); `args = [params[name] for name in compiled.positiontup]`; `Select` runs `conn.fetch(compiled.string, *args)`, every other statement runs `conn.execute(...)` and parses the row count from the status string. This is a transport change only: keep every statement, including the `metadata` column and the `user_facts` table, because WS-17.5, WS-17.6 and WS-17.11 fix those with their own RED tests; fixing them here would make those tests pass before their fix.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`, then `tests/bootstrap/test_service_conformance.py -k memory` in its own process. Expected: all pass (the `session()`-stub decay tests and the `func.exp(` source check in `test_sprint5_regression.py` are unaffected).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.3: Slot 018 puts every memory PHI table under RLS and records the user_facts decision
**Covers:** none (schema prerequisite for MVP-060 and MVP-062: facts persist as `memory_records` rows, there is no `user_facts` table)
**Severity:** high
**Wave:** 2 (phase A, schema lane, after slot 017)
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: Alembic is the schema of record; repositories adapt; facts are `memory_records` rows with tier `tier_5_semantic` and content_type `fact`); D-02 (solace_app, NOBYPASSRLS)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/018_memory.py` (`upgrade()`/`downgrade()` bodies only; `revision`/`down_revision` untouched)
- Create: `tests/realinfra/memory-service/conftest.py` (fixtures `superuser_conn`, `seed_users`, `cpm_default`, `phi_codec`)
- Create: `tests/realinfra/memory-service/test_mvp_018_memory_slot.py`
- Test: `tests/realinfra/memory-service/test_mvp_018_memory_slot.py`, `tests/migrations/test_chain_real_pg.py` (WS-02, run only)

**Interfaces:**
- Consumes: IF-19 `pg_migrated -> PgHandle(dsn_app, dsn_migrator, dsn_superuser, app_pool)`; IF-03 NULLIF policy form.
- Produces: RLS enabled and forced on `session_summaries`, `therapeutic_events`, `memory_user_profiles` with policy `<table>_by_user` USING and WITH CHECK `user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid`. Realinfra conftest fixtures reused by WS-17.5, 17.6, 17.11, 17.15, 17.16: `superuser_conn` (asyncpg connection to `dsn_superuser`), `seed_users` (async `seed_users(n) -> list[UUID]` inserting synthetic users as superuser with `id, created_at, updated_at, encryption_key_id, email, password_hash, display_name`; extend the column list if slot 012 adds a NOT NULL column without a default, the asyncpg error names it), `cpm_default` (sets POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE from `urllib.parse.urlsplit(pg_migrated.dsn_app)` plus `POSTGRES_SSL_MODE=disable`, calls `ConnectionPoolManager.close_all_pools()` then `get_pool("default")`, closes all pools on teardown), `phi_codec` (sets synthetic `ENCRYPTION_MASTER_KEY` = a fixed 64-hex test value, `ENCRYPTION_SEARCH_HASH_SALT`, `ENCRYPTION_KEY_ID=test-k1`, returns `solace_security.phi_codec.get_phi_codec()`).

- [ ] **Step 1: Write the failing test** in `tests/realinfra/memory-service/test_mvp_018_memory_slot.py`
```python
"""Slot 018: memory PHI tables are RLS-isolated per user (WS-17.3)."""
from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]

RLS_TABLES = ("memory_records", "session_summaries", "therapeutic_events", "memory_user_profiles")
INSERT_SUMMARY = (
    "INSERT INTO session_summaries (id, user_id, created_at, updated_at, encryption_key_id, "
    "session_id, session_number, summary_text, session_date) "
    "VALUES ($1, $2, now(), now(), 'test-k1', $3, 1, 'enc:v1:synthetic', now())"
)


async def test_slot018_rls_is_forced_with_a_nullif_policy(superuser_conn) -> None:
    rows = await superuser_conn.fetch(
        "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, p.qual, p.with_check "
        "FROM pg_class c JOIN pg_policies p ON p.tablename = c.relname "
        "WHERE c.relname = ANY($1::text[])", list(RLS_TABLES))
    by_table = {r["relname"]: r for r in rows}
    assert set(by_table) == set(RLS_TABLES)
    for row in by_table.values():
        assert row["relrowsecurity"] and row["relforcerowsecurity"]
        assert "NULLIF" in row["qual"].upper() and "NULLIF" in row["with_check"].upper()


async def test_slot018_solace_app_cannot_write_or_read_another_users_summary(pg_migrated, seed_users) -> None:
    owner, intruder = await seed_users(2)
    conn = await asyncpg.connect(pg_migrated.dsn_app)
    try:
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", str(intruder))
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute(INSERT_SUMMARY, uuid4(), owner, uuid4())
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", str(owner))
        await conn.execute(INSERT_SUMMARY, uuid4(), owner, uuid4())
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", str(intruder))
        assert await conn.fetchval("SELECT count(*) FROM session_summaries") == 0
    finally:
        await conn.close()


async def test_slot018_records_the_no_user_facts_decision(superuser_conn) -> None:
    # Guard, not a defect test: it passes today and must keep passing.
    assert await superuser_conn.fetchval("SELECT to_regclass('public.user_facts')") is None
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/memory-service/test_mvp_018_memory_slot.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `session_summaries`, `therapeutic_events` and `memory_user_profiles` have no policy (only `memory_records` appears); the intruder's INSERT for the owner succeeds (no `InsufficientPrivilegeError`).
- [ ] **Step 3: Implement** `upgrade()` in `018_memory.py`: for each of the three tables run `ALTER TABLE ... ENABLE ROW LEVEL SECURITY`, `... FORCE ROW LEVEL SECURITY`, `DROP POLICY IF EXISTS <t>_by_user`, `CREATE POLICY <t>_by_user ... USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid) WITH CHECK (same)`; `downgrade()` drops the three policies and runs `NO FORCE` and `DISABLE` in reverse order. The `memory_records` policy belongs to 009 (WS-02) and is not touched here. No `user_facts` table and no new columns: repositories adapt to the schema (D-01).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations/test_chain_real_pg.py -o "addopts=" -p no:logging -q`. Expected: all pass (upgrade head, downgrade base, upgrade head). WS-02 reviews the slot diff.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.4: The memory API authorizes the caller's own UUID; HTTP-level tests
**Covers:** MVP-059
**Severity:** high
**Wave:** 2
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-21 (default assumed: the existing CLINICIAN/ADMIN bypass stays until WS-17.12 replaces it with `can_access_subject`)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/api.py` (new `_authorize(current_user, subject_user_id, *, staff_roles)`; the comparisons at lines 55, 80, 111, 138, 162, 187, 214, 243 and 280 all call it; DELETE passes `staff_roles=(Role.ADMIN.value,)`)
- Modify: `services/memory_service/tests/conftest.py` (fixtures `memory_service_inmem` and `memory_api`)
- Create: `services/memory_service/tests/test_mvp_059_api_ownership.py`
- Test: `services/memory_service/tests/test_mvp_059_api_ownership.py`

**Interfaces:**
- Consumes: `solace_security.middleware.AuthenticatedUser`, `get_current_user` (overridden in the fixture), `solace_security.auth.TokenType`.
- Produces: `_authorize(current_user: AuthenticatedUser, subject_user_id: UUID, *, staff_roles: tuple[str, ...]) -> None` (403 unless `str(subject_user_id) == str(current_user.user_id)` or a staff role is held); fixture `memory_service_inmem() -> MemoryService` (in-memory, consolidation and decay off); fixture `memory_api(user_id: str, roles: list[str] | None = None) -> TestClient` (router at `/api/v1/memory`, `app.state.memory_service = memory_service_inmem`, `get_current_user` overridden with `AuthenticatedUser(user_id=user_id, token_type=TokenType.ACCESS, roles=roles or ["user"])`). The provider-contract verifications with real obo tokens are WS-17.17 (`orchestrator__memory`, for WS-11.12) and WS-17.18 (`therapy__memory`); they were split out so this task and the W2 chain behind it (WS-17.5 to WS-17.9) do not wait on the consumers' contract files.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_059_api_ownership.py`
```python
"""MVP-059: a user's own token reaches their own memory (WS-17.4)."""
from __future__ import annotations

from uuid import uuid4

import pytest

OWN_ROUTES = [
    ("/api/v1/memory/context", lambda u: {"user_id": u, "current_message": "synthetic hello"}),
    ("/api/v1/memory/store", lambda u: {"user_id": u, "content": "synthetic note"}),
    ("/api/v1/memory/retrieve", lambda u: {"user_id": u}),
    ("/api/v1/memory/session/start", lambda u: {"user_id": u}),
    ("/api/v1/memory/session/end", lambda u: {"user_id": u, "session_id": str(uuid4())}),
    ("/api/v1/memory/session/message",
     lambda u: {"user_id": u, "session_id": str(uuid4()), "role": "user", "content": "synthetic"}),
    ("/api/v1/memory/consolidate", lambda u: {"user_id": u, "session_id": str(uuid4())}),
]


@pytest.mark.parametrize("path,body", OWN_ROUTES)
def test_mvp_059_own_user_token_is_not_forbidden(memory_api, path, body) -> None:
    user_id = str(uuid4())
    response = memory_api(user_id).post(path, json=body(user_id))
    assert response.status_code in (200, 201), response.text


def test_mvp_059_own_profile_and_self_delete(memory_api) -> None:
    user_id = str(uuid4())
    client = memory_api(user_id)
    assert client.get(f"/api/v1/memory/profile/{user_id}").status_code == 200
    assert client.delete(f"/api/v1/memory/user/{user_id}").status_code == 204


def test_mvp_059_another_users_memory_is_still_forbidden(memory_api) -> None:
    client = memory_api(str(uuid4()))
    assert client.post("/api/v1/memory/context", json={"user_id": str(uuid4())}).status_code == 403
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_059_api_ownership.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Every own-user request returns 403 (`UUID != str` is always True, `services/memory_service/src/api.py:55`) and self-delete returns 403 (MVP-059); the cross-user case passes as a guard.
- [ ] **Step 3: Implement** `_authorize` in `services/memory_service/src/api.py` and call it in every route before any service call; compare string forms so pydantic `UUID` bodies and `AuthenticatedUser.user_id: str` agree; keep the existing staff bypass unchanged (WS-17.12 scopes it).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.5: Memory writes match slot 018 through the PHI codec, and write failures are surfaced
**Covers:** MVP-060
**Severity:** high
**Wave:** 2
**Depends on:** WS-17.3, WS-17.4, WS-09.7, WS-04.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: repository SQL matches the Alembic schema; no `user_facts` table)
**Real infra:** yes
**Files:**
- Modify: `services/memory_service/src/infrastructure/postgres_repo.py` (`memory_records` and `session_summaries` Table objects mirror the migrated columns with no Python-side defaults: memory_records `id, user_id, session_id, tier, content, content_type, retention_category, importance_score, emotional_valence, record_metadata (JSONB), retention_strength, accessed_at, is_archived, is_safety_critical, created_at, updated_at, encryption_key_id`; session_summaries `id, user_id, session_id, session_number, summary_text, key_topics, message_count, duration_minutes, session_date, retention_strength, created_at, updated_at, encryption_key_id`; `from solace_security.phi_codec import PHICodec` (the WS-09.15 ADOPTERS marker for this file); `FieldCodec` Protocol; `MEMORY_PHI_FIELDS`; `__init__(self, *, codec: PHICodec | FieldCodec, pool_name: str = "default")` (the union keeps the import used and lets the tests pass `FakeCodec`); no `get_phi_encryptor`, `configure_phi_encryption`, `FieldEncryptor` or `create_field_encryptor` anywhere in `services/memory_service/src`; rewrite `store_memory_record`, `store_session_summary` (upsert on `session_id` via `sqlalchemy.dialects.postgresql.insert(...).on_conflict_do_update`) and `store_user_fact` (now a `memory_records` row). Do not touch the read methods, the `user_facts` Table or `delete_user_data`: WS-17.6 and WS-17.11 own them.)
- Modify: `services/memory_service/src/domain/service.py` (`store_memory`, `add_message`, `consolidate`: durable write first, cache second; a failure raises `MemoryPersistenceError` and logs `memory_persist_failed` with `tier` and `error_type` only, never `str(exc)`)
- Modify: `services/memory_service/src/domain/models.py` (`MemoryRecord.is_safety_critical: bool = False`; `class MemoryPersistenceError(RuntimeError)`)
- Modify: `services/memory_service/src/api.py` (catch `MemoryPersistenceError` in every route and raise `HTTPException(503, detail="Memory store unavailable")`)
- Modify: `services/memory_service/src/main.py` (`build_memory_service`: `PostgresRepository(codec=runtime.phi_codec)` only when `runtime.pg_pool is not None and runtime.phi_codec is not None`; when `is_strict_environment(settings.environment)` and either is `None`, raise `StartupDependencyError("memory-service requires Postgres and the PHI codec")` instead of running without durable storage)
- Modify: `services/memory_service/tests/conftest.py` (`FakeCodec` and fixture `fake_codec`)
- Modify: `services/memory_service/tests/test_infrastructure.py` (`TestPostgresRepository` passes `codec=FakeCodec()`)
- Create: `services/memory_service/tests/test_mvp_060_write_path.py`
- Create: `tests/realinfra/memory-service/test_mvp_060_memory_write_path.py`
- Test: both new files

**Interfaces:**
- Consumes: IF-13 `PHICodec.encrypt(value, *, field, owner_id)`, `.decrypt(...)`, `.active_key_id`; IF-01 `runtime.phi_codec`, `is_strict_environment`, `StartupDependencyError` (`from solace_infrastructure.runtime import ...`); `solace_common.request_context.acting_as_user`.
- Produces: the WS-09.15 (CR-10) adoption for `services/memory_service/src/infrastructure/postgres_repo.py`: the file imports `solace_security.phi_codec`, writes `codec.active_key_id` to `encryption_key_id`, and memory code calls none of the retired encryptor helpers; a strict build without Postgres or the codec fails startup. `class FieldCodec(Protocol)` with `active_key_id: str`, `encrypt(value: str | None, *, field: str, owner_id: str | UUID) -> str | None`, `decrypt(...)`; `MEMORY_PHI_FIELDS = {"memory_records": ("content", "record_metadata"), "session_summaries": ("summary_text",)}` with AAD field names `"memory_records.content"`, `"memory_records.record_metadata"`, `"session_summaries.summary_text"` and `owner_id=user_id`; `record_metadata` is stored as a JSON string holding `codec.encrypt(json.dumps(metadata, default=str), ...)`; `MemoryPersistenceError`; HTTP 503 `{"detail": "Memory store unavailable"}` on any durable failure (IF-11 consumers treat it as degraded `memory_unavailable`). `FakeCodec` (conftest): `active_key_id = "fake-k1"`, encrypt returns `"enc:v1:fake-k1:" + urlsafe_b64encode(f"{field}|{owner_id}|{value}")`, decrypt reverses it and raises `ValueError("AAD mismatch")` when field or owner differ.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_060_write_path.py`
```python
"""MVP-060: memory writes reach the migrated schema, encrypted, or fail loudly (WS-17.5)."""
from __future__ import annotations

import ast
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from services.memory_service.src.domain.models import MemoryRecord, MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.infrastructure.postgres_repo import PostgresRepository


class _FailingRepo:
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def store_memory_record(self, record) -> None:
        raise RuntimeError("write rejected")


def _store_kwargs(user_id):
    return dict(user_id=user_id, session_id=None, content="synthetic", content_type="message",
                tier="tier_3_session", retention_category="medium_term",
                importance_score=Decimal("0.5"), metadata={})


async def test_mvp_060_write_failure_is_raised_and_not_cached() -> None:
    service = MemoryService(settings=MemoryServiceSettings(enable_auto_consolidation=False, enable_decay=False),
                            postgres_repo=_FailingRepo())
    user_id = uuid4()
    with pytest.raises(Exception) as exc_info:
        await service.store_memory(**_store_kwargs(user_id))
    assert type(exc_info.value).__name__ == "MemoryPersistenceError"
    assert service._tier_3_session.get(user_id, []) == []


def test_mvp_060_api_returns_503_on_a_failed_durable_write(memory_api, memory_service_inmem) -> None:
    memory_service_inmem._postgres_repo = _FailingRepo()
    user_id = str(uuid4())
    response = memory_api(user_id).post("/api/v1/memory/store", json={"user_id": user_id, "content": "synthetic"})
    assert response.status_code == 503
    assert "write rejected" not in response.text


def _insert_values(sql: str, args: tuple) -> dict:
    columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
    return dict(zip(columns, args))


async def test_mvp_060_insert_uses_migrated_columns_and_the_codec(recording_cpm, fake_codec) -> None:
    user_id = uuid4()
    record = MemoryRecord(user_id=user_id, session_id=uuid4(), content="synthetic sadness note",
                          metadata={"role": "user"})
    await PostgresRepository(codec=fake_codec).store_memory_record(record)
    values = _insert_values(*recording_cpm.connection.executed[0])
    assert "metadata" not in values
    assert values["encryption_key_id"] == "fake-k1"
    assert {"updated_at", "accessed_at", "is_safety_critical"} <= set(values)
    assert fake_codec.decrypt(values["content"], field="memory_records.content", owner_id=user_id) == "synthetic sadness note"
    assert json.loads(fake_codec.decrypt(values["record_metadata"], field="memory_records.record_metadata",
                                         owner_id=user_id)) == {"role": "user"}


LEGACY = {"configure_phi_encryption", "get_phi_encryptor", "FieldEncryptor", "create_field_encryptor"}


def test_cr10_repository_adopts_the_if13_codec_and_no_legacy_encryptor() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    repo_tree = ast.parse((src / "infrastructure" / "postgres_repo.py").read_text(encoding="utf-8"))
    assert "solace_security.phi_codec" in {n.module for n in ast.walk(repo_tree) if isinstance(n, ast.ImportFrom)}
    offenders = []
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        names |= {getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if names & LEGACY:
            offenders.append(path.name)
    assert offenders == []


def test_mvp_060_strict_build_without_the_codec_fails_instead_of_dropping_durability() -> None:
    from services.memory_service.src import main as memory_main
    from solace_infrastructure.runtime import StartupDependencyError

    runtime = SimpleNamespace(pg_pool=object(), redis=None, phi_codec=None, event_plane=None)
    with pytest.raises(StartupDependencyError):
        memory_main.build_memory_service(runtime, memory_main.SETTINGS.model_copy(update={"environment": "staging"}))
```
and in `tests/realinfra/memory-service/test_mvp_060_memory_write_path.py`
```python
"""MVP-060 on the migrated schema as solace_app under RLS (WS-17.5)."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from services.memory_service.src import main as memory_main
from services.memory_service.src.domain.models import MemoryRecord
from solace_common.request_context import acting_as_user

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


def _service(phi_codec):
    runtime = SimpleNamespace(pg_pool=object(), redis=None, phi_codec=phi_codec, event_plane=None)
    return memory_main.build_memory_service(runtime, memory_main.SETTINGS)


async def test_mvp_060_store_persists_a_ciphertext_row(cpm_default, seed_users, phi_codec, superuser_conn) -> None:
    (user_id,) = await seed_users(1)
    service = _service(phi_codec)
    with acting_as_user(str(user_id)):
        result = await service.store_memory(
            user_id=user_id, session_id=uuid4(), content="synthetic worry about exams", content_type="message",
            tier="tier_3_session", retention_category="medium_term", importance_score=Decimal("0.5"),
            metadata={"role": "user"})
    row = await superuser_conn.fetchrow(
        "SELECT content, record_metadata::text AS meta, encryption_key_id, is_safety_critical "
        "FROM memory_records WHERE id = $1", result.record_id)
    assert row is not None
    assert row["content"].startswith("enc:v1:") and "exams" not in row["content"]
    assert row["meta"].startswith('"enc:v1:')
    assert row["encryption_key_id"] == phi_codec.active_key_id
    assert row["is_safety_critical"] is False


async def test_mvp_060_summary_and_fact_land_in_migrated_tables(cpm_default, seed_users, phi_codec, superuser_conn) -> None:
    (user_id,) = await seed_users(1)
    repo = _service(phi_codec)._postgres_repo
    session_id = uuid4()
    with acting_as_user(str(user_id)):
        await repo.store_session_summary({"session_id": session_id, "user_id": user_id, "session_number": 1,
                                          "summary_text": "synthetic summary", "message_count": 4})
        await repo.store_user_fact({"user_id": user_id, "category": "general", "content": "synthetic fact",
                                    "importance": Decimal("0.7"), "source_session_id": session_id})
    assert await superuser_conn.fetchval("SELECT count(*) FROM session_summaries WHERE user_id = $1", user_id) == 1
    assert await superuser_conn.fetchval(
        "SELECT count(*) FROM memory_records WHERE user_id = $1 AND tier = 'tier_5_semantic' "
        "AND content_type = 'fact'", user_id) == 1


async def test_mvp_060_rls_rejects_a_write_for_another_user(cpm_default, seed_users, phi_codec) -> None:
    owner, intruder = await seed_users(2)
    repo = _service(phi_codec)._postgres_repo
    with acting_as_user(str(intruder)):
        with pytest.raises(Exception) as exc_info:
            await repo.store_memory_record(MemoryRecord(user_id=owner, content="synthetic"))
    assert "row-level security" in str(exc_info.value)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_060_write_path.py -o "addopts=" -p no:logging -q`, then `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/memory-service/test_mvp_060_memory_write_path.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `store_memory` does not raise and caches the record (the exception is logged and `stored=True` returned); the API returns 201; `PostgresRepository(codec=...)` raises `TypeError: unexpected keyword argument 'codec'`; `postgres_repo.py` has no `solace_security.phi_codec` import (the CR-10 test fails on its first assertion; the legacy-name half passes as a guard because WS-17.1 deleted the `configure_phi_encryption` block from `main.py`); the strict build returns a service with no Postgres repository instead of raising; on real Postgres the row is `None` because the INSERT fails with `UndefinedColumnError: column "metadata"` and is swallowed; the summary insert fails on NOT NULL `encryption_key_id`; the fact insert fails on the missing `user_facts` table.
- [ ] **Step 3: Implement** the schema-aligned Table objects and `PostgresRepository(*, codec, pool_name="default")` in `postgres_repo.py`: every INSERT supplies all NOT NULL columns that have no server default (`id, user_id, tier, content, content_type, retention_category, accessed_at, created_at, updated_at, encryption_key_id` for memory_records; `id, user_id, session_id, session_number, summary_text, session_date, created_at, updated_at, encryption_key_id` for summaries), with `encryption_key_id=codec.active_key_id`, `updated_at=now`, `key_topics=[]` (minimum necessary: nothing reads it); `store_user_fact` writes `tier="tier_5_semantic"`, `content_type="fact"`, `retention_category=fact.get("retention", "long_term")` and puts `category` and `source_session_id` inside the encrypted metadata. In `service.py` call the repository before `_store_to_tier`/appending to tier lists, and wrap repository exceptions in `MemoryPersistenceError` with the original as `__cause__`. In `main.py`, `build_memory_service` checks `is_strict_environment(settings.environment)` before composing and raises `StartupDependencyError` when `runtime.pg_pool` or `runtime.phi_codec` is `None`.
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`; then `tests/integration/test_session_lifecycle.py` in its own process; then the realinfra directory `tests/realinfra/memory-service` with `$env:SOLACE_REQUIRE_REALINFRA="1"`; then `tests/solace_security/test_mvp_055_codec_adoption.py` if WS-09.15 has landed (the memory ADOPTERS row must pass). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.6: Memory reads decode the migrated schema, filter before LIMIT and never fall back silently
**Covers:** MVP-061, MVP-283, MVP-149
**Severity:** high
**Wave:** 2
**Depends on:** WS-17.5
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: tier 4 reads come from `session_summaries`, tier 5 facts from `memory_records`)
**Real infra:** yes
**Files:**
- Modify: `services/memory_service/src/infrastructure/postgres_repo.py` (`get_user_records(user_id, *, tier=None, session_id=None, since=None, min_importance=None, limit=50, offset=0)` with every filter in SQL, newest first, decrypting `content` and `record_metadata`; `get_session_records(session_id)` decrypts; `get_user_summaries(user_id, *, limit=10, exclude_session_id=None)` decrypts `summary_text`; `get_user_facts(user_id, *, limit=20)` reads `memory_records` tier-5 fact rows)
- Modify: `services/memory_service/src/domain/service.py` (`retrieve_memories` passes `session_id`, `since` (from `time_range_hours`) and `min_importance` to the repository, fetches `max(limit * 5, 100)` candidates when `query` is set, reads tier 4 from summaries, raises `MemoryPersistenceError` instead of falling back to the process cache when a repository is configured; `_row_to_memory_record` maps `id` to `record_id` and `record_metadata` to `metadata`; `get_user_profile` takes facts from `get_user_facts`)
- Create: `services/memory_service/tests/test_mvp_061_read_path.py`
- Create: `tests/realinfra/memory-service/test_mvp_061_memory_read_path.py`
- Test: both new files

**Interfaces:**
- Consumes: `MemoryPersistenceError`, `FieldCodec`, realinfra fixtures from WS-17.3.
- Produces: repository rows keyed by schema column names with PHI decrypted (`id`, `user_id`, `session_id`, `tier`, `content`, `content_type`, `retention_category`, `importance_score`, `emotional_valence`, `record_metadata` (dict), `retention_strength`, `is_safety_critical`, `created_at`, `accessed_at`); the keyword signatures above (WS-17.9 and WS-17.15 call them).

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_061_read_path.py`
```python
"""MVP-061/MVP-283: durable reads decode the schema and filter before LIMIT (WS-17.6)."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService

SETTINGS = MemoryServiceSettings(enable_auto_consolidation=False, enable_decay=False)


def test_mvp_061_row_converter_reads_schema_keys() -> None:
    now = datetime.now(timezone.utc)
    row = {"id": uuid4(), "user_id": uuid4(), "session_id": None, "tier": "tier_3_session",
           "content": "synthetic", "content_type": "message", "retention_category": "medium_term",
           "importance_score": 0.5, "emotional_valence": None, "record_metadata": {"role": "user"},
           "retention_strength": 1.0, "is_safety_critical": False, "created_at": now, "accessed_at": now}
    record = MemoryService._row_to_memory_record(row)
    assert record.record_id == row["id"]
    assert record.metadata == {"role": "user"}


class _RecordingRepo:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def get_user_records(self, user_id, **kwargs):
        self.calls.append(kwargs)
        return []


async def test_mvp_283_filters_reach_the_query_instead_of_running_after_limit() -> None:
    repo = _RecordingRepo()
    service = MemoryService(settings=SETTINGS, postgres_repo=repo)
    session_id = uuid4()
    await service.retrieve_memories(user_id=uuid4(), session_id=session_id, tiers=["tier_3_session"],
                                    query=None, limit=20, min_importance=Decimal("0.2"), time_range_hours=48)
    call = repo.calls[0]
    assert call["session_id"] == session_id
    assert call["since"] is not None
    assert call["min_importance"] == Decimal("0.2")


async def test_mvp_061_read_failure_is_not_masked_by_the_process_cache() -> None:
    class _BrokenRepo:
        async def get_user_records(self, user_id, **kwargs):
            raise RuntimeError("read failed")

    service = MemoryService(settings=SETTINGS, postgres_repo=_BrokenRepo())
    with pytest.raises(Exception) as exc_info:
        await service.retrieve_memories(user_id=uuid4(), session_id=None, tiers=["tier_3_session"], query=None,
                                        limit=5, min_importance=Decimal("0"), time_range_hours=None)
    assert type(exc_info.value).__name__ == "MemoryPersistenceError"
```
and in `tests/realinfra/memory-service/test_mvp_061_memory_read_path.py`
```python
"""MVP-061/MVP-283/MVP-149 against the migrated schema (WS-17.6)."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from services.memory_service.src import main as memory_main
from solace_common.request_context import acting_as_user

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


def _service(phi_codec):
    runtime = SimpleNamespace(pg_pool=object(), redis=None, phi_codec=phi_codec, event_plane=None)
    return memory_main.build_memory_service(runtime, memory_main.SETTINGS)


async def _store(service, user_id, session_id, text, tier="tier_3_session"):
    return await service.store_memory(user_id=user_id, session_id=session_id, content=text, content_type="message",
                                      tier=tier, retention_category="medium_term",
                                      importance_score=Decimal("0.5"), metadata={"role": "user"})


async def test_mvp_283_older_session_is_found_after_a_restart(cpm_default, seed_users, phi_codec) -> None:
    (user_id,) = await seed_users(1)
    old_session, new_session = uuid4(), uuid4()
    with acting_as_user(str(user_id)):
        writer = _service(phi_codec)
        await _store(writer, user_id, old_session, "synthetic note about my sister")
        for i in range(25):
            await _store(writer, user_id, new_session, f"synthetic filler {i}")
        fresh = _service(phi_codec)  # empty process cache, as after a restart
        result = await fresh.retrieve_memories(user_id=user_id, session_id=old_session, tiers=["tier_3_session"],
                                               query=None, limit=20, min_importance=Decimal("0"),
                                               time_range_hours=None)
    assert [r.content for r in result.records] == ["synthetic note about my sister"]


async def test_mvp_149_summary_is_returned_as_tier_4(cpm_default, seed_users, phi_codec) -> None:
    (user_id,) = await seed_users(1)
    service = _service(phi_codec)
    with acting_as_user(str(user_id)):
        await service._postgres_repo.store_session_summary({"session_id": uuid4(), "user_id": user_id,
                                                            "session_number": 1, "summary_text": "synthetic recap"})
        result = await _service(phi_codec).retrieve_memories(user_id=user_id, session_id=None,
                                                             tiers=["tier_4_episodic"], query=None, limit=5,
                                                             min_importance=Decimal("0"), time_range_hours=None)
    assert [r.content for r in result.records] == ["synthetic recap"]


async def test_mvp_149_another_users_rows_are_invisible(cpm_default, seed_users, phi_codec) -> None:
    owner, other = await seed_users(2)
    service = _service(phi_codec)
    with acting_as_user(str(owner)):
        await _store(service, owner, uuid4(), "synthetic private note")
    with acting_as_user(str(other)):
        assert await service._postgres_repo.get_user_records(owner) == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_061_read_path.py -o "addopts=" -p no:logging -q`, then `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/memory-service/test_mvp_061_memory_read_path.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `KeyError: 'record_id'` in the converter; the recorded call has no `session_id`/`since`/`min_importance` keys (only `tier` and `limit`); the broken repository is swallowed and the cache fallback returns no error; on real Postgres the restart read returns `[]` (KeyError caught, empty cache), and tier 4 returns nothing (summaries are never read by `retrieve_memories`).
- [ ] **Step 3: Implement** the filtered, decrypting reads in `postgres_repo.py` and the read path in `service.py`. Rows whose ciphertext fails to decrypt raise (no silent skip). Records from tier 4 are built from summary rows (`content=summary_text`, `content_type="session_summary"`, `created_at=session_date`). Leave `delete_user_data` and the `user_facts` Table untouched (WS-17.11).
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run the memory suite (`services/memory_service/tests`), `tests/integration/test_session_lifecycle.py`, and `tests/realinfra/memory-service` (with `$env:SOLACE_REQUIRE_REALINFRA="1"`), one process each, using the same command prefix as Step 2. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.7: Crisis retention through the shared lexicon; safety-critical records are never evicted
**Covers:** MVP-144
**Severity:** medium
**Wave:** 2
**Depends on:** WS-17.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: provisional lexicon v1; any non-NONE screen level, or any legacy `CRISIS_KEYWORDS` hit, makes a record permanent; over-retention is the accepted failure mode)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/domain/service.py` (new `_classify_crisis(content: str) -> tuple[bool, tuple[str, ...], str]`; `store_memory` (lines 144-148) and `add_message` (lines 367-372) set `retention_category="permanent"`, `importance_score=Decimal("1.0")`, `is_safety_critical=True` and metadata `crisis_categories` (category codes, sorted) plus `lexicon_version`; `_enforce_tier_limit` and the `tier_list[-max_records:]` eviction in `add_message` evict the oldest non-safety-critical records first and never drop a safety-critical or permanent record; keep `CRISIS_KEYWORDS` as an additional trigger)
- Create: `services/memory_service/tests/test_mvp_144_crisis_retention.py`
- Test: `services/memory_service/tests/test_mvp_144_crisis_retention.py`

**Interfaces:**
- Consumes: IF-07 `solace_clinical.screen_text(text) -> ScreenResult(level, categories, rule_ids, negated_rule_ids, lexicon_version)`, `solace_clinical.CATEGORIES`, `solace_common.enums.CrisisLevel`; `tests/clinical/crisis_phrases_v1.yaml` (WS-12).
- Produces: crisis records carry `is_safety_critical=True` (persisted to `memory_records.is_safety_critical` by WS-17.5's INSERT) and metadata `crisis_categories: list[str]` with codes from `CATEGORIES` only, never matched text; legacy-keyword-only hits record `crisis_source="legacy_keyword"` and an empty category list.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_144_crisis_retention.py`
```python
"""MVP-144: crisis content is retained permanently and never evicted (WS-17.7)."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService
from solace_clinical import CATEGORIES

PHRASES = yaml.safe_load((Path(__file__).resolve().parents[3] / "tests" / "clinical"
                          / "crisis_phrases_v1.yaml").read_text(encoding="utf-8"))


async def _say(service, user_id, session_id, text):
    await service.add_message(user_id=user_id, session_id=session_id, role="user", content=text,
                              emotion_detected=None, importance_override=None, metadata={})
    return service._tier_3_session[user_id][-1]


@pytest.mark.parametrize("text", ["I've been feeling suicidal", "I just want to end it all",
                                  "I took an overdose last night", "I\u2019m going to kill myself"])
async def test_mvp_144_crisis_phrasings_are_retained_permanently(memory_service_inmem, text) -> None:
    record = await _say(memory_service_inmem, uuid4(), uuid4(), text)
    assert record.retention_category == "permanent"
    assert record.importance_score == Decimal("1.0")
    assert record.is_safety_critical is True
    assert record.metadata["crisis_categories"]
    assert set(record.metadata["crisis_categories"]) <= set(CATEGORIES)
    assert text not in json.dumps(record.metadata)


async def test_mvp_144_safety_critical_records_survive_tier_eviction() -> None:
    service = MemoryService(settings=MemoryServiceSettings(max_tier_records_per_user=5,
                                                           enable_auto_consolidation=False, enable_decay=False))
    user_id, session_id = uuid4(), uuid4()
    await _say(service, user_id, session_id, "I've been feeling suicidal")
    for i in range(10):
        await _say(service, user_id, session_id, f"synthetic everyday note {i}")
    kept = service._tier_3_session[user_id]
    assert any(r.content == "I've been feeling suicidal" for r in kept)
    assert len([r for r in kept if not r.is_safety_critical]) == 5


@pytest.mark.parametrize("text", [
    "I'm not going to pretend everything is fine. Work has been awful for weeks, my sleep is gone "
    "and some nights I want to die",
    'My sister told me "I want to kill myself" last night and I keep thinking about it',
], ids=["negation-in-an-earlier-clause", "quoted-speech"])
async def test_mvp_144_disclosures_after_a_negation_or_in_quotes_are_over_retained(memory_service_inmem, text) -> None:
    record = await _say(memory_service_inmem, uuid4(), uuid4(), text)
    assert record.retention_category == "permanent"
    assert record.is_safety_critical is True


@pytest.mark.clinical
@pytest.mark.parametrize("phrase", PHRASES["positive"], ids=lambda p: p["text"][:40])
async def test_mvp_144_every_positive_phrase_is_retained(memory_service_inmem, phrase) -> None:
    record = await _say(memory_service_inmem, uuid4(), uuid4(), phrase["text"])
    assert record.retention_category == "permanent" and record.is_safety_critical is True
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_144_crisis_retention.py -o "addopts=" -p no:logging -q`
Expected: FAIL. "feeling suicidal", "end it all" and "overdose" are stored `medium_term` (`'suicide' in 'feeling suicidal'` is False); no record has `is_safety_critical` set or `crisis_categories`; the crisis record is evicted by age once ten newer notes arrive; many phrase-set positives are not retained; the negation-first and quoted-speech messages are already `permanent` through the legacy keywords but fail on `is_safety_critical` (WS-17.5 added the field with default `False`; nothing sets it). Those two cases stay green after Step 3 only if retention never applies the lexicon's negation or quotation suppression (review focus 1: memory over-retains).
- [ ] **Step 3: Implement** `_classify_crisis` with `screen = screen_text(content)` and `is_crisis = screen.level is not CrisisLevel.NONE or any(kw in normalize_text(content) for kw in self.CRISIS_KEYWORDS)`; categories `tuple(sorted(screen.categories))`. Apply it in `store_memory` and `add_message` before persistence so WS-17.5's INSERT writes `is_safety_critical`. Eviction: partition the tier list into safety-critical/permanent and the rest, trim only the rest to `max_tier_records_per_user`, keep chronological order. Never lower a lexicon-derived retention.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run the memory suite (`services/memory_service/tests`) and `tests/integration/test_session_lifecycle.py`, one process each, with the Step 2 prefix. Expected: all pass (`test_safety_override.py` still passes: the legacy keywords remain triggers).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.8: The context assembler renders chronological order and keeps the newest turns within budget
**Covers:** MVP-147 (also fixes the ordering half of MVP-146, which closes in WS-17.9)
**Severity:** medium
**Wave:** 2
**Depends on:** none
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/domain/context_assembler.py` (`assemble()` keeps priority-based selection in `_assemble_sections` but renders the kept sections in `_DISPLAY_ORDER = ("system_prompt", "safety_context", "user_profile", "therapeutic_context", "retrieved_context", "recent_messages", "current_input")`; `_build_recent_messages_section` (lines 236-269) fills the budget newest-first, renders oldest-to-newest, compresses only the oldest kept message, always keeps the newest message (compressed if it alone exceeds the budget) and skips a trailing user record whose content equals `current_message`)
- Create: `services/memory_service/tests/test_mvp_147_context_order.py`
- Test: `services/memory_service/tests/test_mvp_147_context_order.py`, `services/memory_service/tests/test_context_assembler.py`

**Interfaces:**
- Consumes: none new.
- Produces: `ContextAssembler.assemble(...)` signature unchanged; output order system, safety, profile, therapeutic, retrieved, history, current message.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_147_context_order.py`
```python
"""MVP-147 (+ MVP-146 ordering): newest turns kept, rendered in order (WS-17.8)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from services.memory_service.src.domain.context_assembler import ContextAssembler, ContextAssemblerSettings

BASE = datetime(2026, 10, 1, tzinfo=timezone.utc)


@dataclass
class _Record:
    content: str
    metadata: dict[str, Any]
    created_at: datetime
    record_id: Any = field(default_factory=uuid4)


def _turns(n: int, size: int) -> list[_Record]:
    return [_Record(content=f"MSG{i:02d} " + ("pills saved up for tonight " if i == 18 else "") + "x" * size,
                    metadata={"role": "user" if i % 2 == 0 else "assistant"},
                    created_at=BASE + timedelta(minutes=i)) for i in range(n)]


async def _assemble(history, budget, current="what now"):
    return (await ContextAssembler(ContextAssemblerSettings()).assemble(
        user_id=uuid4(), session_id=uuid4(), current_message=current, token_budget=budget,
        include_safety=False, include_therapeutic=False, retrieval_query=None, priority_topics=[],
        working_memory=[], session_memory=history, user_profile={})).assembled_context


async def test_mvp_147_newest_turns_survive_a_tight_budget() -> None:
    context = await _assemble(_turns(20, 760), budget=4000)
    assert "MSG18" in context and "pills saved up" in context and "MSG19" in context
    assert "MSG00" not in context


async def test_mvp_147_history_is_rendered_oldest_to_newest() -> None:
    context = await _assemble(_turns(20, 760), budget=4000)
    assert context.index("MSG18") < context.index("MSG19")


async def test_mvp_146_current_message_follows_the_history() -> None:
    context = await _assemble(_turns(3, 20), budget=8000)
    assert context.index("[System]") < context.index("[Conversation History]") < context.index("[Current Message]")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_147_context_order.py -o "addopts=" -p no:logging -q`
Expected: FAIL. At budget 4000 the loop keeps MSG00-MSG07 and drops MSG18 ("pills") and MSG19; `[Current Message]` sorts before `[Conversation History]` (both priority 10 vs 8 under the stable sort).
- [ ] **Step 3: Implement** in `context_assembler.py`: iterate `recent` in reverse, accumulate lines until the budget is reached, then reverse the kept lines for rendering; after `_assemble_sections` selects and truncates by priority, emit the selected sections sorted by `_DISPLAY_ORDER`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (the relevance-formula source check in `test_sprint5_regression.py` is unaffected).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.9: Context assembly reads durable history, the previous summary, known facts and safety categories
**Covers:** MVP-146
**Severity:** medium
**Wave:** 2
**Depends on:** WS-17.6, WS-17.7, WS-17.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-04 (default assumed: minimum-necessary context to the LLM; the safety section carries category codes and a date, never crisis text)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/domain/service.py` (`assemble_context`, lines 256-287: with a repository, load the session's newest `max_recent_messages` turns via `get_user_records(user_id, tier="tier_3_session", session_id=session_id, limit=...)` and merge with in-process T2/T3 by `record_id`; the latest other-session summary via `get_user_summaries(user_id, limit=1, exclude_session_id=session_id)`; up to 10 facts via `get_user_facts(user_id, limit=10)`; `profile["safety"]["crisis_history"]` built from `get_safety_critical_categories(user_id)`)
- Modify: `services/memory_service/src/domain/context_assembler.py` (`assemble(..., previous_summary: str | None = None, facts: Sequence[str] = ())`; sections `[Previous Session Summary]` (priority 7) and `[Known Facts]` (priority 6), displayed after the therapeutic section)
- Modify: `services/memory_service/src/infrastructure/postgres_repo.py` (`get_safety_critical_categories(user_id) -> tuple[list[str], datetime | None]`: union of decrypted `crisis_categories` over the user's `is_safety_critical` rows and the latest `created_at`)
- Create: `services/memory_service/tests/test_mvp_146_durable_context.py`
- Test: `services/memory_service/tests/test_mvp_146_durable_context.py`

**Interfaces:**
- Consumes: WS-17.6 repository read signatures; WS-17.7 `crisis_categories`.
- Produces: `ContextAssembler.assemble(..., previous_summary=None, facts=())` (keyword defaults keep existing callers working); `PostgresRepository.get_safety_critical_categories(user_id)`.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_146_durable_context.py`
```python
"""MVP-146: context comes from durable memory, not only process-local tiers (WS-17.9)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from services.memory_service.src.domain.context_assembler import ContextAssembler
from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService

NOW = datetime(2026, 10, 2, tzinfo=timezone.utc)


class _DurableRepo:
    def __init__(self, user_id, session_id) -> None:
        self.rows = [{"id": uuid4(), "user_id": user_id, "session_id": session_id, "tier": "tier_3_session",
                      "content": text, "content_type": "message", "retention_category": "medium_term",
                      "importance_score": 0.5, "emotional_valence": None, "record_metadata": {"role": role},
                      "retention_strength": 1.0, "is_safety_critical": False,
                      "created_at": NOW + timedelta(minutes=i), "accessed_at": NOW}
                     for i, (role, text) in enumerate([("user", "synthetic turn one"),
                                                       ("assistant", "synthetic turn two"),
                                                       ("user", "synthetic turn three")])]

    async def get_user_records(self, user_id, **kwargs):
        return list(reversed(self.rows)) if kwargs.get("tier") == "tier_3_session" else []

    async def get_user_summaries(self, user_id, *, limit=10, exclude_session_id=None):
        return [{"summary_text": "Previous: synthetic sleep worries", "session_date": NOW}]

    async def get_user_facts(self, user_id, *, limit=20):
        return [{"content": "synthetic fact: works night shifts"}]

    async def get_safety_critical_categories(self, user_id):
        return (["suicidal_ideation"], NOW - timedelta(days=3))


async def test_mvp_146_cold_cache_context_reads_durable_memory() -> None:
    user_id, session_id = uuid4(), uuid4()
    service = MemoryService(settings=MemoryServiceSettings(enable_auto_consolidation=False, enable_decay=False),
                            context_assembler=ContextAssembler(), postgres_repo=_DurableRepo(user_id, session_id))
    result = await service.assemble_context(user_id=user_id, session_id=session_id, current_message="and today?",
                                            token_budget=8000, include_safety=True, include_therapeutic=True,
                                            retrieval_query=None, priority_topics=[])
    context = result.assembled_context
    assert context.index("synthetic turn one") < context.index("synthetic turn three")
    assert "Previous: synthetic sleep worries" in context
    assert "works night shifts" in context
    assert "[Safety Context" in context and "suicidal_ideation" in context
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_146_durable_context.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `assemble_context` passes only `_tier_2_working`, `_tier_3_session` and `_user_profiles` (all empty), so the context holds only the system prompt and the current message; no summary, facts or safety section.
- [ ] **Step 3: Implement** the durable sources in `service.py` and the two new sections in `context_assembler.py`. The safety section text is `Crisis History: prior crisis indicators on record (categories: <codes>; most recent: <YYYY-MM-DD>)`. Repository read failures raise `MemoryPersistenceError` (no silent empty context).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run the memory suite (`services/memory_service/tests`) and `tests/integration/test_session_lifecycle.py`, one process each, with the Step 2 prefix. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.10: Per-user BM25 ranking with no process-wide index
**Covers:** MVP-278, MVP-280, MVP-281, MVP-282
**Severity:** low
**Wave:** 3
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-04 (default assumed: retrieval keyword/BM25 only; Weaviate keyword search is not on the read path)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/infrastructure/hybrid_search.py` (new `rank_bm25(documents: Sequence[tuple[UUID, str]], query: str, *, limit: int, k1: float = 1.5, b: float = 0.75) -> list[tuple[UUID, float]]` building a fresh `BM25Index` per call)
- Modify: `services/memory_service/src/domain/service.py` (`_semantic_filter` (lines 669-698) ranks only the caller's candidate records with `rank_bm25`, orders by score, falls back to substring only when BM25 returns nothing; Weaviate keyword branch removed; `search_engine` constructor parameter and `_search_engine` attribute removed)
- Modify: `services/memory_service/src/main.py` (`build_memory_service` no longer constructs `HybridSearchEngine`)
- Create: `services/memory_service/tests/test_mvp_280_per_user_ranking.py`
- Test: `services/memory_service/tests/test_mvp_280_per_user_ranking.py`

**Interfaces:**
- Consumes: none new.
- Produces: `rank_bm25(...)`; `MemoryService(...)` without `search_engine`.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_280_per_user_ranking.py`
```python
"""MVP-278/280/281/282: ranking is per user and leaves no shared index behind (WS-17.10)."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from services.memory_service.src import main as memory_main
from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.infrastructure.hybrid_search import HybridSearchEngine
from services.memory_service.src.infrastructure.weaviate_repo import SearchResult as VectorHit

SESSION = uuid4()


def _service():
    runtime = SimpleNamespace(pg_pool=None, redis=None, phi_codec=None, event_plane=None)
    return memory_main.build_memory_service(runtime, memory_main.SETTINGS)


async def _say(service, user_id, text):
    await service.add_message(user_id=user_id, session_id=SESSION, role="user", content=text,
                              emotion_detected=None, importance_override=Decimal("0.5"), metadata={})


async def _query(service, user_id, text):
    result = await service.retrieve_memories(user_id=user_id, session_id=None, tiers=["tier_3_session"], query=text,
                                             limit=10, min_importance=Decimal("0"), time_range_hours=None)
    return [r.content for r in result.records]


def _indexed_ids(service):
    ids = set()
    for value in vars(service).values():
        if isinstance(value, HybridSearchEngine):
            ids |= set(value._bm25._documents)
    return ids


async def test_mvp_278_query_is_ranked_not_substring_filtered() -> None:
    service, user_id = _service(), uuid4()
    for text in ("attacks of panic at night", "synthetic grocery list", "panic attacks in meetings and panic at home"):
        await _say(service, user_id, text)
    contents = await _query(service, user_id, "panic attacks")
    assert contents[0] == "panic attacks in meetings and panic at home"
    assert "attacks of panic at night" in contents
    assert "synthetic grocery list" not in contents


async def test_mvp_280_another_users_documents_never_enter_an_index() -> None:
    service, a, b = _service(), uuid4(), uuid4()
    for i in range(30):
        await _say(service, b, f"panic attacks again {i}")
    await _say(service, a, "my panic attacks are back")
    await _query(service, b, "panic attacks")
    assert await _query(service, a, "panic attacks") == ["my panic attacks are back"]
    assert _indexed_ids(service) == set()


async def test_mvp_281_erasure_leaves_no_indexed_text() -> None:
    service, user_id = _service(), uuid4()
    await _say(service, user_id, "boss yelled at me again")
    await _query(service, user_id, "boss yelled")
    await service.delete_user_data(user_id=user_id)
    assert _indexed_ids(service) == set()


async def test_mvp_282_a_weaviate_hit_does_not_hide_other_matches() -> None:
    class _Weaviate:
        def __init__(self) -> None:
            self.first_id = None

        async def search_by_keyword(self, keyword, user_id, limit=20, **kwargs):
            return [VectorHit(record_id=self.first_id, content="", score=1.0, distance=0.0)]

    weaviate = _Weaviate()
    service = MemoryService(settings=MemoryServiceSettings(enable_auto_consolidation=False, enable_decay=False),
                            weaviate_repo=weaviate)
    user_id = uuid4()
    for text in ("sleep was bad", "cannot sleep before exams", "sleep schedule ideas"):
        await _say(service, user_id, text)
    weaviate.first_id = service._tier_3_session[user_id][0].record_id
    assert len(await _query(service, user_id, "sleep")) == 3
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_280_per_user_ranking.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `SearchResult` has no `score`, so the shared engine raises `AttributeError`, logs `hybrid_search_failed_using_fallback` and falls back to substring matching ("attacks of panic at night" is missed); the shared `HybridSearchEngine` built by `build_memory_service` holds every queried user's documents, including after `delete_user_data`; the Weaviate branch returns only its single hit.
- [ ] **Step 3: Implement** `rank_bm25` and the new `_semantic_filter(records, query)`: build `(record_id, content)` pairs from the caller's candidates (already filtered to `user_id`), rank, keep records with score above 0 in score order, substring fallback only when the ranking is empty. Delete the Weaviate keyword call from the read path (D-04) and the shared engine from the service and from `build_memory_service`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`, then `tests/integration/test_session_lifecycle.py` in its own process. Expected: all pass (`TestHybridSearchEngine` in `test_infrastructure.py` still exercises the engine class directly).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.11: Memory erasure completes for every store and runs from user.deleted
**Covers:** MVP-062 (supports MVP-029, owned by WS-14)
**Severity:** high
**Wave:** 3
**Depends on:** WS-17.10, WS-14.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: every memory PHI row, key and vector is hard-deleted; memory keeps no pseudonymized copy)
**Real infra:** yes
**Files:**
- Modify: `services/memory_service/src/domain/service.py` (new `async def purge_user_state(self, user_id: UUID) -> None` (tiers, sessions, profiles, session counts); new `def build_erasure(self) -> UserDataErasure`; `delete_user_data` (lines 546-599) purges in-process state, runs the erasure inside `acting_as_user(str(user_id))`, purges again, and raises `ErasureIncomplete` only after purging; `initialize()` keeps an uninitialized Weaviate repository registered and re-raises in strict environments)
- Modify: `services/memory_service/src/domain/erasure.py` (Postgres deleter sums the four table counts)
- Modify: `services/memory_service/src/infrastructure/postgres_repo.py` (`delete_user_data` deletes from `memory_records`, `session_summaries`, `therapeutic_events`, `memory_user_profiles` and returns a 4-tuple; the `user_facts` Table object is removed)
- Modify: `services/memory_service/src/infrastructure/weaviate_repo.py` (new `class WeaviateUnavailableError(RuntimeError)`; `initialize()` (lines 80-105) raises it instead of swallowing; `delete_user_data` calls `initialize()` once when not initialized before failing)
- Modify: `services/memory_service/src/infrastructure/redis_cache.py` (`delete_user_cache` raises `RuntimeError` when no client is attached)
- Modify: `services/memory_service/src/main.py` (`register_memory_erasure(plane, memory_service) -> None` calls `erasure_consumer.register_user_erasure_consumer(plane, service_name="memory-service", erasure=memory_service.build_erasure(), purge_caches=memory_service.purge_user_state)` with `from solace_infrastructure.gdpr import consumer as erasure_consumer`; the lifespan calls it when `runtime.event_plane is not None`; readiness check `weaviate` (critical) when Weaviate is enabled)
- Modify: `services/memory_service/src/api.py` (DELETE `/user/{user_id}` maps `ErasureIncomplete` to `HTTPException(500, detail="User data erasure incomplete")`)
- Modify: `services/memory_service/tests/test_gdpr_erasure.py` (`test_domain_delete_user_data_propagates_incomplete_erasure` now asserts the caches were purged; `test_delete_user_cache_noop_when_uninitialized` becomes a `RuntimeError` assertion)
- Create: `services/memory_service/tests/test_mvp_062_erasure_every_store.py`
- Create: `tests/realinfra/memory-service/test_mvp_062_memory_erasure.py`
- Test: the two new files and `services/memory_service/tests/test_gdpr_erasure.py`, `services/memory_service/tests/test_rls_guc.py`

**Interfaces:**
- Consumes: IF-16 `solace_infrastructure.gdpr.consumer.register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches=None)`; IF-05 `runtime.event_plane`; `acting_as_user`; IF-19 `redis_url`.
- Produces: `MemoryService.purge_user_state(user_id)`, `MemoryService.build_erasure()`, `main.register_memory_erasure(plane, memory_service)`, `WeaviateUnavailableError`; erasure store names `memory_postgres`, `memory_redis` and, when enabled, `memory_weaviate`.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_062_erasure_every_store.py`
```python
"""MVP-062: memory erasure can complete and leaves nothing in process (WS-17.11)."""
from __future__ import annotations

from uuid import uuid4

import pytest

from services.memory_service.src import main as memory_main
from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.infrastructure.redis_cache import RedisCache, RedisSettings
from services.memory_service.src.infrastructure.weaviate_repo import WeaviateRepository, WeaviateSettings
from solace_infrastructure.gdpr import ErasureIncomplete

SETTINGS = MemoryServiceSettings(enable_auto_consolidation=False, enable_decay=False)


class _RaisingPostgres:
    async def delete_user_data(self, user_id):
        raise RuntimeError("db unavailable")


class _FakePostgres:
    async def delete_user_data(self, user_id):
        return (1, 0, 0, 0)


class _FakeRedis:
    async def delete_user_cache(self, user_id):
        return 2


async def test_mvp_062_in_process_state_is_purged_even_when_a_store_fails() -> None:
    service = MemoryService(settings=SETTINGS, postgres_repo=_RaisingPostgres())
    user_id = uuid4()
    service._tier_3_session[user_id] = ["seed"]
    service._user_profiles[user_id] = {"facts": {"k": "v"}}
    with pytest.raises(ErasureIncomplete):
        await service.delete_user_data(user_id=user_id)
    assert user_id not in service._tier_3_session and user_id not in service._user_profiles


async def test_mvp_062_weaviate_initialize_failure_is_loud(monkeypatch) -> None:
    import weaviate

    def _refuse(**kwargs):
        raise ConnectionError("weaviate refused")

    monkeypatch.setattr(weaviate, "connect_to_local", _refuse)
    with pytest.raises(Exception) as exc_info:
        await WeaviateRepository(WeaviateSettings(enabled=True)).initialize()
    assert type(exc_info.value).__name__ == "WeaviateUnavailableError"


async def test_mvp_062_redis_erase_without_a_client_fails_loud() -> None:
    with pytest.raises(RuntimeError):
        await RedisCache(RedisSettings()).delete_user_cache(uuid4())


def test_mvp_062_lifespan_wiring_registers_the_erasure_consumer(monkeypatch) -> None:
    captured = {}

    def _capture(plane, *, service_name, erasure, purge_caches=None):
        captured.update(plane=plane, service_name=service_name, stores=erasure.store_names, purge=purge_caches)

    monkeypatch.setattr("solace_infrastructure.gdpr.consumer.register_user_erasure_consumer", _capture)
    service = MemoryService(settings=SETTINGS, postgres_repo=_FakePostgres(), redis_cache=_FakeRedis())
    plane = object()
    memory_main.register_memory_erasure(plane, service)
    assert captured["plane"] is plane and captured["service_name"] == "memory-service"
    assert captured["stores"] == ("memory_postgres", "memory_redis")
    assert captured["purge"] == service.purge_user_state
```
and in `tests/realinfra/memory-service/test_mvp_062_memory_erasure.py`
```python
"""MVP-062 on the migrated schema and a real Redis (WS-17.11)."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
import redis.asyncio as aioredis

from services.memory_service.src import main as memory_main
from solace_common.request_context import acting_as_user

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
TABLES = ("memory_records", "session_summaries", "therapeutic_events", "memory_user_profiles")


async def test_mvp_062_erasure_removes_every_store_and_spares_others(
        cpm_default, seed_users, phi_codec, redis_url, superuser_conn) -> None:
    target, other = await seed_users(2)
    client = aioredis.from_url(redis_url, decode_responses=True)
    runtime = SimpleNamespace(pg_pool=object(), redis=client, phi_codec=phi_codec, event_plane=None)
    service = memory_main.build_memory_service(runtime, memory_main.SETTINGS)
    await service.initialize()
    for user_id in (target, other):
        with acting_as_user(str(user_id)):
            await service.add_message(user_id=user_id, session_id=uuid4(), role="user",
                                      content="synthetic erasure probe", emotion_detected=None,
                                      importance_override=None, metadata={})
            await service._postgres_repo.store_session_summary(
                {"session_id": uuid4(), "user_id": user_id, "session_number": 1, "summary_text": "synthetic"})
    await superuser_conn.execute(
        "INSERT INTO therapeutic_events (id, user_id, created_at, updated_at, encryption_key_id, event_type, "
        "title, description, occurred_at, ingested_at) "
        "VALUES ($1, $2, now(), now(), 'test-k1', 'note', 'synthetic', 'synthetic', now(), now())",
        uuid4(), target)
    with acting_as_user(str(target)):
        report = await service.build_erasure().erase(str(target))
    for table in TABLES:
        assert await superuser_conn.fetchval(f"SELECT count(*) FROM {table} WHERE user_id = $1", target) == 0
    assert await superuser_conn.fetchval("SELECT count(*) FROM memory_records WHERE user_id = $1", other) == 1
    assert [k async for k in client.scan_iter(match=f"solace:memory:*{target}*")] == []
    assert [k async for k in client.scan_iter(match=f"solace:memory:*{other}*")] != []
    assert set(report.deleted) == {"memory_postgres", "memory_redis"}
    await client.aclose()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_062_erasure_every_store.py -o "addopts=" -p no:logging -q`, then `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/memory-service/test_mvp_062_memory_erasure.py -o "addopts=" -p no:logging -q`
Expected: FAIL. In-process tiers survive a failed durable erase; `initialize()` swallows the Weaviate connect error; the uninitialized Redis delete returns 0; `register_memory_erasure` does not exist; on real Postgres `erase()` raises `ErasureIncomplete` because `DELETE FROM user_facts` hits `UndefinedTableError` and rolls back the whole transaction.
- [ ] **Step 3: Implement** the changes listed above. `purge_user_state` is idempotent and async (the IF-16 `purge_caches` type). `build_erasure()` passes `self._postgres_repo`, `self._redis` and (only when configured) `self._weaviate_repo` to `build_user_data_erasure`. Erasure never logs content; it logs store names and counts only.
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run the memory suite (`services/memory_service/tests`), `tests/integration/test_session_lifecycle.py` and `tests/realinfra/memory-service` (with `$env:SOLACE_REQUIRE_REALINFRA="1"`), one process each, with the Step 2 prefixes. Expected: all pass (`test_rls_guc.py` still sees GUC = target during the erase and the caller restored after).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.12: Session ownership, server-owned roles and clinician scoping in memory
**Covers:** MVP-143, MVP-285 (supports MVP-120 and MVP-141, owned by WS-07)
**Severity:** medium
**Wave:** 3
**Depends on:** WS-17.11
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-21 (default assumed: a clinician reaches a user's memory only when `can_access_subject` allows it; admins have no memory read access; DELETE stays self-or-admin)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/domain/models.py` (`class SessionOwnershipError(LookupError)`)
- Modify: `services/memory_service/src/domain/service.py` (`_require_session_owner(user_id, session_id)` called by `end_session` (lines 319-359), `add_message` (lines 361-425) and `consolidate`; caller metadata merged as `{**_strip_reserved(metadata), "role": role, "emotion": emotion_detected}` in `add_message` (line 377) and reserved keys stripped in `store_memory`; reserved keys `role`, `emotion`, `crisis_categories`, `lexicon_version`, `crisis_source`, `is_safety_critical`)
- Modify: `services/memory_service/src/schemas.py` (`AddMessageRequest.role` pattern `^(user|assistant)$`)
- Modify: `services/memory_service/src/api.py` (`_authorize` replaced: read and write routes call `can_access_subject(current_user, str(subject))` then `set_subject_context(str(subject))`; DELETE keeps self-or-admin then `set_subject_context`; `SessionOwnershipError` maps to `HTTPException(404, detail="Session not found")`)
- Create: `services/memory_service/tests/test_mvp_143_session_ownership.py`
- Test: `services/memory_service/tests/test_mvp_143_session_ownership.py`

**Interfaces:**
- Consumes: IF-04 W3 `solace_security.middleware.can_access_subject(principal, subject_user_id) -> bool` and `set_subject_context(subject_user_id)` (imported into `api.py` by name).
- Produces: `SessionOwnershipError`; HTTP 404 for a foreign session; `AddMessageRequest.role` in {user, assistant} (IF-11 consumers WS-11 and WS-15 never send system turns).

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_143_session_ownership.py`
```python
"""MVP-143/MVP-285 (+ MVP-120 scoping): session and role integrity (WS-17.12)."""
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from services.memory_service.src import api as memory_api_module
from services.memory_service.src.schemas import AddMessageRequest


async def _say(service, user_id, session_id, text, metadata=None):
    await service.add_message(user_id=user_id, session_id=session_id, role="user", content=text,
                              emotion_detected=None, importance_override=None, metadata=metadata or {})


async def test_mvp_143_end_session_rejects_another_users_session(memory_service_inmem) -> None:
    owner, intruder = uuid4(), uuid4()
    started = await memory_service_inmem.start_session(user_id=owner, session_type="therapeutic", initial_context={})
    await _say(memory_service_inmem, owner, started.session_id, "synthetic owner worry")
    with pytest.raises(LookupError):
        await memory_service_inmem.end_session(user_id=intruder, session_id=started.session_id,
                                               trigger_consolidation=False, include_summary=True)
    assert started.session_id in memory_service_inmem._active_sessions


async def test_mvp_143_add_message_cannot_inject_into_another_users_session(memory_service_inmem) -> None:
    owner, intruder = uuid4(), uuid4()
    started = await memory_service_inmem.start_session(user_id=owner, session_type="therapeutic", initial_context={})
    with pytest.raises(LookupError):
        await _say(memory_service_inmem, intruder, started.session_id, "synthetic injected text")
    assert memory_service_inmem._active_sessions[started.session_id].messages == []


async def test_mvp_285_caller_metadata_cannot_override_the_server_role(memory_service_inmem) -> None:
    user_id = uuid4()
    await _say(memory_service_inmem, user_id, uuid4(), "synthetic", {"role": "system", "is_safety_critical": True})
    record = memory_service_inmem._tier_3_session[user_id][-1]
    assert record.metadata["role"] == "user"
    assert "is_safety_critical" not in record.metadata


def test_mvp_285_system_role_is_rejected_at_the_api() -> None:
    with pytest.raises(ValidationError):
        AddMessageRequest(user_id=uuid4(), session_id=uuid4(), role="system", content="synthetic")


def test_supports_mvp_120_unassigned_clinician_is_denied(memory_api, monkeypatch) -> None:
    monkeypatch.setattr(memory_api_module, "can_access_subject", lambda principal, subject: False, raising=False)
    client = memory_api(str(uuid4()), roles=["clinician"])
    assert client.post("/api/v1/memory/context", json={"user_id": str(uuid4())}).status_code == 403
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_143_session_ownership.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The intruder ends the owner's session (summary returned, session deleted) and appends into it; caller metadata `role="system"` wins over the server role; `role="system"` passes validation; the clinician gets 200 through the role bypass.
- [ ] **Step 3: Implement** the ownership check (raise `SessionOwnershipError` when an active session exists for a different user), the metadata merge order and reserved-key stripping, the schema pattern, and the IF-04 scoping in `api.py`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run the memory suite (`services/memory_service/tests`) and `tests/integration/test_session_lifecycle.py`, one process each, with the Step 2 prefix. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.13: No PHI egress to embedding APIs and no raw query text in logs
**Covers:** MVP-145, MVP-277, MVP-279
**Severity:** medium
**Wave:** 3
**Depends on:** WS-17.12
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-04 (default assumed: no embedding vendor is approved; `MEMORY_EMBEDDINGS_PROVIDER` accepts only `none`; OPENAI_API_KEY is never read)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/domain/service.py` (delete `_generate_embedding` (lines 119-136) and the OpenAI branch in `store_memory`; Weaviate, when enabled, receives `embedding=[]`)
- Modify: `services/memory_service/src/main.py` (`MemoryServiceAppSettings.embeddings_provider: Literal["none"] = "none"`, env `MEMORY_EMBEDDINGS_PROVIDER`; the lifespan logs `memory_semantic_retrieval_disabled` once with `embeddings_provider="none"`)
- Modify: `services/memory_service/src/api.py` (`/retrieve` logs `query_present` and `query_length`; no route logs request text)
- Create: `services/memory_service/tests/test_mvp_145_no_phi_egress.py`
- Test: `services/memory_service/tests/test_mvp_145_no_phi_egress.py`

**Interfaces:**
- Consumes: IF-02 `MEMORY_EMBEDDINGS_PROVIDER`.
- Produces: `MemoryServiceAppSettings.embeddings_provider` (only `none` is valid; any other value fails settings load at startup).

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_145_no_phi_egress.py`
```python
"""MVP-145/277/279: no direct embedding egress, honest settings, no query in logs (WS-17.13)."""
from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from services.memory_service.src import api as memory_api_module
from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.main import MemoryServiceAppSettings


class RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, level):
        def _log(event, **kwargs):
            self.calls.append((level, event, kwargs))
        return _log


async def test_mvp_145_store_never_calls_an_external_embedding_api(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-not-a-key")
    urls: list[str] = []

    class _RecordingClient:
        def __init__(self, *args, **kwargs) -> None: ...
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        async def post(self, url, **kwargs):
            urls.append(url)
            raise AssertionError("no network in tests")

    class _Weaviate:
        async def store_vector(self, record): return record.record_id
        async def initialize(self) -> None: ...
        async def close(self) -> None: ...

    monkeypatch.setattr(httpx, "AsyncClient", _RecordingClient)
    service = MemoryService(settings=MemoryServiceSettings(enable_auto_consolidation=False, enable_decay=False),
                            weaviate_repo=_Weaviate())
    await service.store_memory(user_id=uuid4(), session_id=None, content="synthetic diary entry",
                               content_type="message", tier="tier_3_session", retention_category="medium_term",
                               importance_score=Decimal("0.5"), metadata={})
    assert urls == []


def test_mvp_277_only_the_none_embeddings_provider_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_EMBEDDINGS_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        MemoryServiceAppSettings()
    monkeypatch.setenv("MEMORY_EMBEDDINGS_PROVIDER", "none")
    assert MemoryServiceAppSettings().embeddings_provider == "none"


def test_mvp_279_retrieve_does_not_log_the_query_text(memory_api, monkeypatch) -> None:
    recorder = RecordingLogger()
    monkeypatch.setattr(memory_api_module, "logger", recorder)
    user_id = str(uuid4())
    memory_api(user_id).post("/api/v1/memory/retrieve",
                             json={"user_id": user_id, "query": "Dr. Moreno put me on lithium"})
    assert recorder.calls
    assert "lithium" not in repr(recorder.calls)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_145_no_phi_egress.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `urls == ["https://api.openai.com/v1/embeddings"]` (the branch is keyed only on the Weaviate repo being present); `MEMORY_EMBEDDINGS_PROVIDER=openai` is ignored (no such field, extra env ignored); `retrieve_memories_requested` logs `query=` with the full text.
- [ ] **Step 3: Implement** the deletions and the setting; the startup log replaces today's silent missing-key path (MVP-277: the read path never had vector search, and now the configuration says so).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.14: Audit PHI access on every memory route
**Covers:** none (supports MVP-091, owned by WS-09)
**Severity:** high (supporting a blocker; closes no WS-17 entry)
**Wave:** 3
**Depends on:** WS-17.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: audit rows hold ids, route templates and status only)
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/api.py` (every route except `/status` gains `dependencies=[Depends(audit_phi_access("memory", subject=...))]`: path routes pass `subject=lambda request: request.path_params.get("user_id")`; body routes rely on the request-context subject set by `set_subject_context`)
- Create: `services/memory_service/tests/test_mvp_091_memory_phi_audit.py`
- Test: `services/memory_service/tests/test_mvp_091_memory_phi_audit.py`

**Interfaces:**
- Consumes: IF-14 `solace_security.audit_runtime.audit_phi_access(resource_type, *, subject=None)` and `audit(event_type, *, actor_id, actor_type, action, outcome, resource_type, resource_id=None, subject_user_id=None, details=None)`.
- Produces: one audit row per memory PHI request (resource type `memory`).

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_091_memory_phi_audit.py`
```python
"""Supports MVP-091: memory PHI reads and writes are audited (WS-17.14)."""
from __future__ import annotations

from uuid import uuid4


def test_supports_mvp_091_context_read_writes_one_audit_row(memory_api, monkeypatch) -> None:
    captured = []

    async def _audit(event_type, **kwargs):
        captured.append((event_type, kwargs))

    monkeypatch.setattr("solace_security.audit_runtime.audit", _audit)
    user_id = str(uuid4())
    response = memory_api(user_id).post("/api/v1/memory/context",
                                        json={"user_id": user_id, "current_message": "synthetic private text"})
    assert response.status_code == 200
    assert len(captured) == 1
    _, kwargs = captured[0]
    assert kwargs["resource_type"] == "memory"
    assert kwargs["subject_user_id"] == user_id
    assert "synthetic private text" not in repr(kwargs)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_091_memory_phi_audit.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `captured == []`: no memory route records an audit event.
- [ ] **Step 3: Implement** the dependencies in `api.py`, keeping the router signature and response models unchanged.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run the memory suite (`services/memory_service/tests`) with the Step 2 prefix. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.15: Consolidate the previous session automatically with a crisis-aware summary
**Covers:** MVP-142, MVP-148
**Severity:** medium
**Wave:** 5
**Depends on:** none
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: the summary flags crisis by category codes from lexicon v1; clinician sign-off of the summary wording is recorded in docs/CLINICAL-VALIDATION.md by WS-12)
**Real infra:** yes
**Files:**
- Modify: `services/memory_service/src/domain/service.py` (`_consolidate_previous_session(user_id, current_session_id)` runs on the first `assemble_context` call per `(user_id, session_id)` in this process (bounded set of 10,000 entries): it asks the repository for the latest unsummarized earlier session, loads its records and runs `consolidate`; `consolidate` loads session records from the repository when one is configured instead of reading only `_tier_3_session`)
- Modify: `services/memory_service/src/domain/consolidation.py` (`generate_summary` (lines 150-185) screens user messages with `solace_clinical.screen_text` and, on any non-NONE level, starts the summary with `Safety: crisis indicators detected (categories: <codes>; highest level: <LEVEL>).` and adds `crisis` to `key_topics`; `_identify_techniques` reads assistant messages only)
- Modify: `services/memory_service/src/infrastructure/postgres_repo.py` (`get_latest_unsummarized_session(user_id, *, exclude_session_id) -> UUID | None` with a `NOT EXISTS` against `session_summaries`; delete the dead `store_therapeutic_event`, `get_user_events`, `get_crisis_events`)
- Create: `services/memory_service/tests/test_mvp_142_previous_session_consolidation.py`
- Create: `tests/realinfra/memory-service/test_mvp_142_unsummarized_session.py`
- Test: both new files

**Interfaces:**
- Consumes: IF-07 `screen_text`, `CrisisLevel`; WS-17.6 `get_session_records`, `store_session_summary`.
- Produces: `PostgresRepository.get_latest_unsummarized_session(user_id, *, exclude_session_id)`.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_142_previous_session_consolidation.py`
```python
"""MVP-142/MVP-148: sessions are consolidated and summaries flag crisis (WS-17.15)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from services.memory_service.src.domain.consolidation import ConsolidationPipeline
from services.memory_service.src.domain.context_assembler import ContextAssembler
from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService


@dataclass
class _Rec:
    content: str
    role: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = {"role": self.role}


class _ConsolidationRepo:
    def __init__(self, user_id, previous_session) -> None:
        self.previous_session, self.user_id = previous_session, user_id
        self.summaries: list[dict] = []
        self.facts: list[dict] = []

    async def get_latest_unsummarized_session(self, user_id, *, exclude_session_id):
        return self.previous_session

    async def get_session_records(self, session_id, limit=500):
        now = datetime.now(timezone.utc)
        return [{"id": uuid4(), "user_id": self.user_id, "session_id": session_id, "tier": "tier_3_session",
                 "content": text, "content_type": "message", "retention_category": "medium_term",
                 "importance_score": 0.5, "emotional_valence": None, "record_metadata": {"role": "user"},
                 "retention_strength": 1.0, "is_safety_critical": False, "created_at": now, "accessed_at": now}
                for text in ("I feel stressed about work", "my sister is visiting", "I am worried about money",
                             "I have been sleeping badly")]

    async def store_session_summary(self, data): self.summaries.append(data)
    async def store_user_fact(self, data): self.facts.append(data)
    async def get_user_records(self, user_id, **kwargs): return []
    async def get_user_summaries(self, user_id, **kwargs): return []
    async def get_user_facts(self, user_id, **kwargs): return []
    async def get_safety_critical_categories(self, user_id): return ([], None)


async def test_mvp_142_previous_session_is_consolidated_when_a_new_session_starts() -> None:
    user_id, previous, current = uuid4(), uuid4(), uuid4()
    repo = _ConsolidationRepo(user_id, previous)
    service = MemoryService(settings=MemoryServiceSettings(enable_decay=False), context_assembler=ContextAssembler(),
                            consolidation_pipeline=ConsolidationPipeline(), postgres_repo=repo)
    await service.assemble_context(user_id=user_id, session_id=current, current_message="hello again",
                                   token_budget=8000, include_safety=True, include_therapeutic=True,
                                   retrieval_query=None, priority_topics=[])
    assert [s["session_id"] for s in repo.summaries] == [previous]


async def test_mvp_148_summary_flags_crisis_by_category_code_only() -> None:
    records = [_Rec("I want to kill myself tonight", "user"), _Rec("I hear how much pain you are in", "assistant"),
               _Rec("I have a plan and the pills are ready", "user")]
    result = await ConsolidationPipeline().generate_summary(records)
    first_sentence = result.summary.split(".")[0]
    assert result.summary.startswith("Safety: crisis indicators detected")
    assert "suicide_plan" in first_sentence or "suicidal_ideation" in first_sentence
    assert "pills" not in first_sentence and "kill" not in first_sentence
    assert "crisis" in result.key_topics
    assert "problem_solving" not in result.techniques_used
```
and in `tests/realinfra/memory-service/test_mvp_142_unsummarized_session.py`
```python
"""MVP-142: the unsummarized-session query is correct under RLS (WS-17.15)."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from services.memory_service.src import main as memory_main
from solace_common.request_context import acting_as_user

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_142_latest_unsummarized_session_is_per_user(cpm_default, seed_users, phi_codec) -> None:
    owner, other = await seed_users(2)
    runtime = SimpleNamespace(pg_pool=object(), redis=None, phi_codec=phi_codec, event_plane=None)
    service = memory_main.build_memory_service(runtime, memory_main.SETTINGS)
    summarized, pending, current, foreign = uuid4(), uuid4(), uuid4(), uuid4()
    for user_id, session_id in ((owner, summarized), (owner, pending), (owner, current), (other, foreign)):
        with acting_as_user(str(user_id)):
            await service.store_memory(user_id=user_id, session_id=session_id, content="synthetic",
                                       content_type="message", tier="tier_3_session",
                                       retention_category="medium_term", importance_score=Decimal("0.5"),
                                       metadata={"role": "user"})
    repo = service._postgres_repo
    with acting_as_user(str(owner)):
        await repo.store_session_summary({"session_id": summarized, "user_id": owner, "session_number": 1,
                                          "summary_text": "synthetic"})
        assert await repo.get_latest_unsummarized_session(owner, exclude_session_id=current) == pending
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_142_previous_session_consolidation.py -o "addopts=" -p no:logging -q`, then `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/memory-service/test_mvp_142_unsummarized_session.py -o "addopts=" -p no:logging -q`
Expected: FAIL. No summary is stored (nothing triggers consolidation); the summary reads `Session with 3 exchanges. Therapeutic approaches: problem_solving...` with no crisis indication; `get_latest_unsummarized_session` does not exist.
- [ ] **Step 3: Implement** the lazy consolidation, the crisis-aware summary and the repository query. The summary carries category codes and the highest level, never message text, and runs before the `Session with N exchanges` sentence. Remove the dead therapeutic-event writers; verify with `Select-String -Path services\memory_service\src\infrastructure\postgres_repo.py -Pattern "store_therapeutic_event|get_crisis_events"` returning no match (the erasure DELETE on `therapeutic_events` stays).
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run the memory suite (`services/memory_service/tests`), `tests/integration/test_session_lifecycle.py` and `tests/realinfra/memory-service` (with `$env:SOLACE_REQUIRE_REALINFRA="1"`), one process each, with the Step 2 prefixes; also `tests/clinical` in its own process. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.16: Honest, idempotent decay; safety categories never decay; the end_session race closed
**Covers:** MVP-275, MVP-276, MVP-284, MVP-286
**Severity:** low
**Wave:** 5
**Depends on:** WS-17.15
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-05 (default assumed: decay archival disabled, `MEMORY_SERVICE_ENABLE_DECAY_ARCHIVAL=false`; safety-critical and never-decay categories keep strength 1.0)
**Real infra:** yes
**Files:**
- Modify: `services/memory_service/src/infrastructure/postgres_repo.py` (`apply_batch_decay` (lines 393-442) sets `retention_strength = greatest(0, least(1, exp(-rate * days_since(coalesce(accessed_at, created_at)))))` without multiplying the stored strength, still built with `func.exp`/`func.greatest`, and excludes never-decay categories and `is_safety_critical` rows; new `archive_below_threshold(user_id, threshold: Decimal) -> int` for non-critical rows)
- Modify: `services/memory_service/src/domain/decay_manager.py` (`NEVER_DECAY_CATEGORIES` gains `safety`, `crisis`, `emergency`)
- Modify: `services/memory_service/src/domain/service.py` (`_apply_decay_model` (lines 757-800) reports `archived` as rows actually archived, 0 unless `enable_decay_archival`; the in-memory path computes strength from the base the same way; `end_session` pops the session under `_tier_lock` before any await and returns an empty result if another caller already ended it)
- Modify: `services/memory_service/src/domain/models.py` (`MemoryServiceSettings.enable_decay_archival: bool = False`)
- Create: `services/memory_service/tests/test_mvp_276_decay_rules.py`
- Create: `tests/realinfra/memory-service/test_mvp_275_batch_decay.py`
- Test: both new files; `services/memory_service/tests/test_rev29_a4_reconcile.py`, `services/memory_service/tests/test_decay_unit_consistency.py`

**Interfaces:**
- Consumes: WS-17.7 `is_safety_critical`.
- Produces: `MemoryServiceSettings.enable_decay_archival` (env `MEMORY_SERVICE_ENABLE_DECAY_ARCHIVAL`); `PostgresRepository.archive_below_threshold(user_id, threshold)`.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_276_decay_rules.py`
```python
"""MVP-276/284/286: safety never decays, counts are honest, end_session is race-free (WS-17.16)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from services.memory_service.src.domain.consolidation import ConsolidationPipeline
from services.memory_service.src.domain.decay_manager import DecayAction, DecayManager
from services.memory_service.src.domain.models import MemoryRecord, MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService


@pytest.mark.parametrize("category", ["safety", "crisis", "emergency"])
def test_mvp_276_safety_categories_never_decay(category) -> None:
    result = DecayManager().apply_decay(uuid4(), Decimal("1.0"), category,
                                        datetime.now(timezone.utc) - timedelta(days=100))
    assert result.new_strength == Decimal("1.0") and result.action == DecayAction.KEEP


async def test_mvp_286_archived_count_reports_only_real_archives() -> None:
    service = MemoryService(settings=MemoryServiceSettings(), consolidation_pipeline=ConsolidationPipeline())
    user_id, session_id = uuid4(), uuid4()
    service._tier_3_session[user_id] = [MemoryRecord(user_id=user_id, session_id=session_id, content="synthetic",
                                                     created_at=datetime.now(timezone.utc) - timedelta(days=100))]
    result = await service.consolidate(user_id, session_id, False, False, False, True)
    assert result.memories_archived == 0


async def test_mvp_284_concurrent_end_session_does_not_raise() -> None:
    class _SlowPipeline(ConsolidationPipeline):
        async def generate_summary(self, records):
            await asyncio.sleep(0)
            return await super().generate_summary(records)

    service = MemoryService(settings=MemoryServiceSettings(enable_auto_consolidation=False),
                            consolidation_pipeline=_SlowPipeline())
    user_id = uuid4()
    started = await service.start_session(user_id=user_id, session_type="therapeutic", initial_context={})
    await service.add_message(user_id=user_id, session_id=started.session_id, role="user", content="synthetic",
                              emotion_detected=None, importance_override=None, metadata={})
    results = await asyncio.gather(*(service.end_session(user_id, started.session_id, False, True) for _ in range(2)))
    assert sorted(r.message_count for r in results) == [0, 1]
```
and in `tests/realinfra/memory-service/test_mvp_275_batch_decay.py`
```python
"""MVP-275/MVP-276: batch decay is idempotent and skips safety-critical rows (WS-17.16)."""
from __future__ import annotations

import math
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from services.memory_service.src import main as memory_main
from solace_common.request_context import acting_as_user

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def _stored(service, user_id, text):
    return await service.store_memory(user_id=user_id, session_id=uuid4(), content=text, content_type="message",
                                      tier="tier_3_session", retention_category="medium_term",
                                      importance_score=Decimal("0.5"), metadata={"role": "user"})


async def test_mvp_275_repeated_batch_decay_does_not_compound(cpm_default, seed_users, phi_codec, superuser_conn) -> None:
    (user_id,) = await seed_users(1)
    service = memory_main.build_memory_service(
        SimpleNamespace(pg_pool=object(), redis=None, phi_codec=phi_codec, event_plane=None), memory_main.SETTINGS)
    with acting_as_user(str(user_id)):
        plain = await _stored(service, user_id, "synthetic plain note")
        crisis = await _stored(service, user_id, "I've been feeling suicidal")
    await superuser_conn.execute("UPDATE memory_records SET accessed_at = now() - interval '10 days', "
                                 "created_at = now() - interval '10 days' WHERE user_id = $1", user_id)
    with acting_as_user(str(user_id)):
        for _ in range(10):
            await service._postgres_repo.apply_batch_decay(user_id, decay_factor=Decimal("0.05"))
    strengths = {r["id"]: r["retention_strength"] for r in await superuser_conn.fetch(
        "SELECT id, retention_strength FROM memory_records WHERE user_id = $1", user_id)}
    assert strengths[plain.record_id] == pytest.approx(math.exp(-0.5), abs=0.01)
    assert strengths[crisis.record_id] == pytest.approx(1.0)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_276_decay_rules.py -o "addopts=" -p no:logging -q`, then `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/memory-service/test_mvp_275_batch_decay.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `crisis`/`safety`/`emergency` decay at the medium-term fallback rate; `memories_archived` counts records below 0.3 that were never archived; the second concurrent `end_session` raises `KeyError`; ten SQL runs leave the plain row near 0.0067 (`0.6065 ** 10`) instead of 0.6065.
- [ ] **Step 3: Implement** the formula, exclusions, honest counts and the pop-under-lock in the files above; keep `func.exp(`, `func.greatest` and the per-category rates in the Core statement so `test_sprint5_regression.py` and `test_rev29_a4_reconcile.py` stay valid.
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run the memory suite (`services/memory_service/tests`), `tests/integration/test_session_lifecycle.py` and `tests/realinfra/memory-service` (with `$env:SOLACE_REQUIRE_REALINFRA="1"`), one process each, with the Step 2 prefixes. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.17: Memory verifies the orchestrator-to-memory provider contract with obo tokens and uuid5 session ids
**Covers:** none (IF-11/IF-20 provider side of `orchestrator__memory`, requested by WS-11 for WS-11.12; supports MVP-065, owned by WS-11, and MVP-357, owned by WS-05)
**Severity:** high
**Wave:** 2
**Depends on:** WS-17.4, WS-11.12
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-03 (default assumed: the orchestrator calls memory with an `orchestrator-service` obo token bound to the user)
**Real infra:** no
**Files:**
- Create: `services/memory_service/tests/test_mvp_065_contract_orchestrator_memory.py`
- Test: `services/memory_service/tests/test_mvp_065_contract_orchestrator_memory.py`

**Interfaces:**
- Consumes: `solace_security.middleware.get_current_user` accepting obo service tokens (WS-07, W1, IF-04); IF-04 `ServiceTokenManager.create_service_token(service_name, target_service=None, *, on_behalf_of=None)`; IF-20 `solace_testing.contracts.verify_provider(app, contracts_dir, *, auth_headers)` (WS-05.7); consumer contract files `tests/contracts/orchestrator__memory/context.json` and `session_message.json` (WS-11.12, same wave); IF-06 client session ids mapped to `uuid5(NAMESPACE_URL, "solace:session:" + raw)` by the orchestrator (`session_uuid`, WS-10); fixture `memory_service_inmem` and `_authorize` (WS-17.4).
- Produces: provider verification of `tests/contracts/orchestrator__memory/{context,session_message}.json` with an obo token, and acceptance of a uuid5 client session id that memory has never seen through `/session/start`, for WS-11.12 (WS-11's cross-workstream request to WS-17); the `orchestrator__memory` entry that WS-05.10's coverage guard looks for (a memory test file containing `verify_provider(` and `orchestrator__memory`).

- [ ] **Step 1: Write the test** in `services/memory_service/tests/test_mvp_065_contract_orchestrator_memory.py`
```python
"""IF-11/IF-20 provider side of WS-11.12: memory honours orchestrator__memory with obo tokens (WS-17.17)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.memory_service.src.api import router
from solace_security.auth import AuthSettings
from solace_security.middleware import _get_jwt_manager
from solace_security.service_auth import ServiceTokenManager
from solace_testing.contracts import verify_provider

pytestmark = pytest.mark.contract
REPO = Path(__file__).resolve().parents[3]
CONTRACTS = REPO / "tests" / "contracts" / "orchestrator__memory"
USER = "8a3f0d4e-0000-4000-8000-000000000001"


def _obo_headers(monkeypatch, user_id: str) -> dict[str, str]:
    monkeypatch.setenv("AUTH_SECRET_KEY", "synthetic-contract-test-secret-0123456789abcdef")
    _get_jwt_manager.cache_clear()
    credentials = ServiceTokenManager(auth_settings=AuthSettings()).create_service_token(
        "orchestrator-service", target_service="memory-service", on_behalf_of=user_id)
    return {"Authorization": f"Bearer {credentials.token}"}


def _app(memory_service_inmem) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/memory")
    app.state.memory_service = memory_service_inmem
    return app


async def test_mvp_065_memory_satisfies_the_orchestrator_contracts(memory_service_inmem, monkeypatch) -> None:
    contracts = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(CONTRACTS.glob("*.json"))}
    assert {"context.json", "session_message.json"} <= set(contracts), f"WS-11.12 contract files missing in {CONTRACTS}"
    owners = {c["interaction"]: c["request"]["body_example"]["user_id"] for c in contracts.values()}
    outcome = verify_provider(_app(memory_service_inmem), CONTRACTS,
                              auth_headers=lambda interaction: _obo_headers(monkeypatch, owners[interaction]))
    if inspect.isawaitable(outcome):
        await outcome


def test_mvp_065_a_uuid5_client_session_id_is_accepted_without_session_start(memory_service_inmem, monkeypatch) -> None:
    session_id = str(uuid5(NAMESPACE_URL, "solace:session:s1"))  # IF-06 session_uuid("s1")
    client = TestClient(_app(memory_service_inmem))
    headers = _obo_headers(monkeypatch, USER)
    for role, content in (("user", "synthetic hello there"), ("assistant", "synthetic reply")):
        response = client.post("/api/v1/memory/session/message", headers=headers, json={
            "user_id": USER, "session_id": session_id, "role": role, "content": content,
            "metadata": {"turn_id": "00000000-0000-4000-e000-000000000001", "thread_id": "t-1"}})
        assert response.status_code == 201, response.text
        assert response.json()["session_id"] == session_id
    stored = memory_service_inmem._tier_3_session[UUID(USER)]
    assert [r.content for r in stored if str(r.session_id) == session_id] == ["synthetic hello there", "synthetic reply"]
```
- [ ] **Step 2: Run it and record the result**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_065_contract_orchestrator_memory.py -o "addopts=" -p no:logging -q`
Expected: this is a provider-contract lock, not a defect fix. On today's code the first test stops at the file assertion (WS-11.12 has not written the files) and, with the files present, both tests fail with 403: `/context` and `/session/message` compare a `UUID` body to the `str` principal id (`services/memory_service/src/api.py:111`, `:187`), so the obo user is refused (MVP-059); that red proof is WS-17.4 Step 2. With WS-17.4 and WS-11.12 landed both may pass at the first run; record that in the ledger. No session start is needed for a uuid5 session id (`services/memory_service/src/domain/service.py:386` treats an unknown session as optional, and WS-17.12 rejects only a session owned by another user). A failure now is a real mismatch between WS-11.12's examples and memory's `ContextAssemblyRequest`/`AddMessageRequest`.
- [ ] **Step 3: Implement** nothing in production code. A Step 2 mismatch is reported, not patched here: to WS-11 under R-17-11 when the orchestrator's example departs from memory's schema (IF-11 keeps memory canonical), or as a ledger fix task `WS-17.F<k>` when memory's own schema is wrong. Do not create or edit files under `tests/contracts/`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`, then the same prefix with `tests/contracts/test_contract_coverage.py -k memory` in its own process. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.18: Memory verifies the therapy-to-memory provider contract with forwarded obo tokens
**Covers:** none (IF-11/IF-20 provider side of `therapy__memory`; supports MVP-357, owned by WS-05, and WS-15.10's consumer test)
**Severity:** medium
**Wave:** 2
**Depends on:** WS-17.4, WS-15.10, WS-05.10
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-03 (default assumed: therapy forwards the inbound obo token, so memory sees an `orchestrator-service` obo principal bound to the user)
**Real infra:** no
**Files:**
- Create: `services/memory_service/tests/test_mvp_if11_memory_provider_contract.py`
- Test: `services/memory_service/tests/test_mvp_if11_memory_provider_contract.py`

**Interfaces:**
- Consumes: IF-20 `solace_testing.contracts.verify_provider(app, contracts_dir, *, auth_headers)` (WS-05.7); `tests/contracts/therapy__memory/context.json` (created by WS-05.10 in W2 phase A from WS-17.19's published examples; WS-15.10's consumer test reads it first); fixture `memory_service_inmem` and `_authorize` (WS-17.4).
- Produces: the `therapy__memory` provider verification that WS-05.10's coverage guard looks for (a memory test file containing `verify_provider(` and `therapy__memory`).

- [ ] **Step 1: Write the test** in `services/memory_service/tests/test_mvp_if11_memory_provider_contract.py`
```python
"""IF-11/IF-20: memory honours the therapy contracts with forwarded obo tokens (WS-17.18)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from fastapi import FastAPI

from services.memory_service.src.api import router
from solace_security.auth import AuthSettings
from solace_security.middleware import _get_jwt_manager
from solace_security.service_auth import ServiceTokenManager
from solace_testing.contracts import verify_provider

pytestmark = pytest.mark.contract
CONTRACTS = Path(__file__).resolve().parents[3] / "tests" / "contracts" / "therapy__memory"


async def test_if11_memory_satisfies_the_therapy_contracts(memory_service_inmem, monkeypatch) -> None:
    files = sorted(CONTRACTS.glob("*.json"))
    assert files, f"no contract files in {CONTRACTS}"
    monkeypatch.setenv("AUTH_SECRET_KEY", "synthetic-contract-test-secret-0123456789abcdef")
    _get_jwt_manager.cache_clear()
    owners = {}
    for path in files:
        contract = json.loads(path.read_text(encoding="utf-8"))
        owners[contract["interaction"]] = contract["request"]["body_example"]["user_id"]
    tokens = ServiceTokenManager(auth_settings=AuthSettings())

    def auth_headers(interaction: str) -> dict[str, str]:
        credentials = tokens.create_service_token(
            "orchestrator-service", target_service="memory-service", on_behalf_of=owners[interaction])
        return {"Authorization": f"Bearer {credentials.token}"}

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/memory")
    app.state.memory_service = memory_service_inmem
    outcome = verify_provider(app, CONTRACTS, auth_headers=auth_headers)
    if inspect.isawaitable(outcome):
        await outcome
```
- [ ] **Step 2: Run it and record the result**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_if11_memory_provider_contract.py -o "addopts=" -p no:logging -q`
Expected: this is a provider-contract lock, not a defect fix. On today's code every interaction returns 403 (`UUID != str`, MVP-059); that red proof is WS-17.4 Step 2. With WS-17.4 and WS-15.10 landed it may pass at the first run; record that in the ledger. A failure now is a real schema mismatch between therapy's example and memory's `ContextAssemblyRequest`/`ContextAssemblyResponse`.
- [ ] **Step 3: Implement** nothing in production code. A Step 2 mismatch is reported, not patched here: to WS-15 under R-17-12 when therapy's example departs from memory's schema (IF-11 keeps memory canonical), or as a ledger fix task `WS-17.F<k>` when memory's own schema is wrong. If `tests/contracts/therapy__memory/` is empty although WS-05.10 is done, stop and record a ledger Ruling against WS-05.10; do not create files under `tests/contracts/`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`, then the same prefix with `tests/contracts/test_contract_coverage.py -k memory` in its own process. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-17.19: Memory publishes the canonical therapy-to-memory context examples
**Covers:** none (IF-11/IF-20 provider shapes of `therapy__memory`, requested by WS-05 as its request 16 for WS-05.10; supports MVP-357, owned by WS-05)
**Severity:** medium
**Wave:** 0
**Depends on:** WS-17.1
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/memory_service/src/schemas.py` (module constants `THERAPY_CONTEXT_REQUEST_EXAMPLE = {"user_id": "00000000-0000-4000-a000-000000000001", "session_id": "00000000-0000-4000-b000-000000000001", "current_message": "example message"}` and `THERAPY_CONTEXT_RESPONSE_EXAMPLE = {"user_id": "00000000-0000-4000-a000-000000000001", "assembled_context": "example context", "total_tokens": 12, "sources_used": ["tier_3_session"], "assembly_time_ms": 3, "retrieval_count": 1}`; `ContextAssemblyRequest` and `ContextAssemblyResponse` gain `model_config = ConfigDict(json_schema_extra={"examples": [<the matching constant>]})`; field names, types, defaults and validation stay as they are)
- Create: `services/memory_service/tests/test_mvp_357_context_schema_examples.py`
- Test: `services/memory_service/tests/test_mvp_357_context_schema_examples.py`

**Interfaces:**
- Consumes: none.
- Produces: `services.memory_service.src.schemas.THERAPY_CONTEXT_REQUEST_EXAMPLE` and `THERAPY_CONTEXT_RESPONSE_EXAMPLE`, the provider's canonical `/context` examples, also exposed as the `examples` of both models' JSON schema (so memory's OpenAPI shows them). For WS-05.10 (WS-05's request 16): it copies the request example verbatim into `tests/contracts/therapy__memory/context.json` and builds the 200 `body_schema` from the response example's fields, instead of re-deriving them from memory's source. `user_id` stays the only required request field, as R-17-04 asks. WS-17.18 later verifies the provider against that file.

- [ ] **Step 1: Write the failing test** in `services/memory_service/tests/test_mvp_357_context_schema_examples.py`
```python
"""Supports MVP-357 (WS-05.10): memory publishes the canonical therapy__memory /context examples (IF-11 provider shapes)."""
from __future__ import annotations

from uuid import UUID

from services.memory_service.src.schemas import (
    THERAPY_CONTEXT_REQUEST_EXAMPLE,
    THERAPY_CONTEXT_RESPONSE_EXAMPLE,
    ContextAssemblyRequest,
    ContextAssemblyResponse,
)

USER = UUID("00000000-0000-4000-a000-000000000001")


def test_mvp_357_request_example_validates_and_names_the_user() -> None:
    request = ContextAssemblyRequest.model_validate(THERAPY_CONTEXT_REQUEST_EXAMPLE)
    assert request.user_id == USER and request.session_id == UUID("00000000-0000-4000-b000-000000000001")
    assert set(THERAPY_CONTEXT_REQUEST_EXAMPLE) == {"user_id", "session_id", "current_message"}
    schema = ContextAssemblyRequest.model_json_schema()
    assert schema["required"] == ["user_id"] and schema["examples"] == [THERAPY_CONTEXT_REQUEST_EXAMPLE]


def test_mvp_357_response_example_validates_and_carries_the_contract_fields() -> None:
    response = ContextAssemblyResponse.model_validate(THERAPY_CONTEXT_RESPONSE_EXAMPLE)
    assert response.user_id == USER
    assert set(THERAPY_CONTEXT_RESPONSE_EXAMPLE) == {
        "user_id", "assembled_context", "total_tokens", "sources_used", "assembly_time_ms", "retrieval_count"}
    assert ContextAssemblyResponse.model_json_schema()["examples"] == [THERAPY_CONTEXT_RESPONSE_EXAMPLE]


def test_mvp_357_examples_are_synthetic() -> None:
    assert THERAPY_CONTEXT_REQUEST_EXAMPLE["current_message"] == "example message"
    assert THERAPY_CONTEXT_RESPONSE_EXAMPLE["assembled_context"] == "example context"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests/test_mvp_357_context_schema_examples.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ImportError: cannot import name 'THERAPY_CONTEXT_REQUEST_EXAMPLE' from 'services.memory_service.src.schemas'`. With only the constants added, the `examples` assertions fail with `KeyError: 'examples'`: neither model carries a JSON-schema example today (`schemas.py:89-111`), so the contract writer has no provider-owned example and would re-derive one from memory's source, which is how consumer contracts drift from the provider (MVP-357).
- [ ] **Step 3: Implement** the two constants and the two `model_config` lines (import `ConfigDict` from pydantic). Change no field, default or validator; the W3 `AddMessageRequest.role` change stays WS-17.12's.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `Set-Location D:\Repo\Contextual-Chatbot; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q`, then the same prefix with `tests/alignment` in its own process (it imports memory-service internals). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-17.19:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| Id | Owner | Change | Files | Needed by |
|---|---|---|---|---|
| R-17-01 | WS-03 | W0 phase A2 env sync for memory-service per IF-02: POSTGRES_* as solace_app with TLS, `REDIS_URL`, `KAFKA_BOOTSTRAP_SERVERS`, `MEMORY_ENVIRONMENT=${ENVIRONMENT}`, registry entries `WEAVIATE_ENABLED` (dev default false) and `MEMORY_EMBEDDINGS_PROVIDER` (dev default none); retire `MEMORY_DB_*`, `MEMORY_REDIS_HOST/PORT`, `MEMORY_KAFKA_BOOTSTRAP_SERVERS`; drop the hard `depends_on: weaviate` (optional, D-04) | `deploy/env/env-contract.yaml`, `docker-compose.yml` | WS-17.1 |
| R-17-02 | WS-03 | W3 phase A: mark `OPENAI_API_KEY` retired and drop its ".env.example" placeholder; W5 phase A: register `MEMORY_SERVICE_ENABLE_DECAY_ARCHIVAL` (not secret, not required, dev default false) | `deploy/env/env-contract.yaml`, `.env.example` | WS-17.13, WS-17.16 |
| R-17-03 | WS-04 | W0 phase A2: the conformance test expects memory SPEC flags `uses_postgres=True, uses_redis=True, event_plane="publish_consume", handles_phi=True, audited=True, rate_limited=False, edge=False`; retire or replace the `configure_phi_encryption` AST check in `tests/integration/test_phi_encryption_activation.py`, because IF-01 adoption removes that call from every service main | `tests/bootstrap/test_service_conformance.py`, `tests/integration/test_phi_encryption_activation.py` | WS-17.1 |
| R-17-04 | WS-05 | W2 phase A: `verify_provider` (WS-05.7) and realinfra fixtures `pg_migrated` (with `dsn_app`, `dsn_superuser`) and `redis_url` (WS-05.2). Contract files: `orchestrator__memory` (context, session_message) are now produced by WS-11.12; `tests/contracts/therapy__memory/context.json` (with `body_example.user_id`) is produced by WS-05.10 (W2 phase A, at WS-15's request R14), which copies memory's published examples from WS-17.19 (WS-05's request 16); WS-17.18 depends on WS-05.10 and on WS-15.10, whose consumer test reads the file | `src/solace_testing/contracts.py`, `src/solace_testing/realinfra.py`, `tests/contracts/therapy__memory/context.json` | WS-17.3, WS-17.17, WS-17.18 |
| R-17-05 | WS-06 | IF-05 registration window: services call `ServiceEventPlane.subscribe(...)` from their lifespan after `start_service_runtime` returns, so consumer start must wait for subscriptions (or `subscribe` must be accepted until consumers start) | `src/solace_events/service_plane.py` | WS-17.11 |
| R-17-06 | WS-14 | W3 phase A (planned as WS-14.13): IF-16 helper `register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches=None)` running the erasure under `acting_as_user`, writing `ACCOUNT_ERASED_IN_SERVICE` and publishing `user.erasure.completed` | `src/solace_infrastructure/gdpr/consumer.py` | WS-17.11 |
| R-17-07 | WS-07 | `acting_as_user` (W0 A1); obo service tokens accepted by `get_current_user` (W1); `can_access_subject(principal, subject_user_id) -> bool` and `set_subject_context(subject_user_id)` exported from `solace_security.middleware` (W3 A) | `src/solace_common/request_context.py`, `src/solace_security/middleware.py` | WS-17.5, WS-17.12, WS-17.17, WS-17.18 |
| R-17-08 | WS-09 | W2 phase A: `PHICodec` with `encrypt/decrypt(value, *, field, owner_id)` and `active_key_id`; W3 phase A: `audit_phi_access(resource_type, *, subject=None)` calling module-level `audit(...)` | `src/solace_security/phi_codec.py`, `src/solace_security/audit_runtime.py` | WS-17.5, WS-17.14 |
| R-17-09 | WS-12 | W1 phase A1: `solace_clinical.screen_text`, `normalize_text`, `CATEGORIES`, `LEXICON_VERSION` and `tests/clinical/crisis_phrases_v1.yaml`; W5: add memory crisis retention and the crisis-aware summary rule to the D-05 record | `src/solace_clinical/__init__.py`, `tests/clinical/crisis_phrases_v1.yaml`, `docs/CLINICAL-VALIDATION.md` | WS-17.7, WS-17.15 |
| R-17-10 | WS-02 | W0 phase A2: scaffold `018_memory` (down `017_orchestrator_checkpoints`); 009 rewrites the `memory_records` policy with NULLIF and grants DML on memory tables to solace_app by default privileges; 001 loses the duplicate `memory_records.access_count` | `migrations/versions/009_platform_roles_rls.py`, `migrations/versions/001_initial_schema.py` | WS-17.3 |
| R-17-11 | WS-11 | W2: `MemoryServiceClient` writes turns with POST `/api/v1/memory/session/message` using roles `user`/`assistant` only and no `metadata.role`, sends UUID session ids (uuid5 mapping per IF-06) and obo tokens, and maps memory 503 to degraded `memory_unavailable` and 404 to a foreign session | `services/orchestrator_service/src/infrastructure/clients.py` | WS-17.12 |
| R-17-12 | WS-15 | W2: therapy's memory client uses memory's canonical schemas (tier values `tier_4_episodic`/`tier_5_semantic`, `ContextAssemblyResponse` fields) and forwarded obo headers | `services/therapy_service/src/infrastructure/context_assembler.py` | WS-17.18 |

## New interfaces

- `services.memory_service.src.main.build_memory_service(runtime: ServiceRuntime, settings: MemoryServiceAppSettings) -> MemoryService` and `register_memory_erasure(plane, memory_service) -> None` (WS-17.1, WS-17.11).
- `PostgresRepository(*, codec: PHICodec | FieldCodec, pool_name: str = "default")`, `FieldCodec` Protocol, `MEMORY_PHI_FIELDS` with AAD field names `memory_records.content`, `memory_records.record_metadata`, `session_summaries.summary_text` (WS-17.5).
- Repository reads `get_user_records(user_id, *, tier=None, session_id=None, since=None, min_importance=None, limit=50, offset=0)`, `get_user_summaries(user_id, *, limit=10, exclude_session_id=None)`, `get_user_facts(user_id, *, limit=20)`, `get_safety_critical_categories(user_id)`, `get_latest_unsummarized_session(user_id, *, exclude_session_id)`, `archive_below_threshold(user_id, threshold)` (WS-17.6, 17.9, 17.15, 17.16).
- `MemoryPersistenceError` (HTTP 503 `{"detail": "Memory store unavailable"}`), `SessionOwnershipError` (HTTP 404 `{"detail": "Session not found"}`), `WeaviateUnavailableError`.
- `MemoryService.purge_user_state(user_id)` and `MemoryService.build_erasure()` (IF-16 `purge_caches` and `erasure` arguments).
- `rank_bm25(documents, query, *, limit, k1=1.5, b=0.75)`.
- `ContextAssembler.assemble(..., previous_summary=None, facts=())` and the display order system, safety, profile, therapeutic, previous summary, facts, retrieved, history, current message.
- `AddMessageRequest.role` restricted to `user|assistant` (IF-11 provider schema change, W3).
- `schemas.THERAPY_CONTEXT_REQUEST_EXAMPLE` and `THERAPY_CONTEXT_RESPONSE_EXAMPLE`, also the JSON-schema `examples` of `ContextAssemblyRequest`/`ContextAssemblyResponse` (WS-17.19, W0); WS-05.10 copies them into the `therapy__memory` contract.
- `build_memory_service` raises `StartupDependencyError` in a strict environment without `runtime.pg_pool` or `runtime.phi_codec` (WS-17.5).
- Settings `MemoryServiceAppSettings.embeddings_provider: Literal["none"]` (`MEMORY_EMBEDDINGS_PROVIDER`), `WeaviateSettings.enabled` (`WEAVIATE_ENABLED`), `MemoryServiceSettings.enable_decay_archival` (`MEMORY_SERVICE_ENABLE_DECAY_ARCHIVAL`).
- Realinfra fixtures in `tests/realinfra/memory-service/conftest.py`: `superuser_conn`, `seed_users`, `cpm_default`, `phi_codec` (the same names and semantics as WS-18's `tests/realinfra/personality-analytics/conftest.py`).

## Self-review

**Coverage**

| Register id | Severity | Task |
|---|---|---|
| MVP-059 | high | WS-17.4 |
| MVP-060 | high | WS-17.5 |
| MVP-061 | high | WS-17.6 |
| MVP-062 | high | WS-17.11 |
| MVP-142 | medium | WS-17.15 |
| MVP-143 | medium | WS-17.12 |
| MVP-144 | medium | WS-17.7 |
| MVP-145 | medium | WS-17.13 |
| MVP-146 | medium | WS-17.9 (ordering half also fixed in WS-17.8) |
| MVP-147 | medium | WS-17.8 |
| MVP-148 | medium | WS-17.15 |
| MVP-149 | medium | WS-17.6 (HTTP-level tests land in WS-17.4) |
| MVP-275 | low | WS-17.16 |
| MVP-276 | low | WS-17.16 |
| MVP-277 | low | WS-17.13 (moved from W5 to W3: the same change as MVP-145) |
| MVP-278 | low | WS-17.10 (moved from W5 to W3: the same function as MVP-280/281) |
| MVP-279 | low | WS-17.13 |
| MVP-280 | low | WS-17.10 |
| MVP-281 | low | WS-17.10 |
| MVP-282 | low | WS-17.10 (moved from W5 to W3: the same function) |
| MVP-283 | low | WS-17.6 |
| MVP-284 | low | WS-17.16 |
| MVP-285 | low | WS-17.12 |
| MVP-286 | low | WS-17.16 |

All 24 ids are covered; no id is left uncovered.

**Incoming requests**
- WS-11.12 (provider verification of `orchestrator__memory/{context,session_message}.json`, obo token, uuid5 client session id): WS-17.17, W2, file `services/memory_service/tests/test_mvp_065_contract_orchestrator_memory.py`.
- WS-09.15 (CR-10, IF-13 adoption in the ADOPTERS file `services/memory_service/src/infrastructure/postgres_repo.py`): WS-17.5, W2 (the import, `encryption_key_id = codec.active_key_id`, no retired encryptor names; pinned by `test_cr10_repository_adopts_the_if13_codec_and_no_legacy_encryptor`).
- WS-05.10 coverage guard (provider verification per REST pair): `orchestrator__memory` in WS-17.17, `therapy__memory` in WS-17.18.
- WS-05.10 (request 16 of WS-05: memory publishes the canonical `therapy__memory` context examples in `services/memory_service/src/schemas.py`): WS-17.19, W0.

**Review focus (numbered as in the skeleton)**
- 1 (crisis disclosures outside the phrase set): memory does not screen ingress, but its retention must over-retain: WS-17.7 (curly-apostrophe and paraphrase variants, the whole phrase set, a disclosure after a negated earlier clause, quoted speech); WS-17.15 (crisis-aware summaries).
- 2 (alert-path partial failures): not in WS-17 code; memory pages nobody.
- 3 (foreign-but-valid identifiers, context-less work under RLS): WS-17.4 (another user's memory is 403), WS-17.5 (RLS rejects a write for another user), WS-17.6 (another user's rows are invisible), WS-17.11 (erasure under `acting_as_user` spares other users), WS-17.12 (a foreign session id, an unassigned clinician), WS-17.15 and WS-17.16 (background work per user on real Postgres).
- 4 (PHI escaping through unplanned sinks): WS-17.5 (503 body without exception text, ciphertext in raw SQL), WS-17.7 (metadata carries category codes, never matched text), WS-17.13 (no embedding egress, no query text in logs), WS-17.15 (summaries flag crisis by category code only).
- 5 (strict boots with partially valid configuration): WS-17.1 (Weaviate off unless enabled), WS-17.5 (a strict build without Postgres or the codec fails instead of running non-durable), WS-17.11 (a Weaviate init failure is loud), WS-17.13 (only the `none` embeddings provider is accepted).

REVIEW_FOCUS: 1=WS-17.7,WS-17.15; 3=WS-17.4,WS-17.5,WS-17.6,WS-17.11,WS-17.12,WS-17.15,WS-17.16; 4=WS-17.5,WS-17.7,WS-17.13,WS-17.15; 5=WS-17.1,WS-17.5,WS-17.11,WS-17.13

**Interface consistency**
- IF-01 flags match the skeleton notes: Postgres and Redis used, event plane `publish_consume` (erasure consumer and completion event), PHI codec and audit on; internal service (no CORS, no rate limiting).
- IF-03: every RLS access goes through `ConnectionPoolManager.acquire/transaction`; background and consumer work runs under `acting_as_user`; slot 018 uses the NULLIF policy form.
- IF-11: memory stays the canonical provider for `/context` and `/session/message`; status codes 503 (store unavailable) and 404 (foreign session) are additive; the W3 role restriction is announced to WS-11 and WS-15 (R-17-11, R-17-12).
- IF-13: field names and owner AAD are declared in the repository module; key id written to `encryption_key_id`.
- IF-16: store names `memory_postgres`, `memory_redis`, `memory_weaviate` are unchanged from today's registry; `purge_caches` is `MemoryService.purge_user_state`.
- WS-18 uses the same realinfra fixture names, the same `FakeCodec` semantics and the same IF-16 registration shape (`register_personality_erasure`).

**Open risks**
- WS-17.2 deliberately keeps the known-wrong statements (`metadata` column, `user_facts` table) until WS-17.5/17.6/17.11 so their RED tests stay meaningful; memory writes keep failing (status quo) until WS-17.5 lands in W2.
- Redis working-memory blobs (tier 2) still hold plaintext turn text for up to 4 hours; no register entry covers this. Recommended for the W2 or W3 bug-finding round (codec-encrypt the items or drop the tier-2 write-through).
- Retention and summaries depend on the provisional lexicon (D-05); over-retention is the accepted failure mode and the phrase set guards recall.
- Facts as `memory_records` rows is a D-01 decision recorded in slot 018; if the owner wants a `user_facts` table, slot 018 can add it before the first deployment.
- R-17-05 (when services may subscribe) is unresolved in IF-05 text; WS-17.11 assumes subscription from the lifespan after the runtime starts.
- The dead `services/memory_service/src/config.py` declares `MEMORY_DB_*`/`MEMORY_REDIS_*` prefixes; if WS-03's retired-name scan counts declarations as reads, it will flag that file and WS-17 removes those classes on request.
- `can_access_subject` and `verify_provider` signatures are frozen by WS-07/WS-05 at their wave start; the tests above call them exactly as IF-04/IF-20 describe and tolerate a coroutine `verify_provider`.
- Lazy previous-session consolidation adds work to the first `/context` call of a session; it is keyword-only and bounded, and the W5 e2e journey measures it against the 8 s agent timeout.
- The Weaviate-enabled path is not exercised on real infrastructure because D-04 keeps it off.
- `tests/contracts/therapy__memory/context.json` (resolved): WS-05.10 writes it from WS-17.19's published examples; WS-17.18 depends on WS-05.10 and WS-15.10 and stops for a ledger Ruling only if the directory is still empty.

## Review responses

- Automated check, Error: plan a task for WS-09's CR-10 request (needed by WS-09.15: adopt IF-13 in the ADOPTERS file, stop the retired encryptor calls, write `codec.active_key_id`): fixed. WS-17.5 already rewrote `services/memory_service/src/infrastructure/postgres_repo.py` (the memory ADOPTERS path, WS-09-phi-codec-audit-llm.md:1709) through the codec with `encryption_key_id=codec.active_key_id`, but used only a local `FieldCodec` Protocol, so WS-09.15's import check would have failed. WS-17.5 now imports `solace_security.phi_codec.PHICodec`, its Produces line names WS-09.15, the new test `test_cr10_repository_adopts_the_if13_codec_and_no_legacy_encryptor` pins the import and the absence of `configure_phi_encryption`/`FieldEncryptor` (today in `services/memory_service/src/main.py:95-100`; WS-17.1 deletes that block), and Step 5 runs WS-09.15's lock once it lands. The request's Files column is prose ("the eight ADOPTERS files", WS-09-phi-codec-audit-llm.md:1770), so a file-matching check cannot see the match.
- Automated check, Error: plan a task for WS-11's request (needed by WS-11.12: provider verification of `orchestrator__memory/{context,session_message}.json`, obo token, uuid5 client session id) in `services/memory_service/tests/test_mvp_065_contract_orchestrator_memory.py`: fixed. New task WS-17.17 (W2) creates that file (`test_mvp_065_memory_satisfies_the_orchestrator_contracts`, `test_mvp_065_a_uuid5_client_session_id_is_accepted_without_session_start`), depends on WS-17.4 and WS-11.12 (which creates the two contract files, WS-11-orchestrator-edge-contracts.md:1189) and names WS-11.12 in Produces. Both tests fail today with 403 because `/context` and `/session/message` compare a `UUID` body with the `str` principal id (`services/memory_service/src/api.py:111`, `:187`); no session start is needed afterwards (`service.py:386` treats an unknown session as optional). The old combined provider test in WS-17.4 is split: orchestrator into WS-17.17, therapy into new task WS-17.18 (depends on WS-15.10). WS-17.4 keeps only the MVP-059 ownership test and `Depends on: none`, so it and the W2 chain behind it (WS-17.5 to WS-17.9) do not wait on consumer contract files. The old version asserted that both contract directories were non-empty with no dependency on their producers.
- Automated check, Warnings: none reported.
- Self-found while resolving the above: WS-17.11 consumed WS-14's IF-16 helper without naming its task; it now depends on WS-14.13. WS-17.5 consumed `runtime.phi_codec` and IF-13 without naming them; it now depends on WS-09.7 and WS-04.8 (both W2 phase A). R-17-04 is corrected: WS-11.12 produces the orchestrator files, and the therapy file is an open ownership question.
- Reconciliation 5 (WS-05.10 named this workstream's `services/memory_service/src/schemas.py` as the source of the `therapy__memory` contract): fixed by a new task. WS-05's request 16 asks memory, the IF-11 provider, to publish the canonical `/context` examples, and new task WS-17.19 (W0, memory lane, after WS-17.1) adds `THERAPY_CONTEXT_REQUEST_EXAMPLE` and `THERAPY_CONTEXT_RESPONSE_EXAMPLE` as module constants and as the JSON-schema `examples` of `ContextAssemblyRequest`/`ContextAssemblyResponse`, RED first (`ImportError`, then `KeyError: 'examples'`), keeping `user_id` the only required request field (R-17-04). Its Produces line names WS-05.10, and WS-05.10 depends on it; W0 is used because WS-05.10 is a W2 phase-A task and WS-17's W2 work is phase B. R-17-04 now names WS-05.10 as the producer of `therapy__memory/context.json`, so WS-17.18 depends on WS-05.10 (and still on WS-15.10), its Consumes and Step 3 no longer say that no task writes the file, and the open risk is closed. The File map, the waves line, New interfaces and the incoming-requests list now include WS-17.19.
- decision 2026-10-19 D-04: checked the D-04 gates of WS-17.1, WS-17.9, WS-17.10 and WS-17.13 and the rest of this plan. No text names Portkey or any LLM route as the production path. The parts of D-04 these tasks rely on (memory's OpenAI embeddings disabled, `WEAVIATE_ENABLED=false`, keyword/BM25 retrieval, minimum-necessary context) are unchanged by the decision, so no task changed. The real-user LLM route (Anthropic on AWS Bedrock in strict environments) is WS-09.18's.
