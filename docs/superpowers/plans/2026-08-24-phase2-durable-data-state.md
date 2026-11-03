# Phase 2 — Durable Data & State Implementation Plan

**Goal:** Make every conversation turn, session, therapy record, and assembled context survive a process restart and be correct across ≥2 replicas, with no silently-swallowed write ever returning success.

**Architecture:** Six microservices carry state that today lives only in in-process dicts or is written-through with the failure path swallowed. This phase wires the orchestrator to persist each turn to the Memory Service and to use a durable Postgres LangGraph checkpointer; makes the Memory Service read persisted tiers (+ real semantic search) during context assembly and stop swallowing Postgres write failures; makes the Diagnosis live-session Redis store degrade gracefully instead of 500-ing when Redis is down; and makes Therapy persist treatment plans and a full session transcript. All persistence is fail-loud in production and best-effort-but-observable on the user-facing hot path where a hard failure would otherwise drop a live response.

**Tech Stack:** Python 3.12, FastAPI, LangGraph (`langgraph-checkpoint-postgres` `AsyncPostgresSaver`), PostgreSQL + RLS (SQLAlchemy async / asyncpg), Redis (`redis.asyncio`, `fakeredis.aioredis` in tests), Weaviate, structlog, Pydantic v2 / pydantic-settings, pytest / pytest-asyncio.

**Spec:** `docs/ENTERPRISE-READINESS-ASSESSMENT.md` (blocker IDs BLK-15…BLK-19, BLK-21 with exact `file:line` and fix column) and `docs/ENTERPRISE-RELEASE-PLAN.md` → "Phase 2 — Durable data & state" (Closes/Exit criteria).

---

## Global Constraints

- **TDD, red→green.** Every task writes a failing test first, runs it and sees it FAIL for the stated reason, then makes the minimal change and sees it PASS. No implementation code is written before its failing test exists.
- **Prove on REAL infra, not mocks, wherever the harness allows.** The pytest harness on the dev machine has no live Postgres/Redis/Kafka, so: Redis paths are proven with `fakeredis.aioredis` (a real Redis protocol implementation, not a `Mock`); Postgres-repository row round-trips are proven against fake repo doubles that return real asyncpg-shaped `dict` rows; and the durable **behavioural** proofs that require a live server (LangGraph `AsyncPostgresSaver.setup()` against real PG, RLS-scoped reads, cross-replica recovery) are written as `@pytest.mark.integration` tests that run in staging (infra-gated), matching the repo's existing B.3 convention. The in-harness test asserts the exact call contract that the live test then exercises end-to-end.
- **Fail loud in production.** A durable write that fails on a background/record path re-raises (never returns a success DTO). On the user-facing synchronous response path (orchestrator turn persistence), a memory-write failure is logged at `error` and does not crash the reply — but it is never reported as stored.
- **No silent crisis-path drops.** Crisis turns keep their existing fail-loud persistence semantics (diagnosis `_persist_session(fail_loud=crisis_turn)`, therapy `_handle_crisis(crisis=True)`); nothing in this phase weakens them except the *explicitly intended* graceful degrade when Redis is entirely unavailable (BLK-18), where durable persistence is impossible anyway and the Kafka crisis-event path remains the alert channel.
- **NO git commits.** This project forbids commits. The regression gate at the end of every task is running the affected service's test suite (green), NOT `git commit`.
- **Test command (Windows PowerShell; the Bash tool hook breaks on `;`).** `python` on PATH is a broken anaconda shim — call the interpreter by full path. PYTHONPATH MUST include both repo root and `src/`:

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q
  ```

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `services/memory_service/src/domain/service.py` | Modify | Re-raise Postgres write failures in `store_memory` / `add_message` (BLK-19); read persisted tiers + semantic search in `assemble_context` (BLK-17); fix `id`→`record_id` row mapping. |
| `services/memory_service/src/infrastructure/rag_retriever.py` | Create | `MemoryRAGRetriever` — a `RetrieverProtocol` adapter over `PostgresRepository` + `WeaviateRepository` that gives the previously-dead `RAGPipeline` a real backing store (BLK-17). |
| `services/memory_service/src/main.py` | Modify | Build a `RAGPipeline` from the retriever adapter and inject it into `MemoryService` (BLK-17). |
| `services/memory_service/tests/test_blk19_write_failloud.py` | Create | Prove a Postgres write failure never returns `stored=True` / HTTP-201. |
| `services/memory_service/tests/test_blk17_persisted_context.py` | Create | Prove `assemble_context` surfaces persisted-tier + semantic-search content after the in-memory cache is empty. |
| `services/diagnosis_service/src/infrastructure/redis_session_repository.py` | Modify | `try_save_session` returns `False` and `save_session_cas_merge` no-ops when the store is uninitialized (Redis down) — stop 500-ing `/assess` and `/session/start` (BLK-18). |
| `services/diagnosis_service/tests/test_blk18_redis_down_degrade.py` | Create | Prove the store degrades gracefully and `start_session` succeeds with Redis down. |
| `services/orchestrator_service/src/langgraph/memory_writer.py` | Create | `persist_turn(state)` — writes the user message + assistant reply to the Memory Service after each graph turn (BLK-15). |
| `services/orchestrator_service/src/infrastructure/clients.py` | Modify | `MemoryServiceClient.store_memory` accepts an optional `session_id` and forwards it (BLK-15). |
| `services/orchestrator_service/src/langgraph/graph_builder.py` | Modify | `invoke` calls `persist_turn` after the graph runs (BLK-15); async `setup_checkpointer()` / `aclose_checkpointer()` use `AsyncPostgresSaver` correctly and hold it for the process lifetime (BLK-16). |
| `services/orchestrator_service/src/main.py` | Modify | Pass the Postgres DSN into `OrchestratorGraphBuilder` and `await setup_checkpointer()` before compile; close it on shutdown (BLK-16). |
| `services/orchestrator_service/tests/test_blk15_turn_persistence.py` | Create | Prove a graph turn stores user + assistant messages to the Memory Service. |
| `services/orchestrator_service/tests/test_blk16_pg_checkpointer.py` | Create | Prove the Postgres checkpointer is entered via `async with` and `setup()` is awaited and the saver is held. |
| `services/therapy_service/src/domain/entities.py` | Modify | Add a `transcript` field to `TherapySessionEntity` (BLK-21). |
| `services/therapy_service/src/domain/service.py` | Modify | Populate the full session entity (transcript + topics/skills/insights/summary/risk) at `end_session`; persist the treatment plan via `unit_of_work.treatment_plans` and recover it cross-worker in `start_session` (BLK-21). |
| `services/therapy_service/src/infrastructure/postgres_repository.py` | Modify | Serialize/deserialize the `transcript` column in the therapy-session repo (BLK-21). |
| `migrations/versions/009_therapy_session_transcript.py` | Create | Add the `transcript` JSONB column to `therapy_sessions` (BLK-21). |
| `services/therapy_service/tests/test_end_session_persistence.py` | Modify | Make the `MagicMock` unit-of-work's `treatment_plans.save/get` async so the new plan-persistence call in `start_session` is awaited. |
| `services/therapy_service/tests/test_blk21_transcript_plan.py` | Create | Prove the transcript is populated + persisted at `end_session` and the treatment plan is persisted + recovered cross-worker. |

---

### Task 1: BLK-19 — Memory Service must not swallow Postgres write failures

A durable-write failure in `store_memory` / `add_message` is caught by `except Exception: logger.exception(...)` and the method still returns `stored=True` (HTTP 201). Make the durable-tier write fail loud (re-raise) so the API surfaces a 500 and the caller can retry — a message is never reported stored when the durable write failed.

**Files:**
- Modify `services/memory_service/src/domain/service.py` — `store_memory` (lines 157–161), `add_message` (lines 390–394).
- Create `services/memory_service/tests/test_blk19_write_failloud.py`.

**Interfaces:**
- Consumes: `PostgresRepository.store_memory_record(record) -> UUID` (may raise on DB failure).
- Produces: `MemoryService.store_memory(...) -> StoreMemoryResult` and `MemoryService.add_message(...) -> AddMessageResult` — both now propagate a durable-write exception instead of returning a success DTO. In-memory tiers are still updated first (the raise happens after the cache write, which is intended — the next read-through/retry reconciles).

**Steps:**

- [ ] Write the failing test file `services/memory_service/tests/test_blk19_write_failloud.py`:

  ```python
  """BLK-19: Memory Postgres write failures must not be swallowed into a 201 'stored'."""
  from __future__ import annotations

  from decimal import Decimal
  from unittest.mock import AsyncMock
  from uuid import uuid4

  import pytest

  from services.memory_service.src.domain.service import MemoryService
  from services.memory_service.src.domain.models import MemoryServiceSettings


  def _service_with_failing_pg() -> MemoryService:
      repo = AsyncMock()
      repo.store_memory_record = AsyncMock(side_effect=RuntimeError("pg pool exhausted"))
      return MemoryService(settings=MemoryServiceSettings(), postgres_repo=repo)


  @pytest.mark.asyncio
  async def test_store_memory_reraises_on_durable_write_failure() -> None:
      svc = _service_with_failing_pg()
      await svc.initialize()
      with pytest.raises(RuntimeError, match="pg pool exhausted"):
          await svc.store_memory(
              user_id=uuid4(), session_id=uuid4(), content="durable please",
              content_type="message", tier="tier_3_session",
              retention_category="medium_term", importance_score=Decimal("0.5"),
              metadata={"role": "user"},
          )


  @pytest.mark.asyncio
  async def test_add_message_reraises_on_durable_write_failure() -> None:
      svc = _service_with_failing_pg()
      await svc.initialize()
      with pytest.raises(RuntimeError, match="pg pool exhausted"):
          await svc.add_message(
              user_id=uuid4(), session_id=uuid4(), role="user",
              content="durable please", emotion_detected=None,
              importance_override=Decimal("0.5"), metadata={},
          )
  ```

  Note: `MemoryService.initialize()` calls `postgres_repo.initialize()`; the `AsyncMock` returns a coroutine for it, so init succeeds.

- [ ] Run it — expect FAIL (both currently return a success DTO; no exception is raised):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/memory_service/tests/test_blk19_write_failloud.py -o "addopts=" -p no:logging -q
  ```

- [ ] In `services/memory_service/src/domain/service.py` `store_memory`, replace the swallow at lines 157–161:

  ```python
          # Persist tiers 3-5 to postgres
          if self._postgres_repo and tier in ("tier_3_session", "tier_4_episodic", "tier_5_semantic"):
              try:
                  await self._postgres_repo.store_memory_record(record)
              except Exception:
                  # BLK-19: never report a durable write as 'stored'. Re-raise so the
                  # API surfaces a 500 and the caller retries instead of losing the message.
                  logger.exception("postgres_store_failed", record_id=str(record.record_id), tier=tier)
                  raise
  ```

- [ ] In the same file `add_message`, replace the swallow at lines 390–394:

  ```python
          # Persist to postgres
          if self._postgres_repo:
              try:
                  await self._postgres_repo.store_memory_record(record)
              except Exception:
                  # BLK-19: fail loud — a session message must never be reported added
                  # while the durable write failed.
                  logger.exception("postgres_add_message_failed", record_id=str(record.record_id))
                  raise
  ```

  Leave the best-effort Weaviate index (`store_memory`) and the best-effort Redis write-through (`add_message`) as-is — those are caches, not the durable record; only the Postgres write is load-bearing.

- [ ] Run the new test — expect PASS.

- [ ] **Regression gate** — run the full Memory Service suite (must stay green; existing tests use no `postgres_repo`, so the raise path is not hit there):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q
  ```

---

### Task 2: BLK-17 (part 1) — `assemble_context` reads persisted tiers

`assemble_context` passes only the in-process dicts (`self._tier_2_working`, `self._tier_3_session`, `self._user_profiles`) to the `ContextAssembler`, so after a restart or on a second worker the LLM gets empty long-term context. Read the persisted session/episodic/semantic tiers from Postgres and feed them into assembly. Also fix `_row_to_memory_record`, which reads `row["record_id"]` while `get_user_records` returns the column as `id` (a latent `KeyError` that silently drops every persisted row into the in-memory fallback).

**Files:**
- Modify `services/memory_service/src/domain/service.py` — `assemble_context` (lines 256–287), `_row_to_memory_record` (lines 645–662); add `_gather_persisted_context`.
- Create `services/memory_service/tests/test_blk17_persisted_context.py`.

**Interfaces:**
- Consumes: `PostgresRepository.get_user_records(user_id, tier, limit) -> list[dict]` (rows keyed `id`, `user_id`, `session_id`, `tier`, `content`, `content_type`, `retention_category`, `importance_score`, `metadata`, `created_at`).
- Produces: `MemoryService._gather_persisted_context(user_id, session_id) -> tuple[list[MemoryRecord], list[MemoryRecord]]` returning `(session_records, retrieval_records)`; `assemble_context` now passes the union of in-memory + persisted records as `session_memory` so `ContextAssembler` sees durable content.

**Steps:**

- [ ] Write the failing test `services/memory_service/tests/test_blk17_persisted_context.py`:

  ```python
  """BLK-17: assemble_context must surface persisted-tier content, not just in-memory dicts."""
  from __future__ import annotations

  from datetime import datetime, timezone
  from decimal import Decimal
  from unittest.mock import AsyncMock
  from uuid import uuid4

  import pytest

  from services.memory_service.src.domain.service import MemoryService
  from services.memory_service.src.domain.context_assembler import ContextAssembler
  from services.memory_service.src.domain.models import MemoryServiceSettings


  def _pg_row(user_id, session_id, content, tier="tier_3_session"):
      # asyncpg-shaped row: the memory_records column is 'id', not 'record_id'.
      return {
          "id": uuid4(), "user_id": user_id, "session_id": session_id, "tier": tier,
          "content": content, "content_type": "message", "retention_category": "medium_term",
          "importance_score": Decimal("0.7"), "metadata": {"role": "user"},
          "created_at": datetime.now(timezone.utc), "accessed_at": datetime.now(timezone.utc),
          "retention_strength": Decimal("1.0"), "emotional_valence": None,
      }


  @pytest.mark.asyncio
  async def test_assemble_context_includes_persisted_tier_after_cache_empty() -> None:
      user_id, session_id = uuid4(), uuid4()
      repo = AsyncMock()
      repo.initialize = AsyncMock()

      async def _get_user_records(uid, tier=None, limit=50, offset=0):
          if tier == "tier_3_session":
              return [_pg_row(user_id, session_id, "I have trouble sleeping most nights")]
          return []

      repo.get_user_records = AsyncMock(side_effect=_get_user_records)

      svc = MemoryService(
          settings=MemoryServiceSettings(),
          context_assembler=ContextAssembler(),
          postgres_repo=repo,
      )
      await svc.initialize()
      # NB: no in-memory tier state was populated — this simulates a fresh worker.
      result = await svc.assemble_context(
          user_id=user_id, session_id=session_id, current_message="how do I sleep better?",
          token_budget=8000, include_safety=False, include_therapeutic=False,
          retrieval_query=None, priority_topics=[],
      )
      assert "trouble sleeping" in result.assembled_context
      assert "recent_messages" in result.sources_used
  ```

- [ ] Run it — expect FAIL (`assembled_context` contains only the current message; persisted content is never read):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/memory_service/tests/test_blk17_persisted_context.py -o "addopts=" -p no:logging -q
  ```

- [ ] Fix `_row_to_memory_record` (lines 645–662) to accept the real `id` column:

  ```python
      @staticmethod
      def _row_to_memory_record(row: dict[str, Any]) -> MemoryRecord:
          """Convert a postgres row dict back to a MemoryRecord.

          The memory_records PK column is ``id`` (see postgres_repo table def); tolerate
          both keys so persisted rows are not silently dropped into the in-memory fallback.
          """
          return MemoryRecord(
              record_id=row.get("record_id") or row["id"],
              user_id=row["user_id"],
              session_id=row.get("session_id"),
              tier=row.get("tier", "tier_3_session"),
              content=row.get("content", ""),
              content_type=row.get("content_type", "message"),
              retention_category=row.get("retention_category", "medium_term"),
              importance_score=Decimal(str(row.get("importance_score", "0.5"))),
              emotional_valence=Decimal(str(row["emotional_valence"])) if row.get("emotional_valence") is not None else None,
              metadata=row.get("metadata", {}),
              created_at=row.get("created_at", datetime.now(timezone.utc)),
              accessed_at=row.get("accessed_at", datetime.now(timezone.utc)),
              retention_strength=Decimal(str(row.get("retention_strength", "1.0"))),
          )
  ```

- [ ] Add a `_gather_persisted_context` helper (place it just after `_row_to_memory_record`):

  ```python
      async def _gather_persisted_context(
          self, user_id: UUID, session_id: UUID | None
      ) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
          """Load durable tiers so context survives a restart / a second worker (BLK-17).

          Returns (session_records, retrieval_records):
            * session_records  = persisted tier_3 (+ tier_4 summaries) for the recent-
              messages section, scoped to session_id when provided;
            * retrieval_records = persisted tier_4/tier_5 (episodic summaries + semantic
              facts) that the assembler's retrieved-context section can rank.
          No-op (empty) when no Postgres repo is configured.
          """
          if self._postgres_repo is None:
              return [], []
          session_records: list[MemoryRecord] = []
          retrieval_records: list[MemoryRecord] = []
          try:
              for tier in ("tier_3_session", "tier_4_episodic", "tier_5_semantic"):
                  rows = await self._postgres_repo.get_user_records(user_id, tier=tier, limit=50)
                  for row in rows:
                      record = self._row_to_memory_record(row)
                      if tier == "tier_3_session":
                          if session_id is None or record.session_id == session_id:
                              session_records.append(record)
                      else:
                          retrieval_records.append(record)
          except Exception:
              logger.exception("assemble_context_persisted_read_failed", user_id=str(user_id))
          return session_records, retrieval_records
  ```

- [ ] Rewire `assemble_context` (lines 256–287) to merge persisted records into what the assembler sees. Replace the `if self._context_assembler:` block's `assemble(...)` call arguments:

  ```python
      async def assemble_context(self, user_id: UUID, session_id: UUID | None,
                                 current_message: str | None, token_budget: int,
                                 include_safety: bool, include_therapeutic: bool,
                                 retrieval_query: str | None,
                                 priority_topics: list[str]) -> ContextAssemblyResult:
          """Assemble context for LLM within token budget."""
          start_time = time.perf_counter()
          self._stats["total_assemblies"] += 1
          # BLK-17: read the durable tiers so a fresh worker / post-restart request is
          # not handed empty long-term context.
          persisted_session, persisted_retrieval = await self._gather_persisted_context(user_id, session_id)
          if self._context_assembler:
              # De-dup persisted rows against the in-memory session cache by record_id.
              in_mem_session = self._tier_3_session.get(user_id, [])
              seen = {r.record_id for r in in_mem_session}
              merged_session = in_mem_session + [r for r in persisted_session if r.record_id not in seen]
              result = await self._context_assembler.assemble(
                  user_id=user_id, session_id=session_id, current_message=current_message,
                  token_budget=token_budget, include_safety=include_safety,
                  include_therapeutic=include_therapeutic, retrieval_query=retrieval_query,
                  priority_topics=priority_topics, working_memory=self._tier_2_working.get(user_id, []),
                  session_memory=merged_session + persisted_retrieval,
                  user_profile=self._user_profiles.get(user_id, {}),
              )
              assembly_time_ms = int((time.perf_counter() - start_time) * 1000)
              return ContextAssemblyResult(
                  context_id=result.context_id, assembled_context=result.assembled_context,
                  total_tokens=result.total_tokens, token_breakdown=result.token_breakdown,
                  sources_used=result.sources_used, assembly_time_ms=assembly_time_ms,
                  retrieval_count=result.retrieval_count,
              )
          context = self._build_basic_context(user_id, session_id, current_message, token_budget)
          assembly_time_ms = int((time.perf_counter() - start_time) * 1000)
          estimated_tokens = len(context) // 4  # chars/4 matches ContextAssembler._estimate_tokens
          return ContextAssemblyResult(
              assembled_context=context, total_tokens=estimated_tokens,
              token_breakdown={"basic": estimated_tokens}, sources_used=["working_memory"],
              assembly_time_ms=assembly_time_ms,
          )
  ```

- [ ] Run the new test — expect PASS.

- [ ] **Regression gate** — Memory Service suite green:

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q
  ```

---

### Task 3: BLK-17 (part 2) — wire the RAGPipeline for real semantic retrieval

`RAGPipeline` (`infrastructure/rag_pipeline.py`) is fully built but has zero callers (`retriever=None`), so it is dead code the assessment flags as "wire or delete." Give it a real backing store — a Postgres/Weaviate retriever adapter — and call it from `assemble_context` when a `retrieval_query` is present, so semantic hits are ranked into the retrieved-context section instead of only substring matches.

**Files:**
- Create `services/memory_service/src/infrastructure/rag_retriever.py`.
- Modify `services/memory_service/src/domain/service.py` — constructor takes an optional `rag_pipeline`; `assemble_context` uses it.
- Modify `services/memory_service/src/main.py` — build the retriever + pipeline and inject.
- Extend `services/memory_service/tests/test_blk17_persisted_context.py` with a RAG test.

**Interfaces:**
- Produces: `MemoryRAGRetriever.retrieve(query, query_embedding, user_id, limit) -> list[dict]` — items shaped `{"doc_id": UUID, "content": str, "score": float, "source": str, "metadata": dict}` (matches `RAGPipeline._retrieve_documents`).
- Consumes: `WeaviateRepository.search_by_keyword(keyword, user_id, limit)` (returns objects with `.record_id`, `.content`, `.score`), falling back to `PostgresRepository.get_user_records`.

**Steps:**

- [ ] Write the adapter test first — append to `services/memory_service/tests/test_blk17_persisted_context.py`:

  ```python
  from services.memory_service.src.infrastructure.rag_retriever import MemoryRAGRetriever


  @pytest.mark.asyncio
  async def test_rag_retriever_falls_back_to_postgres_when_no_weaviate() -> None:
      user_id = uuid4()
      repo = AsyncMock()
      repo.get_user_records = AsyncMock(return_value=[_pg_row(user_id, uuid4(), "panic attacks at work", "tier_5_semantic")])
      retriever = MemoryRAGRetriever(postgres_repo=repo, weaviate_repo=None)
      docs = await retriever.retrieve(query="work stress", query_embedding=None, user_id=user_id, limit=5)
      assert docs and docs[0]["content"] == "panic attacks at work"
      assert docs[0]["source"] == "postgres"
  ```

- [ ] Run it — expect FAIL (module does not exist):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/memory_service/tests/test_blk17_persisted_context.py::test_rag_retriever_falls_back_to_postgres_when_no_weaviate -o "addopts=" -p no:logging -q
  ```

- [ ] Create `services/memory_service/src/infrastructure/rag_retriever.py`:

  ```python
  """Solace-AI Memory Service - RAG retriever adapter (BLK-17).

  Backs the RAGPipeline's RetrieverProtocol with the service's real stores:
  Weaviate keyword/vector search when available, else the Postgres persisted tiers.
  Wiring this (previously the pipeline had retriever=None → dead code) gives
  assemble_context a real semantic-retrieval path.
  """
  from __future__ import annotations

  from typing import Any
  from uuid import UUID, uuid4

  import structlog

  logger = structlog.get_logger(__name__)


  class MemoryRAGRetriever:
      """RetrieverProtocol adapter over PostgresRepository + WeaviateRepository."""

      def __init__(self, postgres_repo: Any = None, weaviate_repo: Any = None) -> None:
          self._postgres_repo = postgres_repo
          self._weaviate_repo = weaviate_repo

      async def retrieve(
          self, query: str, query_embedding: list[float] | None, user_id: UUID, limit: int
      ) -> list[dict[str, Any]]:
          # 1. Weaviate keyword search (no embedding needed) when available.
          if self._weaviate_repo is not None:
              try:
                  hits = await self._weaviate_repo.search_by_keyword(
                      keyword=query, user_id=user_id, limit=limit
                  )
                  if hits:
                      return [
                          {
                              "doc_id": getattr(h, "record_id", uuid4()),
                              "content": getattr(h, "content", ""),
                              "score": float(getattr(h, "score", 0.0)),
                              "source": "weaviate",
                              "metadata": getattr(h, "metadata", {}) or {},
                          }
                          for h in hits
                      ]
              except Exception as e:
                  logger.warning("rag_weaviate_retrieve_failed", error=str(e))

          # 2. Postgres persisted tiers (semantic + episodic) as the durable fallback.
          if self._postgres_repo is not None:
              docs: list[dict[str, Any]] = []
              try:
                  for tier in ("tier_5_semantic", "tier_4_episodic", "tier_3_session"):
                      rows = await self._postgres_repo.get_user_records(user_id, tier=tier, limit=limit)
                      for row in rows:
                          docs.append({
                              "doc_id": row.get("record_id") or row["id"],
                              "content": row.get("content", ""),
                              "score": float(row.get("importance_score", 0.5)),
                              "source": "postgres",
                              "metadata": row.get("metadata", {}) or {},
                          })
                      if len(docs) >= limit:
                          break
              except Exception as e:
                  logger.warning("rag_postgres_retrieve_failed", error=str(e))
              return docs[:limit]

          return []
  ```

- [ ] Run the adapter test — expect PASS.

- [ ] Add an optional `rag_pipeline` to `MemoryService.__init__` (after the `decay_manager` param) and store it:

  ```python
                   decay_manager: DecayManager | None = None,
                   rag_pipeline: Any | None = None) -> None:
  ```
  ```python
          self._decay_manager = decay_manager
          self._rag_pipeline = rag_pipeline  # BLK-17: optional RAGPipeline for semantic retrieval
  ```

- [ ] In `assemble_context`, when a `retrieval_query` is present and a pipeline is wired, fetch semantic docs and append them (as `MemoryRecord`s) to `persisted_retrieval` before assembly. Insert immediately after the `_gather_persisted_context` call:

  ```python
          persisted_session, persisted_retrieval = await self._gather_persisted_context(user_id, session_id)
          if retrieval_query and self._rag_pipeline is not None:
              try:
                  rag = await self._rag_pipeline.retrieve_and_grade(retrieval_query, user_id)
                  for doc in rag.documents:
                      persisted_retrieval.append(MemoryRecord(
                          record_id=doc.doc_id if isinstance(doc.doc_id, UUID) else uuid4(),
                          user_id=user_id, session_id=session_id, tier="tier_5_semantic",
                          content=doc.content, content_type="fact",
                          importance_score=Decimal(str(round(doc.grade_score or doc.score, 4))),
                          metadata=doc.metadata,
                      ))
              except Exception:
                  logger.exception("assemble_context_rag_failed", user_id=str(user_id))
  ```

- [ ] Wire the pipeline in `services/memory_service/src/main.py`. After `postgres_repo` is created (around line 151) and before `memory_service = MemoryService(...)`:

  ```python
      # BLK-17: give the (previously dead) RAGPipeline a real backing store so
      # assemble_context has a semantic-retrieval path.
      rag_pipeline = None
      try:
          from .infrastructure.rag_pipeline import RAGPipeline
          from .infrastructure.rag_retriever import MemoryRAGRetriever
          rag_pipeline = RAGPipeline(
              retriever=MemoryRAGRetriever(postgres_repo=postgres_repo, weaviate_repo=weaviate_repo)
          )
          logger.info("memory_rag_pipeline_wired")
      except Exception as e:
          logger.debug("memory_rag_pipeline_unavailable", error=str(e))
  ```

  and add `rag_pipeline=rag_pipeline,` to the `MemoryService(...)` call.

- [ ] Add a service-level RAG test to `services/memory_service/tests/test_blk17_persisted_context.py`:

  ```python
  @pytest.mark.asyncio
  async def test_assemble_context_uses_rag_pipeline_when_query_present() -> None:
      from services.memory_service.src.infrastructure.rag_pipeline import RAGPipeline

      user_id = uuid4()
      repo = AsyncMock()
      repo.initialize = AsyncMock()
      repo.get_user_records = AsyncMock(return_value=[])
      rag_repo = AsyncMock()
      rag_repo.get_user_records = AsyncMock(
          return_value=[_pg_row(user_id, uuid4(), "client responds well to grounding exercises", "tier_5_semantic")]
      )
      pipeline = RAGPipeline(retriever=MemoryRAGRetriever(postgres_repo=rag_repo, weaviate_repo=None))

      svc = MemoryService(
          settings=MemoryServiceSettings(), context_assembler=ContextAssembler(),
          postgres_repo=repo, rag_pipeline=pipeline,
      )
      await svc.initialize()
      result = await svc.assemble_context(
          user_id=user_id, session_id=uuid4(), current_message="what helps?",
          token_budget=8000, include_safety=False, include_therapeutic=False,
          retrieval_query="grounding", priority_topics=[],
      )
      assert "grounding exercises" in result.assembled_context
  ```

  (`RAGPipeline` default `SimpleGrader` scores the keyword "grounding" as relevant; `min_relevant_documents` default is 3, so the pipeline may report `INSUFFICIENT_CONTEXT` but still carries the single relevant doc in `context.documents` — which is what assembly consumes.)

- [ ] Run the extended test file — expect PASS.

- [ ] **Regression gate** — Memory Service suite green:

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/memory_service/tests -o "addopts=" -p no:logging -q
  ```

---

### Task 4: BLK-18 — Diagnosis live-session store degrades gracefully when Redis is down

When Redis is unreachable the store stays uninitialized (`self._initialized is False`, `self._client is None`); `try_save_session` then raises `RuntimeError`, which `save_session_cas_merge` propagates and `DiagnosisService._persist_session` does not catch (it only catches `SessionPersistConflict`), so `/assess` and `/session/start` return 500 on any Redis blip. Make the save path a no-op / `False` when uninitialized (gate on `_initialized`). The GDPR `delete_user_data` path keeps its fail-loud `RuntimeError` (erasure must never silently report success).

**Files:**
- Modify `services/diagnosis_service/src/infrastructure/redis_session_repository.py` — `try_save_session` (lines 179–180), `save_session_cas_merge` (lines 214–226).
- Create `services/diagnosis_service/tests/test_blk18_redis_down_degrade.py`.

**Interfaces:**
- `DiagnosisSessionStore.try_save_session(session) -> bool` — returns `False` (was: raised) when uninitialized.
- `DiagnosisSessionStore.save_session_cas_merge(session, max_retries=5) -> None` — no-ops when uninitialized (does not loop to `SessionPersistConflict`).

**Steps:**

- [ ] Write the failing test `services/diagnosis_service/tests/test_blk18_redis_down_degrade.py`:

  ```python
  """BLK-18: diagnosis must not 500 when Redis (live-session store) is down."""
  from __future__ import annotations

  from uuid import uuid4

  import pytest

  from services.diagnosis_service.src.domain.models import SessionState
  from services.diagnosis_service.src.domain.service import DiagnosisService
  from services.diagnosis_service.src.infrastructure.redis_session_repository import (
      DiagnosisSessionStore,
  )


  @pytest.mark.asyncio
  async def test_try_save_returns_false_when_uninitialized() -> None:
      store = DiagnosisSessionStore()  # no client injected, initialize() never called -> Redis down
      assert store._initialized is False
      assert await store.try_save_session(SessionState(user_id=uuid4())) is False


  @pytest.mark.asyncio
  async def test_cas_merge_is_noop_when_uninitialized() -> None:
      store = DiagnosisSessionStore()
      # Must NOT raise (previously looped to SessionPersistConflict / RuntimeError).
      await store.save_session_cas_merge(SessionState(user_id=uuid4()))


  @pytest.mark.asyncio
  async def test_start_session_survives_redis_down() -> None:
      store = DiagnosisSessionStore()  # uninitialized == Redis unreachable
      svc = DiagnosisService(session_store=store)
      await svc.initialize()
      # Previously raised RuntimeError("...redis not initialized...") -> API 500.
      result = await svc.start_session(uuid4(), "assessment", {}, None)
      assert result.session_id is not None
  ```

- [ ] Run it — expect FAIL (`try_save_session` raises `RuntimeError`; `start_session` raises through the uncaught RuntimeError):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/diagnosis_service/tests/test_blk18_redis_down_degrade.py -o "addopts=" -p no:logging -q
  ```

- [ ] In `redis_session_repository.py` `try_save_session`, replace lines 179–180:

  ```python
          if not self._initialized or self._client is None:
              # BLK-18: Redis unreachable -> degrade gracefully instead of raising.
              # The in-memory working copy remains authoritative for this turn; the
              # crisis alert path is Kafka, not this cache, so no crisis is dropped.
              logger.debug("diagnosis_session_save_skipped_uninitialized", session_id=str(session.session_id))
              return False
  ```

- [ ] In `save_session_cas_merge`, add the guard as the first statement (before the retry loop at line 219):

  ```python
      async def save_session_cas_merge(self, session: SessionState, max_retries: int = 5) -> None:
          """Durable write-through that never loses a safety flag (REV-16). On a CAS
          conflict, reload committed state, UNION the ``safety_flags`` into ``session``,
          rebase onto the committed version, and retry. Raises
          :class:`SessionPersistConflict` if it cannot converge.

          BLK-18: no-op when the store is uninitialized (Redis down) so callers that
          only catch SessionPersistConflict do not surface a 500 on a Redis blip."""
          if not self._initialized or self._client is None:
              return
          for _ in range(max_retries):
              if await self.try_save_session(session):
                  return
              committed = await self.get_session(session.session_id)
              if committed is not None:
                  merge_safety_flags(session, committed)
                  session.version = committed.version
          raise SessionPersistConflict(session.session_id)
  ```

  Leave `delete_user_data` (lines 258–281) unchanged — it must keep raising `RuntimeError` when uninitialized (GDPR fail-loud).

- [ ] Run the new test — expect PASS.

- [ ] **Regression gate** — Diagnosis Service suite green (verify `test_session_durability.py` and `test_phase_c_session_reconcile.py` still pass — the initialized/fakeredis paths are unaffected):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/diagnosis_service/tests -o "addopts=" -p no:logging -q
  ```

---

### Task 5: BLK-15 — Orchestrator persists each turn to the Memory Service

`MemoryServiceClient.store_memory` has zero callers, so no conversation turn is ever stored — the memory write path is orphaned. After each graph turn, store the user message and the assistant reply. Centralize it in `graph_builder.invoke` so both the `/chat` REST path and the WebSocket path (and `/batch`) get it for free.

**Files:**
- Modify `services/orchestrator_service/src/infrastructure/clients.py` — `MemoryServiceClient.store_memory` (lines 364–371) to accept `session_id`.
- Create `services/orchestrator_service/src/langgraph/memory_writer.py`.
- Modify `services/orchestrator_service/src/langgraph/graph_builder.py` — `invoke` (lines 488–503) calls `persist_turn`.
- Create `services/orchestrator_service/tests/test_blk15_turn_persistence.py`.

**Interfaces:**
- `MemoryServiceClient.store_memory(user_id: UUID, content: str, memory_type: str, session_id: UUID | None = None) -> ServiceResponse[dict]` — posts `{user_id, content, content_type, session_id?}` to `/api/v1/memory/store`.
- `persist_turn(state: OrchestratorState, *, client: MemoryServiceClient | None = None) -> dict[str, bool]` — returns `{"user_stored": bool, "assistant_stored": bool}`; stores `state["current_message"]` as `user_message` and `state["final_response"]` as `assistant_message`, both to `tier_3_session` via the client.

**Steps:**

- [ ] Write the failing test `services/orchestrator_service/tests/test_blk15_turn_persistence.py`:

  ```python
  """BLK-15: the orchestrator must store each turn (user msg + assistant reply) to Memory."""
  from __future__ import annotations

  from unittest.mock import AsyncMock, patch
  from uuid import uuid4

  import pytest

  from services.orchestrator_service.src.langgraph import (
      OrchestratorGraphBuilder,
      create_initial_state,
  )
  from services.orchestrator_service.src.langgraph.memory_writer import persist_turn


  @pytest.mark.asyncio
  async def test_persist_turn_stores_user_and_assistant() -> None:
      client = AsyncMock()
      client.store_memory = AsyncMock(return_value=type("R", (), {"success": True})())
      state = create_initial_state(str(uuid4()), str(uuid4()), "I feel anxious")
      state["final_response"] = "I hear you. Let's take a breath together."

      result = await persist_turn(state, client=client)

      assert result == {"user_stored": True, "assistant_stored": True}
      assert client.store_memory.await_count == 2
      types_stored = {c.kwargs.get("memory_type") or c.args[2] for c in client.store_memory.await_args_list}
      assert types_stored == {"user_message", "assistant_message"}


  @pytest.mark.asyncio
  async def test_graph_invoke_persists_turn() -> None:
      with patch(
          "services.orchestrator_service.src.langgraph.memory_writer.MemoryServiceClient"
      ) as mock_cls:
          instance = AsyncMock()
          instance.store_memory = AsyncMock(return_value=type("R", (), {"success": True})())
          mock_cls.return_value = instance
          import services.orchestrator_service.src.langgraph.memory_writer as mw
          mw._shared_writer_client = None  # reset the module singleton for isolation

          builder = OrchestratorGraphBuilder()
          builder.compile()
          state = create_initial_state(str(uuid4()), str(uuid4()), "Hello there")
          await builder.invoke(state)

          assert instance.store_memory.await_count == 2  # user + assistant
  ```

- [ ] Run it — expect FAIL (`memory_writer` does not exist; `invoke` never calls it):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/orchestrator_service/tests/test_blk15_turn_persistence.py -o "addopts=" -p no:logging -q
  ```

- [ ] Extend `MemoryServiceClient.store_memory` in `clients.py` (lines 364–371):

  ```python
      async def store_memory(
          self, user_id: UUID, content: str, memory_type: str, session_id: UUID | None = None
      ) -> ServiceResponse[dict[str, Any]]:
          """Store memory entry."""
          payload: dict[str, Any] = {
              "user_id": str(user_id), "content": content, "content_type": memory_type,
          }
          if session_id is not None:
              payload["session_id"] = str(session_id)
          return await self.post("/api/v1/memory/store", payload)
  ```

- [ ] Create `services/orchestrator_service/src/langgraph/memory_writer.py`:

  ```python
  """Solace-AI Orchestrator Service - Turn persistence to the Memory Service (BLK-15).

  After each graph turn the orchestrator stores the user message and the assistant
  reply to the Memory Service so conversation turns survive a restart / a second
  worker (the memory write path was previously orphaned — store_memory had zero
  callers). Failure is logged loud but does NOT crash the user-facing reply.
  """
  from __future__ import annotations

  from typing import Any
  from uuid import UUID

  import structlog
  from pydantic import Field
  from pydantic_settings import BaseSettings, SettingsConfigDict

  from ..infrastructure.clients import MemoryServiceClient

  logger = structlog.get_logger(__name__)


  class MemoryWriterSettings(BaseSettings):
      """Configuration for post-turn memory persistence."""
      enable_persistence: bool = Field(default=True)
      model_config = SettingsConfigDict(env_prefix="ORCHESTRATOR_MEMORY_WRITER_", env_file=".env", extra="ignore")


  _shared_writer_client: MemoryServiceClient | None = None
  _settings = MemoryWriterSettings()


  def _get_writer_client() -> MemoryServiceClient:
      """Reuse one MemoryServiceClient (httpx pool) across turns."""
      global _shared_writer_client
      if _shared_writer_client is None:
          _shared_writer_client = MemoryServiceClient()
      return _shared_writer_client


  async def persist_turn(
      state: dict[str, Any], *, client: MemoryServiceClient | None = None
  ) -> dict[str, bool]:
      """Store the user message and assistant reply for the just-completed turn.

      Returns {"user_stored": bool, "assistant_stored": bool}. Best-effort per message
      (a store failure is recorded as False, never raised) so a Memory outage cannot
      500 the chat response.
      """
      result = {"user_stored": False, "assistant_stored": False}
      if not _settings.enable_persistence:
          return result
      writer = client or _get_writer_client()
      raw_user = state.get("user_id")
      if not raw_user:
          return result
      try:
          user_id = UUID(str(raw_user))
      except (ValueError, TypeError):
          logger.warning("persist_turn_bad_user_id", user_id=str(raw_user))
          return result
      raw_session = state.get("session_id")
      try:
          session_id = UUID(str(raw_session)) if raw_session else None
      except (ValueError, TypeError):
          session_id = None

      user_message = state.get("current_message", "") or ""
      assistant_message = state.get("final_response", "") or ""

      if user_message:
          try:
              resp = await writer.store_memory(user_id, user_message, "user_message", session_id=session_id)
              result["user_stored"] = bool(getattr(resp, "success", False))
          except Exception as e:  # noqa: BLE001 - never crash the reply on a memory outage
              logger.error("persist_turn_user_store_failed", error=str(e))
      if assistant_message:
          try:
              resp = await writer.store_memory(user_id, assistant_message, "assistant_message", session_id=session_id)
              result["assistant_stored"] = bool(getattr(resp, "success", False))
          except Exception as e:  # noqa: BLE001
              logger.error("persist_turn_assistant_store_failed", error=str(e))
      if not (result["user_stored"] and result["assistant_stored"]):
          logger.warning("persist_turn_incomplete", **result)
      return result
  ```

- [ ] Call it from `graph_builder.invoke`. In `services/orchestrator_service/src/langgraph/graph_builder.py`, after the crisis-emit line (currently line 502), add the persist call:

  ```python
          result = await graph.ainvoke(state, config=config)
          await self._maybe_emit_crisis_event(result)
          # BLK-15: store this turn (user msg + assistant reply) to the Memory Service.
          try:
              from .memory_writer import persist_turn
              await persist_turn(result)
          except Exception as e:  # noqa: BLE001 - persistence must not break the response
              logger.error("turn_persistence_failed", error=str(e))
          return result
  ```

- [ ] Run the new test — expect PASS.

- [ ] **Regression gate** — Orchestrator Service suite green (existing `test_graph_builder.py` invokes the graph; `persist_turn` uses a real `MemoryServiceClient` whose HTTP call fails fast against no server and is swallowed, so those tests stay green):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q
  ```

---

### Task 6: BLK-16 — Orchestrator uses a durable Postgres LangGraph checkpointer

The graph always compiles with `InMemorySaver` (state lost on restart, not shared across workers), and the Postgres path is doubly broken: `AsyncPostgresSaver.from_conn_string(...)` returns an async context manager that is never entered and `.setup()` is never called. Fix the usage (`async with ... as saver: await saver.setup()`, held for the process lifetime) and pass the DSN from `main.py`.

**Files:**
- Modify `services/orchestrator_service/src/langgraph/graph_builder.py` — `_create_checkpointer` (lines 384–402); add `setup_checkpointer` / `aclose_checkpointer`.
- Modify `services/orchestrator_service/src/main.py` — lifespan (lines 157–163, shutdown 211–218).
- Create `services/orchestrator_service/tests/test_blk16_pg_checkpointer.py`.

**Interfaces:**
- `OrchestratorGraphBuilder(settings, postgres_connection_string: str | None)` — unchanged signature.
- `async OrchestratorGraphBuilder.setup_checkpointer() -> Any` — when a DSN is set and the postgres checkpointer package is available, enters `AsyncPostgresSaver.from_conn_string(dsn)`, awaits `saver.setup()`, stores the saver as `self._checkpointer` (held open), and returns it; otherwise returns the existing in-memory checkpointer.
- `async OrchestratorGraphBuilder.aclose_checkpointer() -> None` — exits the held context manager on shutdown.

**Steps:**

- [ ] Write the failing test `services/orchestrator_service/tests/test_blk16_pg_checkpointer.py` (a fake `AsyncPostgresSaver` proves the async-with + `setup()` + lifetime-hold contract without a live PG; the real-PG durability proof is the staging integration test below):

  ```python
  """BLK-16: the Postgres checkpointer must be entered via async-with and setup() awaited."""
  from __future__ import annotations

  import pytest

  import services.orchestrator_service.src.langgraph.graph_builder as gb
  from services.orchestrator_service.src.langgraph.graph_builder import (
      GraphBuilderSettings,
      OrchestratorGraphBuilder,
  )


  class _FakeSaver:
      def __init__(self) -> None:
          self.setup_called = False

      async def setup(self) -> None:
          self.setup_called = True


  class _FakeCM:
      """Mimics AsyncPostgresSaver.from_conn_string(...)'s async context manager."""
      def __init__(self, dsn: str) -> None:
          self.dsn = dsn
          self.entered = False
          self.exited = False
          self.saver = _FakeSaver()

      async def __aenter__(self) -> _FakeSaver:
          self.entered = True
          return self.saver

      async def __aexit__(self, *exc) -> None:
          self.exited = True


  class _FakeAsyncPostgresSaver:
      last_cm: _FakeCM | None = None

      @classmethod
      def from_conn_string(cls, dsn: str) -> _FakeCM:
          cls.last_cm = _FakeCM(dsn)
          return cls.last_cm


  @pytest.mark.asyncio
  async def test_setup_checkpointer_enters_cm_and_calls_setup(monkeypatch) -> None:
      monkeypatch.setattr(gb, "_POSTGRES_CHECKPOINT_AVAILABLE", True)
      monkeypatch.setattr(gb, "AsyncPostgresSaver", _FakeAsyncPostgresSaver, raising=False)

      builder = OrchestratorGraphBuilder(
          GraphBuilderSettings(enable_checkpointing=True),
          postgres_connection_string="postgresql://localhost:5432/orchestrator",
      )
      saver = await builder.setup_checkpointer()

      cm = _FakeAsyncPostgresSaver.last_cm
      assert cm is not None and cm.entered is True        # entered via async with
      assert saver.setup_called is True                   # setup() awaited
      assert builder.get_checkpointer() is saver          # held for process lifetime
      assert cm.exited is False                            # not closed yet

      await builder.aclose_checkpointer()
      assert cm.exited is True                             # closed on shutdown


  @pytest.mark.asyncio
  async def test_setup_checkpointer_noop_without_dsn() -> None:
      builder = OrchestratorGraphBuilder(GraphBuilderSettings(enable_checkpointing=True))
      saver = await builder.setup_checkpointer()
      # No DSN -> keep the in-memory checkpointer created in __init__.
      assert saver is builder.get_checkpointer()
      assert type(saver).__name__ == "MemorySaver"
  ```

- [ ] Run it — expect FAIL (`setup_checkpointer` / `aclose_checkpointer` do not exist):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/orchestrator_service/tests/test_blk16_pg_checkpointer.py -o "addopts=" -p no:logging -q
  ```

- [ ] Rewrite `_create_checkpointer` in `graph_builder.py` (lines 384–402) so the postgres branch defers creation to the async `setup_checkpointer` instead of calling `from_conn_string` un-entered:

  ```python
      def _create_checkpointer(self) -> Any:
          """Create the synchronous checkpointer. The durable AsyncPostgresSaver is
          created in setup_checkpointer() (it needs async ``async with`` + ``setup()``),
          so when a DSN is configured we start with InMemorySaver as a placeholder and
          swap it in setup_checkpointer() before compile."""
          if not self._settings.enable_checkpointing:
              return None
          if self._postgres_conn_str and not _POSTGRES_CHECKPOINT_AVAILABLE:
              logger.warning(
                  "postgres_checkpointer_unavailable",
                  msg="langgraph-checkpoint-postgres not installed, falling back to in-memory",
              )
          import os
          if not self._postgres_conn_str and os.environ.get("ENVIRONMENT", "development").lower() == "production":
              logger.warning(
                  "in_memory_checkpointer_in_production",
                  msg="No PERSISTENCE_POSTGRES_URL configured; InMemorySaver loses state on restart.",
              )
          return InMemorySaver()
  ```

  Add a `self._pg_cm = None` line in `__init__` (right after `self._checkpointer = self._create_checkpointer()`):

  ```python
          self._checkpointer = self._create_checkpointer()
          self._pg_cm: Any = None  # held AsyncPostgresSaver context manager (BLK-16)
  ```

- [ ] Add the async setup/close methods to `OrchestratorGraphBuilder` (place after `get_checkpointer`, around line 486):

  ```python
      async def setup_checkpointer(self) -> Any:
          """Create the durable Postgres checkpointer (BLK-16).

          Enters ``AsyncPostgresSaver.from_conn_string(dsn)`` via async-with, awaits
          ``setup()`` (creates the checkpoint tables), and HOLDS the context manager
          open for the process lifetime (closed in aclose_checkpointer on shutdown).
          No-op when no DSN is configured or the postgres package is unavailable —
          the in-memory checkpointer created in __init__ is kept."""
          if not self._postgres_conn_str or not _POSTGRES_CHECKPOINT_AVAILABLE:
              return self._checkpointer
          self._pg_cm = AsyncPostgresSaver.from_conn_string(self._postgres_conn_str)
          saver = await self._pg_cm.__aenter__()
          await saver.setup()
          self._checkpointer = saver
          self._compiled = None  # force recompile with the durable checkpointer
          logger.info("postgres_checkpointer_ready")
          return saver

      async def aclose_checkpointer(self) -> None:
          """Close the held AsyncPostgresSaver context manager on shutdown (BLK-16)."""
          if self._pg_cm is not None:
              try:
                  await self._pg_cm.__aexit__(None, None, None)
              finally:
                  self._pg_cm = None
  ```

- [ ] Run the new test — expect PASS.

- [ ] Wire `main.py`. Replace the graph-builder construction block (lines 157–163) so the DSN is passed and the checkpointer is set up before compile:

  ```python
      from .langgraph import OrchestratorGraphBuilder
      from .langgraph.graph_builder import GraphBuilderSettings
      from .config import PersistenceSettings

      graph_settings = GraphBuilderSettings(enable_checkpointing=settings.enable_checkpointing)
      # BLK-16: pass a Postgres DSN so conversation state is durable + shared across
      # replicas. Only when the persistence backend is configured for postgres.
      _persist = PersistenceSettings()
      _pg_dsn = _persist.postgres_url if _persist.checkpoint_backend == "postgres" else None
      graph_builder = OrchestratorGraphBuilder(graph_settings, postgres_connection_string=_pg_dsn)
      try:
          await graph_builder.setup_checkpointer()
      except Exception as e:
          if settings.environment == "production":
              raise RuntimeError(f"Durable checkpointer required in production: {e}") from e
          logger.warning("postgres_checkpointer_setup_failed", error=str(e))
      graph_builder.build()
      compiled_graph = graph_builder.compile()
  ```

- [ ] In the shutdown half of the lifespan (after `if event_bridge: await event_bridge.stop()`, around line 212), close the checkpointer:

  ```python
      logger.info("orchestrator_service_stopping")
      if event_bridge:
          await event_bridge.stop()
      await graph_builder.aclose_checkpointer()  # BLK-16: release the held AsyncPostgresSaver
  ```

- [ ] **Staging integration proof (infra-gated).** Add `services/orchestrator_service/tests/test_blk16_pg_checkpointer.py::test_real_pg_checkpoint_survives_rebuild` marked `@pytest.mark.integration`, requiring `ORCHESTRATOR_PG_DSN` env: build a graph with the real DSN, `await setup_checkpointer()`, invoke a turn on `thread_id="t-blk16"`, drop the builder, build a *fresh* `OrchestratorGraphBuilder` on the same DSN, `await setup_checkpointer()`, and assert `await get_checkpointer().aget({"configurable": {"thread_id": "t-blk16"}})` returns the persisted checkpoint. This is the Exit-criteria proof ("checkpointer is Postgres-backed … across a restart and a second worker") and runs in staging CI, not the dev harness.

- [ ] **Regression gate** — Orchestrator Service suite green (skip the integration marker):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -m "not integration" -q
  ```

---

### Task 7: BLK-21 (part 1) — Therapy session transcript field, populated + persisted at `end_session`

`TherapySessionEntity` has no transcript field, so the conversation content is lost when the session ends, and `end_session` persists only a skeletal entity (session ids + phase + skills). Add a `transcript` field, populate the full entity at `end_session` (transcript from the live session's messages, plus topics/skills/insights/summary/risk/safety), and serialize the new column in the Postgres repo + a migration.

**Files:**
- Modify `services/therapy_service/src/domain/entities.py` — `TherapySessionEntity` (add field after line 136).
- Modify `services/therapy_service/src/domain/service.py` — `end_session` entity construction (lines 376–387).
- Modify `services/therapy_service/src/infrastructure/postgres_repository.py` — `PostgresTherapySessionRepository.save` (lines 387–447) + `_row_to_entity` (lines 623–675).
- Create `migrations/versions/009_therapy_session_transcript.py`.
- Create `services/therapy_service/tests/test_blk21_transcript_plan.py`.

**Interfaces:**
- `TherapySessionEntity.transcript: list[dict[str, Any]]` (default `[]`) — the ordered `{role, content}` turns for the session.
- `end_session` now saves an entity carrying `transcript = list(session.messages)` plus `topics_covered`, `skills_practiced`, `insights_gained`, `summary`, `current_risk`, `safety_flags`, `mood_rating_start/end`.

**Steps:**

- [ ] Write the failing test `services/therapy_service/tests/test_blk21_transcript_plan.py` (transcript half):

  ```python
  """BLK-21: therapy end_session must persist a full session transcript + entity."""
  from __future__ import annotations

  from uuid import uuid4

  import pytest

  from services.therapy_service.src.domain.service import TherapyOrchestrator
  from services.therapy_service.src.domain.session_manager import SessionManager
  from services.therapy_service.src.domain.technique_selector import TechniqueSelector
  from services.therapy_service.src.infrastructure.repository import (
      TreatmentPlanRepository, TherapySessionRepository, TechniqueRepository,
      OutcomeMeasureRepository, UnitOfWork,
  )


  def _uow() -> UnitOfWork:
      return UnitOfWork(
          treatment_plans=TreatmentPlanRepository(), sessions=TherapySessionRepository(),
          techniques=TechniqueRepository(), outcomes=OutcomeMeasureRepository(),
      )


  @pytest.mark.asyncio
  async def test_end_session_persists_transcript() -> None:
      uow = _uow()
      orch = TherapyOrchestrator(
          technique_selector=TechniqueSelector(), session_manager=SessionManager(),
          unit_of_work=uow,
      )
      await orch.initialize()
      user_id, plan_id = uuid4(), uuid4()
      start = await orch.start_session(user_id, plan_id, {"diagnosis": "Depression"})

      # Record two turns onto the live session.
      session = orch._session_manager.get_session(start.session_id)
      session.messages.append({"role": "user", "content": "I felt low all week"})
      session.messages.append({"role": "assistant", "content": "Thank you for sharing that."})

      await orch.end_session(start.session_id, user_id)

      persisted = await uow.sessions.get(start.session_id)
      assert persisted is not None
      assert len(persisted.transcript) == 2
      assert persisted.transcript[0]["content"] == "I felt low all week"
      await orch.shutdown()
  ```

- [ ] Run it — expect FAIL (`TherapySessionEntity` has no `transcript`; `end_session` doesn't populate messages):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest "services/therapy_service/tests/test_blk21_transcript_plan.py::test_end_session_persists_transcript" -o "addopts=" -p no:logging -q
  ```

- [ ] Add the `transcript` field to `TherapySessionEntity` in `entities.py` (after `next_session_focus` on line 136):

  ```python
      summary: str = ""
      next_session_focus: str = ""
      # BLK-21: the full ordered conversation transcript ({role, content} turns) so the
      # session content is not lost when the live session is deleted at end_session.
      transcript: list[dict[str, Any]] = field(default_factory=list)
      version: int = 1
  ```

- [ ] Populate the full entity at `end_session` in `service.py` (replace the `entity = TherapySessionEntity(...)` construction at lines 377–386):

  ```python
                  from .entities import TherapySessionEntity
                  entity = TherapySessionEntity(
                      session_id=session_id,
                      user_id=user_id,
                      treatment_plan_id=session.treatment_plan_id,
                      session_number=session.session_number,
                      current_phase=SessionPhase.POST_SESSION,
                      started_at=session.started_at,
                      ended_at=datetime.now(timezone.utc),
                      topics_covered=list(getattr(session, "topics_covered", [])),
                      skills_practiced=list(getattr(session, "skills_practiced", [])),
                      insights_gained=list(getattr(session, "insights_gained", [])),
                      current_risk=session.current_risk,
                      safety_flags=list(getattr(session, "safety_flags", [])),
                      session_rating=getattr(session, "session_rating", None),
                      summary=summary or "",
                      # BLK-21: full transcript so the clinical record is complete.
                      transcript=list(getattr(session, "messages", [])),
                  )
  ```

- [ ] Run the transcript test — expect PASS (in-memory `UnitOfWork` round-trips the entity object, so `transcript` survives trivially).

- [ ] Add the `transcript` column to the Postgres repo `save` in `postgres_repository.py`. Extend the INSERT column list + values placeholders (currently `$1..$22`) with `transcript` as `$23`, add the `RETURNING`/UPDATE clause, and add the parameter. Concretely: add `transcript` to the column list after `next_session_focus,`, bump `version` to `$23` by inserting `$23` before it in the VALUES tuple (i.e. change the tuple to end `... $22, $23`), add `transcript = EXCLUDED.transcript,` to the `DO UPDATE SET` block, and add `json.dumps(session.transcript),` to the parameter list immediately before `session.version,`:

  ```python
              query = f"""
                  INSERT INTO {self._table} (
                      session_id, user_id, treatment_plan_id, session_number,
                      current_phase, started_at, ended_at,
                      mood_rating_start, mood_rating_end,
                      agenda_items, topics_covered, interventions,
                      homework_assigned, skills_practiced, insights_gained,
                      current_risk, safety_flags, session_rating, alliance_score,
                      summary, next_session_focus, transcript, version
                  ) VALUES (
                      $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                      $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23
                  )
                  ON CONFLICT (session_id) DO UPDATE SET
                      current_phase = EXCLUDED.current_phase,
                      ended_at = EXCLUDED.ended_at,
                      mood_rating_start = EXCLUDED.mood_rating_start,
                      mood_rating_end = EXCLUDED.mood_rating_end,
                      agenda_items = EXCLUDED.agenda_items,
                      topics_covered = EXCLUDED.topics_covered,
                      interventions = EXCLUDED.interventions,
                      homework_assigned = EXCLUDED.homework_assigned,
                      skills_practiced = EXCLUDED.skills_practiced,
                      insights_gained = EXCLUDED.insights_gained,
                      current_risk = EXCLUDED.current_risk,
                      safety_flags = EXCLUDED.safety_flags,
                      session_rating = EXCLUDED.session_rating,
                      alliance_score = EXCLUDED.alliance_score,
                      summary = EXCLUDED.summary,
                      next_session_focus = EXCLUDED.next_session_focus,
                      transcript = EXCLUDED.transcript,
                      version = EXCLUDED.version
                  RETURNING *
              """
  ```
  and in the `conn.fetchrow(query, ...)` argument list, insert before `session.version,`:
  ```python
                  session.next_session_focus,
                  json.dumps(session.transcript),
                  session.version,
  ```

- [ ] Deserialize it in `_row_to_entity` (`postgres_repository.py`, before the `return TherapySessionEntity(...)` on line 652):

  ```python
          transcript = row.get("transcript", [])
          if isinstance(transcript, str):
              transcript = json.loads(transcript)
  ```
  and add `transcript=transcript,` to the `TherapySessionEntity(...)` constructor (after `next_session_focus=...`).

- [ ] Create the migration `migrations/versions/009_therapy_session_transcript.py` (chain onto `008`; check `008_rls_personality_tables.py`'s `revision` value and set it as `down_revision`):

  ```python
  """009: add transcript column to therapy_sessions (BLK-21).

  Revision ID: 009_therapy_session_transcript
  Revises: 008_rls_personality_tables
  """
  from __future__ import annotations

  import sqlalchemy as sa
  from alembic import op
  from sqlalchemy.dialects.postgresql import JSONB

  revision = "009_therapy_session_transcript"
  down_revision = "008_rls_personality_tables"
  branch_labels = None
  depends_on = None


  def upgrade() -> None:
      op.add_column(
          "therapy_sessions",
          sa.Column("transcript", JSONB, nullable=False, server_default="[]"),
      )


  def downgrade() -> None:
      op.drop_column("therapy_sessions", "transcript")
  ```

  Confirm the exact `revision`/`down_revision` identifiers by opening `migrations/versions/008_rls_personality_tables.py` and matching its `revision` string.

- [ ] **Regression gate** — Therapy Service suite green:

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/therapy_service/tests -o "addopts=" -p no:logging -q
  ```

---

### Task 8: BLK-21 (part 2) — Persist treatment plans via `unit_of_work.treatment_plans` + cross-worker recovery

Treatment plans live only in `self._treatment_plans: dict` — never persisted — so a `start_session` on another worker (or after a restart) fails "plan not found." Persist the plan through `unit_of_work.treatment_plans` when it is created, and on a cache miss recover it from the repository before creating a new one.

**Files:**
- Modify `services/therapy_service/src/domain/service.py` — `start_session` (lines 185–199); add `_treatment_plan_entity` + `_entity_to_treatment_plan_dto` helpers.
- Modify `services/therapy_service/tests/test_end_session_persistence.py` — make the `MagicMock` uow async-compatible for the new plan calls.
- Extend `services/therapy_service/tests/test_blk21_transcript_plan.py` with plan-persistence + recovery tests.

**Interfaces:**
- `_treatment_plan_entity(user_id, plan_id, dto) -> TreatmentPlanEntity` — builds a persistable entity from the DTO the service already holds.
- `_entity_to_treatment_plan_dto(entity) -> TreatmentPlanDTO` — inverse, for cache-hydration on recovery.
- `start_session` behaviour: (1) in-memory cache → (2) `await uow.treatment_plans.get(plan_id)` recovery → (3) create + `await uow.treatment_plans.save(entity)`.

**Steps:**

- [ ] Extend `services/therapy_service/tests/test_blk21_transcript_plan.py` with the plan tests:

  ```python
  @pytest.mark.asyncio
  async def test_start_session_persists_treatment_plan() -> None:
      uow = _uow()
      orch = TherapyOrchestrator(
          technique_selector=TechniqueSelector(), session_manager=SessionManager(),
          unit_of_work=uow,
      )
      await orch.initialize()
      user_id, plan_id = uuid4(), uuid4()
      await orch.start_session(user_id, plan_id, {"diagnosis": "Anxiety"})

      persisted_plan = await uow.treatment_plans.get(plan_id)
      assert persisted_plan is not None
      assert persisted_plan.user_id == user_id
      await orch.shutdown()


  @pytest.mark.asyncio
  async def test_start_session_recovers_plan_from_repo_cross_worker() -> None:
      # Worker A persists the plan.
      uow = _uow()
      user_id, plan_id = uuid4(), uuid4()
      orch_a = TherapyOrchestrator(
          technique_selector=TechniqueSelector(), session_manager=SessionManager(),
          unit_of_work=uow,
      )
      await orch_a.initialize()
      await orch_a.start_session(user_id, plan_id, {"diagnosis": "Anxiety"})

      # Worker B shares the repo but has a cold in-memory cache.
      orch_b = TherapyOrchestrator(
          technique_selector=TechniqueSelector(), session_manager=SessionManager(),
          unit_of_work=uow,
      )
      await orch_b.initialize()
      assert plan_id not in orch_b._treatment_plans  # cold cache
      result = await orch_b.start_session(user_id, plan_id, {})
      assert result.session_id is not None
      assert plan_id in orch_b._treatment_plans      # recovered from the repo, not re-created
      await orch_a.shutdown()
      await orch_b.shutdown()
  ```

- [ ] Run these two — expect FAIL (plan never persisted; no repo recovery):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest "services/therapy_service/tests/test_blk21_transcript_plan.py::test_start_session_persists_treatment_plan" "services/therapy_service/tests/test_blk21_transcript_plan.py::test_start_session_recovers_plan_from_repo_cross_worker" -o "addopts=" -p no:logging -q
  ```

- [ ] Add the mapping helpers to `service.py` (next to `_treatment_plan_to_dto`, after line 600). Import `TreatmentPlanEntity` at the top of the method via the existing `from .entities import ...` pattern used in `end_session`:

  ```python
      def _treatment_plan_entity(self, user_id: UUID, plan_id: UUID, dto: TreatmentPlanDTO):
          """Build a persistable TreatmentPlanEntity from the DTO the service holds (BLK-21)."""
          from .entities import TreatmentPlanEntity
          from ..schemas import TreatmentPhase
          phase_by_int = {1: TreatmentPhase.FOUNDATION, 2: TreatmentPhase.ACTIVE_TREATMENT,
                          3: TreatmentPhase.CONSOLIDATION, 4: TreatmentPhase.MAINTENANCE}
          return TreatmentPlanEntity(
              plan_id=plan_id,
              user_id=user_id,
              primary_diagnosis=dto.primary_diagnosis,
              severity=dto.severity,
              primary_modality=dto.primary_modality,
              adjunct_modalities=list(dto.adjunct_modalities),
              current_phase=phase_by_int.get(dto.current_phase, TreatmentPhase.FOUNDATION),
              total_sessions_completed=dto.sessions_completed,
              skills_acquired=list(dto.skills_acquired),
          )

      def _entity_to_treatment_plan_dto(self, entity) -> TreatmentPlanDTO:
          """Hydrate the in-memory DTO cache from a recovered TreatmentPlanEntity (BLK-21)."""
          phase_int = {"foundation": 1, "active_treatment": 2, "consolidation": 3, "maintenance": 4}
          phase_value = entity.current_phase.value if hasattr(entity.current_phase, "value") else str(entity.current_phase)
          return TreatmentPlanDTO(
              plan_id=entity.plan_id,
              user_id=entity.user_id,
              primary_diagnosis=entity.primary_diagnosis,
              severity=entity.severity,
              primary_modality=entity.primary_modality,
              adjunct_modalities=list(entity.adjunct_modalities),
              current_phase=phase_int.get(phase_value, 1),
              sessions_completed=entity.total_sessions_completed,
              skills_acquired=list(entity.skills_acquired),
          )
  ```

- [ ] Rewrite the plan-resolution block in `start_session` (lines 185–199) to add repo recovery + persistence:

  ```python
          treatment_plan = self._treatment_plans.get(treatment_plan_id)
          if not treatment_plan and self._unit_of_work:
              # BLK-21: cross-worker recovery — the plan may have been created on
              # another worker / before a restart and only lives in Postgres now.
              try:
                  recovered = await self._unit_of_work.treatment_plans.get(treatment_plan_id)
              except Exception as e:
                  logger.warning("treatment_plan_recovery_failed", plan_id=str(treatment_plan_id), error=str(e))
                  recovered = None
              if recovered is not None:
                  treatment_plan = self._entity_to_treatment_plan_dto(recovered)
                  self._treatment_plans[treatment_plan_id] = treatment_plan
          if not treatment_plan:
              if self._treatment_planner:
                  plan = self._treatment_planner.create_plan(
                      user_id=user_id,
                      diagnosis=context.get("diagnosis", "Depression"),
                      severity=SeverityLevel(context.get("severity", SeverityLevel.MODERATE.value)) if isinstance(context.get("severity"), str) else context.get("severity", SeverityLevel.MODERATE),
                      modality=TherapyModality(context.get("modality", TherapyModality.CBT.value)) if isinstance(context.get("modality"), str) else context.get("modality", TherapyModality.CBT),
                      phq9_score=context.get("phq9_score"),
                      initial_goals=context.get("initial_goals"),
                  )
                  treatment_plan = self._treatment_plan_to_dto(plan, treatment_plan_id)
              else:
                  treatment_plan = self._create_mock_treatment_plan(user_id, treatment_plan_id, context)
              self._treatment_plans[treatment_plan_id] = treatment_plan
              # BLK-21: persist the newly-created plan so a later start_session on any
              # worker recovers it instead of failing "plan not found".
              if self._unit_of_work:
                  await self._unit_of_work.treatment_plans.save(
                      self._treatment_plan_entity(user_id, treatment_plan_id, treatment_plan)
                  )
  ```

- [ ] Update `services/therapy_service/tests/test_end_session_persistence.py` so its `MagicMock` unit-of-work supports the new awaited plan calls (the only existing test that passes a `uow` *and* calls `start_session`). After `uow = MagicMock()` / `uow.sessions.save = AsyncMock(...)`, add:

  ```python
      from unittest.mock import AsyncMock, MagicMock
      ...
      uow = MagicMock()
      uow.sessions.save = AsyncMock(side_effect=RuntimeError("DB pool exhausted"))
      uow.treatment_plans.save = AsyncMock()          # BLK-21: start_session persists the plan
      uow.treatment_plans.get = AsyncMock(return_value=None)  # cold cache, no cross-worker recovery
  ```

- [ ] Run the plan tests — expect PASS.

- [ ] **Regression gate** — Therapy Service suite green (confirms `test_end_session_persistence.py` and the ~30 `start_session` tests in `test_service.py`, which pass no `uow`, are unaffected because the plan-persistence branch is guarded by `if self._unit_of_work`):

  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/therapy_service/tests -o "addopts=" -p no:logging -q
  ```

---

## Phase Exit Verification

Run all four affected service suites; every one must be green (integration-marked tests excluded — they run in staging):

```powershell
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/memory_service/tests services/diagnosis_service/tests services/orchestrator_service/tests services/therapy_service/tests -o "addopts=" -p no:logging -m "not integration" -q
```

Maps to `docs/ENTERPRISE-RELEASE-PLAN.md` Phase 2 Exit:
- conversation turns stored + retrieved across restart / second worker → BLK-15 (Task 5) + BLK-16 (Task 6, staging proof);
- checkpointer Postgres-backed → BLK-16 (Task 6);
- therapy record complete → BLK-21 (Tasks 7–8);
- diagnosis degrades without Redis → BLK-18 (Task 4);
- a failed write never returns 201 → BLK-19 (Task 1); durable long-term context is read back → BLK-17 (Tasks 2–3).

**Infra-gated (staging) follow-ups, not code-doable in the dev harness:** the real-Postgres `AsyncPostgresSaver` restart/cross-replica proof (Task 6), the real-PG RLS-scoped `assemble_context` read (Task 2), and the therapy `transcript`/plan round-trip against the live `therapy_sessions` schema after migration `009` (Tasks 7–8). The therapy Postgres repo's column set already diverges from migration `001` (a pre-existing schema-truth gap); reconciling it is out of Phase 2 scope and tracked separately.
