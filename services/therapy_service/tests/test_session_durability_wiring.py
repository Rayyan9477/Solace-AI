"""REV-16: therapy session durability wired end-to-end through the orchestrator.

The orchestrator now write-throughs session state to the Redis store with CAS-merge
on the load-bearing safety fields, and reads-through on a cache miss. These tests
drive the real ``TherapyOrchestrator`` with a fakeredis-backed store and prove:

  * ``start_session`` persists the session durably;
  * a crisis message durably records the escalation (``current_risk`` HIGH +
    ``safety_flags``) — the state a concurrent write must never clobber;
  * a FRESH worker (new SessionManager, same Redis server) recovers the live
    session — including the safety escalation — via read-through.

When no store is injected the orchestrator behaves exactly as before (covered by
the rest of the therapy suite, which stays green).
"""
from __future__ import annotations

from uuid import uuid4

import fakeredis.aioredis
import pytest

from services.therapy_service.src.domain.service import TherapyOrchestrator
from services.therapy_service.src.domain.session_manager import SessionManager
from services.therapy_service.src.domain.technique_selector import TechniqueSelector
from services.therapy_service.src.infrastructure.redis_session_repository import (
    RedisSessionRepository,
)
from services.therapy_service.src.schemas import RiskLevel


def _store(server: fakeredis.FakeServer) -> RedisSessionRepository:
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return RedisSessionRepository(client=client)


def _orchestrator(manager: SessionManager, store: RedisSessionRepository) -> TherapyOrchestrator:
    return TherapyOrchestrator(
        technique_selector=TechniqueSelector(),
        session_manager=manager,
        session_store=store,
    )


@pytest.mark.asyncio
async def test_start_session_persists_durably() -> None:
    server = fakeredis.FakeServer()
    store = _store(server)
    orch = _orchestrator(SessionManager(), store)
    await orch.initialize()

    start = await orch.start_session(uuid4(), uuid4(), {})

    stored = await store.get_session(start.session_id)
    assert stored is not None
    assert stored.version >= 1  # a durable CAS write happened


@pytest.mark.asyncio
async def test_crisis_escalation_is_durably_persisted() -> None:
    server = fakeredis.FakeServer()
    store = _store(server)
    orch = _orchestrator(SessionManager(), store)
    await orch.initialize()
    uid = uuid4()
    start = await orch.start_session(uid, uuid4(), {})

    await orch.process_message(start.session_id, uid, "I want to kill myself", [])

    stored = await store.get_session(start.session_id)
    assert stored is not None
    assert stored.current_risk == RiskLevel.HIGH  # escalation survived to the store
    assert any("kill myself" in flag for flag in stored.safety_flags)


@pytest.mark.asyncio
async def test_fresh_worker_recovers_live_session_via_read_through() -> None:
    server = fakeredis.FakeServer()

    # Worker A starts a session and escalates it.
    orch_a = _orchestrator(SessionManager(), _store(server))
    await orch_a.initialize()
    uid = uuid4()
    start = await orch_a.start_session(uid, uuid4(), {})
    await orch_a.process_message(start.session_id, uid, "I want to kill myself", [])

    # Worker B is a fresh process (empty in-memory cache) sharing the same Redis.
    mgr_b = SessionManager()
    orch_b = _orchestrator(mgr_b, _store(server))
    assert mgr_b.get_session(start.session_id) is None  # nothing cached yet

    recovered = await orch_b._load_session(start.session_id)

    assert recovered is not None
    assert recovered.current_risk == RiskLevel.HIGH  # durable safety state recovered
    # Read-through hydrated worker B's cache for subsequent in-request mutations.
    assert mgr_b.get_session(start.session_id) is not None
