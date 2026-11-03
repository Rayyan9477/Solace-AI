"""REV-16: durable, concurrency-safe live-session store for the diagnosis service.

Diagnosis held live session state (phase, symptoms, differential, reasoning,
safety_flags) purely in-memory — lost on restart, never shared across workers, and
subject to the same whole-object clobber race as therapy. These tests prove the new
Redis store (fakeredis) round-trips the nested DTOs, rejects a stale CAS write, and —
via ``save_session_cas_merge`` — never loses a concurrently-added ``safety_flag``.
The last test drives the real ``DiagnosisService`` to prove start persists and a
fresh worker recovers the live session via read-through.
"""
from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import fakeredis.aioredis
import pytest

from services.diagnosis_service.src.domain.models import SessionState
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.infrastructure.redis_session_repository import (
    DiagnosisSessionStore,
    merge_safety_flags,
)
from services.diagnosis_service.src.schemas import (
    DiagnosisPhase,
    DifferentialDTO,
    HypothesisDTO,
    ReasoningStep,
    ReasoningStepResultDTO,
    SymptomDTO,
    SymptomType,
)


def _store(server: fakeredis.FakeServer) -> DiagnosisSessionStore:
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return DiagnosisSessionStore(client=client)


def _rich_session() -> SessionState:
    return SessionState(
        user_id=uuid4(),
        session_number=2,
        phase=DiagnosisPhase.RAPPORT,
        symptoms=[
            SymptomDTO(
                symptom_id=uuid4(), name="low mood", description="persistent sadness",
                symptom_type=SymptomType.EMOTIONAL,
            )
        ],
        differential=DifferentialDTO(
            primary=HypothesisDTO(hypothesis_id=uuid4(), name="MDD", confidence=Decimal("0.6"))
        ),
        messages=[{"role": "user", "content": "I feel hopeless"}],
        reasoning_history=[
            ReasoningStepResultDTO(step=ReasoningStep.ANALYZE, input_summary="i", output_summary="o")
        ],
        safety_flags=["suicidal_ideation"],
    )


@pytest.mark.asyncio
async def test_session_round_trips_nested_dtos_through_redis() -> None:
    server = fakeredis.FakeServer()
    store = _store(server)
    session = _rich_session()

    assert await store.try_save_session(session) is True
    assert session.version == 1

    fresh = _store(server)  # simulate a different worker
    loaded = await fresh.get_session(session.session_id)
    assert loaded is not None
    assert loaded.phase == DiagnosisPhase.RAPPORT
    assert loaded.symptoms[0].name == "low mood"
    assert loaded.differential.primary.name == "MDD"
    assert loaded.reasoning_history[0].step == ReasoningStep.ANALYZE
    assert loaded.safety_flags == ["suicidal_ideation"]
    assert loaded.version == 1


@pytest.mark.asyncio
async def test_try_save_detects_stale_write_conflict() -> None:
    server = fakeredis.FakeServer()
    repo_a, repo_b = _store(server), _store(server)
    session = _rich_session()
    await repo_a.try_save_session(session)  # v1

    a = await repo_a.get_session(session.session_id)
    b = await repo_b.get_session(session.session_id)
    assert await repo_b.try_save_session(b) is True    # B -> v2
    assert await repo_a.try_save_session(a) is False    # A stale -> rejected
    assert (await repo_a.get_session(session.session_id)).version == 2


@pytest.mark.asyncio
async def test_cas_merge_prevents_safety_flag_clobber() -> None:
    server = fakeredis.FakeServer()
    repo_a, repo_b = _store(server), _store(server)
    session = _rich_session()
    session.safety_flags = ["baseline"]
    session.version = 0
    await repo_a.try_save_session(session)  # v1, flags=[baseline]
    sid = session.session_id

    a = await repo_a.get_session(sid)
    b = await repo_b.get_session(sid)

    b.safety_flags.append("self_harm")
    assert await repo_b.try_save_session(b) is True  # v2

    a.safety_flags.append("hopelessness")
    await repo_a.save_session_cas_merge(a)  # converges, unions

    final = await repo_a.get_session(sid)
    assert set(final.safety_flags) == {"baseline", "self_harm", "hopelessness"}


def test_merge_safety_flags_unions() -> None:
    target = _rich_session()
    target.safety_flags = ["a"]
    committed = _rich_session()
    committed.safety_flags = ["a", "b"]
    merge_safety_flags(target, committed)
    assert set(target.safety_flags) == {"a", "b"}


@pytest.mark.asyncio
async def test_diagnosis_service_persists_and_fresh_worker_recovers() -> None:
    server = fakeredis.FakeServer()

    svc_a = DiagnosisService(session_store=_store(server))
    await svc_a.initialize()
    uid = uuid4()
    start = await svc_a.start_session(uid, "assessment", {}, None)

    # Persisted durably by start_session.
    persisted = await svc_a._session_store.get_session(start.session_id)
    assert persisted is not None and persisted.version >= 1

    # A fresh worker (empty cache, same Redis) recovers the live session.
    svc_b = DiagnosisService(session_store=_store(server))
    assert svc_b._active_sessions.get(start.session_id) is None
    recovered = await svc_b._load_session(start.session_id)
    assert recovered is not None
    assert recovered.user_id == uid
    assert svc_b._active_sessions.get(start.session_id) is not None  # cache hydrated


@pytest.mark.asyncio
async def test_gdpr_erasure_registers_and_purges_live_session_store() -> None:
    """Phase C gate P0: the live-session Redis store (special-category PHI) MUST be
    a registered GDPR erasure store, and erasure must actually purge it."""
    from services.diagnosis_service.src.domain.erasure import (
        DIAGNOSIS_SESSION_STORE,
        build_user_data_erasure,
    )

    server = fakeredis.FakeServer()
    store = _store(server)
    session = _rich_session()
    uid = session.user_id
    await store.try_save_session(session)
    assert await store.get_session(session.session_id) is not None

    class _FakePgRepo:
        async def delete_user_data(self, user_id):  # noqa: ANN001
            return 0

    erasure = build_user_data_erasure(_FakePgRepo(), session_store=store)
    assert DIAGNOSIS_SESSION_STORE in erasure.store_names

    report = await erasure.erase(str(uid))
    assert report.complete is True
    # The user's live-session PHI is gone from Redis (not a false "complete").
    assert await store.get_session(session.session_id) is None
    assert await store.get_session_count(uid) == 0
