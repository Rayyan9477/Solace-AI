"""Phase C security gate — diagnosis cluster (P1-5 + P2-9).

P1-5: ``DiagnosisService._load_session`` trusted a stale in-memory cache on a HIT
and never reconciled the durable version, so on a non-sticky multi-worker LB a
worker could operate on — and re-persist over — downgraded clinical state that
another worker had already escalated. These tests prove that on a HIT the durable
copy is adopted when it is NEWER, and kept (cache preserved) when it is not.

P2-9: the post-``_update_session`` persist swallowed ``SessionPersistConflict``.
For a turn that carries a crisis safety escalation there is no guaranteed "next
turn" to re-persist, so swallowing CAS exhaustion could silently un-persist the
escalation. These tests prove a crisis-flagged turn whose persist hits CAS
exhaustion RAISES, while a non-crisis turn stays best-effort (swallowed).
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import fakeredis.aioredis
import pytest

from services.diagnosis_service.src.domain.models import SessionState
from services.diagnosis_service.src.domain.service import DiagnosisService
from services.diagnosis_service.src.infrastructure.redis_session_repository import (
    DiagnosisSessionStore,
    SessionPersistConflict,
)
from services.diagnosis_service.src.schemas import DiagnosisPhase


def _store(server: fakeredis.FakeServer) -> DiagnosisSessionStore:
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return DiagnosisSessionStore(client=client)


# --------------------------------------------------------------------------- #
# P1-5 — _load_session reconciles the durable version on a cache HIT           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_load_session_adopts_newer_durable_version_on_hit() -> None:
    """Two workers, one Redis server. Worker A caches v1 (no flags); worker B
    escalates the durable session to v2 (adds a crisis flag). Worker A's next
    _load_session HIT must adopt the newer durable copy, not the stale cache."""
    server = fakeredis.FakeServer()
    store_a, store_b = _store(server), _store(server)
    svc_a = DiagnosisService(session_store=store_a)

    uid = uuid4()
    session = SessionState(user_id=uid, phase=DiagnosisPhase.RAPPORT, safety_flags=[])
    sid = session.session_id
    await store_a.try_save_session(session)  # durable v1
    assert session.version == 1
    # Worker A holds the v1 object in its live cache.
    svc_a._active_sessions[sid] = session

    # Worker B advances the durable session: crisis escalation -> v2.
    b = await store_b.get_session(sid)
    b.safety_flags.append("suicidal_ideation")
    assert await store_b.try_save_session(b) is True
    assert b.version == 2

    # Worker A's cached copy is now stale (v1). A HIT must reconcile.
    loaded = await svc_a._load_session(sid)
    assert loaded is not None
    assert loaded.version == 2
    assert "suicidal_ideation" in loaded.safety_flags
    # The in-process cache was replaced with the durable copy.
    assert svc_a._active_sessions[sid] is loaded
    assert svc_a._active_sessions[sid].version == 2


@pytest.mark.asyncio
async def test_load_session_keeps_cache_when_durable_not_newer() -> None:
    """On a HIT where the durable version is NOT newer, the cached working copy is
    kept by identity (no needless clobber of in-flight mutations)."""
    server = fakeredis.FakeServer()
    store_a = _store(server)
    svc_a = DiagnosisService(session_store=store_a)

    session = SessionState(user_id=uuid4(), phase=DiagnosisPhase.RAPPORT)
    sid = session.session_id
    await store_a.try_save_session(session)  # durable v1
    svc_a._active_sessions[sid] = session  # cache holds the same v1 object

    loaded = await svc_a._load_session(sid)
    # Same version durable -> keep the cached object (identity preserved).
    assert loaded is session
    assert svc_a._active_sessions[sid] is session


# --------------------------------------------------------------------------- #
# P2-9 — crisis-turn persist fails loud on CAS exhaustion                      #
# --------------------------------------------------------------------------- #


class _AlwaysConflictStore(DiagnosisSessionStore):
    """A durable store under perpetual CAS contention: every optimistic save
    conflicts, so ``save_session_cas_merge`` exhausts its retry budget and raises
    ``SessionPersistConflict``. ``get_session`` still works (fakeredis-backed)."""

    async def try_save_session(self, session):  # noqa: ANN001, ANN201
        return False


def _crisis_extractor():
    """Symptom extractor that flags a crisis on this turn (no symptoms needed)."""

    class _Extractor:
        async def extract(self, message, history, existing):  # noqa: ANN001, ANN201
            return SimpleNamespace(
                symptoms=[],
                temporal_info={},
                contextual_factors=[],
                risk_indicators=["suicidal_ideation"],
            )

    return _Extractor()


def _benign_extractor():
    class _Extractor:
        async def extract(self, message, history, existing):  # noqa: ANN001, ANN201
            return SimpleNamespace(
                symptoms=[], temporal_info={}, contextual_factors=[], risk_indicators=[]
            )

    return _Extractor()


@pytest.mark.asyncio
async def test_crisis_turn_persist_raises_on_cas_exhaustion() -> None:
    """A turn whose safety_flags carry a crisis escalation must NOT silently
    succeed when the durable persist exhausts its CAS retry budget — it raises so
    the escalation is retried rather than dropped."""
    server = fakeredis.FakeServer()
    store = _AlwaysConflictStore(client=fakeredis.aioredis.FakeRedis(server=server, decode_responses=True))
    svc = DiagnosisService(session_store=store, symptom_extractor=_crisis_extractor())
    await svc.initialize()

    uid = uuid4()
    start = await svc.start_session(uid, "assessment", {}, None)  # persist swallowed (non-crisis)

    with pytest.raises(SessionPersistConflict):
        await svc.assess(
            user_id=uid,
            session_id=start.session_id,
            message="I want to end my life",
            conversation_history=[],
            existing_symptoms=[],
            current_phase=DiagnosisPhase.RAPPORT,
            current_differential=None,
            user_context={},
        )

    # The escalation was recorded on the in-memory working copy before the raise.
    cached = svc._active_sessions[start.session_id]
    assert "suicidal_ideation" in cached.safety_flags
    assert cached.phase == DiagnosisPhase.CRISIS


@pytest.mark.asyncio
async def test_non_crisis_turn_persist_stays_best_effort_on_cas_exhaustion() -> None:
    """A non-crisis turn keeps the existing best-effort behavior: CAS exhaustion is
    logged loud but swallowed (the next turn re-persists), so assess() still
    returns normally."""
    server = fakeredis.FakeServer()
    store = _AlwaysConflictStore(client=fakeredis.aioredis.FakeRedis(server=server, decode_responses=True))
    svc = DiagnosisService(session_store=store, symptom_extractor=_benign_extractor())
    await svc.initialize()

    uid = uuid4()
    start = await svc.start_session(uid, "assessment", {}, None)

    # Does not raise despite the store being under perpetual CAS contention.
    result = await svc.assess(
        user_id=uid,
        session_id=start.session_id,
        message="I had a rough day at work",
        conversation_history=[],
        existing_symptoms=[],
        current_phase=DiagnosisPhase.RAPPORT,
        current_differential=None,
        user_context={},
    )
    assert result is not None
    assert not any(f in ("suicidal_ideation", "self_harm", "harm_to_others") for f in result.safety_flags)
