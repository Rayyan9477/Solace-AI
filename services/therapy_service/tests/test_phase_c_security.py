"""Phase C security-remediation regression tests for the therapy service.

Covers three findings from ``docs/PHASE-C-SECURITY-FINDINGS.md`` (therapy cluster):

  * P1-2 - ``TherapyOrchestrator.erase_user_data`` must run the RLS-protected
    DELETEs under the TARGET user's GUC (``app.current_user_id``), not the
    authenticated caller's, so an admin erasing another user is not RLS-filtered
    to the admin's own (empty) row set and silently deletes nothing while
    attesting the erasure complete. The caller's request context is restored
    afterwards.
  * P1-5 - ``_load_session`` must reconcile a cache HIT against the durable store:
    when another worker has advanced the durable session to a newer version, the
    stale (possibly risk-downgraded) cached copy must be replaced with the durable
    one so the request never acts on stale clinical state.
  * P2-9 - on the CRISIS persist path a CAS-exhaustion (``SessionPersistConflict``)
    must NOT be swallowed - it must fail loud so the escalation is retried and
    never silently dropped while the turn reports success. The normal (non-crisis)
    persist stays best-effort.

RLS enforcement itself is a Postgres-only no-op under fakeredis/in-memory fakes,
so these unit-test the GUC-SETTING / code behavior; real-Postgres RLS filtering is
verified at staging (Stage 3).
"""
from __future__ import annotations

from uuid import uuid4

import fakeredis.aioredis
import pytest

from services.therapy_service.src.domain.models import SessionState
from services.therapy_service.src.domain.service import TherapyOrchestrator
from services.therapy_service.src.domain.session_manager import SessionManager
from services.therapy_service.src.infrastructure.gdpr import build_user_data_erasure
from services.therapy_service.src.infrastructure.redis_session_repository import (
    RedisSessionRepository,
    SessionPersistConflict,
)
from services.therapy_service.src.schemas import RiskLevel
from solace_common.request_context import (
    get_current_user_id,
    reset_current_user_id,
    set_current_user_id,
)


def _store(server: fakeredis.FakeServer) -> RedisSessionRepository:
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return RedisSessionRepository(client=client)


# ---------------------------------------------------------------------------
# P1-2: GDPR erase runs its RLS-protected deleters under the TARGET user's GUC.
# ---------------------------------------------------------------------------

class _GucCapturingRepo:
    """A store deleter that records the RLS GUC active while it deletes."""

    def __init__(self) -> None:
        self.guc_seen: str | None = None
        self.called = False

    async def delete_user_data(self, user_id) -> int:
        self.called = True
        self.guc_seen = get_current_user_id()
        return 1


@pytest.mark.asyncio
async def test_erase_user_data_scopes_guc_to_target_and_restores() -> None:
    repos = {
        "treatment_plan_repo": _GucCapturingRepo(),
        "session_repo": _GucCapturingRepo(),
        "outcome_repo": _GucCapturingRepo(),
        "redis_session_repo": _GucCapturingRepo(),
    }
    erasure = build_user_data_erasure(**repos)
    orch = TherapyOrchestrator(user_data_erasure=erasure)

    admin_id = str(uuid4())
    target_id = uuid4()
    # The request is authenticated as an ADMIN erasing another user, so the ambient
    # GUC is the admin's id -- NOT the user being erased.
    token = set_current_user_id(admin_id)
    try:
        report = await orch.erase_user_data(target_id)

        assert report.complete is True
        # Every RLS-protected deleter ran under the TARGET user's GUC, so the
        # DELETEs are scoped to the erased user's rows -- not RLS-filtered to the
        # admin's own.
        for name, repo in repos.items():
            assert repo.called, f"{name} deleter was not run"
            assert repo.guc_seen == str(target_id), (
                f"{name} deleter ran under GUC {repo.guc_seen!r}, expected target "
                f"{target_id!r}"
            )
        # The caller's request context is restored once the erase returns (before
        # the test's own outer reset below).
        assert get_current_user_id() == admin_id
    finally:
        reset_current_user_id(token)


@pytest.mark.asyncio
async def test_erase_user_data_restores_guc_even_on_failure() -> None:
    class _RaisingRepo:
        async def delete_user_data(self, user_id) -> int:
            raise RuntimeError("store offline")

    from solace_infrastructure.gdpr.erasure import ErasureIncomplete

    repos = {
        "treatment_plan_repo": _GucCapturingRepo(),
        "session_repo": _GucCapturingRepo(),
        "outcome_repo": _GucCapturingRepo(),
        "redis_session_repo": _RaisingRepo(),
    }
    erasure = build_user_data_erasure(**repos)
    orch = TherapyOrchestrator(user_data_erasure=erasure)

    caller_id = str(uuid4())
    target_id = uuid4()
    token = set_current_user_id(caller_id)
    try:
        with pytest.raises(ErasureIncomplete):
            await orch.erase_user_data(target_id)
        # Even when the erase fails partway, the caller's GUC is restored (finally).
        assert get_current_user_id() == caller_id
    finally:
        reset_current_user_id(token)


# ---------------------------------------------------------------------------
# P1-5: _load_session reconciles a cache HIT against a newer durable version.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_session_adopts_newer_durable_version_on_cache_hit() -> None:
    server = fakeredis.FakeServer()
    # This worker's store + its own in-process working cache.
    store_a = _store(server)
    mgr = SessionManager()
    orch = TherapyOrchestrator(session_manager=mgr, session_store=store_a)

    sid, uid = uuid4(), uuid4()
    # This worker cached the session at version 1 (e.g. a low-risk snapshot).
    cached = SessionState(
        session_id=sid, user_id=uid, current_risk=RiskLevel.LOW, version=1
    )
    mgr.put_session(cached)

    # A DIFFERENT worker (second store, same Redis server) has since advanced the
    # durable session to version 2 -- escalating risk and adding a safety flag.
    store_b = _store(server)
    newer = SessionState(
        session_id=sid,
        user_id=uid,
        current_risk=RiskLevel.CRITICAL,
        version=2,
        safety_flags=["advanced_by_other_worker"],
    )
    await store_b.save_session(newer)

    loaded = await orch._load_session(sid)

    assert loaded is not None
    assert loaded.version == 2  # adopted the durable copy, not the stale cache
    assert loaded.current_risk == RiskLevel.CRITICAL
    assert "advanced_by_other_worker" in loaded.safety_flags
    # The working cache was reconciled to the durable copy for in-request mutations.
    assert mgr.get_session(sid).version == 2


@pytest.mark.asyncio
async def test_load_session_keeps_cache_when_durable_not_newer() -> None:
    server = fakeredis.FakeServer()
    store_a = _store(server)
    mgr = SessionManager()
    orch = TherapyOrchestrator(session_manager=mgr, session_store=store_a)

    sid, uid = uuid4(), uuid4()
    # Cache holds the freshest state (version 3, escalated)...
    cached = SessionState(
        session_id=sid, user_id=uid, current_risk=RiskLevel.HIGH, version=3
    )
    mgr.put_session(cached)
    # ...a stale/older durable copy must NOT clobber it (would downgrade risk).
    store_b = _store(server)
    older = SessionState(
        session_id=sid, user_id=uid, current_risk=RiskLevel.NONE, version=2
    )
    await store_b.save_session(older)

    loaded = await orch._load_session(sid)

    assert loaded is cached  # same object retained, not replaced by stale durable
    assert loaded.current_risk == RiskLevel.HIGH
    assert loaded.version == 3


# ---------------------------------------------------------------------------
# P2-9: crisis persist fails loud on CAS exhaustion; normal persist stays best-effort.
# ---------------------------------------------------------------------------

class _ForcedConflictStore:
    """A session store whose CAS write-through never converges."""

    def __init__(self) -> None:
        self.attempts = 0

    async def get_session(self, session_id):
        return None

    async def save_session_cas_merge(self, session, max_retries: int = 5) -> None:
        self.attempts += 1
        raise SessionPersistConflict(session.session_id)

    async def delete_session(self, session_id) -> None:
        return None


def _session() -> SessionState:
    return SessionState(user_id=uuid4())


@pytest.mark.asyncio
async def test_persist_session_crisis_path_raises_on_cas_exhaustion() -> None:
    store = _ForcedConflictStore()
    orch = TherapyOrchestrator(session_manager=SessionManager(), session_store=store)

    with pytest.raises(SessionPersistConflict):
        await orch._persist_session(_session(), crisis=True)
    assert store.attempts == 1


@pytest.mark.asyncio
async def test_persist_session_normal_path_swallows_cas_exhaustion() -> None:
    store = _ForcedConflictStore()
    orch = TherapyOrchestrator(session_manager=SessionManager(), session_store=store)

    # Best-effort: the non-crisis persist logs but never raises.
    await orch._persist_session(_session(), crisis=False)
    assert store.attempts == 1


@pytest.mark.asyncio
async def test_crisis_message_fails_loud_when_escalation_cannot_persist() -> None:
    """End-to-end: a crisis turn whose durable escalation cannot converge must
    surface the failure (propagate) rather than report success while the crisis
    escalation is silently lost."""
    store = _ForcedConflictStore()
    orch = TherapyOrchestrator(
        session_manager=SessionManager(), session_store=store
    )
    await orch.initialize()
    uid = uuid4()
    # start_session persists best-effort (non-crisis) -> the forced conflict is
    # swallowed there, so the session still starts.
    start = await orch.start_session(uid, uuid4(), {})

    with pytest.raises(SessionPersistConflict):
        await orch.process_message(start.session_id, uid, "I want to kill myself", [])
