"""Durability tests for the Redis-backed therapy SessionRepository (REV-16).

REV-16: therapy ``SessionManager`` held live session state (phase, agenda,
safety_flags, engagement) in an in-memory ``dict`` that was lost on restart and
never shared across workers — the ``SessionRepository`` ABC existed but had no
implementation. These tests prove the new Redis implementation survives a
simulated worker/process boundary: state written through one repository
instance is readable through a *fresh* instance backed by the same Redis server
(fakeredis stands in for the real server, which is not available in CI).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import fakeredis.aioredis
import pytest

from services.therapy_service.src.domain.models import SessionState
from services.therapy_service.src.domain.session_manager import SessionRepository
from services.therapy_service.src.infrastructure.redis_session_repository import (
    RedisSessionRepository,
    max_risk,
    merge_safety_fields,
)
from services.therapy_service.src.schemas import (
    HomeworkDTO,
    RiskLevel,
    SessionPhase,
    TechniqueDTO,
    TherapyModality,
)


def _make_client(server: fakeredis.FakeServer) -> fakeredis.aioredis.FakeRedis:
    """A FakeRedis client bound to a shared server (simulates one worker)."""
    return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)


def _make_session() -> SessionState:
    """A mid-flight session carrying phase, safety flags and nested DTOs."""
    technique = TechniqueDTO(
        technique_id=uuid4(),
        name="Cognitive Restructuring",
        modality=TherapyModality.CBT,
        category="cognitive_restructuring",
        description="Challenge automatic thoughts",
        duration_minutes=15,
        requires_homework=True,
        contraindications=["acute_psychosis"],
    )
    homework = HomeworkDTO(
        homework_id=uuid4(),
        title="Thought record",
        description="Log three automatic thoughts",
        technique_id=technique.technique_id,
        due_date=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
    )
    return SessionState(
        session_id=uuid4(),
        user_id=uuid4(),
        treatment_plan_id=uuid4(),
        session_number=3,
        current_phase=SessionPhase.WORKING,
        started_at=datetime(2026, 7, 28, 9, 30, tzinfo=timezone.utc),
        mood_rating=4,
        agenda_items=["work stress", "sleep"],
        topics_covered=["cognitive distortions"],
        skills_practiced=["thought record"],
        insights_gained=["catastrophizing pattern"],
        techniques_used=[technique],
        homework_assigned=[homework],
        messages=[{"role": "user", "content": "I feel hopeless"}],
        phase_history=[{"from_phase": "opening", "to_phase": "working"}],
        current_risk=RiskLevel.HIGH,
        safety_flags=["suicidal_ideation", "self_harm"],
        engagement_score=0.72,
        session_rating=None,
    )


def test_repository_implements_port() -> None:
    """The Redis repository must satisfy the existing SessionRepository ABC."""
    assert issubclass(RedisSessionRepository, SessionRepository)


@pytest.mark.asyncio
async def test_session_survives_worker_boundary() -> None:
    """State written by worker A is fully recoverable by a fresh worker B."""
    server = fakeredis.FakeServer()
    repo_a = RedisSessionRepository(client=_make_client(server))
    session = _make_session()

    await repo_a.save_session(session)

    # Fresh repository instance on the same server == a different worker/process
    repo_b = RedisSessionRepository(client=_make_client(server))
    loaded = await repo_b.get_session(session.session_id)

    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.user_id == session.user_id
    assert loaded.treatment_plan_id == session.treatment_plan_id
    assert loaded.session_number == 3
    # Clinically load-bearing fields must survive the restart:
    assert loaded.current_phase == SessionPhase.WORKING
    assert loaded.current_risk == RiskLevel.HIGH
    assert loaded.safety_flags == ["suicidal_ideation", "self_harm"]
    assert loaded.mood_rating == 4
    assert loaded.agenda_items == ["work stress", "sleep"]
    assert loaded.engagement_score == pytest.approx(0.72)
    assert loaded.started_at == session.started_at
    assert loaded.messages == [{"role": "user", "content": "I feel hopeless"}]
    assert loaded.phase_history == [{"from_phase": "opening", "to_phase": "working"}]


@pytest.mark.asyncio
async def test_nested_dtos_round_trip() -> None:
    """TechniqueDTO / HomeworkDTO nested payloads survive serialization."""
    server = fakeredis.FakeServer()
    repo_a = RedisSessionRepository(client=_make_client(server))
    session = _make_session()

    await repo_a.save_session(session)

    repo_b = RedisSessionRepository(client=_make_client(server))
    loaded = await repo_b.get_session(session.session_id)

    assert loaded is not None
    assert len(loaded.techniques_used) == 1
    technique = loaded.techniques_used[0]
    assert isinstance(technique, TechniqueDTO)
    assert technique.technique_id == session.techniques_used[0].technique_id
    assert technique.modality == TherapyModality.CBT
    assert technique.contraindications == ["acute_psychosis"]

    assert len(loaded.homework_assigned) == 1
    homework = loaded.homework_assigned[0]
    assert isinstance(homework, HomeworkDTO)
    assert homework.homework_id == session.homework_assigned[0].homework_id
    assert homework.due_date == session.homework_assigned[0].due_date


@pytest.mark.asyncio
async def test_get_session_missing_returns_none() -> None:
    server = fakeredis.FakeServer()
    repo = RedisSessionRepository(client=_make_client(server))
    assert await repo.get_session(uuid4()) is None


@pytest.mark.asyncio
async def test_update_then_reload_reflects_latest_state() -> None:
    """Re-saving (write-through on a phase transition) updates the durable copy."""
    server = fakeredis.FakeServer()
    repo_a = RedisSessionRepository(client=_make_client(server))
    session = _make_session()
    await repo_a.save_session(session)

    session.current_phase = SessionPhase.CLOSING
    session.safety_flags.append("hopelessness")
    await repo_a.save_session(session)

    repo_b = RedisSessionRepository(client=_make_client(server))
    loaded = await repo_b.get_session(session.session_id)
    assert loaded is not None
    assert loaded.current_phase == SessionPhase.CLOSING
    assert "hopelessness" in loaded.safety_flags


@pytest.mark.asyncio
async def test_delete_session_removes_durable_copy() -> None:
    server = fakeredis.FakeServer()
    repo_a = RedisSessionRepository(client=_make_client(server))
    session = _make_session()
    await repo_a.save_session(session)

    repo_b = RedisSessionRepository(client=_make_client(server))
    await repo_b.delete_session(session.session_id)

    assert await repo_a.get_session(session.session_id) is None


@pytest.mark.asyncio
async def test_get_active_sessions_by_user_and_global() -> None:
    server = fakeredis.FakeServer()
    repo = RedisSessionRepository(client=_make_client(server))

    user_a = uuid4()
    s1 = _make_session()
    s1.user_id = user_a
    s2 = _make_session()
    s2.user_id = user_a
    s3 = _make_session()  # different user
    for s in (s1, s2, s3):
        await repo.save_session(s)

    fresh = RedisSessionRepository(client=_make_client(server))
    for_user = await fresh.get_active_sessions(user_a)
    assert {s.session_id for s in for_user} == {s1.session_id, s2.session_id}
    assert all(s.user_id == user_a for s in for_user)

    all_sessions = await fresh.get_active_sessions()
    assert {s.session_id for s in all_sessions} == {
        s1.session_id, s2.session_id, s3.session_id
    }


@pytest.mark.asyncio
async def test_get_session_count_is_monotonic_across_restart() -> None:
    """Session count mirrors cumulative session_number: it never drops on delete
    and survives a restart (fresh instance)."""
    server = fakeredis.FakeServer()
    repo = RedisSessionRepository(client=_make_client(server))
    user_a = uuid4()

    s1 = _make_session()
    s1.user_id = user_a
    s2 = _make_session()
    s2.user_id = user_a
    await repo.save_session(s1)
    await repo.save_session(s2)
    # Re-saving an existing session (a plain update) must not inflate the count.
    await repo.save_session(s1)

    fresh = RedisSessionRepository(client=_make_client(server))
    assert await fresh.get_session_count(user_a) == 2

    # Deleting an active session does not roll the cumulative counter back.
    await fresh.delete_session(s1.session_id)
    assert await fresh.get_session_count(user_a) == 2

    assert await fresh.get_session_count(uuid4()) == 0


# ---------------------------------------------------------------------------
# REV-16 optimistic-concurrency (CAS) + safety-field merge
# ---------------------------------------------------------------------------
def test_merge_safety_fields_unions_flags_and_keeps_higher_risk() -> None:
    """merge_safety_fields never loses a flag and never downgrades risk."""
    target = _make_session()
    target.safety_flags = ["suicidal_ideation"]
    target.current_risk = RiskLevel.LOW

    committed = _make_session()
    committed.safety_flags = ["self_harm", "suicidal_ideation"]  # overlap + new
    committed.current_risk = RiskLevel.HIGH

    merge_safety_fields(target, committed)

    assert set(target.safety_flags) == {"suicidal_ideation", "self_harm"}  # union, no dup
    assert target.current_risk == RiskLevel.HIGH  # escalated, never downgraded
    assert max_risk(RiskLevel.HIGH, RiskLevel.LOW) == RiskLevel.HIGH
    assert max_risk(RiskLevel.LOW, RiskLevel.CRITICAL) == RiskLevel.CRITICAL


@pytest.mark.asyncio
async def test_try_save_first_write_and_version_bump() -> None:
    server = fakeredis.FakeServer()
    repo = RedisSessionRepository(client=_make_client(server))
    session = _make_session()
    assert session.version == 0

    assert await repo.try_save_session(session) is True
    assert session.version == 1  # base 0 -> committed 1

    reloaded = await repo.get_session(session.session_id)
    assert reloaded.version == 1
    assert reloaded.safety_flags == session.safety_flags


@pytest.mark.asyncio
async def test_try_save_detects_stale_write_conflict() -> None:
    """A CAS-save against a version another worker already advanced is rejected."""
    server = fakeredis.FakeServer()
    repo_a = RedisSessionRepository(client=_make_client(server))
    repo_b = RedisSessionRepository(client=_make_client(server))

    session = _make_session()
    await repo_a.try_save_session(session)  # committed version 1

    a = await repo_a.get_session(session.session_id)  # both read v1
    b = await repo_b.get_session(session.session_id)

    assert await repo_b.try_save_session(b) is True   # B commits -> v2
    # A still holds v1; its CAS-save must be rejected (no silent clobber).
    assert await repo_a.try_save_session(a) is False
    assert a.version == 1  # rolled back, not advanced

    committed = await repo_a.get_session(session.session_id)
    assert committed.version == 2  # B's write stands


@pytest.mark.asyncio
async def test_cas_retry_merge_prevents_safety_flag_clobber() -> None:
    """The exact REV-16 clobber race: two workers concurrently add different
    safety_flags; the CAS + merge retry loop keeps BOTH (no lost escalation)."""
    server = fakeredis.FakeServer()
    repo_a = RedisSessionRepository(client=_make_client(server))
    repo_b = RedisSessionRepository(client=_make_client(server))

    session = _make_session()
    session.safety_flags = ["baseline"]
    session.current_risk = RiskLevel.LOW
    session.version = 0
    await repo_a.try_save_session(session)  # committed v1, flags=[baseline]
    sid = session.session_id

    # Both workers read the same v1 snapshot.
    a = await repo_a.get_session(sid)
    b = await repo_b.get_session(sid)

    # Worker B escalates first and commits (v1 -> v2).
    b.safety_flags.append("self_harm")
    assert await repo_b.try_save_session(b) is True

    # Worker A (stale v1) tries to add a DIFFERENT flag + escalate risk. The
    # encapsulated CAS-retry-merge (used by the orchestrator write-through)
    # converges without losing either worker's escalation.
    a.safety_flags.append("suicidal_ideation")
    a.current_risk = RiskLevel.HIGH
    await repo_a.save_session_cas_merge(a)

    final = await repo_a.get_session(sid)
    # NEITHER worker's safety flag was lost, and risk stayed escalated.
    assert set(final.safety_flags) == {"baseline", "self_harm", "suicidal_ideation"}
    assert final.current_risk == RiskLevel.HIGH
