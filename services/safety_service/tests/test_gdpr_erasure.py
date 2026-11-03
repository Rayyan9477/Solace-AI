"""REV-17 / GDPR right-to-erasure for the safety service.

Covers three layers:

1. Each Postgres PHI repo grew a ``delete_user_data(user_id) -> int`` that
   issues a single ``DELETE FROM <table> WHERE user_id = $1`` and returns the
   asyncpg row count. Exercised against a fake asyncpg pool (no live infra),
   asserting the SQL/params CONSTRUCTED and the parsed count.
2. The in-memory fakes' ``delete_user_data`` removes ALL of one user's rows and
   NONE of another user's.
3. ``build_user_data_erasure`` registers all four safety stores, and
   ``erase_user_data`` fans out through the shared orchestrator: full success
   reports complete + emits audit; any store raising fails loud with
   ErasureIncomplete AFTER still attempting the healthy stores.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from services.safety_service.src.domain.entities import (
    IncidentSeverity,
    SafetyAssessment,
    SafetyIncident,
    SafetyPlan,
    UserRiskProfile,
)
from services.safety_service.src.infrastructure.repository import (
    PostgresSafetyAssessmentRepository,
    PostgresSafetyIncidentRepository,
    PostgresSafetyPlanRepository,
    PostgresUserRiskProfileRepository,
)
from services.safety_service.src.domain.escalation import (
    EscalationRecord,
    InMemoryEscalationRepository,
)
from services.safety_service.src.infrastructure.gdpr import (
    STORE_ESCALATIONS,
    STORE_SAFETY_ASSESSMENTS,
    STORE_SAFETY_INCIDENTS,
    STORE_SAFETY_PLANS,
    STORE_USER_RISK_PROFILES,
    build_user_data_erasure,
    erase_user_data,
)
from services.safety_service.tests.fixtures import (
    InMemorySafetyAssessmentRepository,
    InMemorySafetyIncidentRepository,
    InMemorySafetyPlanRepository,
    InMemoryUserRiskProfileRepository,
)
from solace_infrastructure.gdpr import ErasureIncomplete


# ---------------------------------------------------------------------------
# Fake asyncpg pool for the Postgres delete_user_data unit tests
# ---------------------------------------------------------------------------
class _FakeConn:
    """Honors the asyncpg ``execute`` contract for the single DELETE shape."""

    def __init__(self, pool: "_FakePool") -> None:
        self._pool = pool

    async def __aenter__(self) -> "_FakeConn":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        self._pool.executed.append((sql, args))
        normalized = " ".join(sql.split()).lower()
        if normalized.startswith("delete from") and "where user_id = $1" in normalized:
            uid = args[0]
            before = len(self._pool.rows)
            self._pool.rows[:] = [r for r in self._pool.rows if r["user_id"] != uid]
            return f"DELETE {before - len(self._pool.rows)}"
        raise NotImplementedError(f"fake: unsupported SQL: {sql!r}")


class _FakePool:
    """Minimal fake of asyncpg.Pool exposing ``acquire()``."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def acquire(self) -> _FakeConn:
        return _FakeConn(self)


@pytest.fixture
def legacy_client_path():
    """Force _acquire() onto the legacy ``self._client.acquire()`` path.

    The pool-manager flag defaults to enabled; with it on, _acquire() would
    ignore the injected fake client and reach for ConnectionPoolManager.
    """
    from solace_infrastructure.feature_flags import FeatureFlags

    FeatureFlags.disable_flag("use_connection_pool_manager")
    try:
        yield
    finally:
        FeatureFlags.enable_flag("use_connection_pool_manager")


_POSTGRES_CASES = [
    (PostgresSafetyAssessmentRepository, "public.safety_assessments"),
    (PostgresSafetyIncidentRepository, "public.safety_incidents"),
    (PostgresUserRiskProfileRepository, "public.user_risk_profiles"),
    (PostgresSafetyPlanRepository, "public.safety_plans"),
]


@pytest.mark.parametrize("repo_cls,table", _POSTGRES_CASES)
@pytest.mark.asyncio
async def test_postgres_delete_user_data_sql_and_count(
    legacy_client_path, repo_cls, table
) -> None:
    """DELETE targets only user_id=$1 and returns the asyncpg row count."""
    user_a, user_b = uuid4(), uuid4()
    rows = [
        {"user_id": user_a},
        {"user_id": user_a},
        {"user_id": user_b},
    ]
    pool = _FakePool(rows)
    repo = repo_cls(client=pool, schema="public")

    deleted = await repo.delete_user_data(user_a)

    assert deleted == 2
    # Only user_b's row remains — no over-reach onto other users.
    assert pool.rows == [{"user_id": user_b}]
    sql, args = pool.executed[-1]
    assert " ".join(sql.split()) == f"DELETE FROM {table} WHERE user_id = $1"
    assert args == (user_a,)


@pytest.mark.asyncio
async def test_postgres_delete_user_data_accepts_str_user_id(legacy_client_path) -> None:
    """The orchestrator passes ``str(user_id)``; the repo must coerce to UUID
    so the DELETE binds the same value the rows were stored under."""
    user_a = uuid4()
    pool = _FakePool([{"user_id": user_a}])
    repo = PostgresSafetyAssessmentRepository(client=pool, schema="public")

    deleted = await repo.delete_user_data(str(user_a))

    assert deleted == 1
    _sql, args = pool.executed[-1]
    assert args == (user_a,)
    assert isinstance(args[0], UUID)


# ---------------------------------------------------------------------------
# In-memory fakes: delete ALL of the target user, NONE of another user
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inmemory_assessment_delete_scoped_to_user() -> None:
    repo = InMemorySafetyAssessmentRepository()
    ua, ub = uuid4(), uuid4()
    await repo.save(SafetyAssessment(user_id=ua, content_assessed="a1"))
    await repo.save(SafetyAssessment(user_id=ua, content_assessed="a2"))
    await repo.save(SafetyAssessment(user_id=ub, content_assessed="b1"))

    deleted = await repo.delete_user_data(ua)

    assert deleted == 2
    assert await repo.get_by_user(ua) == []
    assert len(await repo.get_by_user(ub)) == 1


@pytest.mark.asyncio
async def test_inmemory_incident_delete_scoped_to_user() -> None:
    repo = InMemorySafetyIncidentRepository()
    ua, ub = uuid4(), uuid4()
    for _ in range(2):
        await repo.save(SafetyIncident(
            user_id=ua, severity=IncidentSeverity.HIGH, crisis_level="HIGH", description="x"
        ))
    await repo.save(SafetyIncident(
        user_id=ub, severity=IncidentSeverity.HIGH, crisis_level="HIGH", description="y"
    ))

    deleted = await repo.delete_user_data(ua)

    assert deleted == 2
    assert await repo.get_by_user(ua) == []
    assert len(await repo.get_by_user(ub)) == 1


@pytest.mark.asyncio
async def test_inmemory_plan_delete_scoped_to_user() -> None:
    repo = InMemorySafetyPlanRepository()
    ua, ub = uuid4(), uuid4()
    await repo.save(SafetyPlan(user_id=ua))
    await repo.save(SafetyPlan(user_id=ua))
    await repo.save(SafetyPlan(user_id=ub))

    deleted = await repo.delete_user_data(ua)

    assert deleted == 2
    assert await repo.get_by_user(ua) == []
    assert len(await repo.get_by_user(ub)) == 1


@pytest.mark.asyncio
async def test_inmemory_profile_delete_scoped_to_user() -> None:
    repo = InMemoryUserRiskProfileRepository()
    ua, ub = uuid4(), uuid4()
    await repo.save(UserRiskProfile(user_id=ua))
    await repo.save(UserRiskProfile(user_id=ub))

    deleted = await repo.delete_user_data(ua)

    assert deleted == 1
    assert await repo.get_by_user(ua) is None
    assert await repo.get_by_user(ub) is not None


@pytest.mark.asyncio
async def test_inmemory_delete_absent_user_returns_zero() -> None:
    repo = InMemorySafetyAssessmentRepository()
    assert await repo.delete_user_data(uuid4()) == 0


# ---------------------------------------------------------------------------
# Factory registration + orchestrated erase entry point
# ---------------------------------------------------------------------------
def _fresh_repos() -> tuple[
    InMemorySafetyAssessmentRepository,
    InMemorySafetyIncidentRepository,
    InMemoryUserRiskProfileRepository,
    InMemorySafetyPlanRepository,
    InMemoryEscalationRepository,
]:
    return (
        InMemorySafetyAssessmentRepository(),
        InMemorySafetyIncidentRepository(),
        InMemoryUserRiskProfileRepository(),
        InMemorySafetyPlanRepository(),
        InMemoryEscalationRepository(),
    )


@pytest.mark.asyncio
async def test_inmemory_escalation_delete_scoped_to_user() -> None:
    repo = InMemoryEscalationRepository()
    ua, ub = uuid4(), uuid4()
    await repo.save_escalation(EscalationRecord(user_id=ua, crisis_level="HIGH", reason="c1"))
    await repo.save_escalation(EscalationRecord(user_id=ua, crisis_level="HIGH", reason="c2"))
    await repo.save_escalation(EscalationRecord(user_id=ub, crisis_level="HIGH", reason="c3"))

    deleted = await repo.delete_user_data(ua)

    assert deleted == 2
    assert await repo.get_active_escalations(ua) == []
    assert len(await repo.get_active_escalations(ub)) == 1


def test_build_user_data_erasure_registers_all_five_stores() -> None:
    a, i, p, pl, e = _fresh_repos()
    erasure = build_user_data_erasure(
        assessment_repo=a, incident_repo=i, profile_repo=p, plan_repo=pl,
        escalation_repo=e,
    )
    assert set(erasure.store_names) == {
        STORE_SAFETY_ASSESSMENTS,
        STORE_SAFETY_INCIDENTS,
        STORE_USER_RISK_PROFILES,
        STORE_SAFETY_PLANS,
        STORE_ESCALATIONS,
    }
    assert len(erasure.store_names) == 5


@pytest.mark.asyncio
async def test_erase_user_data_fans_out_across_all_stores() -> None:
    a, i, p, pl, e = _fresh_repos()
    ua, ub = uuid4(), uuid4()
    # user A rows across every store
    await a.save(SafetyAssessment(user_id=ua, content_assessed="x"))
    await a.save(SafetyAssessment(user_id=ua, content_assessed="y"))
    await i.save(SafetyIncident(
        user_id=ua, severity=IncidentSeverity.HIGH, crisis_level="HIGH", description="d"
    ))
    await p.save(UserRiskProfile(user_id=ua))
    await pl.save(SafetyPlan(user_id=ua))
    await e.save_escalation(EscalationRecord(user_id=ua, crisis_level="HIGH", reason="crisis"))
    # user B must survive
    await a.save(SafetyAssessment(user_id=ub, content_assessed="z"))
    await p.save(UserRiskProfile(user_id=ub))
    await e.save_escalation(EscalationRecord(user_id=ub, crisis_level="HIGH", reason="keep"))

    report = await erase_user_data(
        ua, assessment_repo=a, incident_repo=i, profile_repo=p, plan_repo=pl,
        escalation_repo=e,
    )

    assert report.complete is True
    assert report.total_deleted == 6
    assert report.deleted == {
        STORE_SAFETY_ASSESSMENTS: 2,
        STORE_SAFETY_INCIDENTS: 1,
        STORE_USER_RISK_PROFILES: 1,
        STORE_SAFETY_PLANS: 1,
        STORE_ESCALATIONS: 1,
    }
    # user B untouched
    assert len(await a.get_by_user(ub)) == 1
    assert await p.get_by_user(ub) is not None
    assert len(await e.get_active_escalations(ub)) == 1
    # user A gone everywhere
    assert await a.get_by_user(ua) == []
    assert await i.get_by_user(ua) == []
    assert await pl.get_by_user(ua) == []
    assert await p.get_by_user(ua) is None
    assert await e.get_active_escalations(ua) == []


@pytest.mark.asyncio
async def test_erase_user_data_emits_audit_only_on_full_success() -> None:
    a, i, p, pl, e = _fresh_repos()
    ua = uuid4()
    await a.save(SafetyAssessment(user_id=ua, content_assessed="x"))
    events: list[Any] = []

    report = await erase_user_data(
        ua, assessment_repo=a, incident_repo=i, profile_repo=p, plan_repo=pl,
        escalation_repo=e, audit_emit=lambda r: events.append(r),
    )

    assert report.complete is True
    assert report.audit_emitted is True
    assert events == [report]


@pytest.mark.asyncio
async def test_erase_user_data_fails_loud_but_still_erases_healthy_stores() -> None:
    a, i, _p, pl, e = _fresh_repos()
    ua = uuid4()
    await a.save(SafetyAssessment(user_id=ua, content_assessed="x"))
    await pl.save(SafetyPlan(user_id=ua))

    class _BoomProfileRepo(InMemoryUserRiskProfileRepository):
        async def delete_user_data(self, user_id: UUID) -> int:
            raise RuntimeError("profile store down")

    boom_profile = _BoomProfileRepo()
    audit_calls: list[Any] = []

    with pytest.raises(ErasureIncomplete) as excinfo:
        await erase_user_data(
            ua, assessment_repo=a, incident_repo=i, profile_repo=boom_profile,
            plan_repo=pl, escalation_repo=e, audit_emit=lambda r: audit_calls.append(r),
        )

    report = excinfo.value.report
    assert report.complete is False
    assert STORE_USER_RISK_PROFILES in report.failed
    # Fail-loud, NOT fail-fast: the healthy stores were still attempted...
    assert report.deleted.get(STORE_SAFETY_ASSESSMENTS) == 1
    assert report.deleted.get(STORE_SAFETY_PLANS) == 1
    # ...and their rows really are gone.
    assert await a.get_by_user(ua) == []
    assert await pl.get_by_user(ua) == []
    # No completion audit on a partial erasure.
    assert audit_calls == []
