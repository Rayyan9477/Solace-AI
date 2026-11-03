"""REV-17 GDPR right-to-erasure for the therapy service.

Proves the therapy service can erase ALL of one user's PHI across every store it
owns: the three Postgres tables (treatment_plans, therapy_sessions,
outcome_measures) and the Redis durable session store. Each repository grows a
``delete_user_data(user_id) -> int`` that deletes ALL of a user's rows (not just
one record) and nothing belonging to another user; a factory wires all four into
the shared :class:`UserDataErasure` orchestrator, and the orchestrator's
``erase_user_data`` domain method drives them under the fail-loud contract.

RED-first: these assertions fail until ``delete_user_data`` exists on each repo,
``build_user_data_erasure`` exists, and ``TherapyOrchestrator.erase_user_data``
is wired.
"""
from __future__ import annotations

from uuid import uuid4

import fakeredis.aioredis
import pytest

from services.therapy_service.src.domain.models import SessionState
from services.therapy_service.src.infrastructure import postgres_repository as pr
from services.therapy_service.src.infrastructure.postgres_repository import (
    PostgresOutcomeMeasureRepository,
    PostgresTherapySessionRepository,
    PostgresTreatmentPlanRepository,
)
from services.therapy_service.src.infrastructure.redis_session_repository import (
    RedisSessionRepository,
)
from services.therapy_service.src.infrastructure.gdpr import build_user_data_erasure
from services.therapy_service.src.domain.service import TherapyOrchestrator
from services.therapy_service.src import api as therapy_api
from services.therapy_service.src.main import create_application

from solace_infrastructure.gdpr.erasure import (
    ErasureIncomplete,
    ErasureReport,
    UserDataErasure,
)
from solace_security.middleware import get_current_user


# ---------------------------------------------------------------------------
# Fake asyncpg connection: a per-table list of rows filtered by user_id, so a
# real WHERE user_id = $1 DELETE genuinely removes ALL of one user's rows and
# leaves another user's rows intact.
# ---------------------------------------------------------------------------

class _FakeConn:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.executed: list[tuple] = []

    async def execute(self, query: str, *args):
        self.executed.append((query, args))
        if "DELETE FROM" in query and "user_id = $1" in query:
            uid = args[0]
            before = len(self.rows)
            self.rows[:] = [r for r in self.rows if r.get("user_id") != uid]
            return f"DELETE {before - len(self.rows)}"
        return "DELETE 0"


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *a) -> bool:
        return False


class _FakeClient:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


@pytest.fixture(autouse=True)
def _force_legacy_client(monkeypatch):
    # Force the legacy client path so the fake connection is used (mirrors the
    # diagnosis-service postgres persistence test).
    monkeypatch.setattr(pr, "FeatureFlags", None)


def _seed(user_a, user_b, n_a: int, n_b: int) -> _FakeConn:
    rows = [{"user_id": user_a} for _ in range(n_a)] + [
        {"user_id": user_b} for _ in range(n_b)
    ]
    return _FakeConn(rows)


# ---------------------------------------------------------------------------
# Postgres delete_user_data: delete ALL of a user's rows, none of another's.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_treatment_plan_delete_user_data_removes_all_for_user() -> None:
    user_a, user_b = uuid4(), uuid4()
    conn = _seed(user_a, user_b, 3, 2)
    repo = PostgresTreatmentPlanRepository(_FakeClient(conn))

    removed = await repo.delete_user_data(user_a)

    assert removed == 3
    assert len(conn.rows) == 2
    assert all(r["user_id"] == user_b for r in conn.rows)
    query = conn.executed[0][0]
    assert "DELETE FROM" in query and "treatment_plans" in query
    assert "WHERE user_id = $1" in query


@pytest.mark.asyncio
async def test_therapy_session_delete_user_data_removes_all_for_user() -> None:
    user_a, user_b = uuid4(), uuid4()
    conn = _seed(user_a, user_b, 4, 1)
    repo = PostgresTherapySessionRepository(_FakeClient(conn))

    removed = await repo.delete_user_data(user_a)

    assert removed == 4
    assert len(conn.rows) == 1
    assert all(r["user_id"] == user_b for r in conn.rows)
    query = conn.executed[0][0]
    assert "DELETE FROM" in query and "therapy_sessions" in query
    assert "WHERE user_id = $1" in query


@pytest.mark.asyncio
async def test_outcome_measure_delete_user_data_removes_all_for_user() -> None:
    user_a, user_b = uuid4(), uuid4()
    conn = _seed(user_a, user_b, 5, 3)
    repo = PostgresOutcomeMeasureRepository(_FakeClient(conn))

    removed = await repo.delete_user_data(user_a)

    assert removed == 5
    assert len(conn.rows) == 3
    assert all(r["user_id"] == user_b for r in conn.rows)
    query = conn.executed[0][0]
    assert "DELETE FROM" in query and "outcome_measures" in query
    assert "WHERE user_id = $1" in query


@pytest.mark.asyncio
async def test_treatment_plan_delete_user_data_returns_zero_when_none() -> None:
    user_a, user_b = uuid4(), uuid4()
    conn = _seed(user_a, user_b, 0, 2)
    repo = PostgresTreatmentPlanRepository(_FakeClient(conn))

    removed = await repo.delete_user_data(user_a)

    assert removed == 0
    assert len(conn.rows) == 2


# ---------------------------------------------------------------------------
# Redis per-user purge: delete every session in the user index + the index key
# + the per-user counter; leave another user's sessions intact.
# ---------------------------------------------------------------------------

def _redis_repo(server: fakeredis.FakeServer) -> RedisSessionRepository:
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return RedisSessionRepository(client=client)


@pytest.mark.asyncio
async def test_redis_delete_user_data_purges_all_user_sessions() -> None:
    server = fakeredis.FakeServer()
    repo = _redis_repo(server)

    user_a, user_b = uuid4(), uuid4()
    a1, a2, a3 = SessionState(user_id=user_a), SessionState(user_id=user_a), SessionState(user_id=user_a)
    b1, b2 = SessionState(user_id=user_b), SessionState(user_id=user_b)
    for s in (a1, a2, a3, b1, b2):
        await repo.save_session(s)

    removed = await repo.delete_user_data(user_a)

    assert removed == 3
    # A fresh instance (a different worker) sees user_a fully purged...
    fresh = _redis_repo(server)
    assert await fresh.get_active_sessions(user_a) == []
    assert await fresh.get_session_count(user_a) == 0
    assert await fresh.get_session(a1.session_id) is None
    # ...and user_b entirely intact.
    remaining = await fresh.get_active_sessions(user_b)
    assert {s.session_id for s in remaining} == {b1.session_id, b2.session_id}
    assert await fresh.get_session_count(user_b) == 2


# ---------------------------------------------------------------------------
# Factory + orchestrator: registry has all stores; erase() fans out + fail-loud.
# ---------------------------------------------------------------------------

class _FakeUserRepo:
    """A repo whose delete_user_data returns a fixed count and records the arg."""

    def __init__(self, count: int, *, raises: bool = False) -> None:
        self._count = count
        self._raises = raises
        self.called_with = None

    async def delete_user_data(self, user_id) -> int:
        self.called_with = user_id
        if self._raises:
            raise RuntimeError("store offline")
        return self._count


def _build(**overrides) -> tuple[UserDataErasure, dict[str, _FakeUserRepo]]:
    repos = {
        "treatment_plan_repo": _FakeUserRepo(overrides.get("plans", 2)),
        "session_repo": _FakeUserRepo(overrides.get("sessions", 3)),
        "outcome_repo": _FakeUserRepo(overrides.get("outcomes", 1)),
        "redis_session_repo": _FakeUserRepo(overrides.get("redis", 4),
                                            raises=overrides.get("redis_raises", False)),
    }
    erasure = build_user_data_erasure(**repos)
    return erasure, repos


def test_factory_registers_all_four_stores() -> None:
    erasure, _ = _build()
    assert set(erasure.store_names) == {
        "treatment_plans", "therapy_sessions", "outcome_measures", "therapy_redis",
    }


@pytest.mark.asyncio
async def test_factory_erase_fans_out_to_all_stores() -> None:
    erasure, repos = _build()
    user_id = uuid4()

    report = await erasure.erase(str(user_id))

    assert report.complete is True
    assert report.deleted == {
        "treatment_plans": 2,
        "therapy_sessions": 3,
        "outcome_measures": 1,
        "therapy_redis": 4,
    }
    assert report.total_deleted == 10
    # Every store actually received the user id (as a UUID).
    for repo in repos.values():
        assert repo.called_with == user_id


@pytest.mark.asyncio
async def test_factory_erase_is_fail_loud_and_attempts_all() -> None:
    erasure, repos = _build(redis_raises=True)
    user_id = uuid4()

    with pytest.raises(ErasureIncomplete) as exc:
        await erasure.erase(str(user_id))

    report = exc.value.report
    assert report.complete is False
    assert "therapy_redis" in report.failed
    # The three healthy stores were still attempted despite the redis failure.
    assert report.deleted == {
        "treatment_plans": 2, "therapy_sessions": 3, "outcome_measures": 1,
    }
    for repo in repos.values():
        assert repo.called_with == user_id


@pytest.mark.asyncio
async def test_orchestrator_erase_user_data_routes_through_erasure() -> None:
    erasure, repos = _build()
    orchestrator = TherapyOrchestrator(user_data_erasure=erasure)
    user_id = uuid4()

    report = await orchestrator.erase_user_data(user_id)

    assert report.complete is True
    assert report.total_deleted == 10
    for repo in repos.values():
        assert repo.called_with == user_id


@pytest.mark.asyncio
async def test_orchestrator_erase_user_data_requires_configuration() -> None:
    orchestrator = TherapyOrchestrator()
    with pytest.raises(RuntimeError):
        await orchestrator.erase_user_data(uuid4())


# ---------------------------------------------------------------------------
# DELETE /users/{user_id} endpoint: self-or-admin gated, 409 on partial erasure.
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402


class _FakeOrchestrator:
    def __init__(self, *, report=None, exc=None) -> None:
        self._report = report
        self._exc = exc
        self.called_with = None

    async def erase_user_data(self, user_id, *, audit_emit=None):
        self.called_with = user_id
        if self._exc is not None:
            raise self._exc
        return self._report


class _FakeUser:
    def __init__(self, user_id, roles=()) -> None:
        self.user_id = str(user_id)
        self._roles = list(roles)

    def has_any_role(self, roles) -> bool:
        return bool(set(self._roles) & set(roles))


def _client(orchestrator, user) -> TestClient:
    app = create_application()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[therapy_api.get_therapy_orchestrator] = lambda: orchestrator
    return TestClient(app, raise_server_exceptions=False)


def test_endpoint_owner_can_erase_own_data() -> None:
    user_id = uuid4()
    report = ErasureReport(
        user_id=str(user_id),
        deleted={"treatment_plans": 2, "therapy_sessions": 3,
                 "outcome_measures": 1, "therapy_redis": 4},
        complete=True,
    )
    orch = _FakeOrchestrator(report=report)
    client = _client(orch, _FakeUser(user_id))

    resp = client.delete(f"/api/v1/therapy/users/{user_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["complete"] is True
    assert body["total_deleted"] == 10
    assert orch.called_with == user_id


def test_endpoint_rejects_non_owner_non_admin() -> None:
    user_id = uuid4()
    orch = _FakeOrchestrator(report=ErasureReport(user_id=str(user_id), complete=True))
    client = _client(orch, _FakeUser(uuid4()))  # a different, non-admin user

    resp = client.delete(f"/api/v1/therapy/users/{user_id}")

    assert resp.status_code == 403
    # The erasure must NOT run for an unauthorized caller.
    assert orch.called_with is None


def test_endpoint_admin_can_erase_other_user() -> None:
    user_id = uuid4()
    report = ErasureReport(user_id=str(user_id), deleted={"therapy_redis": 1}, complete=True)
    orch = _FakeOrchestrator(report=report)
    client = _client(orch, _FakeUser(uuid4(), roles=["admin"]))

    resp = client.delete(f"/api/v1/therapy/users/{user_id}")

    assert resp.status_code == 200
    assert orch.called_with == user_id


def test_endpoint_reports_partial_erasure_as_conflict() -> None:
    user_id = uuid4()
    partial = ErasureReport(
        user_id=str(user_id),
        deleted={"treatment_plans": 2},
        failed={"therapy_redis": "store offline"},
        complete=False,
    )
    orch = _FakeOrchestrator(exc=ErasureIncomplete(partial))
    client = _client(orch, _FakeUser(user_id))

    resp = client.delete(f"/api/v1/therapy/users/{user_id}")

    # A partial erasure is never reported as success.
    assert resp.status_code == 409
