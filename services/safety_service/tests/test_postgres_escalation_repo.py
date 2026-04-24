"""H-04 regression: PostgresEscalationRepository round-trip.

We cannot spin up a real Postgres container in this environment, so we
exercise the repository against a ``FakeAsyncpgPool`` that honours the
same ``acquire()`` / ``execute`` / ``fetchrow`` / ``fetch`` API asyncpg
exposes. The test proves the SQL parameterisation is correct and that
the ``_row_to_record`` mapping reverses the stored form — the two
places most likely to break when the schema drifts.

The alternative (Testcontainers with a real Postgres) is planned for
Sprint 7 integration. For MVP Sprint 2 this fake-backed unit test is
adequate: a) the interface contract is enforced, b) the JSONB
serialisation round-trip is validated, c) SQL text is exercised but
not executed against a real engine. Follow-up integration tests will
catch Postgres-specific edge cases.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from services.safety_service.src.domain.escalation import (
    EscalationPriority,
    EscalationRecord,
    EscalationStatus,
    NotificationType,
    PostgresEscalationRepository,
)


class FakeAsyncpgConnection:
    """In-memory stand-in for asyncpg.Connection.

    Stores rows in a ``dict`` keyed by ``escalation_id``. Supports the
    three SQL shapes PostgresEscalationRepository actually issues:
      - ``INSERT ... ON CONFLICT DO UPDATE`` (save_escalation)
      - ``SELECT * FROM escalations WHERE escalation_id = $1``
      - ``SELECT * FROM escalations WHERE status = ANY(...) ...``
      - ``UPDATE escalations SET status = $2 WHERE escalation_id = $1``
    """

    def __init__(self, rows: dict[Any, dict[str, Any]]) -> None:
        self._rows = rows

    async def __aenter__(self) -> FakeAsyncpgConnection:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        sql_lower = sql.lower().strip()
        if "insert into escalations" in sql_lower:
            # Match the column order in save_escalation
            row = {
                "escalation_id": args[0],
                "user_id": args[1],
                "session_id": args[2],
                "crisis_level": args[3],
                "priority": args[4],
                "status": args[5],
                "reason": args[6],
                "assigned_clinician_id": args[7],
                "notifications_sent": args[8],
                "actions_taken": args[9],
                "context": args[10],
                "created_at": args[11],
                "acknowledged_at": args[12],
                "resolved_at": args[13],
            }
            existing = self._rows.get(args[0])
            if existing:
                # ON CONFLICT DO UPDATE — update only the mutable fields
                existing.update(
                    {
                        "status": args[5],
                        "assigned_clinician_id": args[7],
                        "notifications_sent": args[8],
                        "actions_taken": args[9],
                        "acknowledged_at": args[12],
                        "resolved_at": args[13],
                    }
                )
            else:
                self._rows[args[0]] = row
            return "INSERT 1"
        if sql_lower.startswith("update escalations set status"):
            eid, status = args[0], args[1]
            if eid in self._rows:
                self._rows[eid]["status"] = status
            return "UPDATE 1"
        raise NotImplementedError(f"fake: unsupported SQL: {sql!r}")

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        if "where escalation_id =" in sql.lower():
            return self._rows.get(args[0])
        raise NotImplementedError(f"fake: unsupported fetchrow SQL: {sql!r}")

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        statuses = args[0]
        out = [r for r in self._rows.values() if r["status"] in statuses]
        if len(args) >= 2:
            user_id = args[1]
            out = [r for r in out if r["user_id"] == user_id]
        out.sort(key=lambda r: r["created_at"])
        return out


class FakeAsyncpgPool:
    """Minimal fake of ``asyncpg.Pool`` exposing just ``acquire()``."""

    def __init__(self) -> None:
        self._rows: dict[Any, dict[str, Any]] = {}

    def acquire(self) -> FakeAsyncpgConnection:
        return FakeAsyncpgConnection(self._rows)


@pytest.fixture
def pool() -> FakeAsyncpgPool:
    return FakeAsyncpgPool()


@pytest.fixture
def repo(pool: FakeAsyncpgPool) -> PostgresEscalationRepository:
    return PostgresEscalationRepository(pool)


def _make_record(**overrides: Any) -> EscalationRecord:
    base: dict[str, Any] = {
        "escalation_id": uuid4(),
        "user_id": uuid4(),
        "session_id": uuid4(),
        "crisis_level": "HIGH",
        "priority": EscalationPriority.HIGH,
        "status": EscalationStatus.PENDING,
        "reason": "Suicidal ideation expressed",
        "assigned_clinician_id": uuid4(),
        "notifications_sent": [NotificationType.EMAIL, NotificationType.SMS],
        "actions_taken": ["HIGH workflow initiated", "Clinician assigned"],
        "context": {"session_phase": "assessment", "turn": 7},
        "created_at": datetime.now(UTC),
        "acknowledged_at": None,
        "resolved_at": None,
    }
    base.update(overrides)
    return EscalationRecord(**base)


class TestPostgresEscalationRepository:
    """H-04: round-trip an escalation through the Postgres repo."""

    @pytest.mark.asyncio
    async def test_save_and_get_round_trips(
        self, repo: PostgresEscalationRepository
    ) -> None:
        original = _make_record()
        await repo.save_escalation(original)

        loaded = await repo.get_escalation(original.escalation_id)
        assert loaded is not None
        assert loaded.escalation_id == original.escalation_id
        assert loaded.user_id == original.user_id
        assert loaded.crisis_level == "HIGH"
        assert loaded.priority == EscalationPriority.HIGH
        assert loaded.status == EscalationStatus.PENDING
        assert loaded.reason == "Suicidal ideation expressed"
        assert loaded.assigned_clinician_id == original.assigned_clinician_id
        # Notifications / actions / context all round-trip through JSONB serialisation
        assert NotificationType.EMAIL in loaded.notifications_sent
        assert NotificationType.SMS in loaded.notifications_sent
        assert loaded.actions_taken == original.actions_taken
        assert loaded.context == original.context

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(
        self, repo: PostgresEscalationRepository
    ) -> None:
        result = await repo.get_escalation(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_save_is_upsert(
        self, repo: PostgresEscalationRepository
    ) -> None:
        """Re-saving the same escalation_id updates mutable fields."""
        record = _make_record(
            status=EscalationStatus.PENDING,
            actions_taken=["HIGH workflow initiated"],
        )
        await repo.save_escalation(record)

        record.status = EscalationStatus.IN_PROGRESS
        record.actions_taken = [
            "HIGH workflow initiated",
            "Notification sent",
        ]
        record.acknowledged_at = datetime.now(UTC)
        await repo.save_escalation(record)

        loaded = await repo.get_escalation(record.escalation_id)
        assert loaded is not None
        assert loaded.status == EscalationStatus.IN_PROGRESS
        assert "Notification sent" in loaded.actions_taken
        assert loaded.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_get_active_filters_out_resolved(
        self, repo: PostgresEscalationRepository
    ) -> None:
        active = _make_record(status=EscalationStatus.PENDING)
        in_progress = _make_record(status=EscalationStatus.IN_PROGRESS)
        resolved = _make_record(status=EscalationStatus.RESOLVED)
        closed = _make_record(status=EscalationStatus.CLOSED)

        for r in (active, in_progress, resolved, closed):
            await repo.save_escalation(r)

        results = await repo.get_active_escalations()
        ids = {r.escalation_id for r in results}
        assert active.escalation_id in ids
        assert in_progress.escalation_id in ids
        assert resolved.escalation_id not in ids
        assert closed.escalation_id not in ids

    @pytest.mark.asyncio
    async def test_get_active_scoped_by_user_id(
        self, repo: PostgresEscalationRepository
    ) -> None:
        user_a = uuid4()
        user_b = uuid4()
        r_a1 = _make_record(user_id=user_a, status=EscalationStatus.PENDING)
        r_a2 = _make_record(user_id=user_a, status=EscalationStatus.IN_PROGRESS)
        r_b = _make_record(user_id=user_b, status=EscalationStatus.PENDING)

        for r in (r_a1, r_a2, r_b):
            await repo.save_escalation(r)

        results = await repo.get_active_escalations(user_id=user_a)
        ids = {r.escalation_id for r in results}
        assert ids == {r_a1.escalation_id, r_a2.escalation_id}

    @pytest.mark.asyncio
    async def test_update_status_persists(
        self, repo: PostgresEscalationRepository
    ) -> None:
        record = _make_record(status=EscalationStatus.PENDING)
        await repo.save_escalation(record)
        await repo.update_status(record.escalation_id, EscalationStatus.RESOLVED)
        loaded = await repo.get_escalation(record.escalation_id)
        assert loaded is not None
        assert loaded.status == EscalationStatus.RESOLVED

    @pytest.mark.asyncio
    async def test_jsonb_serialisation_is_json_string(
        self, pool: FakeAsyncpgPool, repo: PostgresEscalationRepository
    ) -> None:
        """The repo serialises JSONB columns via ``json.dumps``; the stored
        column should therefore be a string that round-trips through
        ``json.loads``. This guards against accidentally passing native
        lists/dicts that asyncpg can't bind to a ``$N::jsonb`` cast."""
        record = _make_record(
            context={"session_phase": "closing", "turn_count": 12}
        )
        await repo.save_escalation(record)
        stored_row = pool._rows[record.escalation_id]
        for key in ("notifications_sent", "actions_taken", "context"):
            stored_val = stored_row[key]
            assert isinstance(stored_val, str), (
                f"{key} stored form must be a JSON string for asyncpg JSONB binding, "
                f"got {type(stored_val).__name__}"
            )
            # Parseable back to the original shape
            parsed = json.loads(stored_val)
            assert parsed is not None

    def test_inherits_escalation_repository_interface(
        self, repo: PostgresEscalationRepository
    ) -> None:
        """Safety net: swapping InMemoryEscalationRepository for
        PostgresEscalationRepository in EscalationManager must not
        require code changes at call sites."""
        from services.safety_service.src.domain.escalation import (
            EscalationRepository,
        )

        assert isinstance(repo, EscalationRepository)
        for method in ("save_escalation", "get_escalation",
                       "get_active_escalations", "update_status"):
            assert hasattr(repo, method), f"missing {method}"
