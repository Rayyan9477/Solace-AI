"""
REV-18 regression tests - Memory decay time-unit consistency.

The per-category decay rates (long_term 0.02, medium_term 0.05, short_term 0.15)
are defined PER DAY (see DecaySettings, decay_interval_hours=24). Every decay path
must therefore measure elapsed time in DAYS. Previously the SQL batch path, the
in-memory service cache path, and RetentionPolicy.calculate_decay applied those
per-day rates per HOUR, decaying 24x too fast and erasing long-term memories in
days. These tests pin the half-life in days and would FAIL under the old (hours)
inconsistency.
"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from services.memory_service.src.domain.decay_manager import DecayManager
from services.memory_service.src.domain.models import MemoryRecord, MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.domain.value_objects import RetentionPolicy
from services.memory_service.src.infrastructure.postgres_repo import PostgresRepository


class TestHalfLifePinnedInDays:
    """RetentionPolicy.calculate_decay measures elapsed time in DAYS."""

    def test_medium_term_half_life_is_days(self) -> None:
        """Medium-term rate 0.05/day => half-life ln(2)/0.05 ~= 13.86 DAYS -> R ~= 0.5."""
        policy = RetentionPolicy.medium_term()
        half_life_days = math.log(2) / 0.05  # ~13.863
        result = policy.calculate_decay(Decimal("1.0"), days_elapsed=half_life_days)
        assert abs(result - Decimal("0.5")) < Decimal("0.01"), (
            f"After one half-life in days, retention should be ~0.5, got {result}"
        )

    def test_long_term_not_erased_after_30_days(self) -> None:
        """Long-term rate 0.02/day (half-life ~34.7d): after 30 days it must stay
        above the delete threshold, not be erased."""
        policy = RetentionPolicy.long_term()
        result = policy.calculate_decay(Decimal("1.0"), days_elapsed=30)
        expected = Decimal(str(math.exp(-0.02 * 30)))  # e^-0.6 ~= 0.5488
        assert abs(result - expected) < Decimal("0.01")
        assert result > policy.delete_threshold, (
            f"30-day-old long-term memory must survive, got {result}"
        )
        assert policy.get_action(result) == "retain"


class TestServiceCacheDecayInDays:
    """The in-memory service decay path measures elapsed time in DAYS."""

    @pytest.mark.asyncio
    async def test_long_term_memory_survives_30_days_in_cache(self) -> None:
        """A 30-day-old long_term memory must NOT be decayed to the floor / archived.
        Under the old per-hour code, 30 days = 720 'hours' -> e^(-0.02*720) ~= 0,
        clamped to 0.1 and archived. Under the corrected per-day code it stays ~0.55."""
        service = MemoryService(settings=MemoryServiceSettings(enable_decay=True))
        user_id = uuid4()
        record = MemoryRecord(
            user_id=user_id,
            tier="tier_4_episodic",
            content="a durable long-term memory",
            retention_category="long_term",
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
            retention_strength=Decimal("1.0"),
        )
        service._tier_4_episodic[user_id] = [record]

        decayed, archived = await service._apply_decay_model(user_id)

        assert decayed == 1
        assert archived == 0, "30-day-old long-term memory must not be archived"
        assert record.retention_strength > Decimal("0.3"), (
            f"long-term strength should stay ~0.55 after 30 days, got {record.retention_strength}"
        )
        expected = Decimal(str(math.exp(-0.02 * 30)))
        assert abs(record.retention_strength - expected) < Decimal("0.02")


class _CapturingSession:
    """Fake AsyncSession that records executed statements (no DB)."""

    def __init__(self, sink: list) -> None:
        self._sink = sink

    async def execute(self, stmt):  # noqa: ANN001
        self._sink.append(stmt)

        class _Result:
            rowcount = 0

        return _Result()


class _StubRepo:
    """Minimal stand-in for PostgresRepository.apply_batch_decay (no engine/DB)."""

    def __init__(self) -> None:
        self._stats = {"updates": 0}
        self.captured: list = []

    @asynccontextmanager
    async def session(self):
        yield _CapturingSession(self.captured)


class TestSqlBatchDecayInDays:
    """The SQL batch decay expression converts seconds to DAYS (/86400), not hours."""

    @pytest.mark.asyncio
    async def test_sql_expression_divides_seconds_by_days(self) -> None:
        stub = _StubRepo()
        await PostgresRepository.apply_batch_decay(stub, uuid4(), Decimal("0.05"))

        assert stub.captured, "apply_batch_decay should execute an update statement"
        compiled = stub.captured[0].compile(dialect=postgresql.dialect())
        param_values = list(compiled.params.values())

        assert 86400.0 in param_values, (
            "SQL decay must divide epoch seconds by 86400 (per-day), "
            f"params were {param_values}"
        )
        assert 3600.0 not in param_values, (
            "SQL decay must NOT divide by 3600 (per-hour), which decays 24x too fast"
        )


class TestCrossPathConsistency:
    """All per-day decay paths must agree for the same elapsed days + rate."""

    def test_all_paths_agree_in_days(self) -> None:
        rate = 0.05
        days = 10.0

        dm = DecayManager()
        dm_ret = dm.calculate_retention(days, Decimal("1.0"), Decimal("0.05"))
        vo_ret = RetentionPolicy.medium_term().calculate_decay(Decimal("1.0"), days_elapsed=days)
        # Mirror of the SQL expression: strength * exp(-rate * seconds/86400)
        sql_ret = math.exp(-rate * days)

        assert abs(float(dm_ret) - sql_ret) < 0.001
        assert abs(float(vo_ret) - sql_ret) < 0.001
