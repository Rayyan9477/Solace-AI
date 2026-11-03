"""
REV-29 + A4 clinical-accuracy reconciliation (Milestone-2 Phase C.5).

Two documented clinical CLAIMS that had drifted from CODE:

(REV-29) DOC claim: "diagnoses / clinical memories NEVER decay."
    CODE-before: clinical content merely received a 0.5 decay MODIFIER and still
    decayed over time. This pins the corrected behavior: a memory whose
    retention_category is "clinical" or "diagnosis" (like "permanent") never
    decays -- zero decay applied, strength unchanged, KEEP action.

(A4) DOC claim: the SQL batch decay uses the SAME per-category rates as the
    DecayManager. CODE-before: apply_batch_decay applied a FLAT rate (~0.05/day)
    to ALL records, diverging ~2.5x from short_term (0.15). This pins that the
    SQL path builds a per-CATEGORY rate matching DecayManager (short_term 0.15,
    medium_term 0.05, long_term 0.02) keyed off accessed_at, and that
    permanent/clinical/diagnosis categories carry ZERO decay.

Under the old (pre-fix) code these tests FAIL:
  * clinical/diagnosis decayed (non-zero decay_applied),
  * the SQL params contained only the flat 0.05, never 0.15 / 0.02.
"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from services.memory_service.src.domain.decay_manager import (
    DecayAction,
    DecayManager,
    RetentionCategory,
)
from services.memory_service.src.infrastructure.postgres_repo import PostgresRepository


# --------------------------------------------------------------------------- #
# REV-29: clinical / diagnosis retention categories NEVER decay (DecayManager)
# --------------------------------------------------------------------------- #
class TestClinicalNeverDecaysInDecayManager:
    @pytest.mark.parametrize("category", ["clinical", "diagnosis"])
    def test_clinical_category_does_not_decay(self, category: str) -> None:
        dm = DecayManager()
        created = datetime.now(timezone.utc) - timedelta(days=365)
        accessed = datetime.now(timezone.utc) - timedelta(days=365)
        result = dm.apply_decay(
            uuid4(), Decimal("1.0"), category, created, accessed,
        )
        assert result.decay_applied == Decimal("0"), (
            f"{category} memory must NOT decay, got decay_applied={result.decay_applied}"
        )
        assert result.new_strength == Decimal("1.0"), (
            f"{category} memory strength must stay 1.0, got {result.new_strength}"
        )
        assert result.action == DecayAction.KEEP
        assert result.retention_category == RetentionCategory.PERMANENT

    def test_short_term_still_decays_at_its_per_category_rate(self) -> None:
        """The never-decay override must NOT disable normal decay for short_term."""
        dm = DecayManager()
        rate = dm.get_decay_rate(RetentionCategory.SHORT_TERM)
        assert rate == Decimal("0.15")
        created = datetime.now(timezone.utc) - timedelta(days=10)
        result = dm.apply_decay(uuid4(), Decimal("1.0"), "short_term", created, created)
        expected = Decimal(str(math.exp(-0.15 * 10)))
        assert result.decay_applied > Decimal("0"), "short_term must decay"
        assert abs(result.new_strength - expected) < Decimal("0.01"), (
            f"short_term should decay at 0.15/day, got {result.new_strength} expected {expected}"
        )


# --------------------------------------------------------------------------- #
# A4: SQL batch decay uses per-category rates matching DecayManager (not flat)
# --------------------------------------------------------------------------- #
class _CapturingSession:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    async def execute(self, stmt):  # noqa: ANN001
        self._sink.append(stmt)

        class _Result:
            rowcount = 0

        return _Result()


class _StubRepo:
    """Stand-in for PostgresRepository (no engine/DB); exposes captured stmts."""

    def __init__(self) -> None:
        self._stats = {"updates": 0}
        self.captured: list = []
        # apply_batch_decay may read a settings/decay-settings attribute; leave
        # unset so it falls back to defaults if the impl needs it.

    @asynccontextmanager
    async def session(self):
        yield _CapturingSession(self.captured)


class TestSqlBatchDecayIsPerCategory:
    @pytest.mark.asyncio
    async def test_sql_rates_match_decay_manager_not_flat(self) -> None:
        stub = _StubRepo()
        # decay_factor kept for backward-compat signature; the per-category rates
        # must come from the same source as DecayManager regardless of it.
        await PostgresRepository.apply_batch_decay(stub, uuid4(), Decimal("0.05"))

        assert stub.captured, "apply_batch_decay should execute an update statement"
        compiled = stub.captured[0].compile(dialect=postgresql.dialect())
        param_values = [
            float(v) if isinstance(v, (int, float, Decimal)) else v
            for v in compiled.params.values()
        ]

        dm = DecayManager()
        short = float(dm.get_decay_rate(RetentionCategory.SHORT_TERM))   # 0.15
        medium = float(dm.get_decay_rate(RetentionCategory.MEDIUM_TERM))  # 0.05
        long_ = float(dm.get_decay_rate(RetentionCategory.LONG_TERM))     # 0.02

        assert short in param_values, (
            "SQL decay must use per-category short_term rate 0.15 (matching "
            f"DecayManager), not a flat rate. params={param_values}"
        )
        assert long_ in param_values, (
            "SQL decay must use per-category long_term rate 0.02 (matching "
            f"DecayManager). params={param_values}"
        )
        assert medium in param_values, "medium_term rate 0.05 must be present"
        # Time base is DAYS off accessed_at, not hours.
        assert 86400.0 in param_values
        assert 3600.0 not in param_values

    @pytest.mark.asyncio
    async def test_sql_never_decay_categories_are_zeroed(self) -> None:
        """permanent / clinical / diagnosis carry ZERO decay in the SQL path."""
        stub = _StubRepo()
        await PostgresRepository.apply_batch_decay(stub, uuid4(), Decimal("0.05"))
        compiled = stub.captured[0].compile(dialect=postgresql.dialect())
        param_str_values = [v for v in compiled.params.values() if isinstance(v, str)]

        assert "clinical" in param_str_values, (
            "SQL path must reference the 'clinical' never-decay category. "
            f"string params={param_str_values}"
        )
        assert "diagnosis" in param_str_values, (
            "SQL path must reference the 'diagnosis' never-decay category. "
            f"string params={param_str_values}"
        )
