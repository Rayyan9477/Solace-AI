"""
Unit tests for Memory Service - Ebbinghaus Decay Formula.
Verifies decay formula uses stability * e^(-lambda*t), permanent skipping,
long-term decay behavior, stability reinforcement, and non-compounding decay.
"""
from __future__ import annotations

import math
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4

from services.memory_service.src.domain.decay_manager import (
    DecayManager,
    DecaySettings,
    DecayResult,
    DecayAction,
    RetentionCategory,
    StabilityRecord,
)


@pytest.fixture
def default_manager() -> DecayManager:
    """Create decay manager with default settings."""
    return DecayManager()


@pytest.fixture
def custom_manager() -> DecayManager:
    """Create decay manager with explicit settings for deterministic tests."""
    settings = DecaySettings(
        short_term_rate=Decimal("0.15"),
        medium_term_rate=Decimal("0.05"),
        long_term_rate=Decimal("0.02"),
        reinforcement_multiplier=Decimal("1.5"),
        max_stability=Decimal("10.0"),
        archive_threshold=Decimal("0.3"),
        delete_threshold=Decimal("0.1"),
    )
    return DecayManager(settings=settings)


class TestEbbinghausDecayFormula:
    """Verify decay formula uses stability * e^(-lambda*t), NOT retention_strength * e^(-lambda*t)."""

    def test_calculate_retention_basic(self, default_manager: DecayManager) -> None:
        """R(t) = e^(-lambda*t) * S with stability=1.0 should equal e^(-lambda*t)."""
        stability = Decimal("1.0")
        decay_rate = Decimal("0.05")
        time_elapsed_days = 10.0

        result = default_manager.calculate_retention(time_elapsed_days, stability, decay_rate)
        expected = Decimal(str(math.exp(-0.05 * 10.0))) * Decimal("1.0")

        assert abs(result - expected) < Decimal("0.0001"), (
            f"Expected retention ~{expected}, got {result}"
        )

    def test_calculate_retention_with_stability(self, default_manager: DecayManager) -> None:
        """Higher stability should yield higher retention for the same elapsed time."""
        decay_rate = Decimal("0.05")
        time_elapsed_days = 10.0

        low_stability = Decimal("1.0")
        high_stability = Decimal("2.0")

        result_low = default_manager.calculate_retention(time_elapsed_days, low_stability, decay_rate)
        result_high = default_manager.calculate_retention(time_elapsed_days, high_stability, decay_rate)

        assert result_high > result_low, (
            f"Higher stability ({high_stability}) should yield higher retention: "
            f"{result_high} vs {result_low}"
        )

    def test_calculate_retention_formula_is_exp_times_stability(self, default_manager: DecayManager) -> None:
        """Verify the formula is exactly e^(-lambda*t) * S, capped at 1.0."""
        stability = Decimal("0.8")
        decay_rate = Decimal("0.1")
        time_elapsed_days = 5.0

        result = default_manager.calculate_retention(time_elapsed_days, stability, decay_rate)
        raw = Decimal(str(math.exp(-0.1 * 5.0))) * stability
        expected = min(Decimal("1.0"), raw)

        assert abs(result - expected) < Decimal("0.0001"), (
            f"Formula should be min(1.0, e^(-lambda*t)*S). Expected {expected}, got {result}"
        )

    def test_medium_term_decay_10_days(self, custom_manager: DecayManager) -> None:
        """Medium-term (lambda=0.05): after 10 days, R = e^(-0.05*10) * 1.0 ~ 0.6065."""
        stability = Decimal("1.0")
        decay_rate = Decimal("0.05")
        time_elapsed_days = 10.0

        result = custom_manager.calculate_retention(time_elapsed_days, stability, decay_rate)
        expected = Decimal(str(math.exp(-0.5)))  # e^(-0.5) ~ 0.6065

        assert abs(result - expected) < Decimal("0.01"), (
            f"Medium-term 10-day retention should be ~{expected:.4f}, got {result}"
        )

    def test_zero_stability_returns_zero(self, default_manager: DecayManager) -> None:
        """Zero stability should return zero retention."""
        result = default_manager.calculate_retention(5.0, Decimal("0"), Decimal("0.05"))
        assert result == Decimal("0")

    def test_zero_time_elapsed_returns_stability(self, default_manager: DecayManager) -> None:
        """Zero time elapsed: e^(0) = 1.0, so R = 1.0 * S = S (capped at 1.0)."""
        stability = Decimal("0.8")
        result = default_manager.calculate_retention(0.0, stability, Decimal("0.1"))
        expected = min(Decimal("1.0"), stability)
        assert abs(result - expected) < Decimal("0.0001")

    def test_retention_capped_at_one(self, default_manager: DecayManager) -> None:
        """Even with high stability and zero time, retention should not exceed 1.0."""
        stability = Decimal("3.0")
        result = default_manager.calculate_retention(0.0, stability, Decimal("0.05"))
        assert result <= Decimal("1.0"), f"Retention must be capped at 1.0, got {result}"


class TestPermanentNeverDecays:
    """Permanent records should never decay regardless of time elapsed."""

    def test_permanent_category_returns_unchanged_strength(self, default_manager: DecayManager) -> None:
        """Apply decay to permanent item; strength should remain unchanged."""
        item_id = uuid4()
        original_strength = Decimal("0.7")
        created_at = datetime.now(timezone.utc) - timedelta(days=365)

        result = default_manager.apply_decay(
            item_id=item_id,
            current_strength=original_strength,
            retention_category="permanent",
            created_at=created_at,
        )

        assert result.new_strength == original_strength
        assert result.decay_applied == Decimal("0")
        assert result.action == DecayAction.KEEP
        assert result.retention_category == RetentionCategory.PERMANENT

    def test_permanent_via_mark_returns_unchanged(self, default_manager: DecayManager) -> None:
        """Items marked permanent via mark_permanent() should also skip decay."""
        item_id = uuid4()
        default_manager.mark_permanent(item_id)
        original_strength = Decimal("0.5")
        created_at = datetime.now(timezone.utc) - timedelta(days=100)

        result = default_manager.apply_decay(
            item_id=item_id,
            current_strength=original_strength,
            retention_category="medium_term",
            created_at=created_at,
        )

        assert result.new_strength == original_strength
        assert result.decay_applied == Decimal("0")
        assert result.retention_category == RetentionCategory.PERMANENT

    def test_permanent_is_permanent_check(self, default_manager: DecayManager) -> None:
        """is_permanent should return True for marked items."""
        item_id = uuid4()
        assert default_manager.is_permanent(item_id) is False
        default_manager.mark_permanent(item_id)
        assert default_manager.is_permanent(item_id) is True

    def test_unmark_permanent_allows_decay(self, default_manager: DecayManager) -> None:
        """After unmark_permanent, item should decay normally."""
        item_id = uuid4()
        default_manager.mark_permanent(item_id)
        assert default_manager.is_permanent(item_id) is True
        removed = default_manager.unmark_permanent(item_id)
        assert removed is True
        assert default_manager.is_permanent(item_id) is False


class TestLongTermDecaysSlowly:
    """Long-term (lambda=0.02) should decay, but slowly. Only 'permanent' is skipped."""

    def test_long_term_does_decay(self, custom_manager: DecayManager) -> None:
        """Long-term items ARE subject to decay (bug fix: only permanent skips decay)."""
        item_id = uuid4()
        original_strength = Decimal("1.0")
        created_at = datetime.now(timezone.utc) - timedelta(days=30)

        result = custom_manager.apply_decay(
            item_id=item_id,
            current_strength=original_strength,
            retention_category="long_term",
            created_at=created_at,
        )

        assert result.decay_applied > Decimal("0"), (
            "Long-term items must decay (only 'permanent' is exempt)"
        )
        assert result.retention_category == RetentionCategory.LONG_TERM

    def test_long_term_decays_slower_than_short_term(self, custom_manager: DecayManager) -> None:
        """Long-term decay rate (0.02) should produce less decay than short-term (0.15)."""
        created_at = datetime.now(timezone.utc) - timedelta(days=20)
        original_strength = Decimal("1.0")

        long_term_result = custom_manager.apply_decay(
            item_id=uuid4(),
            current_strength=original_strength,
            retention_category="long_term",
            created_at=created_at,
        )
        short_term_result = custom_manager.apply_decay(
            item_id=uuid4(),
            current_strength=original_strength,
            retention_category="short_term",
            created_at=created_at,
        )

        assert long_term_result.new_strength > short_term_result.new_strength, (
            f"Long-term ({long_term_result.new_strength}) should retain more than "
            f"short-term ({short_term_result.new_strength})"
        )

    def test_long_term_decay_rate_is_0_02(self, custom_manager: DecayManager) -> None:
        """Verify long_term rate from settings is 0.02."""
        rate = custom_manager.get_decay_rate(RetentionCategory.LONG_TERM)
        assert rate == Decimal("0.02")

    def test_permanent_decay_rate_is_zero(self, custom_manager: DecayManager) -> None:
        """Permanent decay rate should be exactly zero."""
        rate = custom_manager.get_decay_rate(RetentionCategory.PERMANENT)
        assert rate == Decimal("0")


class TestStabilityIncreasesWithAccess:
    """Stability should increase via reinforcement_multiplier on each access."""

    def test_initial_stability_is_1(self, default_manager: DecayManager) -> None:
        """Default stability for a new item should be 1.0."""
        item_id = uuid4()
        stability = default_manager.get_stability(item_id)
        assert stability == Decimal("1.0")

    def test_reinforce_once_multiplies_by_1_5(self, custom_manager: DecayManager) -> None:
        """After one reinforcement, stability = 1.0 * 1.5 = 1.5."""
        item_id = uuid4()
        new_stability = custom_manager.reinforce(item_id)
        assert new_stability == Decimal("1.5"), f"Expected 1.5, got {new_stability}"

    def test_reinforce_twice_multiplies_by_1_5_squared(self, custom_manager: DecayManager) -> None:
        """After two reinforcements, stability = 1.0 * 1.5 * 1.5 = 2.25."""
        item_id = uuid4()
        custom_manager.reinforce(item_id)
        new_stability = custom_manager.reinforce(item_id)
        expected = Decimal("1.0") * Decimal("1.5") * Decimal("1.5")
        assert abs(new_stability - expected) < Decimal("0.001"), (
            f"Expected {expected}, got {new_stability}"
        )

    def test_stability_capped_at_max(self, custom_manager: DecayManager) -> None:
        """Stability should never exceed max_stability (10.0 in custom settings)."""
        item_id = uuid4()
        # Reinforce many times to try to exceed max
        for _ in range(50):
            custom_manager.reinforce(item_id)
        stability = custom_manager.get_stability(item_id)
        assert stability <= Decimal("10.0"), f"Stability exceeds max: {stability}"

    def test_set_stability_clamps_minimum(self, custom_manager: DecayManager) -> None:
        """set_stability should clamp to minimum of 0.1."""
        item_id = uuid4()
        custom_manager.set_stability(item_id, Decimal("0.01"))
        stability = custom_manager.get_stability(item_id)
        assert stability == Decimal("0.1"), f"Expected minimum 0.1, got {stability}"

    def test_reinforcement_tracks_access_count(self, custom_manager: DecayManager) -> None:
        """Access count should increment on each reinforcement."""
        item_id = uuid4()
        custom_manager.reinforce(item_id)
        custom_manager.reinforce(item_id)
        custom_manager.reinforce(item_id)
        record = custom_manager._stability_records[item_id]
        assert record.access_count == 3


class TestDecayDoesNotCompound:
    """Applying decay should use time from reference (created_at/accessed_at),
    NOT from previous decay result. Bug C-20 fix verification."""

    def test_decay_uses_absolute_time_not_previous_result(self, custom_manager: DecayManager) -> None:
        """Calling apply_decay twice with same item should produce consistent results
        because decay is computed from the reference time, not from previous strength."""
        item_id = uuid4()
        original_strength = Decimal("1.0")
        created_at = datetime.now(timezone.utc) - timedelta(days=10)

        result1 = custom_manager.apply_decay(
            item_id=item_id,
            current_strength=original_strength,
            retention_category="medium_term",
            created_at=created_at,
        )

        # Apply decay again with same parameters -- same reference time
        result2 = custom_manager.apply_decay(
            item_id=item_id,
            current_strength=original_strength,
            retention_category="medium_term",
            created_at=created_at,
        )

        # Both calls compute from the same reference time, so results should be equal
        assert abs(result1.new_strength - result2.new_strength) < Decimal("0.001"), (
            f"Decay should not compound: first={result1.new_strength}, second={result2.new_strength}"
        )

    def test_decay_from_same_reference_gives_same_result(self, custom_manager: DecayManager) -> None:
        """Two different items with identical parameters should produce identical decay."""
        created_at = datetime.now(timezone.utc) - timedelta(days=5)
        original_strength = Decimal("1.0")

        result_a = custom_manager.apply_decay(
            item_id=uuid4(),
            current_strength=original_strength,
            retention_category="short_term",
            created_at=created_at,
        )
        result_b = custom_manager.apply_decay(
            item_id=uuid4(),
            current_strength=original_strength,
            retention_category="short_term",
            created_at=created_at,
        )

        assert abs(result_a.new_strength - result_b.new_strength) < Decimal("0.001")

    def test_decay_result_independent_of_current_strength_argument(self, custom_manager: DecayManager) -> None:
        """The new_strength is computed from e^(-lambda*t)*S, not from current_strength.
        So passing different current_strength values should NOT affect new_strength."""
        item_id_a = uuid4()
        item_id_b = uuid4()
        created_at = datetime.now(timezone.utc) - timedelta(days=10)

        result_a = custom_manager.apply_decay(
            item_id=item_id_a,
            current_strength=Decimal("1.0"),
            retention_category="medium_term",
            created_at=created_at,
        )
        result_b = custom_manager.apply_decay(
            item_id=item_id_b,
            current_strength=Decimal("0.5"),
            retention_category="medium_term",
            created_at=created_at,
        )

        # new_strength is computed from formula, independent of current_strength
        assert abs(result_a.new_strength - result_b.new_strength) < Decimal("0.001"), (
            f"new_strength should not depend on current_strength input: "
            f"a={result_a.new_strength}, b={result_b.new_strength}"
        )


class TestDecayActionThresholds:
    """Verify decay action determination based on retention strength thresholds."""

    def test_high_strength_keeps(self, default_manager: DecayManager) -> None:
        """Strength >= 0.7 should result in KEEP."""
        action = default_manager._determine_action(Decimal("0.8"))
        assert action == DecayAction.KEEP

    def test_medium_strength_keeps(self, default_manager: DecayManager) -> None:
        """Strength >= archive_threshold (0.3) should still KEEP."""
        action = default_manager._determine_action(Decimal("0.35"))
        assert action == DecayAction.KEEP

    def test_low_strength_archives(self, default_manager: DecayManager) -> None:
        """Strength >= delete_threshold but < archive_threshold should ARCHIVE."""
        action = default_manager._determine_action(Decimal("0.15"))
        assert action == DecayAction.ARCHIVE

    def test_very_low_strength_deletes(self, default_manager: DecayManager) -> None:
        """Strength < delete_threshold (0.1) should DELETE."""
        action = default_manager._determine_action(Decimal("0.05"))
        assert action == DecayAction.DELETE


class TestDecayModifiers:
    """Verify emotional and clinical decay modifiers."""

    def test_emotional_modifier_reduces_decay_rate(self, custom_manager: DecayManager) -> None:
        """Emotional content should have a reduced decay rate (slower decay)."""
        base_rate = custom_manager.get_decay_rate(RetentionCategory.MEDIUM_TERM, is_emotional=False)
        emotional_rate = custom_manager.get_decay_rate(RetentionCategory.MEDIUM_TERM, is_emotional=True)
        assert emotional_rate < base_rate, (
            f"Emotional rate ({emotional_rate}) should be less than base ({base_rate})"
        )

    def test_clinical_modifier_reduces_decay_rate(self, custom_manager: DecayManager) -> None:
        """Clinical content should have a reduced decay rate."""
        base_rate = custom_manager.get_decay_rate(RetentionCategory.MEDIUM_TERM, is_clinical=False)
        clinical_rate = custom_manager.get_decay_rate(RetentionCategory.MEDIUM_TERM, is_clinical=True)
        assert clinical_rate < base_rate, (
            f"Clinical rate ({clinical_rate}) should be less than base ({base_rate})"
        )

    def test_both_modifiers_stack(self, custom_manager: DecayManager) -> None:
        """Emotional + clinical modifiers should both apply."""
        base_rate = custom_manager.get_decay_rate(RetentionCategory.MEDIUM_TERM)
        both_rate = custom_manager.get_decay_rate(
            RetentionCategory.MEDIUM_TERM, is_emotional=True, is_clinical=True
        )
        emotional_only = custom_manager.get_decay_rate(
            RetentionCategory.MEDIUM_TERM, is_emotional=True
        )
        assert both_rate < emotional_only < base_rate


class TestBatchProcessing:
    """Verify batch decay processing."""

    def test_batch_processes_all_items(self, custom_manager: DecayManager) -> None:
        """Batch processing should handle all items."""
        now = datetime.now(timezone.utc)
        items = [
            (uuid4(), Decimal("1.0"), "medium_term", now - timedelta(days=5), None),
            (uuid4(), Decimal("0.8"), "short_term", now - timedelta(days=10), None),
            (uuid4(), Decimal("0.9"), "long_term", now - timedelta(days=15), None),
        ]
        result = custom_manager.process_batch(items)
        assert result.items_processed == 3

    def test_batch_safety_override_marks_permanent(self, custom_manager: DecayManager) -> None:
        """Batch processing with safety override should mark crisis categories permanent."""
        now = datetime.now(timezone.utc)
        crisis_id = uuid4()
        items = [
            (crisis_id, Decimal("1.0"), "crisis", now - timedelta(days=5), None),
        ]
        result = custom_manager.process_batch(items, apply_safety_override=True)
        assert result.items_processed == 1
        assert custom_manager.is_permanent(crisis_id) is True
