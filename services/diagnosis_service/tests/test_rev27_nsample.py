"""
REV-27 (follow-up): true N-sample Bayesian calibration Prove-It tests.

REV-27 reconciled calibration to N=3 + an `Uncertain (<0.60)` agreement
label, but the agreement signal was a single-shot CONSISTENCY-SCORE PROXY --
no independent samples were ever drawn. These tests close that gap: the
calibrator now draws N independent estimates from an injectable/seedable
sampler and derives agreement from their DISTRIBUTION (mean + spread):

  * tight cluster of high estimates -> high agreement -> "High"
  * wide disagreement across samples -> spread penalty -> "Uncertain"
  * degenerate N=1 / no sampler injected -> collapses to the legacy
    consistency proxy (backward compatible, inert by default)

The sampler is injectable so the N-sample path is deterministic under test.
"""
from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from services.diagnosis_service.src.domain.confidence import (
    ConfidenceCalibrator,
    ConfidenceSettings,
    SeededGaussianSampler,
)
from services.diagnosis_service.src.schemas import (
    HypothesisDTO,
    SymptomDTO,
    SymptomType,
    SeverityLevel,
)


def _gad_case() -> tuple[HypothesisDTO, list[SymptomDTO]]:
    """A well-supported hypothesis whose base consistency proxy is high."""
    hypothesis = HypothesisDTO(
        hypothesis_id=uuid4(),
        name="Generalized Anxiety Disorder",
        confidence=Decimal("0.7"),
        criteria_met=["anxiety", "worry"],
        criteria_missing=[],
    )
    symptoms = [
        SymptomDTO(
            symptom_id=uuid4(),
            name="anxiety",
            description="persistent worry",
            symptom_type=SymptomType.EMOTIONAL,
            severity=SeverityLevel.MODERATE,
            onset="6 months ago",
            duration="6 months",
        ),
    ]
    return hypothesis, symptoms


class TestNSampleAggregation:
    """The N-sample path draws independent estimates and aggregates them."""

    @pytest.mark.asyncio
    async def test_injected_sampler_records_sample_count(self) -> None:
        """An injected sampler yielding 3 estimates surfaces sample_count == 3."""
        cal = ConfidenceCalibrator(
            ConfidenceSettings(),
            sampler=lambda base, n: [0.90, 0.88, 0.92],
        )
        hyp, symptoms = _gad_case()
        result = await cal.calibrate(hyp, symptoms, {})
        assert result.sample_count == 3

    @pytest.mark.asyncio
    async def test_tight_high_cluster_is_high_agreement(self) -> None:
        """N samples clustered high (near-zero spread) -> 'High', not Uncertain."""
        cal = ConfidenceCalibrator(
            ConfidenceSettings(),
            sampler=lambda base, n: [0.90, 0.90, 0.90],
        )
        hyp, symptoms = _gad_case()
        result = await cal.calibrate(hyp, symptoms, {})
        assert result.agreement_label == "High"
        assert result.sample_spread == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_wide_disagreement_is_uncertain(self) -> None:
        """Widely scattered samples -> spread penalty drags agreement < 0.60."""
        cal = ConfidenceCalibrator(
            ConfidenceSettings(),
            sampler=lambda base, n: [0.95, 0.15, 0.60],
        )
        hyp, symptoms = _gad_case()
        result = await cal.calibrate(hyp, symptoms, {})
        assert result.agreement_label == "Uncertain"
        assert result.sample_spread > 0.0

    def test_seeded_gaussian_sampler_is_deterministic(self) -> None:
        """Same seed -> identical draws (deterministic under test)."""
        sampler = SeededGaussianSampler(seed=42, sigma=0.1)
        first = sampler(0.7, 3)
        second = sampler(0.7, 3)
        assert first == second
        assert len(first) == 3
        assert all(0.0 <= v <= 1.0 for v in first)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different sample sets."""
        assert SeededGaussianSampler(seed=1)(0.7, 5) != SeededGaussianSampler(seed=2)(0.7, 5)


class TestBackwardCompatibleDegenerate:
    """No sampler injected -> inert: collapses to the legacy consistency proxy."""

    @pytest.mark.asyncio
    async def test_default_calibrator_zero_spread(self) -> None:
        """Default (degenerate) sampler yields identical copies -> zero spread."""
        cal = ConfidenceCalibrator()
        hyp, symptoms = _gad_case()
        result = await cal.calibrate(hyp, symptoms, {})
        assert result.sample_spread == pytest.approx(0.0, abs=1e-9)
        assert result.sample_count == ConfidenceSettings().calibration_samples

    @pytest.mark.asyncio
    async def test_degenerate_agreement_matches_consistency_proxy(self) -> None:
        """Degenerate agreement == base consistency score (unchanged behavior)."""
        cal = ConfidenceCalibrator()
        hyp, symptoms = _gad_case()
        consistency = await cal.analyze_consistency(hyp, symptoms)
        result = await cal.calibrate(hyp, symptoms, {})
        assert result.sample_agreement == pytest.approx(consistency.consistency_score, abs=1e-9)

    def test_degenerate_n1_returns_single_estimate(self) -> None:
        """Explicit N=1 degenerate aggregate is the point estimate, zero spread."""
        cal = ConfidenceCalibrator(ConfidenceSettings(calibration_samples=1))
        agg = cal._aggregate_samples(0.42, 1)
        assert agg.samples == [0.42]
        assert agg.mean == pytest.approx(0.42)
        assert agg.spread == pytest.approx(0.0)
        assert agg.agreement == pytest.approx(0.42)
