"""
REV-27 clinical-accuracy reconciliation Prove-It tests.

Reconciles two documented clinical CLAIMS (docs/SYSTEM-DESIGN-SUMMARY.md
"Bayesian Confidence Calibration") against the confidence calibrator:

  1. Sample-consistency uses N=3 samples (spec: "N=3-5 LLM samples").
     Code had drifted to calibration_samples=5.
  2. "<60% agreement -> Uncertain" is a documented low-confidence safety
     label that had no implementation. Below the agreement threshold the
     calibration result must be surfaced as labelled "Uncertain".
"""
from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from services.diagnosis_service.src.domain.confidence import (
    ConfidenceCalibrator,
    ConfidenceSettings,
)
from services.diagnosis_service.src.schemas import (
    HypothesisDTO,
    SymptomDTO,
    SymptomType,
    SeverityLevel,
)


class TestCalibrationSampleCount:
    """Claim (1): calibration uses N=3 samples per the documented spec."""

    def test_calibration_samples_default_is_three(self) -> None:
        """calibration_samples must default to 3 (documented spec lower bound)."""
        assert ConfidenceSettings().calibration_samples == 3


class TestUncertainAgreementLabel:
    """Claim (2): agreement below 60% is labelled 'Uncertain'."""

    def setup_method(self) -> None:
        self.calibrator = ConfidenceCalibrator(ConfidenceSettings())

    # ---- Direct helper band mapping (spec bands) ----

    def test_below_threshold_is_uncertain(self) -> None:
        """Agreement < 0.60 -> 'Uncertain'."""
        assert self.calibrator._determine_agreement_label(0.50) == "Uncertain"

    def test_just_below_threshold_is_uncertain(self) -> None:
        """0.5999 agreement -> 'Uncertain' (below 60%)."""
        assert self.calibrator._determine_agreement_label(0.5999) == "Uncertain"

    def test_boundary_060_is_not_uncertain(self) -> None:
        """Exactly 0.60 agreement is the moderate band, not 'Uncertain'."""
        assert self.calibrator._determine_agreement_label(0.60) == "Moderate"

    def test_moderate_band(self) -> None:
        """60-80% agreement -> 'Moderate'."""
        assert self.calibrator._determine_agreement_label(0.70) == "Moderate"
        assert self.calibrator._determine_agreement_label(0.80) == "Moderate"

    def test_high_band(self) -> None:
        """>80% agreement -> 'High'."""
        assert self.calibrator._determine_agreement_label(0.81) == "High"
        assert self.calibrator._determine_agreement_label(0.95) == "High"

    # ---- End-to-end calibrate() surfaces the label ----

    @pytest.mark.asyncio
    async def test_low_agreement_calibration_is_labelled_uncertain(self) -> None:
        """A low-consistency hypothesis yields a result labelled 'Uncertain'.

        Adjustment Disorder matches no symptom-consistency rule, so
        symptom_consistency collapses to met_ratio (0 here); with no
        temporal info the overall agreement lands well below 60%.
        """
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(),
            name="Adjustment Disorder",
            confidence=Decimal("0.5"),
            criteria_met=[],
            criteria_missing=["stressor", "impairment", "onset_within_3mo"],
        )
        symptoms = [
            SymptomDTO(
                symptom_id=uuid4(),
                name="restlessness",
                description="unsettled",
                symptom_type=SymptomType.EMOTIONAL,
                severity=SeverityLevel.MILD,
            ),
        ]
        result = await self.calibrator.calibrate(hypothesis, symptoms, {})
        assert result.agreement_label == "Uncertain"

    @pytest.mark.asyncio
    async def test_high_agreement_calibration_is_not_uncertain(self) -> None:
        """A well-supported, temporally-anchored hypothesis is not 'Uncertain'."""
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
        result = await self.calibrator.calibrate(hypothesis, symptoms, {})
        assert result.agreement_label != "Uncertain"
