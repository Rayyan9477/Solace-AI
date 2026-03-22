"""
Solace-AI Diagnosis Service - Confidence Threshold Unit Tests.
Verify confidence thresholds match spec: HIGH>=0.70, MEDIUM>=0.50, LOW>=0.30, ESCALATE<0.30.
"""
from __future__ import annotations

import pytest

from services.diagnosis_service.src.domain.confidence import (
    ConfidenceCalibrator,
    ConfidenceSettings,
)
from services.diagnosis_service.src.schemas import ConfidenceLevel


class TestConfidenceThresholds:
    """Verify confidence thresholds match unified spec.

    Spec thresholds:
        >= 0.70 = HIGH
        >= 0.50 = MEDIUM
        >= 0.30 = LOW
        <  0.30 = ESCALATE
    """

    def setup_method(self) -> None:
        self.calibrator = ConfidenceCalibrator(ConfidenceSettings())

    # ---- Interior values ----

    def test_high_confidence(self) -> None:
        """Score of 0.75 should be HIGH."""
        level = self.calibrator._determine_confidence_level(0.75)
        assert level == ConfidenceLevel.HIGH

    def test_medium_confidence(self) -> None:
        """Score of 0.55 should be MEDIUM."""
        level = self.calibrator._determine_confidence_level(0.55)
        assert level == ConfidenceLevel.MEDIUM

    def test_low_confidence(self) -> None:
        """Score of 0.35 should be LOW."""
        level = self.calibrator._determine_confidence_level(0.35)
        assert level == ConfidenceLevel.LOW

    def test_escalate_confidence(self) -> None:
        """Score of 0.20 should be ESCALATE."""
        level = self.calibrator._determine_confidence_level(0.20)
        assert level == ConfidenceLevel.ESCALATE

    # ---- Boundary values (inclusive lower bounds) ----

    def test_boundary_070_is_high(self) -> None:
        """Exactly 0.70 should be HIGH (inclusive lower bound)."""
        level = self.calibrator._determine_confidence_level(0.70)
        assert level == ConfidenceLevel.HIGH

    def test_boundary_050_is_medium(self) -> None:
        """Exactly 0.50 should be MEDIUM (inclusive lower bound)."""
        level = self.calibrator._determine_confidence_level(0.50)
        assert level == ConfidenceLevel.MEDIUM

    def test_boundary_030_is_low(self) -> None:
        """Exactly 0.30 should be LOW (inclusive lower bound)."""
        level = self.calibrator._determine_confidence_level(0.30)
        assert level == ConfidenceLevel.LOW

    # ---- Just below boundaries ----

    def test_just_below_070_is_medium(self) -> None:
        """0.6999 should be MEDIUM, not HIGH."""
        level = self.calibrator._determine_confidence_level(0.6999)
        assert level == ConfidenceLevel.MEDIUM

    def test_just_below_050_is_low(self) -> None:
        """0.4999 should be LOW, not MEDIUM."""
        level = self.calibrator._determine_confidence_level(0.4999)
        assert level == ConfidenceLevel.LOW

    def test_just_below_030_is_escalate(self) -> None:
        """0.2999 should be ESCALATE, not LOW."""
        level = self.calibrator._determine_confidence_level(0.2999)
        assert level == ConfidenceLevel.ESCALATE

    # ---- Extreme values ----

    def test_zero_confidence_is_escalate(self) -> None:
        """Score of 0.0 should be ESCALATE."""
        level = self.calibrator._determine_confidence_level(0.0)
        assert level == ConfidenceLevel.ESCALATE

    def test_max_confidence_is_high(self) -> None:
        """Score of 1.0 should be HIGH."""
        level = self.calibrator._determine_confidence_level(1.0)
        assert level == ConfidenceLevel.HIGH

    def test_near_zero_is_escalate(self) -> None:
        """Score of 0.01 should be ESCALATE."""
        level = self.calibrator._determine_confidence_level(0.01)
        assert level == ConfidenceLevel.ESCALATE

    def test_near_max_is_high(self) -> None:
        """Score of 0.95 should be HIGH."""
        level = self.calibrator._determine_confidence_level(0.95)
        assert level == ConfidenceLevel.HIGH
