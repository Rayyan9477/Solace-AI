"""Alignment tests for clinical threshold consistency with design spec.

Verifies that confidence calibration thresholds, risk level enums,
PHQ-9 severity mappings, and PCL-5 scoring parameters match the
Solace-AI system design specification.
"""

import pytest

from solace_common.enums import CrisisLevel
from services.diagnosis_service.src.schemas import ConfidenceLevel
from services.diagnosis_service.src.domain.confidence import ConfidenceCalibrator
from services.diagnosis_service.src.domain.severity import SeverityAssessor


class TestClinicalAlignment:
    """Tests that clinical thresholds and enums match the design specification."""

    def test_confidence_thresholds_match_spec(self) -> None:
        """Verify ConfidenceCalibrator thresholds match the unified spec:

        >= 0.70 = HIGH
        >= 0.50 = MEDIUM
        >= 0.30 = LOW
        < 0.30  = ESCALATE

        These thresholds determine when diagnoses are considered confident
        enough to act on and when clinical escalation is required.
        """
        calibrator = ConfidenceCalibrator()

        # At and above 0.70 -> HIGH
        assert calibrator._determine_confidence_level(0.70) == ConfidenceLevel.HIGH
        assert calibrator._determine_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert calibrator._determine_confidence_level(1.00) == ConfidenceLevel.HIGH

        # 0.50 to just below 0.70 -> MEDIUM
        assert calibrator._determine_confidence_level(0.50) == ConfidenceLevel.MEDIUM
        assert calibrator._determine_confidence_level(0.60) == ConfidenceLevel.MEDIUM
        assert calibrator._determine_confidence_level(0.69) == ConfidenceLevel.MEDIUM

        # 0.30 to just below 0.50 -> LOW
        assert calibrator._determine_confidence_level(0.30) == ConfidenceLevel.LOW
        assert calibrator._determine_confidence_level(0.40) == ConfidenceLevel.LOW
        assert calibrator._determine_confidence_level(0.49) == ConfidenceLevel.LOW

        # Below 0.30 -> ESCALATE
        assert calibrator._determine_confidence_level(0.29) == ConfidenceLevel.ESCALATE
        assert calibrator._determine_confidence_level(0.10) == ConfidenceLevel.ESCALATE
        assert calibrator._determine_confidence_level(0.00) == ConfidenceLevel.ESCALATE

    def test_risk_level_enum_values(self) -> None:
        """Verify CrisisLevel has exactly NONE, LOW, ELEVATED, HIGH, CRITICAL.

        These five levels map to the system design spec's risk tiers and
        drive safety escalation logic across all services.
        """
        expected_names = {"NONE", "LOW", "ELEVATED", "HIGH", "CRITICAL"}
        actual_names = {level.name for level in CrisisLevel}

        assert actual_names == expected_names, (
            f"CrisisLevel enum mismatch.\n"
            f"  Expected: {sorted(expected_names)}\n"
            f"  Actual:   {sorted(actual_names)}"
        )

        # Also verify string values match names (canonical convention)
        for level in CrisisLevel:
            assert level.value == level.name, (
                f"CrisisLevel.{level.name} has value '{level.value}', "
                f"expected '{level.name}'"
            )

    def test_phq9_moderately_severe_maps_to_3(self) -> None:
        """Verify PHQ-9 severity scoring maps MODERATELY_SEVERE to score 3.

        The SeverityAssessor uses a severity_to_score mapping when inferring
        questionnaire responses from symptoms. MODERATELY_SEVERE must map
        to 3 (the maximum per-item score on PHQ-9) to correctly reflect
        clinical severity.
        """
        assessor = SeverityAssessor()

        # Access the inferred severity-to-score mapping
        from solace_common.enums import SeverityLevel

        # Build a test symptom that maps to a PHQ-9 item
        from services.diagnosis_service.src.schemas import SymptomDTO
        from uuid import uuid4

        symptom = SymptomDTO(
            symptom_id=uuid4(),
            name="anhedonia",
            description="Loss of interest",
            symptom_type="emotional",
            severity=SeverityLevel.MODERATELY_SEVERE,
        )

        responses = assessor._infer_responses_from_symptoms([symptom])

        # PHQ-9 item 1 maps to 'anhedonia'; score should be 3
        assert responses.get("phq9_1") == 3, (
            f"Expected PHQ-9 item phq9_1 score=3 for MODERATELY_SEVERE, "
            f"got {responses.get('phq9_1')}"
        )

    def test_pcl5_max_score_is_40(self) -> None:
        """Verify 10-item PCL-5 has max_score=40.

        The abbreviated PCL-5 uses 10 items scored 0-4, giving a maximum
        possible score of 40. This is critical for correct PTSD severity
        classification thresholds.
        """
        assessor = SeverityAssessor()

        # Score with empty responses to get the QuestionnaireResult template
        result = assessor._score_pcl5({})

        assert result.max_score == 40, (
            f"Expected PCL-5 max_score=40, got {result.max_score}"
        )
        assert result.questionnaire == "PCL-5", (
            f"Expected questionnaire name 'PCL-5', got '{result.questionnaire}'"
        )

        # Verify there are exactly 10 items
        assert len(assessor._pcl5_items) == 10, (
            f"Expected 10 PCL-5 items, got {len(assessor._pcl5_items)}"
        )
