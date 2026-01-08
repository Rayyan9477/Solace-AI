"""
Tests for contraindication.py - Therapeutic technique contraindication checking.
"""
import pytest
from decimal import Decimal
from uuid import uuid4
from services.safety_service.src.ml.contraindication import (
    ContraindicationChecker,
    ContraindicationConfig,
    ContraindicationResult,
    ContraindicationType,
    TherapyTechnique,
    MentalHealthCondition,
    ContraindicationMatch,
)


class TestContraindicationChecker:
    """Test suite for ContraindicationChecker."""

    @pytest.fixture
    def checker(self) -> ContraindicationChecker:
        """Create contraindication checker with default config."""
        return ContraindicationChecker()

    def test_initialization(self, checker: ContraindicationChecker) -> None:
        """Test checker initializes correctly."""
        assert checker is not None
        assert len(checker._rules) > 0

    def test_check_no_contraindications(self, checker: ContraindicationChecker) -> None:
        """Test technique with no contraindications."""
        result = checker.check(
            technique=TherapyTechnique.GROUNDING_TECHNIQUES,
            conditions=[]
        )

        assert result.is_safe is True
        assert result.safety_level == "SAFE"
        assert len(result.contraindications) == 0

    def test_check_absolute_contraindication(self, checker: ContraindicationChecker) -> None:
        """Test absolute contraindication blocking."""
        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS]
        )

        assert result.is_safe is False
        assert result.safety_level == "UNSAFE"
        assert len(result.contraindications) > 0
        assert result.contraindications[0].contraindication_type == ContraindicationType.ABSOLUTE

    def test_check_relative_contraindication(self, checker: ContraindicationChecker) -> None:
        """Test relative contraindication caution."""
        result = checker.check(
            technique=TherapyTechnique.COGNITIVE_RESTRUCTURING,
            conditions=[MentalHealthCondition.SEVERE_DEPRESSION]
        )

        assert result.safety_level in ["CAUTION", "UNSAFE"]
        assert len(result.contraindications) > 0

    def test_check_exposure_therapy_crisis(self, checker: ContraindicationChecker) -> None:
        """Test exposure therapy contraindicated during crisis."""
        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACUTE_CRISIS]
        )

        assert result.is_safe is False
        assert result.safety_level == "UNSAFE"

    def test_check_emdr_dissociation(self, checker: ContraindicationChecker) -> None:
        """Test EMDR contraindicated with dissociative disorder."""
        result = checker.check(
            technique=TherapyTechnique.EMDR,
            conditions=[MentalHealthCondition.DISSOCIATIVE_DISORDER]
        )

        assert result.is_safe is False
        assert len(result.contraindications) > 0

    def test_check_multiple_conditions(self, checker: ContraindicationChecker) -> None:
        """Test checking with multiple conditions."""
        result = checker.check(
            technique=TherapyTechnique.MINDFULNESS_MEDITATION,
            conditions=[
                MentalHealthCondition.SEVERE_PTSD,
                MentalHealthCondition.ACUTE_CRISIS
            ]
        )

        # May have multiple contraindications
        assert result is not None

    def test_calculate_risk_score(self, checker: ContraindicationChecker) -> None:
        """Test risk score calculation."""
        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS]
        )

        assert Decimal("0.0") <= result.risk_score <= Decimal("1.0")
        assert result.risk_score >= Decimal("0.9")  # Absolute contraindication

    def test_get_safe_alternatives(self, checker: ContraindicationChecker) -> None:
        """Test getting safe alternative techniques."""
        alternatives = checker.get_safe_alternatives(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACUTE_CRISIS]
        )

        assert len(alternatives) > 0
        # Should suggest alternatives like grounding or distress tolerance
        assert any(alt in [TherapyTechnique.GROUNDING_TECHNIQUES,
                          TherapyTechnique.DBT_DISTRESS_TOLERANCE]
                  for alt in alternatives)

    def test_get_safe_alternatives_safe_technique(self, checker: ContraindicationChecker) -> None:
        """Test alternatives when technique is already safe."""
        alternatives = checker.get_safe_alternatives(
            technique=TherapyTechnique.GROUNDING_TECHNIQUES,
            conditions=[]
        )

        assert TherapyTechnique.GROUNDING_TECHNIQUES in alternatives

    def test_clinical_notes_unsafe(self, checker: ContraindicationChecker) -> None:
        """Test clinical notes for unsafe technique."""
        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS]
        )

        assert "ABSOLUTE CONTRAINDICATION" in result.clinical_notes
        assert len(result.clinical_notes) > 0

    def test_clinical_notes_caution(self, checker: ContraindicationChecker) -> None:
        """Test clinical notes for caution level."""
        result = checker.check(
            technique=TherapyTechnique.COGNITIVE_RESTRUCTURING,
            conditions=[MentalHealthCondition.SEVERE_DEPRESSION]
        )

        if result.safety_level == "CAUTION":
            assert "CAUTION" in result.clinical_notes

    def test_clinical_notes_safe(self, checker: ContraindicationChecker) -> None:
        """Test clinical notes for safe technique."""
        result = checker.check(
            technique=TherapyTechnique.GROUNDING_TECHNIQUES,
            conditions=[]
        )

        assert "appropriate" in result.clinical_notes.lower()

    def test_check_with_user_id(self, checker: ContraindicationChecker) -> None:
        """Test check with user ID for logging."""
        user_id = uuid4()

        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS],
            user_id=user_id
        )

        assert result is not None

    def test_check_with_context(self, checker: ContraindicationChecker) -> None:
        """Test check with additional context."""
        context = {"session_number": 1, "severity_score": 0.8}

        result = checker.check(
            technique=TherapyTechnique.DBT_DIARY_CARD,
            conditions=[MentalHealthCondition.ACUTE_CRISIS],
            context=context
        )

        assert result is not None

    def test_contraindication_match_attributes(self, checker: ContraindicationChecker) -> None:
        """Test contraindication match has all attributes."""
        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS]
        )

        if result.contraindications:
            match = result.contraindications[0]
            assert match.technique == TherapyTechnique.EXPOSURE_THERAPY
            assert match.condition == MentalHealthCondition.ACTIVE_PSYCHOSIS
            assert match.contraindication_type in ContraindicationType
            assert Decimal("0.0") <= match.severity <= Decimal("1.0")
            assert len(match.rationale) > 0

    def test_result_attributes(self, checker: ContraindicationChecker) -> None:
        """Test result has all required attributes."""
        result = checker.check(
            technique=TherapyTechnique.MINDFULNESS_MEDITATION,
            conditions=[MentalHealthCondition.SEVERE_PTSD]
        )

        assert result.check_id is not None
        assert result.timestamp is not None
        assert result.technique == TherapyTechnique.MINDFULNESS_MEDITATION
        assert isinstance(result.is_safe, bool)
        assert result.safety_level in ["SAFE", "CAUTION", "UNSAFE"]
        assert Decimal("0.0") <= result.risk_score <= Decimal("1.0")

    def test_config_absolute_checks_disabled(self) -> None:
        """Test disabling absolute contraindication checks."""
        config = ContraindicationConfig(enable_absolute_checks=False)
        checker = ContraindicationChecker(config)

        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS]
        )

        # Should not detect absolute contraindications
        absolute_contras = [c for c in result.contraindications
                           if c.contraindication_type == ContraindicationType.ABSOLUTE]
        assert len(absolute_contras) == 0

    def test_dbt_diary_card_prerequisites(self, checker: ContraindicationChecker) -> None:
        """Test DBT diary card requires prerequisites."""
        result = checker.check(
            technique=TherapyTechnique.DBT_DIARY_CARD,
            conditions=[MentalHealthCondition.ACUTE_CRISIS]
        )

        if result.contraindications:
            assert result.contraindications[0].contraindication_type == ContraindicationType.TECHNIQUE_SPECIFIC
            assert len(result.contraindications[0].prerequisites) > 0

    def test_behavioral_activation_suicidal_ideation(self, checker: ContraindicationChecker) -> None:
        """Test behavioral activation insufficient for suicidal ideation."""
        result = checker.check(
            technique=TherapyTechnique.BEHAVIORAL_ACTIVATION,
            conditions=[MentalHealthCondition.SUICIDAL_IDEATION]
        )

        # Should have severity mismatch contraindication
        assert len(result.contraindications) > 0

    def test_alternative_techniques_provided(self, checker: ContraindicationChecker) -> None:
        """Test alternatives are provided in contraindications."""
        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS]
        )

        if result.contraindications:
            assert len(result.contraindications[0].alternative_techniques) > 0

    def test_severity_threshold_matching(self, checker: ContraindicationChecker) -> None:
        """Test severity thresholds determine safety level."""
        result = checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS]
        )

        # Absolute contraindication should result in UNSAFE
        if result.risk_score >= checker._config.absolute_block_threshold:
            assert result.safety_level == "UNSAFE"
