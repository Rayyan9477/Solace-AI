"""
Unit tests for Domain Value Objects.
Tests Technique, OutcomeMeasure, and related value objects.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
import pytest

from services.therapy_service.src.domain.value_objects import (
    Technique, OutcomeMeasure, TherapeuticRationale, SessionContext,
)
from services.therapy_service.src.schemas import (
    TherapyModality, TechniqueCategory, DifficultyLevel, DeliveryMode,
    OutcomeInstrument, SeverityLevel,
)


class TestTechnique:
    """Tests for Technique value object."""

    def test_create_technique(self) -> None:
        """Test technique creation."""
        technique = Technique(
            technique_id=uuid4(),
            name="Thought Record",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.COGNITIVE_RESTRUCTURING,
            description="Structured worksheet for examining thoughts",
            duration_minutes=15,
        )
        assert technique.name == "Thought Record"
        assert technique.modality == TherapyModality.CBT
        assert technique.duration_minutes == 15

    def test_technique_immutable(self) -> None:
        """Test technique is immutable."""
        technique = Technique(
            technique_id=uuid4(),
            name="Test",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.COGNITIVE_RESTRUCTURING,
            description="Test",
        )
        with pytest.raises(Exception):
            technique.name = "Modified"

    def test_duration_validation(self) -> None:
        """Test duration validation."""
        with pytest.raises(ValueError):
            Technique(
                technique_id=uuid4(),
                name="Test",
                modality=TherapyModality.CBT,
                category=TechniqueCategory.COGNITIVE_RESTRUCTURING,
                description="Test",
                duration_minutes=0,
            )

    def test_effectiveness_validation(self) -> None:
        """Test effectiveness rating validation."""
        with pytest.raises(ValueError):
            Technique(
                technique_id=uuid4(),
                name="Test",
                modality=TherapyModality.CBT,
                category=TechniqueCategory.COGNITIVE_RESTRUCTURING,
                description="Test",
                effectiveness_rating=Decimal("1.5"),
            )

    def test_is_applicable_for_severity(self) -> None:
        """Test severity applicability."""
        beginner_technique = Technique(
            technique_id=uuid4(),
            name="Test",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.RELAXATION,
            description="Test",
            difficulty=DifficultyLevel.BEGINNER,
        )
        assert beginner_technique.is_applicable_for(SeverityLevel.MILD) is True
        assert beginner_technique.is_applicable_for(SeverityLevel.SEVERE) is False

    def test_has_contraindication(self) -> None:
        """Test contraindication checking."""
        technique = Technique(
            technique_id=uuid4(),
            name="Exposure Therapy",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.EXPOSURE,
            description="Gradual exposure",
            contraindications=("psychosis", "acute crisis"),
        )
        assert technique.has_contraindication("psychosis") is True
        assert technique.has_contraindication("Acute Crisis") is True
        assert technique.has_contraindication("depression") is False

    def test_meets_prerequisites(self) -> None:
        """Test prerequisite checking."""
        technique = Technique(
            technique_id=uuid4(),
            name="Advanced CBT",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.COGNITIVE_RESTRUCTURING,
            description="Advanced technique",
            prerequisites=("basic_cbt", "thought_records"),
        )
        assert technique.meets_prerequisites(["basic_cbt", "thought_records"]) is True
        assert technique.meets_prerequisites(["basic_cbt"]) is False
        assert technique.meets_prerequisites([]) is False

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        technique_id = uuid4()
        technique = Technique(
            technique_id=technique_id,
            name="Test",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.RELAXATION,
            description="Test description",
        )
        data = technique.to_dict()
        assert data["technique_id"] == str(technique_id)
        assert data["name"] == "Test"
        assert data["modality"] == "cbt"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        technique_id = uuid4()
        data = {
            "technique_id": str(technique_id),
            "name": "Test",
            "modality": "cbt",
            "category": "relaxation",
            "description": "Test",
        }
        technique = Technique.from_dict(data)
        assert technique.technique_id == technique_id
        assert technique.modality == TherapyModality.CBT


class TestOutcomeMeasure:
    """Tests for OutcomeMeasure value object."""

    def test_create_phq9_measure(self) -> None:
        """Test PHQ-9 measure creation."""
        measure = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=14,
        )
        assert measure.raw_score == 14
        assert measure.instrument == OutcomeInstrument.PHQ9

    def test_phq9_score_validation(self) -> None:
        """Test PHQ-9 score validation."""
        with pytest.raises(ValueError):
            OutcomeMeasure(
                measure_id=uuid4(),
                instrument=OutcomeInstrument.PHQ9,
                raw_score=30,  # Max is 27
            )

    def test_gad7_score_validation(self) -> None:
        """Test GAD-7 score validation."""
        with pytest.raises(ValueError):
            OutcomeMeasure(
                measure_id=uuid4(),
                instrument=OutcomeInstrument.GAD7,
                raw_score=25,  # Max is 21
            )

    def test_severity_category_phq9(self) -> None:
        """Test PHQ-9 severity categorization."""
        minimal = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=3)
        mild = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=7)
        moderate = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=12)
        mod_severe = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=17)
        severe = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=22)
        assert minimal.severity_category == SeverityLevel.MINIMAL
        assert mild.severity_category == SeverityLevel.MILD
        assert moderate.severity_category == SeverityLevel.MODERATE
        assert mod_severe.severity_category == SeverityLevel.MODERATELY_SEVERE
        assert severe.severity_category == SeverityLevel.SEVERE

    def test_is_clinical_phq9(self) -> None:
        """Test PHQ-9 clinical status."""
        subclinical = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=8)
        clinical = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=12)
        assert subclinical.is_clinical is False
        assert clinical.is_clinical is True

    def test_is_clinical_ors(self) -> None:
        """Test ORS clinical status (higher is better)."""
        clinical = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.ORS, raw_score=20)
        non_clinical = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.ORS, raw_score=30)
        assert clinical.is_clinical is True
        assert non_clinical.is_clinical is False

    def test_normalized_score_phq9(self) -> None:
        """Test PHQ-9 normalized score."""
        measure = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=14)
        normalized = measure.normalized_score
        assert 0 <= normalized <= 100
        assert normalized > 40  # 14/27 * 100 ≈ 51.9

    def test_calculate_change_improved(self) -> None:
        """Test change calculation for improvement."""
        previous = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=16,
        )
        current = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=8,
        )
        change = current.calculate_change(previous)
        assert change["raw_change"] == -8
        assert change["reliable_change"] is True
        assert change["status"] == "improved"
        assert change["clinically_significant"] is True

    def test_calculate_change_deteriorated(self) -> None:
        """Test change calculation for deterioration."""
        previous = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=8,
        )
        current = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=16,
        )
        change = current.calculate_change(previous)
        assert change["raw_change"] == 8
        assert change["status"] == "deteriorated"

    def test_calculate_change_no_change(self) -> None:
        """Test change calculation for no reliable change."""
        previous = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=10,
        )
        current = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=12,
        )
        change = current.calculate_change(previous)
        assert change["reliable_change"] is False
        assert change["status"] == "no_change"

    def test_calculate_change_different_instruments(self) -> None:
        """Test change calculation fails for different instruments."""
        phq9 = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=10)
        gad7 = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.GAD7, raw_score=10)
        with pytest.raises(ValueError):
            gad7.calculate_change(phq9)

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        measure_id = uuid4()
        measure = OutcomeMeasure(
            measure_id=measure_id,
            instrument=OutcomeInstrument.PHQ9,
            raw_score=14,
        )
        data = measure.to_dict()
        assert data["measure_id"] == str(measure_id)
        assert data["raw_score"] == 14
        assert data["severity_category"] == "MODERATE"
        assert data["is_clinical"] is True

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        measure_id = uuid4()
        data = {
            "measure_id": str(measure_id),
            "instrument": "phq9",
            "raw_score": 10,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        measure = OutcomeMeasure.from_dict(data)
        assert measure.measure_id == measure_id
        assert measure.raw_score == 10


class TestTherapeuticRationale:
    """Tests for TherapeuticRationale value object."""

    def test_create_rationale(self) -> None:
        """Test rationale creation."""
        technique_id = uuid4()
        rationale = TherapeuticRationale(
            technique_id=technique_id,
            selection_reason="High efficacy for depression",
            clinical_factors=("moderate_depression", "first_episode"),
            confidence_score=Decimal("0.85"),
        )
        assert rationale.technique_id == technique_id
        assert rationale.confidence_score == Decimal("0.85")

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        technique_id = uuid4()
        rationale = TherapeuticRationale(
            technique_id=technique_id,
            selection_reason="Test reason",
        )
        data = rationale.to_dict()
        assert data["technique_id"] == str(technique_id)
        assert data["selection_reason"] == "Test reason"


class TestSessionContext:
    """Tests for SessionContext value object."""

    def test_create_context(self) -> None:
        """Test context creation."""
        context = SessionContext(
            session_number=5,
            treatment_phase="active_treatment",
            presenting_concerns=("anxiety", "sleep_issues"),
            mood_state="low",
        )
        assert context.session_number == 5
        assert context.mood_state == "low"
        assert "anxiety" in context.presenting_concerns

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        context = SessionContext(
            session_number=3,
            treatment_phase="foundation",
        )
        data = context.to_dict()
        assert data["session_number"] == 3
        assert data["treatment_phase"] == "foundation"
