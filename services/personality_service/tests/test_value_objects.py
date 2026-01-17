"""
Unit tests for Personality Service Domain Value Objects.
Tests immutable value objects for OCEAN scores and communication styles.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
import pytest

from services.personality_service.src.domain.value_objects import (
    TraitScore, OceanScores, CommunicationStyle, AssessmentMetadata,
)
from services.personality_service.src.schemas import (
    PersonalityTrait, CommunicationStyleType, AssessmentSource,
)


class TestTraitScore:
    """Tests for TraitScore value object."""

    def test_valid_trait_score(self) -> None:
        """Test valid trait score creation."""
        score = TraitScore(
            trait=PersonalityTrait.OPENNESS,
            value=Decimal("0.75"),
            confidence_lower=Decimal("0.65"),
            confidence_upper=Decimal("0.85"),
            sample_count=5,
            evidence_markers=("insight_words", "question_marks"),
        )
        assert score.trait == PersonalityTrait.OPENNESS
        assert score.value == Decimal("0.75")
        assert score.confidence_lower == Decimal("0.65")
        assert score.confidence_upper == Decimal("0.85")
        assert score.sample_count == 5
        assert len(score.evidence_markers) == 2

    def test_trait_score_is_frozen(self) -> None:
        """Test trait score immutability."""
        score = TraitScore(trait=PersonalityTrait.OPENNESS, value=Decimal("0.5"))
        with pytest.raises(Exception):
            score.value = Decimal("0.6")

    def test_invalid_value_range(self) -> None:
        """Test validation of value range."""
        with pytest.raises(ValueError):
            TraitScore(trait=PersonalityTrait.OPENNESS, value=Decimal("1.5"))
        with pytest.raises(ValueError):
            TraitScore(trait=PersonalityTrait.OPENNESS, value=Decimal("-0.1"))

    def test_invalid_confidence_bounds(self) -> None:
        """Test validation of confidence bounds."""
        with pytest.raises(ValueError):
            TraitScore(
                trait=PersonalityTrait.OPENNESS,
                value=Decimal("0.5"),
                confidence_lower=Decimal("0.8"),
                confidence_upper=Decimal("0.3"),
            )

    def test_confidence_width(self) -> None:
        """Test confidence width calculation."""
        score = TraitScore(
            trait=PersonalityTrait.OPENNESS,
            value=Decimal("0.5"),
            confidence_lower=Decimal("0.3"),
            confidence_upper=Decimal("0.7"),
        )
        assert score.confidence_width == Decimal("0.4")

    def test_is_high_confidence(self) -> None:
        """Test high confidence detection."""
        high_conf = TraitScore(
            trait=PersonalityTrait.OPENNESS,
            value=Decimal("0.5"),
            confidence_lower=Decimal("0.45"),
            confidence_upper=Decimal("0.55"),
        )
        low_conf = TraitScore(
            trait=PersonalityTrait.OPENNESS,
            value=Decimal("0.5"),
            confidence_lower=Decimal("0.2"),
            confidence_upper=Decimal("0.8"),
        )
        assert high_conf.is_high_confidence is True
        assert low_conf.is_high_confidence is False

    def test_with_evidence(self) -> None:
        """Test adding evidence markers."""
        score = TraitScore(
            trait=PersonalityTrait.OPENNESS,
            value=Decimal("0.5"),
            evidence_markers=("marker1",),
        )
        new_score = score.with_evidence(("marker2", "marker3"))
        assert len(new_score.evidence_markers) == 3
        assert "marker1" in new_score.evidence_markers
        assert "marker2" in new_score.evidence_markers

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        original = TraitScore(
            trait=PersonalityTrait.CONSCIENTIOUSNESS,
            value=Decimal("0.65"),
            confidence_lower=Decimal("0.55"),
            confidence_upper=Decimal("0.75"),
            sample_count=3,
            evidence_markers=("achievement",),
        )
        data = original.to_dict()
        restored = TraitScore.from_dict(data)
        assert restored.trait == original.trait
        assert restored.value == original.value
        assert restored.sample_count == original.sample_count


class TestOceanScores:
    """Tests for OceanScores value object."""

    def test_valid_ocean_scores(self) -> None:
        """Test valid OCEAN scores creation."""
        scores = OceanScores(
            openness=Decimal("0.72"),
            conscientiousness=Decimal("0.45"),
            extraversion=Decimal("0.38"),
            agreeableness=Decimal("0.81"),
            neuroticism=Decimal("0.56"),
            overall_confidence=Decimal("0.75"),
        )
        assert scores.openness == Decimal("0.72")
        assert scores.conscientiousness == Decimal("0.45")
        assert scores.extraversion == Decimal("0.38")
        assert scores.agreeableness == Decimal("0.81")
        assert scores.neuroticism == Decimal("0.56")
        assert scores.overall_confidence == Decimal("0.75")

    def test_ocean_scores_is_frozen(self) -> None:
        """Test OCEAN scores immutability."""
        scores = OceanScores.neutral()
        with pytest.raises(Exception):
            scores.openness = Decimal("0.8")

    def test_invalid_trait_value(self) -> None:
        """Test validation of trait values."""
        with pytest.raises(ValueError):
            OceanScores(
                openness=Decimal("1.5"),
                conscientiousness=Decimal("0.5"),
                extraversion=Decimal("0.5"),
                agreeableness=Decimal("0.5"),
                neuroticism=Decimal("0.5"),
            )

    def test_neutral_scores(self) -> None:
        """Test neutral scores factory method."""
        scores = OceanScores.neutral()
        assert scores.openness == Decimal("0.5")
        assert scores.conscientiousness == Decimal("0.5")
        assert scores.extraversion == Decimal("0.5")
        assert scores.agreeableness == Decimal("0.5")
        assert scores.neuroticism == Decimal("0.5")
        assert scores.overall_confidence == Decimal("0.3")

    def test_get_trait(self) -> None:
        """Test get_trait method."""
        scores = OceanScores(
            openness=Decimal("0.8"),
            conscientiousness=Decimal("0.4"),
            extraversion=Decimal("0.6"),
            agreeableness=Decimal("0.7"),
            neuroticism=Decimal("0.3"),
        )
        assert scores.get_trait(PersonalityTrait.OPENNESS) == Decimal("0.8")
        assert scores.get_trait(PersonalityTrait.NEUROTICISM) == Decimal("0.3")

    def test_dominant_traits(self) -> None:
        """Test dominant_traits method."""
        scores = OceanScores(
            openness=Decimal("0.85"),
            conscientiousness=Decimal("0.45"),
            extraversion=Decimal("0.75"),
            agreeableness=Decimal("0.60"),
            neuroticism=Decimal("0.30"),
        )
        dominant = scores.dominant_traits()
        assert PersonalityTrait.OPENNESS in dominant
        assert PersonalityTrait.EXTRAVERSION in dominant
        assert PersonalityTrait.AGREEABLENESS not in dominant

    def test_low_traits(self) -> None:
        """Test low_traits method."""
        scores = OceanScores(
            openness=Decimal("0.85"),
            conscientiousness=Decimal("0.25"),
            extraversion=Decimal("0.75"),
            agreeableness=Decimal("0.60"),
            neuroticism=Decimal("0.20"),
        )
        low = scores.low_traits()
        assert PersonalityTrait.CONSCIENTIOUSNESS in low
        assert PersonalityTrait.NEUROTICISM in low
        assert PersonalityTrait.OPENNESS not in low

    def test_trait_vector(self) -> None:
        """Test trait vector property."""
        scores = OceanScores(
            openness=Decimal("0.1"),
            conscientiousness=Decimal("0.2"),
            extraversion=Decimal("0.3"),
            agreeableness=Decimal("0.4"),
            neuroticism=Decimal("0.5"),
        )
        vector = scores.trait_vector
        assert vector == (Decimal("0.1"), Decimal("0.2"), Decimal("0.3"), Decimal("0.4"), Decimal("0.5"))

    def test_distance_to(self) -> None:
        """Test Euclidean distance calculation."""
        scores1 = OceanScores.neutral()
        scores2 = OceanScores(
            openness=Decimal("0.6"),
            conscientiousness=Decimal("0.5"),
            extraversion=Decimal("0.5"),
            agreeableness=Decimal("0.5"),
            neuroticism=Decimal("0.5"),
        )
        distance = scores1.distance_to(scores2)
        assert distance > Decimal("0")
        assert scores1.distance_to(scores1) == Decimal("0")

    def test_aggregate_with(self) -> None:
        """Test score aggregation with EMA."""
        scores1 = OceanScores.neutral()
        scores2 = OceanScores(
            openness=Decimal("0.8"),
            conscientiousness=Decimal("0.8"),
            extraversion=Decimal("0.8"),
            agreeableness=Decimal("0.8"),
            neuroticism=Decimal("0.8"),
        )
        aggregated = scores1.aggregate_with(scores2, alpha=Decimal("0.5"))
        assert aggregated.openness == Decimal("0.65")

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        original = OceanScores(
            openness=Decimal("0.72"),
            conscientiousness=Decimal("0.45"),
            extraversion=Decimal("0.38"),
            agreeableness=Decimal("0.81"),
            neuroticism=Decimal("0.56"),
            overall_confidence=Decimal("0.75"),
        )
        data = original.to_dict()
        restored = OceanScores.from_dict(data)
        assert restored.openness == original.openness
        assert restored.overall_confidence == original.overall_confidence


class TestCommunicationStyle:
    """Tests for CommunicationStyle value object."""

    def test_valid_communication_style(self) -> None:
        """Test valid communication style creation."""
        style_id = uuid4()
        style = CommunicationStyle(
            style_id=style_id,
            warmth=Decimal("0.8"),
            structure=Decimal("0.7"),
            complexity=Decimal("0.6"),
            directness=Decimal("0.4"),
            energy=Decimal("0.5"),
            validation_level=Decimal("0.9"),
            style_type=CommunicationStyleType.AMIABLE,
        )
        assert style.style_id == style_id
        assert style.warmth == Decimal("0.8")
        assert style.style_type == CommunicationStyleType.AMIABLE

    def test_communication_style_is_frozen(self) -> None:
        """Test communication style immutability."""
        style = CommunicationStyle(style_id=uuid4())
        with pytest.raises(Exception):
            style.warmth = Decimal("0.9")

    def test_invalid_parameter_value(self) -> None:
        """Test validation of parameter values."""
        with pytest.raises(ValueError):
            CommunicationStyle(style_id=uuid4(), warmth=Decimal("1.5"))

    def test_is_warm(self) -> None:
        """Test warm style detection."""
        warm = CommunicationStyle(style_id=uuid4(), warmth=Decimal("0.8"))
        not_warm = CommunicationStyle(style_id=uuid4(), warmth=Decimal("0.5"))
        assert warm.is_warm is True
        assert not_warm.is_warm is False

    def test_is_structured(self) -> None:
        """Test structured style detection."""
        structured = CommunicationStyle(style_id=uuid4(), structure=Decimal("0.8"))
        not_structured = CommunicationStyle(style_id=uuid4(), structure=Decimal("0.5"))
        assert structured.is_structured is True
        assert not_structured.is_structured is False

    def test_needs_validation(self) -> None:
        """Test validation needs detection."""
        needs_val = CommunicationStyle(style_id=uuid4(), validation_level=Decimal("0.8"))
        no_val = CommunicationStyle(style_id=uuid4(), validation_level=Decimal("0.5"))
        assert needs_val.needs_validation is True
        assert no_val.needs_validation is False

    def test_from_ocean(self) -> None:
        """Test deriving style from OCEAN scores."""
        scores = OceanScores(
            openness=Decimal("0.8"),
            conscientiousness=Decimal("0.8"),
            extraversion=Decimal("0.3"),
            agreeableness=Decimal("0.7"),
            neuroticism=Decimal("0.4"),
        )
        style = CommunicationStyle.from_ocean(scores)
        assert style.style_type == CommunicationStyleType.ANALYTICAL
        assert style.warmth >= Decimal("0")
        assert style.structure >= Decimal("0")

    def test_from_ocean_expressive(self) -> None:
        """Test deriving expressive style from OCEAN scores."""
        scores = OceanScores(
            openness=Decimal("0.6"),
            conscientiousness=Decimal("0.5"),
            extraversion=Decimal("0.8"),
            agreeableness=Decimal("0.8"),
            neuroticism=Decimal("0.3"),
        )
        style = CommunicationStyle.from_ocean(scores)
        assert style.style_type == CommunicationStyleType.EXPRESSIVE

    def test_from_ocean_driver(self) -> None:
        """Test deriving driver style from OCEAN scores."""
        scores = OceanScores(
            openness=Decimal("0.6"),
            conscientiousness=Decimal("0.6"),
            extraversion=Decimal("0.8"),
            agreeableness=Decimal("0.4"),
            neuroticism=Decimal("0.3"),
        )
        style = CommunicationStyle.from_ocean(scores)
        assert style.style_type == CommunicationStyleType.DRIVER

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        original = CommunicationStyle(
            style_id=uuid4(),
            warmth=Decimal("0.7"),
            structure=Decimal("0.6"),
            complexity=Decimal("0.5"),
            directness=Decimal("0.4"),
            energy=Decimal("0.3"),
            validation_level=Decimal("0.8"),
            style_type=CommunicationStyleType.ANALYTICAL,
            custom_params=(("key1", "value1"),),
        )
        data = original.to_dict()
        restored = CommunicationStyle.from_dict(data)
        assert restored.warmth == original.warmth
        assert restored.style_type == original.style_type


class TestAssessmentMetadata:
    """Tests for AssessmentMetadata value object."""

    def test_valid_metadata(self) -> None:
        """Test valid metadata creation."""
        assessment_id = uuid4()
        metadata = AssessmentMetadata(
            assessment_id=assessment_id,
            source=AssessmentSource.TEXT_ANALYSIS,
            text_length=500,
            model_version="1.0.0",
            processing_time_ms=150.5,
            context_tags=("therapy", "initial"),
        )
        assert metadata.assessment_id == assessment_id
        assert metadata.source == AssessmentSource.TEXT_ANALYSIS
        assert metadata.text_length == 500
        assert metadata.processing_time_ms == 150.5

    def test_invalid_text_length(self) -> None:
        """Test validation of text length."""
        with pytest.raises(ValueError):
            AssessmentMetadata(
                assessment_id=uuid4(),
                source=AssessmentSource.TEXT_ANALYSIS,
                text_length=-1,
            )

    def test_is_text_sufficient(self) -> None:
        """Test text sufficiency check."""
        sufficient = AssessmentMetadata(
            assessment_id=uuid4(),
            source=AssessmentSource.TEXT_ANALYSIS,
            text_length=100,
        )
        insufficient = AssessmentMetadata(
            assessment_id=uuid4(),
            source=AssessmentSource.TEXT_ANALYSIS,
            text_length=30,
        )
        assert sufficient.is_text_sufficient is True
        assert insufficient.is_text_sufficient is False

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        original = AssessmentMetadata(
            assessment_id=uuid4(),
            source=AssessmentSource.ENSEMBLE,
            text_length=200,
            model_version="2.0.0",
            processing_time_ms=75.0,
            context_tags=("session_1",),
        )
        data = original.to_dict()
        restored = AssessmentMetadata.from_dict(data)
        assert restored.source == original.source
        assert restored.text_length == original.text_length
