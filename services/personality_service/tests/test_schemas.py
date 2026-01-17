"""
Unit tests for Personality Service Schemas.
Tests Pydantic models, enums, and DTOs.
"""
from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
import pytest

from services.personality_service.src.schemas import (
    PersonalityTrait, AssessmentSource, CommunicationStyleType, EmotionCategory,
    TraitScoreDTO, OceanScoresDTO, StyleParametersDTO, EmotionStateDTO,
    EmpathyComponentsDTO, DetectPersonalityRequest, DetectPersonalityResponse,
    GetStyleRequest, GetStyleResponse, AdaptResponseRequest, ProfileSummaryDTO,
)


class TestPersonalityEnums:
    """Tests for personality-related enums."""

    def test_personality_traits(self) -> None:
        """Test PersonalityTrait enum values."""
        assert PersonalityTrait.OPENNESS.value == "openness"
        assert PersonalityTrait.CONSCIENTIOUSNESS.value == "conscientiousness"
        assert PersonalityTrait.EXTRAVERSION.value == "extraversion"
        assert PersonalityTrait.AGREEABLENESS.value == "agreeableness"
        assert PersonalityTrait.NEUROTICISM.value == "neuroticism"
        assert len(PersonalityTrait) == 5

    def test_assessment_sources(self) -> None:
        """Test AssessmentSource enum values."""
        assert AssessmentSource.TEXT_ANALYSIS.value == "text_analysis"
        assert AssessmentSource.LLM_ZERO_SHOT.value == "llm_zero_shot"
        assert AssessmentSource.ENSEMBLE.value == "ensemble"

    def test_communication_styles(self) -> None:
        """Test CommunicationStyleType enum values."""
        assert CommunicationStyleType.ANALYTICAL.value == "analytical"
        assert CommunicationStyleType.EXPRESSIVE.value == "expressive"
        assert CommunicationStyleType.DRIVER.value == "driver"
        assert CommunicationStyleType.AMIABLE.value == "amiable"
        assert CommunicationStyleType.BALANCED.value == "balanced"

    def test_emotion_categories(self) -> None:
        """Test EmotionCategory enum values."""
        assert EmotionCategory.JOY.value == "joy"
        assert EmotionCategory.SADNESS.value == "sadness"
        assert EmotionCategory.NEUTRAL.value == "neutral"


class TestTraitScoreDTO:
    """Tests for TraitScoreDTO."""

    def test_valid_trait_score(self) -> None:
        """Test valid trait score creation."""
        score = TraitScoreDTO(
            trait=PersonalityTrait.OPENNESS,
            value=0.75,
            confidence_lower=0.65,
            confidence_upper=0.85,
            sample_count=5,
            evidence_markers=["insight_words", "question_marks"],
        )
        assert score.trait == PersonalityTrait.OPENNESS
        assert score.value == 0.75
        assert score.confidence_lower == 0.65
        assert score.confidence_upper == 0.85
        assert score.sample_count == 5
        assert len(score.evidence_markers) == 2

    def test_trait_score_defaults(self) -> None:
        """Test trait score defaults."""
        score = TraitScoreDTO(
            trait=PersonalityTrait.NEUROTICISM,
            value=0.5,
            confidence_lower=0.4,
            confidence_upper=0.6,
        )
        assert score.sample_count == 1
        assert score.evidence_markers == []

    def test_confidence_validation(self) -> None:
        """Test confidence bounds are valid."""
        score = TraitScoreDTO(
            trait=PersonalityTrait.EXTRAVERSION,
            value=0.5,
            confidence_lower=0.7,
            confidence_upper=0.5,
        )
        assert score.confidence_upper >= score.confidence_lower


class TestOceanScoresDTO:
    """Tests for OceanScoresDTO."""

    def test_valid_ocean_scores(self) -> None:
        """Test valid OCEAN scores creation."""
        scores = OceanScoresDTO(
            openness=0.72,
            conscientiousness=0.45,
            extraversion=0.38,
            agreeableness=0.81,
            neuroticism=0.56,
            overall_confidence=0.75,
        )
        assert scores.openness == 0.72
        assert scores.conscientiousness == 0.45
        assert scores.extraversion == 0.38
        assert scores.agreeableness == 0.81
        assert scores.neuroticism == 0.56
        assert scores.overall_confidence == 0.75

    def test_get_trait(self) -> None:
        """Test get_trait method."""
        scores = OceanScoresDTO(
            openness=0.8,
            conscientiousness=0.4,
            extraversion=0.6,
            agreeableness=0.7,
            neuroticism=0.3,
        )
        assert scores.get_trait(PersonalityTrait.OPENNESS) == 0.8
        assert scores.get_trait(PersonalityTrait.NEUROTICISM) == 0.3

    def test_dominant_traits(self) -> None:
        """Test dominant_traits method."""
        scores = OceanScoresDTO(
            openness=0.85,
            conscientiousness=0.45,
            extraversion=0.75,
            agreeableness=0.60,
            neuroticism=0.30,
        )
        dominant = scores.dominant_traits(threshold=0.7)
        assert PersonalityTrait.OPENNESS in dominant
        assert PersonalityTrait.EXTRAVERSION in dominant
        assert PersonalityTrait.AGREEABLENESS not in dominant

    def test_low_traits(self) -> None:
        """Test low_traits method."""
        scores = OceanScoresDTO(
            openness=0.85,
            conscientiousness=0.25,
            extraversion=0.75,
            agreeableness=0.60,
            neuroticism=0.20,
        )
        low = scores.low_traits(threshold=0.3)
        assert PersonalityTrait.CONSCIENTIOUSNESS in low
        assert PersonalityTrait.NEUROTICISM in low
        assert PersonalityTrait.OPENNESS not in low


class TestStyleParametersDTO:
    """Tests for StyleParametersDTO."""

    def test_default_style_parameters(self) -> None:
        """Test default style parameters."""
        params = StyleParametersDTO()
        assert params.warmth == 0.5
        assert params.structure == 0.5
        assert params.complexity == 0.5
        assert params.directness == 0.5
        assert params.energy == 0.5
        assert params.validation_level == 0.5
        assert params.style_type == CommunicationStyleType.BALANCED

    def test_custom_style_parameters(self) -> None:
        """Test custom style parameters."""
        params = StyleParametersDTO(
            warmth=0.8,
            structure=0.7,
            complexity=0.6,
            directness=0.4,
            energy=0.3,
            validation_level=0.9,
            style_type=CommunicationStyleType.AMIABLE,
        )
        assert params.warmth == 0.8
        assert params.style_type == CommunicationStyleType.AMIABLE


class TestEmotionStateDTO:
    """Tests for EmotionStateDTO."""

    def test_emotion_state_creation(self) -> None:
        """Test emotion state creation."""
        state = EmotionStateDTO(
            primary_emotion=EmotionCategory.SADNESS,
            secondary_emotion=EmotionCategory.FEAR,
            intensity=0.7,
            valence=-0.5,
            arousal=0.6,
            confidence=0.8,
        )
        assert state.primary_emotion == EmotionCategory.SADNESS
        assert state.secondary_emotion == EmotionCategory.FEAR
        assert state.intensity == 0.7
        assert state.valence == -0.5


class TestRequestResponseDTOs:
    """Tests for request/response DTOs."""

    def test_detect_personality_request(self) -> None:
        """Test detect personality request."""
        user_id = uuid4()
        request = DetectPersonalityRequest(
            user_id=user_id,
            text="This is a sample text for personality analysis that should be long enough.",
            include_evidence=True,
        )
        assert request.user_id == user_id
        assert len(request.text) > 10
        assert request.include_evidence is True

    def test_detect_personality_request_min_length(self) -> None:
        """Test minimum text length validation."""
        with pytest.raises(ValueError):
            DetectPersonalityRequest(
                user_id=uuid4(),
                text="short",
            )

    def test_get_style_request(self) -> None:
        """Test get style request."""
        user_id = uuid4()
        request = GetStyleRequest(user_id=user_id, context="therapy session")
        assert request.user_id == user_id
        assert request.context == "therapy session"

    def test_adapt_response_request(self) -> None:
        """Test adapt response request."""
        user_id = uuid4()
        request = AdaptResponseRequest(
            user_id=user_id,
            base_response="This is the base therapeutic response.",
            include_empathy=True,
        )
        assert request.user_id == user_id
        assert len(request.base_response) > 0
        assert request.include_empathy is True


class TestProfileSummaryDTO:
    """Tests for ProfileSummaryDTO."""

    def test_profile_summary(self) -> None:
        """Test profile summary creation."""
        user_id = uuid4()
        scores = OceanScoresDTO(
            openness=0.8,
            conscientiousness=0.6,
            extraversion=0.4,
            agreeableness=0.7,
            neuroticism=0.3,
        )
        params = StyleParametersDTO()
        summary = ProfileSummaryDTO(
            user_id=user_id,
            ocean_scores=scores,
            style_parameters=params,
            dominant_traits=[PersonalityTrait.OPENNESS],
            assessment_count=5,
            stability_score=0.85,
            last_updated=datetime.now(timezone.utc),
            version=3,
        )
        assert summary.user_id == user_id
        assert summary.assessment_count == 5
        assert summary.stability_score == 0.85
        assert summary.version == 3
        assert PersonalityTrait.OPENNESS in summary.dominant_traits
