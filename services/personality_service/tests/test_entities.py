"""
Unit tests for Personality Service Domain Entities.
Tests PersonalityProfile, TraitAssessment, and related entities.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
import pytest

from services.personality_service.src.domain.entities import (
    TraitAssessment, PersonalityProfile, ProfileSnapshot, ProfileComparison,
)
from services.personality_service.src.domain.value_objects import (
    OceanScores, CommunicationStyle, AssessmentMetadata,
)
from services.personality_service.src.schemas import (
    PersonalityTrait, AssessmentSource, CommunicationStyleType,
)


class TestTraitAssessment:
    """Tests for TraitAssessment entity."""

    def test_create_assessment(self) -> None:
        """Test assessment creation."""
        user_id = uuid4()
        scores = OceanScores.neutral()
        assessment = TraitAssessment(
            user_id=user_id,
            ocean_scores=scores,
            source=AssessmentSource.TEXT_ANALYSIS,
        )
        assert assessment.user_id == user_id
        assert assessment.ocean_scores == scores
        assert assessment.source == AssessmentSource.TEXT_ANALYSIS
        assert assessment.version == 1

    def test_add_evidence(self) -> None:
        """Test adding evidence markers."""
        assessment = TraitAssessment()
        assessment.add_evidence("high_openness")
        assessment.add_evidence("social_focus")
        assessment.add_evidence("high_openness")
        assert len(assessment.evidence) == 2
        assert "high_openness" in assessment.evidence

    def test_get_trait_value(self) -> None:
        """Test getting specific trait value."""
        scores = OceanScores(
            openness=Decimal("0.8"),
            conscientiousness=Decimal("0.6"),
            extraversion=Decimal("0.4"),
            agreeableness=Decimal("0.7"),
            neuroticism=Decimal("0.3"),
        )
        assessment = TraitAssessment(ocean_scores=scores)
        assert assessment.get_trait_value(PersonalityTrait.OPENNESS) == Decimal("0.8")
        assert assessment.get_trait_value(PersonalityTrait.NEUROTICISM) == Decimal("0.3")

    def test_confidence_property(self) -> None:
        """Test confidence property."""
        scores = OceanScores(
            openness=Decimal("0.5"),
            conscientiousness=Decimal("0.5"),
            extraversion=Decimal("0.5"),
            agreeableness=Decimal("0.5"),
            neuroticism=Decimal("0.5"),
            overall_confidence=Decimal("0.75"),
        )
        assessment = TraitAssessment(ocean_scores=scores)
        assert assessment.confidence == Decimal("0.75")

    def test_is_reliable(self) -> None:
        """Test reliability check."""
        reliable = TraitAssessment(
            ocean_scores=OceanScores(
                openness=Decimal("0.5"), conscientiousness=Decimal("0.5"),
                extraversion=Decimal("0.5"), agreeableness=Decimal("0.5"),
                neuroticism=Decimal("0.5"), overall_confidence=Decimal("0.6"),
            )
        )
        unreliable = TraitAssessment(
            ocean_scores=OceanScores(
                openness=Decimal("0.5"), conscientiousness=Decimal("0.5"),
                extraversion=Decimal("0.5"), agreeableness=Decimal("0.5"),
                neuroticism=Decimal("0.5"), overall_confidence=Decimal("0.3"),
            )
        )
        assert reliable.is_reliable is True
        assert unreliable.is_reliable is False

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        original = TraitAssessment(
            user_id=uuid4(),
            ocean_scores=OceanScores.neutral(),
            source=AssessmentSource.ENSEMBLE,
            evidence=["marker1", "marker2"],
        )
        data = original.to_dict()
        restored = TraitAssessment.from_dict(data)
        assert restored.user_id == original.user_id
        assert restored.source == original.source
        assert restored.evidence == original.evidence


class TestPersonalityProfile:
    """Tests for PersonalityProfile aggregate root."""

    def test_create_profile(self) -> None:
        """Test profile creation."""
        user_id = uuid4()
        profile = PersonalityProfile.create_for_user(user_id)
        assert profile.user_id == user_id
        assert profile.ocean_scores is None
        assert profile.assessment_count == 0
        assert profile.version == 1

    def test_add_first_assessment(self) -> None:
        """Test adding first assessment."""
        profile = PersonalityProfile()
        scores = OceanScores(
            openness=Decimal("0.7"),
            conscientiousness=Decimal("0.6"),
            extraversion=Decimal("0.5"),
            agreeableness=Decimal("0.8"),
            neuroticism=Decimal("0.3"),
        )
        assessment = TraitAssessment(ocean_scores=scores)
        profile.add_assessment(assessment)
        assert profile.ocean_scores is not None
        assert profile.assessment_count == 1
        assert profile.communication_style is not None
        assert profile.version == 2

    def test_add_multiple_assessments(self) -> None:
        """Test adding multiple assessments with aggregation."""
        profile = PersonalityProfile()
        scores1 = OceanScores.neutral()
        scores2 = OceanScores(
            openness=Decimal("0.8"),
            conscientiousness=Decimal("0.8"),
            extraversion=Decimal("0.8"),
            agreeableness=Decimal("0.8"),
            neuroticism=Decimal("0.8"),
        )
        profile.add_assessment(TraitAssessment(ocean_scores=scores1))
        profile.add_assessment(TraitAssessment(ocean_scores=scores2))
        assert profile.assessment_count == 2
        assert profile.ocean_scores.openness > Decimal("0.5")

    def test_stability_score_computation(self) -> None:
        """Test stability score computation."""
        profile = PersonalityProfile()
        assert profile.stability_score == Decimal("0.0")
        scores = OceanScores.neutral()
        for _ in range(5):
            profile.add_assessment(TraitAssessment(ocean_scores=scores))
        assert profile.stability_score > Decimal("0.5")

    def test_is_stable(self) -> None:
        """Test stability check."""
        profile = PersonalityProfile()
        scores = OceanScores.neutral()
        for _ in range(5):
            profile.add_assessment(TraitAssessment(ocean_scores=scores))
        assert profile.is_stable is True

    def test_has_sufficient_data(self) -> None:
        """Test sufficient data check."""
        profile = PersonalityProfile()
        assert profile.has_sufficient_data is False
        for _ in range(3):
            profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        assert profile.has_sufficient_data is True

    def test_dominant_traits(self) -> None:
        """Test dominant traits property."""
        profile = PersonalityProfile()
        scores = OceanScores(
            openness=Decimal("0.85"),
            conscientiousness=Decimal("0.45"),
            extraversion=Decimal("0.75"),
            agreeableness=Decimal("0.60"),
            neuroticism=Decimal("0.30"),
        )
        profile.add_assessment(TraitAssessment(ocean_scores=scores))
        dominant = profile.dominant_traits
        assert PersonalityTrait.OPENNESS in dominant
        assert PersonalityTrait.EXTRAVERSION in dominant

    def test_style_type(self) -> None:
        """Test style type property."""
        profile = PersonalityProfile()
        assert profile.style_type == CommunicationStyleType.BALANCED
        scores = OceanScores(
            openness=Decimal("0.8"),
            conscientiousness=Decimal("0.8"),
            extraversion=Decimal("0.3"),
            agreeableness=Decimal("0.6"),
            neuroticism=Decimal("0.4"),
        )
        profile.add_assessment(TraitAssessment(ocean_scores=scores))
        assert profile.style_type == CommunicationStyleType.ANALYTICAL

    def test_get_recent_assessments(self) -> None:
        """Test getting recent assessments."""
        profile = PersonalityProfile()
        for i in range(10):
            profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        recent = profile.get_recent_assessments(5)
        assert len(recent) == 5

    def test_get_trait_history(self) -> None:
        """Test getting trait history."""
        profile = PersonalityProfile()
        for i in range(5):
            scores = OceanScores(
                openness=Decimal(str(0.5 + i * 0.1)),
                conscientiousness=Decimal("0.5"),
                extraversion=Decimal("0.5"),
                agreeableness=Decimal("0.5"),
                neuroticism=Decimal("0.5"),
            )
            profile.add_assessment(TraitAssessment(ocean_scores=scores))
        history = profile.get_trait_history(PersonalityTrait.OPENNESS)
        assert len(history) == 5

    def test_calculate_trait_trend(self) -> None:
        """Test calculating trait trend."""
        profile = PersonalityProfile()
        for i in range(5):
            scores = OceanScores(
                openness=Decimal(str(0.5 + i * 0.1)),
                conscientiousness=Decimal("0.5"),
                extraversion=Decimal("0.5"),
                agreeableness=Decimal("0.5"),
                neuroticism=Decimal("0.5"),
            )
            profile.add_assessment(TraitAssessment(ocean_scores=scores))
        trend = profile.calculate_trait_trend(PersonalityTrait.OPENNESS)
        assert trend == "increasing"

    def test_reset_scores(self) -> None:
        """Test resetting profile scores."""
        profile = PersonalityProfile()
        profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        profile.reset_scores()
        assert profile.ocean_scores is None
        assert profile.assessment_count == 0
        assert len(profile.assessment_history) == 0

    def test_max_history_size(self) -> None:
        """Test history size limit."""
        profile = PersonalityProfile(_max_history_size=5)
        for _ in range(10):
            profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        assert len(profile.assessment_history) == 5

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        profile = PersonalityProfile.create_for_user(uuid4())
        profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        data = profile.to_dict()
        restored = PersonalityProfile.from_dict(data)
        assert restored.user_id == profile.user_id
        assert restored.assessment_count == profile.assessment_count


class TestProfileSnapshot:
    """Tests for ProfileSnapshot entity."""

    def test_create_snapshot_from_profile(self) -> None:
        """Test creating snapshot from profile."""
        profile = PersonalityProfile.create_for_user(uuid4())
        profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        snapshot = ProfileSnapshot.from_profile(profile, reason="checkpoint")
        assert snapshot.profile_id == profile.profile_id
        assert snapshot.user_id == profile.user_id
        assert snapshot.assessment_count == profile.assessment_count
        assert snapshot.reason == "checkpoint"

    def test_snapshot_to_dict(self) -> None:
        """Test snapshot serialization."""
        profile = PersonalityProfile.create_for_user(uuid4())
        profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        snapshot = ProfileSnapshot.from_profile(profile)
        data = snapshot.to_dict()
        assert "snapshot_id" in data
        assert "ocean_scores" in data


class TestProfileComparison:
    """Tests for ProfileComparison entity."""

    def test_create_comparison(self) -> None:
        """Test creating profile comparison."""
        profile = PersonalityProfile.create_for_user(uuid4())
        profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        baseline = ProfileSnapshot.from_profile(profile)
        scores_new = OceanScores(
            openness=Decimal("1.0"),
            conscientiousness=Decimal("0.5"),
            extraversion=Decimal("0.5"),
            agreeableness=Decimal("0.5"),
            neuroticism=Decimal("0.5"),
        )
        for _ in range(3):
            profile.add_assessment(TraitAssessment(ocean_scores=scores_new))
        current = ProfileSnapshot.from_profile(profile)
        comparison = ProfileComparison(
            profile_id=profile.profile_id,
            baseline_snapshot=baseline,
            current_snapshot=current,
        )
        assert PersonalityTrait.OPENNESS in comparison.significant_changes

    def test_has_significant_changes(self) -> None:
        """Test significant changes detection."""
        profile = PersonalityProfile.create_for_user(uuid4())
        profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        baseline = ProfileSnapshot.from_profile(profile)
        current = ProfileSnapshot.from_profile(profile)
        comparison = ProfileComparison(
            profile_id=profile.profile_id,
            baseline_snapshot=baseline,
            current_snapshot=current,
        )
        assert comparison.has_significant_changes is False

    def test_comparison_to_dict(self) -> None:
        """Test comparison serialization."""
        profile = PersonalityProfile.create_for_user(uuid4())
        profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        snapshot = ProfileSnapshot.from_profile(profile)
        comparison = ProfileComparison(
            profile_id=profile.profile_id,
            baseline_snapshot=snapshot,
            current_snapshot=snapshot,
        )
        data = comparison.to_dict()
        assert "trait_changes" in data
        assert "significant_changes" in data
