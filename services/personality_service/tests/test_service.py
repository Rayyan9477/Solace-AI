"""
Unit tests for Personality Service Orchestrator.
Tests profile management, detection orchestration, and response adaptation.
"""
from __future__ import annotations
from uuid import uuid4
import pytest

from services.personality_service.src.schemas import (
    PersonalityTrait, AssessmentSource, OceanScoresDTO, StyleParametersDTO,
    DetectPersonalityRequest, GetStyleRequest, AdaptResponseRequest,
)
from services.personality_service.src.domain.service import (
    PersonalityOrchestrator, PersonalityServiceSettings,
    PersonalityProfile, ProfileStore,
)


class TestProfileStore:
    """Tests for ProfileStore."""

    @pytest.fixture
    def store(self) -> ProfileStore:
        """Create profile store."""
        return ProfileStore()

    @pytest.mark.asyncio
    async def test_save_and_get(self, store: ProfileStore) -> None:
        """Test saving and retrieving profile."""
        user_id = uuid4()
        profile = PersonalityProfile(user_id=user_id)
        profile.ocean_scores = OceanScoresDTO(
            openness=0.7, conscientiousness=0.6, extraversion=0.5,
            agreeableness=0.8, neuroticism=0.4,
        )
        saved = await store.save(profile)
        assert saved.user_id == user_id
        retrieved = store.get(user_id)
        assert retrieved is not None
        assert retrieved.user_id == user_id
        assert retrieved.ocean_scores.openness == 0.7

    def test_get_nonexistent(self, store: ProfileStore) -> None:
        """Test getting nonexistent profile."""
        result = store.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_exists(self, store: ProfileStore) -> None:
        """Test exists check."""
        user_id = uuid4()
        assert store.exists(user_id) is False
        await store.save(PersonalityProfile(user_id=user_id))
        assert store.exists(user_id) is True

    @pytest.mark.asyncio
    async def test_delete(self, store: ProfileStore) -> None:
        """Test deleting profile."""
        user_id = uuid4()
        await store.save(PersonalityProfile(user_id=user_id))
        assert store.exists(user_id) is True
        result = await store.delete(user_id)
        assert result is True
        assert store.exists(user_id) is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store: ProfileStore) -> None:
        """Test deleting nonexistent profile."""
        result = await store.delete(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_count(self, store: ProfileStore) -> None:
        """Test counting profiles."""
        assert store.count() == 0
        await store.save(PersonalityProfile(user_id=uuid4()))
        await store.save(PersonalityProfile(user_id=uuid4()))
        assert store.count() == 2


class TestPersonalityOrchestrator:
    """Tests for PersonalityOrchestrator."""

    @pytest.fixture
    def settings(self) -> PersonalityServiceSettings:
        """Create service settings."""
        return PersonalityServiceSettings(enable_llm_detection=False)

    @pytest.fixture
    def orchestrator(self, settings: PersonalityServiceSettings) -> PersonalityOrchestrator:
        """Create orchestrator without LLM."""
        return PersonalityOrchestrator(settings=settings, llm_client=None)

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test orchestrator initialization."""
        await orchestrator.initialize()
        assert orchestrator.is_initialized is True

    @pytest.mark.asyncio
    async def test_detect_personality(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test personality detection."""
        await orchestrator.initialize()
        user_id = uuid4()
        request = DetectPersonalityRequest(
            user_id=user_id,
            text="I love learning new things and exploring creative ideas. It makes me happy to discover new perspectives.",
        )
        response = await orchestrator.detect_personality(request)
        assert response.user_id == user_id
        assert response.ocean_scores is not None
        assert response.confidence > 0
        assert response.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_detect_personality_creates_profile(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test detection creates user profile."""
        await orchestrator.initialize()
        user_id = uuid4()
        request = DetectPersonalityRequest(
            user_id=user_id,
            text="I work hard to achieve my goals. Organization and planning are important to me.",
        )
        await orchestrator.detect_personality(request)
        profile = await orchestrator.get_profile(user_id)
        assert profile is not None
        assert profile.user_id == user_id
        assert profile.assessment_count == 1

    @pytest.mark.asyncio
    async def test_detect_personality_updates_profile(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test multiple detections update profile."""
        await orchestrator.initialize()
        user_id = uuid4()
        request1 = DetectPersonalityRequest(
            user_id=user_id,
            text="I enjoy spending time with friends and family. Social gatherings make me happy.",
        )
        request2 = DetectPersonalityRequest(
            user_id=user_id,
            text="I feel anxious sometimes. Uncertainty makes me worry about the future.",
        )
        await orchestrator.detect_personality(request1)
        await orchestrator.detect_personality(request2)
        profile = await orchestrator.get_profile(user_id)
        assert profile is not None
        assert profile.assessment_count == 2
        assert profile.version >= 2

    @pytest.mark.asyncio
    async def test_get_style(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test getting style parameters."""
        await orchestrator.initialize()
        user_id = uuid4()
        detect_request = DetectPersonalityRequest(
            user_id=user_id,
            text="I think deeply about complex problems. Understanding different perspectives is fascinating.",
        )
        await orchestrator.detect_personality(detect_request)
        style_request = GetStyleRequest(user_id=user_id)
        response = await orchestrator.get_style(style_request)
        assert response.user_id == user_id
        assert response.style_parameters is not None
        assert isinstance(response.recommendations, list)
        assert response.profile_confidence > 0

    @pytest.mark.asyncio
    async def test_get_style_no_profile(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test getting style without profile."""
        await orchestrator.initialize()
        user_id = uuid4()
        request = GetStyleRequest(user_id=user_id)
        response = await orchestrator.get_style(request)
        assert response.profile_confidence == 0.0
        assert "Insufficient" in response.recommendations[0]

    @pytest.mark.asyncio
    async def test_adapt_response(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test response adaptation."""
        await orchestrator.initialize()
        user_id = uuid4()
        detect_request = DetectPersonalityRequest(
            user_id=user_id,
            text="I feel worried and anxious about many things. Uncertainty is difficult for me.",
        )
        await orchestrator.detect_personality(detect_request)
        adapt_request = AdaptResponseRequest(
            user_id=user_id,
            base_response="Let's explore what's troubling you.",
            include_empathy=True,
        )
        response = await orchestrator.adapt_response(adapt_request)
        assert response.user_id == user_id
        assert len(response.adapted_content) > 0
        assert response.applied_style is not None
        assert response.empathy_components is not None

    @pytest.mark.asyncio
    async def test_adapt_response_with_style(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test response adaptation with provided style."""
        await orchestrator.initialize()
        user_id = uuid4()
        style = StyleParametersDTO(warmth=0.9, validation_level=0.8)
        adapt_request = AdaptResponseRequest(
            user_id=user_id,
            base_response="Here is some information.",
            style_parameters=style,
            include_empathy=False,
        )
        response = await orchestrator.adapt_response(adapt_request)
        assert response.applied_style.warmth == 0.9
        assert response.empathy_components is None

    @pytest.mark.asyncio
    async def test_get_profile_nonexistent(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test getting nonexistent profile."""
        await orchestrator.initialize()
        profile = await orchestrator.get_profile(uuid4())
        assert profile is None

    @pytest.mark.asyncio
    async def test_get_statistics(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test getting service statistics."""
        await orchestrator.initialize()
        status = await orchestrator.get_status()
        assert status["initialized"] is True
        assert status["statistics"]["total_requests"] == 0
        assert status["profiles_count"] == 0

    @pytest.mark.asyncio
    async def test_statistics_update(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test statistics update after operations."""
        await orchestrator.initialize()
        user_id = uuid4()
        request = DetectPersonalityRequest(
            user_id=user_id,
            text="I enjoy creative activities and exploring new ideas. Art and music inspire me.",
        )
        await orchestrator.detect_personality(request)
        status = await orchestrator.get_status()
        assert status["statistics"]["total_requests"] == 1
        assert status["statistics"]["total_detections"] == 1
        assert status["profiles_count"] == 1

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator: PersonalityOrchestrator) -> None:
        """Test orchestrator shutdown."""
        await orchestrator.initialize()
        await orchestrator.shutdown()
        assert orchestrator.is_initialized is False


class TestPersonalityServiceSettings:
    """Tests for PersonalityServiceSettings."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = PersonalityServiceSettings()
        assert settings.enable_profile_caching is True
        assert settings.profile_cache_ttl_seconds == 600
        assert settings.enable_llm_detection is True
        assert settings.profile_update_threshold == 0.15

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = PersonalityServiceSettings(
            enable_llm_detection=False,
            profile_cache_ttl_seconds=300,
        )
        assert settings.enable_llm_detection is False
        assert settings.profile_cache_ttl_seconds == 300


class TestPersonalityProfile:
    """Tests for PersonalityProfile dataclass."""

    def test_create_profile(self) -> None:
        """Test creating profile."""
        user_id = uuid4()
        profile = PersonalityProfile(user_id=user_id)
        assert profile.user_id == user_id
        assert profile.assessment_count == 0
        assert profile.version == 1
        assert profile.ocean_scores is None

    def test_profile_with_scores(self) -> None:
        """Test profile with OCEAN scores."""
        user_id = uuid4()
        scores = OceanScoresDTO(
            openness=0.8, conscientiousness=0.6, extraversion=0.4,
            agreeableness=0.7, neuroticism=0.3,
        )
        profile = PersonalityProfile(user_id=user_id, ocean_scores=scores)
        assert profile.ocean_scores is not None
        assert profile.ocean_scores.openness == 0.8
