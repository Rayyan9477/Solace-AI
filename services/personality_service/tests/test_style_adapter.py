"""
Unit tests for Style Adapter.
Tests personality-to-style mapping and empathy generation.
"""
from __future__ import annotations
import pytest

from services.personality_service.src.schemas import (
    PersonalityTrait, CommunicationStyleType, EmotionCategory,
    OceanScoresDTO, StyleParametersDTO, EmotionStateDTO,
)
from services.personality_service.src.domain.style_adapter import (
    StyleAdapter, StyleAdapterSettings, StyleMapper,
    RecommendationGenerator, EmpathyAdapter,
)


class TestStyleMapper:
    """Tests for StyleMapper."""

    @pytest.fixture
    def mapper(self) -> StyleMapper:
        """Create style mapper."""
        return StyleMapper()

    def test_map_default_scores(self, mapper: StyleMapper) -> None:
        """Test mapping default OCEAN scores."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.style_type == CommunicationStyleType.BALANCED
        assert 0.3 <= style.warmth <= 0.7
        assert 0.3 <= style.structure <= 0.7

    def test_map_high_neuroticism(self, mapper: StyleMapper) -> None:
        """Test mapping high neuroticism increases warmth."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.85,
        )
        style = mapper.map_to_style(scores)
        assert style.warmth > 0.6
        assert style.validation_level > 0.6

    def test_map_high_conscientiousness(self, mapper: StyleMapper) -> None:
        """Test mapping high conscientiousness increases structure."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.85,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.structure > 0.6

    def test_map_high_openness(self, mapper: StyleMapper) -> None:
        """Test mapping high openness increases complexity."""
        scores = OceanScoresDTO(
            openness=0.85,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.complexity > 0.6

    def test_map_high_extraversion(self, mapper: StyleMapper) -> None:
        """Test mapping high extraversion increases energy."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.85,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.energy > 0.5

    def test_map_low_extraversion(self, mapper: StyleMapper) -> None:
        """Test mapping low extraversion decreases energy."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.2,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.energy < 0.5

    def test_determine_analytical_style(self, mapper: StyleMapper) -> None:
        """Test determining analytical style type."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.85,
            extraversion=0.3,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.style_type == CommunicationStyleType.ANALYTICAL

    def test_determine_expressive_style(self, mapper: StyleMapper) -> None:
        """Test determining expressive style type."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.85,
            agreeableness=0.85,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.style_type == CommunicationStyleType.EXPRESSIVE

    def test_determine_driver_style(self, mapper: StyleMapper) -> None:
        """Test determining driver style type."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.85,
            agreeableness=0.3,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.style_type == CommunicationStyleType.DRIVER

    def test_determine_amiable_style(self, mapper: StyleMapper) -> None:
        """Test determining amiable style type."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.2,
            agreeableness=0.85,
            neuroticism=0.5,
        )
        style = mapper.map_to_style(scores)
        assert style.style_type == CommunicationStyleType.AMIABLE


class TestRecommendationGenerator:
    """Tests for RecommendationGenerator."""

    @pytest.fixture
    def generator(self) -> RecommendationGenerator:
        """Create recommendation generator."""
        return RecommendationGenerator()

    def test_generate_high_neuroticism_recommendations(self, generator: RecommendationGenerator) -> None:
        """Test recommendations for high neuroticism."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.85,
        )
        recommendations = generator.generate(scores)
        assert any("reassurance" in r.lower() or "validation" in r.lower() for r in recommendations)

    def test_generate_low_extraversion_recommendations(self, generator: RecommendationGenerator) -> None:
        """Test recommendations for low extraversion."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.2,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        recommendations = generator.generate(scores)
        assert any("concise" in r.lower() for r in recommendations)

    def test_generate_high_openness_recommendations(self, generator: RecommendationGenerator) -> None:
        """Test recommendations for high openness."""
        scores = OceanScoresDTO(
            openness=0.85,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        recommendations = generator.generate(scores)
        assert any("metaphor" in r.lower() or "abstract" in r.lower() for r in recommendations)

    def test_generate_high_conscientiousness_recommendations(self, generator: RecommendationGenerator) -> None:
        """Test recommendations for high conscientiousness."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.85,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        recommendations = generator.generate(scores)
        assert any("structure" in r.lower() or "timeline" in r.lower() for r in recommendations)

    def test_generate_balanced_scores(self, generator: RecommendationGenerator) -> None:
        """Test few recommendations for balanced scores."""
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        recommendations = generator.generate(scores)
        assert len(recommendations) <= 5


class TestEmpathyAdapter:
    """Tests for EmpathyAdapter."""

    @pytest.fixture
    def adapter(self) -> EmpathyAdapter:
        """Create empathy adapter."""
        return EmpathyAdapter()

    def test_generate_components_sadness(self, adapter: EmpathyAdapter) -> None:
        """Test empathy components for sadness."""
        emotion = EmotionStateDTO(
            primary_emotion=EmotionCategory.SADNESS,
            intensity=0.7,
            valence=-0.5,
        )
        style = StyleParametersDTO(warmth=0.8)
        components = adapter.generate_components(emotion, style)
        assert "sadness" in components.cognitive_content.lower() or "feeling" in components.cognitive_content.lower()
        assert components.selected_strategy in ["validation_first", "balanced", "cognitive_focus", "compassionate_action"]

    def test_generate_components_anger(self, adapter: EmpathyAdapter) -> None:
        """Test empathy components for anger."""
        emotion = EmotionStateDTO(
            primary_emotion=EmotionCategory.ANGER,
            intensity=0.6,
            valence=-0.4,
        )
        style = StyleParametersDTO(warmth=0.6)
        components = adapter.generate_components(emotion, style)
        assert "frustrated" in components.cognitive_content.lower()

    def test_generate_components_joy(self, adapter: EmpathyAdapter) -> None:
        """Test empathy components for joy."""
        emotion = EmotionStateDTO(
            primary_emotion=EmotionCategory.JOY,
            intensity=0.8,
            valence=0.7,
        )
        style = StyleParametersDTO(warmth=0.7)
        components = adapter.generate_components(emotion, style)
        assert "wonderful" in components.affective_content.lower() or "happiness" in components.cognitive_content.lower()

    def test_strategy_high_intensity(self, adapter: EmpathyAdapter) -> None:
        """Test validation_first strategy for high intensity."""
        emotion = EmotionStateDTO(
            primary_emotion=EmotionCategory.SADNESS,
            intensity=0.85,
            valence=-0.6,
        )
        style = StyleParametersDTO()
        components = adapter.generate_components(emotion, style)
        assert components.selected_strategy == "validation_first"

    def test_strategy_low_intensity(self, adapter: EmpathyAdapter) -> None:
        """Test compassionate_action strategy for low intensity."""
        emotion = EmotionStateDTO(
            primary_emotion=EmotionCategory.NEUTRAL,
            intensity=0.2,
            valence=0.0,
        )
        style = StyleParametersDTO()
        components = adapter.generate_components(emotion, style)
        assert components.selected_strategy == "compassionate_action"


class TestStyleAdapter:
    """Tests for StyleAdapter orchestrator."""

    @pytest.fixture
    def adapter(self) -> StyleAdapter:
        """Create style adapter."""
        return StyleAdapter()

    @pytest.mark.asyncio
    async def test_initialize(self, adapter: StyleAdapter) -> None:
        """Test adapter initialization."""
        await adapter.initialize()
        assert adapter._initialized is True

    def test_get_style_parameters(self, adapter: StyleAdapter) -> None:
        """Test getting style parameters."""
        scores = OceanScoresDTO(
            openness=0.7,
            conscientiousness=0.6,
            extraversion=0.5,
            agreeableness=0.8,
            neuroticism=0.4,
        )
        style = adapter.get_style_parameters(scores)
        assert isinstance(style, StyleParametersDTO)
        assert 0.0 <= style.warmth <= 1.0

    def test_get_recommendations(self, adapter: StyleAdapter) -> None:
        """Test getting recommendations."""
        scores = OceanScoresDTO(
            openness=0.85,
            conscientiousness=0.85,
            extraversion=0.2,
            agreeableness=0.5,
            neuroticism=0.85,
        )
        recommendations = adapter.get_recommendations(scores)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_get_empathy_components(self, adapter: StyleAdapter) -> None:
        """Test getting empathy components."""
        emotion = EmotionStateDTO(
            primary_emotion=EmotionCategory.FEAR,
            intensity=0.6,
            valence=-0.3,
        )
        style = StyleParametersDTO(warmth=0.8, validation_level=0.7)
        components = adapter.get_empathy_components(emotion, style)
        assert isinstance(components.cognitive_content, str)
        assert isinstance(components.affective_content, str)

    def test_adapt_response_high_warmth(self, adapter: StyleAdapter) -> None:
        """Test response adaptation with high warmth."""
        style = StyleParametersDTO(warmth=0.85, validation_level=0.8)
        base = "Let's explore this together."
        adapted = adapter.adapt_response(base, style)
        assert "I hear you" in adapted or "That makes sense" in adapted

    def test_adapt_response_high_structure(self, adapter: StyleAdapter) -> None:
        """Test response adaptation with high structure."""
        style = StyleParametersDTO(structure=0.85)
        base = "First we should carefully consider this important matter. Then we can explore that option thoroughly. After that we can decide on the best approach. Finally we implement the solution step by step. This requires attention to detail."
        adapted = adapter.adapt_response(base, style)
        assert "\n" in adapted or len(adapted) >= len(base)

    @pytest.mark.asyncio
    async def test_shutdown(self, adapter: StyleAdapter) -> None:
        """Test adapter shutdown."""
        await adapter.initialize()
        await adapter.shutdown()
        assert adapter._initialized is False


class TestStyleAdapterSettings:
    """Tests for StyleAdapterSettings."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = StyleAdapterSettings()
        assert settings.high_trait_threshold == 0.7
        assert settings.low_trait_threshold == 0.3
        assert settings.default_warmth == 0.6
        assert settings.default_validation == 0.5

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = StyleAdapterSettings(
            high_trait_threshold=0.75,
            warmth_neuroticism_boost=0.3,
        )
        assert settings.high_trait_threshold == 0.75
        assert settings.warmth_neuroticism_boost == 0.3
