"""Tests for MoEL empathy generation."""
from __future__ import annotations
import pytest

from services.personality_service.src.schemas import (
    EmotionCategory, EmotionStateDTO, EmpathyComponentsDTO, StyleParametersDTO,
    CommunicationStyleType,
)
from services.personality_service.src.ml.empathy import (
    MoELSettings, MoELEmpathyGenerator, MoELOutput, EmotionIntensity,
    EmotionDistributionComputer, ListenerBank, SoftCombiner, StrategySelector,
    EmotionListener,
)


def create_emotion_state(
    primary: EmotionCategory = EmotionCategory.NEUTRAL,
    secondary: EmotionCategory | None = None,
    intensity: float = 0.5,
    valence: float = 0.0,
    confidence: float = 0.7,
) -> EmotionStateDTO:
    """Helper to create EmotionStateDTO."""
    return EmotionStateDTO(
        primary_emotion=primary,
        secondary_emotion=secondary,
        intensity=intensity,
        valence=valence,
        confidence=confidence,
    )


def create_style_params(
    warmth: float = 0.5,
    structure: float = 0.5,
    validation_level: float = 0.5,
) -> StyleParametersDTO:
    """Helper to create StyleParametersDTO."""
    return StyleParametersDTO(
        warmth=warmth,
        structure=structure,
        complexity=0.5,
        directness=0.5,
        energy=0.5,
        validation_level=validation_level,
        style_type=CommunicationStyleType.BALANCED,
    )


class TestMoELSettings:
    """Tests for MoELSettings."""

    def test_default_settings(self) -> None:
        settings = MoELSettings()
        assert settings.num_listeners == 32
        assert settings.attention_heads == 8
        assert settings.enable_soft_combination is True
        assert settings.cognitive_weight == 0.33
        assert settings.affective_weight == 0.34

    def test_custom_settings(self) -> None:
        settings = MoELSettings(
            num_listeners=16,
            temperature=0.5,
        )
        assert settings.num_listeners == 16
        assert settings.temperature == 0.5


class TestEmotionDistributionComputer:
    """Tests for EmotionDistributionComputer."""

    def test_compute_single_emotion(self) -> None:
        computer = EmotionDistributionComputer()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            intensity=0.8,
        )
        distribution = computer.compute(emotion_state)
        assert distribution[EmotionCategory.SADNESS] > 0.5
        assert sum(distribution.values()) == pytest.approx(1.0, rel=0.01)

    def test_compute_with_secondary_emotion(self) -> None:
        computer = EmotionDistributionComputer()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            secondary=EmotionCategory.FEAR,
            intensity=0.6,
        )
        distribution = computer.compute(emotion_state)
        assert distribution[EmotionCategory.SADNESS] > 0
        assert distribution[EmotionCategory.FEAR] > 0
        assert sum(distribution.values()) == pytest.approx(1.0, rel=0.01)

    def test_get_intensity_level_mild(self) -> None:
        computer = EmotionDistributionComputer()
        level = computer.get_intensity_level(0.2)
        assert level == EmotionIntensity.MILD

    def test_get_intensity_level_moderate(self) -> None:
        computer = EmotionDistributionComputer()
        level = computer.get_intensity_level(0.4)
        assert level == EmotionIntensity.MODERATE

    def test_get_intensity_level_strong(self) -> None:
        computer = EmotionDistributionComputer()
        level = computer.get_intensity_level(0.6)
        assert level == EmotionIntensity.STRONG

    def test_get_intensity_level_intense(self) -> None:
        computer = EmotionDistributionComputer()
        level = computer.get_intensity_level(0.9)
        assert level == EmotionIntensity.INTENSE


class TestListenerBank:
    """Tests for ListenerBank."""

    def test_get_all_listeners(self) -> None:
        bank = ListenerBank()
        listeners = bank.get_all_listeners()
        assert len(listeners) == len(EmotionCategory)

    def test_get_listener_for_emotion(self) -> None:
        bank = ListenerBank()
        listener = bank.get_listener(EmotionCategory.SADNESS)
        assert listener.emotion == EmotionCategory.SADNESS
        assert len(listener.cognitive_templates) > 0
        assert len(listener.affective_templates) > 0
        assert len(listener.compassionate_templates) > 0

    def test_listener_has_templates(self) -> None:
        bank = ListenerBank()
        for emotion in EmotionCategory:
            listener = bank.get_listener(emotion)
            assert isinstance(listener, EmotionListener)
            assert listener.emotion == emotion


class TestSoftCombiner:
    """Tests for SoftCombiner."""

    def test_combine_responses(self) -> None:
        settings = MoELSettings()
        combiner = SoftCombiner(settings)
        bank = ListenerBank()
        weights = {EmotionCategory.SADNESS: 0.8, EmotionCategory.NEUTRAL: 0.2}
        style = create_style_params()
        cog, aff, comp = combiner.combine_responses(
            bank, weights, EmotionIntensity.MODERATE, style
        )
        assert len(cog) > 0
        assert len(aff) > 0
        assert len(comp) > 0

    def test_combine_responses_high_warmth(self) -> None:
        settings = MoELSettings()
        combiner = SoftCombiner(settings)
        bank = ListenerBank()
        weights = {EmotionCategory.JOY: 1.0}
        style = create_style_params(warmth=0.9)
        cog, aff, comp = combiner.combine_responses(
            bank, weights, EmotionIntensity.STRONG, style
        )
        assert len(cog) > 0


class TestStrategySelector:
    """Tests for StrategySelector."""

    def test_select_validation_first_high_intensity(self) -> None:
        selector = StrategySelector()
        emotion_state = create_emotion_state(intensity=0.85)
        style = create_style_params()
        strategy = selector.select(emotion_state, style)
        assert strategy == "validation_first"

    def test_select_affective_focus_negative_valence(self) -> None:
        selector = StrategySelector()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            intensity=0.5,
            valence=-0.7,
        )
        style = create_style_params()
        strategy = selector.select(emotion_state, style)
        assert strategy == "affective_focus"

    def test_select_cognitive_focus_high_structure(self) -> None:
        selector = StrategySelector()
        emotion_state = create_emotion_state(intensity=0.5, valence=0.0)
        style = create_style_params(structure=0.8)
        strategy = selector.select(emotion_state, style)
        assert strategy == "cognitive_focus"

    def test_select_compassionate_action_low_intensity(self) -> None:
        selector = StrategySelector()
        emotion_state = create_emotion_state(intensity=0.2)
        style = create_style_params(structure=0.5)
        strategy = selector.select(emotion_state, style)
        assert strategy == "compassionate_action"

    def test_select_balanced_default(self) -> None:
        selector = StrategySelector()
        emotion_state = create_emotion_state(intensity=0.5, valence=0.0)
        style = create_style_params(structure=0.5)
        strategy = selector.select(emotion_state, style)
        assert strategy == "balanced"


class TestMoELEmpathyGenerator:
    """Tests for MoELEmpathyGenerator."""

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        assert generator._initialized is True

    @pytest.mark.asyncio
    async def test_generate_returns_components(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            intensity=0.7,
        )
        style = create_style_params()
        result = await generator.generate(emotion_state, style)
        assert isinstance(result, EmpathyComponentsDTO)
        assert len(result.cognitive_content) > 0
        assert len(result.affective_content) > 0
        assert len(result.compassionate_content) > 0

    @pytest.mark.asyncio
    async def test_generate_sadness_response(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            intensity=0.8,
        )
        style = create_style_params(warmth=0.7)
        result = await generator.generate(emotion_state, style)
        # Sadness responses may contain various keywords: sad, loss, difficult, sorry, etc.
        cognitive_lower = result.cognitive_content.lower()
        affective_lower = result.affective_content.lower()
        assert any(kw in cognitive_lower or kw in affective_lower for kw in ["sad", "loss", "difficult", "sorry"])

    @pytest.mark.asyncio
    async def test_generate_anger_response(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.ANGER,
            intensity=0.7,
        )
        style = create_style_params()
        result = await generator.generate(emotion_state, style)
        assert len(result.cognitive_content) > 0

    @pytest.mark.asyncio
    async def test_generate_fear_response(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.FEAR,
            intensity=0.6,
        )
        style = create_style_params()
        result = await generator.generate(emotion_state, style)
        assert len(result.cognitive_content) > 0

    @pytest.mark.asyncio
    async def test_generate_joy_response(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.JOY,
            intensity=0.8,
            valence=0.7,
        )
        style = create_style_params()
        result = await generator.generate(emotion_state, style)
        assert len(result.cognitive_content) > 0

    @pytest.mark.asyncio
    async def test_generate_neutral_response(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.NEUTRAL,
            intensity=0.3,
        )
        style = create_style_params()
        result = await generator.generate(emotion_state, style)
        assert len(result.cognitive_content) > 0

    @pytest.mark.asyncio
    async def test_generate_with_high_warmth(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            intensity=0.6,
        )
        style = create_style_params(warmth=0.9)
        result = await generator.generate(emotion_state, style)
        assert "I'm here" in result.affective_content or len(result.affective_content) > 0

    @pytest.mark.asyncio
    async def test_generate_with_high_validation(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.FEAR,
            intensity=0.5,
        )
        style = create_style_params(validation_level=0.9)
        result = await generator.generate(emotion_state, style)
        assert len(result.cognitive_content) > 0

    @pytest.mark.asyncio
    async def test_generate_with_weights(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            secondary=EmotionCategory.FEAR,
            intensity=0.7,
        )
        style = create_style_params()
        components, output = await generator.generate_with_weights(emotion_state, style)
        assert isinstance(components, EmpathyComponentsDTO)
        assert isinstance(output, MoELOutput)
        assert output.emotion_weights[EmotionCategory.SADNESS] > 0
        assert len(output.combined_response) > 0

    @pytest.mark.asyncio
    async def test_generate_strategy_selection(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.ANGER,
            intensity=0.9,
        )
        style = create_style_params()
        result = await generator.generate(emotion_state, style)
        assert result.selected_strategy == "validation_first"

    @pytest.mark.asyncio
    async def test_generate_without_soft_combination(self) -> None:
        settings = MoELSettings(enable_soft_combination=False)
        generator = MoELEmpathyGenerator(settings=settings)
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            intensity=0.6,
        )
        style = create_style_params()
        result = await generator.generate(emotion_state, style)
        assert len(result.cognitive_content) > 0

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        await generator.shutdown()
        assert generator._initialized is False

    @pytest.mark.asyncio
    async def test_generate_all_emotions(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        style = create_style_params()
        for emotion in EmotionCategory:
            emotion_state = create_emotion_state(primary=emotion, intensity=0.6)
            result = await generator.generate(emotion_state, style)
            assert isinstance(result, EmpathyComponentsDTO)
            assert len(result.selected_strategy) > 0

    @pytest.mark.asyncio
    async def test_generate_confidence(self) -> None:
        generator = MoELEmpathyGenerator()
        await generator.initialize()
        emotion_state = create_emotion_state(
            primary=EmotionCategory.SADNESS,
            intensity=0.7,
            confidence=0.9,
        )
        style = create_style_params()
        _, output = await generator.generate_with_weights(emotion_state, style)
        assert output.confidence > 0.5
