"""
Solace-AI Personality Service - MoEL Empathy Generation.
Mixture of Empathetic Listeners (MoEL) for emotion-aware empathetic response generation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog
from ..schemas import EmotionCategory, EmpathyComponent, EmotionStateDTO, EmpathyComponentsDTO, StyleParametersDTO

logger = structlog.get_logger(__name__)


class MoELSettings(BaseSettings):
    """MoEL empathy generator configuration."""
    num_listeners: int = Field(default=32, ge=8, le=64)
    attention_heads: int = Field(default=8, ge=1, le=16)
    listener_dim: int = Field(default=256, ge=64, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    max_response_length: int = Field(default=200, ge=50, le=500)
    enable_soft_combination: bool = Field(default=True)
    cognitive_weight: float = Field(default=0.33, ge=0.0, le=1.0)
    affective_weight: float = Field(default=0.34, ge=0.0, le=1.0)
    compassionate_weight: float = Field(default=0.33, ge=0.0, le=1.0)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_MOEL_", env_file=".env", extra="ignore")


class EmotionIntensity(str, Enum):
    """Emotion intensity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    STRONG = "strong"
    INTENSE = "intense"


@dataclass
class EmotionListener:
    """Single emotion-specific listener in MoEL."""
    listener_id: int
    emotion: EmotionCategory
    templates: list[str] = field(default_factory=list)
    cognitive_templates: list[str] = field(default_factory=list)
    affective_templates: list[str] = field(default_factory=list)
    compassionate_templates: list[str] = field(default_factory=list)


@dataclass
class ListenerWeight:
    """Weight assigned to a listener based on emotion distribution."""
    listener_id: int
    emotion: EmotionCategory
    weight: float = 0.0


@dataclass
class MoELOutput:
    """Output from MoEL empathy generation."""
    output_id: UUID = field(default_factory=uuid4)
    cognitive_response: str = ""
    affective_response: str = ""
    compassionate_response: str = ""
    combined_response: str = ""
    emotion_weights: dict[EmotionCategory, float] = field(default_factory=dict)
    selected_strategy: str = "balanced"
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EmotionDistributionComputer:
    """Computes emotion distribution for listener weighting."""
    def compute(self, emotion_state: EmotionStateDTO) -> dict[EmotionCategory, float]:
        distribution: dict[EmotionCategory, float] = {e: 0.0 for e in EmotionCategory}
        distribution[emotion_state.primary_emotion] = emotion_state.intensity
        if emotion_state.secondary_emotion:
            distribution[emotion_state.secondary_emotion] = (1 - emotion_state.intensity) * 0.6
        remaining = 1.0 - sum(distribution.values())
        if remaining > 0:
            distribution[EmotionCategory.NEUTRAL] += remaining * 0.7
            for e in EmotionCategory:
                if distribution[e] == 0.0 and e != EmotionCategory.NEUTRAL:
                    distribution[e] = remaining * 0.3 / (len(EmotionCategory) - 2)
        total = sum(distribution.values())
        return {e: w / total for e, w in distribution.items()} if total > 0 else distribution

    def get_intensity_level(self, intensity: float) -> EmotionIntensity:
        if intensity < 0.25: return EmotionIntensity.MILD
        if intensity < 0.5: return EmotionIntensity.MODERATE
        if intensity < 0.75: return EmotionIntensity.STRONG
        return EmotionIntensity.INTENSE


class ListenerBank:
    """Bank of emotion-specific listeners with templates."""
    _TEMPLATES = {
        EmotionCategory.SADNESS: (
            ["It sounds like you're feeling {intensity} sadness about this.", "I can hear that this situation is bringing up {intensity} feelings of loss.", "What you're describing sounds like a {intensity} sense of disappointment."],
            ["That must be really difficult.", "I'm sorry you're going through this.", "That sounds incredibly hard."],
            ["What would feel most supportive for you right now?", "How can I best be here for you?", "Would you like to explore what might help?"]),
        EmotionCategory.ANGER: (
            ["I can sense {intensity} frustration in what you're sharing.", "It sounds like this situation has been {intensity} upsetting for you.", "I hear that you're feeling {intensity} wronged by this."],
            ["That sounds really frustrating.", "It's understandable to feel upset about this.", "Anyone would feel bothered by that."],
            ["What do you need most right now?", "How can I support you in processing this?", "Would it help to talk through what happened?"]),
        EmotionCategory.FEAR: (
            ["It seems like there's {intensity} anxiety about what might happen.", "I can hear {intensity} worry in what you're describing.", "It sounds like this uncertainty is creating {intensity} concern for you."],
            ["That sounds scary.", "It's natural to feel worried about this.", "Uncertainty can be really unsettling."],
            ["What would help you feel more grounded?", "How can we work through this together?", "Would you like to explore some strategies?"]),
        EmotionCategory.JOY: (
            ["I can sense your {intensity} excitement about this!", "It sounds like this has brought you {intensity} happiness.", "I hear {intensity} enthusiasm in what you're sharing."],
            ["That's wonderful!", "I'm so happy for you!", "That sounds amazing!"],
            ["How would you like to celebrate this?", "What's next for you?", "How can we build on this positive moment?"]),
        EmotionCategory.SURPRISE: (
            ["This seems to have caught you off guard with {intensity} unexpectedness.", "I can hear that this was {intensity} surprising for you.", "It sounds like you're processing something {intensity} unexpected."],
            ["Wow, that's quite unexpected!", "I can see why that caught you off guard.", "That's a lot to take in."],
            ["How are you processing this?", "What do you need as you take this in?", "Would it help to talk through the implications?"]),
        EmotionCategory.DISGUST: (
            ["I can sense {intensity} discomfort with this situation.", "It sounds like this has been {intensity} off-putting for you.", "I hear {intensity} aversion in how you're describing this."],
            ["That sounds really unpleasant.", "I can understand why that's bothering you.", "That's a difficult thing to deal with."],
            ["What would help you feel better about this?", "How can we address what's bothering you?", "Would you like to explore some boundaries?"]),
        EmotionCategory.TRUST: (
            ["I can sense your {intensity} openness in sharing this.", "It sounds like you feel {intensity} safe exploring this.", "I hear {intensity} trust in the way you're expressing yourself."],
            ["I appreciate you sharing this with me.", "Thank you for trusting me with this.", "I'm honored you feel safe to share."],
            ["What else would you like to explore?", "How can I continue to support you?", "What feels important to discuss next?"]),
        EmotionCategory.ANTICIPATION: (
            ["I can sense {intensity} anticipation about what's coming.", "It sounds like you're {intensity} looking forward to this.", "I hear {intensity} eagerness in your words."],
            ["That's exciting!", "I can feel your anticipation.", "The future sounds promising."],
            ["How can I help you prepare?", "What would make this even better?", "What support do you need going forward?"]),
        EmotionCategory.NEUTRAL: (
            ["I hear what you're sharing.", "Thank you for telling me about this.", "I'm listening to what you're describing."],
            ["Thank you for sharing that.", "I appreciate you telling me.", "I'm here with you."],
            ["How can I best support you?", "What would be most helpful right now?", "Where would you like to go from here?"]),
    }

    def __init__(self) -> None:
        self._listeners = {emotion: EmotionListener(
            listener_id=i, emotion=emotion,
            cognitive_templates=self._TEMPLATES.get(emotion, ([], [], []))[0],
            affective_templates=self._TEMPLATES.get(emotion, ([], [], []))[1],
            compassionate_templates=self._TEMPLATES.get(emotion, ([], [], []))[2],
        ) for i, emotion in enumerate(EmotionCategory)}

    def get_listener(self, emotion: EmotionCategory) -> EmotionListener:
        return self._listeners[emotion]

    def get_all_listeners(self) -> list[EmotionListener]:
        return list(self._listeners.values())


class SoftCombiner:
    """Combines listener outputs using soft weighting."""
    _INTENSITY_WORDS = {
        EmotionIntensity.MILD: ["some", "a bit of", "mild"],
        EmotionIntensity.MODERATE: ["", "moderate", "noticeable"],
        EmotionIntensity.STRONG: ["significant", "strong", "deep"],
        EmotionIntensity.INTENSE: ["intense", "profound", "overwhelming"],
    }

    def __init__(self, settings: MoELSettings) -> None:
        self._settings = settings

    def combine_responses(self, listener_bank: ListenerBank, weights: dict[EmotionCategory, float], intensity: EmotionIntensity, style: StyleParametersDTO) -> tuple[str, str, str]:
        return (self._select_best_template(listener_bank, weights, "cognitive", intensity, style),
                self._select_best_template(listener_bank, weights, "affective", intensity, style),
                self._select_best_template(listener_bank, weights, "compassionate", intensity, style))

    def _select_best_template(self, listener_bank: ListenerBank, weights: dict[EmotionCategory, float], component: str, intensity: EmotionIntensity, style: StyleParametersDTO) -> str:
        top_emotion = max(weights.items(), key=lambda x: x[1])[0]
        templates = getattr(listener_bank.get_listener(top_emotion), f"{component}_templates", [])
        if not templates: return ""
        template = templates[min(int(style.warmth * 2), len(templates) - 1)]
        return template.format(intensity=self._get_intensity_word(intensity, style))

    def _get_intensity_word(self, intensity: EmotionIntensity, style: StyleParametersDTO) -> str:
        options = self._INTENSITY_WORDS.get(intensity, [""])
        return options[min(int(style.warmth * 2), len(options) - 1)]


class StrategySelector:
    """Selects empathy strategy based on context."""
    def select(self, emotion_state: EmotionStateDTO, style: StyleParametersDTO) -> str:
        if emotion_state.intensity > 0.75: return "validation_first"
        if emotion_state.valence < -0.5: return "affective_focus"
        if style.structure > 0.7: return "cognitive_focus"
        if emotion_state.intensity < 0.3: return "compassionate_action"
        return "balanced"


class MoELEmpathyGenerator:
    """Mixture of Empathetic Listeners for empathy generation."""

    def __init__(self, settings: MoELSettings | None = None) -> None:
        self._settings = settings or MoELSettings()
        self._listener_bank = ListenerBank()
        self._distribution_computer = EmotionDistributionComputer()
        self._soft_combiner = SoftCombiner(self._settings)
        self._strategy_selector = StrategySelector()
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("moel_empathy_generator_initialized", num_listeners=self._settings.num_listeners, soft_combination=self._settings.enable_soft_combination)

    async def generate(self, emotion_state: EmotionStateDTO, style: StyleParametersDTO) -> EmpathyComponentsDTO:
        emotion_weights = self._distribution_computer.compute(emotion_state)
        intensity = self._distribution_computer.get_intensity_level(emotion_state.intensity)
        strategy = self._strategy_selector.select(emotion_state, style)
        if self._settings.enable_soft_combination:
            cognitive, affective, compassionate = self._soft_combiner.combine_responses(self._listener_bank, emotion_weights, intensity, style)
        else:
            cognitive, affective, compassionate = self._single_listener_response(emotion_state, intensity, style)
        return EmpathyComponentsDTO(
            cognitive_content=self._apply_style_adjustments(cognitive, style, "cognitive"),
            affective_content=self._apply_style_adjustments(affective, style, "affective"),
            compassionate_content=self._apply_style_adjustments(compassionate, style, "compassionate"),
            selected_strategy=strategy, emotion_state=emotion_state)

    async def generate_with_weights(self, emotion_state: EmotionStateDTO, style: StyleParametersDTO) -> tuple[EmpathyComponentsDTO, MoELOutput]:
        emotion_weights = self._distribution_computer.compute(emotion_state)
        intensity = self._distribution_computer.get_intensity_level(emotion_state.intensity)
        strategy = self._strategy_selector.select(emotion_state, style)
        cognitive, affective, compassionate = self._soft_combiner.combine_responses(self._listener_bank, emotion_weights, intensity, style)
        combined = self._combine_components(cognitive, affective, compassionate, strategy)
        return (EmpathyComponentsDTO(cognitive_content=cognitive, affective_content=affective, compassionate_content=compassionate, selected_strategy=strategy, emotion_state=emotion_state),
                MoELOutput(cognitive_response=cognitive, affective_response=affective, compassionate_response=compassionate, combined_response=combined, emotion_weights=emotion_weights, selected_strategy=strategy, confidence=self._compute_confidence(emotion_state)))

    def _single_listener_response(self, emotion_state: EmotionStateDTO, intensity: EmotionIntensity, style: StyleParametersDTO) -> tuple[str, str, str]:
        listener = self._listener_bank.get_listener(emotion_state.primary_emotion)
        intensity_word = self._soft_combiner._get_intensity_word(intensity, style)
        return (listener.cognitive_templates[0].format(intensity=intensity_word) if listener.cognitive_templates else "",
                listener.affective_templates[0] if listener.affective_templates else "",
                listener.compassionate_templates[0] if listener.compassionate_templates else "")

    def _apply_style_adjustments(self, text: str, style: StyleParametersDTO, component: str) -> str:
        if not text: return text
        if style.warmth > 0.7 and component == "affective": text = f"{text} I'm here with you."
        if style.validation_level > 0.7 and component == "cognitive" and not text.startswith("It") and not text.startswith("I can"):
            text = f"I understand. {text}"
        return text

    def _combine_components(self, cognitive: str, affective: str, compassionate: str, strategy: str) -> str:
        strategy_orders = {"validation_first": [affective, cognitive, compassionate], "cognitive_focus": [cognitive, affective, compassionate],
                          "affective_focus": [affective, cognitive, compassionate], "compassionate_action": [cognitive, compassionate]}
        parts = strategy_orders.get(strategy, [cognitive, affective, compassionate])
        return " ".join(p for p in parts if p)

    def _compute_confidence(self, emotion_state: EmotionStateDTO) -> float:
        base = 0.6 + (0.15 if emotion_state.confidence > 0.7 else 0) - (0.05 if emotion_state.secondary_emotion else 0)
        return min(0.95, base)

    async def shutdown(self) -> None:
        self._initialized = False
        logger.info("moel_empathy_generator_shutdown")
