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
        EmotionCategory.PROUD: (
            ["I can sense {intensity} pride in what you've accomplished.", "It sounds like you feel {intensity} proud of yourself.", "I hear {intensity} satisfaction in your achievement."],
            ["That's something to be really proud of!", "You deserve to feel good about this.", "What an accomplishment!"],
            ["How would you like to build on this success?", "What does this achievement mean to you?", "How can we celebrate this?"]),
        EmotionCategory.GRATEFUL: (
            ["I can sense {intensity} gratitude in what you're sharing.", "It sounds like you feel {intensity} thankful for this.", "I hear {intensity} appreciation in your words."],
            ["That's a beautiful feeling to have.", "Gratitude can be so powerful.", "It's wonderful that you recognize this."],
            ["How can we nurture this feeling?", "What else are you grateful for?", "How does this gratitude affect you?"]),
        EmotionCategory.HOPEFUL: (
            ["I can sense {intensity} hope about what's ahead.", "It sounds like you're feeling {intensity} optimistic.", "I hear {intensity} hopefulness in what you're describing."],
            ["That's a wonderful outlook!", "Hope can be incredibly sustaining.", "It's great to hear that optimism."],
            ["What would help strengthen that hope?", "How can we build on this feeling?", "What steps feel right to you?"]),
        EmotionCategory.CONTENT: (
            ["I can sense {intensity} contentment in how you're feeling.", "It sounds like you're in a {intensity} peaceful place.", "I hear {intensity} satisfaction in your words."],
            ["That sounds like a really good place to be.", "Contentment is a gift.", "I'm glad you're feeling at peace."],
            ["What helps maintain this feeling?", "How can we preserve this sense of calm?", "What matters most to you right now?"]),
        EmotionCategory.CONFIDENT: (
            ["I can sense {intensity} confidence in how you're approaching this.", "It sounds like you feel {intensity} sure of yourself.", "I hear {intensity} self-assurance in your words."],
            ["That confidence is well-earned!", "It's great to see you believe in yourself.", "Your conviction really shows."],
            ["How can we channel this confidence?", "What's your next move?", "How can I support your plans?"]),
        EmotionCategory.CARING: (
            ["I can sense {intensity} care and concern for others.", "It sounds like you feel {intensity} connected to those around you.", "I hear {intensity} compassion in what you're sharing."],
            ["Your caring nature really shows.", "It's beautiful how much you care.", "That kind of compassion matters so much."],
            ["How can you take care of yourself too?", "What would support you in caring for others?", "How can I help you with this?"]),
        EmotionCategory.FAITHFUL: (
            ["I can sense {intensity} devotion in what you're describing.", "It sounds like your faith gives you {intensity} strength.", "I hear {intensity} commitment in your words."],
            ["That kind of devotion is admirable.", "Your faith is clearly important to you.", "Commitment like that is powerful."],
            ["How does this faith sustain you?", "What role does this play in your life?", "How can I support you in this?"]),
        EmotionCategory.IMPRESSED: (
            ["I can sense you're {intensity} struck by this.", "It sounds like this has made a {intensity} impression on you.", "I hear {intensity} admiration in your words."],
            ["That does sound impressive!", "I can see why that stood out to you.", "What an inspiring thing to witness."],
            ["What does this inspire in you?", "How has this changed your perspective?", "What would you like to do with this feeling?"]),
        EmotionCategory.EXCITED: (
            ["I can sense {intensity} excitement about this!", "It sounds like you're {intensity} energized by what's happening.", "I hear {intensity} enthusiasm in everything you're saying."],
            ["That's so exciting!", "Your energy is contagious!", "How thrilling!"],
            ["How can we channel this excitement?", "What's the first thing you want to do?", "How can I help you make the most of this?"]),
        EmotionCategory.NOSTALGIC: (
            ["I can sense {intensity} nostalgia in what you're remembering.", "It sounds like these memories carry {intensity} meaning for you.", "I hear {intensity} longing for the past in your words."],
            ["Those memories sound really meaningful.", "The past can hold such treasures.", "It's natural to miss those times."],
            ["What do these memories mean to you now?", "How can we honor those feelings?", "What from the past would you like to carry forward?"]),
        EmotionCategory.LONELY: (
            ["I can sense {intensity} loneliness in what you're sharing.", "It sounds like you're feeling {intensity} isolated right now.", "I hear {intensity} disconnection in your words."],
            ["Loneliness can be so painful.", "I'm sorry you're feeling so alone.", "You don't have to go through this by yourself."],
            ["What would help you feel more connected?", "How can I be here for you right now?", "What kind of connection do you need?"]),
        EmotionCategory.EMBARRASSED: (
            ["I can sense {intensity} embarrassment about this situation.", "It sounds like this has been {intensity} uncomfortable for you.", "I hear {intensity} self-consciousness in what you're sharing."],
            ["That sounds really uncomfortable.", "It's okay to feel embarrassed sometimes.", "Everyone goes through moments like this."],
            ["How can we work through this feeling?", "What would help you feel more at ease?", "Would it help to talk about what happened?"]),
        EmotionCategory.GUILTY: (
            ["I can sense {intensity} guilt weighing on you.", "It sounds like you're carrying {intensity} responsibility for this.", "I hear {intensity} remorse in what you're sharing."],
            ["Carrying guilt can be so heavy.", "It takes courage to acknowledge these feelings.", "I hear how much this is affecting you."],
            ["What would help you process this guilt?", "How can we work through this together?", "What do you need to move forward?"]),
        EmotionCategory.ASHAMED: (
            ["I can sense {intensity} shame around this experience.", "It sounds like this has been {intensity} painful for your sense of self.", "I hear {intensity} distress about who you feel you are."],
            ["Shame can be incredibly painful.", "You are more than this moment.", "I'm here without judgment."],
            ["What would help you be gentler with yourself?", "How can we work through this together?", "What do you need to feel safe right now?"]),
        EmotionCategory.JEALOUS: (
            ["I can sense {intensity} jealousy about this situation.", "It sounds like you're feeling {intensity} envious of what others have.", "I hear {intensity} comparison in what you're describing."],
            ["Jealousy is a really human feeling.", "It's okay to want what others have.", "Those feelings can be really uncomfortable."],
            ["What does this jealousy tell you about what you want?", "How can we explore these feelings?", "What matters most to you here?"]),
        EmotionCategory.DEVASTATED: (
            ["I can sense {intensity} devastation from what happened.", "It sounds like this has been {intensity} crushing for you.", "I hear {intensity} heartbreak in your words."],
            ["I'm so sorry you're going through this.", "That sounds absolutely devastating.", "My heart goes out to you."],
            ["What do you need most right now?", "How can I support you through this?", "What would feel most comforting?"]),
        EmotionCategory.FURIOUS: (
            ["I can sense {intensity} fury about this situation.", "It sounds like you're feeling {intensity} outraged.", "I hear {intensity} intense anger in what you're describing."],
            ["That level of anger is completely understandable.", "Anyone would be furious about that.", "Your rage makes sense given what happened."],
            ["What do you need right now?", "How can we work with this anger safely?", "What would help you feel heard?"]),
        EmotionCategory.TERRIFIED: (
            ["I can sense {intensity} terror about this.", "It sounds like you're feeling {intensity} overwhelmed by fear.", "I hear {intensity} dread in what you're sharing."],
            ["That sounds absolutely terrifying.", "Fear like that can be paralyzing.", "I'm here with you through this."],
            ["What would help you feel safer right now?", "How can we address this fear together?", "What do you need to feel more grounded?"]),
        EmotionCategory.ANXIOUS: (
            ["I can sense {intensity} anxiety about what's happening.", "It sounds like you're feeling {intensity} worried and on edge.", "I hear {intensity} unease in your words."],
            ["Anxiety can be so overwhelming.", "It's exhausting to feel this way.", "I understand how consuming worry can be."],
            ["What would help ease your anxiety right now?", "How can we break this down together?", "What grounding strategies work for you?"]),
        EmotionCategory.SENTIMENTAL: (
            ["I can sense {intensity} sentimentality in what you're sharing.", "It sounds like this has {intensity} emotional significance for you.", "I hear {intensity} tenderness in your words."],
            ["Those feelings are really touching.", "It's beautiful to feel so deeply.", "Sentimentality shows how much things matter to you."],
            ["What makes this so meaningful to you?", "How would you like to honor these feelings?", "What would feel right to do with this emotion?"]),
        EmotionCategory.ANNOYED: (
            ["I can sense {intensity} annoyance about this.", "It sounds like this is {intensity} irritating for you.", "I hear {intensity} frustration building."],
            ["That does sound really annoying.", "I can see why that would get under your skin.", "It's okay to be bothered by that."],
            ["What would help resolve this annoyance?", "How can we address what's bothering you?", "What do you need right now?"]),
        EmotionCategory.DISAPPOINTED: (
            ["I can sense {intensity} disappointment about how things turned out.", "It sounds like this wasn't what you {intensity} expected.", "I hear {intensity} letdown in your words."],
            ["Disappointment can be really hard.", "It's tough when things don't go as hoped.", "I'm sorry things didn't work out."],
            ["How can we process this disappointment?", "What would help you move forward?", "What did you learn from this experience?"]),
        EmotionCategory.APPREHENSIVE: (
            ["I can sense {intensity} apprehension about what's ahead.", "It sounds like you're feeling {intensity} uneasy about the future.", "I hear {intensity} wariness in what you're describing."],
            ["It's natural to feel cautious.", "Apprehension shows you're taking this seriously.", "Uncertainty can be really uncomfortable."],
            ["What would help you feel more prepared?", "How can we address your concerns?", "What information would help ease your mind?"]),
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
        if "{intensity}" in template:
            return template.format(intensity=self._get_intensity_word(intensity, style))
        return template

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
                          "affective_focus": [affective, cognitive, compassionate], "compassionate_action": [compassionate, affective, cognitive]}
        parts = strategy_orders.get(strategy, [cognitive, affective, compassionate])
        return " ".join(p for p in parts if p)

    def _compute_confidence(self, emotion_state: EmotionStateDTO) -> float:
        base = 0.6 + (0.15 if emotion_state.confidence > 0.7 else 0) - (0.05 if emotion_state.secondary_emotion else 0)
        return min(0.95, base)

    async def shutdown(self) -> None:
        self._initialized = False
        logger.info("moel_empathy_generator_shutdown")
