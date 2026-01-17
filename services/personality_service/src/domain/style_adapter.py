"""
Solace-AI Personality Service - Communication Style Adapter.
Maps Big Five personality traits to communication style parameters for response adaptation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import (
    PersonalityTrait, CommunicationStyleType, EmotionCategory, EmpathyComponent,
    OceanScoresDTO, StyleParametersDTO, EmotionStateDTO, EmpathyComponentsDTO,
)

logger = structlog.get_logger(__name__)


class StyleAdapterSettings(BaseSettings):
    """Style adapter configuration."""
    high_trait_threshold: float = Field(default=0.7)
    low_trait_threshold: float = Field(default=0.3)
    warmth_neuroticism_boost: float = Field(default=0.2)
    structure_conscientiousness_boost: float = Field(default=0.25)
    energy_extraversion_factor: float = Field(default=0.4)
    directness_agreeableness_factor: float = Field(default=-0.2)
    complexity_openness_boost: float = Field(default=0.3)
    default_warmth: float = Field(default=0.6)
    default_validation: float = Field(default=0.5)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_STYLE_", env_file=".env", extra="ignore")


@dataclass
class StyleRecommendation:
    """Recommendation for communication style adaptation."""
    category: str
    recommendation: str
    priority: int = 1
    trait_source: PersonalityTrait | None = None


@dataclass
class EmotionMapping:
    """Maps emotion to empathy strategy weights."""
    emotion: EmotionCategory
    cognitive_weight: float = 0.33
    affective_weight: float = 0.33
    compassionate_weight: float = 0.34


class StyleMapper:
    """Maps OCEAN traits to style parameters."""

    def __init__(self, settings: StyleAdapterSettings | None = None) -> None:
        self._settings = settings or StyleAdapterSettings()

    def map_to_style(self, scores: OceanScoresDTO) -> StyleParametersDTO:
        """Map OCEAN scores to communication style parameters."""
        warmth = self._compute_warmth(scores)
        structure = self._compute_structure(scores)
        complexity = self._compute_complexity(scores)
        directness = self._compute_directness(scores)
        energy = self._compute_energy(scores)
        validation_level = self._compute_validation(scores)
        style_type = self._determine_style_type(scores)
        return StyleParametersDTO(
            warmth=warmth,
            structure=structure,
            complexity=complexity,
            directness=directness,
            energy=energy,
            validation_level=validation_level,
            style_type=style_type,
        )

    def _compute_warmth(self, scores: OceanScoresDTO) -> float:
        """Compute warmth parameter from agreeableness and neuroticism."""
        base = self._settings.default_warmth
        agreeableness_contribution = (scores.agreeableness - 0.5) * 0.4
        neuroticism_boost = 0.0
        if scores.neuroticism > self._settings.high_trait_threshold:
            neuroticism_boost = self._settings.warmth_neuroticism_boost
        return self._clamp(base + agreeableness_contribution + neuroticism_boost)

    def _compute_structure(self, scores: OceanScoresDTO) -> float:
        """Compute structure level from conscientiousness."""
        base = 0.5
        conscientiousness_boost = 0.0
        if scores.conscientiousness > self._settings.high_trait_threshold:
            conscientiousness_boost = self._settings.structure_conscientiousness_boost
        elif scores.conscientiousness < self._settings.low_trait_threshold:
            conscientiousness_boost = -0.15
        return self._clamp(base + (scores.conscientiousness - 0.5) * 0.5 + conscientiousness_boost)

    def _compute_complexity(self, scores: OceanScoresDTO) -> float:
        """Compute complexity/abstraction level from openness."""
        base = 0.5
        openness_boost = 0.0
        if scores.openness > self._settings.high_trait_threshold:
            openness_boost = self._settings.complexity_openness_boost
        return self._clamp(base + (scores.openness - 0.5) * 0.4 + openness_boost)

    def _compute_directness(self, scores: OceanScoresDTO) -> float:
        """Compute directness from agreeableness (inverse) and neuroticism."""
        base = 0.5
        agreeableness_factor = (0.5 - scores.agreeableness) * self._settings.directness_agreeableness_factor * -1
        neuroticism_reduction = 0.0
        if scores.neuroticism > self._settings.high_trait_threshold:
            neuroticism_reduction = -0.15
        return self._clamp(base + agreeableness_factor + neuroticism_reduction)

    def _compute_energy(self, scores: OceanScoresDTO) -> float:
        """Compute energy level from extraversion."""
        base = 0.4
        extraversion_factor = (scores.extraversion - 0.5) * self._settings.energy_extraversion_factor * 2
        return self._clamp(base + extraversion_factor)

    def _compute_validation(self, scores: OceanScoresDTO) -> float:
        """Compute validation level from neuroticism and agreeableness."""
        base = self._settings.default_validation
        neuroticism_boost = 0.0
        if scores.neuroticism > self._settings.high_trait_threshold:
            neuroticism_boost = 0.25
        agreeableness_boost = (scores.agreeableness - 0.5) * 0.3
        return self._clamp(base + neuroticism_boost + agreeableness_boost)

    def _determine_style_type(self, scores: OceanScoresDTO) -> CommunicationStyleType:
        """Determine primary communication style type."""
        high_e = scores.extraversion > self._settings.high_trait_threshold
        high_a = scores.agreeableness > self._settings.high_trait_threshold
        high_c = scores.conscientiousness > self._settings.high_trait_threshold
        low_e = scores.extraversion < self._settings.low_trait_threshold
        if high_c and not high_e:
            return CommunicationStyleType.ANALYTICAL
        if high_e and not high_a:
            return CommunicationStyleType.DRIVER
        if high_e and high_a:
            return CommunicationStyleType.EXPRESSIVE
        if high_a and low_e:
            return CommunicationStyleType.AMIABLE
        return CommunicationStyleType.BALANCED

    def _clamp(self, value: float) -> float:
        """Clamp value between 0 and 1."""
        return max(0.0, min(1.0, value))


class RecommendationGenerator:
    """Generates style recommendations from OCEAN scores."""

    def __init__(self, settings: StyleAdapterSettings | None = None) -> None:
        self._settings = settings or StyleAdapterSettings()

    def generate(self, scores: OceanScoresDTO) -> list[str]:
        """Generate communication style recommendations."""
        recommendations: list[StyleRecommendation] = []
        if scores.neuroticism > self._settings.high_trait_threshold:
            recommendations.append(StyleRecommendation(category="safety", recommendation="Use extra validation and reassurance", priority=1, trait_source=PersonalityTrait.NEUROTICISM))
            recommendations.append(StyleRecommendation(category="pacing", recommendation="Pace responses gently, avoid overwhelming", priority=2, trait_source=PersonalityTrait.NEUROTICISM))
        if scores.extraversion < self._settings.low_trait_threshold:
            recommendations.append(StyleRecommendation(category="length", recommendation="Keep responses concise, allow processing time", priority=1, trait_source=PersonalityTrait.EXTRAVERSION))
            recommendations.append(StyleRecommendation(category="questions", recommendation="Ask one question at a time", priority=2, trait_source=PersonalityTrait.EXTRAVERSION))
        if scores.openness > self._settings.high_trait_threshold:
            recommendations.append(StyleRecommendation(category="content", recommendation="Use metaphors, analogies, and abstract concepts", priority=2, trait_source=PersonalityTrait.OPENNESS))
            recommendations.append(StyleRecommendation(category="approach", recommendation="Explore novel perspectives and creative exercises", priority=3, trait_source=PersonalityTrait.OPENNESS))
        if scores.conscientiousness > self._settings.high_trait_threshold:
            recommendations.append(StyleRecommendation(category="structure", recommendation="Provide clear structure and timelines", priority=1, trait_source=PersonalityTrait.CONSCIENTIOUSNESS))
            recommendations.append(StyleRecommendation(category="homework", recommendation="Include detailed, specific homework assignments", priority=2, trait_source=PersonalityTrait.CONSCIENTIOUSNESS))
        if scores.agreeableness < self._settings.low_trait_threshold:
            recommendations.append(StyleRecommendation(category="approach", recommendation="Be direct and evidence-based, avoid excessive warmth", priority=2, trait_source=PersonalityTrait.AGREEABLENESS))
        recommendations.sort(key=lambda r: r.priority)
        return [r.recommendation for r in recommendations[:5]]


class EmpathyAdapter:
    """Adapts empathy components based on emotion and personality."""
    _EMOTION_MAPPINGS: dict[EmotionCategory, EmotionMapping] = {
        EmotionCategory.SADNESS: EmotionMapping(EmotionCategory.SADNESS, 0.3, 0.5, 0.2),
        EmotionCategory.ANGER: EmotionMapping(EmotionCategory.ANGER, 0.4, 0.35, 0.25),
        EmotionCategory.FEAR: EmotionMapping(EmotionCategory.FEAR, 0.35, 0.4, 0.25),
        EmotionCategory.JOY: EmotionMapping(EmotionCategory.JOY, 0.25, 0.5, 0.25),
        EmotionCategory.SURPRISE: EmotionMapping(EmotionCategory.SURPRISE, 0.5, 0.3, 0.2),
        EmotionCategory.DISGUST: EmotionMapping(EmotionCategory.DISGUST, 0.4, 0.35, 0.25),
        EmotionCategory.TRUST: EmotionMapping(EmotionCategory.TRUST, 0.3, 0.35, 0.35),
        EmotionCategory.ANTICIPATION: EmotionMapping(EmotionCategory.ANTICIPATION, 0.35, 0.3, 0.35),
        EmotionCategory.NEUTRAL: EmotionMapping(EmotionCategory.NEUTRAL, 0.33, 0.33, 0.34),
    }

    def generate_components(self, emotion: EmotionStateDTO, style: StyleParametersDTO) -> EmpathyComponentsDTO:
        """Generate empathy components for response."""
        mapping = self._EMOTION_MAPPINGS.get(emotion.primary_emotion, self._EMOTION_MAPPINGS[EmotionCategory.NEUTRAL])
        strategy = self._select_strategy(emotion, mapping)
        cognitive = self._generate_cognitive(emotion, style)
        affective = self._generate_affective(emotion, style)
        compassionate = self._generate_compassionate(emotion, style)
        return EmpathyComponentsDTO(cognitive_content=cognitive, affective_content=affective, compassionate_content=compassionate, selected_strategy=strategy, emotion_state=emotion)

    def _select_strategy(self, emotion: EmotionStateDTO, mapping: EmotionMapping) -> str:
        """Select empathy strategy based on emotion intensity."""
        if emotion.intensity > 0.7:
            return "validation_first"
        if emotion.intensity < 0.3:
            return "compassionate_action"
        if mapping.cognitive_weight > 0.4:
            return "cognitive_focus"
        return "balanced"

    def _generate_cognitive(self, emotion: EmotionStateDTO, style: StyleParametersDTO) -> str:
        """Generate cognitive empathy content."""
        templates = {
            EmotionCategory.SADNESS: "It sounds like you're feeling {intensity} sadness about this situation.",
            EmotionCategory.ANGER: "I can hear that you're feeling {intensity} frustrated.",
            EmotionCategory.FEAR: "It seems like there's some {intensity} anxiety or worry present.",
            EmotionCategory.JOY: "I can sense your {intensity} happiness and excitement.",
            EmotionCategory.NEUTRAL: "I hear what you're sharing.",
        }
        intensity_word = "significant" if emotion.intensity > 0.6 else "some" if emotion.intensity > 0.3 else "mild"
        template = templates.get(emotion.primary_emotion, templates[EmotionCategory.NEUTRAL])
        return template.format(intensity=intensity_word)

    def _generate_affective(self, emotion: EmotionStateDTO, style: StyleParametersDTO) -> str:
        """Generate affective empathy content."""
        if emotion.valence < -0.3:
            base = "That must be really difficult."
            if style.warmth > 0.7:
                return f"{base} I'm here with you in this."
            return base
        if emotion.valence > 0.3:
            return "That's wonderful to hear."
        return "Thank you for sharing that with me."

    def _generate_compassionate(self, emotion: EmotionStateDTO, style: StyleParametersDTO) -> str:
        """Generate compassionate empathy content."""
        if emotion.intensity > 0.6:
            return "What would feel most supportive for you right now?"
        return "How can I best support you with this?"


class StyleAdapter:
    """Main style adapter orchestrating personality-to-style mapping."""

    def __init__(self, settings: StyleAdapterSettings | None = None) -> None:
        self._settings = settings or StyleAdapterSettings()
        self._mapper = StyleMapper(self._settings)
        self._recommender = RecommendationGenerator(self._settings)
        self._empathy_adapter = EmpathyAdapter()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the style adapter."""
        self._initialized = True
        logger.info("style_adapter_initialized")

    def get_style_parameters(self, scores: OceanScoresDTO) -> StyleParametersDTO:
        """Get communication style parameters for OCEAN scores."""
        return self._mapper.map_to_style(scores)

    def get_recommendations(self, scores: OceanScoresDTO) -> list[str]:
        """Get communication recommendations for OCEAN scores."""
        return self._recommender.generate(scores)

    def get_empathy_components(self, emotion: EmotionStateDTO, style: StyleParametersDTO) -> EmpathyComponentsDTO:
        """Get empathy components for emotion and style."""
        return self._empathy_adapter.generate_components(emotion, style)

    def adapt_response(self, base_response: str, style: StyleParametersDTO) -> str:
        """Adapt response text according to style parameters."""
        adapted = base_response
        if style.warmth > 0.7 and not base_response.startswith("I "):
            adapted = f"I hear you. {adapted}"
        if style.structure > 0.7 and len(base_response) > 200:
            adapted = self._add_structure(adapted)
        if style.validation_level > 0.7:
            adapted = self._add_validation(adapted)
        return adapted

    def _add_structure(self, text: str) -> str:
        """Add structural elements to response."""
        sentences = text.split(". ")
        if len(sentences) > 3:
            mid = len(sentences) // 2
            sentences.insert(mid, "\n\n")
        return ". ".join(sentences)

    def _add_validation(self, text: str) -> str:
        """Add validation prefix if not present."""
        validation_starters = ["I understand", "I hear", "That makes sense", "It's understandable"]
        if not any(text.startswith(v) for v in validation_starters):
            return f"That makes sense. {text}"
        return text

    async def shutdown(self) -> None:
        """Shutdown the style adapter."""
        self._initialized = False
        logger.info("style_adapter_shutdown")
