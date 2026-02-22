"""
Solace-AI Orchestrator Service - Style Applicator.
Applies personality-based communication styles to responses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import re
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..langgraph.state_schema import (
    OrchestratorState,
    AgentType,
    AgentResult,
)

logger = structlog.get_logger(__name__)


class CommunicationStyle(str, Enum):
    """Communication style types based on personality."""
    ANALYTICAL = "analytical"
    EXPRESSIVE = "expressive"
    DRIVER = "driver"
    AMIABLE = "amiable"
    BALANCED = "balanced"


class StyleApplicatorSettings(BaseSettings):
    """Configuration for the style applicator."""
    enable_warmth_adjustment: bool = Field(default=True)
    enable_complexity_adjustment: bool = Field(default=True)
    enable_structure_adjustment: bool = Field(default=True)
    enable_directness_adjustment: bool = Field(default=True)
    min_warmth_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_warmth_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    default_style: str = Field(default="balanced")
    preserve_crisis_content: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_STYLE_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class StyleParameters:
    """Parameters for style application."""
    warmth: float = 0.6
    structure: float = 0.5
    complexity: float = 0.5
    directness: float = 0.5
    energy: float = 0.5
    validation_level: float = 0.6
    style_type: CommunicationStyle = CommunicationStyle.BALANCED

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleParameters:
        """Create from dictionary."""
        style_type_value = data.get("style_type", "balanced")
        if isinstance(style_type_value, str):
            style_type = CommunicationStyle(style_type_value.lower())
        else:
            style_type = CommunicationStyle.BALANCED
        return cls(
            warmth=float(data.get("warmth", 0.6)),
            structure=float(data.get("structure", 0.5)),
            complexity=float(data.get("complexity", 0.5)),
            directness=float(data.get("directness", 0.5)),
            energy=float(data.get("energy", 0.5)),
            validation_level=float(data.get("validation_level", 0.6)),
            style_type=style_type,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "warmth": self.warmth,
            "structure": self.structure,
            "complexity": self.complexity,
            "directness": self.directness,
            "energy": self.energy,
            "validation_level": self.validation_level,
            "style_type": self.style_type.value,
        }


@dataclass
class StyledResponse:
    """Result of style application."""
    content: str
    original_content: str
    style_applied: CommunicationStyle
    adjustments_made: list[str]
    warmth_level: float
    complexity_level: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "original_content": self.original_content,
            "style_applied": self.style_applied.value,
            "adjustments_made": self.adjustments_made,
            "warmth_level": self.warmth_level,
            "complexity_level": self.complexity_level,
            "metadata": self.metadata,
        }


class WarmthAdjuster:
    """Adjusts response warmth based on personality."""

    WARM_OPENERS: list[str] = [
        "I really appreciate you sharing this.",
        "Thank you for trusting me with this.",
        "I'm glad you felt comfortable telling me.",
        "It means a lot that you opened up.",
    ]
    WARM_CLOSERS: list[str] = [
        "I'm here for you.",
        "We're in this together.",
        "Please know I care about how you're doing.",
        "Take care of yourself.",
    ]
    NEUTRAL_OPENERS: list[str] = [
        "I understand.",
        "I see.",
        "Thank you for sharing.",
    ]

    def adjust(self, content: str, warmth: float, is_crisis: bool) -> tuple[str, bool]:
        """Adjust content warmth level."""
        if is_crisis:
            return content, False
        adjusted = content
        adjustment_made = False
        if warmth >= 0.7:
            if not self._has_warm_opening(content):
                opener = self._select_warm_opener(warmth)
                adjusted = f"{opener} {adjusted}"
                adjustment_made = True
        elif warmth <= 0.4:
            adjusted = self._reduce_warmth(adjusted)
            adjustment_made = True
        return adjusted, adjustment_made

    def _has_warm_opening(self, content: str) -> bool:
        """Check if content has warm opening."""
        content_lower = content.lower()
        warm_indicators = [
            "appreciate", "thank you for", "glad you",
            "means a lot", "trust me", "care about",
        ]
        return any(ind in content_lower[:100] for ind in warm_indicators)

    def _select_warm_opener(self, warmth: float) -> str:
        """Select warm opener based on level."""
        if warmth >= 0.8:
            return self.WARM_OPENERS[1]
        return self.WARM_OPENERS[0]

    def _reduce_warmth(self, content: str) -> str:
        """Reduce warmth for lower warmth preferences."""
        replacements = [
            ("I really appreciate", "I acknowledge"),
            ("Thank you so much for", "Thank you for"),
            ("I'm so glad", "I'm glad"),
            ("wonderful", "good"),
            ("amazing", "notable"),
        ]
        result = content
        for old, new in replacements:
            result = result.replace(old, new)
        return result


class ComplexityAdjuster:
    """Adjusts response complexity based on personality."""

    SIMPLIFICATION_REPLACEMENTS: list[tuple[str, str]] = [
        ("utilize", "use"),
        ("implement", "do"),
        ("facilitate", "help"),
        ("conceptualize", "think about"),
        ("subsequently", "then"),
        ("furthermore", "also"),
        ("nevertheless", "but"),
        ("approximately", "about"),
    ]

    COMPLEXITY_REPLACEMENTS: list[tuple[str, str]] = [
        ("help", "facilitate"),
        ("think about", "conceptualize"),
        ("then", "subsequently"),
        ("also", "furthermore"),
        ("but", "nevertheless"),
        ("about", "approximately"),
        ("show", "demonstrate"),
        ("get", "obtain"),
        ("start", "initiate"),
        ("end", "conclude"),
        ("make", "construct"),
        ("find", "ascertain"),
        ("begin", "commence"),
    ]

    def adjust(self, content: str, complexity: float) -> tuple[str, bool]:
        """Adjust content complexity level."""
        adjusted = content
        adjustment_made = False
        if complexity <= 0.4:
            adjusted = self._simplify(content)
            adjustment_made = adjusted != content
        elif complexity >= 0.7:
            adjusted = self._increase_complexity(content)
            adjustment_made = adjusted != content
        return adjusted, adjustment_made

    def _simplify(self, content: str) -> str:
        """Simplify content for lower complexity preferences."""
        result = content
        for complex_word, simple_word in self.SIMPLIFICATION_REPLACEMENTS:
            result = re.sub(
                rf'\b{complex_word}\b',
                simple_word,
                result,
                flags=re.IGNORECASE
            )
        return result

    def _increase_complexity(self, content: str) -> str:
        """Increase complexity for higher complexity preferences."""
        result = content
        for simple_word, complex_word in self.COMPLEXITY_REPLACEMENTS:
            result = re.sub(
                rf'\b{simple_word}\b',
                complex_word,
                result,
                flags=re.IGNORECASE
            )
        return result


class StructureAdjuster:
    """Adjusts response structure based on personality."""

    def adjust(self, content: str, structure: float) -> tuple[str, bool]:
        """Adjust content structure level."""
        adjusted = content
        adjustment_made = False
        if structure >= 0.7:
            adjusted = self._add_structure(content)
            adjustment_made = adjusted != content
        elif structure <= 0.3:
            adjusted = self._reduce_structure(content)
            adjustment_made = adjusted != content
        return adjusted, adjustment_made

    def _add_structure(self, content: str) -> str:
        """Add structure for high structure preferences."""
        sentences = content.split(". ")
        if len(sentences) >= 4:
            return content
        return content

    def _reduce_structure(self, content: str) -> str:
        """Reduce structure for low structure preferences."""
        content = re.sub(r'\d+\.\s+', '', content)
        content = re.sub(r'•\s+', '', content)
        return content


class DirectnessAdjuster:
    """Adjusts response directness based on personality."""
    INDIRECT_REPLACEMENTS = [("You should", "You might consider"), ("You need to", "It could help to")]
    DIRECT_REPLACEMENTS = [("You might consider", "Consider"), ("It could help to", "Try to")]

    def adjust(self, content: str, directness: float) -> tuple[str, bool]:
        """Adjust content directness level."""
        if directness <= 0.4:
            for old, new in self.INDIRECT_REPLACEMENTS:
                content = content.replace(old, new)
        elif directness >= 0.7:
            for old, new in self.DIRECT_REPLACEMENTS:
                content = content.replace(old, new)
        return content, directness <= 0.4 or directness >= 0.7


class StyleApplicator:
    """Applies personality-based styles to responses."""

    def __init__(self, settings: StyleApplicatorSettings | None = None) -> None:
        self._settings = settings or StyleApplicatorSettings()
        self._warmth_adjuster = WarmthAdjuster()
        self._complexity_adjuster = ComplexityAdjuster()
        self._structure_adjuster = StructureAdjuster()
        self._directness_adjuster = DirectnessAdjuster()
        self._application_count = 0

    def apply(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Apply style to response content.
        Main processing function.
        """
        self._application_count += 1
        content = state.get("final_response", "")
        personality_style = state.get("personality_style", {})
        safety_flags = state.get("safety_flags", {})
        logger.info(
            "style_applicator_processing",
            content_length=len(content),
            has_style=bool(personality_style),
        )
        params = StyleParameters.from_dict(personality_style) if personality_style else StyleParameters()
        is_crisis = safety_flags.get("crisis_detected", False)
        result = self._apply_style(content, params, is_crisis)
        agent_result = AgentResult(
            agent_type=AgentType.PERSONALITY,
            success=True,
            confidence=0.85,
            metadata={
                "style_applied": result.style_applied.value,
                "adjustments_count": len(result.adjustments_made),
            },
        )
        logger.info(
            "style_applicator_complete",
            style=result.style_applied.value,
            adjustments=result.adjustments_made,
        )
        return {
            "final_response": result.content,
            "agent_results": [agent_result.to_dict()],
            "metadata": {
                **state.get("metadata", {}),
                "style_application": result.to_dict(),
            },
        }

    def _apply_style(
        self,
        content: str,
        params: StyleParameters,
        is_crisis: bool,
    ) -> StyledResponse:
        """Apply style transformations to content."""
        adjustments: list[str] = []
        styled = content
        if is_crisis and self._settings.preserve_crisis_content:
            return StyledResponse(
                content=styled,
                original_content=content,
                style_applied=params.style_type,
                adjustments_made=["crisis_preserved"],
                warmth_level=params.warmth,
                complexity_level=params.complexity,
                metadata={"crisis_mode": True},
            )
        if self._settings.enable_warmth_adjustment:
            styled, adjusted = self._warmth_adjuster.adjust(styled, params.warmth, is_crisis)
            if adjusted:
                adjustments.append("warmth")
        if self._settings.enable_complexity_adjustment:
            styled, adjusted = self._complexity_adjuster.adjust(styled, params.complexity)
            if adjusted:
                adjustments.append("complexity")
        if self._settings.enable_structure_adjustment:
            styled, adjusted = self._structure_adjuster.adjust(styled, params.structure)
            if adjusted:
                adjustments.append("structure")
        if self._settings.enable_directness_adjustment:
            styled, adjusted = self._directness_adjuster.adjust(styled, params.directness)
            if adjusted:
                adjustments.append("directness")
        return StyledResponse(
            content=styled,
            original_content=content,
            style_applied=params.style_type,
            adjustments_made=adjustments,
            warmth_level=params.warmth,
            complexity_level=params.complexity,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get applicator statistics."""
        return {
            "total_applications": self._application_count,
            "settings": {
                "warmth_enabled": self._settings.enable_warmth_adjustment,
                "complexity_enabled": self._settings.enable_complexity_adjustment,
                "structure_enabled": self._settings.enable_structure_adjustment,
                "directness_enabled": self._settings.enable_directness_adjustment,
            },
        }


def style_applicator_node(state: OrchestratorState) -> dict[str, Any]:
    """LangGraph node function for style application."""
    return StyleApplicator().apply(state)
