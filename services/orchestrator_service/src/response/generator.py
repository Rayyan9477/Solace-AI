"""
Solace-AI Orchestrator Service - Response Generator.
Generates final responses with formatting, structure, and therapeutic considerations.
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
    ProcessingPhase,
    AgentResult,
    IntentType,
    RiskLevel,
)

logger = structlog.get_logger(__name__)


class ResponseType(str, Enum):
    """Types of generated responses."""
    CONVERSATIONAL = "conversational"
    THERAPEUTIC = "therapeutic"
    EDUCATIONAL = "educational"
    CRISIS = "crisis"
    ASSESSMENT = "assessment"
    PROGRESS = "progress"


class ResponseFormat(str, Enum):
    """Response format options."""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    STRUCTURED = "structured"


class GeneratorSettings(BaseSettings):
    """Configuration for the response generator."""
    default_format: str = Field(default="plain")
    max_response_length: int = Field(default=2000, ge=100, le=5000)
    min_response_length: int = Field(default=50, ge=10, le=500)
    enable_empathy_phrases: bool = Field(default=True)
    enable_validation_statements: bool = Field(default=True)
    enable_follow_up_questions: bool = Field(default=True)
    enable_resource_linking: bool = Field(default=True)
    empathy_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    warmth_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_GENERATOR_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class ResponseContext:
    """Context for response generation."""
    intent: IntentType
    risk_level: RiskLevel
    personality_warmth: float
    personality_validation: float
    has_active_treatment: bool
    message_count: int
    is_first_message: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedResponse:
    """Result of response generation."""
    content: str
    response_type: ResponseType
    format_applied: ResponseFormat
    empathy_applied: bool
    validation_applied: bool
    follow_up_included: bool
    word_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "response_type": self.response_type.value,
            "format_applied": self.format_applied.value,
            "empathy_applied": self.empathy_applied,
            "validation_applied": self.validation_applied,
            "follow_up_included": self.follow_up_included,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }


class EmpathyEnhancer:
    """Enhances responses with empathetic language."""
    EMPATHY_PREFIXES = ["I hear you.", "Thank you for sharing that with me.", "I appreciate you opening up about this."]
    VALIDATION_PHRASES = ["Your feelings are valid.", "What you're experiencing is understandable.", "It's okay to feel this way."]

    def __init__(self, settings: GeneratorSettings) -> None:
        self._settings = settings

    def enhance(self, content: str, warmth: float, validation_level: float, response_type: ResponseType) -> tuple[str, bool, bool]:
        """Enhance content with empathy and validation."""
        if response_type == ResponseType.CRISIS:
            return content, False, False
        empathy_applied, validation_applied = False, False
        enhanced = content
        if self._settings.enable_empathy_phrases and warmth >= self._settings.warmth_threshold:
            if not self._starts_with_empathy(content):
                enhanced = f"{self.EMPATHY_PREFIXES[0 if warmth < 0.8 else 2]} {enhanced}"
                empathy_applied = True
        if self._settings.enable_validation_statements and validation_level >= 0.6:
            if not self._contains_validation(enhanced):
                sentences = enhanced.split(". ")
                if len(sentences) > 1:
                    sentences.insert(1, self.VALIDATION_PHRASES[1])
                    enhanced = ". ".join(sentences)
                    validation_applied = True
        return enhanced, empathy_applied, validation_applied

    def _starts_with_empathy(self, content: str) -> bool:
        content_lower = content.lower()
        return any(content_lower.startswith(ind) for ind in ["i hear", "thank you for", "i appreciate", "it sounds like", "i understand"])

    def _contains_validation(self, content: str) -> bool:
        content_lower = content.lower()
        return any(ind in content_lower for ind in ["your feelings", "it makes sense", "understandable", "it's okay", "valid"])


class FollowUpGenerator:
    """Generates appropriate follow-up questions."""
    FOLLOW_UPS = {
        IntentType.EMOTIONAL_SUPPORT: ["How are you feeling right now?", "What brought you here today?"],
        IntentType.SYMPTOM_DISCUSSION: ["When did you first notice these feelings?", "What brought you here today?"],
        IntentType.TREATMENT_INQUIRY: ["Is there a particular approach you're curious about?", "What brought you here today?"],
        IntentType.COPING_STRATEGY: ["Which of these strategies feels most relevant?", "What brought you here today?"],
        IntentType.GENERAL_CHAT: ["How are you feeling today?", "What brought you here today?"],
    }

    def __init__(self, settings: GeneratorSettings) -> None:
        self._settings = settings

    def generate(self, intent: IntentType, is_first_message: bool, content: str) -> str | None:
        """Generate appropriate follow-up question."""
        if not self._settings.enable_follow_up_questions or content.strip().endswith("?"):
            return None
        follow_ups = self.FOLLOW_UPS.get(intent, self.FOLLOW_UPS[IntentType.GENERAL_CHAT])
        return follow_ups[-1] if is_first_message else follow_ups[0]


class ContentFormatter:
    """Formats response content according to specified format."""

    def __init__(self, settings: GeneratorSettings) -> None:
        self._settings = settings

    def format(self, content: str, response_format: ResponseFormat, response_type: ResponseType) -> str:
        """Format content according to type and format."""
        formatted = re.sub(r'\s+', ' ', content).strip()
        if formatted and formatted[-1] not in '.!?':
            formatted += '.'
        if len(formatted) > self._settings.max_response_length:
            truncated = formatted[:self._settings.max_response_length]
            last_sentence = truncated.rfind('. ')
            if last_sentence > self._settings.max_response_length // 2:
                truncated = truncated[:last_sentence + 1]
            formatted = truncated
        return formatted


class ResponseGenerator:
    """Generates final formatted responses for the orchestrator."""

    def __init__(self, settings: GeneratorSettings | None = None) -> None:
        self._settings = settings or GeneratorSettings()
        self._empathy_enhancer = EmpathyEnhancer(self._settings)
        self._follow_up_generator = FollowUpGenerator(self._settings)
        self._formatter = ContentFormatter(self._settings)
        self._generation_count = 0

    def generate(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Generate final response from aggregated content.
        Main LangGraph node function.
        """
        self._generation_count += 1
        content = state.get("final_response", "")
        intent = IntentType(state.get("intent", "general_chat"))
        safety_flags = state.get("safety_flags", {})
        personality_style = state.get("personality_style", {})
        messages = state.get("messages", [])
        logger.info(
            "generator_processing",
            content_length=len(content),
            intent=intent.value,
        )
        context = self._build_context(intent, safety_flags, personality_style, messages)
        result = self._generate_response(content, context)
        agent_result = AgentResult(
            agent_type=AgentType.AGGREGATOR,
            success=True,
            response_content=result.content,
            confidence=0.9,
            metadata={
                "response_type": result.response_type.value,
                "empathy_applied": result.empathy_applied,
                "validation_applied": result.validation_applied,
            },
        )
        logger.info(
            "generator_complete",
            response_type=result.response_type.value,
            word_count=result.word_count,
            empathy_applied=result.empathy_applied,
        )
        return {
            "final_response": result.content,
            "processing_phase": ProcessingPhase.RESPONSE_GENERATION.value,
            "agent_results": [agent_result.to_dict()],
            "metadata": {
                **state.get("metadata", {}),
                "generation": result.to_dict(),
            },
        }

    def _build_context(
        self,
        intent: IntentType,
        safety_flags: dict[str, Any],
        personality_style: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponseContext:
        """Build context for response generation."""
        risk_level = RiskLevel(safety_flags.get("risk_level", "NONE"))
        warmth = personality_style.get("warmth", 0.6)
        validation = personality_style.get("validation_level", 0.6)
        has_treatment = False
        message_count = len(messages)
        is_first = message_count <= 1
        return ResponseContext(
            intent=intent,
            risk_level=risk_level,
            personality_warmth=warmth,
            personality_validation=validation,
            has_active_treatment=has_treatment,
            message_count=message_count,
            is_first_message=is_first,
        )

    def _generate_response(
        self,
        content: str,
        context: ResponseContext,
    ) -> GeneratedResponse:
        """Generate the response with all enhancements."""
        response_type = self._determine_response_type(context)
        response_format = ResponseFormat(self._settings.default_format)
        enhanced_content, empathy_applied, validation_applied = self._empathy_enhancer.enhance(
            content,
            context.personality_warmth,
            context.personality_validation,
            response_type,
        )
        follow_up = self._follow_up_generator.generate(
            context.intent,
            context.is_first_message,
            enhanced_content,
        )
        follow_up_included = False
        if follow_up:
            enhanced_content = f"{enhanced_content} {follow_up}"
            follow_up_included = True
        final_content = self._formatter.format(
            enhanced_content,
            response_format,
            response_type,
        )
        word_count = len(final_content.split())
        return GeneratedResponse(
            content=final_content,
            response_type=response_type,
            format_applied=response_format,
            empathy_applied=empathy_applied,
            validation_applied=validation_applied,
            follow_up_included=follow_up_included,
            word_count=word_count,
            metadata={"intent": context.intent.value},
        )

    def _determine_response_type(self, context: ResponseContext) -> ResponseType:
        """Determine response type from context."""
        if context.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return ResponseType.CRISIS
        type_mapping: dict[IntentType, ResponseType] = {
            IntentType.CRISIS_DISCLOSURE: ResponseType.CRISIS,
            IntentType.EMOTIONAL_SUPPORT: ResponseType.THERAPEUTIC,
            IntentType.SYMPTOM_DISCUSSION: ResponseType.ASSESSMENT,
            IntentType.TREATMENT_INQUIRY: ResponseType.THERAPEUTIC,
            IntentType.PROGRESS_UPDATE: ResponseType.PROGRESS,
            IntentType.ASSESSMENT_REQUEST: ResponseType.ASSESSMENT,
            IntentType.COPING_STRATEGY: ResponseType.THERAPEUTIC,
            IntentType.PSYCHOEDUCATION: ResponseType.EDUCATIONAL,
            IntentType.SESSION_MANAGEMENT: ResponseType.CONVERSATIONAL,
            IntentType.GENERAL_CHAT: ResponseType.CONVERSATIONAL,
        }
        return type_mapping.get(context.intent, ResponseType.CONVERSATIONAL)

    def get_statistics(self) -> dict[str, Any]:
        """Get generator statistics."""
        return {
            "total_generations": self._generation_count,
            "settings": {
                "max_response_length": self._settings.max_response_length,
                "empathy_enabled": self._settings.enable_empathy_phrases,
                "follow_up_enabled": self._settings.enable_follow_up_questions,
            },
        }


def generator_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for response generation.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    generator = ResponseGenerator()
    return generator.generate(state)
