"""
Solace-AI Orchestrator Service - Chat Agent.
Handles general conversation with empathetic, supportive responses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..langgraph.state_schema import (
    OrchestratorState,
    AgentType,
    AgentResult,
    MessageEntry,
)

if TYPE_CHECKING:
    from services.shared.infrastructure.llm_client import UnifiedLLMClient

logger = structlog.get_logger(__name__)

# Module-level LLM client reference, set during orchestrator startup
_llm_client: UnifiedLLMClient | None = None


def configure_chat_agent_llm(client: UnifiedLLMClient | None) -> None:
    """Set the LLM client for the chat agent. Called during orchestrator startup."""
    global _llm_client
    _llm_client = client


CHAT_SYSTEM_PROMPT = (
    "You are Solace, a warm and compassionate AI mental health support companion. "
    "You are having a general conversation (not a formal therapy session). "
    "Your responses should:\n"
    "- Be empathetic, warm, and naturally conversational\n"
    "- Validate the person's feelings without being clinical\n"
    "- Gently encourage them to share more when appropriate\n"
    "- Keep responses concise (2-4 sentences typically)\n"
    "- Never diagnose, prescribe medication, or provide clinical assessments\n"
    "- If someone shares something concerning, gently encourage professional support\n"
    "- Use the person's context and conversation history to make responses "
    "feel personal and connected"
)


class ConversationTone(str, Enum):
    """Tone categories for conversation."""
    WARM = "warm"
    NEUTRAL = "neutral"
    PROFESSIONAL = "professional"
    ENCOURAGING = "encouraging"
    REFLECTIVE = "reflective"


class TopicCategory(str, Enum):
    """Categories of conversation topics."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    GRATITUDE = "gratitude"
    SMALL_TALK = "small_talk"
    CHECK_IN = "check_in"
    CLARIFICATION = "clarification"
    GENERAL = "general"


class ChatAgentSettings(BaseSettings):
    """Configuration for the chat agent."""
    default_warmth: float = Field(default=0.7, ge=0.0, le=1.0)
    default_validation_level: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_personality_adaptation: bool = Field(default=True)
    max_response_length: int = Field(default=500, ge=100, le=2000)
    include_follow_up_questions: bool = Field(default=True)
    empathy_phrases_enabled: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_CHAT_AGENT_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class TopicClassification:
    """Result of topic classification."""
    category: TopicCategory
    confidence: float
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "keywords": self.keywords,
        }


@dataclass
class ChatResponse:
    """Generated chat response."""
    content: str
    tone: ConversationTone
    topic: TopicCategory
    includes_follow_up: bool
    empathy_applied: bool
    warmth_level: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "tone": self.tone.value,
            "topic": self.topic.value,
            "includes_follow_up": self.includes_follow_up,
            "empathy_applied": self.empathy_applied,
            "warmth_level": self.warmth_level,
        }


class TopicClassifier:
    """Classifies conversation topics from message content."""

    GREETING_PATTERNS = ["hello", "hi ", " hi", "hey", "good morning", "good afternoon", "good evening"]
    FAREWELL_PATTERNS = ["goodbye", "bye", "see you", "take care", "goodnight"]
    GRATITUDE_PATTERNS = ["thank you", "thanks", "appreciate", "grateful"]
    CHECK_IN_PATTERNS = ["how are you", "how's it going", "what's up", "how have you been"]
    CLARIFICATION_PATTERNS = ["what do you mean", "can you explain", "i don't understand", "clarify"]

    def classify(self, message: str) -> TopicClassification:
        """Classify the topic of a message."""
        message_lower = message.lower().strip()
        message_with_spaces = f" {message_lower} "
        if self._matches_greeting(message_lower):
            return TopicClassification(
                category=TopicCategory.GREETING,
                confidence=0.9,
                keywords=self._extract_matches(message_lower, self.GREETING_PATTERNS),
            )
        if self._matches_patterns(message_lower, self.FAREWELL_PATTERNS):
            return TopicClassification(
                category=TopicCategory.FAREWELL,
                confidence=0.9,
                keywords=self._extract_matches(message_lower, self.FAREWELL_PATTERNS),
            )
        if self._matches_patterns(message_lower, self.GRATITUDE_PATTERNS):
            return TopicClassification(
                category=TopicCategory.GRATITUDE,
                confidence=0.85,
                keywords=self._extract_matches(message_lower, self.GRATITUDE_PATTERNS),
            )
        if self._matches_patterns(message_lower, self.CHECK_IN_PATTERNS):
            return TopicClassification(
                category=TopicCategory.CHECK_IN,
                confidence=0.85,
                keywords=self._extract_matches(message_lower, self.CHECK_IN_PATTERNS),
            )
        if self._matches_patterns(message_lower, self.CLARIFICATION_PATTERNS):
            return TopicClassification(
                category=TopicCategory.CLARIFICATION,
                confidence=0.8,
                keywords=self._extract_matches(message_lower, self.CLARIFICATION_PATTERNS),
            )
        if len(message_lower.split()) <= 5:
            return TopicClassification(
                category=TopicCategory.SMALL_TALK,
                confidence=0.6,
            )
        return TopicClassification(
            category=TopicCategory.GENERAL,
            confidence=0.5,
        )

    def _matches_greeting(self, text: str) -> bool:
        """Check if text is a greeting - handles 'hi' specially to avoid false positives."""
        text_with_spaces = f" {text} "
        for pattern in ["hello", "hey", "good morning", "good afternoon", "good evening"]:
            if pattern in text:
                return True
        if " hi " in text_with_spaces or text == "hi" or text.startswith("hi ") or text.startswith("hi,"):
            return True
        return False

    def _matches_patterns(self, text: str, patterns: list[str]) -> bool:
        """Check if text matches any pattern."""
        return any(pattern in text for pattern in patterns)

    def _extract_matches(self, text: str, patterns: list[str]) -> list[str]:
        """Extract matched patterns from text."""
        return [p for p in patterns if p in text]


class ResponseGenerator:
    """Generates empathetic chat responses based on topic and personality."""

    def __init__(self, settings: ChatAgentSettings) -> None:
        self._settings = settings

    def generate(
        self,
        message: str,
        topic: TopicClassification,
        personality_style: dict[str, Any],
        conversation_context: str,
        memory_context: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Generate a response based on topic and personality."""
        warmth = personality_style.get("warmth", self._settings.default_warmth)
        validation_level = personality_style.get("validation_level", self._settings.default_validation_level)
        tone = self._determine_tone(topic.category, warmth)
        
        # Incorporate memory context if available
        context_aware_prefix = ""
        if memory_context and memory_context.get("retrieval_count", 0) > 0:
            context_aware_prefix = self._build_context_aware_prefix(conversation_context, memory_context)
        
        content = self._generate_content(message, topic, warmth, validation_level, context_aware_prefix)
        
        if self._settings.include_follow_up_questions and topic.category not in (
            TopicCategory.FAREWELL,
            TopicCategory.GRATITUDE,
        ):
            follow_up = self._generate_follow_up(topic.category, message)
            if follow_up:
                content = f"{content} {follow_up}"
        return ChatResponse(
            content=content,
            tone=tone,
            topic=topic.category,
            includes_follow_up=self._settings.include_follow_up_questions,
            empathy_applied=self._settings.empathy_phrases_enabled,
            warmth_level=warmth,
        )

    def _determine_tone(self, topic: TopicCategory, warmth: float) -> ConversationTone:
        """Determine appropriate tone for response."""
        if topic == TopicCategory.GREETING:
            return ConversationTone.WARM if warmth > 0.6 else ConversationTone.NEUTRAL
        if topic == TopicCategory.FAREWELL:
            return ConversationTone.ENCOURAGING
        if topic == TopicCategory.GRATITUDE:
            return ConversationTone.WARM
        if topic == TopicCategory.CHECK_IN:
            return ConversationTone.REFLECTIVE
        return ConversationTone.WARM if warmth > 0.5 else ConversationTone.NEUTRAL

    def _build_context_aware_prefix(self, conversation_context: str, memory_context: dict[str, Any]) -> str:
        """Build a prefix based on retrieved memory context."""
        if not conversation_context:
            return ""
        
        # Extract key information from memory context
        retrieval_count = memory_context.get("retrieval_count", 0)
        
        if retrieval_count > 0:
            # Add subtle context awareness (not explicitly mentioning memory retrieval)
            return "Based on our previous conversations, "
        
        return ""

    def _generate_content(
        self,
        message: str,
        topic: TopicClassification,
        warmth: float,
        validation_level: float,
        context_aware_prefix: str = "",
    ) -> str:
        """Generate response content based on topic."""
        if topic.category == TopicCategory.GREETING:
            return self._greeting_response(warmth)
        if topic.category == TopicCategory.FAREWELL:
            return self._farewell_response(warmth)
        if topic.category == TopicCategory.GRATITUDE:
            return self._gratitude_response(warmth)
        if topic.category == TopicCategory.CHECK_IN:
            return self._check_in_response(warmth)
        if topic.category == TopicCategory.CLARIFICATION:
            return self._clarification_response(message)
        return self._general_response(message, warmth, validation_level, context_aware_prefix)

    def _greeting_response(self, warmth: float) -> str:
        """Generate greeting response."""
        if warmth > 0.7:
            return "Hello! It's wonderful to connect with you. I'm here and ready to listen."
        if warmth > 0.5:
            return "Hello! Good to hear from you. How can I help you today?"
        return "Hello. I'm here to assist you."

    def _farewell_response(self, warmth: float) -> str:
        """Generate farewell response."""
        if warmth > 0.7:
            return (
                "Take care of yourself. Remember, I'm here whenever you need someone to talk to. "
                "You've taken an important step today."
            )
        if warmth > 0.5:
            return "Take care. Feel free to reach out anytime you'd like to talk."
        return "Goodbye. I'm available whenever you need support."

    def _gratitude_response(self, warmth: float) -> str:
        """Generate gratitude response."""
        if warmth > 0.7:
            return (
                "You're very welcome. It means a lot that you shared this with me. "
                "I'm glad I could be here for you."
            )
        if warmth > 0.5:
            return "You're welcome. I'm happy I could help."
        return "You're welcome."

    def _check_in_response(self, warmth: float) -> str:
        """Generate check-in response."""
        if warmth > 0.7:
            return (
                "Thank you for asking. I'm here and fully focused on being present with you. "
                "More importantly, how are you doing today?"
            )
        return "I'm here and ready to listen. How are you feeling today?"

    def _clarification_response(self, message: str) -> str:
        """Generate clarification response."""
        return (
            "I want to make sure I understand you correctly. "
            "Could you tell me more about what you mean?"
        )

    def _general_response(self, message: str, warmth: float, validation_level: float, context_aware_prefix: str = "") -> str:
        """Generate general conversation response."""
        empathy_prefix = ""
        if self._settings.empathy_phrases_enabled and validation_level > 0.5:
            empathy_prefix = "Thank you for sharing that with me. "
        
        # Add context-aware prefix if available
        full_prefix = context_aware_prefix + empathy_prefix if context_aware_prefix else empathy_prefix
        
        if warmth > 0.7:
            return (
                f"{full_prefix}I appreciate you opening up. "
                "I'm here to listen and support you in whatever way I can."
            )
        if warmth > 0.5:
            return f"{full_prefix}I hear you. Tell me more about what's on your mind."
        return f"{full_prefix}I understand. How can I help you with this?"

    def _generate_follow_up(self, topic: TopicCategory, message: str) -> str:
        """Generate appropriate follow-up question."""
        if topic == TopicCategory.GREETING:
            return "How are you feeling today?"
        if topic == TopicCategory.CHECK_IN:
            return "What's been on your mind lately?"
        if topic == TopicCategory.GENERAL:
            return "Would you like to tell me more about that?"
        return ""


class ChatAgent:
    """
    General conversation agent for the orchestrator.
    Provides empathetic, supportive responses for non-clinical content.
    """

    def __init__(
        self,
        settings: ChatAgentSettings | None = None,
        llm_client: UnifiedLLMClient | None = None,
    ) -> None:
        self._settings = settings or ChatAgentSettings()
        self._topic_classifier = TopicClassifier()
        self._response_generator = ResponseGenerator(self._settings)
        self._llm_client = llm_client
        self._message_count = 0

    async def process(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Process state and generate conversational response.
        This is the main LangGraph node function.

        Args:
            state: Current orchestrator state

        Returns:
            State updates dictionary
        """
        self._message_count += 1
        message = state.get("current_message", "")
        personality_style = state.get("personality_style", {})
        conversation_context = state.get("conversation_context", "")
        memory_context = state.get("memory_context", {})
        assembled_context = state.get("assembled_context", "")

        logger.info(
            "chat_agent_processing",
            message_length=len(message),
            has_personality_style=bool(personality_style),
            has_memory_context=bool(memory_context),
            has_llm=self._llm_client is not None,
        )
        topic = self._topic_classifier.classify(message)

        # Use LLM for nuanced topics, templates for formulaic ones
        _llm_topics = (
            TopicCategory.GENERAL,
            TopicCategory.SMALL_TALK,
            TopicCategory.CHECK_IN,
            TopicCategory.CLARIFICATION,
        )
        if (
            self._llm_client is not None
            and self._llm_client.is_available
            and topic.category in _llm_topics
        ):
            llm_response = await self._generate_llm_response(
                message, topic, assembled_context, conversation_context, personality_style,
            )
            if llm_response:
                warmth = personality_style.get("warmth", self._settings.default_warmth)
                return self._build_state_update(
                    ChatResponse(
                        content=llm_response,
                        tone=self._response_generator._determine_tone(topic.category, warmth),
                        topic=topic.category,
                        includes_follow_up=False,
                        empathy_applied=True,
                        warmth_level=warmth,
                    ),
                    topic,
                )

        # Fallback to template responses
        response = self._response_generator.generate(
            message=message,
            topic=topic,
            personality_style=personality_style,
            conversation_context=conversation_context,
            memory_context=memory_context,
        )
        return self._build_state_update(response, topic)

    async def _generate_llm_response(
        self,
        message: str,
        topic: TopicClassification,
        assembled_context: str,
        conversation_context: str,
        personality_style: dict[str, Any],
    ) -> str:
        """Generate response using LLM with context. Returns empty string on failure."""
        try:
            warmth = personality_style.get("warmth", self._settings.default_warmth)
            style_instruction = ""
            if warmth > 0.7:
                style_instruction = "\nUse a very warm, nurturing tone."
            elif warmth < 0.4:
                style_instruction = "\nUse a calm, professional tone."

            system_prompt = CHAT_SYSTEM_PROMPT + style_instruction
            context = assembled_context or conversation_context or None

            response = await self._llm_client.generate(
                system_prompt=system_prompt,
                user_message=message,
                context=context,
                service_name="orchestrator_chat_agent",
                task_type="therapy",
                max_tokens=self._settings.max_response_length,
            )
            if response:
                logger.info(
                    "chat_agent_llm_response_generated",
                    topic=topic.category.value,
                    length=len(response),
                )
            return response
        except Exception as e:
            logger.warning("chat_agent_llm_failed", error=str(e), topic=topic.category.value)
            return ""

    def _build_state_update(
        self,
        response: ChatResponse,
        topic: TopicClassification,
    ) -> dict[str, Any]:
        """Build state update from chat response."""
        agent_result = AgentResult(
            agent_type=AgentType.CHAT,
            success=True,
            response_content=response.content,
            confidence=topic.confidence,
            metadata={
                "topic": topic.category.value,
                "tone": response.tone.value,
                "warmth_applied": response.warmth_level,
                "empathy_applied": response.empathy_applied,
            },
        )
        logger.info(
            "chat_agent_complete",
            topic=topic.category.value,
            tone=response.tone.value,
            response_length=len(response.content),
        )
        return {
            "agent_results": [agent_result.to_dict()],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_messages": self._message_count,
            "default_warmth": self._settings.default_warmth,
            "personality_adaptation_enabled": self._settings.enable_personality_adaptation,
        }


async def chat_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for chat agent processing.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    agent = ChatAgent(llm_client=_llm_client)
    return await agent.process(state)
