"""
Solace-AI Orchestrator Service - Supervisor Agent.
Supervisor node for intent classification, agent selection, and routing decisions.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .state_schema import (
    OrchestratorState,
    IntentType,
    AgentType,
    ProcessingPhase,
    RiskLevel,
    AgentResult,
)

if TYPE_CHECKING:
    from services.shared.infrastructure.llm_client import UnifiedLLMClient

logger = structlog.get_logger(__name__)

# Module-level LLM client reference, set during orchestrator startup
_supervisor_llm_client: UnifiedLLMClient | None = None


def configure_supervisor_llm(client: UnifiedLLMClient | None) -> None:
    """Set the LLM client for the supervisor. Called during orchestrator startup."""
    global _supervisor_llm_client
    _supervisor_llm_client = client


# Valid IntentType values for LLM response parsing
_INTENT_VALUES: set[str] = {it.value for it in IntentType}

INTENT_CLASSIFICATION_PROMPT = (
    "You are an intent classifier for a mental health support platform. "
    "Classify the user's message into exactly one of these intents:\n"
    "- crisis_disclosure: Expressing suicidal thoughts, self-harm intent, or immediate danger\n"
    "- emotional_support: Seeking comfort, validation, or emotional processing\n"
    "- symptom_discussion: Discussing specific mental health symptoms or conditions\n"
    "- treatment_inquiry: Asking about therapy techniques, treatments, or what to do\n"
    "- progress_update: Sharing updates on how they're doing or feeling\n"
    "- assessment_request: Requesting a formal assessment or screening\n"
    "- coping_strategy: Seeking specific coping skills or techniques\n"
    "- psychoeducation: Asking informational questions about mental health\n"
    "- session_management: Administrative requests (scheduling, session changes)\n"
    "- general_chat: Casual conversation, greetings, or off-topic discussion\n\n"
    'Respond with ONLY valid JSON: {"intent": "<intent_value>", "confidence": <0.0-1.0>}'
)


class SupervisorSettings(BaseSettings):
    """Configuration for the supervisor agent."""
    intent_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_parallel_agents: int = Field(default=4, ge=1, le=8)
    enable_personality_adaptation: bool = Field(default=True)
    enable_diagnosis_agent: bool = Field(default=True)
    enable_therapy_agent: bool = Field(default=True)
    default_intent: str = Field(default="general_chat")
    crisis_keywords: list[str] = Field(default_factory=lambda: [
        "suicide", "kill myself", "end my life", "want to die", "self-harm",
        "hurt myself", "cutting", "overdose", "no reason to live",
    ])
    emotional_keywords: list[str] = Field(default_factory=lambda: [
        "depressed", "anxious", "scared", "lonely", "overwhelmed", "hopeless",
        "worthless", "panic", "crying", "can't cope", "stressed", "angry",
    ])
    symptom_keywords: list[str] = Field(default_factory=lambda: [
        "symptom", "feeling", "sleep", "appetite", "energy", "concentration",
        "mood", "thoughts", "behavior", "diagnosis", "condition",
    ])
    treatment_keywords: list[str] = Field(default_factory=lambda: [
        "therapy", "treatment", "medication", "exercise", "technique",
        "coping", "strategy", "help", "advice", "what can I do",
    ])
    model_config = SettingsConfigDict(env_prefix="ORCHESTRATOR_SUPERVISOR_", env_file=".env", extra="ignore")


@dataclass
class SupervisorDecision:
    """Decision output from the supervisor agent."""
    intent: IntentType
    confidence: float
    selected_agents: list[AgentType]
    routing_reason: str
    requires_safety_override: bool = False
    processing_priority: Literal["normal", "high", "critical"] = "normal"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "selected_agents": [a.value for a in self.selected_agents],
            "routing_reason": self.routing_reason,
            "requires_safety_override": self.requires_safety_override,
            "processing_priority": self.processing_priority,
            "metadata": self.metadata,
        }


class IntentClassifier:
    """Classifies user intent from message content."""

    def __init__(
        self,
        settings: SupervisorSettings,
        llm_client: UnifiedLLMClient | None = None,
    ) -> None:
        self._settings = settings
        self._llm_client = llm_client
        self._crisis_keywords = set(kw.lower() for kw in settings.crisis_keywords)
        self._emotional_keywords = set(kw.lower() for kw in settings.emotional_keywords)
        self._symptom_keywords = set(kw.lower() for kw in settings.symptom_keywords)
        self._treatment_keywords = set(kw.lower() for kw in settings.treatment_keywords)

    def classify(self, message: str, conversation_context: str = "") -> tuple[IntentType, float, list[str]]:
        """
        Classify user intent from message.

        Args:
            message: User's current message
            conversation_context: Previous conversation context

        Returns:
            Tuple of (intent_type, confidence, matched_keywords)
        """
        message_lower = message.lower()
        full_context = f"{conversation_context} {message_lower}"
        matched_keywords: list[str] = []
        if self._check_crisis_indicators(message_lower):
            keywords = [kw for kw in self._crisis_keywords if kw in message_lower]
            matched_keywords.extend(keywords)
            return IntentType.CRISIS_DISCLOSURE, 0.95, matched_keywords
        emotional_matches = self._count_keyword_matches(message_lower, self._emotional_keywords)
        symptom_matches = self._count_keyword_matches(message_lower, self._symptom_keywords)
        treatment_matches = self._count_keyword_matches(message_lower, self._treatment_keywords)
        if emotional_matches >= 2:
            keywords = [kw for kw in self._emotional_keywords if kw in message_lower]
            matched_keywords.extend(keywords)
            confidence = min(0.5 + (emotional_matches * 0.15), 0.9)
            return IntentType.EMOTIONAL_SUPPORT, confidence, matched_keywords
        if symptom_matches >= 2 or "how am i doing" in message_lower:
            keywords = [kw for kw in self._symptom_keywords if kw in message_lower]
            matched_keywords.extend(keywords)
            confidence = min(0.5 + (symptom_matches * 0.15), 0.85)
            return IntentType.SYMPTOM_DISCUSSION, confidence, matched_keywords
        if treatment_matches >= 1:
            keywords = [kw for kw in self._treatment_keywords if kw in message_lower]
            matched_keywords.extend(keywords)
            confidence = min(0.5 + (treatment_matches * 0.15), 0.85)
            return IntentType.TREATMENT_INQUIRY, confidence, matched_keywords
        if self._is_progress_related(message_lower):
            return IntentType.PROGRESS_UPDATE, 0.75, ["progress"]
        if self._is_assessment_request(message_lower):
            return IntentType.ASSESSMENT_REQUEST, 0.80, ["assessment"]
        if self._is_coping_request(message_lower):
            return IntentType.COPING_STRATEGY, 0.75, ["coping"]
        if self._is_psychoeducation_request(message_lower):
            return IntentType.PSYCHOEDUCATION, 0.70, ["education"]
        if emotional_matches == 1:
            keywords = [kw for kw in self._emotional_keywords if kw in message_lower]
            return IntentType.EMOTIONAL_SUPPORT, 0.60, keywords
        return IntentType.GENERAL_CHAT, 0.50, []

    def _check_crisis_indicators(self, message: str) -> bool:
        """Check for crisis-related content using word-boundary matching."""
        for keyword in self._crisis_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', message, re.IGNORECASE):
                return True
        return False

    def _count_keyword_matches(self, message: str, keywords: set[str]) -> int:
        """Count keyword matches in message using word-boundary matching."""
        return sum(1 for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', message, re.IGNORECASE))

    def _is_progress_related(self, message: str) -> bool:
        """Check if message is about progress tracking."""
        progress_indicators = ["progress", "better", "worse", "improved", "getting", "feeling lately"]
        return any(ind in message for ind in progress_indicators)

    def _is_assessment_request(self, message: str) -> bool:
        """Check if message is requesting an assessment."""
        assessment_indicators = ["assess", "evaluate", "check", "test", "questionnaire", "screening"]
        return any(ind in message for ind in assessment_indicators)

    def _is_coping_request(self, message: str) -> bool:
        """Check if message is requesting coping strategies."""
        coping_indicators = ["cope", "manage", "handle", "deal with", "when i feel"]
        return any(ind in message for ind in coping_indicators)

    def _is_psychoeducation_request(self, message: str) -> bool:
        """Check if message is asking for educational content."""
        edu_indicators = ["what is", "explain", "tell me about", "how does", "why do i"]
        return any(ind in message for ind in edu_indicators)

    async def refine_with_llm(
        self,
        message: str,
        keyword_intent: IntentType,
        keyword_confidence: float,
    ) -> tuple[IntentType, float]:
        """Refine intent classification using LLM when keyword confidence is low."""
        if self._llm_client is None or not self._llm_client.is_available:
            return keyword_intent, keyword_confidence
        try:
            response = await self._llm_client.generate(
                system_prompt=INTENT_CLASSIFICATION_PROMPT,
                user_message=message,
                service_name="orchestrator_supervisor",
                task_type="structured",
                max_tokens=100,
            )
            if not response:
                return keyword_intent, keyword_confidence

            parsed = json.loads(response.strip())
            llm_intent_str = parsed.get("intent", "")
            llm_confidence = float(parsed.get("confidence", 0.5))

            if llm_intent_str not in _INTENT_VALUES:
                logger.debug("llm_intent_unknown_value", value=llm_intent_str)
                return keyword_intent, keyword_confidence

            llm_intent = IntentType(llm_intent_str)

            # LLM agrees with keyword classifier - boost confidence
            if llm_intent == keyword_intent:
                boosted = min(max(keyword_confidence, llm_confidence) + 0.1, 1.0)
                logger.info(
                    "llm_intent_confirmed",
                    intent=keyword_intent.value,
                    boosted_confidence=boosted,
                )
                return keyword_intent, boosted

            # LLM disagrees - use LLM only if substantially more confident
            if llm_confidence > keyword_confidence + 0.15:
                logger.info(
                    "llm_intent_override",
                    keyword_intent=keyword_intent.value,
                    llm_intent=llm_intent.value,
                    llm_confidence=llm_confidence,
                )
                return llm_intent, llm_confidence

            return keyword_intent, keyword_confidence
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("llm_intent_parse_failed", error=str(e))
            return keyword_intent, keyword_confidence
        except Exception as e:
            logger.warning("llm_intent_classification_failed", error=str(e))
            return keyword_intent, keyword_confidence


class AgentRouter:
    """Routes requests to appropriate agents based on intent and context."""

    def __init__(self, settings: SupervisorSettings) -> None:
        self._settings = settings

    def select_agents(
        self,
        intent: IntentType,
        safety_flags: dict[str, Any],
        has_active_treatment: bool = False,
    ) -> tuple[list[AgentType], str]:
        """
        Select agents to handle the request.

        Args:
            intent: Classified user intent
            safety_flags: Current safety assessment
            has_active_treatment: Whether user has an active treatment plan

        Returns:
            Tuple of (selected agents list, routing reason)
        """
        if safety_flags.get("crisis_detected") or safety_flags.get("risk_level") in ("HIGH", "CRITICAL"):
            return [AgentType.SAFETY], "Crisis detected - routing to safety agent exclusively"
        agent_mapping: dict[IntentType, tuple[list[AgentType], str]] = {
            IntentType.CRISIS_DISCLOSURE: (
                [AgentType.SAFETY],
                "Crisis disclosure - immediate safety assessment required"
            ),
            IntentType.EMOTIONAL_SUPPORT: (
                [AgentType.THERAPY, AgentType.PERSONALITY] if self._settings.enable_therapy_agent else [AgentType.CHAT, AgentType.PERSONALITY],
                "Emotional support - therapy agent with personality adaptation"
            ),
            IntentType.SYMPTOM_DISCUSSION: (
                [AgentType.DIAGNOSIS, AgentType.THERAPY] if self._settings.enable_diagnosis_agent else [AgentType.CHAT],
                "Symptom discussion - diagnosis assessment with therapy support"
            ),
            IntentType.TREATMENT_INQUIRY: (
                [AgentType.THERAPY, AgentType.PERSONALITY] if self._settings.enable_therapy_agent else [AgentType.CHAT],
                "Treatment inquiry - therapy agent for intervention guidance"
            ),
            IntentType.PROGRESS_UPDATE: (
                [AgentType.THERAPY, AgentType.DIAGNOSIS] if has_active_treatment else [AgentType.CHAT],
                "Progress update - therapy and diagnosis agents for tracking"
            ),
            IntentType.ASSESSMENT_REQUEST: (
                [AgentType.DIAGNOSIS] if self._settings.enable_diagnosis_agent else [AgentType.CHAT],
                "Assessment request - diagnosis agent for formal assessment"
            ),
            IntentType.COPING_STRATEGY: (
                [AgentType.THERAPY, AgentType.PERSONALITY] if self._settings.enable_therapy_agent else [AgentType.CHAT],
                "Coping strategy - therapy agent for evidence-based techniques"
            ),
            IntentType.PSYCHOEDUCATION: (
                [AgentType.THERAPY, AgentType.DIAGNOSIS],
                "Psychoeducation - combined clinical education"
            ),
            IntentType.SESSION_MANAGEMENT: (
                [AgentType.CHAT],
                "Session management - general chat handler"
            ),
            IntentType.GENERAL_CHAT: (
                [AgentType.CHAT, AgentType.PERSONALITY] if self._settings.enable_personality_adaptation else [AgentType.CHAT],
                "General chat - conversational agent with personality"
            ),
        }
        agents, reason = agent_mapping.get(intent, ([AgentType.CHAT], "Default routing to chat agent"))
        if self._settings.enable_personality_adaptation and AgentType.PERSONALITY not in agents:
            if len(agents) < self._settings.max_parallel_agents:
                agents = agents + [AgentType.PERSONALITY]
        return agents[:self._settings.max_parallel_agents], reason


class SupervisorAgent:
    """
    Supervisor agent for orchestrating multi-agent processing.
    Handles intent classification, agent routing, and coordination.
    """

    def __init__(
        self,
        settings: SupervisorSettings | None = None,
        llm_client: UnifiedLLMClient | None = None,
    ) -> None:
        self._settings = settings or SupervisorSettings()
        self._intent_classifier = IntentClassifier(self._settings, llm_client=llm_client)
        self._agent_router = AgentRouter(self._settings)
        self._decision_count = 0

    async def process(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Process state and make supervisor decision.
        This is the main LangGraph node function.

        Args:
            state: Current orchestrator state

        Returns:
            State updates dictionary
        """
        self._decision_count += 1
        message = state.get("current_message", "")
        context = state.get("conversation_context", "")
        safety_flags = state.get("safety_flags", {})
        active_treatment = state.get("active_treatment")
        logger.info(
            "supervisor_processing",
            message_length=len(message),
            has_context=bool(context),
            safety_risk=safety_flags.get("risk_level", "NONE"),
        )
        decision = await self.make_decision(message, context, safety_flags, active_treatment)
        agent_result = AgentResult(
            agent_type=AgentType.SUPERVISOR,
            success=True,
            response_content=None,
            confidence=decision.confidence,
            processing_time_ms=0.0,
            metadata=decision.to_dict(),
        )
        logger.info(
            "supervisor_decision_made",
            intent=decision.intent.value,
            confidence=decision.confidence,
            selected_agents=[a.value for a in decision.selected_agents],
            routing_reason=decision.routing_reason,
        )
        return {
            "intent": decision.intent.value,
            "intent_confidence": decision.confidence,
            "selected_agents": [a.value for a in decision.selected_agents],
            "processing_phase": ProcessingPhase.AGENT_ROUTING.value,
            "agent_results": [agent_result.to_dict()],
            "metadata": {**state.get("metadata", {}), "supervisor_decision": decision.to_dict()},
        }

    async def make_decision(
        self,
        message: str,
        conversation_context: str = "",
        safety_flags: dict[str, Any] | None = None,
        active_treatment: dict[str, Any] | None = None,
    ) -> SupervisorDecision:
        """
        Make a routing decision for the given message.

        Args:
            message: User's current message
            conversation_context: Previous conversation context
            safety_flags: Current safety assessment
            active_treatment: Active treatment plan if any

        Returns:
            SupervisorDecision with routing information
        """
        safety_flags = safety_flags or {}
        intent, confidence, matched_keywords = self._intent_classifier.classify(message, conversation_context)

        # Refine with LLM when keyword confidence is below threshold
        if confidence < self._settings.intent_confidence_threshold:
            intent, confidence = await self._intent_classifier.refine_with_llm(
                message, intent, confidence,
            )

        has_active_treatment = active_treatment is not None
        selected_agents, routing_reason = self._agent_router.select_agents(
            intent, safety_flags, has_active_treatment
        )
        requires_safety_override = (
            safety_flags.get("crisis_detected", False) or
            safety_flags.get("risk_level") in ("HIGH", "CRITICAL") or
            intent == IntentType.CRISIS_DISCLOSURE
        )
        if requires_safety_override:
            priority = "critical"
        elif intent in (IntentType.EMOTIONAL_SUPPORT, IntentType.SYMPTOM_DISCUSSION):
            priority = "high"
        else:
            priority = "normal"
        return SupervisorDecision(
            intent=intent,
            confidence=confidence,
            selected_agents=selected_agents,
            routing_reason=routing_reason,
            requires_safety_override=requires_safety_override,
            processing_priority=priority,
            metadata={"matched_keywords": matched_keywords, "decision_number": self._decision_count},
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get supervisor agent statistics."""
        return {
            "total_decisions": self._decision_count,
            "settings": {
                "confidence_threshold": self._settings.intent_confidence_threshold,
                "max_parallel_agents": self._settings.max_parallel_agents,
                "personality_enabled": self._settings.enable_personality_adaptation,
            },
        }


async def supervisor_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for supervisor processing.
    Creates a SupervisorAgent and processes the state.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    supervisor = SupervisorAgent(llm_client=_supervisor_llm_client)
    return await supervisor.process(state)
