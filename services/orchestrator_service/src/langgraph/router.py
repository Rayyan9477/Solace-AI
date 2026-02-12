"""
Solace-AI Orchestrator Service - Router.
Intent classification and agent routing for multi-agent orchestration.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
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

logger = structlog.get_logger(__name__)


class RoutingStrategy(str, Enum):
    """Strategies for agent routing."""
    SINGLE = "single"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    FALLBACK = "fallback"


class RouterSettings(BaseSettings):
    """Configuration for the router."""
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_parallel_routing: bool = Field(default=True)
    max_parallel_agents: int = Field(default=4, ge=1, le=8)
    fallback_agent: str = Field(default="chat")
    enable_personality: bool = Field(default=True)
    enable_diagnosis: bool = Field(default=True)
    enable_therapy: bool = Field(default=True)
    safety_always_first: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_ROUTER_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class RouteDecision:
    """Result of routing decision."""
    primary_agents: list[AgentType]
    secondary_agents: list[AgentType]
    strategy: RoutingStrategy
    reasoning: str
    priority: Literal["normal", "high", "critical"]
    requires_safety: bool = True
    estimated_agents_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_agents": [a.value for a in self.primary_agents],
            "secondary_agents": [a.value for a in self.secondary_agents],
            "strategy": self.strategy.value,
            "reasoning": self.reasoning,
            "priority": self.priority,
            "requires_safety": self.requires_safety,
            "estimated_agents_count": self.estimated_agents_count,
            "metadata": self.metadata,
        }

    def all_agents(self) -> list[AgentType]:
        """Get all agents in routing order."""
        return self.primary_agents + self.secondary_agents


@dataclass
class IntentAnalysis:
    """Analysis of user intent from message."""
    intent: IntentType
    confidence: float
    matched_patterns: list[str]
    emotional_indicators: list[str]
    risk_indicators: list[str]
    topic_signals: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "emotional_indicators": self.emotional_indicators,
            "risk_indicators": self.risk_indicators,
            "topic_signals": self.topic_signals,
        }


class IntentAnalyzer:
    """Analyzes user messages for intent classification."""

    CRISIS_PATTERNS: list[str] = [
        "suicide", "kill myself", "end my life", "want to die",
        "self-harm", "hurt myself", "no reason to live", "end it all",
    ]
    EMOTIONAL_PATTERNS: list[str] = [
        "depressed", "anxious", "scared", "lonely", "overwhelmed",
        "hopeless", "worthless", "panic", "crying", "stressed",
    ]
    SYMPTOM_PATTERNS: list[str] = [
        "symptom", "sleep", "appetite", "energy", "concentration",
        "mood", "thoughts", "diagnosis", "feeling lately",
    ]
    TREATMENT_PATTERNS: list[str] = [
        "therapy", "treatment", "medication", "technique",
        "coping", "strategy", "help me", "advice",
    ]
    EDUCATION_PATTERNS: list[str] = [
        "what is", "explain", "tell me about", "how does", "why do",
    ]
    ASSESSMENT_PATTERNS: list[str] = [
        "assess", "evaluate", "screening", "questionnaire", "test",
    ]

    def analyze(self, message: str, context: str = "") -> IntentAnalysis:
        """Analyze message for intent classification."""
        message_lower = message.lower()
        context_lower = context.lower() if context else ""
        full_text = f"{context_lower} {message_lower}"
        matched_patterns: list[str] = []
        emotional_indicators: list[str] = []
        risk_indicators: list[str] = []
        topic_signals: list[str] = []
        for pattern in self.CRISIS_PATTERNS:
            if pattern in message_lower:
                risk_indicators.append(pattern)
        if risk_indicators:
            matched_patterns.extend(risk_indicators)
            return IntentAnalysis(
                intent=IntentType.CRISIS_DISCLOSURE,
                confidence=0.95,
                matched_patterns=matched_patterns,
                emotional_indicators=emotional_indicators,
                risk_indicators=risk_indicators,
                topic_signals=topic_signals,
            )
        for pattern in self.EMOTIONAL_PATTERNS:
            if pattern in message_lower:
                emotional_indicators.append(pattern)
        for pattern in self.SYMPTOM_PATTERNS:
            if pattern in full_text:
                topic_signals.append(f"symptom:{pattern}")
        for pattern in self.TREATMENT_PATTERNS:
            if pattern in message_lower:
                topic_signals.append(f"treatment:{pattern}")
        for pattern in self.EDUCATION_PATTERNS:
            if pattern in message_lower:
                topic_signals.append(f"education:{pattern}")
        for pattern in self.ASSESSMENT_PATTERNS:
            if pattern in message_lower:
                topic_signals.append(f"assessment:{pattern}")
        intent, confidence = self._determine_intent(
            emotional_indicators, topic_signals, message_lower
        )
        matched_patterns = emotional_indicators + [t.split(":")[1] for t in topic_signals]
        return IntentAnalysis(
            intent=intent,
            confidence=confidence,
            matched_patterns=matched_patterns,
            emotional_indicators=emotional_indicators,
            risk_indicators=risk_indicators,
            topic_signals=topic_signals,
        )

    def _determine_intent(self, emotional: list[str], topics: list[str], message: str) -> tuple[IntentType, float]:
        """Determine intent from indicators."""
        counts = {k: sum(1 for t in topics if t.startswith(f"{k}:")) for k in ["symptom", "treatment", "education", "assessment"]}
        if len(emotional) >= 2:
            return IntentType.EMOTIONAL_SUPPORT, min(0.5 + len(emotional) * 0.15, 0.90)
        if counts["assessment"] > 0:
            return IntentType.ASSESSMENT_REQUEST, 0.80
        if counts["symptom"] >= 2:
            return IntentType.SYMPTOM_DISCUSSION, min(0.5 + counts["symptom"] * 0.15, 0.85)
        if counts["treatment"] >= 1:
            return IntentType.TREATMENT_INQUIRY, min(0.5 + counts["treatment"] * 0.15, 0.85)
        if counts["education"] >= 1:
            return IntentType.PSYCHOEDUCATION, 0.75
        if len(emotional) == 1:
            return IntentType.EMOTIONAL_SUPPORT, 0.65
        if any(p in message for p in ["cope with", "manage", "handle", "deal with", "when i feel"]):
            return IntentType.COPING_STRATEGY, 0.75
        if any(p in message for p in ["progress", "better", "worse", "improved", "lately"]):
            return IntentType.PROGRESS_UPDATE, 0.70
        return IntentType.GENERAL_CHAT, 0.50


class AgentSelector:
    """Selects appropriate agents based on intent and context."""

    def __init__(self, settings: RouterSettings) -> None:
        self._settings = settings

    def select(
        self,
        analysis: IntentAnalysis,
        safety_flags: dict[str, Any],
        has_treatment: bool = False,
    ) -> RouteDecision:
        """Select agents for processing."""
        if self._requires_immediate_safety(safety_flags, analysis):
            return RouteDecision(
                primary_agents=[AgentType.SAFETY],
                secondary_agents=[],
                strategy=RoutingStrategy.SINGLE,
                reasoning="Crisis detected - immediate safety response required",
                priority="critical",
                requires_safety=True,
                estimated_agents_count=1,
                metadata={"crisis_triggered": True},
            )
        primary, secondary, reasoning = self._map_intent_to_agents(
            analysis.intent, has_treatment
        )
        if self._settings.enable_personality and AgentType.PERSONALITY not in primary:
            if len(primary) + len(secondary) < self._settings.max_parallel_agents:
                secondary.append(AgentType.PERSONALITY)
        strategy = self._determine_strategy(primary, secondary)
        priority = self._determine_priority(analysis.intent, analysis.confidence)
        return RouteDecision(
            primary_agents=primary,
            secondary_agents=secondary,
            strategy=strategy,
            reasoning=reasoning,
            priority=priority,
            requires_safety=True,
            estimated_agents_count=len(primary) + len(secondary),
            metadata={"intent_analysis": analysis.to_dict()},
        )

    def _requires_immediate_safety(self, safety_flags: dict[str, Any], analysis: IntentAnalysis) -> bool:
        """Check if immediate safety response is required."""
        return (safety_flags.get("crisis_detected") or safety_flags.get("risk_level") in ("HIGH", "CRITICAL") or
                analysis.intent == IntentType.CRISIS_DISCLOSURE)

    def _map_intent_to_agents(
        self,
        intent: IntentType,
        has_treatment: bool,
    ) -> tuple[list[AgentType], list[AgentType], str]:
        """Map intent to agent lists."""
        mappings: dict[IntentType, tuple[list[AgentType], list[AgentType], str]] = {
            IntentType.CRISIS_DISCLOSURE: (
                [AgentType.SAFETY],
                [],
                "Crisis disclosure - safety agent primary",
            ),
            IntentType.EMOTIONAL_SUPPORT: (
                [AgentType.THERAPY] if self._settings.enable_therapy else [AgentType.CHAT],
                [AgentType.PERSONALITY],
                "Emotional support - therapy with personality adaptation",
            ),
            IntentType.SYMPTOM_DISCUSSION: (
                [AgentType.DIAGNOSIS] if self._settings.enable_diagnosis else [AgentType.CHAT],
                [AgentType.THERAPY] if self._settings.enable_therapy else [],
                "Symptom discussion - diagnosis with therapy support",
            ),
            IntentType.TREATMENT_INQUIRY: (
                [AgentType.THERAPY] if self._settings.enable_therapy else [AgentType.CHAT],
                [AgentType.PERSONALITY],
                "Treatment inquiry - therapy for intervention guidance",
            ),
            IntentType.PROGRESS_UPDATE: (
                [AgentType.THERAPY, AgentType.DIAGNOSIS] if has_treatment else [AgentType.CHAT],
                [],
                "Progress update - therapy and diagnosis tracking",
            ),
            IntentType.ASSESSMENT_REQUEST: (
                [AgentType.DIAGNOSIS] if self._settings.enable_diagnosis else [AgentType.CHAT],
                [],
                "Assessment request - formal diagnosis assessment",
            ),
            IntentType.COPING_STRATEGY: (
                [AgentType.THERAPY] if self._settings.enable_therapy else [AgentType.CHAT],
                [AgentType.PERSONALITY],
                "Coping strategy - evidence-based techniques",
            ),
            IntentType.PSYCHOEDUCATION: (
                [AgentType.THERAPY, AgentType.DIAGNOSIS],
                [],
                "Psychoeducation - combined clinical education",
            ),
            IntentType.SESSION_MANAGEMENT: (
                [AgentType.CHAT],
                [],
                "Session management - general handler",
            ),
            IntentType.GENERAL_CHAT: (
                [AgentType.CHAT],
                [AgentType.PERSONALITY],
                "General chat - conversational with personality",
            ),
        }
        return mappings.get(intent, ([AgentType.CHAT], [], "Default routing"))

    def _determine_strategy(self, primary: list[AgentType], secondary: list[AgentType]) -> RoutingStrategy:
        """Determine routing strategy."""
        total = len(primary) + len(secondary)
        if total == 1:
            return RoutingStrategy.SINGLE
        return RoutingStrategy.PARALLEL if self._settings.enable_parallel_routing and total <= self._settings.max_parallel_agents else RoutingStrategy.SEQUENTIAL

    def _determine_priority(self, intent: IntentType, confidence: float) -> Literal["normal", "high", "critical"]:
        """Determine processing priority."""
        if intent == IntentType.CRISIS_DISCLOSURE:
            return "critical"
        return "high" if intent in (IntentType.EMOTIONAL_SUPPORT, IntentType.SYMPTOM_DISCUSSION) else "normal"


class Router:
    """Routes messages to appropriate agents based on intent and context."""

    def __init__(self, settings: RouterSettings | None = None) -> None:
        self._settings = settings or RouterSettings()
        self._analyzer = IntentAnalyzer()
        self._selector = AgentSelector(self._settings)
        self._route_count = 0

    def route(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Route message to appropriate agents.
        Main LangGraph node function.
        """
        self._route_count += 1
        message = state.get("current_message", "")
        context = state.get("conversation_context", "")
        safety_flags = state.get("safety_flags", {})
        has_treatment = state.get("active_treatment") is not None
        logger.info(
            "router_processing",
            message_length=len(message),
            has_context=bool(context),
            safety_risk=safety_flags.get("risk_level", "NONE"),
        )
        analysis = self._analyzer.analyze(message, context)
        decision = self._selector.select(analysis, safety_flags, has_treatment)
        agent_result = AgentResult(
            agent_type=AgentType.SUPERVISOR,
            success=True,
            confidence=analysis.confidence,
            metadata={
                "routing_decision": decision.to_dict(),
                "route_number": self._route_count,
            },
        )
        logger.info(
            "router_decision_made",
            intent=analysis.intent.value,
            confidence=analysis.confidence,
            primary_agents=[a.value for a in decision.primary_agents],
            strategy=decision.strategy.value,
        )
        return {
            "intent": analysis.intent.value,
            "intent_confidence": analysis.confidence,
            "selected_agents": [a.value for a in decision.all_agents()],
            "processing_phase": ProcessingPhase.AGENT_ROUTING.value,
            "agent_results": [agent_result.to_dict()],
            "metadata": {
                **state.get("metadata", {}),
                "routing": decision.to_dict(),
            },
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "total_routes": self._route_count,
            "settings": {
                "confidence_threshold": self._settings.confidence_threshold,
                "max_parallel_agents": self._settings.max_parallel_agents,
                "parallel_enabled": self._settings.enable_parallel_routing,
            },
        }


def router_node(state: OrchestratorState) -> dict[str, Any]:
    """LangGraph node function for routing."""
    return Router().route(state)
