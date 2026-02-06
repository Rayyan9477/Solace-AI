"""
Solace-AI Orchestrator Service - Therapy Agent.
Coordinates with the Therapy Service for evidence-based therapeutic interventions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..langgraph.state_schema import (
    OrchestratorState,
    AgentType,
    AgentResult,
)

logger = structlog.get_logger(__name__)


class SessionPhase(str, Enum):
    """Phases in a therapy session."""
    PRE_SESSION = "PRE_SESSION"
    OPENING = "OPENING"
    WORKING = "WORKING"
    CLOSING = "CLOSING"
    POST_SESSION = "POST_SESSION"
    CRISIS = "CRISIS"


class TherapyModality(str, Enum):
    """Therapeutic modalities available."""
    CBT = "CBT"
    DBT = "DBT"
    ACT = "ACT"
    MI = "MI"
    MINDFULNESS = "MINDFULNESS"
    SFBT = "SFBT"
    PSYCHOEDUCATION = "PSYCHOEDUCATION"


class TechniqueCategory(str, Enum):
    """Categories of therapeutic techniques."""
    COGNITIVE_RESTRUCTURING = "COGNITIVE_RESTRUCTURING"
    BEHAVIORAL_ACTIVATION = "BEHAVIORAL_ACTIVATION"
    EXPOSURE = "EXPOSURE"
    MINDFULNESS_SKILL = "MINDFULNESS_SKILL"
    DISTRESS_TOLERANCE = "DISTRESS_TOLERANCE"
    EMOTION_REGULATION = "EMOTION_REGULATION"
    INTERPERSONAL = "INTERPERSONAL"
    RELAXATION = "RELAXATION"


class TherapyAgentSettings(BaseSettings):
    """Configuration for the therapy agent."""
    service_url: str = Field(default="http://localhost:8003")
    timeout_seconds: float = Field(default=15.0, ge=1.0, le=60.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    fallback_on_service_error: bool = Field(default=True)
    default_modality: str = Field(default="CBT")
    enable_homework: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_THERAPY_AGENT_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class TechniqueDTO:
    """Therapeutic technique data."""
    technique_id: str
    name: str
    modality: TherapyModality
    category: TechniqueCategory
    description: str
    duration_minutes: int = 15
    requires_homework: bool = False
    contraindications: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TechniqueDTO:
        return cls(technique_id=data.get("technique_id", str(uuid4())), name=data.get("name", ""), modality=TherapyModality(data.get("modality", "CBT")), category=TechniqueCategory(data.get("category", "COGNITIVE_RESTRUCTURING")), description=data.get("description", ""), duration_minutes=data.get("duration_minutes", 15), requires_homework=data.get("requires_homework", False), contraindications=data.get("contraindications", []))

    def to_dict(self) -> dict[str, Any]:
        return {"technique_id": self.technique_id, "name": self.name, "modality": self.modality.value, "category": self.category.value, "description": self.description, "duration_minutes": self.duration_minutes, "requires_homework": self.requires_homework, "contraindications": self.contraindications}


@dataclass
class HomeworkDTO:
    """Homework assignment data."""
    homework_id: str
    title: str
    description: str
    technique_id: str
    due_date: str | None = None
    completed: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HomeworkDTO:
        return cls(homework_id=data.get("homework_id", str(uuid4())), title=data.get("title", ""), description=data.get("description", ""), technique_id=data.get("technique_id", ""), due_date=data.get("due_date"), completed=data.get("completed", False))

    def to_dict(self) -> dict[str, Any]:
        return {"homework_id": self.homework_id, "title": self.title, "description": self.description, "technique_id": self.technique_id, "due_date": self.due_date, "completed": self.completed}


@dataclass
class TherapyResponse:
    """Response from therapy service."""
    response_text: str
    current_phase: SessionPhase
    technique_applied: TechniqueDTO | None
    homework_assigned: list[HomeworkDTO]
    safety_alerts: list[str]
    next_steps: list[str]
    processing_time_ms: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TherapyResponse:
        technique_data = data.get("technique_applied")
        return cls(response_text=data.get("response_text", ""), current_phase=SessionPhase(data.get("current_phase", "WORKING")), technique_applied=TechniqueDTO.from_dict(technique_data) if technique_data else None, homework_assigned=[HomeworkDTO.from_dict(h) for h in data.get("homework_assigned", [])], safety_alerts=data.get("safety_alerts", []), next_steps=data.get("next_steps", []), processing_time_ms=float(data.get("processing_time_ms", 0.0)))


class TherapyServiceClient:
    """HTTP client for Therapy Service communication."""

    def __init__(self, settings: TherapyAgentSettings) -> None:
        self._settings = settings
        self._base_url = settings.service_url.rstrip("/")

    async def process_message(
        self,
        session_id: str,
        user_id: str,
        message: str,
        conversation_history: list[dict[str, Any]],
    ) -> TherapyResponse:
        """Process a message in the therapy session."""
        url = f"{self._base_url}/sessions/{session_id}/message"
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            "conversation_history": conversation_history,
        }
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return TherapyResponse.from_dict(response.json())
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "therapy_service_http_error",
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
                except httpx.RequestError as e:
                    logger.warning(
                        "therapy_service_request_error",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
        raise RuntimeError("Therapy service request failed after retries")

    async def get_techniques(self, modality: str | None = None) -> list[TechniqueDTO]:
        """Get available therapeutic techniques."""
        url = f"{self._base_url}/techniques"
        params = {"modality": modality} if modality else {}
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [TechniqueDTO.from_dict(t) for t in data.get("techniques", [])]


class TherapyAgent:
    """
    Therapy coordination agent for the orchestrator.
    Coordinates with Therapy Service for evidence-based interventions.
    """

    def __init__(self, settings: TherapyAgentSettings | None = None) -> None:
        self._settings = settings or TherapyAgentSettings()
        self._client = TherapyServiceClient(self._settings)
        self._session_count = 0

    async def process(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Process state and provide therapeutic response.
        This is the main LangGraph node function.

        Args:
            state: Current orchestrator state

        Returns:
            State updates dictionary
        """
        self._session_count += 1
        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        message = state.get("current_message", "")
        messages = state.get("messages", [])
        intent = state.get("intent", "general_chat")
        active_treatment = state.get("active_treatment")
        memory_context = state.get("memory_context", {})
        assembled_context = state.get("assembled_context", "")
        
        logger.info(
            "therapy_agent_processing",
            user_id=user_id,
            message_length=len(message),
            intent=intent,
            has_memory_context=bool(memory_context),
            context_tokens=memory_context.get("total_tokens", 0) if memory_context else 0,
        )
        try:
            result = await self._process_therapy_request(
                user_id=user_id,
                session_id=session_id,
                message=message,
                messages=messages,
                intent=intent,
                active_treatment=active_treatment,
                assembled_context=assembled_context,
            )
            return self._build_state_update(result)
        except Exception as e:
            logger.error("therapy_agent_error", error=str(e))
            if self._settings.fallback_on_service_error:
                return self._build_fallback_response(message, intent)
            raise

    async def _process_therapy_request(
        self,
        user_id: str,
        session_id: str,
        message: str,
        messages: list[dict[str, Any]],
        intent: str,
        active_treatment: dict[str, Any] | None,
        assembled_context: str = "",
    ) -> TherapyResponse:
        """Process therapy request via Therapy Service."""
        # Note: assembled_context from memory can be logged or used for future enhancements
        # Currently passing to service via conversation_history
        if assembled_context:
            logger.debug(
                "therapy_request_with_memory_context",
                context_length=len(assembled_context),
            )
        
        return await self._client.process_message(
            session_id=session_id,
            user_id=user_id,
            message=message,
            conversation_history=messages[-10:] if messages else [],
        )

    def _build_state_update(self, result: TherapyResponse) -> dict[str, Any]:
        """Build state update from therapy response."""
        technique_data = result.technique_applied.to_dict() if result.technique_applied else None
        homework_data = [h.to_dict() for h in result.homework_assigned]
        agent_result = AgentResult(
            agent_type=AgentType.THERAPY,
            success=True,
            response_content=result.response_text,
            confidence=0.85,
            processing_time_ms=result.processing_time_ms,
            metadata={
                "session_phase": result.current_phase.value,
                "technique_applied": technique_data,
                "homework_count": len(result.homework_assigned),
                "safety_alerts": result.safety_alerts,
                "next_steps": result.next_steps,
            },
        )
        logger.info(
            "therapy_agent_complete",
            phase=result.current_phase.value,
            technique=technique_data.get("name") if technique_data else None,
            homework_assigned=len(result.homework_assigned),
        )
        state_update: dict[str, Any] = {
            "agent_results": [agent_result.to_dict()],
        }
        if homework_data:
            state_update["metadata"] = {
                "assigned_homework": homework_data,
            }
        return state_update

    def _build_fallback_response(self, message: str, intent: str) -> dict[str, Any]:
        """Build fallback response when Therapy Service is unavailable."""
        response = self._generate_fallback_content(message, intent)
        agent_result = AgentResult(
            agent_type=AgentType.THERAPY,
            success=True,
            response_content=response,
            confidence=0.6,
            metadata={"fallback_mode": True, "intent": intent},
        )
        logger.warning("therapy_agent_fallback_used", intent=intent)
        return {
            "agent_results": [agent_result.to_dict()],
        }

    def _generate_fallback_content(self, message: str, intent: str) -> str:
        """Generate fallback therapeutic content based on intent."""
        message_lower = message.lower()
        if intent == "emotional_support":
            return self._emotional_support_response(message_lower)
        if intent == "coping_strategy":
            return self._coping_strategy_response(message_lower)
        if intent == "treatment_inquiry":
            return self._treatment_inquiry_response(message_lower)
        return self._general_therapy_response(message_lower)

    def _emotional_support_response(self, message: str) -> str:
        """Generate emotional support response."""
        if any(word in message for word in ["anxious", "worried", "scared"]):
            return (
                "It sounds like you're experiencing some anxiety right now. "
                "That's a really difficult feeling to sit with. One thing that can help "
                "is to take a moment to ground yourself - notice five things you can see "
                "around you right now. This can help bring you back to the present moment."
            )
        if any(word in message for word in ["sad", "down", "depressed"]):
            return (
                "I hear that you're feeling down right now. Those feelings are valid, "
                "and it takes courage to acknowledge them. Sometimes when we feel low, "
                "even small acts of self-care can make a difference. "
                "Is there something gentle you could do for yourself today?"
            )
        return (
            "It sounds like you're going through a difficult time. "
            "I want you to know that your feelings are valid and you don't have to "
            "face this alone. What matters most to you right now?"
        )

    def _coping_strategy_response(self, message: str) -> str:
        """Generate coping strategy response."""
        if any(word in message for word in ["stress", "overwhelmed"]):
            return (
                "When feeling overwhelmed, it can help to break things down into smaller, "
                "manageable steps. Try this: write down one small thing you can do right now, "
                "just one step. Sometimes taking that first small action can help reduce "
                "the feeling of being overwhelmed."
            )
        return (
            "Building coping skills takes practice, and it's great that you're thinking "
            "about this. One evidence-based technique is the STOP skill: Stop what you're "
            "doing, Take a step back, Observe your thoughts and feelings, and then "
            "Proceed mindfully. Would you like to try this together?"
        )

    def _treatment_inquiry_response(self, message: str) -> str:
        """Generate treatment inquiry response."""
        return (
            "There are several evidence-based approaches that can help with what you're "
            "experiencing. Cognitive Behavioral Therapy (CBT) focuses on the connection "
            "between thoughts, feelings, and behaviors. Mindfulness-based approaches help "
            "you relate differently to difficult thoughts and feelings. "
            "Would you like to explore any of these approaches further?"
        )

    def _general_therapy_response(self, message: str) -> str:
        """Generate general therapeutic response."""
        return (
            "Thank you for sharing that with me. It takes courage to open up about "
            "what you're experiencing. I'm here to support you in exploring these "
            "feelings and finding ways to move forward. "
            "What feels most important to focus on right now?"
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_sessions": self._session_count,
            "service_url": self._settings.service_url,
            "homework_enabled": self._settings.enable_homework,
            "default_modality": self._settings.default_modality,
        }


async def therapy_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for therapy agent processing.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    agent = TherapyAgent()
    return await agent.process(state)
