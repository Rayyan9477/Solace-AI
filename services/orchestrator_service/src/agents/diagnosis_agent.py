"""
Solace-AI Orchestrator Service - Diagnosis Agent.
Coordinates with the Diagnosis Service for symptom assessment and differential diagnosis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
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


class DiagnosisPhase(str, Enum):
    """Phases in the diagnosis conversation."""
    RAPPORT = "RAPPORT"
    HISTORY = "HISTORY"
    ASSESSMENT = "ASSESSMENT"
    DIAGNOSIS = "DIAGNOSIS"
    CLOSURE = "CLOSURE"


class SymptomType(str, Enum):
    """Types of symptoms."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    COGNITIVE = "COGNITIVE"
    BEHAVIORAL = "BEHAVIORAL"
    SOMATIC = "SOMATIC"
    EMOTIONAL = "EMOTIONAL"


class SeverityLevel(str, Enum):
    """Severity levels for symptoms and diagnoses."""
    MINIMAL = "MINIMAL"
    MILD = "MILD"
    MODERATE = "MODERATE"
    MODERATELY_SEVERE = "MODERATELY_SEVERE"
    SEVERE = "SEVERE"


class DiagnosisAgentSettings(BaseSettings):
    """Configuration for the diagnosis agent."""
    service_url: str = Field(default="http://localhost:8002")
    timeout_seconds: float = Field(default=15.0, ge=1.0, le=60.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    enable_differential: bool = Field(default=True)
    enable_reasoning_chain: bool = Field(default=True)
    fallback_on_service_error: bool = Field(default=True)
    min_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_DIAGNOSIS_AGENT_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class SymptomDTO:
    """Symptom data from diagnosis assessment."""
    symptom_id: str
    name: str
    description: str
    symptom_type: SymptomType
    severity: SeverityLevel
    confidence: float
    onset: str | None = None
    duration: str | None = None
    frequency: str | None = None
    triggers: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SymptomDTO:
        """Create from dictionary."""
        return cls(
            symptom_id=data.get("symptom_id", str(uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            symptom_type=SymptomType(data.get("symptom_type", "EMOTIONAL")),
            severity=SeverityLevel(data.get("severity", "MILD")),
            confidence=float(data.get("confidence", 0.5)),
            onset=data.get("onset"),
            duration=data.get("duration"),
            frequency=data.get("frequency"),
            triggers=data.get("triggers", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symptom_id": self.symptom_id,
            "name": self.name,
            "description": self.description,
            "symptom_type": self.symptom_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "onset": self.onset,
            "duration": self.duration,
            "frequency": self.frequency,
            "triggers": self.triggers,
        }


@dataclass
class HypothesisDTO:
    """Diagnostic hypothesis from differential diagnosis."""
    hypothesis_id: str
    name: str
    confidence: float
    dsm5_code: str | None = None
    icd11_code: str | None = None
    criteria_met: list[str] = field(default_factory=list)
    criteria_missing: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    contra_evidence: list[str] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MILD

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HypothesisDTO:
        """Create from dictionary."""
        return cls(
            hypothesis_id=data.get("hypothesis_id", str(uuid4())),
            name=data.get("name", ""),
            confidence=float(data.get("confidence", 0.0)),
            dsm5_code=data.get("dsm5_code"),
            icd11_code=data.get("icd11_code"),
            criteria_met=data.get("criteria_met", []),
            criteria_missing=data.get("criteria_missing", []),
            supporting_evidence=data.get("supporting_evidence", []),
            contra_evidence=data.get("contra_evidence", []),
            severity=SeverityLevel(data.get("severity", "MILD")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "confidence": self.confidence,
            "dsm5_code": self.dsm5_code,
            "icd11_code": self.icd11_code,
            "criteria_met": self.criteria_met,
            "criteria_missing": self.criteria_missing,
            "supporting_evidence": self.supporting_evidence,
            "contra_evidence": self.contra_evidence,
            "severity": self.severity.value,
        }


@dataclass
class DifferentialDTO:
    """Differential diagnosis result."""
    primary: HypothesisDTO | None = None
    alternatives: list[HypothesisDTO] = field(default_factory=list)
    ruled_out: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DifferentialDTO:
        """Create from dictionary."""
        primary_data = data.get("primary")
        return cls(
            primary=HypothesisDTO.from_dict(primary_data) if primary_data else None,
            alternatives=[HypothesisDTO.from_dict(h) for h in data.get("alternatives", [])],
            ruled_out=data.get("ruled_out", []),
            missing_info=data.get("missing_info", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary": self.primary.to_dict() if self.primary else None,
            "alternatives": [h.to_dict() for h in self.alternatives],
            "ruled_out": self.ruled_out,
            "missing_info": self.missing_info,
        }


@dataclass
class AssessmentResult:
    """Result from diagnosis assessment."""
    extracted_symptoms: list[SymptomDTO]
    differential: DifferentialDTO
    next_question: str | None
    response_text: str
    confidence_score: float
    safety_flags: list[str]
    current_phase: DiagnosisPhase
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssessmentResult:
        """Create from API response."""
        return cls(
            extracted_symptoms=[SymptomDTO.from_dict(s) for s in data.get("extracted_symptoms", [])],
            differential=DifferentialDTO.from_dict(data.get("differential", {})),
            next_question=data.get("next_question"),
            response_text=data.get("response_text", ""),
            confidence_score=float(data.get("confidence_score", 0.0)),
            safety_flags=data.get("safety_flags", []),
            current_phase=DiagnosisPhase(data.get("current_phase", "RAPPORT")),
            reasoning_steps=data.get("reasoning_chain", []),
        )


class DiagnosisServiceClient:
    """HTTP client for Diagnosis Service communication."""

    def __init__(self, settings: DiagnosisAgentSettings) -> None:
        self._settings = settings
        self._base_url = settings.service_url.rstrip("/")

    async def assess(
        self,
        user_id: str,
        session_id: str,
        message: str,
        conversation_history: list[dict[str, Any]],
        existing_symptoms: list[dict[str, Any]] | None = None,
        current_phase: str | None = None,
    ) -> AssessmentResult:
        """Perform full diagnostic assessment."""
        url = f"{self._base_url}/assess"
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "conversation_history": conversation_history,
            "existing_symptoms": existing_symptoms or [],
            "current_phase": current_phase,
        }
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return AssessmentResult.from_dict(response.json())
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "diagnosis_service_http_error",
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
                except httpx.RequestError as e:
                    logger.warning(
                        "diagnosis_service_request_error",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
        raise RuntimeError("Diagnosis service assessment failed after retries")

    async def extract_symptoms(
        self,
        message: str,
        conversation_history: list[dict[str, Any]],
    ) -> list[SymptomDTO]:
        """Extract symptoms from message without full assessment."""
        url = f"{self._base_url}/extract-symptoms"
        payload = {
            "message": message,
            "conversation_history": conversation_history,
        }
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return [SymptomDTO.from_dict(s) for s in data.get("symptoms", [])]


class DiagnosisAgent:
    """
    Diagnosis coordination agent for the orchestrator.
    Coordinates with Diagnosis Service for symptom assessment and differential diagnosis.
    """

    def __init__(self, settings: DiagnosisAgentSettings | None = None) -> None:
        self._settings = settings or DiagnosisAgentSettings()
        self._client = DiagnosisServiceClient(self._settings)
        self._assessment_count = 0

    async def process(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Process state and perform diagnostic assessment.
        This is the main LangGraph node function.

        Args:
            state: Current orchestrator state

        Returns:
            State updates dictionary
        """
        self._assessment_count += 1
        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        message = state.get("current_message", "")
        messages = state.get("messages", [])
        metadata = state.get("metadata", {})
        logger.info(
            "diagnosis_agent_processing",
            user_id=user_id,
            message_length=len(message),
        )
        try:
            result = await self._perform_assessment(
                user_id=user_id,
                session_id=session_id,
                message=message,
                messages=messages,
                metadata=metadata,
            )
            return self._build_state_update(result)
        except Exception as e:
            logger.error("diagnosis_agent_error", error=str(e))
            if self._settings.fallback_on_service_error:
                return self._build_fallback_response(message)
            raise

    async def _perform_assessment(
        self,
        user_id: str,
        session_id: str,
        message: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> AssessmentResult:
        """Perform diagnostic assessment via Diagnosis Service."""
        existing_symptoms = metadata.get("existing_symptoms", [])
        current_phase = metadata.get("diagnosis_phase")
        return await self._client.assess(
            user_id=user_id,
            session_id=session_id,
            message=message,
            conversation_history=messages[-10:] if messages else [],
            existing_symptoms=existing_symptoms,
            current_phase=current_phase,
        )

    def _build_state_update(self, result: AssessmentResult) -> dict[str, Any]:
        """Build state update from assessment result."""
        response_content = result.response_text
        if result.next_question:
            response_content = f"{result.response_text}\n\n{result.next_question}"
        differential_data = result.differential.to_dict() if result.differential else None
        symptoms_data = [s.to_dict() for s in result.extracted_symptoms]
        agent_result = AgentResult(
            agent_type=AgentType.DIAGNOSIS,
            success=True,
            response_content=response_content,
            confidence=result.confidence_score,
            metadata={
                "symptoms_count": len(result.extracted_symptoms),
                "differential": differential_data,
                "current_phase": result.current_phase.value,
                "safety_flags": result.safety_flags,
                "reasoning_steps_count": len(result.reasoning_steps),
            },
        )
        logger.info(
            "diagnosis_agent_complete",
            symptoms_found=len(result.extracted_symptoms),
            confidence=result.confidence_score,
            phase=result.current_phase.value,
        )
        return {
            "agent_results": [agent_result.to_dict()],
            "metadata": {
                "diagnosis_phase": result.current_phase.value,
                "existing_symptoms": symptoms_data,
                "differential": differential_data,
            },
        }

    def _build_fallback_response(self, message: str) -> dict[str, Any]:
        """Build fallback response when Diagnosis Service is unavailable."""
        symptom_indicators = [
            "feeling", "sleep", "appetite", "energy", "mood",
            "anxious", "depressed", "worried", "tired", "sad",
        ]
        message_lower = message.lower()
        detected_indicators = [ind for ind in symptom_indicators if ind in message_lower]
        if detected_indicators:
            response = (
                "Based on what you've shared, it sounds like you're experiencing "
                "some changes in how you're feeling. Could you tell me more about "
                "when these feelings started and how they've been affecting your daily life?"
            )
        else:
            response = (
                "I'd like to understand more about what you're going through. "
                "Can you share more about how you've been feeling lately?"
            )
        agent_result = AgentResult(
            agent_type=AgentType.DIAGNOSIS,
            success=True,
            response_content=response,
            confidence=0.5,
            metadata={"fallback_mode": True, "detected_indicators": detected_indicators},
        )
        logger.warning(
            "diagnosis_agent_fallback_used",
            indicators_found=len(detected_indicators),
        )
        return {
            "agent_results": [agent_result.to_dict()],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_assessments": self._assessment_count,
            "service_url": self._settings.service_url,
            "differential_enabled": self._settings.enable_differential,
        }


async def diagnosis_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for diagnosis agent processing.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    agent = DiagnosisAgent()
    return await agent.process(state)
