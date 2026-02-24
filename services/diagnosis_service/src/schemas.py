"""
Solace-AI Diagnosis Service - API Request/Response Schemas.
Pydantic models for diagnosis and assessment operations.
"""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field


class DiagnosisPhase(str, Enum):
    """Diagnosis dialogue phases (AMIE-inspired)."""
    RAPPORT = "rapport"
    HISTORY = "history"
    ASSESSMENT = "assessment"
    DIAGNOSIS = "diagnosis"
    CLOSURE = "closure"


from solace_common.enums import SeverityLevel  # noqa: E402


class ConfidenceLevel(str, Enum):
    """Hypothesis confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReasoningStep(str, Enum):
    """4-step Chain-of-Reasoning steps."""
    ANALYZE = "analyze"
    HYPOTHESIZE = "hypothesize"
    CHALLENGE = "challenge"
    SYNTHESIZE = "synthesize"


class SymptomType(str, Enum):
    """Symptom classification types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    COGNITIVE = "cognitive"
    BEHAVIORAL = "behavioral"
    SOMATIC = "somatic"
    EMOTIONAL = "emotional"


class SymptomDTO(BaseModel):
    """Symptom data transfer object."""
    symptom_id: UUID
    name: str
    description: str
    symptom_type: SymptomType
    severity: SeverityLevel = SeverityLevel.MILD
    onset: str | None = None
    duration: str | None = None
    frequency: str | None = None
    triggers: list[str] = Field(default_factory=list)
    extracted_from: str | None = None
    confidence: Decimal = Field(default=Decimal("0.7"), ge=0, le=1)


class HypothesisDTO(BaseModel):
    """Clinical hypothesis data transfer object."""
    hypothesis_id: UUID
    name: str
    dsm5_code: str | None = None
    icd11_code: str | None = None
    confidence: Decimal = Field(ge=0, le=1)
    confidence_interval: tuple[Decimal, Decimal] | None = None
    criteria_met: list[str] = Field(default_factory=list)
    criteria_missing: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    contra_evidence: list[str] = Field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MILD
    hitop_dimensions: dict[str, Decimal] = Field(default_factory=dict)


class DifferentialDTO(BaseModel):
    """Differential diagnosis list."""
    primary: HypothesisDTO | None = None
    alternatives: list[HypothesisDTO] = Field(default_factory=list)
    ruled_out: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


class ReasoningStepResultDTO(BaseModel):
    """Result from a single reasoning step."""
    step: ReasoningStep
    input_summary: str
    output_summary: str
    duration_ms: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class AssessmentRequest(BaseModel):
    """Request for diagnostic assessment."""
    user_id: UUID
    session_id: UUID
    message: str = Field(..., max_length=50000)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    existing_symptoms: list[SymptomDTO] = Field(default_factory=list)
    current_phase: DiagnosisPhase = DiagnosisPhase.RAPPORT
    current_differential: DifferentialDTO | None = None
    user_context: dict[str, Any] = Field(default_factory=dict)


class AssessmentResponse(BaseModel):
    """Response from diagnostic assessment."""
    assessment_id: UUID
    user_id: UUID
    session_id: UUID
    phase: DiagnosisPhase
    extracted_symptoms: list[SymptomDTO]
    differential: DifferentialDTO
    reasoning_chain: list[ReasoningStepResultDTO]
    next_question: str | None = None
    response_text: str
    confidence_score: Decimal = Field(ge=0, le=1)
    safety_flags: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0


class SymptomExtractionRequest(BaseModel):
    """Request for symptom extraction only."""
    user_id: UUID
    session_id: UUID
    message: str = Field(..., max_length=50000)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    existing_symptoms: list[SymptomDTO] = Field(default_factory=list)


class SymptomExtractionResponse(BaseModel):
    """Response from symptom extraction."""
    extraction_id: UUID
    user_id: UUID
    extracted_symptoms: list[SymptomDTO]
    updated_symptoms: list[SymptomDTO]
    temporal_info: dict[str, str] = Field(default_factory=dict)
    contextual_factors: list[str] = Field(default_factory=list)
    risk_indicators: list[str] = Field(default_factory=list)
    extraction_time_ms: int = 0


class DifferentialRequest(BaseModel):
    """Request for differential generation."""
    user_id: UUID
    session_id: UUID
    symptoms: list[SymptomDTO]
    user_history: dict[str, Any] = Field(default_factory=dict)
    current_differential: DifferentialDTO | None = None


class DifferentialResponse(BaseModel):
    """Response from differential generation."""
    differential_id: UUID
    user_id: UUID
    differential: DifferentialDTO
    hitop_scores: dict[str, Decimal] = Field(default_factory=dict)
    recommended_questions: list[str] = Field(default_factory=list)
    generation_time_ms: int = 0


class SessionStartRequest(BaseModel):
    """Request to start a diagnosis session."""
    user_id: UUID
    session_type: str = "assessment"
    initial_context: dict[str, Any] = Field(default_factory=dict)
    previous_session_id: UUID | None = None


class SessionStartResponse(BaseModel):
    """Response from starting a diagnosis session."""
    session_id: UUID
    user_id: UUID
    session_number: int
    initial_phase: DiagnosisPhase
    greeting: str
    loaded_context: bool = False


class SessionEndRequest(BaseModel):
    """Request to end a diagnosis session."""
    user_id: UUID
    session_id: UUID
    generate_summary: bool = True


class SessionEndResponse(BaseModel):
    """Response from ending a diagnosis session."""
    session_id: UUID
    user_id: UUID
    duration_minutes: int
    messages_exchanged: int
    final_differential: DifferentialDTO | None = None
    summary: str | None = None
    recommendations: list[str] = Field(default_factory=list)


class DiagnosisHistoryRequest(BaseModel):
    """Request for diagnosis history."""
    user_id: UUID
    limit: int = Field(default=10, ge=1, le=100)
    include_symptoms: bool = True
    include_differentials: bool = True


class DiagnosisHistoryResponse(BaseModel):
    """Response with diagnosis history."""
    user_id: UUID
    sessions: list[dict[str, Any]]
    symptom_trends: dict[str, Any] = Field(default_factory=dict)
    longitudinal_patterns: list[str] = Field(default_factory=list)
