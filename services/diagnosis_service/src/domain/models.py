"""
Solace-AI Diagnosis Service - Domain Models.
Data classes for diagnosis operations and results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from ..schemas import (
    DiagnosisPhase, SymptomDTO, HypothesisDTO, DifferentialDTO, ReasoningStepResultDTO,
)


@dataclass
class SessionState:
    """Active diagnosis session state."""
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    phase: DiagnosisPhase = DiagnosisPhase.RAPPORT
    symptoms: list[SymptomDTO] = field(default_factory=list)
    differential: DifferentialDTO | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reasoning_history: list[ReasoningStepResultDTO] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)


@dataclass
class AssessmentResult:
    """Result from diagnostic assessment."""
    assessment_id: UUID = field(default_factory=uuid4)
    phase: DiagnosisPhase = DiagnosisPhase.RAPPORT
    extracted_symptoms: list[SymptomDTO] = field(default_factory=list)
    differential: DifferentialDTO = field(default_factory=DifferentialDTO)
    reasoning_chain: list[ReasoningStepResultDTO] = field(default_factory=list)
    next_question: str | None = None
    response_text: str = ""
    confidence_score: Decimal = Decimal("0.5")
    safety_flags: list[str] = field(default_factory=list)
    processing_time_ms: int = 0


@dataclass
class ExtractionResult:
    """Result from symptom extraction."""
    extraction_id: UUID = field(default_factory=uuid4)
    extracted_symptoms: list[SymptomDTO] = field(default_factory=list)
    updated_symptoms: list[SymptomDTO] = field(default_factory=list)
    temporal_info: dict[str, str] = field(default_factory=dict)
    contextual_factors: list[str] = field(default_factory=list)
    risk_indicators: list[str] = field(default_factory=list)
    extraction_time_ms: int = 0


@dataclass
class DifferentialResult:
    """Result from differential generation."""
    differential_id: UUID = field(default_factory=uuid4)
    differential: DifferentialDTO = field(default_factory=DifferentialDTO)
    hitop_scores: dict[str, Decimal] = field(default_factory=dict)
    recommended_questions: list[str] = field(default_factory=list)
    generation_time_ms: int = 0


@dataclass
class SessionStartResult:
    """Result from session start."""
    session_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    initial_phase: DiagnosisPhase = DiagnosisPhase.RAPPORT
    greeting: str = ""
    loaded_context: bool = False


@dataclass
class SessionEndResult:
    """Result from session end."""
    duration_minutes: int = 0
    messages_exchanged: int = 0
    final_differential: DifferentialDTO | None = None
    summary: str | None = None
    recommendations: list[str] = field(default_factory=list)


@dataclass
class HistoryResult:
    """Result from history query."""
    sessions: list[dict[str, Any]] = field(default_factory=list)
    symptom_trends: dict[str, Any] = field(default_factory=dict)
    longitudinal_patterns: list[str] = field(default_factory=list)


@dataclass
class ChallengeResult:
    """Result from hypothesis challenge."""
    challenges: list[str] = field(default_factory=list)
    alternatives: list[HypothesisDTO] = field(default_factory=list)
    counter_questions: list[str] = field(default_factory=list)
    bias_flags: list[str] = field(default_factory=list)
