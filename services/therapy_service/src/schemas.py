"""
Solace-AI Therapy Service - API Request/Response Schemas.
Pydantic models for therapy session and intervention operations.
"""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field


class SessionPhase(str, Enum):
    """Therapy session phases."""
    PRE_SESSION = "pre_session"
    OPENING = "opening"
    WORKING = "working"
    CLOSING = "closing"
    POST_SESSION = "post_session"


class TherapyModality(str, Enum):
    """Evidence-based therapy modalities."""
    CBT = "cbt"
    DBT = "dbt"
    ACT = "act"
    MI = "mi"
    MINDFULNESS = "mindfulness"


class SeverityLevel(str, Enum):
    """Clinical severity levels."""
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMINENT = "imminent"


class TechniqueDTO(BaseModel):
    """Therapeutic technique data transfer object."""
    technique_id: UUID
    name: str
    modality: TherapyModality
    category: str
    description: str
    duration_minutes: int = 10
    requires_homework: bool = False
    contraindications: list[str] = Field(default_factory=list)


class HomeworkDTO(BaseModel):
    """Homework assignment data transfer object."""
    homework_id: UUID
    title: str
    description: str
    technique_id: UUID
    due_date: datetime | None = None
    completed: bool = False
    completion_date: datetime | None = None
    notes: str | None = None


class TreatmentPlanDTO(BaseModel):
    """Treatment plan data transfer object."""
    plan_id: UUID
    user_id: UUID
    primary_diagnosis: str
    severity: SeverityLevel
    primary_modality: TherapyModality
    adjunct_modalities: list[TherapyModality] = Field(default_factory=list)
    current_phase: int = 1
    sessions_completed: int = 0
    skills_acquired: list[str] = Field(default_factory=list)


class SessionStateDTO(BaseModel):
    """Current session state data transfer object."""
    session_id: UUID
    user_id: UUID
    treatment_plan_id: UUID
    session_number: int
    current_phase: SessionPhase
    mood_rating: int | None = None
    agenda_items: list[str] = Field(default_factory=list)
    topics_covered: list[str] = Field(default_factory=list)
    skills_practiced: list[str] = Field(default_factory=list)
    current_risk: RiskLevel = RiskLevel.NONE
    engagement_score: float = 0.0


class SessionStartRequest(BaseModel):
    """Request to start a therapy session."""
    user_id: UUID
    treatment_plan_id: UUID
    context: dict[str, Any] = Field(default_factory=dict)


class SessionStartResponse(BaseModel):
    """Response from starting a therapy session."""
    session_id: UUID
    user_id: UUID
    treatment_plan_id: UUID
    session_number: int
    current_phase: SessionPhase
    initial_message: str
    suggested_agenda: list[str] = Field(default_factory=list)


class MessageRequest(BaseModel):
    """Request to process user message in therapy session."""
    session_id: UUID
    user_id: UUID
    message: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class TherapyResponse(BaseModel):
    """Response from therapy message processing."""
    session_id: UUID
    user_id: UUID
    response_text: str
    current_phase: SessionPhase
    technique_applied: TechniqueDTO | None = None
    homework_assigned: list[HomeworkDTO] = Field(default_factory=list)
    safety_alerts: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0


class SessionEndRequest(BaseModel):
    """Request to end a therapy session."""
    session_id: UUID
    user_id: UUID
    generate_summary: bool = True


class SessionSummaryDTO(BaseModel):
    """Session summary data transfer object."""
    session_id: UUID
    user_id: UUID
    session_number: int
    duration_minutes: int
    techniques_used: list[TechniqueDTO] = Field(default_factory=list)
    skills_practiced: list[str] = Field(default_factory=list)
    insights_gained: list[str] = Field(default_factory=list)
    homework_assigned: list[HomeworkDTO] = Field(default_factory=list)
    session_rating: float | None = None
    summary_text: str = ""
    next_session_focus: str = ""
