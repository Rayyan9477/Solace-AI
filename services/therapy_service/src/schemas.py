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
    CRISIS = "crisis"


class TreatmentPhase(str, Enum):
    """Treatment plan phases."""
    FOUNDATION = "foundation"
    ACTIVE_TREATMENT = "active_treatment"
    CONSOLIDATION = "consolidation"
    MAINTENANCE = "maintenance"


class SteppedCareLevel(int, Enum):
    """Stepped care intensity levels (PHQ-9 based)."""
    WELLNESS = 0  # PHQ-9: 0-4
    LOW_INTENSITY = 1  # PHQ-9: 5-9
    MEDIUM_INTENSITY = 2  # PHQ-9: 10-14
    HIGH_INTENSITY = 3  # PHQ-9: 15-19
    INTENSIVE_REFERRAL = 4  # PHQ-9: 20+


class TherapyModality(str, Enum):
    """Evidence-based therapy modalities."""
    CBT = "cbt"
    DBT = "dbt"
    ACT = "act"
    MI = "mi"
    MINDFULNESS = "mindfulness"
    SFBT = "sfbt"
    PSYCHOEDUCATION = "psychoeducation"


class TechniqueCategory(str, Enum):
    """Categories of therapeutic techniques."""
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    EXPOSURE = "exposure"
    MINDFULNESS_SKILL = "mindfulness_skill"
    DISTRESS_TOLERANCE = "distress_tolerance"
    EMOTION_REGULATION = "emotion_regulation"
    INTERPERSONAL = "interpersonal"
    VALUES_WORK = "values_work"
    RELAXATION = "relaxation"
    PROBLEM_SOLVING = "problem_solving"
    THOUGHT_RECORDS = "thought_records"
    DEFUSION = "defusion"
    ACCEPTANCE = "acceptance"


class SeverityLevel(str, Enum):
    """Clinical severity levels."""
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"


class ResponseStatus(str, Enum):
    """Treatment response status."""
    NOT_STARTED = "not_started"
    RESPONDING = "responding"
    PARTIAL_RESPONSE = "partial_response"
    NON_RESPONSE = "non_response"
    DETERIORATING = "deteriorating"
    REMISSION = "remission"


class HomeworkStatus(str, Enum):
    """Homework completion status."""
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    NOT_COMPLETED = "not_completed"
    SKIPPED = "skipped"


class GoalStatus(str, Enum):
    """Goal tracking status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    MODIFIED = "modified"
    ABANDONED = "abandoned"


class OutcomeInstrument(str, Enum):
    """Validated outcome measurement instruments."""
    PHQ9 = "phq9"
    GAD7 = "gad7"
    PCL5 = "pcl5"
    ORS = "ors"
    SRS = "srs"
    CORE10 = "core10"
    DASS21 = "dass21"


class DifficultyLevel(str, Enum):
    """Technique difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DeliveryMode(str, Enum):
    """How techniques are delivered."""
    GUIDED = "guided"
    SELF_GUIDED = "self_guided"
    EXERCISE = "exercise"
    PSYCHOEDUCATION = "psychoeducation"
    PRACTICE = "practice"


class RiskLevel(str, Enum):
    """Risk assessment levels. Aligned with canonical CrisisLevel."""
    NONE = "NONE"
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


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


class OutcomeScoreDTO(BaseModel):
    """Outcome measurement score."""
    score_id: UUID
    user_id: UUID
    session_id: UUID | None = None
    instrument: OutcomeInstrument
    total_score: int
    subscale_scores: dict[str, int] = Field(default_factory=dict)
    severity_category: SeverityLevel
    clinically_significant: bool = False
    reliable_change: bool = False
    recorded_at: datetime


class TreatmentGoalDTO(BaseModel):
    """Treatment goal data transfer object."""
    goal_id: UUID
    user_id: UUID
    treatment_plan_id: UUID
    description: str
    goal_type: str = "primary"
    target_outcome: str | None = None
    status: GoalStatus = GoalStatus.NOT_STARTED
    progress_percentage: int = Field(default=0, ge=0, le=100)
    target_date: datetime | None = None
    created_at: datetime
    achieved_at: datetime | None = None


class InterventionDTO(BaseModel):
    """Intervention delivery data transfer object."""
    intervention_id: UUID
    session_id: UUID
    technique: TechniqueDTO
    phase: SessionPhase
    start_time: datetime
    end_time: datetime | None = None
    messages_exchanged: int = 0
    completed: bool = False
    user_engagement_score: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    skills_practiced: list[str] = Field(default_factory=list)
    insights_gained: list[str] = Field(default_factory=list)


class ProgressReportDTO(BaseModel):
    """Progress report data transfer object."""
    report_id: UUID
    user_id: UUID
    treatment_plan_id: UUID
    report_period_start: datetime
    report_period_end: datetime
    sessions_attended: int
    homework_completion_rate: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    outcome_trend: str
    outcome_scores: list[OutcomeScoreDTO] = Field(default_factory=list)
    skills_acquired: list[str] = Field(default_factory=list)
    goals_achieved: list[str] = Field(default_factory=list)
    response_status: ResponseStatus
    recommendations: list[str] = Field(default_factory=list)
    generated_at: datetime


class MilestoneDTO(BaseModel):
    """Treatment milestone data transfer object."""
    milestone_id: UUID
    treatment_plan_id: UUID
    name: str
    description: str
    target_session: int | None = None
    target_week: int | None = None
    achieved: bool = False
    achieved_at: datetime | None = None
    evidence: list[str] = Field(default_factory=list)
