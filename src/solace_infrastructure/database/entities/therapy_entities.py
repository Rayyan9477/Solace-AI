"""
Therapy domain entities for the Solace-AI platform.

Centralized SQLAlchemy ORM models for therapy sessions, treatment plans,
interventions, and homework assignments. Entities with PHI inherit from
ClinicalBase for encryption and audit trail support.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base_models import ClinicalBase
from ..schema_registry import SchemaRegistry


# Enumerations

class TherapyModality(str, Enum):
    CBT = "CBT"
    DBT = "DBT"
    ACT = "ACT"
    MI = "MI"
    MINDFULNESS = "MINDFULNESS"
    SFBT = "SFBT"
    PSYCHOEDUCATION = "PSYCHOEDUCATION"


class SessionPhase(str, Enum):
    PRE_SESSION = "PRE_SESSION"
    OPENING = "OPENING"
    WORKING = "WORKING"
    CLOSING = "CLOSING"
    POST_SESSION = "POST_SESSION"
    CRISIS = "CRISIS"


class TreatmentPhase(str, Enum):
    FOUNDATION = "FOUNDATION"
    ACTIVE_TREATMENT = "ACTIVE_TREATMENT"
    CONSOLIDATION = "CONSOLIDATION"
    MAINTENANCE = "MAINTENANCE"


class ResponseStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RESPONDING = "RESPONDING"
    PARTIAL_RESPONSE = "PARTIAL_RESPONSE"
    NON_RESPONSE = "NON_RESPONSE"
    DETERIORATING = "DETERIORATING"
    REMISSION = "REMISSION"


class HomeworkStatus(str, Enum):
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    NOT_COMPLETED = "NOT_COMPLETED"
    SKIPPED = "SKIPPED"


class GoalStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    ACHIEVED = "ACHIEVED"
    MODIFIED = "MODIFIED"
    ABANDONED = "ABANDONED"


class SessionRiskLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    IMMINENT = "IMMINENT"


class SeverityLevel(str, Enum):
    MINIMAL = "MINIMAL"
    MILD = "MILD"
    MODERATE = "MODERATE"
    MODERATELY_SEVERE = "MODERATELY_SEVERE"
    SEVERE = "SEVERE"


class SteppedCareLevel(str, Enum):
    WELLNESS = "WELLNESS"
    LOW_INTENSITY = "LOW_INTENSITY"
    MEDIUM_INTENSITY = "MEDIUM_INTENSITY"
    HIGH_INTENSITY = "HIGH_INTENSITY"
    INTENSIVE_REFERRAL = "INTENSIVE_REFERRAL"


# Entity Models

@SchemaRegistry.register
class TreatmentPlan(ClinicalBase):
    """Treatment plan entity for therapy planning and tracking.

    Stores comprehensive treatment plans including diagnosis, modality selection,
    phased treatment approach, and outcome measures. All data encrypted as PHI.
    """

    __tablename__ = "treatment_plans"
    __phi_fields__: ClassVar[list[str]] = ["primary_diagnosis", "termination_reason"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    primary_diagnosis: Mapped[str] = mapped_column(
        String(200), nullable=False,
        comment="Primary diagnosis (encrypted as PHI)"
    )
    secondary_diagnoses: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="Secondary diagnoses list (encrypted)"
    )
    severity: Mapped[str] = mapped_column(
        String(30), nullable=False, index=True,
        comment="Severity: MINIMAL, MILD, MODERATE, MODERATELY_SEVERE, SEVERE"
    )
    stepped_care_level: Mapped[str] = mapped_column(
        String(30), nullable=False,
        comment="Stepped care level for treatment intensity"
    )
    primary_modality: Mapped[str] = mapped_column(
        String(30), nullable=False, index=True,
        comment="Primary therapy modality: CBT, DBT, ACT, MI, etc."
    )
    adjunct_modalities: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="Adjunct therapy modalities"
    )
    current_phase: Mapped[str] = mapped_column(
        String(30), nullable=False, default=TreatmentPhase.FOUNDATION.value,
        comment="Current treatment phase"
    )
    phase_sessions_completed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    total_sessions_completed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    session_frequency_per_week: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1,
    )
    response_status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=ResponseStatus.NOT_STARTED.value,
        comment="Treatment response status"
    )

    # Skills tracking
    skills_acquired: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    skills_in_progress: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    contraindications: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )

    # Outcome measures
    baseline_phq9: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latest_phq9: Mapped[int | None] = mapped_column(Integer, nullable=True)
    baseline_gad7: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latest_gad7: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Plan lifecycle
    target_completion: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    review_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True,
    )
    termination_reason: Mapped[str | None] = mapped_column(
        String(500), nullable=True,
    )

    # Goals stored as JSONB array of goal objects
    treatment_goals: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="List of treatment goal objects"
    )

    # Relationships
    sessions: Mapped[list[TherapySession]] = relationship(
        "TherapySession", back_populates="treatment_plan",
    )

    def __repr__(self) -> str:
        return (
            f"<TreatmentPlan(id={self.id}, user_id={self.user_id}, "
            f"modality={self.primary_modality}, phase={self.current_phase})>"
        )


@SchemaRegistry.register
class TherapySession(ClinicalBase):
    """Therapy session entity for tracking individual sessions.

    Stores session state, interventions applied, homework assigned,
    and clinical observations. All data encrypted as PHI.
    """

    __tablename__ = "therapy_sessions"
    __phi_fields__: ClassVar[list[str]] = ["summary", "next_session_focus"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    treatment_plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("treatment_plans.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    session_number: Mapped[int] = mapped_column(
        Integer, nullable=False,
    )
    current_phase: Mapped[str] = mapped_column(
        String(20), nullable=False, default=SessionPhase.PRE_SESSION.value,
        index=True, comment="Current session phase"
    )

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc), index=True,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Mood tracking
    mood_rating_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mood_rating_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Session content (encrypted)
    agenda_items: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    topics_covered: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    skills_practiced: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    insights_gained: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )

    # Safety
    current_risk: Mapped[str] = mapped_column(
        String(20), nullable=False, default=SessionRiskLevel.NONE.value, index=True,
    )
    safety_flags: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )

    # Outcomes
    session_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    alliance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    next_session_focus: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    treatment_plan: Mapped[TreatmentPlan] = relationship(
        "TreatmentPlan", back_populates="sessions",
    )
    interventions: Mapped[list[TherapyIntervention]] = relationship(
        "TherapyIntervention", back_populates="session",
    )
    homework_assignments: Mapped[list[HomeworkAssignment]] = relationship(
        "HomeworkAssignment", back_populates="session",
    )

    def __repr__(self) -> str:
        return (
            f"<TherapySession(id={self.id}, user_id={self.user_id}, "
            f"session_number={self.session_number}, phase={self.current_phase})>"
        )


@SchemaRegistry.register
class TherapyIntervention(ClinicalBase):
    """Intervention record within a therapy session.

    Tracks specific therapeutic techniques applied during a session.
    """

    __tablename__ = "therapy_interventions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("therapy_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    technique_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True,
    )
    technique_name: Mapped[str] = mapped_column(String(200), nullable=False)
    modality: Mapped[str] = mapped_column(String(30), nullable=False)
    phase: Mapped[str] = mapped_column(String(20), nullable=False)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    messages_exchanged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    skills_practiced: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    insights_gained: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )

    # Relationships
    session: Mapped[TherapySession] = relationship(
        "TherapySession", back_populates="interventions",
    )

    def __repr__(self) -> str:
        return (
            f"<TherapyIntervention(id={self.id}, technique={self.technique_name}, "
            f"completed={self.completed})>"
        )


@SchemaRegistry.register
class HomeworkAssignment(ClinicalBase):
    """Homework assignment entity for between-session activities.

    Tracks homework assigned during therapy sessions.
    """

    __tablename__ = "homework_assignments"
    __phi_fields__: ClassVar[list[str]] = ["description", "completion_notes"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("therapy_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    technique_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True,
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=HomeworkStatus.ASSIGNED.value, index=True,
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completion_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    session: Mapped[TherapySession] = relationship(
        "TherapySession", back_populates="homework_assignments",
    )

    def __repr__(self) -> str:
        return (
            f"<HomeworkAssignment(id={self.id}, title={self.title}, "
            f"status={self.status})>"
        )


__all__ = [
    "TherapyModality",
    "SessionPhase",
    "TreatmentPhase",
    "ResponseStatus",
    "HomeworkStatus",
    "GoalStatus",
    "SessionRiskLevel",
    "SeverityLevel",
    "SteppedCareLevel",
    "TreatmentPlan",
    "TherapySession",
    "TherapyIntervention",
    "HomeworkAssignment",
]
