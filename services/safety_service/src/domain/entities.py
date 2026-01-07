"""
Solace-AI Safety Service - Domain Entities.
Core domain entities with identity for safety assessments, plans, and incidents.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
import structlog

logger = structlog.get_logger(__name__)


class SafetyPlanStatus(str, Enum):
    """Status of a safety plan."""
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    UNDER_REVIEW = "UNDER_REVIEW"
    EXPIRED = "EXPIRED"
    ARCHIVED = "ARCHIVED"


class IncidentSeverity(str, Enum):
    """Severity classification for safety incidents."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    """Status of a safety incident."""
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"


class AssessmentType(str, Enum):
    """Type of safety assessment."""
    PRE_CHECK = "PRE_CHECK"
    POST_CHECK = "POST_CHECK"
    CONTINUOUS = "CONTINUOUS"
    TRIGGERED = "TRIGGERED"
    SCHEDULED = "SCHEDULED"


class WarningSign(BaseModel):
    """Individual warning sign in a safety plan."""
    sign_id: UUID = Field(default_factory=uuid4)
    description: str = Field(..., min_length=1, max_length=500)
    severity_level: int = Field(..., ge=1, le=5)
    category: str = Field(..., min_length=1, max_length=100)
    recognition_cues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CopingStrategy(BaseModel):
    """Coping strategy in a safety plan."""
    strategy_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    category: str = Field(default="general")
    effectiveness_rating: int | None = Field(default=None, ge=1, le=10)
    times_used: int = Field(default=0, ge=0)
    last_used_at: datetime | None = Field(default=None)
    is_active: bool = Field(default=True)


class EmergencyContact(BaseModel):
    """Emergency contact in a safety plan."""
    contact_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=200)
    relationship: str = Field(..., min_length=1, max_length=100)
    phone: str | None = Field(default=None)
    email: str | None = Field(default=None)
    availability: str = Field(default="anytime")
    priority_order: int = Field(default=1, ge=1, le=10)
    is_professional: bool = Field(default=False)
    notes: str | None = Field(default=None)


class SafeEnvironmentAction(BaseModel):
    """Action to make environment safer."""
    action_id: UUID = Field(default_factory=uuid4)
    description: str = Field(..., min_length=1, max_length=500)
    category: str = Field(default="general")
    completed: bool = Field(default=False)
    completed_at: datetime | None = Field(default=None)
    priority: int = Field(default=1, ge=1, le=5)


class SafetyPlan(BaseModel):
    """Comprehensive safety plan entity with identity."""
    plan_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    status: SafetyPlanStatus = Field(default=SafetyPlanStatus.DRAFT)
    version: int = Field(default=1, ge=1)
    warning_signs: list[WarningSign] = Field(default_factory=list)
    coping_strategies: list[CopingStrategy] = Field(default_factory=list)
    emergency_contacts: list[EmergencyContact] = Field(default_factory=list)
    safe_environment_actions: list[SafeEnvironmentAction] = Field(default_factory=list)
    reasons_to_live: list[str] = Field(default_factory=list)
    professional_resources: list[dict[str, str]] = Field(default_factory=list)
    clinician_notes: str | None = Field(default=None)
    last_reviewed_at: datetime | None = Field(default=None)
    last_reviewed_by: UUID | None = Field(default=None)
    next_review_due: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if plan has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_complete(self) -> bool:
        """Check if plan has minimum required components."""
        return (
            len(self.warning_signs) >= 1
            and len(self.coping_strategies) >= 2
            and len(self.emergency_contacts) >= 1
        )

    @property
    def days_until_review(self) -> int | None:
        """Days until next review is due."""
        if self.next_review_due is None:
            return None
        delta = self.next_review_due - datetime.now(timezone.utc)
        return max(0, delta.days)

    def activate(self) -> None:
        """Activate the safety plan."""
        if not self.is_complete:
            raise ValueError("Cannot activate incomplete safety plan")
        self.status = SafetyPlanStatus.ACTIVE
        self.updated_at = datetime.now(timezone.utc)
        if self.next_review_due is None:
            self.next_review_due = datetime.now(timezone.utc) + timedelta(days=30)

    def archive(self) -> None:
        """Archive the safety plan."""
        self.status = SafetyPlanStatus.ARCHIVED
        self.updated_at = datetime.now(timezone.utc)

    def add_warning_sign(self, sign: WarningSign) -> None:
        """Add a warning sign to the plan."""
        self.warning_signs.append(sign)
        self.updated_at = datetime.now(timezone.utc)

    def add_coping_strategy(self, strategy: CopingStrategy) -> None:
        """Add a coping strategy to the plan."""
        self.coping_strategies.append(strategy)
        self.updated_at = datetime.now(timezone.utc)

    def add_emergency_contact(self, contact: EmergencyContact) -> None:
        """Add an emergency contact to the plan."""
        self.emergency_contacts.append(contact)
        self.emergency_contacts.sort(key=lambda c: c.priority_order)
        self.updated_at = datetime.now(timezone.utc)


class SafetyAssessment(BaseModel):
    """Safety assessment entity capturing a point-in-time evaluation."""
    assessment_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    session_id: UUID | None = Field(default=None)
    assessment_type: AssessmentType = Field(default=AssessmentType.PRE_CHECK)
    content_assessed: str = Field(..., min_length=1)
    risk_score: Decimal = Field(default=Decimal("0.0"), ge=0, le=1)
    crisis_level: str = Field(default="NONE")
    is_safe: bool = Field(default=True)
    risk_factors: list[dict[str, Any]] = Field(default_factory=list)
    protective_factors: list[dict[str, Any]] = Field(default_factory=list)
    trigger_indicators: list[str] = Field(default_factory=list)
    detection_layers_triggered: list[int] = Field(default_factory=list)
    recommended_action: str = Field(default="continue")
    requires_escalation: bool = Field(default=False)
    requires_human_review: bool = Field(default=False)
    detection_time_ms: int = Field(default=0, ge=0)
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_at: datetime | None = Field(default=None)
    reviewed_by: UUID | None = Field(default=None)
    review_notes: str | None = Field(default=None)

    @field_validator("risk_score", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce risk_score to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class SafetyIncident(BaseModel):
    """Safety incident entity for tracking crisis events."""
    incident_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    session_id: UUID | None = Field(default=None)
    assessment_id: UUID | None = Field(default=None)
    escalation_id: UUID | None = Field(default=None)
    severity: IncidentSeverity = Field(...)
    status: IncidentStatus = Field(default=IncidentStatus.OPEN)
    crisis_level: str = Field(...)
    description: str = Field(..., min_length=1, max_length=2000)
    trigger_indicators: list[str] = Field(default_factory=list)
    risk_factors: list[dict[str, Any]] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    resources_provided: list[str] = Field(default_factory=list)
    assigned_clinician_id: UUID | None = Field(default=None)
    resolution_notes: str | None = Field(default=None)
    follow_up_required: bool = Field(default=False)
    follow_up_due: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: datetime | None = Field(default=None)
    resolved_at: datetime | None = Field(default=None)
    closed_at: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def time_to_acknowledge(self) -> timedelta | None:
        """Time from creation to acknowledgment."""
        if self.acknowledged_at is None:
            return None
        return self.acknowledged_at - self.created_at

    @property
    def time_to_resolve(self) -> timedelta | None:
        """Time from creation to resolution."""
        if self.resolved_at is None:
            return None
        return self.resolved_at - self.created_at

    @property
    def is_overdue(self) -> bool:
        """Check if follow-up is overdue."""
        if not self.follow_up_required or self.follow_up_due is None:
            return False
        return datetime.now(timezone.utc) > self.follow_up_due

    def acknowledge(self, clinician_id: UUID) -> None:
        """Acknowledge the incident."""
        if self.status != IncidentStatus.OPEN:
            raise ValueError(f"Cannot acknowledge incident in {self.status.value} status")
        self.status = IncidentStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)
        self.assigned_clinician_id = clinician_id
        self.actions_taken.append(f"Acknowledged by clinician {clinician_id}")

    def start_progress(self) -> None:
        """Mark incident as in progress."""
        if self.status not in (IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED):
            raise ValueError(f"Cannot start progress from {self.status.value} status")
        self.status = IncidentStatus.IN_PROGRESS
        self.actions_taken.append("Investigation started")

    def resolve(self, notes: str) -> None:
        """Resolve the incident."""
        if self.status not in (IncidentStatus.ACKNOWLEDGED, IncidentStatus.IN_PROGRESS):
            raise ValueError(f"Cannot resolve incident in {self.status.value} status")
        self.status = IncidentStatus.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)
        self.resolution_notes = notes
        self.actions_taken.append(f"Resolved: {notes[:100]}")

    def close(self) -> None:
        """Close the incident."""
        if self.status != IncidentStatus.RESOLVED:
            raise ValueError("Can only close resolved incidents")
        self.status = IncidentStatus.CLOSED
        self.closed_at = datetime.now(timezone.utc)
        self.actions_taken.append("Incident closed")

    def escalate(self, reason: str) -> None:
        """Escalate the incident."""
        self.status = IncidentStatus.ESCALATED
        self.actions_taken.append(f"Escalated: {reason}")


class UserRiskProfile(BaseModel):
    """Aggregated risk profile for a user."""
    profile_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    baseline_risk_level: str = Field(default="NONE")
    current_risk_level: str = Field(default="NONE")
    total_assessments: int = Field(default=0, ge=0)
    total_incidents: int = Field(default=0, ge=0)
    crisis_events_count: int = Field(default=0, ge=0)
    escalations_count: int = Field(default=0, ge=0)
    last_crisis_at: datetime | None = Field(default=None)
    last_assessment_at: datetime | None = Field(default=None)
    high_risk_flag: bool = Field(default=False)
    recent_escalation: bool = Field(default=False)
    active_safety_plan_id: UUID | None = Field(default=None)
    risk_trend: str = Field(default="stable")
    protective_factors_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def record_assessment(self, crisis_level: str, is_escalation: bool = False) -> None:
        """Record a new assessment in the profile."""
        self.total_assessments += 1
        self.last_assessment_at = datetime.now(timezone.utc)
        self.current_risk_level = crisis_level
        if crisis_level in ("HIGH", "CRITICAL"):
            self.crisis_events_count += 1
            self.last_crisis_at = datetime.now(timezone.utc)
            self.high_risk_flag = True
        if is_escalation:
            self.escalations_count += 1
            self.recent_escalation = True
        self.updated_at = datetime.now(timezone.utc)

    def record_incident(self, severity: IncidentSeverity) -> None:
        """Record a new incident in the profile."""
        self.total_incidents += 1
        if severity in (IncidentSeverity.HIGH, IncidentSeverity.CRITICAL):
            self.high_risk_flag = True
        self.updated_at = datetime.now(timezone.utc)

    def clear_recent_flags(self) -> None:
        """Clear recent escalation flag after review."""
        self.recent_escalation = False
        self.updated_at = datetime.now(timezone.utc)
