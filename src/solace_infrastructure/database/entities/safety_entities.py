"""
Safety domain entities for the Solace-AI platform.

Centralized SQLAlchemy ORM models for safety assessments, plans, risk factors,
and contraindication checks. All entities inherit from ClinicalBase to ensure
proper PHI encryption and audit trail support.

These entities replace the fragmented safety_service domain entities and provide
a single source of truth for safety-related database schemas.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base_models import ClinicalBase, SafetyEventBase
from ..schema_registry import SchemaRegistry


# Enumerations for safety domain

class AssessmentType(str, Enum):
    """Type of safety assessment."""
    PRE_CHECK = "PRE_CHECK"
    POST_CHECK = "POST_CHECK"
    CONTINUOUS = "CONTINUOUS"
    TRIGGERED = "TRIGGERED"
    SCHEDULED = "SCHEDULED"


class RiskLevel(str, Enum):
    """Risk level classification for safety assessments. Aligned with canonical CrisisLevel."""
    NONE = "NONE"
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


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


# Entity Models

@SchemaRegistry.register
class SafetyAssessment(ClinicalBase):
    """Safety assessment entity for tracking user safety evaluations.

    Stores comprehensive safety assessments including risk levels, factors,
    and recommended interventions. All data is encrypted at rest as PHI.

    Inherits from ClinicalBase to ensure:
    - Automatic PHI encryption (encryption_key_id required)
    - Audit trail (created_by, updated_by, created_at, updated_at)
    - Soft delete support
    - User association
    """

    __tablename__ = "safety_assessments"
    __phi_fields__: ClassVar[list[str]] = ["content_assessed", "assessment_notes", "review_notes"]

    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Session during which assessment was performed"
    )

    assessment_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of assessment: PRE_CHECK, POST_CHECK, CONTINUOUS, etc."
    )

    # Content assessed (encrypted PHI)
    content_assessed: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="The content that was assessed for safety (encrypted PHI)"
    )

    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Overall risk level: NONE, LOW, ELEVATED, HIGH, CRITICAL"
    )

    risk_score: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Numerical risk score (0.0-1.0)"
    )

    crisis_level: Mapped[str | None] = mapped_column(
        String(20), nullable=True, index=True,
        comment="Crisis level if detected"
    )

    is_safe: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        comment="Whether the content was deemed safe"
    )

    risk_factors: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Detailed risk factors identified"
    )

    protective_factors: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Protective factors present"
    )

    trigger_indicators: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="Trigger indicators detected"
    )

    detection_layers_triggered: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="Which detection layers triggered"
    )

    # Recommendations
    recommended_action: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="Recommended action: CONTINUE, ESCALATE, REFER, etc."
    )

    recommended_interventions: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Recommended interventions based on assessment"
    )

    requires_escalation: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True,
        comment="Whether escalation to human clinician is required"
    )

    requires_human_review: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="Whether human review is recommended"
    )

    detection_time_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="Detection processing time in milliseconds"
    )

    context: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Additional assessment context"
    )

    assessment_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Clinical notes from assessment (encrypted as PHI)"
    )

    # Timing and follow-up
    assessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    next_assessment_due: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True,
    )

    # Review tracking
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    reviewed_by: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    review_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Review notes (encrypted PHI)"
    )

    # Assessment provenance
    assessor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    assessment_method: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="Method used: AI, CLINICAL_INTERVIEW, STANDARDIZED_TOOL, etc."
    )

    # Relationships
    safety_plan: Mapped[SafetyPlan | None] = relationship(
        "SafetyPlan",
        back_populates="assessment",
        uselist=False,
    )

    def __repr__(self) -> str:
        return (
            f"<SafetyAssessment(id={self.id}, user_id={self.user_id}, "
            f"risk_level={self.risk_level}, is_safe={self.is_safe})>"
        )


@SchemaRegistry.register
class SafetyPlan(ClinicalBase):
    """Safety plan entity for comprehensive user safety planning.

    Stores detailed safety plans including warning signs, coping strategies,
    emergency contacts, and safe environment actions. All data encrypted as PHI.

    Inherits from ClinicalBase for PHI protection and audit trail.
    """

    __tablename__ = "safety_plans"
    __phi_fields__: ClassVar[list[str]] = ["clinician_notes"]

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=SafetyPlanStatus.DRAFT.value,
        index=True,
    )

    assessment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("safety_assessments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Plan content (all JSONB, encrypted at application layer)
    warning_signs: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    coping_strategies: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    emergency_contacts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    safe_environment_actions: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    reasons_to_live: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="User's reasons to live (critical for safety plans)"
    )
    professional_resources: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )

    clinician_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Clinician notes about the plan (encrypted PHI)"
    )

    plan_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )

    # Plan validity
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True,
    )

    # Review tracking
    last_reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    last_reviewed_by: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )

    next_review_due: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    assessment: Mapped[SafetyAssessment | None] = relationship(
        "SafetyAssessment",
        back_populates="safety_plan",
    )

    def __repr__(self) -> str:
        return (
            f"<SafetyPlan(id={self.id}, user_id={self.user_id}, "
            f"status={self.status})>"
        )


@SchemaRegistry.register
class RiskFactor(ClinicalBase):
    """Risk factor entity for tracking individual risk indicators.

    Stores specific risk factors identified during assessments with
    severity levels and temporal tracking. All data encrypted as PHI.
    """

    __tablename__ = "risk_factors"

    # Link to assessment
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("safety_assessments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Associated safety assessment"
    )

    # Risk factor details
    factor_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Type of risk factor: SUICIDAL_IDEATION, SUBSTANCE_USE, etc."
    )

    factor_description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Detailed description of the risk factor (encrypted)"
    )

    severity_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Severity level 1-5 (1=minimal, 5=critical)"
    )

    # Temporal tracking
    identified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="When risk factor was identified"
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When risk factor was resolved/mitigated"
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Whether risk factor is still present"
    )

    # Additional context
    context_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional contextual data about the risk factor"
    )

    def __repr__(self) -> str:
        return (
            f"<RiskFactor(id={self.id}, factor_type={self.factor_type}, "
            f"severity_level={self.severity_level}, is_active={self.is_active})>"
        )


@SchemaRegistry.register
class ContraindicationCheck(SafetyEventBase):
    """Contraindication check entity for medication/intervention safety.

    Immutable records of contraindication checks performed before prescribing
    or recommending interventions. Never deleted for compliance.

    Inherits from SafetyEventBase for immutability and RESTRICT deletion.
    """

    __tablename__ = "contraindication_checks"

    # Check metadata
    check_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Type of check: MEDICATION, INTERVENTION, THERAPY_MODALITY, etc."
    )

    # Subject of check
    subject_identifier: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Identifier of medication/intervention being checked"
    )

    subject_details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Details about the medication/intervention"
    )

    # Check results
    contraindications_found: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        index=True,
        comment="Whether any contraindications were found"
    )

    contraindication_details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Detailed contraindications if found (encrypted)"
    )

    # Risk assessment
    risk_assessment: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Overall risk: NONE, LOW, ELEVATED, HIGH, CRITICAL"
    )

    # Recommendations
    recommended_action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="PROCEED, PROCEED_WITH_CAUTION, DO_NOT_PROCEED, CONSULT_SPECIALIST"
    )

    recommendation_rationale: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Rationale for the recommendation"
    )

    # Check provenance
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="When check was performed"
    )

    checked_by_system: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="System/service that performed the check"
    )

    check_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Version of contraindication rules used"
    )

    # Data sources
    data_sources_consulted: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="List of data sources/databases consulted"
    )

    def __repr__(self) -> str:
        return (
            f"<ContraindicationCheck(id={self.id}, subject={self.subject_identifier}, "
            f"found={self.contraindications_found}, risk={self.risk_assessment})>"
        )


# Export all entities
__all__ = [
    # Enumerations
    "AssessmentType",
    "RiskLevel",
    "SafetyPlanStatus",
    "IncidentSeverity",
    # Entities
    "SafetyAssessment",
    "SafetyPlan",
    "RiskFactor",
    "ContraindicationCheck",
]
