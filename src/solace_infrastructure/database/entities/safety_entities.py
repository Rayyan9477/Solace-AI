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
from typing import Any

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
    """Risk level classification for safety assessments."""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
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

    # Primary identification
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # Assessment metadata
    assessment_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of assessment: PRE_CHECK, POST_CHECK, CONTINUOUS, etc."
    )

    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Overall risk level: MINIMAL, LOW, MODERATE, HIGH, CRITICAL"
    )

    # Assessment details (encrypted as PHI)
    risk_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Numerical risk score (0.0-1.0)"
    )

    risk_factors: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Detailed risk factors identified (encrypted)"
    )

    protective_factors: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Protective factors present (encrypted)"
    )

    assessment_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Clinical notes from assessment (encrypted as PHI)"
    )

    # Recommendations
    recommended_interventions: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Recommended interventions based on assessment"
    )

    immediate_actions_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="Whether immediate intervention is required"
    )

    # Timing and follow-up
    assessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="Timestamp when assessment was performed"
    )

    next_assessment_due: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When next assessment is due"
    )

    # Assessment provenance
    assessor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="ID of clinician/system who performed assessment"
    )

    assessment_method: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
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
            f"risk_level={self.risk_level}, assessed_at={self.assessed_at})>"
        )


@SchemaRegistry.register
class SafetyPlan(ClinicalBase):
    """Safety plan entity for comprehensive user safety planning.

    Stores detailed safety plans including warning signs, coping strategies,
    emergency contacts, and safe environment actions. All data encrypted as PHI.

    Inherits from ClinicalBase for PHI protection and audit trail.
    """

    __tablename__ = "safety_plans"

    # Primary identification
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # Plan metadata
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=SafetyPlanStatus.DRAFT.value,
        index=True,
        comment="Plan status: DRAFT, ACTIVE, UNDER_REVIEW, EXPIRED, ARCHIVED"
    )

    # Link to assessment
    assessment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("safety_assessments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Associated safety assessment"
    )

    # Plan content (all encrypted as PHI)
    warning_signs: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="List of warning signs and triggers (encrypted)"
    )

    coping_strategies: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Coping strategies and self-help techniques (encrypted)"
    )

    emergency_contacts: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Emergency contacts list (encrypted as PHI)"
    )

    safe_environment_actions: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Actions to make environment safer (encrypted)"
    )

    professional_resources: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Professional resources and crisis lines"
    )

    # Plan notes
    plan_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Additional notes about the safety plan (encrypted)"
    )

    # Plan validity
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="When plan becomes effective"
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When plan expires (for periodic review)"
    )

    # Review tracking
    last_reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time plan was reviewed"
    )

    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="Clinician who last reviewed the plan"
    )

    # Relationships
    assessment: Mapped[SafetyAssessment | None] = relationship(
        "SafetyAssessment",
        back_populates="safety_plan",
    )

    def __repr__(self) -> str:
        return (
            f"<SafetyPlan(id={self.id}, user_id={self.user_id}, "
            f"status={self.status}, effective_from={self.effective_from})>"
        )


@SchemaRegistry.register
class RiskFactor(ClinicalBase):
    """Risk factor entity for tracking individual risk indicators.

    Stores specific risk factors identified during assessments with
    severity levels and temporal tracking. All data encrypted as PHI.
    """

    __tablename__ = "risk_factors"

    # Primary identification
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

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

    # Primary identification
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

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
        comment="Overall risk: MINIMAL, LOW, MODERATE, HIGH, CRITICAL"
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
