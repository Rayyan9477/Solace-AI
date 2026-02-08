"""
Diagnosis domain entities for the Solace-AI platform.

Centralized SQLAlchemy ORM models for diagnostic sessions, symptoms,
hypotheses, and diagnosis records. Uses AMIE-inspired 4-step reasoning.
All clinical data inherits from ClinicalBase for PHI encryption.
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

from ..base_models import ClinicalBase
from ..schema_registry import SchemaRegistry


# Enumerations

class DiagnosisPhase(str, Enum):
    RAPPORT = "RAPPORT"
    HISTORY = "HISTORY"
    ASSESSMENT = "ASSESSMENT"
    DIAGNOSIS = "DIAGNOSIS"
    CLOSURE = "CLOSURE"


class SymptomType(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    COGNITIVE = "COGNITIVE"
    BEHAVIORAL = "BEHAVIORAL"
    SOMATIC = "SOMATIC"
    EMOTIONAL = "EMOTIONAL"


class DiagnosisSeverity(str, Enum):
    MINIMAL = "MINIMAL"
    MILD = "MILD"
    MODERATE = "MODERATE"
    MODERATELY_SEVERE = "MODERATELY_SEVERE"
    SEVERE = "SEVERE"


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


# Entity Models

@SchemaRegistry.register
class DiagnosisSession(ClinicalBase):
    """Diagnostic session entity for structured clinical interviews.

    Tracks the full diagnostic session including phase progression,
    symptoms extracted, and hypotheses generated via 4-step reasoning.
    """

    __tablename__ = "diagnosis_sessions"
    __phi_fields__: ClassVar[list[str]] = ["summary"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_number: Mapped[int] = mapped_column(
        Integer, nullable=False,
    )
    phase: Mapped[str] = mapped_column(
        String(20), nullable=False, default=DiagnosisPhase.RAPPORT.value,
        index=True, comment="Current diagnostic phase"
    )
    primary_hypothesis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True,
        comment="Current leading hypothesis"
    )

    # Session content (encrypted)
    messages: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="Conversation history (encrypted as PHI)"
    )
    safety_flags: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc), index=True,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True,
    )

    # Summary and recommendations
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommendations: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    session_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )

    # Relationships
    symptoms: Mapped[list[Symptom]] = relationship(
        "Symptom", back_populates="session",
    )
    hypotheses: Mapped[list[Hypothesis]] = relationship(
        "Hypothesis", back_populates="session",
    )
    diagnosis_records: Mapped[list[DiagnosisRecord]] = relationship(
        "DiagnosisRecord", back_populates="session",
    )

    def __repr__(self) -> str:
        return (
            f"<DiagnosisSession(id={self.id}, user_id={self.user_id}, "
            f"phase={self.phase}, session_number={self.session_number})>"
        )


@SchemaRegistry.register
class Symptom(ClinicalBase):
    """Symptom entity extracted during diagnostic sessions.

    Tracks individual symptoms with type, severity, temporal information,
    and provenance from the conversation.
    """

    __tablename__ = "diagnosis_symptoms"
    __phi_fields__: ClassVar[list[str]] = ["description", "extracted_from"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    symptom_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="POSITIVE, NEGATIVE, COGNITIVE, BEHAVIORAL, SOMATIC, EMOTIONAL"
    )
    severity: Mapped[str] = mapped_column(
        String(30), nullable=False,
        comment="Symptom severity level"
    )

    # Temporal information
    onset: Mapped[str | None] = mapped_column(String(200), nullable=True)
    duration: Mapped[str | None] = mapped_column(String(200), nullable=True)
    frequency: Mapped[str | None] = mapped_column(String(200), nullable=True)
    triggers: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )

    # Provenance
    extracted_from: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Source text from which symptom was extracted"
    )
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5,
        comment="Extraction confidence score 0.0-1.0"
    )

    # Validation
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    validation_source: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    session: Mapped[DiagnosisSession] = relationship(
        "DiagnosisSession", back_populates="symptoms",
    )

    def __repr__(self) -> str:
        return (
            f"<Symptom(id={self.id}, name={self.name}, "
            f"type={self.symptom_type}, severity={self.severity})>"
        )


@SchemaRegistry.register
class Hypothesis(ClinicalBase):
    """Diagnostic hypothesis entity for differential diagnosis.

    Tracks candidate diagnoses with DSM-5/ICD-11 codes, evidence,
    confidence levels, and Devil's Advocate challenge results.
    """

    __tablename__ = "diagnosis_hypotheses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    dsm5_code: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    icd11_code: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)

    # Confidence
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5,
        comment="Hypothesis confidence score 0.0-1.0"
    )
    confidence_level: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ConfidenceLevel.MEDIUM.value,
    )

    # Evidence
    criteria_met: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    criteria_missing: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    supporting_evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    contra_evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)

    severity: Mapped[str] = mapped_column(
        String(30), nullable=False, default=DiagnosisSeverity.MODERATE.value,
    )
    hitop_dimensions: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="HiTOP dimensional scores"
    )

    # Devil's Advocate
    challenged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    challenge_results: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    original_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    session: Mapped[DiagnosisSession] = relationship(
        "DiagnosisSession", back_populates="hypotheses",
    )

    def __repr__(self) -> str:
        return (
            f"<Hypothesis(id={self.id}, name={self.name}, "
            f"confidence={self.confidence}, dsm5={self.dsm5_code})>"
        )


@SchemaRegistry.register
class DiagnosisRecord(ClinicalBase):
    """Finalized diagnosis record for clinical documentation.

    Stores confirmed diagnoses after the diagnostic session is complete.
    Immutable once reviewed by a clinician.
    """

    __tablename__ = "diagnosis_records"
    __phi_fields__: ClassVar[list[str]] = ["primary_diagnosis", "clinician_notes"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    primary_diagnosis: Mapped[str] = mapped_column(String(300), nullable=False)
    dsm5_code: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    icd11_code: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(30), nullable=False)

    symptom_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    supporting_evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    differential_diagnoses: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    recommendations: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    assessment_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Clinical review
    clinician_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    session: Mapped[DiagnosisSession] = relationship(
        "DiagnosisSession", back_populates="diagnosis_records",
    )

    def __repr__(self) -> str:
        return (
            f"<DiagnosisRecord(id={self.id}, diagnosis={self.primary_diagnosis}, "
            f"confidence={self.confidence}, reviewed={self.reviewed})>"
        )


__all__ = [
    "DiagnosisPhase",
    "SymptomType",
    "DiagnosisSeverity",
    "ConfidenceLevel",
    "DiagnosisSession",
    "Symptom",
    "Hypothesis",
    "DiagnosisRecord",
]
