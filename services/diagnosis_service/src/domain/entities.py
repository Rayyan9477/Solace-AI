"""
Solace-AI Diagnosis Service - Domain Entities.
Core domain entities following DDD principles for diagnosis workflow.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
import structlog

from ..schemas import DiagnosisPhase, SeverityLevel, SymptomType, ConfidenceLevel

logger = structlog.get_logger(__name__)


@dataclass
class EntityBase:
    """Base class for all domain entities."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1


@dataclass
class SymptomEntity(EntityBase):
    """Domain entity representing a clinical symptom."""
    name: str = ""
    description: str = ""
    symptom_type: SymptomType = SymptomType.EMOTIONAL
    severity: SeverityLevel = SeverityLevel.MILD
    onset: str | None = None
    duration: str | None = None
    frequency: str | None = None
    triggers: list[str] = field(default_factory=list)
    extracted_from: str | None = None
    confidence: Decimal = field(default=Decimal("0.7"))
    session_id: UUID | None = None
    user_id: UUID | None = None
    is_active: bool = True
    validated: bool = False
    validation_source: str | None = None

    def validate(self, source: str) -> None:
        """Mark symptom as clinically validated."""
        self.validated = True
        self.validation_source = source
        self.touch()

    def deactivate(self) -> None:
        """Mark symptom as no longer active."""
        self.is_active = False
        self.touch()

    def update_severity(self, new_severity: SeverityLevel) -> None:
        """Update symptom severity."""
        self.severity = new_severity
        self.touch()

    def add_trigger(self, trigger: str) -> None:
        """Add a trigger to the symptom."""
        if trigger not in self.triggers:
            self.triggers.append(trigger)
            self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "symptom_type": self.symptom_type.value,
            "severity": self.severity.value,
            "onset": self.onset,
            "duration": self.duration,
            "frequency": self.frequency,
            "triggers": self.triggers,
            "extracted_from": self.extracted_from,
            "confidence": str(self.confidence),
            "is_active": self.is_active,
            "validated": self.validated,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class HypothesisEntity(EntityBase):
    """Domain entity representing a diagnostic hypothesis."""
    name: str = ""
    dsm5_code: str | None = None
    icd11_code: str | None = None
    confidence: Decimal = field(default=Decimal("0.5"))
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    criteria_met: list[str] = field(default_factory=list)
    criteria_missing: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    contra_evidence: list[str] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MILD
    hitop_dimensions: dict[str, Decimal] = field(default_factory=dict)
    session_id: UUID | None = None
    challenged: bool = False
    challenge_results: list[str] = field(default_factory=list)
    calibrated: bool = False
    original_confidence: Decimal | None = None

    def apply_challenge(self, challenges: list[str], confidence_adjustment: Decimal) -> None:
        """Apply Devil's Advocate challenge results."""
        self.challenged = True
        self.challenge_results = challenges
        self.original_confidence = self.confidence
        self.confidence = max(Decimal("0.1"), self.confidence + confidence_adjustment)
        self._update_confidence_level()
        self.touch()

    def calibrate(self, calibrated_confidence: Decimal) -> None:
        """Apply calibrated confidence score."""
        self.calibrated = True
        if self.original_confidence is None:
            self.original_confidence = self.confidence
        self.confidence = calibrated_confidence
        self._update_confidence_level()
        self.touch()

    def _update_confidence_level(self) -> None:
        """Update confidence level based on score."""
        if self.confidence >= Decimal("0.8"):
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.confidence >= Decimal("0.6"):
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= Decimal("0.4"):
            self.confidence_level = ConfidenceLevel.MEDIUM
        else:
            self.confidence_level = ConfidenceLevel.LOW

    def add_evidence(self, evidence: str, supporting: bool = True) -> None:
        """Add supporting or contradicting evidence."""
        if supporting:
            if evidence not in self.supporting_evidence:
                self.supporting_evidence.append(evidence)
        else:
            if evidence not in self.contra_evidence:
                self.contra_evidence.append(evidence)
        self.touch()

    def mark_criterion_met(self, criterion: str) -> None:
        """Mark a diagnostic criterion as met."""
        if criterion not in self.criteria_met:
            self.criteria_met.append(criterion)
        if criterion in self.criteria_missing:
            self.criteria_missing.remove(criterion)
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "dsm5_code": self.dsm5_code,
            "icd11_code": self.icd11_code,
            "confidence": str(self.confidence),
            "confidence_level": self.confidence_level.value,
            "criteria_met": self.criteria_met,
            "criteria_missing": self.criteria_missing,
            "supporting_evidence": self.supporting_evidence,
            "contra_evidence": self.contra_evidence,
            "severity": self.severity.value,
            "challenged": self.challenged,
            "calibrated": self.calibrated,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class DiagnosisSessionEntity(EntityBase):
    """Domain entity representing a diagnosis session."""
    user_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    phase: DiagnosisPhase = DiagnosisPhase.RAPPORT
    symptoms: list[SymptomEntity] = field(default_factory=list)
    hypotheses: list[HypothesisEntity] = field(default_factory=list)
    primary_hypothesis_id: UUID | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    is_active: bool = True
    summary: str | None = None
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_symptom(self, symptom: SymptomEntity) -> None:
        """Add a symptom to the session."""
        symptom.session_id = self.id
        symptom.user_id = self.user_id
        self.symptoms.append(symptom)
        self.touch()

    def add_hypothesis(self, hypothesis: HypothesisEntity) -> None:
        """Add a hypothesis to the session."""
        hypothesis.session_id = self.id
        self.hypotheses.append(hypothesis)
        self.touch()

    def set_primary_hypothesis(self, hypothesis_id: UUID) -> None:
        """Set the primary hypothesis."""
        if any(h.id == hypothesis_id for h in self.hypotheses):
            self.primary_hypothesis_id = hypothesis_id
            self.touch()

    def get_primary_hypothesis(self) -> HypothesisEntity | None:
        """Get the primary hypothesis."""
        if self.primary_hypothesis_id is None:
            return self.hypotheses[0] if self.hypotheses else None
        return next((h for h in self.hypotheses if h.id == self.primary_hypothesis_id), None)

    def transition_phase(self, new_phase: DiagnosisPhase) -> None:
        """Transition to a new dialogue phase."""
        logger.debug("phase_transition", from_phase=self.phase.value, to_phase=new_phase.value)
        self.phase = new_phase
        self.touch()

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session."""
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.touch()

    def add_safety_flag(self, flag: str) -> None:
        """Add a safety flag."""
        if flag not in self.safety_flags:
            self.safety_flags.append(flag)
            self.touch()

    def end_session(self, summary: str | None = None, recommendations: list[str] | None = None) -> None:
        """End the session."""
        self.is_active = False
        self.ended_at = datetime.now(timezone.utc)
        if summary:
            self.summary = summary
        if recommendations:
            self.recommendations = recommendations
        self.touch()

    @property
    def duration_minutes(self) -> int:
        """Get session duration in minutes."""
        end = self.ended_at or datetime.now(timezone.utc)
        return int((end - self.started_at).total_seconds() / 60)

    @property
    def active_symptoms(self) -> list[SymptomEntity]:
        """Get only active symptoms."""
        return [s for s in self.symptoms if s.is_active]

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "session_number": self.session_number,
            "phase": self.phase.value,
            "symptom_count": len(self.symptoms),
            "hypothesis_count": len(self.hypotheses),
            "primary_hypothesis_id": str(self.primary_hypothesis_id) if self.primary_hypothesis_id else None,
            "message_count": len(self.messages),
            "safety_flags": self.safety_flags,
            "is_active": self.is_active,
            "duration_minutes": self.duration_minutes,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


@dataclass
class DiagnosisRecordEntity(EntityBase):
    """Domain entity representing a completed diagnosis record."""
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    primary_diagnosis: str = ""
    dsm5_code: str | None = None
    icd11_code: str | None = None
    confidence: Decimal = field(default=Decimal("0.5"))
    severity: SeverityLevel = SeverityLevel.MILD
    symptom_summary: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    differential_diagnoses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    assessment_scores: dict[str, int] = field(default_factory=dict)
    clinician_notes: str | None = None
    reviewed: bool = False
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    def mark_reviewed(self, reviewer: str) -> None:
        """Mark diagnosis as reviewed."""
        self.reviewed = True
        self.reviewed_by = reviewer
        self.reviewed_at = datetime.now(timezone.utc)
        self.touch()

    def add_clinician_note(self, note: str) -> None:
        """Add a clinician note."""
        self.clinician_notes = note
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "session_id": str(self.session_id),
            "primary_diagnosis": self.primary_diagnosis,
            "dsm5_code": self.dsm5_code,
            "icd11_code": self.icd11_code,
            "confidence": str(self.confidence),
            "severity": self.severity.value,
            "symptom_summary": self.symptom_summary,
            "differential_diagnoses": self.differential_diagnoses,
            "recommendations": self.recommendations,
            "reviewed": self.reviewed,
            "created_at": self.created_at.isoformat(),
        }
