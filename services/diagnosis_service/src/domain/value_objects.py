"""
Solace-AI Diagnosis Service - Domain Value Objects.
Immutable value objects for clinical diagnosis domain.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from ..schemas import SeverityLevel, ConfidenceLevel, SymptomType, DiagnosisPhase


@dataclass(frozen=True)
class SeverityScore:
    """Immutable value object for severity scoring."""
    raw_score: int
    max_score: int
    level: SeverityLevel
    instrument: str
    interpretation: str

    @classmethod
    def from_phq9(cls, score: int) -> SeverityScore:
        """Create severity score from PHQ-9."""
        if score >= 20:
            level, interp = SeverityLevel.SEVERE, "Severe depression"
        elif score >= 15:
            level, interp = SeverityLevel.MODERATELY_SEVERE, "Moderately severe depression"
        elif score >= 10:
            level, interp = SeverityLevel.MODERATE, "Moderate depression"
        elif score >= 5:
            level, interp = SeverityLevel.MILD, "Mild depression"
        else:
            level, interp = SeverityLevel.MINIMAL, "Minimal symptoms"
        return cls(raw_score=score, max_score=27, level=level, instrument="PHQ-9", interpretation=interp)

    @classmethod
    def from_gad7(cls, score: int) -> SeverityScore:
        """Create severity score from GAD-7."""
        if score >= 15:
            level, interp = SeverityLevel.SEVERE, "Severe anxiety"
        elif score >= 10:
            level, interp = SeverityLevel.MODERATE, "Moderate anxiety"
        elif score >= 5:
            level, interp = SeverityLevel.MILD, "Mild anxiety"
        else:
            level, interp = SeverityLevel.MINIMAL, "Minimal anxiety"
        return cls(raw_score=score, max_score=21, level=level, instrument="GAD-7", interpretation=interp)

    @classmethod
    def from_pcl5(cls, score: int) -> SeverityScore:
        """Create severity score from PCL-5."""
        if score >= 61:
            level, interp = SeverityLevel.SEVERE, "Severe PTSD symptoms"
        elif score >= 44:
            level, interp = SeverityLevel.MODERATE, "Moderate PTSD symptoms"
        elif score >= 33:
            level, interp = SeverityLevel.MILD, "Mild PTSD symptoms - meets clinical threshold"
        else:
            level, interp = SeverityLevel.MINIMAL, "Below clinical threshold"
        return cls(raw_score=score, max_score=80, level=level, instrument="PCL-5", interpretation=interp)

    @property
    def percentage(self) -> float:
        """Get score as percentage of maximum."""
        return (self.raw_score / self.max_score) * 100 if self.max_score > 0 else 0.0

    @property
    def is_clinical(self) -> bool:
        """Check if score meets clinical threshold."""
        return self.level not in [SeverityLevel.MINIMAL, SeverityLevel.MILD]


@dataclass(frozen=True)
class ConfidenceScore:
    """Immutable value object for diagnostic confidence."""
    value: Decimal
    level: ConfidenceLevel
    interval_low: Decimal
    interval_high: Decimal
    evidence_count: int = 0
    challenge_applied: bool = False

    @classmethod
    def create(cls, value: Decimal, evidence_count: int = 0, challenged: bool = False) -> ConfidenceScore:
        """Create confidence score with automatic level calculation."""
        clamped = max(Decimal("0"), min(Decimal("1"), value))
        if clamped >= Decimal("0.8"):
            level = ConfidenceLevel.VERY_HIGH
        elif clamped >= Decimal("0.6"):
            level = ConfidenceLevel.HIGH
        elif clamped >= Decimal("0.4"):
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        margin = Decimal("0.1") if evidence_count >= 3 else Decimal("0.15")
        interval_low = max(Decimal("0"), clamped - margin)
        interval_high = min(Decimal("1"), clamped + margin)
        return cls(value=clamped, level=level, interval_low=interval_low, interval_high=interval_high,
                   evidence_count=evidence_count, challenge_applied=challenged)

    def with_adjustment(self, adjustment: Decimal) -> ConfidenceScore:
        """Create new confidence score with adjustment."""
        return ConfidenceScore.create(self.value + adjustment, self.evidence_count, self.challenge_applied)

    @property
    def is_reliable(self) -> bool:
        """Check if confidence is sufficiently reliable."""
        return self.evidence_count >= 3 and self.level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]


@dataclass(frozen=True)
class TemporalInfo:
    """Immutable value object for symptom temporal information."""
    onset_date: datetime | None = None
    onset_description: str = ""
    duration_description: str = ""
    duration_days: int | None = None
    frequency: str = ""
    pattern: str = ""
    progression: str = ""

    @classmethod
    def create(cls, onset: str | None = None, duration: str | None = None, frequency: str | None = None) -> TemporalInfo:
        """Create temporal info from string descriptions."""
        duration_days = cls._parse_duration_days(duration) if duration else None
        return cls(onset_description=onset or "", duration_description=duration or "", duration_days=duration_days,
                   frequency=frequency or "")

    @staticmethod
    def _parse_duration_days(duration: str) -> int | None:
        """Parse duration string to approximate days."""
        duration_lower = duration.lower()
        if "week" in duration_lower:
            weeks = 1
            for word in duration_lower.split():
                if word.isdigit():
                    weeks = int(word)
                    break
            return weeks * 7
        if "month" in duration_lower:
            months = 1
            for word in duration_lower.split():
                if word.isdigit():
                    months = int(word)
                    break
            return months * 30
        if "year" in duration_lower:
            years = 1
            for word in duration_lower.split():
                if word.isdigit():
                    years = int(word)
                    break
            return years * 365
        if "day" in duration_lower:
            for word in duration_lower.split():
                if word.isdigit():
                    return int(word)
            return 1
        return None

    @property
    def is_chronic(self) -> bool:
        """Check if condition is chronic (> 6 months)."""
        return self.duration_days is not None and self.duration_days >= 180

    @property
    def meets_duration_criterion(self) -> bool:
        """Check if meets typical duration criterion (> 2 weeks)."""
        return self.duration_days is not None and self.duration_days >= 14


@dataclass(frozen=True)
class DiagnosisCriteria:
    """Immutable value object for diagnostic criteria."""
    code: str
    name: str
    criteria_set: str
    required_criteria: tuple[str, ...]
    supporting_criteria: tuple[str, ...]
    exclusionary_criteria: tuple[str, ...]
    minimum_required: int
    duration_requirement: str = ""

    def evaluate(self, symptoms: list[str]) -> tuple[list[str], list[str], bool]:
        """Evaluate symptoms against criteria."""
        symptom_set = {s.lower() for s in symptoms}
        met = [c for c in self.required_criteria if c.lower() in symptom_set]
        missing = [c for c in self.required_criteria if c.lower() not in symptom_set]
        supporting_met = sum(1 for c in self.supporting_criteria if c.lower() in symptom_set)
        threshold_met = len(met) >= self.minimum_required or (len(met) + supporting_met) >= self.minimum_required
        return met, missing, threshold_met

    @property
    def total_criteria_count(self) -> int:
        """Get total number of criteria."""
        return len(self.required_criteria) + len(self.supporting_criteria)


@dataclass(frozen=True)
class HiTOPDimension:
    """Immutable value object for HiTOP dimensional score."""
    dimension: str
    score: Decimal
    percentile: int
    interpretation: str

    @classmethod
    def create(cls, dimension: str, score: Decimal) -> HiTOPDimension:
        """Create HiTOP dimension with automatic interpretation."""
        clamped = max(Decimal("0"), min(Decimal("1"), score))
        if clamped >= Decimal("0.8"):
            percentile, interp = 90, "Very elevated"
        elif clamped >= Decimal("0.6"):
            percentile, interp = 75, "Elevated"
        elif clamped >= Decimal("0.4"):
            percentile, interp = 50, "Moderate"
        elif clamped >= Decimal("0.2"):
            percentile, interp = 25, "Mild"
        else:
            percentile, interp = 10, "Minimal"
        return cls(dimension=dimension, score=clamped, percentile=percentile, interpretation=interp)


@dataclass(frozen=True)
class ClinicalHypothesis:
    """Immutable value object for a clinical hypothesis."""
    hypothesis_id: UUID
    name: str
    dsm5_code: str | None
    icd11_code: str | None
    confidence: ConfidenceScore
    severity: SeverityLevel
    criteria_met: tuple[str, ...]
    criteria_missing: tuple[str, ...]
    supporting_evidence: tuple[str, ...]
    contra_evidence: tuple[str, ...]
    hitop_dimensions: tuple[HiTOPDimension, ...]

    @classmethod
    def create(cls, name: str, confidence_value: Decimal, dsm5_code: str | None = None, icd11_code: str | None = None,
               severity: SeverityLevel = SeverityLevel.MILD, criteria_met: list[str] | None = None,
               criteria_missing: list[str] | None = None, supporting_evidence: list[str] | None = None,
               contra_evidence: list[str] | None = None) -> ClinicalHypothesis:
        """Create a clinical hypothesis."""
        evidence_count = len(criteria_met or []) + len(supporting_evidence or [])
        confidence = ConfidenceScore.create(confidence_value, evidence_count)
        return cls(
            hypothesis_id=uuid4(), name=name, dsm5_code=dsm5_code, icd11_code=icd11_code, confidence=confidence,
            severity=severity, criteria_met=tuple(criteria_met or []), criteria_missing=tuple(criteria_missing or []),
            supporting_evidence=tuple(supporting_evidence or []), contra_evidence=tuple(contra_evidence or []),
            hitop_dimensions=()
        )

    def with_challenge(self, adjustment: Decimal, challenges: list[str]) -> ClinicalHypothesis:
        """Create new hypothesis with challenge applied."""
        new_confidence = self.confidence.with_adjustment(adjustment)
        new_contra = tuple(list(self.contra_evidence) + challenges)
        return ClinicalHypothesis(
            hypothesis_id=self.hypothesis_id, name=self.name, dsm5_code=self.dsm5_code, icd11_code=self.icd11_code,
            confidence=ConfidenceScore.create(new_confidence.value, new_confidence.evidence_count, True),
            severity=self.severity, criteria_met=self.criteria_met, criteria_missing=self.criteria_missing,
            supporting_evidence=self.supporting_evidence, contra_evidence=new_contra, hitop_dimensions=self.hitop_dimensions
        )

    @property
    def is_primary_candidate(self) -> bool:
        """Check if hypothesis qualifies as primary diagnosis."""
        return self.confidence.is_reliable and len(self.criteria_met) >= 2

    @property
    def evidence_ratio(self) -> float:
        """Get ratio of supporting to contradicting evidence."""
        total_contra = len(self.contra_evidence)
        if total_contra == 0:
            return float("inf") if self.supporting_evidence else 1.0
        return len(self.supporting_evidence) / total_contra


@dataclass(frozen=True)
class SessionProgress:
    """Immutable value object for session progress tracking."""
    session_id: UUID
    current_phase: DiagnosisPhase
    phases_completed: tuple[DiagnosisPhase, ...]
    symptom_count: int
    hypothesis_count: int
    confidence_score: Decimal
    messages_exchanged: int
    duration_minutes: int

    @property
    def completion_percentage(self) -> int:
        """Get session completion percentage."""
        phases = [DiagnosisPhase.RAPPORT, DiagnosisPhase.HISTORY, DiagnosisPhase.ASSESSMENT,
                  DiagnosisPhase.DIAGNOSIS, DiagnosisPhase.CLOSURE]
        completed = len(self.phases_completed)
        return int((completed / len(phases)) * 100)

    @property
    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.current_phase == DiagnosisPhase.CLOSURE
