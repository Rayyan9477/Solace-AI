"""
Solace-AI Therapy Service - Domain Value Objects.
Immutable value objects for therapeutic techniques and outcome measurements.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar
from uuid import UUID, uuid4

from ..schemas import (
    TherapyModality, TechniqueCategory, DifficultyLevel, DeliveryMode,
    OutcomeInstrument, SeverityLevel,
)


@dataclass(frozen=True)
class Technique:
    """
    Immutable therapeutic technique value object.

    Represents a specific evidence-based intervention technique
    with its properties, requirements, and applicability criteria.
    """
    technique_id: UUID
    name: str
    modality: TherapyModality
    category: TechniqueCategory
    description: str
    instructions: str = ""
    duration_minutes: int = 15
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    delivery_mode: DeliveryMode = DeliveryMode.GUIDED
    requires_homework: bool = False
    contraindications: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    target_symptoms: tuple[str, ...] = ()
    effectiveness_rating: Decimal = Decimal("0.7")
    evidence_level: str = "moderate"

    def __post_init__(self) -> None:
        """Validate technique data on initialization."""
        if self.duration_minutes < 1 or self.duration_minutes > 180:
            raise ValueError("Duration must be between 1 and 180 minutes")
        if self.effectiveness_rating < Decimal("0") or self.effectiveness_rating > Decimal("1"):
            raise ValueError("Effectiveness rating must be between 0 and 1")

    def is_applicable_for(self, severity: SeverityLevel) -> bool:
        """Check if technique is appropriate for severity level."""
        severity_restrictions = {
            DifficultyLevel.BEGINNER: [SeverityLevel.MINIMAL, SeverityLevel.MILD, SeverityLevel.MODERATE],
            DifficultyLevel.INTERMEDIATE: [SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.MODERATELY_SEVERE],
            DifficultyLevel.ADVANCED: [SeverityLevel.MODERATE, SeverityLevel.MODERATELY_SEVERE, SeverityLevel.SEVERE],
        }
        allowed_severities = severity_restrictions.get(self.difficulty, [])
        return severity in allowed_severities

    def has_contraindication(self, condition: str) -> bool:
        """Check if technique is contraindicated for a condition."""
        condition_lower = condition.lower()
        return any(contra.lower() in condition_lower or condition_lower in contra.lower()
                   for contra in self.contraindications)

    def meets_prerequisites(self, acquired_skills: list[str]) -> bool:
        """Check if prerequisites are met."""
        if not self.prerequisites:
            return True
        return all(prereq in acquired_skills for prereq in self.prerequisites)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "technique_id": str(self.technique_id),
            "name": self.name,
            "modality": self.modality.value,
            "category": self.category.value,
            "description": self.description,
            "instructions": self.instructions,
            "duration_minutes": self.duration_minutes,
            "difficulty": self.difficulty.value,
            "delivery_mode": self.delivery_mode.value,
            "requires_homework": self.requires_homework,
            "contraindications": list(self.contraindications),
            "prerequisites": list(self.prerequisites),
            "target_symptoms": list(self.target_symptoms),
            "effectiveness_rating": str(self.effectiveness_rating),
            "evidence_level": self.evidence_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Technique:
        """Create from dictionary representation."""
        return cls(
            technique_id=UUID(data["technique_id"]) if isinstance(data["technique_id"], str) else data["technique_id"],
            name=data["name"],
            modality=TherapyModality(data["modality"]) if isinstance(data["modality"], str) else data["modality"],
            category=TechniqueCategory(data["category"]) if isinstance(data["category"], str) else data["category"],
            description=data.get("description", ""),
            instructions=data.get("instructions", ""),
            duration_minutes=data.get("duration_minutes", 15),
            difficulty=DifficultyLevel(data["difficulty"]) if isinstance(data.get("difficulty"), str) else data.get("difficulty", DifficultyLevel.BEGINNER),
            delivery_mode=DeliveryMode(data["delivery_mode"]) if isinstance(data.get("delivery_mode"), str) else data.get("delivery_mode", DeliveryMode.GUIDED),
            requires_homework=data.get("requires_homework", False),
            contraindications=tuple(data.get("contraindications", [])),
            prerequisites=tuple(data.get("prerequisites", [])),
            target_symptoms=tuple(data.get("target_symptoms", [])),
            effectiveness_rating=Decimal(data.get("effectiveness_rating", "0.7")),
            evidence_level=data.get("evidence_level", "moderate"),
        )


@dataclass(frozen=True)
class OutcomeMeasure:
    """
    Immutable outcome measurement value object.

    Represents a single outcome measurement with scoring,
    clinical interpretation, and change detection.
    """
    measure_id: UUID
    instrument: OutcomeInstrument
    raw_score: int
    subscale_scores: tuple[tuple[str, int], ...] = ()
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: UUID | None = None
    notes: str = ""

    INSTRUMENT_CONFIG: ClassVar[dict[OutcomeInstrument, dict[str, Any]]] = {
        OutcomeInstrument.PHQ9: {
            "name": "Patient Health Questionnaire-9",
            "min_score": 0, "max_score": 27, "items": 9,
            "thresholds": [(5, "minimal"), (10, "mild"), (15, "moderate"), (20, "moderately_severe"), (27, "severe")],
            "rci": 6.0, "clinical_cutoff": 10, "higher_worse": True,
        },
        OutcomeInstrument.GAD7: {
            "name": "Generalized Anxiety Disorder-7",
            "min_score": 0, "max_score": 21, "items": 7,
            "thresholds": [(5, "minimal"), (10, "mild"), (15, "moderate"), (21, "severe")],
            "rci": 4.0, "clinical_cutoff": 8, "higher_worse": True,
        },
        OutcomeInstrument.ORS: {
            "name": "Outcome Rating Scale",
            "min_score": 0, "max_score": 40, "items": 4,
            "thresholds": [(25, "clinical"), (40, "non_clinical")],
            "rci": 5.0, "clinical_cutoff": 25, "higher_worse": False,
        },
        OutcomeInstrument.SRS: {
            "name": "Session Rating Scale",
            "min_score": 0, "max_score": 40, "items": 4,
            "thresholds": [(36, "at_risk"), (40, "good")],
            "rci": 5.0, "clinical_cutoff": 36, "higher_worse": False,
        },
        OutcomeInstrument.PCL5: {
            "name": "PTSD Checklist for DSM-5",
            "min_score": 0, "max_score": 80, "items": 20,
            "thresholds": [(33, "below_clinical"), (80, "clinical")],
            "rci": 10.0, "clinical_cutoff": 33, "higher_worse": True,
        },
        OutcomeInstrument.CORE10: {
            "name": "Clinical Outcomes in Routine Evaluation-10",
            "min_score": 0, "max_score": 40, "items": 10,
            "thresholds": [(10, "healthy"), (25, "moderate"), (40, "severe")],
            "rci": 6.0, "clinical_cutoff": 10, "higher_worse": True,
        },
        OutcomeInstrument.DASS21: {
            "name": "Depression Anxiety Stress Scales-21",
            "min_score": 0, "max_score": 126, "items": 21,
            "thresholds": [(21, "normal"), (42, "mild"), (63, "moderate"), (84, "severe"), (126, "extremely_severe")],
            "rci": 8.0, "clinical_cutoff": 21, "higher_worse": True,
        },
    }

    def __post_init__(self) -> None:
        """Validate measure data."""
        config = self.INSTRUMENT_CONFIG.get(self.instrument)
        if config:
            min_score = config["min_score"]
            max_score = config["max_score"]
            if self.raw_score < min_score or self.raw_score > max_score:
                raise ValueError(f"Score {self.raw_score} out of range [{min_score}, {max_score}] for {self.instrument.value}")

    @property
    def severity_category(self) -> SeverityLevel:
        """Determine clinical severity category."""
        config = self.INSTRUMENT_CONFIG.get(self.instrument)
        if not config:
            return SeverityLevel.MODERATE
        thresholds = config["thresholds"]
        for threshold, label in thresholds:
            if self.raw_score <= threshold:
                return self._label_to_severity(label)
        return SeverityLevel.SEVERE

    def _label_to_severity(self, label: str) -> SeverityLevel:
        """Map threshold label to SeverityLevel."""
        mapping = {
            "minimal": SeverityLevel.MINIMAL,
            "mild": SeverityLevel.MILD,
            "moderate": SeverityLevel.MODERATE,
            "moderately_severe": SeverityLevel.MODERATELY_SEVERE,
            "severe": SeverityLevel.SEVERE,
            "normal": SeverityLevel.MINIMAL,
            "healthy": SeverityLevel.MINIMAL,
            "below_clinical": SeverityLevel.MILD,
            "clinical": SeverityLevel.MODERATE,
            "non_clinical": SeverityLevel.MINIMAL,
            "at_risk": SeverityLevel.MILD,
            "good": SeverityLevel.MINIMAL,
            "extremely_severe": SeverityLevel.SEVERE,
        }
        return mapping.get(label, SeverityLevel.MODERATE)

    @property
    def is_clinical(self) -> bool:
        """Check if score is in clinical range."""
        config = self.INSTRUMENT_CONFIG.get(self.instrument)
        if not config:
            return False
        cutoff = config["clinical_cutoff"]
        higher_worse = config.get("higher_worse", True)
        if higher_worse:
            return self.raw_score >= cutoff
        return self.raw_score < cutoff

    @property
    def normalized_score(self) -> float:
        """Calculate normalized score (0-100 scale)."""
        config = self.INSTRUMENT_CONFIG.get(self.instrument)
        if not config:
            return 0.0
        min_score = config["min_score"]
        max_score = config["max_score"]
        normalized = ((self.raw_score - min_score) / (max_score - min_score)) * 100
        if not config.get("higher_worse", True):
            normalized = 100 - normalized
        return round(normalized, 1)

    def calculate_change(self, previous: OutcomeMeasure) -> dict[str, Any]:
        """Calculate change from previous measurement."""
        if self.instrument != previous.instrument:
            raise ValueError("Cannot compare different instruments")
        config = self.INSTRUMENT_CONFIG.get(self.instrument)
        if not config:
            return {"raw_change": self.raw_score - previous.raw_score}
        raw_change = self.raw_score - previous.raw_score
        rci = config["rci"]
        higher_worse = config.get("higher_worse", True)
        reliable_change = abs(raw_change) >= rci
        if higher_worse:
            improved = raw_change < 0
            deteriorated = raw_change > 0
        else:
            improved = raw_change > 0
            deteriorated = raw_change < 0
        clinically_significant = (
            reliable_change and improved and
            previous.is_clinical and not self.is_clinical
        )
        if reliable_change:
            if improved:
                status = "improved"
            elif deteriorated:
                status = "deteriorated"
            else:
                status = "no_change"
        else:
            status = "no_change"
        return {
            "raw_change": raw_change,
            "percent_change": round((raw_change / max(previous.raw_score, 1)) * 100, 1),
            "reliable_change": reliable_change,
            "clinically_significant": clinically_significant,
            "status": status,
            "previous_score": previous.raw_score,
            "current_score": self.raw_score,
            "previous_clinical": previous.is_clinical,
            "current_clinical": self.is_clinical,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "measure_id": str(self.measure_id),
            "instrument": self.instrument.value,
            "raw_score": self.raw_score,
            "subscale_scores": dict(self.subscale_scores),
            "recorded_at": self.recorded_at.isoformat(),
            "session_id": str(self.session_id) if self.session_id else None,
            "notes": self.notes,
            "severity_category": self.severity_category.value,
            "is_clinical": self.is_clinical,
            "normalized_score": self.normalized_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeMeasure:
        """Create from dictionary representation."""
        subscale_scores = data.get("subscale_scores", {})
        if isinstance(subscale_scores, dict):
            subscale_scores = tuple(subscale_scores.items())
        return cls(
            measure_id=UUID(data["measure_id"]) if isinstance(data["measure_id"], str) else data["measure_id"],
            instrument=OutcomeInstrument(data["instrument"]) if isinstance(data["instrument"], str) else data["instrument"],
            raw_score=data["raw_score"],
            subscale_scores=subscale_scores,
            recorded_at=datetime.fromisoformat(data["recorded_at"]) if isinstance(data.get("recorded_at"), str) else data.get("recorded_at", datetime.now(timezone.utc)),
            session_id=UUID(data["session_id"]) if data.get("session_id") and isinstance(data["session_id"], str) else data.get("session_id"),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class TherapeuticRationale:
    """Value object representing the rationale for a therapeutic decision."""
    technique_id: UUID
    selection_reason: str
    clinical_factors: tuple[str, ...] = ()
    personality_factors: tuple[str, ...] = ()
    contextual_factors: tuple[str, ...] = ()
    confidence_score: Decimal = Decimal("0.7")
    alternatives_considered: tuple[UUID, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "technique_id": str(self.technique_id),
            "selection_reason": self.selection_reason,
            "clinical_factors": list(self.clinical_factors),
            "personality_factors": list(self.personality_factors),
            "contextual_factors": list(self.contextual_factors),
            "confidence_score": str(self.confidence_score),
            "alternatives_considered": [str(uid) for uid in self.alternatives_considered],
        }


@dataclass(frozen=True)
class SessionContext:
    """Value object representing session therapeutic context."""
    session_number: int
    treatment_phase: str
    presenting_concerns: tuple[str, ...] = ()
    mood_state: str = ""
    energy_level: str = ""
    sleep_quality: str = ""
    recent_stressors: tuple[str, ...] = ()
    coping_used: tuple[str, ...] = ()
    support_available: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_number": self.session_number,
            "treatment_phase": self.treatment_phase,
            "presenting_concerns": list(self.presenting_concerns),
            "mood_state": self.mood_state,
            "energy_level": self.energy_level,
            "sleep_quality": self.sleep_quality,
            "recent_stressors": list(self.recent_stressors),
            "coping_used": list(self.coping_used),
            "support_available": list(self.support_available),
        }
