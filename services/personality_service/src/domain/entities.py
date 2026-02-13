"""
Solace-AI Personality Service - Domain Entities.
Aggregate roots and entities for personality profile management.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
import structlog

from ..schemas import PersonalityTrait, AssessmentSource, CommunicationStyleType
from .value_objects import OceanScores, CommunicationStyle, TraitScore, AssessmentMetadata

logger = structlog.get_logger(__name__)


@dataclass
class TraitAssessment:
    """
    Entity representing a single personality trait assessment.
    Contains the raw assessment results from a specific source.
    """
    user_id: UUID
    assessment_id: UUID = field(default_factory=uuid4)
    ocean_scores: OceanScores = field(default_factory=OceanScores.neutral)
    source: AssessmentSource = AssessmentSource.TEXT_ANALYSIS
    metadata: AssessmentMetadata | None = None
    evidence: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def add_evidence(self, marker: str) -> None:
        """Add evidence marker to assessment."""
        if marker and marker not in self.evidence:
            self.evidence.append(marker)

    def get_trait_value(self, trait: PersonalityTrait) -> Decimal:
        """Get value for a specific trait."""
        return self.ocean_scores.get_trait(trait)

    @property
    def confidence(self) -> Decimal:
        """Get overall confidence of assessment."""
        return self.ocean_scores.overall_confidence

    @property
    def is_reliable(self) -> bool:
        """Check if assessment has reliable confidence."""
        return self.confidence >= Decimal("0.5")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "assessment_id": str(self.assessment_id), "user_id": str(self.user_id),
            "ocean_scores": self.ocean_scores.to_dict(), "source": self.source.value,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "evidence": self.evidence, "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraitAssessment:
        """Create from dictionary representation."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        metadata = None
        if data.get("metadata"):
            metadata = AssessmentMetadata.from_dict(data["metadata"])
        return cls(
            assessment_id=UUID(data["assessment_id"]) if isinstance(data["assessment_id"], str) else data["assessment_id"],
            user_id=UUID(data["user_id"]) if isinstance(data["user_id"], str) else data["user_id"],
            ocean_scores=OceanScores.from_dict(data["ocean_scores"]),
            source=AssessmentSource(data["source"]) if isinstance(data["source"], str) else data["source"],
            metadata=metadata, evidence=list(data.get("evidence", [])),
            created_at=created_at or datetime.now(timezone.utc),
            version=data.get("version", 1),
        )


@dataclass
class PersonalityProfile:
    """
    Aggregate root for personality profile management.
    Maintains the user's personality state and assessment history.
    """
    user_id: UUID
    profile_id: UUID = field(default_factory=uuid4)
    ocean_scores: OceanScores | None = None
    communication_style: CommunicationStyle | None = None
    assessment_count: int = 0
    stability_score: Decimal = Decimal("0.0")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    assessment_history: list[TraitAssessment] = field(default_factory=list)
    _max_history_size: int = field(default=100, repr=False)
    _pending_events: list[Any] = field(default_factory=list, repr=False)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def add_assessment(self, assessment: TraitAssessment) -> None:
        """Add new assessment and update aggregated scores."""
        if self.ocean_scores is None:
            self.ocean_scores = assessment.ocean_scores
        else:
            self.ocean_scores = self.ocean_scores.aggregate_with(assessment.ocean_scores)
        self.assessment_history.append(assessment)
        if len(self.assessment_history) > self._max_history_size:
            self.assessment_history = self.assessment_history[-self._max_history_size:]
        self.assessment_count += 1
        self.stability_score = self._compute_stability()
        self.communication_style = CommunicationStyle.from_ocean(self.ocean_scores, self.communication_style.style_id if self.communication_style else None)
        self.version += 1
        self.touch()
        logger.debug("assessment_added", profile_id=str(self.profile_id), assessment_count=self.assessment_count)

    def _compute_stability(self) -> Decimal:
        """Compute profile stability score based on assessment variance."""
        if self.assessment_count < 3:
            return Decimal("0.3")
        recent = self.assessment_history[-5:] if len(self.assessment_history) >= 5 else self.assessment_history
        if len(recent) < 2:
            return Decimal("0.5")
        variance_sum = Decimal("0.0")
        for trait in PersonalityTrait:
            values = [a.get_trait_value(trait) for a in recent]
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                variance_sum += variance
        avg_variance = variance_sum / Decimal("5")
        stability = Decimal("1.0") - avg_variance * Decimal("10")
        return max(Decimal("0.0"), min(Decimal("1.0"), stability))

    @property
    def is_stable(self) -> bool:
        """Check if profile is stable (low variance)."""
        return self.stability_score >= Decimal("0.7")

    @property
    def has_sufficient_data(self) -> bool:
        """Check if profile has sufficient assessment data."""
        return self.assessment_count >= 3

    @property
    def dominant_traits(self) -> tuple[PersonalityTrait, ...]:
        """Get dominant personality traits."""
        if self.ocean_scores is None:
            return ()
        return self.ocean_scores.dominant_traits()

    @property
    def style_type(self) -> CommunicationStyleType:
        """Get communication style type."""
        if self.communication_style is None:
            return CommunicationStyleType.BALANCED
        return self.communication_style.style_type

    def get_recent_assessments(self, count: int = 5) -> list[TraitAssessment]:
        """Get most recent assessments."""
        return self.assessment_history[-count:] if self.assessment_history else []

    def get_trait_history(self, trait: PersonalityTrait, limit: int = 10) -> list[tuple[datetime, Decimal]]:
        """Get historical values for a specific trait."""
        history = []
        for assessment in self.assessment_history[-limit:]:
            history.append((assessment.created_at, assessment.get_trait_value(trait)))
        return history

    def calculate_trait_trend(self, trait: PersonalityTrait) -> str:
        """Calculate trend direction for a trait."""
        history = self.get_trait_history(trait, limit=5)
        if len(history) < 2:
            return "stable"
        first_half = sum(v for _, v in history[:len(history)//2]) / max(1, len(history)//2)
        second_half = sum(v for _, v in history[len(history)//2:]) / max(1, len(history) - len(history)//2)
        diff = second_half - first_half
        if diff > Decimal("0.1"):
            return "increasing"
        if diff < Decimal("-0.1"):
            return "decreasing"
        return "stable"

    def reset_scores(self) -> None:
        """Reset profile to initial state."""
        self.ocean_scores = None
        self.communication_style = None
        self.assessment_count = 0
        self.stability_score = Decimal("0.0")
        self.assessment_history.clear()
        self.version += 1
        self.touch()
        logger.info("profile_reset", profile_id=str(self.profile_id))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "profile_id": str(self.profile_id), "user_id": str(self.user_id),
            "ocean_scores": self.ocean_scores.to_dict() if self.ocean_scores else None,
            "communication_style": self.communication_style.to_dict() if self.communication_style else None,
            "assessment_count": self.assessment_count,
            "stability_score": float(self.stability_score),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "assessment_history": [a.to_dict() for a in self.assessment_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalityProfile:
        """Create from dictionary representation."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        ocean_scores = OceanScores.from_dict(data["ocean_scores"]) if data.get("ocean_scores") else None
        comm_style = CommunicationStyle.from_dict(data["communication_style"]) if data.get("communication_style") else None
        history = [TraitAssessment.from_dict(a) for a in data.get("assessment_history", [])]
        return cls(
            profile_id=UUID(data["profile_id"]) if isinstance(data["profile_id"], str) else data["profile_id"],
            user_id=UUID(data["user_id"]) if isinstance(data["user_id"], str) else data["user_id"],
            ocean_scores=ocean_scores, communication_style=comm_style,
            assessment_count=data.get("assessment_count", 0),
            stability_score=Decimal(str(data.get("stability_score", 0))),
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=updated_at or datetime.now(timezone.utc),
            version=data.get("version", 1), assessment_history=history,
        )

    @classmethod
    def create_for_user(cls, user_id: UUID) -> PersonalityProfile:
        """Factory method to create new profile for user."""
        profile = cls(user_id=user_id)
        logger.info("profile_created", profile_id=str(profile.profile_id), user_id=str(user_id))
        return profile


@dataclass
class ProfileSnapshot:
    """Immutable snapshot of a personality profile at a point in time."""
    user_id: UUID
    snapshot_id: UUID = field(default_factory=uuid4)
    profile_id: UUID = field(default_factory=uuid4)
    ocean_scores: OceanScores = field(default_factory=OceanScores.neutral)
    communication_style: CommunicationStyle | None = None
    assessment_count: int = 0
    stability_score: Decimal = Decimal("0.0")
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""

    @classmethod
    def from_profile(cls, profile: PersonalityProfile, reason: str = "") -> ProfileSnapshot:
        """Create snapshot from current profile state."""
        return cls(
            profile_id=profile.profile_id, user_id=profile.user_id,
            ocean_scores=profile.ocean_scores or OceanScores.neutral(),
            communication_style=profile.communication_style,
            assessment_count=profile.assessment_count,
            stability_score=profile.stability_score, reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "snapshot_id": str(self.snapshot_id), "profile_id": str(self.profile_id),
            "user_id": str(self.user_id), "ocean_scores": self.ocean_scores.to_dict(),
            "communication_style": self.communication_style.to_dict() if self.communication_style else None,
            "assessment_count": self.assessment_count,
            "stability_score": float(self.stability_score),
            "captured_at": self.captured_at.isoformat(), "reason": self.reason,
        }


@dataclass
class ProfileComparison:
    """Comparison result between two profile states."""
    profile_id: UUID
    baseline_snapshot: ProfileSnapshot
    current_snapshot: ProfileSnapshot
    trait_changes: dict[PersonalityTrait, Decimal] = field(default_factory=dict)
    significant_changes: list[PersonalityTrait] = field(default_factory=list)
    style_changed: bool = False

    SIGNIFICANCE_THRESHOLD: Decimal = Decimal("0.15")

    def __post_init__(self) -> None:
        """Calculate comparison metrics."""
        if not self.trait_changes:
            for trait in PersonalityTrait:
                baseline = self.baseline_snapshot.ocean_scores.get_trait(trait)
                current = self.current_snapshot.ocean_scores.get_trait(trait)
                change = current - baseline
                self.trait_changes[trait] = change
                if abs(change) >= self.SIGNIFICANCE_THRESHOLD:
                    self.significant_changes.append(trait)
        if self.baseline_snapshot.communication_style and self.current_snapshot.communication_style:
            self.style_changed = (self.baseline_snapshot.communication_style.style_type != self.current_snapshot.communication_style.style_type)

    @property
    def has_significant_changes(self) -> bool:
        """Check if any significant trait changes occurred."""
        return len(self.significant_changes) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "profile_id": str(self.profile_id),
            "baseline_snapshot": self.baseline_snapshot.to_dict(),
            "current_snapshot": self.current_snapshot.to_dict(),
            "trait_changes": {t.value: float(v) for t, v in self.trait_changes.items()},
            "significant_changes": [t.value for t in self.significant_changes],
            "style_changed": self.style_changed,
        }
