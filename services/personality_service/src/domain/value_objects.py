"""
Solace-AI Personality Service - Domain Value Objects.
Immutable value objects for Big Five personality traits and communication styles.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar
from uuid import UUID, uuid4

from ..schemas import PersonalityTrait, CommunicationStyleType, AssessmentSource


@dataclass(frozen=True)
class TraitScore:
    """
    Immutable single trait score with confidence interval.
    Represents a measured personality trait with statistical confidence bounds.
    """
    trait: PersonalityTrait
    value: Decimal
    confidence_lower: Decimal = Decimal("0.0")
    confidence_upper: Decimal = Decimal("1.0")
    sample_count: int = 1
    evidence_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate trait score data."""
        if not Decimal("0.0") <= self.value <= Decimal("1.0"):
            raise ValueError(f"Trait value must be between 0 and 1, got {self.value}")
        if self.confidence_lower > self.confidence_upper:
            raise ValueError("Confidence lower bound cannot exceed upper bound")
        if self.sample_count < 1:
            raise ValueError("Sample count must be at least 1")

    @property
    def confidence_width(self) -> Decimal:
        """Calculate width of confidence interval."""
        return self.confidence_upper - self.confidence_lower

    @property
    def is_high_confidence(self) -> bool:
        """Check if trait has high confidence (narrow interval)."""
        return self.confidence_width < Decimal("0.2")

    def with_evidence(self, markers: tuple[str, ...]) -> TraitScore:
        """Create new TraitScore with added evidence markers."""
        return TraitScore(
            trait=self.trait, value=self.value,
            confidence_lower=self.confidence_lower, confidence_upper=self.confidence_upper,
            sample_count=self.sample_count, evidence_markers=self.evidence_markers + markers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trait": self.trait.value, "value": float(self.value),
            "confidence_lower": float(self.confidence_lower),
            "confidence_upper": float(self.confidence_upper),
            "sample_count": self.sample_count,
            "evidence_markers": list(self.evidence_markers),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraitScore:
        """Create from dictionary representation."""
        return cls(
            trait=PersonalityTrait(data["trait"]) if isinstance(data["trait"], str) else data["trait"],
            value=Decimal(str(data["value"])),
            confidence_lower=Decimal(str(data.get("confidence_lower", 0.0))),
            confidence_upper=Decimal(str(data.get("confidence_upper", 1.0))),
            sample_count=data.get("sample_count", 1),
            evidence_markers=tuple(data.get("evidence_markers", [])),
        )


@dataclass(frozen=True)
class OceanScores:
    """
    Immutable OCEAN (Big Five) personality scores value object.
    Represents a complete personality trait assessment with all five dimensions.
    """
    openness: Decimal
    conscientiousness: Decimal
    extraversion: Decimal
    agreeableness: Decimal
    neuroticism: Decimal
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    overall_confidence: Decimal = Decimal("0.5")
    trait_scores: tuple[TraitScore, ...] = ()

    HIGH_THRESHOLD: ClassVar[Decimal] = Decimal("0.7")
    LOW_THRESHOLD: ClassVar[Decimal] = Decimal("0.3")

    def __post_init__(self) -> None:
        """Validate OCEAN scores."""
        for name in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            value = getattr(self, name)
            if not Decimal("0.0") <= value <= Decimal("1.0"):
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if not Decimal("0.0") <= self.overall_confidence <= Decimal("1.0"):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.overall_confidence}")

    def get_trait(self, trait: PersonalityTrait) -> Decimal:
        """Get score for a specific trait."""
        return getattr(self, trait.value)

    def dominant_traits(self, threshold: Decimal | None = None) -> tuple[PersonalityTrait, ...]:
        """Get traits above threshold."""
        threshold = threshold or self.HIGH_THRESHOLD
        return tuple(t for t in PersonalityTrait if self.get_trait(t) >= threshold)

    def low_traits(self, threshold: Decimal | None = None) -> tuple[PersonalityTrait, ...]:
        """Get traits below threshold."""
        threshold = threshold or self.LOW_THRESHOLD
        return tuple(t for t in PersonalityTrait if self.get_trait(t) <= threshold)

    @property
    def trait_vector(self) -> tuple[Decimal, ...]:
        """Get trait values as ordered vector (O, C, E, A, N)."""
        return (self.openness, self.conscientiousness, self.extraversion, self.agreeableness, self.neuroticism)

    def distance_to(self, other: OceanScores) -> Decimal:
        """Calculate Euclidean distance to another OCEAN profile."""
        squared_sum = sum((a - b) ** 2 for a, b in zip(self.trait_vector, other.trait_vector, strict=True))
        return Decimal(str(float(squared_sum) ** 0.5))

    def aggregate_with(self, other: OceanScores, alpha: Decimal = Decimal("0.3")) -> OceanScores:
        """Create new scores aggregated with another using exponential moving average."""
        one_minus_alpha = Decimal("1.0") - alpha
        return OceanScores(
            openness=self.openness * one_minus_alpha + other.openness * alpha,
            conscientiousness=self.conscientiousness * one_minus_alpha + other.conscientiousness * alpha,
            extraversion=self.extraversion * one_minus_alpha + other.extraversion * alpha,
            agreeableness=self.agreeableness * one_minus_alpha + other.agreeableness * alpha,
            neuroticism=self.neuroticism * one_minus_alpha + other.neuroticism * alpha,
            overall_confidence=min(Decimal("0.9"), self.overall_confidence * Decimal("0.8") + other.overall_confidence * Decimal("0.3")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "openness": float(self.openness), "conscientiousness": float(self.conscientiousness),
            "extraversion": float(self.extraversion), "agreeableness": float(self.agreeableness),
            "neuroticism": float(self.neuroticism), "assessed_at": self.assessed_at.isoformat(),
            "overall_confidence": float(self.overall_confidence),
            "trait_scores": [ts.to_dict() for ts in self.trait_scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OceanScores:
        """Create from dictionary representation."""
        assessed_at = data.get("assessed_at")
        if isinstance(assessed_at, str):
            assessed_at = datetime.fromisoformat(assessed_at)
        trait_scores = tuple(TraitScore.from_dict(ts) for ts in data.get("trait_scores", []))
        return cls(
            openness=Decimal(str(data["openness"])), conscientiousness=Decimal(str(data["conscientiousness"])),
            extraversion=Decimal(str(data["extraversion"])), agreeableness=Decimal(str(data["agreeableness"])),
            neuroticism=Decimal(str(data["neuroticism"])),
            assessed_at=assessed_at or datetime.now(timezone.utc),
            overall_confidence=Decimal(str(data.get("overall_confidence", 0.5))),
            trait_scores=trait_scores,
        )

    @classmethod
    def neutral(cls) -> OceanScores:
        """Create neutral OCEAN scores (all at 0.5)."""
        return cls(
            openness=Decimal("0.5"), conscientiousness=Decimal("0.5"), extraversion=Decimal("0.5"),
            agreeableness=Decimal("0.5"), neuroticism=Decimal("0.5"), overall_confidence=Decimal("0.3"),
        )


@dataclass(frozen=True)
class CommunicationStyle:
    """
    Immutable communication style parameters value object.
    Defines how to adapt responses based on personality traits.
    """
    style_id: UUID
    warmth: Decimal = Decimal("0.5")
    structure: Decimal = Decimal("0.5")
    complexity: Decimal = Decimal("0.5")
    directness: Decimal = Decimal("0.5")
    energy: Decimal = Decimal("0.5")
    validation_level: Decimal = Decimal("0.5")
    style_type: CommunicationStyleType = CommunicationStyleType.BALANCED
    custom_params: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate communication style parameters."""
        for name in ("warmth", "structure", "complexity", "directness", "energy", "validation_level"):
            value = getattr(self, name)
            if not Decimal("0.0") <= value <= Decimal("1.0"):
                raise ValueError(f"{name} must be between 0 and 1, got {value}")

    @property
    def is_warm(self) -> bool:
        """Check if style emphasizes warmth."""
        return self.warmth >= Decimal("0.7")

    @property
    def is_structured(self) -> bool:
        """Check if style emphasizes structure."""
        return self.structure >= Decimal("0.7")

    @property
    def needs_validation(self) -> bool:
        """Check if style requires high validation."""
        return self.validation_level >= Decimal("0.7")

    def get_custom_param(self, key: str, default: str = "") -> str:
        """Get custom parameter value."""
        for k, v in self.custom_params:
            if k == key:
                return v
        return default

    @classmethod
    def from_ocean(cls, scores: OceanScores, style_id: UUID | None = None) -> CommunicationStyle:
        """Derive communication style from OCEAN scores."""
        warmth = Decimal("0.6") + (scores.agreeableness - Decimal("0.5")) * Decimal("0.4")
        if scores.neuroticism > Decimal("0.7"):
            warmth += Decimal("0.2")
        structure = Decimal("0.5") + (scores.conscientiousness - Decimal("0.5")) * Decimal("0.5")
        if scores.conscientiousness > Decimal("0.7"):
            structure += Decimal("0.25")
        complexity = Decimal("0.5") + (scores.openness - Decimal("0.5")) * Decimal("0.4")
        if scores.openness > Decimal("0.7"):
            complexity += Decimal("0.3")
        directness = Decimal("0.5") + (Decimal("0.5") - scores.agreeableness) * Decimal("0.2")
        if scores.neuroticism > Decimal("0.7"):
            directness -= Decimal("0.15")
        energy = Decimal("0.4") + (scores.extraversion - Decimal("0.5")) * Decimal("0.8")
        validation = Decimal("0.5")
        if scores.neuroticism > Decimal("0.7"):
            validation += Decimal("0.25")
        validation += (scores.agreeableness - Decimal("0.5")) * Decimal("0.3")
        clamp = lambda v: max(Decimal("0.0"), min(Decimal("1.0"), v))
        style_type = cls._determine_style_type(scores)
        return cls(
            style_id=style_id or uuid4(),
            warmth=clamp(warmth), structure=clamp(structure), complexity=clamp(complexity),
            directness=clamp(directness), energy=clamp(energy), validation_level=clamp(validation),
            style_type=style_type,
        )

    @staticmethod
    def _determine_style_type(scores: OceanScores) -> CommunicationStyleType:
        """Determine primary communication style type from OCEAN scores."""
        high = Decimal("0.7")
        low = Decimal("0.3")
        high_e = scores.extraversion > high
        high_a = scores.agreeableness > high
        high_c = scores.conscientiousness > high
        low_e = scores.extraversion < low
        if high_c and not high_e:
            return CommunicationStyleType.ANALYTICAL
        if high_e and not high_a:
            return CommunicationStyleType.DRIVER
        if high_e and high_a:
            return CommunicationStyleType.EXPRESSIVE
        if high_a and low_e:
            return CommunicationStyleType.AMIABLE
        return CommunicationStyleType.BALANCED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "style_id": str(self.style_id),
            "warmth": float(self.warmth), "structure": float(self.structure),
            "complexity": float(self.complexity), "directness": float(self.directness),
            "energy": float(self.energy), "validation_level": float(self.validation_level),
            "style_type": self.style_type.value, "custom_params": dict(self.custom_params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunicationStyle:
        """Create from dictionary representation."""
        custom_params = data.get("custom_params", {})
        if isinstance(custom_params, dict):
            custom_params = tuple(custom_params.items())
        return cls(
            style_id=UUID(data["style_id"]) if isinstance(data.get("style_id"), str) else data.get("style_id", uuid4()),
            warmth=Decimal(str(data.get("warmth", 0.5))),
            structure=Decimal(str(data.get("structure", 0.5))),
            complexity=Decimal(str(data.get("complexity", 0.5))),
            directness=Decimal(str(data.get("directness", 0.5))),
            energy=Decimal(str(data.get("energy", 0.5))),
            validation_level=Decimal(str(data.get("validation_level", 0.5))),
            style_type=CommunicationStyleType(data["style_type"]) if isinstance(data.get("style_type"), str) else data.get("style_type", CommunicationStyleType.BALANCED),
            custom_params=custom_params,
        )


@dataclass(frozen=True)
class AssessmentMetadata:
    """
    Immutable metadata for a personality assessment.
    Contains source, quality, and contextual information.
    """
    assessment_id: UUID
    source: AssessmentSource
    text_length: int = 0
    model_version: str = "1.0.0"
    processing_time_ms: float = 0.0
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate assessment metadata."""
        if self.text_length < 0:
            raise ValueError("Text length cannot be negative")
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")

    @property
    def is_text_sufficient(self) -> bool:
        """Check if text length meets minimum threshold."""
        return self.text_length >= 50

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "assessment_id": str(self.assessment_id), "source": self.source.value,
            "text_length": self.text_length, "model_version": self.model_version,
            "processing_time_ms": self.processing_time_ms,
            "assessed_at": self.assessed_at.isoformat(),
            "context_tags": list(self.context_tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssessmentMetadata:
        """Create from dictionary representation."""
        assessed_at = data.get("assessed_at")
        if isinstance(assessed_at, str):
            assessed_at = datetime.fromisoformat(assessed_at)
        return cls(
            assessment_id=UUID(data["assessment_id"]) if isinstance(data["assessment_id"], str) else data["assessment_id"],
            source=AssessmentSource(data["source"]) if isinstance(data["source"], str) else data["source"],
            text_length=data.get("text_length", 0),
            model_version=data.get("model_version", "1.0.0"),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            assessed_at=assessed_at or datetime.now(timezone.utc),
            context_tags=tuple(data.get("context_tags", [])),
        )
