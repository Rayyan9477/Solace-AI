"""
Solace-AI Safety Service - Domain Value Objects.
Immutable value objects for safety domain with validation and comparison support.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from functools import total_ordering
from typing import Any, Self
from pydantic import BaseModel, Field, field_validator


class RiskSeverity(str, Enum):
    """Risk severity classification levels."""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    EXTREME = "EXTREME"


class TriggerCategory(str, Enum):
    """Categories of crisis triggers."""
    KEYWORD = "KEYWORD"
    PATTERN = "PATTERN"
    SENTIMENT = "SENTIMENT"
    BEHAVIORAL = "BEHAVIORAL"
    CONTEXTUAL = "CONTEXTUAL"
    HISTORICAL = "HISTORICAL"
    ESCALATION = "ESCALATION"


class ProtectiveFactorType(str, Enum):
    """Types of protective factors."""
    SOCIAL_SUPPORT = "SOCIAL_SUPPORT"
    FAMILY_CONNECTION = "FAMILY_CONNECTION"
    TREATMENT_ENGAGEMENT = "TREATMENT_ENGAGEMENT"
    COPING_SKILLS = "COPING_SKILLS"
    POSITIVE_OUTLOOK = "POSITIVE_OUTLOOK"
    PROFESSIONAL_HELP = "PROFESSIONAL_HELP"
    MEDICATION_ADHERENCE = "MEDICATION_ADHERENCE"
    PURPOSE = "PURPOSE"


class DetectionLayer(int, Enum):
    """Detection layer identifiers."""
    INPUT_GATE = 1
    PROCESSING_GUARD = 2
    OUTPUT_FILTER = 3
    CONTINUOUS_MONITOR = 4


@total_ordering
class RiskScore(BaseModel):
    """Immutable risk score value object with comparison support."""
    value: Decimal = Field(..., ge=0, le=1)
    confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}

    @field_validator("value", "confidence", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RiskScore):
            return self.value == other.value
        if isinstance(other, (int, float, Decimal)):
            return self.value == Decimal(str(other))
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, RiskScore):
            return self.value < other.value
        if isinstance(other, (int, float, Decimal)):
            return self.value < Decimal(str(other))
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.value, self.confidence))

    @property
    def severity(self) -> RiskSeverity:
        """Determine severity from score value."""
        if self.value >= Decimal("0.9"):
            return RiskSeverity.EXTREME
        if self.value >= Decimal("0.75"):
            return RiskSeverity.SEVERE
        if self.value >= Decimal("0.6"):
            return RiskSeverity.HIGH
        if self.value >= Decimal("0.4"):
            return RiskSeverity.MODERATE
        if self.value >= Decimal("0.2"):
            return RiskSeverity.LOW
        return RiskSeverity.MINIMAL

    @property
    def is_critical(self) -> bool:
        """Check if score indicates critical risk."""
        return self.value >= Decimal("0.9")

    @property
    def is_elevated(self) -> bool:
        """Check if score indicates elevated risk."""
        return self.value >= Decimal("0.5")

    @classmethod
    def zero(cls) -> Self:
        """Create zero risk score."""
        return cls(value=Decimal("0"), confidence=Decimal("1.0"))

    @classmethod
    def maximum(cls) -> Self:
        """Create maximum risk score."""
        return cls(value=Decimal("1.0"), confidence=Decimal("1.0"))

    def weighted_average(self, other: RiskScore, weight: Decimal = Decimal("0.5")) -> RiskScore:
        """Compute weighted average with another score."""
        w = min(max(weight, Decimal("0")), Decimal("1"))
        new_value = self.value * (1 - w) + other.value * w
        new_confidence = (self.confidence + other.confidence) / 2
        return RiskScore(value=new_value, confidence=new_confidence)


class TriggerIndicator(BaseModel):
    """Immutable trigger indicator value object."""
    category: TriggerCategory
    indicator: str = Field(..., min_length=1, max_length=200)
    severity: RiskSeverity = Field(default=RiskSeverity.MODERATE)
    confidence: Decimal = Field(default=Decimal("0.8"), ge=0, le=1)
    layer: DetectionLayer = Field(default=DetectionLayer.INPUT_GATE)
    evidence: str | None = Field(default=None)

    model_config = {"frozen": True}

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TriggerIndicator):
            return self.category == other.category and self.indicator == other.indicator
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.category, self.indicator))

    @classmethod
    def keyword(cls, keyword: str, severity: RiskSeverity = RiskSeverity.HIGH) -> Self:
        """Create keyword trigger indicator."""
        return cls(
            category=TriggerCategory.KEYWORD,
            indicator=f"KEYWORD:{keyword}",
            severity=severity,
            evidence=f"Detected keyword: '{keyword}'",
        )

    @classmethod
    def pattern(cls, pattern_name: str, severity: RiskSeverity = RiskSeverity.HIGH) -> Self:
        """Create pattern trigger indicator."""
        return cls(
            category=TriggerCategory.PATTERN,
            indicator=f"PATTERN:{pattern_name}",
            severity=severity,
            evidence=f"Detected pattern: '{pattern_name}'",
        )


class ProtectiveFactor(BaseModel):
    """Immutable protective factor value object."""
    factor_type: ProtectiveFactorType
    strength: Decimal = Field(..., ge=0, le=1)
    description: str = Field(..., min_length=1, max_length=500)
    source: str = Field(default="assessment")
    verified: bool = Field(default=False)

    model_config = {"frozen": True}

    @field_validator("strength", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProtectiveFactor):
            return self.factor_type == other.factor_type and self.description == other.description
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.factor_type, self.description))

    @property
    def is_strong(self) -> bool:
        """Check if factor provides strong protection."""
        return self.strength >= Decimal("0.7")

    @classmethod
    def social_support(cls, description: str, strength: Decimal = Decimal("0.6")) -> Self:
        """Create social support protective factor."""
        return cls(
            factor_type=ProtectiveFactorType.SOCIAL_SUPPORT,
            strength=strength,
            description=description,
        )

    @classmethod
    def treatment_engagement(cls, description: str, strength: Decimal = Decimal("0.7")) -> Self:
        """Create treatment engagement protective factor."""
        return cls(
            factor_type=ProtectiveFactorType.TREATMENT_ENGAGEMENT,
            strength=strength,
            description=description,
        )


class ContraindicationRule(BaseModel):
    """Immutable contraindication rule for technique safety."""
    technique: str = Field(..., min_length=1, max_length=100)
    crisis_levels: list[str] = Field(..., min_length=1)
    reason: str = Field(..., min_length=1, max_length=500)
    severity: RiskSeverity = Field(default=RiskSeverity.HIGH)
    alternative_techniques: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}

    def is_contraindicated(self, crisis_level: str) -> bool:
        """Check if technique is contraindicated for crisis level."""
        return crisis_level.upper() in [level.upper() for level in self.crisis_levels]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ContraindicationRule):
            return self.technique == other.technique
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.technique)


class SafetyThresholds(BaseModel):
    """Immutable safety thresholds configuration."""
    none_threshold: Decimal = Field(default=Decimal("0.0"), ge=0, le=1)
    low_threshold: Decimal = Field(default=Decimal("0.3"), ge=0, le=1)
    elevated_threshold: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    high_threshold: Decimal = Field(default=Decimal("0.7"), ge=0, le=1)
    critical_threshold: Decimal = Field(default=Decimal("0.9"), ge=0, le=1)

    model_config = {"frozen": True}

    @field_validator("none_threshold", "low_threshold", "elevated_threshold",
                     "high_threshold", "critical_threshold", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    def get_level_for_score(self, score: Decimal | RiskScore) -> str:
        """Get crisis level for a risk score."""
        value = score.value if isinstance(score, RiskScore) else score
        if value >= self.critical_threshold:
            return "CRITICAL"
        if value >= self.high_threshold:
            return "HIGH"
        if value >= self.elevated_threshold:
            return "ELEVATED"
        if value >= self.low_threshold:
            return "LOW"
        return "NONE"


class ResponseModification(BaseModel):
    """Immutable record of response modification."""
    modification_type: str = Field(..., min_length=1, max_length=50)
    original_content: str | None = Field(default=None)
    modified_content: str | None = Field(default=None)
    reason: str = Field(..., min_length=1, max_length=500)
    applied_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    layer: DetectionLayer = Field(default=DetectionLayer.OUTPUT_FILTER)

    model_config = {"frozen": True}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ResponseModification):
            return (self.modification_type == other.modification_type
                    and self.reason == other.reason)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.modification_type, self.reason))


class TrajectoryAnalysis(BaseModel):
    """Immutable trajectory analysis result."""
    trend: str = Field(default="stable")
    deteriorating: bool = Field(default=False)
    risk_delta: Decimal = Field(default=Decimal("0"))
    negative_ratio: Decimal = Field(default=Decimal("0"))
    messages_analyzed: int = Field(default=0, ge=0)
    confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    prediction: str = Field(default="stable")
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}

    @field_validator("risk_delta", "negative_ratio", "confidence", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    @property
    def requires_attention(self) -> bool:
        """Check if trajectory requires clinical attention."""
        return self.deteriorating or self.negative_ratio > Decimal("0.6")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrajectoryAnalysis):
            return self.trend == other.trend and self.deteriorating == other.deteriorating
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.trend, self.deteriorating))
