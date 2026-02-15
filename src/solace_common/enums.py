"""
Solace-AI Canonical Enums.

Shared enum definitions for cross-service consistency.
Import these instead of defining local duplicates.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum


class CrisisLevel(str, Enum):
    """Crisis/risk severity levels with clear escalation boundaries.

    Canonical enum used across all services for safety risk classification.
    Values (ascending severity): NONE < LOW < ELEVATED < HIGH < CRITICAL.
    """

    NONE = "NONE"
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, value: str) -> CrisisLevel:
        """Case-insensitive lookup with common alias mapping.

        Maps common alternative names to canonical CrisisLevel values:
        - "moderate", "medium" -> ELEVATED
        - "severe", "extreme", "imminent" -> CRITICAL
        - "minimal" -> NONE
        """
        _ALIASES: dict[str, CrisisLevel] = {
            "none": cls.NONE,
            "minimal": cls.NONE,
            "low": cls.LOW,
            "moderate": cls.ELEVATED,
            "medium": cls.ELEVATED,
            "elevated": cls.ELEVATED,
            "high": cls.HIGH,
            "severe": cls.CRITICAL,
            "extreme": cls.CRITICAL,
            "imminent": cls.CRITICAL,
            "critical": cls.CRITICAL,
        }
        normalized = value.strip().lower()
        if normalized in _ALIASES:
            return _ALIASES[normalized]
        raise ValueError(
            f"Unknown crisis level: '{value}'. "
            f"Valid values: {', '.join(_ALIASES.keys())}"
        )

    def to_severity_level(self) -> SeverityLevel:
        """Map CrisisLevel to SeverityLevel for cross-domain compatibility.

        NONE -> MINIMAL, LOW -> MILD, ELEVATED -> MODERATE,
        HIGH -> MODERATELY_SEVERE, CRITICAL -> SEVERE.
        """
        _MAP: dict[CrisisLevel, SeverityLevel] = {
            CrisisLevel.NONE: SeverityLevel.MINIMAL,
            CrisisLevel.LOW: SeverityLevel.MILD,
            CrisisLevel.ELEVATED: SeverityLevel.MODERATE,
            CrisisLevel.HIGH: SeverityLevel.MODERATELY_SEVERE,
            CrisisLevel.CRITICAL: SeverityLevel.SEVERE,
        }
        return _MAP[self]

    @classmethod
    def from_score(cls, score: Decimal) -> CrisisLevel:
        """Determine crisis level from a 0.0-1.0 risk score using standard thresholds.

        Thresholds: >=0.9 CRITICAL, >=0.7 HIGH, >=0.5 ELEVATED, >=0.3 LOW, else NONE.
        """
        if score >= Decimal("0.9"):
            return cls.CRITICAL
        if score >= Decimal("0.7"):
            return cls.HIGH
        if score >= Decimal("0.5"):
            return cls.ELEVATED
        if score >= Decimal("0.3"):
            return cls.LOW
        return cls.NONE


# Backward-compatible alias for code that uses the RiskLevel name.
RiskLevel = CrisisLevel


class SeverityLevel(str, Enum):
    """Clinical severity levels for symptoms and conditions.

    Canonical enum used across all services for clinical severity.
    Values (ascending severity): MINIMAL < MILD < MODERATE < MODERATELY_SEVERE < SEVERE.
    """

    MINIMAL = "MINIMAL"
    MILD = "MILD"
    MODERATE = "MODERATE"
    MODERATELY_SEVERE = "MODERATELY_SEVERE"
    SEVERE = "SEVERE"

    @classmethod
    def from_string(cls, value: str) -> SeverityLevel:
        """Case-insensitive lookup with alias mapping."""
        _ALIASES: dict[str, SeverityLevel] = {
            "minimal": cls.MINIMAL,
            "none": cls.MINIMAL,
            "mild": cls.MILD,
            "low": cls.MILD,
            "moderate": cls.MODERATE,
            "medium": cls.MODERATE,
            "moderately_severe": cls.MODERATELY_SEVERE,
            "moderately severe": cls.MODERATELY_SEVERE,
            "high": cls.MODERATELY_SEVERE,
            "severe": cls.SEVERE,
            "critical": cls.SEVERE,
            "extreme": cls.SEVERE,
        }
        normalized = value.strip().lower()
        if normalized in _ALIASES:
            return _ALIASES[normalized]
        raise ValueError(
            f"Unknown severity level: '{value}'. "
            f"Valid values: {', '.join(_ALIASES.keys())}"
        )
