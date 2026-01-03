"""
Solace-AI Value Object Implementation.

Provides immutable, validation-rich value objects for domain modeling.
Value objects are compared by their attribute values, not identity.
They are immutable after creation and self-validating.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

import structlog

logger = structlog.get_logger(__name__)


class ValueObject(BaseModel, ABC):
    """
    Base class for all value objects.

    Value objects are immutable domain primitives that:
    - Are defined by their attributes, not identity
    - Are immutable after creation
    - Self-validate on construction
    - Can be freely substituted if equal

    All subclasses should define their attributes as immutable fields.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
        str_strip_whitespace=True,
    )

    def __eq__(self, other: object) -> bool:
        """Value objects are equal if all their attributes are equal."""
        if not isinstance(other, type(self)):
            return False
        return self.model_dump() == other.model_dump()

    def __hash__(self) -> int:
        """Hash based on all attributes for use in sets and dicts."""
        return hash(tuple(sorted(self.model_dump().items(), key=lambda x: x[0])))


T = TypeVar("T")


class SingleValueObject(ValueObject, Generic[T]):
    """
    Value object wrapping a single primitive value.

    Use for domain primitives that need validation or semantic meaning.
    Example: EmailAddress, PhoneNumber, Money, etc.
    """

    value: T

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"


class EmailAddress(SingleValueObject[str]):
    """
    Validated email address value object.

    Ensures email format compliance and normalizes to lowercase.
    """

    _EMAIL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )

    @field_validator("value", mode="before")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Normalize email to lowercase and validate format."""
        if not isinstance(v, str):
            raise ValueError("Email must be a string")

        normalized = v.strip().lower()

        if not cls._EMAIL_PATTERN.match(normalized):
            raise ValueError(f"Invalid email format: {v}")

        return normalized

    @property
    def domain(self) -> str:
        """Extract email domain."""
        return self.value.split("@")[1]

    @property
    def local_part(self) -> str:
        """Extract local part of email."""
        return self.value.split("@")[0]


class PhoneNumber(SingleValueObject[str]):
    """
    Validated phone number value object.

    Normalizes to E.164 format for storage.
    """

    country_code: str = Field(default="1", description="Country code without +")

    @field_validator("value", mode="before")
    @classmethod
    def normalize_phone(cls, v: str) -> str:
        """Normalize phone number to digits only."""
        if not isinstance(v, str):
            raise ValueError("Phone number must be a string")

        digits = re.sub(r"\D", "", v)

        if len(digits) < 10:
            raise ValueError("Phone number must have at least 10 digits")
        if len(digits) > 15:
            raise ValueError("Phone number too long")

        return digits

    def to_e164(self) -> str:
        """Format as E.164 international format."""
        return f"+{self.country_code}{self.value}"

    def to_national(self) -> str:
        """Format for national display (US format)."""
        if len(self.value) == 10:
            return f"({self.value[:3]}) {self.value[3:6]}-{self.value[6:]}"
        return self.value


class Percentage(SingleValueObject[Decimal]):
    """
    Percentage value object (0-100 scale).

    Validates range and provides conversion utilities.
    """

    @field_validator("value", mode="before")
    @classmethod
    def validate_percentage(cls, v: Any) -> Decimal:
        """Validate percentage is within valid range."""
        decimal_value = Decimal(str(v))

        if decimal_value < 0 or decimal_value > 100:
            raise ValueError("Percentage must be between 0 and 100")

        return decimal_value

    def to_fraction(self) -> Decimal:
        """Convert to decimal fraction (0-1 scale)."""
        return self.value / Decimal("100")

    @classmethod
    def from_fraction(cls, fraction: Decimal | float) -> Percentage:
        """Create from decimal fraction (0-1 scale)."""
        return cls(value=Decimal(str(fraction)) * Decimal("100"))


class Score(ValueObject):
    """
    Normalized score value object (0.0-1.0 scale).

    Used for confidence scores, similarity scores, etc.
    """

    value: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    confidence: Decimal | None = Field(
        default=None, ge=Decimal("0"), le=Decimal("1")
    )

    @field_validator("value", "confidence", mode="before")
    @classmethod
    def to_decimal(cls, v: Any) -> Decimal | None:
        """Convert numeric types to Decimal."""
        if v is None:
            return None
        return Decimal(str(v))

    def to_percentage(self) -> Percentage:
        """Convert to percentage."""
        return Percentage(value=self.value * Decimal("100"))

    @classmethod
    def from_percentage(cls, pct: Percentage) -> Score:
        """Create from percentage."""
        return cls(value=pct.to_fraction())


class DateRange(ValueObject):
    """
    Date range value object with validation.

    Ensures start is before or equal to end.
    """

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_range(self) -> DateRange:
        """Ensure start is not after end."""
        if self.start > self.end:
            raise ValueError("Start date must be before or equal to end date")
        return self

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return (self.end - self.start).total_seconds()

    @property
    def duration_days(self) -> int:
        """Get duration in days."""
        return (self.end - self.start).days

    def contains(self, dt: datetime) -> bool:
        """Check if datetime falls within range."""
        return self.start <= dt <= self.end

    def overlaps(self, other: DateRange) -> bool:
        """Check if this range overlaps with another."""
        return self.start <= other.end and other.start <= self.end


class Severity(str, Enum):
    """Severity level enumeration for mental health context."""

    NONE = "none"
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"
    CRITICAL = "critical"


class SeverityScore(ValueObject):
    """
    Severity assessment value object.

    Combines categorical severity with numerical score.
    """

    level: Severity
    score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    assessment_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def from_score(cls, score: float | Decimal) -> SeverityScore:
        """Create severity from numerical score using standard thresholds."""
        score_decimal = Decimal(str(score))

        if score_decimal <= Decimal("0.1"):
            level = Severity.NONE
        elif score_decimal <= Decimal("0.25"):
            level = Severity.MINIMAL
        elif score_decimal <= Decimal("0.4"):
            level = Severity.MILD
        elif score_decimal <= Decimal("0.6"):
            level = Severity.MODERATE
        elif score_decimal <= Decimal("0.75"):
            level = Severity.MODERATELY_SEVERE
        elif score_decimal <= Decimal("0.9"):
            level = Severity.SEVERE
        else:
            level = Severity.CRITICAL

        return cls(level=level, score=score_decimal)


class HashedValue(SingleValueObject[str]):
    """
    Secure hashed value object.

    Used for storing sensitive data that only needs comparison.
    Value is stored as SHA-256 hash.
    """

    _HASH_ALGORITHM: ClassVar[str] = "sha256"

    @classmethod
    def from_plain_text(cls, plain_text: str, salt: str = "") -> HashedValue:
        """Create hashed value from plain text."""
        combined = f"{salt}{plain_text}"
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        return cls(value=hash_value)

    def matches(self, plain_text: str, salt: str = "") -> bool:
        """Check if plain text matches stored hash."""
        combined = f"{salt}{plain_text}"
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        return self.value == hash_value


class UserId(SingleValueObject[str]):
    """User identifier value object with format validation."""

    @field_validator("value", mode="before")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """Validate user ID format."""
        if not isinstance(v, str):
            raise ValueError("User ID must be a string")

        cleaned = v.strip()
        if not cleaned:
            raise ValueError("User ID cannot be empty")
        if len(cleaned) > 64:
            raise ValueError("User ID too long (max 64 characters)")

        return cleaned


class SessionId(SingleValueObject[str]):
    """Session identifier value object."""

    @field_validator("value", mode="before")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format."""
        if not isinstance(v, str):
            raise ValueError("Session ID must be a string")

        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Session ID cannot be empty")
        if len(cleaned) > 128:
            raise ValueError("Session ID too long (max 128 characters)")

        return cleaned


class CorrelationId(SingleValueObject[str]):
    """Correlation ID for distributed tracing."""

    @classmethod
    def generate(cls) -> CorrelationId:
        """Generate new correlation ID."""
        import uuid

        return cls(value=str(uuid.uuid4()))
