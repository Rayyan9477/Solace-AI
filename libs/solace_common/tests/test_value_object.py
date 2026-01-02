"""
Unit tests for Solace-AI Value Object Module.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from solace_common.src.domain.value_object import (
    CorrelationId,
    DateRange,
    EmailAddress,
    HashedValue,
    Percentage,
    PhoneNumber,
    Score,
    SessionId,
    Severity,
    SeverityScore,
    SingleValueObject,
    UserId,
    ValueObject,
)


class TestValueObject:
    """Tests for ValueObject base class."""

    def test_equality_by_attributes(self) -> None:
        """Test value objects are equal by attributes."""

        class SimpleVO(ValueObject):
            x: int
            y: int

        vo1 = SimpleVO(x=1, y=2)
        vo2 = SimpleVO(x=1, y=2)
        vo3 = SimpleVO(x=1, y=3)

        assert vo1 == vo2
        assert vo1 != vo3

    def test_hashable(self) -> None:
        """Test value objects are hashable."""

        class SimpleVO(ValueObject):
            x: int
            y: int

        vo1 = SimpleVO(x=1, y=2)
        vo2 = SimpleVO(x=1, y=2)

        vo_set = {vo1, vo2}
        assert len(vo_set) == 1

    def test_immutability(self) -> None:
        """Test value objects are immutable."""

        class SimpleVO(ValueObject):
            value: str

        vo = SimpleVO(value="original")
        with pytest.raises(Exception):
            vo.value = "modified"  # type: ignore[misc]


class TestEmailAddress:
    """Tests for EmailAddress value object."""

    def test_valid_email(self) -> None:
        """Test valid email creation."""
        email = EmailAddress(value="test@example.com")

        assert email.value == "test@example.com"
        assert email.domain == "example.com"
        assert email.local_part == "test"

    def test_normalization(self) -> None:
        """Test email is normalized to lowercase."""
        email = EmailAddress(value="  TEST@EXAMPLE.COM  ")

        assert email.value == "test@example.com"

    def test_invalid_email_format(self) -> None:
        """Test invalid email is rejected."""
        with pytest.raises(ValueError):
            EmailAddress(value="not-an-email")

        with pytest.raises(ValueError):
            EmailAddress(value="missing@domain")

        with pytest.raises(ValueError):
            EmailAddress(value="@nodomain.com")

    def test_string_representation(self) -> None:
        """Test string conversion."""
        email = EmailAddress(value="test@example.com")
        assert str(email) == "test@example.com"


class TestPhoneNumber:
    """Tests for PhoneNumber value object."""

    def test_valid_phone(self) -> None:
        """Test valid phone number creation."""
        phone = PhoneNumber(value="1234567890")

        assert phone.value == "1234567890"

    def test_normalization(self) -> None:
        """Test phone is normalized to digits only."""
        phone = PhoneNumber(value="(123) 456-7890")

        assert phone.value == "1234567890"

    def test_e164_format(self) -> None:
        """Test E.164 format output."""
        phone = PhoneNumber(value="1234567890", country_code="1")

        assert phone.to_e164() == "+11234567890"

    def test_national_format(self) -> None:
        """Test national format output."""
        phone = PhoneNumber(value="1234567890")

        assert phone.to_national() == "(123) 456-7890"

    def test_too_short(self) -> None:
        """Test phone number too short is rejected."""
        with pytest.raises(ValueError):
            PhoneNumber(value="12345")

    def test_too_long(self) -> None:
        """Test phone number too long is rejected."""
        with pytest.raises(ValueError):
            PhoneNumber(value="1234567890123456789")


class TestPercentage:
    """Tests for Percentage value object."""

    def test_valid_percentage(self) -> None:
        """Test valid percentage creation."""
        pct = Percentage(value=Decimal("75.5"))

        assert pct.value == Decimal("75.5")

    def test_to_fraction(self) -> None:
        """Test conversion to fraction."""
        pct = Percentage(value=Decimal("50"))

        assert pct.to_fraction() == Decimal("0.5")

    def test_from_fraction(self) -> None:
        """Test creation from fraction."""
        pct = Percentage.from_fraction(0.25)

        assert pct.value == Decimal("25")

    def test_out_of_range(self) -> None:
        """Test percentage out of range is rejected."""
        with pytest.raises(ValueError):
            Percentage(value=Decimal("-10"))

        with pytest.raises(ValueError):
            Percentage(value=Decimal("150"))


class TestScore:
    """Tests for Score value object."""

    def test_valid_score(self) -> None:
        """Test valid score creation."""
        score = Score(value=Decimal("0.85"), confidence=Decimal("0.9"))

        assert score.value == Decimal("0.85")
        assert score.confidence == Decimal("0.9")

    def test_to_percentage(self) -> None:
        """Test conversion to percentage."""
        score = Score(value=Decimal("0.75"))

        pct = score.to_percentage()
        assert pct.value == Decimal("75")

    def test_out_of_range(self) -> None:
        """Test score out of range is rejected."""
        with pytest.raises(ValueError):
            Score(value=Decimal("1.5"))

        with pytest.raises(ValueError):
            Score(value=Decimal("-0.1"))


class TestDateRange:
    """Tests for DateRange value object."""

    def test_valid_range(self) -> None:
        """Test valid date range creation."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        date_range = DateRange(start=start, end=end)

        assert date_range.start == start
        assert date_range.end == end

    def test_duration_days(self) -> None:
        """Test duration calculation in days."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 11, tzinfo=timezone.utc)

        date_range = DateRange(start=start, end=end)

        assert date_range.duration_days == 10

    def test_contains(self) -> None:
        """Test date containment check."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        date_range = DateRange(start=start, end=end)

        inside = datetime(2024, 6, 15, tzinfo=timezone.utc)
        outside = datetime(2025, 1, 1, tzinfo=timezone.utc)

        assert date_range.contains(inside) is True
        assert date_range.contains(outside) is False

    def test_overlaps(self) -> None:
        """Test overlap detection."""
        range1 = DateRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        )
        range2 = DateRange(
            start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        range3 = DateRange(
            start=datetime(2024, 7, 1, tzinfo=timezone.utc),
            end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        assert range1.overlaps(range2) is True
        assert range1.overlaps(range3) is False

    def test_invalid_range(self) -> None:
        """Test invalid range (start after end) is rejected."""
        with pytest.raises(ValueError):
            DateRange(
                start=datetime(2024, 12, 31, tzinfo=timezone.utc),
                end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )


class TestSeverityScore:
    """Tests for SeverityScore value object."""

    def test_from_score_none(self) -> None:
        """Test severity level for zero score."""
        severity = SeverityScore.from_score(0.05)
        assert severity.level == Severity.NONE

    def test_from_score_moderate(self) -> None:
        """Test severity level for moderate score."""
        severity = SeverityScore.from_score(0.5)
        assert severity.level == Severity.MODERATE

    def test_from_score_critical(self) -> None:
        """Test severity level for critical score."""
        severity = SeverityScore.from_score(0.95)
        assert severity.level == Severity.CRITICAL

    def test_all_levels(self) -> None:
        """Test all severity levels are covered."""
        levels = [
            (0.05, Severity.NONE),
            (0.2, Severity.MINIMAL),
            (0.35, Severity.MILD),
            (0.5, Severity.MODERATE),
            (0.7, Severity.MODERATELY_SEVERE),
            (0.85, Severity.SEVERE),
            (0.95, Severity.CRITICAL),
        ]

        for score, expected_level in levels:
            severity = SeverityScore.from_score(score)
            assert severity.level == expected_level


class TestHashedValue:
    """Tests for HashedValue value object."""

    def test_from_plain_text(self) -> None:
        """Test creating hashed value from plain text."""
        hashed = HashedValue.from_plain_text("secret123")

        assert hashed.value is not None
        assert hashed.value != "secret123"
        assert len(hashed.value) == 64  # SHA-256 hex length

    def test_matches_correct(self) -> None:
        """Test matching correct value."""
        hashed = HashedValue.from_plain_text("secret123")

        assert hashed.matches("secret123") is True
        assert hashed.matches("wrong") is False

    def test_with_salt(self) -> None:
        """Test hashing with salt."""
        hashed1 = HashedValue.from_plain_text("secret", salt="salt1")
        hashed2 = HashedValue.from_plain_text("secret", salt="salt2")

        assert hashed1.value != hashed2.value
        assert hashed1.matches("secret", salt="salt1") is True
        assert hashed1.matches("secret", salt="salt2") is False


class TestUserId:
    """Tests for UserId value object."""

    def test_valid_user_id(self) -> None:
        """Test valid user ID creation."""
        user_id = UserId(value="user-123")
        assert user_id.value == "user-123"

    def test_empty_rejected(self) -> None:
        """Test empty user ID is rejected."""
        with pytest.raises(ValueError):
            UserId(value="")

        with pytest.raises(ValueError):
            UserId(value="   ")

    def test_too_long_rejected(self) -> None:
        """Test too long user ID is rejected."""
        with pytest.raises(ValueError):
            UserId(value="x" * 100)


class TestCorrelationId:
    """Tests for CorrelationId value object."""

    def test_generate(self) -> None:
        """Test generating correlation ID."""
        corr_id = CorrelationId.generate()

        assert corr_id.value is not None
        assert len(corr_id.value) == 36  # UUID format

    def test_unique_generation(self) -> None:
        """Test generated IDs are unique."""
        ids = [CorrelationId.generate() for _ in range(100)]
        unique_ids = set(c.value for c in ids)

        assert len(unique_ids) == 100
