"""
Unit tests for User Service domain value objects.

Tests cover immutability, validation, and business logic.
"""
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from src.domain.value_objects import (
    UserRole,
    AccountStatus,
    ConsentType,
    ConsentRecord,
    EmailAddress,
    DisplayName,
    Timezone,
    Locale,
    PasswordPolicy,
)


class TestUserRole:
    """Test cases for UserRole value object."""

    def test_has_admin_access(self):
        """Test admin access check."""
        assert UserRole.ADMIN.has_admin_access() is True
        assert UserRole.SYSTEM.has_admin_access() is True
        assert UserRole.USER.has_admin_access() is False
        assert UserRole.CLINICIAN.has_admin_access() is False

    def test_has_clinical_access(self):
        """Test clinical access check."""
        assert UserRole.CLINICIAN.has_clinical_access() is True
        assert UserRole.ADMIN.has_clinical_access() is True
        assert UserRole.SYSTEM.has_clinical_access() is True
        assert UserRole.USER.has_clinical_access() is False

    def test_can_access_user_data(self):
        """Test user data access permissions."""
        user_id = uuid4()
        other_user_id = uuid4()

        assert UserRole.ADMIN.can_access_user_data(other_user_id, user_id) is True
        assert UserRole.SYSTEM.can_access_user_data(other_user_id, user_id) is True
        assert UserRole.CLINICIAN.can_access_user_data(other_user_id, user_id) is True

        assert UserRole.USER.can_access_user_data(user_id, user_id) is True
        assert UserRole.USER.can_access_user_data(other_user_id, user_id) is False


class TestAccountStatus:
    """Test cases for AccountStatus value object."""

    def test_is_operational(self):
        """Test operational status check."""
        assert AccountStatus.ACTIVE.is_operational() is True
        assert AccountStatus.INACTIVE.is_operational() is False
        assert AccountStatus.SUSPENDED.is_operational() is False

    def test_requires_verification(self):
        """Test verification requirement check."""
        assert AccountStatus.PENDING_VERIFICATION.requires_verification() is True
        assert AccountStatus.ACTIVE.requires_verification() is False

    def test_is_blocked(self):
        """Test blocked status check."""
        assert AccountStatus.SUSPENDED.is_blocked() is True
        assert AccountStatus.INACTIVE.is_blocked() is True
        assert AccountStatus.ACTIVE.is_blocked() is False


class TestConsentType:
    """Test cases for ConsentType value object."""

    def test_is_required(self):
        """Test required consent check."""
        assert ConsentType.TERMS_OF_SERVICE.is_required() is True
        assert ConsentType.PRIVACY_POLICY.is_required() is True
        assert ConsentType.DATA_PROCESSING.is_required() is True
        assert ConsentType.MARKETING_COMMUNICATIONS.is_required() is False

    def test_is_hipaa_protected(self):
        """Test HIPAA protection check."""
        assert ConsentType.CLINICAL_DATA_SHARING.is_hipaa_protected() is True
        assert ConsentType.RESEARCH_PARTICIPATION.is_hipaa_protected() is True
        assert ConsentType.MARKETING_COMMUNICATIONS.is_hipaa_protected() is False


class TestConsentRecord:
    """Test cases for ConsentRecord value object."""

    def test_consent_record_creation(self):
        """Test consent record creation."""
        user_id = uuid4()
        consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
        )

        assert consent.user_id == user_id
        assert consent.granted is True
        assert consent.version == "1.0"

    def test_consent_record_is_immutable(self):
        """Test that consent record is immutable."""
        consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=uuid4(),
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
        )

        with pytest.raises(Exception):
            consent.granted = False

    def test_invalid_version_format_raises_error(self):
        """Test that invalid version format raises error."""
        with pytest.raises(ValueError, match="Invalid version format"):
            ConsentRecord(
                consent_id=uuid4(),
                user_id=uuid4(),
                consent_type=ConsentType.TERMS_OF_SERVICE,
                granted=True,
                version="invalid",
            )

    def test_is_active_returns_true_for_granted_non_expired(self):
        """Test is_active returns True for granted non-expired consent."""
        consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=uuid4(),
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
        )

        assert consent.is_active() is True

    def test_is_active_returns_false_for_revoked(self):
        """Test is_active returns False for revoked consent."""
        consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=uuid4(),
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=False,
            version="1.0",
        )

        assert consent.is_active() is False

    def test_is_active_returns_false_for_expired(self):
        """Test is_active returns False for expired consent."""
        consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=uuid4(),
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        assert consent.is_active() is False

    def test_is_expired_checks_expiry(self):
        """Test is_expired checks expiry correctly."""
        future_consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=uuid4(),
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        assert future_consent.is_expired() is False

        past_consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=uuid4(),
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        assert past_consent.is_expired() is True


class TestEmailAddress:
    """Test cases for EmailAddress value object."""

    def test_valid_email_creation(self):
        """Test creating valid email address."""
        email = EmailAddress(value="test@example.com")

        assert email.value == "test@example.com"
        assert str(email) == "test@example.com"

    def test_email_normalized_to_lowercase(self):
        """Test email is normalized to lowercase."""
        email = EmailAddress(value="Test@Example.COM")

        assert email.value == "test@example.com"

    def test_invalid_email_raises_error(self):
        """Test invalid email raises validation error."""
        with pytest.raises(ValueError, match="Invalid email format"):
            EmailAddress(value="invalid-email")

    def test_email_domain_extraction(self):
        """Test extracting domain from email."""
        email = EmailAddress(value="test@example.com")

        assert email.domain() == "example.com"

    def test_email_is_immutable(self):
        """Test that email is immutable."""
        email = EmailAddress(value="test@example.com")

        with pytest.raises(Exception):
            email.value = "newemail@example.com"


class TestDisplayName:
    """Test cases for DisplayName value object."""

    def test_valid_display_name(self):
        """Test creating valid display name."""
        name = DisplayName(value="John Doe")

        assert name.value == "John Doe"
        assert str(name) == "John Doe"

    def test_display_name_trims_whitespace(self):
        """Test display name trims whitespace."""
        name = DisplayName(value="  John Doe  ")

        assert name.value == "John Doe"

    def test_empty_display_name_raises_error(self):
        """Test empty display name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DisplayName(value="   ")

    def test_display_name_is_immutable(self):
        """Test that display name is immutable."""
        name = DisplayName(value="John Doe")

        with pytest.raises(Exception):
            name.value = "Jane Doe"


class TestTimezone:
    """Test cases for Timezone value object."""

    def test_valid_timezone(self):
        """Test creating valid timezone."""
        tz = Timezone(value="America/New_York")

        assert tz.value == "America/New_York"
        assert str(tz) == "America/New_York"

    def test_default_timezone(self):
        """Test default UTC timezone."""
        tz = Timezone.default()

        assert tz.value == "UTC"

    def test_invalid_timezone_raises_error(self):
        """Test invalid timezone raises error."""
        with pytest.raises(ValueError, match="Invalid timezone"):
            Timezone(value="x" * 60)


class TestLocale:
    """Test cases for Locale value object."""

    def test_valid_locale(self):
        """Test creating valid locale."""
        locale = Locale(value="en-US")

        assert locale.value == "en-US"
        assert str(locale) == "en-US"

    def test_invalid_locale_raises_error(self):
        """Test invalid locale raises error."""
        with pytest.raises(ValueError, match="Invalid locale format"):
            Locale(value="invalid")

    def test_language_code_extraction(self):
        """Test extracting language code."""
        locale = Locale(value="en-US")

        assert locale.language_code() == "en"

    def test_country_code_extraction(self):
        """Test extracting country code."""
        locale = Locale(value="en-US")

        assert locale.country_code() == "US"

    def test_default_locale(self):
        """Test default en-US locale."""
        locale = Locale.default()

        assert locale.value == "en-US"


class TestPasswordPolicy:
    """Test cases for PasswordPolicy value object."""

    def test_default_password_policy(self):
        """Test default password policy."""
        policy = PasswordPolicy.default()

        assert policy.min_length == 8
        assert policy.require_uppercase is True
        assert policy.require_lowercase is True
        assert policy.require_digit is True
        assert policy.require_special is True

    def test_strict_password_policy(self):
        """Test strict password policy."""
        policy = PasswordPolicy.strict()

        assert policy.min_length == 12

    def test_validate_valid_password(self):
        """Test validating valid password."""
        policy = PasswordPolicy.default()

        is_valid, error = policy.validate_password("ValidPass123!")

        assert is_valid is True
        assert error is None

    def test_validate_password_too_short(self):
        """Test password validation fails for short password."""
        policy = PasswordPolicy.default()

        is_valid, error = policy.validate_password("Abc1!")

        assert is_valid is False
        assert "at least 8 characters" in error

    def test_validate_password_missing_uppercase(self):
        """Test password validation fails for missing uppercase."""
        policy = PasswordPolicy.default()

        is_valid, error = policy.validate_password("lowercase123!")

        assert is_valid is False
        assert "uppercase letter" in error

    def test_validate_password_missing_lowercase(self):
        """Test password validation fails for missing lowercase."""
        policy = PasswordPolicy.default()

        is_valid, error = policy.validate_password("UPPERCASE123!")

        assert is_valid is False
        assert "lowercase letter" in error

    def test_validate_password_missing_digit(self):
        """Test password validation fails for missing digit."""
        policy = PasswordPolicy.default()

        is_valid, error = policy.validate_password("ValidPassword!")

        assert is_valid is False
        assert "digit" in error

    def test_validate_password_missing_special(self):
        """Test password validation fails for missing special character."""
        policy = PasswordPolicy.default()

        is_valid, error = policy.validate_password("ValidPassword123")

        assert is_valid is False
        assert "special character" in error

    def test_password_policy_is_immutable(self):
        """Test that password policy is immutable."""
        policy = PasswordPolicy.default()

        with pytest.raises(Exception):
            policy.min_length = 12
