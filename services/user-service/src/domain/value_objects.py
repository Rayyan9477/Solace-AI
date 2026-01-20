"""
Solace-AI User Service - Domain Value Objects.

Value objects are immutable objects defined by their attributes.
They have no identity and are used to describe characteristics of domain entities.

Architecture Layer: Domain
Principles: Immutability, Value Equality, Self-Validation
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator
import structlog

logger = structlog.get_logger(__name__)


class UserRole(str, Enum):
    """
    User role value object for authorization.

    Roles define access levels in the system:
    - USER: Standard patient/user access
    - PREMIUM: Premium subscription with additional features
    - CLINICIAN: Healthcare provider oversight access
    - ADMIN: System administration access
    - SYSTEM: Internal service-to-service communication

    Immutable: Roles cannot be modified, only reassigned.
    """

    USER = "user"
    PREMIUM = "premium"
    CLINICIAN = "clinician"
    ADMIN = "admin"
    SYSTEM = "system"

    def has_admin_access(self) -> bool:
        """Check if role has administrative access."""
        return self in {UserRole.ADMIN, UserRole.SYSTEM}

    def has_clinical_access(self) -> bool:
        """Check if role has clinical oversight access."""
        return self in {UserRole.CLINICIAN, UserRole.ADMIN, UserRole.SYSTEM}

    def can_access_user_data(self, target_user_id: UUID, requesting_user_id: UUID) -> bool:
        """
        Check if role can access specific user data.

        Business Rule:
        - USER can only access own data
        - CLINICIAN can access assigned patient data (validated at service layer)
        - ADMIN/SYSTEM can access all data
        """
        if self in {UserRole.ADMIN, UserRole.SYSTEM}:
            return True
        if self == UserRole.CLINICIAN:
            return True
        return target_user_id == requesting_user_id


class AccountStatus(str, Enum):
    """
    Account status value object representing user account state.

    Status Lifecycle:
    1. PENDING_VERIFICATION -> ACTIVE (email verified)
    2. ACTIVE -> SUSPENDED (violation/investigation)
    3. ACTIVE -> INACTIVE (user deactivation)
    4. SUSPENDED -> ACTIVE (reactivation)
    5. * -> INACTIVE (soft delete)

    Immutable: Status transitions are enforced in User entity.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"

    def is_operational(self) -> bool:
        """Check if account can perform operations."""
        return self == AccountStatus.ACTIVE

    def requires_verification(self) -> bool:
        """Check if account requires email verification."""
        return self == AccountStatus.PENDING_VERIFICATION

    def is_blocked(self) -> bool:
        """Check if account is blocked from access."""
        return self in {AccountStatus.SUSPENDED, AccountStatus.INACTIVE}


class ConsentType(str, Enum):
    """
    Consent type value object for HIPAA/GDPR compliance.

    Consent Types:
    - TERMS_OF_SERVICE: Platform terms acceptance
    - PRIVACY_POLICY: Privacy policy acceptance
    - DATA_PROCESSING: Personal data processing consent (GDPR)
    - CLINICAL_DATA_SHARING: Health data sharing consent (HIPAA)
    - RESEARCH_PARTICIPATION: Research data usage consent
    - MARKETING_COMMUNICATIONS: Marketing email consent
    - ANALYTICS_TRACKING: Analytics and telemetry consent

    Immutable: Consent types are predefined and immutable.
    """

    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"
    DATA_PROCESSING = "data_processing"
    CLINICAL_DATA_SHARING = "clinical_data_sharing"
    RESEARCH_PARTICIPATION = "research_participation"
    MARKETING_COMMUNICATIONS = "marketing_communications"
    ANALYTICS_TRACKING = "analytics_tracking"

    def is_required(self) -> bool:
        """Check if consent is required for platform usage."""
        return self in {
            ConsentType.TERMS_OF_SERVICE,
            ConsentType.PRIVACY_POLICY,
            ConsentType.DATA_PROCESSING,
        }

    def is_hipaa_protected(self) -> bool:
        """Check if consent involves HIPAA-protected health data."""
        return self in {
            ConsentType.CLINICAL_DATA_SHARING,
            ConsentType.RESEARCH_PARTICIPATION,
        }


class ConsentRecord(BaseModel):
    """
    Consent record value object representing user consent audit trail.

    Immutable: Once created, consent records cannot be modified.
    Business Rule: All consent changes create new records for audit compliance.

    Attributes:
        consent_id: Unique identifier for this consent record
        user_id: User who granted/revoked consent
        consent_type: Type of consent
        granted: Whether consent was granted or revoked
        version: Version of terms/policy consented to
        granted_at: Timestamp of consent action
        ip_address: IP address of consent action (audit)
        user_agent: User agent of consent action (audit)
        expires_at: Optional expiry for time-bound consents
    """

    consent_id: UUID = Field(..., description="Unique consent record identifier")
    user_id: UUID = Field(..., description="User granting consent")
    consent_type: ConsentType = Field(..., description="Type of consent")
    granted: bool = Field(..., description="Consent granted (True) or revoked (False)")
    version: str = Field(..., description="Version of consented document")

    granted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Consent timestamp (immutable)"
    )
    ip_address: str | None = Field(default=None, max_length=45, description="IP address for audit")
    user_agent: str | None = Field(default=None, max_length=500, description="User agent for audit")
    expires_at: datetime | None = Field(default=None, description="Optional consent expiry")

    model_config = {"frozen": True}

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format (semantic versioning)."""
        import re
        if not re.match(r"^\d+\.\d+(\.\d+)?$", v):
            raise ValueError("Invalid version format (expected: X.Y or X.Y.Z)")
        return v

    def is_active(self) -> bool:
        """Check if consent is currently active."""
        if not self.granted:
            return False
        if self.expires_at and datetime.now(timezone.utc) >= self.expires_at:
            return False
        return True

    def is_expired(self) -> bool:
        """Check if consent has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) >= self.expires_at


class EmailAddress(BaseModel):
    """
    Email address value object with validation.

    Immutable: Email addresses are normalized and validated on creation.
    Business Rule: Emails are stored lowercase for consistency.
    """

    value: str = Field(..., description="Email address value")

    model_config = {"frozen": True}

    @field_validator("value")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """
        Validate email format using email-validator library.

        Provides production-grade validation with:
        - DNS domain checks (configurable)
        - Internationalized domain support (IDN)
        - Email normalization
        - Deliverability checks (optional)
        """
        try:
            from email_validator import validate_email, EmailNotValidError
            # Validate and normalize email
            # check_deliverability=False for speed (enable in production if needed)
            emailinfo = validate_email(v, check_deliverability=False)
            # Return normalized email (lowercased, properly formatted)
            # Business rule: Emails are stored lowercase for consistency
            return emailinfo.normalized.lower()
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email format: {str(e)}")

    def __str__(self) -> str:
        return self.value

    def domain(self) -> str:
        """Extract domain from email address."""
        return self.value.split("@")[1]


class DisplayName(BaseModel):
    """
    Display name value object with validation.

    Immutable: Display names are validated and trimmed on creation.
    Business Rule: Display names must be 1-100 characters.
    """

    value: str = Field(..., min_length=1, max_length=100, description="Display name value")

    model_config = {"frozen": True}

    @field_validator("value")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and trim display name."""
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Display name cannot be empty")
        return trimmed

    def __str__(self) -> str:
        return self.value


class Timezone(BaseModel):
    """
    Timezone value object with validation.

    Immutable: Timezone strings follow IANA format.
    Business Rule: Defaults to UTC if not specified.
    """

    value: str = Field(..., description="IANA timezone identifier")

    model_config = {"frozen": True}

    @field_validator("value")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone format."""
        if not v or len(v) > 50:
            raise ValueError("Invalid timezone format")
        return v

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def default() -> Timezone:
        """Get default UTC timezone."""
        return Timezone(value="UTC")


class Locale(BaseModel):
    """
    Locale value object with validation.

    Immutable: Locale strings follow ISO 639-1 format (xx-XX).
    Business Rule: Controls language and regional formatting.
    """

    value: str = Field(..., description="Locale identifier (ISO 639-1)")

    model_config = {"frozen": True}

    @field_validator("value")
    @classmethod
    def validate_locale(cls, v: str) -> str:
        """Validate locale format (ISO 639-1)."""
        import re
        if not re.match(r"^[a-z]{2}-[A-Z]{2}$", v):
            raise ValueError("Invalid locale format (expected: xx-XX)")
        return v

    def __str__(self) -> str:
        return self.value

    def language_code(self) -> str:
        """Extract language code from locale."""
        return self.value.split("-")[0]

    def country_code(self) -> str:
        """Extract country code from locale."""
        return self.value.split("-")[1]

    @staticmethod
    def default() -> Locale:
        """Get default en-US locale."""
        return Locale(value="en-US")


class PasswordPolicy(BaseModel):
    """
    Password policy value object defining password requirements.

    Immutable: Password policies are configuration-driven.
    Business Rule: Enforces minimum security standards.
    """

    min_length: int = Field(default=8, ge=6, le=128, description="Minimum password length")
    require_uppercase: bool = Field(default=True, description="Require uppercase letter")
    require_lowercase: bool = Field(default=True, description="Require lowercase letter")
    require_digit: bool = Field(default=True, description="Require digit")
    require_special: bool = Field(default=True, description="Require special character")

    model_config = {"frozen": True}

    def validate_password(self, password: str) -> tuple[bool, str | None]:
        """
        Validate password against policy.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < self.min_length:
            return False, f"Password must be at least {self.min_length} characters"

        if self.require_uppercase and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"

        if self.require_lowercase and not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"

        if self.require_digit and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"

        if self.require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Password must contain at least one special character"

        return True, None

    @staticmethod
    def default() -> PasswordPolicy:
        """Get default password policy."""
        return PasswordPolicy()

    @staticmethod
    def strict() -> PasswordPolicy:
        """Get strict password policy."""
        return PasswordPolicy(
            min_length=12,
            require_uppercase=True,
            require_lowercase=True,
            require_digit=True,
            require_special=True,
        )
