"""
Solace-AI User Service - Domain Entities.

Core business entities following Domain-Driven Design (DDD) principles.
Entities have identity and lifecycle. They encapsulate business rules and invariants.

Architecture Layer: Domain
Principles: Clean Architecture, SOLID, Immutability where possible
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
import structlog

from .value_objects import UserRole, AccountStatus, ConsentType

logger = structlog.get_logger(__name__)

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


class User(BaseModel):
    """
    User aggregate root representing a system user.

    Business Rules:
    - Email must be unique and valid format (validated at repository layer)
    - Password must be hashed (never store plaintext)
    - Account status transitions follow business rules
    - Soft deletes preserve audit trail
    - Email verification required before activation

    Invariants:
    - user_id is immutable once created
    - email must be valid format
    - created_at cannot be changed
    - deleted_at implies status = INACTIVE
    """

    user_id: UUID = Field(default_factory=uuid4, description="Immutable user identifier")
    email: str = Field(..., min_length=5, max_length=255, description="User email (unique, lowercase)")
    password_hash: str = Field(..., description="Bcrypt hashed password (never plaintext)")
    display_name: str = Field(..., min_length=1, max_length=100, description="User display name")
    role: UserRole = Field(default=UserRole.USER, description="User role for authorization")
    status: AccountStatus = Field(default=AccountStatus.PENDING_VERIFICATION, description="Account status")

    # Clinician on-call status for crisis notification routing
    is_on_call: bool = Field(default=False, description="Whether clinician is currently on-call for crisis notifications")
    phone_number: str | None = Field(default=None, max_length=20, description="Phone number for SMS notifications")

    timezone: str = Field(default="UTC", description="User timezone (IANA format)")
    locale: str = Field(default="en-US", description="User locale (ISO 639-1)")
    avatar_url: str | None = Field(default=None, max_length=500, description="Avatar image URL")
    bio: str | None = Field(default=None, max_length=500, description="User biography")

    email_verified: bool = Field(default=False, description="Email verification status")
    email_verification_token: str | None = Field(default=None, description="Email verification token")

    login_attempts: int = Field(default=0, ge=0, description="Failed login attempt counter")
    locked_until: datetime | None = Field(default=None, description="Account lockout expiry")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Entity creation timestamp (immutable)"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    last_login: datetime | None = Field(default=None, description="Last successful login")
    deleted_at: datetime | None = Field(default=None, description="Soft delete timestamp")

    model_config = {"frozen": False, "validate_assignment": True}

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format and normalize to lowercase."""
        if not EMAIL_REGEX.match(v):
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone format (basic check)."""
        if not v or len(v) > 50:
            raise ValueError("Invalid timezone format")
        return v

    @field_validator("locale")
    @classmethod
    def validate_locale(cls, v: str) -> str:
        """Validate locale format (ISO 639-1)."""
        if not re.match(r"^[a-z]{2}-[A-Z]{2}$", v):
            raise ValueError("Invalid locale format (expected: xx-XX)")
        return v

    def activate(self) -> None:
        """
        Activate user account after email verification.

        Business Rule: Can only activate from PENDING_VERIFICATION status.
        """
        if self.status != AccountStatus.PENDING_VERIFICATION:
            raise ValueError(f"Cannot activate user with status: {self.status}")
        if not self.email_verified:
            raise ValueError("Email must be verified before activation")

        self.status = AccountStatus.ACTIVE
        self.updated_at = datetime.now(timezone.utc)
        logger.info("user_activated", user_id=str(self.user_id))

    def suspend(self, reason: str | None = None) -> None:
        """
        Suspend user account.

        Business Rule: Can only suspend ACTIVE accounts.
        """
        if self.status != AccountStatus.ACTIVE:
            raise ValueError(f"Cannot suspend user with status: {self.status}")

        self.status = AccountStatus.SUSPENDED
        self.updated_at = datetime.now(timezone.utc)
        logger.warning("user_suspended", user_id=str(self.user_id), reason=reason)

    def reactivate(self) -> None:
        """
        Reactivate suspended account.

        Business Rule: Can only reactivate SUSPENDED or INACTIVE accounts.
        """
        if self.status not in {AccountStatus.SUSPENDED, AccountStatus.INACTIVE}:
            raise ValueError(f"Cannot reactivate user with status: {self.status}")
        if not self.email_verified:
            raise ValueError("Email must be verified before reactivation")

        self.status = AccountStatus.ACTIVE
        self.login_attempts = 0
        self.locked_until = None
        self.updated_at = datetime.now(timezone.utc)
        logger.info("user_reactivated", user_id=str(self.user_id))

    def soft_delete(self) -> None:
        """
        Soft delete user account (GDPR compliance).

        Business Rule: Preserves audit trail while marking as deleted.
        """
        self.deleted_at = datetime.now(timezone.utc)
        self.status = AccountStatus.INACTIVE
        self.email = f"deleted_{self.user_id}@deleted.solace-ai.com"
        self.email_verified = False
        self.email_verification_token = None
        self.avatar_url = None
        self.bio = None
        self.updated_at = datetime.now(timezone.utc)
        logger.info("user_soft_deleted", user_id=str(self.user_id))

    def record_login_attempt(self, success: bool, max_attempts: int = 5, lockout_duration_minutes: int = 30) -> None:
        """
        Record login attempt and enforce lockout policy.

        Business Rule: Lock account after max failed attempts.

        Args:
            success: Whether login was successful
            max_attempts: Maximum failed attempts before lockout
            lockout_duration_minutes: Duration of lockout period
        """
        if success:
            self.login_attempts = 0
            self.locked_until = None
            self.last_login = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)
            logger.info("login_successful", user_id=str(self.user_id))
        else:
            self.login_attempts += 1
            self.updated_at = datetime.now(timezone.utc)

            if self.login_attempts >= max_attempts:
                from datetime import timedelta
                self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=lockout_duration_minutes)
                logger.warning(
                    "account_locked",
                    user_id=str(self.user_id),
                    attempts=self.login_attempts,
                    locked_until=self.locked_until.isoformat()
                )
            else:
                logger.warning(
                    "login_failed",
                    user_id=str(self.user_id),
                    attempts=self.login_attempts,
                    remaining=max_attempts - self.login_attempts
                )

    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if not self.locked_until:
            return False
        return datetime.now(timezone.utc) < self.locked_until

    def is_active(self) -> bool:
        """Check if account is active and not deleted."""
        return (
            self.status == AccountStatus.ACTIVE
            and not self.deleted_at
            and not self.is_locked()
        )

    def can_login(self) -> tuple[bool, str | None]:
        """
        Check if user can login.

        Returns:
            Tuple of (can_login, error_message)
        """
        if self.deleted_at:
            return False, "Account has been deleted"
        if self.status == AccountStatus.SUSPENDED:
            return False, "Account is suspended"
        if self.status == AccountStatus.PENDING_VERIFICATION:
            return False, "Email verification required"
        if self.is_locked():
            return False, f"Account is locked until {self.locked_until.isoformat()}"
        if self.status != AccountStatus.ACTIVE:
            return False, f"Account is {self.status.value}"
        return True, None

    def update_profile(self, **kwargs: Any) -> None:
        """
        Update user profile fields.

        Business Rule: Only allows updating specific safe fields.
        """
        allowed_fields = {"display_name", "timezone", "locale", "avatar_url", "bio"}

        for key, value in kwargs.items():
            if key not in allowed_fields:
                raise ValueError(f"Cannot update field: {key}")
            if value is not None:
                setattr(self, key, value)

        self.updated_at = datetime.now(timezone.utc)
        logger.info("profile_updated", user_id=str(self.user_id), fields=list(kwargs.keys()))


class UserPreferences(BaseModel):
    """
    User preferences entity for personalization settings.

    Business Rules:
    - Preferences are optional and have defaults
    - Changes trigger updated_at timestamp
    - Privacy settings control data usage
    - Notification preferences control communication channels

    Invariants:
    - user_id is immutable
    - Boolean fields have clear defaults
    - created_at cannot be changed
    """

    user_id: UUID = Field(..., description="Reference to User entity (immutable)")

    notification_email: bool = Field(default=True, description="Enable email notifications")
    notification_sms: bool = Field(default=False, description="Enable SMS notifications")
    notification_push: bool = Field(default=True, description="Enable push notifications")
    notification_channels: list[str] = Field(
        default_factory=lambda: ["email", "in_app"],
        description="Active notification channels"
    )

    session_reminders: bool = Field(default=True, description="Session reminder notifications")
    progress_updates: bool = Field(default=True, description="Progress update notifications")
    marketing_emails: bool = Field(default=False, description="Marketing email consent")

    data_sharing_research: bool = Field(default=False, description="Share data for research (HIPAA)")
    data_sharing_improvement: bool = Field(default=True, description="Share data for service improvement")

    theme: str = Field(default="system", description="UI theme preference")
    language: str = Field(default="en", description="Language preference (ISO 639-1)")

    accessibility_high_contrast: bool = Field(default=False, description="High contrast mode")
    accessibility_large_text: bool = Field(default=False, description="Large text mode")
    accessibility_screen_reader: bool = Field(default=False, description="Screen reader optimization")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Entity creation timestamp (immutable)"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )

    model_config = {"frozen": False, "validate_assignment": True}

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        """Validate theme value."""
        allowed_themes = {"system", "light", "dark"}
        if v not in allowed_themes:
            raise ValueError(f"Invalid theme. Allowed: {allowed_themes}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code (ISO 639-1)."""
        if not re.match(r"^[a-z]{2}$", v):
            raise ValueError("Invalid language code (expected: xx)")
        return v

    def update(self, **kwargs: Any) -> None:
        """
        Update preferences.

        Args:
            **kwargs: Preference fields to update
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown preference field: {key}")
            if key in {"user_id", "created_at"}:
                raise ValueError(f"Cannot update immutable field: {key}")
            setattr(self, key, value)

        self.updated_at = datetime.now(timezone.utc)
        logger.info("preferences_updated", user_id=str(self.user_id), fields=list(kwargs.keys()))

    def get_notification_channels(self) -> list[str]:
        """Get active notification channels based on preferences."""
        channels = []
        if self.notification_email:
            channels.append("email")
        if self.notification_sms:
            channels.append("sms")
        if self.notification_push:
            channels.append("push")
        if "in_app" in self.notification_channels:
            channels.append("in_app")
        return list(set(channels))

    def has_marketing_consent(self) -> bool:
        """Check if user has consented to marketing communications."""
        return self.marketing_emails

    def has_research_consent(self) -> bool:
        """Check if user has consented to research data sharing (HIPAA)."""
        return self.data_sharing_research
