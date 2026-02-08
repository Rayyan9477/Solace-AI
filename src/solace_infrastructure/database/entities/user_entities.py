"""
User domain entities for the Solace-AI platform.

Centralized SQLAlchemy ORM models for users, preferences, and consent records.
User entity is the anchor table for all cross-service foreign keys.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base_models import AuditableModel, EncryptedFieldMixin
from ..schema_registry import SchemaRegistry


# Enumerations

class UserRole(str, Enum):
    USER = "USER"
    CLINICIAN = "CLINICIAN"
    ADMIN = "ADMIN"


class UserStatus(str, Enum):
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    INACTIVE = "INACTIVE"


class ConsentType(str, Enum):
    ANALYTICS_TRACKING = "ANALYTICS_TRACKING"
    DATA_SHARING_RESEARCH = "DATA_SHARING_RESEARCH"
    DATA_SHARING_IMPROVEMENT = "DATA_SHARING_IMPROVEMENT"
    MARKETING_EMAILS = "MARKETING_EMAILS"
    TERMS_OF_SERVICE = "TERMS_OF_SERVICE"
    PRIVACY_POLICY = "PRIVACY_POLICY"


# Entity Models

@SchemaRegistry.register
class User(AuditableModel, EncryptedFieldMixin):
    """User entity - anchor table for all service relationships.

    Stores core user identity, authentication data, and profile info.
    Password hashes and PII fields are encrypted at rest.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True,
    )
    password_hash: Mapped[str] = mapped_column(
        String(512), nullable=False,
    )
    display_name: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default=UserRole.USER.value, index=True,
    )
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=UserStatus.PENDING_VERIFICATION.value, index=True,
    )

    # Contact info
    phone_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    is_on_call: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    # Locale
    timezone: Mapped[str] = mapped_column(String(50), nullable=False, default="UTC")
    locale: Mapped[str] = mapped_column(String(10), nullable=False, default="en")

    # Profile
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Verification
    email_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    email_verification_token: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # Security
    login_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    last_login: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    preferences: Mapped[UserPreferences | None] = relationship(
        "UserPreferences", back_populates="user", uselist=False,
    )
    consent_records: Mapped[list[ConsentRecord]] = relationship(
        "ConsentRecord", back_populates="user",
    )

    def __repr__(self) -> str:
        return (
            f"<User(id={self.id}, email={self.email}, "
            f"role={self.role}, status={self.status})>"
        )


@SchemaRegistry.register
class UserPreferences(AuditableModel):
    """User preferences entity for notification and UI settings."""

    __tablename__ = "user_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )

    # Notification preferences
    notification_email: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notification_sms: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notification_push: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notification_channels: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    session_reminders: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    progress_updates: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    marketing_emails: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Data sharing
    data_sharing_research: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    data_sharing_improvement: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # UI preferences
    theme: Mapped[str] = mapped_column(String(10), nullable=False, default="system")
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")

    # Accessibility
    accessibility_high_contrast: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    accessibility_large_text: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    accessibility_screen_reader: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="preferences")

    def __repr__(self) -> str:
        return f"<UserPreferences(user_id={self.user_id})>"


@SchemaRegistry.register
class ConsentRecord(AuditableModel):
    """Consent record entity for GDPR/HIPAA compliance tracking.

    Immutable audit trail of all consent grants and revocations.
    """

    __tablename__ = "consent_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    consent_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
        comment="Type of consent: ANALYTICS_TRACKING, DATA_SHARING_RESEARCH, etc."
    )
    granted: Mapped[bool] = mapped_column(
        Boolean, nullable=False,
        comment="Whether consent was granted (True) or revoked (False)"
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc), index=True,
    )
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="consent_records")

    def __repr__(self) -> str:
        return (
            f"<ConsentRecord(id={self.id}, user_id={self.user_id}, "
            f"type={self.consent_type}, granted={self.granted})>"
        )


__all__ = [
    "UserRole",
    "UserStatus",
    "ConsentType",
    "User",
    "UserPreferences",
    "ConsentRecord",
]
