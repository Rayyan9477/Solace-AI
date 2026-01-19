"""
Solace-AI User Service - Domain Events.

Domain events represent significant state changes in the domain.
They enable loose coupling between bounded contexts via event-driven architecture.

Architecture Layer: Domain
Principles: Event Sourcing, Pub/Sub, Immutability
Integration: Kafka event bus (topic: solace.users)
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Domain event types for user lifecycle."""

    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_ACTIVATED = "user.activated"
    USER_SUSPENDED = "user.suspended"
    USER_REACTIVATED = "user.reactivated"

    EMAIL_VERIFIED = "user.email_verified"
    PASSWORD_CHANGED = "user.password_changed"

    PREFERENCES_UPDATED = "user.preferences_updated"

    CONSENT_GRANTED = "user.consent_granted"
    CONSENT_REVOKED = "user.consent_revoked"

    LOGIN_SUCCESSFUL = "user.login_successful"
    LOGIN_FAILED = "user.login_failed"
    ACCOUNT_LOCKED = "user.account_locked"


class DomainEvent(BaseModel):
    """
    Base domain event.

    All domain events extend this base class for consistent structure.
    Events are immutable and represent facts that have occurred.

    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of domain event
        aggregate_id: ID of the aggregate root that generated the event
        aggregate_type: Type of aggregate root (always "user")
        occurred_at: Timestamp when event occurred
        version: Event schema version for evolution
        correlation_id: Optional correlation ID for distributed tracing
        metadata: Optional additional metadata
    """

    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of domain event")
    aggregate_id: UUID = Field(..., description="User ID that generated the event")
    aggregate_type: str = Field(default="user", description="Aggregate type (user)")

    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event occurrence timestamp"
    )
    version: str = Field(default="1.0", description="Event schema version")

    correlation_id: UUID | None = Field(default=None, description="Correlation ID for tracing")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"frozen": True}

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "aggregate_id": str(self.aggregate_id),
            "aggregate_type": self.aggregate_type,
            "occurred_at": self.occurred_at.isoformat(),
            "version": self.version,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "metadata": self.metadata,
        }


class UserCreatedEvent(DomainEvent):
    """Event published when a new user is created."""
    event_type: EventType = Field(default=EventType.USER_CREATED)
    email: str
    display_name: str
    role: str
    status: str
    timezone: str
    locale: str

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({"email": self.email, "display_name": self.display_name, "role": self.role,
                     "status": self.status, "timezone": self.timezone, "locale": self.locale})
        return base


class UserUpdatedEvent(DomainEvent):
    """Event published when user profile is updated."""
    event_type: EventType = Field(default=EventType.USER_UPDATED)
    updated_fields: list[str]
    previous_values: dict[str, Any] = Field(default_factory=dict)
    new_values: dict[str, Any] = Field(default_factory=dict)


class UserDeletedEvent(DomainEvent):
    """Event published when user is deleted (soft delete)."""
    event_type: EventType = Field(default=EventType.USER_DELETED)
    reason: str | None = None
    deleted_at: datetime


class UserActivatedEvent(DomainEvent):
    """Event published when user account is activated."""
    event_type: EventType = Field(default=EventType.USER_ACTIVATED)


class UserSuspendedEvent(DomainEvent):
    """Event published when user account is suspended."""
    event_type: EventType = Field(default=EventType.USER_SUSPENDED)
    reason: str | None = None


class UserReactivatedEvent(DomainEvent):
    """Event published when suspended user is reactivated."""
    event_type: EventType = Field(default=EventType.USER_REACTIVATED)


class EmailVerifiedEvent(DomainEvent):
    """Event published when user email is verified."""
    event_type: EventType = Field(default=EventType.EMAIL_VERIFIED)
    email: str


class PasswordChangedEvent(DomainEvent):
    """Event published when user password is changed."""
    event_type: EventType = Field(default=EventType.PASSWORD_CHANGED)
    changed_by: str


class PreferencesUpdatedEvent(DomainEvent):
    """Event published when user preferences are updated."""
    event_type: EventType = Field(default=EventType.PREFERENCES_UPDATED)
    updated_fields: list[str]


class ConsentGrantedEvent(DomainEvent):
    """Event published when user grants consent."""
    event_type: EventType = Field(default=EventType.CONSENT_GRANTED)
    consent_type: str
    version: str
    ip_address: str | None = None


class ConsentRevokedEvent(DomainEvent):
    """Event published when user revokes consent."""
    event_type: EventType = Field(default=EventType.CONSENT_REVOKED)
    consent_type: str
    reason: str | None = None


class LoginSuccessfulEvent(DomainEvent):
    """Event published when user logs in successfully."""
    event_type: EventType = Field(default=EventType.LOGIN_SUCCESSFUL)
    ip_address: str | None = None
    user_agent: str | None = None


class LoginFailedEvent(DomainEvent):
    """Event published when user login attempt fails."""
    event_type: EventType = Field(default=EventType.LOGIN_FAILED)
    attempts: int
    reason: str
    ip_address: str | None = None


class AccountLockedEvent(DomainEvent):
    """Event published when user account is locked due to failed login attempts."""
    event_type: EventType = Field(default=EventType.ACCOUNT_LOCKED)
    locked_until: datetime
    attempts: int
