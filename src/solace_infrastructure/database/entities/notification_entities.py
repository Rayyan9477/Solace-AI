"""
Notification domain entities for the Solace-AI platform.

Centralized SQLAlchemy ORM models for notifications, delivery attempts,
user notification preferences, and notification batches.
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

from ..base_models import AuditableModel
from ..schema_registry import SchemaRegistry


# Enumerations

class NotificationCategory(str, Enum):
    SYSTEM = "SYSTEM"
    CLINICAL = "CLINICAL"
    SAFETY = "SAFETY"
    THERAPY = "THERAPY"
    REMINDER = "REMINDER"
    MARKETING = "MARKETING"
    TRANSACTIONAL = "TRANSACTIONAL"


class NotificationChannel(str, Enum):
    EMAIL = "EMAIL"
    SMS = "SMS"
    PUSH = "PUSH"
    IN_APP = "IN_APP"


class NotificationStatus(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    SENDING = "SENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    BOUNCED = "BOUNCED"
    OPENED = "OPENED"
    CLICKED = "CLICKED"
    UNSUBSCRIBED = "UNSUBSCRIBED"


class DeliveryStatus(str, Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    BOUNCED = "BOUNCED"


class FrequencyPreference(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    HOURLY_DIGEST = "HOURLY_DIGEST"
    DAILY_DIGEST = "DAILY_DIGEST"
    WEEKLY_DIGEST = "WEEKLY_DIGEST"
    DISABLED = "DISABLED"


# Entity Models

@SchemaRegistry.register
class Notification(AuditableModel):
    """Notification entity for tracking all notification lifecycle.

    Stores notification content, delivery status, and tracking data.
    """

    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    category: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="Notification category: SYSTEM, CLINICAL, SAFETY, etc."
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    html_body: Mapped[str | None] = mapped_column(Text, nullable=True)

    channel: Mapped[str] = mapped_column(
        String(10), nullable=False, index=True,
        comment="Delivery channel: EMAIL, SMS, PUSH, IN_APP"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=NotificationStatus.PENDING.value, index=True,
    )
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3,
        comment="Priority 1-5 (1=highest)"
    )

    # References
    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
    )
    reference_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reference_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    template_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    variables: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    notification_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Delivery tracking
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    external_message_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timing
    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True,
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    clicked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    delivery_attempts: Mapped[list[DeliveryAttempt]] = relationship(
        "DeliveryAttempt", back_populates="notification",
    )

    def __repr__(self) -> str:
        return (
            f"<Notification(id={self.id}, user_id={self.user_id}, "
            f"category={self.category}, status={self.status})>"
        )


@SchemaRegistry.register
class DeliveryAttempt(AuditableModel):
    """Delivery attempt entity for tracking individual send attempts."""

    __tablename__ = "delivery_attempts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    notification_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("notifications.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    channel: Mapped[str] = mapped_column(String(10), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=DeliveryStatus.PENDING.value,
    )
    external_message_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    provider_response: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc), index=True,
    )

    # Relationships
    notification: Mapped[Notification] = relationship(
        "Notification", back_populates="delivery_attempts",
    )

    def __repr__(self) -> str:
        return (
            f"<DeliveryAttempt(id={self.id}, notification_id={self.notification_id}, "
            f"attempt={self.attempt_number}, status={self.status})>"
        )


@SchemaRegistry.register
class UserNotificationPreferences(AuditableModel):
    """User notification preferences entity.

    Stores per-user notification channel preferences, quiet hours,
    and category overrides.
    """

    __tablename__ = "user_notification_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    global_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Channel preferences stored as JSONB
    channels: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Per-channel preferences with frequency, quiet hours, categories"
    )
    category_overrides: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Per-category enable/disable overrides"
    )

    # Unsubscribe tracking
    unsubscribed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    unsubscribe_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # User context
    user_timezone: Mapped[str] = mapped_column(
        String(50), nullable=False, default="UTC",
    )
    language: Mapped[str] = mapped_column(
        String(10), nullable=False, default="en",
    )

    def __repr__(self) -> str:
        return (
            f"<UserNotificationPreferences(user_id={self.user_id}, "
            f"global_enabled={self.global_enabled})>"
        )


@SchemaRegistry.register
class NotificationBatch(AuditableModel):
    """Notification batch entity for bulk notification operations."""

    __tablename__ = "notification_batches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    notification_ids: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="List of notification UUIDs in this batch"
    )
    total_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pending_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    delivered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<NotificationBatch(id={self.id}, total={self.total_count}, "
            f"delivered={self.delivered_count}, failed={self.failed_count})>"
        )


__all__ = [
    "NotificationCategory",
    "NotificationChannel",
    "NotificationStatus",
    "DeliveryStatus",
    "FrequencyPreference",
    "Notification",
    "DeliveryAttempt",
    "UserNotificationPreferences",
    "NotificationBatch",
]
