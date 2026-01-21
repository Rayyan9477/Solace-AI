"""
Solace-AI Notification Service - Domain Entities.

Core domain entities with identity for notifications, preferences, and delivery tracking.

Architecture Layer: Domain
Principles: Rich Domain Model, Immutable Value Objects, Entity Identity
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
import structlog

logger = structlog.get_logger(__name__)


class NotificationCategory(str, Enum):
    """Categories of notifications."""
    SYSTEM = "system"
    CLINICAL = "clinical"
    SAFETY = "safety"
    THERAPY = "therapy"
    REMINDER = "reminder"
    MARKETING = "marketing"
    TRANSACTIONAL = "transactional"


class DeliveryStatus(str, Enum):
    """Status of notification delivery."""
    PENDING = "pending"
    QUEUED = "queued"
    SENDING = "sending"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPENED = "opened"
    CLICKED = "clicked"
    UNSUBSCRIBED = "unsubscribed"


class PreferenceChannel(str, Enum):
    """Notification channel preferences."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class FrequencyPreference(str, Enum):
    """Notification frequency preferences."""
    IMMEDIATE = "immediate"
    HOURLY_DIGEST = "hourly_digest"
    DAILY_DIGEST = "daily_digest"
    WEEKLY_DIGEST = "weekly_digest"
    DISABLED = "disabled"


class NotificationEntity(BaseModel):
    """
    Core notification entity with full lifecycle tracking.

    Represents a single notification instance with delivery history.
    """
    notification_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    category: NotificationCategory = Field(...)
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=4000)
    html_body: str | None = Field(default=None, max_length=50000)
    channel: PreferenceChannel = Field(...)
    status: DeliveryStatus = Field(default=DeliveryStatus.PENDING)
    priority: int = Field(default=3, ge=1, le=5)
    correlation_id: UUID | None = Field(default=None)
    reference_id: str | None = Field(default=None)
    reference_type: str | None = Field(default=None)
    template_id: str | None = Field(default=None)
    variables: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    external_message_id: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    scheduled_at: datetime | None = Field(default=None)
    sent_at: datetime | None = Field(default=None)
    delivered_at: datetime | None = Field(default=None)
    opened_at: datetime | None = Field(default=None)
    clicked_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = Field(default=None)

    @property
    def is_expired(self) -> bool:
        """Check if notification has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def can_retry(self) -> bool:
        """Check if notification can be retried."""
        return self.retry_count < self.max_retries and self.status == DeliveryStatus.FAILED

    @property
    def delivery_time_ms(self) -> int | None:
        """Calculate delivery time in milliseconds."""
        if self.sent_at is None or self.delivered_at is None:
            return None
        delta = self.delivered_at - self.sent_at
        return int(delta.total_seconds() * 1000)

    def mark_queued(self) -> None:
        """Mark notification as queued for delivery."""
        self.status = DeliveryStatus.QUEUED
        self.updated_at = datetime.now(timezone.utc)

    def mark_sending(self) -> None:
        """Mark notification as currently sending."""
        self.status = DeliveryStatus.SENDING
        self.sent_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_delivered(self, external_id: str | None = None) -> None:
        """Mark notification as delivered."""
        self.status = DeliveryStatus.DELIVERED
        self.delivered_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        if external_id:
            self.external_message_id = external_id

    def mark_failed(self, error: str) -> None:
        """Mark notification as failed."""
        self.status = DeliveryStatus.FAILED
        self.error_message = error
        self.retry_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def mark_opened(self) -> None:
        """Mark notification as opened."""
        if self.status == DeliveryStatus.DELIVERED:
            self.status = DeliveryStatus.OPENED
            self.opened_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)

    def mark_clicked(self) -> None:
        """Mark notification as clicked."""
        if self.status in (DeliveryStatus.DELIVERED, DeliveryStatus.OPENED):
            self.status = DeliveryStatus.CLICKED
            self.clicked_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)


class ChannelPreference(BaseModel):
    """User preference for a specific notification channel."""
    channel: PreferenceChannel = Field(...)
    enabled: bool = Field(default=True)
    frequency: FrequencyPreference = Field(default=FrequencyPreference.IMMEDIATE)
    quiet_hours_start: int | None = Field(default=None, ge=0, le=23)
    quiet_hours_end: int | None = Field(default=None, ge=0, le=23)
    categories_enabled: list[NotificationCategory] = Field(
        default_factory=lambda: list(NotificationCategory)
    )

    def is_in_quiet_hours(self, current_hour: int) -> bool:
        """Check if current time is within quiet hours."""
        if self.quiet_hours_start is None or self.quiet_hours_end is None:
            return False
        if self.quiet_hours_start <= self.quiet_hours_end:
            return self.quiet_hours_start <= current_hour < self.quiet_hours_end
        return current_hour >= self.quiet_hours_start or current_hour < self.quiet_hours_end


class UserNotificationPreferences(BaseModel):
    """
    Complete notification preferences for a user.

    Manages channel preferences, opt-outs, and delivery rules.
    """
    preferences_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    global_enabled: bool = Field(default=True)
    channels: dict[PreferenceChannel, ChannelPreference] = Field(default_factory=dict)
    category_overrides: dict[NotificationCategory, bool] = Field(default_factory=dict)
    unsubscribed_at: datetime | None = Field(default=None)
    unsubscribe_reason: str | None = Field(default=None)
    timezone: str = Field(default="UTC")
    language: str = Field(default="en")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_channel_enabled(self, channel: PreferenceChannel) -> bool:
        """Check if a channel is enabled."""
        if not self.global_enabled:
            return False
        pref = self.channels.get(channel)
        return pref.enabled if pref else True

    def is_category_enabled(self, category: NotificationCategory) -> bool:
        """Check if a category is enabled."""
        if not self.global_enabled:
            return False
        return self.category_overrides.get(category, True)

    def can_send(
        self,
        channel: PreferenceChannel,
        category: NotificationCategory,
        current_hour: int,
    ) -> bool:
        """Check if notification can be sent based on preferences."""
        if not self.global_enabled:
            return False
        if not self.is_category_enabled(category):
            return False
        pref = self.channels.get(channel)
        if pref is None:
            return True
        if not pref.enabled:
            return False
        if category not in pref.categories_enabled:
            return False
        if pref.is_in_quiet_hours(current_hour):
            return False
        return True

    def set_channel_preference(self, preference: ChannelPreference) -> None:
        """Set or update a channel preference."""
        self.channels[preference.channel] = preference
        self.updated_at = datetime.now(timezone.utc)

    def unsubscribe(self, reason: str | None = None) -> None:
        """Unsubscribe user from all notifications."""
        self.global_enabled = False
        self.unsubscribed_at = datetime.now(timezone.utc)
        self.unsubscribe_reason = reason
        self.updated_at = datetime.now(timezone.utc)

    def resubscribe(self) -> None:
        """Resubscribe user to notifications."""
        self.global_enabled = True
        self.unsubscribed_at = None
        self.unsubscribe_reason = None
        self.updated_at = datetime.now(timezone.utc)


class DeliveryAttempt(BaseModel):
    """Record of a single delivery attempt."""
    attempt_id: UUID = Field(default_factory=uuid4)
    notification_id: UUID = Field(...)
    channel: PreferenceChannel = Field(...)
    attempt_number: int = Field(..., ge=1)
    status: DeliveryStatus = Field(...)
    external_message_id: str | None = Field(default=None)
    provider_response: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    duration_ms: int = Field(default=0, ge=0)
    attempted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NotificationBatch(BaseModel):
    """Batch of notifications for bulk processing."""
    batch_id: UUID = Field(default_factory=uuid4)
    notifications: list[NotificationEntity] = Field(default_factory=list)
    total_count: int = Field(default=0, ge=0)
    pending_count: int = Field(default=0, ge=0)
    delivered_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = Field(default=None)

    def add_notification(self, notification: NotificationEntity) -> None:
        """Add notification to batch."""
        self.notifications.append(notification)
        self.total_count += 1
        self.pending_count += 1

    def update_counts(self) -> None:
        """Update counts based on notification statuses."""
        self.pending_count = sum(
            1 for n in self.notifications if n.status == DeliveryStatus.PENDING
        )
        self.delivered_count = sum(
            1 for n in self.notifications if n.status == DeliveryStatus.DELIVERED
        )
        self.failed_count = sum(
            1 for n in self.notifications if n.status == DeliveryStatus.FAILED
        )
        if self.pending_count == 0:
            self.completed_at = datetime.now(timezone.utc)

    @property
    def is_complete(self) -> bool:
        """Check if batch processing is complete."""
        return self.completed_at is not None
