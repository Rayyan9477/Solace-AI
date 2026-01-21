"""
Unit tests for notification domain entities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from domain.entities import (
    NotificationCategory,
    DeliveryStatus,
    PreferenceChannel,
    FrequencyPreference,
    NotificationEntity,
    ChannelPreference,
    UserNotificationPreferences,
    DeliveryAttempt,
    NotificationBatch,
)


class TestNotificationCategory:
    """Tests for NotificationCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        assert NotificationCategory.SYSTEM == "system"
        assert NotificationCategory.CLINICAL == "clinical"
        assert NotificationCategory.SAFETY == "safety"
        assert NotificationCategory.THERAPY == "therapy"
        assert NotificationCategory.REMINDER == "reminder"


class TestDeliveryStatus:
    """Tests for DeliveryStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert DeliveryStatus.PENDING == "pending"
        assert DeliveryStatus.DELIVERED == "delivered"
        assert DeliveryStatus.FAILED == "failed"
        assert DeliveryStatus.OPENED == "opened"
        assert DeliveryStatus.CLICKED == "clicked"


class TestNotificationEntity:
    """Tests for NotificationEntity model."""

    def test_create_notification(self):
        """Test creating a notification entity."""
        user_id = uuid4()
        notification = NotificationEntity(
            user_id=user_id,
            category=NotificationCategory.SYSTEM,
            title="Test Notification",
            body="This is a test notification body",
            channel=PreferenceChannel.EMAIL,
        )

        assert notification.user_id == user_id
        assert notification.category == NotificationCategory.SYSTEM
        assert notification.title == "Test Notification"
        assert notification.status == DeliveryStatus.PENDING
        assert notification.priority == 3
        assert notification.retry_count == 0
        assert notification.notification_id is not None

    def test_notification_lifecycle(self):
        """Test notification status transitions."""
        notification = NotificationEntity(
            user_id=uuid4(),
            category=NotificationCategory.CLINICAL,
            title="Clinical Alert",
            body="Patient update required",
            channel=PreferenceChannel.EMAIL,
        )

        assert notification.status == DeliveryStatus.PENDING

        notification.mark_queued()
        assert notification.status == DeliveryStatus.QUEUED

        notification.mark_sending()
        assert notification.status == DeliveryStatus.SENDING
        assert notification.sent_at is not None

        notification.mark_delivered("msg-123")
        assert notification.status == DeliveryStatus.DELIVERED
        assert notification.delivered_at is not None
        assert notification.external_message_id == "msg-123"

    def test_notification_failure_and_retry(self):
        """Test notification failure handling."""
        notification = NotificationEntity(
            user_id=uuid4(),
            category=NotificationCategory.SYSTEM,
            title="Test",
            body="Test body",
            channel=PreferenceChannel.EMAIL,
            max_retries=3,
        )

        notification.mark_failed("Connection timeout")
        assert notification.status == DeliveryStatus.FAILED
        assert notification.retry_count == 1
        assert notification.error_message == "Connection timeout"
        assert notification.can_retry is True

        notification.mark_failed("Connection timeout")
        notification.mark_failed("Connection timeout")
        assert notification.retry_count == 3
        assert notification.can_retry is False

    def test_notification_expiration(self):
        """Test notification expiration."""
        notification = NotificationEntity(
            user_id=uuid4(),
            category=NotificationCategory.REMINDER,
            title="Reminder",
            body="Don't forget",
            channel=PreferenceChannel.PUSH,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert notification.is_expired is True

        notification2 = NotificationEntity(
            user_id=uuid4(),
            category=NotificationCategory.REMINDER,
            title="Reminder",
            body="Don't forget",
            channel=PreferenceChannel.PUSH,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert notification2.is_expired is False

    def test_delivery_time_calculation(self):
        """Test delivery time calculation."""
        notification = NotificationEntity(
            user_id=uuid4(),
            category=NotificationCategory.SYSTEM,
            title="Test",
            body="Test body",
            channel=PreferenceChannel.EMAIL,
        )

        assert notification.delivery_time_ms is None

        notification.mark_sending()
        assert notification.delivery_time_ms is None

        notification.mark_delivered()
        assert notification.delivery_time_ms is not None
        assert notification.delivery_time_ms >= 0


class TestChannelPreference:
    """Tests for ChannelPreference model."""

    def test_create_channel_preference(self):
        """Test creating channel preference."""
        pref = ChannelPreference(
            channel=PreferenceChannel.EMAIL,
            enabled=True,
            frequency=FrequencyPreference.IMMEDIATE,
        )

        assert pref.channel == PreferenceChannel.EMAIL
        assert pref.enabled is True
        assert pref.frequency == FrequencyPreference.IMMEDIATE

    def test_quiet_hours(self):
        """Test quiet hours functionality."""
        pref = ChannelPreference(
            channel=PreferenceChannel.PUSH,
            enabled=True,
            quiet_hours_start=22,
            quiet_hours_end=8,
        )

        assert pref.is_in_quiet_hours(23) is True
        assert pref.is_in_quiet_hours(3) is True
        assert pref.is_in_quiet_hours(10) is False
        assert pref.is_in_quiet_hours(21) is False

    def test_quiet_hours_same_day(self):
        """Test quiet hours within same day."""
        pref = ChannelPreference(
            channel=PreferenceChannel.PUSH,
            enabled=True,
            quiet_hours_start=12,
            quiet_hours_end=14,
        )

        assert pref.is_in_quiet_hours(13) is True
        assert pref.is_in_quiet_hours(11) is False
        assert pref.is_in_quiet_hours(15) is False


class TestUserNotificationPreferences:
    """Tests for UserNotificationPreferences model."""

    def test_create_preferences(self):
        """Test creating user preferences."""
        user_id = uuid4()
        prefs = UserNotificationPreferences(user_id=user_id)

        assert prefs.user_id == user_id
        assert prefs.global_enabled is True
        assert prefs.timezone == "UTC"
        assert prefs.language == "en"

    def test_channel_enabled(self):
        """Test channel enabled check."""
        prefs = UserNotificationPreferences(user_id=uuid4())

        assert prefs.is_channel_enabled(PreferenceChannel.EMAIL) is True

        prefs.set_channel_preference(ChannelPreference(
            channel=PreferenceChannel.SMS,
            enabled=False,
        ))
        assert prefs.is_channel_enabled(PreferenceChannel.SMS) is False

    def test_can_send_with_preferences(self):
        """Test can_send with various preferences."""
        prefs = UserNotificationPreferences(user_id=uuid4())
        prefs.set_channel_preference(ChannelPreference(
            channel=PreferenceChannel.EMAIL,
            enabled=True,
            categories_enabled=[NotificationCategory.SYSTEM, NotificationCategory.CLINICAL],
        ))

        assert prefs.can_send(PreferenceChannel.EMAIL, NotificationCategory.SYSTEM, 10) is True
        assert prefs.can_send(PreferenceChannel.EMAIL, NotificationCategory.MARKETING, 10) is False

    def test_unsubscribe_resubscribe(self):
        """Test unsubscribe and resubscribe."""
        prefs = UserNotificationPreferences(user_id=uuid4())

        prefs.unsubscribe("User requested")
        assert prefs.global_enabled is False
        assert prefs.unsubscribed_at is not None
        assert prefs.unsubscribe_reason == "User requested"
        assert prefs.can_send(PreferenceChannel.EMAIL, NotificationCategory.SYSTEM, 10) is False

        prefs.resubscribe()
        assert prefs.global_enabled is True
        assert prefs.unsubscribed_at is None


class TestDeliveryAttempt:
    """Tests for DeliveryAttempt model."""

    def test_create_delivery_attempt(self):
        """Test creating delivery attempt record."""
        notification_id = uuid4()
        attempt = DeliveryAttempt(
            notification_id=notification_id,
            channel=PreferenceChannel.EMAIL,
            attempt_number=1,
            status=DeliveryStatus.DELIVERED,
            external_message_id="msg-456",
            duration_ms=150,
        )

        assert attempt.notification_id == notification_id
        assert attempt.attempt_number == 1
        assert attempt.status == DeliveryStatus.DELIVERED
        assert attempt.duration_ms == 150


class TestNotificationBatch:
    """Tests for NotificationBatch model."""

    def test_create_batch(self):
        """Test creating notification batch."""
        batch = NotificationBatch()

        assert batch.total_count == 0
        assert batch.pending_count == 0
        assert batch.is_complete is False

    def test_add_notifications(self):
        """Test adding notifications to batch."""
        batch = NotificationBatch()

        for i in range(5):
            notification = NotificationEntity(
                user_id=uuid4(),
                category=NotificationCategory.SYSTEM,
                title=f"Notification {i}",
                body="Test body",
                channel=PreferenceChannel.EMAIL,
            )
            batch.add_notification(notification)

        assert batch.total_count == 5
        assert batch.pending_count == 5

    def test_update_counts(self):
        """Test updating batch counts."""
        batch = NotificationBatch()

        for i in range(3):
            notification = NotificationEntity(
                user_id=uuid4(),
                category=NotificationCategory.SYSTEM,
                title=f"Notification {i}",
                body="Test body",
                channel=PreferenceChannel.EMAIL,
            )
            batch.add_notification(notification)

        batch.notifications[0].mark_delivered()
        batch.notifications[1].mark_delivered()
        batch.notifications[2].mark_failed("Error")

        batch.update_counts()

        assert batch.delivered_count == 2
        assert batch.failed_count == 1
        assert batch.pending_count == 0
        assert batch.is_complete is True
