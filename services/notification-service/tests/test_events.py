"""
Unit tests for notification domain events.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from events import (
    NotificationEventType,
    NotificationEvent,
    NotificationCreatedEvent,
    NotificationQueuedEvent,
    NotificationSentEvent,
    NotificationDeliveredEvent,
    NotificationFailedEvent,
    NotificationBouncedEvent,
    NotificationOpenedEvent,
    NotificationClickedEvent,
    PreferencesUpdatedEvent,
    BatchStartedEvent,
    BatchCompletedEvent,
    NotificationEventPublisher,
    NotificationEventHandler,
    DeliveryMetricsHandler,
    get_event_publisher,
)


class TestNotificationEventType:
    """Tests for NotificationEventType enum."""

    def test_all_event_types_exist(self):
        """Test all expected event types exist."""
        assert NotificationEventType.NOTIFICATION_CREATED == "notification.created"
        assert NotificationEventType.NOTIFICATION_SENT == "notification.sent"
        assert NotificationEventType.NOTIFICATION_DELIVERED == "notification.delivered"
        assert NotificationEventType.NOTIFICATION_FAILED == "notification.failed"
        assert NotificationEventType.NOTIFICATION_OPENED == "notification.opened"
        assert NotificationEventType.NOTIFICATION_CLICKED == "notification.clicked"


class TestNotificationEvent:
    """Tests for NotificationEvent base class."""

    def test_create_base_event(self):
        """Test creating base notification event."""
        event = NotificationEvent(
            event_type=NotificationEventType.NOTIFICATION_CREATED,
            user_id=uuid4(),
            notification_id=uuid4(),
        )

        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.source == "notification-service"
        assert event.version == "1.0.0"

    def test_event_serialization(self):
        """Test event JSON serialization."""
        user_id = uuid4()
        notification_id = uuid4()
        event = NotificationEvent(
            event_type=NotificationEventType.NOTIFICATION_CREATED,
            user_id=user_id,
            notification_id=notification_id,
        )

        json_str = event.to_json()
        assert str(user_id) in json_str
        assert str(notification_id) in json_str


class TestNotificationCreatedEvent:
    """Tests for NotificationCreatedEvent."""

    def test_create_event(self):
        """Test creating notification created event."""
        event = NotificationCreatedEvent(
            user_id=uuid4(),
            notification_id=uuid4(),
            channel="email",
            category="system",
            priority=2,
            template_id="welcome_email",
        )

        assert event.event_type == NotificationEventType.NOTIFICATION_CREATED
        assert event.channel == "email"
        assert event.category == "system"
        assert event.priority == 2
        assert event.template_id == "welcome_email"


class TestNotificationQueuedEvent:
    """Tests for NotificationQueuedEvent."""

    def test_create_event(self):
        """Test creating notification queued event."""
        event = NotificationQueuedEvent(
            notification_id=uuid4(),
            channel="sms",
            queue_position=42,
        )

        assert event.event_type == NotificationEventType.NOTIFICATION_QUEUED
        assert event.channel == "sms"
        assert event.queue_position == 42


class TestNotificationSentEvent:
    """Tests for NotificationSentEvent."""

    def test_create_event(self):
        """Test creating notification sent event."""
        event = NotificationSentEvent(
            notification_id=uuid4(),
            channel="push",
            external_message_id="fcm-123456",
            provider="firebase",
        )

        assert event.event_type == NotificationEventType.NOTIFICATION_SENT
        assert event.external_message_id == "fcm-123456"
        assert event.provider == "firebase"


class TestNotificationDeliveredEvent:
    """Tests for NotificationDeliveredEvent."""

    def test_create_event(self):
        """Test creating notification delivered event."""
        event = NotificationDeliveredEvent(
            notification_id=uuid4(),
            channel="email",
            external_message_id="msg-789",
            delivery_time_ms=250,
        )

        assert event.event_type == NotificationEventType.NOTIFICATION_DELIVERED
        assert event.delivery_time_ms == 250


class TestNotificationFailedEvent:
    """Tests for NotificationFailedEvent."""

    def test_create_event(self):
        """Test creating notification failed event."""
        event = NotificationFailedEvent(
            notification_id=uuid4(),
            channel="email",
            error_code="SMTP_TIMEOUT",
            error_message="Connection timed out",
            retry_count=2,
            will_retry=True,
        )

        assert event.event_type == NotificationEventType.NOTIFICATION_FAILED
        assert event.error_code == "SMTP_TIMEOUT"
        assert event.will_retry is True


class TestNotificationBouncedEvent:
    """Tests for NotificationBouncedEvent."""

    def test_create_event(self):
        """Test creating notification bounced event."""
        event = NotificationBouncedEvent(
            notification_id=uuid4(),
            channel="email",
            bounce_type="hard",
            bounce_reason="Invalid address",
        )

        assert event.event_type == NotificationEventType.NOTIFICATION_BOUNCED
        assert event.bounce_type == "hard"


class TestNotificationOpenedEvent:
    """Tests for NotificationOpenedEvent."""

    def test_create_event(self):
        """Test creating notification opened event."""
        event = NotificationOpenedEvent(
            notification_id=uuid4(),
            user_id=uuid4(),
            channel="email",
            user_agent="Mozilla/5.0",
        )

        assert event.event_type == NotificationEventType.NOTIFICATION_OPENED
        assert event.user_agent == "Mozilla/5.0"


class TestNotificationClickedEvent:
    """Tests for NotificationClickedEvent."""

    def test_create_event(self):
        """Test creating notification clicked event."""
        event = NotificationClickedEvent(
            notification_id=uuid4(),
            user_id=uuid4(),
            channel="email",
            link_url="https://example.com/action",
            link_id="cta_button",
        )

        assert event.event_type == NotificationEventType.NOTIFICATION_CLICKED
        assert event.link_url == "https://example.com/action"


class TestPreferencesUpdatedEvent:
    """Tests for PreferencesUpdatedEvent."""

    def test_create_event(self):
        """Test creating preferences updated event."""
        event = PreferencesUpdatedEvent(
            user_id=uuid4(),
            changes={"email_enabled": False},
            previous_values={"email_enabled": True},
        )

        assert event.event_type == NotificationEventType.PREFERENCES_UPDATED
        assert event.changes["email_enabled"] is False


class TestBatchStartedEvent:
    """Tests for BatchStartedEvent."""

    def test_create_event(self):
        """Test creating batch started event."""
        batch_id = uuid4()
        event = BatchStartedEvent(
            batch_id=batch_id,
            total_notifications=100,
            channel="email",
            category="marketing",
        )

        assert event.event_type == NotificationEventType.BATCH_STARTED
        assert event.batch_id == batch_id
        assert event.total_notifications == 100


class TestBatchCompletedEvent:
    """Tests for BatchCompletedEvent."""

    def test_create_event(self):
        """Test creating batch completed event."""
        batch_id = uuid4()
        event = BatchCompletedEvent(
            batch_id=batch_id,
            total_notifications=100,
            delivered_count=95,
            failed_count=5,
            processing_time_ms=15000,
        )

        assert event.event_type == NotificationEventType.BATCH_COMPLETED
        assert event.delivered_count == 95
        assert event.failed_count == 5


class TestDeliveryMetricsHandler:
    """Tests for DeliveryMetricsHandler."""

    @pytest.mark.asyncio
    async def test_tracks_metrics(self):
        """Test that handler tracks delivery metrics."""
        handler = DeliveryMetricsHandler()

        await handler.handle(NotificationCreatedEvent(
            notification_id=uuid4(),
            channel="email",
            category="system",
        ))
        await handler.handle(NotificationSentEvent(
            notification_id=uuid4(),
            channel="email",
        ))
        await handler.handle(NotificationDeliveredEvent(
            notification_id=uuid4(),
            channel="email",
            delivery_time_ms=100,
        ))
        await handler.handle(NotificationOpenedEvent(
            notification_id=uuid4(),
            channel="email",
        ))

        metrics = handler.get_metrics()
        assert metrics["created"] == 1
        assert metrics["sent"] == 1
        assert metrics["delivered"] == 1
        assert metrics["opened"] == 1
        assert metrics["avg_delivery_time_ms"] == 100


class TestNotificationEventPublisher:
    """Tests for NotificationEventPublisher."""

    @pytest.mark.asyncio
    async def test_publish_sync(self):
        """Test synchronous event publishing."""
        publisher = NotificationEventPublisher()
        received_events = []

        async def callback(event: NotificationEvent) -> None:
            received_events.append(event)

        publisher.register_callback(callback)

        event = NotificationCreatedEvent(
            notification_id=uuid4(),
            channel="email",
            category="system",
        )
        await publisher.publish_sync(event)

        assert len(received_events) == 1
        assert received_events[0].event_id == event.event_id

    @pytest.mark.asyncio
    async def test_get_recent_events(self):
        """Test retrieving recent events."""
        publisher = NotificationEventPublisher()

        for i in range(5):
            event = NotificationCreatedEvent(
                notification_id=uuid4(),
                channel="email",
                category="system",
            )
            await publisher.publish_sync(event)

        recent = publisher.get_recent_events(limit=3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self):
        """Test filtering events by type."""
        publisher = NotificationEventPublisher()

        await publisher.publish_sync(NotificationCreatedEvent(
            notification_id=uuid4(),
            channel="email",
            category="system",
        ))
        await publisher.publish_sync(NotificationDeliveredEvent(
            notification_id=uuid4(),
            channel="email",
        ))
        await publisher.publish_sync(NotificationCreatedEvent(
            notification_id=uuid4(),
            channel="sms",
            category="clinical",
        ))

        created_events = publisher.get_recent_events(
            event_type=NotificationEventType.NOTIFICATION_CREATED
        )
        assert len(created_events) == 2


class TestGetEventPublisher:
    """Tests for get_event_publisher singleton."""

    def test_returns_singleton(self):
        """Test that get_event_publisher returns same instance."""
        import events
        events._publisher = None

        publisher1 = get_event_publisher()
        publisher2 = get_event_publisher()

        assert publisher1 is publisher2
