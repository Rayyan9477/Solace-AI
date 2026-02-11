"""
Solace-AI Notification Service - Domain Events.

Notification domain events for inter-service communication and audit logging.

Architecture Layer: Domain
Principles: Event-Driven Architecture, Event Sourcing Compatible, Async Processing
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Awaitable
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class NotificationEventType(str, Enum):
    """Notification event types."""
    NOTIFICATION_CREATED = "notification.created"
    NOTIFICATION_QUEUED = "notification.queued"
    NOTIFICATION_SENT = "notification.sent"
    NOTIFICATION_DELIVERED = "notification.delivered"
    NOTIFICATION_FAILED = "notification.failed"
    NOTIFICATION_BOUNCED = "notification.bounced"
    NOTIFICATION_OPENED = "notification.opened"
    NOTIFICATION_CLICKED = "notification.clicked"
    NOTIFICATION_UNSUBSCRIBED = "notification.unsubscribed"
    BATCH_STARTED = "notification.batch.started"
    BATCH_COMPLETED = "notification.batch.completed"
    PREFERENCES_UPDATED = "notification.preferences.updated"
    CHANNEL_HEALTH_CHANGED = "notification.channel.health_changed"
    TEMPLATE_RENDERED = "notification.template.rendered"
    DELIVERY_RETRIED = "notification.delivery.retried"


class NotificationEvent(BaseModel):
    """Base class for all notification domain events."""
    event_id: UUID = Field(default_factory=uuid4)
    event_type: NotificationEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: UUID | None = Field(default=None)
    notification_id: UUID | None = Field(default=None)
    correlation_id: UUID = Field(default_factory=uuid4)
    source: str = Field(default="notification-service")
    version: str = Field(default="1.0.0")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize event to JSON."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> NotificationEvent:
        """Deserialize event from JSON."""
        return cls.model_validate_json(data)


class NotificationCreatedEvent(NotificationEvent):
    """Event fired when a notification is created."""
    event_type: NotificationEventType = Field(default=NotificationEventType.NOTIFICATION_CREATED)
    channel: str = Field(...)
    category: str = Field(...)
    priority: int = Field(default=3)
    template_id: str | None = Field(default=None)
    scheduled_at: datetime | None = Field(default=None)


class NotificationQueuedEvent(NotificationEvent):
    """Event fired when a notification is queued for delivery."""
    event_type: NotificationEventType = Field(default=NotificationEventType.NOTIFICATION_QUEUED)
    channel: str = Field(...)
    queue_position: int | None = Field(default=None)
    estimated_delivery: datetime | None = Field(default=None)


class NotificationSentEvent(NotificationEvent):
    """Event fired when a notification is sent to the delivery channel."""
    event_type: NotificationEventType = Field(default=NotificationEventType.NOTIFICATION_SENT)
    channel: str = Field(...)
    external_message_id: str | None = Field(default=None)
    provider: str | None = Field(default=None)


class NotificationDeliveredEvent(NotificationEvent):
    """Event fired when a notification is confirmed delivered."""
    event_type: NotificationEventType = Field(default=NotificationEventType.NOTIFICATION_DELIVERED)
    channel: str = Field(...)
    external_message_id: str | None = Field(default=None)
    delivery_time_ms: int = Field(default=0, ge=0)


class NotificationFailedEvent(NotificationEvent):
    """Event fired when a notification delivery fails."""
    event_type: NotificationEventType = Field(default=NotificationEventType.NOTIFICATION_FAILED)
    channel: str = Field(...)
    error_code: str | None = Field(default=None)
    error_message: str = Field(...)
    retry_count: int = Field(default=0, ge=0)
    will_retry: bool = Field(default=False)


class NotificationBouncedEvent(NotificationEvent):
    """Event fired when a notification bounces."""
    event_type: NotificationEventType = Field(default=NotificationEventType.NOTIFICATION_BOUNCED)
    channel: str = Field(...)
    bounce_type: str = Field(...)
    bounce_reason: str | None = Field(default=None)


class NotificationOpenedEvent(NotificationEvent):
    """Event fired when a notification is opened."""
    event_type: NotificationEventType = Field(default=NotificationEventType.NOTIFICATION_OPENED)
    channel: str = Field(...)
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_agent: str | None = Field(default=None)


class NotificationClickedEvent(NotificationEvent):
    """Event fired when a notification link is clicked."""
    event_type: NotificationEventType = Field(default=NotificationEventType.NOTIFICATION_CLICKED)
    channel: str = Field(...)
    clicked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    link_url: str | None = Field(default=None)
    link_id: str | None = Field(default=None)


class PreferencesUpdatedEvent(NotificationEvent):
    """Event fired when user notification preferences are updated."""
    event_type: NotificationEventType = Field(default=NotificationEventType.PREFERENCES_UPDATED)
    changes: dict[str, Any] = Field(default_factory=dict)
    previous_values: dict[str, Any] = Field(default_factory=dict)


class BatchStartedEvent(NotificationEvent):
    """Event fired when a notification batch starts processing."""
    event_type: NotificationEventType = Field(default=NotificationEventType.BATCH_STARTED)
    batch_id: UUID = Field(...)
    total_notifications: int = Field(..., ge=0)
    channel: str | None = Field(default=None)
    category: str | None = Field(default=None)


class BatchCompletedEvent(NotificationEvent):
    """Event fired when a notification batch completes."""
    event_type: NotificationEventType = Field(default=NotificationEventType.BATCH_COMPLETED)
    batch_id: UUID = Field(...)
    total_notifications: int = Field(..., ge=0)
    delivered_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(default=0, ge=0)


def to_kafka_event(event: NotificationEvent) -> Any:
    """Convert local notification event to canonical Kafka event for inter-service messaging.

    Only maps delivery-tracking events (sent, delivered, failed) to Kafka.
    Returns None for internal events or if solace_events is not available.
    """
    try:
        from src.solace_events.schemas import (
            EventMetadata as KafkaMetadata,
            NotificationSentKafkaEvent,
            NotificationDeliveredKafkaEvent,
            NotificationFailedKafkaEvent,
        )
    except ImportError:
        logger.debug("solace_events_not_available_for_bridge")
        return None

    if not event.user_id or not event.notification_id:
        return None

    meta = KafkaMetadata(
        event_id=event.event_id, timestamp=event.timestamp,
        correlation_id=event.correlation_id, source_service="notification-service",
    )
    base: dict[str, Any] = {
        "user_id": event.user_id, "session_id": None, "metadata": meta,
        "notification_id": event.notification_id,
    }

    if isinstance(event, NotificationSentEvent):
        return NotificationSentKafkaEvent(**base, channel=event.channel,
            external_message_id=event.external_message_id, provider=event.provider)

    if isinstance(event, NotificationDeliveredEvent):
        return NotificationDeliveredKafkaEvent(**base, channel=event.channel,
            delivery_time_ms=event.delivery_time_ms)

    if isinstance(event, NotificationFailedEvent):
        return NotificationFailedKafkaEvent(**base, channel=event.channel,
            error_code=event.error_code, error_message=event.error_message,
            retry_count=event.retry_count, will_retry=event.will_retry)

    return None


class NotificationEventHandler(ABC):
    """Abstract base class for notification event handlers."""

    @abstractmethod
    async def handle(self, event: NotificationEvent) -> None:
        """Handle a notification event."""


class NotificationEventPublisher:
    """Publisher for notification domain events."""

    def __init__(self) -> None:
        self._handlers: dict[NotificationEventType, list[NotificationEventHandler]] = {}
        self._async_callbacks: list[Callable[[NotificationEvent], Awaitable[None]]] = []
        self._event_queue: asyncio.Queue[NotificationEvent] = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._running = False
        self._event_history: list[NotificationEvent] = []
        self._max_history = 1000

    async def start(self) -> None:
        """Start the event publisher."""
        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("notification_event_publisher_started")

    async def stop(self) -> None:
        """Stop the event publisher."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("notification_event_publisher_stopped")

    def register_handler(
        self, event_type: NotificationEventType, handler: NotificationEventHandler
    ) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug("notification_handler_registered", event_type=event_type.value)

    def register_callback(
        self, callback: Callable[[NotificationEvent], Awaitable[None]]
    ) -> None:
        """Register an async callback for all events."""
        self._async_callbacks.append(callback)

    async def publish(self, event: NotificationEvent) -> None:
        """Publish a notification event asynchronously."""
        await self._event_queue.put(event)
        logger.info(
            "notification_event_published",
            event_type=event.event_type.value,
            event_id=str(event.event_id),
            notification_id=str(event.notification_id) if event.notification_id else None,
        )

    async def publish_sync(self, event: NotificationEvent) -> None:
        """Publish and immediately process a notification event."""
        await self._dispatch_event(event)

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._dispatch_event(event)
                self._event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("notification_event_processing_error", error=str(e))

    async def _dispatch_event(self, event: NotificationEvent) -> None:
        """Dispatch event to handlers and callbacks."""
        self._record_event(event)
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(
                    "notification_handler_error",
                    handler=type(handler).__name__,
                    error=str(e),
                )
        for callback in self._async_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error("notification_callback_error", error=str(e))

    def _record_event(self, event: NotificationEvent) -> None:
        """Record event in history."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

    def get_recent_events(
        self, limit: int = 100, event_type: NotificationEventType | None = None
    ) -> list[NotificationEvent]:
        """Get recent events from history."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


class DeliveryMetricsHandler(NotificationEventHandler):
    """Handler that tracks delivery metrics."""

    def __init__(self) -> None:
        self._metrics: dict[str, int] = {
            "created": 0,
            "sent": 0,
            "delivered": 0,
            "failed": 0,
            "opened": 0,
            "clicked": 0,
        }
        self._delivery_times: list[int] = []

    async def handle(self, event: NotificationEvent) -> None:
        """Track delivery metrics."""
        if event.event_type == NotificationEventType.NOTIFICATION_CREATED:
            self._metrics["created"] += 1
        elif event.event_type == NotificationEventType.NOTIFICATION_SENT:
            self._metrics["sent"] += 1
        elif event.event_type == NotificationEventType.NOTIFICATION_DELIVERED:
            self._metrics["delivered"] += 1
            if isinstance(event, NotificationDeliveredEvent):
                self._delivery_times.append(event.delivery_time_ms)
        elif event.event_type == NotificationEventType.NOTIFICATION_FAILED:
            self._metrics["failed"] += 1
        elif event.event_type == NotificationEventType.NOTIFICATION_OPENED:
            self._metrics["opened"] += 1
        elif event.event_type == NotificationEventType.NOTIFICATION_CLICKED:
            self._metrics["clicked"] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        avg_delivery_time = (
            sum(self._delivery_times) / len(self._delivery_times)
            if self._delivery_times
            else 0
        )
        return {
            **self._metrics,
            "delivery_rate": (
                self._metrics["delivered"] / max(1, self._metrics["sent"])
            ),
            "open_rate": (
                self._metrics["opened"] / max(1, self._metrics["delivered"])
            ),
            "click_rate": (
                self._metrics["clicked"] / max(1, self._metrics["opened"])
            ),
            "avg_delivery_time_ms": avg_delivery_time,
        }


_publisher: NotificationEventPublisher | None = None


def get_event_publisher() -> NotificationEventPublisher:
    """Get singleton event publisher."""
    global _publisher
    if _publisher is None:
        _publisher = NotificationEventPublisher()
    return _publisher


async def initialize_event_publisher() -> NotificationEventPublisher:
    """Initialize and start the event publisher."""
    publisher = get_event_publisher()
    await publisher.start()
    return publisher
