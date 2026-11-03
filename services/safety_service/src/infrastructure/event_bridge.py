"""
Solace-AI Safety Service - Kafka Event Bridge.
Publishes safety domain events to Kafka for inter-service communication.

This bridge connects the safety service's internal event system to the
Kafka event infrastructure, enabling real-time notifications for crisis events.
"""
from __future__ import annotations

from typing import Any

import structlog

from ..events import (
    SafetyEvent,
    SafetyEventHandler,
    EventType,
    get_event_publisher,
    to_kafka_event,
)

from solace_events.schemas import BaseEvent
from solace_events.publisher import EventPublisher, OutboxPoller, create_publisher
from solace_events.config import KafkaSettings, ProducerSettings

# MVP-133: a missing package is now an ImportError in every environment.
_KAFKA_AVAILABLE = True

logger = structlog.get_logger(__name__)

# Event types that should be forwarded to Kafka
_BRIDGED_EVENT_TYPES = frozenset({
    EventType.CRISIS_DETECTED,
    EventType.CRISIS_RESOLVED,
    EventType.ESCALATION_TRIGGERED,
    EventType.ESCALATION_ACKNOWLEDGED,
    EventType.ESCALATION_RESOLVED,
    EventType.SAFETY_CHECK_COMPLETED,
    EventType.RISK_LEVEL_CHANGED,
    EventType.INCIDENT_CREATED,
    EventType.INCIDENT_RESOLVED,
    EventType.SAFETY_PLAN_CREATED,
    EventType.SAFETY_PLAN_ACTIVATED,
    EventType.SAFETY_PLAN_UPDATED,
    EventType.OUTPUT_FILTERED,
    EventType.TRAJECTORY_ALERT,
})


class KafkaEventBridge(SafetyEventHandler):
    """
    Event handler that bridges safety events to Kafka.

    Converts local safety domain events to shared event schemas
    and publishes them to the appropriate Kafka topics for
    inter-service communication.
    """

    def __init__(
        self,
        kafka_settings: KafkaSettings | None = None,
        producer_settings: ProducerSettings | None = None,
        use_mock: bool = False,
        postgres_pool: Any = None,
    ) -> None:
        self._started = False
        self._poller = None
        if not _KAFKA_AVAILABLE:
            logger.warning("kafka_not_available", reason="solace_events not installed")
            self._publisher = None
            return

        self._publisher = create_publisher(
            kafka_settings=kafka_settings,
            producer_settings=producer_settings,
            use_outbox=True,
            use_mock=use_mock,
            postgres_pool=postgres_pool,
            source_service="safety-service",
        )
        logger.info("kafka_event_bridge_initialized", use_mock=use_mock)

    async def start(self) -> None:
        """Start the Kafka publisher and the outbox drain poller."""
        if self._publisher and not self._started:
            await self._publisher.start()
            # REV-01: drain the outbox to Kafka. Without a running poller the
            # publisher writes events to the outbox but never sends them, so no
            # safety event ever reaches the bus.
            self._poller = OutboxPoller(self._publisher)
            await self._poller.start()
            self._started = True
            logger.info("kafka_event_bridge_started")

    async def stop(self) -> None:
        """Stop the outbox poller and the Kafka publisher."""
        if self._publisher and self._started:
            if self._poller:
                await self._poller.stop()
                self._poller = None
            await self._publisher.stop()
            self._started = False
            logger.info("kafka_event_bridge_stopped")

    async def handle(self, event: SafetyEvent) -> None:
        """Handle a safety event by converting and publishing to Kafka."""
        if not self._publisher or not self._started:
            logger.debug("kafka_bridge_not_started", event_type=event.event_type.value)
            return

        try:
            kafka_event = to_kafka_event(event)
            if kafka_event:
                await self._publisher.publish(kafka_event)
                logger.info(
                    "safety_event_published_to_kafka",
                    event_type=event.event_type.value,
                    event_id=str(event.event_id),
                    kafka_event_type=kafka_event.event_type,
                )
            else:
                logger.debug("event_not_bridged_to_kafka", event_type=event.event_type.value)
        except Exception as e:
            logger.error(
                "kafka_publish_failed",
                event_type=event.event_type.value,
                event_id=str(event.event_id),
                error=str(e),
            )


class SafetyCrisisNotificationBridge:
    """
    High-level bridge for crisis notifications.

    Integrates with the safety service's event publisher to automatically
    send crisis and escalation events to Kafka for notification service
    consumption.
    """

    def __init__(
        self,
        kafka_settings: KafkaSettings | None = None,
        producer_settings: ProducerSettings | None = None,
        use_mock: bool = False,
        postgres_pool: Any = None,
    ) -> None:
        self._bridge = KafkaEventBridge(
            kafka_settings=kafka_settings,
            producer_settings=producer_settings,
            use_mock=use_mock,
            postgres_pool=postgres_pool,
        )
        self._registered = False

    async def start(self) -> None:
        """Start the bridge and register with the event publisher."""
        await self._bridge.start()

        if not self._registered:
            publisher = get_event_publisher()
            for event_type in _BRIDGED_EVENT_TYPES:
                publisher.register_handler(event_type, self._bridge)
            self._registered = True
            logger.info(
                "safety_notification_bridge_registered",
                event_types=[et.value for et in _BRIDGED_EVENT_TYPES],
            )

    async def stop(self) -> None:
        """Stop the bridge."""
        await self._bridge.stop()
        logger.info("safety_notification_bridge_stopped")


_notification_bridge: SafetyCrisisNotificationBridge | None = None


def get_notification_bridge() -> SafetyCrisisNotificationBridge:
    """Get the singleton notification bridge instance."""
    global _notification_bridge
    if _notification_bridge is None:
        _notification_bridge = SafetyCrisisNotificationBridge()
    return _notification_bridge


async def initialize_notification_bridge(
    kafka_settings: KafkaSettings | None = None,
    use_mock: bool = False,
    postgres_pool: Any = None,
) -> SafetyCrisisNotificationBridge:
    """Initialize and start the notification bridge."""
    global _notification_bridge
    _notification_bridge = SafetyCrisisNotificationBridge(
        kafka_settings=kafka_settings,
        use_mock=use_mock,
        postgres_pool=postgres_pool,
    )
    await _notification_bridge.start()
    return _notification_bridge
