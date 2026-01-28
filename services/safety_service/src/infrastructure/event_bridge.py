"""
Solace-AI Safety Service - Kafka Event Bridge.
Publishes safety domain events to Kafka for inter-service communication.

This bridge connects the safety service's internal event system to the
Kafka event infrastructure, enabling real-time notifications for crisis events.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog

from ..events import (
    SafetyEvent,
    SafetyEventHandler,
    CrisisDetectedEvent as LocalCrisisDetectedEvent,
    CrisisResolvedEvent as LocalCrisisResolvedEvent,
    EscalationTriggeredEvent as LocalEscalationTriggeredEvent,
    EscalationAcknowledgedEvent as LocalEscalationAcknowledgedEvent,
    EscalationResolvedEvent as LocalEscalationResolvedEvent,
    IncidentCreatedEvent as LocalIncidentCreatedEvent,
    RiskLevelChangedEvent as LocalRiskLevelChangedEvent,
    EventType,
    get_event_publisher,
)

# Import shared event schemas for Kafka publishing
try:
    from solace_events.schemas import (
        BaseEvent,
        CrisisDetectedEvent as KafkaCrisisDetectedEvent,
        EscalationTriggeredEvent as KafkaEscalationTriggeredEvent,
        SafetyAssessmentEvent as KafkaSafetyAssessmentEvent,
        CrisisLevel,
        EventMetadata,
    )
    from solace_events.publisher import EventPublisher, create_publisher
    from solace_events.config import KafkaSettings, ProducerSettings
    _KAFKA_AVAILABLE = True
except ImportError:
    _KAFKA_AVAILABLE = False
    KafkaCrisisDetectedEvent = None
    KafkaEscalationTriggeredEvent = None
    EventPublisher = None

logger = structlog.get_logger(__name__)


def _crisis_level_to_enum(level: str) -> "CrisisLevel":
    """Convert crisis level string to CrisisLevel enum."""
    level_map = {
        "NONE": CrisisLevel.NONE,
        "LOW": CrisisLevel.LOW,
        "ELEVATED": CrisisLevel.ELEVATED,
        "MODERATE": CrisisLevel.ELEVATED,  # Map MODERATE to ELEVATED
        "HIGH": CrisisLevel.HIGH,
        "CRITICAL": CrisisLevel.CRITICAL,
    }
    return level_map.get(level.upper(), CrisisLevel.LOW)


def _priority_to_literal(priority: str) -> str:
    """Convert priority string to valid literal value."""
    valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    upper = priority.upper()
    return upper if upper in valid else "HIGH"


class KafkaEventBridge(SafetyEventHandler):
    """
    Event handler that bridges safety events to Kafka.

    Converts local safety domain events to shared event schemas
    and publishes them to the appropriate Kafka topics for
    inter-service communication.
    """

    def __init__(
        self,
        kafka_settings: "KafkaSettings | None" = None,
        producer_settings: "ProducerSettings | None" = None,
        use_mock: bool = False,
    ) -> None:
        if not _KAFKA_AVAILABLE:
            logger.warning("kafka_not_available", reason="solace_events not installed")
            self._publisher = None
            return

        self._publisher = create_publisher(
            kafka_settings=kafka_settings,
            producer_settings=producer_settings,
            use_outbox=True,  # Use outbox pattern for reliability
            use_mock=use_mock,
        )
        self._started = False
        logger.info("kafka_event_bridge_initialized", use_mock=use_mock)

    async def start(self) -> None:
        """Start the Kafka publisher."""
        if self._publisher and not self._started:
            await self._publisher.start()
            self._started = True
            logger.info("kafka_event_bridge_started")

    async def stop(self) -> None:
        """Stop the Kafka publisher."""
        if self._publisher and self._started:
            await self._publisher.stop()
            self._started = False
            logger.info("kafka_event_bridge_stopped")

    async def handle(self, event: SafetyEvent) -> None:
        """Handle a safety event by publishing to Kafka."""
        if not self._publisher or not self._started:
            logger.debug("kafka_bridge_not_started", event_type=event.event_type.value)
            return

        try:
            kafka_event = self._convert_to_kafka_event(event)
            if kafka_event:
                await self._publisher.publish(kafka_event)
                logger.info(
                    "safety_event_published_to_kafka",
                    event_type=event.event_type.value,
                    event_id=str(event.event_id),
                    kafka_event_type=kafka_event.event_type,
                )
        except Exception as e:
            logger.error(
                "kafka_publish_failed",
                event_type=event.event_type.value,
                event_id=str(event.event_id),
                error=str(e),
            )

    def _convert_to_kafka_event(self, event: SafetyEvent) -> "BaseEvent | None":
        """Convert local safety event to Kafka event schema."""
        if not _KAFKA_AVAILABLE:
            return None

        if isinstance(event, LocalCrisisDetectedEvent):
            return self._convert_crisis_detected(event)
        elif isinstance(event, LocalEscalationTriggeredEvent):
            return self._convert_escalation_triggered(event)
        # Add more conversions as needed
        return None

    def _convert_crisis_detected(
        self, event: LocalCrisisDetectedEvent
    ) -> "KafkaCrisisDetectedEvent":
        """Convert local CrisisDetectedEvent to Kafka schema."""
        return KafkaCrisisDetectedEvent(
            user_id=event.user_id,
            session_id=event.session_id,
            metadata=EventMetadata(
                event_id=event.event_id,
                timestamp=event.timestamp,
                correlation_id=event.correlation_id,
                source_service="safety-service",
            ),
            crisis_level=_crisis_level_to_enum(event.crisis_level),
            trigger_indicators=event.trigger_indicators,
            detection_layer=event.detection_layers[0] if event.detection_layers else 1,
            confidence=event.risk_score,
            escalation_action="escalate" if event.requires_escalation else "monitor",
            requires_human_review=event.requires_human_review,
        )

    def _convert_escalation_triggered(
        self, event: LocalEscalationTriggeredEvent
    ) -> "KafkaEscalationTriggeredEvent":
        """Convert local EscalationTriggeredEvent to Kafka schema."""
        return KafkaEscalationTriggeredEvent(
            user_id=event.user_id,
            session_id=event.session_id,
            metadata=EventMetadata(
                event_id=event.event_id,
                timestamp=event.timestamp,
                correlation_id=event.correlation_id,
                source_service="safety-service",
            ),
            escalation_reason=event.reason,
            priority=_priority_to_literal(event.priority),
            assigned_clinician_id=event.assigned_clinician_id,
            notification_sent=len(event.notification_channels) > 0,
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
        kafka_settings: "KafkaSettings | None" = None,
        producer_settings: "ProducerSettings | None" = None,
        use_mock: bool = False,
    ) -> None:
        self._bridge = KafkaEventBridge(
            kafka_settings=kafka_settings,
            producer_settings=producer_settings,
            use_mock=use_mock,
        )
        self._registered = False

    async def start(self) -> None:
        """Start the bridge and register with the event publisher."""
        await self._bridge.start()

        if not self._registered:
            publisher = get_event_publisher()
            # Register for crisis-related events
            publisher.register_handler(EventType.CRISIS_DETECTED, self._bridge)
            publisher.register_handler(EventType.ESCALATION_TRIGGERED, self._bridge)
            publisher.register_handler(EventType.ESCALATION_ACKNOWLEDGED, self._bridge)
            publisher.register_handler(EventType.ESCALATION_RESOLVED, self._bridge)
            publisher.register_handler(EventType.INCIDENT_CREATED, self._bridge)
            publisher.register_handler(EventType.RISK_LEVEL_CHANGED, self._bridge)
            self._registered = True
            logger.info("safety_notification_bridge_registered")

    async def stop(self) -> None:
        """Stop the bridge."""
        await self._bridge.stop()
        logger.info("safety_notification_bridge_stopped")


# Module-level singleton
_notification_bridge: SafetyCrisisNotificationBridge | None = None


def get_notification_bridge() -> SafetyCrisisNotificationBridge:
    """Get the singleton notification bridge instance."""
    global _notification_bridge
    if _notification_bridge is None:
        _notification_bridge = SafetyCrisisNotificationBridge()
    return _notification_bridge


async def initialize_notification_bridge(
    kafka_settings: "KafkaSettings | None" = None,
    use_mock: bool = False,
) -> SafetyCrisisNotificationBridge:
    """Initialize and start the notification bridge."""
    global _notification_bridge
    _notification_bridge = SafetyCrisisNotificationBridge(
        kafka_settings=kafka_settings,
        use_mock=use_mock,
    )
    await _notification_bridge.start()
    return _notification_bridge
