"""
Solace-AI Personality Service - Kafka Event Bridge.
Bridges personality assessment and profile events to Kafka for inter-service messaging.
"""
from __future__ import annotations

from typing import Any

import structlog

from ..events import DomainEvent, EventBus, to_kafka_event

import os

try:
    from src.solace_events.publisher import EventPublisher, create_publisher
    from src.solace_events.config import KafkaSettings
    _KAFKA_AVAILABLE = True
except ImportError:
    _KAFKA_AVAILABLE = False
    _logger = structlog.get_logger(__name__)
    _logger.error("kafka_import_failed", package="solace_events",
                  hint="Install solace_events or set KAFKA_BOOTSTRAP_SERVERS")
    if os.environ.get("ENVIRONMENT", "").lower() == "production":
        raise RuntimeError("solace_events package required in production")

logger = structlog.get_logger(__name__)


class KafkaEventBridge:
    """Bridges personality events to Kafka via the local EventBus."""

    def __init__(
        self,
        event_bus: EventBus,
        kafka_settings: KafkaSettings | None = None,
        use_mock: bool = False,
    ) -> None:
        self._event_bus = event_bus

        if not _KAFKA_AVAILABLE:
            logger.info("kafka_bridge_disabled", reason="solace_events not installed")
            self._publisher = None
            return

        self._publisher = create_publisher(
            kafka_settings=kafka_settings, use_outbox=True, use_mock=use_mock,
        )
        self._started = False
        logger.info("personality_kafka_bridge_initialized", use_mock=use_mock)

    async def start(self) -> None:
        """Start the Kafka publisher and register with EventBus."""
        if self._publisher and not self._started:
            await self._publisher.start()
            self._started = True
            self._event_bus.subscribe_all(self._handle_event)
            logger.info("personality_kafka_bridge_started")

    async def stop(self) -> None:
        """Stop the Kafka publisher."""
        if self._publisher and self._started:
            await self._publisher.stop()
            self._started = False
            logger.info("personality_kafka_bridge_stopped")

    async def _handle_event(self, event: DomainEvent) -> None:
        """Handle domain event by converting and publishing to Kafka."""
        if not self._publisher or not self._started:
            return
        try:
            kafka_event = to_kafka_event(event)
            if kafka_event:
                await self._publisher.publish(kafka_event)
                logger.debug(
                    "personality_event_bridged",
                    event_type=event.event_type.value,
                    kafka_type=kafka_event.event_type,
                )
        except Exception as e:
            logger.error(
                "personality_kafka_bridge_error",
                event_type=event.event_type.value,
                error=str(e),
            )

    async def bridge_event(self, event: DomainEvent) -> bool:
        """Manually bridge a single event (for use outside EventBus)."""
        if not self._publisher or not self._started:
            return False
        try:
            kafka_event = to_kafka_event(event)
            if kafka_event:
                await self._publisher.publish(kafka_event)
                return True
            return False
        except Exception as e:
            logger.error("personality_kafka_bridge_error", error=str(e))
            return False


_bridge: KafkaEventBridge | None = None


def get_event_bridge() -> KafkaEventBridge | None:
    """Get the singleton event bridge instance."""
    return _bridge


async def initialize_event_bridge(
    event_bus: EventBus,
    kafka_settings: KafkaSettings | None = None,
    use_mock: bool = False,
) -> KafkaEventBridge:
    """Initialize and start the Kafka event bridge."""
    global _bridge
    _bridge = KafkaEventBridge(
        event_bus=event_bus, kafka_settings=kafka_settings, use_mock=use_mock,
    )
    await _bridge.start()
    return _bridge


async def shutdown_event_bridge() -> None:
    """Stop the event bridge."""
    global _bridge
    if _bridge:
        await _bridge.stop()
        _bridge = None
