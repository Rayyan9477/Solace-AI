"""
Solace-AI Diagnosis Service - Kafka Event Bridge.
Bridges diagnosis completion events to Kafka for inter-service messaging.
"""
from __future__ import annotations

from typing import Any

import structlog

from ..events import DiagnosisEvent, to_kafka_event

try:
    from src.solace_events.publisher import EventPublisher, create_publisher
    from src.solace_events.config import KafkaSettings
    _KAFKA_AVAILABLE = True
except ImportError:
    _KAFKA_AVAILABLE = False

logger = structlog.get_logger(__name__)


class KafkaEventBridge:
    """Bridges diagnosis events to Kafka."""

    def __init__(
        self,
        kafka_settings: KafkaSettings | None = None,
        use_mock: bool = False,
    ) -> None:
        if not _KAFKA_AVAILABLE:
            logger.info("kafka_bridge_disabled", reason="solace_events not installed")
            self._publisher = None
            return
        self._publisher = create_publisher(
            kafka_settings=kafka_settings, use_outbox=True, use_mock=use_mock,
        )
        self._started = False
        logger.info("diagnosis_kafka_bridge_initialized")

    async def start(self) -> None:
        """Start the Kafka publisher."""
        if self._publisher and not self._started:
            await self._publisher.start()
            self._started = True
            logger.info("diagnosis_kafka_bridge_started")

    async def stop(self) -> None:
        """Stop the Kafka publisher."""
        if self._publisher and self._started:
            await self._publisher.stop()
            self._started = False
            logger.info("diagnosis_kafka_bridge_stopped")

    async def bridge_event(self, event: DiagnosisEvent) -> bool:
        """Convert and publish a diagnosis event to Kafka.

        Returns True if event was published, False otherwise.
        """
        if not self._publisher or not self._started:
            return False
        try:
            kafka_event = to_kafka_event(event)
            if kafka_event:
                await self._publisher.publish(kafka_event)
                logger.debug(
                    "diagnosis_event_bridged",
                    event_type=event.event_type,
                    kafka_type=kafka_event.event_type,
                )
                return True
            return False
        except Exception as e:
            logger.error(
                "diagnosis_kafka_bridge_error",
                event_type=event.event_type,
                error=str(e),
            )
            return False


_bridge: KafkaEventBridge | None = None


def get_event_bridge() -> KafkaEventBridge | None:
    """Get the singleton event bridge instance."""
    return _bridge


async def initialize_event_bridge(
    kafka_settings: KafkaSettings | None = None,
    use_mock: bool = False,
) -> KafkaEventBridge:
    """Initialize and start the Kafka event bridge."""
    global _bridge
    _bridge = KafkaEventBridge(kafka_settings=kafka_settings, use_mock=use_mock)
    await _bridge.start()
    return _bridge
