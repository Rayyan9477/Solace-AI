"""
Solace-AI Therapy Service - Kafka Event Bridge.
Publishes therapy domain events to Kafka for inter-service communication.

This bridge connects the therapy service's internal EventBus to the
Kafka event infrastructure, enabling real-time event sharing with other services.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog

from ..events import (
    DomainEvent,
    EventBus,
    EventType,
    SessionStartedEvent,
    SessionEndedEvent,
    SessionPhaseChangedEvent,
    InterventionCompletedEvent,
    TreatmentPlanCreatedEvent,
    TreatmentPhaseAdvancedEvent,
    GoalAchievedEvent,
    OutcomeRecordedEvent,
    RiskLevelElevatedEvent,
    SteppedCareChangedEvent,
)

# Import shared event schemas for Kafka publishing
try:
    from solace_events.schemas import (
        BaseEvent,
        TherapySessionStartedEvent as KafkaTherapySessionStartedEvent,
        InterventionDeliveredEvent as KafkaInterventionDeliveredEvent,
        ProgressMilestoneEvent as KafkaProgressMilestoneEvent,
        TherapyModality as KafkaTherapyModality,
        EventMetadata,
    )
    from solace_events.publisher import EventPublisher, create_publisher
    from solace_events.config import KafkaSettings, ProducerSettings, SolaceTopic
    _KAFKA_AVAILABLE = True
except ImportError:
    _KAFKA_AVAILABLE = False
    KafkaTherapySessionStartedEvent = None
    KafkaInterventionDeliveredEvent = None
    EventPublisher = None

logger = structlog.get_logger(__name__)


def _modality_to_enum(modality: str) -> "KafkaTherapyModality":
    """Convert modality string to KafkaTherapyModality enum."""
    if not _KAFKA_AVAILABLE:
        return None
    modality_map = {
        "CBT": KafkaTherapyModality.CBT,
        "DBT": KafkaTherapyModality.DBT,
        "ACT": KafkaTherapyModality.ACT,
        "MI": KafkaTherapyModality.MI,
        "MINDFULNESS": KafkaTherapyModality.MINDFULNESS,
        "PSYCHOEDUCATION": KafkaTherapyModality.PSYCHOEDUCATION,
    }
    return modality_map.get(modality.upper(), KafkaTherapyModality.CBT)


class KafkaTherapyEventBridge:
    """
    Event bridge that publishes therapy domain events to Kafka.

    Subscribes to the local EventBus and forwards events to Kafka
    for inter-service communication.
    """

    def __init__(
        self,
        event_bus: EventBus,
        kafka_settings: "KafkaSettings | None" = None,
        producer_settings: "ProducerSettings | None" = None,
        use_mock: bool = False,
    ) -> None:
        self._event_bus = event_bus
        self._use_mock = use_mock

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
        logger.info("kafka_therapy_event_bridge_initialized", use_mock=use_mock)

    async def start(self) -> None:
        """Start the Kafka publisher and register with EventBus."""
        if self._publisher and not self._started:
            await self._publisher.start()
            self._started = True

            # Register as global handler to receive all events
            self._event_bus.subscribe_all(self._handle_event)

            logger.info("kafka_therapy_event_bridge_started")

    async def stop(self) -> None:
        """Stop the Kafka publisher."""
        if self._publisher and self._started:
            await self._publisher.stop()
            self._started = False
            logger.info("kafka_therapy_event_bridge_stopped")

    async def _handle_event(self, event: DomainEvent) -> None:
        """Handle domain event by publishing to Kafka."""
        if not self._publisher or not self._started:
            return

        try:
            kafka_event = self._convert_to_kafka_event(event)
            if kafka_event:
                # Publish to the therapy topic
                await self._publisher.publish(kafka_event, SolaceTopic.THERAPY.value)
                logger.debug(
                    "therapy_event_published_to_kafka",
                    event_type=event.event_type.value,
                    event_id=str(event.event_id),
                )
        except Exception as e:
            logger.error(
                "kafka_publish_failed",
                event_type=event.event_type.value,
                event_id=str(event.event_id),
                error=str(e),
            )

    def _convert_to_kafka_event(self, event: DomainEvent) -> "BaseEvent | None":
        """Convert local therapy event to Kafka event schema."""
        if not _KAFKA_AVAILABLE:
            return None

        # Only convert events we have Kafka schemas for
        if isinstance(event, SessionStartedEvent):
            return self._convert_session_started(event)
        elif isinstance(event, InterventionCompletedEvent):
            return self._convert_intervention_completed(event)
        elif isinstance(event, (GoalAchievedEvent, TreatmentPhaseAdvancedEvent)):
            return self._convert_progress_milestone(event)

        # For other events, we can create a generic event or skip
        return None

    def _convert_session_started(
        self, event: SessionStartedEvent
    ) -> "KafkaTherapySessionStartedEvent | None":
        """Convert SessionStartedEvent to Kafka schema."""
        if not event.user_id:
            return None

        return KafkaTherapySessionStartedEvent(
            user_id=event.user_id,
            session_id=event.aggregate_id,
            metadata=EventMetadata(
                event_id=event.event_id,
                timestamp=event.timestamp,
                correlation_id=event.correlation_id or event.event_id,
                source_service="therapy-service",
            ),
            treatment_plan_id=UUID(event.payload.get("treatment_plan_id")) if event.payload.get("treatment_plan_id") else None,
            session_number=event.payload.get("session_number", 1),
            planned_focus=[],  # Would be populated from treatment plan
            active_modalities=[],  # Would be populated from treatment plan
        )

    def _convert_intervention_completed(
        self, event: InterventionCompletedEvent
    ) -> "KafkaInterventionDeliveredEvent | None":
        """Convert InterventionCompletedEvent to Kafka schema."""
        if not event.user_id:
            return None

        technique_name = event.payload.get("technique_name", "unknown")
        engagement_score = event.payload.get("engagement_score", 0.0)

        # Map technique name to modality (simplified mapping)
        modality_map = {
            "Thought Record": KafkaTherapyModality.CBT,
            "Behavioral Activation": KafkaTherapyModality.CBT,
            "Cognitive Restructuring": KafkaTherapyModality.CBT,
            "STOP Skill": KafkaTherapyModality.DBT,
            "Distress Tolerance": KafkaTherapyModality.DBT,
            "Mindfulness of Breath": KafkaTherapyModality.MINDFULNESS,
            "Body Scan": KafkaTherapyModality.MINDFULNESS,
            "Values Clarification": KafkaTherapyModality.ACT,
            "Defusion": KafkaTherapyModality.ACT,
            "Change Talk": KafkaTherapyModality.MI,
            "Reflective Listening": KafkaTherapyModality.MI,
        }
        modality = modality_map.get(technique_name, KafkaTherapyModality.CBT)

        return KafkaInterventionDeliveredEvent(
            user_id=event.user_id,
            session_id=UUID(event.payload.get("session_id")) if event.payload.get("session_id") else None,
            metadata=EventMetadata(
                event_id=event.event_id,
                timestamp=event.timestamp,
                correlation_id=event.correlation_id or event.event_id,
                source_service="therapy-service",
            ),
            intervention_id=event.aggregate_id,
            technique=technique_name,
            modality=modality,
            selection_rationale={},  # Would be populated from technique selector
            user_engagement_score=Decimal(str(engagement_score)) if engagement_score else None,
        )

    def _convert_progress_milestone(
        self, event: DomainEvent
    ) -> "KafkaProgressMilestoneEvent | None":
        """Convert goal/phase events to ProgressMilestoneEvent."""
        if not event.user_id:
            return None

        if isinstance(event, GoalAchievedEvent):
            milestone_type = "goal_achieved"
            milestone_description = event.payload.get("description", "Goal achieved")
        elif isinstance(event, TreatmentPhaseAdvancedEvent):
            milestone_type = "phase_advanced"
            from_phase = event.payload.get("from_phase", "")
            to_phase = event.payload.get("to_phase", "")
            milestone_description = f"Advanced from {from_phase} to {to_phase}"
        else:
            return None

        return KafkaProgressMilestoneEvent(
            user_id=event.user_id,
            session_id=None,  # Milestones are plan-level, not session-level
            metadata=EventMetadata(
                event_id=event.event_id,
                timestamp=event.timestamp,
                correlation_id=event.correlation_id or event.event_id,
                source_service="therapy-service",
            ),
            milestone_type=milestone_type,
            milestone_description=milestone_description,
            sessions_to_milestone=event.payload.get("sessions_completed", 0),
            improvement_metrics={},  # Would be populated from outcome tracking
        )


# Module-level singleton
_event_bridge: KafkaTherapyEventBridge | None = None


def get_event_bridge() -> KafkaTherapyEventBridge | None:
    """Get the singleton event bridge instance."""
    return _event_bridge


async def initialize_event_bridge(
    event_bus: EventBus,
    kafka_settings: "KafkaSettings | None" = None,
    use_mock: bool = False,
) -> KafkaTherapyEventBridge:
    """Initialize and start the event bridge."""
    global _event_bridge
    _event_bridge = KafkaTherapyEventBridge(
        event_bus=event_bus,
        kafka_settings=kafka_settings,
        use_mock=use_mock,
    )
    await _event_bridge.start()
    return _event_bridge


async def shutdown_event_bridge() -> None:
    """Stop the event bridge."""
    global _event_bridge
    if _event_bridge:
        await _event_bridge.stop()
        _event_bridge = None
