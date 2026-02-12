"""
Solace-AI Therapy Service - Domain Events.
Event definitions for therapy service domain events and event handling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import UUID, uuid4
import structlog

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Therapy service event types."""
    SESSION_STARTED = "session.started"
    SESSION_PHASE_CHANGED = "session.phase_changed"
    SESSION_ENDED = "session.ended"
    INTERVENTION_STARTED = "intervention.started"
    INTERVENTION_COMPLETED = "intervention.completed"
    HOMEWORK_ASSIGNED = "homework.assigned"
    HOMEWORK_COMPLETED = "homework.completed"
    TREATMENT_PLAN_CREATED = "treatment_plan.created"
    TREATMENT_PLAN_PHASE_ADVANCED = "treatment_plan.phase_advanced"
    TREATMENT_PLAN_GOAL_ADDED = "treatment_plan.goal_added"
    TREATMENT_PLAN_GOAL_ACHIEVED = "treatment_plan.goal_achieved"
    TREATMENT_PLAN_TERMINATED = "treatment_plan.terminated"
    OUTCOME_RECORDED = "outcome.recorded"
    OUTCOME_IMPROVED = "outcome.improved"
    OUTCOME_DETERIORATED = "outcome.deteriorated"
    RISK_LEVEL_ELEVATED = "risk.elevated"
    RISK_LEVEL_CRITICAL = "risk.critical"
    STEPPED_CARE_CHANGED = "stepped_care.changed"
    SKILL_ACQUIRED = "skill.acquired"
    MILESTONE_ACHIEVED = "milestone.achieved"


@dataclass
class DomainEvent:
    """Base domain event."""
    event_id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.SESSION_STARTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    aggregate_id: UUID = field(default_factory=uuid4)
    aggregate_type: str = "therapy"
    user_id: UUID | None = None
    correlation_id: UUID | None = None
    causation_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.event_id), "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(), "aggregate_id": str(self.aggregate_id),
            "aggregate_type": self.aggregate_type,
            "user_id": str(self.user_id) if self.user_id else None,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "metadata": self.metadata, "payload": self.payload, "version": self.version,
        }


@dataclass
class SessionStartedEvent(DomainEvent):
    """Event raised when a therapy session starts."""
    event_type: EventType = field(default=EventType.SESSION_STARTED)
    aggregate_type: str = "therapy_session"

    @classmethod
    def create(cls, session_id: UUID, user_id: UUID, treatment_plan_id: UUID, session_number: int) -> SessionStartedEvent:
        return cls(aggregate_id=session_id, user_id=user_id, payload={"treatment_plan_id": str(treatment_plan_id), "session_number": session_number})


@dataclass
class SessionPhaseChangedEvent(DomainEvent):
    """Event raised when session phase changes."""
    event_type: EventType = field(default=EventType.SESSION_PHASE_CHANGED)
    aggregate_type: str = "therapy_session"

    @classmethod
    def create(cls, session_id: UUID, user_id: UUID, from_phase: str, to_phase: str, trigger: str = "automatic") -> SessionPhaseChangedEvent:
        return cls(aggregate_id=session_id, user_id=user_id, payload={"from_phase": from_phase, "to_phase": to_phase, "trigger": trigger})


@dataclass
class SessionEndedEvent(DomainEvent):
    """Event raised when a therapy session ends."""
    event_type: EventType = field(default=EventType.SESSION_ENDED)
    aggregate_type: str = "therapy_session"

    @classmethod
    def create(cls, session_id: UUID, user_id: UUID, duration_minutes: int, techniques_count: int, skills_practiced: list[str]) -> SessionEndedEvent:
        return cls(aggregate_id=session_id, user_id=user_id, payload={"duration_minutes": duration_minutes, "techniques_count": techniques_count, "skills_practiced": skills_practiced})


@dataclass
class InterventionCompletedEvent(DomainEvent):
    """Event raised when an intervention is completed."""
    event_type: EventType = field(default=EventType.INTERVENTION_COMPLETED)
    aggregate_type: str = "intervention"

    @classmethod
    def create(cls, intervention_id: UUID, session_id: UUID, user_id: UUID, technique_name: str, engagement_score: float) -> InterventionCompletedEvent:
        return cls(aggregate_id=intervention_id, user_id=user_id, payload={"session_id": str(session_id), "technique_name": technique_name, "engagement_score": engagement_score})


@dataclass
class TreatmentPlanCreatedEvent(DomainEvent):
    """Event raised when a treatment plan is created."""
    event_type: EventType = field(default=EventType.TREATMENT_PLAN_CREATED)
    aggregate_type: str = "treatment_plan"

    @classmethod
    def create(cls, plan_id: UUID, user_id: UUID, diagnosis: str, modality: str, stepped_care_level: int) -> TreatmentPlanCreatedEvent:
        return cls(aggregate_id=plan_id, user_id=user_id, payload={"diagnosis": diagnosis, "modality": modality, "stepped_care_level": stepped_care_level})


@dataclass
class TreatmentPhaseAdvancedEvent(DomainEvent):
    """Event raised when treatment plan advances to new phase."""
    event_type: EventType = field(default=EventType.TREATMENT_PLAN_PHASE_ADVANCED)
    aggregate_type: str = "treatment_plan"

    @classmethod
    def create(cls, plan_id: UUID, user_id: UUID, from_phase: str, to_phase: str, sessions_completed: int) -> TreatmentPhaseAdvancedEvent:
        return cls(aggregate_id=plan_id, user_id=user_id, payload={"from_phase": from_phase, "to_phase": to_phase, "sessions_completed": sessions_completed})


@dataclass
class GoalAchievedEvent(DomainEvent):
    """Event raised when a treatment goal is achieved."""
    event_type: EventType = field(default=EventType.TREATMENT_PLAN_GOAL_ACHIEVED)
    aggregate_type: str = "treatment_plan"

    @classmethod
    def create(cls, plan_id: UUID, goal_id: UUID, user_id: UUID, description: str) -> GoalAchievedEvent:
        return cls(aggregate_id=plan_id, user_id=user_id, payload={"goal_id": str(goal_id), "description": description})


@dataclass
class OutcomeRecordedEvent(DomainEvent):
    """Event raised when an outcome measure is recorded."""
    event_type: EventType = field(default=EventType.OUTCOME_RECORDED)
    aggregate_type: str = "outcome_measure"

    @classmethod
    def create(cls, measure_id: UUID, user_id: UUID, instrument: str, raw_score: int, is_clinical: bool) -> OutcomeRecordedEvent:
        return cls(aggregate_id=measure_id, user_id=user_id, payload={"instrument": instrument, "raw_score": raw_score, "is_clinical": is_clinical})


@dataclass
class RiskLevelElevatedEvent(DomainEvent):
    """Event raised when risk level is elevated."""
    event_type: EventType = field(default=EventType.RISK_LEVEL_ELEVATED)
    aggregate_type: str = "therapy_session"

    @classmethod
    def create(cls, session_id: UUID, user_id: UUID, previous_level: str, current_level: str, flags: list[str]) -> RiskLevelElevatedEvent:
        return cls(aggregate_id=session_id, user_id=user_id, payload={"previous_level": previous_level, "current_level": current_level, "flags": flags})


@dataclass
class SteppedCareChangedEvent(DomainEvent):
    """Event raised when stepped care level changes."""
    event_type: EventType = field(default=EventType.STEPPED_CARE_CHANGED)
    aggregate_type: str = "treatment_plan"

    @classmethod
    def create(cls, plan_id: UUID, user_id: UUID, from_level: int, to_level: int, reason: str) -> SteppedCareChangedEvent:
        return cls(aggregate_id=plan_id, user_id=user_id, payload={"from_level": from_level, "to_level": to_level, "reason": reason})


def to_kafka_event(event: DomainEvent) -> Any:
    """Convert local therapy event to canonical Kafka event for inter-service messaging.

    Maps session lifecycle and intervention events to canonical schemas.
    Returns None for internal-only events or if solace_events is not available.
    """
    try:
        from decimal import Decimal
        from src.solace_events.schemas import (
            EventMetadata as KafkaMetadata,
            TherapySessionStartedEvent as KafkaTherapyStarted,
            InterventionDeliveredEvent as KafkaInterventionDelivered,
            ProgressMilestoneEvent as KafkaMilestone,
            TherapyModality,
        )
    except ImportError:
        logger.debug("solace_events_not_available_for_bridge")
        return None

    if not event.user_id:
        return None

    meta = KafkaMetadata(
        event_id=event.event_id, timestamp=event.timestamp,
        correlation_id=event.correlation_id or event.event_id,
        source_service="therapy-service",
    )
    base: dict[str, Any] = {"user_id": event.user_id, "session_id": None, "metadata": meta}

    if event.event_type == EventType.SESSION_STARTED:
        plan_id_str = event.payload.get("treatment_plan_id")
        return KafkaTherapyStarted(
            **base,
            treatment_plan_id=UUID(plan_id_str) if plan_id_str else None,
            session_number=event.payload.get("session_number", 1),
        )

    if event.event_type == EventType.INTERVENTION_COMPLETED:
        modality_str = event.payload.get("modality", "CBT")
        try:
            modality = TherapyModality(modality_str)
        except ValueError:
            modality = TherapyModality.CBT
        return KafkaInterventionDelivered(
            **base, intervention_id=event.aggregate_id,
            technique=event.payload.get("technique_name", "unknown"),
            modality=modality,
            user_engagement_score=Decimal(str(event.payload.get("engagement_score", "0.5"))),
        )

    if event.event_type == EventType.MILESTONE_ACHIEVED:
        return KafkaMilestone(
            **base, milestone_type="goal",
            milestone_description=event.payload.get("description", ""),
            sessions_to_milestone=0,
        )

    return None


EventHandler = Callable[[DomainEvent], Coroutine[Any, Any, None]]


class EventBus:
    """In-memory event bus for domain events."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._published_events: list[DomainEvent] = []

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe handler to specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe handler to all events."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """Unsubscribe handler from event type."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            return True
        return False

    async def publish(self, event: DomainEvent) -> None:
        """Publish event to all subscribed handlers."""
        self._published_events.append(event)
        handlers = self._handlers.get(event.event_type, []) + self._global_handlers
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error("event_handler_error", event_type=event.event_type.value, error=str(e))

    async def publish_batch(self, events: list[DomainEvent]) -> None:
        """Publish multiple events."""
        for event in events:
            await self.publish(event)

    def get_published_events(self) -> list[DomainEvent]:
        """Get all published events (for testing/debugging)."""
        return self._published_events.copy()

    def clear_published_events(self) -> None:
        """Clear published events history."""
        self._published_events.clear()


class EventStore:
    """In-memory event store for event sourcing."""

    def __init__(self) -> None:
        self._events: dict[UUID, list[DomainEvent]] = {}
        self._all_events: list[DomainEvent] = []

    async def append(self, event: DomainEvent) -> None:
        """Append event to store."""
        if event.aggregate_id not in self._events:
            self._events[event.aggregate_id] = []
        self._events[event.aggregate_id].append(event)
        self._all_events.append(event)

    async def get_events(self, aggregate_id: UUID) -> list[DomainEvent]:
        """Get all events for an aggregate."""
        return self._events.get(aggregate_id, []).copy()

    async def get_events_by_type(self, event_type: EventType) -> list[DomainEvent]:
        """Get all events of a specific type."""
        return [e for e in self._all_events if e.event_type == event_type]

    async def get_events_since(self, timestamp: datetime) -> list[DomainEvent]:
        """Get all events since timestamp."""
        return [e for e in self._all_events if e.timestamp >= timestamp]

    async def get_events_for_user(self, user_id: UUID) -> list[DomainEvent]:
        """Get all events for a user."""
        return [e for e in self._all_events if e.user_id == user_id]

    async def count(self) -> int:
        """Count total events."""
        return len(self._all_events)
