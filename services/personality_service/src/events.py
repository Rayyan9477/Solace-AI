"""
Solace-AI Personality Service - Domain Events.
Event definitions for personality service domain events and event handling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import UUID, uuid4
import structlog

from .schemas import PersonalityTrait, AssessmentSource, CommunicationStyleType

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Personality service event types."""
    PROFILE_CREATED = "profile.created"
    PROFILE_UPDATED = "profile.updated"
    PROFILE_DELETED = "profile.deleted"
    ASSESSMENT_COMPLETED = "assessment.completed"
    ASSESSMENT_AGGREGATED = "assessment.aggregated"
    TRAIT_CHANGED = "trait.changed"
    TRAIT_STABILIZED = "trait.stabilized"
    STYLE_DERIVED = "style.derived"
    STYLE_CHANGED = "style.changed"
    STABILITY_ACHIEVED = "stability.achieved"
    STABILITY_LOST = "stability.lost"
    SNAPSHOT_CAPTURED = "snapshot.captured"
    PROFILE_RESET = "profile.reset"


@dataclass
class DomainEvent:
    """Base domain event for personality service."""
    event_id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.PROFILE_UPDATED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    aggregate_id: UUID = field(default_factory=uuid4)
    aggregate_type: str = "personality_profile"
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainEvent:
        """Create event from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]) if isinstance(data["event_id"], str) else data["event_id"],
            event_type=EventType(data["event_type"]) if isinstance(data["event_type"], str) else data["event_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            aggregate_id=UUID(data["aggregate_id"]) if isinstance(data["aggregate_id"], str) else data["aggregate_id"],
            aggregate_type=data.get("aggregate_type", "personality_profile"),
            user_id=UUID(data["user_id"]) if data.get("user_id") and isinstance(data["user_id"], str) else data.get("user_id"),
            correlation_id=UUID(data["correlation_id"]) if data.get("correlation_id") and isinstance(data["correlation_id"], str) else data.get("correlation_id"),
            causation_id=UUID(data["causation_id"]) if data.get("causation_id") and isinstance(data["causation_id"], str) else data.get("causation_id"),
            metadata=data.get("metadata", {}), payload=data.get("payload", {}),
            version=data.get("version", 1),
        )


@dataclass
class ProfileCreatedEvent(DomainEvent):
    """Event raised when a personality profile is created."""
    event_type: EventType = field(default=EventType.PROFILE_CREATED)

    @classmethod
    def create(cls, profile_id: UUID, user_id: UUID) -> ProfileCreatedEvent:
        return cls(aggregate_id=profile_id, user_id=user_id, payload={"profile_id": str(profile_id)})


@dataclass
class ProfileUpdatedEvent(DomainEvent):
    """Event raised when a personality profile is updated."""
    event_type: EventType = field(default=EventType.PROFILE_UPDATED)

    @classmethod
    def create(cls, profile_id: UUID, user_id: UUID, assessment_count: int, version: int) -> ProfileUpdatedEvent:
        return cls(aggregate_id=profile_id, user_id=user_id, payload={"assessment_count": assessment_count, "version": version})


@dataclass
class AssessmentCompletedEvent(DomainEvent):
    """Event raised when a personality assessment is completed."""
    event_type: EventType = field(default=EventType.ASSESSMENT_COMPLETED)
    aggregate_type: str = "trait_assessment"

    @classmethod
    def create(cls, assessment_id: UUID, user_id: UUID, source: AssessmentSource, confidence: float, processing_time_ms: float) -> AssessmentCompletedEvent:
        return cls(aggregate_id=assessment_id, user_id=user_id, payload={"source": source.value, "confidence": confidence, "processing_time_ms": processing_time_ms})


@dataclass
class TraitChangedEvent(DomainEvent):
    """Event raised when a personality trait changes significantly."""
    event_type: EventType = field(default=EventType.TRAIT_CHANGED)

    @classmethod
    def create(cls, profile_id: UUID, user_id: UUID, trait: PersonalityTrait, previous_value: float, new_value: float, change_magnitude: float) -> TraitChangedEvent:
        return cls(aggregate_id=profile_id, user_id=user_id, payload={"trait": trait.value, "previous_value": previous_value, "new_value": new_value, "change_magnitude": change_magnitude})


@dataclass
class StyleChangedEvent(DomainEvent):
    """Event raised when communication style type changes."""
    event_type: EventType = field(default=EventType.STYLE_CHANGED)

    @classmethod
    def create(cls, profile_id: UUID, user_id: UUID, previous_style: CommunicationStyleType, new_style: CommunicationStyleType) -> StyleChangedEvent:
        return cls(aggregate_id=profile_id, user_id=user_id, payload={"previous_style": previous_style.value, "new_style": new_style.value})


@dataclass
class StabilityAchievedEvent(DomainEvent):
    """Event raised when profile achieves stability."""
    event_type: EventType = field(default=EventType.STABILITY_ACHIEVED)

    @classmethod
    def create(cls, profile_id: UUID, user_id: UUID, stability_score: float, assessment_count: int) -> StabilityAchievedEvent:
        return cls(aggregate_id=profile_id, user_id=user_id, payload={"stability_score": stability_score, "assessment_count": assessment_count})


@dataclass
class SnapshotCapturedEvent(DomainEvent):
    """Event raised when a profile snapshot is captured."""
    event_type: EventType = field(default=EventType.SNAPSHOT_CAPTURED)
    aggregate_type: str = "profile_snapshot"

    @classmethod
    def create(cls, snapshot_id: UUID, profile_id: UUID, user_id: UUID, reason: str) -> SnapshotCapturedEvent:
        return cls(aggregate_id=snapshot_id, user_id=user_id, payload={"profile_id": str(profile_id), "reason": reason})


def to_kafka_event(event: DomainEvent) -> Any:
    """Convert local personality event to canonical Kafka event for inter-service messaging.

    Maps profile updates and trait changes to canonical schemas.
    Returns None for internal-only events or if solace_events is not available.
    """
    try:
        from src.solace_events.schemas import (
            EventMetadata as KafkaMetadata,
            PersonalityProfileUpdatedEvent as KafkaProfileUpdated,
            PersonalityTraitChangedEvent as KafkaTraitChanged,
        )
    except ImportError:
        logger.debug("solace_events_not_available_for_bridge")
        return None

    if not event.user_id:
        return None

    meta = KafkaMetadata(
        event_id=event.event_id, timestamp=event.timestamp,
        correlation_id=event.correlation_id or event.event_id,
        source_service="personality-service",
    )
    base: dict[str, Any] = {"user_id": event.user_id, "session_id": None, "metadata": meta}

    if event.event_type == EventType.PROFILE_UPDATED:
        return KafkaProfileUpdated(
            **base, profile_id=event.aggregate_id,
            assessment_count=event.payload.get("assessment_count", 0),
            profile_version=event.payload.get("version", 1),
        )

    if event.event_type == EventType.TRAIT_CHANGED:
        return KafkaTraitChanged(
            **base, profile_id=event.aggregate_id,
            trait_name=event.payload.get("trait", "unknown"),
            previous_value=Decimal(str(event.payload.get("previous_value", "0.5"))),
            new_value=Decimal(str(event.payload.get("new_value", "0.5"))),
            change_magnitude=Decimal(str(event.payload.get("change_magnitude", "0"))),
        )

    return None


EventHandler = Callable[[DomainEvent], Coroutine[Any, Any, None]]


class EventBus:
    """In-memory event bus for personality domain events."""

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
        """Publish multiple events in order."""
        for event in events:
            await self.publish(event)

    def get_published_events(self) -> list[DomainEvent]:
        """Get all published events (for testing/debugging)."""
        return self._published_events.copy()

    def get_events_by_type(self, event_type: EventType) -> list[DomainEvent]:
        """Get published events filtered by type."""
        return [e for e in self._published_events if e.event_type == event_type]

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
        logger.debug("event_stored", event_type=event.event_type.value, aggregate_id=str(event.aggregate_id))

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

    async def get_events_in_range(self, start: datetime, end: datetime) -> list[DomainEvent]:
        """Get events within a time range."""
        return [e for e in self._all_events if start <= e.timestamp <= end]

    async def count(self) -> int:
        """Count total events."""
        return len(self._all_events)

    async def count_by_aggregate(self, aggregate_id: UUID) -> int:
        """Count events for a specific aggregate."""
        return len(self._events.get(aggregate_id, []))


class EventPublisher:
    """Facade for publishing domain events with automatic store persistence."""

    def __init__(self, bus: EventBus, store: EventStore | None = None) -> None:
        self._bus = bus
        self._store = store

    async def publish(self, event: DomainEvent) -> None:
        """Publish event to bus and optionally store."""
        if self._store:
            await self._store.append(event)
        await self._bus.publish(event)

    async def publish_profile_created(self, profile_id: UUID, user_id: UUID) -> None:
        """Publish profile created event."""
        await self.publish(ProfileCreatedEvent.create(profile_id, user_id))

    async def publish_assessment_completed(self, assessment_id: UUID, user_id: UUID, source: AssessmentSource, confidence: float, processing_time_ms: float) -> None:
        """Publish assessment completed event."""
        await self.publish(AssessmentCompletedEvent.create(assessment_id, user_id, source, confidence, processing_time_ms))

    async def publish_trait_changed(self, profile_id: UUID, user_id: UUID, trait: PersonalityTrait, previous: float, new: float) -> None:
        """Publish trait changed event."""
        change = abs(new - previous)
        await self.publish(TraitChangedEvent.create(profile_id, user_id, trait, previous, new, change))

    async def publish_style_changed(self, profile_id: UUID, user_id: UUID, previous: CommunicationStyleType, new: CommunicationStyleType) -> None:
        """Publish style changed event."""
        await self.publish(StyleChangedEvent.create(profile_id, user_id, previous, new))

    async def publish_stability_achieved(self, profile_id: UUID, user_id: UUID, stability_score: float, assessment_count: int) -> None:
        """Publish stability achieved event."""
        await self.publish(StabilityAchievedEvent.create(profile_id, user_id, stability_score, assessment_count))
