"""
Solace-AI Aggregate Root Implementation.

Provides aggregate roots with domain event support for:
- Transactional consistency boundaries
- Domain event collection and dispatch
- Invariant enforcement
- Child entity management
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

import structlog

from .entity import Entity, EntityId, EntityMetadata, MutableEntity

logger = structlog.get_logger(__name__)


class DomainEvent(BaseModel):
    """
    Base class for all domain events.

    Domain events represent significant occurrences within the domain
    that other parts of the system may need to react to.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., description="Event type identifier")
    aggregate_id: str = Field(..., description="ID of aggregate that raised event")
    aggregate_type: str = Field(..., description="Type of aggregate")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    correlation_id: str | None = Field(
        default=None, description="Correlation ID for tracing"
    )
    causation_id: str | None = Field(
        default=None, description="ID of event that caused this event"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, description="Event schema version")

    model_config = ConfigDict(frozen=True, extra="forbid")

    def with_correlation(
        self, correlation_id: str, causation_id: str | None = None
    ) -> DomainEvent:
        """Create copy with correlation context."""
        data = self.model_dump()
        data["correlation_id"] = correlation_id
        data["causation_id"] = causation_id
        return type(self)(**data)


TId = TypeVar("TId", bound=EntityId)


class AggregateRoot(MutableEntity[TId], ABC, Generic[TId]):
    """
    Base class for aggregate roots.

    Aggregate roots are the primary entry points for domain operations.
    They:
    - Define consistency boundaries
    - Collect domain events for later dispatch
    - Enforce aggregate invariants
    - Manage child entities within the aggregate

    Domain events are collected internally and should be retrieved
    and cleared after the aggregate is persisted.
    """

    _pending_events: list[DomainEvent] = PrivateAttr(default_factory=list)
    _entity_type: ClassVar[str] = "AggregateRoot"

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
    )

    def _raise_event(self, event: DomainEvent) -> None:
        """
        Record a domain event to be dispatched after persistence.

        Events are not dispatched immediately to ensure they are only
        published after the aggregate state is successfully persisted.
        """
        self._pending_events.append(event)
        logger.debug(
            "Domain event raised",
            event_type=event.event_type,
            aggregate_id=str(self.id),
            aggregate_type=self.get_entity_type(),
        )

    def collect_events(self) -> list[DomainEvent]:
        """
        Retrieve and clear all pending domain events.

        Should be called after successful persistence to get events
        for publishing to the event bus.
        """
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    def has_pending_events(self) -> bool:
        """Check if there are pending events to dispatch."""
        return len(self._pending_events) > 0

    @abstractmethod
    def validate_invariants(self) -> None:
        """
        Validate all aggregate invariants.

        Must be implemented by subclasses to enforce business rules.
        Should raise InvariantViolationError if any invariant is violated.
        """
        pass


class AggregateEvent(DomainEvent):
    """
    Base class for aggregate-specific events.

    Provides factory method for creating events with aggregate context.
    """

    @classmethod
    def create(
        cls,
        aggregate: AggregateRoot[Any],
        event_type: str | None = None,
        **kwargs: Any,
    ) -> AggregateEvent:
        """Create event with aggregate context."""
        return cls(
            event_type=event_type or cls.__name__,
            aggregate_id=str(aggregate.id),
            aggregate_type=aggregate.get_entity_type(),
            **kwargs,
        )


class EntityCreatedEvent(AggregateEvent):
    """Event raised when an entity is created."""

    entity_data: dict[str, Any] = Field(default_factory=dict)


class EntityUpdatedEvent(AggregateEvent):
    """Event raised when an entity is updated."""

    changed_fields: list[str] = Field(default_factory=list)
    previous_values: dict[str, Any] = Field(default_factory=dict)
    new_values: dict[str, Any] = Field(default_factory=dict)


class EntityDeletedEvent(AggregateEvent):
    """Event raised when an entity is deleted."""

    reason: str | None = None


class EventEnvelope(BaseModel):
    """
    Wrapper for domain events during transport.

    Provides additional metadata for event routing and delivery.
    """

    event: DomainEvent
    destination: str | None = Field(
        default=None, description="Target topic or queue"
    )
    partition_key: str | None = Field(
        default=None, description="Key for ordered delivery"
    )
    headers: dict[str, str] = Field(default_factory=dict)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)

    model_config = ConfigDict(frozen=True)

    @classmethod
    def wrap(
        cls,
        event: DomainEvent,
        destination: str | None = None,
    ) -> EventEnvelope:
        """Create envelope for event."""
        return cls(
            event=event,
            destination=destination,
            partition_key=event.aggregate_id,
            headers={
                "event_type": event.event_type,
                "aggregate_type": event.aggregate_type,
            },
        )


class EventStore(ABC):
    """
    Abstract interface for event storage.

    Defines contract for persisting and retrieving domain events.
    """

    @abstractmethod
    async def append(
        self,
        aggregate_id: str,
        events: list[DomainEvent],
        expected_version: int | None = None,
    ) -> None:
        """
        Append events to the event stream for an aggregate.

        Args:
            aggregate_id: ID of the aggregate
            events: Events to append
            expected_version: Expected current version for optimistic concurrency
        """
        pass

    @abstractmethod
    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
    ) -> list[DomainEvent]:
        """
        Retrieve events for an aggregate from a specific version.

        Args:
            aggregate_id: ID of the aggregate
            from_version: Starting version number (inclusive)

        Returns:
            List of events in order
        """
        pass

    @abstractmethod
    async def get_all_events(
        self,
        from_position: int = 0,
        batch_size: int = 100,
    ) -> list[DomainEvent]:
        """
        Retrieve all events across aggregates.

        Used for projections and read model updates.
        """
        pass


class SnapshotStore(ABC):
    """
    Abstract interface for aggregate snapshots.

    Snapshots are used to optimize event sourcing by storing
    periodic state snapshots to avoid replaying all events.
    """

    @abstractmethod
    async def save_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str,
        state: dict[str, Any],
        version: int,
    ) -> None:
        """Save aggregate state snapshot."""
        pass

    @abstractmethod
    async def get_snapshot(
        self,
        aggregate_id: str,
    ) -> tuple[dict[str, Any], int] | None:
        """
        Get latest snapshot for aggregate.

        Returns:
            Tuple of (state dict, version) or None if no snapshot exists
        """
        pass


class AggregateRepository(ABC, Generic[TId]):
    """
    Abstract repository for aggregate persistence.

    Defines contract for loading and saving aggregates with
    event collection support.
    """

    @abstractmethod
    async def get_by_id(self, aggregate_id: TId) -> AggregateRoot[TId] | None:
        """Load aggregate by ID."""
        pass

    @abstractmethod
    async def save(self, aggregate: AggregateRoot[TId]) -> None:
        """
        Persist aggregate and collect domain events.

        Should handle optimistic concurrency and event publishing.
        """
        pass

    @abstractmethod
    async def delete(self, aggregate: AggregateRoot[TId]) -> None:
        """Remove aggregate."""
        pass


class InMemoryEventStore(EventStore):
    """
    In-memory event store for testing.

    Not for production use - events are lost on restart.
    """

    def __init__(self) -> None:
        self._streams: dict[str, list[DomainEvent]] = {}
        self._all_events: list[DomainEvent] = []

    async def append(
        self,
        aggregate_id: str,
        events: list[DomainEvent],
        expected_version: int | None = None,
    ) -> None:
        """Append events to aggregate stream."""
        from ..exceptions import ConcurrencyError

        stream = self._streams.setdefault(aggregate_id, [])

        if expected_version is not None:
            current_version = len(stream)
            if current_version != expected_version:
                raise ConcurrencyError(
                    entity_type="EventStream",
                    entity_id=aggregate_id,
                    expected_version=expected_version,
                    actual_version=current_version,
                )

        stream.extend(events)
        self._all_events.extend(events)

    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
    ) -> list[DomainEvent]:
        """Get events for aggregate from version."""
        stream = self._streams.get(aggregate_id, [])
        return stream[from_version:]

    async def get_all_events(
        self,
        from_position: int = 0,
        batch_size: int = 100,
    ) -> list[DomainEvent]:
        """Get all events across streams."""
        return self._all_events[from_position : from_position + batch_size]

    def clear(self) -> None:
        """Clear all stored events (for testing)."""
        self._streams.clear()
        self._all_events.clear()
