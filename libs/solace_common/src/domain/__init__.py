"""
Solace-AI Domain Module.

Provides Domain-Driven Design building blocks:
- Entity: Identity-based domain objects
- ValueObject: Immutable value objects
- AggregateRoot: Transactional consistency boundaries with events
"""

from .aggregate import (
    AggregateEvent,
    AggregateRepository,
    AggregateRoot,
    DomainEvent,
    EntityCreatedEvent,
    EntityDeletedEvent,
    EntityUpdatedEvent,
    EventEnvelope,
    EventStore,
    InMemoryEventStore,
    SnapshotStore,
)
from .entity import (
    AuditableMixin,
    Entity,
    EntityId,
    EntityMetadata,
    MutableEntity,
    TimestampedMixin,
)
from .value_object import (
    CorrelationId,
    DateRange,
    EmailAddress,
    HashedValue,
    Percentage,
    PhoneNumber,
    Score,
    SessionId,
    Severity,
    SeverityScore,
    SingleValueObject,
    UserId,
    ValueObject,
)

__all__ = [
    # Entity
    "Entity",
    "EntityId",
    "EntityMetadata",
    "MutableEntity",
    "TimestampedMixin",
    "AuditableMixin",
    # Value Objects
    "ValueObject",
    "SingleValueObject",
    "EmailAddress",
    "PhoneNumber",
    "Percentage",
    "Score",
    "DateRange",
    "Severity",
    "SeverityScore",
    "HashedValue",
    "UserId",
    "SessionId",
    "CorrelationId",
    # Aggregate
    "AggregateRoot",
    "DomainEvent",
    "AggregateEvent",
    "EntityCreatedEvent",
    "EntityUpdatedEvent",
    "EntityDeletedEvent",
    "EventEnvelope",
    "EventStore",
    "SnapshotStore",
    "AggregateRepository",
    "InMemoryEventStore",
]
