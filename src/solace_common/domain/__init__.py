"""Solace-AI Domain Primitives."""

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

__all__ = [
    # Entity
    "Entity",
    "MutableEntity",
    "EntityId",
    "EntityMetadata",
    "TimestampedMixin",
    "AuditableMixin",
    # Value Object
    "ValueObject",
    "SingleValueObject",
    "UserId",
    "SessionId",
    "CorrelationId",
    "EmailAddress",
    "PhoneNumber",
    "Percentage",
    "Score",
    "Severity",
    "SeverityScore",
    "DateRange",
    "HashedValue",
    # Aggregate
    "AggregateRoot",
    "DomainEvent",
    "AggregateEvent",
    "EntityCreatedEvent",
    "EntityUpdatedEvent",
    "EntityDeletedEvent",
    "EventEnvelope",
    "AggregateRepository",
    "EventStore",
    "InMemoryEventStore",
    "SnapshotStore",
]
