"""
Solace-AI Common Library.

Enterprise-grade DDD primitives and shared utilities:
- Entity base classes with identity and lifecycle management
- Value objects with immutability and validation
- Aggregate roots with domain events and invariant protection
- Structured exception hierarchy with correlation tracking
- Utility functions for validation, datetime, crypto, and retry logic
"""

from .domain import (
    AggregateEvent,
    AggregateRepository,
    AggregateRoot,
    DomainEvent,
    Entity,
    EntityCreatedEvent,
    EntityDeletedEvent,
    EntityId,
    EntityMetadata,
    EntityUpdatedEvent,
    EventEnvelope,
    EventStore,
    InMemoryEventStore,
    MutableEntity,
    SnapshotStore,
)
from .exceptions import (
    ApplicationError,
    AuthenticationError,
    AuthorizationError,
    BusinessRuleViolationError,
    CacheError,
    ConcurrencyError,
    ConfigurationError,
    DatabaseError,
    DomainError,
    EntityConflictError,
    EntityNotFoundError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    ExternalServiceError,
    InfrastructureError,
    InvariantViolationError,
    LLMServiceError,
    RateLimitExceededError,
    SafetyError,
    SolaceError,
    ValidationError,
)
from .utils import (
    CollectionUtils,
    CryptoUtils,
    DateTimeUtils,
    RetryConfig,
    StringUtils,
    ValidationPatterns,
    ValidationUtils,
    retry_async,
)
from .domain import (
    AuditableMixin,
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
    TimestampedMixin,
    UserId,
    ValueObject,
)

__version__ = "0.1.0"

__all__ = [
    # Domain - Entity
    "Entity",
    "MutableEntity",
    "EntityId",
    "EntityMetadata",
    # Domain - Value Object
    "ValueObject",
    "SingleValueObject",
    "TimestampedMixin",
    "AuditableMixin",
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
    # Domain - Aggregate
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
    # Exceptions
    "SolaceError",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "DomainError",
    "ValidationError",
    "EntityNotFoundError",
    "EntityConflictError",
    "ConcurrencyError",
    "BusinessRuleViolationError",
    "InvariantViolationError",
    "ApplicationError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitExceededError",
    "SafetyError",
    "InfrastructureError",
    "DatabaseError",
    "CacheError",
    "ExternalServiceError",
    "LLMServiceError",
    "ConfigurationError",
    # Utils
    "ValidationUtils",
    "ValidationPatterns",
    "DateTimeUtils",
    "StringUtils",
    "CryptoUtils",
    "CollectionUtils",
    "RetryConfig",
    "retry_async",
]
