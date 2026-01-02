"""
Solace-AI Common Library.

Provides shared domain primitives, exceptions, and utilities used across
all Solace-AI microservices.

Modules:
    domain: Entity, ValueObject, AggregateRoot base classes
    exceptions: Structured exception hierarchy
    utils: Common utilities (datetime, crypto, validation)
"""

from .domain.aggregate import (
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
from .domain.entity import (
    AuditableMixin,
    Entity,
    EntityId,
    EntityMetadata,
    MutableEntity,
    TimestampedMixin,
)
from .domain.value_object import (
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

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Domain - Entity
    "Entity",
    "EntityId",
    "EntityMetadata",
    "MutableEntity",
    "TimestampedMixin",
    "AuditableMixin",
    # Domain - Value Objects
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
    # Domain - Aggregate
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
    "DateTimeUtils",
    "CryptoUtils",
    "ValidationUtils",
    "ValidationPatterns",
    "StringUtils",
    "RetryConfig",
    "retry_async",
    "CollectionUtils",
]
