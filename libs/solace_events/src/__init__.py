"""
Solace-AI Events Library.

Enterprise-grade event infrastructure for Solace-AI microservices:
- Kafka event publishing with transactional outbox pattern
- Consumer group management with offset tracking
- Dead letter queue handling with retry policies
- Comprehensive event schemas for all domains
"""

from .config import (
    CompressionType,
    ConsumerGroup,
    ConsumerSettings,
    KafkaSettings,
    ProducerSettings,
    SaslMechanism,
    SecurityProtocol,
    SolaceTopic,
    TopicConfig,
    TOPIC_CONFIGS,
    get_all_topics,
    get_dlq_topics,
    get_topic_config,
)
from .consumer import (
    AIOKafkaConsumerAdapter,
    ConsumerMetrics,
    EventConsumer,
    KafkaConsumerAdapter,
    MockKafkaConsumerAdapter,
    OffsetTracker,
    ProcessingResult,
    ProcessingStatus,
    create_consumer,
)
from .dead_letter import (
    DeadLetterHandler,
    DeadLetterRecord,
    DeadLetterStore,
    RetryPolicy,
    RetryStrategy,
    create_dead_letter_handler,
    get_dlq_topic,
)
from .publisher import (
    AIOKafkaProducerAdapter,
    EventPublisher,
    InMemoryOutboxStore,
    KafkaProducerAdapter,
    MockKafkaProducerAdapter,
    OutboxPoller,
    OutboxRecord,
    OutboxStatus,
    OutboxStore,
    create_publisher,
)
from .schemas import (
    AssessmentCompletedEvent,
    BaseEvent,
    ClinicalHypothesis,
    Confidence,
    CrisisDetectedEvent,
    CrisisLevel,
    DiagnosisCompletedEvent,
    ErrorOccurredEvent,
    EscalationTriggeredEvent,
    EVENT_REGISTRY,
    EventMetadata,
    InterventionDeliveredEvent,
    MemoryConsolidatedEvent,
    MemoryRetrievedEvent,
    MemoryStoredEvent,
    MemoryTier,
    MessageReceivedEvent,
    OceanScores,
    PersonalityAssessedEvent,
    ProgressMilestoneEvent,
    ResponseGeneratedEvent,
    RetentionCategory,
    RiskFactor,
    SafetyAssessmentEvent,
    SessionEndedEvent,
    SessionStartedEvent,
    StyleGeneratedEvent,
    SystemHealthEvent,
    TherapyModality,
    TherapySessionStartedEvent,
    deserialize_event,
    get_topic_for_event,
)

__all__ = [
    # Config
    "CompressionType",
    "ConsumerGroup",
    "ConsumerSettings",
    "KafkaSettings",
    "ProducerSettings",
    "SaslMechanism",
    "SecurityProtocol",
    "SolaceTopic",
    "TopicConfig",
    "TOPIC_CONFIGS",
    "get_all_topics",
    "get_dlq_topics",
    "get_topic_config",
    # Publisher
    "AIOKafkaProducerAdapter",
    "EventPublisher",
    "InMemoryOutboxStore",
    "KafkaProducerAdapter",
    "MockKafkaProducerAdapter",
    "OutboxPoller",
    "OutboxRecord",
    "OutboxStatus",
    "OutboxStore",
    "create_publisher",
    # Consumer
    "AIOKafkaConsumerAdapter",
    "ConsumerMetrics",
    "EventConsumer",
    "KafkaConsumerAdapter",
    "MockKafkaConsumerAdapter",
    "OffsetTracker",
    "ProcessingResult",
    "ProcessingStatus",
    "create_consumer",
    # Dead Letter
    "DeadLetterHandler",
    "DeadLetterRecord",
    "DeadLetterStore",
    "RetryPolicy",
    "RetryStrategy",
    "create_dead_letter_handler",
    "get_dlq_topic",
    # Schemas - Base
    "BaseEvent",
    "EventMetadata",
    "EVENT_REGISTRY",
    "deserialize_event",
    "get_topic_for_event",
    # Schemas - Session
    "SessionStartedEvent",
    "SessionEndedEvent",
    "MessageReceivedEvent",
    "ResponseGeneratedEvent",
    # Schemas - Safety
    "CrisisLevel",
    "RiskFactor",
    "SafetyAssessmentEvent",
    "CrisisDetectedEvent",
    "EscalationTriggeredEvent",
    # Schemas - Diagnosis
    "Confidence",
    "ClinicalHypothesis",
    "DiagnosisCompletedEvent",
    "AssessmentCompletedEvent",
    # Schemas - Therapy
    "TherapyModality",
    "TherapySessionStartedEvent",
    "InterventionDeliveredEvent",
    "ProgressMilestoneEvent",
    # Schemas - Memory
    "MemoryTier",
    "RetentionCategory",
    "MemoryStoredEvent",
    "MemoryConsolidatedEvent",
    "MemoryRetrievedEvent",
    # Schemas - Personality
    "OceanScores",
    "PersonalityAssessedEvent",
    "StyleGeneratedEvent",
    # Schemas - System
    "SystemHealthEvent",
    "ErrorOccurredEvent",
]
