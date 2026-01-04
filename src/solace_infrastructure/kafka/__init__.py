"""
Solace-AI Kafka Infrastructure Library.

Enterprise-grade Kafka infrastructure management for Solace-AI microservices:
- Topic management with admin operations and validation
- Schema registry integration with compatibility enforcement
- Custom partitioning strategies for optimal message routing
- HIPAA-compliant retention policy management
- Comprehensive monitoring and alerting
"""

from .topics import (
    CleanupPolicy,
    CompressionCodec,
    TimestampType,
    TopicPriority,
    TopicDefinition,
    TopicAdminSettings,
    TopicMetadata,
    TopicOperationResult,
    TopicValidator,
    KafkaAdminAdapter,
    TopicManager,
    create_topic_manager,
)
from .schemas import (
    SchemaType,
    CompatibilityLevel,
    SchemaFormat,
    SchemaVersion,
    SchemaMetadata,
    CompatibilityResult,
    SchemaRegistrySettings,
    SchemaDefinition,
    SchemaValidator,
    JsonSchemaValidator,
    AvroSchemaValidator,
    SchemaCache,
    SchemaRegistryAdapter,
    SchemaManager,
    create_schema_manager,
)
from .partitioning import (
    PartitionStrategy,
    PartitionResult,
    PartitionerConfig,
    Partitioner,
    RoundRobinPartitioner,
    HashKeyPartitioner,
    UserAffinityPartitioner,
    PriorityPartitioner,
    StickyPartitioner,
    PartitionerFactory,
    TopicPartitionRouter,
    create_partition_router,
)
from .retention import (
    RetentionType,
    ComplianceCategory,
    RetentionPriority,
    RetentionMetrics,
    RetentionPolicy,
    TopicRetentionAssignment,
    RetentionManager,
    PRESET_POLICIES,
    create_retention_manager,
    get_hipaa_policy,
)
from .monitoring import (
    HealthStatus,
    AlertSeverity,
    BrokerMetrics,
    TopicMetrics,
    ConsumerGroupMetrics,
    ConsumerLag,
    KafkaAlert,
    ClusterHealth,
    MonitoringSettings,
    MetricsCollector,
    KafkaMonitorAdapter,
    KafkaMonitor,
    create_kafka_monitor,
)

__all__ = [
    # Topics
    "CleanupPolicy",
    "CompressionCodec",
    "TimestampType",
    "TopicPriority",
    "TopicDefinition",
    "TopicAdminSettings",
    "TopicMetadata",
    "TopicOperationResult",
    "TopicValidator",
    "KafkaAdminAdapter",
    "TopicManager",
    "create_topic_manager",
    # Schemas
    "SchemaType",
    "CompatibilityLevel",
    "SchemaFormat",
    "SchemaVersion",
    "SchemaMetadata",
    "CompatibilityResult",
    "SchemaRegistrySettings",
    "SchemaDefinition",
    "SchemaValidator",
    "JsonSchemaValidator",
    "AvroSchemaValidator",
    "SchemaCache",
    "SchemaRegistryAdapter",
    "SchemaManager",
    "create_schema_manager",
    # Partitioning
    "PartitionStrategy",
    "PartitionResult",
    "PartitionerConfig",
    "Partitioner",
    "RoundRobinPartitioner",
    "HashKeyPartitioner",
    "UserAffinityPartitioner",
    "PriorityPartitioner",
    "StickyPartitioner",
    "PartitionerFactory",
    "TopicPartitionRouter",
    "create_partition_router",
    # Retention
    "RetentionType",
    "ComplianceCategory",
    "RetentionPriority",
    "RetentionMetrics",
    "RetentionPolicy",
    "TopicRetentionAssignment",
    "RetentionManager",
    "PRESET_POLICIES",
    "create_retention_manager",
    "get_hipaa_policy",
    # Monitoring
    "HealthStatus",
    "AlertSeverity",
    "BrokerMetrics",
    "TopicMetrics",
    "ConsumerGroupMetrics",
    "ConsumerLag",
    "KafkaAlert",
    "ClusterHealth",
    "MonitoringSettings",
    "MetricsCollector",
    "KafkaMonitorAdapter",
    "KafkaMonitor",
    "create_kafka_monitor",
]

__version__ = "1.0.0"
