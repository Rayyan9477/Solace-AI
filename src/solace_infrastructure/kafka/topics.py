"""
Solace-AI Kafka Topic Management - Admin operations for topic lifecycle.
Enterprise-grade topic creation, configuration, and validation.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class CleanupPolicy(str, Enum):
    """Kafka topic cleanup policies."""
    DELETE = "delete"
    COMPACT = "compact"
    COMPACT_DELETE = "compact,delete"


class CompressionCodec(str, Enum):
    """Message compression codecs."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"
    PRODUCER = "producer"


class TimestampType(str, Enum):
    """Message timestamp types."""
    CREATE_TIME = "CreateTime"
    LOG_APPEND_TIME = "LogAppendTime"


class TopicPriority(str, Enum):
    """Topic priority levels for resource allocation."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TopicDefinition(BaseModel):
    """Complete topic configuration definition."""

    name: str = Field(..., min_length=1, max_length=249)
    partitions: int = Field(default=4, ge=1, le=256)
    replication_factor: int = Field(default=3, ge=1, le=5)
    cleanup_policy: CleanupPolicy = Field(default=CleanupPolicy.DELETE)
    retention_ms: int = Field(default=604800000, ge=-1)  # 7 days, -1 for infinite
    retention_bytes: int = Field(default=-1, ge=-1)  # -1 for unlimited
    segment_ms: int = Field(default=86400000, ge=1000)  # 1 day
    segment_bytes: int = Field(default=1073741824, ge=1024)  # 1GB
    min_insync_replicas: int = Field(default=2, ge=1)
    compression_type: CompressionCodec = Field(default=CompressionCodec.PRODUCER)
    max_message_bytes: int = Field(default=1048576, ge=1024)  # 1MB
    message_timestamp_type: TimestampType = Field(default=TimestampType.CREATE_TIME)
    min_compaction_lag_ms: int = Field(default=0, ge=0)
    max_compaction_lag_ms: int = Field(default=9223372036854775807, ge=0)
    unclean_leader_election: bool = Field(default=False)
    priority: TopicPriority = Field(default=TopicPriority.NORMAL)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_topic_name(cls, v: str) -> str:
        """Validate and normalize topic name to Kafka conventions."""
        v = v.lower()  # Normalize to lowercase for consistency
        if not v.replace(".", "").replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid topic name: {v}")
        if v.startswith("__"):
            raise ValueError("Topic name cannot start with '__' (reserved)")
        return v

    def to_kafka_config(self) -> dict[str, str]:
        """Convert to Kafka AdminClient config format."""
        return {
            "cleanup.policy": self.cleanup_policy.value,
            "retention.ms": str(self.retention_ms),
            "retention.bytes": str(self.retention_bytes),
            "segment.ms": str(self.segment_ms),
            "segment.bytes": str(self.segment_bytes),
            "min.insync.replicas": str(self.min_insync_replicas),
            "compression.type": self.compression_type.value,
            "max.message.bytes": str(self.max_message_bytes),
            "message.timestamp.type": self.message_timestamp_type.value,
            "min.compaction.lag.ms": str(self.min_compaction_lag_ms),
            "max.compaction.lag.ms": str(self.max_compaction_lag_ms),
            "unclean.leader.election.enable": str(self.unclean_leader_election).lower(),
        }


class TopicAdminSettings(BaseSettings):
    """Settings for Kafka admin operations."""

    bootstrap_servers: str = Field(default="localhost:9092")
    admin_client_id: str = Field(default="solace-admin")
    request_timeout_ms: int = Field(default=30000, ge=1000)
    operation_timeout_ms: int = Field(default=60000, ge=1000)
    connections_max_idle_ms: int = Field(default=540000, ge=1000)
    default_replication_factor: int = Field(default=3, ge=1)
    validate_only: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_prefix="KAFKA_ADMIN_",
        env_file=".env",
        extra="ignore",
    )


@dataclass
class TopicMetadata:
    """Metadata about an existing topic."""
    name: str
    partitions: int
    replication_factor: int
    config: dict[str, str] = field(default_factory=dict)
    partition_info: list[dict[str, Any]] = field(default_factory=list)
    is_internal: bool = False


@dataclass
class TopicOperationResult:
    """Result of a topic operation."""
    success: bool
    topic_name: str
    operation: str
    message: str
    error_code: str | None = None


class TopicValidator:
    """Validates topic configurations and naming conventions."""

    RESERVED_PREFIXES = ("__", "_confluent", "_schemas")
    MAX_NAME_LENGTH = 249
    VALID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")

    @classmethod
    def validate_name(cls, name: str) -> tuple[bool, str | None]:
        """Validate topic name against Kafka conventions."""
        if not name:
            return False, "Topic name cannot be empty"
        if len(name) > cls.MAX_NAME_LENGTH:
            return False, f"Topic name exceeds {cls.MAX_NAME_LENGTH} characters"
        if any(name.startswith(prefix) for prefix in cls.RESERVED_PREFIXES):
            return False, f"Topic name uses reserved prefix"
        if not all(c in cls.VALID_CHARS for c in name):
            return False, "Topic name contains invalid characters"
        if ".." in name:
            return False, "Topic name cannot contain consecutive dots"
        return True, None

    @classmethod
    def validate_config(cls, definition: TopicDefinition) -> list[str]:
        """Validate topic configuration for best practices."""
        warnings: list[str] = []
        if definition.replication_factor < 3:
            warnings.append("Replication factor < 3 risks data loss")
        if definition.min_insync_replicas >= definition.replication_factor:
            warnings.append("min.insync.replicas should be < replication.factor")
        if definition.retention_ms == -1 and definition.retention_bytes == -1:
            warnings.append("Infinite retention may cause storage issues")
        if definition.partitions > 50:
            warnings.append("High partition count increases broker overhead")
        if definition.unclean_leader_election:
            warnings.append("Unclean leader election enabled - potential data loss")
        return warnings


class KafkaAdminAdapter:
    """Adapter for Kafka AdminClient operations."""

    def __init__(self, settings: TopicAdminSettings) -> None:
        self._settings = settings
        self._client: Any = None

    async def connect(self) -> None:
        """Initialize connection to Kafka cluster."""
        try:
            from aiokafka.admin import AIOKafkaAdminClient
            self._client = AIOKafkaAdminClient(
                bootstrap_servers=self._settings.bootstrap_servers,
                client_id=self._settings.admin_client_id,
                request_timeout_ms=self._settings.request_timeout_ms,
            )
            await self._client.start()
            logger.info("kafka_admin_connected", servers=self._settings.bootstrap_servers)
        except ImportError:
            logger.warning("aiokafka_not_available", fallback="mock_mode")
            self._client = None
        except Exception as e:
            logger.error("kafka_admin_connection_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close admin client connection."""
        if self._client:
            await self._client.close()
            logger.info("kafka_admin_disconnected")

    async def create_topic(self, definition: TopicDefinition) -> TopicOperationResult:
        """Create a new Kafka topic."""
        if not self._client:
            logger.debug("mock_create_topic", topic=definition.name)
            return TopicOperationResult(True, definition.name, "create", "Mock created")
        try:
            from aiokafka.admin import NewTopic
            new_topic = NewTopic(
                name=definition.name,
                num_partitions=definition.partitions,
                replication_factor=definition.replication_factor,
                topic_configs=definition.to_kafka_config(),
            )
            await self._client.create_topics([new_topic])
            logger.info("topic_created", topic=definition.name, partitions=definition.partitions)
            return TopicOperationResult(True, definition.name, "create", "Topic created")
        except Exception as e:
            error_msg = str(e)
            logger.error("topic_creation_failed", topic=definition.name, error=error_msg)
            return TopicOperationResult(False, definition.name, "create", error_msg, "CREATE_FAILED")

    async def delete_topic(self, topic_name: str) -> TopicOperationResult:
        """Delete a Kafka topic."""
        if not self._client:
            return TopicOperationResult(True, topic_name, "delete", "Mock deleted")
        try:
            await self._client.delete_topics([topic_name])
            logger.info("topic_deleted", topic=topic_name)
            return TopicOperationResult(True, topic_name, "delete", "Topic deleted")
        except Exception as e:
            logger.error("topic_deletion_failed", topic=topic_name, error=str(e))
            return TopicOperationResult(False, topic_name, "delete", str(e), "DELETE_FAILED")

    async def describe_topic(self, topic_name: str) -> TopicMetadata | None:
        """Get metadata for a topic."""
        if not self._client:
            return TopicMetadata(name=topic_name, partitions=4, replication_factor=3)
        try:
            metadata = await self._client.describe_topics([topic_name])
            if not metadata:
                return None
            topic_meta = metadata[0]
            partitions = [
                {"partition": p.partition, "leader": p.leader, "replicas": list(p.replicas)}
                for p in topic_meta.partitions
            ]
            return TopicMetadata(
                name=topic_meta.topic,
                partitions=len(topic_meta.partitions),
                replication_factor=len(topic_meta.partitions[0].replicas) if topic_meta.partitions else 0,
                partition_info=partitions,
                is_internal=topic_meta.is_internal,
            )
        except Exception as e:
            logger.error("topic_describe_failed", topic=topic_name, error=str(e))
            return None

    async def list_topics(self) -> list[str]:
        """List all topics in the cluster."""
        if not self._client:
            return ["solace.sessions", "solace.safety", "solace.assessments"]
        try:
            topics = await self._client.list_topics()
            return [t for t in topics if not t.startswith("__")]
        except Exception as e:
            logger.error("topic_list_failed", error=str(e))
            return []

    async def alter_configs(self, topic_name: str, configs: dict[str, str]) -> TopicOperationResult:
        """Alter topic configuration."""
        if not self._client:
            return TopicOperationResult(True, topic_name, "alter", "Mock altered")
        try:
            from aiokafka.admin import ConfigResource, ConfigResourceType
            resource = ConfigResource(ConfigResourceType.TOPIC, topic_name)
            await self._client.alter_configs({resource: configs})
            logger.info("topic_config_altered", topic=topic_name, config_keys=list(configs.keys()))
            return TopicOperationResult(True, topic_name, "alter", "Config updated")
        except Exception as e:
            logger.error("topic_alter_failed", topic=topic_name, error=str(e))
            return TopicOperationResult(False, topic_name, "alter", str(e), "ALTER_FAILED")


class TopicManager:
    """High-level topic management with validation and policies."""

    def __init__(self, settings: TopicAdminSettings | None = None) -> None:
        self._settings = settings or TopicAdminSettings()
        self._adapter = KafkaAdminAdapter(self._settings)
        self._validator = TopicValidator()
        self._topic_registry: dict[str, TopicDefinition] = {}

    async def connect(self) -> None:
        """Establish connection to Kafka cluster."""
        await self._adapter.connect()

    async def close(self) -> None:
        """Close connection."""
        await self._adapter.close()

    def register_topic(self, definition: TopicDefinition) -> None:
        """Register a topic definition for management."""
        self._topic_registry[definition.name] = definition
        logger.debug("topic_registered", topic=definition.name)

    def register_solace_topics(self) -> None:
        """Register all standard Solace-AI topics."""
        topics = [
            TopicDefinition(name="solace.sessions", partitions=4, priority=TopicPriority.HIGH),
            TopicDefinition(name="solace.assessments", partitions=4, priority=TopicPriority.HIGH),
            TopicDefinition(name="solace.therapy", partitions=2, priority=TopicPriority.NORMAL),
            TopicDefinition(name="solace.safety", partitions=4, priority=TopicPriority.CRITICAL,
                          retention_ms=31536000000),  # 1 year for safety events
            TopicDefinition(name="solace.memory", partitions=4, priority=TopicPriority.NORMAL),
            TopicDefinition(name="solace.analytics", partitions=2, priority=TopicPriority.LOW),
            TopicDefinition(name="solace.personality", partitions=2, priority=TopicPriority.NORMAL),
        ]
        for topic in topics:
            self.register_topic(topic)
            self.register_topic(TopicDefinition(
                name=f"{topic.name}.dlq", partitions=1, priority=TopicPriority.NORMAL,
                retention_ms=604800000, tags={"type": "dlq", "parent": topic.name}
            ))

    async def create_topic(self, definition: TopicDefinition) -> TopicOperationResult:
        """Create topic with validation."""
        valid, error = self._validator.validate_name(definition.name)
        if not valid:
            return TopicOperationResult(False, definition.name, "create", error or "Invalid", "VALIDATION_FAILED")
        warnings = self._validator.validate_config(definition)
        for warning in warnings:
            logger.warning("topic_config_warning", topic=definition.name, warning=warning)
        result = await self._adapter.create_topic(definition)
        if result.success:
            self._topic_registry[definition.name] = definition
        return result

    async def create_registered_topics(self) -> list[TopicOperationResult]:
        """Create all registered topics that don't exist."""
        existing = set(await self._adapter.list_topics())
        results: list[TopicOperationResult] = []
        for name, definition in self._topic_registry.items():
            if name not in existing:
                result = await self._adapter.create_topic(definition)
                results.append(result)
        return results

    async def ensure_topic(self, definition: TopicDefinition) -> TopicOperationResult:
        """Ensure topic exists, creating if necessary."""
        metadata = await self._adapter.describe_topic(definition.name)
        if metadata:
            logger.debug("topic_exists", topic=definition.name)
            return TopicOperationResult(True, definition.name, "ensure", "Already exists")
        return await self.create_topic(definition)

    async def delete_topic(self, topic_name: str) -> TopicOperationResult:
        """Delete topic and unregister."""
        result = await self._adapter.delete_topic(topic_name)
        if result.success:
            self._topic_registry.pop(topic_name, None)
        return result

    async def get_topic_info(self, topic_name: str) -> TopicMetadata | None:
        """Get topic metadata."""
        return await self._adapter.describe_topic(topic_name)

    async def list_all_topics(self) -> list[str]:
        """List all topics."""
        return await self._adapter.list_topics()

    def get_registered_topics(self) -> dict[str, TopicDefinition]:
        """Get all registered topic definitions."""
        return dict(self._topic_registry)


async def create_topic_manager(settings: TopicAdminSettings | None = None) -> TopicManager:
    """Factory function to create and connect topic manager."""
    manager = TopicManager(settings)
    await manager.connect()
    return manager
