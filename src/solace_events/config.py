"""Solace-AI Events Configuration - Kafka settings and topic management."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class SecurityProtocol(str, Enum):
    """Kafka security protocols."""
    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


class SaslMechanism(str, Enum):
    """SASL authentication mechanisms."""
    PLAIN = "PLAIN"
    SCRAM_SHA_256 = "SCRAM-SHA-256"
    SCRAM_SHA_512 = "SCRAM-SHA-512"


class CompressionType(str, Enum):
    """Kafka message compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class TopicConfig(BaseModel):
    """Configuration for a Kafka topic."""

    name: str = Field(..., min_length=1, max_length=249)
    partitions: int = Field(default=4, ge=1, le=256)
    replication_factor: int = Field(default=3, ge=1, le=5)
    retention_ms: int = Field(default=7776000000, ge=0)  # 90 days
    cleanup_policy: str = Field(default="delete")
    min_insync_replicas: int = Field(default=2, ge=1)

    model_config = ConfigDict(frozen=True)

    @field_validator("name")
    @classmethod
    def validate_topic_name(cls, v: str) -> str:
        """Validate topic name follows Kafka conventions."""
        if not v.replace(".", "").replace("-", "").replace("_", "").isalnum():
            raise ValueError("Topic name contains invalid characters")
        return v.lower()


class SolaceTopic(str, Enum):
    """Predefined Solace-AI Kafka topics."""
    SESSIONS = "solace.sessions"
    ASSESSMENTS = "solace.assessments"
    THERAPY = "solace.therapy"
    SAFETY = "solace.safety"
    MEMORY = "solace.memory"
    ANALYTICS = "solace.analytics"
    PERSONALITY = "solace.personality"

    @property
    def dlq_topic(self) -> str:
        """Get dead letter queue topic name."""
        return f"{self.value}.dlq"

    @classmethod
    def from_string(cls, value: str) -> SolaceTopic:
        """Create topic from string value."""
        for topic in cls:
            if topic.value == value:
                return topic
        raise ValueError(f"Unknown topic: {value}")


TOPIC_CONFIGS: dict[SolaceTopic, TopicConfig] = {
    SolaceTopic.SESSIONS: TopicConfig(name="solace.sessions", partitions=4),
    SolaceTopic.ASSESSMENTS: TopicConfig(name="solace.assessments", partitions=4),
    SolaceTopic.THERAPY: TopicConfig(name="solace.therapy", partitions=2),
    SolaceTopic.SAFETY: TopicConfig(name="solace.safety", partitions=4),
    SolaceTopic.MEMORY: TopicConfig(name="solace.memory", partitions=4),
    SolaceTopic.ANALYTICS: TopicConfig(name="solace.analytics", partitions=2),
    SolaceTopic.PERSONALITY: TopicConfig(name="solace.personality", partitions=2),
}


class KafkaSettings(BaseSettings):
    """Kafka connection and behavior settings loaded from environment."""

    bootstrap_servers: str = Field(default="localhost:9092")
    security_protocol: SecurityProtocol = Field(default=SecurityProtocol.PLAINTEXT)
    sasl_mechanism: SaslMechanism | None = Field(default=None)
    sasl_username: str | None = Field(default=None)
    sasl_password: SecretStr | None = Field(default=None)
    ssl_cafile: str | None = Field(default=None)
    ssl_certfile: str | None = Field(default=None)
    ssl_keyfile: str | None = Field(default=None)
    client_id: str = Field(default="solace-ai")
    request_timeout_ms: int = Field(default=30000, ge=1000)
    metadata_max_age_ms: int = Field(default=300000, ge=1000)

    model_config = SettingsConfigDict(
        env_prefix="KAFKA_",
        env_file=".env",
        extra="ignore",
    )

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters for aiokafka."""
        params: dict[str, Any] = {
            "bootstrap_servers": self.bootstrap_servers,
            "client_id": self.client_id,
            "request_timeout_ms": self.request_timeout_ms,
            "metadata_max_age_ms": self.metadata_max_age_ms,
        }
        if self.security_protocol != SecurityProtocol.PLAINTEXT:
            params["security_protocol"] = self.security_protocol.value
        if self.sasl_mechanism:
            params["sasl_mechanism"] = self.sasl_mechanism.value
            params["sasl_plain_username"] = self.sasl_username
            params["sasl_plain_password"] = self.sasl_password.get_secret_value() if self.sasl_password else None
        if self.ssl_cafile:
            params["ssl_context"] = self._create_ssl_context()
        return params

    def _create_ssl_context(self) -> Any:
        """Create SSL context for secure connections."""
        import ssl
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if self.ssl_cafile:
            ctx.load_verify_locations(self.ssl_cafile)
        if self.ssl_certfile and self.ssl_keyfile:
            ctx.load_cert_chain(self.ssl_certfile, self.ssl_keyfile)
        return ctx


class ProducerSettings(BaseModel):
    """Kafka producer-specific settings."""

    acks: str = Field(default="all")
    compression_type: CompressionType = Field(default=CompressionType.GZIP)
    max_batch_size: int = Field(default=16384, ge=0)
    linger_ms: int = Field(default=5, ge=0)
    retries: int = Field(default=3, ge=0)
    retry_backoff_ms: int = Field(default=100, ge=0)
    max_in_flight: int = Field(default=1, ge=1, le=5)
    enable_idempotence: bool = Field(default=True)

    model_config = ConfigDict(frozen=True)

    def to_producer_params(self) -> dict[str, Any]:
        """Convert to aiokafka producer parameters."""
        return {
            "acks": self.acks,
            "compression_type": self.compression_type.value,
            "max_batch_size": self.max_batch_size,
            "linger_ms": self.linger_ms,
            "max_request_size": 1048576,
            "enable_idempotence": self.enable_idempotence,
        }


class ConsumerSettings(BaseModel):
    """Kafka consumer-specific settings."""

    group_id: str = Field(..., min_length=1)
    auto_offset_reset: str = Field(default="earliest")
    enable_auto_commit: bool = Field(default=False)
    max_poll_records: int = Field(default=100, ge=1, le=1000)
    session_timeout_ms: int = Field(default=10000, ge=6000)
    heartbeat_interval_ms: int = Field(default=3000, ge=1000)
    max_poll_interval_ms: int = Field(default=300000, ge=1000)

    model_config = ConfigDict(frozen=True)

    @field_validator("group_id")
    @classmethod
    def validate_group_id(cls, v: str) -> str:
        """Validate consumer group ID."""
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Group ID contains invalid characters")
        return v

    def to_consumer_params(self) -> dict[str, Any]:
        """Convert to aiokafka consumer parameters."""
        return {
            "group_id": self.group_id,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
            "max_poll_records": self.max_poll_records,
            "session_timeout_ms": self.session_timeout_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "max_poll_interval_ms": self.max_poll_interval_ms,
        }


class ConsumerGroup(str, Enum):
    """Predefined consumer groups for Solace-AI services."""
    ANALYTICS = "solace-group-analytics"
    NOTIFICATIONS = "solace-group-notifications"
    AUDIT = "solace-group-audit"
    CLINICIAN_DASHBOARD = "solace-group-clinician-dashboard"
    MEMORY_CONSOLIDATION = "solace-group-memory-consolidation"
    SAFETY_MONITOR = "solace-group-safety-monitor"


def get_topic_config(topic: SolaceTopic | str) -> TopicConfig:
    """Get configuration for a topic."""
    if isinstance(topic, str):
        topic = SolaceTopic.from_string(topic)
    return TOPIC_CONFIGS.get(topic, TopicConfig(name=topic.value))


def get_all_topics() -> list[str]:
    """Get all predefined topic names."""
    return [t.value for t in SolaceTopic]


def get_dlq_topics() -> list[str]:
    """Get all dead letter queue topic names."""
    return [t.dlq_topic for t in SolaceTopic]
