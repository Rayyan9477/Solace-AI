"""
Solace-AI Memory Service - Configuration.

Centralized configuration for memory service with validation and defaults.
Follows 12-factor app principles with environment-based configuration.
"""
from __future__ import annotations
from decimal import Decimal
from typing import Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class PostgresConfig(BaseSettings):
    """PostgreSQL database configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="solace_memory")
    user: str = Field(default="solace")
    password: str = Field(default="")
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=5, ge=0, le=50)
    pool_timeout: int = Field(default=30, ge=1, le=300)
    echo_sql: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    @property
    def connection_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseSettings):
    """Redis cache configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: str = Field(default="")
    db: int = Field(default=0, ge=0, le=15)
    working_memory_ttl: int = Field(default=3600, ge=60)
    session_ttl: int = Field(default=86400, ge=300)
    embedding_cache_ttl: int = Field(default=3600, ge=60)
    key_prefix: str = Field(default="memory:")
    max_connections: int = Field(default=50, ge=5, le=500)
    socket_timeout: int = Field(default=5, ge=1, le=60)
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    @property
    def connection_url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class WeaviateConfig(BaseSettings):
    """Weaviate vector database configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=8080, ge=1, le=65535)
    grpc_port: int = Field(default=50051, ge=1, le=65535)
    use_https: bool = Field(default=False)
    api_key: str = Field(default="")
    embedding_dimension: int = Field(default=1536, ge=64, le=4096)
    batch_size: int = Field(default=100, ge=1, le=1000)
    consistency_level: str = Field(default="ONE")
    model_config = SettingsConfigDict(env_prefix="WEAVIATE_", env_file=".env", extra="ignore")

    @property
    def http_url(self) -> str:
        protocol = "https" if self.use_https else "http"
        return f"{protocol}://{self.host}:{self.port}"


class KafkaConfig(BaseSettings):
    """Kafka event bus configuration."""
    bootstrap_servers: str = Field(default="localhost:9092")
    topic_prefix: str = Field(default="solace.")
    group_id: str = Field(default="memory-service")
    auto_offset_reset: str = Field(default="earliest")
    enable_auto_commit: bool = Field(default=True)
    session_timeout_ms: int = Field(default=30000, ge=1000)
    heartbeat_interval_ms: int = Field(default=10000, ge=500)
    max_poll_records: int = Field(default=100, ge=1, le=1000)
    model_config = SettingsConfigDict(env_prefix="KAFKA_", env_file=".env", extra="ignore")


class DecayConfig(BaseSettings):
    """Memory decay model configuration."""
    enabled: bool = Field(default=True)
    run_interval_hours: int = Field(default=1, ge=1, le=24)
    batch_size: int = Field(default=1000, ge=100, le=10000)
    permanent_decay_rate: Decimal = Field(default=Decimal("0"))
    long_term_decay_rate: Decimal = Field(default=Decimal("0.001"))
    medium_term_decay_rate: Decimal = Field(default=Decimal("0.01"))
    short_term_decay_rate: Decimal = Field(default=Decimal("0.05"))
    ephemeral_decay_rate: Decimal = Field(default=Decimal("0.2"))
    archive_threshold: Decimal = Field(default=Decimal("0.1"))
    delete_threshold: Decimal = Field(default=Decimal("0.01"))
    emotional_decay_modifier: Decimal = Field(default=Decimal("0.8"))
    access_boost: Decimal = Field(default=Decimal("0.05"))
    model_config = SettingsConfigDict(env_prefix="DECAY_", env_file=".env", extra="ignore")


class ConsolidationConfig(BaseSettings):
    """Memory consolidation configuration."""
    enabled: bool = Field(default=True)
    auto_consolidate_on_session_end: bool = Field(default=True)
    min_messages_for_summary: int = Field(default=5, ge=1)
    max_summary_tokens: int = Field(default=500, ge=50, le=2000)
    extract_facts: bool = Field(default=True)
    build_knowledge_graph: bool = Field(default=True)
    max_facts_per_session: int = Field(default=20, ge=1, le=100)
    fact_confidence_threshold: Decimal = Field(default=Decimal("0.6"))
    model_config = SettingsConfigDict(env_prefix="CONSOLIDATION_", env_file=".env", extra="ignore")


class ContextAssemblyConfig(BaseSettings):
    """Context assembly configuration."""
    default_token_budget: int = Field(default=8000, ge=1000, le=128000)
    system_prompt_allocation: Decimal = Field(default=Decimal("0.15"))
    safety_allocation: Decimal = Field(default=Decimal("0.10"))
    user_profile_allocation: Decimal = Field(default=Decimal("0.10"))
    therapeutic_context_allocation: Decimal = Field(default=Decimal("0.15"))
    recent_messages_allocation: Decimal = Field(default=Decimal("0.30"))
    retrieved_context_allocation: Decimal = Field(default=Decimal("0.15"))
    reserved_tokens: int = Field(default=500, ge=100, le=2000)
    assembly_timeout_ms: int = Field(default=100, ge=10, le=1000)
    max_retrieval_results: int = Field(default=10, ge=1, le=50)
    min_relevance_score: Decimal = Field(default=Decimal("0.5"))
    model_config = SettingsConfigDict(env_prefix="CONTEXT_", env_file=".env", extra="ignore")


class MemoryServiceConfig(BaseSettings):
    """Main memory service configuration."""
    service_name: str = Field(default="memory-service")
    service_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8005, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    enable_events: bool = Field(default=True)
    max_history_per_user: int = Field(default=100, ge=10, le=1000)
    model_config = SettingsConfigDict(env_prefix="MEMORY_SERVICE_", env_file=".env", extra="ignore")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()


class Settings:
    """Aggregated settings for the memory service."""
    _instance: Settings | None = None

    def __init__(self) -> None:
        self.service = MemoryServiceConfig()
        self.postgres = PostgresConfig()
        self.redis = RedisConfig()
        self.weaviate = WeaviateConfig()
        self.kafka = KafkaConfig()
        self.decay = DecayConfig()
        self.consolidation = ConsolidationConfig()
        self.context_assembly = ContextAssemblyConfig()
        logger.info("settings_loaded", environment=self.service.environment, service=self.service.service_name)

    @classmethod
    def get_instance(cls) -> Settings:
        """Get singleton settings instance."""
        if cls._instance is None:
            cls._instance = Settings()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset settings (for testing)."""
        cls._instance = None

    def to_dict(self) -> dict[str, Any]:
        """Export settings as dictionary (excludes secrets)."""
        return {
            "service": {"name": self.service.service_name, "version": self.service.service_version,
                        "environment": self.service.environment, "debug": self.service.debug},
            "postgres": {"host": self.postgres.host, "port": self.postgres.port, "database": self.postgres.database},
            "redis": {"host": self.redis.host, "port": self.redis.port},
            "weaviate": {"host": self.weaviate.host, "port": self.weaviate.port},
            "kafka": {"bootstrap_servers": self.kafka.bootstrap_servers, "group_id": self.kafka.group_id},
            "decay": {"enabled": self.decay.enabled, "run_interval_hours": self.decay.run_interval_hours},
            "consolidation": {"enabled": self.consolidation.enabled, "auto_consolidate": self.consolidation.auto_consolidate_on_session_end},
            "context_assembly": {"default_token_budget": self.context_assembly.default_token_budget},
        }


def get_settings() -> Settings:
    """Get the global settings instance."""
    return Settings.get_instance()
