"""
Solace-AI Analytics Service - Configuration.

Centralized configuration management for analytics service components.
Supports environment-based configuration with validation.

Architecture Layer: Infrastructure
Principles: 12-Factor App, Configuration Externalization, Type Safety
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ServiceConfiguration(BaseSettings):
    """Core service configuration."""
    name: str = Field(default="analytics-service")
    version: str = Field(default="1.0.0")
    env: Environment = Field(default=Environment.DEVELOPMENT)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8009, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    debug: bool = Field(default=False)
    workers: int = Field(default=1, ge=1, le=16)

    model_config = SettingsConfigDict(
        env_prefix="ANALYTICS_SERVICE_",
        env_file=".env",
        extra="ignore",
    )


class ClickHouseConfig(BaseSettings):
    """ClickHouse database configuration."""
    enabled: bool = Field(default=False)
    host: str = Field(default="localhost")
    port: int = Field(default=8123, ge=1, le=65535)
    database: str = Field(default="solace_analytics")
    username: str = Field(default="default")
    password: str = Field(default="")
    secure: bool = Field(default=False)
    verify: bool = Field(default=True)
    connect_timeout: float = Field(default=10.0, ge=1.0, le=60.0)
    query_timeout: float = Field(default=300.0, ge=1.0, le=3600.0)
    max_connections: int = Field(default=10, ge=1, le=100)
    compression: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="CLICKHOUSE_",
        env_file=".env",
        extra="ignore",
    )


class MetricsStoreConfig(BaseSettings):
    """In-memory metrics store configuration."""
    max_windows_per_metric: int = Field(default=1000, ge=100, le=100000)
    default_retention_hours: int = Field(default=168, ge=1, le=8760)
    aggregation_interval_seconds: int = Field(default=60, ge=10, le=3600)
    flush_interval_seconds: int = Field(default=300, ge=60, le=3600)
    max_labels_per_metric: int = Field(default=10, ge=1, le=50)

    model_config = SettingsConfigDict(
        env_prefix="METRICS_STORE_",
        env_file=".env",
        extra="ignore",
    )


class ConsumerConfiguration(BaseSettings):
    """Event consumer configuration."""
    group_id: str = Field(default="analytics-service")
    topics: list[str] = Field(
        default_factory=lambda: [
            "solace.sessions",
            "solace.safety",
            "solace.assessments",
            "solace.therapy",
            "solace.memory",
            "solace.personality",
            "solace.analytics",
        ]
    )
    batch_size: int = Field(default=100, ge=1, le=1000)
    batch_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_ms: int = Field(default=1000, ge=100, le=30000)
    kafka_enabled: bool = Field(default=False)
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    auto_offset_reset: Literal["earliest", "latest"] = Field(default="latest")

    model_config = SettingsConfigDict(
        env_prefix="CONSUMER_",
        env_file=".env",
        extra="ignore",
    )


class ReportConfig(BaseSettings):
    """Report generation configuration."""
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    max_report_rows: int = Field(default=10000, ge=100, le=100000)
    export_formats: list[str] = Field(
        default_factory=lambda: ["json", "csv", "pdf"]
    )
    async_generation_threshold: int = Field(default=5000, ge=1000, le=50000)

    model_config = SettingsConfigDict(
        env_prefix="REPORT_",
        env_file=".env",
        extra="ignore",
    )


class AggregationConfig(BaseSettings):
    """Aggregation processing configuration."""
    windows: list[str] = Field(
        default_factory=lambda: ["minute", "hour", "day", "week"]
    )
    rollup_enabled: bool = Field(default=True)
    rollup_schedule_cron: str = Field(default="0 * * * *")
    percentiles: list[int] = Field(default_factory=lambda: [50, 90, 95, 99])
    histogram_buckets: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    model_config = SettingsConfigDict(
        env_prefix="AGGREGATION_",
        env_file=".env",
        extra="ignore",
    )


class ObservabilityConfig(BaseSettings):
    """Observability configuration."""
    prometheus_enabled: bool = Field(default=True)
    prometheus_endpoint: str = Field(default="/metrics")
    otel_enabled: bool = Field(default=False)
    otel_service_name: str = Field(default="analytics-service")
    otel_exporter_endpoint: str = Field(default="http://localhost:4317")
    log_format: Literal["json", "console"] = Field(default="json")
    trace_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_prefix="OBSERVABILITY_",
        env_file=".env",
        extra="ignore",
    )


class AlertConfig(BaseSettings):
    """Alerting configuration."""
    enabled: bool = Field(default=False)
    check_interval_seconds: int = Field(default=60, ge=10, le=600)
    notification_webhook: str | None = Field(default=None)
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "error_rate": 0.05,
            "latency_p99_ms": 5000,
            "queue_depth": 10000,
        }
    )

    model_config = SettingsConfigDict(
        env_prefix="ALERT_",
        env_file=".env",
        extra="ignore",
    )


class AnalyticsServiceConfig(BaseSettings):
    """Aggregate analytics service configuration."""
    service: ServiceConfiguration = Field(default_factory=ServiceConfiguration)
    clickhouse: ClickHouseConfig = Field(default_factory=ClickHouseConfig)
    metrics_store: MetricsStoreConfig = Field(default_factory=MetricsStoreConfig)
    consumer: ConsumerConfiguration = Field(default_factory=ConsumerConfiguration)
    report: ReportConfig = Field(default_factory=ReportConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @staticmethod
    def load() -> AnalyticsServiceConfig:
        """Load configuration from environment."""
        config = AnalyticsServiceConfig()
        logger.info(
            "analytics_config_loaded",
            service=config.service.name,
            env=config.service.env.value,
            clickhouse_enabled=config.clickhouse.enabled,
            kafka_enabled=config.consumer.kafka_enabled,
        )
        return config

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.service.env == Environment.PRODUCTION

    def get_retention_timedelta(self) -> "timedelta":
        """Get retention as timedelta."""
        from datetime import timedelta
        return timedelta(hours=self.metrics_store.default_retention_hours)


_config: AnalyticsServiceConfig | None = None


def get_config() -> AnalyticsServiceConfig:
    """Get singleton configuration instance."""
    global _config
    if _config is None:
        _config = AnalyticsServiceConfig.load()
    return _config


def reset_config() -> None:
    """Reset configuration (useful for testing)."""
    global _config
    _config = None
