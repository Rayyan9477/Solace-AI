"""
Solace-AI Safety Service - Centralized Configuration.
All safety service configuration with externalized environment support.
"""
from __future__ import annotations
from decimal import Decimal
from typing import Any
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class SafetyServiceConfig(BaseSettings):
    """Main safety service configuration from environment."""
    service_name: str = Field(default="safety-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001, ge=1, le=65535)
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(default="*")
    request_timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    max_request_size_mb: int = Field(default=10, ge=1, le=100)
    model_config = SettingsConfigDict(
        env_prefix="SAFETY_", env_file=".env", extra="ignore", case_sensitive=False
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"Invalid log level: {v}")
        return upper

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class CrisisDetectionConfig(BaseSettings):
    """Crisis detection layer configuration."""
    enable_layer_1: bool = Field(default=True, description="Enable input gate layer")
    enable_layer_2: bool = Field(default=True, description="Enable processing guard")
    enable_layer_3: bool = Field(default=True, description="Enable output filter")
    enable_layer_4: bool = Field(default=True, description="Enable continuous monitor")
    low_threshold: Decimal = Field(default=Decimal("0.3"), ge=0, le=1)
    elevated_threshold: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    high_threshold: Decimal = Field(default=Decimal("0.7"), ge=0, le=1)
    critical_threshold: Decimal = Field(default=Decimal("0.9"), ge=0, le=1)
    keyword_weight: Decimal = Field(default=Decimal("0.4"), ge=0, le=1)
    pattern_weight: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)
    sentiment_weight: Decimal = Field(default=Decimal("0.2"), ge=0, le=1)
    history_weight: Decimal = Field(default=Decimal("0.15"), ge=0, le=1)
    max_detection_time_ms: int = Field(default=10, ge=1, le=100)
    model_config = SettingsConfigDict(
        env_prefix="CRISIS_", env_file=".env", extra="ignore"
    )


class EscalationConfig(BaseSettings):
    """Escalation workflow configuration."""
    auto_escalate_critical: bool = Field(default=True)
    auto_escalate_high: bool = Field(default=True)
    critical_response_sla_minutes: int = Field(default=5, ge=1, le=60)
    high_response_sla_minutes: int = Field(default=15, ge=1, le=120)
    medium_response_sla_minutes: int = Field(default=60, ge=1, le=480)
    low_response_sla_minutes: int = Field(default=240, ge=1, le=1440)
    notification_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    max_notification_retries: int = Field(default=3, ge=1, le=10)
    enable_sms_notifications: bool = Field(default=True)
    enable_email_notifications: bool = Field(default=True)
    enable_pager_notifications: bool = Field(default=True)
    on_call_clinician_pool_size: int = Field(default=3, ge=1, le=20)
    model_config = SettingsConfigDict(
        env_prefix="ESCALATION_", env_file=".env", extra="ignore"
    )


class SafetyMonitoringConfig(BaseSettings):
    """Continuous safety monitoring configuration."""
    enable_pre_check: bool = Field(default=True)
    enable_post_check: bool = Field(default=True)
    enable_continuous_monitoring: bool = Field(default=True)
    trajectory_window_size: int = Field(default=5, ge=2, le=20)
    deterioration_threshold: Decimal = Field(default=Decimal("0.6"), ge=0, le=1)
    max_history_messages: int = Field(default=20, ge=5, le=100)
    assessment_cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    cache_assessments: bool = Field(default=True)
    safe_response_threshold: Decimal = Field(default=Decimal("0.3"), ge=0, le=1)
    model_config = SettingsConfigDict(
        env_prefix="SAFETY_MONITORING_", env_file=".env", extra="ignore"
    )


class SafetyRepositoryConfig(BaseSettings):
    """Repository and persistence configuration."""
    storage_backend: str = Field(default="memory")
    postgres_dsn: SecretStr | None = Field(default=None)
    redis_url: str = Field(default="redis://localhost:6379/0")
    assessment_retention_days: int = Field(default=365, ge=30, le=3650)
    incident_retention_days: int = Field(default=730, ge=90, le=3650)
    safety_plan_retention_days: int = Field(default=1825, ge=365, le=7300)
    enable_audit_logging: bool = Field(default=True)
    audit_log_retention_days: int = Field(default=730, ge=90, le=3650)
    model_config = SettingsConfigDict(
        env_prefix="SAFETY_REPO_", env_file=".env", extra="ignore"
    )


class SafetyEventsConfig(BaseSettings):
    """Event publishing configuration."""
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_topic_prefix: str = Field(default="solace.safety")
    enable_event_publishing: bool = Field(default=True)
    event_batch_size: int = Field(default=100, ge=1, le=1000)
    event_flush_interval_ms: int = Field(default=1000, ge=100, le=10000)
    enable_dead_letter_queue: bool = Field(default=True)
    max_retry_attempts: int = Field(default=3, ge=1, le=10)
    model_config = SettingsConfigDict(
        env_prefix="SAFETY_EVENTS_", env_file=".env", extra="ignore"
    )


class SafetyConfig(BaseModel):
    """Aggregate configuration for safety service."""
    service: SafetyServiceConfig = Field(default_factory=SafetyServiceConfig)
    crisis_detection: CrisisDetectionConfig = Field(default_factory=CrisisDetectionConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    monitoring: SafetyMonitoringConfig = Field(default_factory=SafetyMonitoringConfig)
    repository: SafetyRepositoryConfig = Field(default_factory=SafetyRepositoryConfig)
    events: SafetyEventsConfig = Field(default_factory=SafetyEventsConfig)

    @classmethod
    def load(cls) -> SafetyConfig:
        """Load configuration from environment."""
        config = cls()
        logger.info(
            "safety_config_loaded",
            environment=config.service.environment,
            detection_layers_enabled={
                "layer_1": config.crisis_detection.enable_layer_1,
                "layer_2": config.crisis_detection.enable_layer_2,
                "layer_3": config.crisis_detection.enable_layer_3,
                "layer_4": config.crisis_detection.enable_layer_4,
            },
            auto_escalate_critical=config.escalation.auto_escalate_critical,
        )
        return config

    def to_dict(self, hide_secrets: bool = True) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        data = {
            "service": self.service.model_dump(),
            "crisis_detection": self.crisis_detection.model_dump(),
            "escalation": self.escalation.model_dump(),
            "monitoring": self.monitoring.model_dump(),
            "repository": self.repository.model_dump(),
            "events": self.events.model_dump(),
        }
        if hide_secrets and self.repository.postgres_dsn:
            data["repository"]["postgres_dsn"] = "***"
        return data


_config: SafetyConfig | None = None


def get_safety_config() -> SafetyConfig:
    """Get singleton safety configuration."""
    global _config
    if _config is None:
        _config = SafetyConfig.load()
    return _config


def reset_config() -> None:
    """Reset configuration singleton (for testing)."""
    global _config
    _config = None
