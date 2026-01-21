"""
Solace-AI Notification Service - Configuration.

Centralized configuration management for notification service components.
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
    name: str = Field(default="notification-service")
    version: str = Field(default="1.0.0")
    env: Environment = Field(default=Environment.DEVELOPMENT)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8003, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    debug: bool = Field(default=False)
    workers: int = Field(default=1, ge=1, le=16)

    model_config = SettingsConfigDict(
        env_prefix="NOTIFICATION_SERVICE_",
        env_file=".env",
        extra="ignore",
    )


class EmailChannelConfig(BaseSettings):
    """Email channel configuration."""
    enabled: bool = Field(default=True)
    smtp_host: str = Field(default="localhost")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str = Field(default="")
    smtp_password: str = Field(default="")
    use_tls: bool = Field(default=True)
    use_ssl: bool = Field(default=False)
    from_email: str = Field(default="noreply@solace-ai.com")
    from_name: str = Field(default="Solace-AI")
    reply_to: str | None = Field(default=None)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    batch_size: int = Field(default=100, ge=1, le=1000)

    model_config = SettingsConfigDict(
        env_prefix="EMAIL_",
        env_file=".env",
        extra="ignore",
    )

    @field_validator("from_email", mode="before")
    @classmethod
    def validate_from_email(cls, v: str) -> str:
        """Validate from email format."""
        if v and "@" not in v:
            raise ValueError("Invalid email format")
        return v.strip().lower() if v else v


class SMSChannelConfig(BaseSettings):
    """SMS channel configuration (Twilio-compatible)."""
    enabled: bool = Field(default=False)
    provider: Literal["twilio", "aws_sns", "messagebird"] = Field(default="twilio")
    provider_url: str = Field(default="https://api.twilio.com/2010-04-01")
    account_sid: str = Field(default="")
    auth_token: str = Field(default="")
    from_number: str = Field(default="")
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    max_message_length: int = Field(default=1600, ge=160, le=1600)

    model_config = SettingsConfigDict(
        env_prefix="SMS_",
        env_file=".env",
        extra="ignore",
    )


class PushChannelConfig(BaseSettings):
    """Push notification configuration (Firebase-compatible)."""
    enabled: bool = Field(default=False)
    provider: Literal["firebase", "apns", "onesignal"] = Field(default="firebase")
    firebase_url: str = Field(default="https://fcm.googleapis.com/fcm/send")
    server_key: str = Field(default="")
    project_id: str = Field(default="")
    credentials_file: str | None = Field(default=None)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    ttl_seconds: int = Field(default=86400, ge=0, le=2419200)

    model_config = SettingsConfigDict(
        env_prefix="PUSH_",
        env_file=".env",
        extra="ignore",
    )


class TemplateConfig(BaseSettings):
    """Template engine configuration."""
    templates_dir: str = Field(default="templates")
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    default_locale: str = Field(default="en")
    supported_locales: list[str] = Field(default_factory=lambda: ["en", "es", "fr", "de"])

    model_config = SettingsConfigDict(
        env_prefix="TEMPLATE_",
        env_file=".env",
        extra="ignore",
    )


class QueueConfig(BaseSettings):
    """Message queue configuration for async delivery."""
    enabled: bool = Field(default=False)
    provider: Literal["redis", "rabbitmq", "kafka"] = Field(default="redis")
    redis_url: str = Field(default="redis://localhost:6379/0")
    queue_name: str = Field(default="notifications")
    max_queue_size: int = Field(default=10000, ge=100, le=1000000)
    batch_size: int = Field(default=100, ge=1, le=1000)
    batch_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    dead_letter_queue: str = Field(default="notifications_dlq")

    model_config = SettingsConfigDict(
        env_prefix="QUEUE_",
        env_file=".env",
        extra="ignore",
    )


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    enabled: bool = Field(default=True)
    global_rate_per_minute: int = Field(default=1000, ge=1, le=100000)
    per_user_rate_per_minute: int = Field(default=60, ge=1, le=1000)
    per_channel_rate_per_minute: dict[str, int] = Field(
        default_factory=lambda: {"email": 100, "sms": 30, "push": 200}
    )
    burst_multiplier: float = Field(default=1.5, ge=1.0, le=5.0)

    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_",
        env_file=".env",
        extra="ignore",
    )


class ObservabilityConfig(BaseSettings):
    """Observability configuration."""
    prometheus_enabled: bool = Field(default=True)
    prometheus_endpoint: str = Field(default="/metrics")
    otel_enabled: bool = Field(default=False)
    otel_service_name: str = Field(default="notification-service")
    otel_exporter_endpoint: str = Field(default="http://localhost:4317")
    log_format: Literal["json", "console"] = Field(default="json")

    model_config = SettingsConfigDict(
        env_prefix="OBSERVABILITY_",
        env_file=".env",
        extra="ignore",
    )


class NotificationServiceConfig(BaseSettings):
    """Aggregate notification service configuration."""
    service: ServiceConfiguration = Field(default_factory=ServiceConfiguration)
    email: EmailChannelConfig = Field(default_factory=EmailChannelConfig)
    sms: SMSChannelConfig = Field(default_factory=SMSChannelConfig)
    push: PushChannelConfig = Field(default_factory=PushChannelConfig)
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @staticmethod
    def load() -> NotificationServiceConfig:
        """Load configuration from environment."""
        config = NotificationServiceConfig()
        logger.info(
            "notification_config_loaded",
            service=config.service.name,
            env=config.service.env.value,
            email_enabled=config.email.enabled,
            sms_enabled=config.sms.enabled,
            push_enabled=config.push.enabled,
        )
        return config

    def get_enabled_channels(self) -> list[str]:
        """Get list of enabled channels."""
        channels = []
        if self.email.enabled:
            channels.append("email")
        if self.sms.enabled:
            channels.append("sms")
        if self.push.enabled:
            channels.append("push")
        return channels

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.service.env == Environment.PRODUCTION


_config: NotificationServiceConfig | None = None


def get_config() -> NotificationServiceConfig:
    """Get singleton configuration instance."""
    global _config
    if _config is None:
        _config = NotificationServiceConfig.load()
    return _config


def reset_config() -> None:
    """Reset configuration (useful for testing)."""
    global _config
    _config = None
