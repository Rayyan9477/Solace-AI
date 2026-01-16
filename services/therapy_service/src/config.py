"""
Solace-AI Therapy Service - Configuration.
Centralized service configuration using pydantic-settings.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="solace_therapy")
    user: str = Field(default="solace")
    password: str = Field(default="")
    pool_size: int = Field(default=10, ge=1, le=100)
    pool_max_overflow: int = Field(default=20, ge=0, le=50)
    pool_timeout: int = Field(default=30, ge=5, le=120)
    echo_sql: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_prefix="THERAPY_DB_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis cache configuration settings."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0, ge=0, le=15)
    password: str = Field(default="")
    ssl: bool = Field(default=False)
    session_ttl_seconds: int = Field(default=3600)
    cache_ttl_seconds: int = Field(default=300)

    model_config = SettingsConfigDict(
        env_prefix="THERAPY_REDIS_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class LLMSettings(BaseSettings):
    """LLM provider configuration settings."""
    provider: str = Field(default="anthropic")
    model: str = Field(default="claude-sonnet-4-20250514")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=100, le=8192)
    timeout_seconds: int = Field(default=60, ge=10, le=300)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    api_key: str = Field(default="")

    model_config = SettingsConfigDict(
        env_prefix="THERAPY_LLM_",
        env_file=".env",
        extra="ignore",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        allowed = ["anthropic", "openai", "portkey"]
        if v.lower() not in allowed:
            raise ValueError(f"Provider must be one of: {allowed}")
        return v.lower()


class SessionSettings(BaseSettings):
    """Therapy session configuration settings."""
    max_duration_minutes: int = Field(default=60, ge=15, le=120)
    min_opening_duration_seconds: int = Field(default=180, ge=60, le=600)
    min_working_duration_minutes: int = Field(default=10, ge=5, le=30)
    enable_flexible_transitions: bool = Field(default=True)
    min_engagement_score: float = Field(default=0.3, ge=0.0, le=1.0)
    auto_close_after_minutes: int = Field(default=90, ge=60, le=180)
    max_concurrent_sessions_per_user: int = Field(default=1, ge=1, le=3)

    model_config = SettingsConfigDict(
        env_prefix="THERAPY_SESSION_",
        env_file=".env",
        extra="ignore",
    )


class TreatmentSettings(BaseSettings):
    """Treatment planning configuration settings."""
    default_sessions_per_phase: int = Field(default=4, ge=1, le=12)
    min_sessions_before_advancement: int = Field(default=2, ge=1, le=6)
    enable_auto_advancement: bool = Field(default=True)
    goal_review_interval_days: int = Field(default=14, ge=7, le=30)
    enable_stepped_care: bool = Field(default=True)
    max_goals_per_plan: int = Field(default=10, ge=3, le=20)
    homework_default_due_days: int = Field(default=7, ge=1, le=14)

    model_config = SettingsConfigDict(
        env_prefix="THERAPY_TREATMENT_",
        env_file=".env",
        extra="ignore",
    )


class SafetySettings(BaseSettings):
    """Safety monitoring configuration settings."""
    enable_risk_monitoring: bool = Field(default=True)
    crisis_escalation_enabled: bool = Field(default=True)
    risk_check_interval_messages: int = Field(default=3, ge=1, le=10)
    auto_pause_on_high_risk: bool = Field(default=True)
    crisis_hotline_number: str = Field(default="988")
    require_safety_plan_severe: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="THERAPY_SAFETY_",
        env_file=".env",
        extra="ignore",
    )


class ObservabilitySettings(BaseSettings):
    """Observability and monitoring configuration."""
    log_level: str = Field(default="INFO")
    enable_tracing: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    trace_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    sentry_dsn: str = Field(default="")
    otlp_endpoint: str = Field(default="")

    model_config = SettingsConfigDict(
        env_prefix="THERAPY_OBSERVABILITY_",
        env_file=".env",
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()


class TherapyServiceSettings(BaseSettings):
    """Main therapy service configuration."""
    service_name: str = Field(default="therapy-service")
    service_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8002, ge=1024, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    debug: bool = Field(default=False)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_prefix: str = Field(default="/api/v1")
    docs_enabled: bool = Field(default=True)

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    treatment: TreatmentSettings = Field(default_factory=TreatmentSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    model_config = SettingsConfigDict(
        env_prefix="THERAPY_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v.lower()

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug and not self.is_production

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary (excluding secrets)."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "debug": self.debug,
            "api_prefix": self.api_prefix,
            "docs_enabled": self.docs_enabled,
            "session": {
                "max_duration_minutes": self.session.max_duration_minutes,
                "enable_flexible_transitions": self.session.enable_flexible_transitions,
            },
            "treatment": {
                "enable_stepped_care": self.treatment.enable_stepped_care,
                "enable_auto_advancement": self.treatment.enable_auto_advancement,
            },
            "safety": {
                "enable_risk_monitoring": self.safety.enable_risk_monitoring,
                "crisis_escalation_enabled": self.safety.crisis_escalation_enabled,
            },
        }


@lru_cache
def get_settings() -> TherapyServiceSettings:
    """Get cached service settings singleton."""
    return TherapyServiceSettings()


def reset_settings() -> None:
    """Reset settings cache (for testing)."""
    get_settings.cache_clear()
