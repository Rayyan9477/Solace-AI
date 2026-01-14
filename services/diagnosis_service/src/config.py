"""
Solace-AI Diagnosis Service - Centralized Configuration.
Service configuration with environment-based settings.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="diagnosis_db")
    user: str = Field(default="diagnosis_user")
    password: str = Field(default="")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    echo: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_DB_", env_file=".env", extra="ignore")

    @property
    def connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=2)
    password: str = Field(default="")
    ttl_seconds: int = Field(default=3600)
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_REDIS_", env_file=".env", extra="ignore")

    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class AISettings(BaseSettings):
    """AI/LLM configuration."""
    model_provider: str = Field(default="anthropic")
    model_name: str = Field(default="claude-3-5-sonnet-20241022")
    temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=2048)
    timeout_seconds: int = Field(default=30)
    retry_attempts: int = Field(default=3)
    enable_streaming: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_AI_", env_file=".env", extra="ignore")


class ReasoningSettings(BaseSettings):
    """4-step Chain-of-Reasoning configuration."""
    max_reasoning_time_ms: int = Field(default=10000)
    enable_anti_sycophancy: bool = Field(default=True)
    min_confidence_threshold: float = Field(default=0.3)
    max_hypotheses: int = Field(default=5)
    phase_transition_threshold: float = Field(default=0.7)
    challenge_frequency: float = Field(default=0.3)
    enable_devil_advocate: bool = Field(default=True)
    confidence_calibration_enabled: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_REASONING_", env_file=".env", extra="ignore")


class SafetySettings(BaseSettings):
    """Safety and clinical guardrails configuration."""
    enable_safety_checks: bool = Field(default=True)
    crisis_keywords: list[str] = Field(default_factory=lambda: ["suicide", "self-harm", "kill myself", "end my life"])
    escalation_threshold: float = Field(default=0.8)
    require_clinician_review: bool = Field(default=True)
    max_session_duration_minutes: int = Field(default=60)
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_SAFETY_", env_file=".env", extra="ignore")


class ObservabilitySettings(BaseSettings):
    """Observability configuration."""
    log_level: str = Field(default="INFO")
    enable_tracing: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    otlp_endpoint: str = Field(default="http://localhost:4317")
    service_name: str = Field(default="diagnosis-service")
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_OBS_", env_file=".env", extra="ignore")


class DiagnosisServiceConfig(BaseSettings):
    """Main diagnosis service configuration."""
    service_name: str = Field(default="diagnosis-service")
    service_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8003)
    enable_longitudinal_tracking: bool = Field(default=True)
    enable_hitop_dimensions: bool = Field(default=True)
    enable_clinical_codes: bool = Field(default=True)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ai: AISettings = Field(default_factory=AISettings)
    reasoning: ReasoningSettings = Field(default_factory=ReasoningSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_", env_file=".env", extra="ignore")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary (excluding secrets)."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "enable_longitudinal_tracking": self.enable_longitudinal_tracking,
            "enable_hitop_dimensions": self.enable_hitop_dimensions,
            "enable_clinical_codes": self.enable_clinical_codes,
            "ai_provider": self.ai.model_provider,
            "ai_model": self.ai.model_name,
            "reasoning": {"anti_sycophancy": self.reasoning.enable_anti_sycophancy,
                          "max_hypotheses": self.reasoning.max_hypotheses,
                          "devil_advocate": self.reasoning.enable_devil_advocate},
            "safety": {"enabled": self.safety.enable_safety_checks,
                       "clinician_review": self.safety.require_clinician_review},
        }


@lru_cache
def get_config() -> DiagnosisServiceConfig:
    """Get cached configuration instance."""
    config = DiagnosisServiceConfig()
    logger.info("config_loaded", environment=config.environment, service=config.service_name)
    return config


def reload_config() -> DiagnosisServiceConfig:
    """Reload configuration (clears cache)."""
    get_config.cache_clear()
    return get_config()
