"""
Solace-AI Personality Service - Centralized Configuration.
Service configuration with environment-based settings using pydantic-settings.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class DatabaseSettings(BaseSettings):
    """Database configuration for personality service."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="personality_db")
    user: str = Field(default="personality_user")
    password: str = Field(description="Database password - set via PERSONALITY_DB_PASSWORD env var")
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=100)
    echo: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_DB_", env_file=".env", extra="ignore")

    @property
    def connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis cache configuration for personality service."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=3, ge=0, le=15)
    password: str = Field(description="Redis password - set via PERSONALITY_REDIS_PASSWORD env var")
    ttl_seconds: int = Field(default=3600, ge=60)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_REDIS_", env_file=".env", extra="ignore")

    @property
    def url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class AISettings(BaseSettings):
    """AI/LLM configuration for personality detection."""
    model_provider: str = Field(default="anthropic")
    model_name: str = Field(default="claude-3-5-sonnet-20241022")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=256, le=8192)
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    retry_attempts: int = Field(default=3, ge=1, le=5)
    enable_streaming: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_AI_", env_file=".env", extra="ignore")


class TraitDetectionSettings(BaseSettings):
    """Trait detection algorithm configuration."""
    min_text_length: int = Field(default=50, ge=10)
    max_text_length: int = Field(default=10000, ge=100)
    ensemble_weight_text: float = Field(default=0.4, ge=0.0, le=1.0)
    ensemble_weight_liwc: float = Field(default=0.3, ge=0.0, le=1.0)
    ensemble_weight_llm: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_base: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_sample_factor: float = Field(default=0.1, ge=0.0, le=0.5)
    enable_llm_detection: bool = Field(default=True)
    enable_roberta_detection: bool = Field(default=True)
    enable_liwc_detection: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=60)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_DETECTION_", env_file=".env", extra="ignore")


class StyleAdaptationSettings(BaseSettings):
    """Style adaptation configuration."""
    high_trait_threshold: float = Field(default=0.7, ge=0.5, le=0.9)
    low_trait_threshold: float = Field(default=0.3, ge=0.1, le=0.5)
    warmth_neuroticism_boost: float = Field(default=0.2, ge=0.0, le=0.5)
    structure_conscientiousness_boost: float = Field(default=0.25, ge=0.0, le=0.5)
    energy_extraversion_factor: float = Field(default=0.4, ge=0.0, le=1.0)
    directness_agreeableness_factor: float = Field(default=-0.2, ge=-0.5, le=0.5)
    complexity_openness_boost: float = Field(default=0.3, ge=0.0, le=0.5)
    default_warmth: float = Field(default=0.6, ge=0.0, le=1.0)
    default_validation: float = Field(default=0.5, ge=0.0, le=1.0)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_STYLE_", env_file=".env", extra="ignore")


class ProfileSettings(BaseSettings):
    """Personality profile management configuration."""
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=600, ge=60)
    min_assessment_interval_seconds: int = Field(default=60, ge=0)
    update_threshold: float = Field(default=0.15, ge=0.05, le=0.5)
    max_history_size: int = Field(default=100, ge=10, le=1000)
    stability_threshold: float = Field(default=0.7, ge=0.5, le=0.95)
    min_assessments_for_stability: int = Field(default=3, ge=2, le=10)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_PROFILE_", env_file=".env", extra="ignore")


class ObservabilitySettings(BaseSettings):
    """Observability configuration."""
    log_level: str = Field(default="INFO")
    enable_tracing: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    otlp_endpoint: str = Field(default="http://localhost:4317")
    service_name: str = Field(default="personality-service")
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_OBS_", env_file=".env", extra="ignore")


class PersonalityServiceConfig(BaseSettings):
    """Main personality service configuration."""
    service_name: str = Field(default="personality-service")
    service_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8007, ge=1024, le=65535)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ai: AISettings = Field(default_factory=AISettings)
    detection: TraitDetectionSettings = Field(default_factory=TraitDetectionSettings)
    style: StyleAdaptationSettings = Field(default_factory=StyleAdaptationSettings)
    profile: ProfileSettings = Field(default_factory=ProfileSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_", env_file=".env", extra="ignore")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary (excluding secrets)."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "ai": {
                "provider": self.ai.model_provider,
                "model": self.ai.model_name,
                "temperature": self.ai.temperature,
            },
            "detection": {
                "llm_enabled": self.detection.enable_llm_detection,
                "roberta_enabled": self.detection.enable_roberta_detection,
                "liwc_enabled": self.detection.enable_liwc_detection,
                "min_text_length": self.detection.min_text_length,
            },
            "style": {
                "high_threshold": self.style.high_trait_threshold,
                "low_threshold": self.style.low_trait_threshold,
            },
            "profile": {
                "caching_enabled": self.profile.enable_caching,
                "max_history": self.profile.max_history_size,
                "stability_threshold": self.profile.stability_threshold,
            },
            "observability": {
                "log_level": self.observability.log_level,
                "tracing_enabled": self.observability.enable_tracing,
                "metrics_enabled": self.observability.enable_metrics,
            },
        }

    def validate_ensemble_weights(self) -> bool:
        """Validate that ensemble weights sum to approximately 1.0."""
        total = (self.detection.ensemble_weight_text + self.detection.ensemble_weight_liwc + self.detection.ensemble_weight_llm)
        return 0.99 <= total <= 1.01


@lru_cache
def get_config() -> PersonalityServiceConfig:
    """Get cached configuration instance."""
    config = PersonalityServiceConfig()
    logger.info("config_loaded", environment=config.environment, service=config.service_name, port=config.port)
    return config


def reload_config() -> PersonalityServiceConfig:
    """Reload configuration (clears cache)."""
    get_config.cache_clear()
    return get_config()


def get_detection_settings() -> TraitDetectionSettings:
    """Get detection settings from main config."""
    return get_config().detection


def get_style_settings() -> StyleAdaptationSettings:
    """Get style adaptation settings from main config."""
    return get_config().style


def get_profile_settings() -> ProfileSettings:
    """Get profile settings from main config."""
    return get_config().profile
