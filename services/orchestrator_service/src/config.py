"""
Solace-AI Orchestrator Service - Configuration.
Centralized configuration management with validation and environment loading.
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from functools import lru_cache
import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ServiceEndpoints(BaseSettings):
    """External service endpoint configuration."""
    personality_service_url: str = Field(default="http://localhost:8002")
    diagnosis_service_url: str = Field(default="http://localhost:8003")
    treatment_service_url: str = Field(default="http://localhost:8004")
    memory_service_url: str = Field(default="http://localhost:8005")
    user_service_url: str = Field(default="http://localhost:8006")
    model_config = SettingsConfigDict(env_prefix="SERVICE_", env_file=".env", extra="ignore")


class LLMSettings(BaseSettings):
    """LLM provider configuration."""
    provider: str = Field(default="anthropic")
    model_name: str = Field(default="claude-3-sonnet-20240229")
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=60, ge=5, le=300)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")


class SafetySettings(BaseSettings):
    """Safety and crisis detection configuration."""
    enable_crisis_detection: bool = Field(default=True)
    crisis_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    escalation_cooldown_minutes: int = Field(default=30, ge=1)
    max_risk_level_before_block: str = Field(default="critical")
    enable_content_filtering: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="SAFETY_", env_file=".env", extra="ignore")


class WebSocketSettings(BaseSettings):
    """WebSocket connection configuration."""
    heartbeat_interval_seconds: int = Field(default=30, ge=5, le=120)
    connection_timeout_seconds: int = Field(default=300, ge=30)
    max_connections_per_user: int = Field(default=5, ge=1, le=20)
    max_message_size_bytes: int = Field(default=65536, ge=1024)
    enable_compression: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="WS_", env_file=".env", extra="ignore")


class PersistenceSettings(BaseSettings):
    """State persistence configuration."""
    enable_checkpointing: bool = Field(default=True)
    checkpoint_backend: str = Field(default="memory")
    redis_url: str = Field(default="redis://localhost:6379/0")
    postgres_url: str = Field(default="postgresql://localhost:5432/orchestrator")
    checkpoint_ttl_hours: int = Field(default=24, ge=1)
    max_checkpoint_size_mb: int = Field(default=10, ge=1, le=100)
    model_config = SettingsConfigDict(env_prefix="PERSISTENCE_", env_file=".env", extra="ignore")


class OrchestratorConfig(BaseSettings):
    """Master configuration aggregating all settings."""
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    service_name: str = Field(default="orchestrator-service")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001, ge=1, le=65535)
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(default="*")
    request_timeout_ms: int = Field(default=60000, ge=1000)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="ORCHESTRATOR_", env_file=".env", extra="ignore")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION


class ConfigLoader:
    """Loads and manages configuration with caching."""

    def __init__(self) -> None:
        self._config: OrchestratorConfig | None = None
        self._endpoints: ServiceEndpoints | None = None
        self._llm: LLMSettings | None = None
        self._safety: SafetySettings | None = None
        self._websocket: WebSocketSettings | None = None
        self._persistence: PersistenceSettings | None = None

    def load(self) -> OrchestratorConfig:
        """Load main configuration."""
        if self._config is None:
            self._config = OrchestratorConfig()
            logger.info("config_loaded", environment=self._config.environment.value)
        return self._config

    def endpoints(self) -> ServiceEndpoints:
        """Load service endpoints configuration."""
        if self._endpoints is None:
            self._endpoints = ServiceEndpoints()
        return self._endpoints

    def llm(self) -> LLMSettings:
        """Load LLM configuration."""
        if self._llm is None:
            self._llm = LLMSettings()
        return self._llm

    def safety(self) -> SafetySettings:
        """Load safety configuration."""
        if self._safety is None:
            self._safety = SafetySettings()
        return self._safety

    def websocket(self) -> WebSocketSettings:
        """Load WebSocket configuration."""
        if self._websocket is None:
            self._websocket = WebSocketSettings()
        return self._websocket

    def persistence(self) -> PersistenceSettings:
        """Load persistence configuration."""
        if self._persistence is None:
            self._persistence = PersistenceSettings()
        return self._persistence

    def to_dict(self) -> dict[str, Any]:
        """Export all configuration as dictionary."""
        return {
            "main": self.load().model_dump(),
            "endpoints": self.endpoints().model_dump(),
            "llm": self.llm().model_dump(),
            "safety": self.safety().model_dump(),
            "websocket": self.websocket().model_dump(),
            "persistence": self.persistence().model_dump(),
        }


@lru_cache(maxsize=1)
def get_config() -> ConfigLoader:
    """Get cached configuration loader instance."""
    return ConfigLoader()
