"""
Solace-AI Configuration Service - Centralized Configuration Management.
Enterprise-grade hierarchical configuration with validation, caching, and hot-reload.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
from collections.abc import Callable, Awaitable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Generic
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)
T = TypeVar("T")


class ConfigEnvironment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigSource(str, Enum):
    """Configuration source types."""
    ENVIRONMENT = "environment"
    FILE = "file"
    SECRETS_MANAGER = "secrets_manager"
    VAULT = "vault"
    REMOTE = "remote"
    DEFAULT = "default"


class ConfigMetadata(BaseModel):
    """Metadata for configuration values."""
    source: ConfigSource = Field(default=ConfigSource.DEFAULT)
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="1.0.0")
    checksum: str | None = Field(default=None)
    ttl_seconds: int | None = Field(default=None)
    sensitive: bool = Field(default=False)


class ConfigValue(BaseModel, Generic[T]):
    """Wrapper for configuration values with metadata tracking."""
    key: str
    value: T
    metadata: ConfigMetadata = Field(default_factory=ConfigMetadata)

    @property
    def is_expired(self) -> bool:
        if self.metadata.ttl_seconds is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.metadata.loaded_at).total_seconds()
        return elapsed > self.metadata.ttl_seconds


class ConfigServiceSettings(BaseSettings):
    """Configuration service settings from environment."""
    environment: ConfigEnvironment = Field(default=ConfigEnvironment.DEVELOPMENT)
    service_name: str = Field(default="config-service")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8008, ge=1, le=65535)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    config_path: str = Field(default="./config")
    cache_ttl_seconds: int = Field(default=300, ge=0)
    hot_reload_enabled: bool = Field(default=True)
    hot_reload_interval_seconds: int = Field(default=30, ge=5)
    vault_enabled: bool = Field(default=False)
    vault_url: str = Field(default="http://localhost:8200")
    vault_token: SecretStr | None = Field(default=None)
    vault_mount_path: str = Field(default="secret")
    aws_secrets_enabled: bool = Field(default=False)
    aws_region: str = Field(default="us-east-1")
    redis_url: str = Field(default="redis://localhost:6379/0")
    encryption_key: SecretStr | None = Field(default=None)
    cors_origins: str = Field(default="*")
    model_config = SettingsConfigDict(
        env_prefix="CONFIG_", env_file=".env", extra="ignore", case_sensitive=False
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="solace")
    username: str = Field(default="solace")
    password: SecretStr = Field(default=SecretStr("changeme"))
    pool_size: int = Field(default=10, ge=1, le=100)
    pool_max_overflow: int = Field(default=20, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=1)
    ssl_mode: str = Field(default="prefer")

    def get_dsn(self, hide_password: bool = True) -> str:
        pwd = "***" if hide_password else self.password.get_secret_value()
        return f"postgresql://{self.username}:{pwd}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis connection configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: SecretStr | None = Field(default=None)
    db: int = Field(default=0, ge=0, le=15)
    ssl: bool = Field(default=False)
    max_connections: int = Field(default=50, ge=1)


class KafkaConfig(BaseModel):
    """Kafka connection configuration."""
    bootstrap_servers: str = Field(default="localhost:9092")
    security_protocol: str = Field(default="PLAINTEXT")
    sasl_mechanism: str | None = Field(default=None)
    sasl_username: str | None = Field(default=None)
    sasl_password: SecretStr | None = Field(default=None)
    consumer_group_id: str = Field(default="solace-consumers")
    auto_offset_reset: str = Field(default="earliest")


class WeaviateConfig(BaseModel):
    """Weaviate vector database configuration."""
    url: str = Field(default="http://localhost:8080")
    api_key: SecretStr | None = Field(default=None)
    grpc_host: str = Field(default="localhost")
    grpc_port: int = Field(default=50051, ge=1, le=65535)
    default_vectorizer: str = Field(default="text2vec-openai")
    batch_size: int = Field(default=100, ge=1)


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    default_provider: str = Field(default="anthropic")
    anthropic_api_key: SecretStr | None = Field(default=None)
    openai_api_key: SecretStr | None = Field(default=None)
    default_model: str = Field(default="claude-3-opus-20240229")
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)


class SecurityConfig(BaseModel):
    """Security and compliance configuration."""
    jwt_secret_key: SecretStr = Field(default=SecretStr("change-in-production"))
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1)
    jwt_refresh_token_expire_days: int = Field(default=7, ge=1)
    encryption_algorithm: str = Field(default="AES-256-GCM")
    phi_encryption_enabled: bool = Field(default=True)
    audit_logging_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)


class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration."""
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1, le=65535)
    tracing_enabled: bool = Field(default=True)
    tracing_endpoint: str = Field(default="http://localhost:4317")
    tracing_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    profiling_enabled: bool = Field(default=False)


class ApplicationConfig(BaseModel):
    """Complete application configuration."""
    environment: ConfigEnvironment = Field(default=ConfigEnvironment.DEVELOPMENT)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


class ConfigurationManager:
    """Centralized configuration management with caching and hot-reload."""

    def __init__(self, settings: ConfigServiceSettings | None = None) -> None:
        self._settings = settings or ConfigServiceSettings()
        self._config: ApplicationConfig | None = None
        self._cache: dict[str, ConfigValue[Any]] = {}
        self._listeners: list[Callable[[str, Any], Awaitable[None]]] = []
        self._reload_task: asyncio.Task | None = None
        self._checksum: str | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def settings(self) -> ConfigServiceSettings:
        return self._settings

    @property
    def config(self) -> ApplicationConfig:
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self._config

    @property
    def environment(self) -> ConfigEnvironment:
        return self._settings.environment

    @property
    def is_production(self) -> bool:
        return self._settings.environment == ConfigEnvironment.PRODUCTION

    async def load(self) -> ApplicationConfig:
        """Load configuration from all sources."""
        async with self._lock:
            config_data: dict[str, Any] = {}
            config_path = Path(self._settings.config_path)
            if config_path.exists():
                for source in self._get_config_files(config_path):
                    file_config = self._load_config_file(source)
                    config_data = self._merge_config(config_data, file_config)
                    logger.info("config_file_loaded", path=str(source))
            env_config = self._load_from_environment()
            config_data = self._merge_config(config_data, env_config)
            self._config = ApplicationConfig(**config_data)
            self._checksum = self._compute_checksum(config_data)
            self._initialized = True
            logger.info("configuration_loaded", environment=self._settings.environment.value,
                        checksum=self._checksum[:12])
            if self._settings.hot_reload_enabled and not self._reload_task:
                self._reload_task = asyncio.create_task(self._hot_reload_loop())
            return self._config

    async def reload(self) -> bool:
        """Reload configuration and notify listeners if changed."""
        old_checksum = self._checksum
        await self.load()
        if self._checksum != old_checksum:
            logger.info("configuration_changed", old_checksum=old_checksum[:12] if old_checksum else None,
                        new_checksum=self._checksum[:12] if self._checksum else None)
            for listener in self._listeners:
                await listener("config_reloaded", self._config)
            return True
        return False

    def get(self, key: str, default: T = None) -> T:
        """Get configuration value by dot-notation key."""
        if key in self._cache:
            cached = self._cache[key]
            if not cached.is_expired:
                return cached.value
            del self._cache[key]
        value = self._resolve_key(key, default)
        self._cache[key] = ConfigValue(key=key, value=value, metadata=ConfigMetadata(
            ttl_seconds=self._settings.cache_ttl_seconds))
        return value

    def get_section(self, section: str) -> dict[str, Any]:
        """Get entire configuration section."""
        if self._config is None:
            return {}
        try:
            obj = getattr(self._config, section, None)
            if obj is None:
                return {}
            if isinstance(obj, BaseModel):
                return obj.model_dump()
            return dict(obj) if isinstance(obj, dict) else {}
        except AttributeError:
            return {}

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """Set configuration value at runtime."""
        parts = key.split(".")
        if len(parts) < 2:
            raise ValueError("Key must be in format 'section.key'")
        section, attr = parts[0], ".".join(parts[1:])
        if self._config and hasattr(self._config, section):
            section_obj = getattr(self._config, section)
            if hasattr(section_obj, attr):
                setattr(section_obj, attr, value)
                self._cache[key] = ConfigValue(key=key, value=value, metadata=ConfigMetadata(
                    source=ConfigSource.REMOTE))
                logger.info("config_value_set", key=key, persist=persist)

    def register_listener(self, callback: Callable[[str, Any], Awaitable[None]]) -> None:
        """Register callback for configuration changes."""
        self._listeners.append(callback)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
        self._cache.clear()
        logger.info("configuration_manager_shutdown")

    def _get_config_files(self, config_path: Path) -> list[Path]:
        """Get ordered list of config files to load."""
        files: list[Path] = []
        base_file = config_path / "config.json"
        if base_file.exists():
            files.append(base_file)
        env_file = config_path / f"config.{self._settings.environment.value}.json"
        if env_file.exists():
            files.append(env_file)
        local_file = config_path / "config.local.json"
        if local_file.exists():
            files.append(local_file)
        return files

    def _load_config_file(self, path: Path) -> dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("config_file_load_error", path=str(path), error=str(e))
            return {}

    def _load_from_environment(self) -> dict[str, Any]:
        """Load configuration overrides from environment variables."""
        return {"environment": self._settings.environment.value}

    def _merge_config(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _resolve_key(self, key: str, default: T) -> T:
        """Resolve dot-notation key to configuration value."""
        if self._config is None:
            return default
        parts = key.split(".")
        obj: Any = self._config
        for part in parts:
            if isinstance(obj, BaseModel):
                obj = getattr(obj, part, None)
            elif isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return default
            if obj is None:
                return default
        return obj

    def _compute_checksum(self, config_data: dict[str, Any]) -> str:
        """Compute configuration checksum for change detection."""
        serialized = json.dumps(config_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def _hot_reload_loop(self) -> None:
        """Background task for hot configuration reload."""
        while True:
            try:
                await asyncio.sleep(self._settings.hot_reload_interval_seconds)
                await self.reload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("hot_reload_error", error=str(e))


_manager: ConfigurationManager | None = None


def get_config_manager() -> ConfigurationManager:
    """Get singleton configuration manager instance."""
    global _manager
    if _manager is None:
        _manager = ConfigurationManager()
    return _manager


async def initialize_config(settings: ConfigServiceSettings | None = None) -> ConfigurationManager:
    """Initialize and load configuration manager."""
    global _manager
    _manager = ConfigurationManager(settings)
    await _manager.load()
    return _manager
