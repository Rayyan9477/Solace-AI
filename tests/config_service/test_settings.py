"""Unit tests for Configuration Service - Settings Module."""
from __future__ import annotations
import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import SecretStr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services"))

from config_service.src.settings import (
    ConfigEnvironment,
    ConfigSource,
    ConfigMetadata,
    ConfigValue,
    ConfigServiceSettings,
    DatabaseConfig,
    RedisConfig,
    KafkaConfig,
    WeaviateConfig,
    LLMConfig,
    SecurityConfig,
    ObservabilityConfig,
    ApplicationConfig,
    ConfigurationManager,
    get_config_manager,
    initialize_config,
)


class TestConfigEnvironment:
    """Tests for ConfigEnvironment enum."""

    def test_environment_values(self) -> None:
        assert ConfigEnvironment.DEVELOPMENT.value == "development"
        assert ConfigEnvironment.STAGING.value == "staging"
        assert ConfigEnvironment.PRODUCTION.value == "production"
        assert ConfigEnvironment.TESTING.value == "testing"

    def test_environment_from_string(self) -> None:
        env = ConfigEnvironment("production")
        assert env == ConfigEnvironment.PRODUCTION


class TestConfigSource:
    """Tests for ConfigSource enum."""

    def test_source_values(self) -> None:
        assert ConfigSource.ENVIRONMENT.value == "environment"
        assert ConfigSource.FILE.value == "file"
        assert ConfigSource.VAULT.value == "vault"
        assert ConfigSource.DEFAULT.value == "default"


class TestConfigMetadata:
    """Tests for ConfigMetadata model."""

    def test_default_metadata(self) -> None:
        meta = ConfigMetadata()
        assert meta.source == ConfigSource.DEFAULT
        assert meta.version == "1.0.0"
        assert meta.checksum is None
        assert meta.ttl_seconds is None
        assert meta.sensitive is False
        assert isinstance(meta.loaded_at, datetime)

    def test_metadata_with_values(self) -> None:
        meta = ConfigMetadata(
            source=ConfigSource.VAULT,
            version="2.0.0",
            checksum="abc123",
            ttl_seconds=300,
            sensitive=True,
        )
        assert meta.source == ConfigSource.VAULT
        assert meta.version == "2.0.0"
        assert meta.checksum == "abc123"
        assert meta.ttl_seconds == 300
        assert meta.sensitive is True


class TestConfigValue:
    """Tests for ConfigValue model."""

    def test_config_value_not_expired(self) -> None:
        cv = ConfigValue[str](key="test.key", value="test_value")
        assert cv.is_expired is False

    def test_config_value_with_ttl(self) -> None:
        cv = ConfigValue[int](
            key="test.number",
            value=42,
            metadata=ConfigMetadata(ttl_seconds=3600),
        )
        assert cv.key == "test.number"
        assert cv.value == 42
        assert cv.is_expired is False


class TestConfigServiceSettings:
    """Tests for ConfigServiceSettings."""

    def test_default_settings(self) -> None:
        settings = ConfigServiceSettings()
        assert settings.environment == ConfigEnvironment.DEVELOPMENT
        assert settings.service_name == "config-service"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8001
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.hot_reload_enabled is True

    def test_log_level_validation(self) -> None:
        settings = ConfigServiceSettings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ValueError):
            ConfigServiceSettings(log_level="invalid")

    def test_cors_origins_list(self) -> None:
        settings = ConfigServiceSettings(cors_origins="http://localhost,http://example.com")
        assert settings.cors_origins_list == ["http://localhost", "http://example.com"]


class TestDatabaseConfig:
    """Tests for DatabaseConfig model."""

    def test_default_database_config(self) -> None:
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "solace"
        assert config.username == "solace"
        assert config.pool_size == 10

    def test_dsn_with_hidden_password(self) -> None:
        config = DatabaseConfig(password=SecretStr("secret123"))
        dsn = config.get_dsn(hide_password=True)
        assert "***" in dsn
        assert "secret123" not in dsn

    def test_dsn_with_visible_password(self) -> None:
        config = DatabaseConfig(password=SecretStr("secret123"))
        dsn = config.get_dsn(hide_password=False)
        assert "secret123" in dsn


class TestRedisConfig:
    """Tests for RedisConfig model."""

    def test_default_redis_config(self) -> None:
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.ssl is False
        assert config.max_connections == 50


class TestKafkaConfig:
    """Tests for KafkaConfig model."""

    def test_default_kafka_config(self) -> None:
        config = KafkaConfig()
        assert config.bootstrap_servers == "localhost:9092"
        assert config.security_protocol == "PLAINTEXT"
        assert config.consumer_group_id == "solace-consumers"


class TestWeaviateConfig:
    """Tests for WeaviateConfig model."""

    def test_default_weaviate_config(self) -> None:
        config = WeaviateConfig()
        assert config.url == "http://localhost:8080"
        assert config.grpc_port == 50051
        assert config.batch_size == 100


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_default_llm_config(self) -> None:
        config = LLMConfig()
        assert config.default_provider == "anthropic"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.max_retries == 3


class TestSecurityConfig:
    """Tests for SecurityConfig model."""

    def test_default_security_config(self) -> None:
        config = SecurityConfig()
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_access_token_expire_minutes == 30
        assert config.phi_encryption_enabled is True
        assert config.audit_logging_enabled is True


class TestObservabilityConfig:
    """Tests for ObservabilityConfig model."""

    def test_default_observability_config(self) -> None:
        config = ObservabilityConfig()
        assert config.metrics_enabled is True
        assert config.metrics_port == 9090
        assert config.tracing_enabled is True
        assert config.tracing_sample_rate == 0.1


class TestApplicationConfig:
    """Tests for ApplicationConfig model."""

    def test_default_application_config(self) -> None:
        config = ApplicationConfig()
        assert config.environment == ConfigEnvironment.DEVELOPMENT
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.kafka, KafkaConfig)
        assert isinstance(config.weaviate, WeaviateConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.observability, ObservabilityConfig)


class TestConfigurationManager:
    """Tests for ConfigurationManager."""

    @pytest.fixture
    def manager(self) -> ConfigurationManager:
        return ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))

    @pytest.mark.asyncio
    async def test_load_configuration(self, manager: ConfigurationManager) -> None:
        config = await manager.load()
        assert isinstance(config, ApplicationConfig)
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_get_value_after_load(self, manager: ConfigurationManager) -> None:
        await manager.load()
        env = manager.get("environment")
        assert env == "development"

    @pytest.mark.asyncio
    async def test_get_section(self, manager: ConfigurationManager) -> None:
        await manager.load()
        section = manager.get_section("database")
        assert "host" in section
        assert "port" in section

    @pytest.mark.asyncio
    async def test_get_nonexistent_section(self, manager: ConfigurationManager) -> None:
        await manager.load()
        section = manager.get_section("nonexistent")
        assert section == {}

    @pytest.mark.asyncio
    async def test_is_production_false(self, manager: ConfigurationManager) -> None:
        await manager.load()
        assert manager.is_production is False

    @pytest.mark.asyncio
    async def test_reload_no_change(self, manager: ConfigurationManager) -> None:
        await manager.load()
        changed = await manager.reload()
        assert changed is False

    @pytest.mark.asyncio
    async def test_shutdown(self, manager: ConfigurationManager) -> None:
        await manager.load()
        await manager.shutdown()
        assert len(manager._cache) == 0

    @pytest.mark.asyncio
    async def test_register_listener(self, manager: ConfigurationManager) -> None:
        callback_called = False

        async def listener(event: str, data: any) -> None:
            nonlocal callback_called
            callback_called = True

        manager.register_listener(listener)
        assert len(manager._listeners) == 1

    @pytest.mark.asyncio
    async def test_load_with_config_file(self, manager: ConfigurationManager) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)
            config_file = config_path / "config.json"
            config_file.write_text(json.dumps({"database": {"host": "custom-host"}}))
            manager._settings = ConfigServiceSettings(config_path=str(config_path), hot_reload_enabled=False)
            await manager.load()
            section = manager.get_section("database")
            assert section["host"] == "custom-host"


class TestGetConfigManager:
    """Tests for get_config_manager singleton."""

    def test_get_config_manager_returns_instance(self) -> None:
        import config_service.src.settings as settings_module
        settings_module._manager = None
        mgr = get_config_manager()
        assert isinstance(mgr, ConfigurationManager)

    def test_get_config_manager_singleton(self) -> None:
        import config_service.src.settings as settings_module
        settings_module._manager = None
        mgr1 = get_config_manager()
        mgr2 = get_config_manager()
        assert mgr1 is mgr2


class TestInitializeConfig:
    """Tests for initialize_config function."""

    @pytest.mark.asyncio
    async def test_initialize_config(self) -> None:
        import config_service.src.settings as settings_module
        settings_module._manager = None
        mgr = await initialize_config(ConfigServiceSettings(hot_reload_enabled=False))
        assert isinstance(mgr, ConfigurationManager)
        assert mgr._initialized is True
        await mgr.shutdown()
