"""Comprehensive unit tests for Configuration Service - Settings Module.

This module provides exhaustive coverage for:
- Edge cases and boundary conditions
- Error handling and validation
- Concurrent access patterns
- Configuration loading edge cases
- Security considerations
"""
from __future__ import annotations
import asyncio
import json
import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from pydantic import SecretStr, ValidationError

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


class TestConfigEnvironmentEdgeCases:
    """Edge case tests for ConfigEnvironment."""

    def test_all_environment_values_unique(self) -> None:
        """Ensure all environment values are unique."""
        values = [e.value for e in ConfigEnvironment]
        assert len(values) == len(set(values))

    def test_environment_string_conversion(self) -> None:
        """Test string conversion works both ways."""
        for env in ConfigEnvironment:
            assert ConfigEnvironment(env.value) == env

    def test_environment_from_invalid_string_raises(self) -> None:
        """Test invalid environment string raises error."""
        with pytest.raises(ValueError):
            ConfigEnvironment("invalid_environment")

    def test_environment_case_sensitivity(self) -> None:
        """Test environment values are case-sensitive."""
        with pytest.raises(ValueError):
            ConfigEnvironment("PRODUCTION")  # Should be lowercase

    def test_environment_iteration(self) -> None:
        """Test all environments can be iterated."""
        expected = {"development", "staging", "production", "testing"}
        actual = {e.value for e in ConfigEnvironment}
        assert actual == expected


class TestConfigSourceComplete:
    """Complete tests for ConfigSource enum."""

    def test_all_source_values_exist(self) -> None:
        """Ensure all expected sources exist."""
        expected_sources = {"environment", "file", "secrets_manager", "vault", "remote", "default"}
        actual_sources = {s.value for s in ConfigSource}
        assert actual_sources == expected_sources

    def test_source_priority_implied_order(self) -> None:
        """Document implied source priority order."""
        # This documents expected override behavior
        priority_order = [
            ConfigSource.DEFAULT,
            ConfigSource.FILE,
            ConfigSource.ENVIRONMENT,
            ConfigSource.VAULT,
            ConfigSource.SECRETS_MANAGER,
            ConfigSource.REMOTE,
        ]
        assert len(priority_order) == len(ConfigSource)


class TestConfigMetadataComplete:
    """Complete tests for ConfigMetadata model."""

    def test_default_timestamp_is_utc(self) -> None:
        """Ensure default timestamp is UTC."""
        meta = ConfigMetadata()
        assert meta.loaded_at.tzinfo == timezone.utc

    def test_checksum_none_by_default(self) -> None:
        """Test checksum is None by default."""
        meta = ConfigMetadata()
        assert meta.checksum is None

    def test_sensitive_flag_default_false(self) -> None:
        """Test sensitive flag defaults to False."""
        meta = ConfigMetadata()
        assert meta.sensitive is False

    def test_metadata_with_all_fields(self) -> None:
        """Test metadata with all fields populated."""
        now = datetime.now(timezone.utc)
        meta = ConfigMetadata(
            source=ConfigSource.VAULT,
            loaded_at=now,
            version="2.1.0",
            checksum="sha256:abc123def456",
            ttl_seconds=600,
            sensitive=True,
        )
        assert meta.source == ConfigSource.VAULT
        assert meta.loaded_at == now
        assert meta.version == "2.1.0"
        assert meta.checksum == "sha256:abc123def456"
        assert meta.ttl_seconds == 600
        assert meta.sensitive is True

    def test_ttl_can_be_zero(self) -> None:
        """Test TTL can be zero (immediate expiry)."""
        meta = ConfigMetadata(ttl_seconds=0)
        assert meta.ttl_seconds == 0


class TestConfigValueExpiration:
    """Expiration tests for ConfigValue."""

    def test_value_expires_after_ttl(self) -> None:
        """Test value expires after TTL."""
        past = datetime.now(timezone.utc) - timedelta(seconds=100)
        meta = ConfigMetadata(loaded_at=past, ttl_seconds=50)
        cv = ConfigValue[str](key="test", value="value", metadata=meta)
        assert cv.is_expired is True

    def test_value_not_expired_within_ttl(self) -> None:
        """Test value not expired within TTL."""
        recent = datetime.now(timezone.utc) - timedelta(seconds=10)
        meta = ConfigMetadata(loaded_at=recent, ttl_seconds=100)
        cv = ConfigValue[str](key="test", value="value", metadata=meta)
        assert cv.is_expired is False

    def test_value_never_expires_without_ttl(self) -> None:
        """Test value never expires if TTL is None."""
        past = datetime.now(timezone.utc) - timedelta(days=365)
        meta = ConfigMetadata(loaded_at=past, ttl_seconds=None)
        cv = ConfigValue[str](key="test", value="value", metadata=meta)
        assert cv.is_expired is False

    def test_value_types_preserved(self) -> None:
        """Test different value types are preserved."""
        # Integer
        cv_int = ConfigValue[int](key="int_key", value=42)
        assert cv_int.value == 42
        assert isinstance(cv_int.value, int)

        # Float
        cv_float = ConfigValue[float](key="float_key", value=3.14)
        assert cv_float.value == 3.14

        # Boolean
        cv_bool = ConfigValue[bool](key="bool_key", value=True)
        assert cv_bool.value is True

        # Dict
        cv_dict = ConfigValue[dict](key="dict_key", value={"nested": "value"})
        assert cv_dict.value == {"nested": "value"}


class TestConfigServiceSettingsValidation:
    """Validation tests for ConfigServiceSettings."""

    def test_port_boundary_min(self) -> None:
        """Test port minimum boundary."""
        settings = ConfigServiceSettings(port=1)
        assert settings.port == 1

    def test_port_boundary_max(self) -> None:
        """Test port maximum boundary."""
        settings = ConfigServiceSettings(port=65535)
        assert settings.port == 65535

    def test_port_below_min_raises(self) -> None:
        """Test port below minimum raises error."""
        with pytest.raises(ValidationError):
            ConfigServiceSettings(port=0)

    def test_port_above_max_raises(self) -> None:
        """Test port above maximum raises error."""
        with pytest.raises(ValidationError):
            ConfigServiceSettings(port=65536)

    def test_all_valid_log_levels(self) -> None:
        """Test all valid log levels are accepted."""
        valid_levels = ["debug", "DEBUG", "info", "INFO", "warning", "WARNING",
                       "error", "ERROR", "critical", "CRITICAL"]
        for level in valid_levels:
            settings = ConfigServiceSettings(log_level=level)
            assert settings.log_level == level.upper()

    def test_cache_ttl_can_be_zero(self) -> None:
        """Test cache TTL can be zero (no caching)."""
        settings = ConfigServiceSettings(cache_ttl_seconds=0)
        assert settings.cache_ttl_seconds == 0

    def test_hot_reload_interval_minimum(self) -> None:
        """Test hot reload interval has minimum of 5."""
        settings = ConfigServiceSettings(hot_reload_interval_seconds=5)
        assert settings.hot_reload_interval_seconds == 5

    def test_hot_reload_interval_below_min_raises(self) -> None:
        """Test hot reload interval below minimum raises error."""
        with pytest.raises(ValidationError):
            ConfigServiceSettings(hot_reload_interval_seconds=4)

    def test_cors_origins_single(self) -> None:
        """Test single CORS origin."""
        settings = ConfigServiceSettings(cors_origins="http://localhost:3000")
        assert settings.cors_origins_list == ["http://localhost:3000"]

    def test_cors_origins_multiple(self) -> None:
        """Test multiple CORS origins."""
        origins = "http://localhost:3000, http://example.com , https://api.example.com"
        settings = ConfigServiceSettings(cors_origins=origins)
        expected = ["http://localhost:3000", "http://example.com", "https://api.example.com"]
        assert settings.cors_origins_list == expected

    def test_cors_origins_empty_handling(self) -> None:
        """Test empty CORS origins handling."""
        settings = ConfigServiceSettings(cors_origins=",,,")
        assert settings.cors_origins_list == []

    def test_vault_settings_defaults(self) -> None:
        """Test Vault settings have correct defaults."""
        settings = ConfigServiceSettings()
        assert settings.vault_enabled is False
        assert settings.vault_url == "http://localhost:8200"
        assert settings.vault_mount_path == "secret"

    def test_aws_secrets_defaults(self) -> None:
        """Test AWS Secrets settings have correct defaults."""
        settings = ConfigServiceSettings()
        assert settings.aws_secrets_enabled is False
        assert settings.aws_region == "us-east-1"


class TestDatabaseConfigComplete:
    """Complete tests for DatabaseConfig."""

    def test_dsn_postgresql_prefix(self) -> None:
        """Test DSN has postgresql prefix."""
        config = DatabaseConfig()
        dsn = config.get_dsn()
        assert dsn.startswith("postgresql://")

    def test_dsn_includes_all_components(self) -> None:
        """Test DSN includes all connection components."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            database="test_db",
            username="test_user",
            password=SecretStr("test_pass"),
        )
        dsn = config.get_dsn(hide_password=False)
        assert "db.example.com" in dsn
        assert "5433" in dsn
        assert "test_db" in dsn
        assert "test_user" in dsn
        assert "test_pass" in dsn

    def test_pool_size_boundaries(self) -> None:
        """Test pool size boundaries."""
        config_min = DatabaseConfig(pool_size=1)
        assert config_min.pool_size == 1

        config_max = DatabaseConfig(pool_size=100)
        assert config_max.pool_size == 100

    def test_pool_size_out_of_bounds_raises(self) -> None:
        """Test pool size out of bounds raises error."""
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=0)
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=101)

    def test_pool_max_overflow_boundaries(self) -> None:
        """Test pool max overflow boundaries."""
        config_min = DatabaseConfig(pool_max_overflow=0)
        assert config_min.pool_max_overflow == 0

        config_max = DatabaseConfig(pool_max_overflow=100)
        assert config_max.pool_max_overflow == 100

    def test_ssl_mode_default(self) -> None:
        """Test SSL mode default."""
        config = DatabaseConfig()
        assert config.ssl_mode == "prefer"


class TestRedisConfigComplete:
    """Complete tests for RedisConfig."""

    def test_db_boundaries(self) -> None:
        """Test Redis DB boundaries (0-15)."""
        config_min = RedisConfig(db=0)
        assert config_min.db == 0

        config_max = RedisConfig(db=15)
        assert config_max.db == 15

    def test_db_out_of_bounds_raises(self) -> None:
        """Test Redis DB out of bounds raises error."""
        with pytest.raises(ValidationError):
            RedisConfig(db=-1)
        with pytest.raises(ValidationError):
            RedisConfig(db=16)

    def test_password_optional(self) -> None:
        """Test Redis password is optional."""
        config = RedisConfig()
        assert config.password is None

    def test_ssl_default_false(self) -> None:
        """Test SSL defaults to False."""
        config = RedisConfig()
        assert config.ssl is False


class TestKafkaConfigComplete:
    """Complete tests for KafkaConfig."""

    def test_security_protocol_default(self) -> None:
        """Test default security protocol."""
        config = KafkaConfig()
        assert config.security_protocol == "PLAINTEXT"

    def test_sasl_settings_optional(self) -> None:
        """Test SASL settings are optional."""
        config = KafkaConfig()
        assert config.sasl_mechanism is None
        assert config.sasl_username is None
        assert config.sasl_password is None

    def test_consumer_group_id_default(self) -> None:
        """Test default consumer group ID."""
        config = KafkaConfig()
        assert config.consumer_group_id == "solace-consumers"

    def test_auto_offset_reset_default(self) -> None:
        """Test default auto offset reset."""
        config = KafkaConfig()
        assert config.auto_offset_reset == "earliest"


class TestWeaviateConfigComplete:
    """Complete tests for WeaviateConfig."""

    def test_grpc_port_boundaries(self) -> None:
        """Test gRPC port boundaries."""
        config_min = WeaviateConfig(grpc_port=1)
        assert config_min.grpc_port == 1

        config_max = WeaviateConfig(grpc_port=65535)
        assert config_max.grpc_port == 65535

    def test_batch_size_minimum(self) -> None:
        """Test batch size minimum."""
        config = WeaviateConfig(batch_size=1)
        assert config.batch_size == 1

    def test_batch_size_zero_raises(self) -> None:
        """Test batch size zero raises error."""
        with pytest.raises(ValidationError):
            WeaviateConfig(batch_size=0)

    def test_api_key_optional(self) -> None:
        """Test API key is optional."""
        config = WeaviateConfig()
        assert config.api_key is None


class TestLLMConfigComplete:
    """Complete tests for LLMConfig."""

    def test_temperature_boundaries(self) -> None:
        """Test temperature boundaries (0.0-2.0)."""
        config_min = LLMConfig(temperature=0.0)
        assert config_min.temperature == 0.0

        config_max = LLMConfig(temperature=2.0)
        assert config_max.temperature == 2.0

    def test_temperature_out_of_bounds_raises(self) -> None:
        """Test temperature out of bounds raises error."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_minimum(self) -> None:
        """Test max tokens minimum."""
        config = LLMConfig(max_tokens=1)
        assert config.max_tokens == 1

    def test_max_retries_can_be_zero(self) -> None:
        """Test max retries can be zero."""
        config = LLMConfig(max_retries=0)
        assert config.max_retries == 0

    def test_api_keys_optional(self) -> None:
        """Test API keys are optional."""
        config = LLMConfig()
        assert config.anthropic_api_key is None
        assert config.openai_api_key is None


class TestSecurityConfigComplete:
    """Complete tests for SecurityConfig."""

    def test_jwt_token_expire_minimum(self) -> None:
        """Test JWT token expire minimum."""
        config = SecurityConfig(jwt_access_token_expire_minutes=1)
        assert config.jwt_access_token_expire_minutes == 1

        config2 = SecurityConfig(jwt_refresh_token_expire_days=1)
        assert config2.jwt_refresh_token_expire_days == 1

    def test_phi_encryption_default_enabled(self) -> None:
        """Test PHI encryption enabled by default."""
        config = SecurityConfig()
        assert config.phi_encryption_enabled is True

    def test_audit_logging_default_enabled(self) -> None:
        """Test audit logging enabled by default."""
        config = SecurityConfig()
        assert config.audit_logging_enabled is True

    def test_rate_limit_boundaries(self) -> None:
        """Test rate limit boundaries."""
        config = SecurityConfig(rate_limit_requests=1, rate_limit_window_seconds=1)
        assert config.rate_limit_requests == 1
        assert config.rate_limit_window_seconds == 1


class TestObservabilityConfigComplete:
    """Complete tests for ObservabilityConfig."""

    def test_tracing_sample_rate_boundaries(self) -> None:
        """Test tracing sample rate boundaries."""
        config_min = ObservabilityConfig(tracing_sample_rate=0.0)
        assert config_min.tracing_sample_rate == 0.0

        config_max = ObservabilityConfig(tracing_sample_rate=1.0)
        assert config_max.tracing_sample_rate == 1.0

    def test_tracing_sample_rate_out_of_bounds_raises(self) -> None:
        """Test tracing sample rate out of bounds raises error."""
        with pytest.raises(ValidationError):
            ObservabilityConfig(tracing_sample_rate=-0.1)
        with pytest.raises(ValidationError):
            ObservabilityConfig(tracing_sample_rate=1.1)

    def test_profiling_default_disabled(self) -> None:
        """Test profiling disabled by default."""
        config = ObservabilityConfig()
        assert config.profiling_enabled is False


class TestApplicationConfigComplete:
    """Complete tests for ApplicationConfig."""

    def test_all_subconfigs_initialized(self) -> None:
        """Test all sub-configs are initialized."""
        config = ApplicationConfig()
        assert config.database is not None
        assert config.redis is not None
        assert config.kafka is not None
        assert config.weaviate is not None
        assert config.llm is not None
        assert config.security is not None
        assert config.observability is not None

    def test_custom_subconfigs(self) -> None:
        """Test custom sub-configs are applied."""
        custom_db = DatabaseConfig(host="custom-host", port=5433)
        config = ApplicationConfig(database=custom_db)
        assert config.database.host == "custom-host"
        assert config.database.port == 5433


class TestConfigurationManagerLifecycle:
    """Lifecycle tests for ConfigurationManager."""

    @pytest.fixture
    def manager(self) -> ConfigurationManager:
        """Create manager with hot reload disabled."""
        return ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))

    @pytest.mark.asyncio
    async def test_load_sets_initialized_flag(self, manager: ConfigurationManager) -> None:
        """Test load sets initialized flag."""
        await manager.load()
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_config_property_before_load_raises(self) -> None:
        """Test accessing config before load raises error."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            _ = manager.config

    @pytest.mark.asyncio
    async def test_shutdown_clears_cache(self, manager: ConfigurationManager) -> None:
        """Test shutdown clears cache."""
        await manager.load()
        manager._cache["test"] = ConfigValue(key="test", value="value")
        await manager.shutdown()
        assert len(manager._cache) == 0

    @pytest.mark.asyncio
    async def test_shutdown_cancels_reload_task(self) -> None:
        """Test shutdown cancels hot reload task."""
        manager = ConfigurationManager(ConfigServiceSettings(
            hot_reload_enabled=True,
            hot_reload_interval_seconds=5
        ))
        await manager.load()
        assert manager._reload_task is not None
        await manager.shutdown()
        assert manager._reload_task.cancelled() or manager._reload_task.done()


class TestConfigurationManagerCaching:
    """Caching tests for ConfigurationManager."""

    @pytest.fixture
    def manager(self) -> ConfigurationManager:
        return ConfigurationManager(ConfigServiceSettings(
            hot_reload_enabled=False,
            cache_ttl_seconds=60
        ))

    @pytest.mark.asyncio
    async def test_get_caches_value(self, manager: ConfigurationManager) -> None:
        """Test get caches value."""
        await manager.load()
        value1 = manager.get("environment")
        assert "environment" in manager._cache
        value2 = manager.get("environment")
        assert value1 == value2

    @pytest.mark.asyncio
    async def test_expired_cache_refetches(self, manager: ConfigurationManager) -> None:
        """Test expired cache refetches value."""
        await manager.load()
        manager.get("environment")
        # Manually expire the cache entry
        cache_entry = manager._cache["environment"]
        cache_entry.metadata = ConfigMetadata(
            loaded_at=datetime.now(timezone.utc) - timedelta(seconds=120),
            ttl_seconds=60
        )
        # Should refetch
        value = manager.get("environment")
        assert value == "development"


class TestConfigurationManagerMerging:
    """Configuration merging tests."""

    @pytest.mark.asyncio
    async def test_deep_merge_nested_dicts(self) -> None:
        """Test deep merge of nested dictionaries."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        base = {"database": {"host": "localhost", "port": 5432, "extra": {"nested": "value"}}}
        override = {"database": {"host": "custom-host", "extra": {"nested": "override"}}}
        result = manager._merge_config(base, override)
        assert result["database"]["host"] == "custom-host"
        assert result["database"]["port"] == 5432
        assert result["database"]["extra"]["nested"] == "override"

    @pytest.mark.asyncio
    async def test_merge_replaces_non_dict_values(self) -> None:
        """Test merge replaces non-dict values."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        base = {"value": [1, 2, 3]}
        override = {"value": [4, 5, 6]}
        result = manager._merge_config(base, override)
        assert result["value"] == [4, 5, 6]


class TestConfigurationManagerListeners:
    """Listener tests for ConfigurationManager."""

    @pytest.fixture
    def manager(self) -> ConfigurationManager:
        return ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))

    @pytest.mark.asyncio
    async def test_listener_called_on_change(self, manager: ConfigurationManager) -> None:
        """Test listener called on configuration change."""
        callback_events = []

        async def listener(event: str, data) -> None:
            callback_events.append(event)

        manager.register_listener(listener)
        await manager.load()
        # Force a checksum change
        manager._checksum = "old_checksum"
        await manager.reload()
        assert "config_reloaded" in callback_events

    @pytest.mark.asyncio
    async def test_multiple_listeners_called(self, manager: ConfigurationManager) -> None:
        """Test multiple listeners are all called."""
        call_counts = {"listener1": 0, "listener2": 0}

        async def listener1(event: str, data) -> None:
            call_counts["listener1"] += 1

        async def listener2(event: str, data) -> None:
            call_counts["listener2"] += 1

        manager.register_listener(listener1)
        manager.register_listener(listener2)
        await manager.load()
        manager._checksum = "old_checksum"
        await manager.reload()
        assert call_counts["listener1"] == 1
        assert call_counts["listener2"] == 1


class TestConfigurationManagerFileLoading:
    """File loading tests for ConfigurationManager."""

    @pytest.mark.asyncio
    async def test_load_base_config_file(self) -> None:
        """Test loading base config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)
            config_file = config_path / "config.json"
            config_file.write_text(json.dumps({
                "database": {"host": "file-host"},
                "redis": {"port": 6380}
            }))
            manager = ConfigurationManager(ConfigServiceSettings(
                config_path=str(config_path),
                hot_reload_enabled=False
            ))
            await manager.load()
            assert manager.get_section("database")["host"] == "file-host"
            assert manager.get_section("redis")["port"] == 6380

    @pytest.mark.asyncio
    async def test_load_environment_specific_file(self) -> None:
        """Test loading environment-specific config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)
            base_file = config_path / "config.json"
            base_file.write_text(json.dumps({"database": {"host": "base-host"}}))
            env_file = config_path / "config.testing.json"
            env_file.write_text(json.dumps({"database": {"host": "testing-host"}}))
            manager = ConfigurationManager(ConfigServiceSettings(
                config_path=str(config_path),
                environment=ConfigEnvironment.TESTING,
                hot_reload_enabled=False
            ))
            await manager.load()
            assert manager.get_section("database")["host"] == "testing-host"

    @pytest.mark.asyncio
    async def test_load_local_override_file(self) -> None:
        """Test loading local override file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)
            base_file = config_path / "config.json"
            base_file.write_text(json.dumps({"database": {"host": "base-host"}}))
            local_file = config_path / "config.local.json"
            local_file.write_text(json.dumps({"database": {"host": "local-host"}}))
            manager = ConfigurationManager(ConfigServiceSettings(
                config_path=str(config_path),
                hot_reload_enabled=False
            ))
            await manager.load()
            assert manager.get_section("database")["host"] == "local-host"

    @pytest.mark.asyncio
    async def test_invalid_json_file_handled(self) -> None:
        """Test invalid JSON file is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)
            config_file = config_path / "config.json"
            config_file.write_text("{invalid json")
            manager = ConfigurationManager(ConfigServiceSettings(
                config_path=str(config_path),
                hot_reload_enabled=False
            ))
            # Should not raise, just log error and continue
            await manager.load()
            assert manager._initialized is True


class TestConfigurationManagerRuntimeSet:
    """Runtime set tests for ConfigurationManager."""

    @pytest.fixture
    def manager(self) -> ConfigurationManager:
        return ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))

    @pytest.mark.asyncio
    async def test_set_updates_cache(self, manager: ConfigurationManager) -> None:
        """Test set updates cache."""
        await manager.load()
        manager.set("database.host", "new-host")
        assert manager._cache["database.host"].value == "new-host"

    @pytest.mark.asyncio
    async def test_set_invalid_key_format_raises(self, manager: ConfigurationManager) -> None:
        """Test set with invalid key format raises error."""
        await manager.load()
        with pytest.raises(ValueError, match="must be in format"):
            manager.set("invalid", "value")


class TestConfigurationManagerChecksum:
    """Checksum tests for ConfigurationManager."""

    def test_checksum_consistent_for_same_data(self) -> None:
        """Test checksum is consistent for same data."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        data = {"key": "value", "nested": {"inner": 123}}
        checksum1 = manager._compute_checksum(data)
        checksum2 = manager._compute_checksum(data)
        assert checksum1 == checksum2

    def test_checksum_different_for_different_data(self) -> None:
        """Test checksum differs for different data."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        checksum1 = manager._compute_checksum(data1)
        checksum2 = manager._compute_checksum(data2)
        assert checksum1 != checksum2

    def test_checksum_order_independent(self) -> None:
        """Test checksum is order-independent for dict keys."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}
        checksum1 = manager._compute_checksum(data1)
        checksum2 = manager._compute_checksum(data2)
        assert checksum1 == checksum2


class TestConfigurationManagerConcurrency:
    """Concurrency tests for ConfigurationManager."""

    @pytest.mark.asyncio
    async def test_concurrent_loads_use_lock(self) -> None:
        """Test concurrent loads use lock correctly."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))

        async def load_config():
            await manager.load()
            return manager._initialized

        # Run multiple concurrent loads
        results = await asyncio.gather(*[load_config() for _ in range(5)])
        assert all(results)

    @pytest.mark.asyncio
    async def test_concurrent_get_operations(self) -> None:
        """Test concurrent get operations work correctly."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        await manager.load()

        async def get_value():
            return manager.get("environment")

        results = await asyncio.gather(*[get_value() for _ in range(10)])
        assert all(r == "development" for r in results)


class TestModuleLevelFunctions:
    """Tests for module-level functions."""

    def test_get_config_manager_creates_singleton(self) -> None:
        """Test get_config_manager creates singleton."""
        import config_service.src.settings as settings_module
        settings_module._manager = None
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        assert manager1 is manager2
        settings_module._manager = None

    @pytest.mark.asyncio
    async def test_initialize_config_creates_and_loads(self) -> None:
        """Test initialize_config creates and loads manager."""
        import config_service.src.settings as settings_module
        settings_module._manager = None
        manager = await initialize_config(ConfigServiceSettings(hot_reload_enabled=False))
        assert manager._initialized is True
        await manager.shutdown()
        settings_module._manager = None


class TestEdgeCasesAndErrorHandling:
    """Edge cases and error handling tests."""

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_key_returns_default(self) -> None:
        """Test resolving nonexistent key returns default."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        await manager.load()
        result = manager.get("nonexistent.deep.path", "default_value")
        assert result == "default_value"

    @pytest.mark.asyncio
    async def test_get_section_returns_empty_for_nonexistent(self) -> None:
        """Test get_section returns empty dict for nonexistent section."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        await manager.load()
        section = manager.get_section("nonexistent_section")
        assert section == {}

    @pytest.mark.asyncio
    async def test_resolve_key_with_dict_intermediate(self) -> None:
        """Test resolving key through dict intermediate."""
        manager = ConfigurationManager(ConfigServiceSettings(hot_reload_enabled=False))
        await manager.load()
        # Test that the resolution handles both BaseModel and dict types
        result = manager.get("database.host")
        assert result == "localhost"

    def test_settings_property_returns_settings(self) -> None:
        """Test settings property returns settings object."""
        settings = ConfigServiceSettings(port=9999)
        manager = ConfigurationManager(settings)
        assert manager.settings.port == 9999

    def test_environment_property(self) -> None:
        """Test environment property."""
        manager = ConfigurationManager(ConfigServiceSettings(
            environment=ConfigEnvironment.PRODUCTION
        ))
        assert manager.environment == ConfigEnvironment.PRODUCTION

    def test_is_production_property(self) -> None:
        """Test is_production property."""
        dev_manager = ConfigurationManager(ConfigServiceSettings(
            environment=ConfigEnvironment.DEVELOPMENT
        ))
        assert dev_manager.is_production is False

        prod_manager = ConfigurationManager(ConfigServiceSettings(
            environment=ConfigEnvironment.PRODUCTION
        ))
        assert prod_manager.is_production is True
