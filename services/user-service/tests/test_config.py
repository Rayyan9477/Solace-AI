"""
Unit tests for User Service configuration.

Tests cover configuration loading, validation, and defaults.
"""
import pytest
import os
from unittest.mock import patch

from src.config import (
    DatabaseConfig,
    RedisConfig,
    SecurityConfig,
    ServiceConfig,
    KafkaConfig,
    UserServiceSettings,
)


class TestDatabaseConfig:
    """Test cases for DatabaseConfig."""

    def test_database_config_defaults(self):
        """Test database config with default values."""
        with patch.dict(os.environ, {"DB_PASSWORD": "test_password"}, clear=True):
            config = DatabaseConfig()

            assert config.host == "localhost"
            assert config.port == 5432
            assert config.name == "solace_users"
            assert config.user == "postgres"
            assert config.pool_size == 10

    def test_database_url_generation(self):
        """Test database URL generation."""
        with patch.dict(os.environ, {"DB_PASSWORD": "test_password"}, clear=True):
            config = DatabaseConfig()

            url = config.url

            assert "postgresql+asyncpg://" in url
            assert "test_password" in url
            assert "solace_users" in url

    def test_database_config_from_env(self):
        """Test database config loaded from environment variables."""
        with patch.dict(os.environ, {
            "DB_HOST": "db.example.com",
            "DB_PORT": "5433",
            "DB_NAME": "custom_db",
            "DB_USER": "custom_user",
            "DB_PASSWORD": "custom_password",
        }, clear=True):
            config = DatabaseConfig()

            assert config.host == "db.example.com"
            assert config.port == 5433
            assert config.name == "custom_db"
            assert config.user == "custom_user"


class TestRedisConfig:
    """Test cases for RedisConfig."""

    def test_redis_config_defaults(self):
        """Test redis config with default values."""
        config = RedisConfig()

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None

    def test_redis_url_generation(self):
        """Test redis URL generation."""
        config = RedisConfig()

        url = config.url

        assert url == "redis://localhost:6379/0"

    def test_redis_url_with_password(self):
        """Test redis URL generation with password."""
        with patch.dict(os.environ, {"REDIS_PASSWORD": "secret"}, clear=True):
            config = RedisConfig()

            url = config.url

            assert ":secret@" in url


class TestSecurityConfig:
    """Test cases for SecurityConfig."""

    def test_security_config_requires_jwt_secret(self):
        """Test that security config requires JWT secret."""
        with pytest.raises(Exception):
            SecurityConfig()

    def test_security_config_with_valid_jwt_secret(self):
        """Test security config with valid JWT secret."""
        with patch.dict(os.environ, {"SECURITY_JWT_SECRET": "a" * 32}, clear=True):
            config = SecurityConfig()

            assert config.jwt_secret == "a" * 32
            assert config.jwt_algorithm == "HS256"
            assert config.jwt_expiry_minutes == 60

    def test_security_config_validates_jwt_secret_length(self):
        """Test that JWT secret must be at least 32 characters."""
        with pytest.raises(ValueError, match="at least 32 characters"):
            with patch.dict(os.environ, {"SECURITY_JWT_SECRET": "short"}, clear=True):
                SecurityConfig()

    def test_security_config_defaults(self):
        """Test security config defaults."""
        with patch.dict(os.environ, {"SECURITY_JWT_SECRET": "a" * 32}, clear=True):
            config = SecurityConfig()

            assert config.password_min_length == 8
            assert config.password_require_special is True
            assert config.max_login_attempts == 5
            assert config.lockout_duration_minutes == 30


class TestServiceConfig:
    """Test cases for ServiceConfig."""

    def test_service_config_defaults(self):
        """Test service config with default values."""
        config = ServiceConfig()

        assert config.name == "user-service"
        assert config.env == "development"
        assert config.port == 8001
        assert config.host == "0.0.0.0"
        assert config.log_level == "INFO"

    def test_is_production_check(self):
        """Test production environment check."""
        config = ServiceConfig()

        assert config.is_production() is False

        with patch.dict(os.environ, {"USER_SERVICE_ENV": "production"}, clear=True):
            prod_config = ServiceConfig()
            assert prod_config.is_production() is True

    def test_is_development_check(self):
        """Test development environment check."""
        config = ServiceConfig()

        assert config.is_development() is True


class TestKafkaConfig:
    """Test cases for KafkaConfig."""

    def test_kafka_config_defaults(self):
        """Test kafka config with default values."""
        config = KafkaConfig()

        assert config.bootstrap_servers == "localhost:9092"
        assert config.topic_users == "solace.users"
        assert config.producer_acks == "all"
        assert config.compression_type == "gzip"
        assert config.enable is True

    def test_kafka_config_from_env(self):
        """Test kafka config loaded from environment variables."""
        with patch.dict(os.environ, {
            "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
            "KAFKA_TOPIC_USERS": "custom.topic",
            "KAFKA_ENABLE": "false",
        }, clear=True):
            config = KafkaConfig()

            assert config.bootstrap_servers == "kafka:9092"
            assert config.topic_users == "custom.topic"
            assert config.enable is False


class TestUserServiceSettings:
    """Test cases for UserServiceSettings."""

    def test_user_service_settings_loads_all_configs(self):
        """Test that user service settings loads all nested configs."""
        with patch.dict(os.environ, {
            "DB_PASSWORD": "test_password",
            "SECURITY_JWT_SECRET": "a" * 32,
        }, clear=True):
            settings = UserServiceSettings()

            assert settings.database is not None
            assert settings.redis is not None
            assert settings.security is not None
            assert settings.service is not None
            assert settings.kafka is not None

    def test_user_service_settings_load_method(self):
        """Test user service settings load method."""
        with patch.dict(os.environ, {
            "DB_PASSWORD": "test_password",
            "SECURITY_JWT_SECRET": "a" * 32,
        }, clear=True):
            settings = UserServiceSettings.load()

            assert settings is not None
            assert isinstance(settings, UserServiceSettings)
