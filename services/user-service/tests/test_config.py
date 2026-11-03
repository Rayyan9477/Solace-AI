"""
Unit tests for User Service configuration.

Tests cover configuration loading, validation, and defaults.
"""
import pytest
import os
from unittest.mock import patch

from src.config import (
    SecurityConfig,
    ServiceConfig,
    KafkaConfig,
    UserServiceSettings,
)


class TestSecurityConfig:
    """Test cases for SecurityConfig."""

    def test_security_config_requires_jwt_secret(self):
        """Test that security config requires JWT secret."""
        with patch.dict(os.environ, {}, clear=True):
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
            "USER_KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
            "USER_KAFKA_TOPIC_USERS": "custom.topic",
            "USER_KAFKA_ENABLE": "false",
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
            "SECURITY_JWT_SECRET": "b" * 32,
            "USER_SERVICE_JWT_SECRET_KEY": "c" * 32,
            "USER_SERVICE_FIELD_ENCRYPTION_KEY": "d" * 32,
        }, clear=True):
            settings = UserServiceSettings()

            assert settings.security is not None
            assert settings.service is not None
            assert settings.kafka is not None

    def test_user_service_settings_load_method(self):
        """Test user service settings load method."""
        with patch.dict(os.environ, {
            "SECURITY_JWT_SECRET": "b" * 32,
            "USER_SERVICE_JWT_SECRET_KEY": "c" * 32,
            "USER_SERVICE_FIELD_ENCRYPTION_KEY": "d" * 32,
        }, clear=True):
            settings = UserServiceSettings.load()

            assert settings is not None
            assert isinstance(settings, UserServiceSettings)
