"""
Unit tests for Service Configuration.
Tests TherapyServiceSettings and related configuration classes.
"""
from __future__ import annotations
import os
import pytest

from services.therapy_service.src.config import (
    TherapyServiceSettings, DatabaseSettings, RedisSettings,
    LLMSettings, SessionSettings, TreatmentSettings,
    SafetySettings, ObservabilitySettings,
    get_settings, reset_settings,
)


class TestDatabaseSettings:
    """Tests for DatabaseSettings."""

    def test_default_settings(self) -> None:
        """Test default database settings."""
        settings = DatabaseSettings()
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.database == "solace_therapy"
        assert settings.pool_size == 10

    def test_connection_string(self) -> None:
        """Test connection string generation."""
        settings = DatabaseSettings(
            host="db.example.com",
            port=5433,
            database="therapy_db",
            user="app_user",
            password="secret",
        )
        conn_str = settings.connection_string
        assert "postgresql+asyncpg://" in conn_str
        assert "app_user:secret" in conn_str
        assert "db.example.com:5433" in conn_str
        assert "therapy_db" in conn_str


class TestRedisSettings:
    """Tests for RedisSettings."""

    def test_default_settings(self) -> None:
        """Test default Redis settings."""
        settings = RedisSettings()
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.db == 0
        assert settings.ssl is False

    def test_url_without_password(self) -> None:
        """Test Redis URL without password."""
        settings = RedisSettings(host="redis.example.com", port=6380)
        url = settings.url
        assert url == "redis://redis.example.com:6380/0"

    def test_url_with_password(self) -> None:
        """Test Redis URL with password."""
        settings = RedisSettings(password="secret")
        url = settings.url
        assert ":secret@" in url

    def test_url_with_ssl(self) -> None:
        """Test Redis URL with SSL."""
        settings = RedisSettings(ssl=True)
        url = settings.url
        assert url.startswith("rediss://")


class TestLLMSettings:
    """Tests for LLMSettings."""

    def test_default_settings(self) -> None:
        """Test default LLM settings."""
        settings = LLMSettings()
        assert settings.provider == "anthropic"
        assert settings.temperature == 0.7
        assert settings.max_tokens == 2048

    def test_provider_validation(self) -> None:
        """Test provider validation."""
        settings = LLMSettings(provider="OpenAI")
        assert settings.provider == "openai"

    def test_invalid_provider(self) -> None:
        """Test invalid provider rejected."""
        with pytest.raises(ValueError):
            LLMSettings(provider="invalid_provider")


class TestSessionSettings:
    """Tests for SessionSettings."""

    def test_default_settings(self) -> None:
        """Test default session settings."""
        settings = SessionSettings()
        assert settings.max_duration_minutes == 60
        assert settings.enable_flexible_transitions is True
        assert settings.min_engagement_score == 0.3

    def test_constraints(self) -> None:
        """Test setting constraints."""
        settings = SessionSettings(
            max_duration_minutes=90,
            min_engagement_score=0.5,
        )
        assert settings.max_duration_minutes == 90
        assert settings.min_engagement_score == 0.5


class TestTreatmentSettings:
    """Tests for TreatmentSettings."""

    def test_default_settings(self) -> None:
        """Test default treatment settings."""
        settings = TreatmentSettings()
        assert settings.default_sessions_per_phase == 4
        assert settings.enable_stepped_care is True
        assert settings.goal_review_interval_days == 14

    def test_custom_settings(self) -> None:
        """Test custom treatment settings."""
        settings = TreatmentSettings(
            default_sessions_per_phase=6,
            enable_auto_advancement=False,
        )
        assert settings.default_sessions_per_phase == 6
        assert settings.enable_auto_advancement is False


class TestSafetySettings:
    """Tests for SafetySettings."""

    def test_default_settings(self) -> None:
        """Test default safety settings."""
        settings = SafetySettings()
        assert settings.enable_risk_monitoring is True
        assert settings.crisis_escalation_enabled is True
        assert settings.crisis_hotline_number == "988"

    def test_custom_settings(self) -> None:
        """Test custom safety settings."""
        settings = SafetySettings(
            risk_check_interval_messages=5,
            auto_pause_on_high_risk=False,
        )
        assert settings.risk_check_interval_messages == 5
        assert settings.auto_pause_on_high_risk is False


class TestObservabilitySettings:
    """Tests for ObservabilitySettings."""

    def test_default_settings(self) -> None:
        """Test default observability settings."""
        settings = ObservabilitySettings()
        assert settings.log_level == "INFO"
        assert settings.enable_tracing is True
        assert settings.trace_sample_rate == 0.1

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        settings = ObservabilitySettings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_invalid_log_level(self) -> None:
        """Test invalid log level rejected."""
        with pytest.raises(ValueError):
            ObservabilitySettings(log_level="invalid")


class TestTherapyServiceSettings:
    """Tests for TherapyServiceSettings."""

    def test_default_settings(self) -> None:
        """Test default service settings."""
        settings = TherapyServiceSettings()
        assert settings.service_name == "therapy-service"
        assert settings.environment == "development"
        assert settings.port == 8002

    def test_nested_settings(self) -> None:
        """Test nested settings access."""
        settings = TherapyServiceSettings()
        assert settings.database.host == "localhost"
        assert settings.redis.port == 6379
        assert settings.llm.provider == "anthropic"
        assert settings.session.max_duration_minutes == 60
        assert settings.treatment.enable_stepped_care is True
        assert settings.safety.enable_risk_monitoring is True

    def test_environment_validation(self) -> None:
        """Test environment validation."""
        settings = TherapyServiceSettings(environment="Production")
        assert settings.environment == "production"

    def test_invalid_environment(self) -> None:
        """Test invalid environment rejected."""
        with pytest.raises(ValueError):
            TherapyServiceSettings(environment="invalid")

    def test_is_production(self) -> None:
        """Test is_production property."""
        dev_settings = TherapyServiceSettings(environment="development")
        prod_settings = TherapyServiceSettings(environment="production")
        assert dev_settings.is_production is False
        assert prod_settings.is_production is True

    def test_is_debug(self) -> None:
        """Test is_debug property."""
        dev_settings = TherapyServiceSettings(environment="development", debug=True)
        prod_settings = TherapyServiceSettings(environment="production", debug=True)
        assert dev_settings.is_debug is True
        assert prod_settings.is_debug is False

    def test_to_dict(self) -> None:
        """Test configuration export."""
        settings = TherapyServiceSettings()
        data = settings.to_dict()
        assert "service_name" in data
        assert "service_version" in data
        assert "environment" in data
        assert "session" in data
        assert "treatment" in data
        assert "safety" in data
        assert "database" not in data
        assert "password" not in str(data)

    def test_cors_origins_default(self) -> None:
        """Test default CORS origins."""
        settings = TherapyServiceSettings()
        assert "*" in settings.cors_origins


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_singleton(self) -> None:
        """Test settings singleton pattern."""
        reset_settings()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reset_settings(self) -> None:
        """Test settings reset."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        assert settings1 is not settings2
