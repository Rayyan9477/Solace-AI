"""
Unit tests for Solace-AI Safety Service Configuration.
Tests centralized configuration management.
"""
from __future__ import annotations
import pytest
from decimal import Decimal
from services.safety_service.src.config import (
    SafetyServiceConfig, CrisisDetectionConfig, EscalationConfig,
    SafetyMonitoringConfig, SafetyRepositoryConfig, SafetyEventsConfig,
    SafetyConfig, get_safety_config, reset_config,
)


class TestSafetyServiceConfig:
    """Tests for SafetyServiceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SafetyServiceConfig()
        assert config.service_name == "safety-service"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.port == 8001

    def test_cors_origins_list_wildcard(self) -> None:
        """Test CORS origins list with wildcard."""
        config = SafetyServiceConfig(cors_origins="*")
        assert config.cors_origins_list == ["*"]

    def test_cors_origins_list_multiple(self) -> None:
        """Test CORS origins list with multiple values."""
        config = SafetyServiceConfig(cors_origins="http://localhost,https://app.com")
        assert len(config.cors_origins_list) == 2
        assert "http://localhost" in config.cors_origins_list

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        config = SafetyServiceConfig(log_level="debug")
        assert config.log_level == "DEBUG"

    def test_invalid_log_level(self) -> None:
        """Test invalid log level raises error."""
        with pytest.raises(ValueError, match="Invalid log level"):
            SafetyServiceConfig(log_level="invalid")


class TestCrisisDetectionConfig:
    """Tests for CrisisDetectionConfig."""

    def test_default_layers_enabled(self) -> None:
        """Test all detection layers enabled by default."""
        config = CrisisDetectionConfig()
        assert config.enable_layer_1 is True
        assert config.enable_layer_2 is True
        assert config.enable_layer_3 is True
        assert config.enable_layer_4 is True

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        config = CrisisDetectionConfig()
        assert config.low_threshold == Decimal("0.3")
        assert config.elevated_threshold == Decimal("0.5")
        assert config.high_threshold == Decimal("0.7")
        assert config.critical_threshold == Decimal("0.9")

    def test_default_weights(self) -> None:
        """Test default weight values."""
        config = CrisisDetectionConfig()
        assert config.keyword_weight == Decimal("0.4")
        assert config.pattern_weight == Decimal("0.25")
        assert config.sentiment_weight == Decimal("0.2")
        assert config.history_weight == Decimal("0.15")

    def test_custom_thresholds(self) -> None:
        """Test custom threshold configuration."""
        config = CrisisDetectionConfig(
            low_threshold=Decimal("0.25"),
            critical_threshold=Decimal("0.85"),
        )
        assert config.low_threshold == Decimal("0.25")
        assert config.critical_threshold == Decimal("0.85")


class TestEscalationConfig:
    """Tests for EscalationConfig."""

    def test_default_auto_escalate(self) -> None:
        """Test auto escalate defaults."""
        config = EscalationConfig()
        assert config.auto_escalate_critical is True
        assert config.auto_escalate_high is True

    def test_default_sla_times(self) -> None:
        """Test default SLA response times."""
        config = EscalationConfig()
        assert config.critical_response_sla_minutes == 5
        assert config.high_response_sla_minutes == 15
        assert config.medium_response_sla_minutes == 60
        assert config.low_response_sla_minutes == 240

    def test_notification_settings(self) -> None:
        """Test notification settings."""
        config = EscalationConfig()
        assert config.enable_sms_notifications is True
        assert config.enable_email_notifications is True
        assert config.enable_pager_notifications is True

    def test_clinician_pool_size(self) -> None:
        """Test clinician pool size."""
        config = EscalationConfig()
        assert config.on_call_clinician_pool_size == 3


class TestSafetyMonitoringConfig:
    """Tests for SafetyMonitoringConfig."""

    def test_default_checks_enabled(self) -> None:
        """Test checks enabled by default."""
        config = SafetyMonitoringConfig()
        assert config.enable_pre_check is True
        assert config.enable_post_check is True
        assert config.enable_continuous_monitoring is True

    def test_default_window_settings(self) -> None:
        """Test default window settings."""
        config = SafetyMonitoringConfig()
        assert config.trajectory_window_size == 5
        assert config.max_history_messages == 20

    def test_default_thresholds(self) -> None:
        """Test default thresholds."""
        config = SafetyMonitoringConfig()
        assert config.deterioration_threshold == Decimal("0.6")
        assert config.safe_response_threshold == Decimal("0.3")

    def test_cache_settings(self) -> None:
        """Test cache settings."""
        config = SafetyMonitoringConfig()
        assert config.cache_assessments is True
        assert config.assessment_cache_ttl_seconds == 300


class TestSafetyRepositoryConfig:
    """Tests for SafetyRepositoryConfig."""

    def test_default_storage_backend(self) -> None:
        """Test default storage backend."""
        config = SafetyRepositoryConfig()
        assert config.storage_backend == "memory"

    def test_default_retention_days(self) -> None:
        """Test default retention periods."""
        config = SafetyRepositoryConfig()
        assert config.assessment_retention_days == 365
        assert config.incident_retention_days == 730
        assert config.safety_plan_retention_days == 1825

    def test_audit_logging_enabled(self) -> None:
        """Test audit logging enabled by default."""
        config = SafetyRepositoryConfig()
        assert config.enable_audit_logging is True


class TestSafetyEventsConfig:
    """Tests for SafetyEventsConfig."""

    def test_default_kafka_settings(self) -> None:
        """Test default Kafka settings."""
        config = SafetyEventsConfig()
        assert config.kafka_bootstrap_servers == "localhost:9092"
        assert config.kafka_topic_prefix == "solace.safety"

    def test_event_publishing_enabled(self) -> None:
        """Test event publishing enabled by default."""
        config = SafetyEventsConfig()
        assert config.enable_event_publishing is True

    def test_batch_settings(self) -> None:
        """Test batch settings."""
        config = SafetyEventsConfig()
        assert config.event_batch_size == 100
        assert config.event_flush_interval_ms == 1000

    def test_retry_settings(self) -> None:
        """Test retry settings."""
        config = SafetyEventsConfig()
        assert config.enable_dead_letter_queue is True
        assert config.max_retry_attempts == 3


class TestSafetyConfig:
    """Tests for aggregate SafetyConfig."""

    def test_load_config(self) -> None:
        """Test loading configuration."""
        config = SafetyConfig.load()
        assert config.service is not None
        assert config.crisis_detection is not None
        assert config.escalation is not None
        assert config.monitoring is not None
        assert config.repository is not None
        assert config.events is not None

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        config = SafetyConfig.load()
        data = config.to_dict()
        assert "service" in data
        assert "crisis_detection" in data
        assert "escalation" in data

    def test_to_dict_hides_secrets(self) -> None:
        """Test that secrets are hidden in dictionary output."""
        config = SafetyConfig.load()
        config.repository.postgres_dsn = "secret_connection_string"
        data = config.to_dict(hide_secrets=True)
        assert data["repository"]["postgres_dsn"] == "***"


class TestConfigSingleton:
    """Tests for configuration singleton."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        reset_config()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        reset_config()

    def test_get_safety_config_singleton(self) -> None:
        """Test singleton returns same instance."""
        config1 = get_safety_config()
        config2 = get_safety_config()
        assert config1 is config2

    def test_reset_config(self) -> None:
        """Test resetting configuration."""
        config1 = get_safety_config()
        reset_config()
        config2 = get_safety_config()
        assert config1 is not config2
