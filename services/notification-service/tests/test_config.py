"""
Unit tests for notification service configuration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from pydantic import ValidationError

from config import (
    Environment,
    ServiceConfiguration,
    EmailChannelConfig,
    SMSChannelConfig,
    PushChannelConfig,
    TemplateConfig,
    QueueConfig,
    RateLimitConfig,
    ObservabilityConfig,
    NotificationServiceConfig,
    get_config,
    reset_config,
)


class TestEnvironment:
    """Tests for Environment enum."""

    def test_all_environments_exist(self):
        """Test all expected environments exist."""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"


class TestServiceConfiguration:
    """Tests for ServiceConfiguration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ServiceConfiguration()

        assert config.name == "notification-service"
        assert config.version == "1.0.0"
        assert config.env == Environment.DEVELOPMENT
        assert config.host == "0.0.0.0"
        assert config.port == 8003
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.workers == 1

    def test_port_validation(self):
        """Test port validation."""
        with pytest.raises(ValidationError):
            ServiceConfiguration(port=0)

        with pytest.raises(ValidationError):
            ServiceConfiguration(port=70000)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ServiceConfiguration(
            name="custom-service",
            env=Environment.PRODUCTION,
            port=9000,
            log_level="ERROR",
            workers=4,
        )

        assert config.name == "custom-service"
        assert config.env == Environment.PRODUCTION
        assert config.port == 9000
        assert config.workers == 4


class TestEmailChannelConfig:
    """Tests for EmailChannelConfig."""

    def test_default_values(self):
        """Test default email configuration."""
        config = EmailChannelConfig()

        assert config.enabled is True
        assert config.smtp_host == "localhost"
        assert config.smtp_port == 587
        assert config.use_tls is True
        assert config.from_email == "noreply@solace-ai.com"
        assert config.max_retries == 3
        assert config.timeout_seconds == 30.0

    def test_email_validation(self):
        """Test email validation."""
        with pytest.raises(ValidationError):
            EmailChannelConfig(from_email="invalid-email")

    def test_custom_values(self):
        """Test custom email configuration."""
        config = EmailChannelConfig(
            smtp_host="mail.example.com",
            smtp_port=465,
            use_ssl=True,
            from_email="alerts@example.com",
            batch_size=50,
        )

        assert config.smtp_host == "mail.example.com"
        assert config.smtp_port == 465
        assert config.from_email == "alerts@example.com"


class TestSMSChannelConfig:
    """Tests for SMSChannelConfig."""

    def test_default_values(self):
        """Test default SMS configuration."""
        config = SMSChannelConfig()

        assert config.enabled is False
        assert config.provider == "twilio"
        assert config.max_message_length == 1600

    def test_custom_values(self):
        """Test custom SMS configuration."""
        config = SMSChannelConfig(
            enabled=True,
            provider="aws_sns",
            account_sid="test_sid",
            from_number="+15551234567",
        )

        assert config.enabled is True
        assert config.provider == "aws_sns"


class TestPushChannelConfig:
    """Tests for PushChannelConfig."""

    def test_default_values(self):
        """Test default push configuration."""
        config = PushChannelConfig()

        assert config.enabled is False
        assert config.provider == "firebase"
        assert config.ttl_seconds == 86400

    def test_custom_values(self):
        """Test custom push configuration."""
        config = PushChannelConfig(
            enabled=True,
            provider="apns",
            project_id="my-project",
            ttl_seconds=3600,
        )

        assert config.enabled is True
        assert config.provider == "apns"
        assert config.ttl_seconds == 3600


class TestTemplateConfig:
    """Tests for TemplateConfig."""

    def test_default_values(self):
        """Test default template configuration."""
        config = TemplateConfig()

        assert config.templates_dir == "templates"
        assert config.cache_enabled is True
        assert config.default_locale == "en"
        assert "en" in config.supported_locales

    def test_custom_values(self):
        """Test custom template configuration."""
        config = TemplateConfig(
            templates_dir="/custom/templates",
            cache_ttl_seconds=7200,
            supported_locales=["en", "de", "ja"],
        )

        assert config.templates_dir == "/custom/templates"
        assert config.cache_ttl_seconds == 7200
        assert len(config.supported_locales) == 3


class TestQueueConfig:
    """Tests for QueueConfig."""

    def test_default_values(self):
        """Test default queue configuration."""
        config = QueueConfig()

        assert config.enabled is False
        assert config.provider == "redis"
        assert config.queue_name == "notifications"
        assert config.batch_size == 100

    def test_custom_values(self):
        """Test custom queue configuration."""
        config = QueueConfig(
            enabled=True,
            provider="rabbitmq",
            batch_size=50,
            max_queue_size=50000,
        )

        assert config.enabled is True
        assert config.provider == "rabbitmq"


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_values(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()

        assert config.enabled is True
        assert config.global_rate_per_minute == 1000
        assert config.per_user_rate_per_minute == 60
        assert config.burst_multiplier == 1.5

    def test_per_channel_rates(self):
        """Test per-channel rate limits."""
        config = RateLimitConfig()

        assert config.per_channel_rate_per_minute["email"] == 100
        assert config.per_channel_rate_per_minute["sms"] == 30
        assert config.per_channel_rate_per_minute["push"] == 200


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_default_values(self):
        """Test default observability configuration."""
        config = ObservabilityConfig()

        assert config.prometheus_enabled is True
        assert config.prometheus_endpoint == "/metrics"
        assert config.otel_enabled is False
        assert config.log_format == "json"


class TestNotificationServiceConfig:
    """Tests for NotificationServiceConfig aggregate."""

    def test_default_configuration(self):
        """Test default aggregate configuration."""
        config = NotificationServiceConfig()

        assert config.service.name == "notification-service"
        assert config.email.enabled is True
        assert config.sms.enabled is False
        assert config.push.enabled is False

    def test_get_enabled_channels(self):
        """Test getting enabled channels."""
        config = NotificationServiceConfig()

        channels = config.get_enabled_channels()
        assert "email" in channels
        assert "sms" not in channels
        assert "push" not in channels

    def test_get_enabled_channels_all(self):
        """Test getting all enabled channels."""
        config = NotificationServiceConfig()
        config.email.enabled = True
        config.sms.enabled = True
        config.push.enabled = True

        channels = config.get_enabled_channels()
        assert len(channels) == 3

    def test_is_production(self):
        """Test production environment check."""
        config = NotificationServiceConfig()
        assert config.is_production() is False

        config.service.env = Environment.PRODUCTION
        assert config.is_production() is True

    def test_load_configuration(self):
        """Test configuration loading."""
        config = NotificationServiceConfig.load()

        assert config is not None
        assert config.service.name == "notification-service"


class TestConfigSingleton:
    """Tests for configuration singleton."""

    def test_get_config_returns_instance(self):
        """Test get_config returns configuration."""
        reset_config()
        config = get_config()
        assert config is not None
        assert isinstance(config, NotificationServiceConfig)

    def test_get_config_returns_same_instance(self):
        """Test get_config returns singleton."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config(self):
        """Test configuration reset."""
        reset_config()
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2
