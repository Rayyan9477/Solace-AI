"""
Unit tests for notification channels module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from domain.channels import (
    ChannelType,
    ChannelStatus,
    ChannelConfig,
    EmailConfig,
    SMSConfig,
    PushConfig,
    DeliveryResult,
    EmailChannel,
    SMSChannel,
    PushChannel,
    ChannelRegistry,
    ChannelError,
    DeliveryError,
)


class TestChannelConfig:
    """Tests for channel configuration models."""

    def test_default_channel_config(self):
        """Test default channel configuration values."""
        config = ChannelConfig()
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.timeout_seconds == 30.0

    def test_email_config_defaults(self):
        """Test email configuration defaults."""
        config = EmailConfig()
        assert config.smtp_host == "localhost"
        assert config.smtp_port == 587
        assert config.use_tls is True
        assert config.from_email == "noreply@solace-ai.com"
        assert config.from_name == "Solace-AI"

    def test_email_config_custom(self):
        """Test custom email configuration."""
        config = EmailConfig(
            smtp_host="smtp.example.com",
            smtp_port=465,
            smtp_username="user",
            smtp_password="pass",
            from_email="custom@example.com",
        )
        assert config.smtp_host == "smtp.example.com"
        assert config.smtp_port == 465
        assert config.from_email == "custom@example.com"

    def test_sms_config_defaults(self):
        """Test SMS configuration defaults."""
        config = SMSConfig()
        assert "twilio" in config.provider_url.lower()
        assert config.enabled is True

    def test_push_config_defaults(self):
        """Test push configuration defaults."""
        config = PushConfig()
        assert "fcm" in config.firebase_url.lower()
        assert config.enabled is True


class TestDeliveryResult:
    """Tests for DeliveryResult model."""

    def test_create_successful_result(self):
        """Test creating a successful delivery result."""
        result = DeliveryResult(
            channel_type=ChannelType.EMAIL,
            recipient="test@example.com",
            success=True,
            message_id="msg-123",
        )
        assert result.success is True
        assert result.channel_type == ChannelType.EMAIL
        assert result.recipient == "test@example.com"
        assert result.error_message is None

    def test_create_failed_result(self):
        """Test creating a failed delivery result."""
        result = DeliveryResult(
            channel_type=ChannelType.SMS,
            recipient="+1234567890",
            success=False,
            error_message="Connection timeout",
        )
        assert result.success is False
        assert result.error_message == "Connection timeout"


class TestEmailChannel:
    """Tests for EmailChannel."""

    @pytest.fixture
    def email_config(self):
        return EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="test",
            smtp_password="secret",
        )

    @pytest.fixture
    def email_channel(self, email_config):
        return EmailChannel(email_config)

    def test_channel_type(self, email_channel):
        """Test email channel type."""
        assert email_channel.channel_type == ChannelType.EMAIL

    def test_channel_is_available(self, email_channel):
        """Test channel availability."""
        assert email_channel.is_available is True
        assert email_channel.status == ChannelStatus.ACTIVE

    def test_disabled_channel_not_available(self, email_config):
        """Test disabled channel is not available."""
        email_config.enabled = False
        channel = EmailChannel(email_config)
        assert channel.is_available is False
        assert channel.status == ChannelStatus.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_send_returns_result_on_unavailable(self, email_config):
        """Test send returns failure when channel unavailable."""
        email_config.enabled = False
        channel = EmailChannel(email_config)
        result = await channel.send(
            recipient="test@example.com",
            subject="Test",
            body="Test body",
        )
        assert result.success is False
        assert "unavailable" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_send_email_success(self, email_channel):
        """Test successful email sending."""
        with patch("aiosmtplib.SMTP") as mock_smtp:
            mock_instance = AsyncMock()
            mock_smtp.return_value.__aenter__.return_value = mock_instance
            mock_instance.send_message.return_value = (250, "OK")

            result = await email_channel.send(
                recipient="test@example.com",
                subject="Test Subject",
                body="Test body",
                html_body="<p>Test body</p>",
            )

            assert result.success is True
            assert result.channel_type == ChannelType.EMAIL

    @pytest.mark.asyncio
    async def test_send_email_with_retry(self, email_config):
        """Test email sending with retry on failure."""
        email_config.max_retries = 2
        email_config.retry_delay_seconds = 0.01
        channel = EmailChannel(email_config)

        with patch("aiosmtplib.SMTP") as mock_smtp:
            mock_instance = AsyncMock()
            mock_smtp.return_value.__aenter__.return_value = mock_instance
            mock_instance.send_message.side_effect = [
                Exception("Temp failure"),
                (250, "OK"),
            ]

            result = await channel.send(
                recipient="test@example.com",
                subject="Test",
                body="Test",
            )

            assert result.success is True
            assert mock_instance.send_message.call_count == 2


class TestSMSChannel:
    """Tests for SMSChannel."""

    @pytest.fixture
    def sms_config(self):
        return SMSConfig(
            account_sid="TEST_SID",
            auth_token="TEST_TOKEN",
            from_number="+15551234567",
        )

    @pytest.fixture
    def sms_channel(self, sms_config):
        return SMSChannel(sms_config)

    def test_channel_type(self, sms_channel):
        """Test SMS channel type."""
        assert sms_channel.channel_type == ChannelType.SMS

    @pytest.mark.asyncio
    async def test_send_sms_success(self, sms_channel):
        """Test successful SMS sending."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"sid": "SM123", "num_segments": 1}
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await sms_channel.send(
                recipient="+1234567890",
                subject="Alert",
                body="Test message",
            )

            assert result.success is True
            assert result.message_id == "SM123"


class TestPushChannel:
    """Tests for PushChannel."""

    @pytest.fixture
    def push_config(self):
        return PushConfig(
            server_key="TEST_SERVER_KEY",
            project_id="test-project",
            use_v1_api=False,
        )

    @pytest.fixture
    def push_channel(self, push_config):
        return PushChannel(push_config)

    def test_channel_type(self, push_channel):
        """Test push channel type."""
        assert push_channel.channel_type == ChannelType.PUSH

    @pytest.mark.asyncio
    async def test_send_push_success(self, push_channel):
        """Test successful push notification sending."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "success": 1,
                "failure": 0,
                "message_id": "MSG123",
            }
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await push_channel.send(
                recipient="device_token_123",
                subject="Push Title",
                body="Push body",
                metadata={"key": "value"},
            )

            assert result.success is True


class TestChannelRegistry:
    """Tests for ChannelRegistry."""

    @pytest.fixture
    def registry(self):
        return ChannelRegistry()

    def test_registry_initially_empty(self, registry):
        """Test registry starts empty."""
        assert len(registry.list_channels()) == 0

    def test_register_channel(self, registry):
        """Test registering a channel."""
        config = EmailConfig()
        channel = EmailChannel(config)
        registry.register_channel(channel)

        channels = registry.list_channels()
        assert len(channels) == 1
        assert channels[0][0] == ChannelType.EMAIL

    def test_get_channel(self, registry):
        """Test getting a registered channel."""
        config = EmailConfig()
        channel = EmailChannel(config)
        registry.register_channel(channel)

        retrieved = registry.get_channel(ChannelType.EMAIL)
        assert retrieved is not None
        assert retrieved.channel_type == ChannelType.EMAIL

    def test_get_channel_not_found(self, registry):
        """Test getting non-existent channel returns None."""
        result = registry.get_channel(ChannelType.EMAIL)
        assert result is None

    def test_get_available_channels(self, registry):
        """Test getting available channels."""
        email_config = EmailConfig(enabled=True)
        sms_config = SMSConfig(enabled=False)

        registry.register_channel(EmailChannel(email_config))
        registry.register_channel(SMSChannel(sms_config))

        available = registry.get_available_channels()
        assert len(available) == 1
        assert available[0].channel_type == ChannelType.EMAIL

    @pytest.mark.asyncio
    async def test_health_check_all(self, registry):
        """Test health check for all channels."""
        registry.register_channel(EmailChannel(EmailConfig()))
        registry.register_channel(SMSChannel(SMSConfig()))

        results = await registry.health_check_all()
        assert ChannelType.EMAIL in results
        assert ChannelType.SMS in results


class TestChannelTypes:
    """Tests for ChannelType enum."""

    def test_all_channel_types_are_strings(self):
        """Test all channel types have string values."""
        for channel_type in ChannelType:
            assert isinstance(channel_type.value, str)

    def test_expected_channel_types_exist(self):
        """Test expected channel types are defined."""
        expected = ["email", "sms", "push", "webhook"]
        actual = [c.value for c in ChannelType]
        for expected_type in expected:
            assert expected_type in actual


class TestChannelExceptions:
    """Tests for channel exceptions."""

    def test_channel_error(self):
        """Test ChannelError exception."""
        error = ChannelError(ChannelType.EMAIL, "Test error")
        assert "[email]" in str(error).lower()
        assert "test error" in str(error).lower()

    def test_delivery_error(self):
        """Test DeliveryError exception."""
        error = DeliveryError(ChannelType.SMS, "+1234567890", "Timeout")
        assert "+1234567890" in str(error)
        assert "timeout" in str(error).lower()
