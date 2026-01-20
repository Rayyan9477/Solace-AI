"""
Pytest configuration and fixtures for notification service tests.
"""
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from domain.templates import TemplateRegistry, TemplateEngine
from domain.channels import (
    ChannelRegistry,
    EmailChannel,
    SMSChannel,
    PushChannel,
    EmailConfig,
    SMSConfig,
    PushConfig,
    ChannelType,
)
from domain.service import NotificationService


@pytest.fixture
def template_registry():
    """Create a template registry with default templates."""
    return TemplateRegistry()


@pytest.fixture
def template_engine():
    """Create a template engine."""
    return TemplateEngine()


@pytest.fixture
def email_config():
    """Create email configuration."""
    return EmailConfig(
        smtp_host="smtp.test.com",
        smtp_port=587,
        smtp_username="test",
        smtp_password="secret",
    )


@pytest.fixture
def sms_config():
    """Create SMS configuration."""
    return SMSConfig(
        account_sid="TEST_SID",
        auth_token="TEST_TOKEN",
        from_number="+15551234567",
    )


@pytest.fixture
def push_config():
    """Create push configuration."""
    return PushConfig(
        server_key="TEST_SERVER_KEY",
        project_id="test-project",
    )


@pytest.fixture
def channel_registry(email_config):
    """Create a channel registry with email channel."""
    registry = ChannelRegistry()
    registry.register_channel(EmailChannel(email_config))
    return registry


@pytest.fixture
def notification_service(template_registry, channel_registry):
    """Create a notification service."""
    return NotificationService(template_registry, channel_registry)


@pytest.fixture
def mock_notification_service():
    """Create a mock notification service."""
    service = MagicMock(spec=NotificationService)
    service.send_notification = AsyncMock()
    service.send_email = AsyncMock()
    service.send_sms = AsyncMock()
    service.send_push = AsyncMock()
    service.get_available_channels = MagicMock(return_value=[ChannelType.EMAIL])
    service.get_available_templates = MagicMock(return_value=[])
    return service
