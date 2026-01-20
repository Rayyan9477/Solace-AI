"""
Unit tests for notification service module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from domain.service import (
    NotificationService,
    NotificationRequest,
    NotificationResult,
    NotificationRecipient,
    NotificationPriority,
    NotificationStatus,
    create_notification_service,
)
from domain.templates import TemplateType, TemplateRegistry, RenderedTemplate
from domain.channels import (
    ChannelType,
    ChannelRegistry,
    DeliveryResult,
    EmailChannel,
    EmailConfig,
)


class TestNotificationRecipient:
    """Tests for NotificationRecipient model."""

    def test_create_recipient_with_email(self):
        """Test creating recipient with email."""
        recipient = NotificationRecipient(
            email="Test@Example.com",
            name="Test User",
        )
        assert recipient.email == "test@example.com"
        assert recipient.name == "Test User"

    def test_create_recipient_with_all_channels(self):
        """Test creating recipient with all channel targets."""
        recipient = NotificationRecipient(
            user_id=uuid4(),
            email="user@example.com",
            phone="+1234567890",
            device_token="device_token_123",
            name="Full User",
        )
        assert recipient.get_channel_target(ChannelType.EMAIL) == "user@example.com"
        assert recipient.get_channel_target(ChannelType.SMS) == "+1234567890"
        assert recipient.get_channel_target(ChannelType.PUSH) == "device_token_123"

    def test_get_channel_target_returns_none_if_missing(self):
        """Test getting channel target returns None if not set."""
        recipient = NotificationRecipient(email="test@example.com")
        assert recipient.get_channel_target(ChannelType.SMS) is None
        assert recipient.get_channel_target(ChannelType.PUSH) is None


class TestNotificationRequest:
    """Tests for NotificationRequest model."""

    def test_create_request_minimal(self):
        """Test creating request with minimal fields."""
        request = NotificationRequest(
            template_type=TemplateType.WELCOME,
            recipients=[NotificationRecipient(email="test@example.com")],
        )
        assert request.template_type == TemplateType.WELCOME
        assert len(request.recipients) == 1
        assert request.priority == NotificationPriority.NORMAL
        assert ChannelType.EMAIL in request.channels

    def test_create_request_full(self):
        """Test creating request with all fields."""
        request = NotificationRequest(
            template_type=TemplateType.CLINICIAN_ALERT,
            recipients=[
                NotificationRecipient(email="clinician@example.com"),
                NotificationRecipient(phone="+1234567890"),
            ],
            channels=[ChannelType.EMAIL, ChannelType.SMS],
            variables={"patient_name": "John Doe"},
            priority=NotificationPriority.URGENT,
            correlation_id=uuid4(),
        )
        assert len(request.recipients) == 2
        assert len(request.channels) == 2
        assert request.priority == NotificationPriority.URGENT


class TestNotificationService:
    """Tests for NotificationService."""

    @pytest.fixture
    def template_registry(self):
        return TemplateRegistry()

    @pytest.fixture
    def channel_registry(self):
        registry = ChannelRegistry()
        email_channel = EmailChannel(EmailConfig())
        registry.register_channel(email_channel)
        return registry

    @pytest.fixture
    def service(self, template_registry, channel_registry):
        return NotificationService(template_registry, channel_registry)

    @pytest.mark.asyncio
    async def test_send_notification_success(self, service):
        """Test successful notification sending."""
        with patch.object(EmailChannel, "send") as mock_send:
            mock_send.return_value = DeliveryResult(
                channel_type=ChannelType.EMAIL,
                recipient="test@example.com",
                success=True,
                message_id="MSG123",
            )

            request = NotificationRequest(
                template_type=TemplateType.WELCOME,
                recipients=[NotificationRecipient(email="test@example.com", name="Test")],
                variables={
                    "display_name": "Test",
                    "getting_started_link": "https://app.solace-ai.com",
                },
            )

            result = await service.send_notification(request)

            assert result.status == NotificationStatus.DELIVERED
            assert result.successful_deliveries == 1
            assert result.failed_deliveries == 0

    @pytest.mark.asyncio
    async def test_send_notification_partial_delivery(self, service):
        """Test partial delivery when some deliveries fail."""
        call_count = [0]

        async def mock_send(*args, **kwargs):
            call_count[0] += 1
            success = call_count[0] == 1
            return DeliveryResult(
                channel_type=ChannelType.EMAIL,
                recipient=kwargs.get("recipient", "test@example.com"),
                success=success,
                error_message=None if success else "Failed",
            )

        with patch.object(EmailChannel, "send", side_effect=mock_send):
            request = NotificationRequest(
                template_type=TemplateType.WELCOME,
                recipients=[
                    NotificationRecipient(email="user1@example.com", name="User1"),
                    NotificationRecipient(email="user2@example.com", name="User2"),
                ],
                variables={
                    "display_name": "User",
                    "getting_started_link": "https://app.solace-ai.com",
                },
            )

            result = await service.send_notification(request)

            assert result.status == NotificationStatus.PARTIALLY_DELIVERED
            assert result.successful_deliveries == 1
            assert result.failed_deliveries == 1

    @pytest.mark.asyncio
    async def test_send_notification_template_not_found(self, service):
        """Test notification fails when template not found."""
        service._templates._templates.clear()

        request = NotificationRequest(
            template_type=TemplateType.WELCOME,
            recipients=[NotificationRecipient(email="test@example.com")],
        )

        result = await service.send_notification(request)

        assert result.status == NotificationStatus.FAILED
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_send_notification_missing_variables(self, service):
        """Test notification fails when required variables missing."""
        request = NotificationRequest(
            template_type=TemplateType.WELCOME,
            recipients=[NotificationRecipient(email="test@example.com")],
            variables={},  # Missing required variables
        )

        result = await service.send_notification(request)

        assert result.status == NotificationStatus.FAILED
        assert "missing" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_send_notification_no_valid_targets(self, service):
        """Test notification fails when no valid delivery targets."""
        request = NotificationRequest(
            template_type=TemplateType.WELCOME,
            recipients=[NotificationRecipient(phone="+1234567890")],  # No email for email channel
            channels=[ChannelType.EMAIL],
            variables={
                "display_name": "Test",
                "getting_started_link": "https://example.com",
            },
        )

        result = await service.send_notification(request)

        assert result.status == NotificationStatus.FAILED
        assert "no valid delivery targets" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_send_email_convenience_method(self, service):
        """Test send_email convenience method."""
        with patch.object(EmailChannel, "send") as mock_send:
            mock_send.return_value = DeliveryResult(
                channel_type=ChannelType.EMAIL,
                recipient="user@example.com",
                success=True,
            )

            result = await service.send_email(
                to_email="user@example.com",
                template_type=TemplateType.WELCOME,
                variables={
                    "display_name": "User",
                    "getting_started_link": "https://example.com",
                },
            )

            assert result.status == NotificationStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_send_clinician_alert(self, service):
        """Test send_clinician_alert method."""
        with patch.object(EmailChannel, "send") as mock_send:
            mock_send.return_value = DeliveryResult(
                channel_type=ChannelType.EMAIL,
                recipient="clinician@example.com",
                success=True,
            )

            result = await service.send_clinician_alert(
                clinician_email="clinician@example.com",
                patient_name="John Doe",
                patient_id="P123",
                severity="HIGH",
                alert_type="Risk Assessment",
                alert_details="Elevated risk indicators detected",
                dashboard_link="https://dashboard.example.com",
            )

            assert result.status == NotificationStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_send_crisis_escalation(self, service):
        """Test send_crisis_escalation method."""
        with patch.object(EmailChannel, "send") as mock_send:
            mock_send.return_value = DeliveryResult(
                channel_type=ChannelType.EMAIL,
                recipient="clinician@example.com",
                success=True,
            )

            result = await service.send_crisis_escalation(
                clinician_email="clinician@example.com",
                patient_name="Jane Doe",
                patient_id="P456",
                risk_level="CRITICAL",
                assessment_summary="Immediate intervention required",
                dashboard_link="https://dashboard.example.com",
            )

            assert result.status == NotificationStatus.DELIVERED

    def test_get_available_channels(self, service):
        """Test getting available channels."""
        channels = service.get_available_channels()
        assert ChannelType.EMAIL in channels

    def test_get_available_templates(self, service):
        """Test getting available templates."""
        templates = service.get_available_templates()
        assert len(templates) > 0
        assert TemplateType.WELCOME in templates


class TestCreateNotificationService:
    """Tests for factory function."""

    def test_create_with_email_only(self):
        """Test creating service with email channel only."""
        service = create_notification_service(
            email_config=EmailConfig(),
        )
        assert ChannelType.EMAIL in service.get_available_channels()

    def test_create_with_no_channels(self):
        """Test creating service with no enabled channels."""
        service = create_notification_service()
        assert len(service.get_available_channels()) == 0

    def test_create_with_disabled_channel(self):
        """Test creating service with disabled channel."""
        email_config = EmailConfig(enabled=False)
        service = create_notification_service(email_config=email_config)
        assert ChannelType.EMAIL not in service.get_available_channels()


class TestNotificationPriority:
    """Tests for NotificationPriority enum."""

    def test_priority_ordering(self):
        """Test priority values are defined."""
        priorities = [p.value for p in NotificationPriority]
        assert "low" in priorities
        assert "normal" in priorities
        assert "high" in priorities
        assert "urgent" in priorities
        assert "critical" in priorities


class TestNotificationStatus:
    """Tests for NotificationStatus enum."""

    def test_status_values(self):
        """Test status values are defined."""
        statuses = [s.value for s in NotificationStatus]
        assert "pending" in statuses
        assert "delivered" in statuses
        assert "failed" in statuses
        assert "partially_delivered" in statuses
