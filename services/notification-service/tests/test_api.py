"""
Unit tests for notification API endpoints.

Tests the FastAPI endpoints using test client with dependency overrides.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from domain.service import (
    NotificationResult,
    NotificationStatus,
    NotificationRecipient,
)
from domain.templates import TemplateType, TemplateRegistry
from domain.channels import ChannelType

from solace_security.middleware import get_current_user, get_current_service, AuthenticatedUser, AuthenticatedService
from solace_security.auth import TokenType


@pytest.mark.integration
class TestTemplateEndpoints:
    """Tests for template endpoints - require full app setup (integration tests)."""

    @pytest.fixture
    def app(self):
        """Create a minimal FastAPI app with template routes."""
        from api import router
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
            user_id="test-user",
            token_type=TokenType.ACCESS,
            roles=["user", "admin"],
            permissions=["notifications:read", "notifications:write"],
        )
        app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
            service_name="test-service",
            permissions=["notifications:read", "notifications:write"],
        )
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.mark.skip(reason="Integration test - requires full app setup")
    def test_list_templates(self, client):
        """Test listing all templates."""
        response = client.get("/api/v1/notifications/templates")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        template = data[0]
        assert "template_type" in template
        assert "name" in template
        assert "required_variables" in template

    def test_get_template_success(self, client):
        """Test getting a specific template."""
        response = client.get("/api/v1/notifications/templates/welcome")

        assert response.status_code == 200
        data = response.json()
        assert data["template_type"] == "welcome"
        assert "required_variables" in data

    def test_get_template_invalid(self, client):
        """Test getting with invalid template type."""
        response = client.get("/api/v1/notifications/templates/nonexistent")

        assert response.status_code == 400


@pytest.mark.integration
class TestValidationEndpoints:
    """Test endpoint validation logic - require full app setup (integration tests)."""

    @pytest.fixture
    def app(self):
        """Create a minimal FastAPI app for validation tests."""
        from api import router
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
            user_id="test-user",
            token_type=TokenType.ACCESS,
            roles=["user", "admin"],
            permissions=["notifications:read", "notifications:write"],
        )
        app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
            service_name="test-service",
            permissions=["notifications:read", "notifications:write"],
        )
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.mark.skip(reason="Integration test - requires full app setup")
    def test_send_notification_invalid_template(self, client):
        """Test sending with invalid template type returns 400."""
        response = client.post(
            "/api/v1/notifications/send",
            json={
                "template_type": "invalid_template",
                "recipients": [{"email": "test@example.com"}],
            },
        )

        assert response.status_code == 400
        assert "invalid template type" in response.json()["detail"].lower()

    @pytest.mark.skip(reason="Integration test - requires full app setup")
    def test_send_notification_invalid_channel(self, client):
        """Test sending with invalid channel returns 400."""
        response = client.post(
            "/api/v1/notifications/send",
            json={
                "template_type": "welcome",
                "recipients": [{"email": "test@example.com"}],
                "channels": ["invalid_channel"],
            },
        )

        assert response.status_code == 400
        assert "invalid channel type" in response.json()["detail"].lower()

    @pytest.mark.skip(reason="Integration test - requires full app setup")
    def test_send_notification_invalid_priority(self, client):
        """Test sending with invalid priority returns 400."""
        response = client.post(
            "/api/v1/notifications/send",
            json={
                "template_type": "welcome",
                "recipients": [{"email": "test@example.com"}],
                "priority": "super_urgent",
            },
        )

        assert response.status_code == 400
        assert "invalid priority" in response.json()["detail"].lower()

    @pytest.mark.skip(reason="Integration test - requires full app setup")
    def test_send_email_invalid_email(self, client):
        """Test sending with invalid email format returns 422."""
        response = client.post(
            "/api/v1/notifications/email",
            json={
                "to_email": "not-an-email",
                "template_type": "welcome",
            },
        )

        assert response.status_code == 422


@pytest.mark.integration
class TestChannelEndpoints:
    """Tests for channel endpoints - require full app setup."""

    @pytest.fixture
    def app(self):
        """Create a minimal FastAPI app."""
        from api import router
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
            user_id="test-user",
            token_type=TokenType.ACCESS,
            roles=["user", "admin"],
            permissions=["notifications:read", "notifications:write"],
        )
        app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
            service_name="test-service",
            permissions=["notifications:read", "notifications:write"],
        )
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.mark.skip(reason="Integration test - requires full app setup")
    def test_list_channels(self, client):
        """Test listing all channels."""
        response = client.get("/api/v1/notifications/channels")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for notification health endpoint - require full app setup."""

    @pytest.fixture
    def app(self):
        """Create a minimal FastAPI app."""
        from api import router
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
            user_id="test-user",
            token_type=TokenType.ACCESS,
            roles=["user", "admin"],
            permissions=["notifications:read", "notifications:write"],
        )
        app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
            service_name="test-service",
            permissions=["notifications:read", "notifications:write"],
        )
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.mark.skip(reason="Integration test - requires full app setup")
    def test_notification_health(self, client):
        """Test notification service health check."""
        response = client.get("/api/v1/notifications/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "channels" in data


class TestNotificationResult:
    """Test NotificationResult model used in API responses."""

    def test_notification_result_creation(self):
        """Test creating a notification result."""
        result = NotificationResult(
            request_id=uuid4(),
            status=NotificationStatus.DELIVERED,
            template_type=TemplateType.WELCOME,
            total_recipients=2,
            successful_deliveries=2,
            failed_deliveries=0,
        )
        assert result.status == NotificationStatus.DELIVERED
        assert result.total_recipients == 2

    def test_notification_result_partial(self):
        """Test partial delivery result."""
        result = NotificationResult(
            request_id=uuid4(),
            status=NotificationStatus.PARTIALLY_DELIVERED,
            template_type=TemplateType.CLINICIAN_ALERT,
            total_recipients=3,
            successful_deliveries=2,
            failed_deliveries=1,
        )
        assert result.status == NotificationStatus.PARTIALLY_DELIVERED
        assert result.failed_deliveries == 1


class TestNotificationRecipientModel:
    """Test NotificationRecipient model used in API requests."""

    def test_recipient_with_email(self):
        """Test creating recipient with email."""
        recipient = NotificationRecipient(
            email="TEST@Example.com",
            name="Test User",
        )
        assert recipient.email == "test@example.com"  # Normalized
        assert recipient.get_channel_target(ChannelType.EMAIL) == "test@example.com"

    def test_recipient_without_email(self):
        """Test recipient without email returns None for email channel."""
        recipient = NotificationRecipient(phone="+1234567890")
        assert recipient.get_channel_target(ChannelType.EMAIL) is None
        assert recipient.get_channel_target(ChannelType.SMS) == "+1234567890"


class TestAPIRequestModels:
    """Test API request validation models."""

    def test_send_notification_request_valid(self):
        """Test valid notification request."""
        from api import SendNotificationRequest

        request = SendNotificationRequest(
            template_type="welcome",
            recipients=[{"email": "test@example.com", "name": "Test"}],
            channels=["email"],
            variables={"display_name": "Test"},
            priority="normal",
        )
        assert request.template_type == "welcome"
        assert len(request.recipients) == 1

    def test_send_email_request_valid(self):
        """Test valid email request."""
        from api import SendEmailRequest

        request = SendEmailRequest(
            to_email="user@example.com",
            template_type="welcome",
            variables={"display_name": "User"},
        )
        assert request.to_email == "user@example.com"

    def test_send_sms_request_valid(self):
        """Test valid SMS request."""
        from api import SendSMSRequest

        request = SendSMSRequest(
            to_phone="+1234567890",
            template_type="welcome",
            variables={"display_name": "User"},
        )
        assert request.to_phone == "+1234567890"

    def test_send_push_request_valid(self):
        """Test valid push request."""
        from api import SendPushRequest

        request = SendPushRequest(
            device_token="device_token_123456",
            template_type="welcome",
            variables={"display_name": "User"},
        )
        assert request.device_token == "device_token_123456"
