"""
Unit tests for User Service domain events.

Tests cover event creation, immutability, and serialization.
"""
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.events import (
    EventType,
    DomainEvent,
    UserCreatedEvent,
    UserUpdatedEvent,
    UserDeletedEvent,
    UserActivatedEvent,
    UserSuspendedEvent,
    EmailVerifiedEvent,
    PasswordChangedEvent,
    PreferencesUpdatedEvent,
    ConsentGrantedEvent,
    ConsentRevokedEvent,
    LoginSuccessfulEvent,
    LoginFailedEvent,
    AccountLockedEvent,
)


class TestDomainEvent:
    """Test cases for base DomainEvent."""

    def test_domain_event_creation(self):
        """Test creating a domain event."""
        user_id = uuid4()
        event = DomainEvent(
            event_type=EventType.USER_CREATED,
            aggregate_id=user_id,
        )

        assert event.event_id is not None
        assert event.event_type == EventType.USER_CREATED
        assert event.aggregate_id == user_id
        assert event.aggregate_type == "user"
        assert event.occurred_at is not None
        assert event.version == "1.0"

    def test_domain_event_is_immutable(self):
        """Test that domain event is immutable."""
        event = DomainEvent(
            event_type=EventType.USER_CREATED,
            aggregate_id=uuid4(),
        )

        with pytest.raises(Exception):
            event.event_type = EventType.USER_DELETED

    def test_domain_event_to_dict(self):
        """Test converting domain event to dictionary."""
        user_id = uuid4()
        event = DomainEvent(
            event_type=EventType.USER_CREATED,
            aggregate_id=user_id,
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == EventType.USER_CREATED.value
        assert event_dict["aggregate_id"] == str(user_id)
        assert event_dict["aggregate_type"] == "user"
        assert "occurred_at" in event_dict


class TestUserCreatedEvent:
    """Test cases for UserCreatedEvent."""

    def test_user_created_event_creation(self):
        """Test creating a user created event."""
        user_id = uuid4()
        event = UserCreatedEvent(
            aggregate_id=user_id,
            email="test@example.com",
            display_name="Test User",
            role="user",
            status="pending_verification",
            timezone="UTC",
            locale="en-US",
        )

        assert event.event_type == EventType.USER_CREATED
        assert event.email == "test@example.com"
        assert event.display_name == "Test User"

    def test_user_created_event_to_dict(self):
        """Test converting user created event to dictionary."""
        user_id = uuid4()
        event = UserCreatedEvent(
            aggregate_id=user_id,
            email="test@example.com",
            display_name="Test User",
            role="user",
            status="pending_verification",
            timezone="UTC",
            locale="en-US",
        )

        event_dict = event.to_dict()

        assert event_dict["email"] == "test@example.com"
        assert event_dict["display_name"] == "Test User"
        assert event_dict["role"] == "user"


class TestUserUpdatedEvent:
    """Test cases for UserUpdatedEvent."""

    def test_user_updated_event_creation(self):
        """Test creating a user updated event."""
        user_id = uuid4()
        event = UserUpdatedEvent(
            aggregate_id=user_id,
            updated_fields=["display_name", "timezone"],
            previous_values={"display_name": "Old Name"},
            new_values={"display_name": "New Name"},
        )

        assert event.event_type == EventType.USER_UPDATED
        assert "display_name" in event.updated_fields


class TestUserDeletedEvent:
    """Test cases for UserDeletedEvent."""

    def test_user_deleted_event_creation(self):
        """Test creating a user deleted event."""
        user_id = uuid4()
        deleted_at = datetime.now(timezone.utc)
        event = UserDeletedEvent(
            aggregate_id=user_id,
            reason="User requested deletion",
            deleted_at=deleted_at,
        )

        assert event.event_type == EventType.USER_DELETED
        assert event.reason == "User requested deletion"
        assert event.deleted_at == deleted_at


class TestConsentEvents:
    """Test cases for consent-related events."""

    def test_consent_granted_event_creation(self):
        """Test creating a consent granted event."""
        user_id = uuid4()
        event = ConsentGrantedEvent(
            aggregate_id=user_id,
            consent_type="terms_of_service",
            version="1.0",
            ip_address="192.168.1.1",
        )

        assert event.event_type == EventType.CONSENT_GRANTED
        assert event.consent_type == "terms_of_service"
        assert event.version == "1.0"

    def test_consent_revoked_event_creation(self):
        """Test creating a consent revoked event."""
        user_id = uuid4()
        event = ConsentRevokedEvent(
            aggregate_id=user_id,
            consent_type="marketing_communications",
            reason="User opted out",
        )

        assert event.event_type == EventType.CONSENT_REVOKED
        assert event.consent_type == "marketing_communications"


class TestAuthenticationEvents:
    """Test cases for authentication-related events."""

    def test_login_successful_event_creation(self):
        """Test creating a login successful event."""
        user_id = uuid4()
        event = LoginSuccessfulEvent(
            aggregate_id=user_id,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert event.event_type == EventType.LOGIN_SUCCESSFUL
        assert event.ip_address == "192.168.1.1"

    def test_login_failed_event_creation(self):
        """Test creating a login failed event."""
        user_id = uuid4()
        event = LoginFailedEvent(
            aggregate_id=user_id,
            attempts=3,
            reason="Invalid password",
            ip_address="192.168.1.1",
        )

        assert event.event_type == EventType.LOGIN_FAILED
        assert event.attempts == 3

    def test_account_locked_event_creation(self):
        """Test creating an account locked event."""
        user_id = uuid4()
        locked_until = datetime.now(timezone.utc)
        event = AccountLockedEvent(
            aggregate_id=user_id,
            locked_until=locked_until,
            attempts=5,
        )

        assert event.event_type == EventType.ACCOUNT_LOCKED
        assert event.attempts == 5


class TestPreferencesUpdatedEvent:
    """Test cases for PreferencesUpdatedEvent."""

    def test_preferences_updated_event_creation(self):
        """Test creating a preferences updated event."""
        user_id = uuid4()
        event = PreferencesUpdatedEvent(
            aggregate_id=user_id,
            updated_fields=["theme", "notification_email"],
        )

        assert event.event_type == EventType.PREFERENCES_UPDATED
        assert "theme" in event.updated_fields
