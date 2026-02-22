"""
Unit tests for User Service domain entities.

Tests cover business rules, invariants, and state transitions.
"""
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from src.domain.entities import User, UserPreferences
from src.domain.value_objects import UserRole, AccountStatus, ConsentType


class TestUserEntity:
    """Test cases for User entity."""

    def test_user_creation_with_defaults(self):
        """Test user creation with default values."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
        )

        assert user.user_id is not None
        assert user.email == "test@example.com"
        assert user.display_name == "Test User"
        assert user.role == UserRole.USER
        assert user.status == AccountStatus.PENDING_VERIFICATION
        assert user.email_verified is False
        assert user.login_attempts == 0
        assert user.deleted_at is None

    def test_email_normalization_to_lowercase(self):
        """Test that email is normalized to lowercase."""
        user = User(
            email="Test@Example.COM",
            password_hash="hashed_password",
            display_name="Test User",
        )

        assert user.email == "test@example.com"

    def test_invalid_email_format_raises_error(self):
        """Test that invalid email format raises validation error."""
        with pytest.raises(ValueError, match="Invalid email format"):
            User(
                email="invalid-email",
                password_hash="hashed_password",
                display_name="Test User",
            )

    def test_invalid_timezone_raises_error(self):
        """Test that invalid timezone raises validation error."""
        with pytest.raises(ValueError, match="Invalid timezone"):
            User(
                email="test@example.com",
                password_hash="hashed_password",
                display_name="Test User",
                timezone="x" * 60,
            )

    def test_invalid_locale_format_raises_error(self):
        """Test that invalid locale format raises validation error."""
        with pytest.raises(ValueError, match="Invalid locale format"):
            User(
                email="test@example.com",
                password_hash="hashed_password",
                display_name="Test User",
                locale="invalid",
            )

    def test_activate_user_from_pending_verification(self):
        """Test activating user from PENDING_VERIFICATION status."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            email_verified=True,
        )

        user.activate()

        assert user.status == AccountStatus.ACTIVE

    def test_activate_user_without_email_verification_fails(self):
        """Test that activation fails without email verification."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            email_verified=False,
        )

        with pytest.raises(ValueError, match="Email must be verified"):
            user.activate()

    def test_activate_user_from_invalid_status_fails(self):
        """Test that activation fails from invalid status."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            status=AccountStatus.ACTIVE,
            email_verified=True,
        )

        with pytest.raises(ValueError, match="Cannot activate user"):
            user.activate()

    def test_suspend_active_user(self):
        """Test suspending an active user."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            status=AccountStatus.ACTIVE,
        )

        user.suspend(reason="Policy violation")

        assert user.status == AccountStatus.SUSPENDED

    def test_suspend_non_active_user_fails(self):
        """Test that suspending non-active user fails."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            status=AccountStatus.PENDING_VERIFICATION,
        )

        with pytest.raises(ValueError, match="Cannot suspend user"):
            user.suspend()

    def test_reactivate_suspended_user(self):
        """Test reactivating a suspended user."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            status=AccountStatus.SUSPENDED,
            email_verified=True,
        )

        user.reactivate()

        assert user.status == AccountStatus.ACTIVE
        assert user.login_attempts == 0
        assert user.locked_until is None

    def test_soft_delete_user(self):
        """Test soft deleting a user."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
        )
        original_user_id = user.user_id

        user.soft_delete()

        assert user.deleted_at is not None
        assert user.status == AccountStatus.INACTIVE
        assert user.email == f"deleted_{original_user_id}@deleted.solace-ai.com"
        assert user.display_name == "Deleted User"
        assert user.password_hash == "DELETED"
        assert user.phone_number is None
        assert user.email_verified is False
        assert user.avatar_url is None
        assert user.bio is None
        assert user.is_on_call is False

    def test_record_successful_login(self):
        """Test recording successful login attempt."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            login_attempts=3,
        )

        user.record_login_attempt(success=True)

        assert user.login_attempts == 0
        assert user.last_login is not None
        assert user.locked_until is None

    def test_record_failed_login_increments_attempts(self):
        """Test recording failed login attempt."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
        )

        user.record_login_attempt(success=False)

        assert user.login_attempts == 1

    def test_account_locked_after_max_attempts(self):
        """Test that account is locked after max failed attempts."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
        )

        for _ in range(5):
            user.record_login_attempt(success=False, max_attempts=5)

        assert user.locked_until is not None
        assert user.is_locked() is True

    def test_is_active_checks_all_conditions(self):
        """Test that is_active checks all conditions."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            status=AccountStatus.ACTIVE,
        )

        assert user.is_active() is True

        user.deleted_at = datetime.now(timezone.utc)
        assert user.is_active() is False

    def test_can_login_returns_correct_status(self):
        """Test that can_login returns correct status and message."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            status=AccountStatus.ACTIVE,
        )

        can_login, message = user.can_login()
        assert can_login is True
        assert message is None

        user.status = AccountStatus.SUSPENDED
        can_login, message = user.can_login()
        assert can_login is False
        assert "suspended" in message.lower()

    def test_update_profile_updates_allowed_fields(self):
        """Test updating profile with allowed fields."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
        )

        user.update_profile(display_name="New Name", timezone="America/New_York")

        assert user.display_name == "New Name"
        assert user.timezone == "America/New_York"

    def test_update_profile_rejects_disallowed_fields(self):
        """Test that updating disallowed fields raises error."""
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
        )

        with pytest.raises(ValueError, match="Cannot update field"):
            user.update_profile(email="newemail@example.com")


class TestUserPreferencesEntity:
    """Test cases for UserPreferences entity."""

    def test_preferences_creation_with_defaults(self):
        """Test preferences creation with default values."""
        user_id = uuid4()
        prefs = UserPreferences(user_id=user_id)

        assert prefs.user_id == user_id
        assert prefs.notification_email is True
        assert prefs.notification_sms is False
        assert prefs.theme == "system"
        assert prefs.language == "en"
        assert prefs.accessibility_high_contrast is False

    def test_invalid_theme_raises_error(self):
        """Test that invalid theme raises validation error."""
        with pytest.raises(ValueError, match="Invalid theme"):
            UserPreferences(user_id=uuid4(), theme="invalid")

    def test_invalid_language_code_raises_error(self):
        """Test that invalid language code raises validation error."""
        with pytest.raises(ValueError, match="Invalid language code"):
            UserPreferences(user_id=uuid4(), language="invalid")

    def test_update_preferences(self):
        """Test updating preferences."""
        prefs = UserPreferences(user_id=uuid4())

        prefs.update(theme="dark", notification_email=False)

        assert prefs.theme == "dark"
        assert prefs.notification_email is False

    def test_update_immutable_field_raises_error(self):
        """Test that updating immutable field raises error."""
        prefs = UserPreferences(user_id=uuid4())

        with pytest.raises(ValueError, match="Cannot update immutable field"):
            prefs.update(user_id=uuid4())

    def test_update_unknown_field_raises_error(self):
        """Test that updating unknown field raises error."""
        prefs = UserPreferences(user_id=uuid4())

        with pytest.raises(ValueError, match="Unknown preference field"):
            prefs.update(unknown_field="value")

    def test_get_notification_channels(self):
        """Test getting active notification channels."""
        prefs = UserPreferences(
            user_id=uuid4(),
            notification_email=True,
            notification_sms=False,
            notification_push=True,
        )

        channels = prefs.get_notification_channels()

        assert "email" in channels
        assert "push" in channels
        assert "sms" not in channels
        assert "in_app" in channels

    def test_has_marketing_consent(self):
        """Test checking marketing consent."""
        prefs = UserPreferences(user_id=uuid4(), marketing_emails=True)

        assert prefs.has_marketing_consent() is True

        prefs.marketing_emails = False
        assert prefs.has_marketing_consent() is False

    def test_has_research_consent(self):
        """Test checking research consent."""
        prefs = UserPreferences(user_id=uuid4(), data_sharing_research=True)

        assert prefs.has_research_consent() is True

        prefs.data_sharing_research = False
        assert prefs.has_research_consent() is False
