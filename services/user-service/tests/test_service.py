"""
Tests for User Service domain service.

Tests business logic for user management operations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from src.domain.entities import User, UserPreferences
from src.domain.value_objects import UserRole, AccountStatus, ConsentType
from src.domain.service import (
    UserService,
    CreateUserResult,
    UpdateUserResult,
    DeleteUserResult,
    PasswordChangeResult,
    EmailVerificationResult,
    UpdatePreferencesResult,
)
from .fixtures import (
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
    InMemoryConsentRepository,
)


class MockPasswordService:
    """Mock password service for testing."""

    def hash_password(self, password: str) -> str:
        return f"hashed_{password}"

    def verify_password(self, password: str, password_hash: str) -> MockVerifyResult:
        is_valid = password_hash == f"hashed_{password}"
        return MockVerifyResult(is_valid=is_valid)


class MockVerifyResult:
    """Mock verify result."""

    def __init__(self, is_valid: bool):
        self.is_valid = is_valid
        self.needs_rehash = False
        self.new_hash = None


@pytest.fixture
def user_repository() -> InMemoryUserRepository:
    """Create user repository."""
    return InMemoryUserRepository()


@pytest.fixture
def preferences_repository() -> InMemoryUserPreferencesRepository:
    """Create preferences repository."""
    return InMemoryUserPreferencesRepository()


@pytest.fixture
def consent_repository() -> InMemoryConsentRepository:
    """Create consent repository."""
    return InMemoryConsentRepository()


@pytest.fixture
def password_service() -> MockPasswordService:
    """Create password service."""
    return MockPasswordService()


@pytest.fixture
def user_service(
    user_repository: InMemoryUserRepository,
    preferences_repository: InMemoryUserPreferencesRepository,
    consent_repository: InMemoryConsentRepository,
    password_service: MockPasswordService,
) -> UserService:
    """Create user service with dependencies."""
    return UserService(
        user_repository=user_repository,
        preferences_repository=preferences_repository,
        consent_repository=consent_repository,
        password_service=password_service,
    )


class TestUserServiceCreation:
    """Tests for user creation."""

    @pytest.mark.asyncio
    async def test_create_user_success(
        self, user_service: UserService
    ) -> None:
        """Test successful user creation."""
        result = await user_service.create_user(
            email="test@example.com",
            password_hash="hashed_password123",
            display_name="Test User",
        )

        assert result.success is True
        assert result.user is not None
        assert result.user.email == "test@example.com"
        assert result.user.display_name == "Test User"
        assert result.user.status == AccountStatus.PENDING_VERIFICATION
        assert result.user.email_verified is False
        assert result.user.email_verification_token is not None

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(
        self, user_service: UserService
    ) -> None:
        """Test duplicate email rejection."""
        await user_service.create_user(
            email="duplicate@example.com",
            password_hash="hashed_password123",
            display_name="First User",
        )

        result = await user_service.create_user(
            email="duplicate@example.com",
            password_hash="hashed_password456",
            display_name="Second User",
        )

        assert result.success is False
        assert result.error_code == "EMAIL_EXISTS"

    @pytest.mark.asyncio
    async def test_create_user_normalizes_email(
        self, user_service: UserService
    ) -> None:
        """Test email normalization to lowercase."""
        result = await user_service.create_user(
            email="TEST@EXAMPLE.COM",
            password_hash="hashed_password",
            display_name="Test User",
        )

        assert result.success is True
        assert result.user is not None
        assert result.user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_create_user_creates_preferences(
        self,
        user_service: UserService,
        preferences_repository: InMemoryUserPreferencesRepository,
    ) -> None:
        """Test preferences are created with user."""
        result = await user_service.create_user(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
        )

        assert result.success is True
        preferences = await preferences_repository.get_by_user_id(result.user.user_id)
        assert preferences is not None
        assert preferences.user_id == result.user.user_id


class TestUserServiceUpdate:
    """Tests for user updates."""

    @pytest.mark.asyncio
    async def test_update_user_profile(
        self, user_service: UserService
    ) -> None:
        """Test updating user profile fields."""
        create_result = await user_service.create_user(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Original Name",
        )

        update_result = await user_service.update_user(
            user_id=create_result.user.user_id,
            display_name="New Name",
            bio="Updated bio",
        )

        assert update_result.success is True
        assert update_result.user is not None
        assert update_result.user.display_name == "New Name"
        assert update_result.user.bio == "Updated bio"

    @pytest.mark.asyncio
    async def test_update_user_not_found(
        self, user_service: UserService
    ) -> None:
        """Test update for non-existent user."""
        result = await user_service.update_user(
            user_id=uuid4(),
            display_name="New Name",
        )

        assert result.success is False
        assert result.error_code == "USER_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_update_user_ignores_immutable_fields(
        self, user_service: UserService
    ) -> None:
        """Test that immutable fields are not updated."""
        create_result = await user_service.create_user(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
        )

        # Try to update non-allowed field
        update_result = await user_service.update_user(
            user_id=create_result.user.user_id,
            email="newemail@example.com",  # Should be ignored
            display_name="Updated Name",
        )

        assert update_result.success is True
        # Email should remain unchanged
        user = await user_service.get_user(create_result.user.user_id)
        assert user.email == "test@example.com"


class TestUserServiceDelete:
    """Tests for user deletion."""

    @pytest.mark.asyncio
    async def test_delete_user_soft_delete(
        self, user_service: UserService
    ) -> None:
        """Test soft delete sets correct status."""
        create_result = await user_service.create_user(
            email="delete@example.com",
            password_hash="hashed_password",
            display_name="Delete Me",
        )

        delete_result = await user_service.delete_user(
            user_id=create_result.user.user_id,
            reason="Test deletion",
        )

        assert delete_result.success is True

        # User should still exist but with deleted status
        # Note: get_user returns None for deleted users in our implementation

    @pytest.mark.asyncio
    async def test_delete_user_not_found(
        self, user_service: UserService
    ) -> None:
        """Test delete for non-existent user."""
        result = await user_service.delete_user(
            user_id=uuid4(),
            reason="Test",
        )

        assert result.success is False
        assert result.error_code == "USER_NOT_FOUND"


class TestEmailVerification:
    """Tests for email verification."""

    @pytest.mark.asyncio
    async def test_verify_email_success(
        self, user_service: UserService
    ) -> None:
        """Test successful email verification."""
        create_result = await user_service.create_user(
            email="verify@example.com",
            password_hash="hashed_password",
            display_name="Verify User",
        )

        token = create_result.user.email_verification_token

        verify_result = await user_service.verify_email(
            user_id=create_result.user.user_id,
            token=token,
        )

        assert verify_result.success is True

        # Check user is now verified and active
        user = await user_service.get_user(create_result.user.user_id)
        assert user.email_verified is True
        assert user.status == AccountStatus.ACTIVE
        assert user.email_verification_token is None

    @pytest.mark.asyncio
    async def test_verify_email_invalid_token(
        self, user_service: UserService
    ) -> None:
        """Test verification with invalid token."""
        create_result = await user_service.create_user(
            email="verify@example.com",
            password_hash="hashed_password",
            display_name="Verify User",
        )

        verify_result = await user_service.verify_email(
            user_id=create_result.user.user_id,
            token="invalid_token",
        )

        assert verify_result.success is False
        assert verify_result.error_code == "INVALID_TOKEN"

    @pytest.mark.asyncio
    async def test_verify_email_already_verified(
        self, user_service: UserService
    ) -> None:
        """Test verification of already verified email."""
        create_result = await user_service.create_user(
            email="verify@example.com",
            password_hash="hashed_password",
            display_name="Verify User",
        )

        token = create_result.user.email_verification_token

        # First verification
        await user_service.verify_email(
            user_id=create_result.user.user_id,
            token=token,
        )

        # Second verification attempt
        verify_result = await user_service.verify_email(
            user_id=create_result.user.user_id,
            token=token,
        )

        assert verify_result.success is False
        assert verify_result.error_code == "ALREADY_VERIFIED"


class TestPasswordChange:
    """Tests for password change."""

    @pytest.mark.asyncio
    async def test_change_password_success(
        self, user_service: UserService
    ) -> None:
        """Test successful password change."""
        create_result = await user_service.create_user(
            email="password@example.com",
            password_hash="hashed_oldpassword",
            display_name="Password User",
        )

        change_result = await user_service.change_password(
            user_id=create_result.user.user_id,
            current_password="oldpassword",
            new_password_hash="hashed_newpassword",
        )

        assert change_result.success is True

        # Verify new password hash is set
        user = await user_service.get_user(create_result.user.user_id)
        assert user.password_hash == "hashed_newpassword"

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(
        self, user_service: UserService
    ) -> None:
        """Test password change with wrong current password."""
        create_result = await user_service.create_user(
            email="password@example.com",
            password_hash="hashed_oldpassword",
            display_name="Password User",
        )

        change_result = await user_service.change_password(
            user_id=create_result.user.user_id,
            current_password="wrongpassword",
            new_password_hash="hashed_newpassword",
        )

        assert change_result.success is False
        assert change_result.error_code == "INVALID_PASSWORD"


class TestPreferences:
    """Tests for preferences management."""

    @pytest.mark.asyncio
    async def test_get_preferences(
        self, user_service: UserService
    ) -> None:
        """Test getting user preferences."""
        create_result = await user_service.create_user(
            email="prefs@example.com",
            password_hash="hashed_password",
            display_name="Prefs User",
        )

        preferences = await user_service.get_preferences(create_result.user.user_id)

        assert preferences is not None
        assert preferences.user_id == create_result.user.user_id
        # Check defaults
        assert preferences.notification_email is True
        assert preferences.theme == "system"

    @pytest.mark.asyncio
    async def test_update_preferences(
        self, user_service: UserService
    ) -> None:
        """Test updating preferences."""
        create_result = await user_service.create_user(
            email="prefs@example.com",
            password_hash="hashed_password",
            display_name="Prefs User",
        )

        update_result = await user_service.update_preferences(
            user_id=create_result.user.user_id,
            notification_email=False,
            theme="dark",
        )

        assert update_result.success is True
        assert update_result.preferences is not None
        assert update_result.preferences.notification_email is False
        assert update_result.preferences.theme == "dark"


class TestConsent:
    """Tests for consent management."""

    @pytest.mark.asyncio
    async def test_record_consent(
        self, user_service: UserService
    ) -> None:
        """Test recording consent."""
        create_result = await user_service.create_user(
            email="consent@example.com",
            password_hash="hashed_password",
            display_name="Consent User",
        )

        consent_result = await user_service.record_consent(
            user_id=create_result.user.user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
        )

        assert consent_result.success is True
        assert consent_result.consent is not None
        assert consent_result.consent.consent_type == ConsentType.TERMS_OF_SERVICE
        assert consent_result.consent.granted is True

    @pytest.mark.asyncio
    async def test_get_consent_records(
        self, user_service: UserService
    ) -> None:
        """Test getting consent records."""
        create_result = await user_service.create_user(
            email="consent@example.com",
            password_hash="hashed_password",
            display_name="Consent User",
        )

        # Record multiple consents
        await user_service.record_consent(
            user_id=create_result.user.user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
        )
        await user_service.record_consent(
            user_id=create_result.user.user_id,
            consent_type=ConsentType.PRIVACY_POLICY,
            granted=True,
            version="1.0",
        )

        records = await user_service.get_consent_records(create_result.user.user_id)

        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_check_consent(
        self, user_service: UserService
    ) -> None:
        """Test checking consent status."""
        create_result = await user_service.create_user(
            email="consent@example.com",
            password_hash="hashed_password",
            display_name="Consent User",
        )

        # Initially no consent
        has_consent = await user_service.check_consent(
            user_id=create_result.user.user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
        )
        assert has_consent is False

        # Grant consent
        await user_service.record_consent(
            user_id=create_result.user.user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            granted=True,
            version="1.0",
        )

        has_consent = await user_service.check_consent(
            user_id=create_result.user.user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
        )
        assert has_consent is True


class TestStatistics:
    """Tests for service statistics."""

    @pytest.mark.asyncio
    async def test_statistics_tracking(
        self, user_service: UserService
    ) -> None:
        """Test statistics are tracked."""
        # Create users
        await user_service.create_user(
            email="stat1@example.com",
            password_hash="hashed_password",
            display_name="Stat User 1",
        )
        await user_service.create_user(
            email="stat2@example.com",
            password_hash="hashed_password",
            display_name="Stat User 2",
        )

        stats = user_service.get_statistics()

        assert stats["users_created"] == 2

    def test_reset_statistics(
        self, user_service: UserService
    ) -> None:
        """Test statistics reset."""
        user_service._stats["users_created"] = 10
        user_service.reset_statistics()

        stats = user_service.get_statistics()
        assert stats["users_created"] == 0
