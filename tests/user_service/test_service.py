"""
Tests for User Service domain service.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from services.user_service.src.domain.service import (
    UserService,
    UserServiceSettings,
    User,
    UserPreferences,
    UserRole,
    AccountStatus,
)
from services.user_service.src.infrastructure.repository import (
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
)
from services.user_service.src.auth import SessionManager, SessionConfig


@pytest.fixture
def user_repo() -> InMemoryUserRepository:
    """Create a fresh user repository for each test."""
    return InMemoryUserRepository()


@pytest.fixture
def prefs_repo() -> InMemoryUserPreferencesRepository:
    """Create a fresh preferences repository for each test."""
    return InMemoryUserPreferencesRepository()


@pytest.fixture
def session_manager() -> SessionManager:
    """Create a session manager for testing."""
    return SessionManager(SessionConfig())


@pytest.fixture
def user_service(user_repo: InMemoryUserRepository, prefs_repo: InMemoryUserPreferencesRepository, session_manager: SessionManager) -> UserService:
    """Create a user service for testing."""
    return UserService(
        settings=UserServiceSettings(),
        user_repository=user_repo,
        preferences_repository=prefs_repo,
        session_manager=session_manager,
    )


class TestUserServiceInitialization:
    """Tests for UserService initialization."""

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized_flag(self, user_service: UserService) -> None:
        """Test that initialize sets the initialized flag."""
        assert not user_service._initialized
        await user_service.initialize()
        assert user_service._initialized

    @pytest.mark.asyncio
    async def test_shutdown_clears_initialized_flag(self, user_service: UserService) -> None:
        """Test that shutdown clears the initialized flag."""
        await user_service.initialize()
        assert user_service._initialized
        await user_service.shutdown()
        assert not user_service._initialized

    @pytest.mark.asyncio
    async def test_get_status_returns_operational_when_initialized(self, user_service: UserService) -> None:
        """Test that get_status returns operational status when initialized."""
        await user_service.initialize()
        status = await user_service.get_status()
        assert status["status"] == "operational"
        assert status["initialized"] is True


class TestUserCreation:
    """Tests for user creation."""

    @pytest.mark.asyncio
    async def test_create_user_success(self, user_service: UserService) -> None:
        """Test successful user creation."""
        await user_service.initialize()
        result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        assert result.user is not None
        assert result.error is None
        assert result.user.email == "test@example.com"
        assert result.user.display_name == "Test User"
        assert result.user.role == UserRole.USER
        assert result.user.status == AccountStatus.PENDING_VERIFICATION

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, user_service: UserService) -> None:
        """Test that duplicate email returns error."""
        await user_service.initialize()
        await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss456!",
            display_name="Another User",
        )
        assert result.user is None
        assert result.error == "Email already registered"

    @pytest.mark.asyncio
    async def test_create_user_weak_password(self, user_service: UserService) -> None:
        """Test that weak password returns error."""
        await user_service.initialize()
        result = await user_service.create_user(
            email="test@example.com",
            password="weak",
            display_name="Test User",
        )
        assert result.user is None
        assert "Password must be at least" in result.error

    @pytest.mark.asyncio
    async def test_create_user_password_without_special_char(self, user_service: UserService) -> None:
        """Test that password without special character returns error."""
        await user_service.initialize()
        result = await user_service.create_user(
            email="test@example.com",
            password="LongPassword123",
            display_name="Test User",
        )
        assert result.user is None
        assert "special character" in result.error

    @pytest.mark.asyncio
    async def test_create_user_creates_preferences(self, user_service: UserService, prefs_repo: InMemoryUserPreferencesRepository) -> None:
        """Test that user creation also creates preferences."""
        await user_service.initialize()
        result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        assert result.user is not None
        prefs = await prefs_repo.get_by_user(result.user.user_id)
        assert prefs is not None
        assert prefs.notification_email is True


class TestUserRetrieval:
    """Tests for user retrieval."""

    @pytest.mark.asyncio
    async def test_get_user_success(self, user_service: UserService) -> None:
        """Test successful user retrieval."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        user = await user_service.get_user(create_result.user.user_id)
        assert user is not None
        assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, user_service: UserService) -> None:
        """Test user retrieval returns None for non-existent user."""
        await user_service.initialize()
        user = await user_service.get_user(uuid4())
        assert user is None


class TestUserUpdate:
    """Tests for user update."""

    @pytest.mark.asyncio
    async def test_update_user_success(self, user_service: UserService) -> None:
        """Test successful user update."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.update_user(
            create_result.user.user_id,
            {"display_name": "Updated Name", "bio": "My bio"},
        )
        assert result.user is not None
        assert result.error is None
        assert result.user.display_name == "Updated Name"
        assert result.user.bio == "My bio"

    @pytest.mark.asyncio
    async def test_update_user_not_found(self, user_service: UserService) -> None:
        """Test update returns error for non-existent user."""
        await user_service.initialize()
        result = await user_service.update_user(uuid4(), {"display_name": "New Name"})
        assert result.error == "User not found"


class TestUserDeletion:
    """Tests for user deletion."""

    @pytest.mark.asyncio
    async def test_delete_user_success(self, user_service: UserService) -> None:
        """Test successful user deletion."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.delete_user(
            create_result.user.user_id,
            "SecureP@ss123!",
            "Testing deletion",
        )
        assert result.success is True
        user = await user_service.get_user(create_result.user.user_id)
        assert user is None

    @pytest.mark.asyncio
    async def test_delete_user_wrong_password(self, user_service: UserService) -> None:
        """Test deletion with wrong password fails."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.delete_user(
            create_result.user.user_id,
            "WrongPassword!",
        )
        assert result.success is False
        assert result.error == "Invalid password"


class TestPreferences:
    """Tests for user preferences."""

    @pytest.mark.asyncio
    async def test_get_preferences(self, user_service: UserService) -> None:
        """Test getting user preferences."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        prefs = await user_service.get_preferences(create_result.user.user_id)
        assert prefs is not None
        assert prefs.notification_email is True

    @pytest.mark.asyncio
    async def test_update_preferences(self, user_service: UserService) -> None:
        """Test updating user preferences."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.update_preferences(
            create_result.user.user_id,
            {"notification_email": False, "theme": "dark"},
        )
        assert result.preferences is not None
        assert result.preferences.notification_email is False
        assert result.preferences.theme == "dark"


class TestConsent:
    """Tests for consent management."""

    @pytest.mark.asyncio
    async def test_record_consent(self, user_service: UserService) -> None:
        """Test recording consent."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.record_consent(
            user_id=create_result.user.user_id,
            consent_type="terms_of_service",
            granted=True,
            version="1.0",
            ip_address="127.0.0.1",
        )
        assert result.consent is not None
        assert result.consent.consent_type == "terms_of_service"
        assert result.consent.granted is True

    @pytest.mark.asyncio
    async def test_get_consent_records(self, user_service: UserService) -> None:
        """Test getting consent records."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        await user_service.record_consent(
            user_id=create_result.user.user_id,
            consent_type="terms_of_service",
            granted=True,
            version="1.0",
        )
        records = await user_service.get_consent_records(create_result.user.user_id)
        assert len(records) == 1
        assert records[0].consent_type == "terms_of_service"


class TestPasswordManagement:
    """Tests for password management."""

    @pytest.mark.asyncio
    async def test_change_password_success(self, user_service: UserService) -> None:
        """Test successful password change."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.change_password(
            create_result.user.user_id,
            "SecureP@ss123!",
            "NewSecureP@ss456!",
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, user_service: UserService) -> None:
        """Test password change with wrong current password."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.change_password(
            create_result.user.user_id,
            "WrongPassword!",
            "NewSecureP@ss456!",
        )
        assert result.success is False
        assert result.error == "Current password is incorrect"


class TestEmailVerification:
    """Tests for email verification."""

    @pytest.mark.asyncio
    async def test_verify_email_success(self, user_service: UserService, user_repo: InMemoryUserRepository) -> None:
        """Test successful email verification."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        user = await user_repo.get_by_id(create_result.user.user_id)
        token = user.email_verification_token
        result = await user_service.verify_email(create_result.user.user_id, token)
        assert result.success is True
        user = await user_repo.get_by_id(create_result.user.user_id)
        assert user.email_verified is True
        assert user.status == AccountStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_verify_email_invalid_token(self, user_service: UserService) -> None:
        """Test email verification with invalid token."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        result = await user_service.verify_email(create_result.user.user_id, "invalid-token")
        assert result.success is False
        assert result.error == "Invalid verification token"


class TestProgress:
    """Tests for user progress."""

    @pytest.mark.asyncio
    async def test_get_progress(self, user_service: UserService) -> None:
        """Test getting user progress."""
        await user_service.initialize()
        create_result = await user_service.create_user(
            email="test@example.com",
            password="SecureP@ss123!",
            display_name="Test User",
        )
        progress = await user_service.get_progress(create_result.user.user_id)
        assert progress is not None
        assert progress.user_id == create_result.user.user_id
        assert progress.total_sessions == 0
