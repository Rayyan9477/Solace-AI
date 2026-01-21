"""
Tests for User Service authentication and session management.

Tests session lifecycle, authentication workflows, and token validation.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock
from uuid import UUID, uuid4

import pytest

from src.auth import (
    SessionConfig,
    UserSession,
    SessionManager,
    AuthenticationService,
    AuthenticationResult,
    TokenRefreshResult,
    SessionValidationResult,
    create_session_manager,
    create_authentication_service,
)
from src.domain.entities import User
from src.domain.value_objects import UserRole, AccountStatus


class TestSessionConfig:
    """Tests for SessionConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SessionConfig()

        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7
        assert config.session_timeout_minutes == 60
        assert config.max_sessions_per_user == 5
        assert config.enable_session_tracking is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SessionConfig(
            secret_key="a" * 32,
            access_token_expire_minutes=15,
            refresh_token_expire_days=14,
            session_timeout_minutes=120,
            max_sessions_per_user=10,
            enable_session_tracking=False,
        )

        assert config.access_token_expire_minutes == 15
        assert config.refresh_token_expire_days == 14
        assert config.session_timeout_minutes == 120
        assert config.max_sessions_per_user == 10
        assert config.enable_session_tracking is False


class TestUserSession:
    """Tests for UserSession model."""

    @pytest.fixture
    def user_id(self) -> UUID:
        """Create a sample user ID."""
        return uuid4()

    def test_session_creation(self, user_id: UUID) -> None:
        """Test session is created with defaults."""
        session = UserSession(user_id=user_id)

        assert session.session_id is not None
        assert session.user_id == user_id
        assert session.created_at is not None
        assert session.last_activity is not None
        assert session.is_active is True
        assert session.ip_address is None
        assert session.user_agent is None
        assert session.device_info == {}

    def test_session_with_metadata(self, user_id: UUID) -> None:
        """Test session with full metadata."""
        session = UserSession(
            user_id=user_id,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            device_info={"os": "Windows", "browser": "Chrome"},
        )

        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0"
        assert session.device_info == {"os": "Windows", "browser": "Chrome"}

    def test_update_activity(self, user_id: UUID) -> None:
        """Test activity timestamp update."""
        session = UserSession(user_id=user_id)
        original_activity = session.last_activity

        # Simulate some time passing
        import time
        time.sleep(0.01)

        session.update_activity()

        assert session.last_activity > original_activity

    def test_is_expired_when_active(self, user_id: UUID) -> None:
        """Test session not expired when recent."""
        session = UserSession(user_id=user_id)

        assert session.is_expired(timeout_minutes=60) is False

    def test_is_expired_when_inactive(self, user_id: UUID) -> None:
        """Test session expired after timeout."""
        session = UserSession(user_id=user_id)
        # Set last activity to past
        session.last_activity = datetime.now(timezone.utc) - timedelta(minutes=61)

        assert session.is_expired(timeout_minutes=60) is True

    def test_is_expired_when_deactivated(self, user_id: UUID) -> None:
        """Test deactivated session is expired."""
        session = UserSession(user_id=user_id)
        session.deactivate()

        assert session.is_expired(timeout_minutes=60) is True
        assert session.is_active is False

    def test_deactivate(self, user_id: UUID) -> None:
        """Test session deactivation."""
        session = UserSession(user_id=user_id)
        assert session.is_active is True

        session.deactivate()

        assert session.is_active is False


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def config(self) -> SessionConfig:
        """Create session config."""
        return SessionConfig(
            secret_key="a" * 32,
            max_sessions_per_user=3,
            session_timeout_minutes=60,
        )

    @pytest.fixture
    def session_manager(self, config: SessionConfig) -> SessionManager:
        """Create session manager instance."""
        return SessionManager(config)

    @pytest.fixture
    def sample_user(self) -> User:
        """Create a sample user."""
        return User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            role=UserRole.USER,
            status=AccountStatus.ACTIVE,
        )

    @pytest.mark.asyncio
    async def test_create_session(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test creating a new session."""
        session = await session_manager.create_session(
            user=sample_user,
            ip_address="192.168.1.1",
            user_agent="Test Agent",
        )

        assert session.session_id is not None
        assert session.user_id == sample_user.user_id
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Test Agent"
        assert session.is_active is True

    @pytest.mark.asyncio
    async def test_create_session_with_device_info(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test creating session with device info."""
        device_info = {"os": "macOS", "browser": "Safari"}
        session = await session_manager.create_session(
            user=sample_user,
            device_info=device_info,
        )

        assert session.device_info == device_info

    @pytest.mark.asyncio
    async def test_get_session(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test getting session by ID."""
        created = await session_manager.create_session(user=sample_user)

        retrieved = await session_manager.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(
        self, session_manager: SessionManager
    ) -> None:
        """Test getting non-existent session returns None."""
        result = await session_manager.get_session(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_session(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test validating active session."""
        session = await session_manager.create_session(user=sample_user)

        is_valid = await session_manager.validate_session(session.session_id)

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_nonexistent_session(
        self, session_manager: SessionManager
    ) -> None:
        """Test validating non-existent session."""
        is_valid = await session_manager.validate_session(uuid4())

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_update_activity(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test updating session activity."""
        session = await session_manager.create_session(user=sample_user)
        original_activity = session.last_activity

        import time
        time.sleep(0.01)

        await session_manager.update_activity(session.session_id)

        updated = await session_manager.get_session(session.session_id)
        assert updated.last_activity > original_activity

    @pytest.mark.asyncio
    async def test_get_user_sessions(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test getting all sessions for a user."""
        # Create multiple sessions
        session1 = await session_manager.create_session(user=sample_user)
        session2 = await session_manager.create_session(user=sample_user)

        sessions = await session_manager.get_user_sessions(sample_user.user_id)

        assert len(sessions) == 2
        session_ids = [s.session_id for s in sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

    @pytest.mark.asyncio
    async def test_get_user_sessions_active_only(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test getting only active sessions."""
        session1 = await session_manager.create_session(user=sample_user)
        session2 = await session_manager.create_session(user=sample_user)

        # Revoke one session
        await session_manager.revoke_session(session1.session_id)

        sessions = await session_manager.get_user_sessions(
            sample_user.user_id, active_only=True
        )

        assert len(sessions) == 1
        assert sessions[0].session_id == session2.session_id

    @pytest.mark.asyncio
    async def test_revoke_session(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test revoking a session."""
        session = await session_manager.create_session(user=sample_user)

        result = await session_manager.revoke_session(session.session_id)

        assert result is True

        # Session should no longer be valid
        is_valid = await session_manager.validate_session(session.session_id)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_session(
        self, session_manager: SessionManager
    ) -> None:
        """Test revoking non-existent session."""
        result = await session_manager.revoke_session(uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_all_user_sessions(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test revoking all sessions for a user."""
        await session_manager.create_session(user=sample_user)
        await session_manager.create_session(user=sample_user)
        await session_manager.create_session(user=sample_user)

        revoked = await session_manager.revoke_all_user_sessions(sample_user.user_id)

        assert revoked == 3

        # All sessions should be gone
        sessions = await session_manager.get_user_sessions(
            sample_user.user_id, active_only=True
        )
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_max_sessions_enforced(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test maximum sessions per user is enforced."""
        # Config has max_sessions_per_user=3
        session1 = await session_manager.create_session(user=sample_user)
        session2 = await session_manager.create_session(user=sample_user)
        session3 = await session_manager.create_session(user=sample_user)

        # Creating 4th session should revoke oldest
        session4 = await session_manager.create_session(user=sample_user)

        sessions = await session_manager.get_user_sessions(sample_user.user_id)

        # Should only have 3 sessions
        assert len(sessions) == 3

        # First session should be revoked
        session_ids = [s.session_id for s in sessions]
        assert session1.session_id not in session_ids
        assert session4.session_id in session_ids

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test cleanup removes expired sessions."""
        session = await session_manager.create_session(user=sample_user)

        # Make session expired
        stored_session = session_manager._sessions[session.session_id]
        stored_session.last_activity = datetime.now(timezone.utc) - timedelta(minutes=61)

        cleaned = await session_manager.cleanup_expired_sessions()

        assert cleaned == 1

    @pytest.mark.asyncio
    async def test_get_statistics(
        self, session_manager: SessionManager, sample_user: User
    ) -> None:
        """Test statistics gathering."""
        await session_manager.create_session(user=sample_user)
        await session_manager.create_session(user=sample_user)

        stats = session_manager.get_statistics()

        assert stats["active_sessions"] == 2
        assert stats["sessions_created"] == 2
        assert stats["sessions_revoked"] == 0


class TestAuthenticationService:
    """Tests for AuthenticationService."""

    @pytest.fixture
    def session_manager(self) -> SessionManager:
        """Create session manager."""
        return SessionManager(SessionConfig(secret_key="a" * 32))

    @pytest.fixture
    def mock_password_service(self) -> Mock:
        """Create mock password service."""
        service = Mock()
        service.verify_password = Mock(
            return_value=Mock(is_valid=True, needs_rehash=False)
        )
        return service

    @pytest.fixture
    def mock_jwt_service(self) -> Mock:
        """Create mock JWT service."""
        service = Mock()
        service.generate_token_pair = Mock(
            return_value=Mock(
                access_token="access_token_123",
                refresh_token="refresh_token_456",
            )
        )
        service.verify_token = Mock(
            return_value=Mock(
                user_id=uuid4(),
                email="test@example.com",
                role="user",
                session_id=None,
            )
        )
        return service

    @pytest.fixture
    def auth_service(
        self,
        session_manager: SessionManager,
        mock_password_service: Mock,
        mock_jwt_service: Mock,
    ) -> AuthenticationService:
        """Create authentication service."""
        return AuthenticationService(
            session_manager=session_manager,
            password_service=mock_password_service,
            jwt_service=mock_jwt_service,
        )

    @pytest.fixture
    def sample_user(self) -> User:
        """Create a sample user."""
        return User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            role=UserRole.USER,
            status=AccountStatus.ACTIVE,
        )

    @pytest.mark.asyncio
    async def test_authenticate_success(
        self,
        auth_service: AuthenticationService,
        sample_user: User,
        mock_password_service: Mock,
    ) -> None:
        """Test successful authentication."""
        result = await auth_service.authenticate(
            user=sample_user,
            password="password123",
            ip_address="192.168.1.1",
        )

        assert result.success is True
        assert result.user == sample_user
        assert result.session is not None
        assert result.access_token == "access_token_123"
        assert result.refresh_token == "refresh_token_456"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_authenticate_invalid_password(
        self,
        auth_service: AuthenticationService,
        sample_user: User,
        mock_password_service: Mock,
    ) -> None:
        """Test authentication with invalid password."""
        mock_password_service.verify_password.return_value.is_valid = False

        result = await auth_service.authenticate(
            user=sample_user,
            password="wrong_password",
        )

        assert result.success is False
        assert result.error == "Invalid credentials"
        assert result.error_code == "INVALID_CREDENTIALS"

    @pytest.mark.asyncio
    async def test_authenticate_without_password_service(
        self,
        session_manager: SessionManager,
        sample_user: User,
    ) -> None:
        """Test authentication fails without password service."""
        service = AuthenticationService(
            session_manager=session_manager,
            password_service=None,
        )

        result = await service.authenticate(
            user=sample_user,
            password="password123",
        )

        assert result.success is False
        assert result.error_code == "SERVICE_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_authenticate_blocked_user(
        self,
        auth_service: AuthenticationService,
        mock_password_service: Mock,
    ) -> None:
        """Test authentication for blocked user."""
        blocked_user = User(
            email="blocked@example.com",
            password_hash="hash",
            display_name="Blocked User",
            status=AccountStatus.SUSPENDED,
        )

        result = await auth_service.authenticate(
            user=blocked_user,
            password="password123",
        )

        assert result.success is False
        assert result.error_code == "LOGIN_BLOCKED"

    @pytest.mark.asyncio
    async def test_logout(
        self,
        auth_service: AuthenticationService,
        sample_user: User,
    ) -> None:
        """Test logout revokes session."""
        # First authenticate
        auth_result = await auth_service.authenticate(
            user=sample_user,
            password="password123",
        )

        # Then logout
        result = await auth_service.logout(auth_result.session.session_id)

        assert result is True

        # Session should be invalid
        is_valid = await auth_service._session_manager.validate_session(
            auth_result.session.session_id
        )
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_logout_all(
        self,
        auth_service: AuthenticationService,
        sample_user: User,
    ) -> None:
        """Test logout all revokes all sessions."""
        # Create multiple sessions
        await auth_service.authenticate(user=sample_user, password="password123")
        await auth_service.authenticate(user=sample_user, password="password123")

        # Logout all
        count = await auth_service.logout_all(sample_user.user_id)

        assert count == 2

    @pytest.mark.asyncio
    async def test_validate_token_without_jwt_service(
        self,
        session_manager: SessionManager,
    ) -> None:
        """Test validate token fails without JWT service."""
        service = AuthenticationService(
            session_manager=session_manager,
            jwt_service=None,
        )

        result = await service.validate_token("some_token")

        assert result.valid is False
        assert result.error_code == "SERVICE_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_refresh_tokens_without_jwt_service(
        self,
        session_manager: SessionManager,
    ) -> None:
        """Test refresh tokens fails without JWT service."""
        service = AuthenticationService(
            session_manager=session_manager,
            jwt_service=None,
        )

        result = await service.refresh_tokens("some_token")

        assert result.success is False
        assert result.error_code == "SERVICE_UNAVAILABLE"


class TestResultTypes:
    """Tests for result type dataclasses."""

    def test_authentication_result_defaults(self) -> None:
        """Test AuthenticationResult default values."""
        result = AuthenticationResult()

        assert result.success is False
        assert result.user is None
        assert result.session is None
        assert result.access_token is None
        assert result.refresh_token is None
        assert result.error is None
        assert result.error_code is None

    def test_authentication_result_success(self) -> None:
        """Test AuthenticationResult for success."""
        user = User(
            email="test@example.com",
            password_hash="hash",
            display_name="Test",
        )
        result = AuthenticationResult(
            success=True,
            user=user,
            access_token="token",
        )

        assert result.success is True
        assert result.user == user

    def test_token_refresh_result_defaults(self) -> None:
        """Test TokenRefreshResult default values."""
        result = TokenRefreshResult()

        assert result.success is False
        assert result.access_token is None
        assert result.refresh_token is None
        assert result.error is None

    def test_session_validation_result_defaults(self) -> None:
        """Test SessionValidationResult default values."""
        result = SessionValidationResult()

        assert result.valid is False
        assert result.user_id is None
        assert result.session_id is None
        assert result.email is None
        assert result.role is None


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_session_manager(self) -> None:
        """Test session manager factory."""
        manager = create_session_manager()

        assert manager is not None
        assert isinstance(manager, SessionManager)

    def test_create_session_manager_with_config(self) -> None:
        """Test session manager factory with config."""
        config = SessionConfig(
            secret_key="a" * 32,
            max_sessions_per_user=10,
        )
        manager = create_session_manager(config)

        assert manager._config.max_sessions_per_user == 10

    def test_create_authentication_service(self) -> None:
        """Test authentication service factory."""
        session_manager = SessionManager()
        service = create_authentication_service(
            session_manager=session_manager,
        )

        assert service is not None
        assert isinstance(service, AuthenticationService)
        assert service._session_manager == session_manager

    def test_create_authentication_service_with_services(self) -> None:
        """Test authentication service factory with optional services."""
        session_manager = SessionManager()
        jwt_service = Mock()
        password_service = Mock()

        service = create_authentication_service(
            session_manager=session_manager,
            jwt_service=jwt_service,
            password_service=password_service,
        )

        assert service._jwt_service == jwt_service
        assert service._password_service == password_service
