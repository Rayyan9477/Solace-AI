"""
Solace-AI User Service - Authentication and Session Management.

Handles user authentication workflows, session lifecycle, and token management.
Integrates with JWT service for token operations.

Architecture Layer: Application
Principles: Single Responsibility, Separation of Concerns
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from .domain.entities import User
from .domain.value_objects import UserRole

logger = structlog.get_logger(__name__)


# --- Configuration ---


class SessionConfig(BaseSettings):
    """Session management configuration."""

    secret_key: str = Field(
        default="your-secret-key-change-in-production-32-chars",
        min_length=32,
        description="Secret key for session signing",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Access token expiry in minutes",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Refresh token expiry in days",
    )
    session_timeout_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Session inactivity timeout in minutes",
    )
    max_sessions_per_user: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent sessions per user",
    )
    enable_session_tracking: bool = Field(
        default=True,
        description="Enable session tracking",
    )

    model_config = SettingsConfigDict(
        env_prefix="SESSION_",
        env_file=".env",
        extra="ignore",
    )


# --- Session Models ---


class UserSession(BaseModel):
    """
    User session representation.

    Tracks active user sessions with metadata for audit and security.
    """

    session_id: UUID = Field(default_factory=uuid4, description="Unique session ID")
    user_id: UUID = Field(..., description="User identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session creation time",
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last activity time",
    )
    ip_address: str | None = Field(default=None, description="Client IP address")
    user_agent: str | None = Field(default=None, description="Client user agent")
    device_info: dict[str, str] = Field(
        default_factory=dict,
        description="Device information",
    )
    is_active: bool = Field(default=True, description="Session active status")

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if session has expired due to inactivity."""
        if not self.is_active:
            return True
        expiry_time = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now(timezone.utc) > expiry_time

    def deactivate(self) -> None:
        """Deactivate the session."""
        self.is_active = False


# --- Result Types ---


@dataclass
class AuthenticationResult:
    """Result of authentication operation."""

    success: bool = False
    user: User | None = None
    session: UserSession | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class TokenRefreshResult:
    """Result of token refresh operation."""

    success: bool = False
    access_token: str | None = None
    refresh_token: str | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class SessionValidationResult:
    """Result of session validation."""

    valid: bool = False
    user_id: UUID | None = None
    session_id: UUID | None = None
    email: str | None = None
    role: str | None = None
    error: str | None = None
    error_code: str | None = None


# --- Session Manager ---


class SessionManager:
    """
    Manages user sessions with support for multiple concurrent sessions.

    Responsibilities:
    - Session creation and tracking
    - Session validation and refresh
    - Session revocation
    - Concurrent session limiting

    Thread-safe using asyncio locks.
    """

    def __init__(self, config: SessionConfig | None = None) -> None:
        """
        Initialize session manager.

        Args:
            config: Session configuration
        """
        self._config = config or SessionConfig()
        self._sessions: dict[UUID, UserSession] = {}
        self._user_sessions: dict[UUID, list[UUID]] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "sessions_created": 0,
            "sessions_revoked": 0,
            "sessions_expired": 0,
        }

    async def create_session(
        self,
        user: User,
        ip_address: str | None = None,
        user_agent: str | None = None,
        device_info: dict[str, str] | None = None,
    ) -> UserSession:
        """
        Create a new session for user.

        Enforces maximum concurrent sessions by removing oldest
        session if limit is reached.

        Args:
            user: User entity
            ip_address: Client IP address
            user_agent: Client user agent
            device_info: Additional device information

        Returns:
            Created session
        """
        async with self._lock:
            # Check and enforce session limit
            user_session_ids = self._user_sessions.get(user.user_id, [])
            while len(user_session_ids) >= self._config.max_sessions_per_user:
                # Revoke oldest session
                oldest_session_id = user_session_ids[0]
                await self._revoke_session_internal(oldest_session_id)
                user_session_ids = self._user_sessions.get(user.user_id, [])

            # Create new session
            session = UserSession(
                user_id=user.user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                device_info=device_info or {},
            )

            # Store session
            self._sessions[session.session_id] = session
            if user.user_id not in self._user_sessions:
                self._user_sessions[user.user_id] = []
            self._user_sessions[user.user_id].append(session.session_id)

            self._stats["sessions_created"] += 1

            logger.info(
                "session_created",
                user_id=str(user.user_id),
                session_id=str(session.session_id),
                ip_address=ip_address,
            )

            return session

    async def get_session(self, session_id: UUID) -> UserSession | None:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session or None if not found/expired
        """
        session = self._sessions.get(session_id)
        if session and session.is_active:
            if session.is_expired(self._config.session_timeout_minutes):
                await self.revoke_session(session_id)
                return None
            return session
        return None

    async def validate_session(self, session_id: UUID) -> bool:
        """
        Validate session is active.

        Args:
            session_id: Session identifier

        Returns:
            True if session is valid and active
        """
        session = await self.get_session(session_id)
        return session is not None

    async def update_activity(self, session_id: UUID) -> None:
        """
        Update session last activity timestamp.

        Args:
            session_id: Session identifier
        """
        session = self._sessions.get(session_id)
        if session and session.is_active:
            session.update_activity()

    async def get_user_sessions(
        self,
        user_id: UUID,
        active_only: bool = True,
    ) -> list[UserSession]:
        """
        Get all sessions for a user.

        Args:
            user_id: User identifier
            active_only: Only return active sessions

        Returns:
            List of user sessions
        """
        session_ids = self._user_sessions.get(user_id, [])
        sessions = []
        for sid in session_ids:
            session = self._sessions.get(sid)
            if session:
                if active_only and not session.is_active:
                    continue
                if active_only and session.is_expired(self._config.session_timeout_minutes):
                    continue
                sessions.append(session)
        return sessions

    async def revoke_session(self, session_id: UUID) -> bool:
        """
        Revoke a specific session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was revoked
        """
        async with self._lock:
            return await self._revoke_session_internal(session_id)

    async def _revoke_session_internal(self, session_id: UUID) -> bool:
        """
        Internal session revocation (must be called with lock held).

        Args:
            session_id: Session identifier

        Returns:
            True if session was revoked
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.deactivate()

        # Remove from user sessions list
        user_sessions = self._user_sessions.get(session.user_id, [])
        if session_id in user_sessions:
            user_sessions.remove(session_id)

        self._stats["sessions_revoked"] += 1

        logger.info(
            "session_revoked",
            session_id=str(session_id),
            user_id=str(session.user_id),
        )

        return True

    async def revoke_all_user_sessions(self, user_id: UUID) -> int:
        """
        Revoke all sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of sessions revoked
        """
        async with self._lock:
            session_ids = self._user_sessions.get(user_id, [])[:]
            revoked = 0
            for sid in session_ids:
                if await self._revoke_session_internal(sid):
                    revoked += 1

            logger.info(
                "all_user_sessions_revoked",
                user_id=str(user_id),
                count=revoked,
            )

            return revoked

    async def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        async with self._lock:
            expired = []
            for session_id, session in self._sessions.items():
                if session.is_active and session.is_expired(
                    self._config.session_timeout_minutes
                ):
                    session.deactivate()
                    expired.append(session_id)

            self._stats["sessions_expired"] += len(expired)

            if expired:
                logger.info("expired_sessions_cleaned", count=len(expired))

            return len(expired)

    def get_statistics(self) -> dict[str, Any]:
        """Get session manager statistics."""
        active_count = sum(
            1 for s in self._sessions.values()
            if s.is_active and not s.is_expired(self._config.session_timeout_minutes)
        )
        return {
            "active_sessions": active_count,
            "total_sessions": len(self._sessions),
            **self._stats,
        }


# --- Authentication Service ---


class AuthenticationService:
    """
    High-level authentication service coordinating login/logout flows.

    Integrates with:
    - JWTService for token operations
    - PasswordService for credential verification
    - SessionManager for session tracking
    - UserService for user operations
    """

    def __init__(
        self,
        session_manager: SessionManager,
        jwt_service: Any | None = None,
        password_service: Any | None = None,
    ) -> None:
        """
        Initialize authentication service.

        Args:
            session_manager: Session manager instance
            jwt_service: JWT service for token operations
            password_service: Password service for credential verification
        """
        self._session_manager = session_manager
        self._jwt_service = jwt_service
        self._password_service = password_service

    async def authenticate(
        self,
        user: User,
        password: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuthenticationResult:
        """
        Authenticate user with credentials.

        Args:
            user: User entity
            password: Plaintext password
            ip_address: Client IP
            user_agent: Client user agent

        Returns:
            AuthenticationResult with tokens and session
        """
        if not self._password_service:
            return AuthenticationResult(
                error="Password service not configured",
                error_code="SERVICE_UNAVAILABLE",
            )

        # Verify credentials
        verify_result = self._password_service.verify_password(
            password, user.password_hash
        )
        if not verify_result.is_valid:
            return AuthenticationResult(
                error="Invalid credentials",
                error_code="INVALID_CREDENTIALS",
            )

        # Check if user can login
        can_login, error = user.can_login()
        if not can_login:
            return AuthenticationResult(
                error=error or "Login not allowed",
                error_code="LOGIN_BLOCKED",
            )

        # Create session
        session = await self._session_manager.create_session(
            user=user,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Generate tokens
        if self._jwt_service:
            token_pair = self._jwt_service.generate_token_pair(
                user_id=user.user_id,
                email=user.email,
                role=user.role.value,
                session_id=session.session_id,
            )
            access_token = token_pair.access_token
            refresh_token = token_pair.refresh_token
        else:
            access_token = None
            refresh_token = None

        logger.info(
            "user_authenticated",
            user_id=str(user.user_id),
            session_id=str(session.session_id),
        )

        return AuthenticationResult(
            success=True,
            user=user,
            session=session,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    async def refresh_tokens(
        self,
        refresh_token: str,
    ) -> TokenRefreshResult:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            TokenRefreshResult with new tokens
        """
        if not self._jwt_service:
            return TokenRefreshResult(
                error="JWT service not configured",
                error_code="SERVICE_UNAVAILABLE",
            )

        try:
            # Verify refresh token
            from .infrastructure.jwt_service import TokenType

            payload = self._jwt_service.verify_token(
                refresh_token,
                expected_type=TokenType.REFRESH,
            )

            # Validate session if session_id present
            if payload.session_id:
                session_valid = await self._session_manager.validate_session(
                    payload.session_id
                )
                if not session_valid:
                    return TokenRefreshResult(
                        error="Session expired",
                        error_code="SESSION_EXPIRED",
                    )

                # Update session activity
                await self._session_manager.update_activity(payload.session_id)

            # Generate new tokens
            token_pair = self._jwt_service.generate_token_pair(
                user_id=payload.user_id,
                email=payload.email,
                role=payload.role,
                session_id=payload.session_id,
            )

            logger.info(
                "tokens_refreshed",
                user_id=str(payload.user_id),
            )

            return TokenRefreshResult(
                success=True,
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
            )

        except Exception as e:
            logger.warning("token_refresh_failed", error=str(e))
            return TokenRefreshResult(
                error=str(e),
                error_code="REFRESH_FAILED",
            )

    async def validate_token(
        self,
        access_token: str,
    ) -> SessionValidationResult:
        """
        Validate access token and session.

        Args:
            access_token: Access token to validate

        Returns:
            SessionValidationResult with user info
        """
        if not self._jwt_service:
            return SessionValidationResult(
                error="JWT service not configured",
                error_code="SERVICE_UNAVAILABLE",
            )

        try:
            from .infrastructure.jwt_service import TokenType

            payload = self._jwt_service.verify_token(
                access_token,
                expected_type=TokenType.ACCESS,
            )

            # Validate session if session_id present
            if payload.session_id:
                session_valid = await self._session_manager.validate_session(
                    payload.session_id
                )
                if not session_valid:
                    return SessionValidationResult(
                        error="Session expired",
                        error_code="SESSION_EXPIRED",
                    )

                # Update session activity
                await self._session_manager.update_activity(payload.session_id)

            return SessionValidationResult(
                valid=True,
                user_id=payload.user_id,
                session_id=payload.session_id,
                email=payload.email,
                role=payload.role,
            )

        except Exception as e:
            return SessionValidationResult(
                error=str(e),
                error_code="VALIDATION_FAILED",
            )

    async def logout(
        self,
        session_id: UUID,
    ) -> bool:
        """
        Logout by revoking session.

        Args:
            session_id: Session to revoke

        Returns:
            True if session was revoked
        """
        result = await self._session_manager.revoke_session(session_id)
        if result:
            logger.info("user_logged_out", session_id=str(session_id))
        return result

    async def logout_all(
        self,
        user_id: UUID,
    ) -> int:
        """
        Logout all sessions for user.

        Args:
            user_id: User identifier

        Returns:
            Number of sessions revoked
        """
        count = await self._session_manager.revoke_all_user_sessions(user_id)
        logger.info("user_logged_out_all", user_id=str(user_id), count=count)
        return count


# --- Factory Functions ---


def create_session_manager(config: SessionConfig | None = None) -> SessionManager:
    """Create session manager instance."""
    return SessionManager(config)


def create_authentication_service(
    session_manager: SessionManager,
    jwt_service: Any | None = None,
    password_service: Any | None = None,
) -> AuthenticationService:
    """Create authentication service instance."""
    return AuthenticationService(
        session_manager=session_manager,
        jwt_service=jwt_service,
        password_service=password_service,
    )
