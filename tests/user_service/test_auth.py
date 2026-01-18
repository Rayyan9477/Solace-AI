"""
Tests for User Service authentication and session management.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4

from services.user_service.src.auth import (
    SessionManager,
    SessionConfig,
    TokenService,
    TokenClaims,
    TokenType,
    TokenCodec,
    AuthResult,
    TokenValidationResult,
)
from services.user_service.src.domain.service import User, UserRole, AccountStatus


def create_test_user(user_id: UUID | None = None, email: str = "test@example.com") -> User:
    """Create a test user."""
    return User(
        user_id=user_id or uuid4(),
        email=email,
        password_hash="test:hash",
        display_name="Test User",
        role=UserRole.USER,
        status=AccountStatus.ACTIVE,
    )


@pytest.fixture
def session_config() -> SessionConfig:
    """Create session config for testing."""
    return SessionConfig(
        secret_key="test-secret-key-for-testing",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7,
    )


@pytest.fixture
def token_service(session_config: SessionConfig) -> TokenService:
    """Create token service for testing."""
    return TokenService(session_config)


@pytest.fixture
def session_manager(session_config: SessionConfig) -> SessionManager:
    """Create session manager for testing."""
    return SessionManager(session_config)


class TestTokenCodec:
    """Tests for TokenCodec."""

    def test_base64url_encode_decode(self) -> None:
        """Test base64url encoding and decoding."""
        original = b"test data with special chars: +/="
        encoded = TokenCodec.base64url_encode(original)
        decoded = TokenCodec.base64url_decode(encoded)
        assert decoded == original

    def test_base64url_handles_padding(self) -> None:
        """Test that padding is handled correctly."""
        test_cases = [b"a", b"ab", b"abc", b"abcd"]
        for data in test_cases:
            encoded = TokenCodec.base64url_encode(data)
            decoded = TokenCodec.base64url_decode(encoded)
            assert decoded == data

    def test_get_hash_func(self) -> None:
        """Test getting hash functions."""
        assert TokenCodec.get_hash_func("HS256") is not None
        assert TokenCodec.get_hash_func("HS384") is not None
        assert TokenCodec.get_hash_func("HS512") is not None

    def test_get_hash_func_invalid(self) -> None:
        """Test invalid algorithm raises error."""
        with pytest.raises(ValueError):
            TokenCodec.get_hash_func("INVALID")


class TestTokenClaims:
    """Tests for TokenClaims."""

    def test_to_dict(self) -> None:
        """Test converting claims to dictionary."""
        now = datetime.now(timezone.utc)
        claims = TokenClaims(
            sub="user123",
            exp=now + timedelta(minutes=30),
            iat=now,
            jti="token123",
            email="test@example.com",
        )
        data = claims.to_dict()
        assert data["sub"] == "user123"
        assert data["email"] == "test@example.com"
        assert "exp" in data
        assert "iat" in data

    def test_from_dict(self) -> None:
        """Test creating claims from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "sub": "user123",
            "exp": int(now.timestamp()) + 1800,
            "iat": int(now.timestamp()),
            "jti": "token123",
            "token_type": "access",
            "iss": "test-issuer",
            "aud": "test-audience",
            "email": "test@example.com",
            "roles": ["user", "premium"],
        }
        claims = TokenClaims.from_dict(data)
        assert claims.sub == "user123"
        assert claims.email == "test@example.com"
        assert "user" in claims.roles

    def test_is_expired(self) -> None:
        """Test expiration check."""
        now = datetime.now(timezone.utc)
        expired_claims = TokenClaims(
            sub="user123",
            exp=now - timedelta(minutes=5),
            iat=now - timedelta(minutes=35),
            jti="token123",
        )
        valid_claims = TokenClaims(
            sub="user123",
            exp=now + timedelta(minutes=30),
            iat=now,
            jti="token123",
        )
        assert expired_claims.is_expired() is True
        assert valid_claims.is_expired() is False

    def test_is_expired_with_leeway(self) -> None:
        """Test expiration check with leeway."""
        now = datetime.now(timezone.utc)
        claims = TokenClaims(
            sub="user123",
            exp=now - timedelta(seconds=10),
            iat=now - timedelta(minutes=30),
            jti="token123",
        )
        assert claims.is_expired(leeway_seconds=0) is True
        assert claims.is_expired(leeway_seconds=30) is False


class TestTokenService:
    """Tests for TokenService."""

    def test_create_access_token(self, token_service: TokenService) -> None:
        """Test creating access token."""
        user = create_test_user()
        session_id = uuid4()
        token = token_service.create_access_token(user, session_id)
        assert token is not None
        assert len(token.split(".")) == 3

    def test_create_refresh_token(self, token_service: TokenService) -> None:
        """Test creating refresh token."""
        user = create_test_user()
        session_id = uuid4()
        token = token_service.create_refresh_token(user, session_id)
        assert token is not None
        assert len(token.split(".")) == 3

    def test_validate_token_success(self, token_service: TokenService) -> None:
        """Test validating a valid token."""
        user = create_test_user()
        session_id = uuid4()
        token = token_service.create_access_token(user, session_id)
        result = token_service.validate_token(token)
        assert result.valid is True
        assert result.claims is not None
        assert result.claims.sub == str(user.user_id)

    def test_validate_token_invalid_format(self, token_service: TokenService) -> None:
        """Test validating token with invalid format."""
        result = token_service.validate_token("invalid.token")
        assert result.valid is False
        assert result.error_code == "INVALID_FORMAT"

    def test_validate_token_invalid_signature(self, token_service: TokenService) -> None:
        """Test validating token with invalid signature."""
        user = create_test_user()
        session_id = uuid4()
        token = token_service.create_access_token(user, session_id)
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + ".invalidsignature"
        result = token_service.validate_token(tampered)
        assert result.valid is False
        assert result.error_code == "INVALID_SIGNATURE"

    def test_validate_token_revoked(self, token_service: TokenService) -> None:
        """Test validating a revoked token."""
        user = create_test_user()
        session_id = uuid4()
        token = token_service.create_access_token(user, session_id)
        result = token_service.validate_token(token)
        jti = result.claims.jti
        token_service.revoke_token(jti)
        result = token_service.validate_token(token)
        assert result.valid is False
        assert result.error_code == "TOKEN_REVOKED"

    def test_validate_token_different_algorithm(self, token_service: TokenService, session_config: SessionConfig) -> None:
        """Test validating token with wrong algorithm."""
        import json
        import base64
        header = {"alg": "HS384", "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload = {"sub": "test", "exp": 9999999999, "iat": 0, "jti": "test"}
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"{header_b64}.{payload_b64}.signature"
        result = token_service.validate_token(token)
        assert result.valid is False
        assert result.error_code == "INVALID_ALGORITHM"


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager: SessionManager) -> None:
        """Test creating a session."""
        user = create_test_user()
        result = await session_manager.create_session(user, ip_address="127.0.0.1")
        assert result.success is True
        assert result.access_token is not None
        assert result.refresh_token is not None
        assert result.session is not None

    @pytest.mark.asyncio
    async def test_create_session_enforces_max_sessions(self, session_config: SessionConfig) -> None:
        """Test that max sessions per user is enforced."""
        session_config.max_sessions_per_user = 2
        manager = SessionManager(session_config)
        user = create_test_user()
        session1 = await manager.create_session(user)
        session2 = await manager.create_session(user)
        session3 = await manager.create_session(user)
        assert session1.success is True
        assert session2.success is True
        assert session3.success is True
        sessions = await manager.get_user_sessions(user.user_id)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_validate_session(self, session_manager: SessionManager) -> None:
        """Test validating a session."""
        user = create_test_user()
        create_result = await session_manager.create_session(user)
        validation = await session_manager.validate_session(create_result.access_token)
        assert validation.valid is True
        assert validation.claims is not None

    @pytest.mark.asyncio
    async def test_refresh_session(self, session_manager: SessionManager) -> None:
        """Test refreshing a session."""
        user = create_test_user()
        create_result = await session_manager.create_session(user)
        refresh_result = await session_manager.refresh_session(create_result.refresh_token)
        assert refresh_result.success is True
        assert refresh_result.access_token is not None

    @pytest.mark.asyncio
    async def test_refresh_session_with_access_token_fails(self, session_manager: SessionManager) -> None:
        """Test that refreshing with access token fails."""
        user = create_test_user()
        create_result = await session_manager.create_session(user)
        refresh_result = await session_manager.refresh_session(create_result.access_token)
        assert refresh_result.success is False
        assert refresh_result.error_code == "INVALID_TOKEN_TYPE"

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, session_manager: SessionManager) -> None:
        """Test getting user sessions."""
        user = create_test_user()
        await session_manager.create_session(user)
        await session_manager.create_session(user)
        sessions = await session_manager.get_user_sessions(user.user_id)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_get_user_sessions_active_only(self, session_manager: SessionManager) -> None:
        """Test getting only active sessions."""
        user = create_test_user()
        result1 = await session_manager.create_session(user)
        await session_manager.create_session(user)
        await session_manager.revoke_session(user.user_id, result1.session.session_id)
        active_sessions = await session_manager.get_user_sessions(user.user_id, active_only=True)
        all_sessions = await session_manager.get_user_sessions(user.user_id, active_only=False)
        assert len(active_sessions) == 1
        assert len(all_sessions) == 2

    @pytest.mark.asyncio
    async def test_revoke_session(self, session_manager: SessionManager) -> None:
        """Test revoking a session."""
        user = create_test_user()
        create_result = await session_manager.create_session(user)
        result = await session_manager.revoke_session(user.user_id, create_result.session.session_id)
        assert result is True
        sessions = await session_manager.get_user_sessions(user.user_id)
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_revoke_session_wrong_user(self, session_manager: SessionManager) -> None:
        """Test revoking session for wrong user fails."""
        user1 = create_test_user()
        user2 = create_test_user(email="other@example.com")
        create_result = await session_manager.create_session(user1)
        result = await session_manager.revoke_session(user2.user_id, create_result.session.session_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_all_sessions(self, session_manager: SessionManager) -> None:
        """Test revoking all sessions for a user."""
        user = create_test_user()
        await session_manager.create_session(user)
        await session_manager.create_session(user)
        await session_manager.create_session(user)
        revoked = await session_manager.revoke_all_sessions(user.user_id)
        assert revoked == 3
        sessions = await session_manager.get_user_sessions(user.user_id)
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, session_config: SessionConfig) -> None:
        """Test cleaning up expired sessions."""
        session_config.session_timeout_minutes = 0
        manager = SessionManager(session_config)
        user = create_test_user()
        await manager.create_session(user)
        import asyncio
        await asyncio.sleep(0.1)
        cleaned = await manager.cleanup_expired_sessions()
        assert cleaned >= 1

    def test_get_stats(self, session_manager: SessionManager) -> None:
        """Test getting session manager stats."""
        stats = session_manager.get_stats()
        assert "active_sessions" in stats
        assert "total_sessions" in stats
        assert "sessions_created" in stats


class TestAuthResult:
    """Tests for AuthResult."""

    def test_success_result(self) -> None:
        """Test successful auth result."""
        user = create_test_user()
        result = AuthResult(
            success=True,
            user=user,
            access_token="token",
        )
        assert result.success is True
        assert result.user is not None

    def test_failure_result(self) -> None:
        """Test failure auth result."""
        result = AuthResult(
            success=False,
            error="Invalid credentials",
            error_code="INVALID_CREDENTIALS",
        )
        assert result.success is False
        assert result.error == "Invalid credentials"


class TestTokenValidationResult:
    """Tests for TokenValidationResult."""

    def test_valid_result(self) -> None:
        """Test valid token result."""
        claims = TokenClaims(
            sub="user123",
            exp=datetime.now(timezone.utc) + timedelta(minutes=30),
            iat=datetime.now(timezone.utc),
            jti="token123",
        )
        result = TokenValidationResult(valid=True, claims=claims)
        assert result.valid is True
        assert result.claims is not None

    def test_invalid_result(self) -> None:
        """Test invalid token result."""
        result = TokenValidationResult(
            valid=False,
            error="Token expired",
            error_code="TOKEN_EXPIRED",
        )
        assert result.valid is False
        assert result.error == "Token expired"
