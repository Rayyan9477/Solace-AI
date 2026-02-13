"""Unit tests for authentication module."""

from __future__ import annotations
import time
from datetime import datetime, timezone
import pytest
from solace_security.auth import (
    TokenType,
    AuthSettings,
    TokenPayload,
    TokenPair,
    AuthenticationResult,
    PasswordHasher,
    JWTManager,
    InMemoryTokenBlacklist,
    APIKeyGenerator,
    SessionManager,
    create_jwt_manager,
    create_session_manager,
)


class TestAuthSettings:
    """Tests for AuthSettings configuration."""

    def test_default_settings(self):
        # Use for_development() since secret_key is now required
        settings = AuthSettings.for_development()
        assert settings.algorithm == "HS256"
        assert settings.access_token_expire_minutes == 30
        assert settings.refresh_token_expire_days == 7
        assert settings.issuer == "solace-ai"

    def test_custom_settings(self):
        settings = AuthSettings(
            secret_key="test-secret-key-32-bytes-long!!!",  # Must be 32+ bytes
            algorithm="HS512",
            access_token_expire_minutes=60,
            issuer="custom-issuer",
        )
        assert settings.algorithm == "HS512"
        assert settings.access_token_expire_minutes == 60
        assert settings.issuer == "custom-issuer"

    def test_secret_key_required(self):
        """Test that secret_key is required and validates length."""
        import pytest

        with pytest.raises(Exception):  # ValidationError for missing required field
            AuthSettings()

    def test_secret_key_minimum_length(self):
        """Test that secret_key must be at least 32 bytes."""
        import pytest

        with pytest.raises(Exception, match="at least 32"):
            AuthSettings(secret_key="short-key")


class TestTokenType:
    """Tests for TokenType enum."""

    def test_token_types(self):
        assert TokenType.ACCESS.value == "access"
        assert TokenType.REFRESH.value == "refresh"
        assert TokenType.API_KEY.value == "api_key"
        assert TokenType.SERVICE.value == "service"


class TestTokenPayload:
    """Tests for TokenPayload model."""

    def test_create_payload(self):
        payload = TokenPayload(
            sub="user123",
            type=TokenType.ACCESS,
            exp=int(time.time()) + 3600,
            roles=["user"],
        )
        assert payload.sub == "user123"
        assert payload.type == TokenType.ACCESS
        assert not payload.is_expired

    def test_payload_expiration(self):
        past_exp = int(time.time()) - 100
        payload = TokenPayload(sub="user123", type=TokenType.ACCESS, exp=past_exp)
        assert payload.is_expired

    def test_payload_timestamps(self):
        now = int(time.time())
        exp = now + 3600
        payload = TokenPayload(sub="user123", type=TokenType.ACCESS, iat=now, exp=exp)
        assert isinstance(payload.issued_at, datetime)
        assert isinstance(payload.expires_at, datetime)


class TestAuthenticationResult:
    """Tests for AuthenticationResult model."""

    def test_success_result(self):
        payload = TokenPayload(
            sub="user123", type=TokenType.ACCESS, exp=int(time.time()) + 3600
        )
        result = AuthenticationResult.ok("user123", payload)
        assert result.success
        assert result.user_id == "user123"
        assert result.payload is not None

    def test_failure_result(self):
        result = AuthenticationResult.fail("TOKEN_EXPIRED", "Token has expired")
        assert not result.success
        assert result.error_code == "TOKEN_EXPIRED"
        assert result.error_message == "Token has expired"


class TestPasswordHasher:
    """Tests for PasswordHasher."""

    def test_hash_password(self):
        password = "SecurePassword123!"
        hashed = PasswordHasher.hash_password(password)
        assert hashed.startswith("$sha256$")
        assert len(hashed) > 50

    def test_verify_password_correct(self):
        password = "SecurePassword123!"
        hashed = PasswordHasher.hash_password(password)
        assert PasswordHasher.verify_password(password, hashed)

    def test_verify_password_incorrect(self):
        password = "SecurePassword123!"
        hashed = PasswordHasher.hash_password(password)
        assert not PasswordHasher.verify_password("WrongPassword", hashed)

    def test_verify_password_invalid_hash(self):
        assert not PasswordHasher.verify_password("password", "invalid_hash")

    def test_needs_rehash_valid(self):
        password = "SecurePassword123!"
        hashed = PasswordHasher.hash_password(password)
        assert not PasswordHasher.needs_rehash(hashed)

    def test_needs_rehash_invalid(self):
        assert PasswordHasher.needs_rehash("invalid_hash")
        assert PasswordHasher.needs_rehash("$md5$1000$salt$hash")


class TestJWTManager:
    """Tests for JWTManager."""

    @pytest.fixture
    def jwt_manager(self):
        return JWTManager(
            AuthSettings(secret_key="test-secret-key-32-bytes-long!!!"),
            token_blacklist=InMemoryTokenBlacklist(),
        )

    def test_create_access_token(self, jwt_manager):
        token = jwt_manager.create_access_token(
            "user123", roles=["user"], permissions=["read"]
        )
        assert token
        assert isinstance(token, str)

    def test_create_refresh_token(self, jwt_manager):
        token = jwt_manager.create_refresh_token("user123", session_id="session123")
        assert token
        assert isinstance(token, str)

    def test_create_token_pair(self, jwt_manager):
        pair = jwt_manager.create_token_pair("user123", roles=["user"])
        assert isinstance(pair, TokenPair)
        assert pair.access_token
        assert pair.refresh_token
        assert pair.token_type == "Bearer"

    def test_create_api_key(self, jwt_manager):
        token = jwt_manager.create_api_key(
            "user123", "my-api-key", permissions=["read"]
        )
        assert token
        result = jwt_manager.decode_token(token, TokenType.API_KEY)
        assert result.success
        assert result.payload.metadata["api_key_name"] == "my-api-key"

    def test_create_service_token(self, jwt_manager):
        token = jwt_manager.create_service_token("auth-service", permissions=["admin"])
        assert token
        result = jwt_manager.decode_token(token, TokenType.SERVICE)
        assert result.success
        assert result.payload.sub == "service:auth-service"

    def test_validate_access_token(self, jwt_manager):
        token = jwt_manager.create_access_token("user123", roles=["admin"])
        result = jwt_manager.validate_access_token(token)
        assert result.success
        assert result.user_id == "user123"
        assert "admin" in result.payload.roles

    def test_validate_refresh_token(self, jwt_manager):
        token = jwt_manager.create_refresh_token("user123")
        result = jwt_manager.validate_refresh_token(token)
        assert result.success
        assert result.payload.type == TokenType.REFRESH

    def test_wrong_token_type(self, jwt_manager):
        access_token = jwt_manager.create_access_token("user123")
        result = jwt_manager.decode_token(access_token, TokenType.REFRESH)
        assert not result.success
        assert result.error_code == "INVALID_TOKEN_TYPE"

    def test_invalid_token(self, jwt_manager):
        result = jwt_manager.decode_token("invalid.token.here")
        assert not result.success
        assert result.error_code == "DECODE_ERROR"

    def test_refresh_access_token(self, jwt_manager):
        refresh = jwt_manager.create_refresh_token("user123")
        new_access, result = jwt_manager.refresh_access_token(refresh, roles=["user"])
        assert result.success
        assert new_access
        access_result = jwt_manager.validate_access_token(new_access)
        assert access_result.success


class TestAPIKeyGenerator:
    """Tests for APIKeyGenerator."""

    def test_generate_api_key(self):
        raw_key, key_hash = APIKeyGenerator.generate()
        assert raw_key.startswith("sk_")
        assert len(key_hash) == 64

    def test_hash_key(self):
        raw_key, _ = APIKeyGenerator.generate()
        hashed = APIKeyGenerator.hash_key(raw_key)
        assert len(hashed) == 64

    def test_validate_format_valid(self):
        raw_key, _ = APIKeyGenerator.generate()
        assert APIKeyGenerator.validate_format(raw_key)

    def test_validate_format_invalid(self):
        assert not APIKeyGenerator.validate_format("invalid_key")
        assert not APIKeyGenerator.validate_format("sk_tooshort")


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def session_manager(self):
        return SessionManager()

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        session_id = await session_manager.create_session("user123", {"device": "web"})
        assert session_id
        assert len(session_id) == 36

    @pytest.mark.asyncio
    async def test_validate_session(self, session_manager):
        session_id = await session_manager.create_session("user123")
        assert await session_manager.validate_session(session_id)
        assert not await session_manager.validate_session("invalid-session")

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager):
        session_id = await session_manager.create_session("user123", {"device": "mobile"})
        session = await session_manager.get_session(session_id)
        assert session["user_id"] == "user123"
        assert session["metadata"]["device"] == "mobile"

    @pytest.mark.asyncio
    async def test_update_activity(self, session_manager):
        session_id = await session_manager.create_session("user123")
        initial_session = await session_manager.get_session(session_id)
        initial_activity = initial_session["last_activity"]
        await session_manager.update_activity(session_id)
        updated_session = await session_manager.get_session(session_id)
        assert updated_session["last_activity"] >= initial_activity

    @pytest.mark.asyncio
    async def test_invalidate_session(self, session_manager):
        session_id = await session_manager.create_session("user123")
        assert await session_manager.validate_session(session_id)
        assert await session_manager.invalidate_session(session_id)
        assert not await session_manager.validate_session(session_id)

    @pytest.mark.asyncio
    async def test_invalidate_user_sessions(self, session_manager):
        await session_manager.create_session("user123")
        await session_manager.create_session("user123")
        await session_manager.create_session("user456")
        count = await session_manager.invalidate_user_sessions("user123")
        assert count == 2
        assert len(await session_manager.get_user_sessions("user123")) == 0
        assert len(await session_manager.get_user_sessions("user456")) == 1

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, session_manager):
        await session_manager.create_session("user123")
        await session_manager.create_session("user123")
        sessions = await session_manager.get_user_sessions("user123")
        assert len(sessions) == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_jwt_manager(self):
        # Must provide settings with valid secret_key
        settings = AuthSettings.for_development()
        manager = create_jwt_manager(settings)
        assert isinstance(manager, JWTManager)

    def test_create_session_manager(self):
        manager = create_session_manager()
        assert isinstance(manager, SessionManager)
