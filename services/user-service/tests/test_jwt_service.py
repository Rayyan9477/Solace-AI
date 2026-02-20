"""
Unit tests for JWT Service.

Tests cover token generation, verification, refresh, and error handling.
"""
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.infrastructure.jwt_service import (
    JWTService,
    JWTConfig,
    TokenType,
    TokenPayload,
    TokenPair,
    TokenExpiredError,
    TokenInvalidError,
    create_jwt_service,
)


@pytest.fixture
def jwt_config():
    """Create JWT configuration for testing."""
    return JWTConfig(
        secret_key="test-secret-key-for-jwt-testing-only",
        algorithm="HS256",
        access_token_expire_minutes=15,
        refresh_token_expire_days=30,
    )


@pytest.fixture
def jwt_service(jwt_config):
    """Create JWT service instance."""
    return JWTService(jwt_config)


class TestJWTService:
    """Test cases for JWTService."""

    def test_generate_token_pair(self, jwt_service):
        """Test generating access and refresh token pair."""
        user_id = uuid4()
        email = "test@example.com"
        role = "user"

        token_pair = jwt_service.generate_token_pair(user_id, email, role)

        assert isinstance(token_pair, TokenPair)
        assert token_pair.access_token
        assert token_pair.refresh_token
        assert token_pair.token_type == "Bearer"
        assert token_pair.expires_in == 900  # 15 minutes

    @pytest.mark.asyncio
    async def test_verify_access_token_success(self, jwt_service):
        """Test successfully verifying access token."""
        user_id = uuid4()
        email = "test@example.com"
        role = "user"

        token_pair = jwt_service.generate_token_pair(user_id, email, role)

        payload = await jwt_service.verify_token(token_pair.access_token, TokenType.ACCESS)

        assert payload.user_id == user_id
        assert payload.email == email
        assert payload.role == role
        assert payload.token_type == TokenType.ACCESS

    @pytest.mark.asyncio
    async def test_verify_refresh_token_success(self, jwt_service):
        """Test successfully verifying refresh token."""
        user_id = uuid4()
        email = "test@example.com"
        role = "user"

        token_pair = jwt_service.generate_token_pair(user_id, email, role)

        payload = await jwt_service.verify_token(token_pair.refresh_token, TokenType.REFRESH)

        assert payload.user_id == user_id
        assert payload.email == email
        assert payload.role == role
        assert payload.token_type == TokenType.REFRESH

    @pytest.mark.asyncio
    async def test_verify_token_wrong_type_fails(self, jwt_service):
        """Test that using access token as refresh token fails."""
        user_id = uuid4()
        email = "test@example.com"
        role = "user"

        token_pair = jwt_service.generate_token_pair(user_id, email, role)

        with pytest.raises(TokenInvalidError, match="Expected refresh token"):
            await jwt_service.verify_token(token_pair.access_token, TokenType.REFRESH)

    @pytest.mark.asyncio
    async def test_verify_expired_token_fails(self, jwt_service):
        """Test that expired token verification fails."""
        # Create config with very short expiry
        short_config = JWTConfig(
            secret_key="test-secret-key",
            access_token_expire_minutes=-1,  # Negative = already expired
        )
        short_service = JWTService(short_config)

        user_id = uuid4()
        token_pair = short_service.generate_token_pair(user_id, "test@example.com", "user")

        with pytest.raises(TokenExpiredError, match="Token has expired"):
            await short_service.verify_token(token_pair.access_token)

    @pytest.mark.asyncio
    async def test_verify_invalid_token_fails(self, jwt_service):
        """Test that invalid token fails verification."""
        with pytest.raises(TokenInvalidError, match="Invalid token"):
            await jwt_service.verify_token("invalid.token.here")

    @pytest.mark.asyncio
    async def test_verify_tampered_token_fails(self, jwt_service):
        """Test that tampered token fails verification."""
        user_id = uuid4()
        token_pair = jwt_service.generate_token_pair(user_id, "test@example.com", "user")

        # Tamper with token
        tampered = token_pair.access_token + "tampered"

        with pytest.raises(TokenInvalidError):
            await jwt_service.verify_token(tampered)

    @pytest.mark.asyncio
    async def test_refresh_access_token(self, jwt_service):
        """Test refreshing access token from refresh token."""
        user_id = uuid4()
        email = "test@example.com"
        role = "user"

        token_pair = jwt_service.generate_token_pair(user_id, email, role)

        new_access_token = await jwt_service.refresh_access_token(token_pair.refresh_token)

        assert new_access_token
        # Note: Tokens may be identical if generated in same second (JWT uses second precision)
        # What matters is that the refreshed token is valid

        # Verify new token is valid and contains correct payload
        payload = await jwt_service.verify_token(new_access_token, TokenType.ACCESS)
        assert payload.user_id == user_id
        assert payload.email == email

    def test_extract_token_from_header_success(self):
        """Test extracting token from authorization header."""
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        header = f"Bearer {token}"

        extracted = JWTService.extract_token_from_header(header)

        assert extracted == token

    def test_extract_token_from_header_missing_fails(self):
        """Test that missing authorization header fails."""
        with pytest.raises(TokenInvalidError, match="Missing authorization header"):
            JWTService.extract_token_from_header("")

    def test_extract_token_from_header_invalid_format_fails(self):
        """Test that invalid header format fails."""
        with pytest.raises(TokenInvalidError, match="Invalid authorization header format"):
            JWTService.extract_token_from_header("InvalidFormat token")

    def test_decode_token_unsafe(self, jwt_service):
        """Test unsafe token decoding (for debugging)."""
        user_id = uuid4()
        token_pair = jwt_service.generate_token_pair(user_id, "test@example.com", "user")

        payload = jwt_service.decode_token_unsafe(token_pair.access_token)

        assert payload is not None
        assert payload["user_id"] == str(user_id)
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"

    def test_create_jwt_service_factory(self):
        """Test factory function creates service."""
        service = create_jwt_service(
            secret_key="test-secret",
            algorithm="HS256",
            access_token_expire_minutes=30,
        )

        assert isinstance(service, JWTService)
        assert service.config.secret_key == "test-secret"
        assert service.config.access_token_expire_minutes == 30
