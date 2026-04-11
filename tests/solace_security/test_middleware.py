"""
Unit tests for Solace-AI Security Middleware.
Tests JWT manager initialization and authentication dependency wiring.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from solace_security.auth import (
    AuthenticationResult,
    InMemoryTokenBlacklist,
    JWTManager,
    RedisTokenBlacklist,
    TokenPayload,
    TokenType,
)
from solace_security.middleware import (
    AuthenticatedService,
    AuthenticatedUser,
    _extract_token,
    _get_jwt_manager,
    _handle_auth_failure,
)


class TestJWTManagerInitialization:
    """Tests for the cached _get_jwt_manager singleton factory."""

    def setup_method(self) -> None:
        """Clear lru_cache before each test for isolation."""
        _get_jwt_manager.cache_clear()
        # Ensure AUTH_SECRET_KEY is set for tests
        os.environ.setdefault(
            "AUTH_SECRET_KEY",
            "test-middleware-secret-key-32-bytes-long!!",
        )

    def teardown_method(self) -> None:
        """Clean up cache after each test."""
        _get_jwt_manager.cache_clear()

    def test_get_jwt_manager_returns_valid_instance(self) -> None:
        """Verify _get_jwt_manager returns a JWTManager instance."""
        manager = _get_jwt_manager()
        assert isinstance(manager, JWTManager)

    def test_jwt_manager_has_token_blacklist(self) -> None:
        """Verify JWTManager was initialized with InMemoryTokenBlacklist."""
        manager = _get_jwt_manager()
        assert hasattr(manager, "_blacklist")
        assert isinstance(manager._blacklist, InMemoryTokenBlacklist)

    def test_jwt_manager_is_cached_singleton(self) -> None:
        """Verify repeated calls return the same cached instance."""
        manager1 = _get_jwt_manager()
        manager2 = _get_jwt_manager()
        assert manager1 is manager2

    def test_cache_clear_creates_new_instance(self) -> None:
        """Verify cache_clear forces a new instance."""
        manager1 = _get_jwt_manager()
        _get_jwt_manager.cache_clear()
        manager2 = _get_jwt_manager()
        assert manager1 is not manager2


class TestJWTManagerRedisBlacklist:
    """Tests for C-01: middleware must use Redis-backed blacklist when available.

    In multi-worker deployments, InMemoryTokenBlacklist is per-process. Token
    revocation in worker A is invisible to worker B, violating HIPAA audit
    requirements for session revocation. When REDIS_URL is set, _get_jwt_manager
    must build a RedisTokenBlacklist instead.
    """

    def setup_method(self) -> None:
        """Clear cache and ensure AUTH_SECRET_KEY set."""
        _get_jwt_manager.cache_clear()
        os.environ.setdefault(
            "AUTH_SECRET_KEY",
            "test-middleware-secret-key-32-bytes-long!!",
        )

    def teardown_method(self) -> None:
        """Clean up cache and env vars after each test."""
        _get_jwt_manager.cache_clear()
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("AUTH_BLACKLIST_BACKEND", None)

    def test_uses_redis_blacklist_when_redis_url_set(self) -> None:
        """Setting REDIS_URL should produce a RedisTokenBlacklist."""
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"
        mock_client = MagicMock()
        with patch(
            "redis.asyncio.from_url", return_value=mock_client
        ) as from_url_mock:
            manager = _get_jwt_manager()
            from_url_mock.assert_called_once()
            assert isinstance(manager._blacklist, RedisTokenBlacklist)

    def test_falls_back_to_inmemory_when_redis_url_absent(self) -> None:
        """No REDIS_URL should keep the InMemoryTokenBlacklist fallback."""
        os.environ.pop("REDIS_URL", None)
        manager = _get_jwt_manager()
        assert isinstance(manager._blacklist, InMemoryTokenBlacklist)

    def test_falls_back_to_inmemory_when_redis_import_fails(self) -> None:
        """If redis.asyncio.from_url raises, we fall back to InMemoryTokenBlacklist."""
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"
        with patch(
            "redis.asyncio.from_url", side_effect=RuntimeError("no redis")
        ):
            manager = _get_jwt_manager()
            assert isinstance(manager._blacklist, InMemoryTokenBlacklist)

    def test_explicit_inmemory_backend_override(self) -> None:
        """AUTH_BLACKLIST_BACKEND=memory should force InMemoryTokenBlacklist even with REDIS_URL set."""
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"
        os.environ["AUTH_BLACKLIST_BACKEND"] = "memory"
        manager = _get_jwt_manager()
        assert isinstance(manager._blacklist, InMemoryTokenBlacklist)


class TestExtractToken:
    """Tests for the _extract_token helper."""

    def test_extract_valid_bearer_token(self) -> None:
        """Verify bearer token is extracted from credentials."""
        from unittest.mock import MagicMock

        creds = MagicMock()
        creds.scheme = "Bearer"
        creds.credentials = "test-token-value"
        token = _extract_token(creds)
        assert token == "test-token-value"

    def test_extract_token_raises_on_none(self) -> None:
        """Verify HTTPException raised when credentials are None."""
        with pytest.raises(HTTPException) as exc_info:
            _extract_token(None)
        assert exc_info.value.status_code == 401
        assert "Missing authentication" in exc_info.value.detail

    def test_extract_token_raises_on_wrong_scheme(self) -> None:
        """Verify HTTPException raised for non-Bearer scheme."""
        from unittest.mock import MagicMock

        creds = MagicMock()
        creds.scheme = "Basic"
        with pytest.raises(HTTPException) as exc_info:
            _extract_token(creds)
        assert exc_info.value.status_code == 401
        assert "Invalid authentication scheme" in exc_info.value.detail


class TestHandleAuthFailure:
    """Tests for _handle_auth_failure error mapping."""

    def test_expired_token_maps_to_401(self) -> None:
        """Verify TOKEN_EXPIRED maps to 401."""
        result = AuthenticationResult.fail("TOKEN_EXPIRED", "Token has expired")
        with pytest.raises(HTTPException) as exc_info:
            _handle_auth_failure(result)
        assert exc_info.value.status_code == 401

    def test_invalid_token_maps_to_401(self) -> None:
        """Verify INVALID_TOKEN maps to 401."""
        result = AuthenticationResult.fail("INVALID_TOKEN", "Invalid token")
        with pytest.raises(HTTPException) as exc_info:
            _handle_auth_failure(result)
        assert exc_info.value.status_code == 401

    def test_unknown_error_code_maps_to_401(self) -> None:
        """Verify unknown error codes still raise 401."""
        result = AuthenticationResult.fail("UNKNOWN_CODE", "Something went wrong")
        with pytest.raises(HTTPException) as exc_info:
            _handle_auth_failure(result)
        assert exc_info.value.status_code == 401

    def test_error_message_used_in_detail(self) -> None:
        """Verify custom error message is used in HTTPException detail."""
        result = AuthenticationResult.fail("DECODE_ERROR", "Custom decode error msg")
        with pytest.raises(HTTPException) as exc_info:
            _handle_auth_failure(result)
        assert exc_info.value.detail == "Custom decode error msg"


class TestAuthenticatedUser:
    """Tests for AuthenticatedUser model and permission resolution."""

    def test_from_payload(self) -> None:
        """Verify AuthenticatedUser is created correctly from TokenPayload."""
        import time

        payload = TokenPayload(
            sub="user-123",
            type=TokenType.ACCESS,
            exp=int(time.time()) + 3600,
            roles=["user"],
            permissions=["read:data"],
            session_id="session-abc",
            metadata={"key": "value"},
        )
        user = AuthenticatedUser.from_payload(payload)
        assert user.user_id == "user-123"
        assert user.token_type == TokenType.ACCESS
        assert "user" in user.roles
        assert "read:data" in user.permissions
        assert user.session_id == "session-abc"

    def test_has_role(self) -> None:
        """Verify has_role checks role membership."""
        user = AuthenticatedUser(
            user_id="u1",
            token_type=TokenType.ACCESS,
            roles=["admin", "user"],
        )
        assert user.has_role("admin") is True
        assert user.has_role("superadmin") is False

    def test_has_any_role(self) -> None:
        """Verify has_any_role checks any role membership."""
        user = AuthenticatedUser(
            user_id="u1",
            token_type=TokenType.ACCESS,
            roles=["user"],
        )
        assert user.has_any_role(["admin", "user"]) is True
        assert user.has_any_role(["admin", "superadmin"]) is False

    def test_is_service_token(self) -> None:
        """Verify is_service_token property."""
        service_user = AuthenticatedUser(
            user_id="svc-1",
            token_type=TokenType.SERVICE,
        )
        assert service_user.is_service_token is True

        regular_user = AuthenticatedUser(
            user_id="u1",
            token_type=TokenType.ACCESS,
        )
        assert regular_user.is_service_token is False


class TestAuthenticatedService:
    """Tests for AuthenticatedService model."""

    def test_from_payload_strips_service_prefix(self) -> None:
        """Verify service: prefix is stripped from subject."""
        import time

        payload = TokenPayload(
            sub="service:safety-service",
            type=TokenType.SERVICE,
            exp=int(time.time()) + 3600,
            permissions=["internal:process"],
            metadata={"service_name": "safety-service"},
        )
        service = AuthenticatedService.from_payload(payload)
        assert service.service_name == "safety-service"
        assert "internal:process" in service.permissions

    def test_from_payload_uses_metadata_service_name(self) -> None:
        """Verify metadata service_name is preferred."""
        import time

        payload = TokenPayload(
            sub="service:raw-name",
            type=TokenType.SERVICE,
            exp=int(time.time()) + 3600,
            metadata={"service_name": "friendly-name"},
        )
        service = AuthenticatedService.from_payload(payload)
        assert service.service_name == "friendly-name"
