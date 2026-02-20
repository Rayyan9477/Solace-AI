"""
Tests for User Service API endpoints.

Tests authentication, user management, preferences, and consent endpoints.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.main import ServiceState
from src.api import router as api_router
from src.infrastructure.jwt_service import (
    JWTService, JWTConfig, TokenExpiredError, TokenInvalidError, InMemoryTokenBlacklist,
)
from src.infrastructure.password_service import PasswordService, PasswordConfig
from src.infrastructure.token_service import create_token_service
from src.infrastructure.encryption_service import create_encryption_service
from src.domain.service import UserService
from src.auth import SessionManager, SessionConfig, AuthenticationService

from .fixtures import (
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
    InMemoryConsentRepository,
)


def _build_test_app() -> FastAPI:
    """Build a test FastAPI app with in-memory repositories and services."""

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        state = ServiceState()

        # Infrastructure
        jwt_config = JWTConfig(
            secret_key="test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars",
        )
        blacklist = InMemoryTokenBlacklist()
        state.jwt_service = JWTService(jwt_config, blacklist=blacklist)
        state.password_service = PasswordService(PasswordConfig())

        from cryptography.fernet import Fernet
        _key = Fernet.generate_key()
        state.token_service = create_token_service(encryption_key=_key)
        state.encryption_service = create_encryption_service(encryption_key=_key)

        # Repositories (in-memory)
        state.user_repository = InMemoryUserRepository()
        state.preferences_repository = InMemoryUserPreferencesRepository()
        state.consent_repository = InMemoryConsentRepository()

        # Domain
        state.user_service = UserService(
            user_repository=state.user_repository,
            preferences_repository=state.preferences_repository,
            consent_repository=state.consent_repository,
            password_service=state.password_service,
        )

        # Application
        session_config = SessionConfig(
            secret_key="test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars",
        )
        state.session_manager = SessionManager(session_config)
        state.auth_service = AuthenticationService(
            session_manager=state.session_manager,
            jwt_service=state.jwt_service,
            password_service=state.password_service,
        )

        state.initialized = True
        app.state.service = state
        yield
        state.initialized = False

    app = FastAPI(lifespan=test_lifespan)

    # Health endpoints (replicate from main.py)
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "user-service"}

    @app.get("/ready")
    async def readiness_check():
        svc_state: ServiceState = app.state.service
        return {"status": "ready", "service": "user-service", "initialized": svc_state.initialized}

    @app.get("/status")
    async def service_status():
        svc_state: ServiceState = app.state.service
        return {
            "status": "operational" if svc_state.initialized else "initializing",
            "service": "user-service",
            "version": "1.0.0",
            "statistics": svc_state.stats,
        }

    app.include_router(api_router, prefix="/api/v1")
    return app


@pytest.fixture
def client():
    """Create test client with in-memory services."""
    app = _build_test_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "user-service"

    def test_readiness_check(self, client: TestClient) -> None:
        """Test readiness endpoint returns ready status."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["initialized"] is True

    def test_status_endpoint(self, client: TestClient) -> None:
        """Test status endpoint returns service information."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert data["service"] == "user-service"
        assert "statistics" in data


class TestRegistration:
    """Tests for user registration."""

    def test_register_success(self, client: TestClient) -> None:
        """Test successful user registration."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["display_name"] == "Test User"
        assert data["status"] == "pending_verification"
        assert "user_id" in data

    def test_register_duplicate_email(self, client: TestClient) -> None:
        """Test registration fails with duplicate email."""
        # First registration
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )

        # Duplicate registration
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "AnotherPass456!",
                "display_name": "Another User",
            },
        )
        assert response.status_code == 409
        assert "already registered" in response.json()["detail"]

    def test_register_invalid_email(self, client: TestClient) -> None:
        """Test registration fails with invalid email."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "invalid-email",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        assert response.status_code == 422

    def test_register_weak_password(self, client: TestClient) -> None:
        """Test registration fails with weak password."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "weak",
                "display_name": "Test User",
            },
        )
        assert response.status_code == 422


class TestLogin:
    """Tests for user login."""

    def test_login_success(self, client: TestClient) -> None:
        """Test successful login."""
        # Register user
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )

        # Login
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "Bearer"
        assert data["expires_in"] > 0

    def test_login_invalid_password(self, client: TestClient) -> None:
        """Test login fails with invalid password."""
        # Register user
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )

        # Login with wrong password
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "WrongPass456!",
            },
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client: TestClient) -> None:
        """Test login fails for nonexistent user."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "SomePass123!",
            },
        )
        assert response.status_code == 401


class TestTokenRefresh:
    """Tests for token refresh."""

    def test_refresh_token_success(self, client: TestClient) -> None:
        """Test successful token refresh."""
        # Register and login
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_invalid_token(self, client: TestClient) -> None:
        """Test refresh fails with invalid token."""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid-token"},
        )
        assert response.status_code == 401


class TestUserProfile:
    """Tests for user profile management."""

    def _get_auth_header(self, client: TestClient) -> dict[str, str]:
        """Register, login, and return auth header."""
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_get_current_user(self, client: TestClient) -> None:
        """Test getting current user profile."""
        headers = self._get_auth_header(client)
        response = client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["display_name"] == "Test User"

    def test_update_profile(self, client: TestClient) -> None:
        """Test updating user profile."""
        headers = self._get_auth_header(client)
        response = client.put(
            "/api/v1/users/me",
            headers=headers,
            json={
                "display_name": "Updated Name",
                "bio": "My bio",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Updated Name"
        assert data["bio"] == "My bio"

    def test_get_profile_unauthorized(self, client: TestClient) -> None:
        """Test getting profile without auth fails."""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401


class TestPasswordChange:
    """Tests for password change."""

    def _get_auth_header(self, client: TestClient) -> dict[str, str]:
        """Register, login, and return auth header."""
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_change_password_success(self, client: TestClient) -> None:
        """Test successful password change."""
        headers = self._get_auth_header(client)
        response = client.post(
            "/api/v1/users/me/password",
            headers=headers,
            json={
                "current_password": "SecurePass123!",
                "new_password": "NewSecurePass456!",
            },
        )
        assert response.status_code == 204

        # Verify new password works
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "NewSecurePass456!",
            },
        )
        assert response.status_code == 200

    def test_change_password_wrong_current(self, client: TestClient) -> None:
        """Test password change fails with wrong current password."""
        headers = self._get_auth_header(client)
        response = client.post(
            "/api/v1/users/me/password",
            headers=headers,
            json={
                "current_password": "WrongPassword123!",
                "new_password": "NewSecurePass456!",
            },
        )
        assert response.status_code == 400


class TestPreferences:
    """Tests for user preferences."""

    def _get_auth_header(self, client: TestClient) -> dict[str, str]:
        """Register, login, and return auth header."""
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_get_preferences(self, client: TestClient) -> None:
        """Test getting user preferences."""
        headers = self._get_auth_header(client)
        response = client.get("/api/v1/users/me/preferences", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "notification_email" in data
        assert "theme" in data

    def test_update_preferences(self, client: TestClient) -> None:
        """Test updating user preferences."""
        headers = self._get_auth_header(client)
        response = client.put(
            "/api/v1/users/me/preferences",
            headers=headers,
            json={
                "notification_email": False,
                "theme": "dark",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["notification_email"] is False
        assert data["theme"] == "dark"


class TestConsent:
    """Tests for consent management."""

    def _get_auth_header(self, client: TestClient) -> dict[str, str]:
        """Register, login, and return auth header."""
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_record_consent(self, client: TestClient) -> None:
        """Test recording user consent."""
        headers = self._get_auth_header(client)
        response = client.post(
            "/api/v1/users/me/consent",
            headers=headers,
            json={
                "consent_type": "terms_of_service",
                "granted": True,
                "version": "1.0",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["consent_type"] == "terms_of_service"
        assert data["granted"] is True

    def test_get_consent_records(self, client: TestClient) -> None:
        """Test getting consent records."""
        headers = self._get_auth_header(client)

        # Record consent
        client.post(
            "/api/v1/users/me/consent",
            headers=headers,
            json={
                "consent_type": "terms_of_service",
                "granted": True,
                "version": "1.0",
            },
        )

        # Get records
        response = client.get("/api/v1/users/me/consent", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["consent_type"] == "terms_of_service"


class TestLogout:
    """Tests for logout."""

    def test_logout_success(self, client: TestClient) -> None:
        """Test successful logout."""
        # Register and login
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Logout
        response = client.post("/api/v1/auth/logout", headers=headers)
        assert response.status_code == 204
