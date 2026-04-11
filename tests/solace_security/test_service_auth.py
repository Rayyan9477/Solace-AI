"""
Unit tests for solace_security.service_auth.

Specifically exercises NEW-05: the FastAPI dependency returned by
`get_service_auth_dependency` must actually extract the Authorization
header. Before the fix, the inner `_verify_service` coroutine took
`authorization: str | None = None` (no ``Header(None)``), so FastAPI
would not inject the Authorization header and every protected endpoint
would return "Missing Authorization header" even with a valid token.
"""
from __future__ import annotations

import os

import pytest
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from solace_security.auth import (
    AuthSettings,
)
from solace_security.service_auth import (
    ServiceAuthResult,
    ServiceAuthSettings,
    ServiceIdentity,
    ServiceTokenManager,
    get_service_auth_dependency,
)


@pytest.fixture
def test_secret() -> str:
    """Provide a stable 32+ byte secret for the test."""
    return "service-auth-test-secret-key-32+!" + "padding-bytes"


@pytest.fixture
def token_manager(test_secret: str) -> ServiceTokenManager:
    """Build a ServiceTokenManager wired to an in-memory blacklist."""
    os.environ["AUTH_SECRET_KEY"] = test_secret
    auth_settings = AuthSettings()
    service_settings = ServiceAuthSettings(
        service_name="orchestrator-service",
        allowed_services=[svc.value for svc in ServiceIdentity],
    )
    return ServiceTokenManager(
        auth_settings=auth_settings, service_settings=service_settings
    )


@pytest.fixture
def valid_service_token(token_manager: ServiceTokenManager) -> str:
    """Issue a valid service token for the orchestrator service."""
    creds = token_manager.create_service_token(
        service_name="orchestrator-service",
    )
    return creds.token


def _make_app(token_manager: ServiceTokenManager) -> FastAPI:
    """Build a tiny FastAPI app that protects one endpoint with
    ``get_service_auth_dependency`` — exactly as its docstring advertises.
    """
    app = FastAPI()
    verify_service = get_service_auth_dependency(
        token_manager, required_permissions=None
    )

    @app.get("/internal/echo")
    async def echo(
        service_auth: ServiceAuthResult = Depends(verify_service),
    ) -> dict:
        if not service_auth.authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=service_auth.error or "unauthenticated",
            )
        return {"service": service_auth.service_name or "unknown"}

    return app


class TestGetServiceAuthDependency:
    """NEW-05: Ensure the dependency actually extracts the Authorization header."""

    def test_missing_header_returns_401(
        self, token_manager: ServiceTokenManager
    ) -> None:
        app = _make_app(token_manager)
        client = TestClient(app)
        r = client.get("/internal/echo")
        assert r.status_code == 401

    def test_valid_bearer_token_returns_200(
        self,
        token_manager: ServiceTokenManager,
        valid_service_token: str,
    ) -> None:
        """The core NEW-05 failing test: header must be injected by FastAPI.

        Before the fix, the inner `_verify_service` declared
        ``authorization: str | None = None`` without ``Header(None)``,
        so FastAPI treated it as a body parameter and never populated it
        from the request headers. Every valid-token request returned 401
        with "Missing Authorization header".
        """
        app = _make_app(token_manager)
        client = TestClient(app)
        r = client.get(
            "/internal/echo",
            headers={"Authorization": f"Bearer {valid_service_token}"},
        )
        assert r.status_code == 200, (
            f"Expected 200 with valid token, got {r.status_code}: {r.text}"
        )
        body = r.json()
        assert body.get("service") == "orchestrator-service"

    def test_invalid_format_returns_401(
        self, token_manager: ServiceTokenManager
    ) -> None:
        app = _make_app(token_manager)
        client = TestClient(app)
        r = client.get(
            "/internal/echo",
            headers={"Authorization": "NotBearer abc"},
        )
        assert r.status_code == 401
        assert "format" in r.json().get("detail", "").lower() or r.status_code == 401
