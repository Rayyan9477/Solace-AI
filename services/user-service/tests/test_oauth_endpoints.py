"""REV-22: Google OAuth /start + /callback endpoint flow tests (config-gated).

These exercise the endpoints layered on top of the already-landed C.3b id_token
signature verifier (src/infrastructure/oauth_google.py). The Google token
exchange is MOCKED (injected fake httpx client) and id_token verification is
STUBBED (injected verifier) so no network call is made.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse, parse_qs

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.main import ServiceState
from src.api import router as api_router
from src.api_auth import auth_router
from src.oauth import InMemoryOAuthStateStore
from src.infrastructure.jwt_service import (
    JWTService, JWTConfig, InMemoryTokenBlacklist,
)
from src.infrastructure.password_service import PasswordService, PasswordConfig
from src.infrastructure.oauth_google import (
    GOOGLE_AUTH_URL, GoogleOAuthSettings, GoogleIdentity,
)
from src.domain.service import UserService
from src.auth import SessionManager, SessionConfig, AuthenticationService

from fixtures import (
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
    InMemoryConsentRepository,
)


_TEST_SETTINGS = GoogleOAuthSettings(
    client_id="test-google-client-id",
    client_secret="test-google-client-secret",
    redirect_uri="https://auth.example/api/v1/auth/oauth/google/callback",
)


class _FakeTokenResponse:
    status_code = 200
    text = "{}"

    def json(self) -> dict[str, str]:
        return {"id_token": "fake.id.token", "access_token": "fake-google-access"}


class _FakeHttpClient:
    """Stand-in for httpx.AsyncClient — records the POST and returns fixed tokens."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeTokenResponse:
        self.calls.append({"url": url, **kwargs})
        return _FakeTokenResponse()


def _stub_verifier_for(email: str, sub: str = "google-sub-123", name: str | None = "Google User"):
    def _verify(id_token: str, settings: GoogleOAuthSettings, **_: Any) -> GoogleIdentity:
        return GoogleIdentity(
            provider_user_id=sub, email=email, email_verified=True, name=name,
        )
    return _verify


def _build_oauth_app(
    *,
    configured: bool = True,
    http_client: Any | None = None,
    verifier: Any | None = None,
    store: InMemoryOAuthStateStore | None = None,
) -> FastAPI:
    _store = store if store is not None else InMemoryOAuthStateStore()
    _client = http_client if http_client is not None else _FakeHttpClient()

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        state = ServiceState()
        state.jwt_service = JWTService(
            JWTConfig(secret_key="test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars"),
            blacklist=InMemoryTokenBlacklist(),
        )
        state.password_service = PasswordService(PasswordConfig())
        state.user_repository = InMemoryUserRepository()
        state.preferences_repository = InMemoryUserPreferencesRepository()
        state.consent_repository = InMemoryConsentRepository()
        state.user_service = UserService(
            user_repository=state.user_repository,
            preferences_repository=state.preferences_repository,
            consent_repository=state.consent_repository,
            password_service=state.password_service,
        )
        state.session_manager = SessionManager(SessionConfig(
            secret_key="test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars",
        ))
        state.auth_service = AuthenticationService(
            session_manager=state.session_manager,
            jwt_service=state.jwt_service,
            password_service=state.password_service,
        )
        # OAuth wiring (inert unless configured)
        state.oauth_settings = _TEST_SETTINGS if configured else None
        state.oauth_state_store = _store
        state.oauth_http_client_factory = lambda: _client
        state.oauth_id_token_verifier = verifier or _stub_verifier_for("someone@example.com")

        state.initialized = True
        app.state.service = state
        yield
        state.initialized = False

    app = FastAPI(lifespan=test_lifespan)
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(auth_router)
    # stash for test introspection
    app.state._test_store = _store
    app.state._test_client_obj = _client
    return app


class TestConfigGate:
    def test_start_returns_404_when_unconfigured(self) -> None:
        app = _build_oauth_app(configured=False)
        with TestClient(app) as c:
            resp = c.get("/api/v1/auth/oauth/google/start", follow_redirects=False)
            assert resp.status_code == 404

    def test_callback_returns_404_when_unconfigured(self) -> None:
        app = _build_oauth_app(configured=False)
        with TestClient(app) as c:
            resp = c.get(
                "/api/v1/auth/oauth/google/callback",
                params={"code": "x", "state": "y"},
                follow_redirects=False,
            )
            assert resp.status_code == 404


class TestProductionAppGate:
    def test_real_create_app_registers_but_disables_oauth_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The production app wires the routes but they 404 when Google OAuth is unset.

        This is the config-gate contract: existing suites and the frontend
        "do-not-integrate-yet" posture are unaffected because the default env has
        no GOOGLE_OAUTH_CLIENT_ID.
        """
        monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
        from src.main import create_app

        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/auth/oauth/google/start", follow_redirects=False)
            assert resp.status_code == 404


class TestStart:
    def test_start_redirects_to_google_and_persists_state(self) -> None:
        store = InMemoryOAuthStateStore()
        app = _build_oauth_app(store=store)
        with TestClient(app) as c:
            resp = c.get("/api/v1/auth/oauth/google/start", follow_redirects=False)
            assert resp.status_code in (302, 307)
            location = resp.headers["location"]
            assert location.startswith(GOOGLE_AUTH_URL)
            assert "client_id=test-google-client-id" in location
            assert "code_challenge_method=S256" in location
            # state persisted server-side with its PKCE verifier
            qs = parse_qs(urlparse(location).query)
            state = qs["state"][0]
            assert len(store) == 1
            verifier = store._entries[state][0]
            assert verifier and isinstance(verifier, str)


class TestCallback:
    def test_callback_rejects_unknown_state(self) -> None:
        app = _build_oauth_app()
        with TestClient(app) as c:
            resp = c.get(
                "/api/v1/auth/oauth/google/callback",
                params={"code": "abc", "state": "never-issued"},
                follow_redirects=False,
            )
            assert resp.status_code == 400

    def test_callback_creates_user_and_returns_solace_tokens(self) -> None:
        store = InMemoryOAuthStateStore()
        client_obj = _FakeHttpClient()
        app = _build_oauth_app(
            store=store,
            http_client=client_obj,
            verifier=_stub_verifier_for("newuser@example.com", sub="sub-new-1", name="New User"),
        )
        with TestClient(app) as c:
            # /start to obtain a valid persisted state
            start = c.get("/api/v1/auth/oauth/google/start", follow_redirects=False)
            state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]

            resp = c.get(
                "/api/v1/auth/oauth/google/callback",
                params={"code": "auth-code-1", "state": state},
                follow_redirects=False,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["access_token"]
            assert body["refresh_token"]
            assert body["token_type"] == "Bearer"

            # token exchange actually hit the (mocked) Google token endpoint
            assert client_obj.calls, "expected the mocked token exchange to be invoked"
            assert client_obj.calls[0]["data"]["code"] == "auth-code-1"

            # a real user now exists and the issued access token authenticates
            svc = app.state.service
            import asyncio
            user = asyncio.get_event_loop().run_until_complete(
                svc.user_repository.get_by_email("newuser@example.com")
            )
            assert user is not None
            assert user.email_verified is True

            me = c.get(
                "/api/v1/users/me",
                headers={"Authorization": f"Bearer {body['access_token']}"},
            )
            assert me.status_code == 200
            assert me.json()["email"] == "newuser@example.com"

    def test_callback_links_existing_user_without_duplicate(self) -> None:
        store = InMemoryOAuthStateStore()
        app = _build_oauth_app(
            store=store,
            verifier=_stub_verifier_for("existing@example.com", sub="sub-existing"),
        )
        with TestClient(app) as c:
            svc_state = None
            # Pre-create a user with the same email via the register endpoint
            reg = c.post(
                "/api/v1/auth/register",
                json={
                    "email": "existing@example.com",
                    "password": "Password123",
                    "display_name": "Existing",
                },
            )
            assert reg.status_code == 201, reg.text

            start = c.get("/api/v1/auth/oauth/google/start", follow_redirects=False)
            state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
            resp = c.get(
                "/api/v1/auth/oauth/google/callback",
                params={"code": "code-2", "state": state},
                follow_redirects=False,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["access_token"]

            import asyncio
            svc = app.state.service
            count = asyncio.get_event_loop().run_until_complete(
                svc.user_repository.count()
            )
            assert count == 1, "OAuth login on an existing email must not create a duplicate"

    def test_callback_state_is_single_use(self) -> None:
        store = InMemoryOAuthStateStore()
        app = _build_oauth_app(store=store, verifier=_stub_verifier_for("oneuse@example.com"))
        with TestClient(app) as c:
            start = c.get("/api/v1/auth/oauth/google/start", follow_redirects=False)
            state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
            first = c.get(
                "/api/v1/auth/oauth/google/callback",
                params={"code": "c1", "state": state}, follow_redirects=False,
            )
            assert first.status_code == 200
            # replay of the same state must now fail (consumed)
            replay = c.get(
                "/api/v1/auth/oauth/google/callback",
                params={"code": "c1", "state": state}, follow_redirects=False,
            )
            assert replay.status_code == 400

    def test_callback_rejects_suspended_account(self) -> None:
        """Regression (Wave-5 verify): a SUSPENDED local account must NOT obtain
        tokens via the Google flow. The callback enforces can_login() like password
        login, so a suspension/lockout gate cannot be circumvented through OAuth."""
        import asyncio
        from src.domain.entities import AccountStatus
        store = InMemoryOAuthStateStore()
        app = _build_oauth_app(store=store, verifier=_stub_verifier_for("susp@example.com"))
        with TestClient(app) as c:
            reg = c.post("/api/v1/auth/register", json={
                "email": "susp@example.com", "password": "Password123", "display_name": "Susp"})
            assert reg.status_code == 201, reg.text
            svc = app.state.service

            async def _suspend() -> None:
                u = await svc.user_repository.get_by_email("susp@example.com")
                u.email_verified = True
                u.status = AccountStatus.SUSPENDED
                await svc.user_repository.update(u)
            asyncio.get_event_loop().run_until_complete(_suspend())

            start = c.get("/api/v1/auth/oauth/google/start", follow_redirects=False)
            state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
            resp = c.get("/api/v1/auth/oauth/google/callback",
                         params={"code": "c-susp", "state": state}, follow_redirects=False)
            assert resp.status_code == 403, resp.text
            assert "access_token" not in resp.json()

    def test_callback_rejects_locked_account(self) -> None:
        """Regression (Wave-5 verify): a locked-out local account must NOT obtain
        tokens via the Google flow."""
        import asyncio
        from datetime import datetime, timezone, timedelta
        from src.domain.entities import AccountStatus
        store = InMemoryOAuthStateStore()
        app = _build_oauth_app(store=store, verifier=_stub_verifier_for("locked@example.com"))
        with TestClient(app) as c:
            reg = c.post("/api/v1/auth/register", json={
                "email": "locked@example.com", "password": "Password123", "display_name": "Locked"})
            assert reg.status_code == 201, reg.text
            svc = app.state.service

            async def _lock() -> None:
                u = await svc.user_repository.get_by_email("locked@example.com")
                u.email_verified = True
                u.status = AccountStatus.ACTIVE
                u.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                await svc.user_repository.update(u)
            asyncio.get_event_loop().run_until_complete(_lock())

            start = c.get("/api/v1/auth/oauth/google/start", follow_redirects=False)
            state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
            resp = c.get("/api/v1/auth/oauth/google/callback",
                         params={"code": "c-lock", "state": state}, follow_redirects=False)
            assert resp.status_code == 403, resp.text
