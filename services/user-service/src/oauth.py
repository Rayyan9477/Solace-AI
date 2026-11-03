"""REV-22: Google OAuth 2.0 sign-in endpoints for the user-service (config-gated).

This is the *flow* layer on top of the already-landed C.3b primitives in
``src/infrastructure/oauth_google.py`` (authorization-URL builder, PKCE, token
exchange, and the RS256 id_token SIGNATURE verifier). It exposes two routes:

    GET /api/v1/auth/oauth/google/start
        Generate a CSRF ``state`` + PKCE verifier/challenge (S256), persist them
        server-side under a short TTL via an injectable store seam, then redirect
        the browser to Google's consent screen.

    GET /api/v1/auth/oauth/google/callback
        Validate ``state`` (reject on mismatch/expiry / single-use), exchange the
        authorization ``code`` for tokens at Google's token endpoint (injectable
        HTTP client), VERIFY the id_token signature via the existing verifier, then
        LINK-OR-CREATE the local user and mint the normal Solace access + refresh
        tokens — exactly as password login does.

CONFIG-GATE: both routes are inert (HTTP 404) unless Google OAuth is configured
(``GOOGLE_OAUTH_CLIENT_ID`` + ``GOOGLE_OAUTH_CLIENT_SECRET``). The routes are
always *registered* but resolve ``oauth_settings`` from app state at request
time; when that is ``None`` they 404, so existing suites and the frontend
"do-not-integrate-yet" posture are unaffected. Default = disabled.
"""
from __future__ import annotations

import os
import secrets
import time
from typing import Any, Callable

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from .api import TokenResponse
from .domain.value_objects import AccountStatus
from .infrastructure.oauth_google import (
    GoogleIdentity,
    GoogleOAuthSettings,
    build_authorization_url,
    exchange_code_for_tokens,
    verify_id_token,
)

logger = structlog.get_logger(__name__)

# How long a pending authorization (state + PKCE verifier) may live before the
# callback must be received. Short by design: the round trip is interactive.
OAUTH_STATE_TTL_SECONDS = 600

oauth_router = APIRouter()


# --------------------------------------------------------------------------- #
# State + PKCE store seam                                                      #
# --------------------------------------------------------------------------- #
# The ``state`` -> ``code_verifier`` binding created at /start must survive to
# /callback. It is deliberately behind a tiny async seam so a single process
# uses the in-memory default while a multi-instance deployment injects a
# Redis-backed implementation (the browser can hit any instance on callback).


class InMemoryOAuthStateStore:
    """Process-local pending-authorization store with per-entry TTL.

    Suitable for single-instance / dev / tests only — entries are lost on
    restart and not shared across workers. Use ``RedisOAuthStateStore`` when
    more than one instance may serve the callback.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        # state -> (code_verifier, expires_at_monotonic)
        self._entries: dict[str, tuple[str, float]] = {}

    async def put(self, state: str, code_verifier: str, *, ttl_seconds: int = OAUTH_STATE_TTL_SECONDS) -> None:
        self._entries[state] = (code_verifier, self._clock() + ttl_seconds)

    async def pop(self, state: str) -> str | None:
        """Single-use: remove and return the verifier, or ``None`` if missing/expired."""
        entry = self._entries.pop(state, None)
        if entry is None:
            return None
        code_verifier, expires_at = entry
        if self._clock() > expires_at:
            return None
        return code_verifier

    def __len__(self) -> int:  # test/introspection convenience
        return len(self._entries)


class RedisOAuthStateStore:
    """Redis-backed pending-authorization store (multi-instance safe).

    ``redis_client`` is anything implementing async ``set(key, value, ex=...)``,
    ``get(key)`` and ``delete(key)`` (e.g. ``redis.asyncio.Redis``). TTL is
    enforced by Redis key expiry; ``pop`` deletes on read for single use.
    """

    _PREFIX = "oauth:google:state:"

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def put(self, state: str, code_verifier: str, *, ttl_seconds: int = OAUTH_STATE_TTL_SECONDS) -> None:
        await self._redis.set(self._PREFIX + state, code_verifier, ex=ttl_seconds)

    async def pop(self, state: str) -> str | None:
        key = self._PREFIX + state
        code_verifier = await self._redis.get(key)
        if code_verifier is None:
            return None
        try:
            await self._redis.delete(key)
        except Exception:  # best-effort single-use; expiry still bounds reuse
            logger.warning("oauth_state_delete_failed", state_present=True)
        return code_verifier


# --------------------------------------------------------------------------- #
# Config gate + wiring helpers                                                 #
# --------------------------------------------------------------------------- #


def google_oauth_settings_from_env() -> GoogleOAuthSettings | None:
    """Build ``GoogleOAuthSettings`` from the environment, or ``None`` if disabled.

    The feature is OFF unless BOTH ``GOOGLE_OAUTH_CLIENT_ID`` and
    ``GOOGLE_OAUTH_CLIENT_SECRET`` are present. ``GOOGLE_OAUTH_REDIRECT_URI``
    defaults to the local callback path for dev but should be set explicitly in
    any real deployment.
    """
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    redirect_uri = os.getenv(
        "GOOGLE_OAUTH_REDIRECT_URI",
        "http://localhost:8001/api/v1/auth/oauth/google/callback",
    ).strip()
    scope = os.getenv("GOOGLE_OAUTH_SCOPE", "openid email profile").strip()
    return GoogleOAuthSettings(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
    )


def default_httpx_client_factory() -> Any:
    """Create a real ``httpx.AsyncClient`` for the production token exchange."""
    import httpx

    return httpx.AsyncClient(timeout=10.0)


def _require_oauth_settings(request: Request) -> GoogleOAuthSettings:
    """Resolve OAuth settings from app state; 404 when the feature is disabled."""
    settings = getattr(request.app.state.service, "oauth_settings", None)
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Google OAuth is not configured",
        )
    return settings


def _state_store(request: Request) -> Any:
    store = getattr(request.app.state.service, "oauth_state_store", None)
    if store is None:
        store = InMemoryOAuthStateStore()
        request.app.state.service.oauth_state_store = store
    return store


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #


@oauth_router.get("/auth/oauth/google/start", tags=["OAuth"])
async def google_oauth_start(request: Request) -> RedirectResponse:
    """Begin Google sign-in: persist state + PKCE, redirect to the consent screen."""
    settings = _require_oauth_settings(request)
    store = _state_store(request)

    auth_req = build_authorization_url(settings)
    await store.put(auth_req.state, auth_req.code_verifier, ttl_seconds=OAUTH_STATE_TTL_SECONDS)

    logger.info("google_oauth_start")
    return RedirectResponse(url=auth_req.url, status_code=status.HTTP_302_FOUND)


@oauth_router.get(
    "/auth/oauth/google/callback",
    response_model=TokenResponse,
    tags=["OAuth"],
)
async def google_oauth_callback(request: Request, code: str, state: str) -> TokenResponse:
    """Complete Google sign-in: validate state, exchange code, verify id_token, link-or-create.

    Returns the standard Solace access + refresh token pair, identical to
    password login, so the frontend can treat the result the same way.
    """
    settings = _require_oauth_settings(request)
    svc = request.app.state.service
    store = _state_store(request)

    # 1. Validate + consume state (CSRF + single-use + expiry).
    code_verifier = await store.pop(state)
    if not code_verifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state",
        )

    # 2. Exchange the authorization code for tokens (injectable HTTP client).
    client_factory = getattr(svc, "oauth_http_client_factory", None) or default_httpx_client_factory
    http_client = client_factory()
    try:
        tokens = await exchange_code_for_tokens(
            settings, code=code, code_verifier=code_verifier, http_client=http_client,
        )
    except Exception as exc:
        logger.warning("google_oauth_token_exchange_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Google token exchange failed",
        )
    finally:
        aclose = getattr(http_client, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - closing a client must not mask the flow
                pass

    id_token = tokens.get("id_token")
    if not id_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Google token response missing id_token",
        )

    # 3. VERIFY the id_token signature via the existing verifier (injectable for tests).
    verifier = getattr(svc, "oauth_id_token_verifier", None) or verify_id_token
    try:
        identity: GoogleIdentity = verifier(id_token, settings)
    except Exception as exc:
        logger.warning("google_oauth_id_token_invalid", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google id_token verification failed",
        )

    # 4. Link-or-create the local account.
    user = await _link_or_create_user(svc, identity)

    # 4b. Enforce the SAME account-status gate as password login: a SUSPENDED /
    # INACTIVE / locked account MUST NOT obtain tokens via the Google flow (authz
    # bypass caught in Wave-5 adversarial verify). A Google-verified email already
    # satisfies email-verification, so a residual "Email verification required" does
    # not block here; suspension and lockout DO.
    can_login, login_error = user.can_login()
    if not can_login and login_error != "Email verification required":
        logger.warning(
            "google_oauth_login_blocked",
            user_id=str(user.user_id),
            reason=login_error,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not permitted to sign in",
        )

    # 5. Issue the normal Solace tokens (mirror the password-login path).
    await svc.session_manager.create_session(
        user=user,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    token_pair = svc.jwt_service.generate_token_pair(
        user_id=user.user_id,
        email=user.email,
        role=user.role.value,
    )
    try:
        request.app.state.service.increment_stat("logins")
    except Exception:  # stat counter is best-effort
        pass

    logger.info("google_oauth_login_succeeded", user_id=str(user.user_id))
    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        token_type=token_pair.token_type,
        expires_in=token_pair.expires_in,
    )


async def _link_or_create_user(svc: Any, identity: GoogleIdentity) -> Any:
    """Find the local user by (verified) Google email, or create + activate a new one.

    A Google-verified email is authoritative for account matching here. New
    accounts get a random, unusable password hash (login is via Google) and are
    marked email-verified + ACTIVE so token issuance and can_login checks pass.
    """
    email = identity.email.lower().strip()
    user = await svc.user_repository.get_by_email(email)
    if user is not None:
        # Consistency with the create branch: a Google-verified email satisfies the
        # local email-verification requirement, so promote a still-unverified /
        # PENDING_VERIFICATION local account to verified + ACTIVE. Account-status
        # enforcement (suspension / lockout) is NOT done here — the callback does it
        # via can_login() before issuing tokens.
        if identity.email_verified and not user.email_verified:
            user.email_verified = True
            if user.status == AccountStatus.PENDING_VERIFICATION:
                user.activate()
            user = await svc.user_repository.update(user)
        return user

    random_secret = secrets.token_urlsafe(32)
    password_hash = svc.password_service.hash_password(random_secret)
    display_name = identity.name or email.split("@", 1)[0]

    result = await svc.user_service.create_user(
        email=email,
        password_hash=password_hash,
        display_name=display_name,
    )
    if not result.success or not result.user:
        # An EMAIL_EXISTS race means a concurrent request created it — re-read.
        existing = await svc.user_repository.get_by_email(email)
        if existing is not None:
            return existing
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to create user from Google identity",
        )

    user = result.user
    if not user.email_verified:
        user.email_verified = True
        if user.status == AccountStatus.PENDING_VERIFICATION:
            user.activate()
        user = await svc.user_repository.update(user)
    return user
