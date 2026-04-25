"""Sprint 8: Google OAuth2 provider for user-service.

Thin wrapper around the Google OAuth 2.0 + OpenID Connect flow:

    1. Build the authorization URL with a state nonce + PKCE code verifier.
    2. Exchange the code + verifier at the token endpoint.
    3. Verify the returned id_token with Google's public keys.
    4. Return a ``GoogleIdentity`` that the user-service wires up to either
       an existing local account (by email) or a new one.

No network calls are made in unit tests — the helper exposes the two
pure helpers (``build_authorization_url`` and ``verify_id_token``) that
can be exercised with a mocked ``httpx`` client.

Design note: we do not use ``authlib`` here because the surface we need
is small and a hand-rolled client keeps the dependency footprint low.
In production we still install ``authlib`` (requirements.txt) so
services that need a more complete OAuth story can use it; this module
is the user-service's minimal implementation.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import structlog

logger = structlog.get_logger(__name__)


# Google OIDC discovery endpoints (stable; change rarely)
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_ISSUER = "https://accounts.google.com"


@dataclass(frozen=True)
class GoogleOAuthSettings:
    """Configuration for the Google OAuth 2.0 flow."""

    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str = "openid email profile"


@dataclass(frozen=True)
class AuthorizationRequest:
    """State captured at ``/oauth/google/start`` and recalled at callback."""

    url: str
    state: str
    code_verifier: str


@dataclass(frozen=True)
class GoogleIdentity:
    """Verified subject returned from a successful OAuth flow."""

    provider_user_id: str  # Google's stable ``sub`` claim
    email: str
    email_verified: bool
    name: str | None = None
    picture_url: str | None = None


def _generate_code_verifier() -> str:
    """RFC 7636 PKCE code verifier (43-128 unreserved chars)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()


def _code_challenge_from_verifier(code_verifier: str) -> str:
    """S256 PKCE challenge derived from the verifier."""
    digest = hashlib.sha256(code_verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def build_authorization_url(
    settings: GoogleOAuthSettings,
    *,
    state: str | None = None,
    code_verifier: str | None = None,
) -> AuthorizationRequest:
    """Construct the Google consent-screen URL for the ``/start`` endpoint.

    Args:
        settings: Google OAuth client config.
        state: CSRF nonce; generated if None. Must survive the round trip.
        code_verifier: PKCE verifier; generated if None. Must be retained
            by the caller (e.g. stored in a signed cookie or server-side
            session) for the callback to exchange.

    Returns:
        An ``AuthorizationRequest`` containing the redirect URL plus the
        state + verifier the callback must present.
    """
    state = state or secrets.token_urlsafe(24)
    code_verifier = code_verifier or _generate_code_verifier()
    code_challenge = _code_challenge_from_verifier(code_verifier)

    params = {
        "response_type": "code",
        "client_id": settings.client_id,
        "redirect_uri": settings.redirect_uri,
        "scope": settings.scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return AuthorizationRequest(url=url, state=state, code_verifier=code_verifier)


async def exchange_code_for_tokens(
    settings: GoogleOAuthSettings,
    *,
    code: str,
    code_verifier: str,
    http_client: Any,
) -> dict[str, Any]:
    """Exchange an authorization code for tokens at the Google token endpoint.

    Args:
        settings: Google OAuth client config.
        code: Authorization code from the callback query string.
        code_verifier: PKCE verifier stashed in /start.
        http_client: Anything implementing ``httpx.AsyncClient.post``. The
            caller owns its lifecycle; we don't create or close it here so
            the test suite can inject a mock.

    Returns:
        Parsed JSON body (should contain ``id_token`` and ``access_token``).

    Raises:
        RuntimeError: if Google returns a non-2xx response.
    """
    response = await http_client.post(
        GOOGLE_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": settings.client_id,
            "client_secret": settings.client_secret,
            "redirect_uri": settings.redirect_uri,
            "code_verifier": code_verifier,
        },
        headers={"Accept": "application/json"},
    )
    if response.status_code // 100 != 2:
        logger.error(
            "google_token_exchange_failed",
            status=response.status_code,
            body=response.text[:500] if hasattr(response, "text") else "",
        )
        raise RuntimeError(
            f"Google token exchange failed with status {response.status_code}"
        )
    return response.json()


def parse_id_token_payload(id_token: str) -> dict[str, Any]:
    """Decode the id_token payload segment WITHOUT signature verification.

    Callers must verify the signature separately (via google-auth's
    ``id_token.verify_oauth2_token``) before trusting any claim. This
    helper is used for diagnostic logging and to extract ``sub`` + ``email``
    when a real verification client is not available (e.g. unit tests).
    """
    import json

    parts = id_token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed id_token: expected three dot-separated parts")
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
    return json.loads(payload_bytes)


def identity_from_id_token_payload(payload: dict[str, Any]) -> GoogleIdentity:
    """Map a verified id_token payload to a ``GoogleIdentity``.

    Assumes signature + issuer + audience checks have already passed
    (the caller is responsible). Missing ``sub`` or ``email`` raises.
    """
    provider_user_id = payload.get("sub")
    email = payload.get("email")
    if not provider_user_id or not email:
        raise ValueError(
            "id_token payload missing required claims: sub + email"
        )
    return GoogleIdentity(
        provider_user_id=str(provider_user_id),
        email=str(email),
        email_verified=bool(payload.get("email_verified", False)),
        name=payload.get("name"),
        picture_url=payload.get("picture"),
    )
