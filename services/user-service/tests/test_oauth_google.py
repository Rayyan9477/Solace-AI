"""Sprint 8: Google OAuth provider unit tests."""
from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from src.infrastructure.oauth_google import (
    GOOGLE_AUTH_URL,
    AuthorizationRequest,
    GoogleIdentity,
    GoogleOAuthSettings,
    _code_challenge_from_verifier,
    _generate_code_verifier,
    build_authorization_url,
    exchange_code_for_tokens,
    identity_from_id_token_payload,
    parse_id_token_payload,
)


@pytest.fixture
def settings() -> GoogleOAuthSettings:
    return GoogleOAuthSettings(
        client_id="test-google-client-id",
        client_secret="test-google-client-secret",
        redirect_uri="https://auth.example/oauth/google/callback",
    )


class TestPkceVerifier:
    def test_verifier_is_url_safe_and_long_enough(self) -> None:
        verifier = _generate_code_verifier()
        assert 43 <= len(verifier) <= 128, (
            "RFC 7636: code_verifier length must be 43-128 characters"
        )
        # Must contain only unreserved URL-safe characters
        assert all(c.isalnum() or c in "-_" for c in verifier)

    def test_challenge_is_s256_of_verifier(self) -> None:
        import hashlib

        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        digest = hashlib.sha256(verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        assert _code_challenge_from_verifier(verifier) == expected


class TestBuildAuthorizationUrl:
    def test_url_targets_google_auth_endpoint(
        self, settings: GoogleOAuthSettings
    ) -> None:
        req = build_authorization_url(settings)
        assert req.url.startswith(GOOGLE_AUTH_URL)

    def test_url_contains_required_params(
        self, settings: GoogleOAuthSettings
    ) -> None:
        req = build_authorization_url(settings)
        assert "response_type=code" in req.url
        assert "client_id=test-google-client-id" in req.url
        assert "code_challenge_method=S256" in req.url
        assert "scope=openid" in req.url.replace("+", " ")

    def test_state_and_verifier_are_unique_per_call(
        self, settings: GoogleOAuthSettings
    ) -> None:
        req_a = build_authorization_url(settings)
        req_b = build_authorization_url(settings)
        assert req_a.state != req_b.state
        assert req_a.code_verifier != req_b.code_verifier

    def test_caller_supplied_state_and_verifier_are_honoured(
        self, settings: GoogleOAuthSettings
    ) -> None:
        req = build_authorization_url(
            settings, state="fixed-state", code_verifier="fixed-verifier" * 4,
        )
        assert req.state == "fixed-state"
        assert req.code_verifier == "fixed-verifier" * 4
        assert "state=fixed-state" in req.url


class TestExchangeCodeForTokens:
    @pytest.mark.asyncio
    async def test_successful_exchange_returns_json_body(
        self, settings: GoogleOAuthSettings
    ) -> None:
        captured: dict[str, Any] = {}

        class FakeResponse:
            status_code = 200
            text = "{}"

            def json(self) -> dict[str, str]:
                return {"id_token": "fake.id.token", "access_token": "fake-access"}

        class FakeClient:
            async def post(self, url: str, **kwargs: Any) -> FakeResponse:
                captured["url"] = url
                captured["data"] = kwargs.get("data")
                return FakeResponse()

        client = FakeClient()
        tokens = await exchange_code_for_tokens(
            settings,
            code="authorization-code-42",
            code_verifier="verifier-abc",
            http_client=client,
        )
        assert tokens["id_token"] == "fake.id.token"
        assert captured["data"]["grant_type"] == "authorization_code"
        assert captured["data"]["code"] == "authorization-code-42"
        assert captured["data"]["code_verifier"] == "verifier-abc"

    @pytest.mark.asyncio
    async def test_non_2xx_raises(
        self, settings: GoogleOAuthSettings
    ) -> None:
        class FakeResponse:
            status_code = 400
            text = '{"error":"invalid_grant"}'

            def json(self) -> dict[str, str]:
                return {"error": "invalid_grant"}

        class FakeClient:
            async def post(self, url: str, **kwargs: Any) -> FakeResponse:
                return FakeResponse()

        with pytest.raises(RuntimeError):
            await exchange_code_for_tokens(
                settings,
                code="bad-code",
                code_verifier="verifier",
                http_client=FakeClient(),
            )


class TestParseAndMapIdToken:
    def test_parse_id_token_returns_payload(self) -> None:
        payload = {"sub": "1234567890", "email": "alice@example.com",
                   "email_verified": True, "name": "Alice"}
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        body = base64.urlsafe_b64encode(
            json.dumps(payload).encode()).rstrip(b"=").decode()
        id_token = f"{header}.{body}.signature"

        parsed = parse_id_token_payload(id_token)
        assert parsed["sub"] == "1234567890"
        assert parsed["email"] == "alice@example.com"

    def test_parse_id_token_rejects_malformed(self) -> None:
        with pytest.raises(ValueError):
            parse_id_token_payload("not-a-jwt")

    def test_identity_from_payload_maps_claims(self) -> None:
        ident = identity_from_id_token_payload(
            {
                "sub": "1234567890",
                "email": "alice@example.com",
                "email_verified": True,
                "name": "Alice",
                "picture": "https://example.com/avatar.png",
            }
        )
        assert isinstance(ident, GoogleIdentity)
        assert ident.provider_user_id == "1234567890"
        assert ident.email == "alice@example.com"
        assert ident.email_verified is True
        assert ident.name == "Alice"
        assert ident.picture_url == "https://example.com/avatar.png"

    def test_identity_requires_sub_and_email(self) -> None:
        with pytest.raises(ValueError):
            identity_from_id_token_payload({"sub": "123"})
        with pytest.raises(ValueError):
            identity_from_id_token_payload({"email": "a@b.com"})


class TestAuthorizationRequestDataclass:
    def test_authorization_request_is_frozen(self) -> None:
        req = AuthorizationRequest(
            url="https://example", state="state", code_verifier="verifier",
        )
        with pytest.raises(AttributeError):
            req.url = "mutated"  # type: ignore[misc]
