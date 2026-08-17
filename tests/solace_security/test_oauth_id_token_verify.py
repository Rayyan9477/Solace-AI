"""C.3b / REV-22 — Google id_token SIGNATURE verification.

The prior helper decoded the id_token payload without verifying the signature, and no
caller verified it separately — so a forged/tampered id_token would be trusted. This
covers the new verify_id_token: RS256 signature (via injectable JWKS resolver) + audience
+ issuer + expiry + verified-email. Full OAuth start/callback endpoints remain deferred
(need live Google), but the security-critical verification is implemented + tested now.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import jwt
import pytest

from solace_security.keys import generate_rsa_keypair


def _load_oauth():
    # oauth_google.py has no relative imports -> load it standalone (avoids the repo-root
    # `infrastructure` package colliding with user-service's `infrastructure`).
    path = _ROOT / "services" / "user-service" / "src" / "infrastructure" / "oauth_google.py"
    spec = importlib.util.spec_from_file_location("oauth_google_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses needs the module registered during class creation
    spec.loader.exec_module(mod)
    return mod


oauth = _load_oauth()
SETTINGS = oauth.GoogleOAuthSettings(
    client_id="my-client-id", client_secret="s", redirect_uri="https://app/callback"
)


def _id_token(priv: str, kid: str, **overrides) -> str:
    now = int(time.time())
    claims = {
        "sub": "google-123", "email": "user@example.com", "email_verified": True,
        "iss": "https://accounts.google.com", "aud": "my-client-id",
        "exp": now + 3600, "iat": now, "name": "Test User",
    }
    claims.update(overrides)
    return jwt.encode(claims, priv, algorithm="RS256", headers={"kid": kid})


def test_valid_id_token_returns_identity():
    priv, pub, kid = generate_rsa_keypair("g1")
    token = _id_token(priv, kid)
    identity = oauth.verify_id_token(token, SETTINGS, signing_key_resolver=lambda t: pub)
    assert identity.provider_user_id == "google-123"
    assert identity.email == "user@example.com"
    assert identity.email_verified is True


def test_tampered_signature_rejected():
    # signed with a DIFFERENT key than the resolver returns -> signature fails
    priv_attacker, _pub_a, kid = generate_rsa_keypair("g1")
    _priv_real, pub_real, _ = generate_rsa_keypair("g1")
    token = _id_token(priv_attacker, kid)
    with pytest.raises(jwt.InvalidSignatureError):
        oauth.verify_id_token(token, SETTINGS, signing_key_resolver=lambda t: pub_real)


def test_wrong_audience_rejected():
    priv, pub, kid = generate_rsa_keypair("g1")
    token = _id_token(priv, kid, aud="some-other-client")
    with pytest.raises(jwt.InvalidAudienceError):
        oauth.verify_id_token(token, SETTINGS, signing_key_resolver=lambda t: pub)


def test_wrong_issuer_rejected():
    priv, pub, kid = generate_rsa_keypair("g1")
    token = _id_token(priv, kid, iss="https://evil.example")
    with pytest.raises(ValueError):
        oauth.verify_id_token(token, SETTINGS, signing_key_resolver=lambda t: pub)


def test_expired_token_rejected():
    priv, pub, kid = generate_rsa_keypair("g1")
    now = int(time.time())
    token = _id_token(priv, kid, exp=now - 100, iat=now - 200)
    with pytest.raises(jwt.ExpiredSignatureError):
        oauth.verify_id_token(token, SETTINGS, signing_key_resolver=lambda t: pub)


def test_unverified_email_rejected():
    priv, pub, kid = generate_rsa_keypair("g1")
    token = _id_token(priv, kid, email_verified=False)
    with pytest.raises(ValueError):
        oauth.verify_id_token(token, SETTINGS, signing_key_resolver=lambda t: pub)


def test_bare_host_issuer_accepted():
    priv, pub, kid = generate_rsa_keypair("g1")
    token = _id_token(priv, kid, iss="accounts.google.com")
    identity = oauth.verify_id_token(token, SETTINGS, signing_key_resolver=lambda t: pub)
    assert identity.provider_user_id == "google-123"
