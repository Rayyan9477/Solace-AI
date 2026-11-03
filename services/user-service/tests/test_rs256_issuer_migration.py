"""P1-1: RS256 issuer migration for user-service JWTService.

The headline Phase-C finding: user access/refresh tokens were signed HS256 with the
symmetric AUTH_SECRET_KEY that EVERY backend container holds, so one compromised
container could forge any user/admin token. This suite proves the issuer now signs
RS256 (issuer-only private key + kid) and that the SAME token verifies on BOTH the
user-service's own verifier AND the shared solace_security validators, while legacy
HS256 stays accepted by default (dual-accept) and is rejected only in strict mode.
"""
from __future__ import annotations

import json
from uuid import uuid4

import jwt
import pytest

from src.infrastructure.jwt_service import JWTConfig, JWTService, TokenType

from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager
from solace_security.keys import KeyManager, generate_rsa_keypair

SHARED_SECRET = "shared-symmetric-secret-32-bytes-long!!"


def _rs256_issuer() -> tuple[JWTService, str, str, str]:
    """Build an RS256 issuer JWTService. Returns (service, priv, pub, kid)."""
    priv, pub, kid = generate_rsa_keypair("user-k1")
    config = JWTConfig(
        secret_key=SHARED_SECRET,
        algorithm="RS256",
        private_key_pem=priv,
        active_kid=kid,
        public_keys={kid: pub},
    )
    return JWTService(config), priv, pub, kid


def _shared_validator(pub: str, kid: str, *, accept_legacy_hs256: bool) -> JWTManager:
    """A solace_security JWTManager as another service would run it: verify-only keyset."""
    settings = AuthSettings(
        secret_key=SHARED_SECRET,
        algorithm="RS256",
        public_keys_json=json.dumps({kid: pub}),
        accept_legacy_hs256=accept_legacy_hs256,
    )
    return JWTManager(settings, token_blacklist=InMemoryTokenBlacklist())


# (a) RS256 token verifies on BOTH verifiers ------------------------------------

def test_issued_access_token_is_rs256_with_kid():
    svc, _priv, _pub, kid = _rs256_issuer()
    pair = svc.generate_token_pair(uuid4(), "u@example.com", "user")
    hdr = jwt.get_unverified_header(pair.access_token)
    assert hdr["alg"] == "RS256"
    assert hdr["kid"] == kid


@pytest.mark.asyncio
async def test_rs256_token_verifies_on_user_service_verifier():
    svc, _priv, _pub, _kid = _rs256_issuer()
    uid = uuid4()
    pair = svc.generate_token_pair(uid, "u@example.com", "clinician")
    payload = await svc.verify_token(pair.access_token, TokenType.ACCESS)
    assert payload.user_id == uid
    assert payload.email == "u@example.com"
    assert payload.role == "clinician"


def test_rs256_token_verifies_on_shared_solace_security_validator():
    svc, _priv, pub, kid = _rs256_issuer()
    uid = uuid4()
    pair = svc.generate_token_pair(uid, "u@example.com", "user")
    validator = _shared_validator(pub, kid, accept_legacy_hs256=True)
    result = validator.decode_token(pair.access_token)
    assert result.success is True
    # solace_security reads the token subject; must equal the user id (union claim).
    assert result.user_id == str(uid)
    assert "user" in result.payload.roles


# (b) strict mode rejects a forged HS256 token signed with the shared secret -----

def test_strict_validator_rejects_forged_hs256_signed_with_shared_secret():
    _svc, _priv, pub, kid = _rs256_issuer()
    # Attacker holds the shared symmetric secret (every container does today) and forges
    # an admin token. In RS256-strict mode the shared validator must reject it.
    forged = jwt.encode(
        {
            "sub": "attacker", "user_id": str(uuid4()), "email": "a@evil.com",
            "role": "admin", "roles": ["admin"], "type": "access", "jti": "j1",
            "iat": 1, "exp": 9999999999, "iss": "solace-ai", "aud": "solace-ai-api",
        },
        SHARED_SECRET, algorithm="HS256",
    )
    strict = _shared_validator(pub, kid, accept_legacy_hs256=False)
    assert strict.decode_token(forged).success is False


@pytest.mark.asyncio
async def test_user_service_strict_mode_rejects_forged_hs256():
    priv, pub, kid = generate_rsa_keypair("user-k1")
    config = JWTConfig(
        secret_key=SHARED_SECRET,
        algorithm="RS256",
        private_key_pem=priv,
        active_kid=kid,
        public_keys={kid: pub},
        accept_legacy_hs256=False,  # strict opt-in
    )
    svc = JWTService(config)
    forged = jwt.encode(
        {"sub": "attacker", "user_id": str(uuid4()), "email": "a@evil.com",
         "role": "admin", "type": "access", "jti": "j", "iat": 1, "exp": 9999999999,
         "iss": "solace-ai", "aud": "solace-ai-api"},
        SHARED_SECRET, algorithm="HS256",
    )
    from src.infrastructure.jwt_service import TokenInvalidError

    with pytest.raises(TokenInvalidError):
        await svc.verify_token(forged, TokenType.ACCESS)


# (c) default dual-accept still verifies legacy HS256 (no regression) ------------

def test_default_dual_accept_still_verifies_legacy_hs256():
    _svc, _priv, pub, kid = _rs256_issuer()
    legacy = jwt.encode(
        {"sub": "u1", "type": "access", "jti": "j", "iat": 1, "exp": 9999999999,
         "iss": "solace-ai", "aud": "solace-ai-api"},
        SHARED_SECRET, algorithm="HS256",
    )
    # Default (dual-accept) validator must still accept it during rollout.
    dual = _shared_validator(pub, kid, accept_legacy_hs256=True)
    assert dual.decode_token(legacy).success is True


@pytest.mark.asyncio
async def test_hs256_default_issuer_unchanged():
    """With no RS256 material, JWTService still signs HS256 exactly as before."""
    config = JWTConfig(secret_key=SHARED_SECRET, algorithm="HS256")
    svc = JWTService(config)
    pair = svc.generate_token_pair(uuid4(), "u@example.com", "user")
    assert jwt.get_unverified_header(pair.access_token)["alg"] == "HS256"
    payload = await svc.verify_token(pair.access_token, TokenType.ACCESS)
    assert payload.role == "user"
