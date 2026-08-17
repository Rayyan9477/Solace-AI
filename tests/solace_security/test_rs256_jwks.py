"""C.3 — RS256/JWKS asymmetric signing + rotation + HS256->RS256 migration.

HS256 is symmetric: every service shares AUTH_SECRET_KEY and can therefore MINT user
tokens, not just verify them. RS256 fixes this — only the issuer holds the private key;
everyone else verifies with the public key (JWKS). See
docs/superpowers/specs/2026-07-28-rs256-jwks-auth-design.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import json
import jwt
import pytest

from solace_security.auth import AuthSettings, JWTManager, TokenType, InMemoryTokenBlacklist
from solace_security.keys import generate_rsa_keypair, KeyManager, build_jwks

SECRET = "test-secret-key-32-bytes-long!!!"


def _rs256_settings(keypairs, active_kid, *, accept_legacy=False, private_pem=None):
    """keypairs: list of (private_pem, public_pem, kid) from generate_rsa_keypair. active_kid signs."""
    public_keys = {kid: pub for _priv, pub, kid in keypairs}
    priv = private_pem or next(p for p, _pub, kid in keypairs if kid == active_kid)
    return AuthSettings(
        secret_key=SECRET,
        algorithm="RS256",
        private_key_pem=priv,
        active_kid=active_kid,
        public_keys_json=json.dumps(public_keys),
        accept_legacy_hs256=accept_legacy,
    )


def _mgr(settings):
    return JWTManager(settings, token_blacklist=InMemoryTokenBlacklist())


# ---- keys.py unit ----

def test_generate_rsa_keypair_returns_pems():
    priv, pub, kid = generate_rsa_keypair("k1")
    assert kid == "k1"
    assert "BEGIN PRIVATE KEY" in priv
    assert "BEGIN PUBLIC KEY" in pub


def test_build_jwks_shape():
    _priv, pub, kid = generate_rsa_keypair("k1")
    jwks = build_jwks({kid: pub})
    assert "keys" in jwks and len(jwks["keys"]) == 1
    k = jwks["keys"][0]
    assert k["kty"] == "RSA" and k["use"] == "sig" and k["alg"] == "RS256"
    assert k["kid"] == "k1" and k["n"] and k["e"]


def test_keymanager_from_settings_serves_jwks():
    kp = generate_rsa_keypair("k1")
    km = KeyManager.from_settings(_rs256_settings([kp], "k1"))
    jwks = km.jwks()
    assert [k["kid"] for k in jwks["keys"]] == ["k1"]
    assert km.can_sign  # issuer holds the private key


def test_keymanager_sign_verifiable_with_public_key():
    priv, pub, kid = generate_rsa_keypair("k1")
    km = KeyManager(active_kid=kid, private_pem=priv, public_keys={kid: pub})
    token = km.sign({"sub": "u1", "exp": 9999999999})
    assert jwt.get_unverified_header(token)["kid"] == "k1"
    assert jwt.decode(token, pub, algorithms=["RS256"])["sub"] == "u1"


# ---- JWTManager RS256 behavior ----

def test_access_token_is_rs256_with_kid_when_enabled():
    kp = generate_rsa_keypair("k1")
    mgr = _mgr(_rs256_settings([kp], "k1"))
    tok = mgr.create_access_token("u1", roles=["user"])
    hdr = jwt.get_unverified_header(tok)
    assert hdr["alg"] == "RS256" and hdr["kid"] == "k1"
    res = mgr.decode_token(tok)
    assert res.success and res.user_id == "u1"


def test_rotation_prior_kid_still_verifies():
    k1 = generate_rsa_keypair("k1")
    k2 = generate_rsa_keypair("k2")
    # Verifier knows BOTH public keys; issuer now signs with k2.
    verifier = _mgr(_rs256_settings([k1, k2], "k2"))
    # A token minted earlier under k1 (issuer's prior active key).
    old_issuer = _mgr(_rs256_settings([k1, k2], "k1"))
    old_token = old_issuer.create_access_token("u1")
    assert jwt.get_unverified_header(old_token)["kid"] == "k1"
    assert verifier.decode_token(old_token).success


def test_unknown_kid_is_rejected():
    k1 = generate_rsa_keypair("k1")
    k2 = generate_rsa_keypair("k2")
    # token signed by k2, but verifier only knows k1
    issuer = _mgr(_rs256_settings([k2], "k2"))
    tok = issuer.create_access_token("u1")
    verifier = _mgr(_rs256_settings([k1], "k1"))
    assert verifier.decode_token(tok).success is False


# ---- HS256 legacy migration ----

def test_legacy_hs256_access_rejected_when_flag_off():
    kp = generate_rsa_keypair("k1")
    mgr = _mgr(_rs256_settings([kp], "k1", accept_legacy=False))
    legacy = jwt.encode(
        {"sub": "u1", "type": "access", "jti": "j1", "iat": 1, "exp": 9999999999,
         "iss": "solace-ai", "aud": "solace-ai-api"},
        SECRET, algorithm="HS256",
    )
    assert mgr.decode_token(legacy).success is False


def test_legacy_hs256_access_accepted_when_flag_on():
    kp = generate_rsa_keypair("k1")
    mgr = _mgr(_rs256_settings([kp], "k1", accept_legacy=True))
    legacy = jwt.encode(
        {"sub": "u1", "type": "access", "jti": "j1", "iat": 1, "exp": 9999999999,
         "iss": "solace-ai", "aud": "solace-ai-api"},
        SECRET, algorithm="HS256",
    )
    res = mgr.decode_token(legacy)
    assert res.success and res.user_id == "u1"


def test_service_token_stays_hs256_and_verifies_even_with_flag_off():
    kp = generate_rsa_keypair("k1")
    mgr = _mgr(_rs256_settings([kp], "k1", accept_legacy=False))
    svc = mgr.create_service_token("orchestrator-service", permissions=["call:safety"])
    assert jwt.get_unverified_header(svc)["alg"] == "HS256"  # service tokens remain symmetric
    res = mgr.decode_token(svc, expected_type=TokenType.SERVICE)
    assert res.success


# ---- security invariants ----

def test_alg_confusion_hs256_user_token_rejected_in_rs256_mode():
    priv, pub, kid = generate_rsa_keypair("k1")
    mgr = _mgr(_rs256_settings([(priv, pub, kid)], "k1", accept_legacy=False))
    # Classic RS256/HS256 confusion = getting the server to verify an alg=HS256 token with
    # the RSA PUBLIC key as the HMAC secret. Our verifier never does that: the HS256 path
    # uses the shared secret only, and in RS256 mode HS256 *user* tokens are rejected by
    # policy. (PyJWT also refuses to HS256-sign with a PEM key, blocking it at encode time.)
    forged = jwt.encode(
        {"sub": "attacker", "type": "access", "jti": "j", "iat": 1, "exp": 9999999999,
         "iss": "solace-ai", "aud": "solace-ai-api"},
        "attacker-controlled-secret-not-ours-32byte", algorithm="HS256",
    )
    assert mgr.decode_token(forged).success is False


def test_alg_none_is_rejected():
    kp = generate_rsa_keypair("k1")
    mgr = _mgr(_rs256_settings([kp], "k1"))
    unsigned = jwt.encode(
        {"sub": "x", "type": "access", "jti": "j", "iat": 1, "exp": 9999999999,
         "iss": "solace-ai", "aud": "solace-ai-api"},
        key="", algorithm="none",
    )
    assert mgr.decode_token(unsigned).success is False


# ---- dev ergonomics + backward compat ----

def test_dev_autogen_rs256_roundtrips():
    settings = AuthSettings.for_development(algorithm="RS256")
    mgr = _mgr(settings)
    tok = mgr.create_access_token("u1")
    assert jwt.get_unverified_header(tok)["alg"] == "RS256"
    assert mgr.decode_token(tok).success


def test_escaped_pem_from_env_file_is_normalized():
    # .env files carry the PEM on one line with \n escapes; loading must restore newlines.
    priv, pub, kid = generate_rsa_keypair("k1")
    settings = AuthSettings(
        secret_key=SECRET, algorithm="RS256",
        private_key_pem=priv.replace("\n", "\\n"),
        active_kid=kid,
        public_keys_json=json.dumps({kid: pub.replace(chr(10), "\\n")}),
    )
    mgr = _mgr(settings)
    tok = mgr.create_access_token("u1")
    assert mgr.decode_token(tok).success


def test_default_hs256_unchanged():
    mgr = _mgr(AuthSettings(secret_key=SECRET))  # no RS config -> HS256 as before
    tok = mgr.create_access_token("u1")
    assert jwt.get_unverified_header(tok)["alg"] == "HS256"
    assert mgr.decode_token(tok).success
