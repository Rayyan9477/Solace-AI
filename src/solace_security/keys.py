"""RS256 signing keys + JWKS helpers (C.3).

Splits key management out of auth.py: the KeyManager holds the issuer's RS256 signing key
and the verification keyset (kid -> public PEM), supports kid-based rotation, and converts
PEM public keys into a JWKS document for the /.well-known/jwks.json endpoint.
"""
from __future__ import annotations

import json
import os
from typing import Any

import jwt
import structlog
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

logger = structlog.get_logger(__name__)


def _normalize_pem(pem: str) -> str:
    """Restore real newlines in a PEM that was single-lined for an env file (``\\n`` escapes).

    A real PEM (with actual newlines) is unaffected. This lets AUTH_PRIVATE_KEY_PEM /
    AUTH_PUBLIC_KEYS_JSON live on one line in .env files.
    """
    return pem.replace("\\n", "\n") if "\\n" in pem else pem


def generate_rsa_keypair(kid: str, key_size: int = 2048) -> tuple[str, str, str]:
    """Generate a fresh RSA keypair. Returns (private_pem, public_pem, kid)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem, kid


def build_jwks(public_keys: dict[str, str]) -> dict[str, Any]:
    """Convert a {kid: public_pem} map into a JWKS document (RFC 7517)."""
    keys: list[dict[str, Any]] = []
    for kid, pem in public_keys.items():
        pub = serialization.load_pem_public_key(pem.encode())
        jwk = json.loads(RSAAlgorithm.to_jwk(pub))
        keys.append(
            {
                "kty": jwk["kty"],
                "use": "sig",
                "alg": "RS256",
                "kid": kid,
                "n": jwk["n"],
                "e": jwk["e"],
            }
        )
    return {"keys": keys}


class KeyManager:
    """Holds the RS256 signing key (issuer only) and the verification keyset.

    - Signing: the issuer sets ``active_kid`` + ``private_pem`` and calls ``sign`` (stamps
      the ``kid`` header). Verifier-only services leave ``private_pem`` None (``can_sign`` False).
    - Verifying: ``public_keys`` maps ``kid -> public_pem`` and may hold several keys so a
      token signed by a prior kid still verifies during rotation.
    """

    def __init__(
        self,
        *,
        active_kid: str | None = None,
        private_pem: str | None = None,
        public_keys: dict[str, str] | None = None,
    ) -> None:
        self._active_kid = active_kid
        self._private_pem = private_pem
        self._public_keys: dict[str, str] = dict(public_keys or {})

    @property
    def active_kid(self) -> str | None:
        return self._active_kid

    @property
    def can_sign(self) -> bool:
        return bool(self._private_pem and self._active_kid)

    def kids(self) -> list[str]:
        return list(self._public_keys.keys())

    def public_keys(self) -> dict[str, str]:
        return dict(self._public_keys)

    def public_pem_for(self, kid: str | None) -> str | None:
        if not kid:
            return None
        return self._public_keys.get(kid)

    def sign(self, claims: dict[str, Any]) -> str:
        """RS256-sign claims, stamping the active kid in the header."""
        if not self.can_sign:
            raise RuntimeError("KeyManager has no private key / active kid to sign with")
        return jwt.encode(
            claims, self._private_pem, algorithm="RS256", headers={"kid": self._active_kid}
        )

    def jwks(self) -> dict[str, Any]:
        return build_jwks(self._public_keys)

    @classmethod
    def from_settings(cls, settings: Any) -> "KeyManager":
        """Build a KeyManager from AuthSettings. Empty (HS256 mode) unless AUTH_ALGORITHM=RS256.

        In non-production, if RS256 is requested with NO keyset provisioned, an ephemeral
        keypair is generated so dev/tests work without provisioning. A verifier-only service
        (public keyset present, no private key) is a valid production config.
        """
        if getattr(settings, "algorithm", "HS256") != "RS256":
            return cls()

        private_pem = None
        raw_priv = getattr(settings, "private_key_pem", None)
        if raw_priv is not None:
            private_pem = (
                raw_priv.get_secret_value() if hasattr(raw_priv, "get_secret_value") else str(raw_priv)
            )
            private_pem = _normalize_pem(private_pem)
        active_kid = getattr(settings, "active_kid", None)

        public_keys: dict[str, str] = {}
        raw_public = getattr(settings, "public_keys_json", None)
        if raw_public:
            parsed = json.loads(raw_public) if isinstance(raw_public, str) else dict(raw_public)
            public_keys = {kid: _normalize_pem(pem) for kid, pem in parsed.items()}

        if not public_keys:
            env = os.environ.get("ENVIRONMENT", "development").lower()
            if env == "production":
                raise RuntimeError(
                    "AUTH_ALGORITHM=RS256 requires AUTH_PUBLIC_KEYS_JSON (and, for the issuer, "
                    "AUTH_PRIVATE_KEY_PEM + AUTH_ACTIVE_KID) in production."
                )
            priv, pub, kid = generate_rsa_keypair("dev-ephemeral")
            logger.warning("rs256_dev_ephemeral_keys_generated", kid=kid)
            private_pem, active_kid, public_keys = priv, kid, {kid: pub}

        return cls(active_kid=active_kid, private_pem=private_pem, public_keys=public_keys)
