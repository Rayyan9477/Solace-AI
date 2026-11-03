"""
Solace-AI User Service - JWT Service.

Provides JWT token generation, verification, and validation for user authentication.
Implements secure token-based authentication with expiry and refresh capabilities.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import jwt
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TokenType(str, Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"


class JWTConfig(BaseModel):
    """JWT service configuration.

    P1-1 (RS256 issuer migration): user-service is the token ISSUER. When
    ``algorithm="RS256"`` and RS256 signing material is provisioned
    (``private_key_pem`` + ``active_kid`` + ``public_keys``), user access/refresh
    tokens are signed with the issuer-only RS256 private key (a ``kid`` header is
    stamped) instead of the symmetric HS256 secret that every backend container
    holds. Verification stays DUAL-ACCEPT: legacy HS256 tokens are still accepted
    while ``accept_legacy_hs256`` is True (the default), so a rollout can flip the
    issuer to RS256 before flipping validators to strict. HS256 remains the default
    algorithm when no RS256 material is configured (dev/test, backward compatible).
    """
    secret_key: str = Field(..., description="Secret key for signing tokens")
    algorithm: str = Field(default="HS256", description="JWT signing algorithm (HS256 or RS256)")
    access_token_expire_minutes: int = Field(default=15, description="Access token lifetime")
    refresh_token_expire_days: int = Field(default=30, description="Refresh token lifetime")
    issuer: str = Field(default="solace-ai", description="Token issuer")
    audience: str = Field(default="solace-ai-api", description="Token audience")
    # RS256 signing material (issuer only). private_key_pem + active_kid enable signing;
    # public_keys is the {kid: public_pem} verification keyset (issuer verifies its own kid).
    private_key_pem: str | None = Field(default=None, description="RS256 private key PEM (issuer only)")
    active_kid: str | None = Field(default=None, description="Key id stamped in newly-issued RS256 tokens")
    public_keys: dict[str, str] | None = Field(default=None, description="{kid: public_pem} verification keyset")
    # Backward-compat DUAL-ACCEPT: keep accepting legacy HS256 user tokens during the
    # RS256 rollout. Strict rejection (False) is an opt-in OPS/CONFIG flip, not a code default.
    accept_legacy_hs256: bool = Field(default=True, description="Accept legacy HS256 user tokens during the RS256 transition")


@dataclass
class TokenPayload:
    """JWT token payload data."""
    user_id: UUID
    email: str
    role: str
    token_type: TokenType
    jti: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to dictionary."""
        d: dict[str, Any] = {
            "user_id": str(self.user_id),
            "email": self.email,
            "role": self.role,
            "type": self.token_type.value,
        }
        if self.jti:
            d["jti"] = self.jti
        return d


@dataclass
class TokenPair:
    """Access and refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 900  # 15 minutes in seconds


class JWTError(Exception):
    """Base exception for JWT errors."""
    pass


class TokenExpiredError(JWTError):
    """Raised when token has expired."""
    pass


class TokenInvalidError(JWTError):
    """Raised when token is invalid."""
    pass


class TokenBlacklist(ABC):
    """Interface for token revocation blacklisting."""

    @abstractmethod
    async def add(self, jti: str, expires_at: datetime) -> None:
        """Add a token JTI to the blacklist until its natural expiry."""

    @abstractmethod
    async def is_blacklisted(self, jti: str) -> bool:
        """Check if a token JTI is in the blacklist."""


class InMemoryTokenBlacklist(TokenBlacklist):
    """In-memory token blacklist for development/testing."""

    def __init__(self) -> None:
        self._blacklist: dict[str, datetime] = {}

    async def add(self, jti: str, expires_at: datetime) -> None:
        self._blacklist[jti] = expires_at

    async def is_blacklisted(self, jti: str) -> bool:
        if jti in self._blacklist:
            if self._blacklist[jti] > datetime.now(timezone.utc):
                return True
            del self._blacklist[jti]
        return False


class RedisTokenBlacklist(TokenBlacklist):
    """Redis-backed token blacklist for production."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def add(self, jti: str, expires_at: datetime) -> None:
        ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds())
        if ttl > 0:
            await self._redis.setex(f"token_blacklist:{jti}", ttl, "1")

    async def is_blacklisted(self, jti: str) -> bool:
        result = await self._redis.get(f"token_blacklist:{jti}")
        return result is not None


class JWTService:
    """
    JWT Service for token generation and verification.

    Provides secure token-based authentication with:
    - Access tokens (short-lived)
    - Refresh tokens (long-lived)
    - Token validation and verification
    - Payload extraction
    """

    def __init__(
        self,
        config: JWTConfig,
        blacklist: TokenBlacklist | None = None,
        key_manager: Any | None = None,
    ):
        """
        Initialize JWT service.

        Args:
            config: JWT configuration
            blacklist: Optional token blacklist for revocation support
            key_manager: Optional RS256 KeyManager (duck-typed: needs ``can_sign``,
                ``active_kid``, ``sign(claims)`` and ``public_pem_for(kid)``). When
                omitted and ``config.algorithm == "RS256"``, one is built from the
                config's RS256 material.
        """
        self.config = config
        self._blacklist = blacklist
        self.logger = structlog.get_logger(__name__)
        self._keys = key_manager
        if self._keys is None and config.algorithm == "RS256":
            # Build the issuer/verifier keyset from config material. Fail-soft so a
            # misconfigured RS256 setup surfaces at sign/verify time, not import time.
            try:
                from solace_security.keys import KeyManager

                self._keys = KeyManager(
                    active_kid=config.active_kid,
                    private_pem=config.private_key_pem,
                    public_keys=config.public_keys or {},
                )
            except Exception as e:  # pragma: no cover - defensive
                self.logger.warning("rs256_key_manager_init_failed", error=str(e))
                self._keys = None

    def _decode_claims(self, token: str) -> dict[str, Any]:
        """Algorithm-aware signature verification returning the raw claims.

        RS256 tokens are verified against the public keyset by ``kid``; HS256 tokens
        are verified with the shared secret. In RS256 mode, HS256 tokens are rejected
        unless ``accept_legacy_hs256`` is set (DUAL-ACCEPT during rollout). The
        single-algorithm allowlist per branch blocks RS256/HS256 confusion attacks.
        """
        header = jwt.get_unverified_header(token)
        alg = header.get("alg")
        if alg == "RS256":
            public_pem = self._keys.public_pem_for(header.get("kid")) if self._keys else None
            if not public_pem:
                raise TokenInvalidError("Unknown signing key id")
            return jwt.decode(
                token,
                public_pem,
                algorithms=["RS256"],
                audience=self.config.audience,
                issuer=self.config.issuer,
            )
        if alg == "HS256":
            if self.config.algorithm == "RS256" and not self.config.accept_legacy_hs256:
                raise TokenInvalidError("HS256 tokens are no longer accepted")
            return jwt.decode(
                token,
                self.config.secret_key,
                algorithms=["HS256"],
                audience=self.config.audience,
                issuer=self.config.issuer,
            )
        raise TokenInvalidError(f"Unsupported algorithm: {alg}")

    def generate_token_pair(
        self,
        user_id: UUID,
        email: str,
        role: str,
    ) -> TokenPair:
        """
        Generate access and refresh token pair.

        Args:
            user_id: User ID
            email: User email
            role: User role

        Returns:
            TokenPair with access and refresh tokens
        """
        access_token = self._generate_token(
            payload=TokenPayload(
                user_id=user_id,
                email=email,
                role=role,
                token_type=TokenType.ACCESS,
            ),
            expires_delta=timedelta(minutes=self.config.access_token_expire_minutes),
        )

        refresh_token = self._generate_token(
            payload=TokenPayload(
                user_id=user_id,
                email=email,
                role=role,
                token_type=TokenType.REFRESH,
            ),
            expires_delta=timedelta(days=self.config.refresh_token_expire_days),
        )

        self.logger.info(
            "generated_token_pair",
            user_id=str(user_id),
            email=email,
            role=role,
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_expire_minutes * 60,
        )

    async def verify_token(self, token: str, expected_type: TokenType = TokenType.ACCESS) -> TokenPayload:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string
            expected_type: Expected token type

        Returns:
            TokenPayload with user information

        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid or revoked
        """
        try:
            payload = self._decode_claims(token)

            # Check revocation blacklist
            jti = payload.get("jti")
            if jti and self._blacklist:
                if await self._blacklist.is_blacklisted(jti):
                    raise TokenInvalidError("Token has been revoked")

            # Validate token type
            token_type = TokenType(payload.get("type"))
            if token_type != expected_type:
                raise TokenInvalidError(f"Expected {expected_type.value} token, got {token_type.value}")

            # Extract payload
            token_payload = TokenPayload(
                user_id=UUID(payload["user_id"]),
                email=payload["email"],
                role=payload["role"],
                token_type=token_type,
                jti=jti,
            )

            self.logger.debug(
                "token_verified",
                user_id=payload["user_id"],
                token_type=token_type.value,
            )

            return token_payload

        except jwt.ExpiredSignatureError as e:
            self.logger.warning("token_expired", error=str(e))
            raise TokenExpiredError("Token has expired") from e
        except jwt.InvalidTokenError as e:
            self.logger.warning("token_invalid", error=str(e))
            raise TokenInvalidError("Invalid token") from e
        except (KeyError, ValueError) as e:
            self.logger.warning("token_malformed", error=str(e))
            raise TokenInvalidError("Malformed token payload") from e

    async def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding its JTI to the blacklist.

        Args:
            token: JWT token string to revoke

        Returns:
            True if token was revoked, False if blacklist unavailable
        """
        if not self._blacklist:
            self.logger.warning("token_revocation_unavailable", hint="No blacklist configured")
            return False

        try:
            payload = self._decode_claims(token)
            jti = payload.get("jti")
            if not jti:
                self.logger.warning("token_has_no_jti")
                return False

            expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            await self._blacklist.add(jti, expires_at)
            self.logger.info("token_revoked", jti=jti, user_id=payload.get("user_id"))
            return True

        except jwt.ExpiredSignatureError:
            return True  # Already expired, no need to revoke
        except (jwt.InvalidTokenError, TokenInvalidError) as e:
            self.logger.warning("revoke_invalid_token", error=str(e))
            return False

    async def refresh_access_token(self, refresh_token: str) -> str:
        """
        Generate new access token from refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New access token

        Raises:
            TokenExpiredError: If refresh token has expired
            TokenInvalidError: If refresh token is invalid
        """
        # Verify refresh token
        payload = await self.verify_token(refresh_token, expected_type=TokenType.REFRESH)

        # Generate new access token
        access_token = self._generate_token(
            payload=TokenPayload(
                user_id=payload.user_id,
                email=payload.email,
                role=payload.role,
                token_type=TokenType.ACCESS,
            ),
            expires_delta=timedelta(minutes=self.config.access_token_expire_minutes),
        )

        self.logger.info(
            "access_token_refreshed",
            user_id=str(payload.user_id),
        )

        return access_token

    def decode_token_unsafe(self, token: str) -> dict[str, Any] | None:
        """
        Decode token without verification (for debugging/inspection).

        Args:
            token: JWT token string

        Returns:
            Decoded payload or None if invalid
        """
        try:
            return jwt.decode(
                token,
                options={"verify_signature": False},
                algorithms=[self.config.algorithm],
            )
        except Exception as e:
            self.logger.warning("token_decode_failed", error=str(e))
            return None

    def _generate_token(
        self,
        payload: TokenPayload,
        expires_delta: timedelta,
    ) -> str:
        """
        Generate JWT token with payload and expiry.

        Args:
            payload: Token payload data
            expires_delta: Token expiration time delta

        Returns:
            Encoded JWT token string
        """
        now = datetime.now(timezone.utc)
        expires_at = now + expires_delta
        jti = str(uuid4())

        jwt_payload = {
            **payload.to_dict(),
            # UNION claim schema (P1-1): user-service's own verifier reads
            # user_id/email/role; the shared solace_security validators (used by every
            # OTHER service via get_current_user) read sub/roles. Emitting both lets the
            # SAME token verify on both without either side breaking.
            "sub": str(payload.user_id),
            "roles": [payload.role],
            "jti": jti,
            "exp": expires_at,
            "iat": now,
            "nbf": now,
            "iss": self.config.issuer,
            "aud": self.config.audience,
        }

        # RS256 issuer path: sign with the issuer-only private key (stamps kid header).
        if self.config.algorithm == "RS256":
            if not (self._keys and self._keys.can_sign):
                raise RuntimeError(
                    "JWTConfig.algorithm=RS256 but no RS256 signing key is configured "
                    "(private_key_pem + active_kid). This issuer cannot mint user tokens."
                )
            return self._keys.sign(jwt_payload)

        # Legacy/default HS256 path (unchanged).
        return jwt.encode(
            jwt_payload,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    @staticmethod
    def extract_token_from_header(authorization: str) -> str:
        """
        Extract JWT token from Authorization header.

        Args:
            authorization: Authorization header value (e.g., "Bearer <token>")

        Returns:
            Extracted token

        Raises:
            TokenInvalidError: If header format is invalid
        """
        if not authorization:
            raise TokenInvalidError("Missing authorization header")

        parts = authorization.split()

        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise TokenInvalidError("Invalid authorization header format")

        return parts[1]


def create_jwt_service(
    secret_key: str,
    algorithm: str = "HS256",
    access_token_expire_minutes: int = 15,
    refresh_token_expire_days: int = 30,
) -> JWTService:
    """
    Factory function to create JWT service.

    Args:
        secret_key: Secret key for signing tokens
        algorithm: JWT signing algorithm
        access_token_expire_minutes: Access token lifetime
        refresh_token_expire_days: Refresh token lifetime

    Returns:
        Configured JWTService instance
    """
    config = JWTConfig(
        secret_key=secret_key,
        algorithm=algorithm,
        access_token_expire_minutes=access_token_expire_minutes,
        refresh_token_expire_days=refresh_token_expire_days,
    )

    return JWTService(config)
