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
    """JWT service configuration."""
    secret_key: str = Field(..., description="Secret key for signing tokens")
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(default=15, description="Access token lifetime")
    refresh_token_expire_days: int = Field(default=30, description="Refresh token lifetime")
    issuer: str = Field(default="solace-ai", description="Token issuer")
    audience: str = Field(default="solace-ai-api", description="Token audience")


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

    def __init__(self, config: JWTConfig, blacklist: TokenBlacklist | None = None):
        """
        Initialize JWT service.

        Args:
            config: JWT configuration
            blacklist: Optional token blacklist for revocation support
        """
        self.config = config
        self._blacklist = blacklist
        self.logger = structlog.get_logger(__name__)

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
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
            )

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
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
            )
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
        except jwt.InvalidTokenError as e:
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
            "jti": jti,
            "exp": expires_at,
            "iat": now,
            "nbf": now,
            "iss": self.config.issuer,
            "aud": self.config.audience,
        }

        token = jwt.encode(
            jwt_payload,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

        return token

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
