"""Solace-AI Authentication - JWT validation and token management."""

from __future__ import annotations
import hashlib
import hmac
import os
import secrets
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
import jwt
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class TokenType(str, Enum):
    """Token types supported by the system."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    SERVICE = "service"


class AuthSettings(BaseSettings):
    """Authentication configuration from environment.

    SECURITY: secret_key MUST be set via AUTH_SECRET_KEY environment variable.
    The key must be at least 32 bytes for HS256 algorithm security.
    """

    secret_key: SecretStr = Field(
        ...,  # Required - no default for security
        description="JWT signing key (min 32 bytes). Set via AUTH_SECRET_KEY env var.",
        min_length=32,
    )
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)
    api_key_expire_days: int = Field(default=365)
    issuer: str = Field(default="solace-ai")
    audience: str = Field(default="solace-ai-api")
    clock_skew_seconds: int = Field(default=30)
    min_password_length: int = Field(default=12)
    max_failed_attempts: int = Field(default=5)
    lockout_duration_minutes: int = Field(default=15)
    model_config = SettingsConfigDict(
        env_prefix="AUTH_", env_file=".env", extra="ignore"
    )

    @classmethod
    def for_development(cls) -> "AuthSettings":
        """Create settings with a development-only key. NOT FOR PRODUCTION."""
        import warnings

        env = os.environ.get("ENVIRONMENT", "development").lower()
        if env == "production":
            raise RuntimeError(
                "AuthSettings.for_development() cannot be used in production. "
                "Set AUTH_SECRET_KEY environment variable with a secure key."
            )
        warnings.warn(
            "Using development AuthSettings with insecure key. NOT FOR PRODUCTION USE.",
            UserWarning,
            stacklevel=2,
        )
        return cls(secret_key=SecretStr("dev-only-insecure-key-32-bytes!!"))

    def model_post_init(self, __context: Any) -> None:
        """Validate secret key after initialization."""
        key_value = self.secret_key.get_secret_value()
        if len(key_value) < 32:
            raise ValueError(
                f"AUTH_SECRET_KEY must be at least 32 bytes, got {len(key_value)}. "
                'Generate a secure key with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )


class TokenPayload(BaseModel):
    """JWT token payload structure."""

    sub: str = Field(..., description="Subject (user ID)")
    jti: str = Field(default_factory=lambda: str(uuid4()), description="JWT ID")
    type: TokenType = Field(..., description="Token type")
    iat: int = Field(default_factory=lambda: int(time.time()), description="Issued at")
    exp: int = Field(..., description="Expiration time")
    iss: str = Field(default="solace-ai", description="Issuer")
    aud: str = Field(default="solace-api", description="Audience")
    roles: list[str] = Field(default_factory=list, description="User roles")
    permissions: list[str] = Field(default_factory=list, description="User permissions")
    session_id: str | None = Field(default=None, description="Session ID")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional claims"
    )

    @property
    def is_expired(self) -> bool:
        return time.time() > self.exp

    @property
    def expires_at(self) -> datetime:
        return datetime.fromtimestamp(self.exp, tz=timezone.utc)

    @property
    def issued_at(self) -> datetime:
        return datetime.fromtimestamp(self.iat, tz=timezone.utc)


class TokenPair(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_expires_in: int


class AuthenticationResult(BaseModel):
    """Result of authentication operation."""

    success: bool
    user_id: str | None = None
    payload: TokenPayload | None = None
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def ok(cls, user_id: str, payload: TokenPayload) -> AuthenticationResult:
        return cls(success=True, user_id=user_id, payload=payload)

    @classmethod
    def fail(cls, code: str, message: str) -> AuthenticationResult:
        return cls(success=False, error_code=code, error_message=message)


class PasswordHasher:
    """Secure password hashing using PBKDF2."""

    _iterations: int = 600000
    _hash_name: str = "sha256"
    _salt_length: int = 32

    @classmethod
    def hash_password(cls, password: str, min_length: int = 12) -> str:
        """Hash a password using PBKDF2.

        Raises:
            ValueError: If password is shorter than min_length.
        """
        if len(password) < min_length:
            raise ValueError(
                f"Password must be at least {min_length} characters, got {len(password)}"
            )
        salt = secrets.token_bytes(cls._salt_length)
        dk = hashlib.pbkdf2_hmac(
            cls._hash_name, password.encode(), salt, cls._iterations
        )
        return f"${cls._hash_name}${cls._iterations}${salt.hex()}${dk.hex()}"

    @classmethod
    def verify_password(cls, password: str, stored_hash: str) -> bool:
        try:
            parts = stored_hash.split("$")
            if len(parts) != 5:
                return False
            _, hash_name, iterations, salt_hex, expected_hex = parts
            salt = bytes.fromhex(salt_hex)
            dk = hashlib.pbkdf2_hmac(
                hash_name, password.encode(), salt, int(iterations)
            )
            return hmac.compare_digest(dk.hex(), expected_hex)
        except (ValueError, TypeError):
            return False

    @classmethod
    def needs_rehash(cls, stored_hash: str) -> bool:
        try:
            parts = stored_hash.split("$")
            if len(parts) != 5:
                return True
            _, hash_name, iterations, _, _ = parts
            return hash_name != cls._hash_name or int(iterations) < cls._iterations
        except (ValueError, TypeError):
            return True


class TokenBlacklist(ABC):
    """Abstract interface for token revocation/blacklisting."""

    @abstractmethod
    async def add(self, jti: str, expires_at: datetime) -> None:
        """Add a token JTI to the blacklist until its natural expiry."""

    @abstractmethod
    async def is_blacklisted(self, jti: str) -> bool:
        """Check if a token JTI has been revoked."""


class InMemoryTokenBlacklist(TokenBlacklist):
    """In-memory token blacklist for testing/development."""

    def __init__(self) -> None:
        self._blacklist: dict[str, datetime] = {}

    async def add(self, jti: str, expires_at: datetime) -> None:
        self._blacklist[jti] = expires_at

    async def is_blacklisted(self, jti: str) -> bool:
        if jti in self._blacklist:
            # Auto-clean expired entries
            if self._blacklist[jti] < datetime.now(timezone.utc):
                del self._blacklist[jti]
                return False
            return True
        return False


class LoginAttemptTracker(ABC):
    """Abstract interface for tracking failed login attempts."""

    @abstractmethod
    async def record_failure(self, user_id: str) -> int:
        """Record a failed login attempt. Returns total failed count."""

    @abstractmethod
    async def is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out due to failed attempts."""

    @abstractmethod
    async def reset(self, user_id: str) -> None:
        """Reset failed attempts on successful login."""


class InMemoryLoginAttemptTracker(LoginAttemptTracker):
    """In-memory login attempt tracker for testing/development."""

    def __init__(self, max_attempts: int = 5, lockout_minutes: int = 15) -> None:
        self._max_attempts = max_attempts
        self._lockout_minutes = lockout_minutes
        self._attempts: dict[str, list[datetime]] = {}

    async def record_failure(self, user_id: str) -> int:
        now = datetime.now(timezone.utc)
        if user_id not in self._attempts:
            self._attempts[user_id] = []
        self._attempts[user_id].append(now)
        # Only count recent attempts within lockout window
        cutoff = now - timedelta(minutes=self._lockout_minutes)
        self._attempts[user_id] = [t for t in self._attempts[user_id] if t > cutoff]
        return len(self._attempts[user_id])

    async def is_locked_out(self, user_id: str) -> bool:
        if user_id not in self._attempts:
            return False
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self._lockout_minutes)
        recent = [t for t in self._attempts[user_id] if t > cutoff]
        return len(recent) >= self._max_attempts

    async def reset(self, user_id: str) -> None:
        self._attempts.pop(user_id, None)


class RedisTokenBlacklist(TokenBlacklist):
    """Redis-backed token blacklist for production use."""

    def __init__(self, redis_client: Any, key_prefix: str = "token_blacklist:") -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    async def add(self, jti: str, expires_at: datetime) -> None:
        key = f"{self._prefix}{jti}"
        ttl_seconds = max(1, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
        await self._redis.set(key, "1", ttl=ttl_seconds)

    async def is_blacklisted(self, jti: str) -> bool:
        key = f"{self._prefix}{jti}"
        result = await self._redis.get(key)
        return result is not None


class RedisLoginAttemptTracker(LoginAttemptTracker):
    """Redis-backed login attempt tracker for production use."""

    def __init__(
        self,
        redis_client: Any,
        max_attempts: int = 5,
        lockout_minutes: int = 15,
        key_prefix: str = "login_attempts:",
    ) -> None:
        self._redis = redis_client
        self._max_attempts = max_attempts
        self._lockout_minutes = lockout_minutes
        self._prefix = key_prefix

    async def record_failure(self, user_id: str) -> int:
        key = f"{self._prefix}{user_id}"
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, timedelta(minutes=self._lockout_minutes))
        return count

    async def is_locked_out(self, user_id: str) -> bool:
        key = f"{self._prefix}{user_id}"
        count = await self._redis.get(key)
        if count is None:
            return False
        return int(count) >= self._max_attempts

    async def reset(self, user_id: str) -> None:
        key = f"{self._prefix}{user_id}"
        await self._redis.delete(key)


class JWTManager:
    """JWT token creation and validation with optional token revocation."""

    def __init__(
        self,
        settings: AuthSettings | None = None,
        token_blacklist: TokenBlacklist | None = None,
    ) -> None:
        self._settings = settings or AuthSettings()
        self._secret = self._settings.secret_key.get_secret_value()
        if token_blacklist is None:
            raise ValueError("token_blacklist is required for JWTManager. Pass a TokenBlacklist implementation.")
        self._blacklist = token_blacklist

    def create_access_token(
        self,
        user_id: str,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = int(time.time())
        exp = now + (self._settings.access_token_expire_minutes * 60)
        payload = TokenPayload(
            sub=user_id,
            type=TokenType.ACCESS,
            iat=now,
            exp=exp,
            iss=self._settings.issuer,
            aud=self._settings.audience,
            roles=roles or [],
            permissions=permissions or [],
            session_id=session_id,
            metadata=metadata or {},
        )
        return self._encode(payload)

    def create_refresh_token(self, user_id: str, session_id: str | None = None) -> str:
        now = int(time.time())
        exp = now + (self._settings.refresh_token_expire_days * 86400)
        payload = TokenPayload(
            sub=user_id,
            type=TokenType.REFRESH,
            iat=now,
            exp=exp,
            iss=self._settings.issuer,
            aud=self._settings.audience,
            session_id=session_id,
        )
        return self._encode(payload)

    def create_token_pair(
        self,
        user_id: str,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
        session_id: str | None = None,
    ) -> TokenPair:
        access = self.create_access_token(user_id, roles, permissions, session_id)
        refresh = self.create_refresh_token(user_id, session_id)
        return TokenPair(
            access_token=access,
            refresh_token=refresh,
            expires_in=self._settings.access_token_expire_minutes * 60,
            refresh_expires_in=self._settings.refresh_token_expire_days * 86400,
        )

    def create_api_key(
        self, user_id: str, name: str, permissions: list[str] | None = None
    ) -> str:
        now = int(time.time())
        exp = now + (self._settings.api_key_expire_days * 86400)
        payload = TokenPayload(
            sub=user_id,
            type=TokenType.API_KEY,
            iat=now,
            exp=exp,
            iss=self._settings.issuer,
            aud=self._settings.audience,
            permissions=permissions or [],
            metadata={"api_key_name": name},
        )
        return self._encode(payload)

    def create_service_token(
        self,
        service_name: str,
        permissions: list[str] | None = None,
        expire_minutes: int = 60,
    ) -> str:
        now = int(time.time())
        exp = now + (expire_minutes * 60)
        payload = TokenPayload(
            sub=f"service:{service_name}",
            type=TokenType.SERVICE,
            iat=now,
            exp=exp,
            iss=self._settings.issuer,
            aud=self._settings.audience,
            permissions=permissions or [],
            metadata={"service_name": service_name},
        )
        return self._encode(payload)

    async def revoke_token(self, token: str) -> bool:
        """Revoke a token by adding its JTI to the blacklist.

        Returns True if successfully revoked, False if no blacklist configured.
        """
        if not self._blacklist:
            logger.warning("token_revocation_unavailable", reason="no blacklist configured")
            return False
        result = self.decode_token_sync(token)
        if result.success and result.payload:
            await self._blacklist.add(result.payload.jti, result.payload.expires_at)
            logger.info("token_revoked", jti=result.payload.jti, user_id=result.user_id)
            return True
        return False

    async def is_token_revoked(self, jti: str) -> bool:
        """Check if a token JTI has been revoked."""
        if not self._blacklist:
            return False
        return await self._blacklist.is_blacklisted(jti)

    def decode_token_sync(
        self, token: str, expected_type: TokenType | None = None
    ) -> AuthenticationResult:
        """Decode and validate a JWT token (synchronous, no revocation check)."""
        try:
            decoded = jwt.decode(
                token,
                self._secret,
                algorithms=[self._settings.algorithm],
                issuer=self._settings.issuer,
                audience=self._settings.audience,
                leeway=self._settings.clock_skew_seconds,
            )
            payload = TokenPayload(**decoded)
            if expected_type and payload.type != expected_type:
                return AuthenticationResult.fail(
                    "INVALID_TOKEN_TYPE",
                    f"Expected {expected_type.value} token, got {payload.type.value}",
                )
            if payload.is_expired:
                return AuthenticationResult.fail("TOKEN_EXPIRED", "Token has expired")
            return AuthenticationResult.ok(payload.sub, payload)
        except jwt.ExpiredSignatureError:
            return AuthenticationResult.fail("TOKEN_EXPIRED", "Token has expired")
        except jwt.InvalidIssuerError:
            return AuthenticationResult.fail("INVALID_ISSUER", "Invalid token issuer")
        except jwt.InvalidAudienceError:
            return AuthenticationResult.fail(
                "INVALID_AUDIENCE", "Invalid token audience"
            )
        except jwt.DecodeError as e:
            return AuthenticationResult.fail(
                "DECODE_ERROR", f"Failed to decode token: {e}"
            )
        except jwt.InvalidTokenError as e:
            return AuthenticationResult.fail("INVALID_TOKEN", f"Invalid token: {e}")

    def decode_token(
        self, token: str, expected_type: TokenType | None = None
    ) -> AuthenticationResult:
        """Decode and validate a JWT token (synchronous, backward compatible).

        Note: For revocation checking, use decode_token_async() instead.
        """
        return self.decode_token_sync(token, expected_type)

    async def decode_token_async(
        self, token: str, expected_type: TokenType | None = None
    ) -> AuthenticationResult:
        """Decode and validate a JWT token with revocation check."""
        result = self.decode_token_sync(token, expected_type)
        if not result.success or not result.payload:
            return result
        # Check revocation blacklist
        if await self.is_token_revoked(result.payload.jti):
            logger.warning("revoked_token_used", jti=result.payload.jti, user_id=result.user_id)
            return AuthenticationResult.fail("TOKEN_REVOKED", "Token has been revoked")
        return result

    def validate_access_token(self, token: str) -> AuthenticationResult:
        return self.decode_token(token, TokenType.ACCESS)

    def validate_refresh_token(self, token: str) -> AuthenticationResult:
        return self.decode_token(token, TokenType.REFRESH)

    def refresh_access_token(
        self,
        refresh_token: str,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> tuple[str, AuthenticationResult]:
        """Refresh access token (synchronous, no rotation)."""
        result = self.validate_refresh_token(refresh_token)
        if not result.success or not result.payload:
            return "", result
        new_access = self.create_access_token(
            result.user_id, roles, permissions, result.payload.session_id
        )
        return new_access, result

    async def refresh_token_pair(
        self,
        refresh_token: str,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> tuple[TokenPair | None, AuthenticationResult]:
        """Refresh with token rotation: issues new access + refresh tokens, revokes old refresh.

        Returns a new TokenPair and the validation result, or (None, error_result).
        """
        result = await self.decode_token_async(refresh_token, TokenType.REFRESH)
        if not result.success or not result.payload:
            return None, result
        # Revoke the old refresh token to prevent reuse
        await self.revoke_token(refresh_token)
        # Issue new pair
        new_pair = self.create_token_pair(
            result.user_id, roles, permissions, result.payload.session_id
        )
        logger.info("token_pair_rotated", user_id=result.user_id)
        return new_pair, result

    def _encode(self, payload: TokenPayload) -> str:
        return jwt.encode(
            payload.model_dump(), self._secret, algorithm=self._settings.algorithm
        )


class APIKeyGenerator:
    """Generate and validate API keys."""

    _prefix: str = "sk_"
    _key_length: int = 32

    @classmethod
    def generate(cls) -> tuple[str, str]:
        key_bytes = secrets.token_bytes(cls._key_length)
        raw_key = cls._prefix + key_bytes.hex()
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        return raw_key, key_hash

    @classmethod
    def hash_key(cls, api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    @classmethod
    def validate_format(cls, api_key: str) -> bool:
        return (
            api_key.startswith(cls._prefix)
            and len(api_key) == len(cls._prefix) + cls._key_length * 2
        )


class SessionStore(ABC):
    """Abstract interface for session persistence.

    Implementations can back sessions with Redis, PostgreSQL, or in-memory storage.
    """

    @abstractmethod
    async def create(self, session_id: str, user_id: str, metadata: dict[str, Any]) -> None:
        """Persist a new session."""

    @abstractmethod
    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve session data by ID. Returns None if not found."""

    @abstractmethod
    async def update_activity(self, session_id: str) -> None:
        """Update last activity timestamp for a session."""

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""

    @abstractmethod
    async def delete_user_sessions(self, user_id: str) -> int:
        """Delete all sessions for a user. Returns count deleted."""

    @abstractmethod
    async def get_user_sessions(self, user_id: str) -> list[str]:
        """Get all session IDs for a user."""


class InMemorySessionStore(SessionStore):
    """In-memory session store for testing/development."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    async def create(self, session_id: str, user_id: str, metadata: dict[str, Any]) -> None:
        self._sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "metadata": metadata,
        }

    async def get(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    async def update_activity(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["last_activity"] = datetime.now(timezone.utc)

    async def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def delete_user_sessions(self, user_id: str) -> int:
        to_remove = [
            sid for sid, data in self._sessions.items()
            if data["user_id"] == user_id
        ]
        for sid in to_remove:
            del self._sessions[sid]
        return len(to_remove)

    async def get_user_sessions(self, user_id: str) -> list[str]:
        return [
            sid for sid, data in self._sessions.items()
            if data["user_id"] == user_id
        ]


class SessionManager:
    """Manage user sessions via pluggable SessionStore backend."""

    def __init__(self, store: SessionStore | None = None) -> None:
        self._store = store or InMemorySessionStore()

    async def create_session(
        self, user_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        session_id = str(uuid4())
        await self._store.create(session_id, user_id, metadata or {})
        logger.info("session_created", user_id=user_id, session_id=session_id)
        return session_id

    async def validate_session(self, session_id: str) -> bool:
        return (await self._store.get(session_id)) is not None

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await self._store.get(session_id)

    async def update_activity(self, session_id: str) -> None:
        await self._store.update_activity(session_id)

    async def invalidate_session(self, session_id: str) -> bool:
        result = await self._store.delete(session_id)
        if result:
            logger.info("session_invalidated", session_id=session_id)
        return result

    async def invalidate_user_sessions(self, user_id: str) -> int:
        count = await self._store.delete_user_sessions(user_id)
        if count:
            logger.info("user_sessions_invalidated", user_id=user_id, count=count)
        return count

    async def get_user_sessions(self, user_id: str) -> list[str]:
        return await self._store.get_user_sessions(user_id)


def create_jwt_manager(
    settings: AuthSettings | None = None,
    token_blacklist: TokenBlacklist | None = None,
) -> JWTManager:
    """Factory function to create JWT manager."""
    if token_blacklist is None:
        token_blacklist = InMemoryTokenBlacklist()
    return JWTManager(settings, token_blacklist=token_blacklist)


def create_session_manager() -> SessionManager:
    """Factory function to create session manager."""
    return SessionManager()


def validate_auth_settings(settings: AuthSettings) -> None:
    """Validate authentication settings at startup.

    Raises ValueError if settings are insecure for the current environment.
    Call this during service lifespan to fail fast on misconfiguration.
    """
    key_value = settings.secret_key.get_secret_value()
    env = os.environ.get("ENVIRONMENT", "development").lower()

    if len(key_value) < 32:
        raise ValueError(
            f"AUTH_SECRET_KEY must be at least 32 bytes, got {len(key_value)}")

    if settings.algorithm not in ("HS256", "HS384", "HS512"):
        raise ValueError(
            f"Unsupported JWT algorithm: {settings.algorithm}. "
            "Use HS256, HS384, or HS512.")

    if settings.access_token_expire_minutes < 5:
        raise ValueError("Access token TTL must be at least 5 minutes")

    if settings.access_token_expire_minutes > 1440:
        raise ValueError("Access token TTL must not exceed 24 hours (1440 minutes)")

    if settings.refresh_token_expire_days > 30:
        raise ValueError("Refresh token TTL must not exceed 30 days")

    if env == "production":
        if key_value.startswith("dev-") or "insecure" in key_value.lower():
            raise ValueError(
                "Production environment detected with development/insecure key. "
                "Set AUTH_SECRET_KEY to a cryptographically random value.")
        if len(key_value) < 64:
            logger.warning("auth_key_length_advisory",
                           length=len(key_value),
                           recommendation="Use at least 64 bytes for production")

    logger.info("auth_settings_validated",
                algorithm=settings.algorithm,
                access_ttl_minutes=settings.access_token_expire_minutes,
                refresh_ttl_days=settings.refresh_token_expire_days)
