"""Solace-AI Authentication - JWT validation and token management."""

from __future__ import annotations
import hashlib
import hmac
import secrets
import time
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
    audience: str = Field(default="solace-api")
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
    def hash_password(cls, password: str) -> str:
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


class JWTManager:
    """JWT token creation and validation."""

    def __init__(self, settings: AuthSettings | None = None) -> None:
        self._settings = settings or AuthSettings()
        self._secret = self._settings.secret_key.get_secret_value()

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

    def decode_token(
        self, token: str, expected_type: TokenType | None = None
    ) -> AuthenticationResult:
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
        result = self.validate_refresh_token(refresh_token)
        if not result.success or not result.payload:
            return "", result
        new_access = self.create_access_token(
            result.user_id, roles, permissions, result.payload.session_id
        )
        return new_access, result

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


class SessionManager:
    """Manage user sessions."""

    def __init__(self) -> None:
        self._active_sessions: dict[str, dict[str, Any]] = {}

    def create_session(
        self, user_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        session_id = str(uuid4())
        self._active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "metadata": metadata or {},
        }
        logger.info("session_created", user_id=user_id, session_id=session_id)
        return session_id

    def validate_session(self, session_id: str) -> bool:
        return session_id in self._active_sessions

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self._active_sessions.get(session_id)

    def update_activity(self, session_id: str) -> None:
        if session_id in self._active_sessions:
            self._active_sessions[session_id]["last_activity"] = datetime.now(
                timezone.utc
            )

    def invalidate_session(self, session_id: str) -> bool:
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            logger.info("session_invalidated", session_id=session_id)
            return True
        return False

    def invalidate_user_sessions(self, user_id: str) -> int:
        to_remove = [
            sid
            for sid, data in self._active_sessions.items()
            if data["user_id"] == user_id
        ]
        for sid in to_remove:
            del self._active_sessions[sid]
        if to_remove:
            logger.info(
                "user_sessions_invalidated", user_id=user_id, count=len(to_remove)
            )
        return len(to_remove)

    def get_user_sessions(self, user_id: str) -> list[str]:
        return [
            sid
            for sid, data in self._active_sessions.items()
            if data["user_id"] == user_id
        ]


def create_jwt_manager(settings: AuthSettings | None = None) -> JWTManager:
    """Factory function to create JWT manager."""
    return JWTManager(settings)


def create_session_manager() -> SessionManager:
    """Factory function to create session manager."""
    return SessionManager()
