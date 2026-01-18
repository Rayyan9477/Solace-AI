"""
Solace-AI User Service - Authentication and Session Management.
Handles user authentication, session lifecycle, and token management.
"""
from __future__ import annotations
import asyncio
import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from .domain.service import User, UserSession

logger = structlog.get_logger(__name__)


class TokenType(str, Enum):
    """Token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFICATION = "email_verification"


class SessionStatus(str, Enum):
    """Session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


class SessionConfig(BaseSettings):
    """Session management configuration."""
    secret_key: str = Field(default="your-secret-key-change-in-production")
    access_token_expire_minutes: int = Field(default=30, ge=1)
    refresh_token_expire_days: int = Field(default=7, ge=1)
    session_timeout_minutes: int = Field(default=60, ge=1)
    max_sessions_per_user: int = Field(default=5, ge=1)
    enable_session_tracking: bool = Field(default=True)
    algorithm: str = Field(default="HS256")
    issuer: str = Field(default="solace-ai-user-service")
    audience: str = Field(default="solace-ai")
    model_config = SettingsConfigDict(env_prefix="SESSION_", env_file=".env", extra="ignore")


class TokenClaims(BaseModel):
    """JWT token claims."""
    sub: str = Field(..., description="Subject (user ID)")
    exp: datetime = Field(..., description="Expiration time")
    iat: datetime = Field(..., description="Issued at")
    jti: str = Field(..., description="JWT ID")
    token_type: TokenType = Field(default=TokenType.ACCESS)
    iss: str = Field(default="solace-ai-user-service")
    aud: str = Field(default="solace-ai")
    email: str | None = None
    roles: list[str] = Field(default_factory=lambda: ["user"])
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JWT payload."""
        return {
            "sub": self.sub,
            "exp": int(self.exp.timestamp()),
            "iat": int(self.iat.timestamp()),
            "jti": self.jti,
            "token_type": self.token_type.value,
            "iss": self.iss,
            "aud": self.aud,
            "email": self.email,
            "roles": self.roles,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenClaims:
        """Create from dictionary."""
        return cls(
            sub=data["sub"],
            exp=datetime.fromtimestamp(data["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(data["iat"], tz=timezone.utc),
            jti=data["jti"],
            token_type=TokenType(data.get("token_type", "access")),
            iss=data.get("iss", ""),
            aud=data.get("aud", ""),
            email=data.get("email"),
            roles=data.get("roles", ["user"]),
            session_id=data.get("session_id"),
        )

    def is_expired(self, leeway_seconds: int = 0) -> bool:
        """Check if token is expired."""
        return datetime.now(timezone.utc) > self.exp + timedelta(seconds=leeway_seconds)


@dataclass
class AuthResult:
    """Authentication result."""
    success: bool = False
    user: User | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    session: UserSession | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class TokenValidationResult:
    """Token validation result."""
    valid: bool = False
    claims: TokenClaims | None = None
    error: str | None = None
    error_code: str | None = None


class TokenCodec:
    """JWT encoding and decoding utilities."""

    @staticmethod
    def base64url_encode(data: bytes) -> str:
        """Base64 URL-safe encode."""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    @staticmethod
    def base64url_decode(data: str) -> bytes:
        """Base64 URL-safe decode."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    @staticmethod
    def get_hash_func(algorithm: str):
        """Get hash function for algorithm."""
        algorithms = {
            "HS256": hashlib.sha256,
            "HS384": hashlib.sha384,
            "HS512": hashlib.sha512,
        }
        if algorithm not in algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        return algorithms[algorithm]


class TokenService:
    """Token creation and validation service."""

    def __init__(self, config: SessionConfig) -> None:
        self._config = config
        self._revoked_tokens: set[str] = set()

    def create_access_token(self, user: User, session_id: UUID) -> str:
        """Create access token."""
        now = datetime.now(timezone.utc)
        claims = TokenClaims(
            sub=str(user.user_id),
            exp=now + timedelta(minutes=self._config.access_token_expire_minutes),
            iat=now,
            jti=secrets.token_hex(16),
            token_type=TokenType.ACCESS,
            iss=self._config.issuer,
            aud=self._config.audience,
            email=user.email,
            roles=[user.role.value],
            session_id=str(session_id),
        )
        return self._encode_token(claims)

    def create_refresh_token(self, user: User, session_id: UUID) -> str:
        """Create refresh token."""
        now = datetime.now(timezone.utc)
        claims = TokenClaims(
            sub=str(user.user_id),
            exp=now + timedelta(days=self._config.refresh_token_expire_days),
            iat=now,
            jti=secrets.token_hex(16),
            token_type=TokenType.REFRESH,
            iss=self._config.issuer,
            aud=self._config.audience,
            email=user.email,
            roles=[user.role.value],
            session_id=str(session_id),
        )
        return self._encode_token(claims)

    def _encode_token(self, claims: TokenClaims) -> str:
        """Encode JWT token."""
        header = {"alg": self._config.algorithm, "typ": "JWT"}
        header_b64 = TokenCodec.base64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = TokenCodec.base64url_encode(json.dumps(claims.to_dict(), separators=(",", ":")).encode())
        message = f"{header_b64}.{payload_b64}"
        hash_func = TokenCodec.get_hash_func(self._config.algorithm)
        signature = hmac.new(self._config.secret_key.encode(), message.encode(), hash_func).digest()
        signature_b64 = TokenCodec.base64url_encode(signature)
        return f"{message}.{signature_b64}"

    def validate_token(self, token: str) -> TokenValidationResult:
        """Validate JWT token."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return TokenValidationResult(error="Invalid token format", error_code="INVALID_FORMAT")
            header_b64, payload_b64, signature_b64 = parts
            header = json.loads(TokenCodec.base64url_decode(header_b64))
            if header.get("alg") != self._config.algorithm:
                return TokenValidationResult(error="Invalid algorithm", error_code="INVALID_ALGORITHM")
            message = f"{header_b64}.{payload_b64}"
            hash_func = TokenCodec.get_hash_func(self._config.algorithm)
            expected_sig = hmac.new(self._config.secret_key.encode(), message.encode(), hash_func).digest()
            actual_sig = TokenCodec.base64url_decode(signature_b64)
            if not hmac.compare_digest(expected_sig, actual_sig):
                return TokenValidationResult(error="Invalid signature", error_code="INVALID_SIGNATURE")
            payload = json.loads(TokenCodec.base64url_decode(payload_b64))
            claims = TokenClaims.from_dict(payload)
            if claims.jti in self._revoked_tokens:
                return TokenValidationResult(error="Token revoked", error_code="TOKEN_REVOKED")
            if claims.is_expired():
                return TokenValidationResult(error="Token expired", error_code="TOKEN_EXPIRED")
            if claims.iss != self._config.issuer:
                return TokenValidationResult(error="Invalid issuer", error_code="INVALID_ISSUER")
            if claims.aud != self._config.audience:
                return TokenValidationResult(error="Invalid audience", error_code="INVALID_AUDIENCE")
            return TokenValidationResult(valid=True, claims=claims)
        except json.JSONDecodeError as e:
            return TokenValidationResult(error=f"Invalid JSON: {e}", error_code="INVALID_JSON")
        except Exception as e:
            logger.error("token_validation_error", error=str(e))
            return TokenValidationResult(error=str(e), error_code="VALIDATION_ERROR")

    def revoke_token(self, jti: str) -> None:
        """Revoke a token by JTI."""
        self._revoked_tokens.add(jti)
        logger.info("token_revoked", jti=jti)


class SessionManager:
    """User session management."""

    def __init__(self, config: SessionConfig | None = None) -> None:
        self._config = config or SessionConfig()
        self._token_service = TokenService(self._config)
        self._sessions: dict[UUID, UserSession] = {}
        self._user_sessions: dict[UUID, list[UUID]] = {}
        self._lock = asyncio.Lock()
        self._stats = {"sessions_created": 0, "sessions_revoked": 0, "tokens_refreshed": 0}

    async def create_session(self, user: User, ip_address: str | None = None, user_agent: str | None = None) -> AuthResult:
        """Create a new session for user."""
        async with self._lock:
            user_session_ids = self._user_sessions.get(user.user_id, [])
            if len(user_session_ids) >= self._config.max_sessions_per_user:
                oldest_session_id = user_session_ids[0]
                await self._revoke_session_internal(user.user_id, oldest_session_id)
            session = UserSession(
                user_id=user.user_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            self._sessions[session.session_id] = session
            if user.user_id not in self._user_sessions:
                self._user_sessions[user.user_id] = []
            self._user_sessions[user.user_id].append(session.session_id)
            access_token = self._token_service.create_access_token(user, session.session_id)
            refresh_token = self._token_service.create_refresh_token(user, session.session_id)
            self._stats["sessions_created"] += 1
            logger.info("session_created", user_id=str(user.user_id), session_id=str(session.session_id))
            return AuthResult(success=True, user=user, access_token=access_token, refresh_token=refresh_token, session=session)

    async def refresh_session(self, refresh_token: str) -> AuthResult:
        """Refresh session using refresh token."""
        validation = self._token_service.validate_token(refresh_token)
        if not validation.valid or not validation.claims:
            return AuthResult(error=validation.error, error_code=validation.error_code)
        if validation.claims.token_type != TokenType.REFRESH:
            return AuthResult(error="Invalid token type", error_code="INVALID_TOKEN_TYPE")
        session_id = UUID(validation.claims.session_id) if validation.claims.session_id else None
        if not session_id or session_id not in self._sessions:
            return AuthResult(error="Session not found", error_code="SESSION_NOT_FOUND")
        session = self._sessions[session_id]
        if not session.is_active:
            return AuthResult(error="Session expired", error_code="SESSION_EXPIRED")
        session.last_activity = datetime.now(timezone.utc)
        from .domain.service import User, UserRole, AccountStatus
        user = User(
            user_id=UUID(validation.claims.sub),
            email=validation.claims.email or "",
            password_hash="",
            display_name="",
            role=UserRole(validation.claims.roles[0]) if validation.claims.roles else UserRole.USER,
            status=AccountStatus.ACTIVE,
        )
        new_access_token = self._token_service.create_access_token(user, session_id)
        self._stats["tokens_refreshed"] += 1
        logger.info("session_refreshed", user_id=str(user.user_id), session_id=str(session_id))
        return AuthResult(success=True, user=user, access_token=new_access_token, session=session)

    async def validate_session(self, access_token: str) -> TokenValidationResult:
        """Validate session using access token."""
        validation = self._token_service.validate_token(access_token)
        if not validation.valid or not validation.claims:
            return validation
        if validation.claims.token_type != TokenType.ACCESS:
            return TokenValidationResult(error="Invalid token type", error_code="INVALID_TOKEN_TYPE")
        session_id = UUID(validation.claims.session_id) if validation.claims.session_id else None
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if not session.is_active:
                return TokenValidationResult(error="Session expired", error_code="SESSION_EXPIRED")
            session.last_activity = datetime.now(timezone.utc)
        return validation

    async def get_user_sessions(self, user_id: UUID, active_only: bool = True) -> list[UserSession]:
        """Get all sessions for a user."""
        session_ids = self._user_sessions.get(user_id, [])
        sessions = []
        for sid in session_ids:
            if sid in self._sessions:
                session = self._sessions[sid]
                if not active_only or session.is_active:
                    sessions.append(session)
        return sessions

    async def revoke_session(self, user_id: UUID, session_id: UUID) -> bool:
        """Revoke a specific session."""
        async with self._lock:
            return await self._revoke_session_internal(user_id, session_id)

    async def _revoke_session_internal(self, user_id: UUID, session_id: UUID) -> bool:
        """Internal session revocation (must be called with lock held)."""
        if session_id not in self._sessions:
            return False
        session = self._sessions[session_id]
        if session.user_id != user_id:
            return False
        session.is_active = False
        self._stats["sessions_revoked"] += 1
        logger.info("session_revoked", user_id=str(user_id), session_id=str(session_id))
        return True

    async def revoke_all_sessions(self, user_id: UUID) -> int:
        """Revoke all sessions for a user."""
        async with self._lock:
            session_ids = self._user_sessions.get(user_id, [])[:]
            revoked = 0
            for sid in session_ids:
                if await self._revoke_session_internal(user_id, sid):
                    revoked += 1
            logger.info("all_sessions_revoked", user_id=str(user_id), count=revoked)
            return revoked

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            timeout = timedelta(minutes=self._config.session_timeout_minutes)
            expired = []
            for session_id, session in self._sessions.items():
                if session.is_active and (now - session.last_activity) > timeout:
                    session.is_active = False
                    expired.append(session_id)
            logger.info("expired_sessions_cleaned", count=len(expired))
            return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get session manager statistics."""
        return {
            "active_sessions": sum(1 for s in self._sessions.values() if s.is_active),
            "total_sessions": len(self._sessions),
            **self._stats,
        }
