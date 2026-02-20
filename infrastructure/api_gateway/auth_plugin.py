"""
Solace-AI API Gateway - JWT Authentication Plugin.
Implements JWT validation, token verification, and role-based access control.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
import base64
import hashlib
import hmac
import json
import secrets
import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class JWTAlgorithm(str, Enum):
    """Supported JWT signing algorithms."""
    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"
    RS256 = "RS256"
    RS384 = "RS384"
    RS512 = "RS512"


class TokenType(str, Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class UserRole(str, Enum):
    """User roles for authorization."""
    USER = "user"
    PREMIUM = "premium"
    CLINICIAN = "clinician"
    ADMIN = "admin"
    SYSTEM = "system"


class JWTConfig(BaseSettings):
    """JWT authentication configuration."""
    secret_key: str = Field(description="JWT secret key - MUST be set via JWT_SECRET_KEY env var")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=1)
    refresh_token_expire_days: int = Field(default=7, ge=1)
    issuer: str = Field(default="solace-ai")
    audience: str = Field(default="solace-ai-api")
    leeway_seconds: int = Field(default=30, ge=0)
    require_expiry: bool = Field(default=True)
    require_subject: bool = Field(default=True)
    cookie_name: str = Field(default="access_token")
    header_name: str = Field(default="Authorization")
    header_prefix: str = Field(default="Bearer")
    model_config = SettingsConfigDict(env_prefix="JWT_", env_file=".env", extra="ignore")

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        valid = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v not in valid:
            raise ValueError(f"algorithm must be one of: {valid}")
        return v


@dataclass
class TokenClaims:
    """JWT token claims."""
    sub: str
    exp: datetime
    iat: datetime
    iss: str
    aud: str
    jti: str
    token_type: TokenType = TokenType.ACCESS
    roles: list[UserRole] = field(default_factory=lambda: [UserRole.USER])
    email: str | None = None
    name: str | None = None
    session_id: str | None = None
    custom_claims: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {"sub": self.sub, "exp": int(self.exp.timestamp()), "iat": int(self.iat.timestamp()), "iss": self.iss, "aud": self.aud, "jti": self.jti, "token_type": self.token_type.value, "roles": [r.value for r in self.roles]}
        if self.email:
            data["email"] = self.email
        if self.name:
            data["name"] = self.name
        if self.session_id:
            data["session_id"] = self.session_id
        if self.custom_claims:
            data.update(self.custom_claims)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenClaims:
        exp = datetime.fromtimestamp(data["exp"], tz=timezone.utc) if isinstance(data.get("exp"), (int, float)) else data.get("exp", datetime.now(timezone.utc))
        iat = datetime.fromtimestamp(data["iat"], tz=timezone.utc) if isinstance(data.get("iat"), (int, float)) else data.get("iat", datetime.now(timezone.utc))
        roles = [UserRole(r) for r in data.get("roles", ["user"])]
        return cls(sub=data["sub"], exp=exp, iat=iat, iss=data.get("iss", ""), aud=data.get("aud", ""), jti=data.get("jti", ""), token_type=TokenType(data.get("token_type", "access")), roles=roles, email=data.get("email"), name=data.get("name"), session_id=data.get("session_id"), custom_claims={k: v for k, v in data.items() if k not in {"sub", "exp", "iat", "iss", "aud", "jti", "token_type", "roles", "email", "name", "session_id"}})

    def is_expired(self, leeway_seconds: int = 0) -> bool:
        return datetime.now(timezone.utc) > self.exp + timedelta(seconds=leeway_seconds)

    def has_role(self, role: UserRole) -> bool:
        return role in self.roles

    def has_any_role(self, roles: list[UserRole]) -> bool:
        return any(r in self.roles for r in roles)


@dataclass
class AuthResult:
    """Authentication result."""
    authenticated: bool
    claims: TokenClaims | None = None
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def success(cls, claims: TokenClaims) -> AuthResult:
        return cls(authenticated=True, claims=claims)

    @classmethod
    def failure(cls, code: str, message: str) -> AuthResult:
        return cls(authenticated=False, error_code=code, error_message=message)


class JWTCodec:
    """JWT encoding and decoding utilities."""

    @staticmethod
    def base64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    @staticmethod
    def base64url_decode(data: str) -> bytes:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    @staticmethod
    def get_hash_algorithm(alg: str):
        algos = {"HS256": hashlib.sha256, "HS384": hashlib.sha384, "HS512": hashlib.sha512}
        if alg not in algos:
            raise ValueError(f"Unsupported algorithm: {alg}")
        return algos[alg]


class JWTAuthPlugin:
    """JWT authentication plugin for API gateway."""

    def __init__(self, config: JWTConfig | None = None, token_blacklist: Any | None = None) -> None:
        self._config = config or JWTConfig()
        self._revoked_tokens: set[str] = set()
        self._token_blacklist = token_blacklist  # Optional Redis-backed TokenBlacklist

    def create_token(self, subject: str, roles: list[UserRole] | None = None, token_type: TokenType = TokenType.ACCESS, email: str | None = None, name: str | None = None, session_id: str | None = None, custom_claims: dict[str, Any] | None = None) -> str:
        now = datetime.now(timezone.utc)
        if token_type == TokenType.ACCESS:
            exp = now + timedelta(minutes=self._config.access_token_expire_minutes)
        elif token_type == TokenType.REFRESH:
            exp = now + timedelta(days=self._config.refresh_token_expire_days)
        else:
            exp = now + timedelta(days=365)
        jti = secrets.token_hex(16)
        claims = TokenClaims(sub=subject, exp=exp, iat=now, iss=self._config.issuer, aud=self._config.audience, jti=jti, token_type=token_type, roles=roles or [UserRole.USER], email=email, name=name, session_id=session_id, custom_claims=custom_claims or {})
        return self._encode(claims)

    def _encode(self, claims: TokenClaims) -> str:
        header = {"alg": self._config.algorithm, "typ": "JWT"}
        header_b64 = JWTCodec.base64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = JWTCodec.base64url_encode(json.dumps(claims.to_dict(), separators=(",", ":")).encode())
        message = f"{header_b64}.{payload_b64}"
        if self._config.algorithm.startswith("HS"):
            hash_func = JWTCodec.get_hash_algorithm(self._config.algorithm)
            signature = hmac.new(self._config.secret_key.encode(), message.encode(), hash_func).digest()
            signature_b64 = JWTCodec.base64url_encode(signature)
        else:
            raise ValueError(f"RS algorithms require cryptography library: {self._config.algorithm}")
        return f"{message}.{signature_b64}"

    def verify_token(self, token: str) -> AuthResult:
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return AuthResult.failure("INVALID_TOKEN", "Invalid token format")
            header_b64, payload_b64, signature_b64 = parts
            header = json.loads(JWTCodec.base64url_decode(header_b64))
            if header.get("alg") != self._config.algorithm:
                return AuthResult.failure("INVALID_ALGORITHM", f"Invalid algorithm: {header.get('alg')}")
            message = f"{header_b64}.{payload_b64}"
            if self._config.algorithm.startswith("HS"):
                hash_func = JWTCodec.get_hash_algorithm(self._config.algorithm)
                expected_sig = hmac.new(self._config.secret_key.encode(), message.encode(), hash_func).digest()
                actual_sig = JWTCodec.base64url_decode(signature_b64)
                if not hmac.compare_digest(expected_sig, actual_sig):
                    return AuthResult.failure("INVALID_SIGNATURE", "Token signature verification failed")
            else:
                return AuthResult.failure("UNSUPPORTED_ALGORITHM", "RS algorithms not implemented")
            payload = json.loads(JWTCodec.base64url_decode(payload_b64))
            claims = TokenClaims.from_dict(payload)
            if claims.jti in self._revoked_tokens:
                return AuthResult.failure("TOKEN_REVOKED", "Token has been revoked")
            if self._config.require_expiry and claims.is_expired(self._config.leeway_seconds):
                return AuthResult.failure("TOKEN_EXPIRED", "Token has expired")
            if self._config.require_subject and not claims.sub:
                return AuthResult.failure("MISSING_SUBJECT", "Token missing subject claim")
            if claims.iss != self._config.issuer:
                return AuthResult.failure("INVALID_ISSUER", f"Invalid issuer: {claims.iss}")
            if claims.aud != self._config.audience:
                return AuthResult.failure("INVALID_AUDIENCE", f"Invalid audience: {claims.aud}")
            logger.info("token_verified", subject=claims.sub, roles=[r.value for r in claims.roles])
            return AuthResult.success(claims)
        except json.JSONDecodeError as e:
            return AuthResult.failure("INVALID_JSON", f"Invalid token payload: {e}")
        except Exception as e:
            logger.error("token_verification_failed", error=str(e))
            return AuthResult.failure("VERIFICATION_FAILED", str(e))

    def extract_token(self, headers: dict[str, str], cookies: dict[str, str] | None = None) -> str | None:
        auth_header = headers.get(self._config.header_name, "")
        if auth_header.startswith(f"{self._config.header_prefix} "):
            return auth_header[len(self._config.header_prefix) + 1:]
        if cookies and self._config.cookie_name in cookies:
            return cookies[self._config.cookie_name]
        return None

    def authenticate(self, headers: dict[str, str], cookies: dict[str, str] | None = None) -> AuthResult:
        token = self.extract_token(headers, cookies)
        if not token:
            return AuthResult.failure("MISSING_TOKEN", "No authentication token provided")
        return self.verify_token(token)

    def authorize(self, claims: TokenClaims, required_roles: list[UserRole] | None = None) -> bool:
        if not required_roles:
            return True
        return claims.has_any_role(required_roles)

    def revoke_token(self, jti: str) -> None:
        self._revoked_tokens.add(jti)
        logger.info("token_revoked", jti=jti)

    def refresh_access_token(self, refresh_token: str) -> tuple[str | None, AuthResult]:
        result = self.verify_token(refresh_token)
        if not result.authenticated or not result.claims:
            return None, result
        if result.claims.token_type != TokenType.REFRESH:
            return None, AuthResult.failure("INVALID_TOKEN_TYPE", "Expected refresh token")
        new_token = self.create_token(subject=result.claims.sub, roles=result.claims.roles, token_type=TokenType.ACCESS, email=result.claims.email, name=result.claims.name, session_id=result.claims.session_id)
        logger.info("token_refreshed", subject=result.claims.sub)
        return new_token, AuthResult.success(result.claims)

    def to_kong_plugin_config(self) -> dict[str, Any]:
        return {"name": "jwt", "config": {"secret_is_base64": False, "claims_to_verify": ["exp"], "key_claim_name": "iss", "header_names": [self._config.header_name], "cookie_names": [self._config.cookie_name], "maximum_expiration": self._config.access_token_expire_minutes * 60}}


def create_solace_auth_plugin(config: JWTConfig | None = None) -> JWTAuthPlugin:
    """Create pre-configured JWT auth plugin for Solace-AI."""
    plugin = JWTAuthPlugin(config)
    logger.info("solace_auth_plugin_created", issuer=plugin._config.issuer, algorithm=plugin._config.algorithm)
    return plugin
