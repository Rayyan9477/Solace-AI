"""
Solace-AI User Service - Service Configuration.

Externalized configuration following 12-factor app principles.
All configuration values are loaded from environment variables.

Architecture Layer: Infrastructure
Principles: Configuration Externalization, Type Safety, Validation
"""
from __future__ import annotations

from typing import Literal

import email_validator
import structlog
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

ALLOW_SPECIAL_USE_TLDS_ENV = "USER_EMAIL_ALLOW_SPECIAL_USE_TLDS"

# email_validator's special-use domain names as the library ships them, captured before any is removed.
# USER_EMAIL_ALLOW_SPECIAL_USE_TLDS may list only these names.
SPECIAL_USE_TLDS: frozenset[str] = frozenset(email_validator.SPECIAL_USE_DOMAIN_NAMES)

# Known unsafe default values that should be rejected
_UNSAFE_SECRET_PATTERNS = [
    "your-super-secret",
    "your-32-byte",
    "changeme",
    "password",
    "secret123",
    "default-key",
    "xxxxxxxx",
]


def _is_unsafe_secret(value: str) -> bool:
    """Check if a secret value matches known unsafe patterns."""
    if not value:
        return True
    value_lower = value.lower()
    return any(pattern in value_lower for pattern in _UNSAFE_SECRET_PATTERNS)


class SecurityConfig(BaseSettings):
    """
    Security configuration for authentication and authorization.

    Environment Variables:
        SECURITY_JWT_SECRET: JWT secret key (REQUIRED - no default)
        SECURITY_JWT_ALGORITHM: JWT algorithm (default: HS256)
        SECURITY_JWT_EXPIRY_MINUTES: JWT expiry in minutes (default: 60)
        SECURITY_REFRESH_TOKEN_EXPIRY_DAYS: Refresh token expiry in days (default: 30)
        SECURITY_PASSWORD_MIN_LENGTH: Minimum password length (default: 8)
        SECURITY_PASSWORD_REQUIRE_SPECIAL: Require special character (default: true)
        SECURITY_MAX_LOGIN_ATTEMPTS: Max failed login attempts (default: 5)
        SECURITY_LOCKOUT_DURATION_MINUTES: Lockout duration in minutes (default: 30)

    SECURITY: JWT_SECRET is required and must be set via environment variable.
    """

    jwt_secret: str = Field(
        ...,  # Required, no default
        min_length=32,
        description="JWT secret key (REQUIRED - set via SECURITY_JWT_SECRET env var)"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_minutes: int = Field(default=60, ge=1, le=1440, description="JWT expiry in minutes")
    refresh_token_expiry_days: int = Field(default=30, ge=1, le=365, description="Refresh token expiry in days")

    password_min_length: int = Field(default=8, ge=6, le=128, description="Minimum password length")
    password_require_special: bool = Field(default=True, description="Require special character")

    max_login_attempts: int = Field(default=5, ge=3, le=10, description="Max failed login attempts")
    lockout_duration_minutes: int = Field(default=30, ge=5, le=1440, description="Lockout duration in minutes")

    model_config = SettingsConfigDict(env_prefix="SECURITY_", env_file=".env", extra="ignore")

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Validate JWT secret is sufficiently strong and not an unsafe default."""
        if len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        if _is_unsafe_secret(v):
            raise ValueError(
                "JWT secret appears to use an unsafe default value. "
                "Please provide a secure, randomly generated secret."
            )
        return v


class CORSConfig(BaseSettings):
    """
    CORS configuration for secure cross-origin requests.

    Environment Variables:
        CORS_ALLOWED_ORIGINS: Comma-separated list of allowed origins (default: http://localhost:3000)
        CORS_ALLOW_CREDENTIALS: Allow credentials in CORS requests (default: true)
        CORS_ALLOWED_METHODS: Comma-separated list of allowed HTTP methods (default: GET,POST,PUT,DELETE,OPTIONS)
        CORS_ALLOWED_HEADERS: Comma-separated list of allowed headers (default: Authorization,Content-Type)

    SECURITY: In production, CORS_ALLOWED_ORIGINS must be explicitly set to trusted domains.
    Using "*" for origins is not allowed when credentials are enabled.
    """

    allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed origins. Set to specific domains in production."
    )
    allow_credentials: bool = Field(default=True, description="Allow credentials in CORS requests")
    allowed_methods: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS",
        description="Comma-separated list of allowed HTTP methods"
    )
    allowed_headers: str = Field(
        default="Authorization,Content-Type,X-Requested-With,X-Request-ID",
        description="Comma-separated list of allowed headers"
    )

    model_config = SettingsConfigDict(env_prefix="CORS_", env_file=".env", extra="ignore")

    def get_allowed_origins(self) -> list[str]:
        """Get list of allowed origins."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    def get_allowed_methods(self) -> list[str]:
        """Get list of allowed methods."""
        return [method.strip() for method in self.allowed_methods.split(",") if method.strip()]

    def get_allowed_headers(self) -> list[str]:
        """Get list of allowed headers."""
        return [header.strip() for header in self.allowed_headers.split(",") if header.strip()]


class ServiceConfig(BaseSettings):
    """
    User service configuration.

    Environment Variables:
        USER_SERVICE_NAME: Service name (default: user-service)
        USER_SERVICE_ENV: Environment: development, test, staging or production (default: development)
        USER_SERVICE_PORT: Service port (default: 8001)
        USER_SERVICE_HOST: Service host (default: 0.0.0.0)
        USER_SERVICE_LOG_LEVEL: Log level (default: INFO)
        USER_SERVICE_ENABLE_AUDIT: Enable audit logging (default: true)
        USER_SERVICE_EMAIL_VERIFICATION_EXPIRY_HOURS: Email verification expiry (default: 24)
        USER_SERVICE_SESSION_TIMEOUT_MINUTES: Session timeout (default: 60)
    """

    name: str = Field(default="user-service", description="Service name")
    env: Literal["development", "test", "staging", "production"] = Field(default="development", description="Environment")
    port: int = Field(default=8001, ge=1, le=65535, description="Service port")
    host: str = Field(default="0.0.0.0", description="Service host")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level"
    )
    enable_audit: bool = Field(default=True, description="Enable audit logging")

    email_verification_expiry_hours: int = Field(default=24, ge=1, le=168, description="Email verification expiry")
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440, description="Session timeout in minutes")

    # CORS configuration
    cors: CORSConfig = Field(default_factory=CORSConfig)

    model_config = SettingsConfigDict(env_prefix="USER_SERVICE_", env_file=".env", extra="ignore")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env == "development"


class EmailValidationConfig(BaseSettings):
    """
    Email address validation.

    Environment Variables:
        USER_EMAIL_ALLOW_SPECIAL_USE_TLDS: Comma-separated special-use TLDs that email addresses may use
            (default: empty). Test-only: the e2e harness registers users at @test.invalid, which
            email_validator (and pydantic EmailStr, which calls it) rejects. Every name must be one of
            email_validator's special-use domain names; any other value fails startup.
    """

    allow_special_use_tlds: str = Field(
        default="",
        description="Test-only comma list of special-use TLDs allowed in email addresses",
    )

    model_config = SettingsConfigDict(env_prefix="USER_EMAIL_", env_file=".env", extra="ignore")


def allow_special_use_tlds(value: str) -> tuple[str, ...]:
    """Let email addresses use the special-use TLDs listed in ``value`` (USER_EMAIL_ALLOW_SPECIAL_USE_TLDS).

    ``value`` is a comma list; whitespace around a name and case are ignored. Empty (the default) changes
    nothing. Every name must be one of :data:`SPECIAL_USE_TLDS`, else ``ValueError`` naming the setting is
    raised and nothing is changed. Otherwise exactly those names are removed from
    ``email_validator.SPECIAL_USE_DOMAIN_NAMES`` (a library setting that the validator reads on every call,
    so the API's validators and pydantic ``EmailStr`` follow it). Idempotent. Returns the allowed names.
    """
    names = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    if not names:
        return ()
    unknown = [name for name in names if name not in SPECIAL_USE_TLDS]
    if unknown:
        raise ValueError(
            f"{ALLOW_SPECIAL_USE_TLDS_ENV} may list only special-use domain names "
            f"({', '.join(sorted(SPECIAL_USE_TLDS))}); not one: {', '.join(repr(name) for name in unknown)}"
        )
    special_use = email_validator.SPECIAL_USE_DOMAIN_NAMES
    for name in names:
        while name in special_use:
            special_use.remove(name)
    logger.warning("email_special_use_tlds_allowed", setting=ALLOW_SPECIAL_USE_TLDS_ENV, tlds=list(names))
    return names


class KafkaConfig(BaseSettings):
    """
    Kafka configuration for event publishing.

    Environment Variables:
        USER_KAFKA_BOOTSTRAP_SERVERS: Kafka bootstrap servers (default: localhost:9092)
        USER_KAFKA_TOPIC_USERS: User events topic (default: solace.users)
        USER_KAFKA_PRODUCER_ACKS: Producer acknowledgments (default: all)
        USER_KAFKA_COMPRESSION_TYPE: Compression type (default: gzip)
        USER_KAFKA_ENABLE: Enable Kafka event publishing (default: true)
    """

    bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    topic_users: str = Field(default="solace.users", description="User events topic")

    producer_acks: Literal["0", "1", "all"] = Field(default="all", description="Producer acknowledgments")
    compression_type: Literal["none", "gzip", "snappy", "lz4", "zstd"] = Field(
        default="gzip",
        description="Compression type"
    )
    enable: bool = Field(default=True, description="Enable Kafka event publishing")

    model_config = SettingsConfigDict(env_prefix="USER_KAFKA_", env_file=".env", extra="ignore")


class UserServiceSettings(BaseSettings):
    """
    Complete user service settings aggregating all configuration sections.

    This is the main configuration class that should be used throughout the application.
    All nested configurations are loaded from environment variables.

    SECURITY: Sensitive configuration values (JWT secrets, encryption keys) MUST be provided via
    environment variables. No defaults are provided for security-sensitive fields to prevent accidental deployment with insecure configurations. Postgres and Redis are
    not configured here: the IF-01 runtime owns both (POSTGRES_*, REDIS_URL).
    """

    # JWT Settings (flat for easy access)
    # REQUIRED: Must be set via USER_SERVICE_JWT_SECRET_KEY environment variable
    jwt_secret_key: str = Field(
        ...,  # Required, no default
        min_length=32,
        description="JWT secret key (REQUIRED - set via USER_SERVICE_JWT_SECRET_KEY env var)"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=15, ge=1, le=1440, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=30, ge=1, le=365, description="Refresh token expiry")

    # Password Settings
    argon2_time_cost: int = Field(default=2, ge=1, le=10, description="Argon2 time cost")
    argon2_memory_cost: int = Field(default=65536, ge=1024, description="Argon2 memory cost in KB")

    # Token Settings
    verification_token_length: int = Field(default=32, ge=16, le=64, description="Verification token length")
    verification_token_expiry_hours: int = Field(default=24, ge=1, le=168, description="Verification token expiry")

    # Encryption Settings
    # REQUIRED: Must be set via USER_SERVICE_FIELD_ENCRYPTION_KEY environment variable
    field_encryption_key: str = Field(
        ...,  # Required, no default
        min_length=32,
        description="Field encryption key (REQUIRED - set via USER_SERVICE_FIELD_ENCRYPTION_KEY env var)"
    )

    # Nested configurations
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    email_validation: EmailValidationConfig = Field(default_factory=EmailValidationConfig)

    model_config = SettingsConfigDict(
        env_prefix="USER_SERVICE_",
        env_file=".env",
        extra="ignore"
    )

    @field_validator("jwt_secret_key", "field_encryption_key")
    @classmethod
    def validate_secrets_not_unsafe(cls, v: str, info) -> str:
        """Validate that secrets don't use known unsafe default patterns."""
        if _is_unsafe_secret(v):
            raise ValueError(
                f"{info.field_name} appears to use an unsafe default value. "
                f"Please provide a secure, randomly generated secret via environment variable."
            )
        return v

    @model_validator(mode='after')
    def validate_nested_secrets(self) -> 'UserServiceSettings':
        """Validate that nested configuration secrets are also secure."""
        # Validate security JWT secret
        if _is_unsafe_secret(self.security.jwt_secret):
            raise ValueError(
                "security.jwt_secret appears to use an unsafe default value. "
                "Please provide a secure secret via SECURITY_JWT_SECRET environment variable."
            )
        return self

    def validate_all(self) -> None:
        """
        Validate all configuration sections.

        Raises:
            ValueError: If any configuration is invalid
        """
        logger.info(
            "configuration_validated",
            service=self.service.name,
            env=self.service.env,
            kafka_enabled=self.kafka.enable,
        )

    @staticmethod
    def load() -> UserServiceSettings:
        """
        Load settings from environment variables.

        Returns:
            UserServiceSettings: Loaded and validated settings
        """
        try:
            settings = UserServiceSettings()
            settings.validate_all()
            return settings
        except Exception as e:
            logger.error("configuration_load_failed", error=str(e))
            raise
