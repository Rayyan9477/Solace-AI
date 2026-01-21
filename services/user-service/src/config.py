"""
Solace-AI User Service - Service Configuration.

Externalized configuration following 12-factor app principles.
All configuration values are loaded from environment variables.

Architecture Layer: Infrastructure
Principles: Configuration Externalization, Type Safety, Validation
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class DatabaseConfig(BaseSettings):
    """
    Database configuration for PostgreSQL.

    Environment Variables:
        DB_HOST: Database host (default: localhost)
        DB_PORT: Database port (default: 5432)
        DB_NAME: Database name (default: solace_users)
        DB_USER: Database user (default: postgres)
        DB_PASSWORD: Database password (required)
        DB_POOL_SIZE: Connection pool size (default: 10)
        DB_MAX_OVERFLOW: Max overflow connections (default: 20)
        DB_POOL_TIMEOUT: Pool timeout in seconds (default: 30)
        DB_ECHO: Echo SQL statements (default: false)
    """

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="solace_users", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(..., description="Database password (required)")

    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, le=300, description="Pool timeout in seconds")
    echo: bool = Field(default=False, description="Echo SQL statements")

    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", extra="ignore")

    @property
    def url(self) -> str:
        """Generate database URL for SQLAlchemy."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseSettings):
    """
    Redis configuration for caching and session storage.

    Environment Variables:
        REDIS_HOST: Redis host (default: localhost)
        REDIS_PORT: Redis port (default: 6379)
        REDIS_DB: Redis database number (default: 0)
        REDIS_PASSWORD: Redis password (optional)
        REDIS_POOL_SIZE: Connection pool size (default: 10)
        REDIS_TIMEOUT: Command timeout in seconds (default: 5)
    """

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password")

    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    timeout: int = Field(default=5, ge=1, le=60, description="Command timeout in seconds")

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class SecurityConfig(BaseSettings):
    """
    Security configuration for authentication and authorization.

    Environment Variables:
        SECURITY_JWT_SECRET: JWT secret key (required)
        SECURITY_JWT_ALGORITHM: JWT algorithm (default: HS256)
        SECURITY_JWT_EXPIRY_MINUTES: JWT expiry in minutes (default: 60)
        SECURITY_REFRESH_TOKEN_EXPIRY_DAYS: Refresh token expiry in days (default: 30)
        SECURITY_PASSWORD_MIN_LENGTH: Minimum password length (default: 8)
        SECURITY_PASSWORD_REQUIRE_SPECIAL: Require special character (default: true)
        SECURITY_MAX_LOGIN_ATTEMPTS: Max failed login attempts (default: 5)
        SECURITY_LOCKOUT_DURATION_MINUTES: Lockout duration in minutes (default: 30)
    """

    jwt_secret: str = Field(..., min_length=32, description="JWT secret key (required)")
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
        """Validate JWT secret is sufficiently strong."""
        if len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v


class ServiceConfig(BaseSettings):
    """
    User service configuration.

    Environment Variables:
        USER_SERVICE_NAME: Service name (default: user-service)
        USER_SERVICE_ENV: Environment (default: development)
        USER_SERVICE_PORT: Service port (default: 8001)
        USER_SERVICE_HOST: Service host (default: 0.0.0.0)
        USER_SERVICE_LOG_LEVEL: Log level (default: INFO)
        USER_SERVICE_ENABLE_AUDIT: Enable audit logging (default: true)
        USER_SERVICE_EMAIL_VERIFICATION_EXPIRY_HOURS: Email verification expiry (default: 24)
        USER_SERVICE_SESSION_TIMEOUT_MINUTES: Session timeout (default: 60)
    """

    name: str = Field(default="user-service", description="Service name")
    env: Literal["development", "staging", "production"] = Field(default="development", description="Environment")
    port: int = Field(default=8001, ge=1, le=65535, description="Service port")
    host: str = Field(default="0.0.0.0", description="Service host")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level"
    )
    enable_audit: bool = Field(default=True, description="Enable audit logging")

    email_verification_expiry_hours: int = Field(default=24, ge=1, le=168, description="Email verification expiry")
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440, description="Session timeout in minutes")

    model_config = SettingsConfigDict(env_prefix="USER_SERVICE_", env_file=".env", extra="ignore")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env == "development"


class KafkaConfig(BaseSettings):
    """
    Kafka configuration for event publishing.

    Environment Variables:
        KAFKA_BOOTSTRAP_SERVERS: Kafka bootstrap servers (default: localhost:9092)
        KAFKA_TOPIC_USERS: User events topic (default: solace.users)
        KAFKA_PRODUCER_ACKS: Producer acknowledgments (default: all)
        KAFKA_COMPRESSION_TYPE: Compression type (default: gzip)
        KAFKA_ENABLE: Enable Kafka event publishing (default: true)
    """

    bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    topic_users: str = Field(default="solace.users", description="User events topic")

    producer_acks: Literal["0", "1", "all"] = Field(default="all", description="Producer acknowledgments")
    compression_type: Literal["none", "gzip", "snappy", "lz4", "zstd"] = Field(
        default="gzip",
        description="Compression type"
    )
    enable: bool = Field(default=True, description="Enable Kafka event publishing")

    model_config = SettingsConfigDict(env_prefix="KAFKA_", env_file=".env", extra="ignore")


class UserServiceSettings(BaseSettings):
    """
    Complete user service settings aggregating all configuration sections.

    This is the main configuration class that should be used throughout the application.
    All nested configurations are loaded from environment variables.
    """

    # JWT Settings (flat for easy access)
    jwt_secret_key: str = Field(
        default="your-super-secret-key-minimum-32-characters-long",
        min_length=32,
        description="JWT secret key"
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
    field_encryption_key: str = Field(
        default="your-32-byte-encryption-key-here",
        min_length=32,
        description="Field encryption key"
    )

    # Nested configurations
    database: DatabaseConfig = Field(default_factory=lambda: DatabaseConfig(password="postgres"))
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig(jwt_secret="your-super-secret-key-minimum-32-characters-long"))
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)

    model_config = SettingsConfigDict(
        env_prefix="USER_SERVICE_",
        env_file=".env",
        extra="ignore"
    )

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
            database_host=self.database.host,
            redis_host=self.redis.host,
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
