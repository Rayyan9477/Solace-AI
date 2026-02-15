"""
Solace-AI User Service - FastAPI Application Entry Point.

Provides user authentication, profile management, preferences, and consent tracking.
Implements JWT-based authentication with Argon2 password hashing.

Architecture: Clean Architecture with DDD
- Domain: Entities, Value Objects, Domain Services
- Application: Use Cases, DTOs
- Infrastructure: Repositories, External Services
- Presentation: REST API
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from cryptography.fernet import Fernet

from .config import UserServiceSettings
from .api import router as api_router
from .infrastructure.jwt_service import JWTService, JWTConfig, TokenInvalidError, TokenExpiredError
from .infrastructure.password_service import PasswordService, PasswordConfig
from .infrastructure.token_service import TokenService, TokenConfig, create_token_service
from .infrastructure.encryption_service import EncryptionService, EncryptionConfig, create_encryption_service
from .infrastructure.repository import (
    RepositoryFactory,
    RepositoryConfig,
    UserRepository,
    UserPreferencesRepository,
    ConsentRepository,
    reset_repositories,
)
from .domain.service import UserService
from .auth import SessionManager, SessionConfig, AuthenticationService

logger = structlog.get_logger(__name__)


class ServiceState:
    """Container for service dependencies following Clean Architecture."""

    def __init__(self) -> None:
        # Configuration
        self.settings: UserServiceSettings | None = None

        # Infrastructure Layer
        self.jwt_service: JWTService | None = None
        self.password_service: PasswordService | None = None
        self.token_service: TokenService | None = None
        self.encryption_service: EncryptionService | None = None

        # Repository Layer
        self.repository_factory: RepositoryFactory | None = None
        self.user_repository: UserRepository | None = None
        self.preferences_repository: UserPreferencesRepository | None = None
        self.consent_repository: ConsentRepository | None = None

        # Domain Services
        self.user_service: UserService | None = None

        # Application Services
        self.session_manager: SessionManager | None = None
        self.auth_service: AuthenticationService | None = None

        # State tracking
        self.initialized: bool = False
        self.start_time: datetime = datetime.now(timezone.utc)
        self._stats: dict[str, int] = {
            "registrations": 0,
            "logins": 0,
            "token_refreshes": 0,
            "password_changes": 0,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Get service statistics."""
        user_service_stats = {}
        if self.user_service:
            user_service_stats = self.user_service.get_statistics()
        session_stats = {}
        if self.session_manager:
            session_stats = self.session_manager.get_statistics()
        return {
            **self._stats,
            **user_service_stats,
            **session_stats,
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
        }

    def increment_stat(self, stat: str) -> None:
        """Increment a statistic counter."""
        if stat in self._stats:
            self._stats[stat] += 1

    def reset(self) -> None:
        """Reset service state for testing."""
        # Reset repositories
        if self.repository_factory:
            self.repository_factory.reset()

        # Reset global repository state
        reset_repositories()

        # Clear statistics
        self._stats = {
            "registrations": 0,
            "logins": 0,
            "token_refreshes": 0,
            "password_changes": 0,
        }


# Module-level state for test access
_state: ServiceState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes and tears down service dependencies following Clean Architecture:
    1. Configuration
    2. Infrastructure Layer (external services)
    3. Repository Layer (data access)
    4. Domain Layer (business logic)
    5. Application Layer (use cases)
    """
    global _state

    # --- 1. Configuration ---
    settings = UserServiceSettings()

    # Configure structured logging with PHI sanitizer
    import logging as _logging
    try:
        from solace_security.phi_protection import phi_sanitizer_processor
        _phi_processor = phi_sanitizer_processor
    except ImportError:
        if settings.service.env == "production":
            raise RuntimeError("PHI log sanitizer required in production - install solace_security")
        _phi_processor = None
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if _phi_processor:
        processors.append(_phi_processor)
    if settings.service.env == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(_logging, settings.service.log_level.upper(), _logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger.info("user_service_starting")

    # --- 2. Infrastructure Layer ---

    # JWT Service
    jwt_config = JWTConfig(
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        access_token_expire_minutes=settings.access_token_expire_minutes,
        refresh_token_expire_days=settings.refresh_token_expire_days,
    )
    jwt_service = JWTService(jwt_config)

    # Password Service
    password_config = PasswordConfig(
        argon2_time_cost=settings.argon2_time_cost,
        argon2_memory_cost=settings.argon2_memory_cost,
    )
    password_service = PasswordService(password_config)

    # Load Fernet keys from environment (MUST be set — keys are persistent across restarts)
    import os
    token_encryption_key = os.environ.get("FERNET_TOKEN_KEY")
    field_encryption_key = os.environ.get("FERNET_FIELD_KEY")
    if not token_encryption_key or not field_encryption_key:
        raise RuntimeError(
            "FERNET_TOKEN_KEY and FERNET_FIELD_KEY environment variables must be set. "
            "Generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    token_encryption_key = token_encryption_key.encode()
    field_encryption_key = field_encryption_key.encode()

    token_service = create_token_service(
        encryption_key=token_encryption_key,
        email_verification_expire_hours=settings.verification_token_expiry_hours,
    )

    encryption_service = create_encryption_service(
        encryption_key=field_encryption_key,
    )

    # --- 3. Repository Layer ---
    repository_config = RepositoryConfig()
    repository_factory = RepositoryFactory(repository_config)

    user_repository = repository_factory.get_user_repository()
    preferences_repository = repository_factory.get_preferences_repository()
    consent_repository = repository_factory.get_consent_repository()

    # --- 4. Domain Layer ---
    user_service = UserService(
        user_repository=user_repository,
        preferences_repository=preferences_repository,
        consent_repository=consent_repository,
        password_service=password_service,
        max_login_attempts=settings.security.max_login_attempts,
        lockout_duration_minutes=settings.security.lockout_duration_minutes,
        verification_token_expiry_hours=settings.service.email_verification_expiry_hours,
    )

    # --- 5. Application Layer ---
    session_config = SessionConfig(
        secret_key=settings.jwt_secret_key,
        access_token_expire_minutes=settings.access_token_expire_minutes,
        refresh_token_expire_days=settings.refresh_token_expire_days,
        session_timeout_minutes=settings.service.session_timeout_minutes,
    )
    session_manager = SessionManager(session_config)

    auth_service = AuthenticationService(
        session_manager=session_manager,
        jwt_service=jwt_service,
        password_service=password_service,
    )

    # --- Store in App State ---
    state = ServiceState()

    # Configuration
    state.settings = settings

    # Infrastructure
    state.jwt_service = jwt_service
    state.password_service = password_service
    state.token_service = token_service
    state.encryption_service = encryption_service

    # Repositories
    state.repository_factory = repository_factory
    state.user_repository = user_repository
    state.preferences_repository = preferences_repository
    state.consent_repository = consent_repository

    # Domain Services
    state.user_service = user_service

    # Application Services
    state.session_manager = session_manager
    state.auth_service = auth_service

    state.initialized = True
    app.state.service = state

    # Store in module-level for test access
    _state = state

    logger.info(
        "user_service_initialized",
        jwt_algorithm=settings.jwt_algorithm,
        access_token_minutes=settings.access_token_expire_minutes,
        refresh_token_days=settings.refresh_token_expire_days,
        session_timeout_minutes=settings.service.session_timeout_minutes,
    )

    yield

    # --- Cleanup ---
    logger.info("user_service_shutting_down", stats=state.stats)
    state.initialized = False
    _state = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    # Load settings early for CORS configuration
    try:
        settings = UserServiceSettings()
        cors_config = settings.service.cors
    except Exception:
        # Fallback for when secrets aren't configured yet (e.g., during testing)
        from .config import CORSConfig
        cors_config = CORSConfig()

    app = FastAPI(
        title="Solace-AI User Service",
        description="User authentication, profile management, and consent tracking",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware with secure configuration
    # SECURITY: Never use "*" for origins when credentials are enabled
    allowed_origins = cors_config.get_allowed_origins()
    if "*" in allowed_origins and cors_config.allow_credentials:
        logger.warning(
            "cors_security_warning",
            message="Using '*' for CORS origins with credentials is insecure. "
                    "Falling back to localhost only."
        )
        allowed_origins = ["http://localhost:3000", "http://localhost:8080"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=cors_config.allow_credentials,
        allow_methods=cors_config.get_allowed_methods(),
        allow_headers=cors_config.get_allowed_headers(),
    )

    # Exception handlers
    @app.exception_handler(TokenExpiredError)
    async def token_expired_handler(request: Request, exc: TokenExpiredError) -> JSONResponse:
        """Handle expired token errors."""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Token has expired", "error_code": "TOKEN_EXPIRED"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(TokenInvalidError)
    async def token_invalid_handler(request: Request, exc: TokenInvalidError) -> JSONResponse:
        """Handle invalid token errors."""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": str(exc), "error_code": "TOKEN_INVALID"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "error_code": "VALIDATION_ERROR"},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        logger.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
        )

    # Health endpoints
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Kubernetes liveness probe."""
        return {"status": "healthy", "service": "user-service"}

    @app.get("/ready", tags=["Health"])
    async def readiness_check(request: Request) -> dict[str, Any]:
        """Kubernetes readiness probe."""
        state: ServiceState = request.app.state.service
        if not state.initialized:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "Service not initialized"},
            )
        return {
            "status": "ready",
            "service": "user-service",
            "initialized": state.initialized,
        }

    @app.get("/status", tags=["Health"])
    async def service_status(request: Request) -> dict[str, Any]:
        """Get service status and statistics."""
        state: ServiceState = request.app.state.service
        return {
            "status": "operational" if state.initialized else "initializing",
            "service": "user-service",
            "version": "1.0.0",
            "statistics": state.stats,
            "settings": {
                "jwt_algorithm": state.settings.jwt_algorithm if state.settings else None,
                "access_token_minutes": state.settings.access_token_expire_minutes if state.settings else None,
            },
        }

    # Include API router
    app.include_router(api_router, prefix="/api/v1")

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8007,
        reload=True,
        log_level="info",
    )
