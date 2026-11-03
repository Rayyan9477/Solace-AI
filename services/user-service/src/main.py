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

from .config import UserServiceSettings, allow_special_use_tlds
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
from .auth import SessionManager, RedisSessionManager, SessionConfig, AuthenticationService

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

        # OAuth (REV-22) — inert unless Google OAuth is configured. When
        # oauth_settings is None the /auth/oauth/google/* routes return 404.
        self.oauth_settings: Any = None
        self.oauth_state_store: Any = None
        self.oauth_http_client_factory: Any = None
        self.oauth_id_token_verifier: Any = None

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

    # REV-32: fail loud at startup if this service's resolved environment disagrees with the
    # deployment's bare ENVIRONMENT (a misnamed/unset USER_SERVICE_ENV would silently pin the
    # service to the development default, disabling the production fail-loud guards).
    # Also runs the broader production-secret validation (self-skips outside prod/staging).
    from solace_security.production_guards import ProductionGuards
    ProductionGuards.assert_startup(settings.service.env, service_name="user-service")

    # Test-only (WS-05.4): USER_EMAIL_ALLOW_SPECIAL_USE_TLDS lets email addresses use the listed
    # special-use TLDs (the e2e harness registers at @test.invalid). Empty by default; a name that is not
    # a special-use name fails startup here, before any I/O.
    allow_special_use_tlds(settings.email_validation.allow_special_use_tlds)

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

    # REV-12: PHI-encryption exemption (documented). This service persists no
    # ClinicalBase entities — its User/Preferences/Consent models are its own and
    # its sensitive fields are protected by the dedicated Fernet EncryptionService
    # below. The global configure_phi_encryption() only affects ClinicalBase, so
    # it is intentionally NOT configured here (least privilege: no PHI master key).
    # Enforced by tests/integration/test_phi_encryption_activation.py.

    # --- 2. Infrastructure Layer ---

    # JWT Service.
    # P1-1 (RS256 issuer migration): user-service is the token ISSUER. When RS256 signing
    # material is provisioned via the shared AUTH_* env (AUTH_ALGORITHM=RS256 +
    # AUTH_PRIVATE_KEY_PEM + AUTH_ACTIVE_KID + AUTH_PUBLIC_KEYS_JSON), the issuer signs user
    # tokens with an RS256 key ONLY it holds — so a compromised peer container can no longer
    # forge user/admin tokens with the shared symmetric secret. Verification stays
    # dual-accept (legacy HS256 accepted while AUTH_ACCEPT_LEGACY_HS256 is on) so a rollout
    # can flip the issuer before flipping validators to strict. Absent RS256 material (the
    # dev/test default), signing stays HS256 exactly as before.
    key_manager: Any = None
    jwt_algorithm = settings.jwt_algorithm
    accept_legacy_hs256 = True
    rs256_public_keys: dict[str, str] | None = None
    rs256_active_kid: str | None = None
    try:
        from solace_security.auth import AuthSettings as _SharedAuthSettings
        from solace_security.keys import KeyManager as _KeyManager

        _shared_auth = _SharedAuthSettings()
        if getattr(_shared_auth, "algorithm", "HS256") == "RS256":
            _km = _KeyManager.from_settings(_shared_auth)
            if _km.can_sign:
                key_manager = _km
                jwt_algorithm = "RS256"
                accept_legacy_hs256 = bool(getattr(_shared_auth, "accept_legacy_hs256", True))
                rs256_public_keys = _km.public_keys()
                rs256_active_kid = _km.active_kid
                logger.info(
                    "jwt_issuer_rs256_enabled",
                    active_kid=rs256_active_kid,
                    accept_legacy_hs256=accept_legacy_hs256,
                )
    except Exception as rs_err:
        logger.warning(
            "jwt_issuer_rs256_unavailable",
            error=str(rs_err),
            hint="Falling back to HS256 issuer signing (set AUTH_ALGORITHM=RS256 + keys to enable).",
        )

    jwt_config = JWTConfig(
        secret_key=settings.jwt_secret_key,
        algorithm=jwt_algorithm,
        access_token_expire_minutes=settings.access_token_expire_minutes,
        refresh_token_expire_days=settings.refresh_token_expire_days,
        active_kid=rs256_active_kid,
        public_keys=rs256_public_keys,
        accept_legacy_hs256=accept_legacy_hs256,
    )
    # Token blacklist (Redis-backed for production, in-memory fallback)
    from .infrastructure.jwt_service import RedisTokenBlacklist, InMemoryTokenBlacklist
    token_blacklist: Any = None
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(
            settings.redis.url,
            decode_responses=True,
            socket_timeout=settings.redis.timeout,
        )
        await redis_client.ping()
        token_blacklist = RedisTokenBlacklist(redis_client)
        logger.info("token_blacklist_initialized", backend="redis")
    except Exception as bl_err:
        logger.warning("redis_blacklist_unavailable", error=str(bl_err), hint="Using in-memory blacklist")
        token_blacklist = InMemoryTokenBlacklist()

    jwt_service = JWTService(jwt_config, blacklist=token_blacklist, key_manager=key_manager)

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
    # The repositories share the service's one "default" pool (ConnectionPoolManager, built from
    # POSTGRES_* as solace_app; R-02.8). There is no private USER_DB_* client.
    repository_factory = RepositoryFactory(RepositoryConfig(use_postgres=True))

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
    # Use Redis-backed sessions when Redis is available, fall back to in-memory
    session_manager: SessionManager | RedisSessionManager
    try:
        import redis.asyncio as _aioredis
        _session_redis = _aioredis.from_url(
            settings.redis.url,
            decode_responses=True,
            socket_timeout=settings.redis.timeout,
        )
        await _session_redis.ping()
        session_manager = RedisSessionManager(_session_redis, session_config)
        logger.info("session_manager_initialized", backend="redis")
    except Exception as _sess_err:
        logger.warning("redis_session_unavailable", error=str(_sess_err), hint="Using in-memory session manager")
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

    # OAuth (REV-22, config-gated). Feature is OFF unless GOOGLE_OAUTH_CLIENT_ID +
    # GOOGLE_OAUTH_CLIENT_SECRET are set — otherwise oauth_settings stays None and the
    # /auth/oauth/google/* routes 404. The state store prefers Redis (multi-instance
    # callbacks land on any worker) and falls back to in-memory.
    from .oauth import (
        google_oauth_settings_from_env,
        default_httpx_client_factory,
        InMemoryOAuthStateStore,
        RedisOAuthStateStore,
    )
    from .infrastructure.oauth_google import verify_id_token as _verify_google_id_token

    state.oauth_settings = google_oauth_settings_from_env()
    if state.oauth_settings is not None:
        oauth_store: Any = InMemoryOAuthStateStore()
        try:
            import redis.asyncio as _oauth_aioredis
            _oauth_redis = _oauth_aioredis.from_url(
                settings.redis.url,
                decode_responses=True,
                socket_timeout=settings.redis.timeout,
            )
            await _oauth_redis.ping()
            oauth_store = RedisOAuthStateStore(_oauth_redis)
            logger.info("oauth_state_store_initialized", backend="redis")
        except Exception as _oauth_redis_err:
            logger.warning(
                "oauth_state_store_redis_unavailable",
                error=str(_oauth_redis_err),
                hint="Using in-memory OAuth state store",
            )
        state.oauth_state_store = oauth_store
        state.oauth_http_client_factory = default_httpx_client_factory
        state.oauth_id_token_verifier = _verify_google_id_token
        logger.info("google_oauth_enabled")

    state.initialized = True
    app.state.service = state

    # Store in module-level for test access
    _state = state

    # Initialize Kafka event bridge for cross-service event publishing
    event_bridge = None
    try:
        from .infrastructure.event_bridge import initialize_event_bridge
        # Get postgres pool for durable event outbox
        _event_pool = None
        try:
            from solace_infrastructure.database.connection_manager import ConnectionPoolManager
            _event_pool = await ConnectionPoolManager.get_pool()
        except Exception:
            logger.debug("event_outbox_pool_not_available", hint="Using in-memory outbox")
        event_bridge = await initialize_event_bridge(postgres_pool=_event_pool)
        state.event_bridge = event_bridge
        logger.info("user_kafka_bridge_started")
    except Exception as e:
        logger.warning("user_kafka_bridge_not_configured", error=str(e))

    logger.info(
        "user_service_initialized",
        jwt_algorithm=settings.jwt_algorithm,
        access_token_minutes=settings.access_token_expire_minutes,
        refresh_token_days=settings.refresh_token_expire_days,
        session_timeout_minutes=settings.service.session_timeout_minutes,
    )

    yield

    # --- Cleanup ---
    if event_bridge:
        await event_bridge.stop()
    logger.info("user_service_shutting_down", stats=state.stats)
    state.initialized = False
    _state = None


def _rate_limiting_disabled() -> bool:
    """Disable rate limiting only under a test harness — never in production.

    Production ALWAYS enforces (and fails loud below), regardless of a stray process-level
    ENVIRONMENT. Outside production, a test context is detected from EITHER the service env
    (USER_SERVICE_ENV) OR the process env (ENVIRONMENT, which the pytest harness sets),
    accepting any value that starts with "test".
    """
    import os

    service_env = os.getenv("USER_SERVICE_ENV", "development")
    if service_env.lower() == "production":
        return False
    candidates = (service_env, os.getenv("ENVIRONMENT", ""))
    return any(str(v).lower().startswith("test") for v in candidates)


def _register_rate_limiting(app: FastAPI, redis_url: str) -> None:
    """P1-7 / P2-4: wire per-IP rate limiting, with tight buckets on the auth routes.

    Mirrors the orchestrator's _register_rate_limiting: disabled only under a test harness
    (so the shared sliding-window counter cannot bleed across tests), enforced everywhere
    else, and FAIL-LOUD in production if it cannot be configured. The user-service auth
    endpoints (/auth/login, /auth/resend-verification) had NO rate limiting, leaving
    password-spray and email-bomb surfaces unbounded — these get their own tight per-IP
    policies below.
    """
    import os

    if _rate_limiting_disabled():
        return

    is_production = os.getenv("USER_SERVICE_ENV", "development").lower() == "production"
    try:
        from infrastructure.api_gateway.rate_limiting import (
            create_solace_rate_limiter,
            RateLimitPolicy,
            RateLimitScope,
            RateLimitWindow,
        )
        from solace_security.rate_limit import RateLimitMiddleware, client_ip

        # Shared async Redis client so multi-instance limits are enforced in Redis, not
        # per-process memory. create_solace_rate_limiter fails loud in production if None.
        rl_redis = None
        try:
            import redis.asyncio as _redis

            rl_redis = _redis.from_url(redis_url, decode_responses=True)
        except Exception:  # noqa: BLE001 - prod fail-loud below handles None
            rl_redis = None

        limiter = create_solace_rate_limiter(redis_client=rl_redis)
        # Tight per-IP auth policies (P2-4). auth-login (20/min) already ships in the shared
        # factory; add a tighter resend-verification bucket to bound email bombing.
        limiter.add_policy(
            RateLimitPolicy(
                name="auth-resend-per-ip",
                limit=5,
                window=RateLimitWindow.MINUTE,
                scope=RateLimitScope.IP,
                route_name="auth-resend",
                tags=["solace", "auth"],
            )
        )

        def _route_resolver(request: Request) -> str | None:
            path = request.url.path
            if path.endswith("/auth/login"):
                return "auth-login"
            if path.endswith("/auth/resend-verification"):
                return "auth-resend"
            return None

        def _principal(request: Request) -> str:
            # Unauthenticated auth routes are the target surface; key on client IP.
            return f"ip:{client_ip(request)}"

        app.add_middleware(
            RateLimitMiddleware,
            limiter=limiter,
            service_name="user-service",
            principal_resolver=_principal,
            route_resolver=_route_resolver,
        )
        logger.info("rate_limiting_enabled", service="user-service")
    except Exception as e:
        if is_production:
            raise RuntimeError(f"Rate limiting is required in production: {e}") from e
        logger.warning("rate_limiting_not_configured", error=str(e))


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

    # Rate limiting is registered BEFORE CORS so CORS ends up the OUTERMOST middleware and
    # its headers are attached even to the limiter's 429 short-circuit (readable cross-origin).
    try:
        _redis_url = UserServiceSettings().redis.url
    except Exception:
        from .config import RedisConfig
        _redis_url = RedisConfig().url
    _register_rate_limiting(app, _redis_url)

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

    @app.get("/.well-known/jwks.json", tags=["Auth"], include_in_schema=False)
    async def jwks() -> dict[str, Any]:
        """Publish the RS256 public verification keyset (JWKS) for token validation (C.3).

        user-service is the token issuer, so it publishes its public keys here for external
        consumers (frontend, OAuth). Internal services load the same keyset from
        AUTH_PUBLIC_KEYS_JSON. Empty ({"keys": []}) while running in HS256 mode.
        """
        from solace_security.auth import AuthSettings
        from solace_security.keys import KeyManager

        try:
            return KeyManager.from_settings(AuthSettings()).jwks()
        except Exception as e:  # never let key config take down the endpoint
            logger.warning("jwks_unavailable", error=str(e))
            return {"keys": []}

    # Include API router
    app.include_router(api_router, prefix="/api/v1")

    # OAuth router (REV-22). Always registered; the handlers config-gate themselves
    # (404 when oauth_settings is None), so this is inert unless Google OAuth is set.
    from .oauth import oauth_router
    app.include_router(oauth_router, prefix="/api/v1")

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
