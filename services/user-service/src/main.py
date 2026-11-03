"""
Solace-AI User Service - FastAPI Application Entry Point.

Provides user authentication, profile management, preferences, and consent tracking.
Implements JWT-based authentication with Argon2 password hashing.

The app is built by the shared IF-01 runtime (``create_service_app``) and the lifespan is
bracketed by ``service_runtime``. The runtime owns logging, the start-up guards, the Postgres
pool, the one Redis client, the health/metrics routes, CORS and the exception handlers; this
module composes the service from ``runtime.redis`` / ``runtime.pg_pool`` and registers every
service-owned stop with ``runtime.register_shutdown`` (MVP-107, MVP-328).

Architecture: Clean Architecture with DDD
- Domain: Entities, Value Objects, Domain Services
- Application: Use Cases, DTOs
- Infrastructure: Repositories, External Services
- Presentation: REST API
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, FastAPI, Request
import structlog

from solace_infrastructure.runtime import (
    ServiceRuntimeSpec,
    StartupDependencyError,
    create_service_app,
    is_strict_environment,
    service_runtime,
)

from .config import ServiceConfig, UserServiceSettings, allow_special_use_tlds
from .api import router as api_router
from .oauth import oauth_router
from .infrastructure.jwt_service import JWTService, JWTConfig
from .infrastructure.password_service import PasswordService, PasswordConfig
from .infrastructure.token_service import TokenService, create_token_service
from .infrastructure.encryption_service import EncryptionService, create_encryption_service
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

_SERVICE_CONFIG = ServiceConfig()

SPEC = ServiceRuntimeSpec(
    service_name="user-service",
    environment=_SERVICE_CONFIG.env,
    log_level=_SERVICE_CONFIG.log_level,
    uses_postgres=True,
    uses_redis=True,
    event_plane="publish_consume",
    handles_phi=True,
    audited=True,
    rate_limited=True,
    edge=True,
)


def _is_strict(environment: str | None) -> bool:
    """Strict when the service's own environment OR the bare ``ENVIRONMENT`` is staging/production."""
    return is_strict_environment(environment or "") or is_strict_environment()


# The readiness check that reports a user event bridge that failed to start (outside strict).
EVENT_BRIDGE_CHECK = "event_bridge"


async def _event_bridge_unavailable() -> tuple[bool, str]:
    """Readiness check for a user event bridge that did not start: never ready."""
    return False, "unavailable"


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


async def build_user_service_state(runtime: Any, app: FastAPI) -> ServiceState:
    """Compose the user service from the runtime's datastores and store it on ``app.state.service``.

    The token blacklist, the session manager and the OAuth state store all share the runtime's
    one Redis client. Without it (no ``REDIS_URL``), a strict environment refuses to start; dev
    and test log one ``identity_redis_unavailable`` warning and use in-memory stores.

    Raises:
        StartupDependencyError: in a strict environment when ``runtime.redis`` is None.
        ValueError: when ``USER_EMAIL_ALLOW_SPECIAL_USE_TLDS`` names a non-special-use TLD.
        RuntimeError: when ``FERNET_TOKEN_KEY`` or ``FERNET_FIELD_KEY`` is unset.
    """
    global _state

    # --- 1. Configuration ---
    settings = UserServiceSettings()

    # Test-only (WS-05.4): USER_EMAIL_ALLOW_SPECIAL_USE_TLDS lets email addresses use the listed
    # special-use TLDs (the e2e harness registers at @test.invalid). Empty by default; a name that is not
    # a special-use name fails startup here, before any I/O.
    allow_special_use_tlds(settings.email_validation.allow_special_use_tlds)

    # MVP-107: identity stores (blacklist, sessions, OAuth state) never silently fall back to
    # process memory in strict: a revoked token or a session must hold across every instance.
    redis_client = runtime.redis
    if redis_client is None:
        if _is_strict(runtime.spec.environment):
            raise StartupDependencyError("redis", "user-service identity stores require REDIS_URL")
        logger.warning(
            "identity_redis_unavailable",
            hint="Using in-memory token blacklist, sessions and OAuth state (dev/test only)",
        )

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
    # Token blacklist on the runtime's Redis client (in-memory only outside strict).
    from .infrastructure.jwt_service import RedisTokenBlacklist, InMemoryTokenBlacklist
    token_blacklist: Any
    if redis_client is not None:
        token_blacklist = RedisTokenBlacklist(redis_client)
    else:
        token_blacklist = InMemoryTokenBlacklist()

    jwt_service = JWTService(jwt_config, blacklist=token_blacklist, key_manager=key_manager)

    # Password Service
    password_config = PasswordConfig(
        argon2_time_cost=settings.argon2_time_cost,
        argon2_memory_cost=settings.argon2_memory_cost,
    )
    password_service = PasswordService(password_config)

    # Load Fernet keys from environment (MUST be set — keys are persistent across restarts)
    token_encryption_key = os.environ.get("FERNET_TOKEN_KEY")
    field_encryption_key = os.environ.get("FERNET_FIELD_KEY")
    if not token_encryption_key or not field_encryption_key:
        raise RuntimeError(
            "FERNET_TOKEN_KEY and FERNET_FIELD_KEY environment variables must be set. "
            "Generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )

    token_service = create_token_service(
        encryption_key=token_encryption_key.encode(),
        email_verification_expire_hours=settings.verification_token_expiry_hours,
    )

    encryption_service = create_encryption_service(
        encryption_key=field_encryption_key.encode(),
    )

    # --- 3. Repository Layer ---
    # The repositories share the service's one "default" pool (ConnectionPoolManager, built from
    # POSTGRES_* as solace_app; R-02.8). There is no private per-service database client.
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
    session_manager: SessionManager | RedisSessionManager
    if redis_client is not None:
        session_manager = RedisSessionManager(redis_client, session_config)
    else:
        session_manager = SessionManager(session_config)

    auth_service = AuthenticationService(
        session_manager=session_manager,
        jwt_service=jwt_service,
        password_service=password_service,
    )

    # --- Store in App State ---
    state = ServiceState()
    state.settings = settings
    state.jwt_service = jwt_service
    state.password_service = password_service
    state.token_service = token_service
    state.encryption_service = encryption_service
    state.repository_factory = repository_factory
    state.user_repository = user_repository
    state.preferences_repository = preferences_repository
    state.consent_repository = consent_repository
    state.user_service = user_service
    state.session_manager = session_manager
    state.auth_service = auth_service

    # OAuth (REV-22, config-gated). Feature is OFF unless GOOGLE_OAUTH_CLIENT_ID +
    # GOOGLE_OAUTH_CLIENT_SECRET are set — otherwise oauth_settings stays None and the
    # /auth/oauth/google/* routes 404. The state store uses the runtime's Redis client
    # (multi-instance callbacks land on any worker); in-memory only outside strict.
    from .oauth import (
        google_oauth_settings_from_env,
        default_httpx_client_factory,
        InMemoryOAuthStateStore,
        RedisOAuthStateStore,
    )
    from .infrastructure.oauth_google import verify_id_token as _verify_google_id_token

    state.oauth_settings = google_oauth_settings_from_env()
    if state.oauth_settings is not None:
        state.oauth_state_store = (
            RedisOAuthStateStore(redis_client) if redis_client is not None else InMemoryOAuthStateStore()
        )
        state.oauth_http_client_factory = default_httpx_client_factory
        state.oauth_id_token_verifier = _verify_google_id_token
        logger.info("google_oauth_enabled")

    state.initialized = True
    app.state.service = state
    _state = state

    logger.info(
        "user_service_initialized",
        identity_store_backend="redis" if redis_client is not None else "memory",
        jwt_algorithm=jwt_algorithm,
        access_token_minutes=settings.access_token_expire_minutes,
        refresh_token_days=settings.refresh_token_expire_days,
        session_timeout_minutes=settings.service.session_timeout_minutes,
    )
    return state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compose the user service inside the IF-01 runtime bracket."""
    async with service_runtime(app, SPEC) as runtime:
        state = await build_user_service_state(runtime, app)

        async def _release_state() -> None:
            global _state
            logger.info("user_service_shutting_down", stats=state.stats)
            state.initialized = False
            _state = None

        # Registered first so it runs last (closers run in reverse registration order).
        runtime.register_shutdown("user_service_state", _release_state)

        # Kafka event bridge for cross-service publishing, on the runtime's pool (durable outbox).
        # Strict: a start failure stops the service (user lifecycle and consent events, incl.
        # erasure, must reach their consumers). Dev/test: the service still boots without a
        # broker, and a critical readiness check keeps /ready at 503 until it is fixed.
        from .infrastructure.event_bridge import initialize_event_bridge

        bridge = None
        if _is_strict(SPEC.environment):
            bridge = await initialize_event_bridge(postgres_pool=runtime.pg_pool)
        else:
            try:
                bridge = await initialize_event_bridge(postgres_pool=runtime.pg_pool)
            except Exception as exc:
                # The exception text can carry a broker or database URL: class name only.
                logger.warning("user_event_bridge_unavailable", error_type=type(exc).__name__)
                runtime.register_readiness_check(EVENT_BRIDGE_CHECK, _event_bridge_unavailable, critical=True)
        if bridge is not None:
            state.event_bridge = bridge
            runtime.register_shutdown("user_event_bridge", bridge.stop)
        yield


jwks_router = APIRouter()


@jwks_router.get("/.well-known/jwks.json", tags=["Auth"], include_in_schema=False)
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


# The OAuth router (REV-22) is always registered; its handlers config-gate themselves
# (404 when oauth_settings is None), so it is inert unless Google OAuth is set.
ROUTERS: list[tuple[APIRouter, str]] = [
    (api_router, "/api/v1"),
    (oauth_router, "/api/v1"),
    (jwks_router, ""),
]


def _rate_limiting_disabled() -> bool:
    """Disable rate limiting only under a test harness — never in a strict environment.

    Strict (staging/production, by USER_SERVICE_ENV OR the bare ENVIRONMENT) ALWAYS enforces
    (and fails loud below). Outside strict, a test context is detected from EITHER the service
    env (USER_SERVICE_ENV) OR the process env (ENVIRONMENT, which the pytest harness sets),
    accepting any value that starts with "test".
    """
    service_env = os.getenv("USER_SERVICE_ENV", "development")
    if _is_strict(service_env):
        return False
    candidates = (service_env, os.getenv("ENVIRONMENT", ""))
    return any(str(v).lower().startswith("test") for v in candidates)


def _register_rate_limiting(app: FastAPI, redis_url: str) -> None:
    """P1-7 / P2-4: wire per-IP rate limiting, with tight buckets on the auth routes.

    Mirrors the orchestrator's _register_rate_limiting: disabled only under a test harness
    (so the shared sliding-window counter cannot bleed across tests), enforced everywhere
    else, and FAIL-LOUD in a strict environment if it cannot be configured. The user-service
    auth endpoints (/auth/login, /auth/resend-verification) had NO rate limiting, leaving
    password-spray and email-bomb surfaces unbounded — these get their own tight per-IP
    policies below.
    """
    if _rate_limiting_disabled():
        return

    fail_loud = _is_strict(SPEC.environment)
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
        except Exception:  # noqa: BLE001 - strict fail-loud below handles None
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
        if fail_loud:
            raise RuntimeError(f"Rate limiting is required in strict environments: {e}") from e
        logger.warning("rate_limiting_not_configured", error=str(e))


def create_app() -> FastAPI:
    """Build a fresh user-service app on the shared IF-01 runtime."""
    app = create_service_app(SPEC, lifespan=lifespan, routers=ROUTERS, title="Solace-AI User Service")
    _register_rate_limiting(app, os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
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
