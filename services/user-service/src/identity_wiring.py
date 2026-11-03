"""
Solace-AI User Service - identity composition.

Builds the identity half of the user service on the IF-01 runtime: the JWT key manager and
token blacklist, the password service, the session manager, the authentication service and the
Google OAuth state store. Every Redis-backed store uses the runtime's one Redis client
(``runtime.redis``). Split verbatim out of ``main.build_user_service_state`` (WS-14.3).

``start_identity`` fills ``app.state.service`` (the ``ServiceState`` that ``main`` created) and
``stop_identity`` is registered with ``runtime.register_shutdown("identity", ...)``.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
import structlog

from solace_infrastructure.runtime import StartupDependencyError, is_strict_environment

from .infrastructure.jwt_service import JWTService, JWTConfig
from .infrastructure.password_service import PasswordService, PasswordConfig
from .auth import SessionManager, RedisSessionManager, SessionConfig, AuthenticationService

logger = structlog.get_logger(__name__)


def _is_strict(environment: str | None) -> bool:
    """Strict when the service's own environment OR the bare ``ENVIRONMENT`` is staging/production."""
    return is_strict_environment(environment or "") or is_strict_environment()


async def start_identity(runtime: Any, app: FastAPI) -> None:
    """Compose the identity stores and services onto ``app.state.service``.

    The token blacklist, the session manager and the OAuth state store all share the runtime's
    one Redis client. Without it (no ``REDIS_URL``), a strict environment refuses to start; dev
    and test log one ``identity_redis_unavailable`` warning and use in-memory stores.

    Raises:
        StartupDependencyError: in a strict environment when ``runtime.redis`` is None.
    """
    state = app.state.service
    settings = state.settings

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

    # --- JWT and password services ---

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

    # --- Sessions and authentication ---
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

    state.jwt_service = jwt_service
    state.password_service = password_service
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


async def stop_identity() -> None:
    """Stop the identity composition.

    The identity stores hold no connection of their own: the Redis client they share is the
    runtime's and the runtime closes it. Registered so every identity stop has one named slot.
    """
    logger.info("identity_stopped")
