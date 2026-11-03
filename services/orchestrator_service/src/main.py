"""
Solace-AI Orchestrator Service - FastAPI Application Entry Point.
LangGraph-based multi-agent orchestration with WebSocket support.

The app is built by the IF-01 runtime (``create_service_app``): logging, the strict start-up
guards, the uniform error bodies, ``/live`` ``/ready`` ``/health`` ``/metrics``, the request
context and, because the orchestrator is an edge service, CORS from ``CORS_ALLOWED_ORIGINS``.
The lifespan is bracketed by ``service_runtime`` and registers every service-owned stop with
``runtime.register_shutdown``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.middleware import Middleware

from services.shared.infrastructure.llm_client import LLMClientSettings, UnifiedLLMClient
from solace_infrastructure.runtime import (
    ServiceRuntimeSpec,
    StartupDependencyError,
    create_service_app,
    is_strict_environment,
    service_runtime,
)
from solace_infrastructure.runtime.http import RequestContextMiddleware

from .agents.chat_agent import configure_chat_agent_llm
from .api import router
from .events import get_event_bus
from .infrastructure.event_bridge import initialize_event_bridge
from .langgraph import OrchestratorGraphBuilder
from .langgraph.graph_builder import GraphBuilderSettings
from .langgraph.supervisor import configure_supervisor_llm

logger = structlog.get_logger(__name__)

SERVICE_NAME = "orchestrator-service"
API_PREFIX = "/api/v1/orchestrator"
# The W0 limiter's Redis connect and command timeouts (WS111-SPE-1): the limiter sits in front
# of every non-exempt route, so a hung Redis costs a request at most this before the limiter
# fails open (RateLimitMiddleware), instead of an unbounded wait.
RATE_LIMIT_REDIS_TIMEOUT_SECONDS = 0.5


class OrchestratorAppSettings(BaseSettings):
    """Orchestrator service application configuration."""

    service_name: str = Field(default=SERVICE_NAME)
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    request_timeout_ms: int = Field(default=60000)
    websocket_timeout_ms: int = Field(default=300000)
    max_connections: int = Field(default=1000)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    enable_checkpointing: bool = Field(default=True)
    # REV-23: server-side replay of the outbound WS message tail on reconnect.
    # OFF by default (feature is inert unless a replay buffer is wired). In-memory
    # store only for now — durable replay is infra-gated.
    enable_ws_replay: bool = Field(default=False)
    ws_replay_buffer_size: int = Field(default=100, ge=1)
    model_config = SettingsConfigDict(env_prefix="ORCHESTRATOR_", env_file=".env", extra="ignore")


_SETTINGS = OrchestratorAppSettings()

SPEC = ServiceRuntimeSpec(
    service_name=SERVICE_NAME,
    environment=_SETTINGS.environment,
    version=_SETTINGS.version,
    log_level=_SETTINGS.log_level,
    debug=_SETTINGS.debug,
    uses_postgres=True,
    uses_redis=True,
    event_plane="publish_consume",
    handles_phi=True,
    audited=True,
    rate_limited=True,
    edge=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compose the orchestrator inside the IF-01 runtime; every stop is a registered closer."""
    async with service_runtime(app, SPEC) as runtime:
        settings = OrchestratorAppSettings()
        # The limiter's Redis client is built with the app (W0, until WS-11.3); close it last.
        rate_limit_redis = getattr(app.state, "rate_limit_redis", None)
        if rate_limit_redis is not None:
            runtime.register_shutdown("rate_limit_redis", rate_limit_redis.aclose)

        # Missing provider keys disable the LLM outside strict; in strict, initialize raises.
        llm_client = UnifiedLLMClient(LLMClientSettings())
        await llm_client.initialize()
        runtime.register_shutdown("llm_client", llm_client.shutdown)
        configure_chat_agent_llm(llm_client)
        configure_supervisor_llm(llm_client)

        graph_builder = OrchestratorGraphBuilder(
            GraphBuilderSettings(enable_checkpointing=settings.enable_checkpointing)
        )
        graph_builder.build()
        app.state.graph_builder = graph_builder
        app.state.compiled_graph = graph_builder.compile()
        app.state.llm_client = llm_client
        app.state.active_connections = {}
        # REV-23: attach the outbound-tail replay buffer only when enabled, so the WS
        # handler stays inert (existing behaviour) by default.
        app.state.replay_buffer = _replay_buffer(settings)

        event_bridge = await initialize_event_bridge(
            event_bus=get_event_bus(), postgres_pool=runtime.pg_pool
        )
        app.state.event_bridge = event_bridge
        runtime.register_shutdown("event_bridge", event_bridge.stop)

        logger.info(
            "orchestrator_service_started",
            checkpointing=settings.enable_checkpointing,
            llm_available=llm_client.is_available,
            ws_replay=app.state.replay_buffer is not None,
        )
        yield


def _replay_buffer(settings: OrchestratorAppSettings):
    """The in-memory WS replay buffer when ``ORCHESTRATOR_ENABLE_WS_REPLAY`` is on, else None."""
    if not settings.enable_ws_replay:
        return None
    from .replay_buffer import InMemoryReplayBufferStore

    return InMemoryReplayBufferStore(max_per_session=settings.ws_replay_buffer_size)


def create_application() -> FastAPI:
    """Build the orchestrator app on the IF-01 runtime, plus the W0 rate limiter."""
    app = create_service_app(
        SPEC,
        lifespan=lifespan,
        routers=[(router, API_PREFIX)],
        title="Solace-AI Orchestrator Service",
    )
    _register_rate_limiting(app)
    return app


def _trusted_proxy_hops() -> int:
    """Number of trusted reverse-proxy hops in front of this service.

    0 => trust only the real socket peer for IP keying (X-Forwarded-For is ignored, as
    it is client-spoofable). n>0 => the genuine client is n positions from the right of XFF.

    P2-12: in production/staging this service sits behind Caddy, which appends exactly one
    XFF hop. If we defaulted to 0 there, every real client would collapse into a single
    rate-limit bucket (the proxy's IP), letting one abuser exhaust the shared limit for all.
    So the EFFECTIVE default is 1 in strict environments and 0 in dev/test (no proxy). An
    explicit ORCHESTRATOR_TRUSTED_PROXY_HOPS always wins so non-Caddy deployments can override.
    """
    explicit = os.getenv("ORCHESTRATOR_TRUSTED_PROXY_HOPS")
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except ValueError:
            return 0

    # No explicit override: default to 1 behind a trusted proxy in strict environments.
    env = os.getenv("ORCHESTRATOR_ENVIRONMENT") or os.getenv("ENVIRONMENT") or ""
    return 1 if is_strict_environment(env) else 0


def _make_principal_resolver(trusted_proxy_hops: int):
    """Rate-limit key: per-user when a valid bearer token is present, else client IP.

    Keys on the signature-verified token subject so a forged/expired token cannot share
    or exhaust another user's bucket (it degrades to IP scope). The route's own auth
    dependency still performs the authoritative (revocation-aware) check.
    """
    from solace_security.rate_limit import client_ip

    def resolve(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            try:
                from solace_security.middleware import _get_jwt_manager
                result = _get_jwt_manager().validate_access_token(token)
                if getattr(result, "success", False) and getattr(result, "user_id", None):
                    return f"user:{result.user_id}"
            except Exception:
                pass
        return f"ip:{client_ip(request, trusted_proxy_hops)}"

    return resolve


def _strict() -> bool:
    """The runtime's strictness rule: the service's environment or the bare ENVIRONMENT (D-14)."""
    return is_strict_environment(SPEC.environment) or is_strict_environment()


def _rate_limiting_disabled() -> bool:
    """Disable rate limiting only under a test harness, never in a strict environment.

    Strict environments (staging, production) ALWAYS enforce, so a stray ENVIRONMENT=test in a
    strict image cannot silently disable it. Outside strict, a test context is detected from
    EITHER the service env (ORCHESTRATOR_ENVIRONMENT) OR the process env (ENVIRONMENT),
    accepting any value that starts with "test".
    """
    service_environment = OrchestratorAppSettings().environment
    if _strict() or is_strict_environment(service_environment):
        return False
    candidates = (service_environment, os.getenv("ENVIRONMENT", ""))
    return any(str(v).strip().lower().startswith("test") for v in candidates)


def _register_rate_limiting(app: FastAPI) -> None:
    """Wire per-user/per-IP rate limiting with Retry-After (REV-24); W0 only.

    WS-11.3 deletes this once the runtime installs the limiter (``install_rate_limiting``).
    The limiter is Redis-backed when ``REDIS_URL`` is set (a lazy client: it connects on first
    use, with ``RATE_LIMIT_REDIS_TIMEOUT_SECONDS`` connect and command timeouts, and the lifespan
    closes it). A Redis error at request time fails open (``RateLimitMiddleware`` logs
    ``rate_limiter_backend_error`` and serves the request). In a strict environment a missing ``REDIS_URL`` raises
    ``StartupDependencyError``: a per-process counter would give every replica its own budget.
    The URL is never logged. Disabled only under a test harness (``_rate_limiting_disabled``),
    so the shared in-memory counter cannot bleed across tests.
    """
    if _rate_limiting_disabled():
        return
    from infrastructure.api_gateway.rate_limiting import create_solace_rate_limiter
    from solace_security.rate_limit import RateLimitMiddleware

    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url and _strict():
        raise StartupDependencyError(
            "rate_limiting", "REDIS_URL is required for the shared rate limiter in strict environments"
        )
    redis_client = None
    if redis_url:
        import redis.asyncio as redis_asyncio

        redis_client = redis_asyncio.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=RATE_LIMIT_REDIS_TIMEOUT_SECONDS,
            socket_timeout=RATE_LIMIT_REDIS_TIMEOUT_SECONDS,
        )
    app.state.rate_limit_redis = redis_client

    hops = _trusted_proxy_hops()
    _insert_inside_request_context(
        app,
        Middleware(
            RateLimitMiddleware,
            limiter=create_solace_rate_limiter(redis_client=redis_client),
            service_name=SERVICE_NAME,
            principal_resolver=_make_principal_resolver(hops),
            trusted_proxy_hops=hops,
        ),
    )
    logger.info(
        "rate_limiting_enabled",
        service=SERVICE_NAME,
        trusted_proxy_hops=hops,
        redis_backed=redis_client is not None,
    )


def _insert_inside_request_context(app: FastAPI, middleware: Middleware) -> None:
    """Put ``middleware`` just inside the runtime's ``RequestContextMiddleware``.

    That is where WS-04.6's ``install_rate_limiting`` puts the runtime's limiter: inside edge
    CORS, so a 429 still carries the CORS headers a cross-origin client needs to read it and its
    Retry-After, and inside the request context, so the 429 carries the request ids.
    ``app.user_middleware`` lists the middleware outermost first.
    """
    stack = app.user_middleware
    for index, entry in enumerate(stack):
        if entry.cls is RequestContextMiddleware:
            stack.insert(index + 1, middleware)
            return
    raise RuntimeError("create_service_app did not install RequestContextMiddleware")


app = create_application()


def run_server() -> None:
    """Run the orchestrator service server."""
    import uvicorn

    settings = OrchestratorAppSettings()
    uvicorn.run(
        "services.orchestrator_service.src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()
