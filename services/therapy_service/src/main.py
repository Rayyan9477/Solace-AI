"""
Solace-AI Therapy Service - FastAPI Application Entry Point.
Evidence-based hybrid (rules+LLM) therapeutic intervention service with CBT/DBT/ACT/MI modalities.

The app is built by the IF-01 runtime (``create_service_app``): logging, exception handlers,
middleware, ``/live`` ``/ready`` ``/health`` ``/metrics`` and the Postgres and Redis clients come
from ``solace_infrastructure.runtime``. This module composes the therapy service on top of them.
"""
from __future__ import annotations
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import FastAPI
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from solace_infrastructure.runtime import (
    ServiceRuntime,
    ServiceRuntimeSpec,
    StartupDependencyError,
    create_service_app,
    is_strict_environment,
    service_runtime,
)

from .domain.service import TherapyOrchestrator
from .events import EventBus
from .infrastructure.redis_session_repository import RedisSessionRepository

logger = structlog.get_logger(__name__)


class TherapyServiceAppSettings(BaseSettings):
    """Therapy service application configuration."""
    service_name: str = Field(default="therapy-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8006)
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(default="")
    request_timeout_ms: int = Field(default=30000)
    max_request_size_mb: int = Field(default=10)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    hybrid_mode: bool = Field(default=True)
    enable_stepped_care: bool = Field(default=True)
    # Event plane: Kafka bootstrap servers come from KAFKA_BOOTSTRAP_SERVERS (KafkaSettings).
    events_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("EVENTS_ENABLED", "THERAPY_KAFKA_ENABLED", "events_enabled"),
    )
    kafka_use_mock: bool = Field(default=False)
    model_config = SettingsConfigDict(
        env_prefix="THERAPY_", env_file=".env", extra="ignore", populate_by_name=True
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


_SETTINGS = TherapyServiceAppSettings()

SPEC = ServiceRuntimeSpec(
    service_name="therapy-service",
    environment=_SETTINGS.environment,
    version=_SETTINGS.version,
    log_level=_SETTINGS.log_level,
    debug=_SETTINGS.debug,
    uses_postgres=True,
    uses_redis=True,
    event_plane="publish_consume",
    handles_phi=True,
    audited=True,
    rate_limited=False,
    edge=False,
)


@dataclass
class TherapyComponents:
    """What the therapy lifespan composes on top of the runtime."""
    orchestrator: TherapyOrchestrator
    event_bus: EventBus
    llm_client: Any
    session_store: RedisSessionRepository | None
    unit_of_work: Any
    user_data_erasure: Any | None


async def build_therapy_components(
    runtime: ServiceRuntime, settings: TherapyServiceAppSettings
) -> TherapyComponents:
    """Compose the therapy service from the runtime's clients and register its checks and stops.

    The session store is built on ``runtime.redis`` and never owns a connection: the runtime
    closes Redis. In a strict environment a missing Redis client or erasure wiring is fatal.
    """
    from .domain.homework import HomeworkManager, HomeworkManagerSettings
    from .domain.progress import ProgressTracker, ProgressTrackerSettings
    from .domain.service import TherapyOrchestratorSettings
    from .domain.session_manager import SessionManager, SessionManagerSettings
    from .domain.technique_selector import TechniqueSelector, TechniqueSelectorSettings
    from .domain.treatment_planner import TreatmentPlanner, TreatmentPlannerSettings
    from .infrastructure.gdpr import build_user_data_erasure
    from .infrastructure.redis_session_repository import build_session_store
    from .infrastructure.repository import create_unit_of_work
    from services.shared.infrastructure import LLMClientSettings, UnifiedLLMClient

    strict = is_strict_environment(settings.environment)
    session_store = build_session_store(runtime.redis)
    if session_store is None and strict:
        raise StartupDependencyError(
            "redis", "therapy session store requires REDIS_URL in strict environments"
        )

    llm_client = UnifiedLLMClient(LLMClientSettings())
    await llm_client.initialize()
    runtime.register_shutdown("therapy_llm_client", llm_client.shutdown)
    event_bus = EventBus()

    context_assembler = None
    try:
        from .infrastructure.context_assembler import ContextAssembler, ContextAssemblerSettings
        context_settings = ContextAssemblerSettings()
        context_assembler = ContextAssembler(context_settings)
        logger.info("context_assembler_initialized", memory_service_url=context_settings.memory_service_url)
    except Exception as e:
        logger.warning("context_assembler_init_failed", error=str(e))

    # MVP-191: strict raises instead of falling back to in-memory repositories.
    unit_of_work = create_unit_of_work(environment=settings.environment)
    memory_not_ready = (
        unit_of_work.backend == "memory" and settings.environment.strip().lower() != "test"
    )

    async def _repository_ready() -> tuple[bool, str]:
        if memory_not_ready:
            return False, "in-memory repositories"
        return True, unit_of_work.backend

    runtime.register_readiness_check("therapy_repository", _repository_ready, critical=True)

    # GDPR right-to-erasure (REV-17) across the Postgres tables and the durable Redis store.
    user_data_erasure = None
    repos = (unit_of_work.treatment_plans, unit_of_work.sessions, unit_of_work.outcomes)
    if session_store is not None and all(hasattr(r, "delete_user_data") for r in repos):
        user_data_erasure = build_user_data_erasure(
            treatment_plan_repo=unit_of_work.treatment_plans,
            session_repo=unit_of_work.sessions,
            outcome_repo=unit_of_work.outcomes,
            redis_session_repo=session_store,
        )
        logger.info("therapy_user_data_erasure_wired", stores=list(user_data_erasure.store_names))
    elif strict:
        raise StartupDependencyError(
            "gdpr_erasure", "therapy erasure needs Postgres-backed repositories and the session store"
        )
    else:
        logger.warning("therapy_user_data_erasure_unavailable")

    orchestrator = TherapyOrchestrator(
        settings=TherapyOrchestratorSettings(),
        technique_selector=TechniqueSelector(TechniqueSelectorSettings()),
        session_manager=SessionManager(SessionManagerSettings()),
        llm_client=llm_client,
        event_bus=event_bus,
        context_assembler=context_assembler,
        unit_of_work=unit_of_work,
        treatment_planner=TreatmentPlanner(TreatmentPlannerSettings()),
        homework_manager=HomeworkManager(HomeworkManagerSettings()),
        progress_tracker=ProgressTracker(ProgressTrackerSettings()),
        user_data_erasure=user_data_erasure,
        # REV-16: durable live-session store; None when Redis is unavailable outside strict
        # (the orchestrator then keeps sessions in memory and /ready reports not ready).
        session_store=session_store,
    )
    await orchestrator.initialize()
    runtime.register_shutdown("therapy_orchestrator", orchestrator.shutdown)

    async def _orchestrator_ready() -> tuple[bool, str]:
        return bool(orchestrator._initialized), "initialized" if orchestrator._initialized else "not initialized"

    async def _session_store_ready() -> tuple[bool, str]:
        if session_store is None:
            return False, "redis unavailable"
        await runtime.redis.ping()
        return True, "ok"

    runtime.register_readiness_check("therapy_orchestrator", _orchestrator_ready, critical=True)
    runtime.register_readiness_check("therapy_session_store", _session_store_ready, critical=True)

    return TherapyComponents(
        orchestrator=orchestrator,
        event_bus=event_bus,
        llm_client=llm_client,
        session_store=session_store,
        unit_of_work=unit_of_work,
        user_data_erasure=user_data_erasure,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: the runtime brackets it; this composes the therapy service."""
    async with service_runtime(app, SPEC) as runtime:
        settings = TherapyServiceAppSettings()
        components = await build_therapy_components(runtime, settings)
        app.state.settings = settings
        app.state.therapy_orchestrator = components.orchestrator
        app.state.llm_client = components.llm_client
        app.state.event_bus = components.event_bus
        app.state.session_store = components.session_store

        if settings.events_enabled:
            from solace_events.config import KafkaSettings
            from .infrastructure.event_bridge import initialize_event_bridge, shutdown_event_bridge
            app.state.event_bridge = await initialize_event_bridge(
                event_bus=components.event_bus,
                kafka_settings=KafkaSettings(),
                use_mock=settings.kafka_use_mock,
                postgres_pool=runtime.pg_pool,
            )
            runtime.register_shutdown("therapy_event_bridge", shutdown_event_bridge)
            logger.info("kafka_event_bridge_started")
        elif is_strict_environment(settings.environment):
            raise StartupDependencyError(
                "events", "therapy events cannot be disabled in strict environments"
            )
        logger.info("therapy_service_started", environment=settings.environment,
                    hybrid_mode=settings.hybrid_mode, stepped_care=settings.enable_stepped_care)
        yield


def create_application() -> FastAPI:
    """Create the therapy FastAPI application on the shared runtime."""
    from .api import router
    return create_service_app(
        SPEC,
        lifespan=lifespan,
        routers=[(router, "/api/v1/therapy")],
        title="Solace-AI Therapy Service",
    )


app = create_application()


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint with service info."""
    return {"service": "therapy-service", "status": "operational", "version": "1.0.0"}


def run_server() -> None:
    """Run the therapy service server."""
    import uvicorn
    settings = TherapyServiceAppSettings()
    uvicorn.run(
        "services.therapy_service.src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()
