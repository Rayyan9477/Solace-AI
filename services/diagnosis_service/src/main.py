"""
Solace-AI Diagnosis Service - FastAPI Application Entry Point.
AMIE-inspired 4-step Chain-of-Reasoning diagnostic assessment service.

The app is built by the IF-01 runtime (``create_service_app``): logging, exception handlers,
middleware, ``/live`` ``/ready`` ``/health`` ``/metrics``, PHI encryption and the Postgres and
Redis clients come from ``solace_infrastructure.runtime``. This module composes the diagnosis
service on top of them. The reasoning knobs live in one place, ``DiagnosisServiceSettings``
(``DIAGNOSIS_SERVICE_*``).
"""
from __future__ import annotations
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

from .domain.service import DiagnosisService, DiagnosisServiceSettings
from .events import EventDispatcher
from .infrastructure.redis_session_repository import DiagnosisSessionStore, build_session_store

if TYPE_CHECKING:
    from services.shared.infrastructure.llm_client import UnifiedLLMClient

    from .infrastructure.repository import DiagnosisRepositoryPort

logger = structlog.get_logger(__name__)


class DiagnosisServiceAppSettings(BaseSettings):
    """Diagnosis service application configuration."""
    service_name: str = Field(default="diagnosis-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8004)
    log_level: str = Field(default="INFO")
    # Event plane: Kafka bootstrap servers come from KAFKA_BOOTSTRAP_SERVERS (KafkaSettings).
    events_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("EVENTS_ENABLED", "events_enabled"),
    )
    model_config = SettingsConfigDict(
        env_prefix="DIAGNOSIS_", env_file=".env", extra="ignore", populate_by_name=True
    )


_SETTINGS = DiagnosisServiceAppSettings()

SPEC = ServiceRuntimeSpec(
    service_name="diagnosis-service",
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
class DiagnosisComponents:
    """What the diagnosis lifespan composes on top of the runtime."""
    service: DiagnosisService
    event_dispatcher: EventDispatcher
    llm_client: UnifiedLLMClient
    session_store: DiagnosisSessionStore | None
    repository: DiagnosisRepositoryPort | None


async def build_diagnosis_components(
    runtime: ServiceRuntime, settings: DiagnosisServiceAppSettings
) -> DiagnosisComponents:
    """Compose the diagnosis service from the runtime's clients and register its checks and stops.

    The session store is built on ``runtime.redis`` and never owns a connection: the runtime
    closes Redis. In a strict environment a missing event plane, Redis client or Postgres pool
    is fatal.
    """
    from .domain.differential import DifferentialGenerator, DifferentialSettings
    from .domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
    from .infrastructure.repository import RepositoryFactory
    from services.shared.infrastructure.llm_client import LLMClientSettings, UnifiedLLMClient

    strict = is_strict_environment(settings.environment)
    if strict and not settings.events_enabled:
        raise StartupDependencyError("events", "diagnosis requires EVENTS_ENABLED in strict environments")

    session_store = build_session_store(runtime.redis)
    if session_store is None and strict:
        raise StartupDependencyError(
            "redis", "diagnosis session store requires REDIS_URL in strict environments"
        )

    repository = RepositoryFactory.get_default() if runtime.pg_pool is not None else None
    if repository is None and strict:
        raise StartupDependencyError(
            "postgres", "diagnosis repository requires Postgres in strict environments"
        )

    llm_client = UnifiedLLMClient(LLMClientSettings())
    await llm_client.initialize()

    service_settings = DiagnosisServiceSettings()
    symptom_extractor = SymptomExtractor(SymptomExtractorSettings(), llm_client=llm_client)
    differential_generator = DifferentialGenerator(DifferentialSettings(), llm_client=llm_client)
    # REV-01: the dispatcher exists BEFORE the service and is injected, so step events are emitted.
    event_dispatcher = EventDispatcher()

    service = DiagnosisService(
        settings=service_settings,
        symptom_extractor=symptom_extractor,
        differential_generator=differential_generator,
        repository=repository,
        event_dispatcher=event_dispatcher,
        # REV-16: durable live-session store; None when Redis is unavailable outside strict
        # (the service then keeps sessions in memory and /ready reports not ready).
        session_store=session_store,
    )
    await service.initialize()

    async def _service_ready() -> tuple[bool, str]:
        ok = bool(service._initialized)
        return ok, "initialized" if ok else "not initialized"

    async def _session_store_ready() -> tuple[bool, str]:
        if session_store is None:
            return False, "redis unavailable"
        await session_store._client.ping()
        return True, "ok"

    async def _repository_ready() -> tuple[bool, str]:
        if repository is None or runtime.pg_pool is None:
            return False, "postgres unavailable"
        await runtime.pg_pool.fetchval("SELECT 1")
        return True, "ok"

    runtime.register_readiness_check("diagnosis_service", _service_ready, critical=True)
    runtime.register_readiness_check("diagnosis_session_store", _session_store_ready, critical=True)
    runtime.register_readiness_check("diagnosis_repository", _repository_ready, critical=True)
    runtime.register_shutdown("diagnosis_service", service.shutdown)
    runtime.register_shutdown("diagnosis_llm_client", llm_client.shutdown)

    logger.info(
        "diagnosis_components_built",
        max_reasoning_time_ms=service_settings.max_reasoning_time_ms,
        anti_sycophancy=service_settings.enable_anti_sycophancy,
        session_store=session_store is not None,
        repository=repository is not None,
        llm_available=bool(getattr(llm_client, "is_available", False)),
    )
    return DiagnosisComponents(
        service=service,
        event_dispatcher=event_dispatcher,
        llm_client=llm_client,
        session_store=session_store,
        repository=repository,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: the runtime brackets it; this composes the diagnosis service."""
    async with service_runtime(app, SPEC) as runtime:
        settings = DiagnosisServiceAppSettings()
        components = await build_diagnosis_components(runtime, settings)
        app.state.settings = settings
        app.state.diagnosis_service = components.service
        app.state.llm_client = components.llm_client
        app.state.event_dispatcher = components.event_dispatcher
        app.state.session_store = components.session_store

        if settings.events_enabled:
            from .infrastructure.event_bridge import initialize_event_bridge
            bridge = await initialize_event_bridge(postgres_pool=runtime.pg_pool)
            components.event_dispatcher.subscribe_all(bridge.bridge_event)
            app.state.event_bridge = bridge
            runtime.register_shutdown("diagnosis_event_bridge", bridge.stop)
            logger.info("diagnosis_kafka_bridge_started")
        logger.info("diagnosis_service_started", environment=settings.environment)
        yield


def create_application() -> FastAPI:
    """Create the diagnosis FastAPI application on the shared runtime."""
    from .api import router
    return create_service_app(
        SPEC,
        lifespan=lifespan,
        routers=[(router, "/api/v1/diagnosis")],
        title="Solace-AI Diagnosis Service",
    )


app = create_application()


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint with service info."""
    return {"service": "diagnosis-service", "status": "operational", "version": "1.0.0"}


def run_server() -> None:
    """Run the diagnosis service server."""
    import uvicorn
    settings = DiagnosisServiceAppSettings()
    uvicorn.run(
        "services.diagnosis_service.src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()
