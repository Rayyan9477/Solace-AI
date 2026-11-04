"""
Solace-AI Personality Service - FastAPI Application Entry Point.
Big Five (OCEAN) personality detection and communication style adaptation service.

The app is built by the IF-01 runtime (``create_service_app``): logging, exception handlers,
middleware, ``/live`` ``/ready`` ``/health`` ``/metrics`` and the Postgres and Redis clients come
from ``solace_infrastructure.runtime``. This module composes the personality service on top of them.
"""
from __future__ import annotations
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
from services.shared.infrastructure import LLMClientSettings, UnifiedLLMClient

from .api import router
from .domain.service import PersonalityOrchestrator, PersonalityServiceSettings
from .events import EventBus

logger = structlog.get_logger(__name__)


class PersonalityServiceAppSettings(BaseSettings):
    """Personality service application configuration."""
    service_name: str = Field(default="personality-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(
        default="development",
        validation_alias=AliasChoices("PERSONALITY_ENVIRONMENT", "ENVIRONMENT", "environment"),
    )
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8007)
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(default="")
    request_timeout_ms: int = Field(default=30000)
    max_request_size_mb: int = Field(default=10)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    enable_llm_detection: bool = Field(default=True)
    # Event plane: Kafka bootstrap servers come from KAFKA_BOOTSTRAP_SERVERS (KafkaSettings).
    events_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("EVENTS_ENABLED", "events_enabled"),
    )
    model_config = SettingsConfigDict(
        env_prefix="PERSONALITY_", env_file=".env", extra="ignore", populate_by_name=True
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


SETTINGS = PersonalityServiceAppSettings()

SPEC = ServiceRuntimeSpec(
    service_name="personality-service",
    environment=SETTINGS.environment,
    version=SETTINGS.version,
    log_level=SETTINGS.log_level,
    debug=SETTINGS.debug,
    uses_postgres=True,
    uses_redis=True,
    event_plane="publish_consume",
    handles_phi=True,
    audited=True,
    rate_limited=False,
    edge=False,
)


def build_personality_orchestrator(
    runtime: ServiceRuntime,
    settings: PersonalityServiceAppSettings,
    *,
    llm_client: UnifiedLLMClient | None,
) -> PersonalityOrchestrator:
    """Compose the orchestrator on the runtime's Postgres pool (the only composition seam)."""
    from .domain.style_adapter import StyleAdapter
    from .domain.trait_detector import TraitDetector, TraitDetectorSettings

    repository = None
    if runtime.pg_pool is not None:
        from solace_infrastructure.database.connection_manager import ConnectionPoolManager
        from .infrastructure.postgres_repository import PostgresPersonalityRepository
        repository = PostgresPersonalityRepository(ConnectionPoolManager)
    trait_detector = TraitDetector(
        TraitDetectorSettings(
            enable_llm_detection=settings.enable_llm_detection and llm_client is not None
        ),
        llm_client,
    )
    return PersonalityOrchestrator(
        settings=PersonalityServiceSettings(enable_llm_detection=settings.enable_llm_detection),
        trait_detector=trait_detector,
        style_adapter=StyleAdapter(),
        llm_client=llm_client,
        repository=repository,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: the IF-01 runtime brackets the personality composition."""
    async with service_runtime(app, SPEC) as runtime:
        settings = SETTINGS
        if not settings.events_enabled and is_strict_environment(settings.environment):
            raise StartupDependencyError(
                "events", "personality-service requires EVENTS_ENABLED=true in strict environments"
            )
        from .config import get_config
        config = get_config()
        if not config.validate_ensemble_weights():
            logger.warning(
                "ensemble_weights_invalid",
                text=config.detection.ensemble_weight_text,
                liwc=config.detection.ensemble_weight_liwc,
                llm=config.detection.ensemble_weight_llm,
            )
        llm_client = UnifiedLLMClient(LLMClientSettings())
        await llm_client.initialize()
        orchestrator = build_personality_orchestrator(runtime, settings, llm_client=llm_client)
        await orchestrator.initialize()
        runtime.register_shutdown("personality_llm_client", llm_client.shutdown)
        runtime.register_shutdown("personality_orchestrator", orchestrator.shutdown)

        async def _orchestrator_ready() -> tuple[bool, str]:
            ok = orchestrator.is_initialized
            return ok, "initialized" if ok else "not_initialized"

        runtime.register_readiness_check("personality_orchestrator", _orchestrator_ready)
        app.state.settings = settings
        app.state.personality_orchestrator = orchestrator
        app.state.llm_client = llm_client

        event_bus = EventBus()
        app.state.event_bus = event_bus
        if settings.events_enabled:
            from .infrastructure.event_bridge import initialize_event_bridge, shutdown_event_bridge
            await initialize_event_bridge(event_bus=event_bus, postgres_pool=runtime.pg_pool)
            runtime.register_shutdown("personality_event_bridge", shutdown_event_bridge)
        logger.info("personality_service_started", environment=settings.environment)
        yield


app = create_service_app(
    SPEC,
    lifespan=lifespan,
    routers=[(router, "/api/v1/personality")],
    title="Solace-AI Personality Service",
)


def run_server() -> None:
    """Run the personality service server."""
    import uvicorn
    settings = PersonalityServiceAppSettings()
    uvicorn.run(
        "services.personality_service.src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()
