"""
Solace-AI Memory Service - FastAPI Application Entry Point.
5-tier memory hierarchy service for therapeutic context management.

The app is built by the IF-01 runtime (``create_service_app``): logging, exception handlers,
middleware, ``/live`` ``/ready`` ``/health`` ``/metrics``, PHI-encryption guards and the
Postgres and Redis clients come from ``solace_infrastructure.runtime``. This module composes the
memory service on top of them. Weaviate is opt-in (D-04, ``WEAVIATE_ENABLED=true``); without it
retrieval is keyword/BM25 only.
"""
from __future__ import annotations
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from solace_infrastructure.runtime import (
    ServiceRuntime,
    ServiceRuntimeSpec,
    StartupDependencyError,
    create_service_app,
    is_strict_environment,
    service_runtime,
)

from .api import router
from .domain.service import MemoryService

logger = structlog.get_logger(__name__)


class MemoryServiceAppSettings(BaseSettings):
    """Memory service application configuration."""
    service_name: str = Field(default="memory-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8005)
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(default="")
    request_timeout_ms: int = Field(default=30000)
    max_request_size_mb: int = Field(default=50)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    context_assembly_timeout_ms: int = Field(default=100)
    consolidation_enabled: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="MEMORY_", env_file=".env", extra="ignore")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


SETTINGS = MemoryServiceAppSettings()

SPEC = ServiceRuntimeSpec(
    service_name="memory-service",
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


def build_memory_service(runtime: ServiceRuntime, settings: MemoryServiceAppSettings) -> MemoryService:
    """Compose the memory service from the runtime's clients (the only composition seam).

    Redis comes from ``runtime.redis`` (the runtime closes it). Weaviate is built only when
    ``WEAVIATE_ENABLED=true`` (D-04). Postgres persistence is built only when the runtime has a
    Postgres pool.
    """
    from .domain.consolidation import ConsolidationPipeline, ConsolidationSettings
    from .domain.context_assembler import ContextAssembler, ContextAssemblerSettings
    from .domain.decay_manager import DecayManager
    from .domain.service import MemoryServiceSettings
    from .infrastructure.hybrid_search import HybridSearchEngine, HybridSearchSettings

    # Shared DecayManager instance for consistent decay across service and consolidation
    decay_manager = DecayManager()
    context_assembler = ContextAssembler(ContextAssemblerSettings())
    consolidation_pipeline = ConsolidationPipeline(ConsolidationSettings(), decay_manager=decay_manager)
    # Hybrid BM25 + semantic search engine for memory retrieval
    search_engine = HybridSearchEngine(HybridSearchSettings())

    redis_cache = None
    if runtime.redis is not None:
        from .infrastructure.redis_cache import RedisCache, RedisSettings
        redis_cache = RedisCache(RedisSettings(), client=runtime.redis)

    weaviate_repo = None
    from .infrastructure.weaviate_repo import WeaviateRepository, WeaviateSettings
    weaviate_settings = WeaviateSettings()
    if weaviate_settings.enabled:
        weaviate_repo = WeaviateRepository(weaviate_settings)
        logger.info("weaviate_repository_created")
    else:
        logger.info("weaviate_disabled", hint="keyword/BM25 retrieval only (D-04)")

    postgres_repo = None
    if runtime.pg_pool is not None:
        from .infrastructure.postgres_repo import PostgresRepository
        postgres_repo = PostgresRepository()
        logger.info("postgres_repository_created")

    return MemoryService(
        settings=MemoryServiceSettings(),
        context_assembler=context_assembler,
        consolidation_pipeline=consolidation_pipeline,
        postgres_repo=postgres_repo,
        search_engine=search_engine,
        weaviate_repo=weaviate_repo,
        redis_cache=redis_cache,
        decay_manager=decay_manager,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: the runtime brackets it; this composes the memory service."""
    async with service_runtime(app, SPEC) as runtime:
        memory_service = build_memory_service(runtime, SETTINGS)
        if memory_service._redis is not None:
            await memory_service._redis.initialize()
        elif is_strict_environment(SETTINGS.environment):
            raise StartupDependencyError("redis", "memory cache requires REDIS_URL in strict environments")
        await memory_service.initialize()

        async def _service_ready() -> tuple[bool, str]:
            ready = memory_service._initialized
            return ready, "initialized" if ready else "not_initialized"

        runtime.register_shutdown("memory_service", memory_service.shutdown)
        runtime.register_readiness_check("memory_service", _service_ready)
        app.state.settings = SETTINGS
        app.state.memory_service = memory_service
        app.state.context_assembler = memory_service._context_assembler
        app.state.consolidation_pipeline = memory_service._consolidation_pipeline
        logger.info("memory_service_started", environment=SETTINGS.environment,
                    tiers_active=5, consolidation_enabled=SETTINGS.consolidation_enabled)
        yield


app = create_service_app(
    SPEC,
    lifespan=lifespan,
    routers=[(router, "/api/v1/memory")],
    title="Solace-AI Memory Service",
)


def run_server() -> None:
    """Run the memory service server."""
    import uvicorn
    settings = MemoryServiceAppSettings()
    uvicorn.run(
        "services.memory_service.src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()
