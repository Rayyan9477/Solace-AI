"""
Solace-AI Personality Service - FastAPI Application Entry Point.
Big Five (OCEAN) personality detection and communication style adaptation service.
"""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class PersonalityServiceAppSettings(BaseSettings):
    """Personality service application configuration."""
    service_name: str = Field(default="personality-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
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
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_", env_file=".env", extra="ignore")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


def configure_logging(settings: PersonalityServiceAppSettings) -> None:
    """Configure structured logging for the personality service."""
    try:
        from solace_security.phi_protection import phi_sanitizer_processor
        _phi_processor = phi_sanitizer_processor
    except ImportError:
        _phi_processor = None
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if _phi_processor:
        processors.append(_phi_processor)
    if settings.debug:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager for startup/shutdown."""
    settings = PersonalityServiceAppSettings()
    configure_logging(settings)
    from .config import get_config
    config = get_config()
    if not config.validate_ensemble_weights():
        logger.warning(
            "ensemble_weights_invalid",
            text=config.detection.ensemble_weight_text,
            liwc=config.detection.ensemble_weight_liwc,
            llm=config.detection.ensemble_weight_llm,
        )
    logger.info("personality_service_starting", environment=settings.environment, host=settings.host, port=settings.port, version=settings.version)
    from .domain.service import PersonalityOrchestrator, PersonalityServiceSettings
    from .domain.trait_detector import TraitDetector, TraitDetectorSettings
    from .domain.style_adapter import StyleAdapter
    llm_client = None
    try:
        from services.shared.infrastructure import UnifiedLLMClient, LLMClientSettings
        llm_client = UnifiedLLMClient(LLMClientSettings())
        await llm_client.initialize()
    except ImportError:
        logger.warning("shared_llm_client_not_available", fallback="using_text_only_detection")
    trait_detector = TraitDetector(TraitDetectorSettings(enable_llm_detection=settings.enable_llm_detection and llm_client is not None), llm_client)
    style_adapter = StyleAdapter()
    personality_orchestrator = PersonalityOrchestrator(
        settings=PersonalityServiceSettings(enable_llm_detection=settings.enable_llm_detection),
        trait_detector=trait_detector,
        style_adapter=style_adapter,
        llm_client=llm_client,
    )
    await personality_orchestrator.initialize()
    app.state.settings = settings
    app.state.personality_orchestrator = personality_orchestrator
    app.state.trait_detector = trait_detector
    app.state.style_adapter = style_adapter
    app.state.llm_client = llm_client
    logger.info("personality_service_started", environment=settings.environment, llm_enabled=llm_client is not None)
    yield
    logger.info("personality_service_stopping")
    await personality_orchestrator.shutdown()
    if llm_client:
        await llm_client.shutdown()
    logger.info("personality_service_stopped")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = PersonalityServiceAppSettings()
    app = FastAPI(
        title="Solace-AI Personality Service",
        description="Big Five (OCEAN) personality detection and communication style adaptation",
        version=settings.version,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Correlation-ID", "X-Process-Time-Ms"],
    )
    from .api import router
    app.include_router(router, prefix="/api/v1/personality")
    _register_middleware(app)
    _register_exception_handlers(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    """Register custom middleware for personality service."""
    @app.middleware("http")
    async def request_tracking_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, correlation_id=correlation_id, path=request.url.path, method=request.method)
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        logger.info("request_completed", status_code=response.status_code, process_time_ms=round(process_time_ms, 2))
        return response


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            errors.append({"field": loc, "message": error["msg"], "type": error["type"]})
        logger.warning("validation_error", path=request.url.path, errors=errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": {"code": "VALIDATION_ERROR", "message": "Request validation failed", "details": errors}},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("value_error", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": {"code": "BAD_REQUEST", "message": str(exc)}})

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_type=type(exc).__name__)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"}})


app = create_application()


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint with service info."""
    return {"service": "personality-service", "status": "operational", "version": "1.0.0"}


@app.get("/ready", include_in_schema=False)
async def readiness() -> JSONResponse:
    """Kubernetes readiness probe."""
    if not hasattr(app.state, "personality_orchestrator") or not app.state.personality_orchestrator.is_initialized:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not_ready"})
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


@app.get("/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health", include_in_schema=False)
async def health_check() -> dict:
    """Health check endpoint with service status details."""
    llm_status = "active" if hasattr(app.state, "llm_client") and app.state.llm_client else "disabled"
    return {
        "status": "healthy",
        "service": "personality-service",
        "version": "1.0.0",
        "personality_model": "Big Five (OCEAN)",
        "detection_methods": ["text_analysis", "liwc_features", "llm_zero_shot"],
        "trait_detector": "active",
        "style_adapter": "active",
        "llm_integration": llm_status,
    }


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
