"""
Solace-AI Diagnosis Service - FastAPI Application Entry Point.
AMIE-inspired 4-step Chain-of-Reasoning diagnostic assessment service.
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


class DiagnosisServiceAppSettings(BaseSettings):
    """Diagnosis service application configuration."""
    service_name: str = Field(default="diagnosis-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8004)
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(default="")
    request_timeout_ms: int = Field(default=30000)
    max_request_size_mb: int = Field(default=10)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    reasoning_timeout_ms: int = Field(default=10000)
    enable_anti_sycophancy: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_", env_file=".env", extra="ignore")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


def configure_logging(settings: DiagnosisServiceAppSettings) -> None:
    """Configure structured logging for the diagnosis service."""
    try:
        from solace_security.phi_protection import phi_sanitizer_processor
        _phi_processor = phi_sanitizer_processor
    except ImportError:
        if settings.environment == "production":
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
    if settings.debug:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager for startup/shutdown."""
    settings = DiagnosisServiceAppSettings()
    configure_logging(settings)
    logger.info("diagnosis_service_starting", environment=settings.environment,
                host=settings.host, port=settings.port, version=settings.version)
    # Activate PHI encryption for ClinicalBase entities
    try:
        from solace_security.encryption import EncryptionSettings, Encryptor, FieldEncryptor
        from solace_infrastructure.database.base_models import configure_phi_encryption
        encryption_settings = EncryptionSettings()
        encryptor = Encryptor(encryption_settings)
        field_encryptor = FieldEncryptor(encryptor, encryption_settings)
        configure_phi_encryption(field_encryptor)
        logger.info("phi_encryption_activated")
    except Exception as e:
        if settings.environment == "production":
            raise RuntimeError(f"PHI encryption is required in production: {e}") from e
        logger.warning("phi_encryption_not_configured", error=str(e))

    # Initialize LLM client for enhanced symptom extraction and differential
    llm_client = None
    try:
        from services.shared.infrastructure.llm_client import UnifiedLLMClient, LLMClientSettings
        llm_client = UnifiedLLMClient(LLMClientSettings())
        await llm_client.initialize()
        logger.info("diagnosis_llm_client_initialized", available=llm_client.is_available)
    except Exception as e:
        if settings.environment == "production":
            raise RuntimeError(f"LLM client required in production: {e}") from e
        logger.warning("diagnosis_llm_client_not_configured", error=str(e))

    from .domain.service import DiagnosisService, DiagnosisServiceSettings
    from .domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
    from .domain.differential import DifferentialGenerator, DifferentialSettings
    symptom_extractor = SymptomExtractor(SymptomExtractorSettings(), llm_client=llm_client)
    differential_generator = DifferentialGenerator(DifferentialSettings(), llm_client=llm_client)
    # Initialize persistent repository
    repository = None
    try:
        from .infrastructure.repository import RepositoryFactory
        repository = RepositoryFactory.get_default()
        logger.info("diagnosis_repository_initialized", type="postgres")
    except Exception as e:
        if settings.environment == "production":
            raise RuntimeError(f"PostgreSQL repository required in production: {e}") from e
        logger.warning("diagnosis_repository_fallback", error=str(e), fallback="in_memory")
    diagnosis_service = DiagnosisService(
        settings=DiagnosisServiceSettings(),
        symptom_extractor=symptom_extractor,
        differential_generator=differential_generator,
        repository=repository,
    )
    await diagnosis_service.initialize()
    app.state.settings = settings
    app.state.diagnosis_service = diagnosis_service
    app.state.symptom_extractor = symptom_extractor
    app.state.differential_generator = differential_generator
    app.state.llm_client = llm_client

    # Initialize Kafka event bridge for cross-service event publishing
    event_bridge = None
    try:
        from .events import EventDispatcher
        from .infrastructure.event_bridge import initialize_event_bridge
        # Get postgres pool for durable event outbox
        _event_pool = None
        try:
            from solace_infrastructure.database.connection_manager import ConnectionPoolManager
            _event_pool = await ConnectionPoolManager.get_pool()
        except Exception:
            logger.debug("event_outbox_pool_not_available", hint="Using in-memory outbox")
        event_dispatcher = EventDispatcher()
        event_bridge = await initialize_event_bridge(postgres_pool=_event_pool)
        event_dispatcher.subscribe_all(event_bridge.bridge_event)
        app.state.event_dispatcher = event_dispatcher
        app.state.event_bridge = event_bridge
        logger.info("diagnosis_kafka_bridge_started")
    except Exception as e:
        logger.warning("diagnosis_kafka_bridge_not_configured", error=str(e))

    logger.info("diagnosis_service_started", environment=settings.environment,
                anti_sycophancy=settings.enable_anti_sycophancy,
                llm_available=llm_client is not None and llm_client.is_available)
    yield
    logger.info("diagnosis_service_stopping")
    if event_bridge:
        await event_bridge.stop()
    await diagnosis_service.shutdown()
    if llm_client:
        await llm_client.shutdown()
    logger.info("diagnosis_service_stopped")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = DiagnosisServiceAppSettings()
    app = FastAPI(
        title="Solace-AI Diagnosis Service",
        description="AMIE-inspired 4-step Chain-of-Reasoning diagnostic assessment",
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
        expose_headers=["X-Request-ID", "X-Correlation-ID", "X-Reasoning-Time-Ms"],
    )
    from .api import router
    app.include_router(router, prefix="/api/v1/diagnosis")
    _register_middleware(app)
    _register_exception_handlers(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    """Register custom middleware for diagnosis service."""
    @app.middleware("http")
    async def request_tracking_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
        )
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        logger.info("request_completed", status_code=response.status_code,
                    process_time_ms=round(process_time_ms, 2))
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
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"code": "BAD_REQUEST", "message": str(exc)}},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_type=type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"}},
        )


app = create_application()


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint with service info."""
    return {"service": "diagnosis-service", "status": "operational", "version": "1.0.0"}


@app.get("/ready", include_in_schema=False)
async def readiness() -> JSONResponse:
    """Kubernetes readiness probe."""
    if not hasattr(app.state, "diagnosis_service") or not app.state.diagnosis_service._initialized:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not_ready"})
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


@app.get("/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health", include_in_schema=False)
async def health_check() -> dict[str, str]:
    """Health check endpoint with service status details."""
    return {
        "status": "healthy",
        "service": "diagnosis-service",
        "version": "1.0.0",
        "reasoning_pipeline": "4-step-chain-active",
        "symptom_extractor": "active",
        "differential_generator": "active",
    }


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
