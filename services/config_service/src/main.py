"""
Solace-AI Configuration Service - FastAPI Application Entry Point.
Centralized configuration, secrets, and feature flag management service.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import structlog
import uvicorn

from .settings import ConfigServiceSettings, initialize_config, get_config_manager
from .secrets import SecretsSettings, create_secrets_manager
from .feature_flags import FeatureFlagSettings, initialize_feature_flags, get_feature_flag_manager
from .api import router

logger = structlog.get_logger(__name__)


def configure_logging(log_level: str) -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, log_level, structlog.stdlib.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager for startup/shutdown."""
    settings = ConfigServiceSettings()
    configure_logging(settings.log_level)
    logger.info("config_service_starting", environment=settings.environment.value,
                host=settings.host, port=settings.port)
    config_manager = await initialize_config(settings)
    flag_manager = await initialize_feature_flags(FeatureFlagSettings())
    app.state.config_manager = config_manager
    app.state.flag_manager = flag_manager
    app.state.secrets_manager = create_secrets_manager(SecretsSettings())
    logger.info("config_service_started", environment=settings.environment.value)
    yield
    logger.info("config_service_stopping")
    await config_manager.shutdown()
    await flag_manager.shutdown()
    logger.info("config_service_stopped")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = ConfigServiceSettings()
    app = FastAPI(
        title="Solace-AI Configuration Service",
        description="Centralized configuration, secrets, and feature flag management",
        version="1.0.0",
        docs_url="/docs" if not settings.environment.value == "production" else None,
        redoc_url="/redoc" if not settings.environment.value == "production" else None,
        openapi_url="/openapi.json" if not settings.environment.value == "production" else None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Correlation-ID"],
    )
    app.include_router(router)
    _register_exception_handlers(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            errors.append({"field": loc, "message": error["msg"], "type": error["type"]})
        logger.warning("validation_error", path=request.url.path, errors=errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": {"code": "VALIDATION_ERROR", "message": "Request validation failed", "details": errors}},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_type=type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"}},
        )


app = create_application()


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint with service info."""
    return {"service": "config-service", "status": "operational", "version": "1.0.0"}


@app.get("/ready", include_in_schema=False)
async def readiness() -> dict[str, str]:
    """Kubernetes readiness probe."""
    config = get_config_manager()
    if not config._initialized:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not_ready"})
    return {"status": "ready"}


@app.get("/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


def run_server() -> None:
    """Run the configuration service server."""
    settings = ConfigServiceSettings()
    uvicorn.run(
        "services.config_service.src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()
