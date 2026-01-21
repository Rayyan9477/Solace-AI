"""
Solace-AI Analytics Service - FastAPI Application.

Real-time analytics service for event processing, metrics aggregation,
and report generation. Consumes events from all Solace topics.

Architecture Layer: Infrastructure
Principles: 12-Factor App, Dependency Injection, Configuration Externalization
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)

_analytics_aggregator: "AnalyticsAggregator | None" = None
_report_service: "ReportService | None" = None
_analytics_consumer: "AnalyticsConsumer | None" = None


class ServiceConfig(BaseSettings):
    """Service configuration."""
    name: str = Field(default="analytics-service")
    env: Literal["development", "staging", "production"] = Field(default="development")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8009, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    model_config = SettingsConfigDict(env_prefix="ANALYTICS_SERVICE_", env_file=".env", extra="ignore")


class MetricsConfig(BaseSettings):
    """Metrics store configuration."""
    max_windows_per_metric: int = Field(default=1000, ge=100, le=100000)
    default_retention_hours: int = Field(default=168, ge=1, le=8760)

    model_config = SettingsConfigDict(env_prefix="METRICS_", env_file=".env", extra="ignore")


class ConsumerConfig(BaseSettings):
    """Event consumer configuration."""
    group_id: str = Field(default="analytics-service")
    batch_size: int = Field(default=100, ge=1, le=1000)
    batch_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    kafka_enabled: bool = Field(default=False)
    kafka_bootstrap_servers: str = Field(default="localhost:9092")

    model_config = SettingsConfigDict(env_prefix="CONSUMER_", env_file=".env", extra="ignore")


class ObservabilityConfig(BaseSettings):
    """Observability configuration for Prometheus and OpenTelemetry."""
    prometheus_enabled: bool = Field(default=True)
    prometheus_endpoint: str = Field(default="/metrics")
    otel_enabled: bool = Field(default=False)
    otel_service_name: str = Field(default="analytics-service")
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", env_file=".env", extra="ignore")


class AnalyticsServiceSettings(BaseSettings):
    """Aggregate analytics service settings."""
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    consumer: ConsumerConfig = Field(default_factory=ConsumerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @staticmethod
    def load() -> "AnalyticsServiceSettings":
        """Load settings from environment."""
        return AnalyticsServiceSettings()


def get_analytics_aggregator() -> "AnalyticsAggregator":
    """Get the global analytics aggregator instance."""
    if _analytics_aggregator is None:
        raise RuntimeError("Analytics aggregator not initialized")
    return _analytics_aggregator


def get_report_service() -> "ReportService":
    """Get the global report service instance."""
    if _report_service is None:
        raise RuntimeError("Report service not initialized")
    return _report_service


def get_analytics_consumer() -> "AnalyticsConsumer":
    """Get the global analytics consumer instance."""
    if _analytics_consumer is None:
        raise RuntimeError("Analytics consumer not initialized")
    return _analytics_consumer


def _create_services(settings: AnalyticsServiceSettings) -> tuple:
    """Create and configure analytics services."""
    try:
        from .aggregations import AnalyticsAggregator, MetricsStore
        from .reports import ReportService
        from .consumer import AnalyticsConsumer, ConsumerConfig as DomainConsumerConfig
    except ImportError:
        from aggregations import AnalyticsAggregator, MetricsStore
        from reports import ReportService
        from consumer import AnalyticsConsumer, ConsumerConfig as DomainConsumerConfig

    metrics_store = MetricsStore(max_windows_per_metric=settings.metrics.max_windows_per_metric)
    aggregator = AnalyticsAggregator(store=metrics_store)

    report_service = ReportService(aggregator)

    consumer_config = DomainConsumerConfig(
        group_id=settings.consumer.group_id,
        batch_size=settings.consumer.batch_size,
        batch_timeout_ms=settings.consumer.batch_timeout_ms,
    )
    consumer = AnalyticsConsumer(aggregator=aggregator, config=consumer_config)

    return aggregator, report_service, consumer


def _setup_prometheus(app: FastAPI, settings: AnalyticsServiceSettings) -> None:
    """Configure Prometheus metrics instrumentation."""
    if not settings.observability.prometheus_enabled:
        logger.info("prometheus_disabled")
        return

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/ready", "/metrics"],
            inprogress_name="analytics_http_requests_inprogress",
            inprogress_labels=True,
        )

        instrumentator.instrument(app).expose(
            app,
            endpoint=settings.observability.prometheus_endpoint,
            include_in_schema=False,
        )
        logger.info(
            "prometheus_enabled",
            endpoint=settings.observability.prometheus_endpoint,
        )
    except ImportError:
        logger.warning("prometheus_not_available", reason="prometheus-fastapi-instrumentator not installed")


def _setup_opentelemetry(app: FastAPI, settings: AnalyticsServiceSettings) -> None:
    """Configure OpenTelemetry tracing instrumentation."""
    if not settings.observability.otel_enabled:
        logger.info("opentelemetry_disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource(attributes={
            SERVICE_NAME: settings.observability.otel_service_name,
            "service.version": "1.0.0",
            "deployment.environment": settings.service.env,
        })

        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.observability.otel_exporter_otlp_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,ready,metrics",
        )

        logger.info(
            "opentelemetry_enabled",
            service_name=settings.observability.otel_service_name,
            endpoint=settings.observability.otel_exporter_otlp_endpoint,
        )
    except ImportError:
        logger.warning("opentelemetry_not_available", reason="opentelemetry packages not installed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _analytics_aggregator, _report_service, _analytics_consumer

    settings = AnalyticsServiceSettings.load()
    logger.info(
        "analytics_service_starting",
        service=settings.service.name,
        env=settings.service.env,
        kafka_enabled=settings.consumer.kafka_enabled,
    )

    _analytics_aggregator, _report_service, _analytics_consumer = _create_services(settings)

    try:
        from .api import set_dependencies
    except ImportError:
        from api import set_dependencies
    set_dependencies(_analytics_aggregator, _report_service, _analytics_consumer)

    await _analytics_consumer.start()
    logger.info("analytics_service_ready")

    yield

    logger.info("analytics_service_shutdown")
    await _analytics_consumer.stop()
    _analytics_aggregator = None
    _report_service = None
    _analytics_consumer = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = AnalyticsServiceSettings.load()

    app = FastAPI(
        title="Solace-AI Analytics Service",
        description="Real-time analytics for event processing, metrics aggregation, and report generation",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.service.env != "production" else None,
        redoc_url="/redoc" if settings.service.env != "production" else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.service.env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _setup_prometheus(app, settings)
    _setup_opentelemetry(app, settings)

    try:
        from .api import router as analytics_router
    except ImportError:
        from api import router as analytics_router
    app.include_router(analytics_router)

    @app.get("/", tags=["health"])
    async def root():
        """Service information endpoint."""
        return {
            "service": settings.service.name,
            "version": "1.0.0",
            "status": "running",
            "environment": settings.service.env,
        }

    @app.get("/health", tags=["health"])
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/ready", tags=["health"])
    async def ready():
        """Readiness probe endpoint."""
        if _analytics_aggregator is None:
            return {"status": "not_ready", "reason": "aggregator_not_initialized"}
        if _analytics_consumer is None:
            return {"status": "not_ready", "reason": "consumer_not_initialized"}
        return {"status": "ready"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = AnalyticsServiceSettings.load()
    uvicorn.run(
        "src.main:app",
        host=settings.service.host,
        port=settings.service.port,
        reload=settings.service.env == "development",
        log_level=settings.service.log_level.lower(),
    )
