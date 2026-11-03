"""
Solace-AI Analytics Service - FastAPI Application.

Real-time analytics service for event processing, metrics aggregation,
and report generation. Consumes events from all Solace topics.

Architecture Layer: Infrastructure
Principles: 12-Factor App, Dependency Injection, Configuration Externalization
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from solace_events.config import KafkaSettings
from solace_events.consumer import create_consumer as create_kafka_consumer
from solace_infrastructure.runtime import (
    ServiceRuntime,
    ServiceRuntimeSpec,
    StartupDependencyError,
    create_service_app,
    is_strict_environment,
    service_runtime,
)

try:
    from .api import router, set_dependencies
except ImportError:
    from api import router, set_dependencies

if TYPE_CHECKING:
    try:
        from .aggregations import AnalyticsAggregator
        from .consumer import AnalyticsConsumer
        from .reports import ReportService
    except ImportError:
        from aggregations import AnalyticsAggregator
        from consumer import AnalyticsConsumer
        from reports import ReportService

logger = structlog.get_logger(__name__)

_analytics_aggregator: "AnalyticsAggregator | None" = None
_report_service: "ReportService | None" = None
_analytics_consumer: "AnalyticsConsumer | None" = None


class ServiceConfig(BaseSettings):
    """Service configuration."""
    name: str = Field(default="analytics-service")
    env: Literal["development", "test", "staging", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("ANALYTICS_SERVICE_ENV", "ENVIRONMENT", "env"),
    )
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8009, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    model_config = SettingsConfigDict(env_prefix="ANALYTICS_SERVICE_", env_file=".env", extra="ignore", populate_by_name=True)


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


def _set_services(aggregator, report_service, consumer) -> None:
    """Publish the composed services to the module-level getters."""
    global _analytics_aggregator, _report_service, _analytics_consumer
    _analytics_aggregator, _report_service, _analytics_consumer = aggregator, report_service, consumer


def _clear_services() -> None:
    """Drop the module-level services on shutdown."""
    _set_services(None, None, None)


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


SETTINGS = AnalyticsServiceSettings.load()

SPEC = ServiceRuntimeSpec(
    service_name="analytics-service",
    environment=SETTINGS.service.env,
    version="1.0.0",
    log_level=SETTINGS.service.log_level,
    uses_postgres=True,
    uses_redis=True,
    event_plane="publish_consume",
    handles_phi=False,
    audited=False,
    rate_limited=False,
    edge=False,
)


async def start_kafka_ingestion(
    runtime: ServiceRuntime,
    consumer: AnalyticsConsumer,
    settings: AnalyticsServiceSettings,
) -> None:
    """Wire the Kafka consumer into analytics and expose its state as the ``kafka_consumer`` check.

    Strict environments refuse to start without a working consumer; development and test
    report it as a failing readiness check instead. WS-18.9 replaces this with the event plane.
    """
    if not settings.consumer.kafka_enabled:
        if is_strict_environment(settings.service.env):
            raise StartupDependencyError(
                "kafka", "analytics-service requires its Kafka consumer in strict environments"
            )

        async def _disabled() -> tuple[bool, str]:
            return False, "kafka consumer disabled"

        runtime.register_readiness_check("kafka_consumer", _disabled, critical=True)
        return

    try:
        kafka_settings = KafkaSettings(bootstrap_servers=settings.consumer.kafka_bootstrap_servers)
        kafka_consumer = create_kafka_consumer(
            group_id=settings.consumer.group_id,
            kafka_settings=kafka_settings,
        )

        # Feed deserialized Kafka events into the analytics processor
        async def _kafka_to_analytics(event):
            await consumer.process_event(event.model_dump())

        kafka_consumer.register_default_handler(_kafka_to_analytics)

        # Wire DLQ handler for failed event processing
        try:
            from solace_events.dead_letter import create_dead_letter_handler
            from solace_events.schemas import get_topic_for_event
            dlq_pool = None
            try:
                from solace_infrastructure.database.connection_manager import ConnectionPoolManager
                dlq_pool = await ConnectionPoolManager.get_pool()
            except Exception:
                pass
            dlq_handler = create_dead_letter_handler(
                consumer_group=settings.consumer.group_id,
                postgres_pool=dlq_pool,
            )
            if hasattr(dlq_handler.store, "ensure_table"):
                await dlq_handler.store.ensure_table()

            async def _handle_dlq(event, error_msg):
                topic = get_topic_for_event(event) or event.event_type
                await dlq_handler.handle_failure(
                    event=event, original_topic=topic, error=Exception(error_msg),
                )

            kafka_consumer.set_dead_letter_handler(_handle_dlq)
            logger.info("analytics_dlq_handler_configured", postgres=dlq_pool is not None)
        except Exception as dlq_err:
            logger.warning("analytics_dlq_handler_failed", error_type=type(dlq_err).__name__)

        try:
            from .consumer import ConsumerConfig as DomainConsumerConfig
        except ImportError:
            from consumer import ConsumerConfig as DomainConsumerConfig
        topics = DomainConsumerConfig().topics
        await kafka_consumer.start(topics)
        task = asyncio.create_task(kafka_consumer.consume_loop())
        runtime.register_background_task("kafka_consumer", task)
        runtime.register_shutdown("kafka_consumer", kafka_consumer.stop)

        async def _consuming() -> tuple[bool, str]:
            return (not task.done()), ("consuming" if not task.done() else "kafka consumer stopped")

        runtime.register_readiness_check("kafka_consumer", _consuming, critical=True)
        logger.info("analytics_kafka_consumer_started", topics=topics)
    except Exception as exc:
        if is_strict_environment(settings.service.env):
            raise StartupDependencyError(
                "kafka", "analytics-service could not start its Kafka consumer"
            ) from exc
        logger.error("analytics_kafka_consumer_failed", error_type=type(exc).__name__)

        async def _not_started() -> tuple[bool, str]:
            return False, "kafka consumer not started"

        runtime.register_readiness_check("kafka_consumer", _not_started, critical=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: the IF-01 runtime brackets the analytics composition."""
    async with service_runtime(app, SPEC) as runtime:
        settings = SETTINGS
        logger.info(
            "analytics_service_starting",
            service=settings.service.name,
            env=settings.service.env,
            kafka_enabled=settings.consumer.kafka_enabled,
        )
        # REV-12: PHI-encryption exemption (documented). This service persists no
        # ClinicalBase entities: analytics events live in ClickHouse (not SQLAlchemy).
        # configure_phi_encryption() only affects ClinicalBase, so it is intentionally
        # NOT configured here (handles_phi=False). Enforced by
        # tests/integration/test_phi_encryption_activation.py.
        aggregator, report_service, consumer = _create_services(settings)
        _set_services(aggregator, report_service, consumer)
        set_dependencies(aggregator, report_service, consumer)
        runtime.register_shutdown("analytics_globals", _clear_services)
        await consumer.start()
        runtime.register_shutdown("analytics_consumer", consumer.stop)
        await start_kafka_ingestion(runtime, consumer, settings)
        logger.info("analytics_service_ready")
        yield


app = create_service_app(
    SPEC,
    lifespan=lifespan,
    routers=[(router, "")],
    title="Solace-AI Analytics Service",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=SETTINGS.service.host,
        port=SETTINGS.service.port,
        reload=SETTINGS.service.env == "development",
        log_level=SETTINGS.service.log_level.lower(),
    )
