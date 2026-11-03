"""
Solace-AI Safety Service - FastAPI Application Entry Point.
Always-active safety monitoring with crisis detection and escalation management.

The app is built by the shared IF-01 runtime (``create_service_app``), and the lifespan is
bracketed by ``service_runtime``. The runtime owns logging, the start-up guards, the Postgres
pool, Redis, ``/live``, ``/ready``, ``/health``, ``/metrics``, the error responses and the
request middleware. This module composes the safety domain on top of it and registers every
service-owned stop with ``runtime.register_shutdown``, so the crisis event bridge, the event
publisher and the service are stopped on every exit (WS-12.1, MVP-088).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from solace_infrastructure.runtime import (
    ServiceRuntimeSpec,
    StartupDependencyError,
    create_service_app,
    is_strict_environment,
    service_runtime,
)

from .api import router

logger = structlog.get_logger(__name__)


class SafetyServiceAppSettings(BaseSettings):
    """Safety service application configuration."""
    service_name: str = Field(default="safety-service")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8002)
    log_level: str = Field(default="INFO")
    request_timeout_ms: int = Field(default=30000)
    max_request_size_mb: int = Field(default=10)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    # Kafka event publishing configuration
    kafka_enabled: bool = Field(default=False)
    kafka_bootstrap_servers: str = Field(default="localhost:29092")
    kafka_use_mock: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="SAFETY_", env_file=".env", extra="ignore")


_SETTINGS = SafetyServiceAppSettings()

SPEC = ServiceRuntimeSpec(
    service_name="safety-service",
    environment=_SETTINGS.environment,
    version="1.0.0",
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

# The readiness check that reports a crisis event bridge that failed to start (outside strict).
EVENT_BRIDGE_CHECK = "event_bridge"


def _is_strict(settings: Any) -> bool:
    """Whether this process is strict, by the runtime's rule (WS121F1-CLO-1).

    Strict when the service's own environment (``SAFETY_ENVIRONMENT``) OR the bare
    ``ENVIRONMENT`` names staging or production. Every strictness decision in this module goes
    through here, so a blank or non-strict ``SAFETY_ENVIRONMENT`` under a strict bare
    ``ENVIRONMENT`` cannot switch the crisis-delivery and PHI guards off while the runtime
    itself boots strict.
    """
    return is_strict_environment(getattr(settings, "environment", "") or "") or is_strict_environment()


def _require_event_delivery_in_strict(settings: Any) -> None:
    """M1-gate P0: crisis events cannot be delivered without the Kafka bridge.

    The bridge (and its outbox poller) is gated behind ``kafka_enabled``, which defaults False.
    In a strict environment (staging or production) that would silently accept crisis events
    into an outbox that never reaches the notification service. Fail loud so a misconfigured
    strict deploy cannot run a crisis-safety service with no event delivery.

    A mock producer is refused in strict too (WS121-CRI-1): it "sends" each crisis row to an
    in-process list and the poller then marks the row PUBLISHED, so no human is paged and the
    row is never redelivered.

    Raises:
        StartupDependencyError: a strict environment with ``kafka_enabled`` False, or with
            ``kafka_use_mock`` True.
    """
    if not _is_strict(settings):
        return
    if not settings.kafka_enabled:
        raise StartupDependencyError(
            EVENT_BRIDGE_CHECK,
            "Kafka event bridge is required in staging and production "
            "(set SAFETY_KAFKA_ENABLED=true); crisis events cannot reach the notification "
            "service without it",
        )
    if getattr(settings, "kafka_use_mock", False):
        raise StartupDependencyError(
            EVENT_BRIDGE_CHECK,
            "a mock Kafka producer is not allowed in staging and production "
            "(unset SAFETY_KAFKA_USE_MOCK)",
        )


async def _event_bridge_unavailable() -> tuple[bool, str]:
    """Readiness check for a crisis event bridge that did not start: never ready."""
    return False, "unavailable"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compose the safety service inside the IF-01 runtime bracket."""
    async with service_runtime(app, SPEC) as runtime:
        settings = SafetyServiceAppSettings()
        strict = _is_strict(settings)
        logger.info("safety_service_starting", environment=settings.environment,
                    version=settings.version, kafka_enabled=settings.kafka_enabled)

        from .config import get_safety_config
        from .domain.escalation import EscalationManager, EscalationSettings
        from .domain.factory import build_crisis_detector, build_escalation_repository
        from .domain.service import SafetyService, SafetyServiceSettings
        from .events import initialize_event_publisher

        # Use centralized SafetyConfig to ensure all sub-components share consistent
        # configuration sourced from the same environment / .env file (M-06).
        safety_config = get_safety_config()
        crisis_detector = build_crisis_detector(safety_config)
        # MVP-088: durable Postgres escalation store whenever the runtime has a pool; a strict
        # environment without one refuses to start, dev/test fall back to in-memory.
        escalation_repo = build_escalation_repository(runtime.pg_pool, environment=settings.environment)
        escalation_manager = EscalationManager(
            EscalationSettings(
                **{k: v for k, v in safety_config.escalation.model_dump().items()
                   if k in EscalationSettings.model_fields},
            ),
            repository=escalation_repo,
        )
        # REV-01: start the local event publisher and inject it so SafetyService
        # actually emits crisis/escalation/audit events.
        event_publisher = await initialize_event_publisher()
        safety_service = SafetyService(
            settings=SafetyServiceSettings(),
            crisis_detector=crisis_detector,
            escalation_manager=escalation_manager,
            event_publisher=event_publisher,
        )
        # Closers run in reverse registration order: the bridge (registered below) stops first,
        # then the publisher, then the service.
        runtime.register_shutdown("safety_service", safety_service.shutdown)
        runtime.register_shutdown("event_publisher", event_publisher.stop)
        await safety_service.initialize()

        # Activate PHI encryption for ClinicalBase entities
        try:
            from solace_infrastructure.database.base_models import configure_phi_encryption
            from solace_security.encryption import EncryptionSettings, Encryptor, FieldEncryptor
            encryption_settings = EncryptionSettings()
            encryptor = Encryptor(encryption_settings)
            field_encryptor = FieldEncryptor(encryptor, encryption_settings)
            configure_phi_encryption(field_encryptor)
            logger.info("phi_encryption_activated")
        except Exception as e:
            if strict:
                raise RuntimeError(f"PHI encryption is required in production: {e}") from e
            logger.warning("phi_encryption_not_configured", error=str(e))

        # M1-gate P0: fail loud if the crisis event bridge is disabled in a strict environment.
        _require_event_delivery_in_strict(settings)

        app.state.settings = settings
        app.state.safety_service = safety_service
        app.state.crisis_detector = crisis_detector
        app.state.escalation_manager = escalation_manager
        app.state.event_publisher = event_publisher

        # Kafka event bridge for notification integration: crisis events go to the durable
        # outbox (owned by safety-service) and the bridge's outbox poller drains it to Kafka.
        if settings.kafka_enabled:
            from solace_events.config import KafkaSettings

            from .infrastructure.event_bridge import initialize_notification_bridge
            bridge_kwargs: dict[str, Any] = {
                "kafka_settings": KafkaSettings(bootstrap_servers=settings.kafka_bootstrap_servers),
                "use_mock": settings.kafka_use_mock,
                "postgres_pool": runtime.pg_pool,
            }
            notification_bridge = None
            if strict:
                # REV-01: in a strict environment a broken crisis->notification bridge means
                # crisis alerts never publish, so a start failure stops the service.
                notification_bridge = await initialize_notification_bridge(**bridge_kwargs)
            else:
                try:
                    notification_bridge = await initialize_notification_bridge(**bridge_kwargs)
                except Exception as exc:
                    # The exception text can carry a broker or database URL: class name only.
                    logger.warning("notification_bridge_unavailable", error_type=type(exc).__name__)
                    runtime.register_readiness_check(
                        EVENT_BRIDGE_CHECK, _event_bridge_unavailable, critical=True)
            if notification_bridge is not None:
                runtime.register_shutdown("notification_bridge", notification_bridge.stop)
                app.state.notification_bridge = notification_bridge
                logger.info("notification_bridge_started")

        logger.info("safety_service_started", environment=settings.environment)
        yield


def create_application() -> FastAPI:
    """Create the safety service app with the shared IF-01 runtime."""
    return create_service_app(
        SPEC,
        lifespan=lifespan,
        routers=[(router, "/api/v1/safety")],
        title="Solace-AI Safety Service",
    )


app = create_application()


def run_server() -> None:
    """Run the safety service server."""
    import uvicorn
    settings = SafetyServiceAppSettings()
    uvicorn.run(
        "services.safety_service.src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    run_server()
