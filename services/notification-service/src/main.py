"""
Solace-AI Notification Service - FastAPI Application.

Multi-channel notification service supporting email, SMS, and push notifications.

The app is built by the shared IF-01 runtime (``create_service_app``), and the lifespan is
bracketed by ``service_runtime``. The runtime owns logging, the start-up guards, the Postgres
pool, Redis, ``/live``, ``/ready``, ``/health``, ``/metrics``, the error responses and the
request middleware. This module composes the notification domain on top of it and registers
every service-owned stop with ``runtime.register_shutdown`` (WS-13.2, MVP-155, MVP-287).

Architecture Layer: Infrastructure
Principles: 12-Factor App, Dependency Injection, Configuration Externalization
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from solace_infrastructure.runtime import (
    ServiceRuntimeSpec,
    create_service_app,
    is_strict_environment,
    service_runtime,
)

from .api import router as notification_router

if TYPE_CHECKING:
    from solace_infrastructure.runtime.bootstrap import ServiceRuntime

    from .consumers import SafetyEventConsumer
    from .domain.service import NotificationService

logger = structlog.get_logger(__name__)


class ServiceConfig(BaseSettings):
    """Service configuration."""
    name: str = Field(default="notification-service")
    env: Literal["development", "test", "staging", "production"] = Field(default="development")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8003, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    model_config = SettingsConfigDict(env_prefix="NOTIFICATION_SERVICE_", env_file=".env", extra="ignore")


class EmailConfig(BaseSettings):
    """Email channel configuration."""
    enabled: bool = Field(default=True)
    smtp_host: str = Field(default="localhost")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str = Field(default="")
    smtp_password: str = Field(default="")
    use_tls: bool = Field(default=True)
    from_email: str = Field(default="noreply@solace-ai.com")
    from_name: str = Field(default="Solace-AI")
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    model_config = SettingsConfigDict(env_prefix="EMAIL_", env_file=".env", extra="ignore")


class SMSConfig(BaseSettings):
    """SMS channel configuration (Twilio-compatible)."""
    enabled: bool = Field(default=False)
    provider_url: str = Field(default="https://api.twilio.com/2010-04-01")
    account_sid: str = Field(default="")
    auth_token: str = Field(default="")
    from_number: str = Field(default="")
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    model_config = SettingsConfigDict(env_prefix="SMS_", env_file=".env", extra="ignore")


class PushConfig(BaseSettings):
    """Push notification configuration (Firebase-compatible)."""
    enabled: bool = Field(default=False)
    firebase_url: str = Field(default="https://fcm.googleapis.com/fcm/send")
    server_key: str = Field(default="")
    project_id: str = Field(default="")
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    model_config = SettingsConfigDict(env_prefix="PUSH_", env_file=".env", extra="ignore")


class KafkaConfig(BaseSettings):
    """Kafka configuration for event consumption."""
    enabled: bool = Field(default=False)
    bootstrap_servers: str = Field(default="localhost:29092")
    consumer_group_id: str = Field(default="notification-service-safety-consumer")
    use_mock: bool = Field(default=False)

    model_config = SettingsConfigDict(env_prefix="KAFKA_", env_file=".env", extra="ignore")


class NotificationServiceSettings(BaseSettings):
    """Aggregate notification service settings."""
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    sms: SMSConfig = Field(default_factory=SMSConfig)
    push: PushConfig = Field(default_factory=PushConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @staticmethod
    def load() -> "NotificationServiceSettings":
        """Load settings from environment."""
        return NotificationServiceSettings()


def _create_notification_service(settings: NotificationServiceSettings) -> "NotificationService":
    """Create and configure the notification service."""
    from .domain.channels import EmailConfig as DomainEmailConfig
    from .domain.channels import SMSConfig as DomainSMSConfig
    from .domain.channels import PushConfig as DomainPushConfig
    from .domain.service import create_notification_service

    email_config = DomainEmailConfig(
        enabled=settings.email.enabled,
        smtp_host=settings.email.smtp_host,
        smtp_port=settings.email.smtp_port,
        smtp_username=settings.email.smtp_username,
        smtp_password=settings.email.smtp_password,
        use_tls=settings.email.use_tls,
        from_email=settings.email.from_email,
        from_name=settings.email.from_name,
        max_retries=settings.email.max_retries,
        timeout_seconds=settings.email.timeout_seconds,
    ) if settings.email.enabled else None

    sms_config = DomainSMSConfig(
        enabled=settings.sms.enabled,
        provider_url=settings.sms.provider_url,
        account_sid=settings.sms.account_sid,
        auth_token=settings.sms.auth_token,
        from_number=settings.sms.from_number,
        max_retries=settings.sms.max_retries,
        timeout_seconds=settings.sms.timeout_seconds,
    ) if settings.sms.enabled else None

    push_config = DomainPushConfig(
        enabled=settings.push.enabled,
        firebase_url=settings.push.firebase_url,
        server_key=settings.push.server_key,
        project_id=settings.push.project_id,
        max_retries=settings.push.max_retries,
        timeout_seconds=settings.push.timeout_seconds,
    ) if settings.push.enabled else None

    return create_notification_service(
        email_config=email_config,
        sms_config=sms_config,
        push_config=push_config,
    )


SETTINGS = NotificationServiceSettings.load()

SPEC = ServiceRuntimeSpec(
    service_name="notification-service",
    environment=SETTINGS.service.env,
    version="1.0.0",
    log_level=SETTINGS.service.log_level,
    uses_postgres=True,
    uses_redis=True,
    event_plane="publish_consume",
    handles_phi=False,
    audited=True,
    rate_limited=False,
    edge=False,
)

# The readiness check (and background task) that report the crisis safety-event consumer.
SAFETY_CONSUMER_CHECK = "safety_consumer"
# The readiness check that reports a delivery-event bridge that failed to start (outside strict).
EVENT_BRIDGE_CHECK = "event_bridge"


def _is_strict(settings: Any) -> bool:
    """Strict when the service's own environment OR the bare ``ENVIRONMENT`` is staging/production.

    A blank or non-strict ``NOTIFICATION_SERVICE_ENV`` under a strict bare ``ENVIRONMENT`` cannot
    switch the crisis-consumer guard off while the runtime itself boots strict.
    """
    return is_strict_environment(getattr(settings.service, "env", "") or "") or is_strict_environment()


def _require_crisis_consumer_in_strict(settings: Any) -> None:
    """M1-gate R2 (P0), D-14: the safety-event consumer must run in staging and production.

    Consumer start is gated behind ``settings.kafka.enabled`` (default False). Without this
    guard a strict deploy that forgets to enable Kafka boots "healthy" and consumes zero
    crisis events, so crisis alerts are never delivered. A mock consumer is refused in strict
    too: it reads from an in-process queue, never from the broker.

    Raises:
        RuntimeError: a strict environment with Kafka disabled or with ``KAFKA_USE_MOCK`` set.
    """
    if not _is_strict(settings):
        return
    if not settings.kafka.enabled:
        raise RuntimeError(
            "the safety-event consumer is required in staging and production; set KAFKA_ENABLED=true"
        )
    if getattr(settings.kafka, "use_mock", False):
        raise RuntimeError(
            "a mock safety-event consumer is not allowed in staging and production; unset KAFKA_USE_MOCK"
        )


async def _event_bridge_unavailable() -> tuple[bool, str]:
    """Readiness check for a delivery-event bridge that did not start: never ready."""
    return False, "unavailable"


async def _start_event_bridge(runtime: "ServiceRuntime", initialize: Any) -> Any:
    """Start the delivery-event bridge on the runtime's pool.

    In a strict environment a start failure propagates (boot fails). Elsewhere (dev and test
    without a broker) it is logged by class name only and a critical readiness check keeps
    ``/ready`` at 503, so the service still boots and serves ``/live``.
    """
    if _is_strict(SETTINGS):
        return await initialize(postgres_pool=runtime.pg_pool)
    try:
        return await initialize(postgres_pool=runtime.pg_pool)
    except Exception as exc:
        logger.warning("notification_event_bridge_unavailable", error_type=type(exc).__name__)
        runtime.register_readiness_check(EVENT_BRIDGE_CHECK, _event_bridge_unavailable, critical=True)
        return None


async def _safety_consumer_unavailable() -> tuple[bool, str]:
    """Readiness check for a safety-event consumer that did not start: never ready."""
    return False, "unavailable"


async def _start_safety_consumer(
    runtime: "ServiceRuntime", notification_service: "NotificationService"
) -> "SafetyEventConsumer | None":
    """Start the crisis safety-event consumer and register it with the runtime.

    Returns None when Kafka is disabled. In a strict environment a start failure propagates
    (boot fails); elsewhere it is logged (class name only: the text can carry a broker URL)
    and a critical readiness check keeps ``/ready`` at 503.
    """
    if not SETTINGS.kafka.enabled:
        return None
    from solace_events.config import KafkaSettings

    from .consumers import initialize_safety_consumer, shutdown_safety_consumer

    kwargs: dict[str, Any] = {
        "notification_service": notification_service,
        "kafka_settings": KafkaSettings(bootstrap_servers=SETTINGS.kafka.bootstrap_servers),
        "use_mock": SETTINGS.kafka.use_mock,
    }
    if _is_strict(SETTINGS):
        consumer = await initialize_safety_consumer(**kwargs)
    else:
        try:
            consumer = await initialize_safety_consumer(**kwargs)
        except Exception as exc:
            logger.warning("safety_consumer_unavailable", error_type=type(exc).__name__)
            runtime.register_readiness_check(
                SAFETY_CONSUMER_CHECK, _safety_consumer_unavailable, critical=True)
            return None

    async def _safety_consumer_ready() -> tuple[bool, str]:
        running = consumer.is_running
        return running, "running" if running else "stopped"

    # The task first, then its graceful stop: closers run in reverse order, so the graceful
    # stop runs before the runtime's cancel of the task (LIF-2).
    runtime.register_background_task(SAFETY_CONSUMER_CHECK, consumer.consume_task)
    runtime.register_shutdown("safety_consumer_stop", shutdown_safety_consumer)
    runtime.register_readiness_check(SAFETY_CONSUMER_CHECK, _safety_consumer_ready, critical=True)
    logger.info("safety_event_consumer_started")
    return consumer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compose the notification service inside the IF-01 runtime bracket."""
    async with service_runtime(app, SPEC) as runtime:
        logger.info("notification_service_starting",
                    service=SETTINGS.service.name,
                    env=SETTINGS.service.env,
                    email_enabled=SETTINGS.email.enabled,
                    sms_enabled=SETTINGS.sms.enabled,
                    push_enabled=SETTINGS.push.enabled,
                    kafka_enabled=SETTINGS.kafka.enabled)

        # REV-12: PHI-encryption exemption (documented). This service persists no
        # ClinicalBase entities: Postgres is used only for the durable event outbox
        # (operational rows). configure_phi_encryption() only affects ClinicalBase, so
        # it is intentionally NOT configured here (least privilege: no PHI master key).
        # Enforced by tests/integration/test_phi_encryption_activation.py.
        notification_service = _create_notification_service(SETTINGS)
        app.state.notification_service = notification_service

        # Kafka event bridge for notification delivery events, through the durable outbox on
        # the runtime's pool.
        from .events import get_event_publisher
        from .infrastructure.event_bridge import initialize_event_bridge

        event_bridge = await _start_event_bridge(runtime, initialize_event_bridge)
        if event_bridge is not None:
            runtime.register_shutdown("event_bridge", event_bridge.stop)
            get_event_publisher().register_callback(event_bridge.bridge_event)

        # M1-gate R2 (P0): fail loud if the crisis consumer is disabled in a strict environment.
        _require_crisis_consumer_in_strict(SETTINGS)
        await _start_safety_consumer(runtime, notification_service)
        logger.info("notification_service_ready")
        yield


app = create_service_app(
    SPEC,
    lifespan=lifespan,
    routers=[(notification_router, "")],
    title="Solace-AI Notification Service",
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
