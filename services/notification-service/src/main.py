"""
Solace-AI Notification Service - FastAPI Application.

Multi-channel notification service supporting email, SMS, and push notifications.

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

_notification_service: "NotificationService | None" = None


class ServiceConfig(BaseSettings):
    """Service configuration."""
    name: str = Field(default="notification-service")
    env: Literal["development", "staging", "production"] = Field(default="development")
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


def get_notification_service() -> "NotificationService":
    """Get the global notification service instance."""
    if _notification_service is None:
        raise RuntimeError("Notification service not initialized")
    return _notification_service


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _notification_service

    settings = NotificationServiceSettings.load()

    # Configure structured logging with PHI sanitizer
    try:
        from solace_security.phi_protection import phi_sanitizer_processor
        _phi_processor = phi_sanitizer_processor
    except ImportError:
        if settings.service.env == "production":
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
    if settings.service.env == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), settings.service.log_level.upper(), __import__("logging").INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger.info("notification_service_starting",
               service=settings.service.name,
               env=settings.service.env,
               email_enabled=settings.email.enabled,
               sms_enabled=settings.sms.enabled,
               push_enabled=settings.push.enabled,
               kafka_enabled=settings.kafka.enabled)

    _notification_service = _create_notification_service(settings)
    logger.info("notification_service_ready")

    # Start Kafka safety event consumer if enabled
    safety_consumer = None
    if settings.kafka.enabled:
        try:
            from .consumers import initialize_safety_consumer, shutdown_safety_consumer
            # Build Kafka settings if solace_events is available
            kafka_settings = None
            try:
                from solace_events.config import KafkaSettings
                kafka_settings = KafkaSettings(
                    bootstrap_servers=settings.kafka.bootstrap_servers,
                )
            except ImportError:
                logger.warning("solace_events_not_available", reason="Cannot configure Kafka settings")

            safety_consumer = await initialize_safety_consumer(
                notification_service=_notification_service,
                kafka_settings=kafka_settings,
                use_mock=settings.kafka.use_mock,
            )
            logger.info("safety_event_consumer_started")
        except Exception as e:
            logger.error("safety_consumer_start_failed", error=str(e))

    yield

    # Stop Kafka consumer
    if safety_consumer:
        try:
            from .consumers import shutdown_safety_consumer
            await shutdown_safety_consumer()
            logger.info("safety_event_consumer_stopped")
        except Exception as e:
            logger.error("safety_consumer_stop_failed", error=str(e))

    logger.info("notification_service_shutdown")
    _notification_service = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = NotificationServiceSettings.load()

    app = FastAPI(
        title="Solace-AI Notification Service",
        description="Multi-channel notification service for email, SMS, and push notifications",
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

    from .api import router as notification_router
    app.include_router(notification_router)

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
        if _notification_service is None:
            return {"status": "not_ready", "reason": "service_not_initialized"}
        return {"status": "ready"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = NotificationServiceSettings.load()
    uvicorn.run(
        "src.main:app",
        host=settings.service.host,
        port=settings.service.port,
        reload=settings.service.env == "development",
        log_level=settings.service.log_level.lower(),
    )
