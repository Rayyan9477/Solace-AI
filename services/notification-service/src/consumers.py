"""
Solace-AI Notification Service - Kafka Event Consumers.

Consumes safety events from Kafka and triggers appropriate notifications
for crisis alerts and escalations.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import shared event infrastructure
import os

try:
    from solace_security.service_auth import ServiceTokenManager, ServiceIdentity
    _SERVICE_AUTH_AVAILABLE = True
except ImportError:
    _SERVICE_AUTH_AVAILABLE = False

try:
    from solace_events.schemas import (
        BaseEvent,
        CrisisDetectedEvent,
        EscalationTriggeredEvent,
        SafetyAssessmentEvent,
        CrisisLevel,
    )
    from solace_events.consumer import EventConsumer, create_consumer
    from solace_events.config import KafkaSettings, ConsumerSettings, SolaceTopic
    from solace_events.dead_letter import DeadLetterRecord, DeadLetterStore
    _KAFKA_AVAILABLE = True
except ImportError:
    _KAFKA_AVAILABLE = False
    CrisisDetectedEvent = None
    EscalationTriggeredEvent = None
    EventConsumer = None
    structlog.get_logger(__name__).error(
        "kafka_import_failed", package="solace_events",
        hint="Install solace_events for Kafka consumer support")
    if os.environ.get("ENVIRONMENT", "").lower() == "production":
        raise RuntimeError("solace_events package required in production")

# Import domain service
from .domain import (
    NotificationService,
    NotificationRequest,
    NotificationRecipient,
    NotificationPriority,
    ChannelType,
    TemplateType,
)

logger = structlog.get_logger(__name__)


@runtime_checkable
class DeadLetterStoreSeam(Protocol):
    """Durable-persistence seam for failed (dead-lettered) events.

    Deliberately structural so the shared ``PostgresDLQStore`` (from
    ``solace_events.postgres_stores``) — whose ``save(DeadLetterRecord)`` matches
    this shape — can be injected for staging/production WITHOUT any consumer code
    change. The default is the in-memory ``DeadLetterStore``, so existing tests
    and dev keep working unchanged.
    """

    async def save(self, record: "DeadLetterRecord") -> None: ...


class UserServiceSettings(BaseSettings):
    """Configuration for User Service integration."""

    user_service_url: str = Field(
        default="http://localhost:8001",
        description="URL of the User Service for on-call clinician lookup",
    )
    request_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="HTTP request timeout in seconds",
    )
    fallback_oncall_email: str = Field(
        default="oncall@solace-ai.com",
        description="Fallback email when user service is unavailable",
    )

    model_config = SettingsConfigDict(
        env_prefix="NOTIFICATION_",
        env_file=".env",
        extra="ignore",
    )


class SafetyEventConsumer:
    """
    Consumes safety events from Kafka and triggers notifications.

    Handles CrisisDetectedEvent and EscalationTriggeredEvent to send
    urgent notifications to clinicians and emergency contacts.
    """

    def __init__(
        self,
        notification_service: NotificationService,
        kafka_settings: "KafkaSettings | None" = None,
        consumer_settings: "ConsumerSettings | None" = None,
        use_mock: bool = False,
        user_service_settings: UserServiceSettings | None = None,
        dlq_store: "DeadLetterStoreSeam | None" = None,
    ) -> None:
        self._user_service_settings = user_service_settings or UserServiceSettings()
        # MVP-287: set every attribute stop() and the lifecycle properties read BEFORE the
        # early return below, so a half-initialised consumer (no Kafka) stops cleanly.
        self._notification_service = notification_service
        self._running = False
        self._stopped = False
        self._consume_task: asyncio.Task | None = None
        # M1-gate R2 (P0): without a dead-letter handler, a handler that raises (e.g.
        # a crisis alert that failed on every channel, REV-35) is marked FAILED and its
        # offset is COMMITTED — the event is silently dropped, not retried. Capture
        # failed events durably so a crisis alert is never lost without a trace.
        self._failed_events: list[dict[str, Any]] = []

        if not _KAFKA_AVAILABLE:
            logger.warning("kafka_not_available", reason="solace_events not installed")
            self._consumer = None
            self._dlq_store: "DeadLetterStoreSeam | None" = dlq_store
            return

        self._consumer = create_consumer(
            group_id="notification-service-safety-consumer",
            kafka_settings=kafka_settings,
            consumer_settings=consumer_settings,
            use_mock=use_mock,
        )

        # Register handlers
        self._consumer.register_handler("safety.crisis.detected", self._handle_crisis_detected)
        self._consumer.register_handler("safety.escalation.triggered", self._handle_escalation_triggered)
        self._consumer.register_handler("safety.assessment.completed", self._handle_assessment_completed)

        # Durable DLQ seam: a failed crisis/escalation alert that only lives in the
        # in-memory list above is LOST on a worker restart. Persist each failure
        # through an injectable store so a durable PostgresDLQStore CAN be wired in
        # staging/prod without code change; the default in-memory store keeps dev
        # and existing tests behaving exactly as before.
        self._dlq_store = dlq_store if dlq_store is not None else DeadLetterStore()
        self._consumer.set_dead_letter_handler(self._handle_dead_letter)

        # Service auth for inter-service calls
        self._token_manager: ServiceTokenManager | None = None
        if _SERVICE_AUTH_AVAILABLE:
            try:
                self._token_manager = ServiceTokenManager()
            except Exception:
                logger.warning("service_token_manager_init_failed", hint="inter-service calls will be unauthenticated")

        logger.info("safety_event_consumer_initialized", use_mock=use_mock)

    def _get_service_auth_headers(self) -> dict[str, str]:
        """Get auth headers for inter-service HTTP calls."""
        if self._token_manager is None:
            return {}
        try:
            creds = self._token_manager.get_or_create_token(
                ServiceIdentity.NOTIFICATION.value,
                ServiceIdentity.USER.value,
            )
            return {
                "Authorization": f"Bearer {creds.token}",
                "X-Service-Name": ServiceIdentity.NOTIFICATION.value,
            }
        except Exception:
            logger.warning("service_auth_header_failed")
            return {}

    async def start(self) -> None:
        """Start consuming safety events."""
        if not self._consumer:
            logger.error("cannot_start_consumer", reason="consumer not initialized, Kafka unavailable")
            raise RuntimeError(
                "SafetyEventConsumer cannot start: Kafka consumer not initialized. "
                "Ensure solace_events is installed and Kafka is reachable."
            )

        await self._consumer.start([SolaceTopic.SAFETY])
        self._stopped = False
        self._running = True
        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info("safety_event_consumer_started", topic="solace.safety")

    @property
    def is_running(self) -> bool:
        """True while the consume loop was started and has not been stopped or exited."""
        return self._running and self._consume_task is not None and not self._consume_task.done()

    @property
    def consume_task(self) -> asyncio.Task | None:
        """The consume-loop task, or None before start() and after stop()."""
        return self._consume_task

    async def stop(self) -> None:
        """Stop consuming events. Idempotent: a second call (or a call before start) is a no-op."""
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        task, self._consume_task = self._consume_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._consumer:
            await self._consumer.stop()
        logger.info("safety_event_consumer_stopped")

    async def _consume_loop(self) -> None:
        """Main consumption loop."""
        if self._consumer:
            await self._consumer.consume_loop()

    async def _handle_crisis_detected(self, event: "CrisisDetectedEvent") -> None:
        """Handle crisis detected event by sending urgent notifications."""
        logger.info(
            "processing_crisis_event",
            event_id=str(event.metadata.event_id),
            user_id=str(event.user_id),
            crisis_level=event.crisis_level.value if hasattr(event.crisis_level, "value") else event.crisis_level,
        )

        try:
            # Determine notification priority based on crisis level
            priority = self._crisis_level_to_priority(event.crisis_level)

            # Build notification variables
            variables = {
                "crisis_level": event.crisis_level.value if hasattr(event.crisis_level, "value") else str(event.crisis_level),
                "user_id": str(event.user_id),
                "session_id": str(event.session_id) if event.session_id else "N/A",
                "trigger_indicators": ", ".join(event.trigger_indicators[:5]),  # Limit to first 5
                "detection_layer": str(event.detection_layer),
                "confidence": f"{float(event.confidence) * 100:.1f}%",
                "escalation_action": event.escalation_action,
                "requires_human_review": "Yes" if event.requires_human_review else "No",
                "timestamp": event.metadata.timestamp.isoformat(),
            }

            # Send notifications based on crisis level
            if event.crisis_level in (CrisisLevel.CRITICAL, CrisisLevel.HIGH):
                await self._send_crisis_notification(
                    event_id=event.metadata.event_id,
                    user_id=event.user_id,
                    priority=priority,
                    variables=variables,
                    send_sms=True,
                    send_push=True,
                )
            else:
                # Lower priority - email only
                await self._send_crisis_notification(
                    event_id=event.metadata.event_id,
                    user_id=event.user_id,
                    priority=priority,
                    variables=variables,
                    send_sms=False,
                    send_push=False,
                )

            logger.info(
                "crisis_notification_sent",
                event_id=str(event.metadata.event_id),
                priority=priority.value,
            )

        except Exception as e:
            logger.error(
                "crisis_notification_failed",
                event_id=str(event.metadata.event_id),
                error=str(e),
            )
            raise

    async def _handle_dead_letter(self, event: Any, error: str) -> None:
        """Durably capture an event whose handler failed, so it is never silently lost.

        M1-gate R2 (P0): a crisis/escalation alert that fails delivery on every channel
        (REV-35 raise) lands here instead of being committed-and-dropped. For crisis
        and escalation events this is a CRITICAL, alertable condition — a clinician was
        NOT notified. The captured record is available for retry/recovery (a durable
        DLQ topic / Postgres store is wired at B.3 on real Kafka).
        """
        event_type = getattr(event, "event_type", None) or type(event).__name__
        event_id = str(getattr(getattr(event, "metadata", None), "event_id", "unknown"))
        record = {"event_type": event_type, "event_id": event_id, "error": error}
        self._failed_events.append(record)
        # Durable persistence: write the failure through the injected DLQ store so it
        # survives a worker restart. A durable PostgresDLQStore keeps the record; the
        # in-memory default keeps existing behavior. Best-effort — persistence must
        # never mask the CRITICAL alert below.
        await self._persist_dead_letter(event, error)
        is_crisis = "crisis" in str(event_type).lower() or "escalation" in str(event_type).lower()
        (logger.critical if is_crisis else logger.error)(
            "notification_dead_letter",
            event_type=event_type,
            event_id=event_id,
            error=error,
            clinician_alerted=False,
        )

    async def _persist_dead_letter(self, event: Any, error: str) -> None:
        """Persist a failed event to the durable DLQ store (best-effort).

        Builds a ``DeadLetterRecord`` from the failed event and hands it to the
        injected store (``PostgresDLQStore`` in staging/prod, in-memory default in
        dev/tests). Any persistence failure is logged but never propagated — the
        crisis/escalation CRITICAL log must still fire.
        """
        store = getattr(self, "_dlq_store", None)
        if store is None:
            return
        try:
            dlq_record = DeadLetterRecord.from_failed_event(
                event=event,
                original_topic=SolaceTopic.SAFETY.value,
                consumer_group="notification-service-safety-consumer",
                error=error if isinstance(error, Exception) else RuntimeError(str(error)),
            )
            await store.save(dlq_record)
        except Exception as persist_err:
            logger.error(
                "dead_letter_persist_failed",
                event_type=getattr(event, "event_type", None) or type(event).__name__,
                error=str(persist_err),
            )

    async def _handle_escalation_triggered(self, event: "EscalationTriggeredEvent") -> None:
        """Handle escalation triggered event by notifying assigned clinician."""
        logger.info(
            "processing_escalation_event",
            event_id=str(event.metadata.event_id),
            user_id=str(event.user_id),
            priority=event.priority,
        )

        try:
            # Map escalation priority to notification priority
            priority = self._escalation_priority_to_notification_priority(event.priority)

            # Build notification variables
            variables = {
                "escalation_reason": event.escalation_reason,
                "priority": event.priority,
                "user_id": str(event.user_id),
                "session_id": str(event.session_id) if event.session_id else "N/A",
                "assigned_clinician_id": str(event.assigned_clinician_id) if event.assigned_clinician_id else "Unassigned",
                "timestamp": event.metadata.timestamp.isoformat(),
            }

            # Always use multi-channel for escalations
            await self._send_escalation_notification(
                event_id=event.metadata.event_id,
                user_id=event.user_id,
                clinician_id=event.assigned_clinician_id,
                priority=priority,
                variables=variables,
            )

            logger.info(
                "escalation_notification_sent",
                event_id=str(event.metadata.event_id),
                clinician_id=str(event.assigned_clinician_id) if event.assigned_clinician_id else None,
            )

        except Exception as e:
            logger.error(
                "escalation_notification_failed",
                event_id=str(event.metadata.event_id),
                error=str(e),
            )
            raise

    async def _handle_assessment_completed(self, event: "SafetyAssessmentEvent") -> None:
        """Handle safety assessment completed event for monitoring."""
        # Only notify for elevated risk levels
        if event.risk_level not in (CrisisLevel.HIGH, CrisisLevel.CRITICAL, CrisisLevel.ELEVATED):
            logger.debug(
                "assessment_below_threshold",
                event_id=str(event.metadata.event_id),
                risk_level=event.risk_level.value if hasattr(event.risk_level, "value") else event.risk_level,
            )
            return

        logger.info(
            "processing_elevated_assessment",
            event_id=str(event.metadata.event_id),
            risk_level=event.risk_level.value if hasattr(event.risk_level, "value") else event.risk_level,
        )

        # Send monitoring notification for elevated assessments
        variables = {
            "risk_level": event.risk_level.value if hasattr(event.risk_level, "value") else str(event.risk_level),
            "risk_score": f"{float(event.risk_score) * 100:.1f}%",
            "user_id": str(event.user_id),
            "session_id": str(event.session_id) if event.session_id else "N/A",
            "recommended_action": event.recommended_action,
            "detection_layer": str(event.detection_layer),
            "timestamp": event.metadata.timestamp.isoformat(),
        }

        await self._send_monitoring_notification(
            event_id=event.metadata.event_id,
            user_id=event.user_id,
            variables=variables,
        )

    async def _send_crisis_notification(
        self,
        event_id: UUID,
        user_id: UUID,
        priority: NotificationPriority,
        variables: dict[str, Any],
        send_sms: bool = True,
        send_push: bool = True,
    ) -> None:
        """Send crisis notification through configured channels."""
        channels = [ChannelType.EMAIL]
        if send_sms:
            channels.append(ChannelType.SMS)
        if send_push:
            channels.append(ChannelType.PUSH)

        # Get on-call clinicians (in production, fetch from user service)
        recipients = await self._get_oncall_clinicians()

        if not recipients:
            logger.warning("no_oncall_clinicians", event_id=str(event_id))
            # Fall back to configured fallback email
            recipients = [
                NotificationRecipient(
                    email=self._user_service_settings.fallback_oncall_email,
                    name="On-Call Team (fallback)",
                )
            ]

        request = NotificationRequest(
            template_type=TemplateType.CRISIS_ALERT,
            recipients=recipients,
            channels=channels,
            variables=variables,
            priority=priority,
            correlation_id=event_id,
        )

        result = await self._notification_service.send_notification(request)
        logger.info(
            "crisis_notification_result",
            event_id=str(event_id),
            successful=result.successful_deliveries,
            failed=result.failed_deliveries,
        )
        # M1-gate P0: a crisis alert that reached NO channel must not be silently
        # committed. Raise so the Kafka consumer marks the message FAILED and it
        # goes to the DLQ / retry path — a crisis notification must never be dropped.
        if result.successful_deliveries == 0:
            raise RuntimeError(
                f"Crisis notification for event {event_id} failed on all channels "
                f"({result.failed_deliveries} failed) — surfacing for DLQ/retry"
            )

    async def _send_escalation_notification(
        self,
        event_id: UUID,
        user_id: UUID,
        clinician_id: UUID | None,
        priority: NotificationPriority,
        variables: dict[str, Any],
    ) -> None:
        """Send escalation notification to assigned clinician."""
        recipients = []

        if clinician_id:
            # Fetch clinician contact from user service
            clinician_contact = await self._get_clinician_contact(clinician_id)
            if clinician_contact:
                recipients.append(clinician_contact)
            else:
                logger.warning(
                    "clinician_contact_not_found",
                    clinician_id=str(clinician_id),
                    fallback="oncall_team",
                )
        else:
            # No assigned clinician - notify on-call team
            recipients = await self._get_oncall_clinicians()

        if not recipients:
            recipients = [
                NotificationRecipient(
                    email=self._user_service_settings.fallback_oncall_email,
                    name="Escalation Team (fallback)",
                )
            ]

        # Escalations always go through all channels
        request = NotificationRequest(
            template_type=TemplateType.ESCALATION_ALERT,
            recipients=recipients,
            channels=[ChannelType.EMAIL, ChannelType.SMS, ChannelType.PUSH],
            variables=variables,
            priority=priority,
            correlation_id=event_id,
        )

        result = await self._notification_service.send_notification(request)
        logger.info(
            "escalation_notification_result",
            event_id=str(event_id),
            successful=result.successful_deliveries,
            failed=result.failed_deliveries,
        )
        # M1-gate P0: an escalation that reached no channel must go to DLQ/retry,
        # not be silently committed — a clinician escalation must not be dropped.
        if result.successful_deliveries == 0:
            raise RuntimeError(
                f"Escalation notification for event {event_id} failed on all channels "
                f"({result.failed_deliveries} failed) — surfacing for DLQ/retry"
            )

    async def _send_monitoring_notification(
        self,
        event_id: UUID,
        user_id: UUID,
        variables: dict[str, Any],
    ) -> None:
        """Send monitoring notification for elevated risk assessments."""
        # Monitoring notifications are email-only, normal priority
        recipients = await self._get_monitoring_team()

        if not recipients:
            recipients = [
                NotificationRecipient(
                    email=self._user_service_settings.fallback_oncall_email,
                    name="Monitoring Team (fallback)",
                )
            ]

        request = NotificationRequest(
            template_type=TemplateType.RISK_ALERT,
            recipients=recipients,
            channels=[ChannelType.EMAIL],
            variables=variables,
            priority=NotificationPriority.NORMAL,
            correlation_id=event_id,
        )

        result = await self._notification_service.send_notification(request)
        logger.info(
            "monitoring_notification_result",
            event_id=str(event_id),
            successful=result.successful_deliveries,
            failed=result.failed_deliveries,
        )

    async def _get_oncall_clinicians(self) -> list[NotificationRecipient]:
        """
        Get list of on-call clinicians for crisis notifications.

        Calls the User Service to get currently on-call clinicians
        with their contact information.

        Returns:
            List of NotificationRecipient objects for on-call clinicians.
            Returns empty list if service is unavailable (triggers fallback).
        """
        try:
            url = f"{self._user_service_settings.user_service_url}/api/v1/users/on-call-clinicians"
            timeout = httpx.Timeout(self._user_service_settings.request_timeout)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=self._get_service_auth_headers())
                response.raise_for_status()
                data = response.json()

            clinicians = data.get("clinicians", [])
            recipients = []

            for clinician in clinicians:
                recipients.append(
                    NotificationRecipient(
                        user_id=UUID(clinician["user_id"]) if clinician.get("user_id") else None,
                        email=clinician.get("email", ""),
                        name=clinician.get("display_name", "Clinician"),
                        phone=clinician.get("phone_number"),
                    )
                )

            logger.info(
                "on_call_clinicians_fetched",
                count=len(recipients),
            )
            return recipients

        except httpx.HTTPStatusError as e:
            logger.warning(
                "user_service_http_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            return []
        except httpx.RequestError as e:
            logger.warning(
                "user_service_request_error",
                error=str(e),
            )
            return []
        except Exception as e:
            logger.error(
                "on_call_clinicians_fetch_failed",
                error=str(e),
            )
            return []

    async def _get_clinician_contact(self, clinician_id: UUID) -> NotificationRecipient | None:
        """Fetch contact info for a specific clinician from User Service."""
        try:
            url = f"{self._user_service_settings.user_service_url}/api/v1/users/{clinician_id}"
            timeout = httpx.Timeout(self._user_service_settings.request_timeout)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=self._get_service_auth_headers())
                response.raise_for_status()
                data = response.json()

            if not data.get("email"):
                return None

            return NotificationRecipient(
                user_id=clinician_id,
                email=data["email"],
                name=data.get("display_name", "Clinician"),
                phone=data.get("phone_number"),
            )
        except Exception as e:
            logger.warning(
                "clinician_contact_lookup_failed",
                clinician_id=str(clinician_id),
                error=str(e),
            )
            return None

    async def _get_monitoring_team(self) -> list[NotificationRecipient]:
        """
        Get list of monitoring team members.

        For now, returns the on-call clinicians as the monitoring team.
        In a full implementation, this would fetch users with a monitoring role.

        Returns:
            List of NotificationRecipient objects for monitoring team.
        """
        # For monitoring, we use the same on-call clinicians
        # In production, this could be a separate role/query
        return await self._get_oncall_clinicians()

    def _crisis_level_to_priority(self, crisis_level: "CrisisLevel") -> NotificationPriority:
        """Map crisis level to notification priority."""
        if not _KAFKA_AVAILABLE:
            return NotificationPriority.HIGH

        level_value = crisis_level.value if hasattr(crisis_level, "value") else str(crisis_level)
        mapping = {
            "CRITICAL": NotificationPriority.CRITICAL,
            "HIGH": NotificationPriority.HIGH,
            "ELEVATED": NotificationPriority.HIGH,
            "LOW": NotificationPriority.NORMAL,
            "NONE": NotificationPriority.LOW,
        }
        return mapping.get(level_value.upper(), NotificationPriority.HIGH)

    def _escalation_priority_to_notification_priority(self, priority: str) -> NotificationPriority:
        """Map escalation priority to notification priority."""
        mapping = {
            "CRITICAL": NotificationPriority.CRITICAL,
            "HIGH": NotificationPriority.HIGH,
            "MEDIUM": NotificationPriority.HIGH,
            "LOW": NotificationPriority.NORMAL,
        }
        return mapping.get(priority.upper(), NotificationPriority.HIGH)


# Module-level singleton
_safety_consumer: SafetyEventConsumer | None = None


def get_safety_consumer() -> SafetyEventConsumer | None:
    """Get the singleton safety event consumer."""
    return _safety_consumer


async def initialize_safety_consumer(
    notification_service: NotificationService,
    kafka_settings: "KafkaSettings | None" = None,
    use_mock: bool = False,
    user_service_settings: UserServiceSettings | None = None,
    dlq_store: "DeadLetterStoreSeam | None" = None,
) -> SafetyEventConsumer:
    """Initialize and start the safety event consumer.

    Pass ``dlq_store`` (e.g. a ``PostgresDLQStore``) to durably persist failed
    crisis/escalation alerts across worker restarts; omit it to keep the
    in-memory default used in dev and tests.
    """
    global _safety_consumer
    consumer = SafetyEventConsumer(
        notification_service=notification_service,
        kafka_settings=kafka_settings,
        use_mock=use_mock,
        user_service_settings=user_service_settings,
        dlq_store=dlq_store,
    )
    # MVP-287: publish the consumer to the module global only once start() returned, so a
    # failed start never leaves a dead consumer behind for get_safety_consumer().
    try:
        await consumer.start()
    except BaseException:
        # Release whatever a partial start opened (the broker connection) before re-raising.
        try:
            await consumer.stop()
        except Exception as exc:
            logger.warning("safety_consumer_cleanup_failed", error_type=type(exc).__name__)
        raise
    _safety_consumer = consumer
    return consumer


async def shutdown_safety_consumer() -> None:
    """Stop the safety event consumer."""
    global _safety_consumer
    if _safety_consumer:
        await _safety_consumer.stop()
        _safety_consumer = None
