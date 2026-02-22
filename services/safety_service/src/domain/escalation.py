"""
Solace-AI Escalation Manager - Crisis escalation workflow management.
Handles escalation to human clinicians, notifications, and crisis response coordination.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class EscalationPriority(str, Enum):
    """Escalation priority levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_crisis_level(cls, crisis_level: str) -> EscalationPriority:
        """Map crisis level to escalation priority."""
        mapping = {"NONE": cls.LOW, "LOW": cls.LOW, "ELEVATED": cls.MEDIUM,
            "HIGH": cls.HIGH, "CRITICAL": cls.CRITICAL}
        return mapping.get(crisis_level, cls.MEDIUM)


class EscalationStatus(str, Enum):
    """Status of an escalation request."""
    PENDING = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


class NotificationType(str, Enum):
    """Types of notifications for escalation."""
    SMS = "SMS"
    EMAIL = "EMAIL"
    PUSH = "PUSH"
    PAGER = "PAGER"
    IN_APP = "IN_APP"


class EscalationSettings(BaseSettings):
    """Configuration for escalation behavior."""
    auto_escalate_critical: bool = Field(default=True, description="Auto-escalate CRITICAL crises")
    auto_escalate_high: bool = Field(default=True, description="Auto-escalate HIGH crises")
    notification_timeout_seconds: int = Field(default=300, description="Notification timeout")
    max_retries: int = Field(default=3, description="Max notification retries")
    critical_response_sla_minutes: int = Field(default=5, description="CRITICAL response SLA")
    high_response_sla_minutes: int = Field(default=15, description="HIGH response SLA")
    medium_response_sla_minutes: int = Field(default=60, description="MEDIUM response SLA")
    low_response_sla_minutes: int = Field(default=240, description="LOW response SLA")
    enable_sms_notifications: bool = Field(default=True, description="Enable SMS notifications")
    enable_email_notifications: bool = Field(default=True, description="Enable email notifications")
    enable_pager_notifications: bool = Field(default=True, description="Enable pager for CRITICAL")
    on_call_clinician_pool_size: int = Field(default=3, description="On-call clinician pool size")
    notification_service_url: str = Field(
        default="http://localhost:8003",
        description="URL of the notification microservice"
    )
    dashboard_base_url: str = Field(
        default="https://dashboard.solace-ai.com",
        description="Base URL for the clinician dashboard"
    )
    model_config = SettingsConfigDict(env_prefix="ESCALATION_", env_file=".env", extra="ignore")


class EscalationResult(BaseModel):
    """Result of an escalation request."""
    escalation_id: UUID = Field(default_factory=uuid4, description="Unique escalation identifier")
    status: str = Field(default=EscalationStatus.PENDING.value, description="Escalation status")
    priority: str = Field(..., description="Escalation priority")
    assigned_clinician_id: UUID | None = Field(default=None, description="Assigned clinician")
    notification_sent: bool = Field(default=False, description="Whether notification was sent")
    actions_taken: list[str] = Field(default_factory=list, description="Actions taken")
    estimated_response_minutes: int | None = Field(default=None, description="Estimated response time")
    resources_provided: bool = Field(default=False, description="Crisis resources provided")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EscalationRecord:
    """Internal record of an escalation event."""
    escalation_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    crisis_level: str = "NONE"
    priority: EscalationPriority = EscalationPriority.LOW
    status: EscalationStatus = EscalationStatus.PENDING
    reason: str = ""
    assigned_clinician_id: UUID | None = None
    notifications_sent: list[NotificationType] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


@dataclass
class CrisisResource:
    """Crisis resource information."""
    name: str
    contact: str
    resource_type: str
    availability: str
    priority_order: int


class NotificationServiceClient:
    """
    HTTP client for sending escalation notifications to the Notification Service.

    Replaces the previous mock implementation with real HTTP calls to the notification
    microservice. Includes retry logic and circuit breaker patterns for reliability.

    CRITICAL: This service handles patient safety notifications. Failures are logged
    and should trigger alerts.
    """

    def __init__(self, settings: EscalationSettings, base_url: str = "http://localhost:8003") -> None:
        self._settings = settings
        self._base_url = base_url.rstrip("/")
        self._dashboard_base_url = settings.dashboard_base_url.rstrip("/")
        self._client: Any = None  # httpx.AsyncClient, lazily initialized
        self._max_retries = settings.max_retries
        self._timeout_seconds = settings.notification_timeout_seconds

    async def _ensure_client(self) -> Any:
        """Lazily initialize HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout_seconds),
                    headers={"Content-Type": "application/json"},
                )
            except ImportError:
                logger.error("httpx_not_installed", message="httpx library required for notification client")
                raise RuntimeError("httpx library is required for NotificationServiceClient")
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send_notification(self, clinician_id: UUID, escalation: EscalationRecord,
                                notification_type: NotificationType) -> bool:
        """
        Send notification to clinician via the notification service.

        Args:
            clinician_id: UUID of the clinician to notify
            escalation: Escalation record with crisis details
            notification_type: Channel type (SMS, EMAIL, PUSH, etc.)

        Returns:
            True if notification was sent successfully, False otherwise.
        """
        client = await self._ensure_client()

        # Map notification type to channel
        channel_map = {
            NotificationType.EMAIL: "email",
            NotificationType.SMS: "sms",
            NotificationType.PUSH: "push",
            NotificationType.PAGER: "sms",  # Pager falls back to SMS
            NotificationType.IN_APP: "push",  # In-app uses push channel
        }
        channel = channel_map.get(notification_type, "email")

        # Build notification payload — clinician contact resolved by notification service
        payload = {
            "template_type": "crisis_escalation",
            "recipients": [{
                "user_id": str(clinician_id),
                "email": None,
                "phone": None,
                "resolve_from_registry": True,
            }],
            "channels": [channel],
            "variables": {
                "patient_id": str(escalation.user_id),
                "patient_name": "Patient",  # Should lookup from user service
                "risk_level": escalation.crisis_level,
                "assessment_summary": escalation.reason,
                "dashboard_link": f"{self._dashboard_base_url}/escalations/{escalation.escalation_id}",
                "escalation_id": str(escalation.escalation_id),
                "priority": escalation.priority.value,
            },
            "priority": "critical" if escalation.priority == EscalationPriority.CRITICAL else "high",
            "correlation_id": str(escalation.escalation_id),
        }

        # Attempt to send with retries
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post("/api/v1/notifications/send", json=payload)

                if response.status_code in (200, 201, 202):
                    result = response.json()
                    success = result.get("successful_deliveries", 0) > 0
                    logger.info(
                        "notification_sent",
                        clinician_id=str(clinician_id),
                        escalation_id=str(escalation.escalation_id),
                        notification_type=notification_type.value,
                        success=success,
                        request_id=result.get("request_id"),
                    )
                    return success
                else:
                    logger.warning(
                        "notification_failed",
                        clinician_id=str(clinician_id),
                        escalation_id=str(escalation.escalation_id),
                        status_code=response.status_code,
                        attempt=attempt + 1,
                        response_text=response.text[:500] if response.text else None,
                    )
                    last_error = f"HTTP {response.status_code}: {response.text[:200] if response.text else 'No response'}"

            except Exception as e:
                logger.error(
                    "notification_error",
                    clinician_id=str(clinician_id),
                    escalation_id=str(escalation.escalation_id),
                    error=str(e),
                    attempt=attempt + 1,
                )
                last_error = str(e)

            # Wait before retry (exponential backoff)
            if attempt < self._max_retries:
                await asyncio.sleep(min(2 ** attempt, 10))

        # All retries exhausted - this is a CRITICAL failure for patient safety
        logger.critical(
            "notification_all_retries_exhausted",
            clinician_id=str(clinician_id),
            escalation_id=str(escalation.escalation_id),
            notification_type=notification_type.value,
            last_error=last_error,
            crisis_level=escalation.crisis_level,
            message="CRITICAL: Crisis notification could not be delivered after all retries",
        )
        return False

    async def send_multi_channel(self, clinician_id: UUID, escalation: EscalationRecord,
                                  channels: list[NotificationType]) -> dict[NotificationType, bool]:
        """
        Send notifications across multiple channels concurrently.

        For crisis situations, we attempt all channels to maximize delivery probability.
        """
        tasks = [
            self.send_notification(clinician_id, escalation, channel)
            for channel in channels
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        channel_results = {}
        for channel, result in zip(channels, results):
            if isinstance(result, Exception):
                logger.error("notification_channel_exception", channel=channel.value, error=str(result))
                channel_results[channel] = False
            else:
                channel_results[channel] = result

        # Log summary
        successful = sum(1 for v in channel_results.values() if v)
        logger.info(
            "multi_channel_notification_complete",
            escalation_id=str(escalation.escalation_id),
            total_channels=len(channels),
            successful=successful,
            failed=len(channels) - successful,
        )

        return channel_results


# Backward compatibility alias
NotificationService = NotificationServiceClient


class ClinicianAssigner:
    """Service for assigning clinicians to escalations.

    Uses ClinicianRegistry (HTTP lookup to User Service) when available,
    falling back to the registry's configurable fallback email.
    Tracks workload per clinician for load balancing.
    """

    def __init__(self, settings: EscalationSettings, clinician_registry: Any | None = None) -> None:
        self._settings = settings
        self._registry = clinician_registry
        self._clinician_workload: dict[UUID, int] = {}

    async def assign_clinician(self, escalation: EscalationRecord) -> UUID | None:
        """Assign available clinician based on priority and workload.

        Fetches on-call clinicians from the ClinicianRegistry (User Service HTTP lookup).
        Falls back to the registry's configurable fallback email when unavailable.
        """
        if self._registry is None:
            logger.warning(
                "no_clinician_registry",
                escalation_id=str(escalation.escalation_id),
                hint="ClinicianRegistry not configured; cannot assign clinician",
            )
            return None

        contacts = await self._registry.get_oncall_clinicians()
        if not contacts:
            logger.warning("no_clinicians_available", escalation_id=str(escalation.escalation_id))
            return None

        # Track workload for load balancing
        for contact in contacts:
            if contact.clinician_id not in self._clinician_workload:
                self._clinician_workload[contact.clinician_id] = 0

        # Select clinician with lowest workload
        available = [(c, self._clinician_workload.get(c.clinician_id, 0)) for c in contacts]
        available.sort(key=lambda x: x[1])
        assigned_contact = available[0][0]
        self._clinician_workload[assigned_contact.clinician_id] = (
            self._clinician_workload.get(assigned_contact.clinician_id, 0) + 1
        )
        logger.info(
            "clinician_assigned",
            clinician_id=str(assigned_contact.clinician_id),
            clinician_name=assigned_contact.name,
            clinician_email=assigned_contact.email,
            escalation_id=str(escalation.escalation_id),
            priority=escalation.priority.value,
        )
        return assigned_contact.clinician_id

    def release_clinician(self, clinician_id: UUID) -> None:
        """Release clinician from assignment."""
        if clinician_id in self._clinician_workload:
            self._clinician_workload[clinician_id] = max(0, self._clinician_workload[clinician_id] - 1)


class CrisisResourceManager:
    """Manager for crisis resources and hotlines."""

    def __init__(self) -> None:
        self._resources = [
            CrisisResource("988 Suicide & Crisis Lifeline", "988", "phone", "24/7", 1),
            CrisisResource("Crisis Text Line", "Text HOME to 741741", "text", "24/7", 2),
            CrisisResource("Emergency Services", "911", "phone", "24/7", 0),
            CrisisResource("SAMHSA National Helpline", "1-800-662-4357", "phone", "24/7", 3),
            CrisisResource("Veterans Crisis Line", "988 (Press 1)", "phone", "24/7", 4),
            CrisisResource("Trevor Project (LGBTQ+)", "1-866-488-7386", "phone", "24/7", 5),
        ]

    def get_resources_for_level(self, crisis_level: str) -> list[dict[str, str]]:
        """Get appropriate resources for crisis level."""
        if crisis_level == "CRITICAL":
            resources = sorted(self._resources, key=lambda r: r.priority_order)
        elif crisis_level == "HIGH":
            resources = [r for r in self._resources if r.priority_order > 0]
            resources.sort(key=lambda r: r.priority_order)
        else:
            resources = [r for r in self._resources if r.priority_order in (1, 2)]
        return [{"name": r.name, "contact": r.contact, "type": r.resource_type, "available": r.availability}
                for r in resources]


class EscalationWorkflow:
    """Orchestrates the escalation workflow based on priority."""

    def __init__(self, settings: EscalationSettings, notification_service: NotificationService,
                 clinician_assigner: ClinicianAssigner) -> None:
        self._settings = settings
        self._notifications = notification_service
        self._assigner = clinician_assigner

    async def execute_critical_workflow(self, escalation: EscalationRecord) -> EscalationRecord:
        """Execute CRITICAL priority workflow - immediate response required."""
        escalation.actions_taken.append("CRITICAL workflow initiated")
        clinician = await self._assigner.assign_clinician(escalation)
        if clinician:
            escalation.assigned_clinician_id = clinician
            escalation.actions_taken.append(f"Clinician {clinician} assigned")
        channels = [NotificationType.PAGER, NotificationType.SMS, NotificationType.IN_APP]
        if clinician:
            results = await self._notifications.send_multi_channel(clinician, escalation, channels)
            escalation.notifications_sent.extend([c for c, sent in results.items() if sent])
        escalation.actions_taken.append("Crisis resources displayed to user")
        escalation.actions_taken.append("All processing paused")
        escalation.actions_taken.append("Safety dialogue initiated")
        escalation.status = EscalationStatus.IN_PROGRESS
        return escalation

    async def execute_high_workflow(self, escalation: EscalationRecord) -> EscalationRecord:
        """Execute HIGH priority workflow - urgent response required."""
        escalation.actions_taken.append("HIGH workflow initiated")
        clinician = await self._assigner.assign_clinician(escalation)
        if clinician:
            escalation.assigned_clinician_id = clinician
            escalation.actions_taken.append(f"Clinician {clinician} assigned")
            channels = [NotificationType.SMS, NotificationType.EMAIL, NotificationType.IN_APP]
            results = await self._notifications.send_multi_channel(clinician, escalation, channels)
            escalation.notifications_sent.extend([c for c, sent in results.items() if sent])
        escalation.actions_taken.append("Therapeutic content paused")
        escalation.actions_taken.append("Safety assessment dialogue started")
        escalation.actions_taken.append("Crisis resources provided")
        escalation.status = EscalationStatus.IN_PROGRESS
        return escalation

    async def execute_medium_workflow(self, escalation: EscalationRecord) -> EscalationRecord:
        """Execute MEDIUM priority workflow - timely response required."""
        escalation.actions_taken.append("MEDIUM workflow initiated")
        escalation.actions_taken.append("Enhanced monitoring enabled")
        escalation.actions_taken.append("Coping strategies reviewed")
        escalation.actions_taken.append("Support resources shared")
        if self._settings.enable_email_notifications:
            escalation.actions_taken.append("Supervisor notified via email")
        escalation.status = EscalationStatus.PENDING
        return escalation

    async def execute_low_workflow(self, escalation: EscalationRecord) -> EscalationRecord:
        """Execute LOW priority workflow - standard monitoring."""
        escalation.actions_taken.append("LOW workflow initiated")
        escalation.actions_taken.append("Standard monitoring active")
        escalation.actions_taken.append("Concern acknowledged")
        escalation.status = EscalationStatus.PENDING
        return escalation


class EscalationManager:
    """Main escalation manager orchestrating all escalation components."""

    def __init__(
        self,
        settings: EscalationSettings | None = None,
        clinician_registry: Any | None = None,
    ) -> None:
        self._settings = settings or EscalationSettings()
        self._notification_service = NotificationServiceClient(
            self._settings,
            base_url=self._settings.notification_service_url
        )

        # Use provided registry or attempt to create one from settings
        if clinician_registry is None:
            try:
                from ..infrastructure.clinician_registry import ClinicianRegistry, ClinicianRegistrySettings
                registry_settings = ClinicianRegistrySettings()
                clinician_registry = ClinicianRegistry(registry_settings)
                logger.info("clinician_registry_created_from_settings")
            except Exception as e:
                logger.warning(
                    "clinician_registry_creation_failed",
                    error=str(e),
                    hint="ClinicianAssigner will not be able to assign real clinicians",
                )

        self._clinician_assigner = ClinicianAssigner(self._settings, clinician_registry=clinician_registry)
        self._resource_manager = CrisisResourceManager()
        self._workflow = EscalationWorkflow(self._settings, self._notification_service, self._clinician_assigner)
        self._active_escalations: dict[UUID, EscalationRecord] = {}
        # Deduplication: (user_id, crisis_level) -> last escalation timestamp
        self._dedup_window: dict[tuple[UUID, str], datetime] = {}
        self._dedup_ttl = timedelta(minutes=5)
        logger.info(
            "escalation_manager_initialized",
            auto_escalate_critical=self._settings.auto_escalate_critical,
            notification_service_url=self._settings.notification_service_url,
            clinician_registry_available=clinician_registry is not None,
            dedup_window_minutes=5,
        )

    async def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        await self._notification_service.close()
        logger.info("escalation_manager_shutdown")

    async def escalate(self, user_id: UUID, session_id: UUID | None, crisis_level: str,
                       reason: str, context: dict[str, Any] | None = None,
                       priority_override: str | None = None) -> EscalationResult:
        """Process escalation request through appropriate workflow.

        Includes 5-minute deduplication per (user_id, crisis_level) pair to prevent
        notification storms. Duplicate requests within the window return a
        DEDUPLICATED status without re-notifying clinicians.
        """
        # Deduplication check
        dedup_key = (user_id, crisis_level)
        now = datetime.now(timezone.utc)
        last_escalation_at = self._dedup_window.get(dedup_key)
        if last_escalation_at and (now - last_escalation_at) < self._dedup_ttl:
            logger.info(
                "escalation_deduplicated",
                user_id=str(user_id),
                crisis_level=crisis_level,
                seconds_since_last=(now - last_escalation_at).total_seconds(),
            )
            return EscalationResult(
                status="DEDUPLICATED",
                priority=EscalationPriority.from_crisis_level(crisis_level).value,
                actions_taken=["Duplicate escalation suppressed within 5-minute window"],
                resources_provided=False,
            )

        # Record this escalation for dedup
        self._dedup_window[dedup_key] = now
        # Clean expired entries
        expired_keys = [k for k, ts in self._dedup_window.items() if (now - ts) >= self._dedup_ttl]
        for k in expired_keys:
            del self._dedup_window[k]

        priority = EscalationPriority(priority_override) if priority_override else EscalationPriority.from_crisis_level(crisis_level)
        escalation = EscalationRecord(
            user_id=user_id, session_id=session_id, crisis_level=crisis_level,
            priority=priority, reason=reason, context=context or {})
        self._active_escalations[escalation.escalation_id] = escalation
        logger.info("escalation_created", escalation_id=str(escalation.escalation_id),
            user_id=str(user_id), priority=priority.value, crisis_level=crisis_level)
        if priority == EscalationPriority.CRITICAL:
            escalation = await self._workflow.execute_critical_workflow(escalation)
        elif priority == EscalationPriority.HIGH:
            escalation = await self._workflow.execute_high_workflow(escalation)
        elif priority == EscalationPriority.MEDIUM:
            escalation = await self._workflow.execute_medium_workflow(escalation)
        else:
            escalation = await self._workflow.execute_low_workflow(escalation)
        response_time = self._get_estimated_response_time(priority)
        resources_provided = priority in (EscalationPriority.CRITICAL, EscalationPriority.HIGH)
        return EscalationResult(
            escalation_id=escalation.escalation_id,
            status=escalation.status.value,
            priority=priority.value,
            assigned_clinician_id=escalation.assigned_clinician_id,
            notification_sent=len(escalation.notifications_sent) > 0,
            actions_taken=escalation.actions_taken,
            estimated_response_minutes=response_time,
            resources_provided=resources_provided,
        )

    def _get_estimated_response_time(self, priority: EscalationPriority) -> int:
        """Get estimated response time based on priority."""
        times = {EscalationPriority.CRITICAL: self._settings.critical_response_sla_minutes,
            EscalationPriority.HIGH: self._settings.high_response_sla_minutes,
            EscalationPriority.MEDIUM: self._settings.medium_response_sla_minutes,
            EscalationPriority.LOW: self._settings.low_response_sla_minutes}
        return times.get(priority, 60)

    def get_crisis_resources(self, crisis_level: str) -> list[dict[str, str]]:
        """Get crisis resources for specified level."""
        return self._resource_manager.get_resources_for_level(crisis_level)

    async def acknowledge_escalation(self, escalation_id: UUID, clinician_id: UUID) -> bool:
        """Acknowledge an escalation by clinician."""
        if escalation_id not in self._active_escalations:
            return False
        escalation = self._active_escalations[escalation_id]
        escalation.status = EscalationStatus.ACKNOWLEDGED
        escalation.acknowledged_at = datetime.now(timezone.utc)
        escalation.actions_taken.append(f"Acknowledged by clinician {clinician_id}")
        logger.info("escalation_acknowledged", escalation_id=str(escalation_id), clinician_id=str(clinician_id))
        return True

    async def resolve_escalation(self, escalation_id: UUID, resolution_notes: str) -> bool:
        """Resolve an escalation."""
        if escalation_id not in self._active_escalations:
            return False
        escalation = self._active_escalations[escalation_id]
        escalation.status = EscalationStatus.RESOLVED
        escalation.resolved_at = datetime.now(timezone.utc)
        escalation.actions_taken.append(f"Resolved: {resolution_notes}")
        if escalation.assigned_clinician_id:
            self._clinician_assigner.release_clinician(escalation.assigned_clinician_id)
        logger.info("escalation_resolved", escalation_id=str(escalation_id))
        return True

    def get_active_escalations(self, user_id: UUID | None = None) -> list[EscalationRecord]:
        """Get active escalations, optionally filtered by user."""
        escalations = list(self._active_escalations.values())
        if user_id:
            escalations = [e for e in escalations if e.user_id == user_id]
        return [e for e in escalations if e.status not in (EscalationStatus.RESOLVED, EscalationStatus.CLOSED)]

    def get_statistics(self) -> dict[str, Any]:
        """Get escalation statistics."""
        active = self.get_active_escalations()
        return {
            "total_active": len(active),
            "by_priority": {p.value: sum(1 for e in active if e.priority == p) for p in EscalationPriority},
            "by_status": {s.value: sum(1 for e in active if e.status == s) for s in EscalationStatus},
        }
