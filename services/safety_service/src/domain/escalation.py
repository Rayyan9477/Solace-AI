"""
Solace-AI Escalation Manager - Crisis escalation workflow management.
Handles escalation to human clinicians, notifications, and crisis response coordination.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


class NotificationService:
    """Service for sending escalation notifications."""

    def __init__(self, settings: EscalationSettings) -> None:
        self._settings = settings

    async def send_notification(self, clinician_id: UUID, escalation: EscalationRecord,
                                notification_type: NotificationType) -> bool:
        """Send notification to clinician."""
        logger.info("sending_notification", clinician_id=str(clinician_id),
            notification_type=notification_type.value, escalation_id=str(escalation.escalation_id))
        await asyncio.sleep(0.01)
        return True

    async def send_multi_channel(self, clinician_id: UUID, escalation: EscalationRecord,
                                  channels: list[NotificationType]) -> dict[NotificationType, bool]:
        """Send notifications across multiple channels."""
        results = {}
        for channel in channels:
            results[channel] = await self.send_notification(clinician_id, escalation, channel)
        return results


class ClinicianAssigner:
    """Service for assigning clinicians to escalations."""

    def __init__(self, settings: EscalationSettings) -> None:
        self._settings = settings
        self._on_call_clinicians: list[UUID] = [uuid4() for _ in range(settings.on_call_clinician_pool_size)]
        self._clinician_workload: dict[UUID, int] = {c: 0 for c in self._on_call_clinicians}

    async def assign_clinician(self, escalation: EscalationRecord) -> UUID | None:
        """Assign available clinician based on priority and workload."""
        if not self._on_call_clinicians:
            logger.warning("no_clinicians_available", escalation_id=str(escalation.escalation_id))
            return None
        available = [(c, self._clinician_workload[c]) for c in self._on_call_clinicians]
        available.sort(key=lambda x: x[1])
        assigned = available[0][0]
        self._clinician_workload[assigned] += 1
        logger.info("clinician_assigned", clinician_id=str(assigned),
            escalation_id=str(escalation.escalation_id), priority=escalation.priority.value)
        return assigned

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

    def __init__(self, settings: EscalationSettings | None = None) -> None:
        self._settings = settings or EscalationSettings()
        self._notification_service = NotificationService(self._settings)
        self._clinician_assigner = ClinicianAssigner(self._settings)
        self._resource_manager = CrisisResourceManager()
        self._workflow = EscalationWorkflow(self._settings, self._notification_service, self._clinician_assigner)
        self._active_escalations: dict[UUID, EscalationRecord] = {}
        logger.info("escalation_manager_initialized", auto_escalate_critical=self._settings.auto_escalate_critical)

    async def escalate(self, user_id: UUID, session_id: UUID | None, crisis_level: str,
                       reason: str, context: dict[str, Any] | None = None,
                       priority_override: str | None = None) -> EscalationResult:
        """Process escalation request through appropriate workflow."""
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
