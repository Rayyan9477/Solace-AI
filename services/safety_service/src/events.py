"""
Solace-AI Safety Service - Domain Events.
Safety domain events for inter-service communication and audit logging.
"""
from __future__ import annotations
import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)
T = TypeVar("T", bound="SafetyEvent")


class EventType(str, Enum):
    """Safety event types."""
    CRISIS_DETECTED = "safety.crisis.detected"
    CRISIS_RESOLVED = "safety.crisis.resolved"
    ESCALATION_TRIGGERED = "safety.escalation.triggered"
    ESCALATION_ACKNOWLEDGED = "safety.escalation.acknowledged"
    ESCALATION_RESOLVED = "safety.escalation.resolved"
    SAFETY_CHECK_COMPLETED = "safety.check.completed"
    SAFETY_PLAN_CREATED = "safety.plan.created"
    SAFETY_PLAN_ACTIVATED = "safety.plan.activated"
    SAFETY_PLAN_UPDATED = "safety.plan.updated"
    RISK_LEVEL_CHANGED = "safety.risk.level_changed"
    INCIDENT_CREATED = "safety.incident.created"
    INCIDENT_RESOLVED = "safety.incident.resolved"
    OUTPUT_FILTERED = "safety.output.filtered"
    TRAJECTORY_ALERT = "safety.trajectory.alert"


class SafetyEvent(BaseModel):
    """Base class for all safety domain events."""
    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: UUID | None = Field(default=None)
    session_id: UUID | None = Field(default=None)
    correlation_id: UUID = Field(default_factory=uuid4)
    source: str = Field(default="safety-service")
    version: str = Field(default="1.0.0")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize event to JSON."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> SafetyEvent:
        """Deserialize event from JSON."""
        return cls.model_validate_json(data)


class CrisisDetectedEvent(SafetyEvent):
    """Event fired when a crisis is detected."""
    event_type: EventType = Field(default=EventType.CRISIS_DETECTED)
    crisis_level: str = Field(...)
    risk_score: Decimal = Field(...)
    trigger_indicators: list[str] = Field(default_factory=list)
    detection_layers: list[int] = Field(default_factory=list)
    detection_time_ms: int = Field(default=0, ge=0)
    content_preview: str | None = Field(default=None)
    requires_escalation: bool = Field(default=False)
    requires_human_review: bool = Field(default=False)


class CrisisResolvedEvent(SafetyEvent):
    """Event fired when a crisis is resolved."""
    event_type: EventType = Field(default=EventType.CRISIS_RESOLVED)
    crisis_level: str = Field(...)
    resolution_notes: str | None = Field(default=None)
    resolved_by: UUID | None = Field(default=None)
    time_to_resolution_minutes: int | None = Field(default=None)


class EscalationTriggeredEvent(SafetyEvent):
    """Event fired when an escalation is triggered."""
    event_type: EventType = Field(default=EventType.ESCALATION_TRIGGERED)
    escalation_id: UUID = Field(...)
    priority: str = Field(...)
    crisis_level: str = Field(...)
    reason: str = Field(...)
    assigned_clinician_id: UUID | None = Field(default=None)
    notification_channels: list[str] = Field(default_factory=list)
    estimated_response_minutes: int | None = Field(default=None)


class EscalationAcknowledgedEvent(SafetyEvent):
    """Event fired when an escalation is acknowledged."""
    event_type: EventType = Field(default=EventType.ESCALATION_ACKNOWLEDGED)
    escalation_id: UUID = Field(...)
    acknowledged_by: UUID = Field(...)
    time_to_acknowledge_seconds: int | None = Field(default=None)


class EscalationResolvedEvent(SafetyEvent):
    """Event fired when an escalation is resolved."""
    event_type: EventType = Field(default=EventType.ESCALATION_RESOLVED)
    escalation_id: UUID = Field(...)
    resolved_by: UUID = Field(...)
    resolution_notes: str = Field(...)
    time_to_resolution_minutes: int | None = Field(default=None)


class SafetyCheckCompletedEvent(SafetyEvent):
    """Event fired when a safety check is completed."""
    event_type: EventType = Field(default=EventType.SAFETY_CHECK_COMPLETED)
    check_type: str = Field(...)
    is_safe: bool = Field(...)
    crisis_level: str = Field(default="NONE")
    risk_score: Decimal = Field(default=Decimal("0.0"))
    detection_time_ms: int = Field(default=0, ge=0)


class RiskLevelChangedEvent(SafetyEvent):
    """Event fired when user's risk level changes."""
    event_type: EventType = Field(default=EventType.RISK_LEVEL_CHANGED)
    previous_level: str = Field(...)
    new_level: str = Field(...)
    change_reason: str | None = Field(default=None)
    risk_trend: str = Field(default="stable")


class IncidentCreatedEvent(SafetyEvent):
    """Event fired when a safety incident is created."""
    event_type: EventType = Field(default=EventType.INCIDENT_CREATED)
    incident_id: UUID = Field(...)
    severity: str = Field(...)
    crisis_level: str = Field(...)
    description: str = Field(...)


class IncidentResolvedEvent(SafetyEvent):
    """Event fired when a safety incident is resolved."""
    event_type: EventType = Field(default=EventType.INCIDENT_RESOLVED)
    incident_id: UUID = Field(...)
    resolution_notes: str = Field(...)
    resolved_by: UUID | None = Field(default=None)
    time_to_resolution_minutes: int | None = Field(default=None)


def to_kafka_event(event: SafetyEvent) -> Any:
    """Convert local safety event to canonical Kafka event for inter-service messaging.

    Returns a canonical BaseEvent subclass, or None if no mapping exists
    or solace_events is not available.
    """
    try:
        from src.solace_events.schemas import (
            EventMetadata as KafkaMetadata,
            CrisisLevel as KafkaCrisisLevel,
            CrisisDetectedEvent as KafkaCrisisDetected,
            CrisisResolvedEvent as KafkaCrisisResolved,
            SafetyAssessmentEvent as KafkaSafetyAssessment,
            EscalationTriggeredEvent as KafkaEscalationTriggered,
            EscalationAcknowledgedEvent as KafkaEscalationAcknowledged,
            EscalationResolvedEvent as KafkaEscalationResolved,
            RiskLevelChangedEvent as KafkaRiskLevelChanged,
            IncidentCreatedEvent as KafkaIncidentCreated,
            IncidentResolvedEvent as KafkaIncidentResolved,
        )
    except ImportError:
        logger.debug("solace_events_not_available_for_bridge")
        return None

    if not event.user_id:
        return None

    meta = KafkaMetadata(
        event_id=event.event_id, timestamp=event.timestamp,
        correlation_id=event.correlation_id, source_service="safety-service",
    )
    base: dict[str, Any] = {"user_id": event.user_id, "session_id": event.session_id, "metadata": meta}

    if isinstance(event, CrisisDetectedEvent):
        return KafkaCrisisDetected(
            **base, crisis_level=KafkaCrisisLevel(event.crisis_level),
            trigger_indicators=event.trigger_indicators,
            detection_layer=event.detection_layers[0] if event.detection_layers else 1,
            confidence=event.risk_score,
            escalation_action="escalate" if event.requires_escalation else "monitor",
            requires_human_review=event.requires_human_review,
        )

    if isinstance(event, CrisisResolvedEvent):
        return KafkaCrisisResolved(
            **base, crisis_level=KafkaCrisisLevel(event.crisis_level),
            resolution_notes=event.resolution_notes, resolved_by=event.resolved_by,
            time_to_resolution_minutes=event.time_to_resolution_minutes,
        )

    if isinstance(event, EscalationTriggeredEvent):
        return KafkaEscalationTriggered(
            **base, escalation_reason=event.reason, priority=event.priority,
            assigned_clinician_id=event.assigned_clinician_id,
            notification_sent=bool(event.notification_channels),
        )

    if isinstance(event, EscalationAcknowledgedEvent):
        return KafkaEscalationAcknowledged(
            **base, escalation_id=event.escalation_id,
            acknowledged_by=event.acknowledged_by,
            time_to_acknowledge_seconds=event.time_to_acknowledge_seconds,
        )

    if isinstance(event, EscalationResolvedEvent):
        return KafkaEscalationResolved(
            **base, escalation_id=event.escalation_id,
            resolved_by=event.resolved_by, resolution_notes=event.resolution_notes,
            time_to_resolution_minutes=event.time_to_resolution_minutes,
        )

    if isinstance(event, SafetyCheckCompletedEvent):
        return KafkaSafetyAssessment(
            **base, risk_level=KafkaCrisisLevel(event.crisis_level),
            risk_score=event.risk_score, detection_layer=1,
            recommended_action="monitor" if event.is_safe else "review",
        )

    if isinstance(event, RiskLevelChangedEvent):
        valid_trends = {"improving", "stable", "worsening"}
        trend = event.risk_trend if event.risk_trend in valid_trends else "stable"
        return KafkaRiskLevelChanged(
            **base, previous_level=KafkaCrisisLevel(event.previous_level),
            new_level=KafkaCrisisLevel(event.new_level),
            change_reason=event.change_reason, risk_trend=trend,
        )

    if isinstance(event, IncidentCreatedEvent):
        return KafkaIncidentCreated(
            **base, incident_id=event.incident_id, severity=event.severity,
            crisis_level=KafkaCrisisLevel(event.crisis_level), description=event.description,
        )

    if isinstance(event, IncidentResolvedEvent):
        return KafkaIncidentResolved(
            **base, incident_id=event.incident_id,
            resolution_notes=event.resolution_notes, resolved_by=event.resolved_by,
            time_to_resolution_minutes=event.time_to_resolution_minutes,
        )

    logger.debug("no_kafka_mapping_for_event", event_type=event.event_type.value)
    return None


class SafetyEventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle(self, event: SafetyEvent) -> None:
        """Handle a safety event."""
        pass


class SafetyEventPublisher:
    """Publisher for safety domain events."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[SafetyEventHandler]] = {}
        self._async_callbacks: list[Callable[[SafetyEvent], Awaitable[None]]] = []
        self._event_queue: asyncio.Queue[SafetyEvent] = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the event publisher."""
        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("safety_event_publisher_started")

    async def stop(self) -> None:
        """Stop the event publisher."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("safety_event_publisher_stopped")

    def register_handler(self, event_type: EventType, handler: SafetyEventHandler) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug("event_handler_registered", event_type=event_type.value)

    def register_callback(self, callback: Callable[[SafetyEvent], Awaitable[None]]) -> None:
        """Register an async callback for all events."""
        self._async_callbacks.append(callback)

    async def publish(self, event: SafetyEvent) -> None:
        """Publish a safety event."""
        await self._event_queue.put(event)
        logger.info("safety_event_published", event_type=event.event_type.value,
                    event_id=str(event.event_id), user_id=str(event.user_id) if event.user_id else None)

    async def publish_sync(self, event: SafetyEvent) -> None:
        """Publish and immediately process a safety event."""
        await self._dispatch_event(event)

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._dispatch_event(event)
                self._event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("event_processing_error", error=str(e))

    async def _dispatch_event(self, event: SafetyEvent) -> None:
        """Dispatch event to handlers and callbacks."""
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error("event_handler_error", handler=type(handler).__name__, error=str(e))
        for callback in self._async_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error("event_callback_error", error=str(e))


class AuditEventHandler(SafetyEventHandler):
    """Handler that logs all safety events for audit purposes."""

    def __init__(self) -> None:
        self._audit_log: list[dict[str, Any]] = []

    async def handle(self, event: SafetyEvent) -> None:
        """Log event for audit trail."""
        audit_entry = {
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": str(event.user_id) if event.user_id else None,
            "session_id": str(event.session_id) if event.session_id else None,
            "correlation_id": str(event.correlation_id),
            "data": event.model_dump(exclude={"event_id", "timestamp", "user_id",
                                              "session_id", "correlation_id"}),
        }
        self._audit_log.append(audit_entry)
        logger.info("safety_event_audited", event_type=event.event_type.value,
                    event_id=str(event.event_id))

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the audit log."""
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()


_publisher: SafetyEventPublisher | None = None


def get_event_publisher() -> SafetyEventPublisher:
    """Get singleton event publisher."""
    global _publisher
    if _publisher is None:
        _publisher = SafetyEventPublisher()
    return _publisher


async def initialize_event_publisher() -> SafetyEventPublisher:
    """Initialize and start the event publisher."""
    publisher = get_event_publisher()
    await publisher.start()
    return publisher
