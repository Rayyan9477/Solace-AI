"""
Solace-AI Diagnosis Service - Domain Events.
Event definitions and dispatcher for diagnosis workflow.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4
import structlog

from .schemas import DiagnosisPhase, SeverityLevel

logger = structlog.get_logger(__name__)

EventHandler = Callable[["DiagnosisEvent"], Awaitable[None]]


@dataclass
class DiagnosisEvent:
    """Base class for all diagnosis events."""
    event_id: UUID = field(default_factory=uuid4)
    event_type: str = "diagnosis_event"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: UUID | None = None
    session_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "metadata": self.metadata,
        }


@dataclass
class SessionStartedEvent(DiagnosisEvent):
    """Event fired when a diagnosis session starts."""
    event_type: str = "session_started"
    session_number: int = 1
    initial_phase: DiagnosisPhase = DiagnosisPhase.RAPPORT
    has_previous_context: bool = False


@dataclass
class SessionEndedEvent(DiagnosisEvent):
    """Event fired when a diagnosis session ends."""
    event_type: str = "session_ended"
    duration_minutes: int = 0
    messages_exchanged: int = 0
    final_phase: DiagnosisPhase = DiagnosisPhase.CLOSURE
    primary_hypothesis: str | None = None


@dataclass
class PhaseTransitionEvent(DiagnosisEvent):
    """Event fired when dialogue phase changes."""
    event_type: str = "phase_transition"
    from_phase: DiagnosisPhase = DiagnosisPhase.RAPPORT
    to_phase: DiagnosisPhase = DiagnosisPhase.HISTORY
    confidence_at_transition: Decimal = Decimal("0.5")


@dataclass
class SymptomExtractedEvent(DiagnosisEvent):
    """Event fired when a symptom is extracted."""
    event_type: str = "symptom_extracted"
    symptom_id: UUID = field(default_factory=uuid4)
    symptom_name: str = ""
    severity: SeverityLevel = SeverityLevel.MILD
    confidence: Decimal = Decimal("0.7")
    extracted_from: str = ""


@dataclass
class HypothesisGeneratedEvent(DiagnosisEvent):
    """Event fired when a hypothesis is generated."""
    event_type: str = "hypothesis_generated"
    hypothesis_id: UUID = field(default_factory=uuid4)
    hypothesis_name: str = ""
    confidence: Decimal = Decimal("0.5")
    dsm5_code: str | None = None
    criteria_met_count: int = 0


@dataclass
class HypothesisChallengedEvent(DiagnosisEvent):
    """Event fired when a hypothesis is challenged."""
    event_type: str = "hypothesis_challenged"
    hypothesis_id: UUID = field(default_factory=uuid4)
    hypothesis_name: str = ""
    original_confidence: Decimal = Decimal("0.5")
    adjusted_confidence: Decimal = Decimal("0.5")
    challenges: list[str] = field(default_factory=list)
    biases_detected: list[str] = field(default_factory=list)


@dataclass
class DiagnosisRecordedEvent(DiagnosisEvent):
    """Event fired when a diagnosis is recorded."""
    event_type: str = "diagnosis_recorded"
    record_id: UUID = field(default_factory=uuid4)
    primary_diagnosis: str = ""
    dsm5_code: str | None = None
    severity: SeverityLevel = SeverityLevel.MILD
    confidence: Decimal = Decimal("0.5")


@dataclass
class SafetyFlagRaisedEvent(DiagnosisEvent):
    """Event fired when a safety flag is raised."""
    event_type: str = "safety_flag_raised"
    flag_type: str = ""
    severity: str = "moderate"
    trigger_text: str = ""
    recommended_action: str = ""


@dataclass
class UserDataDeletedEvent(DiagnosisEvent):
    """Event fired when user data is deleted (GDPR)."""
    event_type: str = "user_data_deleted"
    records_deleted: int = 0
    sessions_deleted: int = 0


def to_kafka_event(event: DiagnosisEvent) -> Any:
    """Convert local diagnosis event to canonical Kafka event for inter-service messaging.

    Maps diagnosis completion events to canonical schemas.
    Returns None for internal-only events or if solace_events is not available.
    """
    try:
        from src.solace_events.schemas import (
            EventMetadata as KafkaMetadata,
            DiagnosisCompletedEvent as KafkaDiagnosisCompleted,
            ClinicalHypothesis, Confidence,
        )
    except ImportError:
        logger.debug("solace_events_not_available_for_bridge")
        return None

    if not event.user_id:
        return None

    meta = KafkaMetadata(
        event_id=event.event_id, timestamp=event.timestamp,
        source_service="diagnosis-service",
    )
    base: dict[str, Any] = {"user_id": event.user_id, "session_id": event.session_id, "metadata": meta}

    if isinstance(event, DiagnosisRecordedEvent):
        primary = None
        if event.primary_diagnosis:
            severity_str = event.severity.value if hasattr(event.severity, "value") else str(event.severity)
            severity_map = {"MILD": "MILD", "MODERATE": "MODERATE", "SEVERE": "SEVERE"}
            primary = ClinicalHypothesis(
                condition_code=event.dsm5_code or "unspecified",
                condition_name=event.primary_diagnosis,
                confidence=Confidence.MODERATE,
                evidence_summary="",
                severity=severity_map.get(severity_str.upper(), "MODERATE"),
            )
        return KafkaDiagnosisCompleted(
            **base, assessment_id=event.record_id, primary_hypothesis=primary,
            stepped_care_level=0,
        )

    return None


class EventDispatcher:
    """Dispatcher for diagnosis domain events."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._event_log: list[DiagnosisEvent] = []
        self._log_events: bool = True
        self._max_log_size: int = 1000
        self._stats = {"events_dispatched": 0, "handlers_called": 0}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)
        logger.debug("handler_subscribed", event_type=event_type)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)
        logger.debug("global_handler_subscribed")

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from a specific event type."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def dispatch(self, event: DiagnosisEvent) -> None:
        """Dispatch an event to all subscribers."""
        self._stats["events_dispatched"] += 1
        if self._log_events:
            self._event_log.append(event)
            if len(self._event_log) > self._max_log_size:
                self._event_log = self._event_log[-self._max_log_size:]
        logger.debug("event_dispatched", event_type=event.event_type, event_id=str(event.event_id))
        handlers = self._handlers.get(event.event_type, []) + self._global_handlers
        for handler in handlers:
            try:
                await handler(event)
                self._stats["handlers_called"] += 1
            except Exception as e:
                logger.error("handler_error", event_type=event.event_type, error=str(e))

    async def dispatch_many(self, events: list[DiagnosisEvent]) -> None:
        """Dispatch multiple events."""
        for event in events:
            await self.dispatch(event)

    def get_event_log(self, event_type: str | None = None, limit: int = 100) -> list[DiagnosisEvent]:
        """Get event log, optionally filtered by type."""
        if event_type:
            filtered = [e for e in self._event_log if e.event_type == event_type]
        else:
            filtered = list(self._event_log)
        return filtered[-limit:]

    def clear_log(self) -> None:
        """Clear the event log."""
        self._event_log.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            **self._stats,
            "registered_handlers": sum(len(h) for h in self._handlers.values()),
            "global_handlers": len(self._global_handlers),
            "log_size": len(self._event_log),
        }


class EventFactory:
    """Factory for creating diagnosis events."""

    @staticmethod
    def session_started(user_id: UUID, session_id: UUID, session_number: int,
                        has_previous: bool = False) -> SessionStartedEvent:
        """Create session started event."""
        return SessionStartedEvent(user_id=user_id, session_id=session_id, session_number=session_number,
                                   has_previous_context=has_previous)

    @staticmethod
    def session_ended(user_id: UUID, session_id: UUID, duration: int, messages: int,
                      final_phase: DiagnosisPhase, primary_hypothesis: str | None = None) -> SessionEndedEvent:
        """Create session ended event."""
        return SessionEndedEvent(user_id=user_id, session_id=session_id, duration_minutes=duration,
                                 messages_exchanged=messages, final_phase=final_phase, primary_hypothesis=primary_hypothesis)

    @staticmethod
    def phase_transition(user_id: UUID, session_id: UUID, from_phase: DiagnosisPhase,
                         to_phase: DiagnosisPhase, confidence: Decimal) -> PhaseTransitionEvent:
        """Create phase transition event."""
        return PhaseTransitionEvent(user_id=user_id, session_id=session_id, from_phase=from_phase,
                                    to_phase=to_phase, confidence_at_transition=confidence)

    @staticmethod
    def symptom_extracted(user_id: UUID, session_id: UUID, symptom_id: UUID, name: str,
                          severity: SeverityLevel, confidence: Decimal, source: str) -> SymptomExtractedEvent:
        """Create symptom extracted event."""
        return SymptomExtractedEvent(user_id=user_id, session_id=session_id, symptom_id=symptom_id,
                                     symptom_name=name, severity=severity, confidence=confidence, extracted_from=source)

    @staticmethod
    def hypothesis_generated(user_id: UUID, session_id: UUID, hypothesis_id: UUID, name: str,
                             confidence: Decimal, dsm5_code: str | None, criteria_count: int) -> HypothesisGeneratedEvent:
        """Create hypothesis generated event."""
        return HypothesisGeneratedEvent(user_id=user_id, session_id=session_id, hypothesis_id=hypothesis_id,
                                        hypothesis_name=name, confidence=confidence, dsm5_code=dsm5_code,
                                        criteria_met_count=criteria_count)

    @staticmethod
    def hypothesis_challenged(user_id: UUID, session_id: UUID, hypothesis_id: UUID, name: str,
                              original: Decimal, adjusted: Decimal, challenges: list[str],
                              biases: list[str]) -> HypothesisChallengedEvent:
        """Create hypothesis challenged event."""
        return HypothesisChallengedEvent(user_id=user_id, session_id=session_id, hypothesis_id=hypothesis_id,
                                         hypothesis_name=name, original_confidence=original, adjusted_confidence=adjusted,
                                         challenges=challenges, biases_detected=biases)

    @staticmethod
    def safety_flag_raised(user_id: UUID, session_id: UUID, flag_type: str,
                           severity: str, trigger: str, action: str) -> SafetyFlagRaisedEvent:
        """Create safety flag raised event."""
        return SafetyFlagRaisedEvent(user_id=user_id, session_id=session_id, flag_type=flag_type,
                                     severity=severity, trigger_text=trigger, recommended_action=action)
