"""
Solace-AI Orchestrator Service - Events.
Domain events for orchestrator lifecycle and agent coordination.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4
import structlog

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Types of orchestrator events."""
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_PROCESSED = "message_processed"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    CRISIS_DETECTED = "crisis_detected"
    SAFETY_ESCALATION = "safety_escalation"
    ROUTING_DECISION = "routing_decision"
    AGGREGATION_COMPLETE = "aggregation_complete"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    ERROR_OCCURRED = "error_occurred"


@dataclass(frozen=True)
class OrchestratorEvent:
    """Base event for orchestrator domain events."""
    event_id: UUID
    event_type: EventType
    timestamp: datetime
    session_id: UUID
    user_id: UUID
    payload: dict[str, Any]
    correlation_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": str(self.session_id),
            "user_id": str(self.user_id),
            "payload": self.payload,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrchestratorEvent:
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=UUID(data["session_id"]),
            user_id=UUID(data["user_id"]),
            payload=data.get("payload", {}),
            correlation_id=UUID(data["correlation_id"]) if data.get("correlation_id") else None,
            metadata=data.get("metadata", {}),
        )


class EventFactory:
    """Factory for creating orchestrator events."""

    @staticmethod
    def session_started(session_id: UUID, user_id: UUID, correlation_id: UUID | None = None) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.SESSION_STARTED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"action": "start"}, correlation_id=correlation_id,
        )

    @staticmethod
    def session_ended(session_id: UUID, user_id: UUID, reason: str = "normal") -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.SESSION_ENDED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"reason": reason},
        )

    @staticmethod
    def message_received(session_id: UUID, user_id: UUID, message_length: int) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.MESSAGE_RECEIVED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"message_length": message_length},
        )

    @staticmethod
    def message_processed(session_id: UUID, user_id: UUID, processing_time_ms: float, intent: str) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.MESSAGE_PROCESSED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"processing_time_ms": processing_time_ms, "intent": intent},
        )

    @staticmethod
    def agent_started(session_id: UUID, user_id: UUID, agent_type: str) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.AGENT_STARTED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"agent_type": agent_type},
        )

    @staticmethod
    def agent_completed(session_id: UUID, user_id: UUID, agent_type: str, confidence: float, processing_time_ms: float) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.AGENT_COMPLETED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id,
            payload={"agent_type": agent_type, "confidence": confidence, "processing_time_ms": processing_time_ms},
        )

    @staticmethod
    def agent_failed(session_id: UUID, user_id: UUID, agent_type: str, error: str) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.AGENT_FAILED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"agent_type": agent_type, "error": error},
        )

    @staticmethod
    def crisis_detected(session_id: UUID, user_id: UUID, risk_level: str, crisis_type: str | None) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.CRISIS_DETECTED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"risk_level": risk_level, "crisis_type": crisis_type},
        )

    @staticmethod
    def safety_escalation(session_id: UUID, user_id: UUID, reason: str, escalation_level: str) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.SAFETY_ESCALATION, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"reason": reason, "escalation_level": escalation_level},
        )

    @staticmethod
    def error_occurred(session_id: UUID, user_id: UUID, error_type: str, error_message: str) -> OrchestratorEvent:
        return OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.ERROR_OCCURRED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"error_type": error_type, "error_message": error_message},
        )


EventHandler = Callable[[OrchestratorEvent], None]


class EventBus:
    """In-process event bus for orchestrator events."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._event_history: list[OrchestratorEvent] = []
        self._max_history: int = 1000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe to specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug("event_handler_subscribed", event_type=event_type.value)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)
        logger.debug("global_event_handler_subscribed")

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe from event type."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def publish(self, event: OrchestratorEvent) -> None:
        """Publish event to all subscribers."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        logger.info("event_published", event_type=event.event_type.value, event_id=str(event.event_id))
        for handler in self._global_handlers:
            self._safe_invoke(handler, event)
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                self._safe_invoke(handler, event)

    def _safe_invoke(self, handler: EventHandler, event: OrchestratorEvent) -> None:
        """Invoke handler with error isolation."""
        try:
            handler(event)
        except Exception as e:
            logger.error("event_handler_error", event_type=event.event_type.value, error=str(e))

    def get_history(self, event_type: EventType | None = None, limit: int = 100) -> list[OrchestratorEvent]:
        """Get event history, optionally filtered by type."""
        filtered = self._event_history if event_type is None else [e for e in self._event_history if e.event_type == event_type]
        return filtered[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()


_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get singleton event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
