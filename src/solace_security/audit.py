"""Solace-AI Audit - Immutable audit logging for HIPAA compliance."""
from __future__ import annotations
import hashlib
import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of auditable events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    PHI_ACCESS = "phi_access"
    PHI_EXPORT = "phi_export"
    CONFIGURATION_CHANGE = "configuration_change"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    API_CALL = "api_call"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    USER_ACTION = "user_action"
    ADMIN_ACTION = "admin_action"


class AuditOutcome(str, Enum):
    """Outcome of audited operation."""
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"
    PARTIAL = "partial"


class AuditSeverity(str, Enum):
    """Severity level of audit event."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditSettings(BaseSettings):
    """Audit logging configuration."""
    enabled: bool = Field(default=True)
    log_phi_access: bool = Field(default=True)
    log_api_calls: bool = Field(default=True)
    retention_days: int = Field(default=2190)
    hash_algorithm: str = Field(default="sha256")
    enable_chain_verification: bool = Field(default=True)
    async_logging: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="AUDIT_", env_file=".env", extra="ignore")


class AuditActor(BaseModel):
    """Actor performing the audited action."""
    actor_id: str = Field(..., description="User/service ID")
    actor_type: str = Field(default="user", description="Type: user, service, system")
    ip_address: str | None = Field(default=None)
    user_agent: str | None = Field(default=None)
    session_id: str | None = Field(default=None)
    roles: list[str] = Field(default_factory=list)


class AuditResource(BaseModel):
    """Resource being accessed or modified."""
    resource_type: str = Field(..., description="Type of resource")
    resource_id: str = Field(..., description="Resource identifier")
    resource_name: str | None = Field(default=None)
    owner_id: str | None = Field(default=None)
    contains_phi: bool = Field(default=False)


class AuditEvent(BaseModel):
    """Immutable audit log entry."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType
    action: str = Field(..., description="Specific action performed")
    outcome: AuditOutcome
    severity: AuditSeverity = Field(default=AuditSeverity.INFO)
    actor: AuditActor
    resource: AuditResource | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = Field(default=None)
    correlation_id: str | None = Field(default=None)
    duration_ms: float | None = Field(default=None)
    error_message: str | None = Field(default=None)
    previous_hash: str | None = Field(default=None)
    event_hash: str | None = Field(default=None)
    model_config = {"frozen": True}

    def compute_hash(self, algorithm: str = "sha256") -> str:
        """Compute hash of event for integrity verification."""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "action": self.action,
            "outcome": self.outcome.value,
            "actor_id": self.actor.actor_id,
            "resource_id": self.resource.resource_id if self.resource else None,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.new(algorithm, canonical.encode()).hexdigest()

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "action": self.action,
            "outcome": self.outcome.value,
            "severity": self.severity.value,
            "actor_id": self.actor.actor_id,
            "actor_type": self.actor.actor_type,
            "ip_address": self.actor.ip_address,
            "resource_type": self.resource.resource_type if self.resource else None,
            "resource_id": self.resource.resource_id if self.resource else None,
            "contains_phi": self.resource.contains_phi if self.resource else False,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "duration_ms": self.duration_ms,
            "event_hash": self.event_hash,
        }


class AuditStore(ABC):
    """Abstract base for audit storage backends."""

    @abstractmethod
    def store(self, event: AuditEvent) -> None:
        """Store audit event."""
        pass

    @abstractmethod
    def query(self, filters: dict[str, Any], limit: int = 100,
              offset: int = 0) -> list[AuditEvent]:
        """Query audit events."""
        pass

    @abstractmethod
    def get_by_id(self, event_id: str) -> AuditEvent | None:
        """Get specific audit event."""
        pass

    @abstractmethod
    def verify_chain(self, start_id: str, end_id: str) -> bool:
        """Verify integrity of audit chain."""
        pass


class InMemoryAuditStore(AuditStore):
    """In-memory audit store for testing and development."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._index: dict[str, int] = {}
        self._lock = threading.Lock()

    def store(self, event: AuditEvent) -> None:
        with self._lock:
            self._index[event.event_id] = len(self._events)
            self._events.append(event)

    def query(self, filters: dict[str, Any], limit: int = 100,
              offset: int = 0) -> list[AuditEvent]:
        with self._lock:
            results = self._events
            if "event_type" in filters:
                results = [e for e in results if e.event_type == filters["event_type"]]
            if "actor_id" in filters:
                results = [e for e in results if e.actor.actor_id == filters["actor_id"]]
            if "resource_id" in filters:
                results = [e for e in results if e.resource and e.resource.resource_id == filters["resource_id"]]
            if "outcome" in filters:
                results = [e for e in results if e.outcome == filters["outcome"]]
            if "start_time" in filters:
                results = [e for e in results if e.timestamp >= filters["start_time"]]
            if "end_time" in filters:
                results = [e for e in results if e.timestamp <= filters["end_time"]]
            return results[offset:offset + limit]

    def get_by_id(self, event_id: str) -> AuditEvent | None:
        with self._lock:
            idx = self._index.get(event_id)
            return self._events[idx] if idx is not None else None

    def verify_chain(self, start_id: str, end_id: str) -> bool:
        with self._lock:
            start_idx = self._index.get(start_id)
            end_idx = self._index.get(end_id)
            if start_idx is None or end_idx is None or start_idx > end_idx:
                return False
            for i in range(start_idx + 1, end_idx + 1):
                event = self._events[i]
                prev_event = self._events[i - 1]
                if event.previous_hash != prev_event.event_hash:
                    return False
            return True

    def get_all(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._index.clear()


class AuditLogger:
    """High-level audit logging interface."""

    def __init__(self, store: AuditStore | None = None,
                 settings: AuditSettings | None = None) -> None:
        self._store = store or InMemoryAuditStore()
        self._settings = settings or AuditSettings()
        self._last_hash: str | None = None
        self._lock = threading.Lock()

    def log(self, event_type: AuditEventType, action: str, outcome: AuditOutcome,
            actor: AuditActor, resource: AuditResource | None = None,
            severity: AuditSeverity = AuditSeverity.INFO, **kwargs: Any) -> AuditEvent:
        """Log an audit event."""
        if not self._settings.enabled:
            return self._create_event(event_type, action, outcome, actor, resource, severity, **kwargs)
        with self._lock:
            event = self._create_event(event_type, action, outcome, actor, resource, severity,
                                       previous_hash=self._last_hash, **kwargs)
            event_hash = event.compute_hash(self._settings.hash_algorithm)
            event = AuditEvent(
                event_id=event.event_id, timestamp=event.timestamp, event_type=event.event_type,
                action=event.action, outcome=event.outcome, severity=event.severity,
                actor=event.actor, resource=event.resource, details=event.details,
                request_id=event.request_id, correlation_id=event.correlation_id,
                duration_ms=event.duration_ms, error_message=event.error_message,
                previous_hash=event.previous_hash, event_hash=event_hash,
            )
            self._store.store(event)
            self._last_hash = event_hash
            logger.info("audit_event", **event.to_log_dict())
            return event

    def _create_event(self, event_type: AuditEventType, action: str, outcome: AuditOutcome,
                      actor: AuditActor, resource: AuditResource | None,
                      severity: AuditSeverity, **kwargs: Any) -> AuditEvent:
        return AuditEvent(
            event_type=event_type, action=action, outcome=outcome,
            severity=severity, actor=actor, resource=resource,
            details=kwargs.get("details", {}), request_id=kwargs.get("request_id"),
            correlation_id=kwargs.get("correlation_id"),
            duration_ms=kwargs.get("duration_ms"), error_message=kwargs.get("error_message"),
            previous_hash=kwargs.get("previous_hash"),
        )

    def log_authentication(self, actor: AuditActor, success: bool,
                           method: str = "password", **kwargs: Any) -> AuditEvent:
        return self.log(
            AuditEventType.AUTHENTICATION, f"login:{method}",
            AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            actor, severity=AuditSeverity.INFO if success else AuditSeverity.WARNING, **kwargs,
        )

    def log_authorization(self, actor: AuditActor, resource: AuditResource,
                          action: str, allowed: bool, **kwargs: Any) -> AuditEvent:
        return self.log(
            AuditEventType.AUTHORIZATION, f"authorize:{action}",
            AuditOutcome.SUCCESS if allowed else AuditOutcome.DENIED,
            actor, resource, severity=AuditSeverity.WARNING if not allowed else AuditSeverity.INFO, **kwargs,
        )

    def log_data_access(self, actor: AuditActor, resource: AuditResource, **kwargs: Any) -> AuditEvent:
        event_type = AuditEventType.PHI_ACCESS if resource.contains_phi else AuditEventType.DATA_ACCESS
        return self.log(event_type, "read", AuditOutcome.SUCCESS, actor, resource, **kwargs)

    def log_data_modification(self, actor: AuditActor, resource: AuditResource,
                              operation: str = "update", **kwargs: Any) -> AuditEvent:
        return self.log(AuditEventType.DATA_MODIFICATION, operation, AuditOutcome.SUCCESS, actor, resource, **kwargs)

    def log_data_deletion(self, actor: AuditActor, resource: AuditResource, **kwargs: Any) -> AuditEvent:
        return self.log(AuditEventType.DATA_DELETION, "delete", AuditOutcome.SUCCESS, actor, resource,
                        severity=AuditSeverity.WARNING, **kwargs)

    def log_phi_export(self, actor: AuditActor, resource: AuditResource,
                       format: str = "json", **kwargs: Any) -> AuditEvent:
        return self.log(AuditEventType.PHI_EXPORT, f"export:{format}", AuditOutcome.SUCCESS, actor, resource,
                        severity=AuditSeverity.WARNING, **kwargs)

    def log_security_event(self, actor: AuditActor, event_name: str, severity: AuditSeverity,
                           resource: AuditResource | None = None, **kwargs: Any) -> AuditEvent:
        return self.log(AuditEventType.SECURITY_EVENT, event_name, AuditOutcome.SUCCESS,
                        actor, resource, severity, **kwargs)

    def query(self, filters: dict[str, Any], limit: int = 100, offset: int = 0) -> list[AuditEvent]:
        return self._store.query(filters, limit, offset)

    def get_by_id(self, event_id: str) -> AuditEvent | None:
        return self._store.get_by_id(event_id)

    def verify_integrity(self, start_id: str, end_id: str) -> bool:
        return self._store.verify_chain(start_id, end_id)


_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger singleton."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def create_audit_logger(store: AuditStore | None = None,
                        settings: AuditSettings | None = None) -> AuditLogger:
    """Factory function to create audit logger."""
    return AuditLogger(store, settings)
