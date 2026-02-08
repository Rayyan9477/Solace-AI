"""Solace-AI Audit - Immutable audit logging for HIPAA compliance."""
from __future__ import annotations
import hashlib
import hmac as hmac_mod
import json
import os
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
    hmac_key: str = Field(
        default="",
        description="HMAC key for audit chain integrity. Set via AUDIT_HMAC_KEY env var.",
    )
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

    def compute_hash(self, algorithm: str = "sha256", hmac_key: str = "") -> str:
        """Compute hash of event for integrity verification.

        Uses HMAC when hmac_key is provided for tamper-resistant chain integrity.
        Falls back to plain hash when no key is available (testing only).
        """
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
        if hmac_key:
            return hmac_mod.new(
                hmac_key.encode(), canonical.encode(), algorithm,
            ).hexdigest()
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


class AsyncAuditStore(ABC):
    """Abstract base for async audit storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize store (create tables, etc.)."""
        pass

    @abstractmethod
    async def store(self, event: AuditEvent) -> None:
        """Store audit event."""
        pass

    @abstractmethod
    async def query(self, filters: dict[str, Any], limit: int = 100,
                    offset: int = 0) -> list[AuditEvent]:
        """Query audit events."""
        pass

    @abstractmethod
    async def get_by_id(self, event_id: str) -> AuditEvent | None:
        """Get specific audit event."""
        pass

    @abstractmethod
    async def verify_chain(self, start_id: str, end_id: str) -> bool:
        """Verify integrity of audit chain."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close store connections."""
        pass


class PostgresAuditStore(AsyncAuditStore):
    """PostgreSQL-backed audit store for production use.

    Stores audit events in an append-only PostgreSQL table with
    HMAC-based chain integrity verification.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS audit_events (
        event_id VARCHAR(36) PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        event_type VARCHAR(50) NOT NULL,
        action VARCHAR(200) NOT NULL,
        outcome VARCHAR(20) NOT NULL,
        severity VARCHAR(20) NOT NULL DEFAULT 'info',
        actor_id VARCHAR(200) NOT NULL,
        actor_type VARCHAR(20) NOT NULL DEFAULT 'user',
        ip_address VARCHAR(45),
        user_agent TEXT,
        session_id VARCHAR(200),
        actor_roles JSONB NOT NULL DEFAULT '[]',
        resource_type VARCHAR(100),
        resource_id VARCHAR(200),
        resource_name VARCHAR(300),
        resource_owner_id VARCHAR(200),
        contains_phi BOOLEAN NOT NULL DEFAULT FALSE,
        details JSONB NOT NULL DEFAULT '{}',
        request_id VARCHAR(200),
        correlation_id VARCHAR(200),
        duration_ms DOUBLE PRECISION,
        error_message TEXT,
        previous_hash VARCHAR(128),
        event_hash VARCHAR(128),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

    CREATE_INDEXES_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)",
        "CREATE INDEX IF NOT EXISTS idx_audit_actor_id ON audit_events(actor_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_resource_id ON audit_events(resource_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_audit_contains_phi ON audit_events(contains_phi) WHERE contains_phi = TRUE",
    ]

    INSERT_SQL = """
    INSERT INTO audit_events (
        event_id, timestamp, event_type, action, outcome, severity,
        actor_id, actor_type, ip_address, user_agent, session_id, actor_roles,
        resource_type, resource_id, resource_name, resource_owner_id, contains_phi,
        details, request_id, correlation_id, duration_ms, error_message,
        previous_hash, event_hash
    ) VALUES (
        $1, $2, $3, $4, $5, $6,
        $7, $8, $9, $10, $11, $12,
        $13, $14, $15, $16, $17,
        $18, $19, $20, $21, $22,
        $23, $24
    )
    """

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None

    async def initialize(self) -> None:
        """Create connection pool and ensure table exists."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg is required for PostgresAuditStore")

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(self.CREATE_TABLE_SQL)
            for idx_sql in self.CREATE_INDEXES_SQL:
                await conn.execute(idx_sql)
        logger.info("postgres_audit_store_initialized")

    async def store(self, event: AuditEvent) -> None:
        """Store audit event in PostgreSQL."""
        if self._pool is None:
            raise RuntimeError("PostgresAuditStore not initialized — call initialize() first")

        async with self._pool.acquire() as conn:
            await conn.execute(
                self.INSERT_SQL,
                event.event_id,
                event.timestamp,
                event.event_type.value,
                event.action,
                event.outcome.value,
                event.severity.value,
                event.actor.actor_id,
                event.actor.actor_type,
                event.actor.ip_address,
                event.actor.user_agent,
                event.actor.session_id,
                json.dumps(event.actor.roles),
                event.resource.resource_type if event.resource else None,
                event.resource.resource_id if event.resource else None,
                event.resource.resource_name if event.resource else None,
                event.resource.owner_id if event.resource else None,
                event.resource.contains_phi if event.resource else False,
                json.dumps(event.details),
                event.request_id,
                event.correlation_id,
                event.duration_ms,
                event.error_message,
                event.previous_hash,
                event.event_hash,
            )

    async def query(self, filters: dict[str, Any], limit: int = 100,
                    offset: int = 0) -> list[AuditEvent]:
        """Query audit events with filters."""
        if self._pool is None:
            raise RuntimeError("PostgresAuditStore not initialized")

        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1

        if "event_type" in filters:
            conditions.append(f"event_type = ${param_idx}")
            params.append(filters["event_type"].value if isinstance(filters["event_type"], Enum) else filters["event_type"])
            param_idx += 1
        if "actor_id" in filters:
            conditions.append(f"actor_id = ${param_idx}")
            params.append(filters["actor_id"])
            param_idx += 1
        if "resource_id" in filters:
            conditions.append(f"resource_id = ${param_idx}")
            params.append(filters["resource_id"])
            param_idx += 1
        if "outcome" in filters:
            conditions.append(f"outcome = ${param_idx}")
            params.append(filters["outcome"].value if isinstance(filters["outcome"], Enum) else filters["outcome"])
            param_idx += 1
        if "start_time" in filters:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(filters["start_time"])
            param_idx += 1
        if "end_time" in filters:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(filters["end_time"])
            param_idx += 1

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM audit_events{where} ORDER BY timestamp ASC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [self._row_to_event(row) for row in rows]

    async def get_by_id(self, event_id: str) -> AuditEvent | None:
        """Get specific audit event by ID."""
        if self._pool is None:
            raise RuntimeError("PostgresAuditStore not initialized")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM audit_events WHERE event_id = $1", event_id,
            )
        return self._row_to_event(row) if row else None

    async def verify_chain(self, start_id: str, end_id: str) -> bool:
        """Verify integrity of audit chain between two events."""
        if self._pool is None:
            raise RuntimeError("PostgresAuditStore not initialized")

        async with self._pool.acquire() as conn:
            start_row = await conn.fetchrow(
                "SELECT timestamp FROM audit_events WHERE event_id = $1", start_id,
            )
            end_row = await conn.fetchrow(
                "SELECT timestamp FROM audit_events WHERE event_id = $1", end_id,
            )
            if not start_row or not end_row:
                return False

            rows = await conn.fetch(
                "SELECT event_hash, previous_hash FROM audit_events "
                "WHERE timestamp >= $1 AND timestamp <= $2 ORDER BY timestamp ASC",
                start_row["timestamp"], end_row["timestamp"],
            )

        if len(rows) < 2:
            return True

        for i in range(1, len(rows)):
            if rows[i]["previous_hash"] != rows[i - 1]["event_hash"]:
                return False
        return True

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _row_to_event(row: Any) -> AuditEvent:
        """Convert a database row to an AuditEvent."""
        actor = AuditActor(
            actor_id=row["actor_id"],
            actor_type=row["actor_type"],
            ip_address=row["ip_address"],
            user_agent=row["user_agent"],
            session_id=row["session_id"],
            roles=json.loads(row["actor_roles"]) if isinstance(row["actor_roles"], str) else (row["actor_roles"] or []),
        )
        resource = None
        if row["resource_type"] and row["resource_id"]:
            resource = AuditResource(
                resource_type=row["resource_type"],
                resource_id=row["resource_id"],
                resource_name=row["resource_name"],
                owner_id=row["resource_owner_id"],
                contains_phi=row["contains_phi"],
            )
        return AuditEvent(
            event_id=row["event_id"],
            timestamp=row["timestamp"],
            event_type=AuditEventType(row["event_type"]),
            action=row["action"],
            outcome=AuditOutcome(row["outcome"]),
            severity=AuditSeverity(row["severity"]),
            actor=actor,
            resource=resource,
            details=json.loads(row["details"]) if isinstance(row["details"], str) else (row["details"] or {}),
            request_id=row["request_id"],
            correlation_id=row["correlation_id"],
            duration_ms=row["duration_ms"],
            error_message=row["error_message"],
            previous_hash=row["previous_hash"],
            event_hash=row["event_hash"],
        )


class AuditLogger:
    """High-level audit logging interface."""

    def __init__(self, store: AuditStore,
                 settings: AuditSettings | None = None) -> None:
        self._store = store
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
            event_hash = event.compute_hash(self._settings.hash_algorithm, self._settings.hmac_key)
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


class AsyncAuditLogger:
    """Async audit logging interface for production use with PostgresAuditStore."""

    def __init__(self, store: AsyncAuditStore,
                 settings: AuditSettings | None = None) -> None:
        self._store = store
        self._settings = settings or AuditSettings()
        self._last_hash: str | None = None
        import asyncio
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the underlying store."""
        await self._store.initialize()

    async def close(self) -> None:
        """Close the underlying store."""
        await self._store.close()

    async def log(self, event_type: AuditEventType, action: str, outcome: AuditOutcome,
                  actor: AuditActor, resource: AuditResource | None = None,
                  severity: AuditSeverity = AuditSeverity.INFO, **kwargs: Any) -> AuditEvent:
        """Log an audit event asynchronously."""
        if not self._settings.enabled:
            return self._create_event(event_type, action, outcome, actor, resource, severity, **kwargs)
        async with self._lock:
            event = self._create_event(event_type, action, outcome, actor, resource, severity,
                                       previous_hash=self._last_hash, **kwargs)
            event_hash = event.compute_hash(self._settings.hash_algorithm, self._settings.hmac_key)
            event = AuditEvent(
                event_id=event.event_id, timestamp=event.timestamp, event_type=event.event_type,
                action=event.action, outcome=event.outcome, severity=event.severity,
                actor=event.actor, resource=event.resource, details=event.details,
                request_id=event.request_id, correlation_id=event.correlation_id,
                duration_ms=event.duration_ms, error_message=event.error_message,
                previous_hash=event.previous_hash, event_hash=event_hash,
            )
            await self._store.store(event)
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

    async def log_authentication(self, actor: AuditActor, success: bool,
                                  method: str = "password", **kwargs: Any) -> AuditEvent:
        return await self.log(
            AuditEventType.AUTHENTICATION, f"login:{method}",
            AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            actor, severity=AuditSeverity.INFO if success else AuditSeverity.WARNING, **kwargs,
        )

    async def log_data_access(self, actor: AuditActor, resource: AuditResource, **kwargs: Any) -> AuditEvent:
        event_type = AuditEventType.PHI_ACCESS if resource.contains_phi else AuditEventType.DATA_ACCESS
        return await self.log(event_type, "read", AuditOutcome.SUCCESS, actor, resource, **kwargs)

    async def log_data_modification(self, actor: AuditActor, resource: AuditResource,
                                     operation: str = "update", **kwargs: Any) -> AuditEvent:
        return await self.log(AuditEventType.DATA_MODIFICATION, operation, AuditOutcome.SUCCESS, actor, resource, **kwargs)

    async def query(self, filters: dict[str, Any], limit: int = 100, offset: int = 0) -> list[AuditEvent]:
        return await self._store.query(filters, limit, offset)

    async def get_by_id(self, event_id: str) -> AuditEvent | None:
        return await self._store.get_by_id(event_id)

    async def verify_integrity(self, start_id: str, end_id: str) -> bool:
        return await self._store.verify_chain(start_id, end_id)


_audit_logger: AuditLogger | None = None
_async_audit_logger: AsyncAuditLogger | None = None
_audit_logger_lock = threading.Lock()


def configure_audit_logger(store: AuditStore,
                           settings: AuditSettings | None = None) -> AuditLogger:
    """Configure and set the global sync audit logger."""
    global _audit_logger
    with _audit_logger_lock:
        _audit_logger = AuditLogger(store, settings)
    return _audit_logger


def configure_async_audit_logger(store: AsyncAuditStore,
                                  settings: AuditSettings | None = None) -> AsyncAuditLogger:
    """Configure and set the global async audit logger."""
    global _async_audit_logger
    _async_audit_logger = AsyncAuditLogger(store, settings)
    return _async_audit_logger


def get_audit_logger() -> AuditLogger:
    """Get global audit logger singleton (thread-safe).

    Raises RuntimeError if not configured. Use configure_audit_logger() first.
    In test environments, use InMemoryAuditStore.
    """
    if _audit_logger is None:
        raise RuntimeError(
            "Audit logger not configured. Call configure_audit_logger() during startup. "
            "For tests, use: configure_audit_logger(InMemoryAuditStore())"
        )
    return _audit_logger


def get_async_audit_logger() -> AsyncAuditLogger:
    """Get global async audit logger singleton.

    Raises RuntimeError if not configured. Use configure_async_audit_logger() first.
    """
    if _async_audit_logger is None:
        raise RuntimeError(
            "Async audit logger not configured. Call configure_async_audit_logger() during startup."
        )
    return _async_audit_logger


def create_audit_logger(store: AuditStore,
                        settings: AuditSettings | None = None) -> AuditLogger:
    """Factory function to create audit logger."""
    return AuditLogger(store, settings)
