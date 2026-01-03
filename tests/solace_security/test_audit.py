"""Unit tests for audit module."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import pytest
from solace_security.audit import (
    AuditEventType,
    AuditOutcome,
    AuditSeverity,
    AuditSettings,
    AuditActor,
    AuditResource,
    AuditEvent,
    InMemoryAuditStore,
    AuditLogger,
    get_audit_logger,
    create_audit_logger,
)


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_event_types(self):
        assert AuditEventType.AUTHENTICATION.value == "authentication"
        assert AuditEventType.AUTHORIZATION.value == "authorization"
        assert AuditEventType.PHI_ACCESS.value == "phi_access"
        assert AuditEventType.DATA_DELETION.value == "data_deletion"


class TestAuditOutcome:
    """Tests for AuditOutcome enum."""

    def test_outcome_values(self):
        assert AuditOutcome.SUCCESS.value == "success"
        assert AuditOutcome.FAILURE.value == "failure"
        assert AuditOutcome.DENIED.value == "denied"


class TestAuditSeverity:
    """Tests for AuditSeverity enum."""

    def test_severity_values(self):
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.CRITICAL.value == "critical"


class TestAuditSettings:
    """Tests for AuditSettings."""

    def test_default_settings(self):
        settings = AuditSettings()
        assert settings.enabled
        assert settings.log_phi_access
        assert settings.retention_days == 2190

    def test_custom_settings(self):
        settings = AuditSettings(retention_days=365, async_logging=True)
        assert settings.retention_days == 365
        assert settings.async_logging


class TestAuditActor:
    """Tests for AuditActor model."""

    def test_create_actor(self):
        actor = AuditActor(
            actor_id="user123",
            actor_type="user",
            ip_address="192.168.1.1",
            roles=["admin"]
        )
        assert actor.actor_id == "user123"
        assert actor.actor_type == "user"
        assert "admin" in actor.roles


class TestAuditResource:
    """Tests for AuditResource model."""

    def test_create_resource(self):
        resource = AuditResource(
            resource_type="user",
            resource_id="user456",
            owner_id="user456",
            contains_phi=True
        )
        assert resource.resource_type == "user"
        assert resource.contains_phi


class TestAuditEvent:
    """Tests for AuditEvent model."""

    @pytest.fixture
    def actor(self):
        return AuditActor(actor_id="user123", actor_type="user")

    @pytest.fixture
    def resource(self):
        return AuditResource(resource_type="session", resource_id="sess123")

    def test_create_event(self, actor, resource):
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            action="read",
            outcome=AuditOutcome.SUCCESS,
            actor=actor,
            resource=resource
        )
        assert event.event_id
        assert event.timestamp
        assert event.event_type == AuditEventType.DATA_ACCESS

    def test_compute_hash(self, actor):
        event = AuditEvent(
            event_type=AuditEventType.AUTHENTICATION,
            action="login",
            outcome=AuditOutcome.SUCCESS,
            actor=actor
        )
        hash1 = event.compute_hash()
        hash2 = event.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_to_log_dict(self, actor, resource):
        event = AuditEvent(
            event_type=AuditEventType.PHI_ACCESS,
            action="read",
            outcome=AuditOutcome.SUCCESS,
            actor=actor,
            resource=resource
        )
        log_dict = event.to_log_dict()
        assert "event_id" in log_dict
        assert log_dict["actor_id"] == "user123"
        assert log_dict["resource_type"] == "session"


class TestInMemoryAuditStore:
    """Tests for InMemoryAuditStore."""

    @pytest.fixture
    def store(self):
        return InMemoryAuditStore()

    @pytest.fixture
    def sample_event(self):
        return AuditEvent(
            event_type=AuditEventType.AUTHENTICATION,
            action="login",
            outcome=AuditOutcome.SUCCESS,
            actor=AuditActor(actor_id="user123", actor_type="user")
        )

    def test_store_event(self, store, sample_event):
        store.store(sample_event)
        retrieved = store.get_by_id(sample_event.event_id)
        assert retrieved is not None
        assert retrieved.event_id == sample_event.event_id

    def test_query_by_event_type(self, store):
        actor = AuditActor(actor_id="user123", actor_type="user")
        store.store(AuditEvent(
            event_type=AuditEventType.AUTHENTICATION, action="login",
            outcome=AuditOutcome.SUCCESS, actor=actor
        ))
        store.store(AuditEvent(
            event_type=AuditEventType.DATA_ACCESS, action="read",
            outcome=AuditOutcome.SUCCESS, actor=actor
        ))
        results = store.query({"event_type": AuditEventType.AUTHENTICATION})
        assert len(results) == 1

    def test_query_by_actor_id(self, store):
        store.store(AuditEvent(
            event_type=AuditEventType.AUTHENTICATION, action="login",
            outcome=AuditOutcome.SUCCESS,
            actor=AuditActor(actor_id="user123", actor_type="user")
        ))
        store.store(AuditEvent(
            event_type=AuditEventType.AUTHENTICATION, action="login",
            outcome=AuditOutcome.SUCCESS,
            actor=AuditActor(actor_id="user456", actor_type="user")
        ))
        results = store.query({"actor_id": "user123"})
        assert len(results) == 1

    def test_query_by_outcome(self, store):
        actor = AuditActor(actor_id="user123", actor_type="user")
        store.store(AuditEvent(
            event_type=AuditEventType.AUTHENTICATION, action="login",
            outcome=AuditOutcome.SUCCESS, actor=actor
        ))
        store.store(AuditEvent(
            event_type=AuditEventType.AUTHENTICATION, action="login",
            outcome=AuditOutcome.FAILURE, actor=actor
        ))
        results = store.query({"outcome": AuditOutcome.FAILURE})
        assert len(results) == 1

    def test_query_pagination(self, store):
        actor = AuditActor(actor_id="user123", actor_type="user")
        for i in range(10):
            store.store(AuditEvent(
                event_type=AuditEventType.DATA_ACCESS, action="read",
                outcome=AuditOutcome.SUCCESS, actor=actor
            ))
        results = store.query({}, limit=5)
        assert len(results) == 5
        results = store.query({}, limit=5, offset=5)
        assert len(results) == 5

    def test_get_all(self, store, sample_event):
        store.store(sample_event)
        all_events = store.get_all()
        assert len(all_events) == 1

    def test_clear(self, store, sample_event):
        store.store(sample_event)
        store.clear()
        assert len(store.get_all()) == 0


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.fixture
    def audit_logger(self):
        return AuditLogger()

    @pytest.fixture
    def actor(self):
        return AuditActor(actor_id="user123", actor_type="user", ip_address="10.0.0.1")

    @pytest.fixture
    def resource(self):
        return AuditResource(resource_type="session", resource_id="sess123", contains_phi=False)

    def test_log_event(self, audit_logger, actor, resource):
        event = audit_logger.log(
            AuditEventType.DATA_ACCESS, "read", AuditOutcome.SUCCESS,
            actor, resource
        )
        assert event.event_id
        assert event.event_hash

    def test_log_authentication_success(self, audit_logger, actor):
        event = audit_logger.log_authentication(actor, success=True, method="password")
        assert event.event_type == AuditEventType.AUTHENTICATION
        assert event.outcome == AuditOutcome.SUCCESS

    def test_log_authentication_failure(self, audit_logger, actor):
        event = audit_logger.log_authentication(actor, success=False)
        assert event.outcome == AuditOutcome.FAILURE
        assert event.severity == AuditSeverity.WARNING

    def test_log_authorization(self, audit_logger, actor, resource):
        event = audit_logger.log_authorization(actor, resource, "read", allowed=True)
        assert event.event_type == AuditEventType.AUTHORIZATION
        assert event.outcome == AuditOutcome.SUCCESS

    def test_log_authorization_denied(self, audit_logger, actor, resource):
        event = audit_logger.log_authorization(actor, resource, "delete", allowed=False)
        assert event.outcome == AuditOutcome.DENIED
        assert event.severity == AuditSeverity.WARNING

    def test_log_data_access(self, audit_logger, actor, resource):
        event = audit_logger.log_data_access(actor, resource)
        assert event.event_type == AuditEventType.DATA_ACCESS

    def test_log_phi_access(self, audit_logger, actor):
        phi_resource = AuditResource(
            resource_type="patient_record", resource_id="pr123", contains_phi=True
        )
        event = audit_logger.log_data_access(actor, phi_resource)
        assert event.event_type == AuditEventType.PHI_ACCESS

    def test_log_data_modification(self, audit_logger, actor, resource):
        event = audit_logger.log_data_modification(actor, resource, operation="update")
        assert event.event_type == AuditEventType.DATA_MODIFICATION
        assert event.action == "update"

    def test_log_data_deletion(self, audit_logger, actor, resource):
        event = audit_logger.log_data_deletion(actor, resource)
        assert event.event_type == AuditEventType.DATA_DELETION
        assert event.severity == AuditSeverity.WARNING

    def test_log_phi_export(self, audit_logger, actor, resource):
        event = audit_logger.log_phi_export(actor, resource, format="csv")
        assert event.event_type == AuditEventType.PHI_EXPORT
        assert event.action == "export:csv"

    def test_log_security_event(self, audit_logger, actor):
        event = audit_logger.log_security_event(
            actor, "suspicious_activity", AuditSeverity.CRITICAL
        )
        assert event.event_type == AuditEventType.SECURITY_EVENT
        assert event.severity == AuditSeverity.CRITICAL

    def test_query(self, audit_logger, actor, resource):
        audit_logger.log(AuditEventType.DATA_ACCESS, "read", AuditOutcome.SUCCESS, actor, resource)
        audit_logger.log(AuditEventType.DATA_ACCESS, "write", AuditOutcome.SUCCESS, actor, resource)
        results = audit_logger.query({"event_type": AuditEventType.DATA_ACCESS})
        assert len(results) == 2

    def test_get_by_id(self, audit_logger, actor, resource):
        event = audit_logger.log(AuditEventType.DATA_ACCESS, "read", AuditOutcome.SUCCESS, actor, resource)
        retrieved = audit_logger.get_by_id(event.event_id)
        assert retrieved is not None
        assert retrieved.event_id == event.event_id

    def test_chain_integrity(self, audit_logger, actor, resource):
        event1 = audit_logger.log(AuditEventType.DATA_ACCESS, "read", AuditOutcome.SUCCESS, actor, resource)
        event2 = audit_logger.log(AuditEventType.DATA_ACCESS, "write", AuditOutcome.SUCCESS, actor, resource)
        event3 = audit_logger.log(AuditEventType.DATA_ACCESS, "delete", AuditOutcome.SUCCESS, actor, resource)
        assert event2.previous_hash == event1.event_hash
        assert event3.previous_hash == event2.event_hash

    def test_verify_integrity(self, audit_logger, actor, resource):
        event1 = audit_logger.log(AuditEventType.DATA_ACCESS, "read", AuditOutcome.SUCCESS, actor, resource)
        event2 = audit_logger.log(AuditEventType.DATA_ACCESS, "write", AuditOutcome.SUCCESS, actor, resource)
        assert audit_logger.verify_integrity(event1.event_id, event2.event_id)


class TestGlobalAuditLogger:
    """Tests for global audit logger singleton."""

    def test_get_audit_logger_singleton(self):
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_audit_logger(self):
        logger = create_audit_logger()
        assert isinstance(logger, AuditLogger)

    def test_create_audit_logger_with_store(self):
        store = InMemoryAuditStore()
        logger = create_audit_logger(store=store)
        actor = AuditActor(actor_id="test", actor_type="user")
        logger.log(AuditEventType.AUTHENTICATION, "login", AuditOutcome.SUCCESS, actor)
        assert len(store.get_all()) == 1
