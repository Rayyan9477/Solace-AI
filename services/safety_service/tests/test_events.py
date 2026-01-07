"""
Unit tests for Solace-AI Safety Service Events.
Tests domain events and event publishing.
"""
from __future__ import annotations
import asyncio
import pytest
from decimal import Decimal
from uuid import uuid4
from services.safety_service.src.events import (
    EventType, SafetyEvent, CrisisDetectedEvent, CrisisResolvedEvent,
    EscalationTriggeredEvent, EscalationAcknowledgedEvent, EscalationResolvedEvent,
    SafetyCheckCompletedEvent, RiskLevelChangedEvent, IncidentCreatedEvent,
    IncidentResolvedEvent, SafetyEventHandler, SafetyEventPublisher,
    AuditEventHandler, get_event_publisher, initialize_event_publisher,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_crisis_detected_value(self) -> None:
        """Test crisis detected event type value."""
        assert EventType.CRISIS_DETECTED.value == "safety.crisis.detected"

    def test_escalation_triggered_value(self) -> None:
        """Test escalation triggered event type value."""
        assert EventType.ESCALATION_TRIGGERED.value == "safety.escalation.triggered"


class TestSafetyEvent:
    """Tests for SafetyEvent base class."""

    def test_create_event(self) -> None:
        """Test creating a safety event."""
        event = SafetyEvent(
            event_type=EventType.CRISIS_DETECTED,
            user_id=uuid4(),
        )
        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.source == "safety-service"

    def test_to_json(self) -> None:
        """Test serializing event to JSON."""
        event = SafetyEvent(
            event_type=EventType.CRISIS_DETECTED,
            user_id=uuid4(),
        )
        json_str = event.to_json()
        assert "event_type" in json_str
        assert "safety.crisis.detected" in json_str

    def test_from_json(self) -> None:
        """Test deserializing event from JSON."""
        event = SafetyEvent(
            event_type=EventType.CRISIS_DETECTED,
            user_id=uuid4(),
        )
        json_str = event.to_json()
        restored = SafetyEvent.from_json(json_str)
        assert restored.event_id == event.event_id


class TestCrisisDetectedEvent:
    """Tests for CrisisDetectedEvent."""

    def test_create_event(self) -> None:
        """Test creating crisis detected event."""
        event = CrisisDetectedEvent(
            user_id=uuid4(),
            crisis_level="CRITICAL",
            risk_score=Decimal("0.95"),
            trigger_indicators=["KEYWORD:suicide"],
            detection_layers=[1, 2],
            requires_escalation=True,
        )
        assert event.event_type == EventType.CRISIS_DETECTED
        assert event.crisis_level == "CRITICAL"
        assert event.requires_escalation is True

    def test_detection_time(self) -> None:
        """Test detection time field."""
        event = CrisisDetectedEvent(
            crisis_level="HIGH",
            risk_score=Decimal("0.8"),
            detection_time_ms=5,
        )
        assert event.detection_time_ms == 5


class TestCrisisResolvedEvent:
    """Tests for CrisisResolvedEvent."""

    def test_create_event(self) -> None:
        """Test creating crisis resolved event."""
        event = CrisisResolvedEvent(
            user_id=uuid4(),
            crisis_level="HIGH",
            resolution_notes="Crisis stabilized",
            resolved_by=uuid4(),
            time_to_resolution_minutes=30,
        )
        assert event.event_type == EventType.CRISIS_RESOLVED
        assert event.resolution_notes == "Crisis stabilized"


class TestEscalationTriggeredEvent:
    """Tests for EscalationTriggeredEvent."""

    def test_create_event(self) -> None:
        """Test creating escalation triggered event."""
        escalation_id = uuid4()
        event = EscalationTriggeredEvent(
            user_id=uuid4(),
            escalation_id=escalation_id,
            priority="CRITICAL",
            crisis_level="CRITICAL",
            reason="Suicidal ideation detected",
            notification_channels=["SMS", "PAGER"],
        )
        assert event.event_type == EventType.ESCALATION_TRIGGERED
        assert event.escalation_id == escalation_id
        assert event.priority == "CRITICAL"


class TestEscalationAcknowledgedEvent:
    """Tests for EscalationAcknowledgedEvent."""

    def test_create_event(self) -> None:
        """Test creating escalation acknowledged event."""
        event = EscalationAcknowledgedEvent(
            escalation_id=uuid4(),
            acknowledged_by=uuid4(),
            time_to_acknowledge_seconds=120,
        )
        assert event.event_type == EventType.ESCALATION_ACKNOWLEDGED


class TestEscalationResolvedEvent:
    """Tests for EscalationResolvedEvent."""

    def test_create_event(self) -> None:
        """Test creating escalation resolved event."""
        event = EscalationResolvedEvent(
            escalation_id=uuid4(),
            resolved_by=uuid4(),
            resolution_notes="User stabilized, follow-up scheduled",
            time_to_resolution_minutes=45,
        )
        assert event.event_type == EventType.ESCALATION_RESOLVED


class TestSafetyCheckCompletedEvent:
    """Tests for SafetyCheckCompletedEvent."""

    def test_create_event(self) -> None:
        """Test creating safety check completed event."""
        event = SafetyCheckCompletedEvent(
            user_id=uuid4(),
            check_type="pre_check",
            is_safe=True,
            crisis_level="NONE",
            risk_score=Decimal("0.1"),
            detection_time_ms=3,
        )
        assert event.event_type == EventType.SAFETY_CHECK_COMPLETED
        assert event.is_safe is True


class TestRiskLevelChangedEvent:
    """Tests for RiskLevelChangedEvent."""

    def test_create_event(self) -> None:
        """Test creating risk level changed event."""
        event = RiskLevelChangedEvent(
            user_id=uuid4(),
            previous_level="LOW",
            new_level="ELEVATED",
            change_reason="Deteriorating messages detected",
            risk_trend="increasing",
        )
        assert event.event_type == EventType.RISK_LEVEL_CHANGED
        assert event.previous_level == "LOW"
        assert event.new_level == "ELEVATED"


class TestIncidentCreatedEvent:
    """Tests for IncidentCreatedEvent."""

    def test_create_event(self) -> None:
        """Test creating incident created event."""
        event = IncidentCreatedEvent(
            user_id=uuid4(),
            incident_id=uuid4(),
            severity="CRITICAL",
            crisis_level="CRITICAL",
            description="Active suicidal ideation",
        )
        assert event.event_type == EventType.INCIDENT_CREATED


class TestIncidentResolvedEvent:
    """Tests for IncidentResolvedEvent."""

    def test_create_event(self) -> None:
        """Test creating incident resolved event."""
        event = IncidentResolvedEvent(
            user_id=uuid4(),
            incident_id=uuid4(),
            resolution_notes="User connected with crisis counselor",
            resolved_by=uuid4(),
            time_to_resolution_minutes=60,
        )
        assert event.event_type == EventType.INCIDENT_RESOLVED


class TestAuditEventHandler:
    """Tests for AuditEventHandler."""

    def test_handle_event(self) -> None:
        """Test handling an event creates audit entry."""
        handler = AuditEventHandler()
        event = CrisisDetectedEvent(
            user_id=uuid4(),
            crisis_level="HIGH",
            risk_score=Decimal("0.8"),
        )
        asyncio.run(handler.handle(event))
        audit_log = handler.get_audit_log()
        assert len(audit_log) == 1
        assert audit_log[0]["event_type"] == EventType.CRISIS_DETECTED.value

    def test_audit_log_contains_metadata(self) -> None:
        """Test audit log contains event metadata."""
        handler = AuditEventHandler()
        user_id = uuid4()
        event = SafetyCheckCompletedEvent(
            user_id=user_id,
            check_type="pre_check",
            is_safe=True,
        )
        asyncio.run(handler.handle(event))
        audit_log = handler.get_audit_log()
        assert audit_log[0]["user_id"] == str(user_id)

    def test_clear_audit_log(self) -> None:
        """Test clearing audit log."""
        handler = AuditEventHandler()
        event = SafetyCheckCompletedEvent(check_type="test", is_safe=True)
        asyncio.run(handler.handle(event))
        handler.clear_audit_log()
        assert len(handler.get_audit_log()) == 0


class TestSafetyEventPublisher:
    """Tests for SafetyEventPublisher."""

    @pytest.fixture
    def publisher(self) -> SafetyEventPublisher:
        """Create a test publisher."""
        return SafetyEventPublisher()

    @pytest.mark.asyncio
    async def test_start_stop(self, publisher: SafetyEventPublisher) -> None:
        """Test starting and stopping publisher."""
        await publisher.start()
        assert publisher._running is True
        await publisher.stop()
        assert publisher._running is False

    @pytest.mark.asyncio
    async def test_register_handler(self, publisher: SafetyEventPublisher) -> None:
        """Test registering a handler."""
        handler = AuditEventHandler()
        publisher.register_handler(EventType.CRISIS_DETECTED, handler)
        assert EventType.CRISIS_DETECTED in publisher._handlers
        assert handler in publisher._handlers[EventType.CRISIS_DETECTED]

    @pytest.mark.asyncio
    async def test_publish_sync(self, publisher: SafetyEventPublisher) -> None:
        """Test synchronous event publishing."""
        handler = AuditEventHandler()
        publisher.register_handler(EventType.CRISIS_DETECTED, handler)
        event = CrisisDetectedEvent(
            crisis_level="HIGH",
            risk_score=Decimal("0.8"),
        )
        await publisher.publish_sync(event)
        assert len(handler.get_audit_log()) == 1

    @pytest.mark.asyncio
    async def test_publish_to_callback(self, publisher: SafetyEventPublisher) -> None:
        """Test publishing with callback."""
        received_events: list[SafetyEvent] = []

        async def callback(event: SafetyEvent) -> None:
            received_events.append(event)

        publisher.register_callback(callback)
        event = CrisisDetectedEvent(
            crisis_level="HIGH",
            risk_score=Decimal("0.8"),
        )
        await publisher.publish_sync(event)
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_publish_async(self, publisher: SafetyEventPublisher) -> None:
        """Test async event publishing through queue."""
        handler = AuditEventHandler()
        publisher.register_handler(EventType.CRISIS_DETECTED, handler)
        await publisher.start()
        event = CrisisDetectedEvent(
            crisis_level="HIGH",
            risk_score=Decimal("0.8"),
        )
        await publisher.publish(event)
        await asyncio.sleep(0.1)
        await publisher.stop()
        assert len(handler.get_audit_log()) == 1


class TestEventPublisherSingleton:
    """Tests for event publisher singleton."""

    def test_get_event_publisher_singleton(self) -> None:
        """Test singleton returns same instance."""
        publisher1 = get_event_publisher()
        publisher2 = get_event_publisher()
        assert publisher1 is publisher2

    @pytest.mark.asyncio
    async def test_initialize_event_publisher(self) -> None:
        """Test initializing event publisher."""
        publisher = await initialize_event_publisher()
        assert publisher._running is True
        await publisher.stop()
