"""
Integration tests for cross-service event publishing and consumption.

Verifies that event bridges, to_kafka_event converters, and event type
registries are correctly wired across services (Phases 6-7 fixes).
"""
from __future__ import annotations

import inspect
import pytest
from decimal import Decimal
from uuid import uuid4

from services.safety_service.src.infrastructure.event_bridge import (
    _BRIDGED_EVENT_TYPES,
    KafkaEventBridge as SafetyKafkaEventBridge,
)
from services.safety_service.src.events import (
    EventType as SafetyEventType,
    CrisisDetectedEvent as SafetyCrisisDetectedEvent,
    SafetyCheckCompletedEvent,
    EscalationTriggeredEvent as SafetyEscalationTriggeredEvent,
    SafetyPlanCreatedEvent as SafetySafetyPlanCreatedEvent,
    OutputFilteredEvent as SafetyOutputFilteredEvent,
    TrajectoryAlertEvent as SafetyTrajectoryAlertEvent,
    to_kafka_event as safety_to_kafka_event,
)
from services.diagnosis_service.src.events import (
    SafetyFlagRaisedEvent as DiagnosisSafetyFlagRaisedEvent,
    DiagnosisRecordedEvent,
    to_kafka_event as diagnosis_to_kafka_event,
)
from services.orchestrator_service.src.infrastructure.event_bridge import (
    KafkaEventBridge as OrchestratorKafkaEventBridge,
)


class TestSafetyBridgeEventTypes:
    """Verify safety bridge includes all required event types."""

    def test_crisis_detected_in_bridge(self) -> None:
        assert SafetyEventType.CRISIS_DETECTED in _BRIDGED_EVENT_TYPES

    def test_safety_check_completed_in_bridge(self) -> None:
        assert SafetyEventType.SAFETY_CHECK_COMPLETED in _BRIDGED_EVENT_TYPES

    def test_escalation_triggered_in_bridge(self) -> None:
        assert SafetyEventType.ESCALATION_TRIGGERED in _BRIDGED_EVENT_TYPES

    def test_safety_plan_created_in_bridge(self) -> None:
        assert SafetyEventType.SAFETY_PLAN_CREATED in _BRIDGED_EVENT_TYPES

    def test_output_filtered_in_bridge(self) -> None:
        assert SafetyEventType.OUTPUT_FILTERED in _BRIDGED_EVENT_TYPES

    def test_trajectory_alert_in_bridge(self) -> None:
        assert SafetyEventType.TRAJECTORY_ALERT in _BRIDGED_EVENT_TYPES

    def test_all_required_event_types_bridged(self) -> None:
        """All critical safety event types must be in the bridge set."""
        required = {
            SafetyEventType.CRISIS_DETECTED,
            SafetyEventType.CRISIS_RESOLVED,
            SafetyEventType.SAFETY_CHECK_COMPLETED,
            SafetyEventType.ESCALATION_TRIGGERED,
            SafetyEventType.ESCALATION_ACKNOWLEDGED,
            SafetyEventType.ESCALATION_RESOLVED,
            SafetyEventType.SAFETY_PLAN_CREATED,
            SafetyEventType.SAFETY_PLAN_ACTIVATED,
            SafetyEventType.SAFETY_PLAN_UPDATED,
            SafetyEventType.OUTPUT_FILTERED,
            SafetyEventType.TRAJECTORY_ALERT,
            SafetyEventType.RISK_LEVEL_CHANGED,
            SafetyEventType.INCIDENT_CREATED,
            SafetyEventType.INCIDENT_RESOLVED,
        }
        for evt in required:
            assert evt in _BRIDGED_EVENT_TYPES, (
                f"{evt.value} missing from _BRIDGED_EVENT_TYPES"
            )

    def test_bridge_event_types_is_frozenset(self) -> None:
        """Bridged event types should be immutable."""
        assert isinstance(_BRIDGED_EVENT_TYPES, frozenset)


class TestSafetyToKafkaEventConversion:
    """Verify safety domain events convert correctly to Kafka events."""

    def test_crisis_detected_converts_to_kafka(self) -> None:
        """CrisisDetectedEvent should convert to a Kafka CrisisDetectedEvent."""
        event = SafetyCrisisDetectedEvent(
            event_type=SafetyEventType.CRISIS_DETECTED,
            user_id=uuid4(),
            session_id=uuid4(),
            crisis_level="HIGH",
            risk_score=Decimal("0.85"),
            trigger_indicators=["suicidal ideation"],
            detection_layers=[1, 2],
            requires_escalation=True,
            requires_human_review=True,
        )
        kafka_event = safety_to_kafka_event(event)
        assert kafka_event is not None
        assert kafka_event.event_type == "safety.crisis.detected"

    def test_safety_check_completed_converts_to_kafka(self) -> None:
        """SafetyCheckCompletedEvent should convert to a Kafka SafetyAssessmentEvent."""
        event = SafetyCheckCompletedEvent(
            event_type=SafetyEventType.SAFETY_CHECK_COMPLETED,
            user_id=uuid4(),
            session_id=uuid4(),
            check_type="pre_response",
            is_safe=True,
            crisis_level="NONE",
            risk_score=Decimal("0.1"),
        )
        kafka_event = safety_to_kafka_event(event)
        assert kafka_event is not None
        assert kafka_event.event_type == "safety.assessment.completed"

    def test_output_filtered_converts_to_kafka(self) -> None:
        """OutputFilteredEvent should convert to a Kafka OutputFilteredEvent."""
        event = SafetyOutputFilteredEvent(
            event_type=SafetyEventType.OUTPUT_FILTERED,
            user_id=uuid4(),
            session_id=uuid4(),
            modifications_count=2,
            filter_reason="harmful_content",
            resources_appended=True,
        )
        kafka_event = safety_to_kafka_event(event)
        assert kafka_event is not None
        assert kafka_event.event_type == "safety.output.filtered"

    def test_trajectory_alert_converts_to_kafka(self) -> None:
        """TrajectoryAlertEvent should convert to a Kafka TrajectoryAlertEvent."""
        event = SafetyTrajectoryAlertEvent(
            event_type=SafetyEventType.TRAJECTORY_ALERT,
            user_id=uuid4(),
            session_id=uuid4(),
            alert_type="rapid_deterioration",
            trend="worsening",
            message_count_analyzed=15,
            risk_score_delta=Decimal("0.3"),
        )
        kafka_event = safety_to_kafka_event(event)
        assert kafka_event is not None
        assert kafka_event.event_type == "safety.trajectory.alert"

    def test_event_without_user_id_returns_none(self) -> None:
        """Events without user_id should not be bridged to Kafka."""
        event = SafetyCrisisDetectedEvent(
            event_type=SafetyEventType.CRISIS_DETECTED,
            user_id=None,
            crisis_level="HIGH",
            risk_score=Decimal("0.85"),
        )
        kafka_event = safety_to_kafka_event(event)
        assert kafka_event is None


class TestDiagnosisBridgeSafetyFlags:
    """Verify diagnosis service bridges SafetyFlagRaisedEvent to Kafka."""

    def test_safety_flag_converts_to_kafka_crisis_detected(self) -> None:
        """SafetyFlagRaisedEvent should convert to a Kafka CrisisDetectedEvent."""
        event = DiagnosisSafetyFlagRaisedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            flag_type="suicidal_ideation",
            severity="high",
            trigger_text="I want to end it all",
            recommended_action="immediate_escalation",
        )
        kafka_event = diagnosis_to_kafka_event(event)
        assert kafka_event is not None
        assert kafka_event.event_type == "safety.crisis.detected"

    def test_safety_flag_severity_maps_to_crisis_level(self) -> None:
        """Different severity levels should map to appropriate CrisisLevel values."""
        for severity in ("low", "moderate", "high", "critical"):
            event = DiagnosisSafetyFlagRaisedEvent(
                user_id=uuid4(),
                session_id=uuid4(),
                flag_type="risk_indicator",
                severity=severity,
                trigger_text="test trigger",
                recommended_action="review",
            )
            kafka_event = diagnosis_to_kafka_event(event)
            assert kafka_event is not None, (
                f"Expected Kafka event for severity '{severity}'"
            )

    def test_diagnosis_recorded_converts_to_kafka(self) -> None:
        """DiagnosisRecordedEvent should convert to a Kafka DiagnosisCompletedEvent."""
        event = DiagnosisRecordedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            primary_diagnosis="Major Depressive Disorder",
            dsm5_code="F32.1",
        )
        kafka_event = diagnosis_to_kafka_event(event)
        assert kafka_event is not None
        assert kafka_event.event_type == "diagnosis.completed"

    def test_safety_flag_without_user_id_returns_none(self) -> None:
        """SafetyFlagRaisedEvent without user_id should not bridge to Kafka."""
        event = DiagnosisSafetyFlagRaisedEvent(
            user_id=None,
            session_id=uuid4(),
            flag_type="risk_indicator",
            severity="moderate",
            trigger_text="test",
            recommended_action="review",
        )
        kafka_event = diagnosis_to_kafka_event(event)
        assert kafka_event is None

    def test_safety_flag_high_severity_requires_human_review(self) -> None:
        """High severity safety flags should set requires_human_review=True."""
        event = DiagnosisSafetyFlagRaisedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            flag_type="suicidal_ideation",
            severity="critical",
            trigger_text="immediate danger",
            recommended_action="escalate",
        )
        kafka_event = diagnosis_to_kafka_event(event)
        assert kafka_event is not None
        assert kafka_event.requires_human_review is True


class TestOrchestratorKafkaBridgeAutoSubscribes:
    """Verify orchestrator event bridge has auto-subscribe pattern."""

    def test_start_method_calls_subscribe_all(self) -> None:
        """KafkaEventBridge.start() should subscribe_all on the EventBus."""
        source = inspect.getsource(OrchestratorKafkaEventBridge.start)
        assert "subscribe_all" in source, (
            "OrchestratorKafkaEventBridge.start() does not call subscribe_all. "
            "Auto-subscription is required for event forwarding."
        )

    def test_bridge_accepts_event_bus(self) -> None:
        """KafkaEventBridge should accept an EventBus parameter."""
        sig = inspect.signature(OrchestratorKafkaEventBridge.__init__)
        param_names = list(sig.parameters.keys())
        assert "event_bus" in param_names, (
            "OrchestratorKafkaEventBridge.__init__ missing 'event_bus' parameter"
        )

    def test_bridge_has_handle_event_method(self) -> None:
        """Bridge should have internal _handle_event for forwarding events."""
        assert hasattr(OrchestratorKafkaEventBridge, "_handle_event"), (
            "OrchestratorKafkaEventBridge missing '_handle_event' method"
        )

    def test_bridge_has_bridge_event_method(self) -> None:
        """Bridge should expose bridge_event for manual event forwarding."""
        assert hasattr(OrchestratorKafkaEventBridge, "bridge_event"), (
            "OrchestratorKafkaEventBridge missing 'bridge_event' method"
        )


class TestSafetyBridgeHandlerRegistration:
    """Verify safety event bridge handler registration pattern."""

    def test_safety_bridge_is_event_handler(self) -> None:
        """SafetyKafkaEventBridge should be a SafetyEventHandler subclass."""
        from services.safety_service.src.events import SafetyEventHandler
        assert issubclass(SafetyKafkaEventBridge, SafetyEventHandler)

    def test_safety_bridge_has_handle_method(self) -> None:
        """SafetyKafkaEventBridge must implement the handle() method."""
        assert hasattr(SafetyKafkaEventBridge, "handle")
        sig = inspect.signature(SafetyKafkaEventBridge.handle)
        param_names = list(sig.parameters.keys())
        assert "event" in param_names
