"""
Unit tests for Intervention Delivery Service.
Tests intervention planning, delivery, crisis handling, and session management.
"""
from __future__ import annotations
import pytest
from uuid import uuid4

from services.therapy_service.src.domain.interventions import (
    InterventionDeliveryService,
    InterventionDeliverySettings,
    InterventionPlan,
    DeliveredIntervention,
    InterventionQueue,
    InterventionType,
    InterventionPriority,
)
from services.therapy_service.src.domain.modalities import InterventionContext, ModalityRegistry
from services.therapy_service.src.schemas import TherapyModality, SessionPhase, SeverityLevel, RiskLevel


class TestInterventionDeliverySettings:
    """Tests for InterventionDeliverySettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings initialization."""
        settings = InterventionDeliverySettings()
        assert settings.max_interventions_per_session == 4
        assert settings.min_time_between_techniques_minutes == 5
        assert settings.enable_crisis_override is True
        assert settings.enable_adaptive_timing is True

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = InterventionDeliverySettings(
            max_interventions_per_session=6,
            enable_crisis_override=False,
        )
        assert settings.max_interventions_per_session == 6
        assert settings.enable_crisis_override is False


class TestInterventionDeliveryService:
    """Tests for InterventionDeliveryService functionality."""

    def test_service_initialization(self) -> None:
        """Test service initializes correctly."""
        service = InterventionDeliveryService()
        stats = service.get_statistics()
        assert stats["active_sessions"] == 0
        assert stats["total_planned"] == 0

    def test_create_session_queue(self) -> None:
        """Test creating a session queue."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        queue = service.create_session_queue(session_id)
        assert queue.session_id == session_id
        assert len(queue.planned) == 0
        assert len(queue.delivered) == 0

    def test_get_queue(self) -> None:
        """Test retrieving a session queue."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        service.create_session_queue(session_id)
        queue = service.get_queue(session_id)
        assert queue is not None
        assert queue.session_id == session_id

    def test_plan_intervention_technique(self) -> None:
        """Test planning a technique intervention."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I have negative thoughts",
        )
        plan = service.plan_intervention(
            session_id=session_id,
            user_id=user_id,
            context=context,
            preferred_type=InterventionType.TECHNIQUE,
        )
        assert plan is not None
        assert plan.intervention_type == InterventionType.TECHNIQUE
        assert plan.technique is not None

    def test_plan_intervention_grounding_for_severe(self) -> None:
        """Test grounding is planned for severe cases."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.SEVERE,
            current_concern="I'm overwhelmed",
        )
        plan = service.plan_intervention(
            session_id=session_id,
            user_id=user_id,
            context=context,
        )
        assert plan is not None
        assert plan.intervention_type == InterventionType.GROUNDING
        assert plan.priority == InterventionPriority.HIGH

    def test_plan_intervention_respects_max(self) -> None:
        """Test planning respects max interventions per session."""
        settings = InterventionDeliverySettings(max_interventions_per_session=2)
        service = InterventionDeliveryService(settings=settings)
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="test",
        )
        # Deliver max interventions
        queue = service.create_session_queue(session_id)
        queue.delivered.append(DeliveredIntervention(session_id=session_id, user_id=user_id))
        queue.delivered.append(DeliveredIntervention(session_id=session_id, user_id=user_id))
        plan = service.plan_intervention(session_id, user_id, context)
        assert plan is None

    def test_deliver_intervention_technique(self) -> None:
        """Test delivering a technique intervention."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="negative thoughts",
        )
        plan = service.plan_intervention(session_id, user_id, context)
        delivered = service.deliver_intervention(session_id, plan, "I feel sad", context)
        assert delivered is not None
        assert len(delivered.response_text) > 0
        assert delivered.completed_at is not None

    def test_deliver_intervention_grounding(self) -> None:
        """Test delivering a grounding intervention."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.SEVERE,
            current_concern="panic",
        )
        plan = service.plan_intervention(session_id, user_id, context)
        delivered = service.deliver_intervention(session_id, plan, "I can't breathe", context)
        assert "breath" in delivered.response_text.lower() or "ground" in delivered.response_text.lower()

    def test_deliver_intervention_reflection(self) -> None:
        """Test delivering a reflection intervention."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        plan = InterventionPlan(
            session_id=session_id,
            user_id=user_id,
            intervention_type=InterventionType.REFLECTION,
        )
        service.create_session_queue(session_id)
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="exploring",
        )
        delivered = service.deliver_intervention(session_id, plan, "I've been thinking", context)
        assert len(delivered.response_text) > 0

    def test_deliver_intervention_validation(self) -> None:
        """Test delivering a validation intervention."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        plan = InterventionPlan(
            session_id=session_id,
            user_id=user_id,
            intervention_type=InterventionType.VALIDATION,
        )
        service.create_session_queue(session_id)
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="validation needed",
        )
        delivered = service.deliver_intervention(session_id, plan, "I feel bad about this", context)
        assert "valid" in delivered.response_text.lower() or "sense" in delivered.response_text.lower()

    def test_handle_crisis(self) -> None:
        """Test crisis handling."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        service.create_session_queue(session_id)
        delivered = service.handle_crisis(session_id, user_id, RiskLevel.HIGH)
        assert delivered.intervention_type == InterventionType.CRISIS
        assert "988" in delivered.response_text or "safe" in delivered.response_text.lower()
        stats = service.get_statistics()
        assert stats["crisis_interventions"] == 1

    def test_get_next_intervention(self) -> None:
        """Test getting next planned intervention."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="test",
        )
        service.plan_intervention(session_id, user_id, context)
        next_plan = service.get_next_intervention(session_id)
        assert next_plan is not None

    def test_get_next_intervention_priority_ordering(self) -> None:
        """Test next intervention respects priority."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        queue = service.create_session_queue(session_id)
        # Add low priority
        queue.planned.append(InterventionPlan(
            session_id=session_id, user_id=user_id,
            intervention_type=InterventionType.REFLECTION,
            priority=InterventionPriority.LOW,
        ))
        # Add high priority
        queue.planned.append(InterventionPlan(
            session_id=session_id, user_id=user_id,
            intervention_type=InterventionType.GROUNDING,
            priority=InterventionPriority.HIGH,
        ))
        next_plan = service.get_next_intervention(session_id)
        assert next_plan.priority == InterventionPriority.HIGH

    def test_rate_intervention(self) -> None:
        """Test rating a delivered intervention."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="test",
        )
        plan = service.plan_intervention(session_id, user_id, context)
        delivered = service.deliver_intervention(session_id, plan, "test input", context)
        result = service.rate_intervention(
            delivered.intervention_id, session_id,
            effectiveness=0.8, engagement=0.9,
        )
        assert result is True
        assert delivered.effectiveness_rating == 0.8
        assert delivered.user_engagement == 0.9

    def test_get_session_summary(self) -> None:
        """Test getting session intervention summary."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="test",
        )
        plan = service.plan_intervention(session_id, user_id, context)
        service.deliver_intervention(session_id, plan, "test", context)
        summary = service.get_session_summary(session_id)
        assert summary["delivered_count"] == 1
        assert "interventions" in summary

    def test_clear_session(self) -> None:
        """Test clearing session queue."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        service.create_session_queue(session_id)
        service.clear_session(session_id)
        assert service.get_queue(session_id) is None

    def test_statistics_tracking(self) -> None:
        """Test statistics are tracked correctly."""
        service = InterventionDeliveryService()
        session_id = uuid4()
        user_id = uuid4()
        context = InterventionContext(
            user_id=user_id,
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="thoughts",
        )
        plan = service.plan_intervention(session_id, user_id, context)
        service.deliver_intervention(session_id, plan, "test", context)
        stats = service.get_statistics()
        assert stats["total_planned"] >= 1
        assert stats["total_delivered"] >= 1


class TestInterventionPlan:
    """Tests for InterventionPlan data structure."""

    def test_plan_defaults(self) -> None:
        """Test intervention plan defaults."""
        plan = InterventionPlan()
        assert plan.intervention_type == InterventionType.TECHNIQUE
        assert plan.priority == InterventionPriority.NORMAL

    def test_plan_with_technique(self) -> None:
        """Test plan with technique attached."""
        registry = ModalityRegistry()
        techniques = registry.get_all_techniques()
        plan = InterventionPlan(
            session_id=uuid4(),
            user_id=uuid4(),
            technique=techniques[0],
            modality=techniques[0].modality,
            estimated_duration_minutes=techniques[0].duration_minutes,
        )
        assert plan.technique is not None
        assert plan.estimated_duration_minutes == techniques[0].duration_minutes


class TestDeliveredIntervention:
    """Tests for DeliveredIntervention data structure."""

    def test_delivered_defaults(self) -> None:
        """Test delivered intervention defaults."""
        delivered = DeliveredIntervention()
        assert delivered.duration_seconds == 0
        assert delivered.user_engagement == 0.0
        assert delivered.follow_up_needed is False

    def test_delivered_with_data(self) -> None:
        """Test delivered intervention with data."""
        delivered = DeliveredIntervention(
            technique_name="Thought Record",
            response_text="Let's examine that thought",
            notes=["Applied CBT technique"],
        )
        assert delivered.technique_name == "Thought Record"
        assert len(delivered.notes) == 1


class TestInterventionQueue:
    """Tests for InterventionQueue data structure."""

    def test_queue_initialization(self) -> None:
        """Test queue initializes empty."""
        session_id = uuid4()
        queue = InterventionQueue(session_id=session_id)
        assert queue.session_id == session_id
        assert len(queue.planned) == 0
        assert len(queue.delivered) == 0
        assert queue.current is None
