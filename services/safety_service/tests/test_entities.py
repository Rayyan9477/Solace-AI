"""
Unit tests for Solace-AI Safety Service Domain Entities.
Tests SafetyAssessment, SafetyPlan, SafetyIncident, and UserRiskProfile entities.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
from services.safety_service.src.domain.entities import (
    SafetyPlan, SafetyPlanStatus, WarningSign, CopingStrategy, EmergencyContact,
    SafeEnvironmentAction, SafetyAssessment, AssessmentType, SafetyIncident,
    IncidentSeverity, IncidentStatus, UserRiskProfile,
)


class TestWarningSign:
    """Tests for WarningSign model."""

    def test_create_warning_sign(self) -> None:
        """Test creating a warning sign."""
        sign = WarningSign(
            description="Isolating from friends",
            severity_level=3,
            category="behavioral",
            recognition_cues=["avoiding calls", "canceling plans"],
        )
        assert sign.description == "Isolating from friends"
        assert sign.severity_level == 3
        assert len(sign.recognition_cues) == 2

    def test_warning_sign_id_generated(self) -> None:
        """Test that warning sign ID is auto-generated."""
        sign = WarningSign(description="Test", severity_level=1, category="test")
        assert sign.sign_id is not None


class TestCopingStrategy:
    """Tests for CopingStrategy model."""

    def test_create_coping_strategy(self) -> None:
        """Test creating a coping strategy."""
        strategy = CopingStrategy(
            name="Deep breathing",
            description="Practice 4-7-8 breathing technique",
            category="relaxation",
            effectiveness_rating=8,
        )
        assert strategy.name == "Deep breathing"
        assert strategy.effectiveness_rating == 8
        assert strategy.is_active is True

    def test_coping_strategy_usage_tracking(self) -> None:
        """Test usage tracking fields."""
        strategy = CopingStrategy(name="Test", description="Test strategy")
        assert strategy.times_used == 0
        assert strategy.last_used_at is None


class TestEmergencyContact:
    """Tests for EmergencyContact model."""

    def test_create_emergency_contact(self) -> None:
        """Test creating an emergency contact."""
        contact = EmergencyContact(
            name="John Doe",
            relationship="friend",
            phone="555-1234",
            priority_order=1,
        )
        assert contact.name == "John Doe"
        assert contact.is_professional is False

    def test_professional_contact(self) -> None:
        """Test professional contact flag."""
        contact = EmergencyContact(
            name="Dr. Smith",
            relationship="therapist",
            phone="555-5678",
            is_professional=True,
            priority_order=1,
        )
        assert contact.is_professional is True


class TestSafetyPlan:
    """Tests for SafetyPlan entity."""

    @pytest.fixture
    def basic_plan(self) -> SafetyPlan:
        """Create a basic safety plan."""
        return SafetyPlan(user_id=uuid4())

    @pytest.fixture
    def complete_plan(self) -> SafetyPlan:
        """Create a complete safety plan."""
        plan = SafetyPlan(user_id=uuid4())
        plan.warning_signs.append(WarningSign(
            description="Warning 1", severity_level=2, category="behavioral"
        ))
        plan.coping_strategies.extend([
            CopingStrategy(name="Strategy 1", description="Description 1"),
            CopingStrategy(name="Strategy 2", description="Description 2"),
        ])
        plan.emergency_contacts.append(EmergencyContact(
            name="Contact 1", relationship="friend", priority_order=1
        ))
        return plan

    def test_create_plan(self, basic_plan: SafetyPlan) -> None:
        """Test creating a safety plan."""
        assert basic_plan.status == SafetyPlanStatus.DRAFT
        assert basic_plan.version == 1
        assert basic_plan.plan_id is not None

    def test_plan_is_complete(self, complete_plan: SafetyPlan) -> None:
        """Test is_complete property."""
        assert complete_plan.is_complete is True

    def test_plan_incomplete(self, basic_plan: SafetyPlan) -> None:
        """Test incomplete plan detection."""
        assert basic_plan.is_complete is False

    def test_plan_expiration(self) -> None:
        """Test plan expiration detection."""
        plan = SafetyPlan(
            user_id=uuid4(),
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert plan.is_expired is True

    def test_plan_not_expired(self) -> None:
        """Test non-expired plan."""
        plan = SafetyPlan(
            user_id=uuid4(),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert plan.is_expired is False

    def test_activate_plan(self, complete_plan: SafetyPlan) -> None:
        """Test activating a complete plan."""
        complete_plan.activate()
        assert complete_plan.status == SafetyPlanStatus.ACTIVE
        assert complete_plan.next_review_due is not None

    def test_activate_incomplete_plan_fails(self, basic_plan: SafetyPlan) -> None:
        """Test that activating incomplete plan raises error."""
        with pytest.raises(ValueError, match="Cannot activate incomplete"):
            basic_plan.activate()

    def test_archive_plan(self, basic_plan: SafetyPlan) -> None:
        """Test archiving a plan."""
        basic_plan.archive()
        assert basic_plan.status == SafetyPlanStatus.ARCHIVED

    def test_add_warning_sign(self, basic_plan: SafetyPlan) -> None:
        """Test adding a warning sign."""
        sign = WarningSign(description="Test", severity_level=2, category="test")
        basic_plan.add_warning_sign(sign)
        assert len(basic_plan.warning_signs) == 1

    def test_add_coping_strategy(self, basic_plan: SafetyPlan) -> None:
        """Test adding a coping strategy."""
        strategy = CopingStrategy(name="Test", description="Test strategy")
        basic_plan.add_coping_strategy(strategy)
        assert len(basic_plan.coping_strategies) == 1

    def test_add_emergency_contact_sorted(self, basic_plan: SafetyPlan) -> None:
        """Test that emergency contacts are sorted by priority."""
        contact1 = EmergencyContact(name="C1", relationship="r1", priority_order=3)
        contact2 = EmergencyContact(name="C2", relationship="r2", priority_order=1)
        basic_plan.add_emergency_contact(contact1)
        basic_plan.add_emergency_contact(contact2)
        assert basic_plan.emergency_contacts[0].name == "C2"

    def test_days_until_review(self) -> None:
        """Test days until review calculation."""
        plan = SafetyPlan(
            user_id=uuid4(),
            next_review_due=datetime.now(timezone.utc) + timedelta(days=5),
        )
        assert plan.days_until_review == 5


class TestSafetyAssessment:
    """Tests for SafetyAssessment entity."""

    def test_create_assessment(self) -> None:
        """Test creating a safety assessment."""
        assessment = SafetyAssessment(
            user_id=uuid4(),
            content_assessed="Test content",
            risk_score=Decimal("0.3"),
            crisis_level="LOW",
        )
        assert assessment.is_safe is True
        assert assessment.assessment_type == AssessmentType.PRE_CHECK

    def test_assessment_with_risk_factors(self) -> None:
        """Test assessment with risk factors."""
        assessment = SafetyAssessment(
            user_id=uuid4(),
            content_assessed="Crisis content",
            risk_score=Decimal("0.85"),
            crisis_level="HIGH",
            is_safe=False,
            risk_factors=[{"type": "keyword", "severity": 0.8}],
            requires_escalation=True,
        )
        assert len(assessment.risk_factors) == 1
        assert assessment.requires_escalation is True

    def test_risk_score_coercion(self) -> None:
        """Test risk score is coerced to Decimal."""
        assessment = SafetyAssessment(
            user_id=uuid4(),
            content_assessed="Test",
            risk_score=0.5,
        )
        assert isinstance(assessment.risk_score, Decimal)


class TestSafetyIncident:
    """Tests for SafetyIncident entity."""

    @pytest.fixture
    def incident(self) -> SafetyIncident:
        """Create a test incident."""
        return SafetyIncident(
            user_id=uuid4(),
            severity=IncidentSeverity.HIGH,
            crisis_level="HIGH",
            description="Test crisis incident",
        )

    def test_create_incident(self, incident: SafetyIncident) -> None:
        """Test creating an incident."""
        assert incident.status == IncidentStatus.OPEN
        assert incident.severity == IncidentSeverity.HIGH

    def test_acknowledge_incident(self, incident: SafetyIncident) -> None:
        """Test acknowledging an incident."""
        clinician_id = uuid4()
        incident.acknowledge(clinician_id)
        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert incident.assigned_clinician_id == clinician_id
        assert incident.acknowledged_at is not None

    def test_acknowledge_non_open_fails(self, incident: SafetyIncident) -> None:
        """Test that acknowledging non-open incident fails."""
        incident.acknowledge(uuid4())
        incident.resolve("Resolved")
        with pytest.raises(ValueError):
            incident.acknowledge(uuid4())

    def test_start_progress(self, incident: SafetyIncident) -> None:
        """Test starting progress on incident."""
        incident.start_progress()
        assert incident.status == IncidentStatus.IN_PROGRESS

    def test_resolve_incident(self, incident: SafetyIncident) -> None:
        """Test resolving an incident."""
        incident.acknowledge(uuid4())
        incident.resolve("Issue resolved successfully")
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_at is not None
        assert incident.resolution_notes == "Issue resolved successfully"

    def test_close_incident(self, incident: SafetyIncident) -> None:
        """Test closing an incident."""
        incident.acknowledge(uuid4())
        incident.resolve("Resolved")
        incident.close()
        assert incident.status == IncidentStatus.CLOSED
        assert incident.closed_at is not None

    def test_close_unresolved_fails(self, incident: SafetyIncident) -> None:
        """Test that closing unresolved incident fails."""
        with pytest.raises(ValueError, match="Can only close resolved"):
            incident.close()

    def test_escalate_incident(self, incident: SafetyIncident) -> None:
        """Test escalating an incident."""
        incident.escalate("Need senior clinician")
        assert incident.status == IncidentStatus.ESCALATED
        assert "Escalated" in incident.actions_taken[-1]

    def test_time_to_acknowledge(self, incident: SafetyIncident) -> None:
        """Test time to acknowledge calculation."""
        assert incident.time_to_acknowledge is None
        incident.acknowledge(uuid4())
        assert incident.time_to_acknowledge is not None

    def test_time_to_resolve(self, incident: SafetyIncident) -> None:
        """Test time to resolve calculation."""
        assert incident.time_to_resolve is None
        incident.acknowledge(uuid4())
        incident.resolve("Done")
        assert incident.time_to_resolve is not None

    def test_is_overdue(self) -> None:
        """Test overdue detection."""
        incident = SafetyIncident(
            user_id=uuid4(),
            severity=IncidentSeverity.HIGH,
            crisis_level="HIGH",
            description="Test",
            follow_up_required=True,
            follow_up_due=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert incident.is_overdue is True


class TestUserRiskProfile:
    """Tests for UserRiskProfile entity."""

    @pytest.fixture
    def profile(self) -> UserRiskProfile:
        """Create a test profile."""
        return UserRiskProfile(user_id=uuid4())

    def test_create_profile(self, profile: UserRiskProfile) -> None:
        """Test creating a risk profile."""
        assert profile.baseline_risk_level == "NONE"
        assert profile.total_assessments == 0
        assert profile.high_risk_flag is False

    def test_record_assessment(self, profile: UserRiskProfile) -> None:
        """Test recording an assessment."""
        profile.record_assessment("LOW")
        assert profile.total_assessments == 1
        assert profile.current_risk_level == "LOW"
        assert profile.last_assessment_at is not None

    def test_record_high_risk_assessment(self, profile: UserRiskProfile) -> None:
        """Test recording high risk assessment sets flags."""
        profile.record_assessment("HIGH", is_escalation=True)
        assert profile.crisis_events_count == 1
        assert profile.escalations_count == 1
        assert profile.high_risk_flag is True
        assert profile.recent_escalation is True

    def test_record_critical_assessment(self, profile: UserRiskProfile) -> None:
        """Test recording critical assessment."""
        profile.record_assessment("CRITICAL")
        assert profile.crisis_events_count == 1
        assert profile.last_crisis_at is not None

    def test_record_incident(self, profile: UserRiskProfile) -> None:
        """Test recording an incident."""
        profile.record_incident(IncidentSeverity.HIGH)
        assert profile.total_incidents == 1
        assert profile.high_risk_flag is True

    def test_clear_recent_flags(self, profile: UserRiskProfile) -> None:
        """Test clearing recent flags."""
        profile.record_assessment("HIGH", is_escalation=True)
        profile.clear_recent_flags()
        assert profile.recent_escalation is False
        assert profile.high_risk_flag is True
