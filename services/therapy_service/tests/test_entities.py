"""
Unit tests for Domain Entities.
Tests TreatmentPlanEntity, TherapySessionEntity, and related entities.
"""
from __future__ import annotations
from decimal import Decimal
from uuid import uuid4
import pytest

from services.therapy_service.src.domain.entities import (
    TreatmentPlanEntity, TherapySessionEntity, TreatmentGoalEntity,
    HomeworkEntity, InterventionEntity,
)
from services.therapy_service.src.schemas import (
    SessionPhase, TreatmentPhase, TherapyModality, SeverityLevel,
    RiskLevel, SteppedCareLevel, GoalStatus, HomeworkStatus, ResponseStatus,
)


class TestTreatmentGoalEntity:
    """Tests for TreatmentGoalEntity."""

    def test_create_goal(self) -> None:
        """Test goal creation."""
        goal = TreatmentGoalEntity(
            description="Reduce anxiety symptoms",
            milestones=["Complete thought records", "Practice grounding"],
        )
        assert goal.description == "Reduce anxiety symptoms"
        assert len(goal.milestones) == 2
        assert goal.status == GoalStatus.NOT_STARTED
        assert goal.progress_percentage == 0

    def test_update_progress(self) -> None:
        """Test progress update."""
        goal = TreatmentGoalEntity(description="Test goal")
        goal.update_progress(50)
        assert goal.progress_percentage == 50
        assert goal.status == GoalStatus.IN_PROGRESS

    def test_progress_to_achieved(self) -> None:
        """Test goal achieved at 100%."""
        goal = TreatmentGoalEntity(description="Test goal")
        goal.update_progress(100)
        assert goal.status == GoalStatus.ACHIEVED

    def test_progress_clamped(self) -> None:
        """Test progress is clamped to 0-100."""
        goal = TreatmentGoalEntity(description="Test goal")
        goal.update_progress(150)
        assert goal.progress_percentage == 100
        goal.update_progress(-10)
        assert goal.progress_percentage == 0

    def test_complete_milestone(self) -> None:
        """Test completing milestones."""
        goal = TreatmentGoalEntity(
            description="Test goal",
            milestones=["Step 1", "Step 2"],
        )
        assert goal.complete_milestone("Step 1") is True
        assert "Step 1" in goal.completed_milestones
        assert goal.complete_milestone("Step 1") is False  # Already completed

    def test_all_milestones_achieves_goal(self) -> None:
        """Test completing all milestones achieves goal."""
        goal = TreatmentGoalEntity(
            description="Test goal",
            milestones=["Step 1", "Step 2"],
        )
        goal.complete_milestone("Step 1")
        goal.complete_milestone("Step 2")
        assert goal.status == GoalStatus.ACHIEVED
        assert goal.progress_percentage == 100


class TestHomeworkEntity:
    """Tests for HomeworkEntity."""

    def test_create_homework(self) -> None:
        """Test homework creation."""
        homework = HomeworkEntity(
            title="Thought Record",
            description="Complete 3 thought records this week",
        )
        assert homework.title == "Thought Record"
        assert homework.status == HomeworkStatus.ASSIGNED

    def test_mark_completed(self) -> None:
        """Test marking homework completed."""
        homework = HomeworkEntity(title="Test Homework")
        homework.mark_completed(notes="Done well", rating=5)
        assert homework.status == HomeworkStatus.COMPLETED
        assert homework.completed_at is not None
        assert homework.rating == 5

    def test_mark_partially_completed(self) -> None:
        """Test marking homework partially completed."""
        homework = HomeworkEntity(title="Test Homework")
        homework.mark_partially_completed(notes="Did half")
        assert homework.status == HomeworkStatus.PARTIALLY_COMPLETED

    def test_is_overdue(self) -> None:
        """Test overdue detection."""
        from datetime import datetime, timezone, timedelta
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        homework = HomeworkEntity(title="Test", due_date=past_date)
        assert homework.is_overdue() is True

    def test_completed_not_overdue(self) -> None:
        """Test completed homework not overdue."""
        from datetime import datetime, timezone, timedelta
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        homework = HomeworkEntity(title="Test", due_date=past_date)
        homework.mark_completed()
        assert homework.is_overdue() is False


class TestInterventionEntity:
    """Tests for InterventionEntity."""

    def test_create_intervention(self) -> None:
        """Test intervention creation."""
        intervention = InterventionEntity(
            technique_name="Thought Record",
            modality=TherapyModality.CBT,
        )
        assert intervention.technique_name == "Thought Record"
        assert intervention.modality == TherapyModality.CBT
        assert intervention.completed is False

    def test_complete_intervention(self) -> None:
        """Test completing intervention."""
        intervention = InterventionEntity(technique_name="Test")
        intervention.complete(engagement=Decimal("0.8"))
        assert intervention.completed is True
        assert intervention.completed_at is not None
        assert intervention.engagement_score == Decimal("0.8")

    def test_engagement_score_clamped(self) -> None:
        """Test engagement score is clamped."""
        intervention = InterventionEntity(technique_name="Test")
        intervention.complete(engagement=Decimal("1.5"))
        assert intervention.engagement_score == Decimal("1")


class TestTherapySessionEntity:
    """Tests for TherapySessionEntity."""

    def test_create_session(self) -> None:
        """Test session creation."""
        user_id = uuid4()
        plan_id = uuid4()
        session = TherapySessionEntity(
            user_id=user_id,
            treatment_plan_id=plan_id,
            session_number=1,
        )
        assert session.user_id == user_id
        assert session.current_phase == SessionPhase.PRE_SESSION
        assert session.is_active is True

    def test_transition_phase_valid(self) -> None:
        """Test valid phase transitions."""
        session = TherapySessionEntity()
        assert session.transition_phase(SessionPhase.OPENING) is True
        assert session.current_phase == SessionPhase.OPENING
        assert session.transition_phase(SessionPhase.WORKING) is True
        assert session.current_phase == SessionPhase.WORKING

    def test_transition_phase_invalid(self) -> None:
        """Test invalid phase transitions."""
        session = TherapySessionEntity()
        assert session.transition_phase(SessionPhase.CLOSING) is False
        assert session.current_phase == SessionPhase.PRE_SESSION

    def test_add_intervention(self) -> None:
        """Test adding intervention."""
        session = TherapySessionEntity()
        intervention = InterventionEntity(technique_name="Grounding")
        session.add_intervention(intervention)
        assert len(session.interventions) == 1
        assert intervention.session_id == session.session_id

    def test_assign_homework(self) -> None:
        """Test assigning homework."""
        session = TherapySessionEntity()
        homework = HomeworkEntity(title="Practice breathing")
        session.assign_homework(homework)
        assert len(session.homework_assigned) == 1
        assert homework.session_id == session.session_id

    def test_record_skill(self) -> None:
        """Test recording skills."""
        session = TherapySessionEntity()
        session.record_skill("grounding")
        session.record_skill("grounding")  # Duplicate
        assert session.skills_practiced == ["grounding"]

    def test_set_risk_level(self) -> None:
        """Test setting risk level."""
        session = TherapySessionEntity()
        session.set_risk_level(RiskLevel.HIGH, "suicidal_ideation")
        assert session.current_risk == RiskLevel.HIGH
        assert "suicidal_ideation" in session.safety_flags

    def test_end_session(self) -> None:
        """Test ending session."""
        session = TherapySessionEntity()
        session.end_session(summary="Good session", next_focus="CBT techniques")
        assert session.is_active is False
        assert session.ended_at is not None
        assert session.summary == "Good session"

    def test_events_recorded(self) -> None:
        """Test domain events are recorded."""
        session = TherapySessionEntity()
        session.transition_phase(SessionPhase.OPENING)
        events = session.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "phase_transitioned"


class TestTreatmentPlanEntity:
    """Tests for TreatmentPlanEntity."""

    def test_create_plan(self) -> None:
        """Test treatment plan creation."""
        user_id = uuid4()
        plan = TreatmentPlanEntity(
            user_id=user_id,
            primary_diagnosis="Major Depressive Disorder",
            severity=SeverityLevel.MODERATE,
            primary_modality=TherapyModality.CBT,
        )
        assert plan.user_id == user_id
        assert plan.current_phase == TreatmentPhase.FOUNDATION
        assert plan.is_active is True

    def test_add_goal(self) -> None:
        """Test adding goals."""
        plan = TreatmentPlanEntity()
        goal = plan.add_goal("Reduce depression", ["Track mood", "Exercise"])
        assert len(plan.goals) == 1
        assert goal.description == "Reduce depression"
        assert len(goal.milestones) == 2

    def test_update_goal_progress(self) -> None:
        """Test updating goal progress."""
        plan = TreatmentPlanEntity()
        goal = plan.add_goal("Test goal")
        assert plan.update_goal_progress(goal.goal_id, 50) is True
        assert goal.progress_percentage == 50

    def test_record_session(self) -> None:
        """Test recording session."""
        plan = TreatmentPlanEntity()
        plan.record_session(skills_practiced=["grounding", "breathing"])
        assert plan.total_sessions_completed == 1
        assert plan.phase_sessions_completed == 1
        assert "grounding" in plan.skills_in_progress

    def test_advance_phase(self) -> None:
        """Test phase advancement."""
        plan = TreatmentPlanEntity()
        assert plan.current_phase == TreatmentPhase.FOUNDATION
        new_phase = plan.advance_phase()
        assert new_phase == TreatmentPhase.ACTIVE_TREATMENT
        assert plan.phase_sessions_completed == 0

    def test_skills_transfer_on_advance(self) -> None:
        """Test skills transfer from in-progress to acquired on advance."""
        plan = TreatmentPlanEntity()
        plan.record_session(skills_practiced=["grounding"])
        plan.advance_phase()
        assert "grounding" in plan.skills_acquired
        assert len(plan.skills_in_progress) == 0

    def test_advance_phase_final(self) -> None:
        """Test can't advance past maintenance."""
        plan = TreatmentPlanEntity(current_phase=TreatmentPhase.MAINTENANCE)
        result = plan.advance_phase()
        assert result is None
        assert plan.current_phase == TreatmentPhase.MAINTENANCE

    def test_update_outcome_score(self) -> None:
        """Test updating outcome scores."""
        plan = TreatmentPlanEntity()
        plan.update_outcome_score(phq9=14, gad7=10)
        assert plan.latest_phq9 == 14
        assert plan.latest_gad7 == 10

    def test_update_stepped_care(self) -> None:
        """Test updating stepped care level."""
        plan = TreatmentPlanEntity(stepped_care_level=SteppedCareLevel.MEDIUM_INTENSITY)
        changed = plan.update_stepped_care(SteppedCareLevel.HIGH_INTENSITY)
        assert changed is True
        assert plan.stepped_care_level == SteppedCareLevel.HIGH_INTENSITY

    def test_terminate_plan(self) -> None:
        """Test terminating plan."""
        plan = TreatmentPlanEntity()
        plan.terminate("Treatment completed")
        assert plan.is_active is False
        assert plan.termination_reason == "Treatment completed"

    def test_goals_achievement_rate(self) -> None:
        """Test goals achievement rate calculation."""
        plan = TreatmentPlanEntity()
        plan.add_goal("Goal 1")
        plan.add_goal("Goal 2")
        plan.goals[0].update_progress(100)
        assert plan.goals_achieved_count == 1
        assert plan.goals_achievement_rate == 0.5

    def test_events_recorded(self) -> None:
        """Test domain events are recorded."""
        plan = TreatmentPlanEntity()
        plan.add_goal("Test")
        plan.record_session()
        events = plan.get_events()
        assert len(events) >= 2
        plan.clear_events()
        assert len(plan.get_events()) == 0
