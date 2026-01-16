"""
Unit tests for Treatment Planner.
Tests stepped care treatment planning, goal management, and phase progression.
"""
from __future__ import annotations
import pytest
from uuid import uuid4

from services.therapy_service.src.domain.treatment_planner import (
    TreatmentPlanner,
    TreatmentPlannerSettings,
    TreatmentPlan,
    TreatmentGoal,
)
from services.therapy_service.src.schemas import (
    TherapyModality, SeverityLevel, TreatmentPhase, GoalStatus,
    SteppedCareLevel, ResponseStatus,
)


class TestTreatmentPlannerSettings:
    """Tests for TreatmentPlannerSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings initialization."""
        settings = TreatmentPlannerSettings()
        assert settings.default_sessions_per_phase == 4
        assert settings.min_sessions_before_advancement == 2
        assert settings.enable_auto_advancement is True
        assert settings.goal_review_interval_days == 14
        assert settings.enable_stepped_care is True

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = TreatmentPlannerSettings(
            default_sessions_per_phase=6,
            enable_auto_advancement=False,
        )
        assert settings.default_sessions_per_phase == 6
        assert settings.enable_auto_advancement is False


class TestTreatmentPlanner:
    """Tests for TreatmentPlanner functionality."""

    def test_planner_initialization(self) -> None:
        """Test planner initializes correctly."""
        planner = TreatmentPlanner()
        assert len(planner._phase_configs) == 4  # FOUNDATION, ACTIVE_TREATMENT, CONSOLIDATION, MAINTENANCE
        assert TreatmentPhase.FOUNDATION in planner._phase_configs

    def test_create_plan_basic(self) -> None:
        """Test basic treatment plan creation."""
        planner = TreatmentPlanner()
        user_id = uuid4()
        plan = planner.create_plan(
            user_id=user_id,
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        assert plan.user_id == user_id
        assert plan.primary_diagnosis == "Depression"
        assert plan.severity == SeverityLevel.MODERATE
        assert plan.primary_modality == TherapyModality.CBT
        assert plan.current_phase == TreatmentPhase.FOUNDATION

    def test_create_plan_with_phq9_sets_stepped_care(self) -> None:
        """Test PHQ-9 score determines stepped care level."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            phq9_score=14,
        )
        assert plan.stepped_care_level == SteppedCareLevel.MEDIUM_INTENSITY
        assert plan.baseline_phq9 == 14

    def test_calculate_stepped_care_level(self) -> None:
        """Test stepped care level calculation from PHQ-9."""
        planner = TreatmentPlanner()
        assert planner.calculate_stepped_care_level(3) == SteppedCareLevel.WELLNESS
        assert planner.calculate_stepped_care_level(7) == SteppedCareLevel.LOW_INTENSITY
        assert planner.calculate_stepped_care_level(12) == SteppedCareLevel.MEDIUM_INTENSITY
        assert planner.calculate_stepped_care_level(17) == SteppedCareLevel.HIGH_INTENSITY
        assert planner.calculate_stepped_care_level(22) == SteppedCareLevel.INTENSIVE_REFERRAL

    def test_create_plan_with_goals(self) -> None:
        """Test plan creation with initial goals."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Anxiety",
            severity=SeverityLevel.MILD,
            modality=TherapyModality.CBT,
            initial_goals=["Reduce anxiety symptoms", "Improve sleep"],
        )
        assert len(plan.goals) == 2
        assert plan.goals[0].description == "Reduce anxiety symptoms"
        assert plan.goals[0].status == GoalStatus.NOT_STARTED

    def test_create_plan_recommends_adjuncts(self) -> None:
        """Test adjunct modality recommendations."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Generalized Anxiety",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        assert TherapyModality.MINDFULNESS in plan.adjunct_modalities

    def test_get_plan(self) -> None:
        """Test retrieving a plan by ID."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        retrieved = planner.get_plan(plan.plan_id)
        assert retrieved is not None
        assert retrieved.plan_id == plan.plan_id

    def test_get_nonexistent_plan(self) -> None:
        """Test retrieving non-existent plan returns None."""
        planner = TreatmentPlanner()
        result = planner.get_plan(uuid4())
        assert result is None

    def test_add_goal(self) -> None:
        """Test adding a goal to a plan."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        goal = planner.add_goal(
            plan.plan_id,
            "Learn cognitive restructuring",
            milestones=["Complete 3 thought records"],
        )
        assert goal is not None
        assert goal.description == "Learn cognitive restructuring"
        assert len(goal.milestones) == 1

    def test_update_goal_progress(self) -> None:
        """Test updating goal progress."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            initial_goals=["Test goal"],
        )
        goal_id = plan.goals[0].goal_id
        result = planner.update_goal_progress(plan.plan_id, goal_id, progress=50)
        assert result is True
        assert plan.goals[0].progress_percentage == 50
        assert plan.goals[0].status == GoalStatus.IN_PROGRESS

    def test_goal_achieved_on_100_percent(self) -> None:
        """Test goal status changes to achieved at 100%."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            initial_goals=["Test goal"],
        )
        goal_id = plan.goals[0].goal_id
        planner.update_goal_progress(plan.plan_id, goal_id, progress=100)
        assert plan.goals[0].status == GoalStatus.ACHIEVED

    def test_record_session_completion(self) -> None:
        """Test recording session completion."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        result = planner.record_session_completion(
            plan.plan_id,
            skills_practiced=["grounding", "thought_record"],
        )
        assert result["success"] is True
        assert result["total_sessions"] == 1
        assert result["phase_sessions"] == 1
        assert "grounding" in plan.skills_in_progress

    def test_advance_phase(self) -> None:
        """Test manual phase advancement."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        # Complete minimum sessions
        planner.record_session_completion(plan.plan_id)
        planner.record_session_completion(plan.plan_id)
        result = planner.advance_phase(plan.plan_id, force=True)
        assert result["success"] is True
        assert plan.current_phase == TreatmentPhase.ACTIVE_TREATMENT

    def test_advance_phase_respects_min_sessions(self) -> None:
        """Test phase advancement respects minimum sessions."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        result = planner.advance_phase(plan.plan_id, force=False)
        assert result["success"] is False
        assert "Min sessions" in result["error"]

    def test_get_phase_recommendations(self) -> None:
        """Test getting phase recommendations."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        recs = planner.get_phase_recommendations(plan.plan_id)
        assert "focus_areas" in recs
        assert "required_skills" in recs
        assert "recommended_techniques" in recs

    def test_delete_plan(self) -> None:
        """Test deleting a plan."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        result = planner.delete_plan(plan.plan_id)
        assert result is True
        assert planner.get_plan(plan.plan_id) is None

    def test_skills_transfer_on_phase_advance(self) -> None:
        """Test skills move from in-progress to acquired on phase advance."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
        )
        planner.record_session_completion(plan.plan_id, skills_practiced=["grounding"])
        planner.record_session_completion(plan.plan_id, skills_practiced=["grounding"])
        planner.advance_phase(plan.plan_id, force=True)
        assert "grounding" in plan.skills_acquired
        assert len(plan.skills_in_progress) == 0


class TestSteppedCareAndOutcomes:
    """Tests for stepped care level management and outcome tracking."""

    def test_update_outcome_score_improves_response(self) -> None:
        """Test outcome score update evaluates treatment response."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            phq9_score=16,
        )
        result = planner.update_outcome_score(plan.plan_id, phq9_score=8)
        assert result["success"] is True
        assert result["response_status"] == ResponseStatus.RESPONDING.value

    def test_update_outcome_triggers_step_down(self) -> None:
        """Test improving score triggers stepped care level reduction."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATELY_SEVERE,
            modality=TherapyModality.CBT,
            phq9_score=18,
        )
        assert plan.stepped_care_level == SteppedCareLevel.HIGH_INTENSITY
        result = planner.update_outcome_score(plan.plan_id, phq9_score=8)
        assert result["step_change"] is True
        assert plan.stepped_care_level == SteppedCareLevel.LOW_INTENSITY

    def test_deterioration_detected(self) -> None:
        """Test deterioration is detected when scores worsen."""
        planner = TreatmentPlanner()
        plan = planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            phq9_score=12,
        )
        result = planner.update_outcome_score(plan.plan_id, phq9_score=18)
        assert result["response_status"] == ResponseStatus.DETERIORATING.value
        assert any("safety" in r.lower() for r in result["recommendations"])

    def test_get_stepped_care_recommendations(self) -> None:
        """Test getting recommendations for stepped care level."""
        planner = TreatmentPlanner()
        recs = planner.get_stepped_care_recommendations(SteppedCareLevel.HIGH_INTENSITY)
        assert recs["session_frequency"] == "2-3x/week"
        assert recs["homework_intensity"] == "daily"

    def test_intensive_referral_flag(self) -> None:
        """Test intensive referral is flagged for severe cases."""
        planner = TreatmentPlanner()
        recs = planner.get_stepped_care_recommendations(SteppedCareLevel.INTENSIVE_REFERRAL)
        assert recs.get("referral_required") is True
        assert recs["human_involvement"] == "required"
