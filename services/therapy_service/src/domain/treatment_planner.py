"""
Solace-AI Therapy Service - Stepped Care Treatment Planning.
Evidence-based treatment planning with phased progression, goal setting, and outcome tracking.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import (
    TherapyModality, SeverityLevel, RiskLevel, SteppedCareLevel,
    TreatmentPhase, GoalStatus, ResponseStatus,
)

logger = structlog.get_logger(__name__)


@dataclass
class TreatmentGoal:
    """A specific treatment goal."""
    goal_id: UUID = field(default_factory=uuid4)
    description: str = ""
    target_date: datetime | None = None
    status: GoalStatus = GoalStatus.NOT_STARTED
    progress_percentage: int = 0
    milestones: list[str] = field(default_factory=list)
    completed_milestones: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PhaseConfig:
    """Configuration for a treatment phase."""
    phase: TreatmentPhase
    min_sessions: int = 2
    max_sessions: int = 8
    required_skills: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)
    advancement_criteria: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreatmentPlan:
    """Comprehensive treatment plan."""
    plan_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    primary_diagnosis: str = ""
    secondary_diagnoses: list[str] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MODERATE
    stepped_care_level: SteppedCareLevel = SteppedCareLevel.MEDIUM_INTENSITY
    primary_modality: TherapyModality = TherapyModality.CBT
    adjunct_modalities: list[TherapyModality] = field(default_factory=list)
    current_phase: TreatmentPhase = TreatmentPhase.FOUNDATION
    phase_sessions_completed: int = 0
    total_sessions_completed: int = 0
    session_frequency_per_week: int = 1
    response_status: ResponseStatus = ResponseStatus.NOT_STARTED
    goals: list[TreatmentGoal] = field(default_factory=list)
    skills_acquired: list[str] = field(default_factory=list)
    skills_in_progress: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    baseline_phq9: int | None = None
    latest_phq9: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    target_completion: datetime | None = None
    review_date: datetime | None = None


class TreatmentPlannerSettings(BaseSettings):
    """Treatment planner configuration."""
    default_sessions_per_phase: int = Field(default=4)
    min_sessions_before_advancement: int = Field(default=2)
    enable_auto_advancement: bool = Field(default=True)
    goal_review_interval_days: int = Field(default=14)
    enable_stepped_care: bool = Field(default=True)
    severity_session_multiplier: dict[str, float] = Field(
        default_factory=lambda: {"minimal": 0.5, "mild": 0.75, "moderate": 1.0, "moderately_severe": 1.25, "severe": 1.5}
    )
    model_config = SettingsConfigDict(env_prefix="TREATMENT_PLANNER_", env_file=".env", extra="ignore")


class TreatmentPlanner:
    """
    Manages stepped care treatment planning.

    Handles treatment plan creation, goal setting, phase advancement,
    and progress monitoring following evidence-based protocols.
    """

    def __init__(self, settings: TreatmentPlannerSettings | None = None) -> None:
        self._settings = settings or TreatmentPlannerSettings()
        self._plans: dict[UUID, TreatmentPlan] = {}
        self._phase_configs = self._initialize_phase_configs()
        logger.info("treatment_planner_initialized", stepped_care=self._settings.enable_stepped_care)

    def _initialize_phase_configs(self) -> dict[TreatmentPhase, PhaseConfig]:
        """Initialize phase configurations aligned with architecture spec."""
        return {
            TreatmentPhase.FOUNDATION: PhaseConfig(
                phase=TreatmentPhase.FOUNDATION, min_sessions=2, max_sessions=4,
                required_skills=["grounding", "safety_planning"],
                focus_areas=["therapeutic_alliance", "assessment", "baseline", "goal_setting", "psychoeducation"],
                advancement_criteria={"alliance_rating": 36, "goals_set": True, "baseline_complete": True},
            ),
            TreatmentPhase.ACTIVE_TREATMENT: PhaseConfig(
                phase=TreatmentPhase.ACTIVE_TREATMENT, min_sessions=6, max_sessions=16,
                required_skills=["cognitive_restructuring", "behavioral_activation", "emotion_regulation"],
                focus_areas=["core_techniques", "skill_practice", "homework_integration", "real_world_application"],
                advancement_criteria={"skills_acquired": 3, "symptom_reduction_percent": 50, "homework_completion": 60},
            ),
            TreatmentPhase.CONSOLIDATION: PhaseConfig(
                phase=TreatmentPhase.CONSOLIDATION, min_sessions=2, max_sessions=4,
                required_skills=["independent_practice", "relapse_prevention", "early_warning_recognition"],
                focus_areas=["skill_generalization", "relapse_prevention", "future_planning", "independence_building"],
                advancement_criteria={"skills_generalized": True, "goals_achieved_percent": 70, "relapse_plan": True},
            ),
            TreatmentPhase.MAINTENANCE: PhaseConfig(
                phase=TreatmentPhase.MAINTENANCE, min_sessions=2, max_sessions=12,
                required_skills=["self_monitoring", "booster_skills"],
                focus_areas=["monthly_checkins", "skill_refreshers", "booster_sessions", "self_management"],
                advancement_criteria={"stable_symptoms": True, "self_management_score": 80},
            ),
        }

    def calculate_stepped_care_level(self, phq9_score: int) -> SteppedCareLevel:
        """
        Calculate stepped care level from PHQ-9 score.

        PHQ-9 Ranges:
        - 0-4: Wellness (Step 0)
        - 5-9: Low Intensity (Step 1)
        - 10-14: Medium Intensity (Step 2)
        - 15-19: High Intensity (Step 3)
        - 20+: Intensive + Referral (Step 4)
        """
        if phq9_score <= 4:
            return SteppedCareLevel.WELLNESS
        elif phq9_score <= 9:
            return SteppedCareLevel.LOW_INTENSITY
        elif phq9_score <= 14:
            return SteppedCareLevel.MEDIUM_INTENSITY
        elif phq9_score <= 19:
            return SteppedCareLevel.HIGH_INTENSITY
        else:
            return SteppedCareLevel.INTENSIVE_REFERRAL

    def get_stepped_care_recommendations(self, level: SteppedCareLevel) -> dict[str, Any]:
        """Get recommendations for a stepped care level."""
        recommendations = {
            SteppedCareLevel.WELLNESS: {
                "description": "Wellness Focus",
                "session_frequency": "monthly",
                "content": ["self_guided_resources", "mood_tracking", "wellness_tips"],
                "homework_intensity": "light",
                "human_involvement": "none",
            },
            SteppedCareLevel.LOW_INTENSITY: {
                "description": "Low Intensity",
                "session_frequency": "bi-weekly",
                "content": ["self_guided_psychoeducation", "basic_coping_skills", "activity_tracking"],
                "homework_intensity": "light",
                "human_involvement": "minimal",
            },
            SteppedCareLevel.MEDIUM_INTENSITY: {
                "description": "Medium Intensity",
                "session_frequency": "1-2x/week",
                "content": ["guided_digital_cbt", "structured_skill_modules", "progress_monitoring"],
                "homework_intensity": "weekly",
                "human_involvement": "periodic_review",
            },
            SteppedCareLevel.HIGH_INTENSITY: {
                "description": "High Intensity",
                "session_frequency": "2-3x/week",
                "content": ["intensive_protocol", "daily_practice", "crisis_plan_active"],
                "homework_intensity": "daily",
                "human_involvement": "coach_checkins",
            },
            SteppedCareLevel.INTENSIVE_REFERRAL: {
                "description": "Intensive Care",
                "session_frequency": "daily_monitoring",
                "content": ["ai_as_adjunct_only", "human_therapist_primary", "safety_planning_priority"],
                "homework_intensity": "as_tolerated",
                "human_involvement": "required",
                "referral_required": True,
            },
        }
        return recommendations.get(level, recommendations[SteppedCareLevel.MEDIUM_INTENSITY])

    def create_plan(
        self,
        user_id: UUID,
        diagnosis: str,
        severity: SeverityLevel,
        modality: TherapyModality,
        phq9_score: int | None = None,
        initial_goals: list[str] | None = None,
        secondary_diagnoses: list[str] | None = None,
        contraindications: list[str] | None = None,
    ) -> TreatmentPlan:
        """
        Create new treatment plan with stepped care routing.

        Args:
            user_id: User identifier
            diagnosis: Primary diagnosis
            severity: Symptom severity
            modality: Primary therapy modality
            phq9_score: Baseline PHQ-9 score for stepped care routing
            initial_goals: Initial treatment goals
            secondary_diagnoses: Secondary diagnoses
            contraindications: Treatment contraindications

        Returns:
            Created treatment plan
        """
        stepped_care_level = self.calculate_stepped_care_level(phq9_score) if phq9_score else self._severity_to_stepped_care(severity)
        initial_phase = self._determine_initial_phase(severity)
        adjunct_modalities = self._recommend_adjunct_modalities(diagnosis, modality, severity)
        target_weeks = self._estimate_treatment_duration(severity)
        session_frequency = self._determine_session_frequency(stepped_care_level)

        plan = TreatmentPlan(
            user_id=user_id,
            primary_diagnosis=diagnosis,
            secondary_diagnoses=secondary_diagnoses or [],
            severity=severity,
            stepped_care_level=stepped_care_level,
            primary_modality=modality,
            adjunct_modalities=adjunct_modalities,
            current_phase=initial_phase,
            session_frequency_per_week=session_frequency,
            baseline_phq9=phq9_score,
            latest_phq9=phq9_score,
            contraindications=contraindications or [],
            target_completion=datetime.now(timezone.utc) + timedelta(weeks=target_weeks),
            review_date=datetime.now(timezone.utc) + timedelta(days=self._settings.goal_review_interval_days),
        )

        if initial_goals:
            for goal_desc in initial_goals:
                plan.goals.append(TreatmentGoal(description=goal_desc))

        self._plans[plan.plan_id] = plan
        logger.info(
            "treatment_plan_created", plan_id=str(plan.plan_id), user_id=str(user_id),
            diagnosis=diagnosis, severity=severity.value, modality=modality.value,
            stepped_care_level=stepped_care_level.value,
        )
        return plan

    def _severity_to_stepped_care(self, severity: SeverityLevel) -> SteppedCareLevel:
        """Map severity level to stepped care level when PHQ-9 unavailable."""
        mapping = {
            SeverityLevel.MINIMAL: SteppedCareLevel.WELLNESS,
            SeverityLevel.MILD: SteppedCareLevel.LOW_INTENSITY,
            SeverityLevel.MODERATE: SteppedCareLevel.MEDIUM_INTENSITY,
            SeverityLevel.MODERATELY_SEVERE: SteppedCareLevel.HIGH_INTENSITY,
            SeverityLevel.SEVERE: SteppedCareLevel.INTENSIVE_REFERRAL,
        }
        return mapping.get(severity, SteppedCareLevel.MEDIUM_INTENSITY)

    def _determine_session_frequency(self, level: SteppedCareLevel) -> int:
        """Determine session frequency per week based on stepped care level."""
        frequency_map = {
            SteppedCareLevel.WELLNESS: 0,  # Monthly
            SteppedCareLevel.LOW_INTENSITY: 1,  # Bi-weekly to weekly
            SteppedCareLevel.MEDIUM_INTENSITY: 2,  # 1-2x per week
            SteppedCareLevel.HIGH_INTENSITY: 3,  # 2-3x per week
            SteppedCareLevel.INTENSIVE_REFERRAL: 5,  # Daily monitoring
        }
        return frequency_map.get(level, 1)

    def _determine_initial_phase(self, severity: SeverityLevel) -> TreatmentPhase:
        """Determine initial treatment phase based on severity."""
        return TreatmentPhase.FOUNDATION

    def _recommend_adjunct_modalities(
        self,
        diagnosis: str,
        primary_modality: TherapyModality,
        severity: SeverityLevel,
    ) -> list[TherapyModality]:
        """Recommend adjunct modalities based on diagnosis."""
        adjuncts = []
        diagnosis_lower = diagnosis.lower()
        if primary_modality != TherapyModality.MINDFULNESS:
            adjuncts.append(TherapyModality.MINDFULNESS)
        if "anxiety" in diagnosis_lower and primary_modality != TherapyModality.ACT:
            adjuncts.append(TherapyModality.ACT)
        if "borderline" in diagnosis_lower or "emotion" in diagnosis_lower:
            if primary_modality != TherapyModality.DBT:
                adjuncts.append(TherapyModality.DBT)
        if "substance" in diagnosis_lower or "addiction" in diagnosis_lower:
            if primary_modality != TherapyModality.MI:
                adjuncts.append(TherapyModality.MI)
        return adjuncts[:2]

    def _estimate_treatment_duration(self, severity: SeverityLevel) -> int:
        """Estimate treatment duration in weeks."""
        base_weeks = 12
        multiplier = self._settings.severity_session_multiplier.get(severity.value, 1.0)
        return int(base_weeks * multiplier)

    def get_plan(self, plan_id: UUID) -> TreatmentPlan | None:
        """Get treatment plan by ID."""
        return self._plans.get(plan_id)

    def get_plans_for_user(self, user_id: UUID) -> list[TreatmentPlan]:
        """Get all treatment plans for a user."""
        return [p for p in self._plans.values() if p.user_id == user_id]

    def add_goal(self, plan_id: UUID, description: str, milestones: list[str] | None = None) -> TreatmentGoal | None:
        """Add a treatment goal."""
        plan = self._plans.get(plan_id)
        if not plan:
            return None
        goal = TreatmentGoal(description=description, milestones=milestones or [])
        plan.goals.append(goal)
        plan.updated_at = datetime.now(timezone.utc)
        logger.debug("goal_added", plan_id=str(plan_id), goal_id=str(goal.goal_id))
        return goal

    def update_goal_progress(self, plan_id: UUID, goal_id: UUID, progress: int, completed_milestone: str | None = None) -> bool:
        """Update goal progress."""
        plan = self._plans.get(plan_id)
        if not plan:
            return False
        for goal in plan.goals:
            if goal.goal_id == goal_id:
                goal.progress_percentage = min(100, max(0, progress))
                goal.updated_at = datetime.now(timezone.utc)
                if completed_milestone and completed_milestone not in goal.completed_milestones:
                    goal.completed_milestones.append(completed_milestone)
                if goal.progress_percentage >= 100:
                    goal.status = GoalStatus.ACHIEVED
                elif goal.progress_percentage > 0:
                    goal.status = GoalStatus.IN_PROGRESS
                plan.updated_at = datetime.now(timezone.utc)
                return True
        return False

    def record_session_completion(self, plan_id: UUID, skills_practiced: list[str] | None = None) -> dict[str, Any]:
        """
        Record session completion and check for phase advancement.

        Args:
            plan_id: Treatment plan ID
            skills_practiced: Skills practiced in session

        Returns:
            Session completion result with advancement status
        """
        plan = self._plans.get(plan_id)
        if not plan:
            return {"success": False, "error": "Plan not found"}
        plan.phase_sessions_completed += 1
        plan.total_sessions_completed += 1
        plan.updated_at = datetime.now(timezone.utc)
        if skills_practiced:
            for skill in skills_practiced:
                if skill not in plan.skills_acquired and skill not in plan.skills_in_progress:
                    plan.skills_in_progress.append(skill)
        advancement_result = self._check_phase_advancement(plan)
        return {
            "success": True,
            "total_sessions": plan.total_sessions_completed,
            "phase_sessions": plan.phase_sessions_completed,
            "current_phase": plan.current_phase.value,
            "advancement": advancement_result,
        }

    def _check_phase_advancement(self, plan: TreatmentPlan) -> dict[str, Any]:
        """Check if plan should advance to next phase."""
        if not self._settings.enable_auto_advancement:
            return {"should_advance": False, "reason": "Auto advancement disabled"}
        config = self._phase_configs.get(plan.current_phase)
        if not config:
            return {"should_advance": False, "reason": "Phase config not found"}
        if plan.phase_sessions_completed < config.min_sessions:
            return {"should_advance": False, "reason": f"Min sessions not met ({plan.phase_sessions_completed}/{config.min_sessions})"}
        criteria_met = self._evaluate_advancement_criteria(plan, config)
        if criteria_met["all_met"]:
            next_phase = self._get_next_phase(plan.current_phase)
            if next_phase and next_phase != plan.current_phase:
                return {
                    "should_advance": True,
                    "next_phase": next_phase.value,
                    "criteria_met": criteria_met["details"],
                }
        return {"should_advance": False, "criteria_met": criteria_met["details"], "reason": "Criteria not fully met"}

    def _evaluate_advancement_criteria(self, plan: TreatmentPlan, config: PhaseConfig) -> dict[str, Any]:
        """Evaluate phase advancement criteria."""
        criteria = config.advancement_criteria
        details = {}
        all_met = True
        if "basic_skills" in criteria:
            required = criteria["basic_skills"]
            acquired = len([s for s in plan.skills_acquired if "grounding" in s.lower() or "safety" in s.lower()])
            details["basic_skills"] = {"required": required, "acquired": acquired, "met": acquired >= required}
            if acquired < required:
                all_met = False
        if "skills_acquired" in criteria:
            required = criteria["skills_acquired"]
            acquired = len(plan.skills_acquired)
            details["skills_acquired"] = {"required": required, "acquired": acquired, "met": acquired >= required}
            if acquired < required:
                all_met = False
        if "goals_achieved_percent" in criteria:
            required = criteria["goals_achieved_percent"]
            achieved = len([g for g in plan.goals if g.status == GoalStatus.ACHIEVED])
            total = len(plan.goals) or 1
            percent = (achieved / total) * 100
            details["goals_achieved"] = {"required_percent": required, "current_percent": percent, "met": percent >= required}
            if percent < required:
                all_met = False
        return {"all_met": all_met, "details": details}

    def _get_next_phase(self, current: TreatmentPhase) -> TreatmentPhase | None:
        """Get next treatment phase following three-phase protocol."""
        order = [TreatmentPhase.FOUNDATION, TreatmentPhase.ACTIVE_TREATMENT,
                 TreatmentPhase.CONSOLIDATION, TreatmentPhase.MAINTENANCE]
        try:
            idx = order.index(current)
            return order[idx + 1] if idx < len(order) - 1 else None
        except ValueError:
            return None

    def advance_phase(self, plan_id: UUID, force: bool = False) -> dict[str, Any]:
        """
        Advance treatment plan to next phase.

        Args:
            plan_id: Treatment plan ID
            force: Force advancement without criteria check

        Returns:
            Advancement result
        """
        plan = self._plans.get(plan_id)
        if not plan:
            return {"success": False, "error": "Plan not found"}
        next_phase = self._get_next_phase(plan.current_phase)
        if not next_phase:
            return {"success": False, "error": "Already at final phase"}
        if not force:
            config = self._phase_configs.get(plan.current_phase)
            if config and plan.phase_sessions_completed < config.min_sessions:
                return {"success": False, "error": f"Min sessions not completed ({plan.phase_sessions_completed}/{config.min_sessions})"}
        previous_phase = plan.current_phase
        plan.current_phase = next_phase
        plan.phase_sessions_completed = 0
        plan.updated_at = datetime.now(timezone.utc)
        for skill in plan.skills_in_progress:
            if skill not in plan.skills_acquired:
                plan.skills_acquired.append(skill)
        plan.skills_in_progress = []
        logger.info("phase_advanced", plan_id=str(plan_id), from_phase=previous_phase.value, to_phase=next_phase.value)
        return {"success": True, "previous_phase": previous_phase.value, "current_phase": next_phase.value}

    def get_phase_recommendations(self, plan_id: UUID) -> dict[str, Any]:
        """Get recommendations for current phase."""
        plan = self._plans.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        config = self._phase_configs.get(plan.current_phase)
        if not config:
            return {"error": "Phase config not found"}
        return {
            "phase": plan.current_phase.value,
            "focus_areas": config.focus_areas,
            "required_skills": config.required_skills,
            "sessions_completed": plan.phase_sessions_completed,
            "sessions_range": {"min": config.min_sessions, "max": config.max_sessions},
            "recommended_techniques": self._get_phase_techniques(plan.current_phase, plan.primary_modality),
        }

    def _get_phase_techniques(self, phase: TreatmentPhase, modality: TherapyModality) -> list[str]:
        """Get recommended techniques for phase and modality."""
        techniques = {
            (TreatmentPhase.FOUNDATION, TherapyModality.CBT): ["Psychoeducation", "Goal Setting", "Mood Monitoring", "Safety Planning"],
            (TreatmentPhase.FOUNDATION, TherapyModality.DBT): ["Psychoeducation", "Mindfulness Basics", "Safety Planning"],
            (TreatmentPhase.ACTIVE_TREATMENT, TherapyModality.CBT): ["Thought Record", "Behavioral Activation", "Cognitive Restructuring", "Exposure"],
            (TreatmentPhase.ACTIVE_TREATMENT, TherapyModality.DBT): ["STOP Skill", "DEAR MAN", "Radical Acceptance", "Emotion Regulation"],
            (TreatmentPhase.ACTIVE_TREATMENT, TherapyModality.ACT): ["Values Clarification", "Cognitive Defusion", "Committed Action"],
            (TreatmentPhase.ACTIVE_TREATMENT, TherapyModality.MI): ["Motivational Interviewing", "Change Talk", "Decisional Balance"],
            (TreatmentPhase.CONSOLIDATION, TherapyModality.CBT): ["Skill Integration", "Relapse Prevention", "Future Planning"],
            (TreatmentPhase.MAINTENANCE, TherapyModality.CBT): ["Booster Review", "Self-Monitoring", "Early Warning Recognition"],
        }
        return techniques.get((phase, modality), ["Mindfulness", "Grounding", "Psychoeducation"])

    def update_outcome_score(self, plan_id: UUID, phq9_score: int) -> dict[str, Any]:
        """
        Update PHQ-9 outcome score and evaluate treatment response.

        Args:
            plan_id: Treatment plan ID
            phq9_score: New PHQ-9 score

        Returns:
            Response evaluation with recommendations
        """
        plan = self._plans.get(plan_id)
        if not plan:
            return {"success": False, "error": "Plan not found"}
        previous_score = plan.latest_phq9
        plan.latest_phq9 = phq9_score
        plan.updated_at = datetime.now(timezone.utc)
        response_status, recommendations = self._evaluate_treatment_response(plan, previous_score, phq9_score)
        plan.response_status = response_status
        new_stepped_care = self.calculate_stepped_care_level(phq9_score)
        step_change = new_stepped_care != plan.stepped_care_level
        if step_change:
            old_level = plan.stepped_care_level
            plan.stepped_care_level = new_stepped_care
            plan.session_frequency_per_week = self._determine_session_frequency(new_stepped_care)
            logger.info("stepped_care_adjusted", plan_id=str(plan_id), old=old_level.value, new=new_stepped_care.value)
        return {
            "success": True,
            "previous_score": previous_score,
            "current_score": phq9_score,
            "response_status": response_status.value,
            "step_change": step_change,
            "recommendations": recommendations,
        }

    def _evaluate_treatment_response(
        self, plan: TreatmentPlan, previous: int | None, current: int
    ) -> tuple[ResponseStatus, list[str]]:
        """Evaluate treatment response based on PHQ-9 change."""
        recommendations = []
        if plan.baseline_phq9 is None or previous is None:
            return ResponseStatus.NOT_STARTED, ["Establish baseline before evaluating response"]
        baseline = plan.baseline_phq9
        reduction_percent = ((baseline - current) / baseline) * 100 if baseline > 0 else 0
        if current < previous:
            if reduction_percent >= 50:
                recommendations = ["Continue current approach", "Begin consolidation phase", "Introduce relapse prevention"]
                return ResponseStatus.RESPONDING, recommendations
            elif reduction_percent >= 25:
                recommendations = ["Augment with adjunct modality", "Increase homework focus", "Extend treatment duration"]
                return ResponseStatus.PARTIAL_RESPONSE, recommendations
        if current > previous:
            if current > baseline:
                recommendations = ["Immediate safety assessment", "Pause standard interventions", "Human clinician consultation"]
                return ResponseStatus.DETERIORATING, recommendations
        if reduction_percent < 25:
            recommendations = ["Reassess diagnosis accuracy", "Consider modality switch", "Explore treatment barriers"]
            return ResponseStatus.NON_RESPONSE, recommendations
        if current <= 4:
            recommendations = ["Transition to maintenance", "Prepare for termination", "Develop relapse prevention plan"]
            return ResponseStatus.REMISSION, recommendations
        return ResponseStatus.RESPONDING, ["Continue monitoring"]

    def delete_plan(self, plan_id: UUID) -> bool:
        """Delete treatment plan."""
        if plan_id in self._plans:
            del self._plans[plan_id]
            logger.info("treatment_plan_deleted", plan_id=str(plan_id))
            return True
        return False
