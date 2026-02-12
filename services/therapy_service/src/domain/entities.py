"""
Solace-AI Therapy Service - Domain Entities.
Aggregate root entities for treatment plans and therapy sessions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar
from uuid import UUID, uuid4
import structlog

logger = structlog.get_logger(__name__)

from ..schemas import (
    SessionPhase, TreatmentPhase, TherapyModality, SeverityLevel,
    RiskLevel, SteppedCareLevel, GoalStatus, ResponseStatus, HomeworkStatus,
)


@dataclass
class TreatmentGoalEntity:
    """Treatment goal entity with lifecycle tracking."""
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
    version: int = 1

    def update_progress(self, progress: int) -> None:
        """Update goal progress and status."""
        self.progress_percentage = max(0, min(100, progress))
        self.updated_at, self.version = datetime.now(timezone.utc), self.version + 1
        self.status = GoalStatus.ACHIEVED if self.progress_percentage >= 100 else (GoalStatus.IN_PROGRESS if self.progress_percentage > 0 else self.status)

    def complete_milestone(self, milestone: str) -> bool:
        """Mark a milestone as completed."""
        if milestone in self.milestones and milestone not in self.completed_milestones:
            self.completed_milestones.append(milestone)
            self.updated_at, self.version = datetime.now(timezone.utc), self.version + 1
            if len(self.completed_milestones) == len(self.milestones):
                self.progress_percentage, self.status = 100, GoalStatus.ACHIEVED
            return True
        return False


@dataclass
class HomeworkEntity:
    """Homework assignment entity."""
    homework_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    technique_id: UUID | None = None
    title: str = ""
    description: str = ""
    status: HomeworkStatus = HomeworkStatus.ASSIGNED
    assigned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: datetime | None = None
    completed_at: datetime | None = None
    completion_notes: str = ""
    rating: int | None = None
    version: int = 1

    def mark_completed(self, notes: str = "", rating: int | None = None) -> None:
        """Mark homework as completed."""
        self.status, self.completed_at = HomeworkStatus.COMPLETED, datetime.now(timezone.utc)
        self.completion_notes, self.rating, self.version = notes, rating, self.version + 1

    def mark_partially_completed(self, notes: str = "") -> None:
        """Mark homework as partially completed."""
        self.status, self.completion_notes, self.version = HomeworkStatus.PARTIALLY_COMPLETED, notes, self.version + 1

    def is_overdue(self) -> bool:
        """Check if homework is overdue."""
        return bool(self.due_date and self.status not in [HomeworkStatus.COMPLETED, HomeworkStatus.SKIPPED] and datetime.now(timezone.utc) > self.due_date)


@dataclass
class InterventionEntity:
    """Intervention delivery entity."""
    intervention_id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    technique_id: UUID = field(default_factory=uuid4)
    technique_name: str = ""
    modality: TherapyModality = TherapyModality.CBT
    phase: SessionPhase = SessionPhase.WORKING
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    messages_exchanged: int = 0
    completed: bool = False
    engagement_score: Decimal = Decimal("0.5")
    skills_practiced: list[str] = field(default_factory=list)
    insights_gained: list[str] = field(default_factory=list)
    version: int = 1

    def complete(self, engagement: Decimal | None = None) -> None:
        """Complete the intervention."""
        self.completed, self.completed_at, self.version = True, datetime.now(timezone.utc), self.version + 1
        if engagement is not None:
            self.engagement_score = max(Decimal("0"), min(Decimal("1"), engagement))

    @property
    def duration_minutes(self) -> int:
        """Calculate intervention duration in minutes."""
        return int(((self.completed_at or datetime.now(timezone.utc)) - self.started_at).total_seconds() / 60)


@dataclass
class TherapySessionEntity:
    """Therapy session aggregate root entity."""
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    treatment_plan_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    current_phase: SessionPhase = SessionPhase.PRE_SESSION
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    mood_rating_start: int | None = None
    mood_rating_end: int | None = None
    agenda_items: list[str] = field(default_factory=list)
    topics_covered: list[str] = field(default_factory=list)
    interventions: list[InterventionEntity] = field(default_factory=list)
    homework_assigned: list[HomeworkEntity] = field(default_factory=list)
    skills_practiced: list[str] = field(default_factory=list)
    insights_gained: list[str] = field(default_factory=list)
    current_risk: RiskLevel = RiskLevel.NONE
    safety_flags: list[str] = field(default_factory=list)
    session_rating: float | None = None
    alliance_score: float | None = None
    summary: str = ""
    next_session_focus: str = ""
    version: int = 1
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    _VALID_TRANSITIONS: ClassVar[dict[SessionPhase, list[SessionPhase]]] = {
        SessionPhase.PRE_SESSION: [SessionPhase.OPENING],
        SessionPhase.OPENING: [SessionPhase.WORKING, SessionPhase.CRISIS],
        SessionPhase.WORKING: [SessionPhase.CLOSING, SessionPhase.CRISIS],
        SessionPhase.CLOSING: [SessionPhase.POST_SESSION],
        SessionPhase.CRISIS: [SessionPhase.CLOSING, SessionPhase.POST_SESSION],
    }

    def transition_phase(self, target_phase: SessionPhase) -> bool:
        """Transition to new session phase."""
        if target_phase in self._VALID_TRANSITIONS.get(self.current_phase, []):
            self._events.append({"type": "phase_transitioned", "from_phase": self.current_phase.value, "to_phase": target_phase.value, "timestamp": datetime.now(timezone.utc).isoformat()})
            self.current_phase, self.version = target_phase, self.version + 1
            return True
        return False

    def add_intervention(self, intervention: InterventionEntity) -> None:
        """Add an intervention to the session."""
        intervention.session_id = self.session_id
        self.interventions.append(intervention)
        self._events.append({"type": "intervention_added", "intervention_id": str(intervention.intervention_id), "technique_name": intervention.technique_name, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.version += 1

    def assign_homework(self, homework: HomeworkEntity) -> None:
        """Assign homework for the session."""
        homework.session_id = self.session_id
        self.homework_assigned.append(homework)
        self._events.append({"type": "homework_assigned", "homework_id": str(homework.homework_id), "title": homework.title, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.version += 1

    def record_skill(self, skill: str) -> None:
        """Record a skill practiced in the session."""
        if skill not in self.skills_practiced:
            self.skills_practiced.append(skill)
            self.version += 1

    def record_insight(self, insight: str) -> None:
        """Record an insight gained in the session."""
        if insight not in self.insights_gained:
            self.insights_gained.append(insight)
            self.version += 1

    def set_risk_level(self, risk: RiskLevel, flag: str | None = None) -> None:
        """Update session risk level."""
        previous_risk = self.current_risk
        self.current_risk = risk
        if flag and flag not in self.safety_flags:
            self.safety_flags.append(flag)
        if risk != previous_risk:
            self._events.append({"type": "risk_level_changed", "previous_risk": previous_risk.value, "current_risk": risk.value, "flag": flag, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.version += 1

    def end_session(self, summary: str = "", next_focus: str = "") -> None:
        """End the therapy session."""
        self.ended_at, self.summary, self.next_session_focus = datetime.now(timezone.utc), summary, next_focus
        if self.current_phase != SessionPhase.POST_SESSION:
            self.current_phase = SessionPhase.POST_SESSION
        self._events.append({"type": "session_ended", "duration_minutes": self.duration_minutes, "interventions_count": len(self.interventions), "timestamp": datetime.now(timezone.utc).isoformat()})
        self.version += 1

    @property
    def duration_minutes(self) -> int:
        return int(((self.ended_at or datetime.now(timezone.utc)) - self.started_at).total_seconds() / 60)

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def get_events(self) -> list[dict[str, Any]]:
        return self._events.copy()

    def clear_events(self) -> None:
        self._events.clear()


@dataclass
class TreatmentPlanEntity:
    """Treatment plan aggregate root entity."""
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
    goals: list[TreatmentGoalEntity] = field(default_factory=list)
    skills_acquired: list[str] = field(default_factory=list)
    skills_in_progress: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    baseline_phq9: int | None = None
    latest_phq9: int | None = None
    baseline_gad7: int | None = None
    latest_gad7: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    target_completion: datetime | None = None
    review_date: datetime | None = None
    is_active: bool = True
    termination_reason: str | None = None
    version: int = 1
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    _PHASE_ORDER: ClassVar[list[TreatmentPhase]] = [TreatmentPhase.FOUNDATION, TreatmentPhase.ACTIVE_TREATMENT, TreatmentPhase.CONSOLIDATION, TreatmentPhase.MAINTENANCE]

    def add_goal(self, description: str, milestones: list[str] | None = None) -> TreatmentGoalEntity:
        """Add a treatment goal."""
        goal = TreatmentGoalEntity(description=description, milestones=milestones or [])
        self.goals.append(goal)
        self.updated_at = datetime.now(timezone.utc)
        self._events.append({"type": "goal_added", "goal_id": str(goal.goal_id), "description": description, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.version += 1
        return goal

    def update_goal_progress(self, goal_id: UUID, progress: int) -> bool:
        """Update progress for a specific goal."""
        for goal in self.goals:
            if goal.goal_id == goal_id:
                goal.update_progress(progress)
                self.updated_at, self.version = datetime.now(timezone.utc), self.version + 1
                return True
        return False

    def record_session(self, skills_practiced: list[str] | None = None) -> None:
        """Record session completion."""
        self.phase_sessions_completed += 1
        self.total_sessions_completed += 1
        self.updated_at = datetime.now(timezone.utc)
        if skills_practiced:
            for skill in skills_practiced:
                if skill not in self.skills_acquired and skill not in self.skills_in_progress:
                    self.skills_in_progress.append(skill)
        self._events.append({"type": "session_recorded", "session_number": self.total_sessions_completed, "phase": self.current_phase.value, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.version += 1

    def advance_phase(self) -> TreatmentPhase | None:
        """Advance to next treatment phase."""
        try:
            current_idx = self._PHASE_ORDER.index(self.current_phase)
            if current_idx < len(self._PHASE_ORDER) - 1:
                previous_phase = self.current_phase
                self.current_phase = self._PHASE_ORDER[current_idx + 1]
                self.phase_sessions_completed = 0
                for skill in self.skills_in_progress:
                    if skill not in self.skills_acquired:
                        self.skills_acquired.append(skill)
                self.skills_in_progress = []
                self.updated_at = datetime.now(timezone.utc)
                self._events.append({"type": "phase_advanced", "from_phase": previous_phase.value, "to_phase": self.current_phase.value, "timestamp": datetime.now(timezone.utc).isoformat()})
                self.version += 1
                return self.current_phase
        except ValueError:
            logger.warning(
                "phase_advance_failed",
                current_phase=self.current_phase.value,
                reason="current phase not found in phase order",
            )
        return None

    def update_outcome_score(self, phq9: int | None = None, gad7: int | None = None) -> None:
        """Update outcome measurement scores."""
        if phq9 is not None:
            self.latest_phq9 = phq9
        if gad7 is not None:
            self.latest_gad7 = gad7
        self.updated_at = datetime.now(timezone.utc)
        self._events.append({"type": "outcome_updated", "phq9": phq9, "gad7": gad7, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.version += 1

    def update_stepped_care(self, new_level: SteppedCareLevel) -> bool:
        """Update stepped care level."""
        if new_level != self.stepped_care_level:
            previous = self.stepped_care_level
            self.stepped_care_level, self.updated_at = new_level, datetime.now(timezone.utc)
            self._events.append({"type": "stepped_care_changed", "from_level": previous.value, "to_level": new_level.value, "timestamp": datetime.now(timezone.utc).isoformat()})
            self.version += 1
            return True
        return False

    def terminate(self, reason: str) -> None:
        """Terminate the treatment plan."""
        self.is_active, self.termination_reason = False, reason
        self.updated_at = datetime.now(timezone.utc)
        self._events.append({"type": "plan_terminated", "reason": reason, "total_sessions": self.total_sessions_completed, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.version += 1

    def get_goal(self, goal_id: UUID) -> TreatmentGoalEntity | None:
        return next((g for g in self.goals if g.goal_id == goal_id), None)

    @property
    def goals_achieved_count(self) -> int:
        return len([g for g in self.goals if g.status == GoalStatus.ACHIEVED])

    @property
    def goals_achievement_rate(self) -> float:
        return self.goals_achieved_count / len(self.goals) if self.goals else 0.0

    def get_events(self) -> list[dict[str, Any]]:
        return self._events.copy()

    def clear_events(self) -> None:
        self._events.clear()
