"""
Solace-AI Therapy Service - Homework Assignment and Tracking.
Manages therapeutic homework creation, assignment, completion tracking, and effectiveness analysis.
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

from ..schemas import TherapyModality, HomeworkStatus, TechniqueCategory

logger = structlog.get_logger(__name__)


class HomeworkType(str, Enum):
    """Types of therapeutic homework."""
    BEHAVIORAL = "behavioral"
    COGNITIVE = "cognitive"
    MINDFULNESS = "mindfulness"
    JOURNALING = "journaling"
    SKILL_PRACTICE = "skill_practice"
    EXPOSURE = "exposure"
    MONITORING = "monitoring"
    VALUES = "values"
    RELAXATION = "relaxation"
    INTERPERSONAL = "interpersonal"


class CompletionStatus(str, Enum):
    """Homework completion status (aligned with HomeworkStatus)."""
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    NOT_COMPLETED = "not_completed"
    SKIPPED = "skipped"


class Difficulty(str, Enum):
    """Homework difficulty level."""
    EASY = "easy"
    MODERATE = "moderate"
    CHALLENGING = "challenging"


@dataclass
class HomeworkStep:
    """A single step in a homework assignment."""
    step_id: UUID = field(default_factory=uuid4)
    instruction: str = ""
    order: int = 0
    completed: bool = False
    notes: str = ""


@dataclass
class HomeworkAssignment:
    """A complete homework assignment."""
    homework_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    technique_id: UUID | None = None
    title: str = ""
    description: str = ""
    homework_type: HomeworkType = HomeworkType.SKILL_PRACTICE
    modality: TherapyModality = TherapyModality.CBT
    difficulty: Difficulty = Difficulty.MODERATE
    estimated_minutes: int = 15
    steps: list[HomeworkStep] = field(default_factory=list)
    status: CompletionStatus = CompletionStatus.ASSIGNED
    assigned_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: datetime | None = None
    started_date: datetime | None = None
    completed_date: datetime | None = None
    completion_percentage: int = 0
    user_rating: int | None = None
    user_feedback: str = ""
    therapist_notes: str = ""
    effectiveness_score: float | None = None


@dataclass
class HomeworkTemplate:
    """Reusable homework template."""
    template_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    homework_type: HomeworkType = HomeworkType.SKILL_PRACTICE
    modality: TherapyModality = TherapyModality.CBT
    difficulty: Difficulty = Difficulty.MODERATE
    estimated_minutes: int = 15
    default_steps: list[str] = field(default_factory=list)
    instructions: str = ""
    target_skills: list[str] = field(default_factory=list)


class HomeworkManagerSettings(BaseSettings):
    """Homework manager configuration."""
    default_due_days: int = Field(default=7)
    reminder_threshold_hours: int = Field(default=24)
    enable_effectiveness_tracking: bool = Field(default=True)
    min_completion_for_effectiveness: float = Field(default=0.5)
    enable_adaptive_difficulty: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="HOMEWORK_", env_file=".env", extra="ignore")


class HomeworkManager:
    """
    Manages homework assignments for therapy sessions.

    Handles assignment creation, tracking, completion monitoring,
    and effectiveness analysis for between-session practice.
    """

    def __init__(self, settings: HomeworkManagerSettings | None = None) -> None:
        self._settings = settings or HomeworkManagerSettings()
        self._assignments: dict[UUID, HomeworkAssignment] = {}
        self._user_assignments: dict[UUID, list[UUID]] = {}
        self._templates = self._initialize_templates()
        self._stats = {"assigned": 0, "completed": 0, "in_progress": 0, "effectiveness_tracked": 0}
        logger.info("homework_manager_initialized", templates=len(self._templates))

    def _initialize_templates(self) -> dict[str, HomeworkTemplate]:
        """Initialize homework templates for common techniques."""
        templates = {
            "thought_record": HomeworkTemplate(
                name="Thought Record", description="Track and challenge automatic negative thoughts",
                homework_type=HomeworkType.COGNITIVE, modality=TherapyModality.CBT, difficulty=Difficulty.MODERATE,
                estimated_minutes=20,
                default_steps=["Identify triggering situation", "Record automatic thought", "Rate emotion intensity 0-100",
                               "Identify cognitive distortion", "Generate alternative thought", "Re-rate emotion"],
                target_skills=["cognitive_restructuring", "thought_challenging"],
            ),
            "behavioral_activation": HomeworkTemplate(
                name="Activity Scheduling", description="Schedule and complete mood-boosting activities",
                homework_type=HomeworkType.BEHAVIORAL, modality=TherapyModality.CBT, difficulty=Difficulty.EASY,
                estimated_minutes=15,
                default_steps=["Choose 3 pleasant activities", "Schedule activities for specific times",
                               "Complete activities as planned", "Rate mood before and after each"],
                target_skills=["behavioral_activation", "activity_scheduling"],
            ),
            "mindfulness_practice": HomeworkTemplate(
                name="Daily Mindfulness", description="Practice mindful awareness exercises",
                homework_type=HomeworkType.MINDFULNESS, modality=TherapyModality.MINDFULNESS, difficulty=Difficulty.EASY,
                estimated_minutes=10,
                default_steps=["Find quiet space", "Set timer for 5-10 minutes", "Focus on breath",
                               "Note when mind wanders", "Gently return attention to breath"],
                target_skills=["mindfulness", "present_moment_awareness"],
            ),
            "emotion_diary": HomeworkTemplate(
                name="Emotion Diary", description="Track emotions throughout the day",
                homework_type=HomeworkType.MONITORING, modality=TherapyModality.DBT, difficulty=Difficulty.EASY,
                estimated_minutes=10,
                default_steps=["Note time of emotion", "Identify emotion name", "Rate intensity 0-10",
                               "Describe triggering event", "Record urges and actions taken"],
                target_skills=["emotion_identification", "self_monitoring"],
            ),
            "values_clarification": HomeworkTemplate(
                name="Values Exploration", description="Reflect on personal values and meaning",
                homework_type=HomeworkType.VALUES, modality=TherapyModality.ACT, difficulty=Difficulty.MODERATE,
                estimated_minutes=25,
                default_steps=["List top 5 life domains", "Rate importance of each 1-10",
                               "Identify core value in each domain", "Note one action aligned with each value"],
                target_skills=["values_clarification", "meaning_making"],
            ),
            "exposure_hierarchy": HomeworkTemplate(
                name="Exposure Practice", description="Gradual exposure to anxiety-provoking situations",
                homework_type=HomeworkType.EXPOSURE, modality=TherapyModality.CBT, difficulty=Difficulty.CHALLENGING,
                estimated_minutes=30,
                default_steps=["Review exposure hierarchy", "Select appropriate level item",
                               "Rate anxiety before (0-100)", "Complete exposure task",
                               "Rate anxiety after", "Record observations"],
                target_skills=["exposure", "anxiety_tolerance"],
            ),
            "stop_skill": HomeworkTemplate(
                name="STOP Skill Practice", description="Practice distress tolerance with STOP",
                homework_type=HomeworkType.SKILL_PRACTICE, modality=TherapyModality.DBT, difficulty=Difficulty.EASY,
                estimated_minutes=5,
                default_steps=["Stop when distress noticed", "Take a step back", "Observe situation and reactions",
                               "Proceed mindfully"],
                target_skills=["distress_tolerance", "emotional_regulation"],
            ),
        }
        return {k: v for k, v in templates.items()}

    def assign_homework(
        self,
        user_id: UUID,
        template_name: str,
        session_id: UUID | None = None,
        technique_id: UUID | None = None,
        custom_instructions: str | None = None,
        due_days: int | None = None,
    ) -> HomeworkAssignment | None:
        """
        Assign homework from template.

        Args:
            user_id: User identifier
            template_name: Name of template to use
            session_id: Associated session ID
            technique_id: Associated technique ID
            custom_instructions: Custom instructions to add
            due_days: Days until due (overrides default)

        Returns:
            Created homework assignment
        """
        template = self._templates.get(template_name)
        if not template:
            logger.warning("template_not_found", template_name=template_name)
            return None
        due_date = datetime.now(timezone.utc) + timedelta(days=due_days or self._settings.default_due_days)
        steps = [HomeworkStep(instruction=step, order=i) for i, step in enumerate(template.default_steps)]
        description = template.description
        if custom_instructions:
            description = f"{description}\n\nAdditional instructions: {custom_instructions}"
        assignment = HomeworkAssignment(
            user_id=user_id, session_id=session_id, technique_id=technique_id,
            title=template.name, description=description, homework_type=template.homework_type,
            modality=template.modality, difficulty=template.difficulty,
            estimated_minutes=template.estimated_minutes, steps=steps, due_date=due_date,
        )
        self._assignments[assignment.homework_id] = assignment
        if user_id not in self._user_assignments:
            self._user_assignments[user_id] = []
        self._user_assignments[user_id].append(assignment.homework_id)
        self._stats["assigned"] += 1
        logger.info("homework_assigned", homework_id=str(assignment.homework_id),
                    user_id=str(user_id), template=template_name)
        return assignment

    def create_custom_homework(
        self,
        user_id: UUID,
        title: str,
        description: str,
        homework_type: HomeworkType,
        steps: list[str],
        modality: TherapyModality = TherapyModality.CBT,
        difficulty: Difficulty = Difficulty.MODERATE,
        estimated_minutes: int = 15,
        due_days: int | None = None,
    ) -> HomeworkAssignment:
        """Create custom homework assignment."""
        due_date = datetime.now(timezone.utc) + timedelta(days=due_days or self._settings.default_due_days)
        homework_steps = [HomeworkStep(instruction=step, order=i) for i, step in enumerate(steps)]
        assignment = HomeworkAssignment(
            user_id=user_id, title=title, description=description, homework_type=homework_type,
            modality=modality, difficulty=difficulty, estimated_minutes=estimated_minutes,
            steps=homework_steps, due_date=due_date,
        )
        self._assignments[assignment.homework_id] = assignment
        if user_id not in self._user_assignments:
            self._user_assignments[user_id] = []
        self._user_assignments[user_id].append(assignment.homework_id)
        self._stats["assigned"] += 1
        logger.info("custom_homework_created", homework_id=str(assignment.homework_id), title=title)
        return assignment

    def get_assignment(self, homework_id: UUID) -> HomeworkAssignment | None:
        """Get homework assignment by ID."""
        return self._assignments.get(homework_id)

    def get_user_assignments(
        self,
        user_id: UUID,
        status: CompletionStatus | None = None,
        include_past_due: bool = True,
    ) -> list[HomeworkAssignment]:
        """Get homework assignments for a user."""
        assignment_ids = self._user_assignments.get(user_id, [])
        assignments = [self._assignments[aid] for aid in assignment_ids if aid in self._assignments]
        if status:
            assignments = [a for a in assignments if a.status == status]
        if not include_past_due:
            now = datetime.now(timezone.utc)
            assignments = [a for a in assignments if not a.due_date or a.due_date > now]
        return sorted(assignments, key=lambda x: x.assigned_date, reverse=True)

    def start_homework(self, homework_id: UUID) -> bool:
        """Mark homework as started."""
        assignment = self._assignments.get(homework_id)
        if not assignment:
            return False
        if assignment.status == CompletionStatus.ASSIGNED:
            assignment.status = CompletionStatus.IN_PROGRESS
            assignment.started_date = datetime.now(timezone.utc)
            self._stats["in_progress"] += 1
            logger.debug("homework_started", homework_id=str(homework_id))
        return True

    def complete_step(self, homework_id: UUID, step_index: int, notes: str = "") -> bool:
        """Mark a homework step as completed."""
        assignment = self._assignments.get(homework_id)
        if not assignment or step_index >= len(assignment.steps):
            return False
        step = assignment.steps[step_index]
        step.completed = True
        step.notes = notes
        completed_steps = sum(1 for s in assignment.steps if s.completed)
        total_steps = len(assignment.steps)
        assignment.completion_percentage = int((completed_steps / total_steps) * 100) if total_steps > 0 else 0
        if assignment.status == CompletionStatus.ASSIGNED:
            assignment.status = CompletionStatus.IN_PROGRESS
            assignment.started_date = datetime.now(timezone.utc)
        return True

    def complete_homework(
        self,
        homework_id: UUID,
        user_rating: int | None = None,
        user_feedback: str = "",
        partial: bool = False,
    ) -> dict[str, Any]:
        """
        Mark homework as completed.

        Args:
            homework_id: Homework assignment ID
            user_rating: User's rating of helpfulness (1-5)
            user_feedback: User's feedback text
            partial: Whether this is partial completion

        Returns:
            Completion result with effectiveness tracking
        """
        assignment = self._assignments.get(homework_id)
        if not assignment:
            return {"success": False, "error": "Assignment not found"}
        completed_steps = sum(1 for s in assignment.steps if s.completed)
        total_steps = len(assignment.steps)
        completion_ratio = completed_steps / total_steps if total_steps > 0 else 1.0
        if partial or completion_ratio < 1.0:
            assignment.status = CompletionStatus.PARTIALLY_COMPLETED
        else:
            assignment.status = CompletionStatus.COMPLETED
            self._stats["completed"] += 1
        assignment.completed_date = datetime.now(timezone.utc)
        assignment.completion_percentage = int(completion_ratio * 100)
        if user_rating:
            assignment.user_rating = max(1, min(5, user_rating))
        assignment.user_feedback = user_feedback
        if self._settings.enable_effectiveness_tracking and completion_ratio >= self._settings.min_completion_for_effectiveness:
            assignment.effectiveness_score = self._calculate_effectiveness(assignment)
            self._stats["effectiveness_tracked"] += 1
        logger.info("homework_completed", homework_id=str(homework_id),
                    status=assignment.status.value, completion=assignment.completion_percentage)
        return {
            "success": True, "status": assignment.status.value, "completion_percentage": assignment.completion_percentage,
            "effectiveness_score": assignment.effectiveness_score,
        }

    def _calculate_effectiveness(self, assignment: HomeworkAssignment) -> float:
        """Calculate homework effectiveness score."""
        score = 0.0
        if assignment.completion_percentage >= 100:
            score += 0.4
        elif assignment.completion_percentage >= 75:
            score += 0.3
        elif assignment.completion_percentage >= 50:
            score += 0.2
        if assignment.user_rating:
            score += (assignment.user_rating / 5) * 0.3
        if assignment.due_date and assignment.completed_date:
            if assignment.completed_date <= assignment.due_date:
                score += 0.2
            else:
                days_late = (assignment.completed_date - assignment.due_date).days
                score += max(0, 0.2 - (days_late * 0.05))
        if assignment.user_feedback:
            score += 0.1
        return min(1.0, score)

    def get_pending_reminders(self, user_id: UUID) -> list[HomeworkAssignment]:
        """Get homework assignments needing reminders."""
        now = datetime.now(timezone.utc)
        threshold = timedelta(hours=self._settings.reminder_threshold_hours)
        assignments = self.get_user_assignments(user_id, status=CompletionStatus.ASSIGNED)
        assignments.extend(self.get_user_assignments(user_id, status=CompletionStatus.IN_PROGRESS))
        return [a for a in assignments if a.due_date and (a.due_date - now) <= threshold and a.due_date > now]

    def get_overdue(self, user_id: UUID) -> list[HomeworkAssignment]:
        """Get overdue homework assignments."""
        now = datetime.now(timezone.utc)
        assignments = self.get_user_assignments(user_id)
        return [a for a in assignments if a.due_date and a.due_date < now
                and a.status in [CompletionStatus.ASSIGNED, CompletionStatus.IN_PROGRESS]]

    def get_completion_stats(self, user_id: UUID) -> dict[str, Any]:
        """Get homework completion statistics for a user."""
        assignments = self.get_user_assignments(user_id)
        if not assignments:
            return {"total": 0, "completed": 0, "completion_rate": 0.0, "avg_effectiveness": None}
        completed = [a for a in assignments if a.status == CompletionStatus.COMPLETED]
        partial = [a for a in assignments if a.status == CompletionStatus.PARTIALLY_COMPLETED]
        effectiveness_scores = [a.effectiveness_score for a in assignments if a.effectiveness_score is not None]
        return {
            "total": len(assignments), "completed": len(completed), "partial": len(partial),
            "completion_rate": len(completed) / len(assignments) if assignments else 0.0,
            "avg_effectiveness": sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else None,
            "by_type": self._group_by_type(assignments),
        }

    def _group_by_type(self, assignments: list[HomeworkAssignment]) -> dict[str, dict[str, int]]:
        """Group assignments by type for statistics."""
        by_type: dict[str, dict[str, int]] = {}
        for a in assignments:
            hw_type = a.homework_type.value
            if hw_type not in by_type:
                by_type[hw_type] = {"total": 0, "completed": 0}
            by_type[hw_type]["total"] += 1
            if a.status == CompletionStatus.COMPLETED:
                by_type[hw_type]["completed"] += 1
        return by_type

    def recommend_difficulty(self, user_id: UUID) -> Difficulty:
        """Recommend difficulty level based on completion history."""
        if not self._settings.enable_adaptive_difficulty:
            return Difficulty.MODERATE
        stats = self.get_completion_stats(user_id)
        completion_rate = stats.get("completion_rate", 0.5)
        if completion_rate >= 0.8:
            return Difficulty.CHALLENGING
        elif completion_rate <= 0.4:
            return Difficulty.EASY
        return Difficulty.MODERATE

    def delete_assignment(self, homework_id: UUID) -> bool:
        """Delete homework assignment."""
        assignment = self._assignments.get(homework_id)
        if not assignment:
            return False
        if assignment.user_id in self._user_assignments:
            self._user_assignments[assignment.user_id] = [
                aid for aid in self._user_assignments[assignment.user_id] if aid != homework_id
            ]
        del self._assignments[homework_id]
        return True

    def get_templates(self) -> list[HomeworkTemplate]:
        """Get all available homework templates."""
        return list(self._templates.values())

    def get_template(self, name: str) -> HomeworkTemplate | None:
        """Get template by name."""
        return self._templates.get(name)
