"""
Unit tests for Homework Manager.
Tests homework assignment, tracking, completion, and effectiveness analysis.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from services.therapy_service.src.domain.homework import (
    HomeworkManager,
    HomeworkManagerSettings,
    HomeworkAssignment,
    HomeworkTemplate,
    HomeworkType,
    CompletionStatus,
    Difficulty,
)


class TestHomeworkManagerSettings:
    """Tests for HomeworkManagerSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings initialization."""
        settings = HomeworkManagerSettings()
        assert settings.default_due_days == 7
        assert settings.reminder_threshold_hours == 24
        assert settings.enable_effectiveness_tracking is True
        assert settings.enable_adaptive_difficulty is True

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = HomeworkManagerSettings(
            default_due_days=14,
            enable_effectiveness_tracking=False,
        )
        assert settings.default_due_days == 14
        assert settings.enable_effectiveness_tracking is False


class TestHomeworkManager:
    """Tests for HomeworkManager functionality."""

    def test_manager_initialization(self) -> None:
        """Test manager initializes with templates."""
        manager = HomeworkManager()
        templates = manager.get_templates()
        assert len(templates) > 0
        assert any(t.name == "Thought Record" for t in templates)

    def test_assign_homework_from_template(self) -> None:
        """Test assigning homework from template."""
        manager = HomeworkManager()
        user_id = uuid4()
        assignment = manager.assign_homework(
            user_id=user_id,
            template_name="thought_record",
        )
        assert assignment is not None
        assert assignment.user_id == user_id
        assert assignment.title == "Thought Record"
        assert assignment.status == CompletionStatus.ASSIGNED
        assert len(assignment.steps) > 0

    def test_assign_homework_invalid_template(self) -> None:
        """Test assigning homework with invalid template returns None."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="nonexistent_template",
        )
        assert assignment is None

    def test_assign_homework_with_custom_instructions(self) -> None:
        """Test assigning homework with custom instructions."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="behavioral_activation",
            custom_instructions="Focus on morning activities",
        )
        assert "Focus on morning activities" in assignment.description

    def test_create_custom_homework(self) -> None:
        """Test creating custom homework assignment."""
        manager = HomeworkManager()
        user_id = uuid4()
        assignment = manager.create_custom_homework(
            user_id=user_id,
            title="Custom Practice",
            description="Practice this custom technique",
            homework_type=HomeworkType.SKILL_PRACTICE,
            steps=["Step 1", "Step 2", "Step 3"],
        )
        assert assignment.title == "Custom Practice"
        assert len(assignment.steps) == 3
        assert assignment.steps[0].instruction == "Step 1"

    def test_get_assignment(self) -> None:
        """Test retrieving assignment by ID."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="mindfulness_practice",
        )
        retrieved = manager.get_assignment(assignment.homework_id)
        assert retrieved is not None
        assert retrieved.homework_id == assignment.homework_id

    def test_get_user_assignments(self) -> None:
        """Test getting all assignments for a user."""
        manager = HomeworkManager()
        user_id = uuid4()
        manager.assign_homework(user_id=user_id, template_name="thought_record")
        manager.assign_homework(user_id=user_id, template_name="mindfulness_practice")
        assignments = manager.get_user_assignments(user_id)
        assert len(assignments) == 2

    def test_get_user_assignments_by_status(self) -> None:
        """Test filtering assignments by status."""
        manager = HomeworkManager()
        user_id = uuid4()
        a1 = manager.assign_homework(user_id=user_id, template_name="thought_record")
        manager.assign_homework(user_id=user_id, template_name="mindfulness_practice")
        manager.complete_homework(a1.homework_id)
        pending = manager.get_user_assignments(user_id, status=CompletionStatus.ASSIGNED)
        assert len(pending) == 1

    def test_start_homework(self) -> None:
        """Test marking homework as started."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="thought_record",
        )
        result = manager.start_homework(assignment.homework_id)
        assert result is True
        assert assignment.status == CompletionStatus.IN_PROGRESS
        assert assignment.started_date is not None

    def test_complete_step(self) -> None:
        """Test completing a homework step."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="thought_record",
        )
        result = manager.complete_step(assignment.homework_id, 0, notes="Completed first step")
        assert result is True
        assert assignment.steps[0].completed is True
        assert assignment.steps[0].notes == "Completed first step"
        assert assignment.completion_percentage > 0

    def test_complete_homework(self) -> None:
        """Test completing homework assignment."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="thought_record",
        )
        # Complete all steps
        for i in range(len(assignment.steps)):
            manager.complete_step(assignment.homework_id, i)
        result = manager.complete_homework(
            assignment.homework_id,
            user_rating=4,
            user_feedback="Very helpful exercise",
        )
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["completion_percentage"] == 100
        assert assignment.user_rating == 4

    def test_partial_completion(self) -> None:
        """Test partial homework completion."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="thought_record",
        )
        manager.complete_step(assignment.homework_id, 0)
        result = manager.complete_homework(assignment.homework_id, partial=True)
        assert result["status"] == "partially_completed"

    def test_effectiveness_tracking(self) -> None:
        """Test effectiveness score calculation."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="thought_record",
        )
        for i in range(len(assignment.steps)):
            manager.complete_step(assignment.homework_id, i)
        result = manager.complete_homework(
            assignment.homework_id,
            user_rating=5,
            user_feedback="Excellent",
        )
        assert result["effectiveness_score"] is not None
        assert result["effectiveness_score"] > 0

    def test_get_pending_reminders(self) -> None:
        """Test getting homework needing reminders."""
        manager = HomeworkManager(HomeworkManagerSettings(reminder_threshold_hours=48))
        user_id = uuid4()
        assignment = manager.assign_homework(
            user_id=user_id,
            template_name="thought_record",
            due_days=1,  # Due tomorrow
        )
        reminders = manager.get_pending_reminders(user_id)
        assert len(reminders) == 1

    def test_get_overdue(self) -> None:
        """Test getting overdue homework."""
        manager = HomeworkManager()
        user_id = uuid4()
        assignment = manager.assign_homework(
            user_id=user_id,
            template_name="thought_record",
        )
        # Manually set due date to past
        assignment.due_date = datetime.now(timezone.utc) - timedelta(days=1)
        overdue = manager.get_overdue(user_id)
        assert len(overdue) == 1

    def test_get_completion_stats(self) -> None:
        """Test getting completion statistics."""
        manager = HomeworkManager()
        user_id = uuid4()
        a1 = manager.assign_homework(user_id=user_id, template_name="thought_record")
        a2 = manager.assign_homework(user_id=user_id, template_name="mindfulness_practice")
        for i in range(len(a1.steps)):
            manager.complete_step(a1.homework_id, i)
        manager.complete_homework(a1.homework_id)
        stats = manager.get_completion_stats(user_id)
        assert stats["total"] == 2
        assert stats["completed"] == 1
        assert stats["completion_rate"] == 0.5

    def test_recommend_difficulty(self) -> None:
        """Test difficulty recommendation based on history."""
        manager = HomeworkManager()
        user_id = uuid4()
        # Create and complete multiple assignments
        for _ in range(5):
            assignment = manager.assign_homework(user_id=user_id, template_name="thought_record")
            for i in range(len(assignment.steps)):
                manager.complete_step(assignment.homework_id, i)
            manager.complete_homework(assignment.homework_id)
        difficulty = manager.recommend_difficulty(user_id)
        assert difficulty == Difficulty.CHALLENGING

    def test_delete_assignment(self) -> None:
        """Test deleting homework assignment."""
        manager = HomeworkManager()
        user_id = uuid4()
        assignment = manager.assign_homework(user_id=user_id, template_name="thought_record")
        result = manager.delete_assignment(assignment.homework_id)
        assert result is True
        assert manager.get_assignment(assignment.homework_id) is None

    def test_get_template(self) -> None:
        """Test getting a specific template."""
        manager = HomeworkManager()
        template = manager.get_template("thought_record")
        assert template is not None
        assert template.name == "Thought Record"
        assert template.homework_type == HomeworkType.COGNITIVE

    def test_due_date_calculation(self) -> None:
        """Test due date is calculated correctly."""
        manager = HomeworkManager()
        assignment = manager.assign_homework(
            user_id=uuid4(),
            template_name="thought_record",
            due_days=5,
        )
        expected_due = datetime.now(timezone.utc) + timedelta(days=5)
        assert assignment.due_date is not None
        assert abs((assignment.due_date - expected_due).total_seconds()) < 60
