"""
Unit tests for Therapy Service Schemas.
Tests Pydantic V2 request/response models and enums.
"""
from __future__ import annotations
import pytest
from uuid import uuid4

from services.therapy_service.src.schemas import (
    SessionPhase, TherapyModality, SeverityLevel, RiskLevel,
    TechniqueDTO, HomeworkDTO, TreatmentPlanDTO, SessionStateDTO, SessionSummaryDTO,
    SessionStartRequest, MessageRequest, SessionEndRequest,
    SessionStartResponse, TherapyResponse,
)


class TestEnums:
    """Tests for enum definitions."""

    def test_session_phase_values(self) -> None:
        """Test SessionPhase enum values."""
        assert SessionPhase.PRE_SESSION == "pre_session"
        assert SessionPhase.OPENING == "opening"
        assert SessionPhase.WORKING == "working"
        assert SessionPhase.CLOSING == "closing"
        assert SessionPhase.POST_SESSION == "post_session"
        assert SessionPhase.CRISIS == "crisis"

    def test_therapy_modality_values(self) -> None:
        """Test TherapyModality enum values."""
        assert TherapyModality.CBT == "cbt"
        assert TherapyModality.DBT == "dbt"
        assert TherapyModality.ACT == "act"
        assert TherapyModality.MI == "mi"
        assert TherapyModality.MINDFULNESS == "mindfulness"

    def test_severity_level_values(self) -> None:
        """Test SeverityLevel enum values."""
        assert SeverityLevel.MINIMAL == "MINIMAL"
        assert SeverityLevel.MILD == "MILD"
        assert SeverityLevel.MODERATE == "MODERATE"
        assert SeverityLevel.MODERATELY_SEVERE == "MODERATELY_SEVERE"
        assert SeverityLevel.SEVERE == "SEVERE"

    def test_risk_level_values(self) -> None:
        """Test RiskLevel enum values."""
        assert RiskLevel.NONE == "NONE"
        assert RiskLevel.LOW == "LOW"
        assert RiskLevel.ELEVATED == "ELEVATED"
        assert RiskLevel.HIGH == "HIGH"
        assert RiskLevel.CRITICAL == "CRITICAL"


class TestTechniqueDTO:
    """Tests for TechniqueDTO model."""

    def test_technique_dto_creation(self) -> None:
        """Test TechniqueDTO creation."""
        technique = TechniqueDTO(
            technique_id=uuid4(),
            name="Thought Record",
            description="Identify and challenge negative thoughts",
            modality=TherapyModality.CBT,
            category="cognitive",
            requires_homework=True,
        )
        assert technique.name == "Thought Record"
        assert technique.modality == TherapyModality.CBT
        assert technique.requires_homework is True

    def test_technique_dto_optional_fields(self) -> None:
        """Test TechniqueDTO optional fields."""
        technique = TechniqueDTO(
            technique_id=uuid4(),
            name="Mindfulness of Breath",
            description="Focus on breath",
            modality=TherapyModality.MINDFULNESS,
            category="mindfulness",
            requires_homework=False,
            contraindications=["dissociation"],
            duration_minutes=10,
        )
        assert technique.contraindications == ["dissociation"]
        assert technique.duration_minutes == 10


class TestHomeworkDTO:
    """Tests for HomeworkDTO model."""

    def test_homework_dto_creation(self) -> None:
        """Test HomeworkDTO creation."""
        homework = HomeworkDTO(
            homework_id=uuid4(),
            title="Practice Thought Records",
            description="Complete 3 thought records this week",
            technique_id=uuid4(),
            completed=False,
        )
        assert homework.title == "Practice Thought Records"
        assert homework.completed is False

    def test_homework_dto_with_due_date(self) -> None:
        """Test HomeworkDTO with due date."""
        from datetime import datetime, timezone
        due = datetime.now(timezone.utc)
        homework = HomeworkDTO(
            homework_id=uuid4(),
            title="Practice",
            description="Description",
            technique_id=uuid4(),
            due_date=due,
            completed=False,
        )
        assert homework.due_date == due


class TestTreatmentPlanDTO:
    """Tests for TreatmentPlanDTO model."""

    def test_treatment_plan_dto_creation(self) -> None:
        """Test TreatmentPlanDTO creation."""
        plan = TreatmentPlanDTO(
            plan_id=uuid4(),
            user_id=uuid4(),
            primary_diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            primary_modality=TherapyModality.CBT,
            adjunct_modalities=[TherapyModality.MINDFULNESS],
            current_phase=1,
            sessions_completed=0,
            skills_acquired=[],
        )
        assert plan.primary_diagnosis == "Depression"
        assert plan.severity == SeverityLevel.MODERATE
        assert plan.primary_modality == TherapyModality.CBT
        assert TherapyModality.MINDFULNESS in plan.adjunct_modalities


class TestSessionStateDTO:
    """Tests for SessionStateDTO model."""

    def test_session_state_dto_creation(self) -> None:
        """Test SessionStateDTO creation."""
        state = SessionStateDTO(
            session_id=uuid4(),
            user_id=uuid4(),
            treatment_plan_id=uuid4(),
            session_number=1,
            current_phase=SessionPhase.WORKING,
            mood_rating=6,
            agenda_items=["Discuss anxiety"],
            topics_covered=["CBT basics"],
            skills_practiced=["Deep breathing"],
            current_risk=RiskLevel.NONE,
            engagement_score=0.75,
        )
        assert state.session_number == 1
        assert state.current_phase == SessionPhase.WORKING
        assert state.mood_rating == 6
        assert state.engagement_score == 0.75


class TestSessionSummaryDTO:
    """Tests for SessionSummaryDTO model."""

    def test_session_summary_dto_creation(self) -> None:
        """Test SessionSummaryDTO creation."""
        summary = SessionSummaryDTO(
            session_id=uuid4(),
            user_id=uuid4(),
            session_number=3,
            duration_minutes=45,
            techniques_used=[],
            skills_practiced=["Grounding"],
            insights_gained=["Identified triggers"],
            homework_assigned=[],
            session_rating=8,
            summary_text="Good session progress",
            next_session_focus="Continue skill practice",
        )
        assert summary.session_number == 3
        assert summary.duration_minutes == 45
        assert "Grounding" in summary.skills_practiced


class TestRequestModels:
    """Tests for request models."""

    def test_session_start_request(self) -> None:
        """Test SessionStartRequest model."""
        request = SessionStartRequest(
            user_id=uuid4(),
            treatment_plan_id=uuid4(),
            context={"diagnosis": "Anxiety"},
        )
        assert request.context["diagnosis"] == "Anxiety"

    def test_message_request(self) -> None:
        """Test MessageRequest model."""
        request = MessageRequest(
            session_id=uuid4(),
            user_id=uuid4(),
            message="I've been feeling better this week",
        )
        assert "feeling better" in request.message

    def test_session_end_request(self) -> None:
        """Test SessionEndRequest model."""
        request = SessionEndRequest(
            session_id=uuid4(),
            user_id=uuid4(),
            generate_summary=True,
        )
        assert request.generate_summary is True


class TestResponseModels:
    """Tests for response models."""

    def test_session_start_response(self) -> None:
        """Test SessionStartResponse model."""
        response = SessionStartResponse(
            session_id=uuid4(),
            user_id=uuid4(),
            treatment_plan_id=uuid4(),
            session_number=1,
            current_phase=SessionPhase.OPENING,
            initial_message="Welcome to your session",
            suggested_agenda=["Check in", "Review homework"],
        )
        assert response.session_number == 1
        assert "Welcome" in response.initial_message

    def test_therapy_response(self) -> None:
        """Test TherapyResponse model."""
        response = TherapyResponse(
            session_id=uuid4(),
            user_id=uuid4(),
            response_text="I hear you saying...",
            current_phase=SessionPhase.WORKING,
            technique_applied=None,
            homework_assigned=[],
            safety_alerts=[],
            next_steps=["Share more about your experience"],
            processing_time_ms=150,
        )
        assert response.current_phase == SessionPhase.WORKING
        assert response.processing_time_ms == 150


class TestValidation:
    """Tests for model validation."""

    def test_invalid_severity_level(self) -> None:
        """Test invalid severity level raises error."""
        with pytest.raises(ValueError):
            TreatmentPlanDTO(
                plan_id=uuid4(),
                user_id=uuid4(),
                primary_diagnosis="Test",
                severity="invalid",  # type: ignore
                primary_modality=TherapyModality.CBT,
            )

    def test_invalid_risk_level(self) -> None:
        """Test invalid risk level raises error."""
        with pytest.raises(ValueError):
            SessionStateDTO(
                session_id=uuid4(),
                user_id=uuid4(),
                treatment_plan_id=uuid4(),
                session_number=1,
                current_phase=SessionPhase.WORKING,
                current_risk="invalid",  # type: ignore
                engagement_score=0.5,
            )
