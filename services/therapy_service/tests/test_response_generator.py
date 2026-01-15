"""
Unit tests for Therapy Service Response Generator.
Tests helper functions for generating therapeutic responses and summaries.
"""
from __future__ import annotations
import pytest
from dataclasses import dataclass, field
from uuid import uuid4, UUID
from datetime import datetime, timezone

from services.therapy_service.src.schemas import (
    SessionPhase, TherapyModality, SeverityLevel, RiskLevel,
    TechniqueDTO, TreatmentPlanDTO, HomeworkDTO,
)
from services.therapy_service.src.domain.response_generator import ResponseGenerator


@dataclass
class MockSession:
    """Mock session for testing."""
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    current_phase: SessionPhase = SessionPhase.WORKING
    topics_covered: list[str] = field(default_factory=list)
    skills_practiced: list[str] = field(default_factory=list)
    insights_gained: list[str] = field(default_factory=list)
    techniques_used: list[TechniqueDTO] = field(default_factory=list)
    homework_assigned: list[HomeworkDTO] = field(default_factory=list)
    session_rating: int = 7
    current_risk: RiskLevel = RiskLevel.NONE


class TestGenerateTherapeuticResponse:
    """Tests for generate_therapeutic_response."""

    def test_pre_session_response(self) -> None:
        """Test response for PRE_SESSION phase."""
        session = MockSession(current_phase=SessionPhase.PRE_SESSION)
        response = ResponseGenerator.generate_therapeutic_response(session, "Hello", None, [])
        assert "prepare" in response.lower()

    def test_opening_response(self) -> None:
        """Test response for OPENING phase."""
        session = MockSession(current_phase=SessionPhase.OPENING)
        response = ResponseGenerator.generate_therapeutic_response(session, "Hello", None, [])
        assert "feeling" in response.lower()

    def test_working_response(self) -> None:
        """Test response for WORKING phase."""
        session = MockSession(current_phase=SessionPhase.WORKING)
        response = ResponseGenerator.generate_therapeutic_response(session, "I'm struggling", None, [])
        assert "explore" in response.lower()

    def test_closing_response(self) -> None:
        """Test response for CLOSING phase."""
        session = MockSession(current_phase=SessionPhase.CLOSING)
        response = ResponseGenerator.generate_therapeutic_response(session, "That was helpful", None, [])
        assert "insight" in response.lower() or "forward" in response.lower()

    def test_post_session_response(self) -> None:
        """Test response for POST_SESSION phase."""
        session = MockSession(current_phase=SessionPhase.POST_SESSION)
        response = ResponseGenerator.generate_therapeutic_response(session, "Thanks", None, [])
        assert "thank" in response.lower()

    def test_response_with_technique(self) -> None:
        """Test response includes technique information."""
        session = MockSession(current_phase=SessionPhase.WORKING)
        technique = TechniqueDTO(
            technique_id=uuid4(),
            name="Thought Record",
            description="Identify and challenge negative thoughts",
            modality=TherapyModality.CBT,
            category="cognitive",
            requires_homework=True,
        )
        response = ResponseGenerator.generate_therapeutic_response(session, "I'm anxious", technique, [])
        assert "Thought Record" in response
        assert "technique" in response.lower()

    def test_closing_includes_insights(self) -> None:
        """Test closing phase includes insights gained."""
        session = MockSession(
            current_phase=SessionPhase.CLOSING,
            insights_gained=["Identified triggers", "Learned coping skills"],
        )
        response = ResponseGenerator.generate_therapeutic_response(session, "Thanks", None, [])
        assert "covered" in response.lower()


class TestGenerateInitialMessage:
    """Tests for generate_initial_message."""

    def test_first_session_message(self) -> None:
        """Test initial message for first session."""
        plan = TreatmentPlanDTO(
            plan_id=uuid4(),
            user_id=uuid4(),
            primary_diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            primary_modality=TherapyModality.CBT,
            adjunct_modalities=[],
            current_phase=1,
            sessions_completed=0,
            skills_acquired=[],
        )
        message = ResponseGenerator.generate_initial_message(1, plan)
        assert "first" in message.lower()
        assert "Depression" in message
        assert "feeling" in message.lower()

    def test_followup_session_message(self) -> None:
        """Test initial message for follow-up session."""
        plan = TreatmentPlanDTO(
            plan_id=uuid4(),
            user_id=uuid4(),
            primary_diagnosis="Anxiety",
            severity=SeverityLevel.MILD,
            primary_modality=TherapyModality.CBT,
            adjunct_modalities=[],
            current_phase=2,
            sessions_completed=2,
            skills_acquired=["breathing"],
        )
        message = ResponseGenerator.generate_initial_message(3, plan)
        assert "session 3" in message.lower()
        assert "back" in message.lower()


class TestGenerateSuggestedAgenda:
    """Tests for generate_suggested_agenda."""

    def test_first_session_agenda(self) -> None:
        """Test agenda for first session."""
        plan = TreatmentPlanDTO(
            plan_id=uuid4(),
            user_id=uuid4(),
            primary_diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            primary_modality=TherapyModality.CBT,
            adjunct_modalities=[],
            current_phase=1,
            sessions_completed=0,
            skills_acquired=[],
        )
        agenda = ResponseGenerator.generate_suggested_agenda(plan, 1)
        assert len(agenda) >= 3
        assert any("mood" in item.lower() for item in agenda)

    def test_followup_session_includes_homework(self) -> None:
        """Test follow-up session agenda includes homework review."""
        plan = TreatmentPlanDTO(
            plan_id=uuid4(),
            user_id=uuid4(),
            primary_diagnosis="Anxiety",
            severity=SeverityLevel.MILD,
            primary_modality=TherapyModality.CBT,
            adjunct_modalities=[],
            current_phase=2,
            sessions_completed=3,
            skills_acquired=[],
        )
        agenda = ResponseGenerator.generate_suggested_agenda(plan, 4)
        assert any("homework" in item.lower() for item in agenda)


class TestGenerateSessionSummary:
    """Tests for generate_session_summary."""

    def test_basic_summary(self) -> None:
        """Test basic session summary generation."""
        session = MockSession(
            session_number=2,
            topics_covered=["CBT basics", "Thought patterns"],
            skills_practiced=["Deep breathing"],
        )
        summary = ResponseGenerator.generate_session_summary(session, 45)
        assert summary.session_number == 2
        assert summary.duration_minutes == 45
        assert "Deep breathing" in summary.skills_practiced

    def test_summary_with_techniques(self) -> None:
        """Test summary includes techniques used."""
        technique = TechniqueDTO(
            technique_id=uuid4(),
            name="Thought Record",
            description="Identify thoughts",
            modality=TherapyModality.CBT,
            category="cognitive",
            requires_homework=True,
        )
        session = MockSession(
            techniques_used=[technique],
            topics_covered=["Cognitive restructuring"],
        )
        summary = ResponseGenerator.generate_session_summary(session, 30)
        assert "Thought Record" in summary.summary_text


class TestGenerateSummaryText:
    """Tests for generate_summary_text."""

    def test_summary_text_with_topics(self) -> None:
        """Test summary text mentions topics covered."""
        session = MockSession(topics_covered=["Topic 1", "Topic 2", "Topic 3"])
        text = ResponseGenerator.generate_summary_text(session)
        assert "3 key areas" in text

    def test_summary_text_with_techniques(self) -> None:
        """Test summary text mentions techniques."""
        technique = TechniqueDTO(
            technique_id=uuid4(),
            name="Test Technique",
            description="Description",
            modality=TherapyModality.CBT,
            category="cognitive",
            requires_homework=False,
        )
        session = MockSession(techniques_used=[technique])
        text = ResponseGenerator.generate_summary_text(session)
        assert "Test Technique" in text

    def test_summary_text_with_skills(self) -> None:
        """Test summary text mentions skills practiced."""
        session = MockSession(skills_practiced=["Skill 1", "Skill 2"])
        text = ResponseGenerator.generate_summary_text(session)
        assert "2 skills" in text


class TestGenerateNextFocus:
    """Tests for generate_next_focus."""

    def test_next_focus_with_skills(self) -> None:
        """Test next focus suggests continuing skills."""
        session = MockSession(skills_practiced=["Breathing", "Grounding"])
        focus = ResponseGenerator.generate_next_focus(session)
        assert "Breathing" in focus
        assert "continue" in focus.lower()

    def test_next_focus_without_skills(self) -> None:
        """Test next focus without skills practiced."""
        session = MockSession(skills_practiced=[])
        focus = ResponseGenerator.generate_next_focus(session)
        assert "building" in focus.lower()


class TestGenerateRecommendations:
    """Tests for generate_recommendations."""

    def test_recommendations_with_risk(self) -> None:
        """Test recommendations include safety for elevated risk."""
        session = MockSession(current_risk=RiskLevel.MEDIUM)
        recs = ResponseGenerator.generate_recommendations(session)
        assert any("safety" in r.lower() for r in recs)

    def test_recommendations_with_homework(self) -> None:
        """Test recommendations include homework completion."""
        homework = HomeworkDTO(
            homework_id=uuid4(),
            title="Practice",
            description="Practice breathing",
            technique_id=uuid4(),
            completed=False,
        )
        session = MockSession(homework_assigned=[homework])
        recs = ResponseGenerator.generate_recommendations(session)
        assert any("homework" in r.lower() for r in recs)

    def test_recommendations_with_skills(self) -> None:
        """Test recommendations include skill practice."""
        session = MockSession(skills_practiced=["Deep breathing"])
        recs = ResponseGenerator.generate_recommendations(session)
        assert any("Deep breathing" in r for r in recs)

    def test_standard_recommendations(self) -> None:
        """Test standard recommendations are included."""
        session = MockSession()
        recs = ResponseGenerator.generate_recommendations(session)
        assert any("mood" in r.lower() for r in recs)
        assert any("next session" in r.lower() for r in recs)


class TestGenerateNextSteps:
    """Tests for generate_next_steps."""

    def test_opening_next_steps(self) -> None:
        """Test next steps for opening phase."""
        session = MockSession(current_phase=SessionPhase.OPENING)
        steps = ResponseGenerator.generate_next_steps(session)
        assert len(steps) > 0

    def test_working_next_steps(self) -> None:
        """Test next steps for working phase."""
        session = MockSession(current_phase=SessionPhase.WORKING)
        steps = ResponseGenerator.generate_next_steps(session)
        assert len(steps) > 0

    def test_closing_next_steps(self) -> None:
        """Test next steps for closing phase."""
        session = MockSession(current_phase=SessionPhase.CLOSING)
        steps = ResponseGenerator.generate_next_steps(session)
        assert len(steps) > 0

    def test_post_session_next_steps(self) -> None:
        """Test next steps for post-session phase."""
        session = MockSession(current_phase=SessionPhase.POST_SESSION)
        steps = ResponseGenerator.generate_next_steps(session)
        assert len(steps) > 0


class TestGenerateCrisisResponse:
    """Tests for generate_crisis_response."""

    def test_crisis_response_includes_resources(self) -> None:
        """Test crisis response includes crisis resources."""
        response = ResponseGenerator.generate_crisis_response(["Crisis keyword detected"])
        assert "988" in response
        assert "741741" in response
        assert "911" in response

    def test_crisis_response_empathetic(self) -> None:
        """Test crisis response is empathetic."""
        response = ResponseGenerator.generate_crisis_response(["Harm language detected"])
        assert "safety" in response.lower()
        assert "support" in response.lower()
