"""
Unit tests for Therapy Service Session Manager.
Tests session lifecycle, state machine, and phase transitions.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from services.therapy_service.src.schemas import SessionPhase, RiskLevel
from services.therapy_service.src.domain.session_manager import (
    SessionManager, SessionManagerSettings, SessionState
)


class TestSessionManagerSettings:
    """Tests for SessionManagerSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings are properly initialized."""
        settings = SessionManagerSettings()
        assert settings.min_opening_duration_sec == 180
        assert settings.max_session_duration_min == 60
        assert settings.enable_flexible_transitions is True
        assert settings.min_engagement_score == 0.3
        assert settings.track_detailed_metrics is True

    def test_custom_settings(self) -> None:
        """Test custom settings override defaults."""
        settings = SessionManagerSettings(
            max_session_duration_min=30,
            min_engagement_score=0.5,
            enable_flexible_transitions=False,
        )
        assert settings.max_session_duration_min == 30
        assert settings.min_engagement_score == 0.5
        assert settings.enable_flexible_transitions is False


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_session_state_creation(self) -> None:
        """Test SessionState is properly created."""
        user_id = uuid4()
        plan_id = uuid4()
        state = SessionState(
            session_id=uuid4(),
            user_id=user_id,
            treatment_plan_id=plan_id,
            session_number=1,
        )
        assert state.user_id == user_id
        assert state.treatment_plan_id == plan_id
        assert state.session_number == 1
        assert state.current_phase == SessionPhase.PRE_SESSION
        assert state.mood_rating is None
        assert state.current_risk == RiskLevel.NONE

    def test_session_state_defaults(self) -> None:
        """Test SessionState default values."""
        state = SessionState(
            session_id=uuid4(),
            user_id=uuid4(),
            treatment_plan_id=uuid4(),
            session_number=1,
        )
        assert len(state.agenda_items) == 0
        assert len(state.topics_covered) == 0
        assert len(state.skills_practiced) == 0
        assert len(state.messages) == 0
        assert state.engagement_score == 0.0


class TestSessionManager:
    """Tests for SessionManager functionality."""

    def test_create_session(self) -> None:
        """Test session creation."""
        manager = SessionManager()
        user_id = uuid4()
        plan_id = uuid4()
        session_id, session = manager.create_session(
            user_id=user_id, treatment_plan_id=plan_id, context={}
        )
        assert session_id is not None
        assert session.user_id == user_id
        assert session.treatment_plan_id == plan_id
        assert session.session_number == 1

    def test_create_multiple_sessions_increments_number(self) -> None:
        """Test session number increments for same user."""
        manager = SessionManager()
        user_id = uuid4()
        plan_id = uuid4()
        _, session1 = manager.create_session(user_id, plan_id, {})
        _, session2 = manager.create_session(user_id, plan_id, {})
        _, session3 = manager.create_session(user_id, plan_id, {})
        assert session1.session_number == 1
        assert session2.session_number == 2
        assert session3.session_number == 3

    def test_get_session(self) -> None:
        """Test session retrieval."""
        manager = SessionManager()
        user_id = uuid4()
        plan_id = uuid4()
        session_id, _ = manager.create_session(user_id, plan_id, {})
        session = manager.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id

    def test_get_nonexistent_session(self) -> None:
        """Test retrieval of non-existent session."""
        manager = SessionManager()
        session = manager.get_session(uuid4())
        assert session is None

    def test_update_state_mood(self) -> None:
        """Test updating mood rating."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.update_state(session_id, {"mood_rating": 7})
        session = manager.get_session(session_id)
        assert session.mood_rating == 7

    def test_update_state_agenda(self) -> None:
        """Test updating agenda items."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.update_state(session_id, {"agenda_items": ["Discuss anxiety"]})
        manager.update_state(session_id, {"agenda_items": ["Practice breathing"]})
        session = manager.get_session(session_id)
        assert len(session.agenda_items) == 2
        assert "Discuss anxiety" in session.agenda_items

    def test_update_state_topics(self) -> None:
        """Test updating topics covered."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.update_state(session_id, {"topics_covered": "Cognitive distortions"})
        session = manager.get_session(session_id)
        assert "Cognitive distortions" in session.topics_covered

    def test_update_state_skills(self) -> None:
        """Test updating skills practiced."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.update_state(session_id, {"skills_practiced": "Deep breathing"})
        session = manager.get_session(session_id)
        assert "Deep breathing" in session.skills_practiced

    def test_update_state_messages(self) -> None:
        """Test updating messages."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.update_state(session_id, {"messages": {"role": "user", "content": "Hello"}})
        session = manager.get_session(session_id)
        assert len(session.messages) == 1

    def test_update_state_risk_level(self) -> None:
        """Test updating risk level."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.update_state(session_id, {"current_risk": RiskLevel.HIGH})
        session = manager.get_session(session_id)
        assert session.current_risk == RiskLevel.HIGH

    def test_delete_session(self) -> None:
        """Test session deletion."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        assert manager.get_session(session_id) is not None
        manager.delete_session(session_id)
        assert manager.get_session(session_id) is None

    def test_get_active_session_count(self) -> None:
        """Test active session count."""
        manager = SessionManager()
        assert manager.get_active_session_count() == 0
        manager.create_session(uuid4(), uuid4(), {})
        assert manager.get_active_session_count() == 1
        manager.create_session(uuid4(), uuid4(), {})
        assert manager.get_active_session_count() == 2


class TestPhaseTransitions:
    """Tests for session phase transitions."""

    def test_transition_to_opening(self) -> None:
        """Test transition from PRE_SESSION to OPENING."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        result = manager.transition_phase(session_id, SessionPhase.OPENING, "session_start")
        assert result.allowed is True
        session = manager.get_session(session_id)
        assert session.current_phase == SessionPhase.OPENING

    def test_transition_to_working(self) -> None:
        """Test transition from OPENING to WORKING."""
        manager = SessionManager(SessionManagerSettings(enable_flexible_transitions=True))
        session_id, session = manager.create_session(uuid4(), uuid4(), {})
        manager.transition_phase(session_id, SessionPhase.OPENING, "start")
        result = manager.transition_phase(session_id, SessionPhase.WORKING, "criteria_met")
        assert result.allowed is True
        session = manager.get_session(session_id)
        assert session.current_phase == SessionPhase.WORKING

    def test_transition_to_closing(self) -> None:
        """Test transition to CLOSING with criteria met."""
        manager = SessionManager(SessionManagerSettings(enable_flexible_transitions=True))
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.transition_phase(session_id, SessionPhase.OPENING, "start")
        manager.transition_phase(session_id, SessionPhase.WORKING, "criteria")
        # Add required criteria for WORKING -> CLOSING transition
        manager.update_state(session_id, {"skills_practiced": "Breathing exercise"})
        manager.update_state(session_id, {"skills_practiced": "Grounding"})
        result = manager.transition_phase(session_id, SessionPhase.CLOSING, "work_complete")
        assert result.allowed is True
        session = manager.get_session(session_id)
        assert session.current_phase == SessionPhase.CLOSING

    def test_transition_to_post_session(self) -> None:
        """Test transition to POST_SESSION."""
        manager = SessionManager(SessionManagerSettings(enable_flexible_transitions=True))
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.transition_phase(session_id, SessionPhase.OPENING, "start")
        manager.transition_phase(session_id, SessionPhase.WORKING, "criteria")
        manager.transition_phase(session_id, SessionPhase.CLOSING, "complete")
        result = manager.transition_phase(session_id, SessionPhase.POST_SESSION, "session_end")
        assert result.allowed is True
        session = manager.get_session(session_id)
        assert session.current_phase == SessionPhase.POST_SESSION

    def test_invalid_transition_rejected(self) -> None:
        """Test invalid phase transition is rejected."""
        manager = SessionManager(SessionManagerSettings(enable_flexible_transitions=False))
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        result = manager.transition_phase(session_id, SessionPhase.POST_SESSION, "invalid")
        assert result.allowed is False

    def test_crisis_override_transition(self) -> None:
        """Test crisis override allows any transition to CLOSING."""
        manager = SessionManager()
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.transition_phase(session_id, SessionPhase.OPENING, "start")
        manager.update_state(session_id, {"current_risk": RiskLevel.HIGH})
        result = manager.transition_phase(session_id, SessionPhase.CLOSING, "crisis_protocol")
        assert result.allowed is True


class TestEngagementScore:
    """Tests for engagement score calculation."""

    def test_engagement_score_increases_with_messages(self) -> None:
        """Test engagement score increases with messages."""
        manager = SessionManager()
        session_id, session = manager.create_session(uuid4(), uuid4(), {})
        initial_score = manager.calculate_engagement_score(session)
        for i in range(5):
            manager.update_state(session_id, {"messages": {"role": "user", "content": f"Message {i}"}})
        session = manager.get_session(session_id)
        final_score = manager.calculate_engagement_score(session)
        assert final_score >= initial_score

    def test_engagement_score_increases_with_skills(self) -> None:
        """Test engagement score increases with skills practiced."""
        manager = SessionManager()
        session_id, session = manager.create_session(uuid4(), uuid4(), {})
        manager.update_state(session_id, {"skills_practiced": "Skill 1"})
        manager.update_state(session_id, {"skills_practiced": "Skill 2"})
        session = manager.get_session(session_id)
        score = manager.calculate_engagement_score(session)
        assert score > 0

    def test_engagement_score_capped_at_one(self) -> None:
        """Test engagement score is capped at 1.0."""
        manager = SessionManager()
        session_id, session = manager.create_session(uuid4(), uuid4(), {})
        for i in range(50):
            manager.update_state(session_id, {"messages": {"role": "user", "content": f"Long message {i}" * 100}})
            manager.update_state(session_id, {"skills_practiced": f"Skill {i}"})
            manager.update_state(session_id, {"insights_gained": f"Insight {i}"})
        session = manager.get_session(session_id)
        score = manager.calculate_engagement_score(session)
        assert score <= 1.0


class TestTransitionValidation:
    """Tests for transition validation criteria."""

    def test_validate_opening_to_working_insufficient_messages(self) -> None:
        """Test OPENING to WORKING requires sufficient messages."""
        manager = SessionManager(SessionManagerSettings(enable_flexible_transitions=False))
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.transition_phase(session_id, SessionPhase.OPENING, "start")
        result = manager.transition_phase(session_id, SessionPhase.WORKING, "test")
        assert result.allowed is False

    def test_validate_working_to_closing_insufficient_time(self) -> None:
        """Test WORKING to CLOSING requires sufficient time."""
        manager = SessionManager(SessionManagerSettings(
            enable_flexible_transitions=False,
            min_opening_duration_sec=180,
        ))
        session_id, _ = manager.create_session(uuid4(), uuid4(), {})
        manager.transition_phase(session_id, SessionPhase.OPENING, "start")
        manager.transition_phase(session_id, SessionPhase.WORKING, "auto")
        session = manager.get_session(session_id)
        session.phase_history.append({
            "from_phase": SessionPhase.OPENING,
            "to_phase": SessionPhase.WORKING,
            "timestamp": datetime.now(timezone.utc),
        })
        result = manager.transition_phase(session_id, SessionPhase.CLOSING, "test")
        assert result.allowed is False
