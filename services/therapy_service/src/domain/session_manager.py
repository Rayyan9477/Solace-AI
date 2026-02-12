"""
Solace-AI Therapy Service - Session State Management.
Manages therapy session lifecycle, phase transitions, and state tracking.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from uuid import UUID
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import SessionPhase, RiskLevel, TechniqueDTO, HomeworkDTO
from .models import SessionState, PhaseTransitionResult

logger = structlog.get_logger(__name__)


class SessionManagerSettings(BaseSettings):
    """Session manager configuration."""
    min_opening_duration_sec: int = Field(default=180)
    max_session_duration_min: int = Field(default=60)
    min_engagement_score: float = Field(default=0.3)
    enable_flexible_transitions: bool = Field(default=True)
    track_detailed_metrics: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="SESSION_MANAGER_", env_file=".env", extra="ignore")


class SessionManager:
    """
    Manages therapy session state and phase transitions.

    Implements session state machine with flexible phase transitions based on
    clinical criteria, user engagement, and safety considerations.
    """

    def __init__(self, settings: SessionManagerSettings | None = None) -> None:
        self._settings = settings or SessionManagerSettings()
        self._active_sessions: dict[UUID, SessionState] = {}
        self._user_session_counts: dict[UUID, int] = {}
        logger.info("session_manager_initialized", settings_applied=True)

    def create_session(
        self,
        user_id: UUID,
        treatment_plan_id: UUID,
        context: dict[str, Any],
    ) -> tuple[UUID, SessionState]:
        """
        Create new therapy session.

        Args:
            user_id: User identifier
            treatment_plan_id: Associated treatment plan
            context: Initial session context

        Returns:
            Tuple of (session_id, session_state)
        """
        session_number = self._user_session_counts.get(user_id, 0) + 1
        self._user_session_counts[user_id] = session_number

        session = SessionState(
            user_id=user_id,
            treatment_plan_id=treatment_plan_id,
            session_number=session_number,
            current_phase=SessionPhase.PRE_SESSION,
        )

        self._active_sessions[session.session_id] = session

        logger.info(
            "session_created",
            session_id=str(session.session_id),
            user_id=str(user_id),
            session_number=session_number
        )

        return session.session_id, session

    def get_session(self, session_id: UUID) -> SessionState | None:
        """
        Retrieve active session state.

        Args:
            session_id: Session identifier

        Returns:
            Session state or None if not found
        """
        return self._active_sessions.get(session_id)

    def update_state(
        self,
        session_id: UUID,
        updates: dict[str, Any],
    ) -> SessionState | None:
        """
        Update session state with new values.

        Args:
            session_id: Session identifier
            updates: Dictionary of state updates

        Returns:
            Updated session state or None if session not found
        """
        session = self._active_sessions.get(session_id)
        if not session:
            logger.warning("session_not_found", session_id=str(session_id))
            return None

        for key, value in updates.items():
            if hasattr(session, key):
                if key == "topics_covered" and isinstance(value, str):
                    if value not in session.topics_covered:
                        session.topics_covered.append(value)
                elif key == "skills_practiced" and isinstance(value, str):
                    if value not in session.skills_practiced:
                        session.skills_practiced.append(value)
                elif key == "insights_gained" and isinstance(value, str):
                    if value not in session.insights_gained:
                        session.insights_gained.append(value)
                elif key == "techniques_used" and isinstance(value, TechniqueDTO):
                    session.techniques_used.append(value)
                elif key == "homework_assigned" and isinstance(value, HomeworkDTO):
                    session.homework_assigned.append(value)
                elif key == "messages" and isinstance(value, dict):
                    session.messages.append(value)
                elif key == "safety_flags" and isinstance(value, str):
                    if value not in session.safety_flags:
                        session.safety_flags.append(value)
                elif key == "agenda_items" and isinstance(value, list):
                    session.agenda_items.extend(value)
                else:
                    setattr(session, key, value)

        logger.debug(
            "session_state_updated",
            session_id=str(session_id),
            updates_applied=list(updates.keys())
        )

        return session

    def transition_phase(
        self,
        session_id: UUID,
        target_phase: SessionPhase,
        trigger: str = "automatic",
    ) -> PhaseTransitionResult:
        """
        Attempt to transition session to new phase.

        Args:
            session_id: Session identifier
            target_phase: Desired phase to transition to
            trigger: Reason for transition attempt

        Returns:
            PhaseTransitionResult with transition status and details
        """
        session = self._active_sessions.get(session_id)
        if not session:
            return PhaseTransitionResult(
                from_phase=SessionPhase.PRE_SESSION,
                to_phase=target_phase,
                trigger=trigger,
                allowed=False,
            )

        current_phase = session.current_phase
        validation = self.validate_transition(session, target_phase)

        if not validation["allowed"]:
            logger.info(
                "phase_transition_blocked",
                session_id=str(session_id),
                from_phase=current_phase.value,
                to_phase=target_phase.value,
                reason=validation["reason"]
            )
            return PhaseTransitionResult(
                from_phase=current_phase,
                to_phase=target_phase,
                trigger=trigger,
                allowed=False,
                criteria_met=validation.get("criteria_met", []),
            )

        session.current_phase = target_phase
        transition_record = {
            "from_phase": current_phase.value,
            "to_phase": target_phase.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
        }
        session.phase_history.append(transition_record)

        logger.info(
            "phase_transition_completed",
            session_id=str(session_id),
            from_phase=current_phase.value,
            to_phase=target_phase.value,
            trigger=trigger
        )

        return PhaseTransitionResult(
            from_phase=current_phase,
            to_phase=target_phase,
            trigger=trigger,
            allowed=True,
            criteria_met=validation.get("criteria_met", []),
        )

    def validate_transition(
        self,
        session: SessionState,
        target_phase: SessionPhase,
    ) -> dict[str, Any]:
        """
        Validate if phase transition is allowed.

        Args:
            session: Current session state
            target_phase: Desired target phase

        Returns:
            Dictionary with 'allowed' boolean and transition details
        """
        current = session.current_phase
        criteria_met = []
        criteria_missing = []

        if current == SessionPhase.PRE_SESSION and target_phase == SessionPhase.OPENING:
            criteria_met.append("session_initialized")
            return {"allowed": True, "criteria_met": criteria_met}

        if current == SessionPhase.OPENING and target_phase == SessionPhase.WORKING:
            duration_sec = (datetime.now(timezone.utc) - session.started_at).total_seconds()
            if duration_sec >= self._settings.min_opening_duration_sec:
                criteria_met.append("min_duration_met")
            if session.mood_rating is not None:
                criteria_met.append("mood_check_completed")
            if len(session.agenda_items) > 0:
                criteria_met.append("agenda_set")
            if len(session.messages) >= 4:
                criteria_met.append("engagement_established")
            if session.current_risk == RiskLevel.NONE or session.current_risk == RiskLevel.LOW:
                criteria_met.append("safety_cleared")

            if len(criteria_met) >= 3 or self._settings.enable_flexible_transitions:
                return {"allowed": True, "criteria_met": criteria_met}
            return {
                "allowed": False,
                "reason": "insufficient_opening_criteria",
                "criteria_met": criteria_met,
            }

        if current == SessionPhase.WORKING and target_phase == SessionPhase.CLOSING:
            if len(session.techniques_used) > 0:
                criteria_met.append("technique_delivered")
            if len(session.skills_practiced) > 0:
                criteria_met.append("skill_practiced")
            duration_min = (datetime.now(timezone.utc) - session.started_at).total_seconds() / 60
            if duration_min >= 15:
                criteria_met.append("sufficient_work_time")
            if session.engagement_score < self._settings.min_engagement_score:
                criteria_met.append("engagement_declining")

            if len(criteria_met) >= 2 or duration_min >= 25:
                return {"allowed": True, "criteria_met": criteria_met}
            return {
                "allowed": False,
                "reason": "insufficient_working_criteria",
                "criteria_met": criteria_met,
            }

        if current == SessionPhase.CLOSING and target_phase == SessionPhase.POST_SESSION:
            if len(session.insights_gained) > 0 or len(session.messages) > 10:
                criteria_met.append("session_reviewed")
            if len(session.homework_assigned) > 0 or session.session_rating is not None:
                criteria_met.append("closure_activities_completed")

            if len(criteria_met) >= 1 or self._settings.enable_flexible_transitions:
                return {"allowed": True, "criteria_met": criteria_met}
            return {
                "allowed": False,
                "reason": "insufficient_closing_criteria",
                "criteria_met": criteria_met,
            }

        if session.current_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if target_phase == SessionPhase.CLOSING:
                return {
                    "allowed": True,
                    "criteria_met": ["crisis_override"],
                    "reason": "crisis_escalation",
                }

        if self._settings.enable_flexible_transitions:
            return {"allowed": True, "criteria_met": ["flexible_mode_enabled"]}

        return {"allowed": False, "reason": "invalid_transition_path"}

    def calculate_engagement_score(self, session: SessionState) -> float:
        """
        Calculate user engagement score for session.

        Args:
            session: Session state

        Returns:
            Engagement score between 0.0 and 1.0
        """
        score = 0.0

        if len(session.messages) > 0:
            user_messages = len([m for m in session.messages if m.get("role") == "user"])
            message_ratio = min(user_messages / max(len(session.messages), 1), 1.0)
            score += message_ratio * 0.3

        avg_length = sum(len(m.get("content", "")) for m in session.messages if m.get("role") == "user")
        avg_length = avg_length / max(len([m for m in session.messages if m.get("role") == "user"]), 1)
        if avg_length > 50:
            score += 0.2
        elif avg_length > 20:
            score += 0.1

        if len(session.skills_practiced) > 0:
            score += min(len(session.skills_practiced) * 0.15, 0.3)

        if len(session.insights_gained) > 0:
            score += min(len(session.insights_gained) * 0.1, 0.2)

        return min(score, 1.0)

    def delete_session(self, session_id: UUID) -> bool:
        """
        Delete session from active sessions.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            del self._active_sessions[session_id]
            logger.info("session_deleted", session_id=str(session_id), user_id=str(session.user_id))
            return True
        return False

    def get_session_metrics(self, session_id: UUID) -> dict[str, Any]:
        """
        Get detailed session metrics.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary of session metrics
        """
        session = self._active_sessions.get(session_id)
        if not session:
            return {}

        duration_sec = (datetime.now(timezone.utc) - session.started_at).total_seconds()
        engagement = self.calculate_engagement_score(session)

        return {
            "session_id": str(session_id),
            "duration_minutes": int(duration_sec / 60),
            "message_count": len(session.messages),
            "current_phase": session.current_phase.value,
            "techniques_used_count": len(session.techniques_used),
            "skills_practiced_count": len(session.skills_practiced),
            "engagement_score": round(engagement, 2),
            "risk_level": session.current_risk.value,
            "homework_assigned_count": len(session.homework_assigned),
        }

    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._active_sessions)

    def get_user_session_count(self, user_id: UUID) -> int:
        """Get total session count for user."""
        return self._user_session_counts.get(user_id, 0)
