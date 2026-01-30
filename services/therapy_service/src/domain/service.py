"""
Solace-AI Therapy Service - Therapeutic Intervention Orchestration.
Evidence-based hybrid (rules+LLM) therapy with CBT/DBT/ACT/MI/Mindfulness modalities.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone, timedelta
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import (
    SessionPhase, TherapyModality, SeverityLevel, RiskLevel,
    TechniqueDTO, HomeworkDTO, TreatmentPlanDTO, SessionStateDTO, SessionSummaryDTO,
)
from .models import SessionStartResult, TherapyMessageResult, SessionEndResult
from .response_generator import ResponseGenerator
from services.shared import ServiceBase

if TYPE_CHECKING:
    from .technique_selector import TechniqueSelector
    from .session_manager import SessionManager
    from services.shared.infrastructure import UnifiedLLMClient
    from ..events import EventBus
    from ..infrastructure.context_assembler import ContextAssembler

logger = structlog.get_logger(__name__)


class TherapyOrchestratorSettings(BaseSettings):
    """Therapy orchestrator configuration."""
    enable_safety_checks: bool = Field(default=True)
    crisis_keywords: list[str] = Field(default_factory=lambda: ["suicide", "kill myself", "end it all", "hurt myself"])
    max_processing_time_ms: int = Field(default=15000)
    enable_homework_assignment: bool = Field(default=True)
    session_timeout_minutes: int = Field(default=60)
    enable_outcome_tracking: bool = Field(default=True)
    # Enable context assembly from Memory Service
    enable_context_assembly: bool = Field(default=True)
    context_assembly_timeout_ms: int = Field(default=5000)
    model_config = SettingsConfigDict(env_prefix="THERAPY_ORCHESTRATOR_", env_file=".env", extra="ignore")


class TherapyOrchestrator(ServiceBase):
    """
    Main orchestrator for therapeutic interventions.

    Coordinates session management, technique selection, intervention delivery,
    safety monitoring, and outcome tracking across evidence-based modalities.
    """

    def __init__(
        self,
        settings: TherapyOrchestratorSettings | None = None,
        technique_selector: TechniqueSelector | None = None,
        session_manager: SessionManager | None = None,
        llm_client: UnifiedLLMClient | None = None,
        event_bus: "EventBus | None" = None,
        context_assembler: "ContextAssembler | None" = None,
    ) -> None:
        self._settings = settings or TherapyOrchestratorSettings()
        self._technique_selector = technique_selector
        self._session_manager = session_manager
        self._llm_client = llm_client
        self._event_bus = event_bus
        self._context_assembler = context_assembler
        self._treatment_plans: dict[UUID, TreatmentPlanDTO] = {}
        self._initialized = False
        self._stats = {
            "sessions_started": 0,
            "sessions_ended": 0,
            "messages_processed": 0,
            "techniques_applied": 0,
            "crisis_interventions": 0,
            "homework_assigned": 0,
            "llm_responses": 0,
            "llm_fallbacks": 0,
            "context_assembly_success": 0,
            "context_assembly_fallback": 0,
        }

    async def initialize(self) -> None:
        """Initialize the therapy orchestrator."""
        logger.info("therapy_orchestrator_initializing")
        self._initialized = True
        logger.info("therapy_orchestrator_initialized", settings={
            "safety_checks": self._settings.enable_safety_checks,
            "homework_enabled": self._settings.enable_homework_assignment,
            "llm_enabled": self._llm_client is not None,
        })

    async def shutdown(self) -> None:
        """Shutdown the therapy orchestrator."""
        logger.info("therapy_orchestrator_shutting_down", stats=self._stats)
        self._initialized = False

    async def start_session(self, user_id: UUID, treatment_plan_id: UUID, context: dict[str, Any]) -> SessionStartResult:
        """Start new therapy session."""
        self._stats["sessions_started"] += 1
        start_time = time.perf_counter()
        if not self._session_manager:
            return SessionStartResult.failure("Session manager not initialized", "SESSION_MANAGER_NOT_INITIALIZED")
        treatment_plan = self._treatment_plans.get(treatment_plan_id)
        if not treatment_plan:
            treatment_plan = self._create_mock_treatment_plan(user_id, treatment_plan_id, context)
            self._treatment_plans[treatment_plan_id] = treatment_plan
        session_id, session = self._session_manager.create_session(user_id=user_id, treatment_plan_id=treatment_plan_id, context=context)
        initial_message = ResponseGenerator.generate_initial_message(session.session_number, treatment_plan)
        suggested_agenda = ResponseGenerator.generate_suggested_agenda(treatment_plan, session.session_number)
        self._session_manager.transition_phase(session_id=session_id, target_phase=SessionPhase.OPENING, trigger="session_start")
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("session_start_completed", session_id=str(session_id), duration_ms=duration_ms)

        # Publish session started event
        if self._event_bus:
            from ..events import SessionStartedEvent
            event = SessionStartedEvent.create(
                session_id=session_id,
                user_id=user_id,
                treatment_plan_id=treatment_plan_id,
                session_number=session.session_number,
            )
            await self._event_bus.publish(event)

        return SessionStartResult(
            session_id=session_id, session_number=session.session_number, initial_message=initial_message,
            suggested_agenda=suggested_agenda, loaded_context=True,
        )

    async def process_message(
        self,
        session_id: UUID,
        user_id: UUID,
        message: str,
        conversation_history: list[dict[str, str]],
    ) -> TherapyMessageResult:
        """
        Process user message in therapy session.

        Args:
            session_id: Session identifier
            user_id: User identifier
            message: User message
            conversation_history: Conversation context

        Returns:
            TherapyMessageResult with response and state updates
        """
        self._stats["messages_processed"] += 1
        start_time = time.perf_counter()

        if not self._session_manager:
            return TherapyMessageResult.failure("Session manager not initialized", "SESSION_MANAGER_NOT_INITIALIZED")

        session = self._session_manager.get_session(session_id)
        if not session:
            return TherapyMessageResult.failure(f"Session {session_id} not found", "SESSION_NOT_FOUND")

        if session.user_id != user_id:
            return TherapyMessageResult.failure("User ID mismatch for session", "USER_ID_MISMATCH")

        self._session_manager.update_state(session_id, {"messages": {"role": "user", "content": message}})

        # Assemble context from Memory Service if enabled
        enriched_history = conversation_history
        memory_context = None
        if self._settings.enable_context_assembly and self._context_assembler:
            try:
                memory_context = await self._context_assembler.assemble(
                    user_id=user_id,
                    session_id=session_id,
                    current_message=message,
                )
                enriched_history = await self._context_assembler.enrich_conversation_history(
                    user_id=user_id,
                    session_id=session_id,
                    conversation_history=conversation_history,
                    current_message=message,
                )
                self._stats["context_assembly_success"] += 1
                logger.debug(
                    "context_assembled_for_message",
                    session_id=str(session_id),
                    total_tokens=memory_context.total_tokens if memory_context else 0,
                    assembly_time_ms=memory_context.assembly_time_ms if memory_context else 0,
                )
            except Exception as e:
                logger.warning("context_assembly_failed", error=str(e), session_id=str(session_id))
                self._stats["context_assembly_fallback"] += 1
                # Continue with original conversation history

        safety_result = await self._check_safety(message, session)
        if safety_result["crisis_detected"]:
            return await self._handle_crisis(session_id, safety_result)

        treatment_plan = self._treatment_plans.get(session.treatment_plan_id)
        if not treatment_plan:
            return TherapyMessageResult.failure("Treatment plan not found", "TREATMENT_PLAN_NOT_FOUND")

        technique_result = await self._select_and_apply_technique(
            session=session,
            message=message,
            treatment_plan=treatment_plan,
            conversation_history=enriched_history,
        )

        response_text = await self._generate_response(
            session=session,
            message=message,
            technique=technique_result.get("technique"),
            treatment_plan=treatment_plan,
            conversation_history=enriched_history,
        )

        homework = None
        if self._settings.enable_homework_assignment and technique_result.get("technique"):
            if technique_result["technique"].requires_homework and session.current_phase == SessionPhase.CLOSING:
                homework = self._create_homework_assignment(technique_result["technique"], session_id)
                if homework:
                    self._session_manager.update_state(session_id, {"homework_assigned": homework})
                    self._stats["homework_assigned"] += 1

        self._maybe_transition_phase(session_id, session)

        engagement = self._session_manager.calculate_engagement_score(session)
        self._session_manager.update_state(session_id, {"engagement_score": engagement})

        session = self._session_manager.get_session(session_id)

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        return TherapyMessageResult(
            response_text=response_text,
            current_phase=session.current_phase if session else SessionPhase.WORKING,
            technique_applied=technique_result.get("technique"),
            homework_assigned=[homework] if homework else [],
            safety_alerts=safety_result.get("alerts", []),
            next_steps=ResponseGenerator.generate_next_steps(session),
            processing_time_ms=processing_time_ms,
        )

    async def end_session(self, session_id: UUID, user_id: UUID, generate_summary: bool = True) -> SessionEndResult:
        """End therapy session."""
        self._stats["sessions_ended"] += 1
        if not self._session_manager:
            return SessionEndResult.failure("Session manager not initialized", "SESSION_MANAGER_NOT_INITIALIZED")
        session = self._session_manager.get_session(session_id)
        if not session:
            return SessionEndResult.failure(f"Session {session_id} not found", "SESSION_NOT_FOUND")
        if session.user_id != user_id:
            return SessionEndResult.failure("User ID mismatch for session", "USER_ID_MISMATCH")
        self._session_manager.transition_phase(session_id=session_id, target_phase=SessionPhase.POST_SESSION, trigger="session_end")
        duration = datetime.now(timezone.utc) - session.started_at
        duration_minutes = int(duration.total_seconds() / 60)
        summary = ResponseGenerator.generate_session_summary(session, duration_minutes) if generate_summary else None
        recommendations = ResponseGenerator.generate_recommendations(session)

        # Publish session ended event before deleting session
        if self._event_bus:
            from ..events import SessionEndedEvent
            techniques_count = len(session.techniques_used) if hasattr(session, 'techniques_used') else 0
            skills_practiced = session.skills_practiced if hasattr(session, 'skills_practiced') else []
            event = SessionEndedEvent.create(
                session_id=session_id,
                user_id=user_id,
                duration_minutes=duration_minutes,
                techniques_count=techniques_count,
                skills_practiced=skills_practiced,
            )
            await self._event_bus.publish(event)

        self._session_manager.delete_session(session_id)
        logger.info("session_end_completed", session_id=str(session_id), duration_minutes=duration_minutes)
        return SessionEndResult(summary=summary, duration_minutes=duration_minutes, recommendations=recommendations)

    async def _check_safety(self, message: str, session: Any) -> dict[str, Any]:
        """Check message for safety concerns."""
        if not self._settings.enable_safety_checks:
            return {"crisis_detected": False, "alerts": []}
        message_lower = message.lower()
        alerts = []
        crisis_detected = False
        for keyword in self._settings.crisis_keywords:
            if keyword in message_lower:
                alerts.append(f"Crisis keyword detected: {keyword}")
                crisis_detected = True
                self._stats["crisis_interventions"] += 1
        if "harm" in message_lower or "danger" in message_lower:
            alerts.append("Potential harm language detected")
            crisis_detected = True
        if crisis_detected:
            session.current_risk = RiskLevel.HIGH
            self._session_manager.update_state(session.session_id, {"current_risk": RiskLevel.HIGH})
            for alert in alerts:
                self._session_manager.update_state(session.session_id, {"safety_flags": alert})
        return {"crisis_detected": crisis_detected, "alerts": alerts, "risk_level": session.current_risk}

    async def _handle_crisis(self, session_id: UUID, safety_result: dict[str, Any]) -> TherapyMessageResult:
        """Handle crisis situation."""
        logger.warning("crisis_intervention_triggered", session_id=str(session_id), alerts=safety_result["alerts"])
        self._session_manager.transition_phase(session_id, SessionPhase.CLOSING, trigger="crisis_protocol")
        return TherapyMessageResult(
            response_text=ResponseGenerator.generate_crisis_response(safety_result["alerts"]),
            current_phase=SessionPhase.CLOSING, technique_applied=None, safety_alerts=safety_result["alerts"],
            next_steps=["Contact crisis services", "Ensure immediate safety", "Consider emergency services"],
            processing_time_ms=50,
        )

    async def _generate_response(
        self,
        session: Any,
        message: str,
        technique: TechniqueDTO | None,
        treatment_plan: TreatmentPlanDTO,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """
        Generate therapeutic response using LLM with fallback to templates.

        Uses Portkey AI Gateway for resilient LLM access with automatic fallbacks.
        Falls back to template-based generation if LLM is unavailable.
        """
        if self._llm_client and self._llm_client.is_available:
            try:
                from ..infrastructure import get_therapy_prompt
                system_prompt = get_therapy_prompt(
                    treatment_plan.primary_modality.value,
                    is_crisis=session.current_risk in [RiskLevel.HIGH, RiskLevel.IMMINENT],
                )
                context = None
                if technique:
                    context = f"Current Technique: {technique.name}\n{technique.description}"
                session_phase_str = session.current_phase.value if hasattr(session.current_phase, "value") else str(session.current_phase)
                context = f"{context}\n\nSession Phase: {session_phase_str}" if context else f"Session Phase: {session_phase_str}"
                response = await self._llm_client.generate(
                    system_prompt=system_prompt,
                    user_message=message,
                    conversation_history=conversation_history,
                    context=context,
                    service_name="therapy_service",
                    metadata={
                        "session_id": str(session.session_id),
                        "user_id": str(session.user_id),
                        "modality": treatment_plan.primary_modality.value,
                        "session_number": session.session_number,
                    },
                )
                if response:
                    self._stats["llm_responses"] += 1
                    logger.debug("llm_response_generated", session_id=str(session.session_id))
                    return response
            except Exception as e:
                logger.warning("llm_generation_failed_using_fallback", error=str(e))
                self._stats["llm_fallbacks"] += 1
        return ResponseGenerator.generate_therapeutic_response(
            session=session,
            message=message,
            technique=technique,
            conversation_history=conversation_history,
        )

    async def _select_and_apply_technique(
        self, session: Any, message: str, treatment_plan: TreatmentPlanDTO, conversation_history: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Select and apply appropriate technique."""
        if not self._technique_selector:
            return {"technique": None, "reasoning": "Technique selector not available"}
        if session.current_phase not in [SessionPhase.WORKING, SessionPhase.CLOSING]:
            return {"technique": None, "reasoning": "Not in intervention phase"}
        user_context = {"current_risk": session.current_risk, "contraindications": [], "preferences": {}, "personality": {}}
        selection = await self._technique_selector.select_technique(
            user_id=session.user_id, diagnosis=treatment_plan.primary_diagnosis, severity=treatment_plan.severity,
            modality=treatment_plan.primary_modality, session_phase=session.current_phase, user_context=user_context,
            treatment_plan={"current_phase": treatment_plan.current_phase, "skills_acquired": treatment_plan.skills_acquired},
        )
        if selection["selected"]:
            self._stats["techniques_applied"] += 1
            self._session_manager.update_state(session.session_id, {"techniques_used": selection["selected"]})
        return {"technique": selection["selected"], "alternatives": selection["alternatives"], "reasoning": selection["reasoning"]}

    def _maybe_transition_phase(self, session_id: UUID, session: Any) -> None:
        """Attempt automatic phase transition if criteria met."""
        if not self._session_manager:
            return
        message_count = len(session.messages)
        duration_sec = (datetime.now(timezone.utc) - session.started_at).total_seconds()
        target_phase = None
        if session.current_phase == SessionPhase.OPENING and message_count >= 6:
            target_phase = SessionPhase.WORKING
        elif session.current_phase == SessionPhase.WORKING and duration_sec >= 900:
            target_phase = SessionPhase.CLOSING
        if target_phase:
            self._session_manager.transition_phase(session_id, target_phase, trigger="automatic")

    def _create_homework_assignment(self, technique: TechniqueDTO, session_id: UUID) -> HomeworkDTO | None:
        """Create homework assignment from technique."""
        if not technique.requires_homework:
            return None
        descriptions = {
            "Thought Record": "Complete a thought record when you notice a strong negative emotion. Identify the situation, thought, emotion, and evidence.",
            "Behavioral Activation": "Schedule and complete 3 pleasant activities this week. Rate your mood before and after each activity.",
            "Mindfulness of Breath": "Practice 5 minutes of mindful breathing each day. Use the technique we practiced in session.",
            "STOP Skill": "Use the STOP skill when you notice distress. Practice at least 3 times this week.",
            "Values Clarification": "Reflect on your top 3 values and write one way you can honor each value this week.",
        }
        return HomeworkDTO(
            homework_id=uuid4(), title=f"Practice: {technique.name}",
            description=descriptions.get(technique.name, f"Practice the {technique.name} technique daily."),
            technique_id=technique.technique_id, due_date=None, completed=False,
        )

    def _create_mock_treatment_plan(
        self,
        user_id: UUID,
        plan_id: UUID,
        context: dict[str, Any],
    ) -> TreatmentPlanDTO:
        """Create mock treatment plan for demonstration."""
        return TreatmentPlanDTO(
            plan_id=plan_id,
            user_id=user_id,
            primary_diagnosis=context.get("diagnosis", "Depression"),
            severity=SeverityLevel.MODERATE,
            primary_modality=TherapyModality.CBT,
            adjunct_modalities=[TherapyModality.MINDFULNESS],
            current_phase=1,
            sessions_completed=0,
            skills_acquired=[],
        )

    async def get_session_state(self, session_id: UUID) -> SessionStateDTO | None:
        """Get current session state as DTO."""
        if not self._session_manager:
            return None

        session = self._session_manager.get_session(session_id)
        if not session:
            return None

        return SessionStateDTO(
            session_id=session.session_id,
            user_id=session.user_id,
            treatment_plan_id=session.treatment_plan_id,
            session_number=session.session_number,
            current_phase=session.current_phase,
            mood_rating=session.mood_rating,
            agenda_items=session.agenda_items,
            topics_covered=session.topics_covered,
            skills_practiced=session.skills_practiced,
            current_risk=session.current_risk,
            engagement_score=session.engagement_score,
        )

    async def get_treatment_plan(self, session_id: UUID) -> TreatmentPlanDTO | None:
        """Get treatment plan for session."""
        if not self._session_manager:
            return None

        session = self._session_manager.get_session(session_id)
        if not session:
            return None

        return self._treatment_plans.get(session.treatment_plan_id)

    async def assign_homework(self, session_id: UUID, homework: HomeworkDTO) -> bool:
        """Manually assign homework to session."""
        if not self._session_manager:
            return False

        session = self._session_manager.get_session(session_id)
        if not session:
            return False

        self._session_manager.update_state(session_id, {"homework_assigned": homework})
        return True

    async def list_techniques(self, modality: TherapyModality | None = None) -> list[TechniqueDTO]:
        """List available techniques."""
        if not self._technique_selector:
            return []
        return self._technique_selector.get_techniques_by_modality(modality)

    async def delete_session(self, session_id: UUID) -> None:
        """Delete session data."""
        if self._session_manager:
            self._session_manager.delete_session(session_id)

    async def get_user_progress(self, user_id: UUID) -> dict[str, Any]:
        """
        Get user's therapy progress summary.

        Returns session statistics, techniques used, and engagement metrics.

        Parameters
        ----------
        user_id : UUID
            The user identifier.

        Returns
        -------
        dict[str, Any]
            Progress summary including:
            - total_sessions: Total completed sessions
            - completed_homework: Number of completed homework assignments
            - techniques_used: List of techniques practiced
            - total_minutes: Total therapy time in minutes
            - streak_days: Consecutive days with sessions
            - last_session: ISO timestamp of last session
            - mood_trend: Overall mood trajectory
            - engagement_score: Average engagement across sessions
        """
        # Get session history from session manager
        completed_sessions = 0
        total_minutes = 0
        techniques_used: set[str] = set()
        last_session = None
        total_engagement = 0.0
        completed_homework = 0
        session_dates: list[datetime] = []
        mood_ratings: list[int] = []

        if self._session_manager:
            # Get user's session history
            user_sessions = self._session_manager.get_user_sessions(user_id)
            completed_sessions = len(user_sessions)

            for session in user_sessions:
                # Calculate session duration
                if hasattr(session, "start_time") and hasattr(session, "end_time"):
                    if session.end_time and session.start_time:
                        duration = (session.end_time - session.start_time).total_seconds() / 60
                        total_minutes += int(duration)

                # Collect techniques used
                if hasattr(session, "skills_practiced"):
                    techniques_used.update(session.skills_practiced or [])

                # Track last session and collect dates for streak calculation
                if hasattr(session, "start_time") and session.start_time:
                    session_dates.append(session.start_time)
                    if last_session is None or session.start_time > last_session:
                        last_session = session.start_time

                # Accumulate engagement scores
                if hasattr(session, "engagement_score"):
                    total_engagement += session.engagement_score or 0.0

                # Count completed homework
                if hasattr(session, "homework_assigned"):
                    for hw in session.homework_assigned or []:
                        if hasattr(hw, "completed") and hw.completed:
                            completed_homework += 1

                # Collect mood ratings for trend calculation
                if hasattr(session, "mood_rating") and session.mood_rating is not None:
                    mood_ratings.append(session.mood_rating)

        avg_engagement = total_engagement / max(completed_sessions, 1)

        # Calculate streak days (consecutive days with sessions)
        streak_days = self._calculate_streak_days(session_dates)

        # Calculate mood trend
        mood_trend = self._calculate_mood_trend(mood_ratings)

        return {
            "user_id": str(user_id),
            "total_sessions": completed_sessions,
            "completed_homework": completed_homework,
            "techniques_used": list(techniques_used),
            "total_minutes": total_minutes,
            "streak_days": streak_days,
            "last_session": last_session.isoformat() if last_session else None,
            "mood_trend": mood_trend,
            "engagement_score": round(avg_engagement, 2),
        }

    def _calculate_streak_days(self, session_dates: list[datetime]) -> int:
        """Calculate consecutive days with sessions ending at today or yesterday."""
        if not session_dates:
            return 0

        # Get unique dates (just the date part)
        unique_dates = sorted(set(d.date() for d in session_dates), reverse=True)
        if not unique_dates:
            return 0

        # Check if streak is current (includes today or yesterday)
        today = datetime.now(timezone.utc).date()
        if unique_dates[0] < today - timedelta(days=1):
            return 0  # Streak broken

        # Count consecutive days
        streak = 1
        for i in range(1, len(unique_dates)):
            if unique_dates[i - 1] - unique_dates[i] == timedelta(days=1):
                streak += 1
            else:
                break

        return streak

    def _calculate_mood_trend(self, mood_ratings: list[int]) -> str:
        """Calculate mood trend from recent ratings."""
        if len(mood_ratings) < 2:
            return "stable"

        # Use last 5 ratings for trend
        recent = mood_ratings[-5:]
        if len(recent) < 2:
            return "stable"

        # Calculate simple linear regression slope
        n = len(recent)
        sum_x = sum(range(n))
        sum_y = sum(recent)
        sum_xy = sum(i * r for i, r in enumerate(recent))
        sum_xx = sum(i * i for i in range(n))

        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return "stable"

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Classify trend based on slope
        if slope > 0.3:
            return "improving"
        elif slope < -0.3:
            return "declining"
        else:
            return "stable"

    async def get_status(self) -> dict[str, Any]:
        """Get service status."""
        active_sessions = self._session_manager.get_active_session_count() if self._session_manager else 0
        return {
            "status": "operational" if self._initialized else "initializing",
            "initialized": self._initialized,
            "statistics": self._stats,
            "active_sessions": active_sessions,
            "treatment_plans_loaded": len(self._treatment_plans),
        }

    @property
    def stats(self) -> dict[str, int]:
        """Get service statistics counters."""
        return self._stats
