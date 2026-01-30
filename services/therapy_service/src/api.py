"""
Solace-AI Therapy Service API - Therapy session and intervention endpoints.
Provides endpoints for evidence-based therapeutic interventions with CBT/DBT/ACT/MI modalities.
"""
from __future__ import annotations
from typing import Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
import structlog

from .domain.service import TherapyOrchestrator
from .schemas import (
    SessionPhase, TherapyModality, TechniqueDTO, HomeworkDTO,
    SessionStartRequest, SessionStartResponse,
    MessageRequest, TherapyResponse,
    SessionEndRequest, SessionSummaryDTO,
    SessionStateDTO, TreatmentPlanDTO,
)

# Authentication dependencies from shared security library
try:
    from solace_security.middleware import (
        AuthenticatedUser,
        AuthenticatedService,
        get_current_user,
        get_current_user_optional,
        get_current_service,
        require_roles,
        require_permissions,
    )
except ImportError:
    from dataclasses import dataclass as _dataclass
    from typing import Optional

    @_dataclass
    class AuthenticatedUser:
        user_id: str
        token_type: str = "access"
        roles: list = None
        permissions: list = None
        session_id: str | None = None
        metadata: dict = None
        def has_role(self, role: str) -> bool:
            return role in (self.roles or [])
        def has_permission(self, perm: str) -> bool:
            return perm in (self.permissions or [])

    @_dataclass
    class AuthenticatedService:
        service_name: str
        permissions: list = None

    async def get_current_user() -> AuthenticatedUser:
        raise HTTPException(status_code=501, detail="Authentication not configured")

    async def get_current_user_optional() -> Optional[AuthenticatedUser]:
        return None

    async def get_current_service() -> AuthenticatedService:
        raise HTTPException(status_code=501, detail="Service auth not configured")

    def require_roles(*roles):
        return get_current_user

    def require_permissions(*perms):
        return get_current_user


logger = structlog.get_logger(__name__)
router = APIRouter(tags=["therapy"])


def get_therapy_orchestrator(request: Request) -> TherapyOrchestrator:
    """Dependency to get therapy orchestrator from app state."""
    if not hasattr(request.app.state, "therapy_orchestrator"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Therapy orchestrator not initialized"
        )
    return request.app.state.therapy_orchestrator


@router.post("/sessions/start", response_model=SessionStartResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    request: SessionStartRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> SessionStartResponse:
    """
    Start a new therapy session.

    Initializes session state, loads treatment plan, performs safety checks,
    and generates initial greeting with suggested agenda.
    """
    logger.info(
        "session_start_requested",
        user_id=str(request.user_id),
        treatment_plan_id=str(request.treatment_plan_id)
    )

    try:
        result = await orchestrator.start_session(
            user_id=request.user_id,
            treatment_plan_id=request.treatment_plan_id,
            context=request.context,
        )

        logger.info(
            "session_started",
            session_id=str(result.session_id),
            session_number=result.session_number
        )

        return SessionStartResponse(
            session_id=result.session_id,
            user_id=request.user_id,
            treatment_plan_id=request.treatment_plan_id,
            session_number=result.session_number,
            current_phase=SessionPhase.OPENING,
            initial_message=result.initial_message,
            suggested_agenda=result.suggested_agenda,
        )

    except ValueError as e:
        logger.warning("session_start_failed", error=str(e), user_id=str(request.user_id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("session_start_error", error=str(e), user_id=str(request.user_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start therapy session"
        )


@router.post("/sessions/{session_id}/message", response_model=TherapyResponse, status_code=status.HTTP_200_OK)
async def process_message(
    session_id: UUID,
    request: MessageRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> TherapyResponse:
    """
    Process user message in therapy session.

    Executes full therapeutic intervention pipeline: safety check, technique selection,
    intervention delivery, and response generation with appropriate modality.
    """
    if session_id != request.session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID mismatch"
        )

    logger.info(
        "message_received",
        session_id=str(session_id),
        user_id=str(request.user_id),
        message_length=len(request.message)
    )

    try:
        result = await orchestrator.process_message(
            session_id=session_id,
            user_id=request.user_id,
            message=request.message,
            conversation_history=request.conversation_history,
        )

        if result.safety_alerts:
            logger.warning(
                "safety_alerts_triggered",
                session_id=str(session_id),
                alerts=result.safety_alerts
            )

        logger.info(
            "message_processed",
            session_id=str(session_id),
            phase=result.current_phase.value,
            technique=result.technique_applied.name if result.technique_applied else None,
            processing_time_ms=result.processing_time_ms
        )

        return TherapyResponse(
            session_id=session_id,
            user_id=request.user_id,
            response_text=result.response_text,
            current_phase=result.current_phase,
            technique_applied=result.technique_applied,
            homework_assigned=result.homework_assigned,
            safety_alerts=result.safety_alerts,
            next_steps=result.next_steps,
            processing_time_ms=result.processing_time_ms,
        )

    except ValueError as e:
        logger.warning("message_processing_failed", error=str(e), session_id=str(session_id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("message_processing_error", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process therapy message"
        )


@router.post("/sessions/{session_id}/end", response_model=SessionSummaryDTO, status_code=status.HTTP_200_OK)
async def end_session(
    session_id: UUID,
    request: SessionEndRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> SessionSummaryDTO:
    """
    End therapy session and generate summary.

    Transitions to post-session phase, generates session summary with techniques used,
    skills practiced, insights, and homework assignments.
    """
    if session_id != request.session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID mismatch"
        )

    logger.info(
        "session_end_requested",
        session_id=str(session_id),
        user_id=str(request.user_id),
        generate_summary=request.generate_summary
    )

    try:
        result = await orchestrator.end_session(
            session_id=session_id,
            user_id=request.user_id,
            generate_summary=request.generate_summary,
        )

        if not result.summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or already ended"
            )

        logger.info(
            "session_ended",
            session_id=str(session_id),
            duration_minutes=result.duration_minutes,
            techniques_count=len(result.summary.techniques_used)
        )

        return result.summary

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("session_end_failed", error=str(e), session_id=str(session_id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("session_end_error", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end therapy session"
        )


@router.get("/sessions/{session_id}/state", response_model=SessionStateDTO, status_code=status.HTTP_200_OK)
async def get_session_state(
    session_id: UUID,
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> SessionStateDTO:
    """
    Get current state of therapy session.

    Returns current phase, mood rating, agenda items, topics covered,
    skills practiced, and risk level.
    """
    logger.info("session_state_requested", session_id=str(session_id))

    try:
        state = await orchestrator.get_session_state(session_id=session_id)

        if state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        return state

    except HTTPException:
        raise
    except Exception as e:
        logger.error("session_state_error", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session state"
        )


@router.get("/sessions/{session_id}/treatment-plan", response_model=TreatmentPlanDTO, status_code=status.HTTP_200_OK)
async def get_treatment_plan(
    session_id: UUID,
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> TreatmentPlanDTO:
    """
    Get treatment plan for active session.

    Returns treatment plan including diagnosis, modality, phase,
    sessions completed, and skills acquired.
    """
    logger.info("treatment_plan_requested", session_id=str(session_id))

    try:
        plan = await orchestrator.get_treatment_plan(session_id=session_id)

        if plan is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Treatment plan not found for session"
            )

        return plan

    except HTTPException:
        raise
    except Exception as e:
        logger.error("treatment_plan_error", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve treatment plan"
        )


@router.post("/sessions/{session_id}/homework", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
async def assign_homework(
    session_id: UUID,
    homework: HomeworkDTO,
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> dict[str, Any]:
    """
    Assign homework to session.

    Manually assign homework assignment with technique reference,
    description, and due date.
    """
    logger.info("homework_assignment_requested", session_id=str(session_id), homework_id=str(homework.homework_id))

    try:
        success = await orchestrator.assign_homework(
            session_id=session_id,
            homework=homework,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        logger.info("homework_assigned", session_id=str(session_id), homework_id=str(homework.homework_id))

        return {
            "session_id": str(session_id),
            "homework_id": str(homework.homework_id),
            "assigned": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("homework_assignment_error", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign homework"
        )


@router.get("/techniques", response_model=list[TechniqueDTO], status_code=status.HTTP_200_OK)
async def list_techniques(
    modality: TherapyModality | None = None,
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> list[TechniqueDTO]:
    """
    List available therapeutic techniques.

    Optionally filter by modality (CBT, DBT, ACT, MI, MINDFULNESS).
    Returns technique metadata including contraindications.
    """
    logger.info("techniques_list_requested", modality=modality.value if modality else None)

    try:
        techniques = await orchestrator.list_techniques(modality=modality)
        return techniques

    except Exception as e:
        logger.error("techniques_list_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve techniques"
        )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_session(
    session_id: UUID,
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> Response:
    """
    Delete therapy session data.

    Removes session from active sessions and clears session state.
    For compliance and privacy purposes.
    """
    logger.info("session_deletion_requested", session_id=str(session_id))

    try:
        await orchestrator.delete_session(session_id=session_id)
        logger.info("session_deleted", session_id=str(session_id))
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except Exception as e:
        logger.error("session_deletion_error", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )


@router.get("/status", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def get_service_status(
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> dict[str, Any]:
    """
    Get therapy service status and statistics.

    Returns operational status, active sessions, and processing statistics.
    """
    return await orchestrator.get_status()


@router.get("/users/{user_id}/progress", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def get_user_progress(
    user_id: UUID,
    orchestrator: TherapyOrchestrator = Depends(get_therapy_orchestrator),
) -> dict[str, Any]:
    """
    Get user's therapy progress summary.

    Returns session counts, techniques used, total minutes, and engagement metrics.
    Used by User Service to aggregate user progress data.
    """
    logger.info("user_progress_requested", user_id=str(user_id))

    try:
        progress = await orchestrator.get_user_progress(user_id=user_id)
        return progress

    except Exception as e:
        logger.error("user_progress_error", error=str(e), user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user progress"
        )
