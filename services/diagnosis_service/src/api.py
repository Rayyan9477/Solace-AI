"""
Solace-AI Diagnosis Service API - Diagnosis and assessment endpoints.
Provides endpoints for AMIE-inspired 4-step reasoning diagnostic operations.
"""
from __future__ import annotations
from typing import Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
import structlog

from .domain.service import DiagnosisService
from .schemas import (
    DiagnosisPhase, SeverityLevel,
    AssessmentRequest, AssessmentResponse,
    SymptomExtractionRequest, SymptomExtractionResponse,
    DifferentialRequest, DifferentialResponse,
    SessionStartRequest, SessionStartResponse,
    SessionEndRequest, SessionEndResponse,
    DiagnosisHistoryRequest, DiagnosisHistoryResponse,
    SymptomDTO, HypothesisDTO, DifferentialDTO,
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
    from solace_security import Role, Permission
except ImportError:
    # Fallback for testing/development without security library
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class AuthenticatedUser:
        user_id: UUID
        email: str
        roles: list[str]
        permissions: list[str]

    @dataclass
    class AuthenticatedService:
        service_id: str
        service_name: str
        permissions: list[str]

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

    class Role:
        ADMIN = "admin"
        CLINICIAN = "clinician"
        USER = "user"

    class Permission:
        DELETE_USER_DATA = "delete:user_data"

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["diagnosis"])


def get_diagnosis_service(request: Request) -> DiagnosisService:
    """Dependency to get diagnosis service from app state."""
    if not hasattr(request.app.state, "diagnosis_service"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Diagnosis service not initialized")
    return request.app.state.diagnosis_service


@router.post("/assess", response_model=AssessmentResponse, status_code=status.HTTP_200_OK)
async def perform_assessment(
    request: AssessmentRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> AssessmentResponse:
    """Perform full 4-step Chain-of-Reasoning diagnostic assessment."""
    # Verify user can only access their own data (unless clinician/admin)
    if request.user_id != current_user.user_id and "clinician" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access other user's assessments")
    logger.info("assessment_requested", user_id=str(request.user_id),
                session_id=str(request.session_id), phase=request.current_phase.value,
                authenticated_user=str(current_user.user_id))
    result = await diagnosis_service.assess(
        user_id=request.user_id,
        session_id=request.session_id,
        message=request.message,
        conversation_history=request.conversation_history,
        existing_symptoms=request.existing_symptoms,
        current_phase=request.current_phase,
        current_differential=request.current_differential,
        user_context=request.user_context,
    )
    return AssessmentResponse(
        assessment_id=result.assessment_id,
        user_id=request.user_id,
        session_id=request.session_id,
        phase=result.phase,
        extracted_symptoms=result.extracted_symptoms,
        differential=result.differential,
        reasoning_chain=result.reasoning_chain,
        next_question=result.next_question,
        response_text=result.response_text,
        confidence_score=result.confidence_score,
        safety_flags=result.safety_flags,
        processing_time_ms=result.processing_time_ms,
    )


@router.post("/extract-symptoms", response_model=SymptomExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_symptoms(
    request: SymptomExtractionRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> SymptomExtractionResponse:
    """Extract symptoms from conversation without full assessment."""
    # Verify user can only access their own data (unless clinician/admin)
    if request.user_id != current_user.user_id and "clinician" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access other user's symptoms")
    logger.info("symptom_extraction_requested", user_id=str(request.user_id),
                session_id=str(request.session_id), authenticated_user=str(current_user.user_id))
    result = await diagnosis_service.extract_symptoms(
        user_id=request.user_id,
        session_id=request.session_id,
        message=request.message,
        conversation_history=request.conversation_history,
        existing_symptoms=request.existing_symptoms,
    )
    return SymptomExtractionResponse(
        extraction_id=result.extraction_id,
        user_id=request.user_id,
        extracted_symptoms=result.extracted_symptoms,
        updated_symptoms=result.updated_symptoms,
        temporal_info=result.temporal_info,
        contextual_factors=result.contextual_factors,
        risk_indicators=result.risk_indicators,
        extraction_time_ms=result.extraction_time_ms,
    )


@router.post("/differential", response_model=DifferentialResponse, status_code=status.HTTP_200_OK)
async def generate_differential(
    request: DifferentialRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> DifferentialResponse:
    """Generate differential diagnosis from symptoms."""
    # Verify user can only access their own data (unless clinician/admin)
    if request.user_id != current_user.user_id and "clinician" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access other user's differential")
    logger.info("differential_requested", user_id=str(request.user_id),
                session_id=str(request.session_id), symptom_count=len(request.symptoms),
                authenticated_user=str(current_user.user_id))
    result = await diagnosis_service.generate_differential(
        user_id=request.user_id,
        session_id=request.session_id,
        symptoms=request.symptoms,
        user_history=request.user_history,
        current_differential=request.current_differential,
    )
    return DifferentialResponse(
        differential_id=result.differential_id,
        user_id=request.user_id,
        differential=result.differential,
        hitop_scores=result.hitop_scores,
        recommended_questions=result.recommended_questions,
        generation_time_ms=result.generation_time_ms,
    )


@router.post("/session/start", response_model=SessionStartResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    request: SessionStartRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> SessionStartResponse:
    """Start a new diagnosis session."""
    # Verify user can only start sessions for themselves (unless clinician/admin)
    if request.user_id != current_user.user_id and "clinician" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot start sessions for other users")
    logger.info("session_start_requested", user_id=str(request.user_id),
                session_type=request.session_type, authenticated_user=str(current_user.user_id))
    result = await diagnosis_service.start_session(
        user_id=request.user_id,
        session_type=request.session_type,
        initial_context=request.initial_context,
        previous_session_id=request.previous_session_id,
    )
    return SessionStartResponse(
        session_id=result.session_id,
        user_id=request.user_id,
        session_number=result.session_number,
        initial_phase=result.initial_phase,
        greeting=result.greeting,
        loaded_context=result.loaded_context,
    )


@router.post("/session/end", response_model=SessionEndResponse, status_code=status.HTTP_200_OK)
async def end_session(
    request: SessionEndRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> SessionEndResponse:
    """End a diagnosis session and optionally generate summary."""
    # Verify user can only end their own sessions (unless clinician/admin)
    if request.user_id != current_user.user_id and "clinician" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot end other user's sessions")
    logger.info("session_end_requested", user_id=str(request.user_id),
                session_id=str(request.session_id), authenticated_user=str(current_user.user_id))
    result = await diagnosis_service.end_session(
        user_id=request.user_id,
        session_id=request.session_id,
        generate_summary=request.generate_summary,
    )
    return SessionEndResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        duration_minutes=result.duration_minutes,
        messages_exchanged=result.messages_exchanged,
        final_differential=result.final_differential,
        summary=result.summary,
        recommendations=result.recommendations,
    )


@router.post("/history", response_model=DiagnosisHistoryResponse, status_code=status.HTTP_200_OK)
async def get_diagnosis_history(
    request: DiagnosisHistoryRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> DiagnosisHistoryResponse:
    """Get diagnosis history for longitudinal tracking."""
    # Verify user can only access their own history (unless clinician/admin)
    if request.user_id != current_user.user_id and "clinician" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access other user's history")
    logger.info("history_requested", user_id=str(request.user_id), limit=request.limit,
                authenticated_user=str(current_user.user_id))
    result = await diagnosis_service.get_history(
        user_id=request.user_id,
        limit=request.limit,
        include_symptoms=request.include_symptoms,
        include_differentials=request.include_differentials,
    )
    return DiagnosisHistoryResponse(
        user_id=request.user_id,
        sessions=result.sessions,
        symptom_trends=result.symptom_trends,
        longitudinal_patterns=result.longitudinal_patterns,
    )


@router.get("/session/{session_id}/state", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def get_session_state(
    session_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> dict[str, Any]:
    """Get current state of a diagnosis session."""
    logger.info("session_state_requested", session_id=str(session_id),
                authenticated_user=str(current_user.user_id))
    result = await diagnosis_service.get_session_state(session_id=session_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    # Verify user owns this session (unless clinician/admin)
    session_user_id = result.get("user_id")
    if session_user_id and str(session_user_id) != str(current_user.user_id) and "clinician" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access other user's session")
    return result


@router.post("/challenge/{session_id}", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def challenge_hypothesis(
    session_id: UUID,
    hypothesis_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> dict[str, Any]:
    """Trigger Devil's Advocate challenge for a specific hypothesis."""
    logger.info("challenge_requested", session_id=str(session_id), hypothesis_id=str(hypothesis_id),
                authenticated_user=str(current_user.user_id))
    result = await diagnosis_service.challenge_hypothesis(
        session_id=session_id,
        hypothesis_id=hypothesis_id,
    )
    return {
        "session_id": str(session_id),
        "hypothesis_id": str(hypothesis_id),
        "challenges": result.challenges,
        "alternative_hypotheses": result.alternatives,
        "counter_questions": result.counter_questions,
        "bias_flags": result.bias_flags,
    }


@router.get("/status", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def get_service_status(
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> dict[str, Any]:
    """Get diagnosis service status and statistics."""
    return await diagnosis_service.get_status()


@router.delete("/user/{user_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_user_data(
    user_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    diagnosis_service: DiagnosisService = Depends(get_diagnosis_service),
) -> Response:
    """Delete all diagnosis data for a user (GDPR compliance)."""
    # Only allow self-deletion or admin deletion
    if user_id != current_user.user_id and "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete other user's data without admin role")
    logger.info("user_data_deletion_requested", user_id=str(user_id),
                authenticated_user=str(current_user.user_id))
    await diagnosis_service.delete_user_data(user_id=user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
