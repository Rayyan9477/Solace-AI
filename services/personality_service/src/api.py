"""
Solace-AI Personality Service - REST API Endpoints.
Personality detection, profile management, and style adaptation endpoints.
"""
from __future__ import annotations
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
import structlog

from .schemas import (
    DetectPersonalityRequest, DetectPersonalityResponse,
    GetStyleRequest, GetStyleResponse,
    AdaptResponseRequest, AdaptResponseResponse,
    GetProfileResponse, UpdateProfileRequest, UpdateProfileResponse,
    ErrorResponse, ErrorDetail,
)
from .domain.service import PersonalityOrchestrator

from solace_infrastructure.gdpr import ErasureIncomplete

from solace_security.middleware import (
    AuthenticatedUser,
    AuthenticatedService,
    get_current_user,
    get_current_service,
)
from solace_security import Role
from solace_common.request_context import reset_current_user_id, set_current_user_id

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["personality"])


def get_orchestrator(request: Request) -> PersonalityOrchestrator:
    """Get personality orchestrator from app state."""
    if not hasattr(request.app.state, "personality_orchestrator"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Personality service not initialized",
        )
    return request.app.state.personality_orchestrator


OrchestratorDep = Annotated[PersonalityOrchestrator, Depends(get_orchestrator)]


@router.post(
    "/detect",
    response_model=DetectPersonalityResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Detect personality traits from text",
    description="Analyze text to detect Big Five (OCEAN) personality traits using ensemble methods.",
)
async def detect_personality(
    request: DetectPersonalityRequest,
    orchestrator: OrchestratorDep,
    service: AuthenticatedService = Depends(get_current_service),
) -> DetectPersonalityResponse:
    """Detect personality traits from user text."""
    logger.info("personality_detection_request", user_id=str(request.user_id), text_length=len(request.text))
    # C.2 RLS: this is a service-to-service call (get_current_service sets no user
    # GUC), but it reads/writes personality_* rows scoped to request.user_id. Publish
    # that user into the request context so the DB layer sets app.current_user_id and
    # RLS admits the row (INSERT WITH CHECK and SELECT/UPDATE USING).
    # P2-7: capture the token and reset in finally so the request-scoped GUC does not
    # rely solely on asyncio task boundaries (a reused task/thread would leak it).
    token = set_current_user_id(str(request.user_id))
    try:
        response = await orchestrator.detect_personality(request)
        return response
    except ValueError as e:
        logger.warning("personality_detection_validation_error", error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("personality_detection_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Personality detection failed")
    finally:
        reset_current_user_id(token)


@router.post(
    "/style",
    response_model=GetStyleResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Profile not found"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Get communication style parameters",
    description="Get personality-adapted communication style parameters for a user.",
)
async def get_style(
    request: GetStyleRequest,
    orchestrator: OrchestratorDep,
    service: AuthenticatedService = Depends(get_current_service),
) -> GetStyleResponse:
    """Get style parameters for user."""
    logger.info("style_request", user_id=str(request.user_id))
    # C.2 RLS: set the per-user GUC so RLS admits this user's personality rows
    # (SELECT is filtered by the same policy as writes).
    # P2-7: reset the request-scoped GUC in finally (see /detect).
    token = set_current_user_id(str(request.user_id))
    try:
        response = await orchestrator.get_style(request)
        return response
    except Exception as e:
        logger.error("style_retrieval_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Style retrieval failed")
    finally:
        reset_current_user_id(token)


@router.post(
    "/adapt",
    response_model=AdaptResponseResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Adapt response to personality",
    description="Adapt a response to match user personality and communication preferences.",
)
async def adapt_response(
    request: AdaptResponseRequest,
    orchestrator: OrchestratorDep,
    service: AuthenticatedService = Depends(get_current_service),
) -> AdaptResponseResponse:
    """Adapt response content to user personality."""
    logger.info("adapt_request", user_id=str(request.user_id), base_length=len(request.base_response))
    # C.2 RLS: set the per-user GUC so RLS admits this user's personality rows.
    # P2-7: reset the request-scoped GUC in finally (see /detect).
    token = set_current_user_id(str(request.user_id))
    try:
        response = await orchestrator.adapt_response(request)
        return response
    except ValueError as e:
        logger.warning("adapt_validation_error", error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("adapt_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Response adaptation failed")
    finally:
        reset_current_user_id(token)


@router.get(
    "/profile/{user_id}",
    response_model=GetProfileResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Profile not found"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Get user personality profile",
    description="Retrieve the personality profile for a user including OCEAN scores and style parameters.",
)
async def get_profile(
    user_id: UUID,
    orchestrator: OrchestratorDep,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> GetProfileResponse:
    """Get user personality profile."""
    logger.info("profile_request", user_id=str(user_id))
    # Enforce ownership: user can only view their own profile
    if str(user_id) != str(current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another user's profile",
        )
    profile = await orchestrator.get_profile(user_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile not found for user {user_id}",
        )
    return GetProfileResponse(profile=profile, exists=True)


@router.post(
    "/profile/update",
    response_model=UpdateProfileResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Update personality profile",
    description="Update a user's personality profile with new assessment data.",
)
async def update_profile(
    request: UpdateProfileRequest,
    orchestrator: OrchestratorDep,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> UpdateProfileResponse:
    """Update user personality profile."""
    logger.info("profile_update_request", user_id=str(request.user_id), source=request.source.value)
    # Enforce ownership: user can only update their own profile
    if str(request.user_id) != str(current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot update another user's profile",
        )
    current_profile = await orchestrator.get_profile(request.user_id)
    previous_version = current_profile.version if current_profile else 0
    # B-3: honor the submitted OCEAN scores directly instead of synthesizing a
    # placeholder text (which fell below min_text_length and dragged toward 0.5).
    updated_profile = await orchestrator.update_profile_from_scores(
        request.user_id, request.ocean_scores, request.source
    )
    if updated_profile is None:
        updated_profile = await orchestrator.get_profile(request.user_id)
    changed_traits = []
    if current_profile and updated_profile:
        for trait in updated_profile.dominant_traits:
            if trait not in current_profile.dominant_traits:
                changed_traits.append(trait)
    return UpdateProfileResponse(
        user_id=request.user_id,
        previous_version=previous_version,
        new_version=updated_profile.version if updated_profile else 1,
        changed_traits=changed_traits,
        update_reason="external_assessment",
    )


@router.delete(
    "/user/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={
        403: {"model": ErrorResponse, "description": "Cannot delete another user's data"},
        500: {"model": ErrorResponse, "description": "Erasure incomplete"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Erase all personality data for a user",
    description="GDPR right-to-erasure (REV-17): purge the user's in-memory profile "
                "cache and all persistent personality records. Self-or-admin only.",
)
async def delete_user_data(
    user_id: UUID,
    orchestrator: OrchestratorDep,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> Response:
    """Delete all personality data for a user (GDPR compliance, REV-17)."""
    # Only allow self-deletion or admin deletion (mirrors memory/diagnosis services).
    if str(user_id) != str(current_user.user_id) and Role.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete other user's data without admin role",
        )
    logger.info(
        "user_data_deletion_requested",
        user_id=str(user_id),
        authenticated_user=str(current_user.user_id),
    )
    try:
        await orchestrator.delete_user_data(user_id)
    except ErasureIncomplete as e:
        # Fail loud: a partial erasure is never reported as success.
        logger.error("user_data_erasure_incomplete", user_id=str(user_id), failed=e.failed)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User data erasure incomplete",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get service status",
    description="Retrieve personality service status and operational statistics.",
)
async def get_status(
    orchestrator: OrchestratorDep,
    _service: AuthenticatedService = Depends(get_current_service),
) -> dict:
    """Get service status and statistics.

    REV-09: gated behind service auth (no unauthenticated operational leak).
    """
    return await orchestrator.get_status()


@router.get(
    "/traits",
    status_code=status.HTTP_200_OK,
    summary="Get available personality traits",
    description="List all available Big Five personality traits with descriptions.",
)
async def list_traits() -> dict:
    """List available personality traits."""
    return {
        "traits": [
            {
                "name": "openness",
                "description": "Intellectual curiosity, creativity, preference for novelty",
                "high_indicators": ["curious", "imaginative", "open to new experiences"],
                "low_indicators": ["practical", "conventional", "prefer routine"],
            },
            {
                "name": "conscientiousness",
                "description": "Organization, dependability, self-discipline",
                "high_indicators": ["organized", "reliable", "goal-oriented"],
                "low_indicators": ["flexible", "spontaneous", "less structured"],
            },
            {
                "name": "extraversion",
                "description": "Sociability, assertiveness, positive emotions",
                "high_indicators": ["outgoing", "energetic", "talkative"],
                "low_indicators": ["reserved", "introspective", "prefer solitude"],
            },
            {
                "name": "agreeableness",
                "description": "Cooperation, trust, empathy toward others",
                "high_indicators": ["cooperative", "trusting", "helpful"],
                "low_indicators": ["competitive", "skeptical", "challenging"],
            },
            {
                "name": "neuroticism",
                "description": "Emotional instability, anxiety, moodiness",
                "high_indicators": ["anxious", "moody", "emotionally reactive"],
                "low_indicators": ["calm", "emotionally stable", "resilient"],
            },
        ],
        "model": "Big Five (OCEAN)",
        "score_range": {"min": 0.0, "max": 1.0},
    }


@router.get(
    "/styles",
    status_code=status.HTTP_200_OK,
    summary="Get available communication styles",
    description="List all available communication style types with descriptions.",
)
async def list_styles() -> dict:
    """List available communication styles."""
    return {
        "styles": [
            {"name": "analytical", "description": "Detail-oriented, logical, systematic approach"},
            {"name": "expressive", "description": "Enthusiastic, emotional, relationship-focused"},
            {"name": "driver", "description": "Direct, results-oriented, decisive"},
            {"name": "amiable", "description": "Supportive, patient, diplomatic"},
            {"name": "balanced", "description": "Adaptable, moderate across all dimensions"},
        ],
        "parameters": [
            {"name": "warmth", "range": [0.0, 1.0], "description": "Emotional warmth level"},
            {"name": "structure", "range": [0.0, 1.0], "description": "Response organization level"},
            {"name": "complexity", "range": [0.0, 1.0], "description": "Conceptual abstraction level"},
            {"name": "directness", "range": [0.0, 1.0], "description": "Communication directness"},
            {"name": "energy", "range": [0.0, 1.0], "description": "Response energy level"},
            {"name": "validation_level", "range": [0.0, 1.0], "description": "Emotional validation emphasis"},
        ],
    }
