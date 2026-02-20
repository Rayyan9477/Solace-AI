"""
Solace-AI Personality Service - REST API Endpoints.
Personality detection, profile management, and style adaptation endpoints.
"""
from __future__ import annotations
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, status
import structlog

from .schemas import (
    DetectPersonalityRequest, DetectPersonalityResponse,
    GetStyleRequest, GetStyleResponse,
    AdaptResponseRequest, AdaptResponseResponse,
    GetProfileResponse, UpdateProfileRequest, UpdateProfileResponse,
    ErrorResponse, ErrorDetail,
)
from .domain.service import PersonalityOrchestrator

from solace_security.middleware import (
    AuthenticatedUser,
    AuthenticatedService,
    get_current_user,
    get_current_service,
)

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
    try:
        response = await orchestrator.detect_personality(request)
        return response
    except ValueError as e:
        logger.warning("personality_detection_validation_error", error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("personality_detection_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Personality detection failed")


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
    try:
        response = await orchestrator.get_style(request)
        return response
    except Exception as e:
        logger.error("style_retrieval_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Style retrieval failed")


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
    try:
        response = await orchestrator.adapt_response(request)
        return response
    except ValueError as e:
        logger.warning("adapt_validation_error", error=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("adapt_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Response adaptation failed")


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
    detect_request = DetectPersonalityRequest(
        user_id=request.user_id,
        text="Profile update from external assessment",
        sources=[request.source],
    )
    await orchestrator.detect_personality(detect_request)
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


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get service status",
    description="Retrieve personality service status and operational statistics.",
)
async def get_status(orchestrator: OrchestratorDep) -> dict:
    """Get service status and statistics."""
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
