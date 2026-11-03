"""
Solace-AI User Service - REST API Endpoints.

Provides user profile, preferences, consent and clinician endpoints. The identity endpoints
live in ``api_auth`` and the shared dependencies and DTOs in ``api_deps`` (WS-14.3).
Uses repository pattern for data access.

Architecture Layer: Presentation
Principles: Clean API Design, Dependency Injection, DTOs
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
import structlog

from .domain.entities import UserPreferences
from .domain.value_objects import ConsentType
from .domain.service import UserService
from .infrastructure.jwt_service import TokenPayload
# Re-exported for importers of ``src.api`` (``oauth.py`` line 36 imports TokenResponse from here
# until WS-08 re-points it; tests import get_current_user).
from .api_deps import (  # noqa: F401
    TokenResponse,
    UserResponse,
    get_current_user,
    get_jwt_service,
    get_password_service,
    get_session_manager,
    get_user_repository,
    get_user_service,
    user_to_response,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


# --- Request/Response Models ---


class UpdateProfileRequest(BaseModel):
    """Update profile request."""
    display_name: str | None = Field(default=None, max_length=100)
    timezone: str | None = Field(default=None)
    locale: str | None = Field(default=None)
    avatar_url: str | None = Field(default=None, max_length=500)
    bio: str | None = Field(default=None, max_length=500)


class PreferencesResponse(BaseModel):
    """User preferences response."""
    user_id: UUID = Field(..., description="User ID")
    notification_email: bool = Field(..., description="Email notifications")
    notification_sms: bool = Field(..., description="SMS notifications")
    notification_push: bool = Field(..., description="Push notifications")
    session_reminders: bool = Field(..., description="Session reminders")
    progress_updates: bool = Field(..., description="Progress updates")
    marketing_emails: bool = Field(..., description="Marketing emails")
    data_sharing_research: bool = Field(..., description="Research data sharing")
    data_sharing_improvement: bool = Field(..., description="Improvement data sharing")
    theme: str = Field(..., description="UI theme")
    language: str = Field(..., description="Language")
    accessibility_high_contrast: bool = Field(..., description="High contrast")
    accessibility_large_text: bool = Field(..., description="Large text")
    accessibility_screen_reader: bool = Field(..., description="Screen reader")


class UpdatePreferencesRequest(BaseModel):
    """Update preferences request."""
    notification_email: bool | None = None
    notification_sms: bool | None = None
    notification_push: bool | None = None
    session_reminders: bool | None = None
    progress_updates: bool | None = None
    marketing_emails: bool | None = None
    data_sharing_research: bool | None = None
    data_sharing_improvement: bool | None = None
    theme: str | None = None
    language: str | None = None
    accessibility_high_contrast: bool | None = None
    accessibility_large_text: bool | None = None
    accessibility_screen_reader: bool | None = None


class ConsentRequest(BaseModel):
    """Record consent request."""
    consent_type: str = Field(..., description="Type of consent")
    granted: bool = Field(..., description="Consent granted")
    version: str = Field(default="1.0", description="Consent version")


class ConsentResponse(BaseModel):
    """Consent record response."""
    consent_id: UUID = Field(..., description="Consent ID")
    user_id: UUID = Field(..., description="User ID")
    consent_type: str = Field(..., description="Consent type")
    granted: bool = Field(..., description="Consent granted")
    version: str = Field(..., description="Consent version")
    granted_at: datetime = Field(..., description="Grant timestamp")
    ip_address: str | None = Field(default=None, description="IP address")


# --- Helper Functions ---


def preferences_to_response(prefs: UserPreferences) -> PreferencesResponse:
    """Convert UserPreferences entity to PreferencesResponse DTO."""
    return PreferencesResponse(
        user_id=prefs.user_id,
        notification_email=prefs.notification_email,
        notification_sms=prefs.notification_sms,
        notification_push=prefs.notification_push,
        session_reminders=prefs.session_reminders,
        progress_updates=prefs.progress_updates,
        marketing_emails=prefs.marketing_emails,
        data_sharing_research=prefs.data_sharing_research,
        data_sharing_improvement=prefs.data_sharing_improvement,
        theme=prefs.theme,
        language=prefs.language,
        accessibility_high_contrast=prefs.accessibility_high_contrast,
        accessibility_large_text=prefs.accessibility_large_text,
        accessibility_screen_reader=prefs.accessibility_screen_reader,
    )


# --- User Profile Endpoints ---


@router.get(
    "/users/me",
    response_model=UserResponse,
    tags=["Users"],
)
async def get_current_user_profile(
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Get current user's profile."""
    user = await user_service.get_user(current_user.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user_to_response(user)


@router.put(
    "/users/me",
    response_model=UserResponse,
    tags=["Users"],
)
async def update_current_user_profile(
    request_data: UpdateProfileRequest,
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Update current user's profile."""
    updates = {k: v for k, v in request_data.model_dump().items() if v is not None}

    result = await user_service.update_user(current_user.user_id, **updates)

    if not result.success or not result.user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if result.error_code == "USER_NOT_FOUND" else status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to update profile",
        )

    logger.info("profile_updated", user_id=str(current_user.user_id), fields=list(updates.keys()))

    return user_to_response(result.user)


# --- Preferences Endpoints ---


@router.get(
    "/users/me/preferences",
    response_model=PreferencesResponse,
    tags=["Preferences"],
)
async def get_preferences(
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> PreferencesResponse:
    """Get current user's preferences."""
    preferences = await user_service.get_preferences(current_user.user_id)
    if not preferences:
        # Create default preferences
        result = await user_service.update_preferences(current_user.user_id)
        preferences = result.preferences

    if not preferences:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preferences not found",
        )

    return preferences_to_response(preferences)


@router.put(
    "/users/me/preferences",
    response_model=PreferencesResponse,
    tags=["Preferences"],
)
async def update_preferences(
    request_data: UpdatePreferencesRequest,
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> PreferencesResponse:
    """Update current user's preferences."""
    updates = {k: v for k, v in request_data.model_dump().items() if v is not None}

    result = await user_service.update_preferences(current_user.user_id, **updates)

    if not result.success or not result.preferences:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to update preferences",
        )

    logger.info("preferences_updated", user_id=str(current_user.user_id), fields=list(updates.keys()))

    return preferences_to_response(result.preferences)


# --- Consent Endpoints ---


@router.get(
    "/users/me/consent",
    response_model=list[ConsentResponse],
    tags=["Consent"],
)
async def get_consent_records(
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> list[ConsentResponse]:
    """Get current user's consent records."""
    records = await user_service.get_consent_records(current_user.user_id)
    return [
        ConsentResponse(
            consent_id=r.consent_id,
            user_id=r.user_id,
            consent_type=r.consent_type.value if hasattr(r.consent_type, 'value') else str(r.consent_type),
            granted=r.granted,
            version=r.version,
            granted_at=r.granted_at,
            ip_address=r.ip_address,
        )
        for r in records
    ]


@router.post(
    "/users/me/consent",
    response_model=ConsentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Consent"],
)
async def record_consent(
    request_data: ConsentRequest,
    request: Request,
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> ConsentResponse:
    """Record a consent decision."""
    # Parse and validate consent type
    try:
        consent_type = ConsentType(request_data.consent_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid consent type: {request_data.consent_type}. Valid types: {[t.value for t in ConsentType]}",
        )

    result = await user_service.record_consent(
        user_id=current_user.user_id,
        consent_type=consent_type,
        granted=request_data.granted,
        version=request_data.version,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    if not result.success or not result.consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to record consent",
        )

    logger.info(
        "consent_recorded",
        user_id=str(current_user.user_id),
        consent_type=request_data.consent_type,
        granted=request_data.granted,
    )

    return ConsentResponse(
        consent_id=result.consent.consent_id,
        user_id=result.consent.user_id,
        consent_type=result.consent.consent_type.value,
        granted=result.consent.granted,
        version=result.consent.version,
        granted_at=result.consent.granted_at,
        ip_address=result.consent.ip_address,
    )


# --- Admin Endpoints (Protected) ---


@router.get(
    # ':uuid' path convertor ensures this only matches real UUIDs, so static
    # sibling routes like /users/on-call-clinicians are not shadowed (REV-03).
    "/users/{user_id:uuid}",
    response_model=UserResponse,
    tags=["Admin"],
)
async def get_user_by_id(
    user_id: UUID,
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Get user by ID (admin only)."""
    # Check if admin or clinician
    if current_user.role not in ["admin", "clinician", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user_to_response(user)


# --- On-Call Clinician Endpoints ---


class OnCallClinicianResponse(BaseModel):
    """Response for on-call clinician lookup."""
    user_id: UUID
    display_name: str
    email: str
    phone_number: str | None = None


class OnCallListResponse(BaseModel):
    """Response containing list of on-call clinicians."""
    clinicians: list[OnCallClinicianResponse]
    count: int


class SetOnCallRequest(BaseModel):
    """Request to set on-call status."""
    is_on_call: bool = Field(..., description="Whether the clinician is on-call")


@router.get(
    "/users/on-call-clinicians",
    response_model=OnCallListResponse,
    tags=["Clinicians"],
)
async def get_on_call_clinicians(
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> OnCallListResponse:
    """
    Get list of currently on-call clinicians for crisis notifications.

    This endpoint is used by the notification service to route crisis alerts
    to available clinicians. Returns contact information for all clinicians
    who have marked themselves as on-call.

    Note: This endpoint requires service-level authentication in production.
    """
    clinicians = await user_service.get_on_call_clinicians()

    return OnCallListResponse(
        clinicians=[
            OnCallClinicianResponse(
                user_id=UUID(c["user_id"]),
                display_name=c["display_name"],
                email=c["email"],
                phone_number=c.get("phone_number"),
            )
            for c in clinicians
        ],
        count=len(clinicians),
    )


@router.put(
    "/users/{user_id}/on-call",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Clinicians"],
)
async def set_on_call_status(
    user_id: UUID,
    request: SetOnCallRequest,
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> None:
    """
    Set a clinician's on-call status.

    Only the clinician themselves or an admin can update on-call status.
    """
    # Check authorization: only self or admin
    if current_user.user_id != user_id:
        user = await user_service.get_user(current_user.user_id)
        if not user or not user.role.has_admin_access():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify another user's on-call status",
            )

    result = await user_service.set_on_call_status(user_id, request.is_on_call)

    if not result.success:
        if result.error_code == "USER_NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error or "User not found",
            )
        if result.error_code == "INVALID_ROLE":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error or "Only clinicians can be set on-call",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Failed to update on-call status",
        )

    logger.info(
        "on_call_status_updated",
        user_id=str(user_id),
        is_on_call=request.is_on_call,
    )
