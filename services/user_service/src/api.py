"""
Solace-AI User Service API - User management and profile endpoints.
Provides CRUD operations for users, preferences, sessions, and consent management.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
import structlog

from .domain.service import UserService

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["users"])


class UserRole(str, Enum):
    """User roles in the system."""
    USER = "user"
    PREMIUM = "premium"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class AccountStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class CreateUserRequest(BaseModel):
    """Request to create a new user."""
    email: str = Field(..., min_length=5, max_length=255, pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password")
    display_name: str = Field(..., min_length=1, max_length=100, description="Display name")
    timezone: str = Field(default="UTC", description="User timezone")
    locale: str = Field(default="en-US", description="User locale")


class UpdateUserRequest(BaseModel):
    """Request to update user profile."""
    display_name: str | None = Field(default=None, max_length=100, description="Display name")
    timezone: str | None = Field(default=None, description="User timezone")
    locale: str | None = Field(default=None, description="User locale")
    avatar_url: str | None = Field(default=None, max_length=500, description="Avatar URL")
    bio: str | None = Field(default=None, max_length=500, description="User bio")


class UserResponse(BaseModel):
    """User profile response."""
    user_id: UUID = Field(..., description="User identifier")
    email: str = Field(..., description="User email")
    display_name: str = Field(..., description="Display name")
    role: UserRole = Field(..., description="User role")
    status: AccountStatus = Field(..., description="Account status")
    timezone: str = Field(..., description="User timezone")
    locale: str = Field(..., description="User locale")
    avatar_url: str | None = Field(default=None, description="Avatar URL")
    bio: str | None = Field(default=None, description="User bio")
    email_verified: bool = Field(..., description="Email verification status")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: datetime | None = Field(default=None, description="Last login timestamp")


class UserPreferencesDTO(BaseModel):
    """User preferences data transfer object."""
    notification_email: bool = Field(default=True, description="Email notifications enabled")
    notification_sms: bool = Field(default=False, description="SMS notifications enabled")
    notification_push: bool = Field(default=True, description="Push notifications enabled")
    notification_channels: list[NotificationChannel] = Field(default_factory=lambda: [NotificationChannel.EMAIL, NotificationChannel.IN_APP])
    session_reminders: bool = Field(default=True, description="Session reminder notifications")
    progress_updates: bool = Field(default=True, description="Progress update notifications")
    marketing_emails: bool = Field(default=False, description="Marketing email opt-in")
    data_sharing_research: bool = Field(default=False, description="Data sharing for research")
    data_sharing_improvement: bool = Field(default=True, description="Data sharing for service improvement")
    theme: str = Field(default="system", description="UI theme preference")
    language: str = Field(default="en", description="Preferred language")
    accessibility_high_contrast: bool = Field(default=False, description="High contrast mode")
    accessibility_large_text: bool = Field(default=False, description="Large text mode")
    accessibility_screen_reader: bool = Field(default=False, description="Screen reader optimization")


class UpdatePreferencesRequest(BaseModel):
    """Request to update user preferences."""
    notification_email: bool | None = None
    notification_sms: bool | None = None
    notification_push: bool | None = None
    notification_channels: list[NotificationChannel] | None = None
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


class ConsentRecord(BaseModel):
    """Consent record for data processing."""
    consent_id: UUID = Field(default_factory=uuid4, description="Consent record identifier")
    consent_type: str = Field(..., description="Type of consent")
    granted: bool = Field(..., description="Whether consent was granted")
    version: str = Field(..., description="Consent version")
    granted_at: datetime = Field(..., description="When consent was granted")
    ip_address: str | None = Field(default=None, description="IP address at time of consent")
    user_agent: str | None = Field(default=None, description="User agent at time of consent")


class ConsentRequest(BaseModel):
    """Request to record consent."""
    consent_type: str = Field(..., description="Type of consent")
    granted: bool = Field(..., description="Whether consent is granted")
    version: str = Field(default="1.0", description="Consent version")


class UserSessionDTO(BaseModel):
    """User session information."""
    session_id: UUID = Field(..., description="Session identifier")
    user_id: UUID = Field(..., description="User identifier")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    ip_address: str | None = Field(default=None, description="Client IP address")
    user_agent: str | None = Field(default=None, description="Client user agent")
    is_active: bool = Field(default=True, description="Session active status")


class UserProgressDTO(BaseModel):
    """User progress summary."""
    user_id: UUID = Field(..., description="User identifier")
    total_sessions: int = Field(default=0, description="Total therapy sessions")
    completed_assessments: int = Field(default=0, description="Completed assessments")
    streak_days: int = Field(default=0, description="Current streak in days")
    total_minutes: int = Field(default=0, description="Total minutes in therapy")
    last_session: datetime | None = Field(default=None, description="Last session timestamp")
    mood_trend: str = Field(default="stable", description="Overall mood trend")
    engagement_score: float = Field(default=0.0, ge=0, le=1, description="Engagement score 0-1")


class PasswordChangeRequest(BaseModel):
    """Request to change password."""
    current_password: str = Field(..., min_length=8, description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")


class DeleteAccountRequest(BaseModel):
    """Request to delete account."""
    password: str = Field(..., description="Password for verification")
    confirmation: str = Field(..., pattern="^DELETE$", description="Type DELETE to confirm")
    reason: str | None = Field(default=None, max_length=500, description="Reason for deletion")


def get_user_service(request: Request) -> UserService:
    """Dependency to get user service from app state."""
    if not hasattr(request.app.state, "user_service"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="User service not initialized")
    return request.app.state.user_service


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: CreateUserRequest,
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Create a new user account."""
    logger.info("create_user_requested", email=request.email)
    result = await user_service.create_user(
        email=request.email,
        password=request.password,
        display_name=request.display_name,
        timezone=request.timezone,
        locale=request.locale,
    )
    if result.error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.error)
    return UserResponse(**result.user.model_dump())


@router.get("/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_user(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Get user profile by ID."""
    logger.info("get_user_requested", user_id=str(user_id))
    result = await user_service.get_user(user_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserResponse(**result.model_dump())


@router.put("/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def update_user(
    user_id: UUID,
    request: UpdateUserRequest,
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Update user profile."""
    logger.info("update_user_requested", user_id=str(user_id))
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    result = await user_service.update_user(user_id, updates)
    if result.error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.error)
    if not result.user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserResponse(**result.user.model_dump())


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    request: DeleteAccountRequest,
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Delete user account (GDPR right to erasure)."""
    logger.info("delete_user_requested", user_id=str(user_id))
    result = await user_service.delete_user(user_id, request.password, request.reason)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.error or "Deletion failed")


@router.get("/{user_id}/preferences", response_model=UserPreferencesDTO, status_code=status.HTTP_200_OK)
async def get_preferences(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service),
) -> UserPreferencesDTO:
    """Get user preferences."""
    logger.info("get_preferences_requested", user_id=str(user_id))
    result = await user_service.get_preferences(user_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserPreferencesDTO(**result.model_dump())


@router.put("/{user_id}/preferences", response_model=UserPreferencesDTO, status_code=status.HTTP_200_OK)
async def update_preferences(
    user_id: UUID,
    request: UpdatePreferencesRequest,
    user_service: UserService = Depends(get_user_service),
) -> UserPreferencesDTO:
    """Update user preferences."""
    logger.info("update_preferences_requested", user_id=str(user_id))
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    result = await user_service.update_preferences(user_id, updates)
    if result.error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.error)
    if not result.preferences:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserPreferencesDTO(**result.preferences.model_dump())


@router.get("/{user_id}/progress", response_model=UserProgressDTO, status_code=status.HTTP_200_OK)
async def get_progress(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service),
) -> UserProgressDTO:
    """Get user progress summary."""
    logger.info("get_progress_requested", user_id=str(user_id))
    result = await user_service.get_progress(user_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserProgressDTO(**result.model_dump())


@router.get("/{user_id}/consent", response_model=list[ConsentRecord], status_code=status.HTTP_200_OK)
async def get_consent_records(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service),
) -> list[ConsentRecord]:
    """Get user consent records."""
    logger.info("get_consent_requested", user_id=str(user_id))
    records = await user_service.get_consent_records(user_id)
    return [ConsentRecord(**r.model_dump()) for r in records]


@router.post("/{user_id}/consent", response_model=ConsentRecord, status_code=status.HTTP_201_CREATED)
async def record_consent(
    user_id: UUID,
    request: ConsentRequest,
    http_request: Request,
    user_service: UserService = Depends(get_user_service),
) -> ConsentRecord:
    """Record user consent."""
    logger.info("record_consent_requested", user_id=str(user_id), consent_type=request.consent_type)
    ip_address = http_request.client.host if http_request.client else None
    user_agent = http_request.headers.get("User-Agent")
    result = await user_service.record_consent(
        user_id=user_id,
        consent_type=request.consent_type,
        granted=request.granted,
        version=request.version,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    if result.error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.error)
    return ConsentRecord(**result.consent.model_dump())


@router.get("/{user_id}/sessions", response_model=list[UserSessionDTO], status_code=status.HTTP_200_OK)
async def get_sessions(
    user_id: UUID,
    active_only: bool = True,
    user_service: UserService = Depends(get_user_service),
) -> list[UserSessionDTO]:
    """Get user sessions."""
    logger.info("get_sessions_requested", user_id=str(user_id), active_only=active_only)
    sessions = await user_service.get_user_sessions(user_id, active_only)
    return [UserSessionDTO(**s.model_dump()) for s in sessions]


@router.delete("/{user_id}/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_session(
    user_id: UUID,
    session_id: UUID,
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Revoke a specific session."""
    logger.info("revoke_session_requested", user_id=str(user_id), session_id=str(session_id))
    result = await user_service.revoke_session(user_id, session_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")


@router.post("/{user_id}/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    user_id: UUID,
    request: PasswordChangeRequest,
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Change user password."""
    logger.info("change_password_requested", user_id=str(user_id))
    result = await user_service.change_password(user_id, request.current_password, request.new_password)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.error or "Password change failed")


@router.post("/{user_id}/verify-email", status_code=status.HTTP_204_NO_CONTENT)
async def verify_email(
    user_id: UUID,
    token: str,
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Verify user email address."""
    logger.info("verify_email_requested", user_id=str(user_id))
    result = await user_service.verify_email(user_id, token)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.error or "Email verification failed")


@router.get("/status", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def get_service_status(user_service: UserService = Depends(get_user_service)) -> dict[str, Any]:
    """Get user service status and statistics."""
    return await user_service.get_status()
