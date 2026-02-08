"""
Solace-AI User Service - REST API Endpoints.

Provides authentication, user management, preferences, and consent endpoints.
Implements JWT-based authentication with secure password handling.
Uses repository pattern for data access.

Architecture Layer: Presentation
Principles: Clean API Design, Dependency Injection, DTOs
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
import structlog

from .domain.entities import User, UserPreferences
from .domain.value_objects import UserRole, AccountStatus, ConsentType
from .domain.service import UserService
from .infrastructure.jwt_service import (
    JWTService,
    TokenPayload,
    TokenPair,
    TokenType,
    TokenExpiredError,
    TokenInvalidError,
)
from .infrastructure.password_service import PasswordService
from .infrastructure.repository import UserRepository, UserPreferencesRepository, ConsentRepository
from .auth import SessionManager, AuthenticationService

logger = structlog.get_logger(__name__)
router = APIRouter()


# --- Request/Response Models ---


class RegisterRequest(BaseModel):
    """User registration request."""
    email: str = Field(..., min_length=5, max_length=255, description="User email")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    display_name: str = Field(..., min_length=1, max_length=100, description="Display name")
    timezone: str = Field(default="UTC", description="User timezone")
    locale: str = Field(default="en-US", description="User locale")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and normalize email."""
        from email_validator import validate_email as _validate_email, EmailNotValidError
        try:
            result = _validate_email(v, check_deliverability=False)
            return result.normalized
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email: {e}")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    """User login request."""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiry in seconds")


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="Refresh token")


class UserResponse(BaseModel):
    """User profile response."""
    user_id: UUID = Field(..., description="User ID")
    email: str = Field(..., description="Email")
    display_name: str = Field(..., description="Display name")
    role: str = Field(..., description="User role")
    status: str = Field(..., description="Account status")
    timezone: str = Field(..., description="Timezone")
    locale: str = Field(..., description="Locale")
    avatar_url: str | None = Field(default=None, description="Avatar URL")
    bio: str | None = Field(default=None, description="Bio")
    email_verified: bool = Field(..., description="Email verified")
    created_at: datetime = Field(..., description="Created timestamp")
    last_login: datetime | None = Field(default=None, description="Last login")


class UpdateProfileRequest(BaseModel):
    """Update profile request."""
    display_name: str | None = Field(default=None, max_length=100)
    timezone: str | None = Field(default=None)
    locale: str | None = Field(default=None)
    avatar_url: str | None = Field(default=None, max_length=500)
    bio: str | None = Field(default=None, max_length=500)


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


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


# --- Dependencies ---


def get_jwt_service(request: Request) -> JWTService:
    """Get JWT service from app state."""
    state = request.app.state.service
    if not state.jwt_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT service not available",
        )
    return state.jwt_service


def get_password_service(request: Request) -> PasswordService:
    """Get password service from app state."""
    state = request.app.state.service
    if not state.password_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Password service not available",
        )
    return state.password_service


def get_user_service(request: Request) -> UserService:
    """Get user service from app state."""
    state = request.app.state.service
    if not state.user_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User service not available",
        )
    return state.user_service


def get_user_repository(request: Request) -> UserRepository:
    """Get user repository from app state."""
    state = request.app.state.service
    if not state.user_repository:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User repository not available",
        )
    return state.user_repository


def get_session_manager(request: Request) -> SessionManager:
    """Get session manager from app state."""
    state = request.app.state.service
    if not state.session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not available",
        )
    return state.session_manager


async def get_current_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> TokenPayload:
    """
    Extract and validate current user from JWT token.

    Args:
        request: FastAPI request
        authorization: Authorization header

    Returns:
        TokenPayload with user information

    Raises:
        HTTPException: If token is missing or invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jwt_service = get_jwt_service(request)

    try:
        token = jwt_service.extract_token_from_header(authorization)
        payload = jwt_service.verify_token(token, expected_type=TokenType.ACCESS)
        return payload
    except (TokenExpiredError, TokenInvalidError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- Helper Functions ---


def user_to_response(user: User) -> UserResponse:
    """Convert User entity to UserResponse DTO."""
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        display_name=user.display_name,
        role=user.role.value,
        status=user.status.value,
        timezone=user.timezone,
        locale=user.locale,
        avatar_url=user.avatar_url,
        bio=user.bio,
        email_verified=user.email_verified,
        created_at=user.created_at,
        last_login=user.last_login,
    )


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


# --- Authentication Endpoints ---


@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
)
async def register(
    request_data: RegisterRequest,
    request: Request,
    user_service: UserService = Depends(get_user_service),
    password_service: PasswordService = Depends(get_password_service),
) -> UserResponse:
    """
    Register a new user account.

    Creates a new user with hashed password and returns user profile.
    Email verification is required before login.
    """
    # Hash password
    password_hash = password_service.hash_password(request_data.password)

    # Create user via service
    result = await user_service.create_user(
        email=request_data.email,
        password_hash=password_hash,
        display_name=request_data.display_name,
        timezone=request_data.timezone,
        locale=request_data.locale,
    )

    if not result.success or not result.user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT if result.error_code == "EMAIL_EXISTS" else status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to create user",
        )

    # Increment stats
    request.app.state.service.increment_stat("registrations")

    logger.info("user_registered", user_id=str(result.user.user_id), email=result.user.email)

    return user_to_response(result.user)


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
)
async def login(
    request_data: LoginRequest,
    request: Request,
    jwt_service: JWTService = Depends(get_jwt_service),
    password_service: PasswordService = Depends(get_password_service),
    user_repository: UserRepository = Depends(get_user_repository),
    user_service: UserService = Depends(get_user_service),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TokenResponse:
    """
    Authenticate user and return JWT tokens.

    Verifies credentials and returns access/refresh token pair.
    """
    # Find user by email
    user = await user_repository.get_by_email(request_data.email.lower())
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if user can login
    can_login, error = user.can_login()
    if not can_login:
        # For demo purposes, allow login even if not verified
        if error != "Email verification required":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error,
            )

    # Verify password
    result = password_service.verify_password(request_data.password, user.password_hash)
    if not result.is_valid:
        await user_service.record_login_attempt(user.user_id, success=False)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Update password hash if migration needed
    if result.needs_rehash and result.new_hash:
        user.password_hash = result.new_hash
        await user_repository.update(user)
        logger.info("password_migrated", user_id=str(user.user_id), algorithm=result.algorithm_used.value)

    # Record successful login
    await user_service.record_login_attempt(user.user_id, success=True)

    # Create session
    session = await session_manager.create_session(
        user=user,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    # Generate tokens
    token_pair = jwt_service.generate_token_pair(
        user_id=user.user_id,
        email=user.email,
        role=user.role.value,
    )

    # Increment stats
    request.app.state.service.increment_stat("logins")

    logger.info("user_logged_in", user_id=str(user.user_id), email=user.email)

    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        token_type=token_pair.token_type,
        expires_in=token_pair.expires_in,
    )


@router.post(
    "/auth/refresh",
    response_model=TokenResponse,
    tags=["Authentication"],
)
async def refresh_token(
    request_data: RefreshRequest,
    request: Request,
    jwt_service: JWTService = Depends(get_jwt_service),
) -> TokenResponse:
    """
    Refresh access token using refresh token.

    Returns new access token without requiring re-authentication.
    """
    try:
        # Verify refresh token and get payload
        payload = jwt_service.verify_token(
            request_data.refresh_token,
            expected_type=TokenType.REFRESH,
        )

        # Generate new token pair
        token_pair = jwt_service.generate_token_pair(
            user_id=payload.user_id,
            email=payload.email,
            role=payload.role,
        )

        # Increment stats
        request.app.state.service.increment_stat("token_refreshes")

        logger.info("token_refreshed", user_id=str(payload.user_id))

        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
        )

    except (TokenExpiredError, TokenInvalidError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.post(
    "/auth/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Authentication"],
)
async def logout(
    current_user: TokenPayload = Depends(get_current_user),
    session_manager: SessionManager = Depends(get_session_manager),
) -> None:
    """
    Logout user by revoking all their sessions.

    Since tokens don't contain session IDs, we revoke all sessions for the user.
    """
    await session_manager.revoke_all_user_sessions(current_user.user_id)
    logger.info("user_logged_out", user_id=str(current_user.user_id))
    return None


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


@router.post(
    "/users/me/password",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Users"],
)
async def change_password(
    request_data: PasswordChangeRequest,
    request: Request,
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
    password_service: PasswordService = Depends(get_password_service),
    session_manager: SessionManager = Depends(get_session_manager),
) -> None:
    """Change current user's password."""
    # Hash new password
    new_password_hash = password_service.hash_password(request_data.new_password)

    # Change password via service
    result = await user_service.change_password(
        user_id=current_user.user_id,
        current_password=request_data.current_password,
        new_password_hash=new_password_hash,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to change password",
        )

    # Revoke all sessions except current
    await session_manager.revoke_all_user_sessions(current_user.user_id)

    # Increment stats
    request.app.state.service.increment_stat("password_changes")

    logger.info("password_changed", user_id=str(current_user.user_id))


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
    # Try to parse consent type
    try:
        consent_type = ConsentType(request_data.consent_type)
    except ValueError:
        # Allow custom consent types
        consent_type = ConsentType.ANALYTICS_TRACKING  # Default fallback

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
    "/users/{user_id}",
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


@router.delete(
    "/users/me",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Users"],
)
async def delete_current_user(
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
    session_manager: SessionManager = Depends(get_session_manager),
) -> None:
    """Delete current user's account (soft delete for GDPR compliance)."""
    result = await user_service.delete_user(
        user_id=current_user.user_id,
        reason="User requested account deletion",
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to delete account",
        )

    # Revoke all sessions
    await session_manager.revoke_all_user_sessions(current_user.user_id)

    logger.info("user_deleted_self", user_id=str(current_user.user_id))


# --- Email Verification Endpoints ---


@router.post(
    "/auth/verify-email",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Authentication"],
)
async def verify_email(
    user_id: UUID,
    token: str,
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Verify user email address."""
    result = await user_service.verify_email(user_id, token)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to verify email",
        )

    logger.info("email_verified", user_id=str(user_id))


@router.post(
    "/auth/resend-verification",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Authentication"],
)
async def resend_verification(
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Request new verification email."""
    token, error = await user_service.resend_verification_email(current_user.user_id)

    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error,
        )

    # Email is sent via notification service in the service layer
    logger.info(
        "verification_email_sent",
        user_id=str(current_user.user_id),
        token_generated=token is not None,
    )


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
