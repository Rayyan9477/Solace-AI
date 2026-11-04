"""
Solace-AI User Service - shared REST API dependencies and DTOs.

The app-state dependency getters, the JWT ``get_current_user`` dependency and the DTOs that more
than one router uses (``UserResponse``, ``user_to_response``, ``TokenResponse``). Split verbatim
out of ``api.py`` (WS-14.3) so the identity routers (``api_auth``, and WS-08's OAuth and MFA
modules) import them without importing each other.

Architecture Layer: Presentation
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from .domain.entities import User
from .domain.service import UserService
from .infrastructure.jwt_service import (
    JWTService,
    TokenPayload,
    TokenType,
    TokenExpiredError,
    TokenInvalidError,
)
from .infrastructure.password_service import PasswordService
from .infrastructure.repository import UserRepository
from .auth import SessionManager


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiry in seconds")


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
        payload = await jwt_service.verify_token(token, expected_type=TokenType.ACCESS)
    except (TokenExpiredError, TokenInvalidError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # P1-3 (C.2 RLS): publish the authenticated user id into the request context so the
    # DB layer sets the Postgres app.current_user_id GUC that RLS on consent/preferences
    # (migration 006) enforces. Without this the user-service's own RLS-protected reads
    # return no rows on real Postgres. Reset per-request in the finally so the contextvar
    # never leaks into an unrelated request handled on the same worker.
    from solace_common.request_context import set_current_user_id, reset_current_user_id

    ctx_token = set_current_user_id(str(payload.user_id))
    try:
        yield payload
    finally:
        reset_current_user_id(ctx_token)


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
