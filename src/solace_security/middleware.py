"""
Solace-AI Authentication Middleware - Shared FastAPI dependencies for service authentication.

Provides reusable authentication dependencies that can be imported and used by any service
to secure their endpoints with JWT token validation.

Usage:
    from solace_security.middleware import get_current_user, require_permissions

    @router.get("/protected")
    async def protected_endpoint(user: AuthenticatedUser = Depends(get_current_user)):
        return {"user_id": user.user_id}

    @router.post("/admin-only")
    async def admin_endpoint(
        user: AuthenticatedUser = Depends(require_permissions(["admin:write"]))
    ):
        return {"admin": True}
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import structlog

from .auth import (
    AuthSettings,
    JWTManager,
    TokenPayload,
    TokenType,
    AuthenticationResult,
)

logger = structlog.get_logger(__name__)

# HTTP Bearer token scheme
_bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    """Represents an authenticated user from a validated JWT token."""

    user_id: str = Field(..., description="User ID from token subject")
    token_type: TokenType = Field(..., description="Type of token used")
    roles: list[str] = Field(default_factory=list, description="User roles")
    permissions: list[str] = Field(default_factory=list, description="User permissions")
    session_id: str | None = Field(default=None, description="Session ID if present")
    metadata: dict = Field(default_factory=dict, description="Additional token metadata")

    @classmethod
    def from_payload(cls, payload: TokenPayload) -> "AuthenticatedUser":
        """Create AuthenticatedUser from token payload."""
        return cls(
            user_id=payload.sub,
            token_type=payload.type,
            roles=payload.roles,
            permissions=payload.permissions,
            session_id=payload.session_id,
            metadata=payload.metadata,
        )

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def has_any_role(self, roles: list[str]) -> bool:
        """Check if user has any of the specified roles."""
        return bool(set(self.roles) & set(roles))

    def has_all_permissions(self, permissions: list[str]) -> bool:
        """Check if user has all specified permissions."""
        return set(permissions).issubset(set(self.permissions))

    @property
    def is_service_token(self) -> bool:
        """Check if this is a service-to-service token."""
        return self.token_type == TokenType.SERVICE


class AuthenticatedService(BaseModel):
    """Represents an authenticated service from a service token."""

    service_name: str = Field(..., description="Name of the authenticated service")
    permissions: list[str] = Field(default_factory=list, description="Service permissions")

    @classmethod
    def from_payload(cls, payload: TokenPayload) -> "AuthenticatedService":
        """Create AuthenticatedService from token payload."""
        service_name = payload.metadata.get("service_name", payload.sub)
        if service_name.startswith("service:"):
            service_name = service_name[8:]
        return cls(
            service_name=service_name,
            permissions=payload.permissions,
        )


@lru_cache()
def _get_jwt_manager() -> JWTManager:
    """Get cached JWT manager instance.

    Uses LRU cache to ensure singleton-like behavior.
    Settings are loaded from environment variables.
    """
    try:
        settings = AuthSettings()
        return JWTManager(settings)
    except Exception as e:
        logger.error("jwt_manager_initialization_failed", error=str(e))
        raise


def _extract_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    """Extract and validate the bearer token from credentials."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Use Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def _handle_auth_failure(result: AuthenticationResult) -> None:
    """Convert authentication failure to HTTP exception."""
    error_map = {
        "TOKEN_EXPIRED": (status.HTTP_401_UNAUTHORIZED, "Token has expired"),
        "INVALID_TOKEN": (status.HTTP_401_UNAUTHORIZED, "Invalid token"),
        "INVALID_TOKEN_TYPE": (status.HTTP_401_UNAUTHORIZED, "Invalid token type"),
        "DECODE_ERROR": (status.HTTP_401_UNAUTHORIZED, "Failed to decode token"),
        "INVALID_ISSUER": (status.HTTP_401_UNAUTHORIZED, "Invalid token issuer"),
        "INVALID_AUDIENCE": (status.HTTP_401_UNAUTHORIZED, "Invalid token audience"),
    }

    status_code, default_detail = error_map.get(
        result.error_code or "",
        (status.HTTP_401_UNAUTHORIZED, "Authentication failed")
    )

    raise HTTPException(
        status_code=status_code,
        detail=result.error_message or default_detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    """
    FastAPI dependency to get the current authenticated user.

    Extracts JWT from Authorization header, validates it, and returns
    the authenticated user information.

    Raises:
        HTTPException: 401 if token is missing, invalid, or expired

    Example:
        @router.get("/me")
        async def get_me(user: AuthenticatedUser = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    token = _extract_token(credentials)
    jwt_manager = _get_jwt_manager()

    result = jwt_manager.validate_access_token(token)
    if not result.success or not result.payload:
        _handle_auth_failure(result)

    # Log successful authentication (without sensitive data)
    logger.debug(
        "user_authenticated",
        user_id=result.user_id,
        token_type=result.payload.type.value,
        path=request.url.path,
    )

    return AuthenticatedUser.from_payload(result.payload)


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser | None:
    """
    FastAPI dependency to optionally get the current user.

    Returns None if no token is provided, but validates if one is present.
    Useful for endpoints that work differently for authenticated vs anonymous users.

    Example:
        @router.get("/content")
        async def get_content(user: AuthenticatedUser | None = Depends(get_current_user_optional)):
            if user:
                return {"personalized": True}
            return {"personalized": False}
    """
    if credentials is None:
        return None

    return await get_current_user(request, credentials)


async def get_current_service(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedService:
    """
    FastAPI dependency for service-to-service authentication.

    Validates that the token is a service token and returns service information.

    Raises:
        HTTPException: 401 if not a valid service token

    Example:
        @router.post("/internal/sync")
        async def internal_sync(service: AuthenticatedService = Depends(get_current_service)):
            return {"service": service.service_name}
    """
    token = _extract_token(credentials)
    jwt_manager = _get_jwt_manager()

    result = jwt_manager.decode_token(token, expected_type=TokenType.SERVICE)
    if not result.success or not result.payload:
        _handle_auth_failure(result)

    logger.debug(
        "service_authenticated",
        service=result.payload.metadata.get("service_name"),
        path=request.url.path,
    )

    return AuthenticatedService.from_payload(result.payload)


def require_roles(*roles: str) -> Callable:
    """
    Create a dependency that requires the user to have specific roles.

    Args:
        *roles: Role names that the user must have (at least one)

    Returns:
        FastAPI dependency function

    Example:
        @router.delete("/users/{user_id}")
        async def delete_user(
            user_id: str,
            admin: AuthenticatedUser = Depends(require_roles("admin", "superadmin"))
        ):
            ...
    """
    async def _check_roles(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        if not user.has_any_role(list(roles)):
            logger.warning(
                "authorization_failed",
                user_id=user.user_id,
                required_roles=roles,
                user_roles=user.roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient privileges. Required roles: {', '.join(roles)}",
            )
        return user

    return _check_roles


def require_permissions(*permissions: str) -> Callable:
    """
    Create a dependency that requires the user to have specific permissions.

    Args:
        *permissions: Permission names that the user must have (all required)

    Returns:
        FastAPI dependency function

    Example:
        @router.post("/assessments")
        async def create_assessment(
            user: AuthenticatedUser = Depends(require_permissions("assessment:create"))
        ):
            ...
    """
    async def _check_permissions(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        if not user.has_all_permissions(list(permissions)):
            logger.warning(
                "authorization_failed",
                user_id=user.user_id,
                required_permissions=permissions,
                user_permissions=user.permissions,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient privileges. Required permissions: {', '.join(permissions)}",
            )
        return user

    return _check_permissions


def require_service_permission(*permissions: str) -> Callable:
    """
    Create a dependency that requires a service token with specific permissions.

    Args:
        *permissions: Permission names that the service must have

    Returns:
        FastAPI dependency function

    Example:
        @router.post("/internal/process")
        async def internal_process(
            service: AuthenticatedService = Depends(require_service_permission("internal:process"))
        ):
            ...
    """
    async def _check_service_permissions(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    ) -> AuthenticatedService:
        service = await get_current_service(request, credentials)
        missing = set(permissions) - set(service.permissions)
        if missing:
            logger.warning(
                "service_authorization_failed",
                service_name=service.service_name,
                required_permissions=permissions,
                missing_permissions=list(missing),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Service lacks required permissions: {', '.join(missing)}",
            )
        return service

    return _check_service_permissions


# Convenience aliases
CurrentUser = AuthenticatedUser
CurrentService = AuthenticatedService
