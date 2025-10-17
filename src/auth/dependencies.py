"""
Authentication Dependencies for FastAPI

Provides dependency injection for authentication and authorization.
"""

from typing import Optional, List
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from src.auth.jwt_utils import jwt_manager
from src.auth.models import TokenData, UserRole
from src.config.security import SecurityConfig, SecurityExceptions

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None
) -> TokenData:
    """
    Get current authenticated user from JWT token
    
    Args:
        credentials: HTTP Bearer credentials
        request: FastAPI request object
        
    Returns:
        TokenData: Current user token data
        
    Raises:
        HTTPException: If authentication fails
    """
    # Handle missing credentials
    if not credentials:
        logger.warning(
            f"Authentication attempt without credentials from IP: {request.client.host if request else 'unknown'}",
            extra={
                "event_type": "auth_missing_credentials",
                "client_ip": request.client.host if request else "unknown",
                "endpoint": str(request.url) if request else "unknown"
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Verify the token
        token_data = jwt_manager.verify_token(credentials.credentials)
        
        logger.info(
            f"Successful authentication for user: {token_data.username}",
            extra={
                "event_type": "auth_success",
                "user_id": token_data.sub,
                "username": token_data.username,
                "role": token_data.role,
                "client_ip": request.client.host if request else "unknown"
            }
        )
        
        return token_data
        
    except SecurityExceptions.InvalidTokenError as e:
        logger.warning(
            f"Authentication failed: {str(e)} from IP: {request.client.host if request else 'unknown'}",
            extra={
                "event_type": "auth_failed",
                "error": str(e),
                "client_ip": request.client.host if request else "unknown",
                "endpoint": str(request.url) if request else "unknown"
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(
            f"Unexpected authentication error: {str(e)}",
            extra={
                "event_type": "auth_error",
                "error": str(e),
                "client_ip": request.client.host if request else "unknown"
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable"
        )


async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """
    Get current active user (additional checks can be added here)
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        TokenData: Verified active user
    """
    # Add additional checks here if needed (e.g., user is active, not suspended)
    return current_user


def require_roles(allowed_roles: List[UserRole]):
    """
    Dependency factory for role-based access control
    
    Args:
        allowed_roles: List of allowed user roles
        
    Returns:
        Dependency function
    """
    async def role_checker(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
        if current_user.role not in [role.value for role in allowed_roles]:
            logger.warning(
                f"Authorization failed: User {current_user.username} with role {current_user.role} "
                f"attempted to access endpoint requiring roles: {[role.value for role in allowed_roles]}",
                extra={
                    "event_type": "auth_role_denied",
                    "user_id": current_user.sub,
                    "username": current_user.username,
                    "user_role": current_user.role,
                    "required_roles": [role.value for role in allowed_roles]
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return role_checker


def require_permissions(required_permissions: List[str]):
    """
    Dependency factory for permission-based access control
    
    Args:
        required_permissions: List of required permissions
        
    Returns:
        Dependency function
    """
    async def permission_checker(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
        if not jwt_manager.validate_token_permissions(current_user, required_permissions):
            logger.warning(
                f"Permission denied: User {current_user.username} lacks required permissions: {required_permissions}",
                extra={
                    "event_type": "auth_permission_denied",
                    "user_id": current_user.sub,
                    "username": current_user.username,
                    "user_permissions": current_user.permissions,
                    "required_permissions": required_permissions
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return permission_checker


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """
    Get current user if authenticated, otherwise return None
    Useful for endpoints that work with or without authentication
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        Optional[TokenData]: Current user or None
    """
    if not credentials:
        return None
    
    try:
        return jwt_manager.verify_token(credentials.credentials)
    except SecurityExceptions.InvalidTokenError:
        return None
    except Exception:
        return None


def validate_api_key(required_key_type: str = "internal_service"):
    """
    Dependency factory for API key validation
    
    Args:
        required_key_type: Type of API key required
        
    Returns:
        Dependency function
    """
    async def api_key_checker(request: Request) -> bool:
        api_key = request.headers.get(SecurityConfig.API_KEY_HEADER)
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        if not SecurityConfig.validate_api_key(api_key, required_key_type):
            logger.warning(
                f"Invalid API key attempt from IP: {request.client.host}",
                extra={
                    "event_type": "invalid_api_key",
                    "client_ip": request.client.host,
                    "key_type": required_key_type
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return True
    
    return api_key_checker


# Pre-defined role dependencies for common use cases
require_admin = require_roles([UserRole.ADMIN])
require_therapist_or_admin = require_roles([UserRole.THERAPIST, UserRole.ADMIN])
require_authenticated = get_current_active_user

# Pre-defined permission dependencies
require_chat_access = require_permissions(["use:chat"])
require_assessment_access = require_permissions(["use:assessment"])
require_profile_write = require_permissions(["write:profile"])
require_session_read = require_permissions(["read:sessions"])