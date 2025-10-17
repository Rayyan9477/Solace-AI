"""
Authentication Models for Solace-AI API

Contains Pydantic models for authentication, user management, and security.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, validator, Field
from enum import Enum
import re
from src.config.security import SecurityConfig


class UserRole(str, Enum):
    """User role enumeration"""
    USER = "user"
    THERAPIST = "therapist"
    ADMIN = "admin"


class UserCreate(BaseModel):
    """Model for user registration"""
    username: str = Field(..., min_length=3, max_length=50, regex=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=SecurityConfig.PASSWORD_MIN_LENGTH, max_length=SecurityConfig.PASSWORD_MAX_LENGTH)
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = UserRole.USER
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements"""
        pattern = SecurityConfig.get_password_regex()
        if not re.match(pattern, v):
            raise ValueError(
                f"Password must be {SecurityConfig.PASSWORD_MIN_LENGTH}-{SecurityConfig.PASSWORD_MAX_LENGTH} characters "
                f"and contain uppercase, lowercase, digit, and special character"
            )
        return v
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        return v


class UserLogin(BaseModel):
    """Model for user login"""
    username: str = Field(..., max_length=50)
    password: str = Field(..., max_length=SecurityConfig.PASSWORD_MAX_LENGTH)


class UserResponse(BaseModel):
    """Model for user data response (excluding sensitive fields)"""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT token response model"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Token payload data"""
    sub: str  # user_id
    username: Optional[str] = None
    role: str = "user"
    permissions: List[str] = []
    exp: int
    iat: int
    jti: str  # JWT ID for token revocation


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class PasswordReset(BaseModel):
    """Password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation with new password"""
    token: str
    new_password: str = Field(..., min_length=SecurityConfig.PASSWORD_MIN_LENGTH, max_length=SecurityConfig.PASSWORD_MAX_LENGTH)
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        """Validate new password meets security requirements"""
        pattern = SecurityConfig.get_password_regex()
        if not re.match(pattern, v):
            raise ValueError(
                f"Password must be {SecurityConfig.PASSWORD_MIN_LENGTH}-{SecurityConfig.PASSWORD_MAX_LENGTH} characters "
                f"and contain uppercase, lowercase, digit, and special character"
            )
        return v


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=SecurityConfig.PASSWORD_MIN_LENGTH, max_length=SecurityConfig.PASSWORD_MAX_LENGTH)
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        """Validate new password meets security requirements"""
        pattern = SecurityConfig.get_password_regex()
        if not re.match(pattern, v):
            raise ValueError(
                f"Password must be {SecurityConfig.PASSWORD_MIN_LENGTH}-{SecurityConfig.PASSWORD_MAX_LENGTH} characters "
                f"and contain uppercase, lowercase, digit, and special character"
            )
        return v


class UserUpdate(BaseModel):
    """Model for updating user profile"""
    full_name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None
    preferences: Optional[Dict[str, Any]] = None


# Enhanced validation models for existing API endpoints
class ChatRequestSecure(BaseModel):
    """Secure chat request with enhanced validation"""
    message: str = Field(..., min_length=1, max_length=SecurityConfig.INPUT_LIMITS["max_message_length"])
    user_id: Optional[str] = Field(None, max_length=SecurityConfig.INPUT_LIMITS["max_user_id_length"])
    metadata: Optional[Dict[str, Any]] = Field(None, max_items=20)
    
    @validator('message')
    def sanitize_message(cls, v):
        """Basic sanitization of message content"""
        # Remove potential script tags and other dangerous content
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>'
        ]
        
        for pattern in dangerous_patterns:
            v = re.sub(pattern, '', v, flags=re.IGNORECASE | re.DOTALL)
        
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Validate user ID format"""
        if v and not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("User ID contains invalid characters")
        return v


class DiagnosticAssessmentRequestSecure(BaseModel):
    """Secure diagnostic assessment request"""
    user_id: str = Field(..., max_length=SecurityConfig.INPUT_LIMITS["max_user_id_length"])
    assessment_type: str = Field(..., max_length=50, regex=r"^[a-zA-Z0-9_]+$")
    responses: Dict[str, Any] = Field(..., max_items=SecurityConfig.INPUT_LIMITS["max_assessment_responses"])
    
    @validator('assessment_type')
    def validate_assessment_type(cls, v):
        """Validate assessment type"""
        allowed_types = ["phq9", "gad7", "big_five", "mbti"]
        if v.lower() not in allowed_types:
            raise ValueError(f"Assessment type must be one of: {allowed_types}")
        return v.lower()


class UserProfileRequestSecure(BaseModel):
    """Secure user profile update request"""
    user_id: Optional[str] = Field(None, max_length=SecurityConfig.INPUT_LIMITS["max_user_id_length"])
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    preferences: Optional[Dict[str, Any]] = Field(None, max_items=50)
    metadata: Optional[Dict[str, Any]] = Field(None, max_items=50)
    
    @validator('name')
    def sanitize_name(cls, v):
        """Sanitize name field"""
        if v:
            # Remove HTML tags and trim whitespace
            v = re.sub(r'<[^>]+>', '', v).strip()
        return v


# API Key authentication model
class APIKeyAuth(BaseModel):
    """API Key authentication for service-to-service communication"""
    api_key: str = Field(..., min_length=32)
    service_name: str = Field(..., max_length=50)


# Audit and logging models
class AuditEvent(BaseModel):
    """Audit event model"""
    event_type: str
    user_id: Optional[str] = None
    ip_address: str
    user_agent: Optional[str] = None
    endpoint: str
    method: str
    status_code: int
    timestamp: datetime
    additional_data: Optional[Dict[str, Any]] = None