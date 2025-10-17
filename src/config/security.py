"""
Security Configuration for Solace-AI API

This module contains all security-related configurations following OWASP guidelines
and healthcare data protection standards.
"""

import os
from datetime import timedelta
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)


class SecurityConfig:
    """Security configuration settings following OWASP guidelines"""
    
    # JWT Configuration - Secure Key Requirement
    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    if not SECRET_KEY:
        raise ValueError(
            "JWT_SECRET_KEY environment variable is required. "
            "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    if len(SECRET_KEY) < 32:
        raise ValueError(
            "JWT_SECRET_KEY must be at least 32 characters long for security. "
            "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    # Password Security
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_MAX_LENGTH = 128
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL = True
    BCRYPT_ROUNDS = 12
    
    # Rate Limiting Configuration
    RATE_LIMIT_CONFIG = {
        "auth": "5/minute",           # Authentication endpoints
        "chat": "30/minute",          # Chat endpoints
        "assessment": "10/minute",    # Assessment endpoints
        "health": "60/minute",        # Health check
        "default": "20/minute",       # Default for other endpoints
        "upload": "5/minute",         # File upload endpoints
        "user_profile": "15/minute",  # User management
    }
    
    # CORS Configuration - Environment Based
    CORS_ORIGINS = {
        "development": [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            # React Native development
            "exp://127.0.0.1:19000",
            "exp://localhost:19000"
        ],
        "production": [
            # Add your production domains here
            "https://solace-ai.com",
            "https://app.solace-ai.com",
            "https://api.solace-ai.com"
        ],
        "staging": [
            "https://staging.solace-ai.com",
            "https://staging-api.solace-ai.com"
        ]
    }
    
    # Get environment-based CORS origins
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    ALLOWED_ORIGINS = CORS_ORIGINS.get(ENVIRONMENT, CORS_ORIGINS["development"])
    
    # Security Headers Configuration
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https:; "
            "font-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self';"
        )
    }
    
    # Input Validation Limits
    INPUT_LIMITS = {
        "max_message_length": 2000,
        "max_user_id_length": 50,
        "max_file_size_mb": 10,
        "max_assessment_responses": 100,
        "max_metadata_size": 1000,
        "allowed_audio_types": ["audio/wav", "audio/mp3", "audio/m4a", "audio/ogg"]
    }
    
    # Session Security
    SESSION_CONFIG = {
        "secure": True,
        "httponly": True,
        "samesite": "lax",
        "max_age": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }
    
    # User Roles and Permissions
    USER_ROLES = {
        "user": ["read:profile", "write:profile", "use:chat", "use:assessment"],
        "therapist": ["read:profile", "write:profile", "use:chat", "use:assessment", "read:sessions", "generate:reports"],
        "admin": ["*"]  # All permissions
    }
    
    # API Key Configuration (for service-to-service communication)
    API_KEY_HEADER = "X-API-Key"
    API_KEYS = {
        "internal_service": os.getenv("INTERNAL_API_KEY", ""),
        "monitoring": os.getenv("MONITORING_API_KEY", ""),
    }
    
    # Audit and Logging Configuration
    AUDIT_CONFIG = {
        "log_failed_auth": True,
        "log_rate_limit_exceeded": True,
        "log_suspicious_activity": True,
        "retention_days": 90,
        "sensitive_fields": ["password", "token", "api_key", "secret"]
    }
    
    # IP Whitelisting for admin endpoints
    ADMIN_IP_WHITELIST = [
        "127.0.0.1",
        "::1",
        # Add production admin IPs here
    ]
    
    @classmethod
    def get_password_regex(cls) -> str:
        """Get password validation regex based on security requirements"""
        patterns = []
        if cls.PASSWORD_REQUIRE_LOWERCASE:
            patterns.append("(?=.*[a-z])")
        if cls.PASSWORD_REQUIRE_UPPERCASE:
            patterns.append("(?=.*[A-Z])")
        if cls.PASSWORD_REQUIRE_DIGITS:
            patterns.append("(?=.*\\d)")
        if cls.PASSWORD_REQUIRE_SPECIAL:
            patterns.append("(?=.*[!@#$%^&*()_+\\-=\\[\\]{};':\"\\\\|,.<>/?])")
        
        length_pattern = f".{{{cls.PASSWORD_MIN_LENGTH},{cls.PASSWORD_MAX_LENGTH}}}"
        return f"^{''.join(patterns)}{length_pattern}$"
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment"""
        return cls.ENVIRONMENT == "development"
    
    @classmethod
    def validate_api_key(cls, api_key: str, key_type: str = "internal_service") -> bool:
        """Validate API key for service authentication"""
        expected_key = cls.API_KEYS.get(key_type)
        return expected_key and api_key == expected_key
    
    @classmethod
    def get_rate_limit(cls, endpoint_type: str = "default") -> str:
        """Get rate limit configuration for endpoint type"""
        return cls.RATE_LIMIT_CONFIG.get(endpoint_type, cls.RATE_LIMIT_CONFIG["default"])


class SecurityExceptions:
    """Custom security-related exceptions"""
    
    class AuthenticationError(Exception):
        """Authentication failed"""
        pass
    
    class AuthorizationError(Exception):
        """Authorization failed"""
        pass
    
    class RateLimitExceeded(Exception):
        """Rate limit exceeded"""
        pass
    
    class InvalidTokenError(Exception):
        """Token validation failed"""
        pass
    
    class WeakPasswordError(Exception):
        """Password doesn't meet security requirements"""
        pass