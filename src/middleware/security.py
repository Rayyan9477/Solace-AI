"""
Security Middleware for Solace-AI API

Implements comprehensive security middleware including:
- Security headers
- Rate limiting
- Request/response logging
- IP filtering
- CSRF protection (SEC-006)
"""

import time
import asyncio
import secrets
import hmac
import hashlib
from typing import Callable, List, Optional, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import logging
from datetime import datetime
import json

from src.config.security import SecurityConfig

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    def __init__(self, app, headers: Optional[Dict[str, str]] = None):
        super().__init__(app)
        self.headers = headers or SecurityConfig.SECURITY_HEADERS
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response"""
        response = await call_next(request)
        
        # Add security headers
        for header_name, header_value in self.headers.items():
            response.headers[header_name] = header_value
        
        # Add HTTPS redirect header in production
        if not SecurityConfig.is_development() and request.url.scheme == "http":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log requests and responses for security auditing"""
    
    def __init__(self, app, sensitive_fields: Optional[List[str]] = None):
        super().__init__(app)
        self.sensitive_fields = sensitive_fields or SecurityConfig.AUDIT_CONFIG["sensitive_fields"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details"""
        start_time = time.time()
        
        # Extract request details
        client_ip = get_remote_address(request)
        user_agent = request.headers.get("user-agent", "Unknown")
        method = request.method
        url = str(request.url)
        
        # Log request
        logger.info(
            f"Request: {method} {url} - IP: {client_ip} - User-Agent: {user_agent}",
            extra={
                "event_type": "http_request",
                "method": method,
                "url": url,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} - Time: {process_time:.3f}s",
                extra={
                    "event_type": "http_response",
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "client_ip": client_ip,
                    "method": method,
                    "url": url,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {str(e)} - Time: {process_time:.3f}s",
                extra={
                    "event_type": "http_error",
                    "error": str(e),
                    "process_time": process_time,
                    "client_ip": client_ip,
                    "method": method,
                    "url": url,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            raise


class IPFilterMiddleware(BaseHTTPMiddleware):
    """Filter requests based on IP address for admin endpoints"""
    
    def __init__(self, app, admin_endpoints: Optional[List[str]] = None):
        super().__init__(app)
        self.admin_endpoints = admin_endpoints or [
            "/api/supervision/configure",
            "/admin/",
        ]
        self.whitelist = SecurityConfig.ADMIN_IP_WHITELIST
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Filter IP addresses for admin endpoints"""
        client_ip = get_remote_address(request)
        request_path = request.url.path
        
        # Check if this is an admin endpoint
        is_admin_endpoint = any(
            request_path.startswith(endpoint) for endpoint in self.admin_endpoints
        )
        
        if is_admin_endpoint and client_ip not in self.whitelist:
            logger.warning(
                f"Blocked admin access from IP: {client_ip} to {request_path}",
                extra={
                    "event_type": "admin_access_blocked",
                    "client_ip": client_ip,
                    "endpoint": request_path,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied"}
            )
        
        return await call_next(request)


class RateLimitingSetup:
    """Setup rate limiting for the application"""
    
    def __init__(self):
        # Create limiter instance
        self.limiter = Limiter(key_func=get_remote_address)
    
    def get_limiter(self) -> Limiter:
        """Get the limiter instance"""
        return self.limiter
    
    def get_rate_limit_for_endpoint(self, endpoint_path: str) -> str:
        """Get rate limit configuration for specific endpoint"""
        if "/api/auth/" in endpoint_path:
            return SecurityConfig.get_rate_limit("auth")
        elif "/api/chat" in endpoint_path:
            return SecurityConfig.get_rate_limit("chat")
        elif "/api/assessment/" in endpoint_path:
            return SecurityConfig.get_rate_limit("assessment")
        elif "/health" in endpoint_path:
            return SecurityConfig.get_rate_limit("health")
        elif "/api/voice/" in endpoint_path:
            return SecurityConfig.get_rate_limit("upload")
        elif "/api/user/" in endpoint_path:
            return SecurityConfig.get_rate_limit("user_profile")
        else:
            return SecurityConfig.get_rate_limit("default")


# Rate limiting error handler
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit exceeded handler"""
    client_ip = get_remote_address(request)
    
    logger.warning(
        f"Rate limit exceeded for IP: {client_ip} on {request.url.path}",
        extra={
            "event_type": "rate_limit_exceeded",
            "client_ip": client_ip,
            "endpoint": request.url.path,
            "limit": str(exc.detail),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please try again later.",
            "retry_after": getattr(exc, 'retry_after', 60)
        },
        headers={"Retry-After": str(getattr(exc, 'retry_after', 60))}
    )


class ContentTypeValidationMiddleware(BaseHTTPMiddleware):
    """Validate content types for POST requests"""
    
    def __init__(self, app):
        super().__init__(app)
        self.allowed_content_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "audio/wav",
            "audio/mp3",
            "audio/m4a",
            "audio/ogg"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate content type for POST/PUT requests"""
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()
            
            # Skip validation for specific endpoints that handle their own validation
            skip_paths = ["/api/voice/transcribe"]
            if any(request.url.path.startswith(path) for path in skip_paths):
                return await call_next(request)
            
            if content_type and not any(
                content_type.startswith(allowed) for allowed in self.allowed_content_types
            ):
                logger.warning(
                    f"Invalid content type: {content_type} from IP: {get_remote_address(request)}",
                    extra={
                        "event_type": "invalid_content_type",
                        "content_type": content_type,
                        "client_ip": get_remote_address(request),
                        "endpoint": request.url.path,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                return JSONResponse(
                    status_code=415,
                    content={"detail": "Unsupported Media Type"}
                )
        
        return await call_next(request)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    CSRF Protection Middleware using Double Submit Cookie Pattern (SEC-006).

    This middleware provides stateless CSRF protection for state-changing requests.
    It is designed to work alongside JWT-based authentication.

    Protection approach:
    1. Requests with valid Authorization headers bypass CSRF check (JWT protected)
    2. State-changing requests (POST, PUT, DELETE, PATCH) require CSRF token
    3. CSRF token is validated against a signed cookie
    """

    # Token configuration
    CSRF_COOKIE_NAME = "csrftoken"
    CSRF_HEADER_NAME = "X-CSRFToken"
    CSRF_FORM_FIELD = "csrf_token"
    TOKEN_LENGTH = 32  # 256 bits of entropy

    # Endpoints that don't require CSRF protection
    EXEMPT_PATHS = [
        "/api/auth/login",
        "/api/auth/register",
        "/api/auth/refresh",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
    ]

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.secret_key = SecurityConfig.SECRET_KEY

    def _generate_csrf_token(self) -> str:
        """Generate a cryptographically secure CSRF token."""
        return secrets.token_urlsafe(self.TOKEN_LENGTH)

    def _sign_token(self, token: str) -> str:
        """Sign a CSRF token with HMAC."""
        signature = hmac.new(
            self.secret_key.encode(),
            token.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{token}.{signature}"

    def _verify_token(self, signed_token: str, submitted_token: str) -> bool:
        """Verify that the submitted token matches the signed cookie."""
        try:
            if not signed_token or not submitted_token:
                return False

            parts = signed_token.split(".")
            if len(parts) != 2:
                return False

            token, signature = parts

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                token.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return False

            # Verify submitted token matches
            return hmac.compare_digest(token, submitted_token)

        except (ValueError, TypeError, AttributeError, UnicodeError):
            return False

    def _is_exempt(self, request: Request) -> bool:
        """Check if the request path is exempt from CSRF protection."""
        path = request.url.path

        # Check explicit exemptions
        for exempt_path in self.EXEMPT_PATHS:
            if path.startswith(exempt_path):
                return True

        return False

    def _has_valid_authorization(self, request: Request) -> bool:
        """Check if request has a valid Authorization header (JWT-protected)."""
        auth_header = request.headers.get("Authorization", "")
        # If there's a Bearer token, the request is JWT-protected
        # which provides CSRF protection implicitly
        return auth_header.startswith("Bearer ") and len(auth_header) > 10

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with CSRF protection."""
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip safe methods
        if request.method in ("GET", "HEAD", "OPTIONS", "TRACE"):
            response = await call_next(request)
            # Set CSRF cookie for GET requests to forms
            if request.method == "GET" and not request.cookies.get(self.CSRF_COOKIE_NAME):
                token = self._generate_csrf_token()
                signed_token = self._sign_token(token)
                response.set_cookie(
                    key=self.CSRF_COOKIE_NAME,
                    value=signed_token,
                    httponly=False,  # Must be readable by JavaScript
                    samesite="strict",
                    secure=not SecurityConfig.is_development(),
                    max_age=3600 * 24  # 24 hours
                )
            return response

        # Skip exempt paths
        if self._is_exempt(request):
            return await call_next(request)

        # Skip JWT-protected requests (Authorization header provides CSRF protection)
        if self._has_valid_authorization(request):
            return await call_next(request)

        # Validate CSRF token for state-changing requests
        cookie_token = request.cookies.get(self.CSRF_COOKIE_NAME, "")
        header_token = request.headers.get(self.CSRF_HEADER_NAME, "")

        # Try to get token from form data if not in header
        submitted_token = header_token
        if not submitted_token:
            try:
                # Check if content-type is form data
                content_type = request.headers.get("content-type", "")
                if "application/x-www-form-urlencoded" in content_type:
                    form_data = await request.form()
                    submitted_token = form_data.get(self.CSRF_FORM_FIELD, "")
            except (ValueError, KeyError, RuntimeError, TypeError):
                pass

        if not self._verify_token(cookie_token, submitted_token):
            client_ip = get_remote_address(request)
            logger.warning(
                f"CSRF validation failed for IP: {client_ip} on {request.url.path}",
                extra={
                    "event_type": "csrf_validation_failed",
                    "client_ip": client_ip,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "CSRF validation failed",
                    "detail": "Missing or invalid CSRF token"
                }
            )

        return await call_next(request)


# Initialize rate limiting setup
rate_limiting_setup = RateLimitingSetup()
limiter = rate_limiting_setup.get_limiter()