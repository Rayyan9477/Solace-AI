"""
Security Middleware for Solace-AI API

Implements comprehensive security middleware including:
- Security headers
- Rate limiting
- Request/response logging
- IP filtering
"""

import time
import asyncio
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


# Initialize rate limiting setup
rate_limiting_setup = RateLimitingSetup()
limiter = rate_limiting_setup.get_limiter()