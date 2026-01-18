"""
Solace-AI API Gateway Infrastructure.
Kong-based API Gateway configuration for routing, authentication, rate limiting, and CORS.
"""
from .kong_config import KongConfig, KongSettings, ServiceConfig, UpstreamConfig
from .routes import (
    RouteConfig,
    RouteManager,
    RouteDefinition,
    ServiceRoutes,
)
from .rate_limiting import (
    RateLimitPolicy,
    RateLimiter,
    RateLimitConfig,
    RateLimitResult,
)
from .auth_plugin import (
    JWTAuthPlugin,
    JWTConfig,
    TokenClaims,
    AuthResult,
)
from .cors import (
    CORSConfig,
    CORSPolicy,
    CORSHandler,
)

__all__ = [
    "KongConfig",
    "KongSettings",
    "ServiceConfig",
    "UpstreamConfig",
    "RouteConfig",
    "RouteManager",
    "RouteDefinition",
    "ServiceRoutes",
    "RateLimitPolicy",
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "JWTAuthPlugin",
    "JWTConfig",
    "TokenClaims",
    "AuthResult",
    "CORSConfig",
    "CORSPolicy",
    "CORSHandler",
]
