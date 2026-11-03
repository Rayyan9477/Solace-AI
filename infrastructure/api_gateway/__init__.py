"""
Solace-AI API Gateway Infrastructure.
Kong-based API Gateway configuration for routing, authentication, rate limiting, and CORS.

Exports are lazy (PEP 562) so importing one submodule (e.g. ``rate_limiting``) does not
load the others (``auth_plugin`` needs python-jose, ``kong_config`` needs httpx).
"""
from __future__ import annotations

import importlib
from typing import Any

_MODULE_NAMES: dict[str, tuple[str, ...]] = {
    ".kong_config": ("KongConfig", "KongSettings", "ServiceConfig", "UpstreamConfig"),
    ".routes": ("RouteConfig", "RouteManager", "RouteDefinition", "ServiceRoutes"),
    ".rate_limiting": ("RateLimitPolicy", "RateLimiter", "RateLimitConfig", "RateLimitResult"),
    ".auth_plugin": ("JWTAuthPlugin", "JWTConfig", "TokenClaims", "AuthResult"),
    ".cors": ("CORSConfig", "CORSPolicy", "CORSHandler"),
}

_EXPORTS: dict[str, tuple[str, str]] = {
    name: (module, name) for module, names in _MODULE_NAMES.items() for name in names
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
