"""Rate-limiting middleware (REV-24 / API-2).

Rate limiting was entirely unimplemented: a complete RateLimiter (sliding window,
Redis-backed for prod) lived in infrastructure.api_gateway.rate_limiting but was wired
into no application, and nothing emitted 429 / Retry-After.

This module provides a reusable Starlette middleware that enforces per-caller request
limits and emits the standard ``X-RateLimit-*`` headers plus ``Retry-After`` on 429.

It stays decoupled from both the limiter implementation and the auth stack:

- ``limiter`` is any object exposing
  ``check_request(service_name, route_name, identifier) -> result`` where ``result``
  has ``.allowed: bool`` and ``.to_headers() -> dict[str, str]`` (the api_gateway
  ``RateLimiter``/``RateLimitResult`` satisfy this).
- Caller identity is produced by an injected ``principal_resolver``; the default keys
  on client IP so anonymous traffic is still bounded. Services that authenticate pass
  a resolver that returns a per-user identifier when a valid token is present.
- A limiter that raises (its backend is down) fails open: the request is served and one
  ``rate_limiter_backend_error`` line names the exception class only. A limiter that
  answers "not allowed" is still a 429.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

PrincipalResolver = Callable[[Request], str]

# Paths that must never be rate limited: infra probes and the API docs UIs.
# Matched exactly (plus the Swagger oauth2-redirect helper) so a business route that
# merely starts with "/docs"/"/redoc" is NOT silently exempted.
_DEFAULT_EXEMPT: frozenset[str] = frozenset(
    {"/", "/health", "/ready", "/live", "/metrics", "/openapi.json",
     "/docs", "/redoc", "/docs/oauth2-redirect"}
)


def client_ip(request: Request, trusted_proxy_hops: int = 0) -> str:
    """Resolve the caller IP for rate-limit keying.

    SECURITY: ``X-Forwarded-For`` is client-controlled and must not be trusted by
    default — trusting the leftmost value lets a caller rotate the header to mint a
    fresh limit bucket per request. So:

    - ``trusted_proxy_hops <= 0`` (default): use the real socket peer
      (``request.client.host``) and ignore XFF entirely.
    - ``trusted_proxy_hops == n > 0``: the app sits behind exactly ``n`` trusted
      reverse proxies (e.g. Caddy), each of which appends one hop to XFF. The genuine
      client is then the entry ``n`` positions from the right; take it, falling back to
      the socket peer if the header is too short to trust.
    """
    if trusted_proxy_hops > 0:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            if len(hops) >= trusted_proxy_hops:
                return hops[-trusted_proxy_hops]
    if request.client:
        return request.client.host
    return "unknown"


def _default_principal(request: Request, trusted_proxy_hops: int = 0) -> str:
    return f"ip:{client_ip(request, trusted_proxy_hops)}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-caller request limits and surface Retry-After on 429."""

    def __init__(
        self,
        app: Any,
        *,
        limiter: Any,
        service_name: str,
        principal_resolver: PrincipalResolver | None = None,
        exempt_paths: Iterable[str] | None = None,
        trusted_proxy_hops: int = 0,
        route_resolver: Callable[[Request], str | None] | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._service_name = service_name
        self._trusted_proxy_hops = trusted_proxy_hops
        self._resolve_principal = principal_resolver or self._default_principal
        self._route_resolver = route_resolver
        self._exempt = frozenset(exempt_paths) if exempt_paths is not None else _DEFAULT_EXEMPT

    def _default_principal(self, request: Request) -> str:
        return _default_principal(request, self._trusted_proxy_hops)

    def _is_exempt(self, path: str) -> bool:
        return path in self._exempt

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Preflight and infra/doc paths bypass limiting entirely.
        if request.method == "OPTIONS" or self._is_exempt(request.url.path):
            return await call_next(request)

        try:
            identifier = self._resolve_principal(request)
        except Exception:
            # Identity resolution must never take down a request; degrade to IP.
            identifier = self._default_principal(request)

        route_name = self._route_resolver(request) if self._route_resolver else None
        # Prefer the async path so a Redis-backed limiter enforces shared, cross-instance
        # limits (atomic INCR); fall back to the sync in-memory path otherwise.
        try:
            if hasattr(self._limiter, "check_request_async"):
                result = await self._limiter.check_request_async(
                    self._service_name, route_name, identifier
                )
            else:
                result = self._limiter.check_request(self._service_name, route_name, identifier)
        except Exception as exc:
            # Fail open (WS111-CRI-1): a limiter backend outage (a Redis failover or blip)
            # must not turn every request, crisis messages on the chat route included, into
            # a 500. Only the exception class is logged: never the key, IP, user or text.
            logger.warning("rate_limiter_backend_error", error_type=type(exc).__name__)
            return await call_next(request)

        if not result.allowed:
            logger.warning(
                "rate_limit_exceeded",
                identifier=identifier,
                path=request.url.path,
                method=request.method,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please retry later.",
                    }
                },
                headers=result.to_headers(),
            )

        response = await call_next(request)
        for key, value in result.to_headers().items():
            response.headers[key] = value
        return response
