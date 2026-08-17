"""Phase B / API-2 (REV-24): rate-limiting middleware with Retry-After.

Rate limiting was entirely unimplemented — a full RateLimiter existed in
infrastructure/api_gateway but was wired into no app, and no middleware emitted 429 /
Retry-After. This proves the reusable RateLimitMiddleware enforces per-caller limits,
returns 429 + Retry-After when exceeded, keeps buckets isolated per principal, and
exempts health/infra paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from infrastructure.api_gateway.rate_limiting import (
    RateLimiter,
    RateLimitPolicy,
    RateLimitWindow,
    RateLimitScope,
)
from solace_security.rate_limit import RateLimitMiddleware, client_ip


def _limiter(limit: int) -> RateLimiter:
    limiter = RateLimiter()
    limiter.add_policy(
        RateLimitPolicy(
            name="test-per-user",
            limit=limit,
            window=RateLimitWindow.MINUTE,
            scope=RateLimitScope.CONSUMER,
            service_name="test-svc",
        )
    )
    return limiter


def _app(limiter: RateLimiter, **mw_kwargs) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        service_name="test-svc",
        **mw_kwargs,
    )

    @app.get("/api/v1/thing")
    async def thing():
        return {"ok": True}

    @app.get("/health", include_in_schema=False)
    async def health():
        return {"status": "healthy"}

    @app.get("/docsearch")
    async def docsearch():
        # A business route that merely starts with "/docs" must NOT be auto-exempt.
        return {"ok": True}

    return app


def _by_user(request):
    return f"user:{request.headers.get('x-test-user', 'anon')}"


def test_requests_under_limit_pass_with_headers():
    client = TestClient(_app(_limiter(3), principal_resolver=_by_user))
    r = client.get("/api/v1/thing", headers={"x-test-user": "u1"})
    assert r.status_code == 200
    assert r.headers["X-RateLimit-Limit"] == "3"
    assert "X-RateLimit-Remaining" in r.headers


def test_request_over_limit_returns_429_with_retry_after():
    client = TestClient(_app(_limiter(2), principal_resolver=_by_user))
    assert client.get("/api/v1/thing", headers={"x-test-user": "u1"}).status_code == 200
    assert client.get("/api/v1/thing", headers={"x-test-user": "u1"}).status_code == 200
    third = client.get("/api/v1/thing", headers={"x-test-user": "u1"})
    assert third.status_code == 429
    assert "Retry-After" in third.headers
    assert int(third.headers["Retry-After"]) >= 1
    assert third.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"


def test_limits_are_isolated_per_principal():
    client = TestClient(_app(_limiter(2), principal_resolver=_by_user))
    # u1 exhausts its bucket
    client.get("/api/v1/thing", headers={"x-test-user": "u1"})
    client.get("/api/v1/thing", headers={"x-test-user": "u1"})
    assert client.get("/api/v1/thing", headers={"x-test-user": "u1"}).status_code == 429
    # u2 is unaffected
    assert client.get("/api/v1/thing", headers={"x-test-user": "u2"}).status_code == 200


def test_exempt_path_is_not_limited():
    client = TestClient(_app(_limiter(1), principal_resolver=_by_user))
    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_default_principal_ignores_spoofable_xff():
    # SECURITY: with no trusted proxy (default), X-Forwarded-For is IGNORED — a caller
    # rotating XFF must NOT mint fresh buckets. All requests key on the socket peer.
    client = TestClient(_app(_limiter(1)))
    assert client.get("/api/v1/thing", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    # Different spoofed XFF, same socket peer -> SAME bucket -> denied (not evaded).
    assert client.get("/api/v1/thing", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 429


def test_trusted_proxy_hops_reads_genuine_client_from_xff():
    # Behind one trusted proxy, the genuine client is the rightmost XFF entry; distinct
    # real clients get distinct buckets, but the value cannot be spoofed from the left.
    client = TestClient(_app(_limiter(1), trusted_proxy_hops=1))
    assert client.get("/api/v1/thing", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert client.get("/api/v1/thing", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 429
    assert client.get("/api/v1/thing", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200


def test_docs_prefix_route_is_not_exempt():
    # A route that merely starts with "/docs" is NOT the Swagger UI and must be limited.
    client = TestClient(_app(_limiter(1)))
    assert client.get("/docsearch").status_code == 200
    assert client.get("/docsearch").status_code == 429


def test_client_ip_default_ignores_xff():
    from starlette.requests import Request

    scope = {
        "type": "http", "headers": [(b"x-forwarded-for", b"9.9.9.9, 10.0.0.1")],
        "client": ("127.0.0.1", 5000),
    }
    # Default (no trusted proxy): ignore the spoofable header, use the socket peer.
    assert client_ip(Request(scope)) == "127.0.0.1"


def test_rate_limited_429_carries_cors_headers_when_cors_is_outermost():
    """G2 P3 regression: CORS must wrap the rate limiter so a 429 short-circuit still
    gets Access-Control-Allow-Origin — otherwise a cross-origin browser can't read it."""
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware, limiter=_limiter(1), service_name="test-svc",
        principal_resolver=_by_user,
    )
    # CORS added LAST -> outermost, exactly like the orchestrator wiring.
    app.add_middleware(CORSMiddleware, allow_origins=["http://app.example"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/v1/thing")
    async def thing():
        return {"ok": True}

    client = TestClient(app)
    origin = {"origin": "http://app.example", "x-test-user": "u1"}
    assert client.get("/api/v1/thing", headers=origin).status_code == 200
    denied = client.get("/api/v1/thing", headers=origin)
    assert denied.status_code == 429
    assert denied.headers.get("access-control-allow-origin") == "http://app.example"
    assert "Retry-After" in denied.headers


def test_client_ip_one_trusted_hop_takes_rightmost():
    from starlette.requests import Request

    scope = {
        "type": "http", "headers": [(b"x-forwarded-for", b"9.9.9.9, 10.0.0.1")],
        "client": ("127.0.0.1", 5000),
    }
    # One trusted proxy appended "10.0.0.1" as the immediate client — take it, not the
    # attacker-controlled leftmost "9.9.9.9".
    assert client_ip(Request(scope), trusted_proxy_hops=1) == "10.0.0.1"
