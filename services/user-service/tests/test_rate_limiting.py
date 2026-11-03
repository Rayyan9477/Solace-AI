"""P1-7 / P2-4: user-service auth endpoints must be rate limited.

The user-service auth surface (/auth/login, /auth/resend-verification) had NO rate
limiting. This proves the middleware is wired (outside the test harness) and that the
tight per-IP auth-resend bucket actually denies after its limit.
"""
from __future__ import annotations

import os

import pytest
from fastapi import FastAPI

from src.main import _register_rate_limiting, _rate_limiting_disabled


def test_rate_limiting_disabled_under_test_harness(monkeypatch):
    # The pytest harness sets ENVIRONMENT=test; limiting must be disabled so the shared
    # sliding-window counter cannot bleed across tests.
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.delenv("USER_SERVICE_ENV", raising=False)
    assert _rate_limiting_disabled() is True


def test_rate_limiting_enforced_in_production_env(monkeypatch):
    monkeypatch.setenv("USER_SERVICE_ENV", "production")
    assert _rate_limiting_disabled() is False


def test_register_adds_rate_limit_middleware_when_enabled(monkeypatch):
    # Force "enabled" and a non-production env so a missing Redis degrades to in-memory
    # instead of fail-loud.
    monkeypatch.setattr("src.main._rate_limiting_disabled", lambda: False)
    monkeypatch.setenv("USER_SERVICE_ENV", "staging")
    monkeypatch.setenv("ENVIRONMENT", "staging")

    app = FastAPI()
    _register_rate_limiting(app, "redis://localhost:6379/0")

    from solace_security.rate_limit import RateLimitMiddleware

    assert any(m.cls is RateLimitMiddleware for m in app.user_middleware)


def test_resend_verification_bucket_denies_after_limit(monkeypatch):
    # ENVIRONMENT=test keeps create_solace_rate_limiter from fail-loud on a None redis
    # client, giving a deterministic in-memory limiter to exercise the policy.
    monkeypatch.setenv("ENVIRONMENT", "test")
    from infrastructure.api_gateway.rate_limiting import (
        create_solace_rate_limiter,
        RateLimitPolicy,
        RateLimitScope,
        RateLimitWindow,
    )

    limiter = create_solace_rate_limiter(redis_client=None)
    limiter.add_policy(
        RateLimitPolicy(
            name="auth-resend-per-ip", limit=5, window=RateLimitWindow.MINUTE,
            scope=RateLimitScope.IP, route_name="auth-resend", tags=["auth"],
        )
    )
    ident = "ip:203.0.113.9"
    results = [
        limiter.check_request("user-service", "auth-resend", ident) for _ in range(6)
    ]
    assert all(r.allowed for r in results[:5])
    assert results[5].allowed is False
    assert results[5].retry_after and results[5].retry_after > 0
