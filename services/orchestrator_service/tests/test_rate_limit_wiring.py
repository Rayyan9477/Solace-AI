"""Phase B / API-2 (REV-24): the orchestrator app must wire the rate-limit middleware.

The RateLimitMiddleware itself is unit-tested in tests/solace_security. This proves the
orchestrator actually mounts it (outside the automated test harness) and skips it under
ENVIRONMENT=test so the shared in-memory counter cannot bleed across tests.
"""
from __future__ import annotations

import pytest


def _has_rate_limit_mw(app) -> bool:
    return any(mw.cls.__name__ == "RateLimitMiddleware" for mw in app.user_middleware)


def test_rate_limiting_mounted_outside_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    from services.orchestrator_service.src.main import create_application

    app = create_application()
    assert _has_rate_limit_mw(app), "RateLimitMiddleware must be mounted outside ENVIRONMENT=test"


def test_rate_limiting_skipped_in_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "test")
    from services.orchestrator_service.src.main import create_application

    app = create_application()
    assert not _has_rate_limit_mw(app), "rate limiting must be off under ENVIRONMENT=test"


def test_rate_limiting_skipped_under_harness_testing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # The pytest harness (root conftest.py) sets ENVIRONMENT="testing", not "test".
    # The gate must recognize any "test*" value or it would run limiting across the suite.
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.delenv("ORCHESTRATOR_ENVIRONMENT", raising=False)
    from services.orchestrator_service.src.main import create_application

    app = create_application()
    assert not _has_rate_limit_mw(app), "rate limiting must be off under ENVIRONMENT=testing"


def test_rate_limiting_enforced_in_production_even_if_process_env_is_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stray process-level ENVIRONMENT=test must NOT disable limiting when the service
    # itself is configured for production (ORCHESTRATOR_ENVIRONMENT=production).
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("ORCHESTRATOR_ENVIRONMENT", "production")
    from services.orchestrator_service.src.main import create_application

    app = create_application()
    assert _has_rate_limit_mw(app), "production must always enforce rate limiting"
