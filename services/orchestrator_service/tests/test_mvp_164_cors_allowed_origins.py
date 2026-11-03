"""MVP-164: browser clients on the configured origin pass CORS preflight; others do not."""
import pytest
from fastapi.testclient import TestClient

ORIGIN = "https://app.example.test"


def _preflight(monkeypatch, origin, allowed=ORIGIN):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", allowed)
    monkeypatch.delenv("ORCHESTRATOR_CORS_ORIGINS", raising=False)
    from services.orchestrator_service.src.main import create_application
    return TestClient(create_application()).options(
        "/api/v1/orchestrator/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})


def test_mvp_164_cors_allowed_origins_drives_preflight(monkeypatch):
    assert _preflight(monkeypatch, ORIGIN).headers.get("access-control-allow-origin") == ORIGIN


def test_mvp_164_unlisted_origin_is_refused(monkeypatch):
    assert "access-control-allow-origin" not in _preflight(monkeypatch, "https://evil.example.test").headers


def test_mvp_164_wildcard_origin_never_carries_credentials(monkeypatch):
    """Outside strict, '*' answers any origin with '*' and never reflects it with credentials."""
    headers = _preflight(monkeypatch, "https://evil.example.test", allowed="*").headers
    assert headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in headers


def test_mvp_164_strict_environment_refuses_a_wildcard_origin(monkeypatch):
    from solace_infrastructure.runtime import StartupDependencyError
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ORCHESTRATOR_ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_URL", "redis://:test-only@redis:6379/0")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    from services.orchestrator_service.src.main import create_application
    with pytest.raises(StartupDependencyError) as info:
        create_application()
    assert info.value.component == "cors"
