"""WS-04.F2 (W0-close review SEC-3): in a strict environment every CORS origin is https and never localhost or
loopback. A copied .env.example (http://localhost:3000) or any plain-http origin used to pass the strict gate, and an
edge service then allowed credentialed requests from an origin a network attacker can inject scripts into."""
import pytest
from contextlib import asynccontextmanager
from solace_infrastructure.runtime import ServiceRuntimeSpec, StartupDependencyError, create_service_app
from solace_infrastructure.runtime.http import cors_allowed_origins

REFUSED = [
    "http://localhost:3000",            # the .env.example dev default
    "http://evil.example",
    "http://app.example.test",
    "https://localhost:3000",
    "https://LOCALHOST",
    "https://localhost.",
    "https://app.localhost",            # RFC 6761: *.localhost is loopback too
    "https://127.0.0.1",
    "https://127.0.0.2:8443",
    "https://[::1]:3000",
    "https://[::ffff:127.0.0.1]",
    "https://0.0.0.0",
    "ws://app.example.test",
    "app.example.test",
    "https://",
    "https://app.example.test:notaport",
    "https://app.example.test, http://localhost:3000",   # one bad entry refuses the list
]
ALLOWED = ["https://app.example.test", "https://app.example.test:8443", "https://app.example.test,https://admin.example.test"]


def _origins(monkeypatch, value: str, *, environment: str, bare: str | None = None) -> list:
    monkeypatch.setenv("ENVIRONMENT", bare if bare is not None else environment)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", value)
    return cors_allowed_origins(ServiceRuntimeSpec(service_name="orchestrator-service", environment=environment, edge=True))


@pytest.mark.parametrize("environment", ["staging", "production"])
@pytest.mark.parametrize("value", REFUSED)
def test_strict_refuses_plain_http_localhost_and_loopback_origins(monkeypatch, environment, value):
    with pytest.raises(StartupDependencyError, match="https origins"):
        _origins(monkeypatch, value, environment=environment)


def test_a_strict_bare_environment_refuses_them_under_a_development_service_env(monkeypatch):
    with pytest.raises(StartupDependencyError, match="https origins"):
        _origins(monkeypatch, "http://localhost:3000", environment="development", bare="staging")


@pytest.mark.parametrize("environment", ["staging", "production"])
@pytest.mark.parametrize("value", ALLOWED)
def test_strict_accepts_https_origins(monkeypatch, environment, value):
    assert _origins(monkeypatch, value, environment=environment) == [o.strip() for o in value.split(",")]


@pytest.mark.parametrize("environment", ["staging", "production"])
@pytest.mark.parametrize("value", ["*", "null", "https://app.example.test,*"])
def test_strict_still_refuses_wildcard_and_null(monkeypatch, environment, value):
    with pytest.raises(StartupDependencyError, match="'\\*' or 'null'"):
        _origins(monkeypatch, value, environment=environment)


@pytest.mark.parametrize("environment", ["development", "test"])
def test_dev_and_test_keep_the_localhost_default(monkeypatch, environment):
    assert _origins(monkeypatch, "http://localhost:3000", environment=environment) == ["http://localhost:3000"]


def test_a_strict_edge_app_refuses_to_build_with_a_localhost_origin(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
    spec = ServiceRuntimeSpec(service_name="orchestrator-service", environment="production", edge=True)

    @asynccontextmanager
    async def lifespan(app):
        yield

    with pytest.raises(StartupDependencyError, match="https origins"):
        create_service_app(spec, lifespan=lifespan, routers=[], title="t")
