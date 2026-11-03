# tests/deploy/test_mvp_225_cors_origins.py
"""MVP-225: strict deploys must set CORS_ALLOWED_ORIGINS for the two edge services; both trust exactly one proxy hop."""
import pytest
from deploy.env.compose_render import ComposeRenderError

@pytest.mark.parametrize("service", ["user-service", "orchestrator-service"])
def test_cors_origins_reach_edge_services(render, service):
    r = render("strict", overrides={"CORS_ALLOWED_ORIGINS": "https://app.example.test"})
    env = r.services[service]["environment"]
    assert env["CORS_ALLOWED_ORIGINS"] == "https://app.example.test"
    assert env["TRUSTED_PROXY_HOPS"] == "1"

def test_strict_render_requires_cors_origins(render):
    with pytest.raises(ComposeRenderError, match="CORS_ALLOWED_ORIGINS"):
        render("strict", drop=["CORS_ALLOWED_ORIGINS"])

# --- WS-03.5 additions ------------------------------------------------------------------------------
EDGE = ("orchestrator-service", "user-service")

@pytest.mark.parametrize("service", EDGE)
def test_dev_edge_services_default_to_localhost_and_no_proxy_hop(render, service):
    env = render("dev").services[service]["environment"]
    assert (env["CORS_ALLOWED_ORIGINS"], env["TRUSTED_PROXY_HOPS"]) == ("http://localhost:3000", "0")

@pytest.mark.parametrize("profile,hops", [("dev", "0"), ("strict", "1")])
def test_orchestrator_legacy_names_follow_the_shared_ones(render, profile, hops):
    r = render(profile, overrides={"CORS_ALLOWED_ORIGINS": "https://app.example.test"})
    env = r.services["orchestrator-service"]["environment"]
    assert env["ORCHESTRATOR_CORS_ORIGINS"] == env["CORS_ALLOWED_ORIGINS"] == "https://app.example.test"
    assert env["ORCHESTRATOR_TRUSTED_PROXY_HOPS"] == env["TRUSTED_PROXY_HOPS"] == hops

@pytest.mark.parametrize("service", EDGE)
@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_render_passes_the_runtime_cors_gate(render, monkeypatch, service, environment):
    """Controller hand-off (WS-05.5): the strict render a synthetic operator env produces must satisfy
    the edge services' fail-closed CORS gate, or the strict boot smoke cannot start them."""
    from solace_infrastructure.runtime.http import cors_allowed_origins
    from solace_infrastructure.runtime.spec import ServiceRuntimeSpec
    env = render("strict", environment=environment).services[service]["environment"]
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", env["CORS_ALLOWED_ORIGINS"])
    origins = cors_allowed_origins(ServiceRuntimeSpec(service_name=service, environment=environment, edge=True))
    assert origins and all(o.startswith("https://") for o in origins)

@pytest.mark.parametrize("service", EDGE)
def test_strict_never_inherits_the_dev_cors_default(render, service):
    env = render("strict").services[service]["environment"]
    assert "localhost" not in env["CORS_ALLOWED_ORIGINS"]
    if service == "orchestrator-service":
        assert "localhost" not in env["ORCHESTRATOR_CORS_ORIGINS"]

def test_cors_and_proxy_hops_are_registered(registry):
    rows = {(v.name, s): v for v in registry if not v.retired for s in v.services}
    for service in EDGE:
        cors = rows[("CORS_ALLOWED_ORIGINS", service)]
        assert (cors.source, cors.format, cors.required_in_strict) == ("CORS_ALLOWED_ORIGINS", "csv", True)
        hops = rows[("TRUSTED_PROXY_HOPS", service)]
        assert (hops.source, hops.dev_default, hops.strict_default) == (None, "0", "1")

def test_runbook_lists_cors_origins_as_required():
    from tests.deploy.conftest import REPO
    assert "CORS_ALLOWED_ORIGINS" in (REPO / "docs/STAGING-BRINGUP.md").read_text(encoding="utf-8")
