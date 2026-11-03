"""MVP-004/MVP-247: every service gate variable is derived from ENVIRONMENT in both profiles, never a literal."""
import re
import pytest
from deploy.env.compose_render import ComposeRenderError
from tests.deploy.conftest import REPO

GATES = {"orchestrator-service": "ORCHESTRATOR_ENVIRONMENT", "user-service": "USER_SERVICE_ENV",
         "safety-service": "SAFETY_ENVIRONMENT", "notification-service": "NOTIFICATION_SERVICE_ENV",
         "diagnosis-service": "DIAGNOSIS_ENVIRONMENT", "memory-service": "MEMORY_ENVIRONMENT",
         "therapy-service": "THERAPY_ENVIRONMENT", "personality-service": "PERSONALITY_ENVIRONMENT",
         "analytics-service": "ANALYTICS_SERVICE_ENV"}

@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_gates_follow_ENVIRONMENT(render, environment):
    r = render("strict", environment=environment)
    for service, gate in GATES.items():
        env = r.services[service]["environment"]
        assert env["ENVIRONMENT"] == environment, service
        assert env[gate] == environment, f"{service}:{gate}"

def test_dev_gates_follow_ENVIRONMENT(render):
    r = render("dev", environment="development")
    for service, gate in GATES.items():
        assert r.services[service]["environment"]["ENVIRONMENT"] == "development", service
        assert r.services[service]["environment"][gate] == "development", f"{service}:{gate}"

def test_strict_render_fails_without_ENVIRONMENT(render):
    with pytest.raises(ComposeRenderError, match="ENVIRONMENT"):
        render("strict", drop=["ENVIRONMENT"])

def test_no_gate_variable_is_a_literal():
    text = (REPO / "docker-compose.yml").read_text(encoding="utf-8") + (REPO / "docker-compose.prod.yml").read_text(encoding="utf-8")
    for gate in GATES.values():
        for m in re.finditer(rf"^\s*{gate}:\s*(.+)$", text, re.M):
            assert m.group(1).strip().strip('"').startswith("${ENVIRONMENT"), (gate, m.group(1))
