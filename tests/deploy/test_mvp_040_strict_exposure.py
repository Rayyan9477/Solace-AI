# tests/deploy/test_mvp_040_strict_exposure.py
"""MVP-040: in strict only Caddy publishes host ports; Grafana never boots on admin/admin; dev ports bind loopback."""
import pytest
from deploy.env.compose_render import ComposeRenderError

def test_only_caddy_publishes_host_ports_in_strict(render):
    published = {name for name, svc in render("strict").services.items() if svc.get("ports")}
    assert published == {"caddy"}

def test_grafana_admin_password_is_required_in_strict(render):
    with pytest.raises(ComposeRenderError, match="GRAFANA_ADMIN_PASSWORD"):
        render("strict", drop=["GRAFANA_ADMIN_PASSWORD"])
    r = render("strict")
    assert r.services["grafana"]["environment"]["GF_SECURITY_ADMIN_PASSWORD"] == r.env["GRAFANA_ADMIN_PASSWORD"] != "admin"

def test_dev_ports_bind_loopback(render):
    for name, svc in render("dev").services.items():
        for port in svc.get("ports", []):
            assert port.get("host_ip") == "127.0.0.1", (name, port)

# --- WS-03.3 additions ------------------------------------------------------------------------------

def test_dev_only_tools_bind_loopback_too(render):
    # mailhog and kafka-ui sit behind the dev profile, so the plain render above never sees them
    r = render("dev", profiles=("dev",))
    assert {"mailhog", "kafka-ui"} <= set(r.services)
    for name in ("mailhog", "kafka-ui"):
        for port in r.services[name].get("ports", []):
            assert port.get("host_ip") == "127.0.0.1", (name, port)

def test_strict_publishes_nothing_but_caddy_even_with_the_dev_profile(render):
    published = {name for name, svc in render("strict", profiles=("dev",)).services.items() if svc.get("ports")}
    assert published == {"caddy"}

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_built_services_never_pull(render, profile):
    # controller ruling: solace/<name>:<tag> is an unqualified Docker Hub reference, so a service that
    # builds its image must never ask a registry for it
    from deploy.env.registry import APP_SERVICES
    services = render(profile).services
    built = {name for name, svc in services.items() if svc.get("build")}
    assert built == {*APP_SERVICES, "migrate"}
    assert {name: services[name].get("pull_policy") for name in built} == {name: "build" for name in built}

# --- WS-03.3 fix round 1 (review batch18: WS033-SEC-2) ----------------------------------------------
# Grafana applies GF_SECURITY_ADMIN_PASSWORD only when it first creates grafana.db, so a strict run on a
# dev-seeded grafana_data volume would keep admin/admin. Strict resets the password on every start.

def test_strict_grafana_resets_the_admin_password_on_every_start(render):
    entrypoint = render("strict").services["grafana"].get("entrypoint") or []
    assert entrypoint[:2] == ["sh", "-c"], entrypoint[:2]
    script = entrypoint[2]
    feed, _, rest = script.partition("|")
    assert "GF_SECURITY_ADMIN_PASSWORD" in feed and "printf" in feed      # the value reaches the CLI on stdin
    assert "admin reset-admin-password --password-from-stdin" in rest     # never as an argv value
    assert "|| {" in rest and "exit 1" in rest                            # a failed reset stops the container
    assert script.rstrip().endswith("exec /run.sh")
    assert render("dev").services["grafana"].get("entrypoint") is None   # dev keeps the image's /run.sh

@pytest.mark.integration
@pytest.mark.e2e
def test_strict_grafana_refuses_admin_admin_on_a_dev_seeded_volume(compose_project, tmp_path):
    from tests.deploy.conftest import PROFILES, ComposeProject, strict_env_file
    health = "wget -q -O /dev/null http://127.0.0.1:3000/api/health"
    login = "wget -q -O /dev/null http://admin:{}@127.0.0.1:3000/api/user"
    compose_project.up(["grafana"])  # dev: no admin password, so grafana.db is seeded with admin/admin
    assert compose_project.poll("grafana", health)
    seeded = compose_project.poll("grafana", login.format("admin"), timeout=20)
    assert seeded, "the dev run must have seeded admin/admin"
    values = strict_env_file(tmp_path / "strict.env")
    strict = ComposeProject(compose_project.name, tmp_path / "strict.env", files=PROFILES["strict"])
    try:
        strict.up(["grafana"])  # same project name, so the same grafana_data volume
        assert strict.poll("grafana", health)
        default_works = strict.poll("grafana", login.format("admin"), timeout=4)
        assert not default_works, "strict must not accept admin/admin"
        strict_works = strict.poll("grafana", login.format(values["GRAFANA_ADMIN_PASSWORD"]), timeout=20)
        assert strict_works, "the strict admin password must work"
    finally:
        strict.down()
