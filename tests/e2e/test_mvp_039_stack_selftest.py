"""Harness self-test on a live compose stack (MVP-039, WS-05.4).

Every test but the last needs the e2e_stack fixture (tests/e2e/conftest.py): the real images built by docker
compose with docker-compose.test.yml, in the profile SOLACE_E2E_PROFILE names.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from solace_testing.e2e import (
    BASE_FILE,
    MVP_SERVICES,
    OVERLAY_ALLOWED_KEYS,
    OVERLAY_APP_SECRET,
    OVERLAY_RESET_KEYS,
    OVERLAY_SERVICES,
    PROD_FILE,
    TEST_FILE,
    compose_config,
    require_docker,
)
from solace_testing.e2e_manifest import hop

REPO = Path(__file__).resolve().parents[2]


def test_mvp_039_no_fixed_container_names_and_only_test_ports_published(e2e_stack):
    ps = [json.loads(l) for l in e2e_stack.compose("ps", "--format", "json").stdout.splitlines() if l.strip()]
    assert ps and all(p["Name"].startswith(e2e_stack.project) for p in ps)
    published = {p["Service"] for p in ps for pub in p.get("Publishers") or [] if pub.get("PublishedPort")}
    assert published <= {"caddy", "mailpit", "fake-llm"}
    assert all(pub["URL"] in ("127.0.0.1", "") for p in ps for pub in p.get("Publishers") or [] if pub.get("PublishedPort"))


def test_mvp_039_sql_runs_as_the_image_superuser(e2e_stack):
    assert e2e_stack.sql("SELECT current_user AS u")[0]["u"] == "postgres"


@pytest.mark.parametrize("service", MVP_SERVICES)
def test_mvp_039_call_reaches_each_service(e2e_stack, service):
    with hop("service_live"):
        assert e2e_stack.call(service, "GET", "/live").status == 200


async def test_mvp_039_edge_routes_to_the_orchestrator_without_internal_paths(e2e_stack):
    assert (await e2e_stack.edge("POST", "/api/v1/orchestrator/chat", json={"message": "hi"})).status_code == 401
    assert (await e2e_stack.edge("GET", "/metrics")).status_code == 404


def test_mvp_039_overlay_changes_only_allowed_keys_in_strict(tmp_path):
    # render 'docker compose -f docker-compose.yml -f docker-compose.prod.yml config --format json' with and without
    # -f docker-compose.test.yml (same generated env file); for every app service the environment differs only in
    # OVERLAY_ALLOWED_KEYS. No container is started. The renders hold generated secrets: only key names are compared
    # in assertion messages.
    require_docker()
    run = tmp_path / "run"
    gen = subprocess.run([sys.executable, "-m", "scripts.test.gen_e2e_env", "--profile", "strict",  # noqa: S603
                          "--environment", "production", "--out", str(run)], cwd=REPO, capture_output=True, text=True,
                         timeout=300)
    assert gen.returncode == 0, gen.stderr[-2000:]
    strict = [BASE_FILE, PROD_FILE]
    render_without = compose_config(strict, run / "e2e.env")
    render_with = compose_config([*strict, TEST_FILE], run / "e2e.env")
    without, with_overlay = render_without["services"], render_with["services"]

    assert set(with_overlay) - set(without) == set(OVERLAY_SERVICES)
    assert set(without) <= set(with_overlay)
    for name in MVP_SERVICES:
        before, after = without[name].get("environment") or {}, with_overlay[name].get("environment") or {}
        changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        assert changed <= OVERLAY_ALLOWED_KEYS, (name, sorted(changed - OVERLAY_ALLOWED_KEYS))
        # only user-service lets the harness's @test.invalid addresses register (WS-14.1 test-only setting)
        tlds = after.get("USER_EMAIL_ALLOW_SPECIAL_USE_TLDS")   # a local, so a failure never prints the environment
        assert tlds == ("invalid" if name == "user-service" else None), name
        # the one non-environment change: the e2e CA mounted at /run/secrets/e2e-ca.crt, next to the base mounts
        mounts_before = {s["source"]: s for s in without[name].get("secrets") or []}
        mounts_after = {s["source"]: s for s in with_overlay[name].get("secrets") or []}
        assert set(mounts_after) - set(mounts_before) == {OVERLAY_APP_SECRET}, (name, sorted(mounts_after))
        assert all(mounts_after[k] == v for k, v in mounts_before.items()), name
        assert mounts_after[OVERLAY_APP_SECRET].get("target", OVERLAY_APP_SECRET) in (
            OVERLAY_APP_SECRET, f"/run/secrets/{OVERLAY_APP_SECRET}"), name
    for name, spec in without.items():
        after = with_overlay[name]
        changed = {k for k in spec.keys() | after.keys() if spec.get(k) != after.get(k)}
        allowed = set(OVERLAY_RESET_KEYS)
        if name in MVP_SERVICES:
            allowed |= {"environment"} | (OVERLAY_ALLOWED_KEYS & {"secrets"})
        elif name == "caddy":
            allowed |= {"environment"}
        assert changed <= allowed, (name, sorted(changed - allowed))   # never build, image or any other strict setting
        assert not after.get("container_name") and not after.get("env_file"), name
        if name != "caddy":
            assert not after.get("ports"), name
    caddy_before, caddy_after = without["caddy"].get("environment") or {}, with_overlay["caddy"].get("environment") or {}
    assert {k for k in caddy_before.keys() | caddy_after.keys() if caddy_before.get(k) != caddy_after.get(k)} \
        <= OVERLAY_ALLOWED_KEYS
    published = [p for name in ("caddy", *OVERLAY_SERVICES) for p in with_overlay[name].get("ports") or []]
    assert published and all(p.get("host_ip") == "127.0.0.1" and not p.get("published") for p in published)
    # top-level secrets: the base ones unchanged, plus the e2e CA and mailpit's pair from the run's secrets dir
    top_before, top_after = render_without.get("secrets") or {}, render_with.get("secrets") or {}
    assert set(top_after) - set(top_before) == {OVERLAY_APP_SECRET, "e2e-mailpit.crt", "e2e-mailpit.key"}
    assert all(top_after[k] == v for k, v in top_before.items())
    assert Path(top_after[OVERLAY_APP_SECRET]["file"]) == run / "secrets" / "e2e-ca.crt"
