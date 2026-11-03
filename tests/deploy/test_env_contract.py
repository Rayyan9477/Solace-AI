"""IF-02: the registry is the deployment env contract; compose, .env.example and code agree with it."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import pytest
import yaml
from deploy.env.registry import APP_SERVICES, FORMATS
from deploy.env.render_env_example import render_env_example
from deploy.env.env_readers import collect_env_readers
from deploy.env.compose_render import ComposeRenderError
from tests.deploy.conftest import REPO

def _index(registry, profile):
    idx = {}
    for v in registry:
        if not v.retired and profile in v.profiles:
            for s in v.services:
                idx.setdefault(s, set()).add(v.name)
    return idx

def test_registry_entries_are_well_formed(registry):
    seen = set()
    for v in registry:
        assert v.format in FORMATS and v.services and v.group, v.name
        for s in v.services:
            assert (v.name, s) not in seen, (v.name, s)
            seen.add((v.name, s))
        if v.secret:
            assert v.strict_default is None and v.example is None, v.name

def test_env_example_is_rendered_from_registry(registry):
    assert (REPO / ".env.example").read_text(encoding="utf-8") == render_env_example(registry)

def test_env_example_holds_no_secret_value(registry):
    lines = dict(l.split("=", 1) for l in (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
                 if l and not l.startswith("#"))
    for v in registry:
        if v.secret and v.source and not v.retired:
            assert lines.get(v.source, "") in ("", "''"), v.source

@pytest.mark.parametrize("profile,environment", [("dev", "development"), ("strict", "staging"), ("strict", "production")])
def test_registered_variables_are_set_for_their_services(render, registry, profile, environment):
    r = render(profile, environment=environment)
    idx = _index(registry, profile)
    missing = sorted((s, n) for s, names in idx.items() if s in r.services
                     for n in names if n not in r.services[s].get("environment", {}))
    assert not missing

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_every_app_variable_is_registered(render, registry, profile):
    r = render(profile)
    idx = _index(registry, profile)
    extra = {s: sorted(set(r.services[s].get("environment", {})) - idx.get(s, set()))
             for s in (*APP_SERVICES, "migrate", "caddy") if s in r.services}
    assert {s: e for s, e in extra.items() if e} == {}

class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that also reads compose's custom tags (!reset, !override) as plain nodes."""

def _construct_tagged(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)

_ComposeLoader.add_multi_constructor("!", _construct_tagged)

def _raw_env_file_services(files):
    """Services that declare env_file in the raw files (the render only sees the .env.prod probe)."""
    found = []
    for name in files:
        services = yaml.load((REPO / name).read_text(encoding="utf-8"), Loader=_ComposeLoader).get("services") or {}
        found += [f"{name}:{s}" for s, d in services.items() if isinstance(d, dict) and "env_file" in d]
    return found

def test_no_service_reads_an_env_file(render):
    assert [s for s, d in render("strict").services.items() if d.get("env_file")] == []
    # also catches entries the render cannot see, e.g. {path: ./.env.real, required: false}
    from tests.deploy.conftest import PROFILES
    assert _raw_env_file_services(PROFILES["strict"]) == []

def test_containers_are_project_scoped(render):
    assert [s for s, d in render("dev").services.items() if d.get("container_name")] == []

def test_no_dev_default_survives_in_strict(render, registry):
    r = render("strict")
    leaks = sorted((s, v.name) for v in registry if v.dev_default is not None and v.strict_default is None
                   and not v.retired and "strict" in v.profiles for s in v.services
                   if s in r.services and r.services[s]["environment"].get(v.name) == v.dev_default)
    assert not leaks

def test_strict_with_only_required_sources_keeps_no_dev_default(render, registry):
    # the operator supplies only what the strict render demands; every other strict row must
    # then hold its strict_default or be empty, never a dev default
    required = {v.source for v in registry if v.required_in_strict and v.source and not v.retired}
    full = render("strict")
    r = render("strict", drop=sorted(k for k in full.env if k not in required and k != "ENVIRONMENT"))
    allowed = {("notification-service", "EMAIL_SMTP_HOST"), ("notification-service", "EMAIL_SMTP_PORT")}  # MVP-042, removed by WS-03.6
    leaks = sorted((s, v.name) for v in registry
                   if "strict" in v.profiles and not v.retired and not v.required_in_strict and v.source not in required
                   for s in v.services if s in r.services and (s, v.name) not in allowed
                   and r.services[s]["environment"].get(v.name) not in (v.strict_default, ""))
    assert not leaks

def test_docker_gate_fails_when_the_cli_is_missing(monkeypatch):
    # opted in (realinfra tier) but no docker CLI: a deploy realinfra/e2e test must FAIL, not
    # xfail or skip, so the tier can never go green without running
    from tests.deploy.conftest import pytest_collection_modifyitems, pytest_runtest_setup

    class _Item:
        path = Path(__file__)
        def __init__(self):
            self.stash, self.added = pytest.Stash(), []
        def get_closest_marker(self, name):
            return pytest.mark.realinfra.mark if name == "realinfra" else None
        def add_marker(self, marker):
            self.added.append(marker)

    monkeypatch.setenv("SOLACE_REQUIRE_REALINFRA", "1")
    monkeypatch.delenv("SOLACE_RUN_DOCKER_TESTS", raising=False)
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)
    item = _Item()
    pytest_collection_modifyitems(None, [item])
    assert item.added == []
    with pytest.raises(pytest.fail.Exception, match="docker CLI not on PATH"):
        pytest_runtest_setup(item)

def test_required_strict_sources_fail_fast(render, registry):
    sources = sorted({v.source for v in registry if v.required_in_strict and v.source and not v.retired})
    def silently_rendered(src):
        try:
            render("strict", drop=[src])
        except ComposeRenderError as exc:
            return None if src in str(exc) else src
        return src
    with ThreadPoolExecutor(max_workers=8) as pool:
        offenders = [s for s in pool.map(silently_rendered, sources) if s]
    assert offenders == []

def test_no_code_reads_a_retired_name(registry):
    readers = collect_env_readers([REPO / "src", REPO / "services"])
    assert {v.name: readers[v.name] for v in registry if v.retired and v.name in readers} == {}
