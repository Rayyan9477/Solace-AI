"""Pure helpers of solace_testing.e2e (MVP-039, WS-05.4). No Docker: the live stack is tests/e2e."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
import yaml

from solace_testing import e2e, e2e_manifest, realinfra

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "docker-compose.test.yml"

DESCRIBE = """
Consumer group 'solace-group-notification' has no active members.

GROUP                     TOPIC          PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG  CONSUMER-ID  HOST  CLIENT-ID
solace-group-notification solace.safety  0          10              12              2    -            -     -
solace-group-notification solace.safety  1          5               5               0    -            -     -
solace-group-notification solace.users   0          -               7               -    -            -     -
"""


# --------------------------------------------------------------------------------------------- parsers

def test_mvp_039_consumer_group_lag_is_summed_per_topic():
    assert e2e.parse_consumer_groups_describe(DESCRIBE) == {"solace.safety": 2, "solace.users": 7}


def test_mvp_039_consumer_group_without_offsets_counts_the_whole_log_as_lag():
    text = ("GROUP TOPIC PARTITION CURRENT-OFFSET LOG-END-OFFSET LAG CONSUMER-ID HOST CLIENT-ID\n"
            "g t 0 - 3 - c1 /10.0.0.5 c\n")
    assert e2e.parse_consumer_groups_describe(text) == {"t": 3}


def test_mvp_039_missing_consumer_group_raises():
    with pytest.raises(ValueError, match="does not exist"):
        e2e.parse_consumer_groups_describe("\nError: Consumer group 'nope' does not exist.\n")
    with pytest.raises(ValueError, match="no GROUP table"):
        e2e.parse_consumer_groups_describe("something else entirely")


# kafka-consumer-groups prints the table on stdout and the Empty / rebalancing notices on stderr.
TABLE = """
GROUP  TOPIC  PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG  CONSUMER-ID  HOST  CLIENT-ID
g      t1     0          4               6               2    -            -     -
g      t1     1          3               3               0    -            -     -
"""
NOTICES = ("Consumer group 'g' has no active members.", "Warning: Consumer group 'g' is rebalancing.")


def _stack_with_kafka_output(stdout: str, stderr: str) -> tuple[e2e.Stack, list]:
    calls: list = []
    stack = e2e.Stack(urls={}, edge_url="", edge_auth_url="", project="solace-e2e-zzut", mailpit_api="",
                      fake_llm_control="", pg_superuser_dsn=None, config=e2e.StackConfig("dev", "development"),
                      mail=None, llm=None)  # type: ignore[arg-type]

    def compose(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=stderr)

    stack.compose = compose  # type: ignore[method-assign]
    return stack, calls


@pytest.mark.parametrize("notice", NOTICES)
def test_mvp_039_kafka_lag_reads_the_table_from_stdout_whatever_stderr_says(notice):
    stack, calls = _stack_with_kafka_output(TABLE, notice + "\n")
    assert stack.kafka_lag("g") == {"t1": 2}
    assert calls[0][:3] == ("exec", "-T", "kafka") and calls[0][-2:] == ("--group", "g")


@pytest.mark.parametrize("notice", NOTICES)
def test_mvp_039_a_notice_after_the_table_is_not_a_row(notice):
    assert e2e.parse_consumer_groups_describe(TABLE + notice + "\n") == {"t1": 2}


def test_mvp_039_kafka_lag_reports_a_missing_group_from_either_stream():
    for stdout, stderr in (("", "Error: Consumer group 'g' does not exist.\n"),
                           ("\nError: Consumer group 'g' does not exist.\n", "")):
        stack, _ = _stack_with_kafka_output(stdout, stderr)
        with pytest.raises(ValueError, match="does not exist"):
            stack.kafka_lag("g")


def test_mvp_039_rows_of_another_group_or_without_a_partition_number_are_skipped():
    text = TABLE + "other  t1  0  0  9  9  -  -  -\ng  t2  x  0  9  9  -  -  -\n"
    assert e2e.parse_consumer_groups_describe(text) == {"t1": 2}              # the group of the first row
    assert e2e.parse_consumer_groups_describe(text, group="g") == {"t1": 2}
    assert e2e.parse_consumer_groups_describe(text, group="other") == {"t1": 9}


def test_mvp_039_provision_output_is_the_user_id_and_the_set_password_url():
    uid = UUID("6f1c1f8e-6a7e-4f3b-9d6e-0c1b2a3d4e5f")
    text = f"user_id={uid}\nset_password_url=http://app.e2e.invalid/set-password?token=Tk_123-abc\n"
    assert e2e.parse_provision_output(text) == (uid, "http://app.e2e.invalid/set-password?token=Tk_123-abc")


@pytest.mark.parametrize("text", [
    "user_id=6f1c1f8e-6a7e-4f3b-9d6e-0c1b2a3d4e5f\n",
    "user_id=not-a-uuid\nset_password_url=http://x/set-password?token=SECRET-TOKEN\n",
    "user_id=6f1c1f8e-6a7e-4f3b-9d6e-0c1b2a3d4e5f\nset_password_url=http://x/set-password?token=SECRET-TOKEN\nextra=1\n",
    "set_password_url=http://x/set-password?token=SECRET-TOKEN\nuser_id=6f1c1f8e-6a7e-4f3b-9d6e-0c1b2a3d4e5f\n",
    # a structlog JSON line on stdout that carries the link, and a ':' separator instead of '='
    '{"event":"account_setup_link_issued","link":"http://x/account-setup/SECRET-TOKEN"}\n'
    "user_id=6f1c1f8e-6a7e-4f3b-9d6e-0c1b2a3d4e5f\nset_password_url=http://x/set-password?token=SECRET-TOKEN\n",
    "user_id=6f1c1f8e-6a7e-4f3b-9d6e-0c1b2a3d4e5f\nset_password_url: http://x/account-setup/SECRET-TOKEN\n",
])
def test_mvp_039_provision_output_must_be_exactly_two_lines_and_never_echoes_the_token(text):
    with pytest.raises(ValueError) as info:
        e2e.parse_provision_output(text)
    assert "SECRET-TOKEN" not in str(info.value)


def _hop_text(response: httpx.Response, statuses: tuple[int, ...]) -> str:
    """The text a failing _expect puts into the HopFailure (and so into the CI log)."""
    with pytest.raises(e2e_manifest.HopFailure) as info, e2e_manifest.hop("account_setup"):
        e2e._expect(response, statuses, "account-setup")
    return str(info.value)


def test_mvp_039_an_unexpected_response_never_echoes_its_body():
    # FastAPI's default 422 echoes the request (here the one-time token and the new password) in detail[].input
    body = {"detail": [{"type": "missing", "loc": ["body", "password"], "msg": "Field required",
                        "input": {"token": "ONE-TIME-TOKEN", "new_password": "Aa1-SECRET-PASSWORD"}}]}
    text = _hop_text(httpx.Response(422, json=body), (200, 204))
    assert "HTTP 422" in text and "missing at body.password" in text
    assert "ONE-TIME-TOKEN" not in text and "Aa1-SECRET-PASSWORD" not in text
    tokens = _hop_text(httpx.Response(200, json={"access_token": "JWT-SECRET", "refresh_token": "R-SECRET"}), (201,))
    assert "HTTP 200" in tokens and "JWT-SECRET" not in tokens and "R-SECRET" not in tokens
    plain = _hop_text(httpx.Response(500, text="Traceback ... token=PLAIN-SECRET"), (200,))
    assert "HTTP 500" in plain and "PLAIN-SECRET" not in plain


def test_mvp_039_extract_link_finds_the_url_by_path_prefix():
    text = ("Hello,\nsee https://app.e2e.invalid/help for help.\n"
            "Acknowledge: http://api.e2e.invalid/api/v1/notifications/crisis-alerts/ack/abc123?sig=x1.\n"
            "(or <http://api.e2e.invalid/other>)")
    assert (e2e.extract_link(text, "/api/v1/notifications/crisis-alerts/ack/")
            == "http://api.e2e.invalid/api/v1/notifications/crisis-alerts/ack/abc123?sig=x1")
    assert e2e.extract_link(text, "/other") == "http://api.e2e.invalid/other"
    with pytest.raises(LookupError, match="/verify-email"):
        e2e.extract_link(text, "/verify-email")


def test_mvp_039_new_email_matches_the_mailpit_allow_list():
    for kind in ("user", "clin", "ops", "fallback", "reject"):
        address = e2e.new_email(kind)
        local, domain = address.split("@")
        assert domain == "test.invalid" and local.startswith(f"{kind}-") and len(local) == len(kind) + 13
    assert e2e.new_email("user") != e2e.new_email("user")
    with pytest.raises(ValueError):
        e2e.new_email("admin")


# --------------------------------------------------------------------------------------------- config

_E2E_VARS = ("SOLACE_E2E_PROFILE", "SOLACE_E2E_ENVIRONMENT", "SOLACE_E2E_LANE", "SOLACE_E2E_KEEP", "SOLACE_E2E_REUSE",
             "SOLACE_E2E_RUN_DIR")


@pytest.fixture
def clean_e2e_env(monkeypatch):
    for name in _E2E_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_mvp_039_stack_config_defaults_to_the_dev_profile_of_lane_a(clean_e2e_env, tmp_path):
    import tempfile

    for name in ("TEMP", "TMP", "TMPDIR"):
        clean_e2e_env.setenv(name, str(tmp_path))
    clean_e2e_env.setattr(tempfile, "tempdir", None)
    cfg = e2e.StackConfig.from_env()
    assert (cfg.profile, cfg.environment, cfg.lane, cfg.keep, cfg.reuse) == ("dev", "development", "a", False, False)
    assert cfg.project == "solace-e2e-a"
    assert cfg.run_dir == tmp_path / "solace-e2e" / "solace-e2e-a"
    assert [Path(f).name for f in cfg.compose_files] == ["docker-compose.yml", "docker-compose.test.yml"]
    assert all(Path(f).is_absolute() for f in cfg.compose_files)


def test_mvp_039_stack_config_strict_defaults_to_production_and_adds_the_prod_file(clean_e2e_env, tmp_path):
    for name, value in {"SOLACE_E2E_PROFILE": "strict", "SOLACE_E2E_LANE": "b2", "SOLACE_E2E_KEEP": "1",
                        "SOLACE_E2E_REUSE": "off", "SOLACE_E2E_RUN_DIR": str(tmp_path / "rd")}.items():
        clean_e2e_env.setenv(name, value)
    cfg = e2e.StackConfig.from_env()
    assert (cfg.profile, cfg.environment, cfg.project, cfg.keep, cfg.reuse) == ("strict", "production", "solace-e2e-b2", True, False)
    assert cfg.run_dir == tmp_path / "rd"
    assert [Path(f).name for f in cfg.compose_files] == ["docker-compose.yml", "docker-compose.prod.yml", "docker-compose.test.yml"]
    clean_e2e_env.setenv("SOLACE_E2E_ENVIRONMENT", "staging")
    assert e2e.StackConfig.from_env().environment == "staging"


@pytest.mark.parametrize("env", [
    {"SOLACE_E2E_PROFILE": "prod"},
    {"SOLACE_E2E_PROFILE": "strict", "SOLACE_E2E_ENVIRONMENT": "development"},
    {"SOLACE_E2E_PROFILE": "dev", "SOLACE_E2E_ENVIRONMENT": "production"},
    {"SOLACE_E2E_LANE": "Lane A"},
    {"SOLACE_E2E_LANE": ""},
])
def test_mvp_039_stack_config_rejects_bad_settings(clean_e2e_env, env):
    for key, value in env.items():
        clean_e2e_env.setenv(key, value)
    with pytest.raises(ValueError):
        e2e.StackConfig.from_env()


def test_mvp_039_compose_runs_from_a_clean_environment():
    # A shell ENVIRONMENT=test (the pytest process) or a dev secret must never reach compose interpolation.
    environ = {"PATH": "/bin", "SYSTEMROOT": "C:\\Windows", "DOCKER_HOST": "npipe:////./pipe/docker_engine",
               "ENVIRONMENT": "test", "POSTGRES_PASSWORD": "leak", "COMPOSE_FILE": "x.yml", "AUTH_SECRET_KEY": "leak",
               "SOLACE_IMAGE_TAG": "ci-123", "HTTPS_PROXY": "http://proxy:3128"}
    env = e2e._compose_env(environ)
    assert env == {"PATH": "/bin", "SYSTEMROOT": "C:\\Windows", "DOCKER_HOST": "npipe:////./pipe/docker_engine",
                   "SOLACE_IMAGE_TAG": "ci-123", "HTTPS_PROXY": "http://proxy:3128"}


def test_mvp_039_env_file_values_are_unescaped_for_harness_use(tmp_path):
    path = tmp_path / "e2e.env"
    path.write_text('# header without an equals sign\nBASIC_AUTH_HASH="$$2b$$10$$abc"\nPLAIN=v=1\nEMPTY=""\n',
                    encoding="utf-8")
    assert e2e.read_env_file(path) == {"BASIC_AUTH_HASH": "$2b$10$abc", "PLAIN": "v=1", "EMPTY": ""}


# --------------------------------------------------------------------------------------------- scrubbing

_REDIS_PASSWORD = "redis-pw-" + "a" * 24
_SECRET_VALUES = {
    "REDIS_PASSWORD": _REDIS_PASSWORD,
    "REDIS_URL": f"redis://:{_REDIS_PASSWORD}@redis:6379/0",   # holds another secret: the longer one goes first
    "AUTH_SECRET_KEY": "auth-" + "b" * 40,
    "BASIC_AUTH_HASH": "$2b$10$" + "c" * 53,
}


def _write_run_env(run_dir: Path) -> None:
    lines = ['ENVIRONMENT="production"', 'E2E_SMTP_TLS_MODE="starttls"']
    lines += [f'{k}="{v.replace("$", "$$")}"' for k, v in _SECRET_VALUES.items()]
    (run_dir / "e2e.env").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("stream", ["stderr", "stdout"])  # stdout is the message when stderr is empty
def test_mvp_039_compose_errors_are_scrubbed_of_every_generated_secret(tmp_path, monkeypatch, stream):
    _write_run_env(tmp_path)
    leak = " ".join(_SECRET_VALUES.values())
    script = f"import sys; sys.{stream}.write({leak!r} + ' production'); sys.exit(1)"
    monkeypatch.setattr(e2e, "_compose_argv", lambda project, env_file, files: [sys.executable, "-c", script])
    config = e2e.StackConfig("strict", "production", lane="zzut", run_dir=tmp_path)
    for failing in (lambda: e2e._compose(config, ["up", "-d"]),
                    lambda: e2e.compose_config(config.compose_files, config.env_file)):
        with pytest.raises(e2e.ComposeError) as info:
            failing()
        text = str(info.value)
        assert text.count("***") == len(_SECRET_VALUES) and "production" in text        # public values stay
        assert not any(value in text for value in _SECRET_VALUES.values())
        assert _REDIS_PASSWORD[:12] not in text                                           # no partial leftovers


def test_mvp_039_no_registry_secret_is_treated_as_public():
    data = yaml.safe_load((REPO / "deploy" / "env" / "env-contract.yaml").read_text(encoding="utf-8"))
    secret_sources = {row.get("source") or row["name"] for row in data["variables"] if row.get("secret")}
    secret_sources |= {"E2E_BASIC_AUTH_PASSWORD", "REDIS_URL"}                               # generator extras
    assert secret_sources and not secret_sources & e2e._PUBLIC_ENV_KEYS


# --------------------------------------------------------------------------------------------- the run dir

@pytest.mark.parametrize("make", [
    lambda: e2e.ROOT, lambda: e2e.ROOT / "tests", lambda: e2e.ROOT.parent, lambda: Path.home(),
    lambda: Path(e2e.ROOT.anchor), lambda: Path(Path.home().anchor),
])
def test_mvp_039_run_dir_is_never_the_repo_home_or_a_drive_root(make):
    with pytest.raises(ValueError, match="run dir"):
        e2e.StackConfig("dev", "development", run_dir=make())


def _marked(path: Path) -> Path:
    path.mkdir()
    (path / e2e.RUN_MARKER).write_text("solace-e2e run dir\n", encoding="utf-8")
    (path / "e2e.env").write_text("", encoding="utf-8")
    return path


def test_mvp_039_only_a_marked_or_empty_run_dir_is_ever_deleted(tmp_path, monkeypatch):
    from scripts.test import gen_e2e_env

    assert e2e.RUN_MARKER == gen_e2e_env.RUN_MARKER
    monkeypatch.setattr(e2e, "_remove_project", lambda config: None)
    monkeypatch.setattr(e2e, "_generate_env", lambda config: (_ for _ in ()).throw(RuntimeError("generator reached")))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("operator data", encoding="utf-8")
    config = e2e.StackConfig("dev", "development", lane="zzut", run_dir=foreign)
    with pytest.raises(RuntimeError, match="not an e2e run dir"):
        e2e._teardown(config, strict=True)                                 # stop_stack and 'down'
    with pytest.raises(RuntimeError, match="not an e2e run dir"):
        e2e._start(config)                                                 # a fresh start clears the run dir first
    assert (foreign / "keep.txt").is_file()

    for directory in (_marked(tmp_path / "marked"), tmp_path / "empty"):
        directory.mkdir(exist_ok=True)
        e2e._teardown(replace(config, run_dir=directory), strict=True)
        assert not directory.exists()
    with pytest.raises(RuntimeError, match="generator reached"):          # marked: cleared, then generated anew
        e2e._start(replace(config, run_dir=_marked(tmp_path / "again")))
    assert not (tmp_path / "again").exists()


# --------------------------------------------------------------------------------------------- the e2e conftest

class _FakeItem:
    def __init__(self, path: Path, obj=None) -> None:
        self.path, self.obj, self._marks = path, obj, []

    def add_marker(self, marker, append: bool = True) -> None:
        mark = getattr(marker, "mark", marker)
        self._marks.append(mark) if append else self._marks.insert(0, mark)

    def iter_markers(self, name=None):
        return (m for m in self._marks if name is None or m.name == name)

    def get_closest_marker(self, name, default=None):
        return next(self.iter_markers(name), default)


def test_mvp_039_the_default_e2e_timeout_bounds_the_test_function_only():
    # The session fixture builds the stack inside the first test's setup, bounded by start_stack's own timeouts
    # (UP_TIMEOUT_S for a cold build); the default 900 s per-test timeout must not cover that setup.
    spec = importlib.util.spec_from_file_location("_e2e_conftest_under_test", REPO / "tests" / "e2e" / "conftest.py")
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    item = _FakeItem(REPO / "tests" / "e2e" / "test_probe.py")
    config = SimpleNamespace(pluginmanager=SimpleNamespace(hasplugin=lambda name: name == "timeout"))
    conftest.pytest_collection_modifyitems(config, [item])
    mark = item.get_closest_marker("timeout")
    assert mark.args == (900,) and mark.kwargs == {"func_only": True}
    assert item.get_closest_marker("e2e") and item.get_closest_marker("integration")


# --------------------------------------------------------------------------------------------- the overlay

def _overlay() -> dict:
    return e2e.load_compose_yaml(OVERLAY.read_text(encoding="utf-8"))


def test_mvp_039_overlay_allowed_keys_are_exactly_what_the_overlay_sets():
    services = _overlay()["services"]
    app_env = {key for name in e2e.MVP_SERVICES for key in services[name]["environment"]}
    app_service_keys = ({key for name in e2e.MVP_SERVICES for key in services[name]}
                        - set(e2e.OVERLAY_RESET_KEYS) - {"environment"})
    caddy_env = set(services["caddy"]["environment"])
    assert app_service_keys == {"secrets"}                     # the e2e CA mount, the one non-env key
    assert app_env | caddy_env | app_service_keys == set(e2e.OVERLAY_ALLOWED_KEYS)
    for name in e2e.MVP_SERVICES:
        assert set(services[name]["environment"]) == app_env, name
    assert services["user-service"]["environment"]["LLM_PORTKEY_GATEWAY_URL"] == "http://fake-llm:8080/v1"
    assert services["notification-service"]["environment"]["EMAIL_SMTP_HOST"] == "mailpit"


def test_mvp_039_overlay_never_touches_build_image_or_other_keys_of_a_base_service():
    services = _overlay()["services"]
    for name, spec in services.items():
        if name in e2e.OVERLAY_SERVICES:
            continue
        allowed = set(e2e.OVERLAY_RESET_KEYS)
        if name in e2e.MVP_SERVICES:
            allowed |= {"environment", "secrets"}
        elif name == "caddy":
            allowed |= {"environment"}
        assert set(e2e.OVERLAY_RESET_KEYS) <= set(spec) <= allowed, (name, set(spec) ^ allowed)
        assert spec["container_name"] is None and spec["env_file"] == []
        assert spec["ports"] == (["127.0.0.1::8080", "127.0.0.1::8081"] if name == "caddy" else []), name
    assert set(e2e.OVERLAY_SERVICES) == {"fake-llm", "mailpit"}


def test_mvp_039_overlay_mounts_the_e2e_ca_on_every_app_service_from_the_run_secrets_dir():
    doc = _overlay()
    for name in e2e.MVP_SERVICES:
        assert doc["services"][name]["secrets"] == [e2e.OVERLAY_APP_SECRET], name   # /run/secrets/e2e-ca.crt
    assert e2e.OVERLAY_APP_SECRET == "e2e-ca.crt"
    source = doc["secrets"][e2e.OVERLAY_APP_SECRET]["file"]
    assert source.startswith("${SOLACE_SECRETS_DIR:?") and source.endswith("}/e2e-ca.crt")
    # the env file points the services at exactly that mount in strict
    assert doc["services"]["user-service"]["environment"]["SSL_CERT_FILE"] == "${E2E_SSL_CERT_FILE:-}"


def test_mvp_039_overlay_test_doubles_publish_only_loopback_ephemeral_ports():
    services = _overlay()["services"]
    assert services["fake-llm"]["ports"] == ["127.0.0.1::8080"] and services["fake-llm"]["networks"] == ["backend"]
    mailpit = services["mailpit"]
    assert mailpit["ports"] == ["127.0.0.1::8025"] and set(mailpit["networks"]) == {"backend", "monitoring"}
    assert mailpit["image"].startswith("axllent/mailpit@sha256:") and len(mailpit["image"].split(":")[-1]) == 64
    # compose needs "$$" for the regex anchor; the container sees a single "$"
    assert mailpit["environment"]["MP_SMTP_ALLOWED_RECIPIENTS"] == r"^(user|clin|fallback|ops)-[A-Za-z0-9.-]*@test\.invalid$$"


COMPOSE_SNIPPET = """\
services:
  postgres:
    image: postgres:16-alpine
    container_name: !reset null
    ports: !reset []
    env_file: !reset
  redis:
    image: !override redis:7-alpine
    command: !override ["redis-server", "--appendonly", "yes"]
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    environment: !override {KAFKA_BROKER_ID: 1}
  mailhog:
    image: !reset
"""


def test_mvp_039_compose_yaml_loader_keeps_reset_and_override_values():
    doc = realinfra.load_compose_yaml(COMPOSE_SNIPPET)
    postgres, redis, kafka = (doc["services"][n] for n in ("postgres", "redis", "kafka"))
    assert postgres == {"image": "postgres:16-alpine", "container_name": None, "ports": [], "env_file": None}
    assert redis == {"image": "redis:7-alpine", "command": ["redis-server", "--appendonly", "yes"]}
    assert kafka["environment"] == {"KAFKA_BROKER_ID": 1} and doc["services"]["mailhog"] == {"image": None}
    assert e2e.load_compose_yaml is realinfra.load_compose_yaml           # one implementation
    # a subclass: plain yaml.safe_load is not patched and still refuses the compose tags
    assert "!reset" not in yaml.SafeLoader.yaml_constructors and "!override" not in yaml.SafeLoader.yaml_constructors
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("a: !reset null")


def test_mvp_039_image_refs_read_compose_tags(tmp_path):
    compose = tmp_path / "docker-compose.test.yml"
    compose.write_text(COMPOSE_SNIPPET, encoding="utf-8")
    refs = realinfra._image_refs([compose])
    assert refs["postgres"] == "postgres:16-alpine" and refs["redis"] == "redis:7-alpine"
    assert refs["kafka"] == "confluentinc/cp-kafka:7.5.0" and "mailhog" not in refs   # !reset image: no image
    assert refs["mailpit"] == realinfra.MAILPIT_FALLBACK_IMAGE


def test_mvp_039_realinfra_imports_on_its_own_and_pins_mailpit_from_the_overlay():
    # A fresh interpreter that imports realinfra first (no solace_testing.e2e): IMAGE_REFS is read from the overlay
    # too (its tags parse), and plain yaml.safe_load still refuses the tags afterwards. The overlay's digest equals
    # MAILPIT_FALLBACK_IMAGE today, so the value alone cannot show the overlay was read: COMPOSE_FILES must name it.
    code = ("import solace_testing.realinfra as r, sys, yaml; "
            "print(r.IMAGE_REFS['mailpit']); "
            "print('solace_testing.e2e' in sys.modules); "
            "print(r.COMPOSE_FILES[-1].as_posix()); "
            "yaml.safe_load('a: !reset null')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120)  # noqa: S603
    lines = proc.stdout.strip().splitlines()
    assert lines[:3] == [_overlay()["services"]["mailpit"]["image"], "False", OVERLAY.as_posix()], proc.stderr[-2000:]
    assert proc.returncode != 0 and "ConstructorError" in proc.stderr


def test_mvp_039_image_refs_take_mailpit_from_the_overlay_file(tmp_path):
    # the real overlay text with a digest that differs from the fallback: _image_refs must return that one
    digest = "sha256:" + "1" * 64
    text = OVERLAY.read_text(encoding="utf-8")
    pinned = _overlay()["services"]["mailpit"]["image"]
    assert pinned in text and digest not in pinned
    overlay = tmp_path / "docker-compose.test.yml"
    overlay.write_text(text.replace(pinned, f"axllent/mailpit@{digest}"), encoding="utf-8")
    refs = realinfra._image_refs([e2e.BASE_FILE, overlay])
    assert refs["mailpit"] == f"axllent/mailpit@{digest}" != realinfra.MAILPIT_FALLBACK_IMAGE


def test_mvp_039_tests_realinfra_collects_in_its_own_process():
    # The CI realinfra tier runs tests/realinfra as its own pytest process; its conftest imports realinfra while
    # docker-compose.test.yml (with !reset tags) is present. --collect-only: no fixture runs, no container starts.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SOLACE_", "PYTEST_"))}
    env.update({"PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src"))), "ENVIRONMENT": "test"})
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/realinfra/realinfra-harness-e2e", "--collect-only",  # noqa: S603
                           "-q", "-o", "addopts=", "-p", "no:logging", "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=REPO, env=env, timeout=300)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "ConstructorError" not in output, output[-3000:]
    assert " collected" in output


# --------------------------------------------------------------------------------------------- manifest-driven tolerance

_MANIFEST = """version: 1
entries:
  - test_id: "tests/e2e/test_mvp_039_stack_selftest.py::test_mvp_039_call_reaches_each_service[notification-service]"
    expected_failing_hop: service_live
    target_wave: 0
    reason: cannot boot before its W0 phase-B adoption (MVP-006)
  - test_id: "tests/e2e/test_mvp_039_stack_selftest.py::test_mvp_039_call_reaches_each_service[memory-service]"
    expected_failing_hop: service_live
    target_wave: 1
    reason: r
    profiles: [strict]
  - test_id: "tests/e2e/test_mvp_039_stack_selftest.py::test_mvp_039_call_reaches_each_service[therapy-service]"
    expected_failing_hop: register
    target_wave: 1
    reason: another hop does not excuse a dead service
shortcuts: []
"""


def test_mvp_039_only_services_the_manifest_expects_down_may_be_down(tmp_path):
    (tmp_path / "expected_failures.yaml").write_text(_MANIFEST, encoding="utf-8")
    down = e2e.services_allowed_down
    assert down("dev", manifest_dir=tmp_path, environ={}) == frozenset({"notification-service"})
    assert down("strict", manifest_dir=tmp_path, environ={}) == frozenset({"notification-service", "memory-service"})
    # due at the barrier of its target wave: the stack must bring the service up
    assert down("strict", manifest_dir=tmp_path, environ={"SOLACE_BARRIER_WAVE": "0"}) == frozenset({"memory-service"})
    assert down("strict", manifest_dir=tmp_path, environ={"SOLACE_BARRIER_WAVE": "1"}) == frozenset()
    assert down("dev", manifest_dir=tmp_path / "none", environ={}) == frozenset()


# --------------------------------------------------------------------------------------------- in-container scripts

class _Echo(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        status = 418 if self.path.startswith("/teapot") else 200
        out = json.dumps({"method": self.command, "path": self.path, "body": body.decode(),
                          "x": self.headers.get("X-Test"), "ct": self.headers.get("Content-Type")}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's name
        self._reply()

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's name
        self._reply()


@pytest.fixture
def echo_port():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _run_call(argv):
    assert argv[0] == "python"
    return subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True, timeout=60)  # noqa: S603


def test_mvp_039_call_script_makes_one_request_and_reports_status_and_body(echo_port):
    argv = e2e._call_argv(echo_port, "POST", "/api/v1/x?q=1", json={"message": "hi \"there\" 'x' $HOME"},
                          headers={"X-Test": "yes"}, timeout=5)
    assert all(arg.isascii() and "\n" not in arg and '"' not in arg for arg in argv[2:]), argv
    proc = _run_call(argv)
    result = e2e._parse_call_output(proc.stdout)
    assert result.status == 200
    assert result.json["method"] == "POST" and result.json["path"] == "/api/v1/x?q=1" and result.json["x"] == "yes"
    assert json.loads(result.json["body"]) == {"message": "hi \"there\" 'x' $HOME"} and result.json["ct"] == "application/json"
    teapot = e2e._parse_call_output(_run_call(e2e._call_argv(echo_port, "GET", "/teapot", timeout=5)).stdout)
    assert teapot.status == 418 and teapot.json["method"] == "GET"


def test_mvp_039_call_script_reports_a_refused_connection(echo_port):
    proc = _run_call(e2e._call_argv(1, "GET", "/live", timeout=2))
    with pytest.raises(ConnectionError):
        e2e._parse_call_output(proc.stdout)


def test_mvp_039_sql_wraps_the_query_and_binds_parameters_as_psql_variables():
    argv, stdin = e2e._psql_invocation("SELECT id FROM users WHERE email = :'p1' AND role = :'p2';", ("a@test.invalid", "x'y"))
    assert argv[:2] == ["psql", "-X"] and "ON_ERROR_STOP=1" in argv and "postgres" in argv and "solace" in argv
    assert argv[argv.index("p1=a@test.invalid") - 1] == "-v" and "p2=x'y" in argv
    assert stdin.strip() == ("WITH t AS (SELECT id FROM users WHERE email = :'p1' AND role = :'p2') "
                             "SELECT coalesce(json_agg(t), '[]'::json) FROM t;")
