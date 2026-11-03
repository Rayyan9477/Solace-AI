# tests/deploy/test_mvp_118_redis_auth.py
"""MVP-118: Redis always requires a password, every client gets it, and the healthcheck fails on NOAUTH."""
import pytest
from deploy.env.compose_render import ComposeRenderError
from deploy.env.registry import APP_SERVICES

LEGACY = {"user-service": {"USER_REDIS_HOST": "redis", "USER_REDIS_PASSWORD": None},
          "therapy-service": {"THERAPY_SESSION_REDIS_HOST": "redis", "THERAPY_SESSION_REDIS_PASSWORD": None},
          "diagnosis-service": {"DIAGNOSIS_SESSION_REDIS_HOST": "redis", "DIAGNOSIS_SESSION_REDIS_PASSWORD": None},
          "memory-service": {"MEMORY_REDIS_HOST": "redis", "MEMORY_REDIS_PASSWORD": None, "REDIS_PASSWORD": None}}

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_redis_requires_a_password(render, profile):
    redis = render(profile).services["redis"]
    assert "requirepass" in " ".join(redis["command"]) and redis["environment"]["REDIS_PASSWORD"]
    hc = " ".join(redis["healthcheck"]["test"])
    assert "PONG" in hc and "REDISCLI_AUTH" in hc

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_every_redis_client_gets_the_password(render, profile):
    r = render(profile)
    pw = r.services["redis"]["environment"]["REDIS_PASSWORD"]
    for svc in APP_SERVICES:
        assert r.services[svc]["environment"]["REDIS_URL"] == f"redis://:{pw}@redis:6379/0", svc
    for svc, expected in LEGACY.items():
        env = r.services[svc]["environment"]
        for name, value in expected.items():
            assert env[name] == (pw if value is None else value), f"{svc}:{name}"

def test_strict_render_requires_REDIS_PASSWORD(render):
    with pytest.raises(ComposeRenderError, match="REDIS_PASSWORD"):
        render("strict", drop=["REDIS_PASSWORD"])

@pytest.mark.integration
@pytest.mark.e2e
def test_healthcheck_needs_auth_and_noauth_is_rejected(compose_project):
    compose_project.up(["redis"])
    assert "NOAUTH" in compose_project.exec("redis", "redis-cli", "ping")
    assert compose_project.exec("redis", "sh", "-c", 'REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli ping').strip() == "PONG"

# --- WS-03.3 additions ------------------------------------------------------------------------------

# Every Redis name a service reads today (the brief's full list), each derived from the one source.
LEGACY_ALL = {
    "user-service": ("USER_REDIS",),
    "diagnosis-service": ("DIAGNOSIS_REDIS", "DIAGNOSIS_SESSION_REDIS"),
    "therapy-service": ("THERAPY_REDIS", "THERAPY_SESSION_REDIS"),
    "personality-service": ("PERSONALITY_REDIS",),
    "memory-service": ("MEMORY_REDIS", "REDIS"),
}

def _redis_mismatches(r):
    """Names (never values) of every Redis client setting that differs from the redis container's."""
    pw = r.services["redis"]["environment"]["REDIS_PASSWORD"]
    url = f"redis://:{pw}@redis:6379/0"
    bad = [f"{svc}:REDIS_URL" for svc in APP_SERVICES if r.services[svc]["environment"].get("REDIS_URL") != url]
    for svc, prefixes in LEGACY_ALL.items():
        env = r.services[svc]["environment"]
        for prefix in prefixes:
            for suffix, expected in (("HOST", "redis"), ("PORT", "6379"), ("PASSWORD", pw)):
                if env.get(f"{prefix}_{suffix}") != expected:
                    bad.append(f"{svc}:{prefix}_{suffix}")
    if r.services["safety-service"]["environment"].get("SAFETY_REPO_REDIS_URL") != url:
        bad.append("safety-service:SAFETY_REPO_REDIS_URL")
    return bad

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_every_legacy_redis_name_follows_the_one_source(render, profile):
    # one REDIS_PASSWORD source reaches every client, so a rotation cannot leave one behind
    assert _redis_mismatches(render(profile)) == []

@pytest.mark.integration
@pytest.mark.e2e
def test_redis_runs_unprivileged_with_the_password_off_argv(compose_project):
    # the image's entrypoint steps down to the redis user only when it is handed redis-server itself
    compose_project.up(["redis"])
    pw = compose_project.exec("redis", "sh", "-c", 'printf %s "$REDIS_PASSWORD"')
    assert pw, "the redis container must see REDIS_PASSWORD"
    uid = compose_project.exec("redis", "sh", "-c", "awk '/^Uid:/{print $2}' /proc/1/status").strip()
    assert uid not in ("", "0"), "redis-server must not run as root"
    argv = compose_project.exec("redis", "sh", "-c", "tr '\\0' ' ' < /proc/1/cmdline")
    assert "redis-server" in argv and pw not in argv, "the password must not be on redis-server's argv"
    conf = compose_project.exec("redis", "sh", "-c", "stat -c '%a %U' /tmp/redis.conf").strip()
    assert conf == "600 redis", "the generated config is private to the redis user"

# --- WS-03.3 fix round 1 (review batch18: WS033-SEC-1, WS033-SPE-1) --------------------------------

URL_SAFE_GUARD = "*[!A-Za-z0-9_-]*"

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_redis_refuses_a_password_that_is_not_url_safe_before_writing_its_config(render, profile):
    # redis-server echoes a bad requirepass line (password included) and a '/' or '+' breaks the
    # clients' URL parsing, so the command fails closed first, with a fixed message
    script = render(profile).services["redis"]["command"][-1]
    assert URL_SAFE_GUARD in script and "exit 1" in script
    assert script.index(URL_SAFE_GUARD) < script.index("printf")
    assert "REDIS_PASSWORD must be URL-safe [A-Za-z0-9_-]" in script

@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.parametrize("bad", ["has/a/slash-in-it", "has a space in it", "has+plus+signs"])
def test_redis_with_a_password_that_is_not_url_safe_fails_closed(compose_project, bad):
    from deploy.env.compose_render import format_env_file
    with compose_project.env_file.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(format_env_file({"REDIS_PASSWORD": bad}))
    proc = compose_project.run("redis", timeout=90)  # a server that starts instead of refusing never exits
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "REDIS_PASSWORD must be URL-safe" in out
    assert bad not in out, "the refused value must never be echoed"

# App services that read Redis while starting. Derived, so the list cannot drift: every IF-01 SPEC
# with uses_redis=True (start_service_runtime pings REDIS_URL before serving; strict exits without it),
# plus the readers outside IF-01: user-service builds its Redis-backed stores in its own lifespan.
NON_IF01_STARTUP_REDIS_READERS = {"user-service"}

def _uses_redis_specs():
    """Compose names of the app services whose main.py builds ServiceRuntimeSpec(..., uses_redis=True)."""
    import ast
    from pathlib import Path
    from scripts.ci.images import IMAGES
    from tests.deploy.conftest import REPO
    found = set()
    for image in IMAGES:
        if image.name not in APP_SERVICES:
            continue
        main = REPO / Path(image.dockerfile).parent / "src" / "main.py"
        tree = ast.parse(main.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "ServiceRuntimeSpec":
                if any(k.arg == "uses_redis" and isinstance(k.value, ast.Constant) and k.value.value is True
                       for k in node.keywords):
                    found.add(image.name)
    return found

def test_the_redis_reader_scan_finds_the_if01_services():
    # guards the scan itself: these SPECs set uses_redis=True today (phase-B adoptions)
    assert {"safety-service", "therapy-service", "diagnosis-service", "memory-service",
            "orchestrator-service"} <= _uses_redis_specs()

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_services_that_read_redis_at_startup_wait_for_it(render, profile):
    services = render(profile).services
    readers = _uses_redis_specs() | NON_IF01_STARTUP_REDIS_READERS
    missing = sorted(s for s in readers
                     if services[s].get("depends_on", {}).get("redis", {}).get("condition") != "service_healthy")
    assert missing == []

# --- WS-03.3 fix rounds 2 and 3 (review batch25 WS033F1-CLO-2, batch27 WS033F2-CLO-1) ----------------

def test_run_reaps_a_cli_that_ignores_every_timeout(monkeypatch):
    # the compose CLI neither exits after its one-off container is removed nor after kill: run() must
    # still give up after bounded waits and raise, saying the CLI outlived the kill
    import subprocess
    from tests.deploy.conftest import ComposeProject, ComposeProjectError

    calls = []

    class _StuckCli:
        def __init__(self, *args, **kwargs):
            self.stdout = self.stderr = None
        def kill(self):
            calls.append("kill")
        def wait(self, timeout=None):
            calls.append(f"wait:{timeout}")
            raise subprocess.TimeoutExpired("docker compose run", timeout)

    monkeypatch.setattr(subprocess, "Popen", _StuckCli)
    monkeypatch.setattr(ComposeProject, "_labelled", lambda self, *args, **kwargs: [])
    project = ComposeProject("solace-deploytest-unit", "unused.env")
    project.run_stop_grace = 2
    with pytest.raises(ComposeProjectError, match=r"did not exit within 1s .*did not exit after kill"):
        project.run("redis", timeout=1)
    assert calls == ["wait:1", "wait:2", "kill", "wait:10"]

_HOLDER = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
_PARENT = ("import subprocess, sys, time; "
           "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], stdout=sys.stdout, stderr=sys.stderr); "
           "print('parent started', flush=True); time.sleep(float(sys.argv[3]))")

def _run_with_a_grandchild_holding_the_output(monkeypatch, tmp_path, parent_seconds):
    """run() a REAL parent that leaves a grandchild holding its inherited stdout/stderr (the compose
    plugin case). Returns (hung, result, error); a thread plus join detects a hang without waiting forever."""
    import os
    import signal
    import sys
    import threading
    import time
    from tests.deploy.conftest import ComposeProject

    pid_file = tmp_path / "holder.pid"
    monkeypatch.setattr(ComposeProject, "_cmd", lambda self, args, profiles=():
                        [sys.executable, "-c", _PARENT, _HOLDER, str(pid_file), str(parent_seconds)])
    monkeypatch.setattr(ComposeProject, "_labelled", lambda self, *args, **kwargs: [])
    project = ComposeProject("solace-deploytest-unit", tmp_path / "unused.env")
    project.run_stop_grace = 1
    box = {}

    def target():
        try:
            box["result"] = project.run("redis", timeout=1)
        except BaseException as exc:  # noqa: BLE001 - handed back to the test
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    try:
        worker.join(20)
        return worker.is_alive(), box.get("result"), box.get("error")
    finally:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)  # TerminateProcess on Windows
            except (OSError, ValueError):
                pass
        worker.join(10)

def test_run_returns_when_a_grandchild_keeps_its_output_open(monkeypatch, tmp_path):
    hung, result, error = _run_with_a_grandchild_holding_the_output(monkeypatch, tmp_path, parent_seconds=0)
    assert not hung, "run() hung on output a grandchild still held"
    assert error is None and result.returncode == 0 and "parent started" in result.stdout

def test_run_raises_when_the_cli_outlives_its_timeout_and_a_grandchild_holds_its_output(monkeypatch, tmp_path):
    from tests.deploy.conftest import ComposeProjectError
    hung, result, error = _run_with_a_grandchild_holding_the_output(monkeypatch, tmp_path, parent_seconds=60)
    assert not hung, "run() hung on output a grandchild still held"
    assert isinstance(error, ComposeProjectError) and "did not exit within 1s" in str(error)
