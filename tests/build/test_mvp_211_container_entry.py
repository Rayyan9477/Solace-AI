"""MVP-211: bearer tokens in query strings never reach container logs; the IF-22 entrypoint loads NAME_FILE secrets."""
from __future__ import annotations

import asyncio
import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import websockets

from scripts.ci import dockerfiles
from scripts.ci.images import IMAGES, probe_env

ROOT = Path(__file__).resolve().parents[2]
APP_IMAGES = [i for i in IMAGES if i.module]
WS_TOKEN, VERIFY_TOKEN = "SYNTHETIC-WS-TOKEN-7f3a", "SYNTHETIC-VERIFY-TOKEN-91c2"


@pytest.mark.parametrize("image", APP_IMAGES, ids=lambda i: i.name)
def test_mvp_211_cmd_flags_and_entry(image) -> None:
    df = dockerfiles.parse(ROOT / image.dockerfile)
    assert "--no-access-log" in df.cmd and "--proxy-headers" in df.cmd
    assert df.cmd[df.cmd.index("--log-level") + 1] == "warning"
    assert df.entrypoint == ["/app/docker/entrypoint.sh"]
    assert df.healthcheck and "/live" in df.healthcheck and "/health" not in df.healthcheck
    if image.name == "orchestrator-service":
        assert df.cmd[df.cmd.index("--ws-max-size") + 1] == "16384"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mvp_211_image_uvicorn_flags_never_log_query_tokens() -> None:
    flags = dockerfiles.uvicorn_flags(dockerfiles.parse(ROOT / "services/orchestrator_service/Dockerfile"))
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ws_probe_app:app", "--app-dir", str(Path(__file__).parent),
         "--host", "127.0.0.1", "--port", str(port), *flags],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)

        async def _ws() -> None:
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/session-1?token={WS_TOKEN}") as ws:
                await ws.send("ping")
                await ws.recv()

        asyncio.run(_ws())
        httpx.post(f"http://127.0.0.1:{port}/auth/verify-email?token={VERIFY_TOKEN}&user_id=u-1", timeout=5)
    finally:
        proc.terminate()
        out = proc.communicate(timeout=15)[0]
    assert WS_TOKEN not in out and VERIFY_TOKEN not in out


def test_f4_uvicorn_flags_boolean_flag_before_app_does_not_swallow_the_app() -> None:
    flags = dockerfiles.uvicorn_flags(["uvicorn", "--proxy-headers", "main:app", "--port", "1"])
    assert flags == ["--proxy-headers"]
    flags = dockerfiles.uvicorn_flags(["uvicorn", "--log-level", "warning", "main:app", "--host", "h", "--no-access-log"])
    assert flags == ["--log-level", "warning", "--no-access-log"]


def test_f4_uvicorn_flags_value_taking_option_with_app_shaped_value() -> None:
    flags = dockerfiles.uvicorn_flags(["uvicorn", "--header", "server:x", "main:app", "--proxy-headers"])
    assert flags == ["--header", "server:x", "--proxy-headers"]


def test_f4_uvicorn_flags_boolean_flag_before_a_dotted_app_attribute() -> None:
    """WS01F4F1-CLO-1: uvicorn accepts `module:attr.sub`; a boolean flag before it must not take it as a value."""
    flags = dockerfiles.uvicorn_flags(["uvicorn", "--proxy-headers", "main:app.asgi", "--log-level", "info"])
    assert flags == ["--proxy-headers", "--log-level", "info"]
    flags = dockerfiles.uvicorn_flags(["uvicorn", "--factory", "pkg.mod:make.app", "--workers", "2"])
    assert flags == ["--factory", "--workers", "2"]


def _loader_copies(image) -> list:
    df = dockerfiles.parse(ROOT / image.dockerfile)
    return [c for c in df.copies if c.dest.endswith("/docker/entrypoint.sh") and c.stage == len(df.from_images) - 1]


# Every registered image whose final stage ships the entrypoint loader (the nine services and the migrate job).
LOADER_IMAGES = [i for i in IMAGES if _loader_copies(i)]


def test_f4_every_service_and_the_migrate_job_ship_the_loader() -> None:
    assert len(LOADER_IMAGES) == 10 and "migrate" in {i.name for i in LOADER_IMAGES}


@pytest.mark.parametrize("image", LOADER_IMAGES, ids=lambda i: i.name)
def test_f4_entrypoint_copy_is_not_chowned_to_the_app_user(image) -> None:
    df = dockerfiles.parse(ROOT / image.dockerfile)
    loader = _loader_copies(image)
    assert len(loader) == 1 and "chown" not in loader[0].flags
    assert df.user == "appuser"


@pytest.mark.parametrize("image", LOADER_IMAGES, ids=lambda i: i.name)
def test_f4_runtime_code_is_root_owned_and_bytecode_is_not_written(image) -> None:
    df = dockerfiles.parse(ROOT / image.dockerfile)
    owned = [c for c in df.runtime_copies if c.stage == len(df.from_images) - 1 and "chown" in c.flags]
    assert not owned, f"appuser must not own code: {[c.sources for c in owned]}"
    assert df.env.get("PYTHONDONTWRITEBYTECODE") == "1"


def test_f4_orchestrator_final_stage_opts_in_to_strict_msgpack() -> None:
    df = dockerfiles.parse(ROOT / "services/orchestrator_service/Dockerfile")  # env describes the final stage
    assert df.env.get("LANGGRAPH_STRICT_MSGPACK") == "true"


def _selected_images() -> list:
    only = {n for n in os.environ.get("SOLACE_F4_IMAGES", "").split(",") if n}
    return [i for i in LOADER_IMAGES if not only or i.name in only]


def _docker_run(tag: str, *argv: str, user: str | None = None, env: dict[str, str] | None = None):
    cmd = ["docker", "run", "--rm", "--entrypoint", argv[0]]
    cmd += ["--user", user] if user else []
    for key, value in (env or {}).items():
        cmd += ["-e", f"{key}={value}"]
    return subprocess.run([*cmd, tag, *argv[1:]], capture_output=True, text=True, timeout=300)


@pytest.mark.realinfra
@pytest.mark.parametrize("image", _selected_images(), ids=lambda i: i.name)
def test_f4_built_image_code_and_loader_are_root_owned_and_unwritable(require_docker, image) -> None:
    tag = f"solace-f4-{image.name}:test"
    build = subprocess.run(["docker", "build", "-q", "-f", image.dockerfile, "-t", tag, "."], cwd=ROOT,
                           capture_output=True, text=True, timeout=1800)
    try:
        assert build.returncode == 0, build.stderr[-2000:]
        run = _docker_run(tag, "stat", "-c", "%U", "/app/docker/entrypoint.sh", "/app/docker", "/app")
        assert run.returncode == 0, run.stderr
        assert run.stdout.split() == ["root", "root", "root"]
        # As the app user nothing under /app is writable, and nothing is owned by anyone but root.
        probe = ("for d in /app /app/src /app/services /app/docker; do [ -d $d ] && test -w $d && echo W:$d; done; "
                 "find /app -writable; find /app ! -user root")
        run = _docker_run(tag, "sh", "-c", probe, user="appuser")
        assert run.returncode == 0, run.stderr
        assert run.stdout.strip() == "", run.stdout[:500]
        if image.name in ("orchestrator-service", "memory-service"):
            code = "; ".join(f"import {m}" for m in (image.module, *image.probes))
            run = _docker_run(tag, "python", "-c", code, user="appuser", env=probe_env())
            assert run.returncode == 0, run.stderr[-1500:]
    finally:
        subprocess.run(["docker", "rmi", tag], capture_output=True, text=True)


def _shell_scripts_shipped() -> set[Path]:
    """Every *.sh in the repository outside VCS, virtualenv and dependency directories (which covers the compose
    bind mounts, directory mounts such as infrastructure/init-db included, and every Dockerfile COPY)."""
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__"}
    found: set[Path] = set()
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        found.update(Path(base) / f for f in files if f.endswith(".sh"))
    return found


def test_f4_shell_scripts_keep_lf_line_endings() -> None:
    rules = [ln.split() for ln in (ROOT / ".gitattributes").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert any(r[0] == "*.sh" and "eol=lf" in r[1:] for r in rules), ".gitattributes must give *.sh eol=lf"
    scripts = _shell_scripts_shipped() | {ROOT / "docker/entrypoint.sh", ROOT / "scripts/db/pg_tls_entrypoint.sh"}
    assert ROOT / "infrastructure/init-db/01-roles.sh" in scripts and len(scripts) >= 3
    for script in sorted(scripts):
        data = script.read_bytes()
        assert b"\r\n" not in data and not data.startswith(b"\xef\xbb\xbf"), f"{script.relative_to(ROOT)} is not clean LF"


def _run_entrypoint(tmp_path: Path, env: dict[str, str], command: list[str]):
    import docker

    image = dockerfiles.parse(ROOT / "services/orchestrator_service/Dockerfile").from_images[0]
    client = docker.from_env()
    container = client.containers.run(
        image, ["sh", "/entry/entrypoint.sh", *command], detach=True, environment=env,
        volumes={str(ROOT / "docker"): {"bind": "/entry", "mode": "ro"},
                 str(tmp_path): {"bind": "/run/secrets", "mode": "ro"}})
    try:
        status = container.wait(timeout=60)["StatusCode"]
        return status, container.logs().decode("utf-8", "replace")
    finally:
        container.remove(force=True)


PRINT_SHA = ["python", "-c", "import hashlib, os; print(hashlib.sha256(os.environ['AUTH_SECRET_KEY'].encode()).hexdigest()); "
             "print('provenance=' + os.environ.get('AUTH_SECRET_KEY_FILE', 'unset'))"]


@pytest.mark.realinfra
@pytest.mark.parametrize("content,expected", [("v1\n", "v1"), ("v1\n\n", "v1\n"),
                                              ("-----BEGIN K-----\nabc\n-----END K-----\n", "-----BEGIN K-----\nabc\n-----END K-----")])
def test_mvp_211_entrypoint_exports_name_file(require_docker, tmp_path: Path, content: str, expected: str) -> None:
    (tmp_path / "auth_secret").write_bytes(content.encode())
    status, logs = _run_entrypoint(tmp_path, {"AUTH_SECRET_KEY_FILE": "/run/secrets/auth_secret"}, PRINT_SHA)
    assert status == 0, logs
    assert hashlib.sha256(expected.encode()).hexdigest() in logs
    assert "provenance=/run/secrets/auth_secret" in logs  # NAME_FILE stays set for WS-04.11


@pytest.mark.realinfra
def test_mvp_211_entrypoint_fails_closed_without_echoing(require_docker, tmp_path: Path) -> None:
    (tmp_path / "s").write_text("SYNTHETIC-MARKER-5512\n")
    status, logs = _run_entrypoint(tmp_path, {"AUTH_SECRET_KEY_FILE": "/run/secrets/missing"}, ["true"])
    assert status == 1 and "AUTH_SECRET_KEY_FILE" in logs
    status, logs = _run_entrypoint(tmp_path, {"AUTH_SECRET_KEY": "x", "AUTH_SECRET_KEY_FILE": "/run/secrets/s"}, ["true"])
    assert status == 1 and "SYNTHETIC-MARKER-5512" not in logs
    status, logs = _run_entrypoint(tmp_path, {"OTHER_SECRET_FILE": "/run/secrets/s",
                                              "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt"},
                                   ["sh", "-c", "test -z \"${SSL_CERT:-}\" && cat /proc/1/cmdline"])
    assert status == 0 and "SYNTHETIC-MARKER-5512" not in logs
    assert "entrypoint.sh" not in logs  # PID 1 is the exec'd command, not the loader shell


@pytest.mark.realinfra
def test_f4_entrypoint_fails_closed_on_nul_byte_without_echoing(require_docker, tmp_path: Path) -> None:
    (tmp_path / "nulbyte").write_bytes(b"SYNTHETIC-NUL-MARK-77\x00b\n")  # not "nul": a reserved device name on Windows
    status, logs = _run_entrypoint(tmp_path, {"AUTH_SECRET_KEY_FILE": "/run/secrets/nulbyte"}, PRINT_SHA)
    assert status == 1
    assert "AUTH_SECRET_KEY_FILE contains a NUL byte" in logs
    assert "SYNTHETIC-NUL-MARK-77" not in logs


@pytest.mark.realinfra
def test_f4_entrypoint_fails_closed_on_nul_in_procfs_style_file(require_docker, tmp_path: Path) -> None:
    # /proc/self/environ reports size 0 but yields NUL-separated content when read: only the real read path sees it.
    status, logs = _run_entrypoint(tmp_path, {"AUTH_SECRET_KEY_FILE": "/proc/self/environ", "SYNTHETIC_PRESENT": "MARK-9931"},
                                   PRINT_SHA)
    assert status == 1, logs
    assert "AUTH_SECRET_KEY_FILE contains a NUL byte" in logs
    assert "MARK-9931" not in logs
