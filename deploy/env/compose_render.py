"""Hermetic ``docker compose config`` rendering for the env-contract tests and the harness.

``render_compose_config`` copies the named compose files into ``workdir``, writes the given
values to ``workdir/render.env`` (every value single-quoted, so compose never interpolates
inside them), creates a ``workdir/.env.prod`` placeholder (older files that still say
``env_file: .env.prod`` render without a real one ever being read), and runs::

    docker compose --project-directory <workdir> -p solace-render --env-file <workdir>/render.env \\
        -f <file> ... [--profile <p> ...] config --format json

with a scrubbed process environment, so neither the developer's ``.env`` nor their shell
variables can leak into the render. The rendered configuration is returned, never printed.

Two normalizations of compose's output are undone so the result says what a container gets:

- compose writes a literal ``$`` in a value as ``$$``; ``environment`` values are returned
  unescaped (a bcrypt hash renders as itself). Other fields keep compose's escaping.
- compose inlines ``env_file`` contents into ``environment`` and drops the ``env_file`` key, so
  the placeholder holds a single probe variable: every service whose environment received it
  is reported with ``env_file: [".env.prod"]`` and the probe is removed. Any other
  ``env_file`` fails the render, because no such file exists in ``workdir``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAME = "solace-render"
RENDER_TIMEOUT_SECONDS = 180

# The only parent-environment variables the docker CLI sees (compared case-insensitively,
# because Windows environment names are case-insensitive). ProgramFiles is where Docker Desktop
# installs the compose CLI plugin on Windows; without it ``docker compose`` is not found.
PASSTHROUGH_ENV = (
    "PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA",
    "ProgramData", "ProgramFiles", "DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_CONTEXT",
)

ENV_FILE_PLACEHOLDER = ".env.prod"
ENV_FILE_PROBE = "SOLACE_RENDER_ENV_FILE_PROBE"

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_IN_VALUE = ("'", "\n", "\r", "\x00")


class ComposeRenderError(RuntimeError):
    """``docker compose config`` failed; the message is compose's stderr (never a value)."""


def format_env_file(values: Mapping[str, str]) -> str:
    """Render ``values`` as a compose env file with one single-quoted ``KEY='value'`` per line.

    Single quotes make compose take the value literally (no ``$`` interpolation). A value that
    cannot be single-quoted (it holds a quote or a line break) is refused without echoing it.
    """
    lines = []
    for key in sorted(values):
        value = values[key]
        if not _KEY.match(key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        if not isinstance(value, str):
            raise TypeError(f"{key}: value must be a string")
        if any(ch in value for ch in _FORBIDDEN_IN_VALUE):
            raise ValueError(f"{key}: value cannot be written single-quoted (quote or line break)")
        lines.append(f"{key}='{value}'")
    return "\n".join(lines) + "\n"


def _scrubbed_env() -> dict[str, str]:
    allowed = {name.upper() for name in PASSTHROUGH_ENV}
    return {k: v for k, v in os.environ.items() if k.upper() in allowed}


def _resolve(file: str | os.PathLike[str]) -> Path:
    path = Path(file)
    return path if path.is_absolute() else REPO_ROOT / path


def render_compose_config(
    files: Sequence[str | os.PathLike[str]],
    env: Mapping[str, str],
    *,
    profiles: Sequence[str] = (),
    workdir: Path,
) -> dict:
    """Render the merged compose configuration of ``files`` under ``env`` and return it as a dict.

    ``files`` are repository-relative (or absolute) compose file paths, merged in order.
    Raises ``ComposeRenderError`` when the docker CLI is missing or compose exits non-zero
    (for example when a ``${VAR:?message}`` variable is unset).
    """
    if not files:
        raise ValueError("at least one compose file is required")
    docker = shutil.which("docker")
    if docker is None:
        raise ComposeRenderError("docker CLI not found on PATH")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for file in files:
        source = _resolve(file)
        target = workdir / source.name
        if target in copied:
            raise ValueError(f"two compose files share the name {source.name}")
        shutil.copyfile(source, target)
        copied.append(target)

    env_file = workdir / "render.env"
    env_file.write_text(format_env_file(env), encoding="utf-8")
    (workdir / ENV_FILE_PLACEHOLDER).write_text(f"{ENV_FILE_PROBE}=1\n", encoding="utf-8")

    cmd = [docker, "compose", "--project-directory", str(workdir), "-p", PROJECT_NAME,
           "--env-file", str(env_file)]
    for path in copied:
        cmd += ["-f", str(path)]
    for profile in profiles:
        cmd += ["--profile", profile]
    cmd += ["config", "--format", "json"]

    try:
        proc = subprocess.run(
            cmd, cwd=workdir, env=_scrubbed_env(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=RENDER_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ComposeRenderError(f"docker compose config timed out after {exc.timeout}s") from None
    if proc.returncode != 0:
        raise ComposeRenderError(proc.stderr.strip() or f"docker compose config exited {proc.returncode}")
    try:
        config = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ComposeRenderError(f"docker compose config produced no JSON ({exc.msg})") from None
    for service in (config.get("services") or {}).values():
        environment = service.get("environment") or {}
        for name, value in environment.items():
            if isinstance(value, str):
                environment[name] = value.replace("$$", "$")
        if ENV_FILE_PROBE in environment:
            del environment[ENV_FILE_PROBE]
            service.setdefault("env_file", []).append(ENV_FILE_PLACEHOLDER)
    return config
