"""IF-19 end-to-end harness (MVP-039, WS-05.4): the compose test stack and the helpers E2E tests use.

The stack is the real images, built from this checkout and started by ``docker compose``:

    docker compose -p solace-e2e-<lane> --project-directory <repo> --env-file <run dir>/e2e.env \\
        -f docker-compose.yml [-f docker-compose.prod.yml] -f docker-compose.test.yml up -d --build

* ``<run dir>`` (``SOLACE_E2E_RUN_DIR``, default ``%TEMP%/solace-e2e/<project>``) holds ``e2e.env`` and
  ``secrets/`` from ``python -m scripts.test.gen_e2e_env``: fresh secrets every run, never printed.
* compose runs from a cleaned environment (:func:`_compose_env`): the pytest process's ``ENVIRONMENT=test`` or a
  developer's secrets can never reach interpolation, and only the generated env file is read.
* Client traffic goes through the edge Caddy (``Stack.edge``) on Docker-assigned 127.0.0.1 ports. Internal access
  is ``docker compose exec`` (``Stack.call``, ``Stack.sql``, ``Stack.kafka_lag``), so no internal port is ever
  published, in either profile; ``Stack.urls`` are the in-network base URLs.
* :func:`start_stack` runs inside ``hop("stack_up")``: generate the env, ``config --quiet`` (render errors before
  any build), ``up -d --build``, the migrate job exited 0, every MVP service answering ``/live`` 200 within
  180 s. A service may stay down only while ``tests/e2e/expected_failures.yaml`` expects its
  ``test_mvp_039_call_reaches_each_service[<service>]`` case to fail at hop ``service_live`` for this profile and
  the entry is not yet due (:func:`services_allowed_down`); that case then reports it.
* :func:`stop_stack` removes the containers, volumes and the run dir unless ``SOLACE_E2E_KEEP`` is set.

``python -m solace_testing.e2e up|down|logs|ps [SERVICE]`` (scripts/test/e2e_stack.ps1) drives the same stack for
an operator; ``up`` keeps it.
"""
from __future__ import annotations

import asyncio
import base64
import json as _json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID

import httpx
import structlog

from solace_testing import e2e_manifest
from solace_testing import realinfra as _realinfra
from solace_testing.e2e_manifest import hop, shortcut_allowed

if TYPE_CHECKING:
    from solace_testing.realinfra import FakeLLMHandle, SmtpSink

_log = structlog.get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = ROOT / "docker-compose.yml"
PROD_FILE = ROOT / "docker-compose.prod.yml"
TEST_FILE = ROOT / "docker-compose.test.yml"
ENV_FILE_NAME = "e2e.env"
# Written by scripts/test/gen_e2e_env.py into every run dir it creates; only a dir holding it (or an empty one) is
# ever deleted by the harness.
RUN_MARKER = ".solace-e2e-run"

SERVICE_PORTS = {"orchestrator-service": 8000, "user-service": 8001, "safety-service": 8002, "notification-service": 8003,
                 "diagnosis-service": 8004, "memory-service": 8005, "therapy-service": 8006, "personality-service": 8007,
                 "analytics-service": 8009}
MVP_SERVICES = tuple(SERVICE_PORTS)

# docker-compose.test.yml, key by key (tests/solace_testing/test_mvp_039_e2e_helpers.py holds the file to these).
OVERLAY_SERVICES = ("fake-llm", "mailpit")                   # the services only the overlay defines
OVERLAY_RESET_KEYS = ("container_name", "ports", "env_file")  # reset on every service of the base and prod files
OVERLAY_APP_ENV_KEYS = frozenset({
    "EMAIL_ENABLED", "EMAIL_SMTP_HOST", "EMAIL_SMTP_PORT", "EMAIL_SMTP_TLS_MODE", "SSL_CERT_FILE",
    "SMS_ENABLED", "PUSH_ENABLED",
    "LLM_PORTKEY_GATEWAY_URL", "LLM_PHI_EGRESS_APPROVED", "LLM_ANTHROPIC_API_KEY", "LLM_ENABLE_CACHING",
    "NOTIFICATION_FALLBACK_ONCALL_EMAIL", "NOTIFICATION_ACK_BASE_URL",
    "WEAVIATE_ENABLED",
})
# user-service only: the test-only WS-14.1 setting that lets the harness's @test.invalid addresses register.
OVERLAY_USER_ENV_KEYS = frozenset({"USER_EMAIL_ALLOW_SPECIAL_USE_TLDS"})
OVERLAY_CADDY_ENV_KEYS = frozenset({"CADDY_API_SITE", "CADDY_AUTH_SITE"})
# The one service key the overlay adds to the app services: ``secrets: [e2e-ca.crt]``, the per-run CA that
# SSL_CERT_FILE names in strict, mounted at /run/secrets/e2e-ca.crt (both profiles, so one overlay serves both).
OVERLAY_APP_SECRET = "e2e-ca.crt"  # noqa: S105 - a compose secret name, not a secret value
OVERLAY_APP_SERVICE_KEYS = frozenset({"secrets"})
# Every key the overlay may add or change on a base or prod service: the environment keys above (app services and
# caddy) and the app services' e2e CA mount.
OVERLAY_ALLOWED_KEYS = OVERLAY_APP_ENV_KEYS | OVERLAY_USER_ENV_KEYS | OVERLAY_CADDY_ENV_KEYS | OVERLAY_APP_SERVICE_KEYS

EDGE_API_PORT = 8080
EDGE_AUTH_PORT = 8081
MAILPIT_API_PORT = 8025
MAILPIT_SMTP = ("mailpit", 1025)
FAKE_LLM_PORT = 8080

EMAIL_KINDS = ("user", "clin", "reject", "ops", "fallback")
EMAIL_DOMAIN = "test.invalid"
SUPERUSER = "postgres"
APP_DATABASE = "solace"

STACK_UP_HOP = "stack_up"
LIVE_HOP = "service_live"
LIVE_TEST_ID = "tests/e2e/test_mvp_039_stack_selftest.py::test_mvp_039_call_reaches_each_service[{service}]"
VERIFY_EMAIL_SHORTCUT = "verify_email_via_sql"

GEN_TIMEOUT_S = 300.0
CONFIG_TIMEOUT_S = 120.0
UP_TIMEOUT_S = 3600.0   # the first run builds every image
DOWN_TIMEOUT_S = 600.0
LIVE_TIMEOUT_S = 180.0
AUX_READY_TIMEOUT_S = 60.0
COMPOSE_TIMEOUT_S = 600.0

_LANE_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,15}")
_FLAG_OFF = frozenset({"", "0", "false", "no", "off"})
_ENVIRONMENTS = {"dev": ("development",), "strict": ("staging", "production")}

# What compose may see of the caller's environment: the OS, the Docker CLI and its proxies, and the image tag
# CI builds with. Everything else (ENVIRONMENT=test of the pytest process, any *_PASSWORD, COMPOSE_FILE, ...) is
# dropped, so interpolation reads only the generated env file.
_ENV_KEEP = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR", "HOME",
    "USERPROFILE", "USERNAME", "USERDOMAIN", "USER", "LOGNAME", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMFILES", "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME",
    "LANG", "LC_ALL", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SOLACE_IMAGE_TAG",
})
_ENV_KEEP_PREFIXES = ("DOCKER_", "BUILDKIT_")
# e2e.env keys whose values are not secret (everything else is scrubbed from error messages).
_PUBLIC_ENV_KEYS = frozenset({
    "ENVIRONMENT", "E2E_SMTP_TLS_MODE", "E2E_SSL_CERT_FILE", "SOLACE_SECRETS_DIR", "ENCRYPTION_KEY_ID",
    "DOMAIN", "CADDY_ADMIN_EMAIL",
})


# Compose YAML with !reset / !override (one implementation, in realinfra) and the Docker requirement.
load_compose_yaml = _realinfra.load_compose_yaml
require_docker = _realinfra.require_docker


# ---------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------


def _flag(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in _FLAG_OFF


def _check_run_dir(path: Path) -> Path:
    """The run dir, absolute. It is deleted at teardown, so it is never a drive root, the home directory or one of
    its parents, nor the repository, one of its parents or anything inside it."""
    resolved = path.expanduser().resolve()
    home, repo = Path.home().resolve(), ROOT.resolve()
    if resolved == Path(resolved.anchor) or resolved in (home, *home.parents):
        raise ValueError(f"the run dir {resolved} is a drive root or the home directory (or a parent of it)")
    if resolved == repo or resolved in repo.parents or repo in resolved.parents:
        raise ValueError(f"the run dir {resolved} is the repository, a parent of it or a directory inside it")
    return resolved


def _remove_run_dir(run_dir: Path) -> None:
    """Delete a run dir, but only one the generator made (it holds RUN_MARKER) or an empty one."""
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise RuntimeError(f"refusing to delete {run_dir}: it is not a directory, so not an e2e run dir")
    if not (run_dir / RUN_MARKER).is_file() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to delete {run_dir}: it is not an e2e run dir (no {RUN_MARKER}); "
                           "point SOLACE_E2E_RUN_DIR at a new or earlier run dir")
    shutil.rmtree(run_dir)


@dataclass(frozen=True)
class StackConfig:
    """Which stack to run. ``project``, ``compose_files`` and ``run_dir`` are derived when not given."""

    profile: str
    environment: str
    lane: str = "a"
    keep: bool = False
    reuse: bool = False
    run_dir: Path = None  # type: ignore[assignment]  # set in __post_init__
    project: str = ""
    compose_files: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.profile not in _ENVIRONMENTS:
            raise ValueError(f"SOLACE_E2E_PROFILE must be dev or strict, not {self.profile!r}")
        if self.environment not in _ENVIRONMENTS[self.profile]:
            raise ValueError(f"profile {self.profile} runs with ENVIRONMENT {' or '.join(_ENVIRONMENTS[self.profile])}, "
                             f"not {self.environment!r}")
        if not isinstance(self.lane, str) or not _LANE_RE.fullmatch(self.lane):
            raise ValueError(f"SOLACE_E2E_LANE must match {_LANE_RE.pattern}, not {self.lane!r}")
        project = self.project or f"solace-e2e-{self.lane}"
        object.__setattr__(self, "project", project)
        if not self.compose_files:
            files = [BASE_FILE, *([PROD_FILE] if self.profile == "strict" else []), TEST_FILE]
            object.__setattr__(self, "compose_files", [str(f) for f in files])
        run_dir = self.run_dir if self.run_dir is not None else Path(tempfile.gettempdir()) / "solace-e2e" / project
        object.__setattr__(self, "run_dir", _check_run_dir(Path(run_dir)))

    @property
    def env_file(self) -> Path:
        return self.run_dir / ENV_FILE_NAME

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> StackConfig:
        """SOLACE_E2E_PROFILE (dev), SOLACE_E2E_ENVIRONMENT (development; production when strict),
        SOLACE_E2E_LANE (a), SOLACE_E2E_KEEP, SOLACE_E2E_REUSE, SOLACE_E2E_RUN_DIR (%TEMP%/solace-e2e/<project>)."""
        env = os.environ if environ is None else environ
        profile = env.get("SOLACE_E2E_PROFILE", "dev").strip()
        environment = env.get("SOLACE_E2E_ENVIRONMENT")
        if environment is None:
            environment = "production" if profile == "strict" else "development"
        run_dir = env.get("SOLACE_E2E_RUN_DIR", "").strip()
        return cls(profile=profile, environment=environment.strip(), lane=env.get("SOLACE_E2E_LANE", "a").strip(),
                   keep=_flag(env.get("SOLACE_E2E_KEEP")), reuse=_flag(env.get("SOLACE_E2E_REUSE")),
                   run_dir=Path(run_dir) if run_dir else None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------------------
# Running compose
# ---------------------------------------------------------------------------------------------------------------


class ComposeError(RuntimeError):
    """A docker compose command failed; the message carries a scrubbed stderr tail, never a secret."""


def _compose_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {k: v for k, v in source.items() if k.upper() in _ENV_KEEP or k.upper().startswith(_ENV_KEEP_PREFIXES)}


def _compose_argv(project: str, env_file: Path, files: Iterable[str]) -> list[str]:
    argv = ["docker", "compose", "-p", project, "--project-directory", str(ROOT), "--env-file", str(env_file)]
    for path in files:
        argv += ["-f", str(path)]
    return argv


def _describe(args: Sequence[str]) -> str:
    """The command for a message: the subcommand and its first two arguments, never a payload."""
    return " ".join(args[:3])


def _scrub(text: str, hidden: Iterable[str]) -> str:
    for value in sorted({v for v in hidden if v}, key=len, reverse=True):
        text = text.replace(value, "***")
    return text


def _tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _run(argv: list[str], args: Sequence[str], *, check: bool, timeout: float, input: str | None = None,
         hidden: Iterable[str] = (), cwd: Path = ROOT) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(argv, cwd=cwd, env=_compose_env(), capture_output=True, text=True, encoding="utf-8",  # noqa: S603
                              errors="replace", timeout=timeout, input=input, check=False)
    except FileNotFoundError:
        raise ComposeError("the docker CLI is not on PATH") from None
    except subprocess.TimeoutExpired:
        raise ComposeError(f"docker compose {_describe(args)} did not finish within {timeout:g}s") from None
    if check and proc.returncode != 0:
        raise ComposeError(f"docker compose {_describe(args)} exited {proc.returncode}:\n"
                           f"{_scrub(_tail(proc.stderr or proc.stdout), hidden)}")
    return proc


def _compose(config: StackConfig, args: Sequence[str], *, check: bool = True, timeout: float = COMPOSE_TIMEOUT_S,
             input: str | None = None) -> subprocess.CompletedProcess:
    hidden = _hidden_values(config.env_file)
    argv = _compose_argv(config.project, config.env_file, config.compose_files) + list(args)
    return _run(argv, args, check=check, timeout=timeout, input=input, hidden=hidden)


def compose_config(files: Sequence[str | Path], env_file: Path, *, project: str = "solace-e2e-render") -> dict:
    """``docker compose config --format json`` of ``files`` with ``env_file``, from the cleaned environment.

    The result holds the generated secrets: compare it, never print it.
    """
    args = ["config", "--format", "json"]
    argv = _compose_argv(project, Path(env_file), [str(f) for f in files]) + args
    proc = _run(argv, args, check=True, timeout=CONFIG_TIMEOUT_S, hidden=_hidden_values(Path(env_file)))
    return _json.loads(proc.stdout)


def _ps(config: StackConfig, *services: str, all_: bool = False) -> list[dict]:
    args = ["ps", *(["-a"] if all_ else []), "--format", "json", *services]
    out = _compose(config, args, timeout=120).stdout.strip()
    if not out:
        return []
    if out.startswith("["):  # older compose prints one JSON array
        return _json.loads(out)
    return [_json.loads(line) for line in out.splitlines() if line.strip()]


# ---------------------------------------------------------------------------------------------------------------
# The env file
# ---------------------------------------------------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    """``NAME=value`` lines of a compose env file, quotes removed and ``$$`` read as ``$`` (single quotes: literal)."""
    values: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        else:
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            value = value.replace("$$", "$")
        values[name] = value
    return values


def _hidden_values(env_file: Path) -> set[str]:
    try:
        values = read_env_file(env_file)
    except OSError:
        return set()
    return {v for k, v in values.items() if k not in _PUBLIC_ENV_KEYS and len(v) >= 8}


# ---------------------------------------------------------------------------------------------------------------
# Scripts run inside the containers
# ---------------------------------------------------------------------------------------------------------------

# One HTTP request from inside a service container (stdlib only; no proxy, no redirect following). The arguments
# arrive as one base64 JSON argv element, the script itself as base64, so nothing needs shell or Windows quoting.
_CALL_SOURCE = r'''
import base64, json, sys, urllib.error, urllib.request
a = json.loads(base64.b64decode(sys.argv[1]))
body = None if a["body"] is None else base64.b64decode(a["body"])
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
request = urllib.request.Request(a["url"], data=body, method=a["method"], headers=a["headers"])
try:
    with opener.open(request, timeout=a["timeout"]) as response:
        status, data = response.status, response.read()
except urllib.error.HTTPError as exc:
    status, data = exc.code, exc.read()
except (urllib.error.URLError, OSError) as exc:
    print(json.dumps({"error": type(exc).__name__ + ": " + str(getattr(exc, "reason", exc))}))
    sys.exit(3)
print(json.dumps({"status": status, "body": base64.b64encode(data).decode("ascii")}))
'''
_CALL_SCRIPT = "import base64;exec(base64.b64decode('" + base64.b64encode(_CALL_SOURCE.encode()).decode() + "'))"
_CALL_UNREACHABLE = 3


def _call_argv(port: int, method: str, path: str, *, json: Any = None, headers: Mapping[str, str] | None = None,
               timeout: float = 10.0) -> list[str]:
    """``python -c <script> <payload>`` for one request to ``http://127.0.0.1:<port><path>``."""
    if not path.startswith("/"):
        raise ValueError(f"path must start with '/', not {path!r}")
    request_headers = dict(headers or {})
    body = None
    if json is not None:
        body = base64.b64encode(_json.dumps(json).encode("utf-8")).decode("ascii")
        request_headers.setdefault("Content-Type", "application/json")
    payload = {"url": f"http://127.0.0.1:{int(port)}{path}", "method": method.upper(), "headers": request_headers,
               "body": body, "timeout": timeout}
    return ["python", "-c", _CALL_SCRIPT, base64.b64encode(_json.dumps(payload).encode("utf-8")).decode("ascii")]


@dataclass(frozen=True)
class HttpResult:
    status: int
    json: Any
    text: str


def _parse_call_output(stdout: str) -> HttpResult:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ConnectionError("the call script printed nothing")
    reply = _json.loads(lines[-1])
    if "error" in reply:
        raise ConnectionError(reply["error"])
    text = base64.b64decode(reply["body"]).decode("utf-8", "replace")
    try:
        parsed = _json.loads(text) if text else None
    except ValueError:
        parsed = None
    return HttpResult(status=int(reply["status"]), json=parsed, text=text)


def _psql_invocation(query: str, params: Sequence[str]) -> tuple[list[str], str]:
    """psql argv (the image superuser over the container's unix socket) and the wrapped query for its stdin.

    Parameters are psql variables ``p1``, ``p2``, ... that the query reads as ``:'p1'`` (quoted literals). The
    query is ``WITH t AS (<query>) SELECT coalesce(json_agg(t), '[]'::json) FROM t``, so a SELECT or a data-
    modifying statement with RETURNING comes back as JSON rows.
    """
    for value in params:
        if not isinstance(value, str):
            raise TypeError(f"sql parameters must be str, not {type(value).__name__}")
    body = query.strip().rstrip(";").strip()
    argv = ["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-U", SUPERUSER, "-d", APP_DATABASE]
    for index, value in enumerate(params, start=1):
        argv += ["-v", f"p{index}={value}"]
    return argv, f"WITH t AS ({body}) SELECT coalesce(json_agg(t), '[]'::json) FROM t;\n"  # noqa: S608 - harness SQL


# ---------------------------------------------------------------------------------------------------------------
# Parsers and identities
# ---------------------------------------------------------------------------------------------------------------


_MISSING_GROUP_RE = re.compile(r"Consumer group '[^']*' does not exist")
_OFFSET_RE = re.compile(r"-|\d+")


def parse_consumer_groups_describe(text: str, *, group: str | None = None) -> dict[str, int]:
    """``{topic: lag}`` from ``kafka-consumer-groups --describe``, summed over partitions.

    Only table rows count: a row's GROUP must be ``group`` (default: the GROUP of the first row), its PARTITION a
    number and its offsets numbers or ``-``. Any other line, such as the "has no active members" or "is rebalancing"
    notice, is skipped wherever it appears. A partition without a committed offset (``-``) counts its whole log as
    lag, so "lag 0" is never claimed for a group that has not consumed.
    """
    missing = _MISSING_GROUP_RE.search(text)
    if missing:
        raise ValueError(f"the consumer group does not exist: {missing.group(0)}")
    header: dict[str, int] | None = None
    totals: dict[str, int] = {}
    for line in text.splitlines():
        cols = line.split()
        if not cols:
            continue
        if cols[0] == "GROUP" and "TOPIC" in cols:
            header = {name: i for i, name in enumerate(cols)}
            continue
        if header is None:
            continue
        need = max(header["GROUP"], header["TOPIC"], header["PARTITION"], header["CURRENT-OFFSET"],
                   header["LOG-END-OFFSET"], header["LAG"])
        if len(cols) <= need or not cols[header["PARTITION"]].isdigit():
            continue
        current, end, lag = cols[header["CURRENT-OFFSET"]], cols[header["LOG-END-OFFSET"]], cols[header["LAG"]]
        if not all(_OFFSET_RE.fullmatch(value) for value in (current, end, lag)):
            continue
        if group is None:
            group = cols[header["GROUP"]]
        if cols[header["GROUP"]] != group:
            continue
        if lag != "-":
            value = int(lag)
        elif end != "-":
            value = int(end) - (int(current) if current != "-" else 0)
        else:
            value = 0
        topic = cols[header["TOPIC"]]
        totals[topic] = totals.get(topic, 0) + value
    if header is None:
        raise ValueError("no GROUP table in the kafka-consumer-groups output")
    return totals


def _consumer_lag(stdout: str, stderr: str, group: str) -> dict[str, int]:
    """kafka_lag's reading of the CLI: a missing group is reported on either stream, the table is stdout only."""
    missing = _MISSING_GROUP_RE.search(stderr) or _MISSING_GROUP_RE.search(stdout)
    if missing:
        raise ValueError(f"the consumer group does not exist: {missing.group(0)}")
    return parse_consumer_groups_describe(stdout, group=group)


# Keys parse_provision_output may name in an error: anything else is shown as a placeholder, because an unexpected
# stdout line (a JSON log line, a "key: value" line) can carry the one-time token.
_LINE_KEY_RE = re.compile(r"[a-z_]{1,40}")


def parse_provision_output(text: str) -> tuple[UUID, str]:
    """``(user_id, set_password_url)`` from ``scripts.provision_clinician``'s two stdout lines (IF-08).

    Errors name only line keys that look like keys (``[a-z_]+`` before an ``=``); any other line is shown as
    ``<unexpected line>``. The URL carries a one-time set-password token, which never reaches an error.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    keys = [line.split("=", 1)[0] if "=" in line else None for line in lines]
    if keys != ["user_id", "set_password_url"]:
        shown = [k if k is not None and _LINE_KEY_RE.fullmatch(k) else "<unexpected line>" for k in keys]
        raise ValueError(f"provision_clinician must print exactly 'user_id=<uuid>' then 'set_password_url=<url>', "
                         f"got {len(lines)} line(s): {shown}")
    try:
        user_id = UUID(lines[0].removeprefix("user_id="))
    except ValueError:
        raise ValueError("provision_clinician printed a user_id that is not a UUID") from None
    url = lines[1].removeprefix("set_password_url=")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("provision_clinician printed a set_password_url that is not an absolute http(s) URL")
    return user_id, url


_URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]]+")


def extract_link(text: str, path_prefix: str) -> str:
    """The first http(s) URL in ``text`` whose path starts with ``path_prefix`` (trailing punctuation dropped)."""
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?")
        if urlsplit(url).path.startswith(path_prefix):
            return url
    raise LookupError(f"no link whose path starts with {path_prefix!r} in the message")


def new_email(kind: str) -> str:
    """``<kind>-<12 hex>@test.invalid``; mailpit accepts user, clin, ops and fallback and refuses reject."""
    if kind not in EMAIL_KINDS:
        raise ValueError(f"kind must be one of {', '.join(EMAIL_KINDS)}, not {kind!r}")
    return f"{kind}-{secrets.token_hex(6)}@{EMAIL_DOMAIN}"


def _new_password() -> str:
    """A fresh password that passes the user-service strength rules (upper, lower, digit, 8+ characters)."""
    return f"Aa1-{secrets.token_urlsafe(18)}"


@dataclass(frozen=True)
class TestUser:
    __test__ = False  # not a pytest test class

    id: UUID
    email: str
    password: str = field(repr=False)
    access_token: str = field(repr=False)
    role: str = "user"


# ---------------------------------------------------------------------------------------------------------------
# The stack
# ---------------------------------------------------------------------------------------------------------------


class Faults:
    """Container faults through compose: ``stop``/``start``/``restart``/``pause``/``unpause`` one or more services."""

    def __init__(self, stack: Stack) -> None:
        self._stack = stack

    def _act(self, action: str, services: tuple[str, ...]) -> None:
        if not services:
            raise ValueError(f"{action}: name at least one service")
        self._stack.compose(action, *services, timeout=300)

    def stop(self, *services: str) -> None:
        self._act("stop", services)

    def start(self, *services: str) -> None:
        self._act("start", services)

    def restart(self, *services: str) -> None:
        self._act("restart", services)

    def pause(self, *services: str) -> None:
        self._act("pause", services)

    def unpause(self, *services: str) -> None:
        self._act("unpause", services)


@dataclass(eq=False)
class Stack:
    """A running stack. ``urls`` are in-network base URLs; Postgres is never published (``pg_superuser_dsn`` None)."""

    urls: dict[str, str]
    edge_url: str
    edge_auth_url: str
    project: str
    mailpit_api: str
    fake_llm_control: str
    pg_superuser_dsn: str | None
    config: StackConfig
    mail: SmtpSink
    llm: FakeLLMHandle
    fault: Faults = field(init=False, repr=False)
    not_live: frozenset[str] = frozenset()  # services the manifest let stay down (see services_allowed_down)

    def __post_init__(self) -> None:
        self.fault = Faults(self)

    def compose(self, *args: str, check: bool = True, timeout: float = COMPOSE_TIMEOUT_S) -> subprocess.CompletedProcess:
        """``docker compose <args>`` for this stack (same project, files and env file as the fixture)."""
        return _compose(self.config, args, check=check, timeout=timeout)

    def call(self, service: str, method: str, path: str, *, json: Any = None, headers: dict | None = None,
             timeout: float = 10.0) -> HttpResult:
        """One HTTP request from inside ``service``'s container to itself (``docker compose exec``)."""
        if service not in SERVICE_PORTS:
            raise ValueError(f"unknown service {service!r}; call() reaches {', '.join(MVP_SERVICES)}")
        argv = _call_argv(SERVICE_PORTS[service], method, path, json=json, headers=headers, timeout=timeout)
        proc = _compose(self.config, ["exec", "-T", service, *argv], check=False, timeout=timeout + 60)
        if proc.returncode not in (0, _CALL_UNREACHABLE):
            raise ComposeError(f"docker compose exec {service} exited {proc.returncode}:\n"
                               f"{_scrub(_tail(proc.stderr), _hidden_values(self.config.env_file))}")
        return _parse_call_output(proc.stdout)

    def sql(self, query: str, *params: str) -> list[dict]:
        """Rows of ``query`` as the image superuser (psql in the postgres container); params bind as ``:'p1'``..."""
        argv, stdin = _psql_invocation(query, params)
        proc = _compose(self.config, ["exec", "-T", "postgres", *argv], check=False, timeout=120, input=stdin)
        if proc.returncode != 0:
            raise RuntimeError(f"psql exited {proc.returncode}:\n"
                               f"{_scrub(_tail(proc.stderr), _hidden_values(self.config.env_file))}")
        return _json.loads(proc.stdout.strip() or "[]")

    def logs(self, service: str | None = None) -> str:
        """The container logs of one service, or of the whole stack (IF-19 read_logs)."""
        return self.compose("logs", "--no-color", *([service] if service else []), timeout=300).stdout

    def kafka_lag(self, group: str) -> dict[str, int]:
        """``{topic: lag}`` of a consumer group, from kafka-consumer-groups in the kafka container."""
        proc = self.compose("exec", "-T", "kafka", "kafka-consumer-groups", "--bootstrap-server", "localhost:9092",
                            "--describe", "--group", group, timeout=120)
        return _consumer_lag(proc.stdout, proc.stderr, group)

    def env_value(self, name: str) -> str:
        """A value of this run's e2e.env, for harness use only: never print or log it."""
        values = read_env_file(self.config.env_file)
        if name not in values:
            raise KeyError(f"{name} is not in this run's {ENV_FILE_NAME}")
        return values[name]

    def _edge_base(self, site: str) -> str:
        bases = {"api": self.edge_url, "auth": self.edge_auth_url}
        if site not in bases:
            raise ValueError(f"site must be 'api' or 'auth', not {site!r}")
        return bases[site]

    @staticmethod
    def _edge_headers(headers: dict | None, token: str | None) -> dict[str, str]:
        out = dict(headers or {})
        if token:
            out["Authorization"] = f"Bearer {token}"
        return out

    async def edge(self, method: str, path: str, *, site: str = "api", json: Any = None, headers: dict | None = None,
                   token: str | None = None) -> httpx.Response:
        """A client request through the edge Caddy (``site`` api or auth), as a browser or app would send it."""
        async with httpx.AsyncClient(base_url=self._edge_base(site), timeout=30.0, trust_env=False) as client:
            return await client.request(method, path, json=json, headers=self._edge_headers(headers, token))

    def _edge_sync(self, method: str, path: str, *, site: str = "api", json: Any = None,
                   token: str | None = None) -> httpx.Response:
        with httpx.Client(base_url=self._edge_base(site), timeout=30.0, trust_env=False) as client:
            return client.request(method, path, json=json, headers=self._edge_headers(None, token))


def services_allowed_down(profile: str, *, manifest_dir: Path | None = None,
                          environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """The MVP services the stack may start without: those whose ``LIVE_TEST_ID`` case the manifest expects to
    fail at hop ``service_live`` in ``profile``, while that entry is not due (``SOLACE_BARRIER_WAVE`` unset or
    below its target wave). Every other service must answer /live, or stack_up fails."""
    env = os.environ if environ is None else environ
    directory = e2e_manifest.DEFAULT_MANIFEST_DIR if manifest_dir is None else Path(manifest_dir)
    path = directory / e2e_manifest.MANIFEST_NAME
    if not path.is_file():
        return frozenset()
    entries, _ = e2e_manifest.load_manifest(path)
    barrier = e2e_manifest.barrier_wave(env)
    down = set()
    for service in MVP_SERVICES:
        nodeid = LIVE_TEST_ID.format(service=service)
        for entry in entries:
            if (entry.expected_failing_hop == LIVE_HOP and profile in entry.profiles and entry.matches(nodeid)
                    and (barrier is None or barrier < entry.target_wave)):
                down.add(service)
    return frozenset(down)


def _generate_env(config: StackConfig) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH")) if p)
    argv = [sys.executable, "-m", "scripts.test.gen_e2e_env", "--profile", config.profile,
            "--environment", config.environment, "--out", str(config.run_dir)]
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",  # noqa: S603
                          errors="replace", timeout=GEN_TIMEOUT_S, check=False)
    if proc.returncode != 0:  # the generator prints names only
        raise RuntimeError(f"python -m scripts.test.gen_e2e_env exited {proc.returncode}:\n{_tail(proc.stderr)}")


def _remove_project(config: StackConfig) -> None:
    """``docker compose -p <project> down -v --remove-orphans`` by project label alone (no files, no env file),
    from a directory without compose files, so a stale or half-written run is removed too."""
    config.run_dir.parent.mkdir(parents=True, exist_ok=True)
    args = ["down", "-v", "--remove-orphans"]
    _run(["docker", "compose", "-p", config.project, *args], args, check=True, timeout=DOWN_TIMEOUT_S,
         cwd=config.run_dir.parent)


def _teardown(config: StackConfig, *, strict: bool) -> None:
    """Remove the project's containers, networks and volumes, then the run dir (and its secrets)."""
    try:
        _remove_project(config)
        _remove_run_dir(config.run_dir)
    except Exception as exc:
        if strict:
            raise
        _log.warning("e2e_teardown_failed", project=config.project, error=type(exc).__name__)


def _published(config: StackConfig, service: str, port: int) -> str:
    """``http://127.0.0.1:<host port>`` for a published container port; anything but loopback is refused."""
    out = _compose(config, ["port", service, str(port)], timeout=60).stdout.strip().splitlines()
    if not out:
        raise RuntimeError(f"{service}:{port} is not published")
    host, _, host_port = out[-1].strip().rpartition(":")
    if host.strip("[]") not in ("127.0.0.1", "::1"):
        raise RuntimeError(f"{service}:{port} is published on {host}, not on loopback only")
    return f"http://127.0.0.1:{int(host_port)}"


def _wait_http(url: str, what: str, timeout: float = AUX_READY_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if httpx.get(url, timeout=2.0, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"{what} did not answer 200 on {url} within {timeout:g}s")
        time.sleep(0.5)


def _check_migrate(config: StackConfig) -> None:
    jobs = [p for p in _ps(config, "migrate", all_=True) if p.get("Service") == "migrate"]
    if not jobs:
        raise RuntimeError("the migrate job did not run")
    state, code = jobs[0].get("State"), jobs[0].get("ExitCode")
    if state != "exited" or code != 0:
        logs = _compose(config, ["logs", "--no-color", "--tail", "40", "migrate"], check=False, timeout=120).stdout
        raise RuntimeError(f"the migrate job is {state} with exit code {code}; log tail:\n"
                           f"{_scrub(logs, _hidden_values(config.env_file))}")


def _wait_live(stack: Stack, allowed_down: frozenset[str], timeout: float = LIVE_TIMEOUT_S) -> frozenset[str]:
    """Poll /live of every MVP service until all answer 200 or ``timeout`` passes; returns the ones still down.

    A service that is down at the deadline fails the stack unless ``allowed_down`` names it. The full timeout is
    spent on an allowed one too, so a slow (not broken) service is never reported down.
    """
    deadline = time.monotonic() + timeout
    pending = set(MVP_SERVICES)
    last: dict[str, str] = {}
    while pending:
        for service in sorted(pending):
            try:
                result = stack.call(service, "GET", "/live", timeout=5.0)
            except (ComposeError, ConnectionError, ValueError) as exc:
                last[service] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
                continue
            if result.status == 200:
                pending.discard(service)
            else:
                last[service] = f"HTTP {result.status}"
        if not pending or time.monotonic() > deadline:
            break
        time.sleep(2.0)
    unexpected = sorted(pending - allowed_down)
    if unexpected:
        hidden = _hidden_values(stack.config.env_file)
        tails = []
        for service in unexpected:
            logs = _compose(stack.config, ["logs", "--no-color", "--tail", "20", service], check=False,
                            timeout=120).stdout
            tails.append(f"--- {service} ({last.get(service, 'no answer')}):\n{_scrub(logs, hidden)}")
        raise RuntimeError(f"not live within {timeout:g}s: {', '.join(unexpected)}\n" + "\n".join(tails))
    if pending:
        _log.warning("e2e_services_down_as_expected", services=sorted(pending), hop=LIVE_HOP)
    return frozenset(pending)


def _build_stack(config: StackConfig) -> Stack:
    edge = _published(config, "caddy", EDGE_API_PORT)
    auth = _published(config, "caddy", EDGE_AUTH_PORT)
    mail_api = _published(config, "mailpit", MAILPIT_API_PORT)
    llm_root = _published(config, "fake-llm", FAKE_LLM_PORT)
    _wait_http(f"{mail_api}/readyz", "mailpit")
    _wait_http(f"{llm_root}/health", "fake-llm")
    return Stack(urls={s: f"http://{s}:{p}" for s, p in SERVICE_PORTS.items()}, edge_url=edge, edge_auth_url=auth,
                 project=config.project, mailpit_api=mail_api, fake_llm_control=f"{llm_root}/__control",
                 pg_superuser_dsn=None, config=config, mail=_realinfra.SmtpSink(mail_api, *MAILPIT_SMTP),
                 llm=_realinfra.FakeLLMHandle(base_url=f"{llm_root}/v1", control_url=f"{llm_root}/__control"))


def _start(config: StackConfig) -> Stack:
    if not (config.reuse and config.env_file.is_file()):
        _remove_project(config)  # a crashed earlier run's volumes hold its passwords
        _remove_run_dir(config.run_dir)
        _generate_env(config)
    _compose(config, ["config", "--quiet"], timeout=CONFIG_TIMEOUT_S)  # render errors before any build
    _compose(config, ["up", "-d", "--remove-orphans", *([] if config.reuse else ["--build"])], timeout=UP_TIMEOUT_S)
    _check_migrate(config)
    stack = _build_stack(config)
    stack.not_live = _wait_live(stack, services_allowed_down(config.profile))
    _log.info("e2e_stack_up", project=config.project, profile=config.profile, environment=config.environment,
              edge=stack.edge_url, auth=stack.edge_auth_url, down=sorted(stack.not_live))
    return stack


def start_stack(config: StackConfig) -> Stack:
    """Generate the env, build and start the stack, and wait for it (hop ``stack_up``).

    On a failure the project and the run dir are removed again unless ``keep`` or ``reuse`` is set.
    """
    with hop(STACK_UP_HOP):
        try:
            return _start(config)
        except BaseException:
            if not (config.keep or config.reuse):
                _teardown(config, strict=False)
            raise


def stop_stack(stack: Stack) -> None:
    """``down -v --remove-orphans`` and remove the run dir, unless ``keep`` is set."""
    if stack.config.keep:
        _log.info("e2e_stack_kept", project=stack.project, run_dir=str(stack.config.run_dir))
        return
    _teardown(stack.config, strict=True)


# ---------------------------------------------------------------------------------------------------------------
# Identity flows (IF-19)
# ---------------------------------------------------------------------------------------------------------------


_ERROR_TYPE_RE = re.compile(r"[a-z][a-z0-9_.]{0,59}")
_ERROR_LOC_RE = re.compile(r"[A-Za-z0-9_]{1,40}")


def _response_summary(response: httpx.Response) -> str:
    """What a failure may say about a response: never its body, which can echo the request (FastAPI's default 422
    puts the submitted token and password in ``detail[].input``) or carry tokens. A JSON ``detail`` list is reduced
    to each entry's ``type`` and ``loc``; anything else to its content type and size."""
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list) and detail:
        parts = []
        for entry in detail[:5]:
            entry = entry if isinstance(entry, dict) else {}
            kind = entry.get("type")
            kind = kind if isinstance(kind, str) and _ERROR_TYPE_RE.fullmatch(kind) else "?"
            loc = entry.get("loc") if isinstance(entry.get("loc"), list) else []
            where = ".".join(str(p) if isinstance(p, int) or (isinstance(p, str) and _ERROR_LOC_RE.fullmatch(p))
                             else "?" for p in loc)
            parts.append(f"{kind} at {where or '?'}")
        return "detail: " + "; ".join(parts)
    kind = response.headers.get("content-type", "no content type").split(";", 1)[0]
    return f"{kind}, {len(response.content)} bytes (body not shown)"


def _expect(response: httpx.Response, statuses: tuple[int, ...], what: str) -> None:
    if response.status_code not in statuses:
        raise AssertionError(f"{what}: HTTP {response.status_code} (expected {statuses}): {_response_summary(response)}")


async def make_user(stack: Stack, role: str = "user", verified: bool = True) -> TestUser:
    """Register through the auth site, verify (shortcut verify_email_via_sql while the manifest lists it, else the
    mailpit link) and log in; hops ``register``, ``verify_email``, ``login``."""
    if role != "user":
        raise ValueError("make_user registers through the public route, which creates role 'user' only; "
                         "use provision_clinician for CLINICIAN or ADMIN")
    email, password = new_email("user"), _new_password()
    with hop("register"):
        response = await stack.edge("POST", "/api/v1/auth/register", site="auth",
                                    json={"email": email, "password": password, "display_name": "E2E User"})
        _expect(response, (201,), "register")
        user_id = UUID(str(response.json()["user_id"]))
    if verified:
        with hop("verify_email"):
            if shortcut_allowed(VERIFY_EMAIL_SHORTCUT):
                rows = stack.sql("UPDATE users SET email_verified = true, status = CASE WHEN status = "
                                 "'pending_verification' THEN 'active' ELSE status END "
                                 "WHERE id = :'p1'::uuid RETURNING id", str(user_id))
                if len(rows) != 1:
                    raise AssertionError(f"verify_email_via_sql updated {len(rows)} rows for the new user")
            else:
                message = await asyncio.to_thread(stack.mail.wait_for, email, "", 60.0)
                query = parse_qs(urlsplit(extract_link(message.text, "/verify-email")).query)
                params = urlencode({"user_id": query["user_id"][0], "token": query["token"][0]})
                response = await stack.edge("POST", f"/api/v1/auth/verify-email?{params}", site="auth")
                _expect(response, (200, 204), "verify-email")
    with hop("login"):
        response = await stack.edge("POST", "/api/v1/auth/login", site="auth",
                                    json={"email": email, "password": password})
        _expect(response, (200,), "login")
        token = response.json()["access_token"]
    return TestUser(id=user_id, email=email, password=password, access_token=token)


def provision_clinician(stack: Stack, email: str, phone: str, on_call: bool = True, backup: bool = False, *,
                        role: str = "CLINICIAN", assign: UUID | None = None) -> TestUser:
    """Provision a privileged account with the IF-08 CLI in the user-service container, set its password through
    the one-time link (``POST /api/v1/users/account-setup``) and log in. The caller names the hop."""
    if role not in ("CLINICIAN", "ADMIN"):
        raise ValueError(f"role must be CLINICIAN or ADMIN, not {role!r}")
    args = ["exec", "-T", "user-service", "python", "-m", "scripts.provision_clinician", "--email", email,
            "--display-name", "E2E Clinician", "--phone", phone, "--role", role]
    if on_call:
        args.append("--on-call")
    if backup:
        args.append("--backup")
    if assign is not None:
        args += ["--assign", str(assign)]
    proc = _compose(stack.config, args, check=False, timeout=180)
    if proc.returncode != 0:  # stdout is never shown: it carries the set-password token
        raise RuntimeError(f"scripts.provision_clinician exited {proc.returncode}:\n"
                           f"{_scrub(_tail(proc.stderr, 10), _hidden_values(stack.config.env_file))}")
    user_id, url = parse_provision_output(proc.stdout)
    token = (parse_qs(urlsplit(url).query).get("token") or [""])[0]
    if not token:
        raise ValueError("the set_password_url carries no token")
    password = _new_password()
    _expect(stack._edge_sync("POST", "/api/v1/users/account-setup", site="auth",
                             json={"token": token, "new_password": password}), (200, 204), "account-setup")
    response = stack._edge_sync("POST", "/api/v1/auth/login", site="auth", json={"email": email, "password": password})
    _expect(response, (200,), "login")
    return TestUser(id=user_id, email=email, password=password, access_token=response.json()["access_token"],
                    role=role.lower())


# ---------------------------------------------------------------------------------------------------------------
# Operator CLI (scripts/test/e2e_stack.ps1)
# ---------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``up`` (kept), ``down``, ``logs [SERVICE]`` or ``ps`` for the stack the SOLACE_E2E_* variables name."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("up", "down", "logs", "ps") or len(args) > (2 if args[0] == "logs" else 1):
        print("usage: python -m solace_testing.e2e up | down | logs [SERVICE] | ps", file=sys.stderr)
        return 2
    try:
        config = StackConfig.from_env()
        action = args[0]
        if action == "up":
            stack = start_stack(replace(config, keep=True))
            print(f"project {stack.project} ({config.profile}, ENVIRONMENT={config.environment}) is up and kept")
            print(f"  edge api  {stack.edge_url}\n  edge auth {stack.edge_auth_url}")
            print(f"  mailpit   {stack.mailpit_api}\n  fake-llm  {stack.fake_llm_control}")
            if stack.not_live:
                print(f"  down (expected by the manifest): {', '.join(sorted(stack.not_live))}")
            return 0
        if action == "down":
            _teardown(config, strict=True)
            print(f"project {config.project} removed")
            return 0
        if not config.env_file.is_file():
            print(f"no run for {config.project} at {config.run_dir}; run 'up' first", file=sys.stderr)
            return 1
        compose_args = ["logs", "--no-color", *args[1:]] if action == "logs" else ["ps", "-a"]
        sys.stdout.write(_compose(config, compose_args, timeout=300).stdout)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "MVP_SERVICES",
    "OVERLAY_ALLOWED_KEYS",
    "OVERLAY_APP_SECRET",
    "OVERLAY_RESET_KEYS",
    "OVERLAY_SERVICES",
    "OVERLAY_USER_ENV_KEYS",
    "SERVICE_PORTS",
    "ComposeError",
    "Faults",
    "HttpResult",
    "Stack",
    "StackConfig",
    "TestUser",
    "compose_config",
    "extract_link",
    "load_compose_yaml",
    "make_user",
    "new_email",
    "parse_consumer_groups_describe",
    "parse_provision_output",
    "provision_clinician",
    "read_env_file",
    "require_docker",
    "services_allowed_down",
    "start_stack",
    "stop_stack",
]


if __name__ == "__main__":
    raise SystemExit(main())
