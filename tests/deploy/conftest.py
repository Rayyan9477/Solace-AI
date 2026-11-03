import contextlib, json, os, secrets, shutil, socket, subprocess, sys, tempfile, time
from dataclasses import dataclass
from pathlib import Path
import pytest

# portkey-ai (runtime lock) installs a top-level ``tests`` package into site-packages, which
# shadows this repository's ``tests`` namespace package; register this conftest under its repo
# module path so test modules can ``from tests.deploy.conftest import REPO``.
sys.modules.setdefault("tests.deploy.conftest", sys.modules[__name__])
from deploy.env.registry import load_registry, synthetic_values, write_synthetic_env
from deploy.env.compose_render import PASSTHROUGH_ENV, format_env_file, render_compose_config

REPO = Path(__file__).resolve().parents[2]
PROFILES = {"dev": ["docker-compose.yml"], "strict": ["docker-compose.yml", "docker-compose.prod.yml"]}
# interpolation names the pre-change files still use, so the RED run renders today's compose (removed in WS-03.15)
LEGACY_RENDER_SOURCES = ("JWT_SECRET_KEY", "FIELD_ENCRYPTION_KEY")
_HERE = Path(__file__).resolve().parent
_DOCKER_MISSING = pytest.StashKey[str]()

def pytest_collection_modifyitems(config, items):
    run = os.environ.get("SOLACE_RUN_DOCKER_TESTS") == "1" or os.environ.get("SOLACE_REQUIRE_REALINFRA") == "1"
    for item in items:
        # this hook sees every collected item; only gate the tests in tests/deploy
        if _HERE not in Path(str(item.path)).resolve().parents:
            continue
        if item.get_closest_marker("realinfra") or item.get_closest_marker("e2e"):
            if not run:
                item.add_marker(pytest.mark.skip(reason="starts containers: set SOLACE_RUN_DOCKER_TESTS=1"))
            elif shutil.which("docker") is None:
                # fail at setup (never xfail or skip), so an opted-in tier cannot pass without docker
                item.stash[_DOCKER_MISSING] = "SOLACE_RUN_DOCKER_TESTS/SOLACE_REQUIRE_REALINFRA set but docker CLI not on PATH"

def pytest_runtest_setup(item):
    reason = item.stash.get(_DOCKER_MISSING, None)
    if reason is not None:
        pytest.fail(reason, pytrace=False)

@dataclass
class Rendered:
    config: dict
    env: dict
    @property
    def services(self) -> dict:
        return self.config["services"]
    def __repr__(self) -> str:  # keep generated values out of assertion reports
        return f"Rendered(services={sorted(self.config.get('services', {}))})"

@pytest.fixture(scope="session")
def registry():
    return load_registry()

@pytest.fixture(scope="session")
def render(registry, tmp_path_factory):
    cache, base = {}, {}
    def _render(profile, *, environment="production", overrides=None, drop=(), fragments=(), profiles=()):
        key = (profile, environment, tuple(sorted((overrides or {}).items())), tuple(drop), tuple(fragments), tuple(profiles))
        if key not in cache:
            env = dict(base.setdefault((profile, environment), synthetic_values(
                registry, profile=profile, environment=environment, extra_sources=LEGACY_RENDER_SOURCES)))
            env.update(overrides or {})
            for k in drop:
                env.pop(k, None)
            workdir = Path(tempfile.mkdtemp(dir=tmp_path_factory.getbasetemp()))
            cache[key] = Rendered(render_compose_config(PROFILES[profile] + list(fragments), env, profiles=profiles, workdir=workdir), env)
        return cache[key]
    return _render

# --- compose_project: one throwaway dev-profile stack per test (real infra, WS-03.2) ---------------

COMPOSE_TIMEOUT_SECONDS = 1200   # includes a first build of an image (the hash-locked venv)
UP_WAIT_TIMEOUT_SECONDS = 300
# Host ports the dev file interpolates for the stateful services; each project publishes them on
# free ports so it never collides with a developer's stack or a parallel run (containers and the
# app services' POSTGRES_PORT stay 5432/6379). Compose-only names: no container sees them (memory-service's
# WEAVIATE_PORT and WEAVIATE_GRPC_PORT are literals, not these interpolations).
HOST_PORT_SOURCES = ("POSTGRES_PORT", "REDIS_PORT", "WEAVIATE_PORT", "WEAVIATE_GRPC_PORT")
PROJECT_LABEL = "com.docker.compose.project"

class ComposeProjectError(RuntimeError):
    """A docker compose step of a ComposeProject failed (the message carries compose's stderr tail)."""

def _scrubbed_env() -> dict:
    # the same allow-list as the renders: no shell variable can override the --env-file
    allowed = {name.upper() for name in PASSTHROUGH_ENV}
    return {k: v for k, v in os.environ.items() if k.upper() in allowed}

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _tail(text: str, lines: int = 40) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])

def _read_back(handle) -> str:
    handle.flush()
    handle.seek(0)
    return handle.read().decode("utf-8", errors="replace")

def _row(pairs):
    # json_agg rows keep column order (and duplicate names) as key/value pairs
    return tuple(value for _, value in pairs)

class ComposeProject:
    """A uniquely named compose project, over the dev profile (``docker-compose.yml``) by default.

    Every call runs ``docker compose --project-directory <repo> -p <name> --env-file <env> -f
    <file> ...`` with a scrubbed process environment. Nothing here prints a value. A second
    ComposeProject with the same name and other ``files`` (for example ``PROFILES["strict"]``)
    drives the same project, so it reuses that project's volumes.
    """

    def __init__(self, name: str, env_file: Path, files=None):
        self.name, self.env_file = name, Path(env_file)
        self.files = list(files or PROFILES["dev"])
        self._docker = shutil.which("docker")
        # compose profiles brought up so far; down() must name them, or it leaves those services behind
        self._profiles: set = set()

    def _cmd(self, args, profiles=()) -> list:
        cmd = [self._docker or "docker", "compose", "--project-directory", str(REPO), "-p", self.name,
               "--env-file", str(self.env_file)]
        for file in self.files:
            cmd += ["-f", str(REPO / file)]
        for profile in profiles:
            cmd += ["--profile", profile]
        return cmd + list(args)

    def _call(self, args, *, profiles=(), timeout=COMPOSE_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
        # S603: a docker argv list (no shell) built from the test code's own arguments
        return subprocess.run(self._cmd(args, profiles), cwd=REPO, env=_scrubbed_env(), stdin=subprocess.DEVNULL,  # noqa: S603
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, check=False)

    def _check(self, what: str, proc: subprocess.CompletedProcess) -> subprocess.CompletedProcess:
        if proc.returncode != 0:
            raise ComposeProjectError(f"{what} failed (exit {proc.returncode}):\n{_tail(proc.stderr)}")
        return proc

    def up(self, services, *, profiles=(), wait=True) -> None:
        """``up -d --build`` the named services (and their dependencies); ``wait``: until healthy."""
        services = list(services)
        self._profiles.update(profiles)
        args = ["up", "-d", "--build"]
        if wait:
            args += ["--wait", "--wait-timeout", str(UP_WAIT_TIMEOUT_SECONDS)]
        self._check(f"docker compose up {' '.join(services)}", self._call(args + services, profiles=profiles))

    # seconds run() waits for the CLI after removing its one-off container, before it kills it
    run_stop_grace: float = 60

    def run(self, service: str, *, timeout: float = COMPOSE_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
        """Run a one-shot service with its own command (``run --rm --build -T``); never raises on its exit code.

        Raises ComposeProjectError when the command has not exited within ``timeout`` seconds (a
        server that starts instead of failing); the fixture's teardown removes the container.

        The CLI's output goes to temporary files, never pipes. On Windows the compose plugin the
        docker CLI starts inherits the output handles and can outlive the CLI; with pipes,
        communicate()'s reader threads then block on EOF and closing a pipe blocks on their lock, so
        run() could hang forever. With files, only the CLI's own exit is awaited, always bounded.
        """
        cmd = self._cmd(["run", "--rm", "--build", "-T", service])
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            # S603: a docker argv list (no shell) built from the test code's own arguments
            proc = subprocess.Popen(cmd, cwd=REPO, env=_scrubbed_env(), stdin=subprocess.DEVNULL,  # noqa: S603
                                    stdout=out, stderr=err)
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                note = self._stop_run(proc, service, grace=self.run_stop_grace)
                raise ComposeProjectError(f"docker compose run {service} did not exit within {timeout}s{note}") from None
            except BaseException:  # e.g. KeyboardInterrupt: never leave the one-off or the CLI running
                self._stop_run(proc, service, grace=5)
                raise
            return subprocess.CompletedProcess(cmd, returncode, _read_back(out), _read_back(err))

    def _stop_run(self, proc, service: str, *, grace: float) -> str:
        """Stop an unfinished ``run``: remove its one-off container, then reap the CLI with bounded waits.

        Removing the one-off container detaches the CLI; a CLI that has still not exited after
        ``grace`` seconds is killed and reaped (at most 10 s more). Returns a note for the error
        message when even the kill did not end it.
        """
        with contextlib.suppress(Exception):
            ids = self._labelled(["ps", "-a"], f"label=com.docker.compose.service={service}",
                                 "label=com.docker.compose.oneoff=True")
            if ids:
                subprocess.run([self._docker or "docker", "rm", "-f", "-v", *ids], env=_scrubbed_env(),  # noqa: S603
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=120, check=False)
        try:
            proc.wait(timeout=grace)
            return ""
        except subprocess.TimeoutExpired:
            pass
        with contextlib.suppress(Exception):
            proc.kill()
        try:
            proc.wait(timeout=10)
            return ""
        except subprocess.TimeoutExpired:
            return " (and the CLI did not exit after kill)"

    def exec(self, service: str, *cmd: str) -> str:
        """``exec -T`` a command in a running service; its stdout then stderr, whatever its exit code."""
        proc = self._call(["exec", "-T", service, *cmd], timeout=120)
        return proc.stdout + proc.stderr

    def poll(self, service: str, script: str, *, timeout: float = 120, interval: float = 2) -> bool:
        """True once ``sh -c script`` exits 0 in the running service, False after ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            if self._call(["exec", "-T", service, "sh", "-c", script], timeout=60).returncode == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)

    def psql(self, sql: str) -> list:
        """Rows of one query run as the image superuser on ``solace`` (JSON-typed: bool, int, str, None).

        The query is wrapped as ``SELECT json_agg(q) FROM (<sql>) q``; rows keep the subquery's order.
        """
        # S608: the SQL is the calling test's own literal, run as the superuser on a throwaway stack
        query = f"SELECT coalesce(json_agg(q), '[]'::json) FROM ({sql.strip().rstrip(';')}) AS q"  # noqa: S608
        proc = self._check("psql", self._call(
            ["exec", "-T", "postgres", "psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1",
             "-U", "postgres", "-d", "solace", "-c", query], timeout=120))
        return json.loads(proc.stdout, object_pairs_hook=_row)

    def _labelled(self, listing, *filters) -> list:
        """IDs from a docker listing (``["ps", "-a"]``, ``["volume", "ls"]``, ...) of this project's objects."""
        args = [self._docker or "docker", *listing, "-q", "--filter", f"label={PROJECT_LABEL}={self.name}"]
        for extra in filters:
            args += ["--filter", extra]
        proc = subprocess.run(args, env=_scrubbed_env(), stdin=subprocess.DEVNULL, capture_output=True,  # noqa: S603
                              text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
        return proc.stdout.split()

    def leftovers(self) -> dict:
        """The containers, volumes and networks still labelled with this project (names only)."""
        found = {kind: self._labelled(listing) for kind, listing in
                 (("containers", ["ps", "-a"]), ("volumes", ["volume", "ls"]), ("networks", ["network", "ls"]))}
        return {kind: ids for kind, ids in found.items() if ids}

    def down(self) -> None:
        """``down -v --remove-orphans``; anything still labelled with the project is then force-removed
        and reported, so a failed teardown fails the test instead of leaking resources."""
        proc = self._call(["down", "-v", "--remove-orphans", "--timeout", "10"], profiles=sorted(self._profiles), timeout=300)
        left = self.leftovers()
        for kind, removal in (("containers", ["rm", "-f", "-v"]), ("volumes", ["volume", "rm", "-f"]),
                              ("networks", ["network", "rm"])):
            if left.get(kind):
                subprocess.run([self._docker or "docker", *removal, *left[kind]], env=_scrubbed_env(),  # noqa: S603
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=120, check=False)
        if proc.returncode != 0 or left:
            raise ComposeProjectError(f"docker compose down left {sorted(left) or 'an error'} "
                                      f"(exit {proc.returncode}):\n{_tail(proc.stderr)}")

@pytest.fixture
def compose_project(tmp_path):
    """A fresh ``solace-deploytest-<8 hex>`` project on the dev profile; torn down with its volumes."""
    env_file = tmp_path / "compose.env"
    write_synthetic_env(env_file, profile="dev", environment="development")
    with env_file.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(format_env_file({name: str(_free_port()) for name in HOST_PORT_SOURCES}))
    project = ComposeProject(f"solace-deploytest-{secrets.token_hex(4)}", env_file)
    try:
        yield project
    finally:
        project.down()

# --- caddy_adapt: the Caddyfile adapted to JSON by the pinned Caddy image (real infra, WS-03.5) -----

CADDY_IMAGE = "caddy:2.8-alpine"

@pytest.fixture
def caddy_adapt():
    """``caddy_adapt(env) -> dict``: ``caddy adapt`` of the repo Caddyfile with ``env`` as ``-e`` values.

    Runs ``docker run --rm`` on the pinned image with the Caddyfile mounted read-only; nothing
    else is mounted and nothing persists. Raises with caddy's stderr tail when adapting fails.
    """
    docker = shutil.which("docker") or "docker"

    def _adapt(env: dict) -> dict:
        cmd = [docker, "run", "--rm", "--network", "none",
               "-v", f"{(REPO / 'Caddyfile').as_posix()}:/etc/caddy/Caddyfile:ro"]
        for key, value in env.items():
            cmd += ["-e", f"{key}={value}"]
        cmd += [CADDY_IMAGE, "caddy", "adapt", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
        # S603: a docker argv list (no shell) built from the test code's own arguments
        proc = subprocess.run(cmd, env=_scrubbed_env(), stdin=subprocess.DEVNULL, capture_output=True,  # noqa: S603
                              text=True, encoding="utf-8", errors="replace", timeout=300, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"caddy adapt failed (exit {proc.returncode}):\n{_tail(proc.stderr)}")
        return json.loads(proc.stdout)
    return _adapt

def strict_env_file(path: Path) -> dict:
    """Write a synthetic strict (production) env file for a ComposeProject over PROFILES["strict"].

    Returns the values so a test can compare them in-process; they are written, never printed.
    SOLACE_SECRETS_DIR points at an empty directory next to the file: only services that mount a
    file secret need its files.
    """
    from deploy.env.registry import load_registry
    secrets_dir = Path(path).parent / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    values = synthetic_values(load_registry(), profile="strict", environment="production",
                              extra_sources=LEGACY_RENDER_SOURCES)
    values["SOLACE_SECRETS_DIR"] = secrets_dir.as_posix()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(format_env_file(values))
    return values
