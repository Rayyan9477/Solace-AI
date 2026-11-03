"""IF-19 real-infra component fixtures (MVP-039): Postgres, Redis, Kafka, an SMTP sink and the fake LLM.

Session fixtures, each backed by a container of its own that is removed together with its volumes when the
session ends:

* ``pg_migrated`` -> :class:`PgHandle`. The postgres image runs ``infrastructure/init-db`` (the IF-03 roles and
  the single database ``solace``) and is migrated to head by ``python -m scripts.db.migrate`` as
  ``solace_migrator``. ``app_pool`` is ConnectionPoolManager's ``'default'`` pool as ``solace_app`` (NOSUPERUSER
  NOBYPASSRLS), so tests see the production codec and RLS.
* ``redis_url`` -> ``redis://:<password>@127.0.0.1:<port>/0`` on a server started with ``--requirepass``.
* ``kafka_bootstrap`` -> :class:`KafkaHandle` on the compose kafka image (testcontainers ``KafkaContainer``).
* ``smtp_sink`` -> :class:`SmtpSink` on mailpit (SMTP 1025, API 8025).
* ``fake_llm`` -> :class:`FakeLLMHandle` on the image built from ``infrastructure/test/fake-llm``.

Every image comes from :data:`IMAGE_REFS`, read from the compose files, so tests run what is deployed.

The containers live for the session, but no test sees another's state: ``_realinfra_isolate`` (autouse wherever a
conftest imports it, as tests/realinfra/conftest.py does) empties the mail sink, resets the fake LLM and flushes
Redis before each test that uses them, and fails a test that leaves the kafka broker paused. Postgres data and
Kafka topics are not reset; tests use their own ids, topics and groups (or ``create_database``).

Docker is required: :func:`require_docker` skips when the daemon is unreachable and fails instead when
``SOLACE_REQUIRE_REALINFRA`` is set (any value but empty, ``0``, ``false``, ``no`` or ``off``). Generated
passwords (``secrets.token_urlsafe(24)``) are never printed or logged: reprs and log lines mask every DSN
password, readiness probes send no credentials, and no password is on a container's command line.

The async fixtures run on the session event loop (``loop_scope="session"``), so ``PgHandle.app_pool`` (and every
``create_database`` pool) is bound to it; tests that use a pool must run on that loop as well.
``tests/realinfra/conftest.py`` marks its coroutine tests so.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
import pytest
import pytest_asyncio
import structlog
import yaml

if TYPE_CHECKING:
    import asyncpg

    from solace_events.schemas import BaseEvent

_log = structlog.get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES: tuple[Path, ...] = (ROOT / "docker-compose.yml", ROOT / "docker-compose.test.yml")
INIT_DB_DIR = ROOT / "infrastructure" / "init-db"
FAKE_LLM_DIR = ROOT / "infrastructure" / "test" / "fake-llm"

# The variables infrastructure/init-db/01-roles.sh reads for each role (WS-02.1, R-02.12). POSTGRES_PASSWORD
# stays the image superuser's.
INIT_DB_ENV = {"superuser": "POSTGRES_PASSWORD", "migrator": "SOLACE_MIGRATOR_PASSWORD", "app": "SOLACE_APP_PASSWORD"}
SUPERUSER = "postgres"
MIGRATOR_ROLE = "solace_migrator"
APP_ROLE = "solace_app"
APP_DATABASE = "solace"
_ROLE_USERS = {"superuser": SUPERUSER, "migrator": MIGRATOR_ROLE, "app": APP_ROLE}

# axllent/mailpit v1.32.0. docker-compose.test.yml (WS-05.4) declares mailpit with the same digest and wins; this
# fallback applies only when that file is absent.
MAILPIT_FALLBACK_IMAGE = "axllent/mailpit@sha256:0790692be935ea4420d085694f7985b1f692ca6a8771ab0e0bd8ba75ac180498"
# The E2E overlay's allow-list (WS-05 conventions, R-13-5): reject-* addresses are refused at RCPT time.
MAILPIT_ALLOWED_RECIPIENTS = r"^(user|clin|fallback|ops)-[A-Za-z0-9.-]*@test\.invalid$"
MAILPIT_SMTP_PORT = 1025
MAILPIT_API_PORT = 8025
FAKE_LLM_PORT = 8080
FAKE_LLM_MODES = frozenset({"ok", "echo", "error", "timeout", "crisis_intent", "reply"})

STARTUP_TIMEOUT_S = 120.0
REACHABLE_TIMEOUT_S = 30.0
MIGRATE_TIMEOUT_S = 600.0
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 4

_DB_NAME_RE = re.compile(r"[a-z_][a-z0-9_]{0,62}")
_DSN_PASSWORD_RE = re.compile(r"(://[^:/@]*):[^@/]*@")
_REQUIRE_OFF = frozenset({"", "0", "false", "no", "off"})
_REQUIRED_IMAGES = ("postgres", "redis", "kafka", "mailpit")


# ---------------------------------------------------------------------------------------------------------------
# Compose YAML and image references
# ---------------------------------------------------------------------------------------------------------------

# Compose's merge tags (docker-compose.test.yml uses both): `!reset` clears a base value, `!override` replaces it.
COMPOSE_TAGS = ("!reset", "!override")


def _construct_untagged(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    """A ``!reset``/``!override`` node loaded as its plain YAML value; ``!reset`` alone (or ``!reset null``) is None."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    tag = loader.resolve(yaml.ScalarNode, node.value, (True, False))
    return loader.yaml_constructors[tag](loader, node)


class ComposeLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` plus compose's ``!reset`` and ``!override`` tags. A subclass: plain ``yaml.safe_load``
    is left unchanged and still refuses them."""


for _tag in COMPOSE_TAGS:
    ComposeLoader.add_constructor(_tag, _construct_untagged)


def load_compose_yaml(text: str) -> Any:
    """Parse compose YAML safely; a ``!reset``/``!override`` value comes back as its plain value."""
    return yaml.load(text, Loader=ComposeLoader)  # noqa: S506 - a SafeLoader subclass


def _image_refs(files: Iterable[Path] = COMPOSE_FILES) -> dict[str, str]:
    """``{service: image}`` from the compose files, later files overriding earlier ones (as compose merges them).

    The services the fixtures run (postgres, redis, kafka, mailpit) must pin a literal reference; an interpolated
    image elsewhere is left out rather than failing every collection that imports this module. The files are read
    with :func:`load_compose_yaml`, so the overlay's ``!reset``/``!override`` tags parse (an image reset to nothing
    is skipped).
    """
    refs: dict[str, str] = {}
    for path in files:
        if not path.is_file():
            continue
        doc = load_compose_yaml(path.read_text(encoding="utf-8")) or {}
        for name, service in (doc.get("services") or {}).items():
            image = (service or {}).get("image")
            if not image:
                continue
            if not isinstance(image, str) or "$" in image:
                if name in _REQUIRED_IMAGES:
                    raise ValueError(f"{path.name}: service {name!r} must pin a literal image reference, "
                                     f"not {image!r}")
                continue
            refs[name] = image
    refs.setdefault("mailpit", MAILPIT_FALLBACK_IMAGE)
    missing = [name for name in _REQUIRED_IMAGES if name not in refs]
    if missing:
        raise RuntimeError(f"the compose files declare no image for {missing} (read {[p.name for p in files]})")
    return refs


IMAGE_REFS: dict[str, str] = _image_refs()


# ---------------------------------------------------------------------------------------------------------------
# Docker requirement and shared helpers
# ---------------------------------------------------------------------------------------------------------------

_docker_error: str | None = None


def _docker_ping() -> bool:
    """True when the Docker daemon answers a ping (the failure reason is kept for require_docker's message)."""
    global _docker_error
    try:
        import docker

        client = docker.from_env(timeout=10)
        try:
            return bool(client.ping())
        finally:
            client.close()
    except Exception as exc:  # any failure (SDK missing, daemon down, bad DOCKER_HOST) means unreachable
        _docker_error = f"{type(exc).__name__}: {exc}"
        return False


def _realinfra_required() -> bool:
    return os.environ.get("SOLACE_REQUIRE_REALINFRA", "").strip().lower() not in _REQUIRE_OFF


def require_docker() -> None:
    """Skip the caller when Docker is unreachable; ``pytest.fail`` instead when SOLACE_REQUIRE_REALINFRA is set."""
    global _docker_error
    _docker_error = None
    if _docker_ping():
        return
    reason = f"Docker is unreachable ({_docker_error or 'ping failed'})"
    if _realinfra_required():
        pytest.fail(f"SOLACE_REQUIRE_REALINFRA is set but {reason}", pytrace=False)
    pytest.skip(reason)


def _mask(dsn: str) -> str:
    """The URL with the password of its userinfo replaced by ``***``."""
    return _DSN_PASSWORD_RE.sub(r"\1:***@", dsn)


def _scrub(text: str, hidden: Iterable[str]) -> str:
    for value in hidden:
        if value:
            text = text.replace(value, "***").replace(quote(value, safe=""), "***")
    return text


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def _host_of(container: Any) -> str:
    """The address that reaches the container's published ports (``localhost`` pinned to 127.0.0.1)."""
    host = container.get_container_host_ip()
    return "127.0.0.1" if host == "localhost" else host


def _log_tail(container: Any, hidden: Iterable[str] = (), lines: int = 40) -> str:
    try:
        raw = container.get_wrapped_container().logs(tail=lines).decode("utf-8", "replace")
    except Exception as exc:  # the tail is diagnostics only
        raw = f"<log tail unavailable: {type(exc).__name__}>"
    return _scrub(raw, hidden)


def _raise_if_exited(container: Any, what: str, hidden: Iterable[str] = ()) -> None:
    wrapped = container.get_wrapped_container()
    wrapped.reload()
    if wrapped.status in {"exited", "dead"}:
        code = wrapped.attrs.get("State", {}).get("ExitCode")
        raise RuntimeError(f"{what} container exited during startup (exit code {code}); log tail:\n"
                           f"{_log_tail(container, hidden)}")


def _start(container: Any, what: str, start: Callable[[], Any] | None = None) -> None:
    """Start ``container``; a failed start removes it (and its volumes) before re-raising the start error."""
    try:
        (start or container.start)()
    except BaseException:
        _discard(container, what)
        raise


def _stop(container: Any, what: str) -> None:
    """Remove the container with its anonymous volumes (``remove(force=True, v=True)``). A failure is raised, so a
    leaked container shows up as a teardown error."""
    with contextlib.suppress(Exception):  # best effort: a paused container is force-removed below anyway
        wrapped = container.get_wrapped_container()
        if wrapped is not None:
            wrapped.reload()
            if wrapped.status == "paused":
                wrapped.unpause()
    container.stop(force=True, delete_volume=True)


def _discard(container: Any, what: str) -> None:
    """:func:`_stop` on a failure path: a removal error is logged, never raised over the original error."""
    try:
        _stop(container, what)
    except Exception as exc:  # the caller re-raises the error that matters; Ryuk reaps what is left
        _log.warning("realinfra_container_remove_failed", container=what, error=type(exc).__name__)


def _wait_until(check: Callable[[], bool], container: Any, what: str, timeout: float = STARTUP_TIMEOUT_S,
                hidden: Iterable[str] = ()) -> None:
    """Poll ``check`` until it returns True; fail fast when the container exits, raise after ``timeout``."""
    deadline = time.monotonic() + timeout
    while True:
        _raise_if_exited(container, what, hidden)
        if check():
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"{what} was not ready within {timeout:.0f}s; log tail:\n{_log_tail(container, hidden)}")
        time.sleep(0.25)


def _http_ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


async def _close_pool(pool: Any) -> None:
    try:
        await asyncio.wait_for(pool.close(), timeout=10)
    except Exception:  # a pool from another (closed) loop cannot be closed gracefully
        pool.terminate()


# ---------------------------------------------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------------------------------------------


def _codec_init() -> Callable[[Any], Any]:
    """The production pool codec: WS-02.6's public ``init_connection`` once it exists, else ConnectionPoolManager's."""
    from solace_infrastructure.database import connection_manager

    public = getattr(connection_manager, "init_connection", None)
    return public if public is not None else connection_manager.ConnectionPoolManager._init_connection


@dataclass(eq=False)
class _PgServer:
    host: str
    port: int
    passwords: dict[str, str] = field(repr=False)  # INIT_DB_ENV role key -> password
    container: Any = field(repr=False)
    pools: list[Any] = field(default_factory=list, repr=False)  # create_database pools, closed at session end

    def dsn(self, role: str, database: str = APP_DATABASE) -> str:
        """``postgresql://`` URL for an INIT_DB_ENV role key, with user, password and database percent-encoded."""
        return (f"postgresql://{quote(_ROLE_USERS[role], safe='')}:{quote(self.passwords[role], safe='')}"
                f"@{_url_host(self.host)}:{self.port}/{quote(database, safe='')}")

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(self.passwords.values())


@dataclass(eq=False, repr=False)
class PgHandle:
    """A migrated database on the session's server. ``repr()`` masks every DSN password."""

    dsn_app: str
    dsn_migrator: str
    dsn_superuser: str
    app_pool: asyncpg.Pool
    _server: _PgServer
    _database: str = APP_DATABASE

    def __repr__(self) -> str:
        return (f"PgHandle(database={self._database!r}, dsn_app={_mask(self.dsn_app)!r}, "
                f"dsn_migrator={_mask(self.dsn_migrator)!r}, dsn_superuser={_mask(self.dsn_superuser)!r}, "
                f"app_pool=<asyncpg.Pool>)")

    @property
    def database(self) -> str:
        return self._database

    async def superuser_fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        """``fetch`` as the image superuser on this database, through a one-off connection with the pool codec."""
        import asyncpg

        conn = await asyncpg.connect(self.dsn_superuser, timeout=30)
        try:
            await _codec_init()(conn)
            return await conn.fetch(sql, *args)
        finally:
            await conn.close()

    async def create_database(self, name: str, *, migrate_to: str = "head") -> PgHandle:
        """A private database with the grants of ``solace``, migrated to ``migrate_to`` by the migrate CLI as
        solace_migrator, with its own solace_app pool (same codec). It lives until the session ends."""
        import asyncpg

        if not _DB_NAME_RE.fullmatch(name):
            raise ValueError(f"create_database: {name!r} must match {_DB_NAME_RE.pattern}")
        server = self._server
        admin = await asyncpg.connect(server.dsn("superuser", "postgres"), timeout=30)
        try:
            await admin.execute(f'CREATE DATABASE "{name}" OWNER {MIGRATOR_ROLE}')
            await admin.execute(f'REVOKE ALL ON DATABASE "{name}" FROM PUBLIC')
            await admin.execute(f'GRANT CONNECT, TEMPORARY ON DATABASE "{name}" TO {MIGRATOR_ROLE}, {APP_ROLE}')
        finally:
            await admin.close()
        conn = await asyncpg.connect(server.dsn("superuser", name), timeout=30)
        try:
            await conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        finally:
            await conn.close()
        expected, actual = await _privileges(server, APP_DATABASE), await _privileges(server, name)
        if actual != expected:
            raise RuntimeError(f"create_database: privileges of {name!r} differ from {APP_DATABASE!r}: "
                               f"{actual} != {expected}")
        await asyncio.to_thread(_run_migrate, server, name, migrate_to)
        pool = await asyncpg.create_pool(server.dsn("app", name), min_size=POOL_MIN_SIZE, max_size=POOL_MAX_SIZE,
                                         init=_codec_init())
        server.pools.append(pool)
        _log.info("realinfra_database_created", database=name, revision=migrate_to, dsn=_mask(server.dsn("app", name)))
        return PgHandle(dsn_app=server.dsn("app", name), dsn_migrator=server.dsn("migrator", name),
                        dsn_superuser=server.dsn("superuser", name), app_pool=pool, _server=server, _database=name)


_PRIVILEGES_SQL = ("SELECT pg_get_userbyid(datdba) AS owner, coalesce(datacl::text[], '{}') AS acl "
                   "FROM pg_database WHERE datname = $1")
_PUBLIC_SCHEMA_SQL = ("SELECT pg_get_userbyid(nspowner) AS owner, coalesce(nspacl::text[], '{}') AS acl "
                      "FROM pg_namespace WHERE nspname = 'public'")


async def _privileges(server: _PgServer, database: str) -> dict[str, Any]:
    """Database owner and ACL as init-db sets them, plus the public schema's owner and PUBLIC entries (the role
    grants that migration 009 adds to the schema are left out, so a database at any revision compares)."""
    import asyncpg

    conn = await asyncpg.connect(server.dsn("superuser", database), timeout=30)
    try:
        db = await conn.fetchrow(_PRIVILEGES_SQL, database)
        schema = await conn.fetchrow(_PUBLIC_SCHEMA_SQL)
    finally:
        await conn.close()
    return {"owner": db["owner"], "acl": sorted(db["acl"]), "public_owner": schema["owner"],
            "public_acl_for_public": sorted(a for a in schema["acl"] if a.startswith("="))}


def _run_migrate(server: _PgServer, database: str, target: str) -> None:
    """``python -m scripts.db.migrate upgrade <target>`` as solace_migrator (ENVIRONMENT=test, cwd = repo root)."""
    dsn = server.dsn("migrator", database)
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL_FILE"}
    env["DATABASE_URL"] = dsn
    env["ENVIRONMENT"] = "test"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH")) if p)
    proc = subprocess.run(  # noqa: S603 - fixed argv: this interpreter and the repo's own module
        [sys.executable, "-m", "scripts.db.migrate", "upgrade", target], cwd=ROOT, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=MIGRATE_TIMEOUT_S, check=False)
    hidden = (*server.secrets, dsn)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        raise RuntimeError(f"python -m scripts.db.migrate upgrade {target} exited {proc.returncode} for "
                           f"{_mask(dsn)}:\n{_scrub(tail, hidden)}")
    reached = next((line.removeprefix("migrated: ").strip() for line in reversed(proc.stdout.splitlines())
                    if line.startswith("migrated: ")), None)
    if reached is None or (target != "head" and reached != target):
        raise RuntimeError(f"python -m scripts.db.migrate upgrade {target} reported {reached!r} for {_mask(dsn)}")


def _start_postgres(passwords: dict[str, str]) -> Any:
    from testcontainers.core.container import DockerContainer

    container = DockerContainer(IMAGE_REFS["postgres"]).with_exposed_ports(5432)
    container.with_env("POSTGRES_USER", SUPERUSER).with_env("POSTGRES_DB", APP_DATABASE)
    for role, variable in INIT_DB_ENV.items():
        container.with_env(variable, passwords[role])
    container.with_volume_mapping(INIT_DB_DIR, "/docker-entrypoint-initdb.d", "ro")
    _start(container, "postgres")
    # pg_isready sends no credentials. It probes TCP on 127.0.0.1, which the image's init-time server does not
    # listen on, so success means init-db has run and the final server is up.
    probe = ["pg_isready", "--quiet", "--host", "127.0.0.1", "--port", "5432",
             "--username", SUPERUSER, "--dbname", APP_DATABASE]

    def ready() -> bool:
        import docker.errors

        try:
            return container.exec(probe).exit_code == 0
        except docker.errors.APIError:  # exited between reload() and exec(); the next check reports it
            return False

    try:
        _wait_until(ready, container, "postgres", hidden=passwords.values())
    except BaseException:
        _discard(container, "postgres")
        raise
    return container


async def _wait_until_reachable(server: _PgServer) -> None:
    """Wait until the published port reaches the server (a login as the superuser through it)."""
    import asyncpg

    deadline = time.monotonic() + REACHABLE_TIMEOUT_S
    while True:
        try:
            conn = await asyncpg.connect(server.dsn("superuser", "postgres"), timeout=5)
            await conn.close()
            return
        except (OSError, TimeoutError, asyncpg.CannotConnectNowError, asyncpg.PostgresConnectionError):
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.5)


async def _register_default_pool(server: _PgServer) -> Any:
    """ConnectionPoolManager ``'default'`` as solace_app (ssl_mode disable, pool 1-4), replacing any stale one."""
    from solace_infrastructure.database.connection_manager import ConnectionPoolManager
    from solace_infrastructure.postgres import PostgresSettings

    # _env_file=None: .env is never read; every connection field is explicit (other fields keep their defaults).
    fields: dict[str, Any] = {
        "_env_file": None, "host": server.host, "port": server.port, "database": APP_DATABASE, "user": APP_ROLE,
        "password": server.passwords["app"], "ssl_mode": "disable",
        "min_pool_size": POOL_MIN_SIZE, "max_pool_size": POOL_MAX_SIZE}
    settings = PostgresSettings(**fields)
    await _forget_default_pool(ConnectionPoolManager)
    await ConnectionPoolManager.register_pool("default", settings, min_size=POOL_MIN_SIZE, max_size=POOL_MAX_SIZE)
    return await ConnectionPoolManager.get_pool("default")


async def _forget_default_pool(manager: Any, pool: Any = None) -> None:
    """Without ``pool``: drop any ``'default'`` registration and close its pool. With ``pool`` (ours): drop the
    registration only while it is still ``pool``, then close ``pool``; another owner's registration is kept."""
    current = manager._pools.get("default")
    if pool is None or current is pool:
        manager._pools.pop("default", None)
        manager._pool_configs.pop("default", None)
        manager._locks.pop("default", None)  # a lock may be bound to an earlier event loop
        if current is not None:
            await _close_pool(current)
    if pool is not None and pool is not current:
        await _close_pool(pool)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_migrated() -> AsyncIterator[PgHandle]:
    """PostgreSQL from the compose image with init-db, migrated to head, ``app_pool`` = ``'default'`` as solace_app."""
    require_docker()
    from solace_infrastructure.database.connection_manager import ConnectionPoolManager

    passwords = {role: secrets.token_urlsafe(24) for role in INIT_DB_ENV}
    container = await asyncio.to_thread(_start_postgres, passwords)
    server: _PgServer | None = None
    app_pool = None
    try:
        server = _PgServer(host=_host_of(container), port=int(container.get_exposed_port(5432)),
                           passwords=passwords, container=container)
        await _wait_until_reachable(server)
        await asyncio.to_thread(_run_migrate, server, APP_DATABASE, "head")
        app_pool = await _register_default_pool(server)
        handle = PgHandle(dsn_app=server.dsn("app"), dsn_migrator=server.dsn("migrator"),
                          dsn_superuser=server.dsn("superuser"), app_pool=app_pool, _server=server)
        _log.info("realinfra_postgres_ready", image=IMAGE_REFS["postgres"], dsn=_mask(handle.dsn_app))
        yield handle
    finally:
        try:
            for pool in server.pools if server is not None else ():
                await _close_pool(pool)
            if app_pool is not None:
                await _forget_default_pool(ConnectionPoolManager, app_pool)
        finally:
            await asyncio.to_thread(_stop, container, "postgres")


# ---------------------------------------------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """``redis://:<password>@127.0.0.1:<port>/0`` on the compose redis image started with ``--requirepass``."""
    require_docker()
    import redis
    from testcontainers.core.container import DockerContainer

    password = secrets.token_urlsafe(24)
    # The password reaches redis-server through the container environment, not the container command.
    container = (DockerContainer(IMAGE_REFS["redis"]).with_exposed_ports(6379).with_env("REDIS_PASSWORD", password)
                 .with_command(["sh", "-c", 'exec docker-entrypoint.sh redis-server --requirepass "$REDIS_PASSWORD"']))
    _start(container, "redis")
    try:
        url = f"redis://:{quote(password, safe='')}@{_url_host(_host_of(container))}:{container.get_exposed_port(6379)}/0"

        def ready() -> bool:
            try:
                client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
                try:
                    return bool(client.ping())
                finally:
                    client.close()
            except redis.exceptions.AuthenticationError:
                raise
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, OSError):
                return False

        _wait_until(ready, container, "redis", hidden=(password,))
        _log.info("realinfra_redis_ready", image=IMAGE_REFS["redis"], url=_mask(url))
        yield url
    finally:
        _stop(container, "redis")


# ---------------------------------------------------------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------------------------------------------------------


def _encode(value: BaseEvent | dict | bytes) -> tuple[bytes, str | None]:
    """The production wire format (solace_events.publisher): JSON of ``to_dict()``, keyed by ``user_id``."""
    from solace_events.schemas import BaseEvent

    if isinstance(value, bytes | bytearray):
        return bytes(value), None
    if isinstance(value, dict):
        return json.dumps(value, default=str).encode("utf-8"), None
    if isinstance(value, BaseEvent):
        return json.dumps(value.to_dict(), default=str).encode("utf-8"), str(value.user_id)
    raise TypeError(f"produce: value must be a BaseEvent, dict or bytes, not {type(value).__name__}")


def _decode(raw: bytes | None) -> dict | None:
    try:
        message = json.loads(raw) if raw is not None else None
    except (ValueError, UnicodeDecodeError):
        return None
    return message if isinstance(message, dict) else None


@dataclass(eq=False)
class KafkaHandle:
    """The session broker. Each call opens and closes its own aiokafka client, so any event loop may use it."""

    bootstrap: str
    _container: Any = field(default=None, repr=False)

    async def produce(self, topic: str, value: BaseEvent | dict | bytes, *, key: str | None = None) -> None:
        """Send one message and wait for the broker's ack (``acks="all"``). A BaseEvent is keyed by its user_id."""
        from aiokafka import AIOKafkaProducer

        payload, default_key = _encode(value)
        key = key if key is not None else default_key
        producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap, acks="all")
        await producer.start()
        try:
            await producer.send_and_wait(topic, value=payload, key=key.encode("utf-8") if key is not None else None)
        finally:
            await producer.stop()

    async def consume_until(self, topic: str, predicate: Callable[[dict], bool], timeout: float = 30.0, *,
                            group_id: str | None = None, settle: float = 1.0) -> list[dict]:
        """Every JSON object on ``topic`` that ``predicate`` accepts, read from the earliest offset (or the group's
        committed offsets).

        * A match within ``timeout`` starts the settle phase. Every match (also one after ``timeout``) restarts a
          ``settle``-second quiet window, and the call returns once a window passes with no new match and the
          consumer has reached the end offsets seen at the first match. A returned list is therefore always followed
          by ``settle`` quiet seconds: every duplicate that reaches the broker within ``settle`` of the latest match
          is counted. The settle phase is bounded at ``timeout + settle`` after the first match (so the call lasts at
          most about ``2 * timeout + settle``); reaching the bound raises TimeoutError naming the cause, matches
          still arriving or end offsets not reached, rather than returning a count that may be short.
        * No match: reading goes on until ``timeout``. ``[]`` is returned only when the consumer was assigned and
          read the topic to the end offsets seen at assignment; otherwise TimeoutError, so ``[]`` always means
          "read, and absent".

        Messages that are not JSON objects are never passed to ``predicate``; its exceptions propagate. With
        ``group_id`` the consumer joins that group and commits what it read before returning (never on an error).
        """
        from aiokafka import AIOKafkaConsumer

        if timeout < 0 or settle < 0:
            raise ValueError("consume_until: timeout and settle must be >= 0")
        consumer = AIOKafkaConsumer(topic, bootstrap_servers=self.bootstrap, group_id=group_id,
                                    auto_offset_reset="earliest", enable_auto_commit=False)
        deadline = time.monotonic() + timeout
        matched: list[dict] = []
        first_match = last_match = 0.0
        start_ends: dict[Any, int] | None = None  # end offsets when first assigned: the "whole topic" for []
        match_ends: dict[Any, int] | None = None  # end offsets at the first match: read at least this far
        caught_up = False

        async def reached(ends: dict[Any, int]) -> bool:
            for tp, end in ends.items():
                if await consumer.position(tp) < end:
                    return False
            return True

        await consumer.start()
        try:
            while True:
                now = time.monotonic()
                if matched:
                    drained = match_ends is not None and await reached(match_ends)
                    if drained and now - last_match >= settle:
                        break
                    if now >= first_match + timeout + settle:  # the settle phase's bound, from the first match
                        if drained:
                            raise TimeoutError(
                                f"consume_until: matches on {topic!r} were still arriving {timeout + settle:g}s after "
                                f"the first one ({len(matched)} so far, the latest {now - last_match:.2f}s ago); pass "
                                f"a larger timeout or a narrower predicate")
                        raise TimeoutError(f"consume_until: {len(matched)} match(es) on {topic!r}, but it did not read "
                                           f"to the end offsets of the first match within {timeout + settle:g}s of it")
                elif now >= deadline:
                    if caught_up:
                        break
                    raise TimeoutError(f"consume_until: did not read {topic!r} to its end within {timeout:g}s "
                                       f"(assigned={bool(consumer.assignment())})")
                horizon = (last_match + settle) if matched else deadline
                poll_s = min(max(horizon - now, 0.05), 0.25)
                batches = await consumer.getmany(timeout_ms=int(poll_s * 1000))
                for records in batches.values():
                    for record in records:
                        message = _decode(record.value)
                        if message is not None and predicate(message):
                            matched.append(message)
                            last_match = time.monotonic()
                            if match_ends is None:
                                first_match = last_match
                                match_ends = await consumer.end_offsets(list(consumer.assignment()))
                if not caught_up and consumer.assignment():
                    if start_ends is None:
                        start_ends = await consumer.end_offsets(list(consumer.assignment()))
                    caught_up = await reached(start_ends)
            if group_id is not None and consumer.assignment():
                await consumer.commit()
        finally:
            await consumer.stop()
        return matched

    async def committed_offset(self, group_id: str, topic: str, partition: int = 0) -> int | None:
        """The group's committed offset (the next offset it will read) for ``topic[partition]``, or None."""
        from aiokafka.admin import AIOKafkaAdminClient
        from aiokafka.errors import KafkaError
        from aiokafka.structs import TopicPartition

        tp = TopicPartition(topic, partition)
        deadline = time.monotonic() + 15.0
        while True:
            admin = AIOKafkaAdminClient(bootstrap_servers=self.bootstrap)
            try:
                await admin.start()
                offsets = await admin.list_consumer_group_offsets(group_id, partitions=[tp])
                break
            except KafkaError as exc:
                if not exc.retriable or time.monotonic() > deadline:
                    raise
            finally:
                await admin.close()
            await asyncio.sleep(0.5)
        meta = offsets.get(tp)
        return None if meta is None or meta.offset < 0 else meta.offset

    def pause(self) -> None:
        """``docker pause`` the broker: clients hang (no refusal) until :meth:`unpause`."""
        self._wrapped().pause()

    def unpause(self) -> None:
        self._wrapped().unpause()

    def _wrapped(self) -> Any:
        if self._container is None:
            raise RuntimeError("this KafkaHandle has no container (it was not built by the kafka_bootstrap fixture)")
        return self._container.get_wrapped_container()

    async def _wait_ready(self, timeout: float = STARTUP_TIMEOUT_S) -> None:
        from aiokafka.admin import AIOKafkaAdminClient
        from aiokafka.errors import KafkaError

        deadline = time.monotonic() + timeout
        while True:
            admin = AIOKafkaAdminClient(bootstrap_servers=self.bootstrap, request_timeout_ms=5000)
            try:
                await admin.start()
                await admin.describe_cluster()
                return
            except (KafkaError, OSError):
                if time.monotonic() > deadline:
                    raise
            finally:
                await admin.close()
            await asyncio.sleep(0.5)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def kafka_bootstrap() -> AsyncIterator[KafkaHandle]:
    """The compose kafka image through testcontainers ``KafkaContainer``, as a :class:`KafkaHandle`."""
    require_docker()
    from testcontainers.kafka import KafkaContainer

    container = KafkaContainer(IMAGE_REFS["kafka"])
    await asyncio.to_thread(_start, container, "kafka", lambda: container.start(timeout=int(STARTUP_TIMEOUT_S)))
    try:
        handle = KafkaHandle(bootstrap=container.get_bootstrap_server(), _container=container)
        await handle._wait_ready()
        _log.info("realinfra_kafka_ready", image=IMAGE_REFS["kafka"], bootstrap=handle.bootstrap)
        yield handle
    finally:
        await asyncio.to_thread(_stop, container, "kafka")


# ---------------------------------------------------------------------------------------------------------------
# SMTP sink (mailpit)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MailMessage:
    id: str
    to: list[str]  # every recipient mailpit recorded: To, Cc, and envelope-only recipients (shown as Bcc)
    subject: str
    text: str
    headers: dict[str, list[str]]


class SmtpSink:
    """Mailpit's HTTP API (``/api/v1``) plus the SMTP address it listens on."""

    def __init__(self, api_url: str, smtp_host: str = "", smtp_port: int = 0) -> None:
        self.api_url = api_url.rstrip("/")
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def __repr__(self) -> str:
        return f"SmtpSink(api_url={self.api_url!r}, smtp_host={self.smtp_host!r}, smtp_port={self.smtp_port!r})"

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.api_url, timeout=10.0)

    def _summaries(self, client: httpx.Client) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        while True:
            response = client.get("/api/v1/messages", params={"start": len(summaries), "limit": 250})
            response.raise_for_status()
            body = response.json()
            batch = body.get("messages") or []
            summaries.extend(batch)
            if not batch or len(summaries) >= int(body.get("messages_count", body.get("total", 0))):
                return summaries

    @staticmethod
    def _recipients(item: dict[str, Any]) -> list[str]:
        return [a["Address"] for key in ("To", "Cc", "Bcc") for a in (item.get(key) or []) if a.get("Address")]

    def messages(self, to: str | None = None, subject_contains: str = "") -> list[MailMessage]:
        """Every captured message to ``to`` (case-insensitive address match) whose subject contains
        ``subject_contains``, oldest first."""
        with self._client() as client:
            found = [s for s in reversed(self._summaries(client))
                     if (to is None or to.lower() in {a.lower() for a in self._recipients(s)})
                     and subject_contains in (s.get("Subject") or "")]
            out = []
            for summary in found:
                detail = client.get(f"/api/v1/message/{summary['ID']}")
                detail.raise_for_status()
                headers = client.get(f"/api/v1/message/{summary['ID']}/headers")
                headers.raise_for_status()
                body = detail.json()
                out.append(MailMessage(id=summary["ID"], to=self._recipients(body), subject=body.get("Subject") or "",
                                       text=body.get("Text") or "", headers=headers.json()))
            return out

    def wait_for(self, to: str, subject_contains: str = "", timeout: float = 60.0) -> MailMessage:
        """The oldest message to ``to`` whose subject contains ``subject_contains``; TimeoutError after ``timeout``."""
        deadline = time.monotonic() + timeout
        while True:
            found = self.messages(to, subject_contains)
            if found:
                return found[0]
            if time.monotonic() > deadline:
                with self._client() as client:
                    held = len(self._summaries(client))
                raise TimeoutError(f"no mail to {to!r} with a subject containing {subject_contains!r} within "
                                   f"{timeout:g}s ({held} message(s) in the sink)")
            time.sleep(0.25)

    def clear(self) -> None:
        """Delete every captured message."""
        with self._client() as client:
            client.delete("/api/v1/messages").raise_for_status()


def _smtp_ok(host: str, port: int) -> bool:
    import smtplib

    try:
        with smtplib.SMTP(host, port, timeout=2) as smtp:
            return smtp.noop()[0] == 250
    except (OSError, smtplib.SMTPException):
        return False


@pytest.fixture(scope="session")
def smtp_sink() -> Iterator[SmtpSink]:
    """Mailpit (SMTP 1025, API 8025) with the E2E recipient allow-list, as an :class:`SmtpSink`."""
    require_docker()
    from testcontainers.core.container import DockerContainer

    container = (DockerContainer(IMAGE_REFS["mailpit"]).with_exposed_ports(MAILPIT_SMTP_PORT, MAILPIT_API_PORT)
                 .with_env("MP_SMTP_ALLOWED_RECIPIENTS", MAILPIT_ALLOWED_RECIPIENTS))
    _start(container, "mailpit")
    try:
        host = _host_of(container)
        sink = SmtpSink(f"http://{_url_host(host)}:{container.get_exposed_port(MAILPIT_API_PORT)}",
                        host, int(container.get_exposed_port(MAILPIT_SMTP_PORT)))
        _wait_until(lambda: _http_ok(f"{sink.api_url}/readyz") and _smtp_ok(sink.smtp_host, sink.smtp_port),
                    container, "mailpit")
        _log.info("realinfra_smtp_sink_ready", image=IMAGE_REFS["mailpit"], api_url=sink.api_url)
        yield sink
    finally:
        _stop(container, "mailpit")


# ---------------------------------------------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeLLMHandle:
    """``base_url`` is the OpenAI-compatible base (``.../v1``); ``control_url`` is the ``/__control`` API."""

    base_url: str
    control_url: str

    def scenario(self, mode: str, match: str | None = None, content: str | None = None) -> None:
        """Set the server's scenario. Anything the server would misread is refused here, before it is sent: an
        unknown mode (answered as "ok"), ``reply`` without str content (an empty answer), content for another
        mode (ignored), and a non-str or empty ``match`` (the server drops every connection, which looks like an
        LLM outage)."""
        if mode not in FAKE_LLM_MODES:
            raise ValueError(f"fake_llm.scenario: mode must be one of {sorted(FAKE_LLM_MODES)}, not {mode!r}")
        if mode == "reply":
            if not isinstance(content, str):
                raise ValueError("fake_llm.scenario: mode 'reply' needs content as a str")
        elif content is not None:
            raise ValueError(f"fake_llm.scenario: content applies only to mode 'reply', not {mode!r}")
        if match is not None and (not isinstance(match, str) or not match):
            raise ValueError("fake_llm.scenario: match must be None or a non-empty str")
        httpx.post(f"{self.control_url}/scenario", json={"mode": mode, "match": match, "content": content},
                   timeout=10.0).raise_for_status()

    def requests(self) -> list[dict]:
        response = httpx.get(f"{self.control_url}/requests", timeout=10.0)
        response.raise_for_status()
        return response.json()

    def reset(self) -> None:
        httpx.post(f"{self.control_url}/reset", timeout=10.0).raise_for_status()


def _fake_llm_image() -> str:
    """Build ``infrastructure/test/fake-llm`` once per content hash and return its tag."""
    import docker
    import docker.errors

    digest = hashlib.sha256()
    for name in ("Dockerfile", "app.py"):
        digest.update(name.encode() + b"\0" + (FAKE_LLM_DIR / name).read_bytes() + b"\0")
    tag = f"solace-test/fake-llm:{digest.hexdigest()[:12]}"
    client = docker.from_env()
    try:
        try:
            client.images.get(tag)
        except docker.errors.ImageNotFound:
            client.images.build(path=str(FAKE_LLM_DIR), tag=tag, rm=True, forcerm=True)
    finally:
        client.close()
    return tag


@pytest.fixture(scope="session")
def fake_llm() -> Iterator[FakeLLMHandle]:
    """The fake LLM container built from ``infrastructure/test/fake-llm``, as a :class:`FakeLLMHandle`."""
    require_docker()
    from testcontainers.core.container import DockerContainer

    container = DockerContainer(_fake_llm_image()).with_exposed_ports(FAKE_LLM_PORT)
    _start(container, "fake-llm")
    try:
        root = f"http://{_url_host(_host_of(container))}:{container.get_exposed_port(FAKE_LLM_PORT)}"
        _wait_until(lambda: _http_ok(f"{root}/health"), container, "fake-llm")
        _log.info("realinfra_fake_llm_ready", base_url=f"{root}/v1")
        yield FakeLLMHandle(base_url=f"{root}/v1", control_url=f"{root}/__control")
    finally:
        _stop(container, "fake-llm")


# ---------------------------------------------------------------------------------------------------------------
# Per-test isolation
# ---------------------------------------------------------------------------------------------------------------


def _check_kafka_running(handle: KafkaHandle) -> None:
    """Fail the test that left the broker paused, after resuming it so later tests are not affected."""
    wrapped = handle._wrapped()
    wrapped.reload()
    if wrapped.status == "paused":
        handle.unpause()
        pytest.fail("the test left the kafka broker paused: call kafka_bootstrap.unpause() in a finally block",
                    pytrace=False)


@pytest.fixture(autouse=True)
def _realinfra_isolate(request: pytest.FixtureRequest) -> Iterator[None]:
    """Give each test clean session fixtures, so no test can pass on what an earlier test left behind: before the
    test, the mail sink is emptied, the fake LLM's scenario and request log are reset, and Redis is flushed (for
    each one the test uses, directly or through its fixtures). After it, a broker it left paused fails the test.

    Autouse wherever it is imported into a conftest (tests/realinfra/conftest.py does); a suite elsewhere that uses
    these fixtures must import it too.
    """
    names = set(request.fixturenames)
    if "smtp_sink" in names:
        request.getfixturevalue("smtp_sink").clear()
    if "fake_llm" in names:
        request.getfixturevalue("fake_llm").reset()
    if "redis_url" in names:
        import redis

        client = redis.Redis.from_url(request.getfixturevalue("redis_url"))
        try:
            client.flushall()
        finally:
            client.close()
    yield
    if "kafka_bootstrap" in names:
        _check_kafka_running(request.getfixturevalue("kafka_bootstrap"))


__all__ = [
    "_realinfra_isolate",
    "IMAGE_REFS",
    "INIT_DB_ENV",
    "ComposeLoader",
    "FakeLLMHandle",
    "KafkaHandle",
    "MailMessage",
    "PgHandle",
    "SmtpSink",
    "fake_llm",
    "kafka_bootstrap",
    "load_compose_yaml",
    "pg_migrated",
    "redis_url",
    "require_docker",
    "smtp_sink",
]
