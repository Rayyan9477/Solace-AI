"""Real-PostgreSQL fixtures for tests/migrations (IF-03 platform roles, created by infrastructure/init-db).

Every server is the pinned ``scripts.db.images.POSTGRES_IMAGE`` with ``infrastructure/init-db`` mounted at
``/docker-entrypoint-initdb.d``, so a test sees exactly the roles and database privileges a deployment gets.

* ``pg_server`` (session): one server; yields :class:`PgServer` with percent-encoded DSN helpers.
* ``start_postgres(*, env, tls_dir=None, tls_mode=None)``: start another server (behind WS-02.5's TLS
  entrypoint wrapper when ``tls_dir`` or ``tls_mode`` is given); the caller stops it.
* ``fresh_database`` (function): a new database owned by ``solace_migrator`` with the same database
  privileges as ``solace``, dropped ``WITH (FORCE)`` afterwards.
* ``migrated_database`` (function): a ``fresh_database`` upgraded to head by ``scripts.db.migrate.run_migrations``.
* ``run_migration(database_url, command, target)``: runs ``python -m scripts.db.migrate`` (WS-02.4) in a subprocess.
* ``insert_minimal``: ``async def insert_minimal(conn, table, **values) -> None`` fills every NOT NULL
  column without a default with a synthetic value of its type.
* ``docker_client`` (function): a docker SDK client for tests that run the image directly; closed afterwards.

Docker is required: the fixtures skip when it is unreachable and fail when SOLACE_REQUIRE_REALINFRA=1.
Passwords are generated per run and never printed: they stay out of reprs, ``docker exec`` command lines
and error messages.
"""
from __future__ import annotations

import datetime as dt
import decimal
import ipaddress
import os
import secrets
import subprocess
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from testcontainers.postgres import PostgresContainer

from scripts.db.images import POSTGRES_IMAGE

ROOT = Path(__file__).resolve().parents[2]
INIT_DB_DIR = ROOT / "infrastructure" / "init-db"
TLS_ENTRYPOINT = ROOT / "scripts" / "db" / "pg_tls_entrypoint.sh"  # created by WS-02.5
TLS_ENTRYPOINT_IN_CONTAINER = "/usr/local/bin/solace-pg-entrypoint.sh"
TLS_SECRETS_IN_CONTAINER = "/run/secrets"

SUPERUSER = "postgres"
APP_ROLE = "solace_app"
MIGRATOR_ROLE = "solace_migrator"
APP_DATABASE = "solace"
STARTUP_TIMEOUT_S = 120.0
REACHABLE_TIMEOUT_S = 30.0


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "realinfra: needs real infrastructure (Docker); fails instead of skipping when SOLACE_REQUIRE_REALINFRA=1")


def require_docker() -> None:
    """Skip the calling test when Docker is unreachable; fail instead when SOLACE_REQUIRE_REALINFRA=1."""
    try:
        import docker

        client = docker.from_env()
        try:
            client.ping()
        finally:
            client.close()
    except Exception as exc:  # any failure (SDK missing, daemon down, bad DOCKER_HOST) means unreachable
        reason = f"Docker is unreachable: {type(exc).__name__}: {exc}"
        if os.environ.get("SOLACE_REQUIRE_REALINFRA") == "1":
            pytest.fail(f"SOLACE_REQUIRE_REALINFRA=1 but {reason}", pytrace=False)
        pytest.skip(reason)


@pytest.fixture
def docker_client() -> Iterator[Any]:
    """A docker SDK client (skips or fails per require_docker()); closed after the test."""
    require_docker()
    import docker

    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()


@dataclass(frozen=True)
class PgServer:
    """A running server with the IF-03 roles. Passwords are excluded from the repr."""

    host: str
    port: int
    superuser_password: str = field(repr=False)
    app_password: str = field(repr=False)
    migrator_password: str = field(repr=False)

    def dsn(self, user: str, password: str, database: str = APP_DATABASE) -> str:
        """``postgresql://`` URL with user, password and database percent-encoded."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return (f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
                f"@{host}:{self.port}/{quote(database, safe='')}")

    def superuser_dsn(self, database: str = APP_DATABASE) -> str:
        return self.dsn(SUPERUSER, self.superuser_password, database)

    def migrator_dsn(self, database: str = APP_DATABASE) -> str:
        return self.dsn(MIGRATOR_ROLE, self.migrator_password, database)

    def app_dsn(self, database: str = APP_DATABASE) -> str:
        return self.dsn(APP_ROLE, self.app_password, database)


class _SolacePostgresContainer(PostgresContainer):
    """PostgresContainer whose readiness probe needs no password.

    The stock probe runs ``PGPASSWORD='<password>' psql ...`` through ``docker exec`` and repeats that
    command in its timeout message. ``pg_isready`` sends no credentials. It probes TCP on 127.0.0.1,
    which the image's init-time server does not listen on, so success means init has finished and the
    final server is up. A container that exits during init fails fast instead of waiting for the timeout.
    """

    def _connect(self) -> None:
        import docker.errors

        probe = ["pg_isready", "--quiet", "--host", "127.0.0.1", "--port", "5432",
                 "--username", self.username, "--dbname", self.dbname]
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            wrapped = self.get_wrapped_container()
            wrapped.reload()
            if wrapped.status in {"exited", "dead"}:
                code = wrapped.attrs.get("State", {}).get("ExitCode")
                raise RuntimeError(f"postgres container exited during init (exit code {code}); "
                                   f"log tail:\n{self._log_tail()}")
            try:
                if self.exec(probe).exit_code == 0:
                    return
            except docker.errors.APIError:
                pass  # exited between reload() and exec(); the next reload() reports it
            if time.monotonic() > deadline:
                raise TimeoutError(f"postgres did not accept TCP connections within {STARTUP_TIMEOUT_S:.0f}s; "
                                   f"log tail:\n{self._log_tail()}")
            time.sleep(0.5)

    def _log_tail(self, lines: int = 40) -> str:
        return self.get_wrapped_container().logs(tail=lines).decode("utf-8", "replace")


def start_postgres(*, env: dict[str, str], tls_dir: Path | None = None,
                   tls_mode: str | None = None) -> PostgresContainer:
    """Start the pinned image with ``infrastructure/init-db`` and return it once init has finished.

    ``env`` must hold ``POSTGRES_PASSWORD`` (the image superuser's) and is passed to the container as is;
    init needs ``SOLACE_APP_PASSWORD`` and ``SOLACE_MIGRATOR_PASSWORD`` (or their ``_FILE`` forms).
    ``POSTGRES_USER`` defaults to ``postgres`` and ``POSTGRES_DB`` to ``solace``. With ``tls_dir`` or
    ``tls_mode`` the server runs behind ``scripts/db/pg_tls_entrypoint.sh`` (WS-02.5): ``tls_dir`` is
    mounted read-only at ``/run/secrets`` and ``tls_mode`` becomes ``SOLACE_PG_TLS``. The caller stops the
    container (``container.stop()`` also deletes its data volume); a failed start removes it here.
    """
    require_docker()
    env = dict(env)
    password = env.pop("POSTGRES_PASSWORD", "")
    if not password:
        raise ValueError("start_postgres: env must set POSTGRES_PASSWORD (the image superuser's password)")
    container = _SolacePostgresContainer(
        POSTGRES_IMAGE, username=env.pop("POSTGRES_USER", SUPERUSER), password=password,
        dbname=env.pop("POSTGRES_DB", APP_DATABASE), driver=None)
    for key, value in env.items():
        container.with_env(key, value)
    container.with_volume_mapping(INIT_DB_DIR, "/docker-entrypoint-initdb.d", "ro")
    if tls_dir is not None or tls_mode is not None:
        if not TLS_ENTRYPOINT.is_file():
            raise FileNotFoundError(f"{TLS_ENTRYPOINT} is missing (WS-02.5 creates it)")
        container.with_volume_mapping(TLS_ENTRYPOINT, TLS_ENTRYPOINT_IN_CONTAINER, "ro")
        if tls_dir is not None:
            container.with_volume_mapping(Path(tls_dir), TLS_SECRETS_IN_CONTAINER, "ro")
        if tls_mode is not None:
            container.with_env("SOLACE_PG_TLS", tls_mode)
        container.with_kwargs(entrypoint=["sh", TLS_ENTRYPOINT_IN_CONTAINER])
        container.with_command(["postgres"])
    try:
        container.start()
    except BaseException:
        container.stop()
        raise
    return container


def _admin_connect(server: PgServer, database: str):
    """Autocommit superuser connection (psycopg, synchronous, so fixtures stay outside the test's event loop)."""
    import psycopg

    return psycopg.connect(host=server.host, port=server.port, user=SUPERUSER, password=server.superuser_password,
                           dbname=database, autocommit=True, connect_timeout=10)


def _wait_until_reachable(server: PgServer) -> None:
    """Wait until the host-side port mapping reaches the server (credentials go in keyword arguments only)."""
    import psycopg

    deadline = time.monotonic() + REACHABLE_TIMEOUT_S
    while True:
        try:
            with _admin_connect(server, "postgres") as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


@pytest.fixture(scope="session")
def pg_server() -> Iterator[PgServer]:
    """One PostgreSQL 16 server for the session, initialised by infrastructure/init-db."""
    superuser_password = secrets.token_urlsafe(18)
    app_password = secrets.token_urlsafe(18)
    migrator_password = "m/@#:%" + secrets.token_urlsafe(12)  # URL-reserved characters exercise DSN encoding (MVP-273)
    container = start_postgres(env={"POSTGRES_PASSWORD": superuser_password, "SOLACE_APP_PASSWORD": app_password,
                                    "SOLACE_MIGRATOR_PASSWORD": migrator_password})
    try:
        server = PgServer(host=container.get_container_host_ip(), port=int(container.get_exposed_port(5432)),
                          superuser_password=superuser_password, app_password=app_password,
                          migrator_password=migrator_password)
        _wait_until_reachable(server)
        yield server
    finally:
        container.stop()


_PRIVILEGES_SQL = ("SELECT pg_get_userbyid(datdba) AS owner, coalesce(datacl::text[], '{}') AS acl "
                   "FROM pg_database WHERE datname = %s")
_PUBLIC_SCHEMA_SQL = ("SELECT pg_get_userbyid(nspowner) AS owner, coalesce(nspacl::text[], '{}') AS acl "
                      "FROM pg_namespace WHERE nspname = 'public'")


def _privileges(server: PgServer, database: str) -> dict[str, Any]:
    with _admin_connect(server, "postgres") as conn:
        owner, acl = conn.execute(_PRIVILEGES_SQL, [database]).fetchone()
    with _admin_connect(server, database) as conn:
        schema_owner, schema_acl = conn.execute(_PUBLIC_SCHEMA_SQL).fetchone()
    return {"owner": owner, "acl": sorted(acl), "public_owner": schema_owner, "public_acl": sorted(schema_acl)}


@pytest.fixture
def fresh_database(pg_server: PgServer) -> Iterator[str]:
    """A new, empty database owned by solace_migrator with the same privileges as ``solace``."""
    from psycopg import sql

    name = f"solace_test_{uuid.uuid4().hex[:12]}"
    db = sql.Identifier(name)
    with _admin_connect(pg_server, "postgres") as conn:
        conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(db, sql.Identifier(MIGRATOR_ROLE)))
    try:
        with _admin_connect(pg_server, "postgres") as conn:
            conn.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(db))
            conn.execute(sql.SQL("GRANT CONNECT, TEMPORARY ON DATABASE {} TO {}, {}").format(
                db, sql.Identifier(MIGRATOR_ROLE), sql.Identifier(APP_ROLE)))
        with _admin_connect(pg_server, name) as conn:
            conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        expected, actual = _privileges(pg_server, APP_DATABASE), _privileges(pg_server, name)
        assert actual == expected, f"fresh_database privileges differ from {APP_DATABASE!r}: {actual} != {expected}"
        yield name
    finally:
        with _admin_connect(pg_server, "postgres") as conn:
            conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(db))


@pytest.fixture
def run_migration() -> Callable[[str, str, str], subprocess.CompletedProcess]:
    """``run_migration(database_url, command, target)``: the migrate entrypoint,
    ``python -m scripts.db.migrate <command> <target>`` with DATABASE_URL set, so every chain test also proves it
    (role preflight, advisory lock, Alembic) end to end."""

    def run(database_url: str, command: str, target: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL_FILE"}
        env["DATABASE_URL"] = database_url
        return subprocess.run(  # noqa: S603 - fixed argv: this interpreter and the repo's own module
            [sys.executable, "-m", "scripts.db.migrate", command, target],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False)

    return run


@pytest.fixture
def migrated_database(pg_server: PgServer, fresh_database: str) -> str:
    """``fresh_database`` upgraded to head as solace_migrator through the entrypoint's API (``run_migrations``)."""
    from scripts.db.migrate import run_migrations

    reached = run_migrations(pg_server.migrator_dsn(fresh_database), "upgrade", "head")
    assert reached is not None, "upgrade head left the database at base"
    return fresh_database


_COLUMNS_SQL = """
SELECT column_name, data_type, udt_schema, udt_name, character_maximum_length,
       is_nullable, column_default, is_identity, is_generated
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = $1
ORDER BY ordinal_position
"""
_ENUM_LABELS_SQL = """
SELECT e.enumlabel FROM pg_enum e
JOIN pg_type t ON t.oid = e.enumtypid JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE t.typname = $1 AND n.nspname = $2 ORDER BY e.enumsortorder
"""


def _text(col: Mapping[str, Any]) -> str:
    value = f"{uuid.uuid4().hex}_{col['column_name']}"  # random part first, so truncation keeps it
    limit = col["character_maximum_length"]
    return value[:limit] if limit else value


_SYNTHETIC: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "uuid": lambda col: uuid.uuid4(),
    "text": _text, "varchar": _text, "bpchar": _text, "name": _text, "citext": _text,
    "int2": lambda col: 1, "int4": lambda col: 1, "int8": lambda col: 1,
    "numeric": lambda col: decimal.Decimal(1), "float4": lambda col: 1.0, "float8": lambda col: 1.0,
    "bool": lambda col: False,
    "json": lambda col: "{}", "jsonb": lambda col: "{}",
    "timestamptz": lambda col: dt.datetime.now(dt.UTC),
    "timestamp": lambda col: dt.datetime.now(dt.UTC).replace(tzinfo=None),
    "date": lambda col: dt.date.today(),
    "time": lambda col: dt.time(0, 0), "timetz": lambda col: dt.time(0, 0, tzinfo=dt.UTC),
    "interval": lambda col: dt.timedelta(0),
    "bytea": lambda col: b"",
    "inet": lambda col: ipaddress.ip_address("127.0.0.1"), "cidr": lambda col: ipaddress.ip_network("127.0.0.0/8"),
}


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def _synthetic_value(conn: Any, table: str, col: Mapping[str, Any]) -> Any:
    if col["data_type"] == "ARRAY":
        return []
    if col["data_type"] == "USER-DEFINED" and col["udt_name"] != "citext":
        labels = await conn.fetch(_ENUM_LABELS_SQL, col["udt_name"], col["udt_schema"])
        if labels:
            return labels[0]["enumlabel"]
    factory = _SYNTHETIC.get(col["udt_name"])
    if factory is None:
        raise TypeError(f"insert_minimal: no synthetic value for {table}.{col['column_name']} "
                        f"({col['data_type']}/{col['udt_name']}); pass it explicitly")
    return factory(col)


async def _insert_minimal(conn: Any, table: str, **values: Any) -> None:
    """INSERT one row into public.<table>: ``values`` plus a synthetic value for every other NOT NULL
    column that has no default and is neither an identity nor a generated column."""
    columns = await conn.fetch(_COLUMNS_SQL, table)
    if not columns:
        raise LookupError(f"insert_minimal: table public.{table} does not exist or is not visible to this role")
    unknown = sorted(set(values) - {c["column_name"] for c in columns})
    if unknown:
        raise ValueError(f"insert_minimal: public.{table} has no column(s) {unknown}")
    row = dict(values)
    for col in columns:
        if (col["column_name"] in row or col["is_nullable"] != "NO" or col["column_default"] is not None
                or col["is_identity"] == "YES" or col["is_generated"] != "NEVER"):
            continue
        row[col["column_name"]] = await _synthetic_value(conn, table, col)
    if not row:
        await conn.execute(f"INSERT INTO {_ident(table)} DEFAULT VALUES")
        return
    names = list(row)
    placeholders = ", ".join(f"${i}" for i in range(1, len(names) + 1))
    # Identifiers are quoted by _ident and every value is a bind parameter.
    await conn.execute(f"INSERT INTO {_ident(table)} ({', '.join(map(_ident, names))}) VALUES ({placeholders})",  # noqa: S608
                       *(row[n] for n in names))


@pytest.fixture
def insert_minimal() -> Callable[..., Awaitable[None]]:
    """``async def insert_minimal(conn, table: str, **values) -> None`` (asyncpg connection)."""
    return _insert_minimal
