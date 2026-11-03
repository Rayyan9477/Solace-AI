"""MVP-249/041: init-db creates exactly the two platform roles and one application database, and fails closed."""
from __future__ import annotations

import secrets
import time
from pathlib import Path

import asyncpg
import docker
import pytest

from scripts.db.images import POSTGRES_IMAGE

ROOT = Path(__file__).resolve().parents[2]
ROLES_SCRIPT = ROOT / "infrastructure" / "init-db" / "01-roles.sh"
ATTRS = ("rolsuper", "rolbypassrls", "rolcreatedb", "rolcreaterole", "rolcanlogin")
EXPECTED_ROLES = {
    "solace_app": dict(rolsuper=False, rolbypassrls=False, rolcreatedb=False, rolcreaterole=False, rolcanlogin=True),
    "solace_migrator": dict(rolsuper=False, rolbypassrls=True, rolcreatedb=False, rolcreaterole=False, rolcanlogin=True),
}


@pytest.mark.realinfra
async def test_mvp_249_init_creates_the_platform_roles(pg_server) -> None:
    conn = await asyncpg.connect(pg_server.superuser_dsn("postgres"))
    try:
        rows = await conn.fetch("SELECT rolname, " + ", ".join(ATTRS) + " FROM pg_roles WHERE rolname = ANY($1::text[])",
                                list(EXPECTED_ROLES))
    finally:
        await conn.close()
    assert {r["rolname"]: {a: r[a] for a in ATTRS} for r in rows} == EXPECTED_ROLES


@pytest.mark.realinfra
async def test_mvp_249_single_application_database_owned_by_migrator(pg_server) -> None:
    conn = await asyncpg.connect(pg_server.superuser_dsn("postgres"))
    try:
        names = {r["datname"] for r in await conn.fetch("SELECT datname FROM pg_database WHERE NOT datistemplate")}
        owner = await conn.fetchval("SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname = 'solace'")
        can_create = await conn.fetchval("SELECT has_database_privilege('solace_app', 'solace', 'CREATE')")
        can_connect = await conn.fetchval("SELECT has_database_privilege('solace_app', 'solace', 'CONNECT')")
    finally:
        await conn.close()
    assert names == {"postgres", "solace"}
    assert owner == "solace_migrator"
    assert (can_create, can_connect) == (False, True)


@pytest.mark.realinfra
async def test_mvp_041_app_role_cannot_create_objects_in_public(pg_server) -> None:
    conn = await asyncpg.connect(pg_server.app_dsn("solace"))
    try:
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await conn.execute("CREATE TABLE mvp041_probe (id int)")
    finally:
        await conn.close()


@pytest.mark.realinfra
def test_mvp_249_init_fails_closed_without_role_passwords(docker_client) -> None:
    container = docker_client.containers.run(
        POSTGRES_IMAGE, detach=True,
        environment={"POSTGRES_PASSWORD": secrets.token_urlsafe(16), "POSTGRES_DB": "solace"},
        volumes={str(ROOT / "infrastructure" / "init-db"): {"bind": "/docker-entrypoint-initdb.d", "mode": "ro"}})
    try:
        status = container.wait(timeout=120)["StatusCode"]
        logs = container.logs().decode("utf-8", "replace")
    finally:
        container.remove(force=True, v=True)
    assert status != 0
    assert "SOLACE_APP_PASSWORD" in logs


def test_mvp_249_init_script_is_lf_posix_sh() -> None:
    raw = ROLES_SCRIPT.read_bytes()
    assert raw.startswith(b"#!/bin/sh\n")  # also rules out a UTF-8 BOM
    assert b"\r" not in raw


def test_mvp_249_init_script_reads_passwords_from_psql_environment() -> None:
    # /proc/<pid>/cmdline is world-readable, so no password may travel on psql's argv.
    script = ROLES_SCRIPT.read_text(encoding="utf-8")
    assert "--set=app_password" not in script and "--set=migrator_password" not in script
    assert "\\getenv app_password SOLACE_APP_PASSWORD" in script
    assert "\\getenv migrator_password SOLACE_MIGRATOR_PASSWORD" in script


def _wait_until_serving(container, timeout: float = 120.0) -> str:
    """'ready' once the final server accepts TCP on 127.0.0.1 (init done), 'exited' if init failed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        container.reload()
        if container.status in ("exited", "dead"):
            return "exited"
        try:
            probe = container.exec_run(["pg_isready", "-q", "-h", "127.0.0.1", "-U", "postgres", "-d", "solace"])
        except docker.errors.APIError:  # exited between reload() and exec_run(); the next reload() reports it
            probe = None
        if probe is not None and probe.exit_code == 0:
            return "ready"
        time.sleep(0.5)
    return "timeout"


@pytest.mark.realinfra
def test_mvp_249_role_passwords_never_reach_psql_argv_or_logs(docker_client, tmp_path: Path) -> None:
    # A psql shim first on PATH records every argv the init process passes to psql, then runs the real psql.
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "psql"
    # /tmp here is inside the throwaway container, not on the host.
    shim.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> /tmp/psql-argv.log\nexec /usr/local/bin/psql "$@"\n',
                    encoding="utf-8", newline="\n")
    shim.chmod(0o755)
    passwords = {"solace_app": "app'\\" + secrets.token_urlsafe(16),
                 "solace_migrator": "m/@#:%'\\" + secrets.token_urlsafe(12)}
    container = docker_client.containers.run(
        POSTGRES_IMAGE, detach=True,
        environment={"POSTGRES_PASSWORD": secrets.token_urlsafe(16), "POSTGRES_DB": "solace",
                     "SOLACE_APP_PASSWORD": passwords["solace_app"],
                     "SOLACE_MIGRATOR_PASSWORD": passwords["solace_migrator"],
                     "PATH": "/ws/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
        volumes={str(ROOT / "infrastructure" / "init-db"): {"bind": "/docker-entrypoint-initdb.d", "mode": "ro"},
                 str(shim_dir): {"bind": "/ws/bin", "mode": "ro"}})
    try:
        state = _wait_until_serving(container)
        argv = container.exec_run(["cat", "/tmp/psql-argv.log"]).output.decode("utf-8", "replace")  # noqa: S108
        logins = {user: container.exec_run(["/usr/local/bin/psql", "-h", "127.0.0.1", "-U", user, "-d", "solace",
                                            "-tAXc", "SELECT current_user"], environment={"PGPASSWORD": pw})
                  for user, pw in passwords.items()}
        logs = container.logs().decode("utf-8", "replace")
    finally:
        container.remove(force=True, v=True)
    assert state == "ready"
    assert "VERBOSITY=terse" in argv  # the shim saw 01-roles.sh's own psql call
    # Compare role names, so a failure never prints a password.
    assert [u for u, pw in passwords.items() if pw in argv] == [], "role passwords were passed on psql's argv"
    assert [u for u, pw in passwords.items() if pw in logs] == [], "role passwords reached the container log"
    assert {u: (r.exit_code, r.output.decode().strip()) for u, r in logins.items()} == {
        "solace_app": (0, "solace_app"), "solace_migrator": (0, "solace_migrator")}


@pytest.mark.realinfra
async def test_mvp_249_insert_minimal_keeps_short_unique_text_distinct(pg_server, fresh_database,
                                                                        insert_minimal) -> None:
    conn = await asyncpg.connect(pg_server.migrator_dsn(fresh_database))
    try:
        await conn.execute("CREATE TABLE short_codes (id serial PRIMARY KEY, shortcode varchar(6) NOT NULL UNIQUE)")
        for _ in range(3):
            await insert_minimal(conn, "short_codes")
        assert await conn.fetchval("SELECT count(DISTINCT shortcode) FROM short_codes") == 3
    finally:
        await conn.close()
