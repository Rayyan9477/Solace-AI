"""MVP-001/041: migrations run from one entrypoint, only as solace_migrator, serialized by an advisory lock."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import asyncpg
import pytest

ROOT = Path(__file__).resolve().parents[2]
HEAD = "021_personality_analytics"


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("POSTGRES_") and k not in ("DATABASE_URL", "DATABASE_URL_FILE")}
    env.update({"PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "test", **extra})
    return env


def _migrate(env: dict[str, str], *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "scripts.db.migrate", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


async def _version(dsn: str):
    conn = await asyncpg.connect(dsn)
    try:
        if await conn.fetchval("SELECT to_regclass('public.alembic_version')") is None:
            return None
        return await conn.fetchval("SELECT version_num FROM alembic_version")
    finally:
        await conn.close()


@pytest.mark.realinfra
async def test_mvp_001_cli_upgrades_to_schema_head_as_migrator(pg_server, fresh_database) -> None:
    r = _migrate(_env(DATABASE_URL=pg_server.migrator_dsn(fresh_database)), "upgrade", "head")
    assert r.returncode == 0, r.stderr[-3000:]
    assert f"migrated: {HEAD}" in r.stdout
    assert pg_server.migrator_password not in r.stdout + r.stderr
    assert await _version(pg_server.superuser_dsn(fresh_database)) == HEAD


@pytest.mark.realinfra
async def test_mvp_001_concurrent_jobs_serialize_on_the_advisory_lock(pg_server, fresh_database) -> None:
    env = _env(DATABASE_URL=pg_server.migrator_dsn(fresh_database))
    procs = [subprocess.Popen([sys.executable, "-m", "scripts.db.migrate", "upgrade", "head"], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outs = [p.communicate(timeout=600) for p in procs]
    assert [p.returncode for p in procs] == [0, 0], [o[1][-2000:] for o in outs]
    assert await _version(pg_server.superuser_dsn(fresh_database)) == HEAD


@pytest.mark.realinfra
async def test_mvp_041_cli_refuses_a_superuser(pg_server, fresh_database) -> None:
    r = _migrate(_env(DATABASE_URL=pg_server.superuser_dsn(fresh_database)), "upgrade", "head")
    assert r.returncode == 1
    assert "solace_migrator" in r.stderr
    assert await _version(pg_server.superuser_dsn(fresh_database)) is None


def test_mvp_001_cli_waits_then_fails_for_an_unreachable_database() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    started = time.monotonic()
    r = _migrate(_env(DATABASE_URL=f"postgresql://solace_migrator:SYNTHETIC-PW-1@127.0.0.1:{port}/solace"),
                 "upgrade", "head", "--wait-timeout", "2", timeout=120)
    assert r.returncode == 1 and time.monotonic() - started < 60
    assert "not reachable" in r.stderr
    assert "SYNTHETIC-PW-1" not in r.stdout + r.stderr


def test_mvp_001_cli_requires_a_database_url() -> None:
    r = _migrate(_env(), "upgrade", "head", timeout=120)
    assert r.returncode == 2 and "DATABASE_URL" in r.stderr


@pytest.mark.realinfra
def test_run_migrations_api_returns_the_reached_revision(pg_server, fresh_database) -> None:
    from scripts.db.migrate import run_migrations

    url = pg_server.migrator_dsn(fresh_database)
    assert run_migrations(url, "upgrade", "head") == HEAD
    assert run_migrations(url, "downgrade", "base") is None


# WS-02.4 additions: the role preflight (WS-02.3 review hand-off: both platform roles exist with the IF-03
# attributes, fail closed otherwise), fail-fast logins, the lock timeout, `current`, in-process API hygiene,
# usage errors, redaction and URL resolution.

# drift -> (SQL that breaks the role, SQL that restores it, what the refusal must say). Roles are cluster-wide,
# so each case restores the shared pg_server in a finally block.
ROLE_DRIFT: dict[str, tuple[str, str, str]] = {
    "app_bypassrls": ("ALTER ROLE solace_app BYPASSRLS", "ALTER ROLE solace_app NOBYPASSRLS",
                      "role solace_app must be NOBYPASSRLS (it is BYPASSRLS)"),
    "app_superuser": ("ALTER ROLE solace_app SUPERUSER", "ALTER ROLE solace_app NOSUPERUSER",
                      "role solace_app must be NOSUPERUSER (it is SUPERUSER)"),
    "app_member_of_migrator": ("GRANT solace_migrator TO solace_app", "REVOKE solace_migrator FROM solace_app",
                               "role solace_app must not be a member of BYPASSRLS role(s) solace_migrator"),
    "app_missing": ("ALTER ROLE solace_app RENAME TO solace_app_renamed", "ALTER ROLE solace_app_renamed RENAME TO solace_app",
                    "role solace_app does not exist"),
    "migrator_member_of_superuser": ("GRANT postgres TO solace_migrator", "REVOKE postgres FROM solace_migrator",
                                     "role solace_migrator must not be a member of superuser role(s) postgres"),
    "migrator_without_bypassrls": ("ALTER ROLE solace_migrator NOBYPASSRLS", "ALTER ROLE solace_migrator BYPASSRLS",
                                   "role solace_migrator must be BYPASSRLS (it is NOBYPASSRLS)"),
    "migrator_createrole": ("ALTER ROLE solace_migrator CREATEROLE", "ALTER ROLE solace_migrator NOCREATEROLE",
                            "role solace_migrator must be NOCREATEROLE (it is CREATEROLE)"),
    # Superuser-equivalent predefined roles (COPY ... TO PROGRAM, server file access); fix round 1, R3.
    "app_server_program": ("GRANT pg_execute_server_program TO solace_app",
                           "REVOKE pg_execute_server_program FROM solace_app",
                           "role solace_app must not be a member of superuser-equivalent role(s) "
                           "pg_execute_server_program"),
    "migrator_server_files": ("GRANT pg_read_server_files, pg_write_server_files TO solace_migrator",
                              "REVOKE pg_read_server_files, pg_write_server_files FROM solace_migrator",
                              "role solace_migrator must not be a member of superuser-equivalent role(s) "
                              "pg_read_server_files, pg_write_server_files"),
}


def _assert_no_password(r: subprocess.CompletedProcess, pg_server) -> None:
    """Fails naming the role whose password was printed, never the password itself."""
    from urllib.parse import quote

    out = r.stdout + r.stderr
    leaked = [name for name, pw in (("superuser", pg_server.superuser_password), ("app", pg_server.app_password),
                                    ("migrator", pg_server.migrator_password))
              if pw in out or quote(pw, safe="") in out]
    assert leaked == [], f"password(s) printed: {leaked}"


async def _admin(pg_server, sql: str) -> None:
    conn = await asyncpg.connect(pg_server.superuser_dsn("postgres"))
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@pytest.mark.realinfra
@pytest.mark.parametrize("drift", sorted(ROLE_DRIFT))
async def test_mvp_041_cli_refuses_platform_roles_that_drifted(pg_server, fresh_database, drift) -> None:
    apply, undo, refusal = ROLE_DRIFT[drift]
    await _admin(pg_server, apply)
    try:
        r = _migrate(_env(DATABASE_URL=pg_server.migrator_dsn(fresh_database)), "upgrade", "head", timeout=300)
    finally:
        await _admin(pg_server, undo)
    _assert_no_password(r, pg_server)
    assert r.returncode == 1, r.stderr[-3000:]
    assert "refusing to migrate" in r.stderr and refusal in r.stderr, r.stderr[-3000:]
    assert await _version(pg_server.superuser_dsn(fresh_database)) is None


@pytest.mark.realinfra
async def test_mvp_041_cli_refuses_the_app_role(pg_server, fresh_database) -> None:
    r = _migrate(_env(DATABASE_URL=pg_server.app_dsn(fresh_database)), "upgrade", "head", timeout=300)
    _assert_no_password(r, pg_server)
    assert r.returncode == 1, r.stderr[-3000:]
    assert "connected as solace_app, but migrations run only as solace_migrator" in r.stderr
    assert await _version(pg_server.superuser_dsn(fresh_database)) is None


# Fix round 1, R1: asyncpg (preflight, lock) takes user, host, port and database from the URL's netloc and path,
# while SQLAlchemy (Alembic's connection) lets query parameters override them, so any query or fragment is refused.
QUERY_SMUGGLING: tuple[str, ...] = ("as_superuser", "other_database", "other_host", "tls_option", "fragment")


@pytest.mark.realinfra
@pytest.mark.parametrize("case", QUERY_SMUGGLING)
async def test_mvp_041_cli_refuses_a_url_whose_query_could_redirect_alembic(pg_server, fresh_database, case) -> None:
    from urllib.parse import quote

    other = f"solace_other_{os.urandom(6).hex()}"
    await _admin(pg_server, f'CREATE DATABASE "{other}" OWNER solace_migrator')
    try:
        suffix = {
            "as_superuser": f"?user=postgres&password={quote(pg_server.superuser_password, safe='')}",
            "other_database": f"?database={other}",
            "other_host": "?host=other-host.invalid&port=6543",
            "tls_option": "?sslmode=require",
            "fragment": "#ignored",
        }[case]
        r = _migrate(_env(DATABASE_URL=pg_server.migrator_dsn(fresh_database) + suffix), "upgrade", "head",
                     timeout=300)
        versions = (await _version(pg_server.superuser_dsn(fresh_database)),
                    await _version(pg_server.superuser_dsn(other)))
    finally:
        await _admin(pg_server, f'DROP DATABASE IF EXISTS "{other}" WITH (FORCE)')
    _assert_no_password(r, pg_server)
    code = r.returncode
    assert code == 2, r.stderr[-3000:]
    assert "must not carry query parameters or a fragment" in r.stderr
    assert versions == (None, None)


@pytest.mark.realinfra
async def test_mvp_041_cli_refuses_a_superuser_session_acting_as_the_migrator(pg_server, fresh_database) -> None:
    # session_user stays postgres (it can RESET ROLE at any time) while current_user becomes solace_migrator.
    await _admin(pg_server, f"ALTER ROLE postgres IN DATABASE \"{fresh_database}\" SET role = 'solace_migrator'")
    try:
        r = _migrate(_env(DATABASE_URL=pg_server.superuser_dsn(fresh_database)), "upgrade", "head", timeout=300)
    finally:
        await _admin(pg_server, f'ALTER ROLE postgres IN DATABASE "{fresh_database}" RESET role')
    _assert_no_password(r, pg_server)
    code = r.returncode
    assert code == 1, r.stderr[-3000:]
    assert "connected as postgres (a superuser) acting as solace_migrator" in r.stderr
    assert await _version(pg_server.superuser_dsn(fresh_database)) is None


@pytest.mark.realinfra
async def test_migrate_async_checks_who_owns_alembic_version_afterwards(pg_server, fresh_database, monkeypatch) -> None:
    """Defence in depth for R1: whatever connection Alembic used, the version table must be the migrator's."""
    import psycopg

    from scripts.db import migrate

    url = pg_server.migrator_dsn(fresh_database)
    monkeypatch.setattr(migrate, "_run_alembic", lambda *_args: None)  # an "upgrade" that touched another database
    with pytest.raises(migrate.MigrateError, match="no alembic_version after upgrade"):
        await migrate.migrate_async(url, "upgrade", "head")

    def foreign_alembic(*_args: object) -> None:  # an "upgrade" that ran as another role
        with psycopg.connect(pg_server.superuser_dsn(fresh_database), autocommit=True) as conn:
            conn.execute("CREATE TABLE alembic_version (version_num varchar(32) PRIMARY KEY)")
            conn.execute("INSERT INTO alembic_version VALUES ('021_personality_analytics')")

    monkeypatch.setattr(migrate, "_run_alembic", foreign_alembic)
    with pytest.raises(migrate.MigrateError, match="alembic_version is owned by postgres, not solace_migrator"):
        await migrate.migrate_async(url, "upgrade", "head")


@pytest.mark.realinfra
async def test_mvp_041_cli_never_runs_as_a_superuser_even_when_it_is_the_expected_role(pg_server, fresh_database) -> None:
    r = _migrate(_env(DATABASE_URL=pg_server.superuser_dsn(fresh_database)), "upgrade", "head",
                 "--expected-role", "postgres", timeout=300)
    _assert_no_password(r, pg_server)
    assert r.returncode == 1, r.stderr[-3000:]
    assert "role postgres must be NOSUPERUSER (it is SUPERUSER)" in r.stderr
    assert await _version(pg_server.superuser_dsn(fresh_database)) is None


@pytest.mark.realinfra
async def test_mvp_001_cli_fails_fast_when_the_login_is_refused(pg_server, fresh_database) -> None:
    wrong = "WRONG-" + os.urandom(8).hex()
    started = time.monotonic()
    r = _migrate(_env(DATABASE_URL=pg_server.dsn("solace_migrator", wrong, fresh_database)), "upgrade", "head",
                 "--wait-timeout", "120", timeout=300)
    assert r.returncode == 1 and time.monotonic() - started < 60, r.stderr[-3000:]
    assert "login refused" in r.stderr
    assert wrong not in r.stdout + r.stderr


@pytest.mark.realinfra
async def test_mvp_001_cli_gives_up_when_another_job_keeps_the_lock(pg_server, fresh_database) -> None:
    holder = await asyncpg.connect(pg_server.migrator_dsn(fresh_database))
    try:
        assert await holder.fetchval("SELECT pg_try_advisory_lock(hashtext('solace_migrate'))") is True
        started = time.monotonic()
        r = _migrate(_env(DATABASE_URL=pg_server.migrator_dsn(fresh_database)), "upgrade", "head",
                     "--lock-timeout", "2", timeout=300)
        assert r.returncode == 1 and time.monotonic() - started < 60, r.stderr[-3000:]
        assert "migration lock" in r.stderr
    finally:
        await holder.close()
    assert await _version(pg_server.superuser_dsn(fresh_database)) is None


@pytest.mark.realinfra
def test_mvp_001_cli_current_reports_the_revision(pg_server, migrated_database) -> None:
    r = _migrate(_env(DATABASE_URL=pg_server.migrator_dsn(migrated_database)), "current", timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert f"migrated: {HEAD}" in r.stdout


@pytest.mark.realinfra
def test_run_migrations_leaves_the_callers_logging_alone(pg_server, migrated_database) -> None:
    import logging

    from scripts.db.migrate import run_migrations

    probe = logging.getLogger("ws024.caller.probe")
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    assert run_migrations(pg_server.migrator_dsn(migrated_database), "current") == HEAD
    assert probe.disabled is False  # alembic.ini's fileConfig would disable every existing logger
    assert root.handlers == handlers and root.level == level


@pytest.mark.realinfra
async def test_migrate_async_keeps_the_lock_until_the_worker_thread_ends(pg_server, fresh_database, monkeypatch) -> None:
    import asyncio
    import threading

    from scripts.db import migrate

    started, release = threading.Event(), threading.Event()

    def blocking_alembic(*_args: object) -> None:
        started.set()
        release.wait(60)

    monkeypatch.setattr(migrate, "_run_alembic", blocking_alembic)
    url = pg_server.migrator_dsn(fresh_database)
    task = asyncio.ensure_future(migrate.migrate_async(url, "upgrade", "head"))
    probe = await asyncpg.connect(url)
    try:
        assert await asyncio.to_thread(started.wait, 60)
        # The worker thread cannot be stopped, so the cancelled job still holds the lock while it runs, however
        # often it is cancelled (a double Ctrl-C cancels the main task twice; fix round 1, R2).
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0.5)
            assert await probe.fetchval("SELECT pg_try_advisory_lock(hashtext('solace_migrate'))") is False
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await probe.fetchval("SELECT pg_try_advisory_lock(hashtext('solace_migrate'))") is True
    finally:
        release.set()
        await probe.close()


async def test_run_migrations_refuses_a_running_event_loop() -> None:
    from scripts.db.migrate import run_migrations

    with pytest.raises(RuntimeError, match="migrate_async"):
        run_migrations("postgresql://solace_migrator:SYNTHETIC-PW-4@127.0.0.1:1/solace", "upgrade", "head")


def test_mvp_001_cli_downgrade_needs_an_explicit_target() -> None:
    r = _migrate(_env(DATABASE_URL="postgresql://solace_migrator:SYNTHETIC-PW-2@127.0.0.1:1/solace"), "downgrade",
                 timeout=120)
    assert r.returncode == 2 and "downgrade" in r.stderr
    assert "SYNTHETIC-PW-2" not in r.stdout + r.stderr


@pytest.mark.parametrize("url", ["mysql://solace_migrator:SYNTHETIC-PW-3@127.0.0.1:1/solace",
                                 "postgresql://solace_migrator:SYNTHETIC-PW-3@127.0.0.1:notaport/solace",
                                 "SYNTHETIC-PW-3",
                                 # an unencoded '@': drivers split at the first '@', urlsplit at the last (R4)
                                 "postgresql://solace_migrator:SYNTHETIC-PW-3@x@127.0.0.1:1/solace"])
def test_mvp_001_cli_rejects_an_unusable_database_url_without_echoing_it(url: str) -> None:
    started = time.monotonic()
    r = _migrate(_env(DATABASE_URL=url), "upgrade", "head", "--wait-timeout", "3", timeout=120)
    assert r.returncode == 2 and "DATABASE_URL" in r.stderr
    assert time.monotonic() - started < 60
    assert "SYNTHETIC-PW-3" not in r.stdout + r.stderr


def test_redact_shows_where_the_url_points_but_no_password() -> None:
    from scripts.db.migrate import redact

    url = "postgresql://solace_migrator:p%2F%40ss%3A%25w0rd@db.internal:6543/solace?password=other&sslmode=require"
    assert redact(url) == "solace_migrator:***@db.internal:6543/solace"
    assert redact("postgresql+asyncpg://solace_app@[::1]/solace") == "solace_app@[::1]:5432/solace"
    for broken in ("postgresql://u:SYNTHETIC-PW-5@h:notaport/d", "SYNTHETIC-PW-5", ""):
        assert "SYNTHETIC-PW-5" not in redact(broken)


def test_resolve_database_url_prefers_database_url_then_postgres_settings(monkeypatch) -> None:
    from scripts.db.migrate import MigrateUsageError, resolve_database_url

    monkeypatch.setenv("POSTGRES_HOST", "from-os-environ")  # only the mapping counts
    assert resolve_database_url({"DATABASE_URL": "postgresql://a:b@h/d", "POSTGRES_PASSWORD": "x"}) == "postgresql://a:b@h/d"
    assert resolve_database_url({}) is None
    assert resolve_database_url({"DATABASE_URL": "", "POSTGRES_USER": "solace_migrator"}) is None
    assert resolve_database_url({"POSTGRES_PASSWORD": "pw"}) == "postgresql://solace:pw@localhost:5432/solace"
    assert resolve_database_url({"POSTGRES_HOST": "postgres", "POSTGRES_PORT": "5433", "POSTGRES_DATABASE": "solace",
                                 "POSTGRES_USER": "solace_migrator", "POSTGRES_PASSWORD": "p/@#:%w"}) == (
        "postgresql://solace_migrator:p%2F%40%23%3A%25w@postgres:5433/solace")
    with pytest.raises(MigrateUsageError, match="POSTGRES_PORT") as excinfo:
        resolve_database_url({"POSTGRES_PASSWORD": "SYNTHETIC-PW-6", "POSTGRES_PORT": "notaport"})
    assert "SYNTHETIC-PW-6" not in str(excinfo.value)
