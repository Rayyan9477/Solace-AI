# tests/migrations/test_mvp_010_tls_real_pg.py
"""MVP-010/271: the bundled Postgres speaks TLS, strict clients verify it, and plaintext is refused."""
from __future__ import annotations

import os
import secrets
import ssl
import subprocess
import sys
from pathlib import Path

import asyncpg
import docker
import pytest

from scripts.db import gen_certs
from scripts.db.images import POSTGRES_IMAGE
from solace_infrastructure.postgres import PostgresSettings

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.realinfra


@pytest.fixture(scope="module")
def tls_pg(tmp_path_factory):
    from conftest import start_postgres  # tests/migrations/conftest.py helper (prepend import mode)

    certs = tmp_path_factory.mktemp("tls")
    assert gen_certs.main(["--out", str(certs), "--san", "localhost", "--ip", "127.0.0.1"]) == 0
    pw = {"POSTGRES_PASSWORD": secrets.token_urlsafe(18), "SOLACE_APP_PASSWORD": secrets.token_urlsafe(18),
          "SOLACE_MIGRATOR_PASSWORD": secrets.token_urlsafe(18)}
    container = start_postgres(env=pw, tls_dir=certs, tls_mode="required")
    try:
        yield {"host": container.get_container_host_ip(), "port": int(container.get_exposed_port(5432)),
               "ca": certs / "postgres-ca.crt", **pw}
    finally:
        container.stop()


def _settings(tls_pg, monkeypatch, ca: Path) -> PostgresSettings:
    monkeypatch.setenv("ENVIRONMENT", "production")
    return PostgresSettings(host=tls_pg["host"], port=tls_pg["port"], user="solace_app",
                            password=tls_pg["SOLACE_APP_PASSWORD"], database="solace", ssl_root_cert=str(ca))


def _dsn(tls_pg) -> str:
    """The solace_app DSN, built here: the test server is reachable only through a loopback host, which a strict
    ``PostgresSettings.get_dsn()`` refuses (WS-02.5 Controller Ruling; see the test after the brief block)."""
    return f"postgresql://solace_app:{tls_pg['SOLACE_APP_PASSWORD']}@{tls_pg['host']}:{tls_pg['port']}/solace"


async def test_mvp_010_strict_client_connects_with_verify_full(tls_pg, monkeypatch) -> None:
    s = _settings(tls_pg, monkeypatch, tls_pg["ca"])
    assert s.get_effective_ssl_mode() == "verify-full"
    conn = await asyncpg.connect(_dsn(tls_pg), ssl=s.get_ssl_context())
    try:
        assert await conn.fetchval("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()") is True
    finally:
        await conn.close()


async def test_mvp_271_a_certificate_from_another_ca_is_rejected(tls_pg, monkeypatch, tmp_path: Path) -> None:
    assert gen_certs.main(["--out", str(tmp_path), "--san", "localhost"]) == 0
    s = _settings(tls_pg, monkeypatch, tmp_path / "postgres-ca.crt")
    with pytest.raises((ssl.SSLCertVerificationError, ssl.SSLError)):
        await asyncpg.connect(_dsn(tls_pg), ssl=s.get_ssl_context())


async def test_mvp_010_required_server_rejects_plaintext_clients(tls_pg, monkeypatch) -> None:
    _settings(tls_pg, monkeypatch, tls_pg["ca"])
    with pytest.raises(asyncpg.exceptions.InvalidAuthorizationSpecificationError):
        await asyncpg.connect(_dsn(tls_pg), ssl=False)


def test_mvp_010_required_server_without_certificates_fails_closed(tmp_path: Path) -> None:
    client = docker.from_env()
    container = client.containers.run(
        POSTGRES_IMAGE, ["postgres"], entrypoint=["sh", "/usr/local/bin/solace-pg-entrypoint.sh"], detach=True,
        environment={"POSTGRES_PASSWORD": secrets.token_urlsafe(16), "SOLACE_PG_TLS": "required"},
        volumes={str(ROOT / "scripts/db/pg_tls_entrypoint.sh"): {"bind": "/usr/local/bin/solace-pg-entrypoint.sh", "mode": "ro"},
                 str(tmp_path): {"bind": "/run/secrets", "mode": "ro"}})
    try:
        status = container.wait(timeout=60)["StatusCode"]
        logs = container.logs().decode("utf-8", "replace")
    finally:
        container.remove(force=True, v=True)  # WS-02.5: v=True also removes the image's anonymous PGDATA volume
    assert status != 0 and "server.crt" in logs


def test_mvp_010_migrate_job_runs_over_verify_full(tls_pg) -> None:
    url = (f"postgresql://solace_migrator:{tls_pg['SOLACE_MIGRATOR_PASSWORD']}@{tls_pg['host']}:{tls_pg['port']}/solace")
    env = {k: v for k, v in os.environ.items() if not k.startswith("POSTGRES_")}
    env.update({"PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "production", "DATABASE_URL": url,
                "POSTGRES_SSL_MODE": "verify-full", "POSTGRES_SSL_ROOT_CERT": str(tls_pg["ca"])})
    r = subprocess.run([sys.executable, "-m", "scripts.db.migrate", "upgrade", "head"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]


# --- WS-02.5 additions -------------------------------------------------------------------------------------------
# The migrate job and migrations/env.py each verify the server (a client that silently fell back to unverified
# TLS would still pass the test above, because the server accepts any TLS client); the required server's hba
# rules; the optional (dev) mode; and the entrypoint's fail-closed cases. Every container is removed with its
# volumes; certificates live in pytest's tmp_path only.

TLS_DIR_IN_CONTAINER = "/var/lib/postgresql/tls"
ENTRYPOINT_IN_CONTAINER = "/usr/local/bin/solace-pg-entrypoint.sh"


def _migrator_url(tls_pg) -> str:
    return f"postgresql://solace_migrator:{tls_pg['SOLACE_MIGRATOR_PASSWORD']}@{tls_pg['host']}:{tls_pg['port']}/solace"


def _strict_env(url: str, ca: Path | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("POSTGRES_") and k != "DATABASE_URL_FILE"}
    env.update({"PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "production", "DATABASE_URL": url,
                "POSTGRES_SSL_MODE": "verify-full"})
    if ca is not None:
        env["POSTGRES_SSL_ROOT_CERT"] = str(ca)
    return env


def _run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, capture_output=True, text=True,  # noqa: S603
                          timeout=600, check=False)


def _other_ca(tmp_path: Path) -> Path:
    assert gen_certs.main(["--out", str(tmp_path), "--san", "localhost", "--ip", "127.0.0.1"]) == 0
    return tmp_path / "postgres-ca.crt"


def test_ruling_strict_settings_refuse_the_loopback_test_host(tls_pg, monkeypatch) -> None:
    """Why the brief block uses _dsn(): strict PostgresSettings never builds a DSN for a local host."""
    from solace_infrastructure.postgres import DatabaseConfigError

    s = _settings(tls_pg, monkeypatch, tls_pg["ca"])
    if tls_pg["host"] not in ("localhost", "127.0.0.1", "::1"):
        pytest.skip("the Docker host is not local here")
    with pytest.raises(DatabaseConfigError, match="POSTGRES_HOST"):
        s.get_dsn()


def test_mvp_271_migrate_job_refuses_a_server_from_another_ca(tls_pg, tmp_path: Path) -> None:
    r = _run(["-m", "scripts.db.migrate", "current", "--wait-timeout", "5"], _strict_env(_migrator_url(tls_pg),
                                                                                         _other_ca(tmp_path)))
    output = r.stdout + r.stderr
    assert r.returncode == 1, output[-3000:]
    # Refused at the entrypoint's own login (step 2), not later by Alembic's engine: both verify.
    assert "cannot connect to" in output and "SSLCertVerificationError" in output, output[-3000:]
    assert "current failed" not in output
    assert tls_pg["SOLACE_MIGRATOR_PASSWORD"] not in output


def test_mvp_271_env_py_verifies_the_server_certificate(tls_pg, tmp_path: Path) -> None:
    """Alembic's own engine (migrations/env.py), without the migrate entrypoint in front of it."""
    alembic = ["-m", "alembic", "-c", str(ROOT / "alembic.ini"), "current"]
    bad = _run(alembic, _strict_env(_migrator_url(tls_pg), _other_ca(tmp_path)))
    assert bad.returncode != 0
    assert "certificate verify failed" in bad.stderr or "SSLCertVerificationError" in bad.stderr, bad.stderr[-3000:]
    good = _run(alembic, _strict_env(_migrator_url(tls_pg), tls_pg["ca"]))
    assert good.returncode == 0, good.stderr[-3000:]


def test_mvp_010_strict_migrate_without_a_root_cert_is_a_usage_error() -> None:
    """No server needed: the TLS context is built before the first login attempt."""
    password = secrets.token_urlsafe(12)
    r = _run(["-m", "scripts.db.migrate", "current", "--wait-timeout", "1"],
             _strict_env(f"postgresql://solace_migrator:{password}@127.0.0.1:1/solace", None))
    output = r.stdout + r.stderr
    assert r.returncode == 2, output[-3000:]
    assert "POSTGRES_SSL_ROOT_CERT" in output and password not in output


async def test_mvp_010_required_server_hba_is_tls_only_over_tcp(tls_pg, monkeypatch) -> None:
    s = _settings(tls_pg, monkeypatch, tls_pg["ca"])
    conn = await asyncpg.connect(host=tls_pg["host"], port=tls_pg["port"], user="postgres", database="postgres",
                                 password=tls_pg["POSTGRES_PASSWORD"], ssl=s.get_ssl_context())
    try:
        assert await conn.fetchval("SHOW ssl") == "on"
        assert await conn.fetchval("SHOW hba_file") == f"{TLS_DIR_IN_CONTAINER}/pg_hba.conf"
        assert await conn.fetchval("SHOW ssl_key_file") == f"{TLS_DIR_IN_CONTAINER}/server.key"
        rules = await conn.fetch("SELECT type, database, user_name, auth_method, error FROM pg_hba_file_rules "
                                 "ORDER BY rule_number")
    finally:
        await conn.close()
    assert [(r["type"], r["database"], r["user_name"], r["auth_method"], r["error"]) for r in rules] == [
        ("local", ["all"], ["all"], "trust", None),
        ("hostssl", ["all"], ["all"], "scram-sha-256", None),
        ("host", ["all"], ["all"], "reject", None),
    ]


async def test_mvp_010_optional_mode_serves_tls_and_plaintext(tmp_path: Path, monkeypatch) -> None:
    """The dev profile (SOLACE_PG_TLS=optional): TLS is offered and verifiable, plaintext still works."""
    from conftest import start_postgres

    certs = tmp_path / "tls"
    assert gen_certs.main(["--out", str(certs), "--san", "localhost", "--ip", "127.0.0.1"]) == 0
    pw = {"POSTGRES_PASSWORD": secrets.token_urlsafe(18), "SOLACE_APP_PASSWORD": secrets.token_urlsafe(18),
          "SOLACE_MIGRATOR_PASSWORD": secrets.token_urlsafe(18)}
    container = start_postgres(env=pw, tls_dir=certs, tls_mode="optional")
    try:
        wrapped = container.get_wrapped_container()
        modes = wrapped.exec_run(["stat", "-c", "%n %a %U", TLS_DIR_IN_CONTAINER, f"{TLS_DIR_IN_CONTAINER}/server.key"])
        assert modes.exit_code == 0
        assert modes.output.decode().split("\n")[:2] == [f"{TLS_DIR_IN_CONTAINER} 700 postgres",
                                                         f"{TLS_DIR_IN_CONTAINER}/server.key 600 postgres"]
        server = {"host": container.get_container_host_ip(), "port": int(container.get_exposed_port(5432)),
                  "SOLACE_APP_PASSWORD": pw["SOLACE_APP_PASSWORD"]}
        s = _settings(server, monkeypatch, certs / "postgres-ca.crt")
        for ssl_arg, expected in ((s.get_ssl_context(), True), (False, False)):
            conn = await asyncpg.connect(_dsn(server), ssl=ssl_arg)
            try:
                assert await conn.fetchval("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()") is expected
            finally:
                await conn.close()
    finally:
        container.stop()


@pytest.mark.parametrize(("tls_mode", "files", "named"), [
    ("required", ("server.key",), "server.crt"),
    ("optional", ("server.crt",), "server.key"),
    ("bogus", ("server.crt", "server.key"), "SOLACE_PG_TLS"),
])
def test_mvp_010_entrypoint_fails_closed(tmp_path: Path, tls_mode: str, files: tuple[str, ...], named: str) -> None:
    """A half-provisioned TLS directory, or an unknown SOLACE_PG_TLS, stops the container before initdb."""
    certs = tmp_path / "certs"
    assert gen_certs.main(["--out", str(certs)]) == 0
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    for name in files:
        (secrets_dir / name).write_bytes((certs / name).read_bytes())
    client = docker.from_env()
    try:
        container = client.containers.run(
            POSTGRES_IMAGE, ["postgres"], entrypoint=["sh", ENTRYPOINT_IN_CONTAINER], detach=True,
            environment={"POSTGRES_PASSWORD": secrets.token_urlsafe(16), "SOLACE_PG_TLS": tls_mode},
            volumes={str(ROOT / "scripts/db/pg_tls_entrypoint.sh"): {"bind": ENTRYPOINT_IN_CONTAINER, "mode": "ro"},
                     str(secrets_dir): {"bind": "/run/secrets", "mode": "ro"}})
        try:
            status = container.wait(timeout=60)["StatusCode"]
            logs = container.logs().decode("utf-8", "replace")
        finally:
            container.remove(force=True, v=True)
    finally:
        client.close()
    assert status == 1, logs[-2000:]
    assert named in logs and "BEGIN" not in logs
