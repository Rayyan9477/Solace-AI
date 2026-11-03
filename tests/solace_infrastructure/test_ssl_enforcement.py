"""H-39 / MVP-271 / MVP-010: the PostgreSQL TLS policy (D-15, D-14).

PHI transiting a service->DB connection unverified is a HIPAA exposure. The policy lives in two module
functions of ``solace_infrastructure.postgres``, ``effective_ssl_mode`` and ``build_ssl_context``;
``PostgresSettings.get_effective_ssl_mode`` / ``get_ssl_context`` delegate to them, ``ConnectionPoolManager.get_pool``
and ``PostgresClient.connect`` hand the context to asyncpg, and migrations/env.py and scripts/db/migrate.py build
theirs with the same function.

These tests pin the contract:
  1. staging and production (strict): the effective mode is always ``verify-full`` whatever was configured (a
     WARNING says so); the context requires a certificate chained to POSTGRES_SSL_ROOT_CERT, and to it alone, plus
     a matching host name; an unset or unreadable root certificate is a ``DatabaseConfigError`` that names
     POSTGRES_SSL_ROOT_CERT, never its value;
  2. development keeps the configured mode (``prefer`` gives None, so a local non-TLS Postgres keeps working) -
     this dev/strict contrast is what gives the strict assertions teeth;
  3. the context actually reaches ``asyncpg.create_pool``.
"""
from __future__ import annotations

import ssl
from pathlib import Path

import pytest

from solace_infrastructure.postgres import PostgresSettings


def _settings(ssl_mode: str = "prefer", ssl_root_cert: Path | str | None = None,
              host: str = "localhost") -> PostgresSettings:
    return PostgresSettings(ssl_mode=ssl_mode, password="test_password", host=host,
                            ssl_root_cert=str(ssl_root_cert) if ssl_root_cert is not None else None)


class _AppRoleConn:
    """Answers verify_app_role's query (WS-02.6) as a plain application role."""

    async def fetchrow(self, sql, *args, **kwargs):
        assert "rolbypassrls" in sql
        return {"login_role": "solace_app", "effective_role": "solace_app", "roles_found": 1, "role_switched": False,
                "is_superuser": False, "bypasses_rls": False, "owns_public_tables": False,
                "member_of_privileged_role": False, "member_of_superuser_equivalent_role": False,
                "can_replicate": False}


class _AcquireCM:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _AppRolePool:
    """A stand-in pool: strict pools check the login role on one connection before use (WS-02.6)."""

    def acquire(self, timeout=None):
        return _AcquireCM(_AppRoleConn())

    async def close(self) -> None: ...


@pytest.fixture
def root_cert(tmp_path: Path) -> Path:
    """A deploy-time CA (and, beside it, server.crt/server.key for host 'postgres') from scripts/db/gen_certs.py."""
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path)]) == 0
    return tmp_path / "postgres-ca.crt"


# --- strict environments (brief block, verbatim) -----------------------------------------------------------------
@pytest.mark.parametrize("env", ["production", "staging"])
@pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"])
def test_mvp_271_strict_effective_mode_is_verify_full(monkeypatch, env, mode) -> None:
    monkeypatch.setenv("ENVIRONMENT", env)
    assert _settings(mode).get_effective_ssl_mode() == "verify-full"


def test_mvp_271_strict_context_verifies_chain_and_hostname(monkeypatch, root_cert) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    ctx = PostgresSettings(password="test_password", ssl_root_cert=str(root_cert)).get_ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True


@pytest.mark.parametrize("root", [None, "does-not-exist.crt"])
def test_mvp_010_strict_without_a_readable_root_cert_fails_loud(monkeypatch, tmp_path, root) -> None:
    from solace_infrastructure.postgres import DatabaseConfigError

    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(DatabaseConfigError, match="POSTGRES_SSL_ROOT_CERT"):
        PostgresSettings(password="test_password", ssl_root_cert=str(tmp_path / root) if root else None).get_ssl_context()


# --- the H-39 tests, moved to the new contract -------------------------------------------------------------------
class TestSslModeEnforcement:
    @pytest.mark.parametrize("permissive", ["disable", "allow", "prefer"])
    def test_development_keeps_permissive_mode(
        self, monkeypatch: pytest.MonkeyPatch, permissive: str
    ) -> None:
        """Development must NOT be upgraded - local non-SSL Postgres keeps working."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        assert _settings(permissive).get_effective_ssl_mode() == permissive

    @pytest.mark.parametrize("strict", ["require", "verify-ca", "verify-full"])
    def test_strict_mode_preserved(
        self, monkeypatch: pytest.MonkeyPatch, strict: str
    ) -> None:
        """Outside strict environments an explicitly strict mode is kept as configured."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        assert _settings(strict).get_effective_ssl_mode() == strict


class TestSslContextEnforcement:
    def test_production_yields_real_ssl_context(
        self, monkeypatch: pytest.MonkeyPatch, root_cert: Path
    ) -> None:
        """Prod + permissive mode -> a real, verifying SSLContext."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        ctx = _settings("prefer", root_cert).get_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_development_permissive_yields_no_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dev + permissive mode -> None (asyncpg negotiates, local dev works)."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        assert _settings("prefer").get_ssl_context() is None

    def test_production_cannot_disable_ssl(
        self, monkeypatch: pytest.MonkeyPatch, root_cert: Path
    ) -> None:
        """Even an explicit 'disable' cannot turn off encryption (or verification) in production."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        ctx = _settings("disable", root_cert).get_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_mode == ssl.CERT_REQUIRED


class TestSslContextReachesPool:
    """The exact SSLContext must be handed to asyncpg.create_pool (H-39 deliverable)."""

    def test_ssl_context_passed_to_create_pool(
        self, monkeypatch: pytest.MonkeyPatch, root_cert: Path
    ) -> None:
        import asyncio

        import asyncpg

        from solace_infrastructure.database.connection_manager import (
            ConnectionPoolManager,
        )

        monkeypatch.setenv("ENVIRONMENT", "production")
        captured: dict = {}

        async def _fake_create_pool(*args, **kwargs):
            captured.update(kwargs)
            return _AppRolePool()  # stand-in pool; its connection answers the strict role check as solace_app

        monkeypatch.setattr(asyncpg, "create_pool", _fake_create_pool)

        name = "h39_ssl_test"

        async def _run() -> None:
            try:
                # A remote host: strict environments refuse a local one (Controller Ruling, below).
                await ConnectionPoolManager.register_pool(name, _settings("prefer", root_cert, host="db.internal"))
                await ConnectionPoolManager.get_pool(name)
            finally:
                # Clean up class-level state so we don't leak into other tests.
                ConnectionPoolManager._pools.pop(name, None)
                ConnectionPoolManager._pool_configs.pop(name, None)

        asyncio.run(_run())

        assert "ssl" in captured
        assert isinstance(captured["ssl"], ssl.SSLContext), (
            "H-39: a real SSLContext must be passed to asyncpg.create_pool in production"
        )
        assert captured["ssl"].verify_mode == ssl.CERT_REQUIRED


# --- WS-02.5 additions -------------------------------------------------------------------------------------------
from solace_infrastructure import postgres as pg  # noqa: E402

STRICT_ENVIRONMENTS = ["production", "staging", "  Production ", "STAGING"]
OTHER_ENVIRONMENTS = ["development", "test", "", "prod", "staging-2"]


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def record(event: str, **kw) -> None:
            self.events.append((level, event, kw))

        return record


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    recorder = _RecordingLogger()
    monkeypatch.setattr(pg, "logger", recorder)
    return recorder


def _handshake(client_ctx: ssl.SSLContext, server_ctx: ssl.SSLContext, server_hostname: str) -> None:
    """A complete TLS handshake over memory BIOs (no sockets); raises what the client raises."""
    c_in, c_out, s_in, s_out = ssl.MemoryBIO(), ssl.MemoryBIO(), ssl.MemoryBIO(), ssl.MemoryBIO()
    client = client_ctx.wrap_bio(c_in, c_out, server_hostname=server_hostname)
    server = server_ctx.wrap_bio(s_in, s_out, server_side=True)
    client_done = server_done = False
    for _ in range(50):
        if not client_done:
            try:
                client.do_handshake()
                client_done = True
            except ssl.SSLWantReadError:
                pass
        s_in.write(c_out.read())
        if not server_done:
            try:
                server.do_handshake()
                server_done = True
            except ssl.SSLWantReadError:
                pass
            except ssl.SSLError:
                pass  # the client aborted; its own error is raised on the next turn
        c_in.write(s_out.read())
        if client_done and server_done:
            return
    raise AssertionError("TLS handshake did not complete")


def _server_ctx(cert_dir: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_dir / "server.crt", cert_dir / "server.key")
    return ctx


@pytest.mark.parametrize("env", STRICT_ENVIRONMENTS)
def test_is_strict_environment_matches_if01(monkeypatch: pytest.MonkeyPatch, env: str) -> None:
    from solace_security.production_guards import is_strict_environment

    assert pg._is_strict_environment(env) is True is is_strict_environment(env)
    monkeypatch.setenv("ENVIRONMENT", env)
    assert pg._is_strict_environment() is True


@pytest.mark.parametrize("env", OTHER_ENVIRONMENTS)
def test_other_environments_are_not_strict(monkeypatch: pytest.MonkeyPatch, env: str) -> None:
    from solace_security.production_guards import is_strict_environment

    assert pg._is_strict_environment(env) is False is is_strict_environment(env)
    monkeypatch.setenv("ENVIRONMENT", env)
    assert pg._is_strict_environment() is False


def test_unset_environment_is_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert pg._is_strict_environment() is False
    assert pg.effective_ssl_mode("prefer") == "prefer"
    assert pg.build_ssl_context("prefer", None) is None


def test_explicit_environment_wins_over_the_variable(monkeypatch: pytest.MonkeyPatch, root_cert: Path) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert pg.effective_ssl_mode("prefer", environment="production") == "verify-full"
    assert pg.build_ssl_context("prefer", str(root_cert), environment="staging").verify_mode == ssl.CERT_REQUIRED
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert pg.effective_ssl_mode("prefer", environment="development") == "prefer"
    assert pg.build_ssl_context("prefer", None, environment="development") is None


def test_strict_upgrade_logs_one_warning(recorded: _RecordingLogger) -> None:
    assert pg.effective_ssl_mode("require", environment="staging") == "verify-full"
    assert recorded.events == [("warning", "ssl_mode_upgraded", {
        "original_mode": "require", "effective_mode": "verify-full", "environment": "staging",
        "reason": "strict environments verify the server certificate and host name (D-15)"})]


def test_strict_verify_full_logs_nothing(recorded: _RecordingLogger) -> None:
    assert pg.effective_ssl_mode("verify-full", environment="production") == "verify-full"
    assert recorded.events == []


def test_strict_upgrade_never_logs_an_unrecognised_value(recorded: _RecordingLogger) -> None:
    assert pg.effective_ssl_mode("hunter2-not-a-mode", environment="production") == "verify-full"
    assert [kw["original_mode"] for _, _, kw in recorded.events] == ["<unrecognised>"]


def test_strict_context_trusts_only_the_root_cert(root_cert: Path) -> None:
    ctx = pg.build_ssl_context("require", str(root_cert), environment="production")
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2
    cas = ctx.get_ca_certs()
    assert len(cas) == 1  # the system store is not loaded
    assert dict(item for rdn in cas[0]["subject"] for item in rdn)["commonName"].startswith("Solace-AI Postgres CA")


def test_strict_context_accepts_the_generated_server_for_its_names(root_cert: Path) -> None:
    ctx = pg.build_ssl_context("verify-full", str(root_cert), environment="production")
    for name in ("postgres", "localhost", "127.0.0.1"):
        _handshake(ctx, _server_ctx(root_cert.parent), name)


def test_strict_context_rejects_another_host_name(root_cert: Path) -> None:
    ctx = pg.build_ssl_context("disable", str(root_cert), environment="production")
    with pytest.raises(ssl.SSLCertVerificationError, match="match"):
        _handshake(ctx, _server_ctx(root_cert.parent), "db.attacker.example")


def test_strict_context_rejects_a_server_from_another_ca(tmp_path: Path, root_cert: Path) -> None:
    from scripts.db import gen_certs

    other = tmp_path / "other"
    assert gen_certs.main(["--out", str(other)]) == 0
    ctx = pg.build_ssl_context("verify-full", str(root_cert), environment="production")
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(ctx, _server_ctx(other), "postgres")


@pytest.mark.parametrize("kind", ["blank", "directory", "not-a-certificate", "a-private-key"])
def test_strict_root_cert_problems_name_the_variable_never_the_value(tmp_path: Path, root_cert: Path,
                                                                     kind: str) -> None:
    secret_looking = tmp_path / "s3cr3t-path-component"
    secret_looking.mkdir()
    value = {"blank": "   ", "directory": str(secret_looking)}.get(kind)
    if kind == "not-a-certificate":
        (secret_looking / "ca.crt").write_text("not a certificate\n", encoding="ascii")
        value = str(secret_looking / "ca.crt")
    elif kind == "a-private-key":
        (secret_looking / "ca.crt").write_bytes((root_cert.parent / "server.key").read_bytes())
        value = str(secret_looking / "ca.crt")
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_SSL_ROOT_CERT") as excinfo:
        pg.build_ssl_context("verify-full", value, environment="production")
    message = str(excinfo.value)
    assert "s3cr3t" not in message and "PRIVATE" not in message
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__ is True


def test_database_config_error_is_a_database_error() -> None:
    from solace_common.exceptions import DatabaseError

    assert issubclass(pg.DatabaseConfigError, DatabaseError)


def test_development_require_keeps_encryption_without_verification() -> None:
    ctx = pg.build_ssl_context("require", None, environment="development")
    assert ctx.verify_mode == ssl.CERT_NONE and ctx.check_hostname is False


@pytest.mark.parametrize(("mode", "check_hostname"), [("verify-ca", False), ("verify-full", True)])
def test_development_verify_modes_with_a_root_cert_trust_only_it(root_cert: Path, mode: str,
                                                                 check_hostname: bool) -> None:
    ctx = pg.build_ssl_context(mode, str(root_cert), environment="development")
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is check_hostname
    assert len(ctx.get_ca_certs()) == 1


@pytest.mark.parametrize("mode", ["disable", "allow", "prefer"])
def test_development_permissive_modes_yield_no_context(mode: str) -> None:
    assert pg.build_ssl_context(mode, None, environment="development") is None


def test_development_unknown_mode_fails_loud_naming_the_variable() -> None:
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_SSL_MODE") as excinfo:
        pg.build_ssl_context("verify_full", None, environment="development")
    assert "verify_full" not in str(excinfo.value)


def test_postgres_client_connect_passes_the_strict_context(monkeypatch: pytest.MonkeyPatch, root_cert: Path) -> None:
    import asyncio

    import asyncpg

    monkeypatch.setenv("ENVIRONMENT", "staging")
    captured: dict = {}

    async def _fake_create_pool(*args, **kwargs):
        captured.update(kwargs)
        return _AppRolePool()

    monkeypatch.setattr(asyncpg, "create_pool", _fake_create_pool)
    asyncio.run(pg.PostgresClient(_settings("require", root_cert, host="db.internal")).connect(max_retries=1))
    assert captured["ssl"].verify_mode == ssl.CERT_REQUIRED and captured["ssl"].check_hostname is True


def test_strict_pool_without_a_root_cert_fails_before_connecting(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import asyncpg

    from solace_infrastructure.database.connection_manager import ConnectionPoolManager

    monkeypatch.setenv("ENVIRONMENT", "production")
    calls: list[dict] = []

    async def _fake_create_pool(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(asyncpg, "create_pool", _fake_create_pool)
    name = "mvp010_no_root_cert"

    async def _run() -> None:
        try:
            # A remote host (WS-02.6: register_pool validates the settings), so the root cert is the only problem.
            await ConnectionPoolManager.register_pool(name, _settings("verify-full", host="db.internal"))
            await ConnectionPoolManager.get_pool(name)
        finally:
            ConnectionPoolManager._pools.pop(name, None)
            ConnectionPoolManager._pool_configs.pop(name, None)
            ConnectionPoolManager._locks.pop(name, None)

    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_SSL_ROOT_CERT"):
        asyncio.run(_run())
    assert calls == []


# --- Controller Ruling (WS-02.5): strict environments refuse a missing or local POSTGRES_HOST -------------------
# Unset, the host silently defaulted to localhost. In staging and production PostgresSettings.validate_for_environment
# (called by get_dsn(), so ConnectionPoolManager.register_pool/get_pool and PostgresClient.connect) raises
# DatabaseConfigError naming POSTGRES_HOST, before any connection attempt; elsewhere localhost stays the default.
import os  # noqa: E402

import asyncpg  # noqa: E402

LOCAL_HOSTS = ["localhost", "127.0.0.1", "::1", "", "   ", "LOCALHOST", " localhost ", "[::1]", "127.0.0.2",
               "0.0.0.0", "::", "::ffff:127.0.0.1", "db.localhost", "/var/run/postgresql",  # noqa: S104 - data
               "db.internal,localhost",
               # Fix round 1 (R1): asyncpg percent-decodes the host and cuts the DSN at '/', '?' and '#', so these
               # reach a unix socket (no TLS) or loopback although the raw string looks remote.
               "%2Fvar%2Frun%2Fpostgresql", "%6Cocalhost", "%31%32%37.0.0.1", "db.internal,%6Cocalhost",
               "localhost/x", "localhost?x=1", "localhost#x", "[%3A%3A1]",
               # Legacy inet_aton spellings of 127.0.0.1 that getaddrinfo still accepts.
               "127.1", "2130706433", "0x7f.0.0.1",
               # Not a plain host: refused rather than guessed at (asyncpg would read each one differently).
               "x@localhost", "localhost:6432", "::abcd", "db.internal,", "l\N{CYRILLIC SMALL LETTER O}calhost",
               "db internal"]
REMOTE_HOSTS = ["postgres", "db.internal", "db.internal.", "solace_postgres", "postgres-primary.svc.cluster.local",
                "10.0.0.5", "[fd00::5]", "db1.internal,db2.internal", "PostgreS"]
SYNTHETIC_PASSWORD = "synthetic-pw-ruling-7"


@pytest.fixture
def no_postgres_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # never read the repository's .env
    for key in [k for k in os.environ if k.startswith("POSTGRES_")]:
        monkeypatch.delenv(key)


@pytest.fixture
def create_pool_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    async def _fake(*args, **kwargs):
        calls.append(kwargs)
        return _AppRolePool()  # strict pools check the login role (WS-02.6)

    monkeypatch.setattr(asyncpg, "create_pool", _fake)
    return calls


def _get_pool(name: str, settings: PostgresSettings | None = None) -> None:
    """register_pool (unless the pool auto-registers from POSTGRES_*) + get_pool, cleaning the class state."""
    import asyncio

    from solace_infrastructure.database.connection_manager import ConnectionPoolManager

    async def _run() -> None:
        try:
            if settings is not None:
                await ConnectionPoolManager.register_pool(name, settings)
            await ConnectionPoolManager.get_pool(name)
        finally:
            ConnectionPoolManager._pools.pop(name, None)
            ConnectionPoolManager._pool_configs.pop(name, None)
            ConnectionPoolManager._locks.pop(name, None)

    asyncio.run(_run())


@pytest.mark.parametrize("env", ["production", "staging"])
def test_ruling_strict_pool_refuses_an_unset_postgres_host(monkeypatch, no_postgres_env, create_pool_calls,
                                                           root_cert, env) -> None:
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("POSTGRES_PASSWORD", SYNTHETIC_PASSWORD)
    monkeypatch.setenv("POSTGRES_SSL_ROOT_CERT", str(root_cert))  # so the host is the only problem
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_HOST") as excinfo:
        _get_pool("ruling_unset_host")  # auto-registers PostgresSettings() from POSTGRES_*
    assert SYNTHETIC_PASSWORD not in str(excinfo.value)
    assert create_pool_calls == []


@pytest.mark.parametrize("host", LOCAL_HOSTS)
def test_ruling_strict_pool_refuses_a_local_postgres_host(monkeypatch, no_postgres_env, create_pool_calls,
                                                          root_cert, host) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("POSTGRES_HOST", host)
    monkeypatch.setenv("POSTGRES_PASSWORD", SYNTHETIC_PASSWORD)
    monkeypatch.setenv("POSTGRES_SSL_ROOT_CERT", str(root_cert))
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_HOST") as excinfo:
        _get_pool("ruling_local_host")
    message = str(excinfo.value)
    assert SYNTHETIC_PASSWORD not in message and str(root_cert) not in message
    if host.strip():
        assert host.strip() not in message  # names the variable, never its value
    assert create_pool_calls == []


def test_ruling_strict_explicit_local_settings_are_refused_too(monkeypatch, no_postgres_env, create_pool_calls,
                                                              root_cert) -> None:
    monkeypatch.setenv("ENVIRONMENT", "staging")
    settings = PostgresSettings(host="127.0.0.1", password=SYNTHETIC_PASSWORD, ssl_root_cert=str(root_cert))
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_HOST"):
        _get_pool("ruling_explicit_local", settings)
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_HOST"):
        settings.get_dsn()
    assert create_pool_calls == []


def test_ruling_strict_client_connect_refuses_a_local_host_before_tls(monkeypatch, no_postgres_env,
                                                                     create_pool_calls) -> None:
    import asyncio

    monkeypatch.setenv("ENVIRONMENT", "production")
    client = pg.PostgresClient(PostgresSettings(password=SYNTHETIC_PASSWORD))  # unset host, no root cert either
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_HOST"):
        asyncio.run(client.connect(max_retries=1))
    assert create_pool_calls == []


@pytest.mark.parametrize("host", REMOTE_HOSTS)
def test_ruling_strict_plain_remote_hosts_are_accepted(monkeypatch, no_postgres_env, host) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert PostgresSettings(host=host, password=SYNTHETIC_PASSWORD).validate_for_environment() is None


def test_ruling_strict_remote_host_is_accepted(monkeypatch, no_postgres_env, create_pool_calls, root_cert) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    settings = PostgresSettings(host="postgres", password=SYNTHETIC_PASSWORD, ssl_root_cert=str(root_cert))
    assert settings.validate_for_environment() is None
    _get_pool("ruling_remote_host", settings)
    assert "@postgres:5432/" in create_pool_calls[0]["dsn"]
    assert create_pool_calls[0]["ssl"].verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize("env", ["development", "test"])
def test_ruling_development_unset_host_still_uses_localhost(monkeypatch, no_postgres_env, create_pool_calls,
                                                            env) -> None:
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("POSTGRES_PASSWORD", SYNTHETIC_PASSWORD)
    _get_pool("ruling_dev_unset_host")
    assert "@localhost:5432/" in create_pool_calls[0]["dsn"]
    assert PostgresSettings(password=SYNTHETIC_PASSWORD).validate_for_environment() is None


def test_ruling_explicit_environment_argument_wins(monkeypatch, no_postgres_env) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    settings = PostgresSettings(password=SYNTHETIC_PASSWORD)
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_HOST"):
        settings.validate_for_environment("production")
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert settings.validate_for_environment("development") is None
