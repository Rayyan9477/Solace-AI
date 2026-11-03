# tests/solace_infrastructure/test_mvp_033_default_pool.py
"""MVP-033/273/058: one default pool, strict environments refuse localhost and missing passwords, DSNs are encoded."""
from __future__ import annotations

import os
from urllib.parse import unquote, urlsplit

import asyncpg
import pytest

from solace_common.request_context import reset_current_user_id, set_current_user_id
from solace_infrastructure.database.connection_manager import ConnectionPoolManager as CPM
from solace_infrastructure.postgres import PostgresSettings


class _FakePool:
    async def close(self) -> None: ...


@pytest.fixture(autouse=True)
def isolated_manager(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # never read the repository's .env
    for key in [k for k in os.environ if k.startswith("POSTGRES_")]:
        monkeypatch.delenv(key)
    saved = (dict(CPM._pools), dict(CPM._pool_configs), dict(CPM._locks), dict(CPM._metrics))
    for d in (CPM._pools, CPM._pool_configs, CPM._locks, CPM._metrics):
        d.clear()
    yield
    for d, old in zip((CPM._pools, CPM._pool_configs, CPM._locks, CPM._metrics), saved):
        d.clear()
        d.update(old)


@pytest.fixture
def create_pool_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    async def _fake(**kwargs):
        calls.append(kwargs)
        return _FakePool()

    monkeypatch.setattr(asyncpg, "create_pool", _fake)
    return calls


async def test_mvp_033_legacy_pool_names_share_the_default_pool(monkeypatch, create_pool_calls) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-pw-1")
    default = await CPM.get_pool("default")
    for name in ("user_db", "diagnosis_db", "therapy_db", "personality_db", "safety_db"):
        assert await CPM.get_pool(name) is default, name
    assert len(create_pool_calls) == 1
    from solace_infrastructure.database.connection_manager import canonical_pool_name

    assert canonical_pool_name("user_db") == "default" and canonical_pool_name("contraindication_db") == "contraindication_db"


@pytest.mark.parametrize("env", ["production", "staging"])
async def test_mvp_033_strict_refuses_a_localhost_database(env, monkeypatch, create_pool_calls) -> None:
    from solace_infrastructure.postgres import DatabaseConfigError

    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-pw-2")
    with pytest.raises(DatabaseConfigError) as exc:
        await CPM.get_pool("user_db")
    assert "POSTGRES_HOST" in str(exc.value) and "synthetic-pw-2" not in str(exc.value)
    assert create_pool_calls == []


@pytest.mark.parametrize("env", ["production", "staging"])
async def test_mvp_033_strict_refuses_a_missing_password(env, monkeypatch, create_pool_calls) -> None:
    from solace_infrastructure.postgres import DatabaseConfigError

    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    with pytest.raises(DatabaseConfigError, match="POSTGRES_PASSWORD"):
        await CPM.get_pool("default")
    assert create_pool_calls == []


async def test_mvp_033_development_still_allows_localhost(monkeypatch, create_pool_calls) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-pw-3")
    await CPM.get_pool("default")
    assert "@localhost:5432/solace" in create_pool_calls[0]["dsn"]


def test_mvp_273_dsn_percent_encodes_credentials() -> None:
    s = PostgresSettings(user="solace_app", password="p@ss/w#rd:%x", host="db.internal", database="solace")
    parts = urlsplit(s.get_dsn())
    assert unquote(parts.password) == "p@ss/w#rd:%x"
    assert (parts.hostname, parts.port, parts.path) == ("db.internal", 5432, "/solace")


async def test_mvp_273_register_conflict_message_carries_no_secret() -> None:
    await CPM.register_pool("conflict_probe", PostgresSettings(password="first-SECRET-1", host="db.internal"))
    with pytest.raises(ValueError) as exc:
        await CPM.register_pool("conflict_probe", PostgresSettings(password="second-SECRET-2", host="db.internal"))
    assert "first-SECRET-1" not in str(exc.value) and "second-SECRET-2" not in str(exc.value)


async def test_mvp_033_init_connection_is_the_public_pool_codec() -> None:
    from solace_infrastructure.database.connection_manager import init_connection

    seen: list[tuple] = []

    class _CodecConn:
        async def set_type_codec(self, typename, *, encoder, decoder, schema, **kwargs):
            seen.append((typename, schema, encoder({"a": 1}), decoder('{"a": 1}')))

    await init_connection(_CodecConn())
    assert [(t, s) for t, s, _, _ in seen] == [("json", "pg_catalog"), ("jsonb", "pg_catalog")]
    assert all(enc == '{"a": 1}' and dec == {"a": 1} for _, _, enc, dec in seen)


class _Conn:
    def __init__(self) -> None:
        self.terminated = False

    async def execute(self, sql, *args):
        if "''" in sql:
            raise OSError("reset failed")
        return "OK"

    def terminate(self) -> None:
        self.terminated = True


class _AcquireCM:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _GucPool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self, timeout=None):
        return _AcquireCM(self._conn)

    def get_size(self):
        return 1

    def get_idle_size(self):
        return 1


async def test_mvp_058_failed_guc_reset_terminates_the_connection(monkeypatch) -> None:
    conn = _Conn()

    async def _get_pool(name="default"):
        return _GucPool(conn)

    monkeypatch.setattr(CPM, "get_pool", _get_pool)
    token = set_current_user_id("00000000-0000-0000-0000-000000000001")
    try:
        async with CPM.acquire("user_db"):
            pass
    finally:
        reset_current_user_id(token)
    assert conn.terminated is True


# --- WS-02.6 additions -------------------------------------------------------------------------------------------
import asyncio  # noqa: E402

from solace_common.exceptions import DatabaseError  # noqa: E402
from solace_infrastructure import postgres as pg  # noqa: E402
from solace_infrastructure.database import connection_manager as cm  # noqa: E402

UID = "00000000-0000-0000-0000-000000000002"
DISTINCT_USER = "svc-login-q7"
DISTINCT_PASSWORD = "synthetic-pw-ws026"
# verify_app_role's row for a plain application role that logged in as itself (fix round 1: the session role,
# the effective role, and whether the role was switched at login are part of the row).
APP_ROLE_ROW = {"login_role": DISTINCT_USER, "effective_role": DISTINCT_USER, "roles_found": 1,
                "role_switched": False, "is_superuser": False, "bypasses_rls": False, "owns_public_tables": False,
                "member_of_privileged_role": False, "member_of_superuser_equivalent_role": False,
                "can_replicate": False}
PRIVILEGE_REASONS = {"is_superuser": "is a superuser", "bypasses_rls": "has BYPASSRLS",
                     "owns_public_tables": "owns tables in schema public",
                     "member_of_privileged_role": "is a member of a superuser or BYPASSRLS role",
                     "role_switched": "switched to another role at login",
                     # Fix round 2 (WS026F1-CLO-1): COPY ... PROGRAM and server-file access bypass RLS.
                     "member_of_superuser_equivalent_role": "is a member of a superuser-equivalent role",
                     "can_replicate": "has REPLICATION"}
EVERY_PRIVILEGE_ROW = {**APP_ROLE_ROW, **{key: True for key in PRIVILEGE_REASONS}}


class _RoleConn:
    def __init__(self, pool: _RolePool) -> None:
        self._pool = pool

    async def fetchrow(self, sql, *args, **kwargs):
        self._pool.role_queries.append(sql)
        self._pool.role_query_args.append(args)
        return self._pool.row


class _RoleAcquire:
    def __init__(self, pool: _RolePool) -> None:
        self._pool = pool

    async def __aenter__(self):
        if self._pool.acquire_error is not None:
            raise self._pool.acquire_error
        return _RoleConn(self._pool)

    async def __aexit__(self, *exc):
        return False


class _RolePool:
    """create_pool's stand-in: one connection answering verify_app_role's query with ``row``."""

    def __init__(self, row: dict | None = None, acquire_error: BaseException | None = None) -> None:
        self.row = dict(APP_ROLE_ROW) if row is None else row
        self.acquire_error = acquire_error
        self.role_queries: list[str] = []
        self.role_query_args: list[tuple] = []
        self.closed = False

    def acquire(self, timeout=None):
        return _RoleAcquire(self)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def root_cert(tmp_path):
    from scripts.db import gen_certs

    out = tmp_path / "tls"
    assert gen_certs.main(["--out", str(out)]) == 0
    return out / "postgres-ca.crt"


@pytest.fixture
def strict_env(monkeypatch, root_cert) -> None:
    """Production with every setting a strict pool needs (remote host, password, root cert, a distinct user)."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_USER", DISTINCT_USER)
    monkeypatch.setenv("POSTGRES_PASSWORD", DISTINCT_PASSWORD)
    monkeypatch.setenv("POSTGRES_SSL_ROOT_CERT", str(root_cert))


class _PoolFactory:
    """create_pool's stand-in: hands out the queued _RolePool objects (a plain app role when none is queued)."""

    def __init__(self) -> None:
        self.queued: list[_RolePool] = []
        self.made: list[_RolePool] = []

    def queue(self, pool: _RolePool) -> _RolePool:
        self.queued.append(pool)
        return pool

    async def create_pool(self, **kwargs) -> _RolePool:
        pool = self.queued.pop(0) if self.queued else _RolePool()
        self.made.append(pool)
        return pool


@pytest.fixture
def role_pools(monkeypatch) -> _PoolFactory:
    factory = _PoolFactory()
    monkeypatch.setattr(asyncpg, "create_pool", factory.create_pool)
    return factory


@pytest.mark.parametrize("reason", sorted(PRIVILEGE_REASONS))
async def test_ws026_strict_pool_refuses_a_privileged_role_and_closes_it(strict_env, role_pools, reason) -> None:
    refused = role_pools.queue(_RolePool(row={**APP_ROLE_ROW, reason: True}))
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_USER") as exc:
        await CPM.get_pool("therapy_db")
    message = str(exc.value)
    assert PRIVILEGE_REASONS[reason] in message
    assert DISTINCT_PASSWORD not in message  # the problem and the role names only (fix round 1)
    assert f"session_user '{DISTINCT_USER}'" in message and f"current_user '{DISTINCT_USER}'" in message
    assert refused.closed is True and len(refused.role_queries) == 1
    assert CPM._pools == {} and role_pools.made == [refused]


async def test_ws026_strict_pool_with_every_privilege_names_every_reason(strict_env, role_pools) -> None:
    role_pools.queue(_RolePool(row=dict(EVERY_PRIVILEGE_ROW)))
    with pytest.raises(pg.DatabaseConfigError) as exc:
        await CPM.get_pool("default")
    assert all(text in str(exc.value) for text in PRIVILEGE_REASONS.values())


async def test_ws026_strict_pool_refuses_a_superuser_login_switched_to_the_app_role(strict_env, role_pools) -> None:
    """WS026-SEC-1: ALTER ROLE <superuser> SET role = solace_app makes current_user the app role at login, but the
    session stays the superuser's (SET ROLE NONE gets it back); both role names are in the message."""
    switched = {**APP_ROLE_ROW, "login_role": "postgres", "effective_role": "solace_app", "roles_found": 2,
                "role_switched": True, "is_superuser": True}
    refused = role_pools.queue(_RolePool(row=switched))
    with pytest.raises(pg.DatabaseConfigError) as exc:
        await CPM.get_pool("default")
    message = str(exc.value)
    assert "switched to another role at login" in message and "is a superuser" in message
    assert "session_user 'postgres'" in message and "current_user 'solace_app'" in message
    assert refused.closed is True and CPM._pools == {}


async def test_ws026_strict_pool_refuses_an_unreadable_role_row(strict_env, role_pools) -> None:
    refused = _RolePool()
    refused.row = None  # current_user missing from pg_roles: fail closed
    role_pools.queue(refused)
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_USER"):
        await CPM.get_pool("default")
    assert refused.closed is True and CPM._pools == {}


async def test_ws026_strict_pool_refuses_when_no_login_role_is_found(strict_env, role_pools) -> None:
    refused = role_pools.queue(_RolePool(row={**APP_ROLE_ROW, "roles_found": 0}))
    with pytest.raises(pg.DatabaseConfigError, match="could not be read from pg_roles"):
        await CPM.get_pool("default")
    assert refused.closed is True and CPM._pools == {}


def test_ws026_role_query_checks_the_session_role_and_counts_noinherit_membership() -> None:
    """WS026-SEC-1 (shape; real PG proves the behaviour): the login (session) role is evaluated as well as the
    current one, and table ownership counts any membership ('MEMBER'), not only inherited privileges ('USAGE')."""
    sql = " ".join(cm._APP_ROLE_SQL.split())
    assert "session_user" in sql and "'USAGE'" not in sql
    assert "c.relowner, 'MEMBER')" in sql


def test_ws026_superuser_equivalent_roles_match_the_migrate_job() -> None:
    """WS026F1-CLO-1: one list of superuser-equivalent predefined roles; scripts/db/migrate.py refuses the same
    ones for the platform roles."""
    from scripts.db.migrate import SUPERUSER_EQUIVALENT_ROLES as MIGRATE_ROLES

    assert cm.SUPERUSER_EQUIVALENT_ROLES == MIGRATE_ROLES
    assert set(cm.SUPERUSER_EQUIVALENT_ROLES) == {"pg_execute_server_program", "pg_read_server_files",
                                                  "pg_write_server_files"}


async def test_ws026_the_role_query_receives_the_superuser_equivalent_roles(strict_env, role_pools) -> None:
    pool = await CPM.get_pool("default")
    assert pool.role_query_args == [(list(cm.SUPERUSER_EQUIVALENT_ROLES),)]
    sql = " ".join(cm._APP_ROLE_SQL.split())
    assert "= ANY($1::text[])" in sql and "rolreplication" in sql


def test_ws026_database_config_error_documents_every_way_it_is_raised() -> None:
    """WS026-COR-1: the class contract covers the settings errors (before any connection) and the strict role
    refusal (after one connection, with the pool it opened closed)."""
    doc = " ".join((pg.DatabaseConfigError.__doc__ or "").split())
    for name in ("POSTGRES_HOST", "POSTGRES_PASSWORD", "POSTGRES_USER", "POSTGRES_SSL_"):
        assert name in doc, name
    assert "verify_app_role" in doc and "after one connection" in doc and "closed" in doc
    connect_doc = " ".join((pg.PostgresClient.connect.__doc__ or "").split())
    assert "member of a superuser or BYPASSRLS role" in connect_doc and "switched" in connect_doc
    assert "superuser-equivalent" in connect_doc


async def test_ws026_strict_pool_accepts_the_app_role_once(strict_env, role_pools) -> None:
    default = await CPM.get_pool("default")
    assert await CPM.get_pool("safety_db") is default
    assert role_pools.made == [default] and len(default.role_queries) == 1 and default.closed is False


async def test_ws026_strict_role_check_without_a_connection_closes_the_pool(strict_env, role_pools) -> None:
    broken = role_pools.queue(_RolePool(acquire_error=ConnectionRefusedError("synthetic refusal")))
    with pytest.raises(DatabaseError) as exc:
        await CPM.get_pool("default")
    assert not isinstance(exc.value, pg.DatabaseConfigError)
    assert broken.closed is True and CPM._pools == {}


async def test_ws026_strict_role_check_cancelled_closes_the_pool(strict_env, role_pools) -> None:
    cancelled = role_pools.queue(_RolePool(acquire_error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await CPM.get_pool("default")
    assert cancelled.closed is True and CPM._pools == {}


async def test_ws026_development_pool_skips_the_role_check(monkeypatch, role_pools) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("POSTGRES_PASSWORD", DISTINCT_PASSWORD)
    role_pools.queue(_RolePool(row=dict(EVERY_PRIVILEGE_ROW)))  # a superuser: allowed in development
    pool = await CPM.get_pool("default")
    assert pool.role_queries == [] and pool.closed is False


async def test_ws026_postgres_client_strict_refuses_a_privileged_role_without_retry(strict_env, role_pools) -> None:
    refused = role_pools.queue(_RolePool(row={**APP_ROLE_ROW, "is_superuser": True}))
    client = pg.PostgresClient(PostgresSettings())
    with pytest.raises(pg.DatabaseConfigError, match="is a superuser"):
        await client.connect(max_retries=3, base_delay=0)
    assert role_pools.made == [refused] and refused.closed is True  # one attempt, and its pool is closed
    assert client.is_connected is False and client._pool is None


async def test_ws026_postgres_client_strict_accepts_the_app_role(strict_env, role_pools) -> None:
    client = pg.PostgresClient(PostgresSettings())
    await client.connect(max_retries=1)
    assert client.is_connected is True and len(role_pools.made[0].role_queries) == 1


async def test_ws026_postgres_client_development_skips_the_role_check(monkeypatch, role_pools) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    role_pools.queue(_RolePool(row={**APP_ROLE_ROW, "bypasses_rls": True}))
    client = pg.PostgresClient(PostgresSettings(password=DISTINCT_PASSWORD))
    await client.connect(max_retries=1)
    assert client.is_connected is True and role_pools.made[0].role_queries == []


async def test_ws026_verify_app_role_reads_the_environment(monkeypatch) -> None:
    pool = _RolePool(row={**APP_ROLE_ROW, "is_superuser": True})
    conn = _RoleConn(pool)
    monkeypatch.setenv("ENVIRONMENT", "test")
    await cm.verify_app_role(conn)  # not strict: no query at all
    assert pool.role_queries == []
    monkeypatch.setenv("ENVIRONMENT", "staging")
    with pytest.raises(pg.DatabaseConfigError):
        await cm.verify_app_role(conn)
    await cm.verify_app_role(conn, environment="development")  # the argument wins
    assert len(pool.role_queries) == 1


@pytest.mark.parametrize("env", ["development", "production"])
async def test_ws026_missing_password_is_a_config_error_naming_only_the_variable(env, monkeypatch,
                                                                                 create_pool_calls) -> None:
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("POSTGRES_HOST", "distinct-host-value.internal")
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_PASSWORD") as exc:
        await CPM.get_pool("default")
    assert "distinct-host-value" not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__ is True  # no pydantic text chained
    assert create_pool_calls == [] and CPM._pool_configs == {}


async def test_ws026_an_invalid_postgres_value_names_the_variable_not_the_value(monkeypatch,
                                                                                create_pool_calls) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("POSTGRES_PASSWORD", DISTINCT_PASSWORD)
    monkeypatch.setenv("POSTGRES_PORT", "not-a-port-x1")
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_PORT") as exc:
        await CPM.get_pool("default")
    assert "not-a-port-x1" not in str(exc.value) and DISTINCT_PASSWORD not in str(exc.value)
    assert create_pool_calls == []


@pytest.mark.parametrize("password", ["", "   "])
async def test_ws026_strict_empty_password_is_refused_before_any_connection(password, monkeypatch,
                                                                            create_pool_calls) -> None:
    monkeypatch.setenv("ENVIRONMENT", "staging")
    settings = PostgresSettings(password=password, host="db.internal")
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_PASSWORD"):
        settings.validate_for_environment()
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_PASSWORD"):
        settings.get_dsn()
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_PASSWORD"):
        await CPM.register_pool("default", settings)
    with pytest.raises(pg.DatabaseConfigError, match="POSTGRES_PASSWORD"):
        await pg.PostgresClient(settings).connect(max_retries=1)
    assert create_pool_calls == [] and CPM._pool_configs == {}
    assert settings.validate_for_environment("development") is None


def test_ws026_strict_host_and_password_problems_are_named_together() -> None:
    with pytest.raises(pg.DatabaseConfigError) as exc:
        PostgresSettings(password="").validate_for_environment("production")
    assert "POSTGRES_HOST" in str(exc.value) and "POSTGRES_PASSWORD" in str(exc.value)


def test_ws026_dsn_encodes_user_and_database_too() -> None:
    s = PostgresSettings(user="app@x/y", password="a b+c?d=e&f", host="db.internal", port=6432, database="d#b/1")
    parts = urlsplit(s.get_dsn())
    assert (unquote(parts.username), unquote(parts.password)) == ("app@x/y", "a b+c?d=e&f")
    assert (parts.hostname, parts.port, unquote(parts.path)) == ("db.internal", 6432, "/d#b/1")
    assert parts.query == "" and parts.fragment == ""
    plain = PostgresSettings(user="solace_app", password="token_urlsafe-Ab9", host="db.internal").get_dsn()
    assert plain == "postgresql://solace_app:token_urlsafe-Ab9@db.internal:5432/solace"


async def test_ws026_conflict_message_names_the_pool_host_and_database_only() -> None:
    first = PostgresSettings(password="first-SECRET-1", host="db.internal", database="solace")
    await CPM.register_pool("conflict_probe", first)
    await CPM.register_pool("conflict_probe", PostgresSettings(password="first-SECRET-1", host="db.internal"))
    with pytest.raises(ValueError) as exc:
        await CPM.register_pool("conflict_probe", PostgresSettings(password="first-SECRET-1", host="db2.internal"))
    message = str(exc.value)
    assert "'conflict_probe'" in message and "db.internal" in message and "db2.internal" in message
    assert "solace" in message and "postgresql://" not in message and "first-SECRET-1" not in message
    with pytest.raises(ValueError):
        await CPM.register_pool("conflict_probe", PostgresSettings(password="first-SECRET-1", host="db.internal",
                                                                   ssl_root_cert="other-ca.crt"))


async def test_ws026_a_legacy_alias_registers_and_conflicts_with_the_default_pool() -> None:
    await CPM.register_pool("user_db", PostgresSettings(password="alias-SECRET-1", host="db.internal"))
    assert CPM.get_registered_pools() == ["default"]
    with pytest.raises(ValueError, match="'default'"):
        await CPM.register_pool("therapy_db", PostgresSettings(password="alias-SECRET-2", host="db.internal"))


class _StatsPool(_FakePool):
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def get_size(self):
        return 1

    def get_idle_size(self):
        return 1

    def get_min_size(self):
        return 1

    def get_max_size(self):
        return 2


async def test_ws026_legacy_names_reach_the_default_pool_in_every_method(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("POSTGRES_PASSWORD", DISTINCT_PASSWORD)
    pool = _StatsPool()

    async def _fake(**kwargs):
        return pool

    monkeypatch.setattr(asyncpg, "create_pool", _fake)
    assert await CPM.get_pool("personality_db") is pool
    assert CPM.get_active_pools() == ["default"]
    assert (await CPM.get_pool_stats("diagnosis_db"))["pool_name"] == "default"
    await CPM.close_pool("therapy_db")
    assert pool.closed is True and CPM.get_active_pools() == []


async def test_ws026_acquire_and_transaction_under_an_alias_use_the_default_pool(monkeypatch) -> None:
    used: list[str] = []

    class _TxConn(_Conn):
        def transaction(self, isolation=None):
            return _AcquireCM(None)

    conn = _TxConn()

    async def _get_pool(name="default"):
        used.append(name)
        return _GucPool(conn)

    monkeypatch.setattr(CPM, "get_pool", _get_pool)
    async with CPM.acquire("user_db"):
        pass
    async with CPM.transaction("safety_db"):
        pass
    assert used == ["default", "default"]
    assert CPM.get_metrics("user_db") is CPM.get_metrics("default") and CPM.get_metrics("default").acquisition_count == 2


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def record(event: str, **kw) -> None:
            self.events.append((level, event, kw))

        return record


class _ResetConn:
    """execute() raises ``reset_error`` on the reset statement (and ``set_error`` on the set statement)."""

    def __init__(self, reset_error: BaseException | None = None, set_error: BaseException | None = None) -> None:
        self.reset_error, self.set_error = reset_error, set_error
        self.executed: list[str] = []
        self.terminated = False

    async def execute(self, sql, *args):
        self.executed.append(sql)
        if "''" in sql and self.reset_error is not None:
            raise self.reset_error
        if "$1" in sql and self.set_error is not None:
            raise self.set_error
        return "OK"

    def terminate(self) -> None:
        self.terminated = True


def _patch_get_pool(monkeypatch, conn) -> None:
    async def _get_pool(name="default"):
        return _GucPool(conn)

    monkeypatch.setattr(CPM, "get_pool", _get_pool)


async def test_ws026_failed_guc_reset_logs_the_class_only_and_never_fails_the_request(monkeypatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(pg, "logger", recorder)
    conn = _ResetConn(reset_error=OSError("reset failed for SECRET-ish detail"))
    _patch_get_pool(monkeypatch, conn)
    token = set_current_user_id(UID)
    try:
        async with CPM.acquire():
            pass
    finally:
        reset_current_user_id(token)
    assert conn.terminated is True
    assert recorder.events == [("warning", "rls_guc_reset_failed",
                                {"error": "OSError", "action": "connection_terminated"})]


async def test_ws026_successful_guc_reset_keeps_the_connection(monkeypatch) -> None:
    conn = _ResetConn()
    _patch_get_pool(monkeypatch, conn)
    token = set_current_user_id(UID)
    try:
        async with CPM.acquire():
            pass
    finally:
        reset_current_user_id(token)
    assert conn.terminated is False and conn.executed[-1] == "SELECT set_config('app.current_user_id', '', false)"


async def test_ws026_cancelled_guc_reset_terminates_and_propagates(monkeypatch) -> None:
    conn = _ResetConn(reset_error=asyncio.CancelledError())
    _patch_get_pool(monkeypatch, conn)
    token = set_current_user_id(UID)
    try:
        with pytest.raises(asyncio.CancelledError):
            async with CPM.acquire():
                pass
    finally:
        reset_current_user_id(token)
    assert conn.terminated is True
    assert id(conn) not in CPM._active_connections and CPM.get_metrics("default").active_connections == 0


async def test_ws026_failed_guc_set_still_resets_and_untracks_the_connection(monkeypatch) -> None:
    conn = _ResetConn(set_error=OSError("set failed"))
    _patch_get_pool(monkeypatch, conn)
    token = set_current_user_id(UID)
    try:
        with pytest.raises(OSError, match="set failed"):
            async with CPM.acquire():
                pytest.fail("the body must not run when the GUC could not be set")
    finally:
        reset_current_user_id(token)
    assert conn.executed[-1] == "SELECT set_config('app.current_user_id', '', false)"
    assert id(conn) not in CPM._active_connections and CPM.get_metrics("default").active_connections == 0


async def test_ws026_postgres_client_failed_guc_reset_terminates_the_connection(monkeypatch) -> None:
    conn = _ResetConn(reset_error=OSError("reset failed"))
    client = pg.PostgresClient(PostgresSettings(password=DISTINCT_PASSWORD))
    client._pool, client._connected = _GucPool(conn), True
    token = set_current_user_id(UID)
    try:
        async with client.acquire():
            pass
    finally:
        reset_current_user_id(token)
    assert conn.terminated is True


async def test_ws026_init_connection_backs_the_pool_hook() -> None:
    seen: list[str] = []

    class _CodecConn:
        async def set_type_codec(self, typename, *, encoder, decoder, schema, **kwargs):
            seen.append(typename)

    await CPM._init_connection(_CodecConn())
    assert seen == ["json", "jsonb"]
    assert cm.init_connection.__name__ == "init_connection" and "init_connection" in cm.__all__
