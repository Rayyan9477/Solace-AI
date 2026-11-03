"""MVP-001/009 (IF-03): the Alembic chain applies, round-trips and leaves one schema on an empty PostgreSQL 16.

Every slot task in every workstream keeps this file green."""
from __future__ import annotations

import ast
import configparser
from pathlib import Path

import asyncpg
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[2]


def chain_head() -> str:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    assert len(heads) == 1, heads
    return heads[0]


async def _fetch(dsn: str, sql: str, *args):
    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


async def _tables(dsn: str) -> set[str]:
    return {r["tablename"] for r in await _fetch(dsn, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}


@pytest.mark.realinfra
async def test_mvp_001_upgrade_head_on_empty_database(pg_server, fresh_database, run_migration) -> None:
    r = run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head")
    assert r.returncode == 0, r.stderr[-3000:]
    su = pg_server.superuser_dsn(fresh_database)
    assert {"users", "user_preferences", "memory_records", "diagnosis_sessions", "therapy_sessions",
            "escalations", "oauth_accounts", "personality_profiles", "alembic_version"} <= await _tables(su)
    cols = await _fetch(su, "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'memory_records' AND column_name = 'access_count'")
    assert len(cols) == 1
    assert (await _fetch(su, "SELECT version_num FROM alembic_version"))[0]["version_num"] == chain_head()


@pytest.mark.realinfra
async def test_mvp_009_upgrade_through_007_needs_no_runtime_outbox(pg_server, fresh_database, run_migration) -> None:
    r = run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "007_outbox_claimed_at")
    assert r.returncode == 0, r.stderr[-3000:]
    rows = await _fetch(pg_server.superuser_dsn(fresh_database), "SELECT to_regclass('public.event_outbox') AS t")
    assert rows[0]["t"] is None


@pytest.mark.realinfra
async def test_mvp_001_downgrade_base_then_upgrade_head_round_trips(pg_server, fresh_database, run_migration) -> None:
    url, su = pg_server.migrator_dsn(fresh_database), pg_server.superuser_dsn(fresh_database)
    assert run_migration(url, "upgrade", "head").returncode == 0
    r = run_migration(url, "downgrade", "base")
    assert r.returncode == 0, r.stderr[-3000:]
    assert await _tables(su) == {"alembic_version"}
    r = run_migration(url, "upgrade", "head")
    assert r.returncode == 0, r.stderr[-3000:]
    assert (await _fetch(su, "SELECT version_num FROM alembic_version"))[0]["version_num"] == chain_head()


def test_mvp_009_env_runs_each_revision_in_its_own_transaction() -> None:
    tree = ast.parse((ROOT / "migrations" / "env.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "configure"]
    assert calls
    for call in calls:
        flags = {k.arg: getattr(k.value, "value", None) for k in call.keywords}
        assert flags.get("transaction_per_migration") is True


def test_mvp_001_alembic_ini_holds_no_credentials() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "alembic.ini", encoding="utf-8")
    assert "@" not in cfg["alembic"].get("sqlalchemy.url", "")


# MVP-041/058 (WS-02.3): app-role grants, NULLIF policies, per-user isolation and the schema-head check.
from solace_infrastructure.database.schema_check import SCHEMA_HEAD, assert_schema_head  # noqa: E402

NULLIF_FRAGMENT = "nullif(current_setting('app.current_user_id'"


@pytest.mark.realinfra
async def test_mvp_041_app_role_has_dml_but_no_ddl(pg_server, fresh_database, run_migration) -> None:
    assert run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head").returncode == 0
    app = await asyncpg.connect(pg_server.app_dsn(fresh_database))
    try:
        assert await app.fetchval("SELECT version_num FROM alembic_version") == SCHEMA_HEAD
        assert await app.fetchval("SELECT has_table_privilege('solace_app', 'users', 'SELECT, INSERT, UPDATE, DELETE')")
        for ddl in ("CREATE TABLE mvp041_probe (id int)", "ALTER TABLE users ADD COLUMN mvp041 int",
                    "DELETE FROM alembic_version"):
            with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
                await app.execute(ddl)
    finally:
        await app.close()
    owners = {r["tableowner"] for r in await _fetch(pg_server.superuser_dsn(fresh_database),
                                                    "SELECT tableowner FROM pg_tables WHERE schemaname = 'public'")}
    assert owners == {"solace_migrator"}


@pytest.mark.realinfra
async def test_mvp_041_audit_events_is_append_only_when_present(pg_server, fresh_database, run_migration) -> None:
    assert run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head").returncode == 0
    su = pg_server.superuser_dsn(fresh_database)
    if (await _fetch(su, "SELECT to_regclass('public.audit_events') AS t"))[0]["t"] is None:
        return  # vacuous until WS-09 fills slot 011
    rows = await _fetch(su, "SELECT has_table_privilege('solace_app', 'audit_events', p) AS ok, p FROM "
                            "unnest(ARRAY['INSERT','SELECT','UPDATE','DELETE','TRUNCATE']) AS p")
    assert {r["p"]: r["ok"] for r in rows} == {"INSERT": True, "SELECT": True, "UPDATE": False, "DELETE": False, "TRUNCATE": False}


@pytest.mark.realinfra
async def test_mvp_058_every_policy_uses_nullif(pg_server, fresh_database, run_migration) -> None:
    assert run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head").returncode == 0
    rows = await _fetch(pg_server.superuser_dsn(fresh_database),
                        "SELECT tablename, qual, with_check FROM pg_policies WHERE schemaname = 'public'")
    assert len(rows) >= 15
    bad = sorted(r["tablename"] for r in rows if NULLIF_FRAGMENT not in (r["qual"] or "").lower()
                 or (r["with_check"] is not None and NULLIF_FRAGMENT not in r["with_check"].lower()))
    assert bad == []


async def _grant_select_for_probe(pg_server, database: str, *tables: str) -> None:
    conn = await asyncpg.connect(pg_server.superuser_dsn(database))
    try:
        for t in tables:  # explicit grants keep the RED reason on the policy, not on missing privileges
            await conn.execute(f"GRANT SELECT, INSERT ON {t} TO solace_app")
    finally:
        await conn.close()


@pytest.mark.realinfra
async def test_mvp_058_reset_guc_returns_no_rows_instead_of_error(pg_server, fresh_database, run_migration) -> None:
    assert run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head").returncode == 0
    await _grant_select_for_probe(pg_server, fresh_database, "diagnosis_sessions")
    app = await asyncpg.connect(pg_server.app_dsn(fresh_database))
    try:
        assert await app.fetchval("SELECT count(*) FROM diagnosis_sessions") == 0  # GUC never set
        await app.execute("SELECT set_config('app.current_user_id', '', false)")
        assert await app.fetchval("SELECT count(*) FROM diagnosis_sessions") == 0  # GUC reset to ''
    finally:
        await app.close()


@pytest.mark.realinfra
async def test_mvp_058_rls_isolates_users_for_the_app_role(pg_server, fresh_database, run_migration, insert_minimal) -> None:
    import uuid

    assert run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head").returncode == 0
    await _grant_select_for_probe(pg_server, fresh_database, "users", "user_preferences")
    a, b = uuid.uuid4(), uuid.uuid4()
    app = await asyncpg.connect(pg_server.app_dsn(fresh_database))
    try:
        await insert_minimal(app, "users", id=a, email=f"{a}@test.invalid")
        await insert_minimal(app, "users", id=b, email=f"{b}@test.invalid")
        await app.execute("SELECT set_config('app.current_user_id', $1, false)", str(a))
        await insert_minimal(app, "user_preferences", user_id=a)
        assert await app.fetchval("SELECT count(*) FROM user_preferences") == 1
        await app.execute("SELECT set_config('app.current_user_id', $1, false)", str(b))
        assert await app.fetchval("SELECT count(*) FROM user_preferences") == 0
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await insert_minimal(app, "user_preferences", user_id=a)  # WITH CHECK: b cannot write a's row
        await app.execute("SELECT set_config('app.current_user_id', '', false)")
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await insert_minimal(app, "user_preferences", user_id=b)  # no user context, no write
    finally:
        await app.close()


@pytest.mark.realinfra
async def test_mvp_041_schema_head_check_passes_as_app_role(pg_server, fresh_database, run_migration) -> None:
    assert run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head").returncode == 0
    pool = await asyncpg.create_pool(pg_server.app_dsn(fresh_database), min_size=1, max_size=1)
    try:
        assert await assert_schema_head(pool, environment="production") == SCHEMA_HEAD
    finally:
        await pool.close()


# MVP-041 (WS-02.3 fix 1): user-scoped rows are actually unreachable across users, not only covered by policy text.
# The tables with a user_id column that are NOT under RLS at head, each with the task that covers it. The task that
# puts one under RLS removes it here; a new user_id table either enables and forces RLS with a NULLIF policy or is
# added here with a reason. This file is WS-02's, so both changes are WS-02 requests.
RLS_DEFERRED: frozenset[str] = frozenset({
    "escalations",              # non-RLS by design (IF-03); WS-12.4 keeps it so in slot 014, WS-12.12 pseudonymizes on erasure
    "notifications",            # non-RLS by design (IF-03); WS-13.3 keeps the slot 015 notification tables non-RLS
    "oauth_accounts",           # non-RLS by design (IF-03): written before authentication (WS-08.12)
    "session_summaries",        # WS-17.3 puts it under RLS in slot 018
    "therapeutic_events",       # WS-17.3 puts it under RLS in slot 018
    "memory_user_profiles",     # WS-17.3 puts it under RLS in slot 018
    "safety_assessments",       # WS-12.12 erases it under acting_as_user; no plan task enables RLS yet
    "safety_plans",             # WS-12.12 erases it under acting_as_user; no plan task enables RLS yet
    "safety_events",            # WS-12 (slot 014): no runtime writer; no plan task enables RLS yet
    "risk_factors",             # WS-12.13 removes the legacy safety engines; no runtime writer, no RLS task yet
    "contraindication_checks",  # WS-12.13 removes the unwired contraindication engine; no runtime writer
    "event_dead_letter",        # non-RLS by design (IF-03): slot 010 (WS-06.1); written by consumers and reapers outside any user context
})
# SECURITY DEFINER functions in public, by regprocedure signature, each with why it cannot leak rows. Empty today.
SECURITY_DEFINER_ALLOWLIST: frozenset[str] = frozenset()


_USER_TABLES_SQL = """
SELECT c.relname,
       c.relrowsecurity AND c.relforcerowsecurity
           AND EXISTS (SELECT 1 FROM pg_policy p WHERE p.polrelid = c.oid) AS protected
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
  AND EXISTS (SELECT 1 FROM pg_attribute a
              WHERE a.attrelid = c.oid AND a.attname = 'user_id' AND a.attnum > 0 AND NOT a.attisdropped)
"""
_POLICY_WITHOUT_RLS_SQL = """
SELECT DISTINCT c.relname FROM pg_policy p
JOIN pg_class c ON c.oid = p.polrelid JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND NOT c.relrowsecurity
"""
_VIEWS_WITHOUT_INVOKER_SQL = """
SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'v'
  AND NOT EXISTS (SELECT 1 FROM pg_options_to_table(c.reloptions) o
                  WHERE o.option_name = 'security_invoker' AND lower(o.option_value) IN ('true', 'on', 'yes', '1'))
"""
_MATVIEWS_SQL = """
SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'm'
"""
_SECURITY_DEFINER_SQL = """
SELECT p.oid::regprocedure::text AS signature FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public' AND p.prosecdef
"""


async def _rls_audit(dsn: str) -> dict[str, list[str]]:
    """Every public object through which one user's rows can reach another as solace_app; {} when sound.

    Policies bind only the tables they sit on, and only once RLS is enabled. A view runs as its owner and a
    SECURITY DEFINER function as its definer, and both are owned by solace_migrator, which has BYPASSRLS. A
    materialized view stores rows and cannot carry RLS at all.
    """
    conn = await asyncpg.connect(dsn)
    try:
        unprotected = {r["relname"] for r in await conn.fetch(_USER_TABLES_SQL) if not r["protected"]}
        findings = {
            "unprotected_user_tables_not_deferred": unprotected - RLS_DEFERRED,
            "deferred_tables_now_protected_or_gone": RLS_DEFERRED - unprotected,
            "policy_on_table_without_rls": {r["relname"] for r in await conn.fetch(_POLICY_WITHOUT_RLS_SQL)},
            "views_without_security_invoker": {r["relname"] for r in await conn.fetch(_VIEWS_WITHOUT_INVOKER_SQL)},
            "materialized_views": {r["relname"] for r in await conn.fetch(_MATVIEWS_SQL)},
            "security_definer_functions": {r["signature"] for r in await conn.fetch(_SECURITY_DEFINER_SQL)}
                                          - SECURITY_DEFINER_ALLOWLIST,
        }
    finally:
        await conn.close()
    return {name: sorted(found) for name, found in findings.items() if found}


@pytest.mark.realinfra
async def test_mvp_041_user_scoped_data_is_rls_protected_at_head(pg_server, fresh_database, run_migration) -> None:
    r = run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head")
    assert r.returncode == 0, r.stderr[-3000:]
    assert await _rls_audit(pg_server.superuser_dsn(fresh_database)) == {}


_PROBE_NULLIF = "user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid"


@pytest.mark.realinfra
async def test_mvp_041_rls_audit_names_every_cross_user_bypass(pg_server, fresh_database, run_migration,
                                                               insert_minimal) -> None:
    """Each probe object lets user B read user A's row as solace_app while the policy itself holds; the audit
    must name every one of them, and must not flag the security_invoker view, which does not leak."""
    import uuid

    assert run_migration(pg_server.migrator_dsn(fresh_database), "upgrade", "head").returncode == 0
    a, b = uuid.uuid4(), uuid.uuid4()
    su = await asyncpg.connect(pg_server.superuser_dsn(fresh_database))
    try:  # the superuser bypasses RLS, so it seeds user A's row directly
        for uid in (a, b):
            await insert_minimal(su, "users", id=uid, email=f"{uid}@test.invalid")
        await insert_minimal(su, "user_preferences", user_id=a)
    finally:
        await su.close()
    mig = await asyncpg.connect(pg_server.migrator_dsn(fresh_database))
    try:  # what a later slot could create as solace_migrator, the BYPASSRLS owner of everything
        for sql in ("CREATE TABLE rls_probe_later (user_id uuid NOT NULL)",
                    "CREATE TABLE rls_probe_inert (user_id uuid NOT NULL)",
                    f"CREATE POLICY rls_probe_inert_by_user ON rls_probe_inert USING ({_PROBE_NULLIF}) "
                    f"WITH CHECK ({_PROBE_NULLIF})",
                    "CREATE VIEW rls_probe_view AS SELECT user_id FROM user_preferences",
                    "CREATE VIEW rls_probe_invoker_view WITH (security_invoker = true) AS "
                    "SELECT user_id FROM user_preferences",
                    "CREATE MATERIALIZED VIEW rls_probe_matview AS SELECT user_id FROM user_preferences",
                    "CREATE FUNCTION rls_probe_definer() RETURNS bigint LANGUAGE sql SECURITY DEFINER "
                    "AS 'SELECT count(*) FROM user_preferences'"):
            await mig.execute(sql)
        await mig.execute("INSERT INTO rls_probe_later (user_id) VALUES ($1)", a)
        await mig.execute("INSERT INTO rls_probe_inert (user_id) VALUES ($1)", a)
    finally:
        await mig.close()
    app = await asyncpg.connect(pg_server.app_dsn(fresh_database))
    try:
        await app.execute("SELECT set_config('app.current_user_id', $1, false)", str(b))
        assert await app.fetchval("SELECT count(*) FROM user_preferences") == 0  # the policy itself holds
        assert await app.fetchval("SELECT count(*) FROM rls_probe_invoker_view") == 0
        for leak in ("SELECT count(*) FROM rls_probe_later", "SELECT count(*) FROM rls_probe_inert",
                     "SELECT count(*) FROM rls_probe_view", "SELECT count(*) FROM rls_probe_matview",
                     "SELECT rls_probe_definer()"):
            assert await app.fetchval(leak) == 1, leak  # user B reads user A's row
    finally:
        await app.close()
    assert await _rls_audit(pg_server.superuser_dsn(fresh_database)) == {
        "unprotected_user_tables_not_deferred": ["rls_probe_inert", "rls_probe_later"],
        "policy_on_table_without_rls": ["rls_probe_inert"],
        "views_without_security_invoker": ["rls_probe_view"],
        "materialized_views": ["rls_probe_matview"],
        "security_definer_functions": ["rls_probe_definer()"],
    }
