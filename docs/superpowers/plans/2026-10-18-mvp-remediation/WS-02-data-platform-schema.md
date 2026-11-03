# WS-02: Schema of record, database roles, TLS and connection layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Alembic is the single schema of record: 001 to head applies and round-trips on an empty PostgreSQL 16, is applied by `python -m scripts.db.migrate` as `solace_migrator` under an advisory lock, and services connect as NOSUPERUSER `solace_app` over verify-full TLS through one `default` pool per service, under RLS policies that tolerate a missing user context.

**Architecture:** `infrastructure/init-db/01-roles.sh` creates the two platform roles (passwords from the environment or `_FILE`) and hands the `solace` database to `solace_migrator`. Slot `009_platform_roles_rls` grants DML to `solace_app` through default privileges and rewrites all fifteen user policies with `NULLIF`, and fixed no-op slots 010 to 021 give each owning workstream one revision on a single linear chain that ends at `SCHEMA_HEAD = '021_personality_analytics'`. `solace_infrastructure.postgres` and `solace_infrastructure.database.connection_manager` enforce the strict-environment policy (verify-full with a root certificate, no localhost, no missing password, no superuser, BYPASSRLS or table-owner role), percent-encode DSNs, alias the legacy pool names to `default` and size pools for a stock `max_connections`; `tests/migrations` carries its own Docker fixtures so every slot change re-proves the chain on real Postgres.

**Covers:** 12 register entries (blocker 4, high 2, medium 2, low 4).

**Waves:** 0 (phase A2: WS-02.1 to WS-02.7, the first platform lane after WS-01's lock), 2 (phase A: WS-02.8, WS-02.9).

**Depends on:** WS-01, for the pinned alembic, SQLAlchemy, asyncpg, psycopg, cryptography and testcontainers versions installed per D-20 (WS-01.2), the lazy `solace_infrastructure` package (WS-01.3, needed by WS-02.7), the import-root and lint guards whose WS-02 cases WS-02.2 and WS-02.5 turn green (WS-01.7), and the migrate image that carries `migrations/`, `alembic.ini` and `scripts/db` (WS-01.9, which in turn waits for WS-02.4).

**Owned paths:** `migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/001_initial_schema.py` to `migrations/versions/008_rls_personality_tables.py`, `migrations/versions/009_platform_roles_rls.py`, `alembic.ini`, `infrastructure/init-db`, `src/solace_infrastructure/database`, `src/solace_infrastructure/postgres.py`, `src/solace_infrastructure/redis.py`, `src/solace_infrastructure/weaviate.py`, `src/solace_infrastructure/feature_flags.py`, `scripts/db`, `tests/migrations`. By the `migrations/versions/*` shared-file protocol, WS-02 also creates the no-op slot files 010 to 021 in W0; each passes to its owner on creation, and the owner edits only `upgrade()` and `downgrade()`. WS-02 reviews every slot diff (review only).

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

**Conventions used in every task.** Commands run from `D:\Repo\Contextual-Chatbot` (the PowerShell tool's working directory) with Windows PowerShell 5.1. The Python interpreter is always `& "D:\tmp\mvp-exec\venv\Scripts\python.exe"`. Real-infra commands add `$env:SOLACE_REQUIRE_REALINFRA="1"` so a missing Docker fails instead of skipping. The Postgres image for every test in this plan is `scripts.db.images.POSTGRES_IMAGE` (`postgres:16-alpine@sha256:16bc17c64a573ef34162af9298258d1aec548232985b33ed7b1eac33ba35c229`, the image already present locally; confirm with `docker image inspect --format "{{index .RepoDigests 0}}" postgres:16-alpine`). Tests generate every password at run time and never print one. New test files are named `test_mvp_<id>_<slug>.py`, except `tests/migrations/test_chain_real_pg.py`, which the skeleton names.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `infrastructure/init-db/01-create-databases.sql` | Modify (delete) | Nine unused per-service databases and a grant to a hard-coded role | WS-02.1 |
| `infrastructure/init-db/01-roles.sh` | Create | Create `solace_migrator` and `solace_app`; hand the database to the migrator | WS-02.1 |
| `scripts/db/__init__.py` | Create | Package marker | WS-02.1 |
| `scripts/db/images.py` | Create | `POSTGRES_IMAGE`, the pinned image reference | WS-02.1 |
| `tests/migrations/conftest.py` | Create | Docker Postgres fixtures (`pg_server`, `fresh_database`, `run_migration`, `insert_minimal`, `migrated_database`) | WS-02.1, WS-02.2, WS-02.4 |
| `tests/migrations/test_mvp_249_init_roles.py` | Create | Roles, database ownership, fail-closed init | WS-02.1 |
| `migrations/versions/001_initial_schema.py` | Modify | Drop the duplicate `memory_records.access_count` | WS-02.2 |
| `migrations/versions/007_outbox_claimed_at.py` | Modify | Guard the index and column on `to_regclass('public.event_outbox')` | WS-02.2 |
| `migrations/env.py` | Modify | Bare `solace_infrastructure` import, required `DATABASE_URL`, per-migration transactions (02.2); TLS connect args (02.5) | WS-02.2, WS-02.5 |
| `alembic.ini` | Modify | No credentials in `sqlalchemy.url` | WS-02.2 |
| `tests/migrations/test_chain_real_pg.py` | Create | Upgrade, round trip, grants, NULLIF, RLS isolation, schema head | WS-02.2, WS-02.3 |
| `migrations/versions/009_platform_roles_rls.py` | Create | Grants, default privileges, NULLIF policy rewrite | WS-02.3 |
| `migrations/versions/010_event_plane.py` to `021_personality_analytics.py` | Create | Twelve no-op slot revisions on the fixed chain | WS-02.3 |
| `src/solace_infrastructure/database/schema_check.py` | Create | `SCHEMA_HEAD`, `SchemaVersionMismatch`, `read_schema_version`, `assert_schema_head` | WS-02.3 |
| `tests/migrations/test_mvp_041_slot_chain.py` | Create | Chain shape, slot ownership, 009 policy coverage, schema-check semantics | WS-02.3 |
| `scripts/db/migrate.py` | Create | Migrate entrypoint (02.4); TLS (02.5) | WS-02.4, WS-02.5 |
| `tests/migrations/test_mvp_001_migrate_entrypoint.py` | Create | CLI behaviour on real Postgres | WS-02.4 |
| `scripts/db/gen_certs.py` | Create | CA and server certificate generation | WS-02.5 |
| `scripts/db/pg_tls_entrypoint.sh` | Create | Postgres container wrapper: TLS files, key permissions, TLS-only hba | WS-02.5 |
| `src/solace_infrastructure/postgres.py` | Modify | TLS policy and `ssl` import (02.5); strict config, DSN encoding, role refusal (02.6); pool defaults (02.8) | WS-02.5, WS-02.6, WS-02.8 |
| `tests/solace_infrastructure/test_ssl_enforcement.py` | Modify | Existing H-39 tests moved to the verify-full contract | WS-02.5, WS-02.6 |
| `tests/migrations/test_mvp_010_gen_certs.py` | Create | Certificate contents | WS-02.5 |
| `tests/migrations/test_mvp_010_tls_real_pg.py` | Create | verify-full, wrong CA, plaintext rejected, fail-closed server | WS-02.5 |
| `src/solace_infrastructure/database/connection_manager.py` | Modify | Default-pool aliases, strict checks, conflict message, GUC reset (02.6); leak detector and timeouts (02.9) | WS-02.6, WS-02.9 |
| `tests/solace_infrastructure/test_mvp_033_default_pool.py` | Create | Aliases, strict refusals, DSN encoding, GUC reset failure | WS-02.6 |
| `tests/migrations/test_mvp_041_pool_role_real_pg.py` | Create | Role refusal and special-character password on real Postgres | WS-02.6 |
| `src/solace_infrastructure/database/__init__.py` | Modify | PEP 562 lazy exports | WS-02.7 |
| `tests/solace_infrastructure/test_mvp_269_database_lazy_exports.py` | Create | Import cost and export completeness | WS-02.7 |
| `tests/solace_infrastructure/test_postgres.py` | Modify | Default pool sizes 1 and 8 | WS-02.8 |
| `tests/solace_infrastructure/test_mvp_140_pool_defaults.py` | Create | Defaults and strictness parity with the runtime | WS-02.8 |
| `tests/migrations/test_mvp_140_pool_budget_real_pg.py` | Create | Nine pools fit a stock `max_connections` | WS-02.8 |
| `tests/solace_infrastructure/test_mvp_272_leak_detection.py` | Create | Leak checker starts and warns; no stack capture when off | WS-02.9 |
| `tests/solace_infrastructure/test_mvp_274_acquire_timeouts.py` | Create | Query timeouts are not relabelled as pool timeouts | WS-02.9 |
| `tests/solace_infrastructure/database/test_rls_guc.py` | Modify | Fake pool keeps working with the explicit acquire context | WS-02.9 |

## Tasks

### Task WS-02.1: Platform roles and a single database from init-db, with real-Postgres fixtures
**Covers:** MVP-249, MVP-041
**Severity:** high
**Wave:** 0
**Phase:** A2 (first WS-02 task)
**Depends on:** WS-01.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-02 (default assumed: one PostgreSQL 16 container with one database `solace`; `solace_migrator` owns every table and is used only by the migrate job; `solace_app` is NOSUPERUSER NOBYPASSRLS with DML only; the image superuser `postgres` is ops-only; the nine per-service databases are dropped)
**Real infra:** yes
**Files:**
- Modify: `infrastructure/init-db/01-create-databases.sql` (delete the file: its nine `CREATE DATABASE` statements and the `GRANT ... TO solace` loop at lines 15 to 28 go away)
- Create: `infrastructure/init-db/01-roles.sh` (POSIX sh, LF line endings, safe whether the postgres entrypoint sources or executes it)
- Create: `scripts/db/__init__.py`, `scripts/db/images.py`, `tests/migrations/conftest.py`
- Test: `tests/migrations/test_mvp_249_init_roles.py`
**Interfaces:**
- Consumes: the postgres image's init contract (`POSTGRES_PASSWORD`, `POSTGRES_DB`, scripts in `/docker-entrypoint-initdb.d` run once on an empty data directory).
- Produces (IF-03 roles): `solace_migrator` = LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS, owner of the `solace` database (and through `pg_database_owner` of schema `public`); `solace_app` = LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; `REVOKE ALL ON DATABASE solace FROM PUBLIC`, `GRANT CONNECT, TEMPORARY` to both roles, `REVOKE CREATE ON SCHEMA public FROM PUBLIC`. Init environment (new, see New interfaces): `SOLACE_APP_PASSWORD` or `SOLACE_APP_PASSWORD_FILE`, and `SOLACE_MIGRATOR_PASSWORD` or `SOLACE_MIGRATOR_PASSWORD_FILE`, both required; setting a variable and its `_FILE` form together, or neither, makes init exit non-zero naming the variable. `scripts.db.images.POSTGRES_IMAGE: Final[str]`. Fixtures in `tests/migrations/conftest.py`: `pg_server` (session) yielding `PgServer(host: str, port: int, superuser_password: str, app_password: str, migrator_password: str)` with `dsn(user, password, database="solace") -> str` (percent-encoded), `superuser_dsn(database="solace")`, `migrator_dsn(database="solace")` and `app_dsn(database="solace")`; helper `start_postgres(*, env: dict[str, str], tls_dir: Path | None = None, tls_mode: str | None = None)` returning a started testcontainers `PostgresContainer`; `fresh_database` (function) yielding a new database name owned by `solace_migrator` with the same database privileges as `solace`, dropped `WITH (FORCE)` afterwards; `insert_minimal` returning `async def insert_minimal(conn, table: str, **values) -> None`, which fills every NOT NULL column without a default from `information_schema.columns` with a synthetic value of its type. Docker is required: the fixtures skip when it is unreachable and fail when `SOLACE_REQUIRE_REALINFRA=1`.
- Produces for requesting tasks: the two roles and the single database `solace` for WS-05.2 and WS-05.4 (WS-05 request #2) and for WS-03.2's compose postgres service. The init variables are `SOLACE_APP_PASSWORD` and `SOLACE_MIGRATOR_PASSWORD` (or their `_FILE` forms), with the image's own `POSTGRES_PASSWORD` staying the superuser's. WS-05's proposed names `POSTGRES_APP_PASSWORD`/`POSTGRES_MIGRATOR_PASSWORD` are not read, because WS-03 already interpolates `SOLACE_*` from the IF-02 sources `POSTGRES_PASSWORD` and `POSTGRES_MIGRATOR_PASSWORD`; WS-05's `INIT_DB_ENV` is asked to match (R-02.12).

Why migrator BYPASSRLS: under FORCE RLS a table owner is filtered like anyone else, so slot data migrations (backfills across users) would silently touch no rows. The owner can already disable RLS on its own tables, so BYPASSRLS adds no power, and services can never use the role (WS-02.6 refuses it).

- [ ] **Step 1: Write the failing test** in `tests/migrations/test_mvp_249_init_roles.py`
```python
"""MVP-249/041: init-db creates exactly the two platform roles and one application database, and fails closed."""
from __future__ import annotations

import secrets
from pathlib import Path

import asyncpg
import docker
import pytest

from scripts.db.images import POSTGRES_IMAGE

ROOT = Path(__file__).resolve().parents[2]
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
def test_mvp_249_init_fails_closed_without_role_passwords() -> None:
    client = docker.from_env()
    container = client.containers.run(
        POSTGRES_IMAGE, detach=True,
        environment={"POSTGRES_PASSWORD": secrets.token_urlsafe(16), "POSTGRES_DB": "solace"},
        volumes={str(ROOT / "infrastructure" / "init-db"): {"bind": "/docker-entrypoint-initdb.d", "mode": "ro"}})
    try:
        status = container.wait(timeout=120)["StatusCode"]
        logs = container.logs().decode("utf-8", "replace")
    finally:
        container.remove(force=True)
    assert status != 0
    assert "SOLACE_APP_PASSWORD" in logs
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations\test_mvp_249_init_roles.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The `pg_server` fixture errors for the first three tests: today's `01-create-databases.sql` runs `GRANT ALL PRIVILEGES ON DATABASE ... TO solace` while the superuser is `postgres`, so init aborts with `role "solace" does not exist` and the container exits before it is ready. The fail-closed test fails on `"SOLACE_APP_PASSWORD" in logs`: the container exits non-zero for the wrong reason.
- [ ] **Step 3: Implement** `01-roles.sh` (a local `_solace_file_env NAME` helper that reads `NAME` or `NAME_FILE`, rejects both or neither with a message naming the variable and exits 1; then `psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --set=app_password="$SOLACE_APP_PASSWORD" --set=migrator_password="$SOLACE_MIGRATOR_PASSWORD"` with a quoted heredoc that uses `:'app_password'`, `:'migrator_password'` and `format(... current_database()) \gexec`, so no value is ever interpolated by the shell or echoed). Delete `01-create-databases.sql`. Write `scripts/db/images.py` and `tests/migrations/conftest.py` per Produces. The `pg_server` fixture generates the superuser and app passwords with `secrets.token_urlsafe(18)` and a migrator password with URL-reserved characters (`"m/@#:%" + secrets.token_urlsafe(12)`), which also exercises MVP-273 in WS-02.6.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations -o "addopts=" -p no:logging -q`. Expected: all pass (the existing `test_extend_rls_phi_tables.py` is unaffected)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-02.2: Repair the migration chain so 001 to head applies and round-trips on an empty database
**Covers:** MVP-001, MVP-009
**Severity:** blocker
**Wave:** 0
**Phase:** A2
**Depends on:** WS-02.1, WS-01.7
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-01 (default assumed: Alembic is the only schema of record; 001 and 007 are repaired in place because no database has applied them; `event_outbox` is created by slot 010, owned by WS-06)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/001_initial_schema.py` (delete line 750, the second `sa.Column("access_count", ...)` on `memory_records`; `_audit_trail_columns()` at line 106 already supplies the identical column)
- Modify: `migrations/versions/007_outbox_claimed_at.py` (lines 42 to 49 become one `DO $$ BEGIN IF to_regclass('public.event_outbox') IS NOT NULL THEN ... END IF; END $$;` block holding both the `ADD COLUMN IF NOT EXISTS` and the `CREATE INDEX IF NOT EXISTS`; docstring says slot `010_event_plane` creates the table, column and index on fresh databases; `downgrade()` unchanged)
- Modify: `migrations/env.py` (insert the repo root and `src/` into `sys.path` if absent; line 13 becomes `from solace_infrastructure.database.base_models import Base`; online mode raises `RuntimeError("DATABASE_URL is required to run migrations")` when it is unset; offline mode falls back to the dialect-only URL `postgresql+asyncpg://`; both `context.configure(...)` calls pass `transaction_per_migration=True`)
- Modify: `alembic.ini` (line 32 becomes `sqlalchemy.url =` with a comment that `DATABASE_URL` supplies it)
- Modify: `tests/migrations/conftest.py` (add fixture `run_migration`, returning `def run_migration(database_url: str, command: str, target: str) -> subprocess.CompletedProcess`, which runs `python -m alembic -c <repo>/alembic.ini <command> <target>` with `DATABASE_URL` set; WS-02.4 switches it to the migrate entrypoint)
- Test: `tests/migrations/test_chain_real_pg.py`
**Interfaces:**
- Consumes: fixtures from WS-02.1; WS-01.7's guard case `tests/build/test_import_roots.py::test_mvp_270_no_src_root_imports[WS-02]`.
- Produces: a chain that applies on an empty PostgreSQL 16, one transaction per revision (a failure leaves the database at the last good revision and a re-run continues), no runtime-created table assumed by any revision, and no credentials in `alembic.ini`. WS-06's slot 010 must create `event_outbox` with `claimed_at` and `idx_outbox_publishing_claimed` (request R-02.5).

If this test exposes a further failure in 002 to 008 on real Postgres, repair it in place in this task (WS-02 owns those files) and name it in the report.

- [ ] **Step 1: Write the failing test** in `tests/migrations/test_chain_real_pg.py`
```python
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
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations\test_chain_real_pg.py -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_import_roots.py -k "WS-02" -o "addopts=" -p no:logging -q`
Expected: FAIL. The three real-PG tests exit non-zero with `sqlalchemy.exc.DuplicateColumnError: A column with name 'access_count' is already present in table 'memory_records'` in stderr (001 builds the table). After fixing 001 alone, `test_mvp_009_upgrade_through_007_needs_no_runtime_outbox` still fails with `relation "event_outbox" does not exist` from 007's index; record both RED outputs. The env test fails (no `transaction_per_migration`), the ini test fails (`solace:solace@localhost`), and the import-root case WS-02 fails on `migrations/env.py: 13:src.solace_infrastructure.database.base_models`.
- [ ] **Step 3: Implement** the edits listed under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations -o "addopts=" -p no:logging -q`, then the same command with `D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-02.3: Slot 009 grants and NULLIF policies, the fixed slot chain 010 to 021, and the schema-head check
**Covers:** MVP-041, MVP-058
**Severity:** high
**Wave:** 0
**Phase:** A2
**Depends on:** WS-02.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-02 (default assumed: services run as `solace_app` from Wave 0; no BYPASSRLS role for services). D-01 (default assumed: owners edit only their slot's `upgrade()` and `downgrade()`)
**Real infra:** yes
**Files:**
- Create: `migrations/versions/009_platform_roles_rls.py`
- Create: `migrations/versions/010_event_plane.py`, `migrations/versions/011_audit_trail.py`, `migrations/versions/012_user_accounts.py`, `migrations/versions/013_identity_mfa.py`, `migrations/versions/014_safety.py`, `migrations/versions/015_notification.py`, `migrations/versions/016_orchestrator_threads.py`, `migrations/versions/017_orchestrator_checkpoints.py`, `migrations/versions/018_memory.py`, `migrations/versions/019_therapy.py`, `migrations/versions/020_diagnosis.py`, `migrations/versions/021_personality_analytics.py`
- Create: `src/solace_infrastructure/database/schema_check.py`
- Test: `tests/migrations/test_mvp_041_slot_chain.py`, `tests/migrations/test_chain_real_pg.py` (add the grant, NULLIF, isolation and schema-head tests below)
**Interfaces:**
- Consumes: the roles from WS-02.1; the `_TABLES_WITH_RLS` lists of 002, 006 and 008.
- Produces (IF-03): slot ids exactly as listed, each file stem equal to its revision id, `down_revision` the previous slot, a docstring line `Owner: WS-NN`, empty `upgrade()`/`downgrade()` for 010 to 021. In 009: `POLICY_TABLES: tuple[str, ...]` (the 15 tables of 002, 006 and 008), `NULLIF_PREDICATE = "user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid"`; `upgrade()` runs, when the dialect is PostgreSQL, `GRANT USAGE ON SCHEMA public TO solace_app`, `GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO solace_app`, `GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO solace_app`, `REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON alembic_version FROM solace_app`, `ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO solace_app` (and `USAGE, SELECT ON SEQUENCES`), and `ALTER POLICY <table>_by_user ON <table> USING (<NULLIF_PREDICATE>) WITH CHECK (<NULLIF_PREDICATE>)` for each table; `downgrade()` restores the original predicate and revokes in reverse order. `solace_infrastructure.database.schema_check`: `SCHEMA_HEAD: Final = "021_personality_analytics"`; `class SchemaVersionMismatch(DatabaseError)` with `.expected: str` and `.found: str | None`; `async def read_schema_version(pool_or_conn) -> str | None` (None when `alembic_version` is missing or empty); `async def assert_schema_head(pool: asyncpg.Pool, *, expected: str = SCHEMA_HEAD, environment: str | None = None) -> str | None`, which returns the found version, raises `SchemaVersionMismatch` in strict environments (ENVIRONMENT staging or production) on any mismatch, and logs a WARNING otherwise. `alembic_version` is SELECT-only for `solace_app`, so the check runs as the app role.

- Produces for requesting tasks: the empty slot `010_event_plane` that WS-06.1 fills later in the same phase-A2 lane (R-02.5; WS-06.1 must list WS-02.3 in its Depends on, requested as R-02.13); `SCHEMA_HEAD` in `solace_infrastructure.database.schema_check` for WS-05.2 and WS-05.4 (WS-05 request #2) and for WS-04's bootstrap check (R-02.3); the slot scaffolds that WS-08.15, WS-10.9 (its `WS-02:slot-017-scaffold` is this task), WS-11.6 (`WS-02:slot-016-scaffold`), WS-13.3, WS-14.1, WS-14.4, WS-16.2 and WS-18.3 fill.

Shared-file handoff: `migrations/versions/010_event_plane.py` to `021_personality_analytics.py` are created here and then owned by the slot owners named in IF-03, who edit only `upgrade()` and `downgrade()` (skeleton shared_files `migrations/versions/*`: "WS-02 ... creates no-op slot files 010-021 ... Each slot file is then owned by its workstream"). Only slot 010 is filled in the same wave (WS-06.1, W0 phase A2), so that edge needs the explicit dependency WS-06.1 -> WS-02.3 (R-02.13).

- [ ] **Step 1: Write the failing test** in `tests/migrations/test_mvp_041_slot_chain.py` and the additions to `tests/migrations/test_chain_real_pg.py`
```python
# tests/migrations/test_mvp_041_slot_chain.py
"""MVP-041/058 (IF-03): one linear chain with fixed slot ids; 009 covers every user policy; schema-head semantics."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import asyncpg
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ROOT / "migrations" / "versions"
EXPECTED_CHAIN = ["001_initial", "002_enable_rls", "003_escalations", "004_oauth_accounts", "005_widen_phi_columns",
                  "006_extend_rls", "007_outbox_claimed_at", "008_rls_personality", "009_platform_roles_rls",
                  "010_event_plane", "011_audit_trail", "012_user_accounts", "013_identity_mfa", "014_safety",
                  "015_notification", "016_orchestrator_threads", "017_orchestrator_checkpoints", "018_memory",
                  "019_therapy", "020_diagnosis", "021_personality_analytics"]
SLOT_OWNERS = {"009_platform_roles_rls": "WS-02", "010_event_plane": "WS-06", "011_audit_trail": "WS-09",
               "012_user_accounts": "WS-14", "013_identity_mfa": "WS-08", "014_safety": "WS-12",
               "015_notification": "WS-13", "016_orchestrator_threads": "WS-11",
               "017_orchestrator_checkpoints": "WS-10", "018_memory": "WS-17", "019_therapy": "WS-15",
               "020_diagnosis": "WS-16", "021_personality_analytics": "WS-18"}


def _load(filename: str):
    spec = importlib.util.spec_from_file_location(f"mig_{filename[:3]}_under_test", VERSIONS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_chain_is_linear_and_ends_at_schema_head() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    script = ScriptDirectory.from_config(cfg)
    assert [r.revision for r in reversed(list(script.walk_revisions("base", "heads")))] == EXPECTED_CHAIN
    from solace_infrastructure.database.schema_check import SCHEMA_HEAD

    assert script.get_heads() == [SCHEMA_HEAD] == ["021_personality_analytics"]


def test_slot_ids_fit_the_alembic_version_column() -> None:
    assert all(len(rev) <= 32 for rev in EXPECTED_CHAIN)


@pytest.mark.parametrize("slot,owner", sorted(SLOT_OWNERS.items()))
def test_slot_file_named_by_revision_and_owned(slot: str, owner: str) -> None:
    text = (VERSIONS / f"{slot}.py").read_text(encoding="utf-8")
    previous = EXPECTED_CHAIN[EXPECTED_CHAIN.index(slot) - 1]
    assert f'revision = "{slot}"' in text and f'down_revision = "{previous}"' in text
    assert f"Owner: {owner}" in text


def test_mvp_058_009_rewrites_every_existing_user_policy() -> None:
    existing = set()
    for name in ("002_enable_rls_clinical_tables.py", "006_extend_rls_phi_tables.py", "008_rls_personality_tables.py"):
        existing |= set(_load(name)._TABLES_WITH_RLS)
    m009 = _load("009_platform_roles_rls.py")
    assert len(existing) == 15 and set(m009.POLICY_TABLES) == existing
    assert m009.NULLIF_PREDICATE == "user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid"


class _Pool:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    async def fetchval(self, sql: str):
        if self._exc:
            raise self._exc
        return self._result


async def test_assert_schema_head_raises_only_in_strict_environments() -> None:
    from solace_infrastructure.database.schema_check import SchemaVersionMismatch, assert_schema_head

    with pytest.raises(SchemaVersionMismatch) as exc:
        await assert_schema_head(_Pool("008_rls_personality"), environment="staging")
    assert (exc.value.expected, exc.value.found) == ("021_personality_analytics", "008_rls_personality")
    assert await assert_schema_head(_Pool("008_rls_personality"), environment="development") == "008_rls_personality"
    with pytest.raises(SchemaVersionMismatch):
        await assert_schema_head(_Pool(exc=asyncpg.exceptions.UndefinedTableError("alembic_version")), environment="production")
    assert await assert_schema_head(_Pool("021_personality_analytics"), environment="production") == "021_personality_analytics"
    with pytest.raises(SchemaVersionMismatch) as newer:  # a restore onto a database migrated past this release
        await assert_schema_head(_Pool("022_future_slot"), environment="production")
    assert newer.value.found == "022_future_slot"
```
```python
# appended to tests/migrations/test_chain_real_pg.py
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
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations\test_mvp_041_slot_chain.py D:\Repo\Contextual-Chatbot\tests\migrations\test_chain_real_pg.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_chain_real_pg.py` cannot be collected (`ModuleNotFoundError: solace_infrastructure.database.schema_check`); comment out that one import line to read the real-PG REDs, then restore it. With it commented: the DML test fails on `permission denied for table alembic_version` (no grants exist for `solace_app`); the policy test lists all 15 tables; the reset-GUC test fails with `InvalidTextRepresentationError: invalid input syntax for type uuid: ""` on the second count; the isolation test fails with the same error at the `''` insert. `test_mvp_041_slot_chain.py`: the chain ends at `008_rls_personality`, the 13 slot files are missing, `009_platform_roles_rls.py` is missing, and `schema_check` is missing. The audit test passes vacuously.
- [ ] **Step 3: Implement** the thirteen revision files and `schema_check.py` per Produces. 009 keeps the `_is_postgres()` guard used by 002 to 008, so the SQLite-based migration unit tests stay a no-op.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations -o "addopts=" -p no:logging -q`, then the same command with `D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-02.4: Migrate entrypoint: wait for Postgres, refuse the wrong role, serialize on an advisory lock
**Covers:** MVP-001
**Severity:** blocker
**Wave:** 0
**Phase:** A2
**Depends on:** WS-02.3
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-02 (default assumed: migrations run only as `solace_migrator`, from a compose one-shot `migrate` service that app services wait on with `service_completed_successfully`)
**Real infra:** yes
**Files:**
- Create: `scripts/db/migrate.py`
- Modify: `tests/migrations/conftest.py` (`run_migration` now runs `python -m scripts.db.migrate <command> <target>`, so every chain test proves the real entrypoint; add fixture `migrated_database`, a `fresh_database` upgraded to head)
- Test: `tests/migrations/test_mvp_001_migrate_entrypoint.py`
**Interfaces:**
- Consumes: `alembic.ini`, `migrations/env.py` (WS-02.2), the roles (WS-02.1).
- Produces (IF-03): `python -m scripts.db.migrate {upgrade,downgrade,current} [target] [--wait-timeout SECONDS (default 120)] [--lock-timeout SECONDS (default 600)] [--expected-role NAME (default solace_migrator)]`, exit codes 0 success, 1 failure, 2 usage (no database URL). Database URL: `DATABASE_URL` (also via `DATABASE_URL_FILE` through the image entrypoint), else `PostgresSettings` from `POSTGRES_*` when `POSTGRES_PASSWORD` is set. Steps: wait for a connection; refuse with exit 1 unless `current_user` equals the expected role and is not a superuser; take `pg_try_advisory_lock(hashtext('solace_migrate'))`, retrying until the lock timeout; run the Alembic command in a worker thread; print `migrated: <version or base>`; release the lock. Output never contains a password (`redact(url) -> str` shows `user:***@host:port/db`). Python API: `run_migrations(database_url: str, command: str, target: str = "head", *, wait_timeout_s: float = 120.0, lock_timeout_s: float = 600.0, expected_role: str = "solace_migrator") -> str | None` (synchronous; call it from sync code or a thread) and `async def migrate_async(...)` with the same parameters; `resolve_database_url(environ: Mapping[str, str]) -> str | None`. WS-05's `pg_migrated` fixture uses `run_migrations` (request R-02.4).
- Produces for requesting tasks: `python -m scripts.db.migrate upgrade|downgrade <rev>` reading `DATABASE_URL`, for WS-05.2 and WS-05.4 (WS-05 request #2) and for WS-03.2's compose `migrate` service; the CLI that WS-01.9 packages as the `solace-migrate` image (R-01.13).

- [ ] **Step 1: Write the failing test** in `tests/migrations/test_mvp_001_migrate_entrypoint.py`
```python
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
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations\test_mvp_001_migrate_entrypoint.py -o "addopts=" -p no:logging -q`
Expected: FAIL: every CLI call exits 1 with `No module named scripts.db.migrate`, so the return-code and message assertions fail; the API test fails with `ModuleNotFoundError`. This is the defect: today the only migration path is a manual `alembic upgrade head` (Makefile line 20, DEPLOY.md) that no container can run.
- [ ] **Step 3: Implement** `scripts/db/migrate.py` per Produces (asyncpg for the wait, role check and lock connection; `asyncio.to_thread(alembic.command.<cmd>, cfg, target)` for the migration, because `env.py` calls `asyncio.run`; `if __name__ == "__main__": raise SystemExit(main())`), and switch the conftest fixtures.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations -o "addopts=" -p no:logging -q`. Expected: all pass (the chain tests now run through the entrypoint)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-02.5: Verified TLS between services and the bundled Postgres
**Covers:** MVP-010, MVP-271
**Severity:** blocker
**Wave:** 0
**Phase:** A2
**Depends on:** WS-02.4, WS-01.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-15 (default assumed: TLS is on for the bundled Postgres with a CA and server certificate generated by `scripts/db/gen_certs.py`; clients use `POSTGRES_SSL_MODE=verify-full` against host `postgres` with `POSTGRES_SSL_ROOT_CERT`; the dev profile may use `prefer`). D-14 (default assumed: staging is as strict as production)
**Real infra:** yes
**Files:**
- Create: `scripts/db/gen_certs.py`, `scripts/db/pg_tls_entrypoint.sh` (POSIX sh, LF line endings)
- Modify: `src/solace_infrastructure/postgres.py` (add `import ssl` for the annotation at line 94, which clears the F821 the CI lint gate reports; module-level `_is_strict_environment(environment: str | None = None) -> bool` (lower-cased ENVIRONMENT in {staging, production}), `effective_ssl_mode(configured: str, *, environment: str | None = None) -> str` and `build_ssl_context(ssl_mode: str, ssl_root_cert: str | None, *, environment: str | None = None) -> ssl.SSLContext | None`; `class DatabaseConfigError(DatabaseError)`; `PostgresSettings.get_effective_ssl_mode()` and `.get_ssl_context()` delegate to the two functions)
- Modify: `migrations/env.py` (the async engine gets `connect_args={"ssl": ctx}` when `build_ssl_context(POSTGRES_SSL_MODE, POSTGRES_SSL_ROOT_CERT)` returns a context)
- Modify: `scripts/db/migrate.py` (its asyncpg connections use the same `build_ssl_context`)
- Test: `tests/migrations/test_mvp_010_gen_certs.py`, `tests/migrations/test_mvp_010_tls_real_pg.py`, `tests/solace_infrastructure/test_ssl_enforcement.py` (existing H-39 tests moved to the new contract: the strict cases now expect `verify-full` instead of `require`; `test_strict_mode_preserved` runs under `development`; the production context tests pass a root certificate generated by `gen_certs` into `tmp_path`)
**Interfaces:**
- Consumes: the migrate entrypoint (WS-02.4); WS-01.7's guard case `tests/build/test_mvp_038_lint_gate.py::test_mvp_038_lint_gate_clean[WS-02]`.
- Produces: in strict environments the effective mode is always `verify-full` (a WARNING is logged when another mode was configured), `check_hostname=True`, `CERT_REQUIRED`, and the CA comes only from `ssl_root_cert`; `DatabaseConfigError` when it is unset or not a readable file (the message names `POSTGRES_SSL_ROOT_CERT`, never a value). Outside strict environments the existing behaviour stays (`prefer` returns None). `gen_certs.py --out DIR [--san NAME ...] [--ip ADDR ...] [--days N] [--force]` writes `postgres-ca.crt`, `server.crt` and `server.key` (ECDSA P-256, server EKU serverAuth, default SANs `postgres` and `localhost` and IP `127.0.0.1`, keys mode 0600 where the OS supports it), keeps no CA private key, refuses to overwrite without `--force` (exit 1), and prints file names only; `main(argv: list[str] | None = None) -> int`. `pg_tls_entrypoint.sh` (run as `sh /usr/local/bin/solace-pg-entrypoint.sh postgres ...`): when `$SOLACE_PG_TLS_DIR` (default `/run/secrets`) holds `server.crt` and `server.key`, it copies them to `/var/lib/postgresql/tls`, sets owner `postgres` and mode 0600 on the key, and appends `-c ssl=on -c ssl_cert_file=... -c ssl_key_file=...`; with `SOLACE_PG_TLS=required` it also writes an hba file (`local all all trust`, `hostssl all all all scram-sha-256`, `host all all all reject`) and appends `-c hba_file=...`, and exits 1 naming `server.crt` when the files are missing; then `exec docker-entrypoint.sh "$@"`.
- Produces for requesting tasks: `python -m scripts.db.gen_certs --out <dir>` writing `postgres-ca.crt`, `server.crt` and `server.key` for WS-05.4's strict test stack (WS-05 request #2) and WS-03.2/WS-03.7's compose TLS files; the `ssl` import that turns WS-01.7's lint case `[WS-02]` green (R-01.13).

- [ ] **Step 1: Write the failing test** in `tests/migrations/test_mvp_010_gen_certs.py`, `tests/migrations/test_mvp_010_tls_real_pg.py` and the updated `tests/solace_infrastructure/test_ssl_enforcement.py`
```python
# tests/migrations/test_mvp_010_gen_certs.py
"""MVP-010: deploy-time CA and server certificate for verify-full against host 'postgres'."""
from __future__ import annotations

from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID


def test_mvp_010_gen_certs_writes_ca_and_server_pair(tmp_path: Path, capsys) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path), "--san", "postgres", "--san", "localhost", "--ip", "127.0.0.1"]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {"postgres-ca.crt", "server.crt", "server.key"}
    ca = x509.load_pem_x509_certificate((tmp_path / "postgres-ca.crt").read_bytes())
    srv = x509.load_pem_x509_certificate((tmp_path / "server.crt").read_bytes())
    san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert set(san.get_values_for_type(x509.DNSName)) == {"postgres", "localhost"}
    assert {str(i) for i in san.get_values_for_type(x509.IPAddress)} == {"127.0.0.1"}
    ca.public_key().verify(srv.signature, srv.tbs_certificate_bytes, ec.ECDSA(srv.signature_hash_algorithm))
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca is True
    assert ExtendedKeyUsageOID.SERVER_AUTH in srv.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    out = capsys.readouterr().out
    assert "BEGIN" not in out and "PRIVATE" not in out


def test_mvp_010_gen_certs_refuses_to_overwrite(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path)]) == 0
    assert gen_certs.main(["--out", str(tmp_path)]) == 1
    assert gen_certs.main(["--out", str(tmp_path), "--force"]) == 0
```
```python
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


async def test_mvp_010_strict_client_connects_with_verify_full(tls_pg, monkeypatch) -> None:
    s = _settings(tls_pg, monkeypatch, tls_pg["ca"])
    assert s.get_effective_ssl_mode() == "verify-full"
    conn = await asyncpg.connect(s.get_dsn(), ssl=s.get_ssl_context())
    try:
        assert await conn.fetchval("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()") is True
    finally:
        await conn.close()


async def test_mvp_271_a_certificate_from_another_ca_is_rejected(tls_pg, monkeypatch, tmp_path: Path) -> None:
    assert gen_certs.main(["--out", str(tmp_path), "--san", "localhost"]) == 0
    s = _settings(tls_pg, monkeypatch, tmp_path / "postgres-ca.crt")
    with pytest.raises((ssl.SSLCertVerificationError, ssl.SSLError)):
        await asyncpg.connect(s.get_dsn(), ssl=s.get_ssl_context())


async def test_mvp_010_required_server_rejects_plaintext_clients(tls_pg, monkeypatch) -> None:
    s = _settings(tls_pg, monkeypatch, tls_pg["ca"])
    with pytest.raises(asyncpg.exceptions.InvalidAuthorizationSpecificationError):
        await asyncpg.connect(s.get_dsn(), ssl=False)


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
        container.remove(force=True)
    assert status != 0 and "server.crt" in logs


def test_mvp_010_migrate_job_runs_over_verify_full(tls_pg) -> None:
    url = (f"postgresql://solace_migrator:{tls_pg['SOLACE_MIGRATOR_PASSWORD']}@{tls_pg['host']}:{tls_pg['port']}/solace")
    env = {k: v for k, v in os.environ.items() if not k.startswith("POSTGRES_")}
    env.update({"PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "production", "DATABASE_URL": url,
                "POSTGRES_SSL_MODE": "verify-full", "POSTGRES_SSL_ROOT_CERT": str(tls_pg["ca"])})
    r = subprocess.run([sys.executable, "-m", "scripts.db.migrate", "upgrade", "head"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
```
```python
# tests/solace_infrastructure/test_ssl_enforcement.py (revised strict cases; development cases unchanged)
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
```
The revised file adds a `root_cert` fixture (`gen_certs.main(["--out", str(tmp_path)])`, returns `tmp_path / "postgres-ca.crt"`), changes `test_strict_mode_preserved` to run under `development`, and makes `test_production_yields_real_ssl_context`, `test_production_cannot_disable_ssl` and `test_ssl_context_passed_to_create_pool` pass `ssl_root_cert` and assert `verify_mode == ssl.CERT_REQUIRED`.
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations\test_mvp_010_gen_certs.py D:\Repo\Contextual-Chatbot\tests\migrations\test_mvp_010_tls_real_pg.py -o "addopts=" -p no:logging -q`, then the same command with `D:\Repo\Contextual-Chatbot\tests\solace_infrastructure\test_ssl_enforcement.py`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_038_lint_gate.py -k "WS-02" -o "addopts=" -p no:logging -q`
Expected: FAIL. `scripts.db.gen_certs` does not exist, so both TLS files error at import; `test_mvp_271_strict_effective_mode_is_verify_full` fails with `'require' == 'verify-full'` for disable, allow, prefer and require (the `require` floor at postgres.py lines 73 to 83 that gives `CERT_NONE` at lines 110 to 113); the strict-without-root-cert test fails with `DID NOT RAISE` (today verify-full silently uses the system store and require skips verification); the lint case fails on `postgres.py:94 F821 Undefined name 'ssl'`.
- [ ] **Step 3: Implement** the files listed. The strict rule lives in `effective_ssl_mode` and `build_ssl_context` only, so `PostgresClient`, `ConnectionPoolManager.get_pool` (which already passes `get_ssl_context()` to `asyncpg.create_pool`), env.py and migrate.py share one policy.
- [ ] **Step 4: Run it and confirm it passes**
Run: the three commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** one process per suite with the real-infra command for `D:\Repo\Contextual-Chatbot\tests\migrations` and the standard command for `D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`, `D:\Repo\Contextual-Chatbot\tests\solace_security` and `D:\Repo\Contextual-Chatbot\services\user-service\tests` (they construct `PostgresSettings` and production guards). Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-02.6: One default pool per service, strict-environment fail-loud, encoded DSNs and a role check
**Covers:** MVP-033, MVP-273, MVP-041, MVP-058
**Severity:** blocker
**Wave:** 0
**Phase:** A2
**Depends on:** WS-02.5
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-14 (default assumed: staging and production are strict). D-02 (default assumed: services connect only as `solace_app`)
**Real infra:** yes
**Files:**
- Modify: `src/solace_infrastructure/postgres.py` (`get_dsn()` percent-encodes user and password with `urllib.parse.quote(..., safe="")`; new `PostgresSettings.validate_for_environment(environment: str | None = None) -> None` raising `DatabaseConfigError` in strict environments when `host` is `localhost`, `127.0.0.1` or `::1` (the default when `POSTGRES_HOST` is unset) or the password is empty; `PostgresClient.connect` calls it and, in strict environments, `verify_app_role` on its first connection)
- Modify: `src/solace_infrastructure/database/connection_manager.py` (`DEFAULT_POOL = "default"`, `LEGACY_POOL_ALIASES = frozenset({"user_db", "diagnosis_db", "therapy_db", "personality_db", "safety_db"})`, `canonical_pool_name(name: str) -> str`, applied at the top of `register_pool`, `get_pool`, `acquire`, `transaction`, `close_pool`, `get_pool_stats` and `health_check`; `register_pool` builds settings through a helper that turns a missing `POSTGRES_PASSWORD` (pydantic `ValidationError`) into `DatabaseConfigError` and calls `validate_for_environment`; its conflict `ValueError` names the pool and host and database only, never a DSN; `get_pool` in strict environments runs `verify_app_role` on one connection of the new pool and closes the pool before raising; `async def verify_app_role(conn, *, environment: str | None = None) -> None` raises `DatabaseConfigError` when `current_user` is a superuser, has BYPASSRLS, or owns any table in schema `public`, and is a no-op outside strict environments; when the RLS reset `set_config('app.current_user_id', '', false)` fails on release, the connection is terminated (`conn.terminate()`) instead of being handed back with a user still set; a public module-level `async def init_connection(conn: asyncpg.Connection) -> None` registers the `json` and `jsonb` codecs (today's `_init_connection` body, lines 275 to 294), and the classmethod `_init_connection` delegates to it, so `create_pool(init=cls._init_connection)` and existing callers keep working)
- Test: `tests/solace_infrastructure/test_mvp_033_default_pool.py`, `tests/migrations/test_mvp_041_pool_role_real_pg.py`, `tests/solace_infrastructure/test_ssl_enforcement.py` (`test_ssl_context_passed_to_create_pool` uses host `db.internal` and a fake pool whose `acquire()` yields a connection answering the role query as a plain app role)
**Interfaces:**
- Consumes: `DatabaseConfigError` and `build_ssl_context` (WS-02.5); `solace_common.request_context.get_current_user_id` (unchanged).
- Produces (IF-03 pools): `await ConnectionPoolManager.get_pool("default")` is the one pool per service, built from `POSTGRES_*`; the legacy names return the same pool object; `DatabaseConfigError` is the strict-environment failure that WS-04's bootstrap maps to `StartupDependencyError`; `verify_app_role` and `canonical_pool_name` (New interfaces). Repositories that use the legacy names (user-service, therapy, diagnosis, personality, safety) need no edits.
- Produces for requesting tasks: `solace_infrastructure.database.connection_manager.init_connection(conn)`, the public pool codec helper that WS-05.2's `pg_migrated` app pool passes as `init=` (WS-05 request #2; WS-05's plan falls back to `_init_connection` until it exists, and WS-15's realinfra test keeps using `_init_connection`, which still works); the `default` pool and strict refusals that WS-04's bootstrap maps (R-02.3) and that WS-14.1, WS-18.1 and WS-11.7 (`WS-02:IF-03-pool`) consume.

- [ ] **Step 1: Write the failing test** in `tests/solace_infrastructure/test_mvp_033_default_pool.py` and `tests/migrations/test_mvp_041_pool_role_real_pg.py`
```python
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
```
```python
# tests/migrations/test_mvp_041_pool_role_real_pg.py
"""MVP-041/273: services refuse superuser, BYPASSRLS and table-owner roles in strict environments; encoded passwords connect."""
from __future__ import annotations

import asyncpg
import pytest

from solace_infrastructure.postgres import PostgresSettings

pytestmark = pytest.mark.realinfra


async def test_mvp_041_verify_app_role_refuses_privileged_roles(pg_server, migrated_database) -> None:
    from solace_infrastructure.database.connection_manager import verify_app_role
    from solace_infrastructure.postgres import DatabaseConfigError

    for dsn in (pg_server.superuser_dsn(migrated_database), pg_server.migrator_dsn(migrated_database)):
        conn = await asyncpg.connect(dsn)
        try:
            with pytest.raises(DatabaseConfigError):
                await verify_app_role(conn, environment="production")
            await verify_app_role(conn, environment="development")  # no-op outside strict
        finally:
            await conn.close()
    conn = await asyncpg.connect(pg_server.app_dsn(migrated_database))
    try:
        await verify_app_role(conn, environment="production")
    finally:
        await conn.close()


async def test_mvp_273_password_with_reserved_characters_connects(pg_server) -> None:
    s = PostgresSettings(host=pg_server.host, port=pg_server.port, user="solace_migrator",
                         password=pg_server.migrator_password, database="solace")
    conn = await asyncpg.connect(s.get_dsn())
    try:
        assert await conn.fetchval("SELECT current_user") == "solace_migrator"
    finally:
        await conn.close()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_infrastructure\test_mvp_033_default_pool.py -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations\test_mvp_041_pool_role_real_pg.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The alias test fails on `get_pool('user_db') is default` (each legacy name auto-registers its own pool from `PostgresSettings()` at connection_manager.py lines 216 to 223, so `create_pool` runs six times); both strict tests fail with `DID NOT RAISE` (a localhost pool and a missing password reach `create_pool`, or raise a raw pydantic `ValidationError`); the DSN test fails because the raw `#` starts a URL fragment (postgres.py lines 89 to 92); the conflict test finds both passwords in the `ValueError` text (lines 172 to 177); the codec test fails with `ImportError: cannot import name 'init_connection'` (only the private classmethod exists, line 275); the GUC test fails on `terminated is True` (the reset failure is only logged at lines 367 to 368). The real-PG role test fails with `ImportError: verify_app_role`; the reserved-character password fails to connect (`ValueError` or `InvalidPasswordError` from the unencoded DSN).
- [ ] **Step 3: Implement** the changes listed under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** one process per suite, each with the standard command: `D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`, `D:\Repo\Contextual-Chatbot\tests\solace_security`, `D:\Repo\Contextual-Chatbot\tests\integration`, `D:\Repo\Contextual-Chatbot\services\user-service\tests`, `D:\Repo\Contextual-Chatbot\services\therapy_service\tests`, `D:\Repo\Contextual-Chatbot\services\diagnosis_service\tests`, `D:\Repo\Contextual-Chatbot\services\personality_service\tests`, `D:\Repo\Contextual-Chatbot\services\safety_service\tests`; then the real-infra command for `D:\Repo\Contextual-Chatbot\tests\migrations`. Expected: all pass. A service test that relied on a per-name pool is a finding for that service's owner, recorded in the ledger, not an edit here.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-02.7: Lazy exports for solace_infrastructure.database (supporting MVP-269)
**Covers:** MVP-269
**Severity:** low
**Wave:** 0
**Phase:** A2
**Depends on:** WS-01.3
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-19 (default assumed: unused engines such as the Weaviate schema manager and the migrations runner stay in place but are never imported at boot)
**Real infra:** no
**Files:**
- Modify: `src/solace_infrastructure/database/__init__.py` (replace the eager imports at lines 38 to 113 with a `_EXPORTS` map and a module-level `__getattr__`, the same pattern as WS-01.3; `__all__` unchanged; `__getattr__` raises `AttributeError` for unknown names so `from solace_infrastructure.database import <submodule>` still falls back to the submodule)
- Test: `tests/solace_infrastructure/test_mvp_269_database_lazy_exports.py`
**Interfaces:**
- Consumes: the lazy `solace_infrastructure` package (WS-01.3).
- Produces: `import solace_infrastructure.database.connection_manager` and `.base_models` (used by `migrations/env.py`) load neither weaviate nor alembic nor the Redis or seed modules. MVP-269 is WS-01's; this is the supporting part in WS-02's file, and WS-01.12 closes the issue.

- [ ] **Step 1: Write the failing test** in `tests/solace_infrastructure/test_mvp_269_database_lazy_exports.py`
```python
"""MVP-269 (supporting): the database package no longer drags weaviate and alembic into every service."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEAVY = ("weaviate", "alembic", "solace_infrastructure.weaviate", "solace_infrastructure.redis",
         "solace_infrastructure.database.migrations_runner", "solace_infrastructure.database.weaviate_schema",
         "solace_infrastructure.database.seed_data", "solace_infrastructure.database.redis_setup")


def _loaded(import_line: str) -> list[str]:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "test"}
    code = f"import json, sys\n{import_line}\nprint(json.dumps(sorted(m for m in sys.modules if m.startswith({HEAVY!r}))))"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", ["solace_infrastructure.database.connection_manager",
                                    "solace_infrastructure.database.base_models"])
def test_mvp_269_database_submodule_import_skips_optional_backends(module: str) -> None:
    assert _loaded(f"import {module}") == []


def test_mvp_269_database_exports_still_resolve() -> None:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "test"}
    code = ("import solace_infrastructure.database as d\n"
            "assert len(d.__all__) == 58 and {'ConnectionPoolManager', 'Base', 'MigrationRunner', 'setup_redis'} <= set(d.__all__)\n"
            "print([n for n in d.__all__ if getattr(d, n, None) is None])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[]"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_infrastructure\test_mvp_269_database_lazy_exports.py -o "addopts=" -p no:logging -q`
Expected: FAIL in both import cases, listing `alembic`, `weaviate`, `solace_infrastructure.database.migrations_runner`, `solace_infrastructure.database.weaviate_schema`, `solace_infrastructure.weaviate`, `solace_infrastructure.redis`, `solace_infrastructure.database.redis_setup` and `solace_infrastructure.database.seed_data`, loaded by `database/__init__.py` lines 61 to 105. The export test passes today (`__all__` at lines 115 to 181 holds 58 unique names) and guards the refactor.
- [ ] **Step 3: Implement** the lazy `__init__.py`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** standard command, one process each, for `D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`, `D:\Repo\Contextual-Chatbot\tests\test_syntax_and_imports.py`, `D:\Repo\Contextual-Chatbot\tests\integration` and `D:\Repo\Contextual-Chatbot\services\memory_service\tests`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-02.8: Pool sizes that fit a stock Postgres max_connections
**Covers:** MVP-140
**Severity:** medium
**Wave:** 2
**Phase:** A
**Depends on:** WS-02.6
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** yes
**Files:**
- Modify: `src/solace_infrastructure/postgres.py` (`min_pool_size` default 1 and `max_pool_size` default 8 at lines 48 and 49, matching IF-02's `POSTGRES_MIN_POOL_SIZE=1` and `POSTGRES_MAX_POOL_SIZE=8`)
- Test: `tests/solace_infrastructure/test_mvp_140_pool_defaults.py`, `tests/migrations/test_mvp_140_pool_budget_real_pg.py`, `tests/solace_infrastructure/test_postgres.py` (its default assertions at lines 29 and 30 change from 5 and 20 to 1 and 8)
**Interfaces:**
- Consumes: the one-pool-per-service rule (WS-02.6), `solace_infrastructure.runtime.is_strict_environment` (IF-01, WS-04, landed in W0).
- Produces: nine services at the defaults hold 9 idle and at most 72 busy connections against the stock `max_connections=100`, leaving room for the migrate job and ops sessions. `postgres._is_strict_environment` is proven equal to the runtime's `is_strict_environment` for every ENVIRONMENT value (kept private to avoid an import cycle between the runtime and the pool).

- [ ] **Step 1: Write the failing test** in `tests/solace_infrastructure/test_mvp_140_pool_defaults.py` and `tests/migrations/test_mvp_140_pool_budget_real_pg.py`
```python
# tests/solace_infrastructure/test_mvp_140_pool_defaults.py
"""MVP-140: default pool sizes fit a stock Postgres; the pool's strictness rule matches the runtime's."""
from __future__ import annotations

import os

import pytest

from solace_infrastructure.postgres import PostgresSettings


def test_mvp_140_default_pool_sizes(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    for key in ("POSTGRES_MIN_POOL_SIZE", "POSTGRES_MAX_POOL_SIZE"):
        monkeypatch.delenv(key, raising=False)
    s = PostgresSettings(password="synthetic-pw")
    assert (s.min_pool_size, s.max_pool_size) == (1, 8)


@pytest.mark.parametrize("env", ["development", "test", "staging", "production", "Production", "STAGING", ""])
def test_mvp_140_pool_strictness_matches_runtime(env: str) -> None:
    from solace_infrastructure.postgres import _is_strict_environment
    from solace_infrastructure.runtime import is_strict_environment

    assert _is_strict_environment(env) == is_strict_environment(env)
```
```python
# tests/migrations/test_mvp_140_pool_budget_real_pg.py
"""MVP-140: nine service pools at the defaults fit a stock max_connections with headroom."""
from __future__ import annotations

import asyncpg
import pytest

from solace_infrastructure.postgres import PostgresSettings


@pytest.mark.realinfra
async def test_mvp_140_nine_service_pools_fit_stock_max_connections(pg_server, migrated_database, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    for key in ("POSTGRES_MIN_POOL_SIZE", "POSTGRES_MAX_POOL_SIZE"):
        monkeypatch.delenv(key, raising=False)
    s = PostgresSettings(host=pg_server.host, port=pg_server.port, user="solace_app",
                         password=pg_server.app_password, database=migrated_database)
    count_sql = "SELECT count(*) FROM pg_stat_activity WHERE usename = 'solace_app' AND datname = $1"
    pools, held = [], []
    su = await asyncpg.connect(pg_server.superuser_dsn(migrated_database))
    try:
        for _ in range(9):
            pools.append(await asyncpg.create_pool(dsn=s.get_dsn(), min_size=s.min_pool_size, max_size=s.max_pool_size))
        assert await su.fetchval(count_sql, migrated_database) == 9
        for pool in pools:
            for _ in range(pool.get_max_size()):
                held.append((pool, await pool.acquire()))
        busy = await su.fetchval(count_sql, migrated_database)
        limit = int(await su.fetchval("SHOW max_connections")) - int(await su.fetchval("SHOW superuser_reserved_connections"))
        assert busy == 72 and busy + 10 <= limit
        extra = await asyncpg.connect(pg_server.migrator_dsn(migrated_database))  # room for migrate and ops
        await extra.close()
    finally:
        for pool, conn in held:
            await pool.release(conn)
        for pool in pools:
            await pool.close()
        await su.close()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_infrastructure\test_mvp_140_pool_defaults.py -o "addopts=" -p no:logging -q`, then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations\test_mvp_140_pool_budget_real_pg.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The defaults test fails with `(5, 20) == (1, 8)`. The budget test fails at the idle count (`45 == 9`); saturating the pools then raises `asyncpg.exceptions.TooManyConnectionsError: sorry, too many clients already` long before 180 connections. The strictness parity test passes if WS-04's runtime has landed; it guards the private helper.
- [ ] **Step 3: Implement** the two default changes and update `tests/solace_infrastructure/test_postgres.py` lines 29 and 30 (open it with the Read and Edit tools).
- [ ] **Step 4: Run it and confirm it passes**
Run: both commands from Step 2. Expected: PASS
- [ ] **Step 5: Regression:** standard command for `D:\Repo\Contextual-Chatbot\tests\solace_infrastructure` and `D:\Repo\Contextual-Chatbot\services\user-service\tests`, and the real-infra command for `D:\Repo\Contextual-Chatbot\tests\migrations`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-02.9: A leak detector that actually runs, and query timeouts reported as query timeouts
**Covers:** MVP-272, MVP-274
**Severity:** low
**Wave:** 2
**Phase:** A
**Depends on:** WS-02.8
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `src/solace_infrastructure/database/connection_manager.py` (`_leak_detection_enabled` defaults to False; `enable_leak_detection()` starts the checker task even on the first call; new class attribute `LEAK_CHECK_INTERVAL_SECONDS = 10.0`, used by `_check_for_connection_leaks`; `_get_stack_trace` runs only when detection is enabled; `acquire` enters the pool's acquire context manager explicitly so only its `__aenter__` sits inside the `asyncio.TimeoutError` handler; a `TimeoutError` raised inside the caller's block propagates unchanged and does not increment `connection_timeouts`; the module docstring says leak detection is opt-in via `enable_leak_detection()`)
- Test: `tests/solace_infrastructure/test_mvp_272_leak_detection.py`, `tests/solace_infrastructure/test_mvp_274_acquire_timeouts.py`, `tests/solace_infrastructure/database/test_rls_guc.py` (keep its fakes working with the explicit context; behaviour assertions unchanged)
**Interfaces:**
- Consumes: the `acquire()` GUC behaviour from WS-02.6.
- Produces: `ConnectionPoolManager.enable_leak_detection()` and `.disable_leak_detection()` start and stop a real checker that logs `potential_connection_leak` for connections held longer than `CONNECTION_LEAK_THRESHOLD_SECONDS`; no stack capture and no CPU cost on the hot path when off.

- [ ] **Step 1: Write the failing test** in `tests/solace_infrastructure/test_mvp_272_leak_detection.py` and `tests/solace_infrastructure/test_mvp_274_acquire_timeouts.py`
```python
# tests/solace_infrastructure/test_mvp_272_leak_detection.py
"""MVP-272: no stack walk per acquire when leak detection is off; when on, the checker runs and warns."""
from __future__ import annotations

import asyncio

import pytest
from structlog.testing import capture_logs

from solace_infrastructure.database.connection_manager import ConnectionPoolManager as CPM


class _Conn:
    async def execute(self, *a):
        return "OK"


class _CM:
    async def __aenter__(self):
        return _Conn()

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def acquire(self, timeout=None):
        return _CM()

    def get_size(self):
        return 1

    def get_idle_size(self):
        return 0


@pytest.fixture
def fake_pool(monkeypatch):
    async def _get_pool(name="default"):
        return _Pool()

    monkeypatch.setattr(CPM, "get_pool", _get_pool)
    yield
    CPM.disable_leak_detection()


async def test_mvp_272_acquire_skips_stack_capture_when_detection_is_off(fake_pool, monkeypatch) -> None:
    def _boom():
        raise AssertionError("stack captured on the hot path")

    monkeypatch.setattr(CPM, "_get_stack_trace", classmethod(lambda cls: _boom()))
    async with CPM.acquire():
        pass


async def test_mvp_272_enabled_checker_warns_about_a_held_connection(fake_pool, monkeypatch) -> None:
    monkeypatch.setattr(CPM, "CONNECTION_LEAK_THRESHOLD_SECONDS", 0.05)
    monkeypatch.setattr(CPM, "LEAK_CHECK_INTERVAL_SECONDS", 0.02)
    CPM.enable_leak_detection()
    assert CPM._leak_check_task is not None and not CPM._leak_check_task.done()
    with capture_logs() as logs:
        async with CPM.acquire():
            await asyncio.sleep(0.3)
    assert any(e["event"] == "potential_connection_leak" for e in logs)
```
```python
# tests/solace_infrastructure/test_mvp_274_acquire_timeouts.py
"""MVP-274: a timeout inside the caller's block is not reported as pool exhaustion."""
from __future__ import annotations

import asyncio

import pytest

from solace_common.exceptions import DatabaseError
from solace_infrastructure.database.connection_manager import ConnectionPoolManager as CPM


class _Conn:
    async def execute(self, *a):
        return "OK"


class _CM:
    def __init__(self, fail: bool) -> None:
        self._fail = fail

    async def __aenter__(self):
        if self._fail:
            raise asyncio.TimeoutError()
        return _Conn()

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def __init__(self, fail_acquire: bool) -> None:
        self._fail = fail_acquire

    def acquire(self, timeout=None):
        return _CM(self._fail)

    def get_size(self):
        return 1

    def get_idle_size(self):
        return 0


def _use(monkeypatch, pool) -> None:
    async def _get_pool(name="default"):
        return pool

    monkeypatch.setattr(CPM, "get_pool", _get_pool)
    CPM.reset_metrics("mvp274")


async def test_mvp_274_query_timeout_inside_the_block_propagates_unchanged(monkeypatch) -> None:
    _use(monkeypatch, _Pool(fail_acquire=False))
    with pytest.raises(asyncio.TimeoutError):
        async with CPM.acquire("mvp274"):
            raise asyncio.TimeoutError()  # e.g. asyncpg command_timeout
    assert CPM.get_metrics("mvp274").connection_timeouts == 0


async def test_mvp_274_pool_acquire_timeout_still_maps_to_database_error(monkeypatch) -> None:
    _use(monkeypatch, _Pool(fail_acquire=True))
    with pytest.raises(DatabaseError, match="Timeout acquiring connection"):
        async with CPM.acquire("mvp274"):
            pass
    assert CPM.get_metrics("mvp274").connection_timeouts == 1
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_infrastructure\test_mvp_272_leak_detection.py D:\Repo\Contextual-Chatbot\tests\solace_infrastructure\test_mvp_274_acquire_timeouts.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The hot-path test raises `AssertionError: stack captured on the hot path` (`_leak_detection_enabled` is True at line 136, so line 343 walks the stack); the checker test fails on `_leak_check_task is not None` (`enable_leak_detection` returns at lines 588 and 589 because the flag is already True); the query-timeout test fails with `DatabaseError: Timeout acquiring connection from pool 'mvp274'` and `connection_timeouts == 1` (the body's timeout is caught by the handler at lines 374 to 387). The pool-timeout test passes today and guards the change.
- [ ] **Step 3: Implement** the changes listed under Files. `reset_metrics` keeps its signature; the explicit `__aenter__`/`__aexit__` pair passes the body's exception to `__aexit__` so asyncpg still releases the connection.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** standard command, one process each, for `D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`, `D:\Repo\Contextual-Chatbot\services\user-service\tests`, `D:\Repo\Contextual-Chatbot\services\therapy_service\tests` and `D:\Repo\Contextual-Chatbot\services\diagnosis_service\tests`; real-infra command for `D:\Repo\Contextual-Chatbot\tests\migrations`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-02.9:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| Id | Owner WS | Exact change | Files | Needed by |
|---|---|---|---|---|
| R-02.1 | WS-01 | Lock pins for alembic 1.16.2, sqlalchemy[asyncio] 2.0.42, asyncpg 0.30.0, psycopg[binary], cryptography, testcontainers 4.14.2 and docker 7.1.0, installed locally per D-20 (WS-01.2); the `solace-migrate` image from `docker/migrate.Dockerfile` carrying `alembic.ini`, `migrations/`, `scripts/db/` and `src/` with CMD `python -m scripts.db.migrate upgrade head` and the NAME_FILE entrypoint (WS-01.9). | `requirements.in`, `requirements.lock`, `docker/migrate.Dockerfile` | WS-02.1, WS-02.4 |
| R-02.2 | WS-03 | W0 phase A2 env sync. Postgres service: image `scripts.db.images.POSTGRES_IMAGE`; `POSTGRES_USER=postgres`, `POSTGRES_PASSWORD=${POSTGRES_SUPERUSER_PASSWORD}`, `POSTGRES_DB=solace`, `SOLACE_APP_PASSWORD=${POSTGRES_PASSWORD}`, `SOLACE_MIGRATOR_PASSWORD=${POSTGRES_MIGRATOR_PASSWORD}` (W3: the `_FILE` forms), `SOLACE_PG_TLS` (`optional` in dev, `required` in strict), `SOLACE_PG_TLS_DIR=/run/secrets`; mount `./infrastructure/init-db` read-only on `/docker-entrypoint-initdb.d` (unchanged), `./scripts/db/pg_tls_entrypoint.sh` read-only on `/usr/local/bin/solace-pg-entrypoint.sh` with `entrypoint: ["sh", "/usr/local/bin/solace-pg-entrypoint.sh"]` and `command: ["postgres"]`, and the three files from `scripts/db/gen_certs.py --out <generated dir> --san postgres` on `/run/secrets`; healthcheck `pg_isready -U postgres -d solace`. A `migrate` one-shot with `DATABASE_URL=postgresql://solace_migrator:${POSTGRES_MIGRATOR_PASSWORD}@postgres:5432/solace`, `ENVIRONMENT`, `POSTGRES_SSL_MODE` and `POSTGRES_SSL_ROOT_CERT`, depending on postgres healthy; every DB-using app service depends on it with `service_completed_successfully` and gets `POSTGRES_HOST=postgres`, `POSTGRES_PORT=5432`, `POSTGRES_DATABASE=solace`, `POSTGRES_USER=solace_app`, `POSTGRES_PASSWORD`, `POSTGRES_SSL_MODE` (verify-full strict, prefer dev), `POSTGRES_SSL_ROOT_CERT=/run/secrets/postgres-ca.crt`, `POSTGRES_MIN_POOL_SIZE=1`, `POSTGRES_MAX_POOL_SIZE=8`. Registry entries for the four new names (`SOLACE_APP_PASSWORD` and `SOLACE_MIGRATOR_PASSWORD`: secret, postgres container only; `SOLACE_PG_TLS`, `SOLACE_PG_TLS_DIR`: not secret). Runbook: `make migrate` replaces the manual `alembic upgrade head`. | `docker-compose.yml`, `docker-compose.prod.yml`, `deploy/env/env-contract.yaml` | WS-02.4, WS-02.5, WS-02.6 |
| R-02.3 | WS-04 | `runtime/datastores.py`: when `spec.uses_postgres`, `pool = await ConnectionPoolManager.get_pool("default")`, then in strict `await assert_schema_head(pool)`; `DatabaseConfigError` and `SchemaVersionMismatch` become `StartupDependencyError` in strict and a failed critical `postgres` readiness check in development; readiness runs `SELECT 1` through the pool; `stop_service_runtime` calls `ConnectionPoolManager.close_all_pools()`. `ProductionGuards` treats any `POSTGRES_SSL_MODE` other than `verify-full`, or an unset `POSTGRES_SSL_ROOT_CERT`, as a strict-environment violation, matching WS-02.5. | `src/solace_infrastructure/runtime/datastores.py`, `src/solace_security/production_guards.py` | WS-02.3, WS-02.6 |
| R-02.4 | WS-05 | `pg_migrated` (IF-19) starts `scripts.db.images.POSTGRES_IMAGE` with `infrastructure/init-db` mounted and generated `SOLACE_APP_PASSWORD`/`SOLACE_MIGRATOR_PASSWORD`, migrates with `scripts.db.migrate.run_migrations(dsn_migrator, "upgrade", "head")` (or a given target for `create_database(..., migrate_to=)`), and builds `app_pool` as `solace_app`; `docker-compose.test.yml` generates per-run TLS files with `scripts/db/gen_certs.py --san postgres` for the strict profile. | `src/solace_testing/realinfra.py`, `docker-compose.test.yml` | WS-02.4, WS-02.5 |
| R-02.5 | WS-06 | Slot `010_event_plane` (scaffolded empty by WS-02.3, `down_revision = "009_platform_roles_rls"`) creates `event_outbox` with `claimed_at` and the partial index `idx_outbox_publishing_claimed ON event_outbox (claimed_at) WHERE status = 'PUBLISHING'`, plus `processed_events` and `event_dead_letter`, because 007 is now a guarded no-op on fresh databases; `ensure_table()` verifies only (solace_app has no DDL). | `migrations/versions/010_event_plane.py` | WS-02.2 |
| R-02.6 | WS-09 | Slot `011_audit_trail` revokes `UPDATE, DELETE, TRUNCATE ON audit_events FROM solace_app` after creating it (009's default privileges grant all DML); `tests/migrations/test_chain_real_pg.py::test_mvp_041_audit_events_is_append_only_when_present` checks it. | `migrations/versions/011_audit_trail.py` | WS-02.3 |
| R-02.7 | WS-10 | Slot `017_orchestrator_checkpoints`: tables get DML for `solace_app` through 009's default privileges; any RLS policy uses `NULLIF(current_setting('app.current_user_id', true), '')::uuid` (the chain test enforces it for every slot); `CONCURRENTLY` statements inside `op.get_context().autocommit_block()` work with `transaction_per_migration=True`. | `migrations/versions/017_orchestrator_checkpoints.py` | WS-02.3 |
| R-02.8 | WS-14 | W0 phase B: the registration `user_preferences` insert runs inside `acting_as_user(user_id)` (with NULLIF policies a context-less insert is rejected by WITH CHECK, as it should be); user-service stops building its separate `USER_DB_*` `PostgresClient` and uses `runtime.pg_pool`, so one pool per service holds. | `services/user-service/src/domain/service.py`, `services/user-service/src/main.py` | WS-02.3 (MVP-058 closure), WS-02.6 (MVP-033 closure), WS-02.8 |
| R-02.9 | WS-11 | W0 phase B: `services/orchestrator_service/src/infrastructure/state.py` stops opening a raw-DSN asyncpg pool without TLS; it uses `runtime.pg_pool` or `PostgresSettings.get_ssl_context()` (MVP-271 bypass). | `services/orchestrator_service/src/infrastructure/state.py` | WS-02.5 |
| R-02.10 | WS-12 | W0 phase B: `services/safety_service/src/infrastructure/database.py` stops calling `asyncpg.create_pool` without `ssl`; it uses `runtime.pg_pool` (MVP-271 bypass). | `services/safety_service/src/infrastructure/database.py` | WS-02.5 |
| R-02.11 | WS-17 | W0 phase B: the memory SQLAlchemy engine in `postgres_repo.py` either moves to the shared pool or passes `connect_args={"ssl": build_ssl_context(...)}`, and its default database becomes `solace` instead of `solace_memory` (MVP-271, MVP-249, MVP-140). | `services/memory_service/src/infrastructure/postgres_repo.py` | WS-02.5, WS-02.8 |
| R-02.12 | WS-05 | (a) `INIT_DB_ENV` (WS-05 plan, WS-05.2) uses the names init-db reads: `{"superuser": "POSTGRES_PASSWORD", "migrator": "SOLACE_MIGRATOR_PASSWORD", "app": "SOLACE_APP_PASSWORD"}`, not `POSTGRES_MIGRATOR_PASSWORD`/`POSTGRES_APP_PASSWORD` (WS-02.1; WS-03 compose interpolates the same names). (b) WS-05.2 adds `WS-02.1`, `WS-02.4` and `WS-02.6` to its Depends on (init-db roles, `run_migrations`, `init_connection`), and WS-05.4 adds `WS-02.5` (`gen_certs`). (c) `create_database` passes `init=init_connection` from `solace_infrastructure.database.connection_manager`. | `src/solace_testing/realinfra.py` | none in WS-02 (without (a) WS-05.2's postgres init fails closed on missing `SOLACE_*` passwords, per WS-02.1) |
| R-02.13 | WS-06 | WS-06.1 adds `WS-02.3` to its Depends on: it fills `migrations/versions/010_event_plane.py`, which WS-02.3 creates earlier in the same W0 phase-A2 lane (the `migrations/versions/*` handoff). | none (WS-06's plan) | WS-02.3 |

Requests received and planned here:
- WS-06's request (010 scaffold, 009 default privileges, 007 guard) is WS-02.3 and WS-02.2; WS-10.9's dependency on the 017 scaffold (`WS-02:slot-017-scaffold`) and WS-11.6's on the 016 scaffold are WS-02.3.
- WS-01's requests (env.py import root, the postgres.py `ssl` F821, lazy database exports, migrate.py and schema_check for the migrate image) are WS-02.2, WS-02.5, WS-02.7, WS-02.4 and WS-02.3.
- WS-05 request #2: the roles and single `solace` database are WS-02.1 (with the `SOLACE_*` init names; WS-05's `POSTGRES_APP_PASSWORD`/`POSTGRES_MIGRATOR_PASSWORD` naming is declined, see R-02.12); `migrate upgrade|downgrade <rev>` reading `DATABASE_URL` is WS-02.4; `SCHEMA_HEAD` is WS-02.3; `gen_certs --out` is WS-02.5; the public `init_connection(conn)` is new in WS-02.6.
- WS-03's prerequisites (migrate CLI with `DATABASE_URL`/`POSTGRES_SSL_*`, init-db `SOLACE_*` names, `gen_certs --out` with SAN `postgres`) are WS-02.4, WS-02.1 and WS-02.5.
- Not planned: WS-09 CR-11, the optional removal of `configure_phi_encryption`, `get_phi_encryptor` and the ClinicalBase listeners from `base_models.py`. It can only run after the W2 phase-B codec adoption, but WS-02's waves are W0 and W2, and a W2 phase-A task may not depend on W2 phase B. WS-09.15 does not need it: its lock scans only `services/`. The controller may schedule it by ledger Ruling after the W2 barrier, together with the unowned `tests/integration/test_phi_at_rest.py`.

## New interfaces

- Init environment on the postgres container: `SOLACE_APP_PASSWORD` / `SOLACE_APP_PASSWORD_FILE` and `SOLACE_MIGRATOR_PASSWORD` / `SOLACE_MIGRATOR_PASSWORD_FILE` (required, exclusive pairs); `SOLACE_PG_TLS` (`required` | `optional`, default `optional`) and `SOLACE_PG_TLS_DIR` (default `/run/secrets`) for `scripts/db/pg_tls_entrypoint.sh`.
- Role attributes refining IF-03: `solace_migrator` also has BYPASSRLS, NOCREATEDB and NOCREATEROLE (so data migrations see every row under FORCE RLS); services never connect as it (`verify_app_role`).
- `scripts.db.images.POSTGRES_IMAGE`; `scripts.db.migrate` CLI and API (`run_migrations`, `migrate_async`, `resolve_database_url`, `redact`); `DATABASE_URL` takes precedence, with a `POSTGRES_*` fallback built by `PostgresSettings.get_dsn()`; exit codes 0, 1 and 2; output `migrated: <version or base>`.
- `scripts.db.gen_certs` CLI and `main(argv)`; file names follow IF-22 (`postgres-ca.crt`, `server.crt`, `server.key`); no CA key is kept, so rotation regenerates all three files.
- `solace_infrastructure.postgres`: `DatabaseConfigError(DatabaseError)`, `effective_ssl_mode`, `build_ssl_context`, `PostgresSettings.validate_for_environment`, private `_is_strict_environment` (parity with IF-01 proven in WS-02.8); defaults `min_pool_size=1`, `max_pool_size=8` from W2.
- `solace_infrastructure.database.connection_manager`: `DEFAULT_POOL`, `LEGACY_POOL_ALIASES`, `canonical_pool_name`, `verify_app_role`, `init_connection` (public json/jsonb codec helper, WS-05's request); strict pool creation refuses superuser, BYPASSRLS and table-owner roles; a failed RLS reset terminates the connection; leak detection is opt-in (W2).
- `solace_infrastructure.database.schema_check`: `read_schema_version`, `SchemaVersionMismatch.expected` and `.found`, `assert_schema_head(..., environment=None) -> str | None`.
- Slot-owner convention: every policy in a slot uses the NULLIF predicate; `tests/migrations/test_chain_real_pg.py` enforces it and the audit append-only rule once `audit_events` exists.

## Self-review

Coverage (register id -> tasks):

| Id | Sev | Tasks |
|---|---|---|
| MVP-001 | blocker | WS-02.2 (chain applies and round-trips), WS-02.4 (applied by the migrate entrypoint; compose job via R-02.2) |
| MVP-009 | blocker | WS-02.2 |
| MVP-010 | blocker | WS-02.5 (compose TLS wiring via R-02.2) |
| MVP-033 | blocker | WS-02.6 (compose `POSTGRES_*` via R-02.2, readiness via R-02.3, user-service pool via R-02.8) |
| MVP-041 | high | WS-02.1 (roles), WS-02.3 (grants, no DDL), WS-02.6 (runtime refusal of privileged roles) |
| MVP-058 | high | WS-02.3 (NULLIF policies, isolation), WS-02.6 (terminate on failed reset; registration via R-02.8) |
| MVP-140 | medium | WS-02.8 (one pool per service from WS-02.6; extra pools removed via R-02.8 and R-02.11) |
| MVP-273 | medium | WS-02.6 |
| MVP-249 | low | WS-02.1 |
| MVP-271 | low | WS-02.5 (bypassing pools via R-02.9, R-02.10, R-02.11) |
| MVP-272 | low | WS-02.9 |
| MVP-274 | low | WS-02.9 |
| MVP-269 (WS-01's) | low | WS-02.7, supporting |

Interface consistency: IF-03 is produced as written (role attributes, NULLIF predicate, `get_pool('default')` with aliases, strict `DatabaseConfigError`, percent-encoded DSNs, `transaction_per_migration=True`, the migrate command and advisory lock, `SCHEMA_HEAD = '021_personality_analytics'`, the slot ids and owners, and the chain test). The additions are listed under New interfaces. IF-02's names are used unchanged (`POSTGRES_*`, `DATABASE_URL` for the migrate job, `POSTGRES_SUPERUSER_PASSWORD` and `POSTGRES_MIGRATOR_PASSWORD` for compose interpolation); the four `SOLACE_*` names on the postgres container are new and requested from WS-03. IF-22's TLS file names and NAME_FILE rule are followed. WS-01 consistency: WS-02.1 depends on WS-01.2, WS-02.2 and WS-02.5 on WS-01.7 (their guard cases), WS-02.7 on WS-01.3; WS-01.9 depends on WS-02.3 and WS-02.4; both plans list the same requests in both directions (R-01.13 and R-02.1).

Plan-checker notes:
- `migrations/versions/010_event_plane.py` appears in WS-02.3 (create) and WS-06.1 (modify) in wave 0. This is the designed `migrations/versions/*` handoff, with WS-02.3 first in the lane. The missing dependency edge WS-06.1 -> WS-02.3 is requested as R-02.13.
- WS-02.3 creating slots 011 to 021, which IF-03 assigns to WS-09, WS-14, WS-08, WS-12, WS-13, WS-11, WS-10, WS-17, WS-15, WS-16 and WS-18, is the step the skeleton's shared_files `migrations/versions/*` protocol gives WS-02. WS-02.3 writes only the scaffold (revision ids, docstring owner, empty bodies), and each owner edits only `upgrade()`/`downgrade()` in its own later slot task.
- `alembic.ini` has one W0 editor, WS-02.2. WS-01.9 only copies it into the migrate image.

REVIEW_FOCUS: 3=WS-02.3,WS-02.6; 5=WS-02.3,WS-02.5,WS-02.6

Open risks:
- Running every service as `solace_app` from W0 will expose context-less queries on RLS tables in consumers, CLIs and registration. They fail closed (no rows, rejected writes) and show up as empty reads unless realinfra tests cover them. R-02.8 covers registration; other owners rely on `acting_as_user`.
- The TLS wrapper and init script are shell files bind-mounted from a Windows working tree (`core.autocrlf=true`). A CRLF checkout breaks them. They are written with LF, and the git-archive build normalizes them, but a developer re-checkout could reintroduce CRLF; WS-02.1 and WS-02.5's real-infra tests catch it immediately.
- Slot bodies may be amended in place until the first real deployment (shared-file protocol). A database migrated before an amendment keeps the same head id with different DDL; only fresh databases are guaranteed to match.
- `verify_app_role` runs one extra query when a strict pool is created, and refuses any role that owns a table. An operator who points a service at the migrator credential gets a startup failure by design.
- The stock `max_connections=100` budget assumes one pool per service. Each removal of an extra pool (user-service, memory, orchestrator state) is another workstream's W0 task; until they land, the budget test passes but a live stack can still exceed it.
- WS-05's plan currently starts init-db with `POSTGRES_APP_PASSWORD`/`POSTGRES_MIGRATOR_PASSWORD`. Until R-02.12 lands, its fixture fails closed at init (by design of WS-02.1), so the mismatch is loud, not silent.

## Review responses

- Error: W0 `010_event_plane.py` edited by WS-02.3 and WS-06.1 without a dependency: fixed. WS-02.3's Produces names WS-06.1. R-02.13 asks WS-06 to add `WS-02.3` to WS-06.1's Depends on, which only WS-06's plan can change. The edit itself is the skeleton's `migrations/versions/*` handoff, and the Self-review explains it.
- Error: plan WS-05's request #2 (init-db roles, migrate CLI, SCHEMA_HEAD, gen_certs, public codec helper): fixed. Most items were already planned and are now named in Produces lines: WS-02.1 roles and the single DB, WS-02.4 CLI, WS-02.3 `SCHEMA_HEAD`, WS-02.5 `gen_certs`. The new `init_connection(conn)` goes into WS-02.6, with the RED test `test_mvp_033_init_connection_is_the_public_pool_codec`. The env-var naming part is rejected. init-db reads `SOLACE_APP_PASSWORD`/`SOLACE_MIGRATOR_PASSWORD` (WS-02.1 Produces; New interfaces). WS-03 already interpolates those names (WS-03 plan: WS-03.2 prerequisites and Produces, and its New interfaces). The skeleton's IF-02 has no `POSTGRES_APP_PASSWORD`. WS-05 is asked to align (R-02.12).
- Error: plan WS-09's CR-11 cleanup (needed by WS-09.15): rejected. WS-09 itself marks CR-11 "Optional ... after W2" with needed-by "none". WS-09.15's lock (`test_mvp_055_no_service_uses_the_retired_orm_listener_path`) scans only `services/` and passes without the deletion. The removal must follow the W2 phase-B codec adoption, ten service `main.py` files still reference `configure_phi_encryption` today (grep over `services/`), and WS-02's waves (W0, W2 phase A) cannot depend on W2 phase B. `tests/integration/test_phi_at_rest.py` is not WS-02-owned. It is recorded under "Not planned" for a controller Ruling after W2.
- Warning: WS-02.3 edits `014_safety.py` (WS-12): rejected. WS-02.3 creates the scaffold, which the skeleton's shared_files `migrations/versions/*` protocol sanctions ("WS-02 ... creates no-op slot files 010-021"); WS-12 then edits only the bodies.
- Warning: WS-02.3 edits `018_memory.py` (WS-17): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `010_event_plane.py` (WS-06): rejected: same protocol. The same-wave edge is handled by R-02.13.
- Warning: WS-02.3 edits `013_identity_mfa.py` (WS-08): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `021_personality_analytics.py` (WS-18): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `020_diagnosis.py` (WS-16): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `016_orchestrator_threads.py` (WS-11): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `017_orchestrator_checkpoints.py` (WS-10): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `015_notification.py` (WS-13): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `012_user_accounts.py` (WS-14): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `019_therapy.py` (WS-15): rejected: same shared_files protocol (scaffold only).
- Warning: WS-02.3 edits `011_audit_trail.py` (WS-09): rejected: same shared_files protocol (scaffold only).
- Warning: W0 `alembic.ini` edited by WS-01.9 and WS-02.2: rejected. WS-01.9 only uses it as a COPY source of `docker/migrate.Dockerfile`, and the WS-01 plan now words it that way. WS-02.2 is the sole editor, and WS-01.9 depends on WS-02.4, which comes after WS-02.2.
- AGY [error/RED] WS-02.7 `len(d.__all__) == 50`: fixed. Verified that `src/solace_infrastructure/database/__init__.py:115-181` holds 58 unique names. The assertion is now `== 58`, so the export test really passes today, as Step 2 states.
