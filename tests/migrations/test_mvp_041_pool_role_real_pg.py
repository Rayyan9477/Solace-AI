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


# --- WS-02.6 additions -------------------------------------------------------------------------------------------
import asyncio  # noqa: E402
import secrets  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

from solace_common.request_context import reset_current_user_id, set_current_user_id  # noqa: E402
from solace_infrastructure.database.connection_manager import (  # noqa: E402
    ConnectionPoolManager,
    check_new_pool_role,
    verify_app_role,
)
from solace_infrastructure.postgres import DatabaseConfigError  # noqa: E402


async def test_ws026_a_refused_real_pool_is_closed_and_the_app_pool_kept(pg_server, migrated_database) -> None:
    refused = await asyncpg.create_pool(pg_server.migrator_dsn(migrated_database), min_size=1, max_size=2)
    try:
        with pytest.raises(DatabaseConfigError, match="has BYPASSRLS") as exc:
            await check_new_pool_role(refused, environment="production")
        assert "owns tables in schema public" in str(exc.value)
        assert "session_user 'solace_migrator'" in str(exc.value)  # the problem and the role names (fix round 1)
        assert pg_server.migrator_password not in str(exc.value)
        assert refused.is_closing()
    finally:
        refused.terminate()  # a no-op once closed
    app = await asyncpg.create_pool(pg_server.app_dsn(migrated_database), min_size=1, max_size=2)
    try:
        await check_new_pool_role(app, environment="production")
        assert not app.is_closing() and await app.fetchval("SELECT current_user") == "solace_app"
    finally:
        await app.close()


async def test_ws026_inherited_ownership_and_privileged_membership_are_refused(pg_server, migrated_database) -> None:
    """A plain role that owns one table, and a plain role that is a member of solace_migrator (inheriting its
    ownership and able to SET ROLE to a BYPASSRLS role), are both refused; each message names only its reasons."""
    suffix = uuid.uuid4().hex[:10]
    owner, member = f"ws026_owner_{suffix}", f"ws026_member_{suffix}"
    passwords = {owner: secrets.token_urlsafe(18), member: secrets.token_urlsafe(18)}
    admin = await asyncpg.connect(pg_server.superuser_dsn(migrated_database))
    try:
        await admin.execute("SET log_statement = 'none'")
        for role, password in passwords.items():
            await admin.execute(f"CREATE ROLE {role} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{password}'")
            await admin.execute(f'GRANT CONNECT ON DATABASE "{migrated_database}" TO {role}')
        await admin.execute(f"CREATE TABLE public.ws026_owned_{suffix} (id int)")
        await admin.execute(f"ALTER TABLE public.ws026_owned_{suffix} OWNER TO {owner}")
        await admin.execute(f"GRANT solace_migrator TO {member}")
        cases = (
            (owner, ["owns tables in schema public"], ["is a superuser", "has BYPASSRLS", "is a member"]),
            (member, ["owns tables in schema public", "is a member of a superuser or BYPASSRLS role"],
             ["is a superuser", "has BYPASSRLS"]),
        )
        for role, expected, absent in cases:
            conn = await asyncpg.connect(pg_server.dsn(role, passwords[role], migrated_database))
            try:
                with pytest.raises(DatabaseConfigError) as exc:
                    await verify_app_role(conn, environment="staging")
                message = str(exc.value)
                assert all(text in message for text in expected), message
                assert not any(text in message for text in absent), message
                assert f"session_user '{role}'" in message and passwords[role] not in message
            finally:
                await conn.close()
    finally:
        await _drop_roles(admin, owner, member)
        await admin.close()


async def _drop_roles(admin, *roles: str) -> None:
    """DROP OWNED BY (objects in this database and the CONNECT grants) and DROP ROLE, for those that exist."""
    existing = [role for role in roles if await admin.fetchval("SELECT 1 FROM pg_roles WHERE rolname = $1", role)]
    if existing:
        await admin.execute(f"DROP OWNED BY {', '.join(existing)}")
        await admin.execute(f"DROP ROLE {', '.join(existing)}")


async def test_ws026_a_superuser_switched_to_the_app_role_at_login_is_refused(pg_server, migrated_database) -> None:
    """WS026-SEC-1: ``ALTER ROLE <superuser> SET role = solace_app`` makes current_user solace_app at login while
    the session stays the superuser's (any ``SET ROLE NONE`` gets it back). A strict pool built on that login is
    refused and closed."""
    probe = f"ws026_su_{uuid.uuid4().hex[:10]}"
    password = secrets.token_urlsafe(18)
    admin = await asyncpg.connect(pg_server.superuser_dsn(migrated_database))
    try:
        await admin.execute("SET log_statement = 'none'")
        await admin.execute(f"CREATE ROLE {probe} LOGIN SUPERUSER PASSWORD '{password}'")
        await admin.execute(f"ALTER ROLE {probe} SET role = 'solace_app'")
        dsn = pg_server.dsn(probe, password, migrated_database)
        conn = await asyncpg.connect(dsn)
        try:  # the precondition: the old current_user-only check saw a plain app role
            assert tuple(await conn.fetchrow("SELECT session_user, current_user")) == (probe, "solace_app")
            await conn.execute("SET ROLE NONE")
            assert await conn.fetchval("SELECT current_user") == probe  # ...and the superuser is one statement away
        finally:
            await conn.close()
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
        try:
            with pytest.raises(DatabaseConfigError) as exc:
                await check_new_pool_role(pool, environment="production")
            message = str(exc.value)
            assert "switched to another role at login" in message and "is a superuser" in message
            assert f"session_user '{probe}'" in message and "current_user 'solace_app'" in message
            assert password not in message
            assert pool.is_closing()
        finally:
            pool.terminate()  # a no-op once closed
    finally:
        await _drop_roles(admin, probe)
        await admin.close()


async def test_ws026_superuser_equivalent_membership_is_refused(pg_server, migrated_database) -> None:
    """WS026F1-CLO-1: a member of pg_execute_server_program (COPY ... PROGRAM) or of pg_read_server_files /
    pg_write_server_files reads or writes anything the server can reach, whatever RLS says. Refused for a direct
    grant (pool closed) and through a chain whose link has INHERIT FALSE."""
    suffix = uuid.uuid4().hex[:10]
    direct, chained, link = f"ws026_exec_{suffix}", f"ws026_chain_{suffix}", f"ws026_link_{suffix}"
    files = f"ws026_files_{suffix}"
    passwords = {role: secrets.token_urlsafe(18) for role in (direct, chained, files)}
    admin = await asyncpg.connect(pg_server.superuser_dsn(migrated_database))
    try:
        await admin.execute("SET log_statement = 'none'")
        for role, password in passwords.items():
            await admin.execute(f"CREATE ROLE {role} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{password}'")
            await admin.execute(f'GRANT CONNECT ON DATABASE "{migrated_database}" TO {role}')
        await admin.execute(f"CREATE ROLE {link} NOLOGIN NOSUPERUSER NOBYPASSRLS")
        await admin.execute(f"GRANT pg_execute_server_program TO {direct}")
        await admin.execute(f"GRANT pg_execute_server_program TO {link}")
        await admin.execute(f"GRANT {link} TO {chained} WITH INHERIT FALSE")
        await admin.execute(f"GRANT pg_read_server_files, pg_write_server_files TO {files}")

        pool = await asyncpg.create_pool(pg_server.dsn(direct, passwords[direct], migrated_database),
                                         min_size=1, max_size=2)
        try:
            with pytest.raises(DatabaseConfigError) as exc:
                await check_new_pool_role(pool, environment="production")
            message = str(exc.value)
            assert "is a member of a superuser-equivalent role" in message
            assert f"session_user '{direct}'" in message and passwords[direct] not in message
            assert pool.is_closing()
        finally:
            pool.terminate()  # a no-op once closed

        for role in (chained, files):
            conn = await asyncpg.connect(pg_server.dsn(role, passwords[role], migrated_database))
            try:
                if role == chained:  # the precondition: nothing is inherited through the INHERIT FALSE link
                    assert await conn.fetchval("SELECT pg_has_role('pg_execute_server_program', 'USAGE')") is False
                    assert await conn.fetchval("SELECT pg_has_role('pg_execute_server_program', 'MEMBER')") is True
                with pytest.raises(DatabaseConfigError) as exc:
                    await verify_app_role(conn, environment="staging")
                message = str(exc.value)
                assert "is a member of a superuser-equivalent role" in message, message
                assert not any(text in message for text in ("is a superuser", "has BYPASSRLS", "owns tables")), message
                assert f"session_user '{role}'" in message and passwords[role] not in message
            finally:
                await conn.close()
    finally:
        await _drop_roles(admin, direct, chained, files, link)
        await admin.close()


async def test_ws026_a_replication_login_is_refused(pg_server, migrated_database) -> None:
    """Fix round 2 hardening (not an RLS bypass on this connection): the app login is NOREPLICATION, as
    scripts/db/migrate.py requires of the platform roles."""
    role, password = f"ws026_repl_{uuid.uuid4().hex[:10]}", secrets.token_urlsafe(18)
    admin = await asyncpg.connect(pg_server.superuser_dsn(migrated_database))
    try:
        await admin.execute("SET log_statement = 'none'")
        await admin.execute(f"CREATE ROLE {role} LOGIN REPLICATION NOSUPERUSER NOBYPASSRLS PASSWORD '{password}'")
        await admin.execute(f'GRANT CONNECT ON DATABASE "{migrated_database}" TO {role}')
        conn = await asyncpg.connect(pg_server.dsn(role, password, migrated_database))
        try:
            with pytest.raises(DatabaseConfigError, match="has REPLICATION"):
                await verify_app_role(conn, environment="production")
        finally:
            await conn.close()
    finally:
        await _drop_roles(admin, role)
        await admin.close()


async def test_ws026_noinherit_membership_in_a_table_owner_is_refused(pg_server, migrated_database) -> None:
    """WS026-SEC-1: a NOINHERIT member of a (non-BYPASSRLS) table owner has none of the owner's privileges until it
    runs SET ROLE, which it may; ownership is therefore checked with 'MEMBER', not 'USAGE'."""
    suffix = uuid.uuid4().hex[:10]
    owner, member = f"ws026_tblowner_{suffix}", f"ws026_noinherit_{suffix}"
    password = secrets.token_urlsafe(18)
    admin = await asyncpg.connect(pg_server.superuser_dsn(migrated_database))
    try:
        await admin.execute("SET log_statement = 'none'")
        await admin.execute(f"CREATE ROLE {owner} NOLOGIN NOSUPERUSER NOBYPASSRLS")
        await admin.execute(f"CREATE ROLE {member} LOGIN NOINHERIT NOSUPERUSER NOBYPASSRLS PASSWORD '{password}'")
        await admin.execute(f'GRANT CONNECT ON DATABASE "{migrated_database}" TO {member}')
        await admin.execute(f"CREATE TABLE public.ws026_owned_{suffix} (id int)")
        await admin.execute(f"ALTER TABLE public.ws026_owned_{suffix} OWNER TO {owner}")
        await admin.execute(f"GRANT {owner} TO {member}")
        conn = await asyncpg.connect(pg_server.dsn(member, password, migrated_database))
        try:
            assert await conn.fetchval(f"SELECT pg_has_role('{owner}', 'USAGE')") is False  # not inherited
            with pytest.raises(DatabaseConfigError) as exc:
                await verify_app_role(conn, environment="production")
            message = str(exc.value)
            assert "owns tables in schema public" in message
            assert not any(text in message for text in ("is a superuser", "has BYPASSRLS", "is a member", "switched"))
            assert f"session_user '{member}'" in message and password not in message
        finally:
            await conn.close()
    finally:
        await _drop_roles(admin, member, owner)
        await admin.close()


def _probe_settings(pg_server, database: str = "solace") -> PostgresSettings:
    return PostgresSettings(_env_file=None, host=pg_server.host, port=pg_server.port, user="solace_migrator",
                            password=pg_server.migrator_password, database=database, ssl_mode="disable",
                            min_pool_size=1, max_pool_size=2)


async def _forget_pool(name: str) -> None:
    await ConnectionPoolManager.close_pool(name)
    for registry in (ConnectionPoolManager._pool_configs, ConnectionPoolManager._locks,
                     ConnectionPoolManager._metrics):
        registry.pop(name, None)


async def test_ws026_the_pool_manager_connects_with_a_reserved_character_password(pg_server) -> None:
    name = f"ws026_encoded_{uuid.uuid4().hex[:8]}"
    try:
        await ConnectionPoolManager.register_pool(name, _probe_settings(pg_server))
        async with ConnectionPoolManager.acquire(name) as conn:
            assert await conn.fetchval("SELECT current_user") == "solace_migrator"
            assert await conn.fetchval("SELECT '{\"a\": 1}'::jsonb") == {"a": 1}  # init_connection's codec
    finally:
        await _forget_pool(name)


async def test_ws026_a_real_connection_whose_guc_reset_fails_is_terminated(pg_server) -> None:
    """MVP-058 on real asyncpg: the reset fails inside an aborted transaction, so the connection is terminated
    (its backend exits) and the next borrower gets a fresh connection with no user set."""
    name = f"ws026_reset_{uuid.uuid4().hex[:8]}"
    uid = "00000000-0000-0000-0000-0000000000a1"
    admin = await asyncpg.connect(pg_server.superuser_dsn())
    try:
        await ConnectionPoolManager.register_pool(name, _probe_settings(pg_server))
        token = set_current_user_id(uid)
        try:
            async with ConnectionPoolManager.acquire(name) as conn:
                pid = conn.get_server_pid()
                assert await conn.fetchval("SELECT current_setting('app.current_user_id', true)") == uid
                await conn.execute("BEGIN")
                with pytest.raises(asyncpg.DivisionByZeroError):
                    await conn.execute("SELECT 1/0")  # the transaction is now aborted: the reset will fail
        finally:
            reset_current_user_id(token)
        deadline = time.monotonic() + 10
        while await admin.fetchval("SELECT count(*) FROM pg_stat_activity WHERE pid = $1", pid):
            assert time.monotonic() < deadline, "the terminated connection's backend is still running"
            await asyncio.sleep(0.1)
        async with ConnectionPoolManager.acquire(name) as conn:
            assert conn.get_server_pid() != pid
            assert await conn.fetchval("SELECT current_setting('app.current_user_id', true)") in (None, "")
    finally:
        await _forget_pool(name)
        await admin.close()
