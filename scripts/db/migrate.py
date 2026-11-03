"""The migrate entrypoint (IF-03; MVP-001, MVP-041): the one way the schema changes.

    python -m scripts.db.migrate {upgrade,downgrade,current} [target]
        [--wait-timeout SECONDS]   (default 120)
        [--lock-timeout SECONDS]   (default 600)
        [--expected-role NAME]     (default solace_migrator)

Every step fails closed:

1. The database URL is ``DATABASE_URL`` (docker/entrypoint.sh also loads it from ``DATABASE_URL_FILE``), else a URL
   built from ``PostgresSettings`` (``POSTGRES_*``) when ``POSTGRES_PASSWORD`` is set. ``postgresql://``,
   ``postgres://`` and ``postgresql+asyncpg://`` URLs are accepted, without a query string or fragment (Alembic's
   SQLAlchemy connection would let query parameters override the user, host, port and database the preflight below
   checks on its asyncpg connection) and with an ``@`` in the password percent-encoded.
   TLS therefore comes only from POSTGRES_SSL_MODE and POSTGRES_SSL_ROOT_CERT: every connection, here and
   Alembic's in migrations/env.py, uses ``solace_infrastructure.postgres.build_ssl_context`` with them (MVP-010,
   MVP-271). In staging and production that is verify-full against the CA file POSTGRES_SSL_ROOT_CERT, and an
   unset or unreadable root certificate is a usage error before any login attempt.
2. Wait at most ``--wait-timeout`` seconds for the server to accept a login. A refused login, a missing database
   or a TLS failure (a certificate that does not verify, a server without TLS) ends the wait at once, because
   waiting cannot fix it.
3. Refuse unless ``current_user`` and ``session_user`` are the expected role, which is neither a superuser nor a
   member of one, and the platform roles exist with the attributes infrastructure/init-db gives them
   (:data:`PLATFORM_ROLE_ATTRIBUTES`) and belong to no superuser or superuser-equivalent role
   (:data:`SUPERUSER_EQUIVALENT_ROLES`); ``solace_app`` must also not be a member of any BYPASSRLS role.
   Migrations must run as ``solace_migrator`` because 009_platform_roles_rls grants ``solace_app`` its DML through
   default privileges ``FOR ROLE solace_migrator``: a table another role creates gives the services no access.
4. Take ``pg_try_advisory_lock(hashtext('solace_migrate'))`` on that connection, retrying for at most
   ``--lock-timeout`` seconds, so jobs against one database run one after another.
5. Run the Alembic command (alembic.ini, migrations/env.py) in a worker thread, because env.py calls
   ``asyncio.run``. The lock is held until that thread has ended, however often the run is cancelled. Then check
   that ``alembic_version`` exists after an upgrade and is owned by the expected role, i.e. that Alembic's own
   connection migrated this database as that role.
6. Print ``migrated: <version or base>`` and release the lock.

Exit codes: 0 success; 1 failure (server not reachable, TLS refused, login or role refused, lock timeout, Alembic
error); 2 usage (no or unusable database URL or TLS settings, bad arguments). Nothing printed holds a password:
URLs are shown only through :func:`redact`, and every message is scrubbed of the URL's password before it is
printed. The CLI sends structlog output to stderr, so stdout carries only the result line.

Python API (WS-05's ``pg_migrated``): :func:`run_migrations` (synchronous), :func:`migrate_async`,
:func:`resolve_database_url` and :func:`redact`. The refusals above raise :class:`MigrateError`
(:class:`MigrateUsageError` for an unusable URL); Alembic's own errors propagate unchanged. The API leaves the
caller's logging alone: only the CLI applies alembic.ini's logging sections.
"""
from __future__ import annotations

import argparse
import asyncio
import configparser
import logging.config
import math
import os
import ssl
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qsl, quote, unquote, urlsplit

import asyncpg
from alembic import command as alembic_command
from alembic.config import Config

ROOT: Final = Path(__file__).resolve().parents[2]
ALEMBIC_INI: Final = ROOT / "alembic.ini"
MIGRATIONS_DIR: Final = ROOT / "migrations"
SRC_DIR: Final = ROOT / "src"

COMMANDS: Final = ("upgrade", "downgrade", "current")
MIGRATOR_ROLE: Final = "solace_migrator"
APP_ROLE: Final = "solace_app"
DEFAULT_WAIT_TIMEOUT_S: Final = 120.0
DEFAULT_LOCK_TIMEOUT_S: Final = 600.0

# The security-relevant attributes infrastructure/init-db/01-roles.sh gives the platform roles (WS-02.1). The
# migrator has BYPASSRLS so that data migrations see every row of a FORCE ROW LEVEL SECURITY table.
PLATFORM_ROLE_ATTRIBUTES: Final[Mapping[str, Mapping[str, bool]]] = {
    MIGRATOR_ROLE: {"rolsuper": False, "rolcreatedb": False, "rolcreaterole": False, "rolreplication": False,
                    "rolbypassrls": True},
    APP_ROLE: {"rolsuper": False, "rolcreatedb": False, "rolcreaterole": False, "rolreplication": False,
               "rolbypassrls": False},
}
_ANY_EXPECTED_ROLE: Final[Mapping[str, bool]] = {"rolsuper": False}  # an --expected-role outside the platform roles
_KEYWORDS: Final = {"rolsuper": "SUPERUSER", "rolcreatedb": "CREATEDB", "rolcreaterole": "CREATEROLE",
                    "rolreplication": "REPLICATION", "rolbypassrls": "BYPASSRLS"}

_ROLES_SQL: Final = """
SELECT r.rolname, r.rolsuper, r.rolcreatedb, r.rolcreaterole, r.rolreplication, r.rolbypassrls,
       ARRAY(SELECT m.rolname FROM pg_roles m
             WHERE m.oid <> r.oid AND m.rolsuper AND pg_has_role(r.oid, m.oid, 'MEMBER')
             ORDER BY m.rolname) AS superuser_memberships,
       ARRAY(SELECT m.rolname FROM pg_roles m
             WHERE m.oid <> r.oid AND m.rolbypassrls AND pg_has_role(r.oid, m.oid, 'MEMBER')
             ORDER BY m.rolname) AS bypassrls_memberships,
       ARRAY(SELECT m.rolname FROM pg_roles m
             WHERE m.rolname = ANY($2::text[]) AND pg_has_role(r.oid, m.oid, 'MEMBER')
             ORDER BY m.rolname) AS server_access_memberships
FROM pg_roles r
WHERE r.rolname = ANY($1::text[])
"""
# Predefined roles the PostgreSQL docs call superuser-equivalent: OS commands through COPY ... TO PROGRAM, and
# reading or writing any file the server can reach.
SUPERUSER_EQUIVALENT_ROLES: Final = ("pg_execute_server_program", "pg_read_server_files", "pg_write_server_files")
_VERSION_TABLE_OWNER_SQL: Final = ("SELECT pg_get_userbyid(relowner) FROM pg_class "
                                   "WHERE oid = to_regclass('public.alembic_version')")
_LOCK_SQL: Final = "SELECT pg_try_advisory_lock(hashtext('solace_migrate'))"
_UNLOCK_SQL: Final = "SELECT pg_advisory_unlock(hashtext('solace_migrate'))"

_SCHEMES: Final = frozenset({"postgresql", "postgres", "postgresql+asyncpg"})
_DEFAULT_PORT: Final = 5432
_RETRY_INTERVAL_S: Final = 0.5
_ATTEMPT_TIMEOUT_S: Final = 10.0
_UNPARSEABLE: Final = "<unparseable database URL>"

# Alembic's `context` proxy is process-global and env.py reads DATABASE_URL from os.environ, so one command at a
# time per process.
_ALEMBIC_LOCK: Final = threading.Lock()


class MigrateError(RuntimeError):
    """The run was refused or could not reach the database (CLI exit 1). The message holds no credentials."""

    exit_code: int = 1


class MigrateUsageError(MigrateError):
    """No usable database URL or settings (CLI exit 2). The message names variables, never values."""

    exit_code = 2


def redact(url: str) -> str:
    """``user:***@host:port/db`` for a database URL.

    The password and the query string (which may carry one too) are never shown; a URL that does not parse is
    shown as a placeholder only."""
    try:
        parts = urlsplit(url)
        port = parts.port or _DEFAULT_PORT
    except ValueError:
        return _UNPARSEABLE
    host = parts.hostname
    if not parts.scheme or not host:
        return _UNPARSEABLE
    userinfo = unquote(parts.username or "") + (":***" if parts.password is not None else "")
    shown_host = f"[{host}]" if ":" in host else host
    return f"{userinfo + '@' if userinfo else ''}{shown_host}:{port}/{unquote(parts.path.lstrip('/'))}"


def _password_forms(url: str) -> set[str]:
    """Every form in which the URL's password could appear in a message, including when the URL does not parse."""
    candidates: set[str] = set()
    userinfo = (url.partition("://")[2] or url).rpartition("@")[0]
    if ":" in userinfo:
        candidates.add(userinfo.partition(":")[2])
    try:
        parts = urlsplit(url)
    except ValueError:
        parts = None
    if parts is not None:
        if parts.password is not None:
            candidates.add(parts.password)
        candidates.update(v for k, v in parse_qsl(parts.query) if k.lower() == "password")
    forms: set[str] = set()
    for raw in candidates:
        for text in (raw, unquote(raw)):
            forms.update({text, quote(text, safe=""), text.replace("%", "%%")})
    forms.discard("")
    return forms


def _scrub(text: str, secrets: Iterable[str]) -> str:
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, "***")
    return text


@dataclass(frozen=True)
class _Database:
    dsn: str = field(repr=False)  # postgresql://, for asyncpg and (through DATABASE_URL) migrations/env.py
    shown: str
    secrets: frozenset[str] = field(repr=False)

    def describe(self, exc: BaseException) -> str:
        text = str(exc)
        return _scrub(f"{type(exc).__name__}: {text}" if text else type(exc).__name__, self.secrets)


def _database(url: str) -> _Database:
    scheme, sep, rest = url.partition("://")
    if not sep or scheme.lower() not in _SCHEMES:
        raise MigrateUsageError("the database URL (DATABASE_URL) must be a postgresql:// URL "
                                "(postgres:// and postgresql+asyncpg:// are accepted too)")
    dsn = f"postgresql://{rest}"
    try:
        parts = urlsplit(dsn)
        _ = parts.port  # raises ValueError for a port that is not a number
    except ValueError:
        raise MigrateUsageError("the database URL (DATABASE_URL) does not parse; "
                                "is the password percent-encoded?") from None
    if parts.netloc.count("@") > 1:
        # asyncpg and SQLAlchemy split at the first '@', urlsplit at the last: they would dial different hosts.
        raise MigrateUsageError("the database URL (DATABASE_URL) does not parse; is the password percent-encoded?")
    if parts.query or parts.fragment or "?" in dsn or "#" in dsn:
        # The preflight and the lock connect through asyncpg, which takes user, host, port and database from the
        # URL itself; Alembic connects through SQLAlchemy, which lets query parameters override all four. A query
        # could therefore pass the preflight as solace_migrator and migrate as another role or database.
        raise MigrateUsageError("the database URL (DATABASE_URL) must not carry query parameters or a fragment; the "
                                "migrate job takes user, host, port and database from the URL itself (TLS comes from "
                                "WS-02.5's ssl context; percent-encode '?' and '#' in a password)")
    if not parts.hostname:
        raise MigrateUsageError("the database URL (DATABASE_URL) names no host")
    return _Database(dsn=dsn, shown=redact(dsn),
                     secrets=frozenset(_password_forms(url) | _password_forms(dsn) | {url, dsn}))


def resolve_database_url(environ: Mapping[str, str]) -> str | None:
    """The database URL from ``environ`` alone (os.environ and .env files are not consulted).

    DATABASE_URL when it is set and not empty; else, when POSTGRES_PASSWORD is set, a URL built from
    ``PostgresSettings`` (POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DATABASE and POSTGRES_USER, or their defaults) with
    user, password and database percent-encoded (the shape of ``PostgresSettings.get_dsn()``, which does not encode
    yet: MVP-273); else None. Raises :class:`MigrateUsageError`, naming the variables only, when the POSTGRES_*
    values do not validate.
    """
    url = (environ.get("DATABASE_URL") or "").strip()
    if url:
        return url
    password = environ.get("POSTGRES_PASSWORD") or ""
    if not password:
        return None
    from pydantic import ValidationError

    from solace_infrastructure.postgres import PostgresSettings

    fields = PostgresSettings.model_fields
    values = {name: environ.get(f"POSTGRES_{name.upper()}") or fields[name].default
              for name in ("host", "port", "database", "user")}
    try:
        settings = PostgresSettings(_env_file=None, password=password, **values)
    except ValidationError as exc:
        names = sorted({f"POSTGRES_{str(error['loc'][0]).upper()}" for error in exc.errors() if error.get("loc")})
        raise MigrateUsageError(f"{', '.join(names) or 'POSTGRES_*'} is not valid for the database URL") from None
    host = f"[{settings.host}]" if ":" in settings.host else settings.host
    return (f"postgresql://{quote(settings.user, safe='')}:{quote(settings.password.get_secret_value(), safe='')}"
            f"@{host}:{settings.port}/{quote(settings.database, safe='')}")


def _note(message: str) -> None:
    print(f"migrate: {message}", file=sys.stderr, flush=True)


def _client_ssl_context() -> ssl.SSLContext | None:
    """The TLS context for this run's connections: ``build_ssl_context(POSTGRES_SSL_MODE, POSTGRES_SSL_ROOT_CERT)``
    from os.environ (ENVIRONMENT decides strictness), exactly as migrations/env.py builds Alembic's. None (the
    default ``prefer`` outside staging and production) lets asyncpg negotiate.

    Raises :class:`MigrateUsageError` (naming the variable, never its value) for unusable TLS settings."""
    if str(SRC_DIR) not in sys.path:  # env.py puts src/ on sys.path too; appending changes no earlier resolution
        sys.path.append(str(SRC_DIR))
    from solace_infrastructure.postgres import DatabaseConfigError, build_ssl_context

    try:
        return build_ssl_context(os.environ.get("POSTGRES_SSL_MODE", "prefer"), os.environ.get("POSTGRES_SSL_ROOT_CERT"))
    except DatabaseConfigError as exc:
        raise MigrateUsageError(str(exc)) from None


async def _connect(dsn: str, *, timeout: float, ssl_context: ssl.SSLContext | None) -> asyncpg.Connection:
    """One login attempt with :func:`_client_ssl_context`'s context (None: asyncpg negotiates, i.e. ``prefer``)."""
    return await asyncpg.connect(dsn, timeout=timeout, ssl=ssl_context)


# Failures that waiting can cure: nothing listening yet, a server still starting, no free connection slot.
_TRANSIENT_ERRORS: Final = (OSError, TimeoutError, asyncpg.exceptions.CannotConnectNowError,
                            asyncpg.exceptions.TooManyConnectionsError, asyncpg.exceptions.PostgresConnectionError)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLError):  # an OSError, but a certificate problem does not go away
        return False
    if type(exc) is ConnectionError and "rejected SSL upgrade" in str(exc):  # asyncpg: the server has no TLS
        return False
    return isinstance(exc, _TRANSIENT_ERRORS)


async def _wait_for_connection(db: _Database, wait_timeout_s: float,
                               ssl_context: ssl.SSLContext | None) -> asyncpg.Connection:
    deadline = time.monotonic() + wait_timeout_s
    attempts = 0
    while True:
        attempts += 1
        remaining = deadline - time.monotonic()
        try:
            return await _connect(db.dsn, timeout=min(_ATTEMPT_TIMEOUT_S, max(1.0, remaining)),
                                  ssl_context=ssl_context)
        except asyncpg.exceptions.InvalidAuthorizationSpecificationError as exc:  # also InvalidPasswordError
            raise MigrateError(f"login refused for {db.shown}: {db.describe(exc)}") from None
        except Exception as exc:
            if not _is_transient(exc):
                raise MigrateError(f"cannot connect to {db.shown}: {db.describe(exc)}") from None
            if time.monotonic() >= deadline:
                raise MigrateError(f"database not reachable at {db.shown} within {wait_timeout_s:g}s "
                                   f"({attempts} attempt(s); last error: {db.describe(exc)})") from None
            if attempts == 1:
                _note(f"waiting up to {wait_timeout_s:g}s for {db.shown} ({db.describe(exc)})")
        await asyncio.sleep(min(_RETRY_INTERVAL_S, max(0.0, deadline - time.monotonic())))


def _keyword(attribute: str, value: bool) -> str:
    return _KEYWORDS[attribute] if value else f"NO{_KEYWORDS[attribute]}"


async def _check_roles(conn: asyncpg.Connection, expected_role: str) -> None:
    """Refuse a login that is not ``expected_role``, a superuser, or platform roles that drifted from IF-03.

    Both ``current_user`` and ``session_user`` must be ``expected_role``: a session that logged in as another role
    and switched (``ALTER ROLE ... SET role``) can switch back at any time."""
    identity = await conn.fetchrow("SELECT current_user AS who, session_user AS login")
    who, login = identity["who"], identity["login"]
    names = list(dict.fromkeys((expected_role, *PLATFORM_ROLE_ATTRIBUTES)))
    rows = {row["rolname"]: row
            for row in await conn.fetch(_ROLES_SQL, [*names, who, login], list(SUPERUSER_EQUIVALENT_ROLES))}
    if who != expected_role or login != expected_role:
        superuser = " (a superuser)" if login in rows and rows[login]["rolsuper"] else ""
        acting = f" acting as {who}" if who != login else ""
        raise MigrateError(f"refusing to migrate: connected as {login}{superuser}{acting}, but migrations run only "
                           f"as {expected_role}, a NOSUPERUSER role (IF-03)")
    problems: list[str] = []
    for name in names:
        row = rows.get(name)
        if row is None:
            problems.append(f"role {name} does not exist (infrastructure/init-db creates it)")
            continue
        for attribute, wanted in PLATFORM_ROLE_ATTRIBUTES.get(name, _ANY_EXPECTED_ROLE).items():
            if row[attribute] is not wanted:
                problems.append(f"role {name} must be {_keyword(attribute, wanted)} "
                                f"(it is {_keyword(attribute, row[attribute])})")
        if row["superuser_memberships"]:
            problems.append(f"role {name} must not be a member of superuser role(s) "
                            f"{', '.join(row['superuser_memberships'])}")
        if name == APP_ROLE and row["bypassrls_memberships"]:
            problems.append(f"role {name} must not be a member of BYPASSRLS role(s) "
                            f"{', '.join(row['bypassrls_memberships'])}")
        if row["server_access_memberships"]:
            problems.append(f"role {name} must not be a member of superuser-equivalent role(s) "
                            f"{', '.join(row['server_access_memberships'])}")
    if problems:
        raise MigrateError("refusing to migrate: " + "; ".join(problems))


async def _check_version_table(conn: asyncpg.Connection, command: str, expected_role: str) -> None:
    """Defence in depth: Alembic opens its own connection, so confirm afterwards that it migrated this database as
    ``expected_role`` (the version table's owner) before reporting success."""
    owner = await conn.fetchval(_VERSION_TABLE_OWNER_SQL)
    if owner is None:
        if command == "upgrade":
            raise MigrateError("no alembic_version after upgrade: Alembic did not migrate the database the "
                               "preflight checked")
        return
    if owner != expected_role:
        raise MigrateError(f"alembic_version is owned by {owner}, not {expected_role}: the migration did not run as "
                           f"the role the preflight checked")


async def _take_lock(conn: asyncpg.Connection, db: _Database, lock_timeout_s: float) -> None:
    deadline = time.monotonic() + lock_timeout_s
    waiting = False
    while not await conn.fetchval(_LOCK_SQL):
        if time.monotonic() >= deadline:
            raise MigrateError(f"could not take the migration lock (pg_advisory_lock(hashtext('solace_migrate'))) "
                               f"on {db.shown} within {lock_timeout_s:g}s: another migrate job still holds it")
        if not waiting:
            _note(f"waiting up to {lock_timeout_s:g}s for the migration lock on {db.shown}: another job holds it")
            waiting = True
        await asyncio.sleep(min(_RETRY_INTERVAL_S, max(0.0, deadline - time.monotonic())))


async def _release_lock(conn: asyncpg.Connection, db: _Database) -> None:
    try:
        released = await conn.fetchval(_UNLOCK_SQL)
    except Exception as exc:  # never mask the command's own outcome; the server drops the lock with the connection
        _note(f"could not release the migration lock ({db.describe(exc)}); the server drops it with the connection")
        return
    if released is not True:
        _note("warning: the migration lock was no longer held when the command ended")


async def _close(conn: asyncpg.Connection) -> None:
    try:
        await asyncio.wait_for(conn.close(), _ATTEMPT_TIMEOUT_S)
    except Exception:  # a broken connection: drop it without the goodbye
        conn.terminate()


async def _read_version(conn: asyncpg.Connection) -> str | None:
    if await conn.fetchval("SELECT to_regclass('public.alembic_version')") is None:
        return None
    rows = await conn.fetch("SELECT version_num FROM public.alembic_version ORDER BY version_num")
    return ",".join(row["version_num"] for row in rows) or None


@contextmanager
def _environ(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _alembic_config() -> Config:
    """alembic.ini's ``[alembic]`` options on a Config without a file name.

    migrations/env.py runs ``logging.config.fileConfig`` on the config file whenever the Config names one, which
    disables every logger the calling process already has; in-process callers keep theirs this way (the CLI
    applies the same logging sections itself, in :func:`main`).
    """
    ini = configparser.ConfigParser(defaults={"here": ROOT.as_posix()})
    with ALEMBIC_INI.open(encoding="utf-8") as handle:
        ini.read_file(handle)
    cfg = Config(ini_section="alembic")
    for key, value in ini.items("alembic"):
        if key != "here":
            cfg.set_main_option(key, value.replace("%", "%%"))  # Config re-interpolates option values
    cfg.set_main_option("script_location", MIGRATIONS_DIR.as_posix().replace("%", "%%"))
    return cfg


def _run_alembic(dsn: str, command: str, target: str) -> None:
    """The Alembic command, in the calling (worker) thread; env.py reads the URL from DATABASE_URL."""
    with _ALEMBIC_LOCK, _environ("DATABASE_URL", dsn):
        cfg = _alembic_config()
        if command == "upgrade":
            alembic_command.upgrade(cfg, target)
        elif command == "downgrade":
            alembic_command.downgrade(cfg, target)
        else:
            alembic_command.current(cfg)


async def _in_worker_thread(fn: Callable[..., None], *args: Any) -> None:
    """Run ``fn`` in a worker thread and, if cancelled (any number of times, e.g. a double Ctrl-C), still wait for it:
    the thread cannot be stopped, and the caller holds the migration lock until it has ended."""
    work = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    try:
        await asyncio.shield(work)
    except asyncio.CancelledError:
        while not work.done():
            try:
                await asyncio.wait({work})
            except asyncio.CancelledError:
                pass  # still running, so the lock must stay held; the cancellation is re-raised below
        if not work.cancelled():
            work.exception()  # retrieved here, so asyncio does not report it; the caller sees the cancellation
        raise


def _check_timeout(name: str, value: float) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise MigrateUsageError(f"{name} must be a finite number of seconds >= 0")


async def migrate_async(database_url: str, command: str, target: str = "head", *,
                        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S, lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
                        expected_role: str = MIGRATOR_ROLE) -> str | None:
    """Run ``command`` (upgrade, downgrade or current; ``target`` is unused by current) through the steps in the
    module docstring and return the revision the database is at afterwards, None for base."""
    if command not in COMMANDS:
        raise MigrateUsageError(f"unknown command {command!r}; use one of {', '.join(COMMANDS)}")
    _check_timeout("wait_timeout_s", wait_timeout_s)
    _check_timeout("lock_timeout_s", lock_timeout_s)
    db = _database(database_url)
    ssl_context = _client_ssl_context()
    _note(f"{command}{'' if command == 'current' else ' ' + target} on {db.shown}")
    conn = await _wait_for_connection(db, wait_timeout_s, ssl_context)
    try:
        await _check_roles(conn, expected_role)
        await _take_lock(conn, db, lock_timeout_s)
        try:
            await _in_worker_thread(_run_alembic, db.dsn, command, target)
            await _check_version_table(conn, command, expected_role)
            return await _read_version(conn)
        finally:
            await _release_lock(conn, db)
    finally:
        await _close(conn)


def run_migrations(database_url: str, command: str, target: str = "head", *,
                   wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S, lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
                   expected_role: str = MIGRATOR_ROLE) -> str | None:
    """:func:`migrate_async` for synchronous code (or a worker thread); refuses to run inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(migrate_async(database_url, command, target, wait_timeout_s=wait_timeout_s,
                                         lock_timeout_s=lock_timeout_s, expected_role=expected_role))
    raise RuntimeError("run_migrations() is synchronous and cannot run inside an event loop; "
                       "await migrate_async() instead, or call run_migrations() from a worker thread")


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number of seconds: {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"must be a finite number of seconds >= 0: {text!r}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.db.migrate",
        description="Apply the Alembic chain as solace_migrator, one job per database at a time (IF-03). The "
                    "database URL comes from DATABASE_URL, else from POSTGRES_* when POSTGRES_PASSWORD is set.")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("target", nargs="?",
                        help="upgrade: revision (default head); downgrade: revision (required, e.g. base); "
                             "current: none")
    parser.add_argument("--wait-timeout", type=_seconds, default=DEFAULT_WAIT_TIMEOUT_S, metavar="SECONDS",
                        help="how long to wait for the server to accept a login (default %(default)g)")
    parser.add_argument("--lock-timeout", type=_seconds, default=DEFAULT_LOCK_TIMEOUT_S, metavar="SECONDS",
                        help="how long to wait for another migrate job to finish (default %(default)g)")
    parser.add_argument("--expected-role", default=MIGRATOR_ROLE, metavar="NAME",
                        help="the only role allowed to migrate (default %(default)s); never a superuser")
    return parser


def _structlog_to_stderr() -> None:
    """solace_infrastructure logs through structlog (e.g. ``ssl_mode_upgraded``), whose default writes to stdout;
    the CLI keeps stdout for its result line. A process that configured structlog itself is left alone."""
    import structlog

    if not structlog.is_configured():
        structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)  # exits 2 on bad arguments
    if args.command == "downgrade" and not args.target:
        parser.error("downgrade needs an explicit target revision, for example: downgrade base")
    if args.command == "current" and args.target:
        parser.error("current takes no target revision")
    try:
        url = resolve_database_url(os.environ)
    except MigrateUsageError as exc:
        _note(f"error: {exc}")
        return exc.exit_code
    if url is None:
        _note("error: no database URL: set DATABASE_URL (docker/entrypoint.sh also loads DATABASE_URL_FILE), or "
              "POSTGRES_PASSWORD with POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DATABASE and POSTGRES_USER")
        return 2
    secrets = _password_forms(url) | {url}
    # Alembic's progress lines go to stderr exactly as with the alembic CLI.
    logging.config.fileConfig(ALEMBIC_INI, disable_existing_loggers=False, encoding="utf-8")
    _structlog_to_stderr()
    try:
        version = run_migrations(url, args.command, args.target or "head", wait_timeout_s=args.wait_timeout,
                                 lock_timeout_s=args.lock_timeout, expected_role=args.expected_role)
    except MigrateError as exc:
        _note(f"error: {_scrub(str(exc), secrets)}")
        return exc.exit_code
    except Exception as exc:  # an Alembic, SQL or driver error: report it scrubbed and exit 1
        _note(f"error: {args.command} failed:\n{_scrub(''.join(traceback.format_exception(exc)), secrets)}")
        return 1
    print(f"migrated: {version or 'base'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
