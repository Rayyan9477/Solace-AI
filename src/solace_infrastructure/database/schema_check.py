"""Startup check that the database carries the schema revision this release was built for (MVP-041, IF-03).

Services never run DDL: the ``migrate`` job applies the Alembic chain as ``solace_migrator``,
and each service, connected as ``solace_app``, compares ``alembic_version`` with
:data:`SCHEMA_HEAD` before it serves traffic. Migration 009 leaves ``alembic_version``
SELECT-only for ``solace_app``, so the check runs as the app role and cannot alter the
recorded revision.

Any difference counts as a mismatch: an older revision (the migration has not run), a newer
one (for example a restore onto a database migrated past this release), or no revision at all
(``alembic_version`` missing, empty, or not readable by this role, as before 009). In strict environments (``ENVIRONMENT`` staging or
production) :func:`assert_schema_head` raises :class:`SchemaVersionMismatch`; elsewhere it logs
a WARNING and returns what it found.

Usage::

    from solace_infrastructure.database.schema_check import assert_schema_head

    await assert_schema_head(pool)  # environment defaults to the ENVIRONMENT variable
"""
from __future__ import annotations

from typing import Any, Final

import asyncpg
import structlog

from solace_common.exceptions import DatabaseError
from solace_security.production_guards import is_strict_environment

logger = structlog.get_logger(__name__)

SCHEMA_HEAD: Final = "021_personality_analytics"

# string_agg yields NULL for an empty table and joins several rows (a branched stamp), which
# then never equals a single expected revision, so neither case can pass as the head.
_VERSION_SQL: Final = "SELECT string_agg(version_num, ',' ORDER BY version_num) FROM alembic_version"


class SchemaVersionMismatch(DatabaseError):  # noqa: N818  (name fixed by IF-03)
    """The database's ``alembic_version`` is not the revision this release expects."""

    error_code = "SCHEMA_VERSION_MISMATCH"

    def __init__(self, *, expected: str, found: str | None) -> None:
        self.expected: str = expected
        self.found: str | None = found
        super().__init__(
            f"database schema revision is {found or 'absent or unreadable'}, this release needs {expected}",
            operation="schema_head_check",
            details={"expected": expected, "found": found},
        )


async def read_schema_version(pool_or_conn: Any) -> str | None:
    """Return the revision stamped in ``alembic_version``, or None when there is none this role can see.

    None covers a missing or empty table and a table the role may not read: before migration 009
    ``solace_app`` has no privilege on ``alembic_version`` at all, which means 009 has not run, so
    the caller treats it as a mismatch like any other. ``pool_or_conn`` is anything with asyncpg's
    ``fetchval`` (a pool or a connection); other errors, such as a lost connection, propagate.
    """
    try:
        version = await pool_or_conn.fetchval(_VERSION_SQL)
    except (asyncpg.exceptions.UndefinedTableError, asyncpg.exceptions.InsufficientPrivilegeError):
        return None
    return version or None


async def assert_schema_head(
    pool: asyncpg.Pool,
    *,
    expected: str = SCHEMA_HEAD,
    environment: str | None = None,
) -> str | None:
    """Compare the database's revision with ``expected`` and return the revision found.

    Raises :class:`SchemaVersionMismatch` on any difference when ``environment`` (default: the
    ``ENVIRONMENT`` variable) is staging or production; otherwise logs a WARNING.
    """
    found = await read_schema_version(pool)
    if found == expected:
        return found
    if is_strict_environment(environment):
        raise SchemaVersionMismatch(expected=expected, found=found)
    logger.warning("schema_version_mismatch", expected=expected, found=found,
                   hint="run the migrate job (alembic upgrade head) before starting services")
    return found
