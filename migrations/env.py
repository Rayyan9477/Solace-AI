"""Alembic environment configuration for Solace-AI database migrations."""
from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from solace_infrastructure.database.base_models import Base  # noqa: E402
from solace_infrastructure.postgres import build_ssl_context  # noqa: E402

config = context.config

# DATABASE_URL supplies the URL (alembic.ini holds none). Coerce a plain postgresql://
# DSN to the asyncpg driver required by run_async_migrations(). Online mode refuses to
# run without it; offline mode only needs the dialect to render SQL.
_db_url = os.environ.get("DATABASE_URL")
if _db_url:
    if _db_url.startswith("postgresql://"):
        _db_url = _db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    # ConfigParser treats '%' as interpolation syntax; a percent-encoded URL must escape it.
    config.set_main_option("sqlalchemy.url", _db_url.replace("%", "%%"))
elif context.is_offline_mode():
    config.set_main_option("sqlalchemy.url", "postgresql+asyncpg://")
else:
    raise RuntimeError("DATABASE_URL is required to run migrations")

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL script without requiring a live database connection.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        transaction_per_migration=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations against the provided connection."""
    context.configure(connection=connection, target_metadata=target_metadata, transaction_per_migration=True)

    with context.begin_transaction():
        context.run_migrations()


def _connect_args() -> dict[str, object]:
    """asyncpg's ``ssl`` for the engine, from POSTGRES_SSL_MODE and POSTGRES_SSL_ROOT_CERT (MVP-010, MVP-271).

    The same policy as every other Postgres client (``build_ssl_context``; scripts/db/migrate.py uses it for its
    own connections): in staging and production the migration connection verifies the server certificate
    against POSTGRES_SSL_ROOT_CERT and the host name (verify-full), and an unset or unreadable root certificate
    stops the run. Elsewhere the default ``prefer`` adds nothing, so asyncpg negotiates as before.
    """
    ctx = build_ssl_context(os.environ.get("POSTGRES_SSL_MODE", "prefer"), os.environ.get("POSTGRES_SSL_ROOT_CERT"))
    return {"ssl": ctx} if ctx is not None else {}


async def run_async_migrations() -> None:
    """Run migrations using an async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=_connect_args(),
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with async engine.

    asyncio.run() is correct here: migrations run as a standalone CLI script
    (alembic upgrade head), not inside a running event loop.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
