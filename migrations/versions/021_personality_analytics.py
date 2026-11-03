"""Slot 021: personality and analytics. Empty until filled. The chain head.

Revision ID: 021_personality_analytics
Revises: 020_diagnosis
Create Date: 2026-11-02

Owner: WS-18

Fixed slot created by WS-02.3 (IF-03, MVP-041). The owner edits only ``upgrade()`` and
``downgrade()`` (D-01); the file name, revision id and ``down_revision`` stay as they are, and
``downgrade()`` must undo ``upgrade()`` cleanly. This is the head the services check at
startup (``SCHEMA_HEAD`` in ``solace_infrastructure.database.schema_check``); a revision after
this one must move ``SCHEMA_HEAD`` with it.

Conventions for every slot: Postgres-only DDL sits behind ``_is_postgres()`` so SQLite-based
unit runs stay a no-op; new tables are owned by ``solace_migrator`` and get DML for
``solace_app`` through 009's default privileges (revoke what a table must not allow); a
per-user RLS policy uses 009's ``NULLIF_PREDICATE`` for USING and WITH CHECK. A table with a
``user_id`` column must ENABLE and FORCE row level security with such a policy, or be added to
``RLS_DEFERRED`` in ``tests/migrations/test_chain_real_pg.py`` with a reason, and every view
is created ``WITH (security_invoker = true)``.
"""
from __future__ import annotations

import sqlalchemy as sa  # noqa: F401  (for the owner's upgrade() and downgrade())
from alembic import context, op  # noqa: F401  (for the owner's upgrade() and downgrade())

# Alembic identifiers
revision = "021_personality_analytics"
down_revision = "020_diagnosis"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    bind = context.get_bind()
    return bind is not None and bind.dialect.name == "postgresql"


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
