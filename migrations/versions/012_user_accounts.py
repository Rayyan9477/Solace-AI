"""Slot 012: user accounts (roster, consent, erasure). Empty until filled.

Revision ID: 012_user_accounts
Revises: 011_audit_trail
Create Date: 2026-11-02

Owner: WS-14

Fixed slot created by WS-02.3 (IF-03, MVP-041). The owner edits only ``upgrade()`` and
``downgrade()`` (D-01); the file name, revision id and ``down_revision`` stay as they are, and
``downgrade()`` must undo ``upgrade()`` cleanly.

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
revision = "012_user_accounts"
down_revision = "011_audit_trail"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    bind = context.get_bind()
    return bind is not None and bind.dialect.name == "postgresql"


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
