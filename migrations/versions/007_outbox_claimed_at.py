"""Add the outbox ``claimed_at`` column for the atomic-claim poller (REV-39).

Revision ID: 007_outbox_claimed_at
Revises: 006_extend_rls
Create Date: 2026-07-30

The transactional-outbox poller now CLAIMS records atomically (PENDING ->
PUBLISHING with a ``claimed_at`` stamp) instead of a bare ``SELECT ... FOR UPDATE
SKIP LOCKED`` whose row locks released the instant the connection returned to the
pool — which let two replicas' pollers claim the same rows and double-publish.
The claim + stale-claim reclaim need a ``claimed_at`` column and a supporting
index (see ``src/solace_events/postgres_stores.py``).

The ``event_outbox`` table is created by slot ``010_event_plane``, which also creates
the ``claimed_at`` column and ``idx_outbox_publishing_claimed`` on fresh databases. So
this revision only touches the table when it already exists (a database that predates
the chain): one guarded ``DO`` block holding the idempotent ``ADD COLUMN IF NOT
EXISTS`` and ``CREATE INDEX IF NOT EXISTS``. On a fresh database it is a no-op.

Postgres-only; a no-op on SQLite (the outbox uses the in-memory store there).
"""
from __future__ import annotations

from alembic import context, op

# Alembic identifiers
revision = "007_outbox_claimed_at"
down_revision = "006_extend_rls"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    bind = context.get_bind()
    return bind is not None and bind.dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return
    op.execute(
        "DO $$ BEGIN "
        "IF to_regclass('public.event_outbox') IS NOT NULL THEN "
        "ALTER TABLE event_outbox ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ; "
        "CREATE INDEX IF NOT EXISTS idx_outbox_publishing_claimed "
        "ON event_outbox (claimed_at) WHERE status = 'PUBLISHING'; "
        "END IF; END $$;"
    )


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute("DROP INDEX IF EXISTS idx_outbox_publishing_claimed;")
    op.execute("ALTER TABLE IF EXISTS event_outbox DROP COLUMN IF EXISTS claimed_at;")
