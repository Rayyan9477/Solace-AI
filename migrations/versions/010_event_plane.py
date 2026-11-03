"""Slot 010: the durable event plane (outbox, processed events, dead letters). Empty until filled.

Revision ID: 010_event_plane
Revises: 009_platform_roles_rls
Create Date: 2026-11-02

Owner: WS-06

Fixed slot created by WS-02.3 (IF-03, MVP-041). The owner edits only ``upgrade()`` and
``downgrade()`` (D-01); the file name, revision id and ``down_revision`` stay as they are, and
``downgrade()`` must undo ``upgrade()`` cleanly. This slot creates ``event_outbox`` on fresh
databases, including the ``claimed_at`` column that 007 only adds to a pre-existing table.
The stale-claim reclaim scan is served by ``ix_event_outbox_claimable (source_service,
created_at)`` (WS-06.1, controller Ruling 2026-11-03), not by a separate
``idx_outbox_publishing_claimed``.

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
revision = "010_event_plane"
down_revision = "009_platform_roles_rls"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    bind = context.get_bind()
    return bind is not None and bind.dialect.name == "postgresql"


def upgrade() -> None:
    # MVP-266 (WS-06.1): the only source of the event tables. Plain CREATE (no IF NOT EXISTS),
    # so a table left by the old runtime ensure_table() DDL fails loudly here. Owned by
    # solace_migrator; solace_app gets DML only, through 009's default privileges. No RLS:
    # IF-03 lists all three as non-RLS tables.
    if not _is_postgres():
        return
    # IF-05 transactional outbox; source_service scopes every claim to its own service's poller.
    op.execute(
        """
        CREATE TABLE event_outbox (
            id UUID PRIMARY KEY,
            source_service VARCHAR(64) NOT NULL,
            event_id UUID NOT NULL,
            event_type VARCHAR(255) NOT NULL,
            event_payload JSONB NOT NULL,
            aggregate_id UUID NOT NULL,
            topic VARCHAR(255) NOT NULL,
            partition_key VARCHAR(255) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            published_at TIMESTAMPTZ,
            status VARCHAR(20) NOT NULL DEFAULT 'PENDING'
                CHECK (status IN ('PENDING', 'PUBLISHING', 'PUBLISHED', 'FAILED')),
            retry_count INT NOT NULL DEFAULT 0,
            last_error TEXT,
            claimed_at TIMESTAMPTZ,
            claim_token UUID,
            next_retry_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_event_outbox_claimable ON event_outbox (source_service, created_at) "
        "WHERE status IN ('PENDING', 'PUBLISHING')"
    )
    op.execute(
        "CREATE INDEX ix_event_outbox_published_at ON event_outbox (published_at) "
        "WHERE status = 'PUBLISHED'"
    )
    op.execute("CREATE INDEX ix_event_outbox_aggregate ON event_outbox (aggregate_id)")

    # Consumer idempotency ledger, written only after handler success.
    op.execute(
        """
        CREATE TABLE processed_events (
            event_id UUID NOT NULL,
            consumer_group VARCHAR(255) NOT NULL,
            processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (event_id, consumer_group)
        )
        """
    )
    op.execute("CREATE INDEX ix_processed_events_processed_at ON processed_events (processed_at)")

    # Durable dead letters. consumer_group is a consumer group or 'outbox:<service>'; dlq_topic is
    # a compatibility label, never a Kafka topic. The event fields are nullable so a malformed
    # message is kept by its raw_payload and Kafka coordinates.
    op.execute(
        """
        CREATE TABLE event_dead_letter (
            id UUID PRIMARY KEY,
            consumer_group VARCHAR(255) NOT NULL,
            original_topic VARCHAR(255) NOT NULL,
            dlq_topic VARCHAR(255) NOT NULL,
            kafka_partition INT,
            kafka_offset BIGINT,
            original_event JSONB,
            raw_payload BYTEA,
            event_type VARCHAR(255),
            event_id UUID,
            user_id UUID,
            error_message TEXT NOT NULL,
            error_type VARCHAR(255) NOT NULL,
            stack_trace TEXT,
            retry_count INT NOT NULL DEFAULT 0,
            redrive_count INT NOT NULL DEFAULT 0,
            first_failure_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_failure_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            dlq_entry_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            next_retry_at TIMESTAMPTZ,
            is_retriable BOOLEAN NOT NULL DEFAULT TRUE,
            resolved BOOLEAN NOT NULL DEFAULT FALSE,
            resolved_at TIMESTAMPTZ,
            resolution_notes TEXT
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_event_dead_letter_redrive ON event_dead_letter (consumer_group, next_retry_at) "
        "WHERE NOT resolved AND is_retriable"
    )
    op.execute(
        "CREATE INDEX ix_event_dead_letter_unresolved ON event_dead_letter (consumer_group) "
        "WHERE NOT resolved"
    )
    op.execute("CREATE INDEX ix_event_dead_letter_user ON event_dead_letter (user_id)")
    op.execute(
        "CREATE INDEX ix_event_dead_letter_resolved_at ON event_dead_letter (resolved_at) "
        "WHERE resolved"
    )


def downgrade() -> None:
    # Drops exactly what upgrade() created (each table's indexes go with it).
    if not _is_postgres():
        return
    op.execute("DROP TABLE event_dead_letter")
    op.execute("DROP TABLE processed_events")
    op.execute("DROP TABLE event_outbox")
