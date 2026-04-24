"""Safety escalations persistence table (H-04).

Revision ID: 003_escalations
Revises: 002_enable_rls
Create Date: 2026-04-24

Creates the ``escalations`` table used by
``PostgresEscalationRepository`` in the safety service. Before this
migration the escalation state lived in process memory
(``InMemoryEscalationRepository``) and was lost on every restart —
a HIPAA-relevant gap because active clinician workflows could be
silently dropped.

Schema matches the ``EscalationRecord`` dataclass:

    escalation_id          UUID (PK)
    user_id                UUID (not null, indexed)
    session_id             UUID (nullable)
    crisis_level           VARCHAR(16) (not null)
    priority               VARCHAR(16) (not null, indexed)
    status                 VARCHAR(32) (not null, indexed)
    reason                 TEXT (not null)
    assigned_clinician_id  UUID (nullable)
    notifications_sent     JSONB (list of NotificationType strings)
    actions_taken          JSONB (list of audit action strings)
    context                JSONB (arbitrary dict)
    created_at             TIMESTAMPTZ (not null)
    acknowledged_at        TIMESTAMPTZ (nullable)
    resolved_at            TIMESTAMPTZ (nullable)

No Row-Level Security policy is added here; ``user_id`` links to the
``users`` table only logically (not as a FK) because the safety service
runs in its own database in multi-db deployments. If the future
deployment model consolidates, a FK + RLS policy can be added in a
follow-up migration.

The migration is Postgres-only for the JSONB columns; it no-ops on
other dialects so SQLite-backed tests don't break.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

# Alembic identifiers
revision = "003_escalations"
down_revision = "002_enable_rls"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    bind = context.get_bind()
    return bind is not None and bind.dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS escalations (
            escalation_id         UUID PRIMARY KEY,
            user_id               UUID NOT NULL,
            session_id            UUID NULL,
            crisis_level          VARCHAR(16) NOT NULL,
            priority              VARCHAR(16) NOT NULL,
            status                VARCHAR(32) NOT NULL,
            reason                TEXT NOT NULL,
            assigned_clinician_id UUID NULL,
            notifications_sent    JSONB NOT NULL DEFAULT '[]'::jsonb,
            actions_taken         JSONB NOT NULL DEFAULT '[]'::jsonb,
            context               JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at            TIMESTAMPTZ NOT NULL,
            acknowledged_at       TIMESTAMPTZ NULL,
            resolved_at           TIMESTAMPTZ NULL
        );
        """
    )
    # Query-heavy columns: active-escalation lookups by user + status.
    op.create_index(
        "ix_escalations_user_id", "escalations", ["user_id"], if_not_exists=True,
    )
    op.create_index(
        "ix_escalations_status", "escalations", ["status"], if_not_exists=True,
    )
    op.create_index(
        "ix_escalations_priority", "escalations", ["priority"], if_not_exists=True,
    )
    op.create_index(
        "ix_escalations_created_at",
        "escalations",
        ["created_at"],
        if_not_exists=True,
    )


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute("DROP INDEX IF EXISTS ix_escalations_created_at;")
    op.execute("DROP INDEX IF EXISTS ix_escalations_priority;")
    op.execute("DROP INDEX IF EXISTS ix_escalations_status;")
    op.execute("DROP INDEX IF EXISTS ix_escalations_user_id;")
    op.execute("DROP TABLE IF EXISTS escalations;")


# Silence linters about unused import while still making it clear we depend on sqlalchemy:
_ = sa
