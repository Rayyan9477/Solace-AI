"""OAuth account linkage table (Sprint 8 Google OAuth).

Revision ID: 004_oauth_accounts
Revises: 003_escalations
Create Date: 2026-04-25

Creates the ``oauth_accounts`` table that links internal users to
external OAuth providers (Google now; Apple post-MVP). One user may
link multiple providers; a (provider, provider_user_id) pair must be
unique.

Schema:
    id                 UUID PK
    user_id            UUID not null, FK -> users.id ON DELETE CASCADE
    provider           VARCHAR(32) not null, e.g. 'google'
    provider_user_id   VARCHAR(255) not null  (Google ``sub`` claim)
    email              VARCHAR(255) nullable  (diagnostic only)
    created_at         TIMESTAMPTZ not null default now()
    updated_at         TIMESTAMPTZ not null default now()

    UNIQUE (provider, provider_user_id)

Postgres-only; no-op on other dialects so SQLite test runs pass.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "004_oauth_accounts"
down_revision = "003_escalations"
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
        CREATE TABLE IF NOT EXISTS oauth_accounts (
            id                UUID PRIMARY KEY,
            user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            provider          VARCHAR(32) NOT NULL,
            provider_user_id  VARCHAR(255) NOT NULL,
            email             VARCHAR(255) NULL,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_oauth_provider_identity UNIQUE (provider, provider_user_id)
        );
        """
    )
    op.create_index(
        "ix_oauth_accounts_user_id", "oauth_accounts", ["user_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_oauth_accounts_email", "oauth_accounts", ["email"],
        if_not_exists=True,
    )


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute("DROP INDEX IF EXISTS ix_oauth_accounts_email;")
    op.execute("DROP INDEX IF EXISTS ix_oauth_accounts_user_id;")
    op.execute("DROP TABLE IF EXISTS oauth_accounts;")


_ = sa  # keep import referenced for readability parity with other migrations
