"""Enable Row-Level Security on highest-risk PHI tables (H-56 partial).

Revision ID: 002_enable_rls
Revises: 001_initial
Create Date: 2026-04-22

This migration enables PostgreSQL Row-Level Security on the three clinical
tables that carry the highest PHI exposure risk:

  - ``diagnosis_sessions`` — conversation history + clinical summary
  - ``therapy_sessions``   — intervention content + session notes
  - ``memory_records``     — long-term retained user context

The policy keys off a per-connection GUC ``app.current_user_id``. Service
code must set this GUC at the beginning of each request scope
(``SET LOCAL app.current_user_id = '<uuid>'``) so that Postgres will only
return rows matching the current user. A service account running without
the GUC set sees nothing, which is the safe default.

Admin / maintenance access goes through ``BYPASSRLS`` on the maintenance
role, granted separately.

This covers only 3 of the ~15 PHI-containing tables. Full RLS rollout
across the remaining tables is scheduled for Sprint 7. Per user
clarification, the sprint plan accepts 3 tables in Sprint 1 as the MVP
cut, with the rest deferred to post-MVP.

The migration is idempotent and a no-op on non-PostgreSQL dialects so
SQLite-backed test runs don't break.
"""
from __future__ import annotations

from alembic import context, op

# Alembic identifiers
revision = "002_enable_rls"
down_revision = "001_initial"
branch_labels = None
depends_on = None


_TABLES_WITH_RLS: list[str] = [
    "diagnosis_sessions",
    "therapy_sessions",
    "memory_records",
]


def _is_postgres() -> bool:
    """Check we're on Postgres. RLS statements are Postgres-only."""
    bind = context.get_bind()
    return bind is not None and bind.dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        # Skip silently on SQLite / other dialects. Tests run on SQLite.
        return

    for table in _TABLES_WITH_RLS:
        # Enable RLS on the table. Existing rows remain queryable only
        # under matching policies.
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        # Force RLS even for the table owner (matters when the service runs
        # with the owning role rather than a separate app role).
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")

        # Per-user access policy. Relies on the per-request GUC
        # ``app.current_user_id`` set by middleware. The cast to uuid
        # normalizes whatever the middleware sets (string uuid format).
        policy_name = f"{table}_by_user"
        op.execute(
            f"""
            CREATE POLICY {policy_name} ON {table}
                USING (user_id = current_setting('app.current_user_id', true)::uuid)
                WITH CHECK (user_id = current_setting('app.current_user_id', true)::uuid);
            """
        )


def downgrade() -> None:
    if not _is_postgres():
        return

    for table in _TABLES_WITH_RLS:
        policy_name = f"{table}_by_user"
        op.execute(f"DROP POLICY IF EXISTS {policy_name} ON {table};")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")
