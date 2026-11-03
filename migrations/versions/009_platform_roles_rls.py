"""Platform roles: DML grants for ``solace_app`` and NULL-safe per-user RLS policies (MVP-041, MVP-058).

Revision ID: 009_platform_roles_rls
Revises: 008_rls_personality
Create Date: 2026-11-02

Owner: WS-02

The services connect as ``solace_app`` (NOSUPERUSER NOBYPASSRLS, created by
``infrastructure/init-db/01-roles.sh``), while ``solace_migrator`` owns every object the
chain creates (D-02). Before this revision ``solace_app`` held no privilege on any table,
so every service query failed with ``permission denied``. This revision grants it:

* ``USAGE`` on schema ``public``, DML (``SELECT, INSERT, UPDATE, DELETE``) on every table
  and ``USAGE, SELECT`` on every sequence that exists now. No DDL: ``solace_app`` owns
  nothing and has no ``CREATE`` on ``public``.
* ``SELECT`` only on ``alembic_version``, so the startup schema-head check
  (``solace_infrastructure.database.schema_check``) runs as the app role, which cannot
  rewrite the recorded revision.
* The same DML through ``ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator``, so every table
  and sequence a later slot (010 to 021) creates is usable by the app role without a grant
  of its own. A slot that needs less (for example the append-only ``audit_events``) revokes
  the extra privileges itself. Migrations must therefore run as ``solace_migrator``.

It also rewrites the ``<table>_by_user`` policy of all 15 tables that 002, 006 and 008 put
under RLS. Their predicate ``user_id = current_setting('app.current_user_id', true)::uuid``
raised ``invalid input syntax for type uuid: ""`` once a pooled connection's GUC had been
reset to ``''``. The predicate is now ``NULLIF_PREDICATE``: an unset or reset GUC compares
``user_id`` with NULL, so a read returns no rows and a write fails the WITH CHECK, which is
the fail-closed default. New per-user policies in later slots use the same predicate.

Postgres-only and a no-op on other dialects, like 002 to 008. ``downgrade()`` restores the
original predicate and then revokes in reverse order.
"""
from __future__ import annotations

from alembic import context, op

# Alembic identifiers
revision = "009_platform_roles_rls"
down_revision = "008_rls_personality"
branch_labels = None
depends_on = None

# The 15 tables whose ``<table>_by_user`` policy was created by 002, 006 and 008.
POLICY_TABLES: tuple[str, ...] = (
    # 002_enable_rls
    "diagnosis_sessions",
    "therapy_sessions",
    "memory_records",
    # 006_extend_rls
    "diagnosis_symptoms",
    "diagnosis_hypotheses",
    "diagnosis_records",
    "treatment_plans",
    "therapy_interventions",
    "homework_assignments",
    "consent_records",
    "user_preferences",
    "user_notification_preferences",
    # 008_rls_personality
    "personality_profiles",
    "trait_assessments",
    "profile_snapshots",
)

NULLIF_PREDICATE = "user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid"
# The predicate 002, 006 and 008 created; downgrade() puts it back.
_ORIGINAL_PREDICATE = "user_id = current_setting('app.current_user_id', true)::uuid"

# Plain literals (role names are the fixed IF-03 names), executed in this order.
_GRANTS: tuple[str, ...] = (
    "GRANT USAGE ON SCHEMA public TO solace_app",
    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO solace_app",
    "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO solace_app",
    "REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON alembic_version FROM solace_app",
    "ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator IN SCHEMA public "
    "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO solace_app",
    "ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator IN SCHEMA public "
    "GRANT USAGE, SELECT ON SEQUENCES TO solace_app",
)

# Reverse order of _GRANTS. The alembic_version REVOKE needs no inverse: the table-wide
# REVOKE below removes the SELECT that is left.
_REVOKES: tuple[str, ...] = (
    "ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator IN SCHEMA public "
    "REVOKE USAGE, SELECT ON SEQUENCES FROM solace_app",
    "ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator IN SCHEMA public "
    "REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM solace_app",
    "REVOKE USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public FROM solace_app",
    "REVOKE SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public FROM solace_app",
    "REVOKE USAGE ON SCHEMA public FROM solace_app",
)


def _is_postgres() -> bool:
    """Check we're on Postgres. Roles, grants and RLS statements are Postgres-only."""
    bind = context.get_bind()
    return bind is not None and bind.dialect.name == "postgresql"


def _alter_policy(table: str, predicate: str) -> str:
    return f"ALTER POLICY {table}_by_user ON {table} USING ({predicate}) WITH CHECK ({predicate})"


def upgrade() -> None:
    if not _is_postgres():
        return
    for statement in _GRANTS:
        op.execute(statement)
    for table in POLICY_TABLES:
        op.execute(_alter_policy(table, NULLIF_PREDICATE))


def downgrade() -> None:
    if not _is_postgres():
        return
    for table in reversed(POLICY_TABLES):
        op.execute(_alter_policy(table, _ORIGINAL_PREDICATE))
    for statement in _REVOKES:
        op.execute(statement)
