"""C.2 — Extend Row-Level Security to the remaining PHI tables (migration 006).

Migration ``006_extend_rls_phi_tables`` mirrors ``002_enable_rls_clinical_tables``:
it enables ``ROW LEVEL SECURITY`` + ``FORCE ROW LEVEL SECURITY`` and installs a
``{table}_by_user`` policy keyed on the per-request ``app.current_user_id`` GUC for
every remaining PHI table that carries a real ``user_id`` column.

These tests lock two invariants without a live Postgres:

  (a) the migration targets EXACTLY the agreed set of user-scoped PHI tables, and
  (b) both ``upgrade()`` and ``downgrade()`` are a strict no-op on non-PostgreSQL
      dialects (RLS DDL is Postgres-only; SQLite-backed test runs must not emit it).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MIGRATION_PATH = _ROOT / "migrations" / "versions" / "006_extend_rls_phi_tables.py"


def _load_migration():
    # Leading-digit filename can't be a normal import; load it standalone.
    spec = importlib.util.spec_from_file_location("migration_006_under_test", _MIGRATION_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mig = _load_migration()


# Only tables written under an authenticated USER request (get_current_user -> GUC set)
# are RLS-safe now. Service-auth'd (safety_*, personality_*) and background-written
# (session_summaries, notifications, ...) tables are DEFERRED until their writers set the
# GUC — enabling RLS on them would silently break those writers on Postgres.
_EXPECTED_TABLES = {
    "diagnosis_symptoms",
    "diagnosis_hypotheses",
    "diagnosis_records",
    "treatment_plans",
    "therapy_interventions",
    "homework_assignments",
    "consent_records",
    "user_preferences",
    "user_notification_preferences",
}

# Tables that MUST NOT be RLS'd in this pass (their writers do not set the GUC).
_DEFERRED_TABLES = {
    "safety_assessments", "safety_plans", "risk_factors", "contraindication_checks",
    "safety_events", "personality_profiles", "trait_assessments", "profile_snapshots",
    "session_summaries", "therapeutic_events", "memory_user_profiles",
    "notifications", "escalations", "oauth_accounts", "delivery_attempts",
}


class _FakeDialect:
    name = "sqlite"


class _FakeBind:
    dialect = _FakeDialect()


class _PgDialect:
    name = "postgresql"


class _PgBind:
    dialect = _PgDialect()


def test_migration_targets_exactly_the_expected_phi_tables() -> None:
    assert set(mig._TABLES_WITH_RLS) == _EXPECTED_TABLES
    # No duplicates crept into the list.
    assert len(mig._TABLES_WITH_RLS) == len(_EXPECTED_TABLES)


def test_deferred_tables_are_not_rls_enabled() -> None:
    # Enabling RLS on a table whose writer doesn't set the GUC breaks that writer on PG.
    for table in _DEFERRED_TABLES:
        assert table not in mig._TABLES_WITH_RLS, f"{table} must be deferred (writer sets no GUC)"


def test_upgrade_is_idempotent_drop_before_create_on_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr(mig.context, "get_bind", lambda: _PgBind())
    monkeypatch.setattr(mig.op, "execute", lambda sql: executed.append(sql))

    mig.upgrade()

    for table in mig._TABLES_WITH_RLS:
        pol = f"{table}_by_user"
        joined = "\n".join(executed)
        assert f"DROP POLICY IF EXISTS {pol} ON {table};" in executed  # idempotent
        assert f"CREATE POLICY {pol} ON {table}" in joined
        drop_i = executed.index(f"DROP POLICY IF EXISTS {pol} ON {table};")
        create_i = next(i for i, s in enumerate(executed) if f"CREATE POLICY {pol}" in s)
        assert drop_i < create_i  # DROP precedes CREATE


def test_revision_chains_onto_current_head() -> None:
    assert mig.revision == "006_extend_rls"
    assert mig.down_revision == "005_widen_phi_columns"


def test_upgrade_is_noop_on_non_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr(mig.context, "get_bind", lambda: _FakeBind())
    monkeypatch.setattr(mig.op, "execute", lambda sql: executed.append(sql))

    mig.upgrade()

    assert executed == []


def test_downgrade_is_noop_on_non_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []
    monkeypatch.setattr(mig.context, "get_bind", lambda: _FakeBind())
    monkeypatch.setattr(mig.op, "execute", lambda sql: executed.append(sql))

    mig.downgrade()

    assert executed == []


def test_is_postgres_false_for_non_postgres_dialect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mig.context, "get_bind", lambda: _FakeBind())
    assert mig._is_postgres() is False
