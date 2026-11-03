# tests/migrations/test_mvp_041_slot_chain.py
"""MVP-041/058 (IF-03): one linear chain with fixed slot ids; 009 covers every user policy; schema-head semantics."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import asyncpg
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ROOT / "migrations" / "versions"
EXPECTED_CHAIN = ["001_initial", "002_enable_rls", "003_escalations", "004_oauth_accounts", "005_widen_phi_columns",
                  "006_extend_rls", "007_outbox_claimed_at", "008_rls_personality", "009_platform_roles_rls",
                  "010_event_plane", "011_audit_trail", "012_user_accounts", "013_identity_mfa", "014_safety",
                  "015_notification", "016_orchestrator_threads", "017_orchestrator_checkpoints", "018_memory",
                  "019_therapy", "020_diagnosis", "021_personality_analytics"]
SLOT_OWNERS = {"009_platform_roles_rls": "WS-02", "010_event_plane": "WS-06", "011_audit_trail": "WS-09",
               "012_user_accounts": "WS-14", "013_identity_mfa": "WS-08", "014_safety": "WS-12",
               "015_notification": "WS-13", "016_orchestrator_threads": "WS-11",
               "017_orchestrator_checkpoints": "WS-10", "018_memory": "WS-17", "019_therapy": "WS-15",
               "020_diagnosis": "WS-16", "021_personality_analytics": "WS-18"}


def _load(filename: str):
    spec = importlib.util.spec_from_file_location(f"mig_{filename[:3]}_under_test", VERSIONS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_chain_is_linear_and_ends_at_schema_head() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    script = ScriptDirectory.from_config(cfg)
    assert [r.revision for r in reversed(list(script.walk_revisions("base", "heads")))] == EXPECTED_CHAIN
    from solace_infrastructure.database.schema_check import SCHEMA_HEAD

    assert script.get_heads() == [SCHEMA_HEAD] == ["021_personality_analytics"]


def test_slot_ids_fit_the_alembic_version_column() -> None:
    assert all(len(rev) <= 32 for rev in EXPECTED_CHAIN)


@pytest.mark.parametrize("slot,owner", sorted(SLOT_OWNERS.items()))
def test_slot_file_named_by_revision_and_owned(slot: str, owner: str) -> None:
    text = (VERSIONS / f"{slot}.py").read_text(encoding="utf-8")
    previous = EXPECTED_CHAIN[EXPECTED_CHAIN.index(slot) - 1]
    assert f'revision = "{slot}"' in text and f'down_revision = "{previous}"' in text
    assert f"Owner: {owner}" in text


def test_mvp_058_009_rewrites_every_existing_user_policy() -> None:
    existing = set()
    for name in ("002_enable_rls_clinical_tables.py", "006_extend_rls_phi_tables.py", "008_rls_personality_tables.py"):
        existing |= set(_load(name)._TABLES_WITH_RLS)
    m009 = _load("009_platform_roles_rls.py")
    assert len(existing) == 15 and set(m009.POLICY_TABLES) == existing
    assert m009.NULLIF_PREDICATE == "user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid"


class _Pool:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    async def fetchval(self, sql: str):
        if self._exc:
            raise self._exc
        return self._result


async def test_assert_schema_head_raises_only_in_strict_environments() -> None:
    from solace_infrastructure.database.schema_check import SchemaVersionMismatch, assert_schema_head

    with pytest.raises(SchemaVersionMismatch) as exc:
        await assert_schema_head(_Pool("008_rls_personality"), environment="staging")
    assert (exc.value.expected, exc.value.found) == ("021_personality_analytics", "008_rls_personality")
    assert await assert_schema_head(_Pool("008_rls_personality"), environment="development") == "008_rls_personality"
    with pytest.raises(SchemaVersionMismatch):
        await assert_schema_head(_Pool(exc=asyncpg.exceptions.UndefinedTableError("alembic_version")), environment="production")
    assert await assert_schema_head(_Pool("021_personality_analytics"), environment="production") == "021_personality_analytics"
    with pytest.raises(SchemaVersionMismatch) as newer:  # a restore onto a database migrated past this release
        await assert_schema_head(_Pool("022_future_slot"), environment="production")
    assert newer.value.found == "022_future_slot"
    denied = asyncpg.exceptions.InsufficientPrivilegeError("permission denied for table alembic_version")
    with pytest.raises(SchemaVersionMismatch) as before_009:  # before 009 solace_app cannot read alembic_version
        await assert_schema_head(_Pool(exc=denied), environment="staging")
    assert (before_009.value.expected, before_009.value.found) == ("021_personality_analytics", None)
    assert await assert_schema_head(_Pool(exc=denied), environment="development") is None


@pytest.mark.parametrize("slot", EXPECTED_CHAIN[EXPECTED_CHAIN.index("010_event_plane"):])
def test_slot_docstring_states_the_rls_rule(slot: str) -> None:
    """MVP-041 fix 1: every slot tells its owner how a user_id table and a view stay RLS-safe."""
    doc = " ".join((ast.get_docstring(ast.parse((VERSIONS / f"{slot}.py").read_text(encoding="utf-8"))) or "").split())
    assert "``user_id`` column must ENABLE and FORCE row level security" in doc
    assert "``RLS_DEFERRED`` in ``tests/migrations/test_chain_real_pg.py`` with a reason" in doc
    assert "``WITH (security_invoker = true)``" in doc
