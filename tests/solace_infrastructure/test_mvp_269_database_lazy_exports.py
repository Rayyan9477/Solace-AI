"""MVP-269 (supporting): the database package no longer drags weaviate and alembic into every service."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEAVY = ("weaviate", "alembic", "solace_infrastructure.weaviate", "solace_infrastructure.redis",
         "solace_infrastructure.database.migrations_runner", "solace_infrastructure.database.weaviate_schema",
         "solace_infrastructure.database.seed_data", "solace_infrastructure.database.redis_setup")


def _loaded(import_line: str) -> list[str]:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "test"}
    code = f"import json, sys\n{import_line}\nprint(json.dumps(sorted(m for m in sys.modules if m.startswith({HEAVY!r}))))"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", ["solace_infrastructure.database.connection_manager",
                                    "solace_infrastructure.database.base_models"])
def test_mvp_269_database_submodule_import_skips_optional_backends(module: str) -> None:
    assert _loaded(f"import {module}") == []


def test_mvp_269_database_exports_still_resolve() -> None:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "test"}
    code = ("import solace_infrastructure.database as d\n"
            "assert len(d.__all__) == 58 and {'ConnectionPoolManager', 'Base', 'MigrationRunner', 'setup_redis'} <= set(d.__all__)\n"
            "print([n for n in d.__all__ if getattr(d, n, None) is None])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[]"
