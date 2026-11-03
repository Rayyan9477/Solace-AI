"""MVP-008: every planned platform path must be visible to git (checkpoint trees and their git-archive builds; D-17: no commits)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

MUST_TRACK = [
    "tests/build/test_mvp_008_git_visibility.py",
    "tests/build/conftest.py",
    "deploy/env/env-contract.yaml",
    "deploy/env/render_env_example.py",
    "deploy/env/generate_dev_env.py",
    "services/orchestrator_service/src/langgraph/checkpointer.py",
    "migrations/versions/017_orchestrator_checkpoints.py",
    "docker/entrypoint.sh",
    "docker/migrate.Dockerfile",
    "scripts/ci/run_suites.py",
    "scripts/db/migrate.py",
    "infrastructure/init-db/01-roles.sh",
    "requirements.lock",
    "requirements-dev.lock",
    "src/solace_infrastructure/runtime/bootstrap.py",
    "tests/e2e/expected_failures.yaml",
    "tests/realinfra/data-platform-schema/test_x.py",
]
MUST_IGNORE = [
    ".env", ".env.prod", "deploy/env/.env.dev", "secrets/x.txt", "server.key", "postgres-ca.crt",
    ".venv/x", "venv/x", "env/bin/python", "build/lib/x.py", "__pycache__/x.pyc",
    "tests/logs/pytest.log", "checkpoint/model.bin", "checkpoint-500/weights.bin", "model.ckpt",
]


def _ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", "-q", path],
        capture_output=True, text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


@pytest.mark.parametrize("path", MUST_TRACK)
def test_mvp_008_planned_path_is_not_gitignored(path: str) -> None:
    assert not _ignored(path), f"{path} is gitignored, so checkpoint trees and their git-archive builds would drop it"


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_mvp_008_secret_and_artifact_paths_stay_ignored(path: str) -> None:
    assert _ignored(path), f"{path} must stay ignored"
