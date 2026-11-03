"""MVP-038/112/246: CI runs every suite from the same registry as make ci-local, and builds images from the repo root."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _wf() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def _runs(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


def test_mvp_038_jobs_exist_and_tests_do_not_wait_for_lint() -> None:
    jobs = _wf()["jobs"]
    assert {"plan", "lint", "unit", "realinfra", "images", "e2e"} <= set(jobs)
    for name in ("unit", "realinfra", "images"):
        needs = jobs[name].get("needs", [])
        assert "lint" not in (needs if isinstance(needs, list) else [needs]), name


def test_mvp_038_unit_matrix_comes_from_the_suite_registry() -> None:
    jobs = _wf()["jobs"]
    assert "run_suites.py --list-json" in _runs(jobs["plan"])
    assert jobs["unit"]["strategy"]["matrix"]["suite"] == "${{ fromJSON(needs.plan.outputs.suites) }}"
    assert "run_suites.py --tier unit --suite" in _runs(jobs["unit"])
    assert "--tier realinfra --require-realinfra" in _runs(jobs["realinfra"])


def test_mvp_038_installs_only_from_the_hashed_dev_lock() -> None:
    text = CI.read_text(encoding="utf-8")
    assert 'pip install -e' not in text and '.[dev]' not in text
    for name, job in _wf()["jobs"].items():
        if name != "plan":
            assert "uv pip sync --system --require-hashes requirements-dev.lock" in _runs(job), name


def test_mvp_246_no_suite_or_file_is_ignored() -> None:
    assert "--ignore" not in CI.read_text(encoding="utf-8")


def test_mvp_112_images_build_from_repo_root_with_probe() -> None:
    jobs = _wf()["jobs"]
    assert "build_images.py --list-json" in _runs(jobs["plan"])
    assert jobs["images"]["strategy"]["matrix"]["image"] == "${{ fromJSON(needs.plan.outputs.images) }}"
    assert re.search(r"build_images\.py --image \"?\$\{\{ matrix\.image \}\}\"? --tag ci --probe", _runs(jobs["images"]))
    assert "./services/" not in CI.read_text(encoding="utf-8")


def test_mvp_038_lint_gate_and_e2e_trigger() -> None:
    jobs = _wf()["jobs"]
    assert "ruff check . --select E9,F63,F7,F82 --no-fix" in _runs(jobs["lint"])
    assert "workflow_dispatch" in jobs["e2e"]["if"] and "schedule" in jobs["e2e"]["if"]
    triggers = _wf().get("on", _wf().get(True))
    assert {"push", "pull_request", "workflow_dispatch", "schedule"} <= set(triggers)
    assert "ci-test-secret" not in CI.read_text(encoding="utf-8")
