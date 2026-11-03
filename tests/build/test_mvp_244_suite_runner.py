"""MVP-244/245: one pytest process per suite, registered markers, and Makefile targets that work."""
from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "ci" / "run_suites.py"
REQUIRED_SUITES = {"tests/solace_events", "tests/solace_infrastructure", "tests/security", "tests/migrations",
                   "tests/config_service", "tests/solace_testing", "tests/system", "tests/scripts",
                   "tests/integration", "tests/build", "tests/test_syntax_and_imports.py",
                   "infrastructure/api_gateway/tests", "services/user-service/tests",
                   "services/notification-service/tests", "services/analytics-service/tests"}


def test_mvp_245_markers_registered_in_pytest_ini() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "pytest.ini", encoding="utf-8")
    names = {re.split(r"[:(]", line)[0].strip() for line in cfg["pytest"]["markers"].splitlines() if line.strip()}
    assert {"realinfra", "e2e", "contract", "clinical", "profile"} <= names


def test_mvp_245_pyproject_has_no_competing_pytest_table() -> None:
    tool = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8")).get("tool", {})
    assert "pytest" not in tool


def test_mvp_244_makefile_exposes_if21_targets_without_timeout_flag() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("lint", "test-unit", "test-realinfra", "test-e2e", "test-clinical", "test-contract",
                   "images", "clean-checkout", "ci-local", "test"):
        assert re.search(rf"^{re.escape(target)}:", text, re.M), target
    assert "--timeout=60" not in text
    assert "tests/ services/" not in text


def test_mvp_244_runner_lists_every_suite() -> None:
    r = subprocess.run([sys.executable, str(RUNNER), "--list-json"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert REQUIRED_SUITES <= set(json.loads(r.stdout))


def test_mvp_244_unit_command_matches_global_constraint() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci.run_suites import build_command, child_env

    assert build_command("tests/solace_events", "unit") == [
        sys.executable, "-m", "pytest", "tests/solace_events", "-o", "addopts=", "-p", "no:logging", "-q",
        "-m", "not realinfra and not e2e"]
    env = child_env("unit", profile=None, require_realinfra=False)
    assert env["ENVIRONMENT"] == "test"
    assert env["PYTHONPATH"] == f"{ROOT}{os.pathsep}{ROOT / 'src'}"


def test_mvp_244_each_suite_runs_in_its_own_process(tmp_path: Path) -> None:
    for name, ok in (("a", True), ("b", False)):
        d = tmp_path / name
        d.mkdir()
        (d / f"test_{name}.py").write_text(
            "import os\n"
            f"def test_{name}():\n"
            f"    open(r'{tmp_path / (name + '.pid')}', 'w').write(str(os.getpid()))\n"
            "    assert os.environ['ENVIRONMENT'] == 'test'\n"
            f"    assert {ok}\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(RUNNER), "--tier", "unit", "--path", str(tmp_path / "a"),
                        "--path", str(tmp_path / "b")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 1
    lines = [l.rstrip() for l in r.stdout.splitlines()]
    assert any(l.startswith("PASS") and l.endswith(str(tmp_path / "a")) for l in lines), r.stdout
    assert any(l.startswith("FAIL") and l.endswith(str(tmp_path / "b")) for l in lines), r.stdout
    assert (tmp_path / "a.pid").read_text() != (tmp_path / "b.pid").read_text()
