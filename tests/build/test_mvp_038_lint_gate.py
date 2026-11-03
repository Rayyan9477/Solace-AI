"""MVP-038: the CI lint gate (ruff E9,F63,F7,F82) is clean; each remaining F821 is owned by one workstream."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LINT_OWNERS = {
    "WS-02": ("src/solace_infrastructure/postgres.py",),
    "WS-13": ("services/notification-service/src/main.py",),
    "WS-18": ("services/analytics-service/src/main.py", "services/analytics-service/src/config.py"),
}


def _ruff() -> list[str]:
    """`python -m ruff`, else the ruff on PATH (a --system-site-packages venv's launcher finds no binary)."""
    if subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True).returncode == 0:
        return [sys.executable, "-m", "ruff"]
    exe = shutil.which("ruff")
    assert exe, "ruff is not runnable; requirements-dev.lock pins it"
    return [exe]


def _findings() -> dict[str, list[str]]:
    r = subprocess.run([*_ruff(), "check", str(ROOT), "--select", "E9,F63,F7,F82",
                        "--no-fix", "--no-cache", "--output-format", "json"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, 1), r.stderr
    out: dict[str, list[str]] = {}
    # No "[]" fallback: ruff prints "[]" when clean, so empty stdout means ruff never ran (a crash also exits 1).
    for f in json.loads(r.stdout):
        rel = Path(f["filename"]).resolve().relative_to(ROOT).as_posix()
        out.setdefault(rel, []).append(f"{f['location']['row']}:{f['code']}")
    return out


@pytest.mark.parametrize("owner", sorted(LINT_OWNERS))
def test_mvp_038_lint_gate_clean(owner: str) -> None:
    assert {f: v for f, v in _findings().items() if f in LINT_OWNERS[owner]} == {}


def test_mvp_038_lint_gate_clean_outside_registered_files() -> None:
    known = {f for files in LINT_OWNERS.values() for f in files}
    assert {f: v for f, v in _findings().items() if f not in known} == {}
