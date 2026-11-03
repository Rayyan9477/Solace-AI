"""MVP-005: ``import tests`` must resolve to the repo's tests/ package.

portkey-ai installs a top-level ``tests`` package into site-packages. Without
``tests/__init__.py`` the repo's ``tests`` is a namespace package and loses to it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_mvp_005_tests_package_resolves_to_the_repo() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "src")])
    out = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util as u; print(u.find_spec('tests').origin)"],
        capture_output=True, text=True, env=env, cwd=str(REPO), check=True,
    ).stdout.strip()
    assert out and out != "None", f"tests has no file origin: {out!r}"
    origin = Path(out).resolve()
    assert REPO in origin.parents, f"tests resolved outside the repo: {origin}"
