"""The local entrypoints (Makefile, ci_local) must call scripts that exist."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module_exists(dotted: str) -> bool:
    return (ROOT / (dotted.replace(".", "/") + ".py")).is_file()


def test_makefile_python_modules_exist():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    mods = re.findall(r"-m (scripts(?:\.\w+)+)", text)
    assert mods, "Makefile should call at least one scripts module"
    missing = [m for m in mods if not _module_exists(m)]
    assert not missing, f"Makefile calls modules that do not exist: {missing}"


def test_ci_local_stage_targets_exist():
    text = (ROOT / "scripts" / "ci" / "ci_local.py").read_text(encoding="utf-8")
    mods = re.findall(r'"-m", "(scripts(?:\.\w+)+)"', text)
    missing = [m for m in mods if not _module_exists(m)]
    assert not missing, f"ci_local calls modules that do not exist: {missing}"
    for rel in re.findall(r'"(scripts/ci/[\w]+\.py)"', text):
        assert (ROOT / rel).is_file(), rel


def test_makefile_images_target_uses_the_ci_builder():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "scripts/ci/build_images.py" in text
