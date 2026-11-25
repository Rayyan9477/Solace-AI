"""The local entrypoints (Makefile, ci_local) must call scripts that exist, and ci_local must run CI's image stage."""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from scripts.ci import ci_local

ROOT = Path(__file__).resolve().parents[2]


def _module_exists(dotted: str) -> bool:
    return (ROOT / (dotted.replace(".", "/") + ".py")).is_file()


def _stage_targets(stages: Iterable[tuple[str, Sequence[str]]]) -> tuple[int, list[str]]:
    """(targets checked, problems) over the real stage commands: every *.py argument is an existing file and
    every ``-m scripts.X`` module resolves to scripts/X.py."""
    checked, problems = 0, []
    for name, cmd in stages:
        for i, arg in enumerate(cmd):
            if arg.endswith(".py"):
                checked += 1
                path = Path(arg) if Path(arg).is_absolute() else ROOT / arg
                if not path.is_file():
                    problems.append(f"{name}: script {arg} does not exist")
            elif arg == "-m" and i + 1 < len(cmd) and cmd[i + 1].startswith("scripts."):
                checked += 1
                if not _module_exists(cmd[i + 1]):
                    problems.append(f"{name}: module {cmd[i + 1]} does not exist")
    return checked, problems


def test_makefile_python_modules_exist():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    mods = re.findall(r"-m (scripts(?:\.\w+)+)", text)
    assert mods, "Makefile should call at least one scripts module"
    missing = [m for m in mods if not _module_exists(m)]
    assert not missing, f"Makefile calls modules that do not exist: {missing}"
    paths = re.findall(r"(scripts/[\w/]+\.py)", text)
    assert paths, "Makefile should call at least one scripts/*.py file"
    assert not [p for p in paths if not (ROOT / p).is_file()], f"Makefile calls missing scripts: {paths}"


def test_ci_local_stage_targets_exist():
    script_stages = [name for name, cmd in ci_local.STAGES if any(a.endswith(".py") or a.startswith("scripts.") for a in cmd)]
    checked, problems = _stage_targets(ci_local.STAGES)
    assert checked >= len(script_stages) > 0, "the check must see the stage commands' targets"
    assert problems == [], problems


def test_the_stage_check_catches_a_missing_script_or_module():
    bad = (
        ("images", ["py", str(ROOT / "scripts" / "ci" / "build_imagez_does_not_exist.py"), "--all"]),
        ("relative", ["py", "scripts/ci/nope.py"]),
        ("module", ["py", "-m", "scripts.build.images"]),
    )
    checked, problems = _stage_targets(bad)
    assert checked == 3
    assert len(problems) == 3, problems


def test_ci_local_images_stage_runs_ci_image_probe():
    """CI's images job runs build_images.py --tag ci --probe (the in-image import of each CMD module)."""
    ci_line = next(line for line in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8").splitlines()
                   if "scripts/ci/build_images.py --image" in line)
    assert "--tag ci" in ci_line and "--probe" in ci_line, ci_line
    images = dict(ci_local.STAGES)["images"]
    assert Path(images[1]).name == "build_images.py"
    assert "--all" in images and "--probe" in images, images
    assert images[images.index("--tag") + 1] == "ci", images


def test_makefile_images_target_uses_the_ci_builder():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "scripts/ci/build_images.py" in text
