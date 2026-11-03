"""MVP-013/006/032: the runtime layout each Dockerfile builds imports its CMD target (replica, no Docker)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci import dockerfiles
from scripts.ci.images import IMAGES, probe_env

ROOT = Path(__file__).resolve().parents[2]
APP_IMAGES = [i for i in IMAGES if i.module]


def _replica(image, tmp_path: Path) -> tuple[Path, dict[str, str]]:
    df = dockerfiles.parse(ROOT / image.dockerfile)
    app_dir = dockerfiles.replicate_runtime_layout(df, ROOT, tmp_path / "app", dockerignore=ROOT / ".dockerignore")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(probe_env())
    env.update({"PYTHONPATH": dockerfiles.pythonpath_for(df, app_dir), "ENVIRONMENT": "test"})
    return app_dir, env


def _import(module: str, app_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    code = ("import importlib, json; m = importlib.import_module(%r); "
            "print(json.dumps({'file': getattr(m, '__file__', ''), 'pkg': m.__package__, "
            "'app': type(getattr(m, 'app', None)).__name__}))") % module
    return subprocess.run([sys.executable, "-c", code], cwd=app_dir, env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("image", APP_IMAGES, ids=lambda i: i.name)
def test_mvp_032_replica_imports_cmd_target_from_image_layout(image, tmp_path: Path) -> None:
    app_dir, env = _replica(image, tmp_path)
    r = _import(image.module, app_dir, env)
    assert r.returncode == 0, r.stderr[-3000:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert Path(info["file"]).resolve().is_relative_to(app_dir.resolve())
    assert info["app"] == "FastAPI"
    if image.merged_src:
        assert info["pkg"] == "src"  # MVP-006: the lifespan's package-relative imports resolve


@pytest.mark.parametrize("image", [i for i in APP_IMAGES if i.probes], ids=lambda i: i.name)
def test_mvp_013_replica_imports_probe_modules(image, tmp_path: Path) -> None:
    app_dir, env = _replica(image, tmp_path)
    for module in image.probes:
        r = _import(module, app_dir, env)
        assert r.returncode == 0, f"{image.name}: {module}: {r.stderr[-2000:]}"
