"""MVP-155 / MVP-006 (supporting): notification-service has exactly one module path, src.main."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src"
FLAT_ROOTS = {"domain", "api", "main", "consumers", "events", "infrastructure"}


def _flat_imports() -> list[str]:
    hits: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        rel = py.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] in FLAT_ROOTS:
                hits.append(f"{rel}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                hits += [f"{rel}:{node.lineno}:{a.name}" for a in node.names if a.name.split(".")[0] in FLAT_ROOTS]
    return hits


def test_mvp_155_no_flat_import_fallbacks() -> None:
    assert _flat_imports() == []


def test_mvp_155_every_module_imports_under_the_image_path() -> None:
    for name in ("src.main", "src.api", "src.consumers", "src.events",
                 "src.domain.channels", "src.infrastructure.event_bridge"):
        assert importlib.import_module(name).__name__ == name


def test_mvp_155_real_lifespan_runs_under_the_image_path(monkeypatch) -> None:
    monkeypatch.setenv("KAFKA_ENABLED", "false")
    import src.main as main
    with TestClient(main.app) as client:
        assert client.get("/api/v1/notifications/health").status_code == 200
