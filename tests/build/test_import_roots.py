"""MVP-270: shared packages are imported only as top-level solace_* (never src.solace_*)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN = ("src", "services", "migrations", "scripts", "infrastructure", "tests")
SKIP = ("services/config_service/", "tests/build/test_import_roots.py")
BRIDGES = tuple(f"services/{s}/src/infrastructure/event_bridge.py" for s in (
    "orchestrator_service", "safety_service", "therapy_service", "diagnosis_service",
    "personality_service", "user-service", "notification-service"))
OWNERS: dict[str, tuple[str, ...]] = {
    "WS-01": ("src/__init__.py",),
    "WS-02": ("migrations/env.py",),
    "WS-05": ("tests/integration/test_persistence_roundtrip.py",),
    "WS-06": BRIDGES,
    "WS-10": ("services/orchestrator_service/src/events.py",),
    "WS-12": ("services/safety_service/src/events.py",),
    "WS-13": ("services/notification-service/src/events.py",),
    "WS-14": ("services/user-service/src/events.py",),
    "WS-15": ("services/therapy_service/src/events.py",),
    "WS-16": ("services/diagnosis_service/src/events.py",),
    "WS-17": ("services/memory_service/src/events.py",),
    "WS-18": ("services/personality_service/src/events.py", "services/analytics-service/src/main.py"),
}
PATCH_CALLS = {"import_module", "patch", "object"}


def _scan_source(source: str, rel: str = "x.py") -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module == "src":
                names = [f"src.{a.name}" for a in node.names]
            elif node.level == 0 and node.module:
                names = [node.module]
            elif node.level == 1 and rel == "src/__init__.py":
                names = [f"src.{node.module or a.name}" for a in node.names]
        elif isinstance(node, ast.Call):
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if func in PATCH_CALLS and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                names = [node.args[0].value]
        hits += [f"{getattr(node, 'lineno', 0)}:{n}" for n in names if n.startswith("src.solace_")]
    return hits


def _violations() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for base in SCAN:
        for path in (ROOT / base).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(SKIP) or "/__pycache__/" in rel:
                continue
            # utf-8-sig: Python accepts a BOM-prefixed source file, but ast.parse(str) rejects U+FEFF.
            hits = _scan_source(path.read_text(encoding="utf-8-sig"), rel)
            if hits:
                out[rel] = hits
    return out


def test_mvp_270_scanner_detects_every_form() -> None:
    for src in ("import src.solace_events", "from src.solace_events.config import X",
                "from src import solace_common", "importlib.import_module('src.solace_security.auth')",
                "patch('src.solace_events.publisher.EventPublisher')"):
        assert _scan_source(src), src
    assert _scan_source("from . import solace_common", "src/__init__.py")
    assert not _scan_source("from solace_events.config import X")


@pytest.mark.parametrize("owner", [
    pytest.param(o, marks=pytest.mark.xfail(strict=True, reason="WS-10.3 (W1 phase B) rewrites the import; marker removed by Ruling"))
    if o == "WS-10" else o for o in sorted(OWNERS)])
def test_mvp_270_no_src_root_imports(owner: str) -> None:
    found = {f: h for f, h in _violations().items() if f in OWNERS[owner]}
    assert found == {}


def test_mvp_270_no_src_root_imports_outside_registered_files() -> None:
    known = {f for files in OWNERS.values() for f in files}
    assert {f: h for f, h in _violations().items() if f not in known} == {}
