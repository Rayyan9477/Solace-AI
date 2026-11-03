"""MVP-133: the seven bridges import bare solace_events and stamp their own service."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BRIDGES = {
    "services/orchestrator_service/src/infrastructure/event_bridge.py": "orchestrator-service",
    "services/safety_service/src/infrastructure/event_bridge.py": "safety-service",
    "services/therapy_service/src/infrastructure/event_bridge.py": "therapy-service",
    "services/diagnosis_service/src/infrastructure/event_bridge.py": "diagnosis-service",
    "services/personality_service/src/infrastructure/event_bridge.py": "personality-service",
    "services/user-service/src/infrastructure/event_bridge.py": "user-service",
    "services/notification-service/src/infrastructure/event_bridge.py": "notification-service",
}
_SERVICE_ARG = {"create_publisher": "source_service", "get_or_create_service_event_plane": "service_name"}


def _tree(path):
    return ast.parse((REPO / path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", sorted(BRIDGES))
def test_mvp_133_bridge_imports_bare_solace_events(path):
    mods = {n.module for n in ast.walk(_tree(path)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not {m for m in mods if m.startswith("src.solace_events")}
    assert any(m.startswith("solace_events") for m in mods)


@pytest.mark.parametrize("path,service", sorted(BRIDGES.items()))
def test_mvp_133_bridge_names_its_service(path, service):
    values = []
    for c in ast.walk(_tree(path)):
        if isinstance(c, ast.Call):
            name = getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            if name in _SERVICE_ARG:
                values += [
                    k.value.value
                    for k in c.keywords
                    if k.arg == _SERVICE_ARG[name] and isinstance(k.value, ast.Constant)
                ]
                values += [a.value for a in c.args[:1] if isinstance(a, ast.Constant)]
    assert values and set(values) == {service}
