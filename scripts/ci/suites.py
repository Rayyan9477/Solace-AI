"""IF-21: the list of test suites, each run in its own pytest process. Stdlib only."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Suite:
    path: str
    tier: str = "unit"


_SERVICES = (
    "analytics-service",
    "diagnosis_service",
    "memory_service",
    "notification-service",
    "orchestrator_service",
    "personality_service",
    "safety_service",
    "therapy_service",
    "user-service",
)

_PATHS = (
    "tests/test_syntax_and_imports.py",
    "tests/alignment",
    "tests/solace_common",
    "tests/solace_events",
    "tests/solace_security",
    "tests/solace_infrastructure",
    "tests/solace_testing",
    "tests/integration",
    "tests/security",
    "tests/migrations",
    "tests/scripts",
    "tests/system",
    "tests/config_service",
    "tests/bootstrap",
    "tests/build",
    "tests/deploy",
    "tests/contracts",
    "tests/clinical",
    "infrastructure/api_gateway/tests",
) + tuple(f"services/{s}/tests" for s in _SERVICES)

SUITES: tuple[Suite, ...] = tuple(Suite(p) for p in _PATHS)

REALINFRA_PATHS = ("tests/realinfra", "tests/migrations", "tests/build")
