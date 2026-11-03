"""Test double for the IF-01 ServiceRuntime (helper module, not collected)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def fake_runtime(
    *,
    redis: Any = None,
    pg_pool: Any = None,
    event_plane: Any = None,
    phi_codec: Any = None,
    environment: str = "test",
) -> SimpleNamespace:
    """Stand-in for ``ServiceRuntime`` exposing the registration surface a composition uses."""
    readiness_checks: dict[str, tuple[Any, bool]] = {}
    shutdowns: dict[str, Any] = {}

    def register_readiness_check(name: str, check: Any, *, critical: bool = True) -> None:
        readiness_checks[name] = (check, critical)

    def register_shutdown(name: str, closer: Any) -> None:
        shutdowns[name] = closer

    return SimpleNamespace(
        redis=redis,
        pg_pool=pg_pool,
        event_plane=event_plane,
        phi_codec=phi_codec,
        spec=SimpleNamespace(environment=environment, service_name="therapy-service"),
        readiness_checks=readiness_checks,
        shutdowns=shutdowns,
        register_readiness_check=register_readiness_check,
        register_shutdown=register_shutdown,
    )
