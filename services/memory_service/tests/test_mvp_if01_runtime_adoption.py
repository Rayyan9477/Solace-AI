"""IF-01/IF-02/D-04 adoption for memory-service (WS-17.1)."""
from __future__ import annotations

import solace_events.schemas as canonical
from fastapi.testclient import TestClient

from services.memory_service.src import events as memory_events
from services.memory_service.src import main as memory_main


def test_if01_memory_spec_declares_every_flag_through_w5() -> None:
    spec = memory_main.SPEC
    assert spec.service_name == "memory-service"
    assert (spec.uses_postgres, spec.uses_redis) == (True, True)
    assert spec.event_plane == "publish_consume"
    assert (spec.handles_phi, spec.audited) == (True, True)
    assert (spec.rate_limited, spec.edge) == (False, False)


def test_if01_memory_main_has_no_local_platform_boilerplate() -> None:
    for name in ("configure_logging", "create_application",
                 "_register_middleware", "_register_exception_handlers"):
        assert not hasattr(memory_main, name), name


def test_if01_lifespan_registers_readiness_and_keeps_weaviate_off(monkeypatch) -> None:
    monkeypatch.delenv("WEAVIATE_ENABLED", raising=False)
    with TestClient(memory_main.app) as client:
        assert client.get("/live").status_code == 200
        checks = client.get("/ready").json()["checks"]
        service = memory_main.app.state.memory_service
    assert checks["memory_service"]["ok"] is True
    assert service._weaviate_repo is None  # D-04: constructed only when WEAVIATE_ENABLED=true


def test_mvp_270_memory_events_use_the_single_import_root() -> None:
    assert issubclass(memory_events.MemoryStoredEvent, canonical.BaseEvent)
