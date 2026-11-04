"""System smoke tests for Solace-AI services (IF-01 liveness contract).

Uses FastAPI TestClient to verify that each underscore service's FastAPI app
imports and answers ``GET /live`` with 200 ``{"status": "alive"}`` without
Docker, Kafka or any other external dependency.

The client is used outside a ``with`` block on purpose: the IF-01 lifespan
(``service_runtime``) is not entered, so no datastore is touched. The
lifespan itself is covered by the bootstrap conformance tests and by the
compose boot smoke in ``tests/e2e/test_w0_boot_smoke.py``.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

UNDERSCORE_SERVICES = (
    "orchestrator_service",
    "safety_service",
    "diagnosis_service",
    "therapy_service",
    "memory_service",
    "personality_service",
)


class TestServiceLiveness:
    """Each underscore service answers GET /live with 200 {"status": "alive"}."""

    @pytest.mark.parametrize("package", UNDERSCORE_SERVICES)
    def test_service_live(self, package: str) -> None:
        app = importlib.import_module(f"services.{package}.src.main").app

        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/live")

        assert response.status_code == 200, f"{package} /live returned {response.status_code}"
        assert response.json() == {"status": "alive"}, f"{package} /live body is not the IF-01 liveness body"
