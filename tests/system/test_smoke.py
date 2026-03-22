"""System smoke tests for Solace-AI services.

Uses FastAPI TestClient to verify basic health endpoints without
requiring Docker, Kafka, or external dependencies. These tests
confirm that each service's FastAPI application boots and responds
to health checks.
"""

import pytest
from fastapi.testclient import TestClient


class TestServiceHealth:
    """Smoke tests verifying each service starts and exposes /health."""

    def test_safety_service_health(self) -> None:
        """Verify the safety service FastAPI app responds to GET /health with 200.

        The safety service is always-active and must be reachable for crisis
        detection to function.
        """
        from services.safety_service.src.main import app

        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/health")

        assert response.status_code == 200, (
            f"Safety service /health returned {response.status_code}"
        )
        data = response.json()
        assert data.get("status") == "healthy", (
            f"Safety service health status is '{data.get('status')}', expected 'healthy'"
        )
        assert data.get("service") == "safety-service", (
            f"Safety service name is '{data.get('service')}', expected 'safety-service'"
        )

    def test_diagnosis_service_health(self) -> None:
        """Verify the diagnosis service FastAPI app responds to GET /health with 200.

        The diagnosis service provides AMIE-inspired diagnostic assessment
        and must be available for clinical workflows.
        """
        from services.diagnosis_service.src.main import app

        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/health")

        assert response.status_code == 200, (
            f"Diagnosis service /health returned {response.status_code}"
        )
        data = response.json()
        assert data.get("status") == "healthy", (
            f"Diagnosis service health status is '{data.get('status')}', expected 'healthy'"
        )
        assert data.get("service") == "diagnosis-service", (
            f"Diagnosis service name is '{data.get('service')}', expected 'diagnosis-service'"
        )

    def test_therapy_service_health(self) -> None:
        """Verify the therapy service FastAPI app responds to GET /health with 200.

        The therapy service delivers evidence-based interventions (CBT, DBT,
        ACT, MI) and must respond to health checks for load balancing.
        """
        from services.therapy_service.src.main import app

        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/health")

        assert response.status_code == 200, (
            f"Therapy service /health returned {response.status_code}"
        )
        data = response.json()
        assert data.get("status") == "healthy", (
            f"Therapy service health status is '{data.get('status')}', expected 'healthy'"
        )
        assert data.get("service") == "therapy-service", (
            f"Therapy service name is '{data.get('service')}', expected 'therapy-service'"
        )
