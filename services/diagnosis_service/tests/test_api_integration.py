"""
Comprehensive API Integration Tests for Diagnosis Service Batch 5.1.
Tests all endpoints, error handling, and full diagnostic workflows.
"""
from __future__ import annotations
import pytest
import pytest_asyncio
from decimal import Decimal
from uuid import uuid4
from typing import AsyncIterator
import httpx
from httpx import ASGITransport
from services.diagnosis_service.src.main import app
from services.diagnosis_service.src.schemas import DiagnosisPhase
from solace_security.middleware import AuthenticatedUser, get_current_user
from solace_security.auth import TokenType


def _mock_user() -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id="test-user",
        token_type=TokenType.ACCESS,
        roles=["user"],
        permissions=["diagnosis:read", "diagnosis:write"],
    )


@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    """Create async test client with manually initialized service."""
    from services.diagnosis_service.src.main import DiagnosisServiceAppSettings
    from services.diagnosis_service.src.domain.service import DiagnosisService, DiagnosisServiceSettings
    from services.diagnosis_service.src.domain.symptom_extractor import SymptomExtractor, SymptomExtractorSettings
    from services.diagnosis_service.src.domain.differential import DifferentialGenerator, DifferentialSettings

    # Initialize service components
    extractor = SymptomExtractor(SymptomExtractorSettings())
    generator = DifferentialGenerator(DifferentialSettings())
    service = DiagnosisService(
        settings=DiagnosisServiceSettings(),
        symptom_extractor=extractor,
        differential_generator=generator,
    )
    await service.initialize()

    # Set app state
    app.state.settings = DiagnosisServiceAppSettings()
    app.state.diagnosis_service = service
    app.state.symptom_extractor = extractor
    app.state.differential_generator = generator

    # Override auth dependency
    app.dependency_overrides[get_current_user] = _mock_user

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Cleanup
    app.dependency_overrides.clear()
    await service.shutdown()


class TestHealthEndpoints:
    """Test health and status endpoints."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """Test root endpoint returns service info."""
        response = await async_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "diagnosis-service"
        assert data["status"] == "operational"

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """Test health check endpoint."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "reasoning_pipeline" in data

    @pytest.mark.asyncio
    async def test_liveness_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """Test liveness probe."""
        response = await async_client.get("/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    @pytest.mark.asyncio
    async def test_readiness_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """Test readiness probe when service is initialized."""
        response = await async_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestSessionEndpoints:
    """Test session management endpoints."""

    @pytest.mark.asyncio
    async def test_start_session_new_user(self, async_client: httpx.AsyncClient) -> None:
        """Test starting session for new user."""
        user_id = str(uuid4())
        response = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
            "initial_context": {},
        })
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["session_number"] == 1
        assert data["initial_phase"] == "rapport"
        assert len(data["greeting"]) > 0

    @pytest.mark.asyncio
    async def test_start_session_returning_user(self, async_client: httpx.AsyncClient) -> None:
        """Test starting session for returning user."""
        user_id = str(uuid4())
        response1 = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        session1_id = response1.json()["session_id"]
        await async_client.post("/api/v1/diagnosis/session/end", json={
            "user_id": user_id,
            "session_id": session1_id,
            "generate_summary": False,
        })
        response2 = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
            "previous_session_id": session1_id,
        })
        assert response2.status_code == 201
        data = response2.json()
        assert data["session_number"] == 2
        assert data["loaded_context"] is True

    @pytest.mark.asyncio
    async def test_end_session_with_summary(self, async_client: httpx.AsyncClient) -> None:
        """Test ending session with summary generation."""
        user_id = str(uuid4())
        start_response = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        session_id = start_response.json()["session_id"]
        end_response = await async_client.post("/api/v1/diagnosis/session/end", json={
            "user_id": user_id,
            "session_id": session_id,
            "generate_summary": True,
        })
        assert end_response.status_code == 200
        data = end_response.json()
        assert "duration_minutes" in data
        assert "recommendations" in data

    @pytest.mark.asyncio
    async def test_get_session_state(self, async_client: httpx.AsyncClient) -> None:
        """Test getting session state."""
        user_id = str(uuid4())
        start_response = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        session_id = start_response.json()["session_id"]
        state_response = await async_client.get(f"/api/v1/diagnosis/session/{session_id}/state")
        assert state_response.status_code == 200
        data = state_response.json()
        assert data["phase"] == "rapport"

    @pytest.mark.asyncio
    async def test_get_session_state_not_found(self, async_client: httpx.AsyncClient) -> None:
        """Test getting non-existent session state."""
        fake_id = str(uuid4())
        response = await async_client.get(f"/api/v1/diagnosis/session/{fake_id}/state")
        assert response.status_code == 404


class TestAssessmentEndpoints:
    """Test assessment and diagnosis endpoints."""

    @pytest.mark.asyncio
    async def test_full_assessment(self, async_client: httpx.AsyncClient) -> None:
        """Test full 4-step assessment."""
        user_id = str(uuid4())
        start_response = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        session_id = start_response.json()["session_id"]
        assess_response = await async_client.post("/api/v1/diagnosis/assess", json={
            "user_id": user_id,
            "session_id": session_id,
            "message": "I've been feeling very sad and hopeless for the past two weeks",
            "conversation_history": [],
            "existing_symptoms": [],
            "current_phase": "rapport",
        })
        assert assess_response.status_code == 200
        data = assess_response.json()
        assert "assessment_id" in data
        assert "reasoning_chain" in data
        assert len(data["reasoning_chain"]) == 4
        assert "extracted_symptoms" in data
        assert "differential" in data
        assert "response_text" in data

    @pytest.mark.asyncio
    async def test_assessment_with_multiple_symptoms(self, async_client: httpx.AsyncClient) -> None:
        """Test assessment with multiple symptom message."""
        user_id = str(uuid4())
        start_response = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        session_id = start_response.json()["session_id"]
        assess_response = await async_client.post("/api/v1/diagnosis/assess", json={
            "user_id": user_id,
            "session_id": session_id,
            "message": "I feel depressed, anxious, can't sleep, have no energy, and lost interest in everything",
            "conversation_history": [],
            "current_phase": "assessment",
        })
        assert assess_response.status_code == 200
        data = assess_response.json()
        assert len(data["extracted_symptoms"]) >= 3

    @pytest.mark.asyncio
    async def test_assessment_phase_progression(self, async_client: httpx.AsyncClient) -> None:
        """Test that assessment progresses through phases."""
        user_id = str(uuid4())
        start_response = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        session_id = start_response.json()["session_id"]
        messages = [
            "I've been feeling very sad lately",
            "It started about three weeks ago after losing my job",
            "I'd rate the intensity as about 7 out of 10",
        ]
        phases = ["rapport", "history", "assessment"]
        for msg, phase in zip(messages, phases):
            response = await async_client.post("/api/v1/diagnosis/assess", json={
                "user_id": user_id,
                "session_id": session_id,
                "message": msg,
                "current_phase": phase,
            })
            assert response.status_code == 200


class TestSymptomExtractionEndpoints:
    """Test symptom extraction endpoints."""

    @pytest.mark.asyncio
    async def test_extract_symptoms_depression(self, async_client: httpx.AsyncClient) -> None:
        """Test symptom extraction for depression indicators."""
        response = await async_client.post("/api/v1/diagnosis/extract-symptoms", json={
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "message": "I feel hopeless and sad all the time, can't enjoy anything",
            "conversation_history": [],
        })
        assert response.status_code == 200
        data = response.json()
        symptom_names = [s["name"] for s in data["extracted_symptoms"]]
        assert "depressed_mood" in symptom_names or "anhedonia" in symptom_names

    @pytest.mark.asyncio
    async def test_extract_symptoms_anxiety(self, async_client: httpx.AsyncClient) -> None:
        """Test symptom extraction for anxiety indicators."""
        response = await async_client.post("/api/v1/diagnosis/extract-symptoms", json={
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "message": "I'm constantly worried and nervous, my heart races a lot",
            "conversation_history": [],
        })
        assert response.status_code == 200
        data = response.json()
        symptom_names = [s["name"] for s in data["extracted_symptoms"]]
        assert "anxiety" in symptom_names

    @pytest.mark.asyncio
    async def test_extract_symptoms_with_existing(self, async_client: httpx.AsyncClient) -> None:
        """Test symptom extraction with existing symptoms."""
        existing = [{
            "symptom_id": str(uuid4()),
            "name": "depressed_mood",
            "description": "Existing symptom",
            "symptom_type": "emotional",
            "severity": "moderate",
            "confidence": "0.8",
        }]
        response = await async_client.post("/api/v1/diagnosis/extract-symptoms", json={
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "message": "I also have trouble sleeping",
            "existing_symptoms": existing,
        })
        assert response.status_code == 200
        data = response.json()
        updated_names = [s["name"] for s in data["updated_symptoms"]]
        assert "depressed_mood" in updated_names

    @pytest.mark.asyncio
    async def test_extract_symptoms_risk_detection(self, async_client: httpx.AsyncClient) -> None:
        """Test risk indicator detection."""
        response = await async_client.post("/api/v1/diagnosis/extract-symptoms", json={
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "message": "Sometimes I think about hurting myself",
        })
        assert response.status_code == 200
        data = response.json()
        assert "self_harm" in data["risk_indicators"]


class TestDifferentialEndpoints:
    """Test differential diagnosis endpoints."""

    @pytest.mark.asyncio
    async def test_generate_differential_depression(self, async_client: httpx.AsyncClient) -> None:
        """Test differential generation for depression symptoms."""
        symptoms = [
            {"symptom_id": str(uuid4()), "name": "depressed_mood",
             "description": "Feeling sad", "symptom_type": "emotional",
             "severity": "moderate", "confidence": "0.8"},
            {"symptom_id": str(uuid4()), "name": "anhedonia",
             "description": "No interest", "symptom_type": "emotional",
             "severity": "moderate", "confidence": "0.7"},
            {"symptom_id": str(uuid4()), "name": "sleep_disturbance",
             "description": "Insomnia", "symptom_type": "somatic",
             "severity": "mild", "confidence": "0.6"},
        ]
        response = await async_client.post("/api/v1/diagnosis/differential", json={
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "symptoms": symptoms,
            "user_history": {},
        })
        assert response.status_code == 200
        data = response.json()
        assert "differential" in data
        assert data["differential"]["primary"] is not None
        hypothesis_name = data["differential"]["primary"]["name"].lower()
        assert "depress" in hypothesis_name or "adjustment" in hypothesis_name

    @pytest.mark.asyncio
    async def test_generate_differential_with_hitop(self, async_client: httpx.AsyncClient) -> None:
        """Test differential includes HiTOP dimensional scores."""
        symptoms = [
            {"symptom_id": str(uuid4()), "name": "anxiety",
             "description": "Anxious", "symptom_type": "emotional",
             "severity": "moderate", "confidence": "0.8"},
        ]
        response = await async_client.post("/api/v1/diagnosis/differential", json={
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "symptoms": symptoms,
        })
        assert response.status_code == 200
        data = response.json()
        assert "hitop_scores" in data
        assert "internalizing" in data["hitop_scores"]

    @pytest.mark.asyncio
    async def test_generate_differential_recommended_questions(self, async_client: httpx.AsyncClient) -> None:
        """Test differential generates recommended questions."""
        symptoms = [
            {"symptom_id": str(uuid4()), "name": "depressed_mood",
             "description": "Sad", "symptom_type": "emotional",
             "severity": "mild", "confidence": "0.6"},
        ]
        response = await async_client.post("/api/v1/diagnosis/differential", json={
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "symptoms": symptoms,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommended_questions"]) > 0


class TestServiceStatusEndpoints:
    """Test service status and administration endpoints."""

    @pytest.mark.asyncio
    async def test_get_status(self, async_client: httpx.AsyncClient) -> None:
        """Test service status endpoint."""
        response = await async_client.get("/api/v1/diagnosis/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "statistics" in data
        assert "active_sessions" in data

    @pytest.mark.asyncio
    async def test_get_history(self, async_client: httpx.AsyncClient) -> None:
        """Test diagnosis history endpoint."""
        user_id = str(uuid4())
        start = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        session_id = start.json()["session_id"]
        await async_client.post("/api/v1/diagnosis/session/end", json={
            "user_id": user_id,
            "session_id": session_id,
            "generate_summary": True,
        })
        history_response = await async_client.post("/api/v1/diagnosis/history", json={
            "user_id": user_id,
            "limit": 10,
        })
        assert history_response.status_code == 200
        data = history_response.json()
        assert len(data["sessions"]) == 1

    @pytest.mark.asyncio
    async def test_delete_user_data(self, async_client: httpx.AsyncClient) -> None:
        """Test GDPR user data deletion."""
        user_id = str(uuid4())
        await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        delete_response = await async_client.delete(f"/api/v1/diagnosis/user/{user_id}")
        assert delete_response.status_code == 204
        history_response = await async_client.post("/api/v1/diagnosis/history", json={
            "user_id": user_id,
            "limit": 10,
        })
        assert len(history_response.json()["sessions"]) == 0


class TestErrorHandling:
    """Test error handling and validation."""

    @pytest.mark.asyncio
    async def test_invalid_uuid_format(self, async_client: httpx.AsyncClient) -> None:
        """Test handling of invalid UUID format."""
        response = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": "not-a-valid-uuid",
            "session_type": "assessment",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_required_field(self, async_client: httpx.AsyncClient) -> None:
        """Test handling of missing required fields."""
        response = await async_client.post("/api/v1/diagnosis/assess", json={
            "session_id": str(uuid4()),
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_phase_value(self, async_client: httpx.AsyncClient) -> None:
        """Test handling of invalid diagnosis phase."""
        response = await async_client.post("/api/v1/diagnosis/assess", json={
            "user_id": str(uuid4()),
            "session_id": str(uuid4()),
            "message": "test",
            "current_phase": "invalid_phase",
        })
        assert response.status_code == 422


class TestChallengeEndpoint:
    """Test Devil's Advocate challenge endpoint."""

    @pytest.mark.asyncio
    async def test_challenge_hypothesis(self, async_client: httpx.AsyncClient) -> None:
        """Test hypothesis challenge endpoint."""
        user_id = str(uuid4())
        start = await async_client.post("/api/v1/diagnosis/session/start", json={
            "user_id": user_id,
            "session_type": "assessment",
        })
        session_id = start.json()["session_id"]
        await async_client.post("/api/v1/diagnosis/assess", json={
            "user_id": user_id,
            "session_id": session_id,
            "message": "I feel very depressed and hopeless",
            "current_phase": "assessment",
        })
        hypothesis_id = str(uuid4())
        response = await async_client.post(
            f"/api/v1/diagnosis/challenge/{session_id}",
            params={"hypothesis_id": hypothesis_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert "challenges" in data or "counter_questions" in data
