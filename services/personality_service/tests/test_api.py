"""
Unit tests for Personality Service API.
Tests REST endpoints using FastAPI TestClient.
"""
from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from services.personality_service.src.schemas import (
    OceanScoresDTO, StyleParametersDTO, ProfileSummaryDTO,
    PersonalityTrait, CommunicationStyleType, EmotionCategory,
)
from services.personality_service.src.api import router
from services.personality_service.src.domain.service import PersonalityOrchestrator


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """Create mock orchestrator."""
    orchestrator = MagicMock(spec=PersonalityOrchestrator)
    orchestrator.is_initialized = True
    return orchestrator


@pytest.fixture
def app(mock_orchestrator: MagicMock):
    """Create test app with mock orchestrator."""
    from fastapi import FastAPI
    from solace_security.middleware import (
        get_current_user, get_current_service,
        AuthenticatedUser, AuthenticatedService,
    )
    from solace_security.auth import TokenType

    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1/personality")
    test_app.state.personality_orchestrator = mock_orchestrator

    # Override auth dependencies for testing
    test_app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id="test-user", token_type=TokenType.ACCESS, roles=["user"],
    )
    test_app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="test-service", permissions=["personality:read", "personality:write"],
    )
    return test_app


@pytest.fixture
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestDetectEndpoint:
    """Tests for /detect endpoint."""

    def test_detect_personality_success(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        """Test successful personality detection."""
        user_id = uuid4()
        mock_response = MagicMock()
        mock_response.user_id = user_id
        mock_response.ocean_scores = OceanScoresDTO(
            openness=0.7, conscientiousness=0.6, extraversion=0.5,
            agreeableness=0.8, neuroticism=0.4,
        )
        mock_response.assessment_source = "ensemble"
        mock_response.confidence = 0.75
        mock_response.evidence = ["insight_words"]
        mock_response.processing_time_ms = 50.5
        mock_orchestrator.detect_personality = AsyncMock(return_value=mock_response)
        response = client.post(
            "/api/v1/personality/detect",
            json={
                "user_id": str(user_id),
                "text": "I love exploring new ideas and learning about creative solutions to problems.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == str(user_id)
        assert "ocean_scores" in data

    def test_detect_personality_validation_error(self, client: TestClient) -> None:
        """Test validation error for short text."""
        response = client.post(
            "/api/v1/personality/detect",
            json={
                "user_id": str(uuid4()),
                "text": "short",
            },
        )
        assert response.status_code == 422


class TestStyleEndpoint:
    """Tests for /style endpoint."""

    def test_get_style_success(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        """Test successful style retrieval."""
        user_id = uuid4()
        mock_response = MagicMock()
        mock_response.user_id = user_id
        mock_response.style_parameters = StyleParametersDTO(
            warmth=0.7, structure=0.6, style_type=CommunicationStyleType.BALANCED,
        )
        mock_response.recommendations = ["Use validation"]
        mock_response.profile_confidence = 0.8
        mock_orchestrator.get_style = AsyncMock(return_value=mock_response)
        response = client.post(
            "/api/v1/personality/style",
            json={"user_id": str(user_id)},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == str(user_id)
        assert "style_parameters" in data


class TestAdaptEndpoint:
    """Tests for /adapt endpoint."""

    def test_adapt_response_success(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        """Test successful response adaptation."""
        user_id = uuid4()
        mock_response = MagicMock()
        mock_response.user_id = user_id
        mock_response.adapted_content = "I hear you. Let's explore this together."
        mock_response.applied_style = StyleParametersDTO()
        mock_response.empathy_components = None
        mock_response.adaptation_confidence = 0.8
        mock_orchestrator.adapt_response = AsyncMock(return_value=mock_response)
        response = client.post(
            "/api/v1/personality/adapt",
            json={
                "user_id": str(user_id),
                "base_response": "Let's explore this together.",
                "include_empathy": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == str(user_id)
        assert "adapted_content" in data


class TestProfileEndpoint:
    """Tests for /profile endpoint."""

    def test_get_profile_success(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        """Test successful profile retrieval."""
        user_id = uuid4()
        from datetime import datetime
        mock_profile = ProfileSummaryDTO(
            user_id=user_id,
            ocean_scores=OceanScoresDTO(
                openness=0.7, conscientiousness=0.6, extraversion=0.5,
                agreeableness=0.8, neuroticism=0.4,
            ),
            style_parameters=StyleParametersDTO(),
            dominant_traits=[PersonalityTrait.OPENNESS, PersonalityTrait.AGREEABLENESS],
            assessment_count=5,
            stability_score=0.85,
            last_updated=datetime.now(timezone.utc),
            version=3,
        )
        mock_orchestrator.get_profile = MagicMock(return_value=mock_profile)
        response = client.get(f"/api/v1/personality/profile/{user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is True
        assert data["profile"]["user_id"] == str(user_id)

    def test_get_profile_not_found(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        """Test profile not found."""
        mock_orchestrator.get_profile = MagicMock(return_value=None)
        response = client.get(f"/api/v1/personality/profile/{uuid4()}")
        assert response.status_code == 404


class TestStatisticsEndpoint:
    """Tests for /statistics endpoint."""

    def test_get_statistics(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        """Test getting service statistics."""
        mock_orchestrator.get_statistics = MagicMock(return_value={
            "initialized": True,
            "total_requests": 100,
            "total_detections": 50,
            "profiles_count": 25,
        })
        response = client.get("/api/v1/personality/statistics")
        assert response.status_code == 200
        data = response.json()
        assert data["initialized"] is True
        assert data["total_requests"] == 100


class TestTraitsEndpoint:
    """Tests for /traits endpoint."""

    def test_list_traits(self, client: TestClient) -> None:
        """Test listing available traits."""
        response = client.get("/api/v1/personality/traits")
        assert response.status_code == 200
        data = response.json()
        assert "traits" in data
        assert len(data["traits"]) == 5
        assert data["model"] == "Big Five (OCEAN)"
        trait_names = [t["name"] for t in data["traits"]]
        assert "openness" in trait_names
        assert "neuroticism" in trait_names


class TestStylesEndpoint:
    """Tests for /styles endpoint."""

    def test_list_styles(self, client: TestClient) -> None:
        """Test listing available styles."""
        response = client.get("/api/v1/personality/styles")
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        assert "parameters" in data
        assert len(data["styles"]) == 5
        style_names = [s["name"] for s in data["styles"]]
        assert "analytical" in style_names
        assert "balanced" in style_names


class TestErrorHandling:
    """Tests for API error handling."""

    def test_service_unavailable(self) -> None:
        """Test error when service not initialized."""
        from fastapi import FastAPI
        test_app = FastAPI()
        test_app.include_router(router, prefix="/api/v1/personality")
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/personality/detect",
            json={
                "user_id": str(uuid4()),
                "text": "Test text that is long enough for analysis.",
            },
        )
        assert response.status_code == 503
