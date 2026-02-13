"""
Unit tests for Solace-AI Safety Service API endpoints.
Tests FastAPI endpoints for safety checking and crisis management.
"""
from __future__ import annotations
import pytest
from uuid import uuid4
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from services.safety_service.src.api import (
    router, SafetyCheckRequest, SafetyCheckResponse, CrisisLevel,
    SafetyCheckType, CrisisDetectionRequest, EscalationRequest,
    SafetyAssessmentRequest, OutputFilterRequest, RiskFactorDTO,
)
from services.safety_service.src.domain.service import SafetyService, SafetyCheckResult
from services.safety_service.src.domain.crisis_detector import (
    CrisisLevel as DomainCrisisLevel, RiskFactor,
)
from services.safety_service.src.domain.escalation import EscalationResult
from solace_security.middleware import AuthenticatedService, get_current_service


@pytest.fixture
def mock_safety_service() -> MagicMock:
    """Create mock safety service."""
    service = MagicMock(spec=SafetyService)
    service._initialized = True
    return service


def _mock_service() -> AuthenticatedService:
    return AuthenticatedService(service_name="test-service", permissions=["safety:read", "safety:write"])


@pytest.fixture
def app(mock_safety_service: MagicMock) -> FastAPI:
    """Create test FastAPI application."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1/safety")
    test_app.state.safety_service = mock_safety_service
    test_app.dependency_overrides[get_current_service] = _mock_service
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestSafetyCheckEndpoint:
    """Tests for /check endpoint."""

    @pytest.mark.asyncio
    async def test_safety_check_safe_content(self, client: TestClient, mock_safety_service: MagicMock) -> None:
        """Test safety check with safe content."""
        mock_safety_service.check_safety = AsyncMock(return_value=SafetyCheckResult(
            is_safe=True,
            crisis_level=DomainCrisisLevel.NONE,
            risk_score=Decimal("0.1"),
            risk_factors=[],
            protective_factors=[],
            recommended_action="continue",
            requires_escalation=False,
            requires_human_review=False,
            detection_time_ms=5,
            detection_layer=1,
        ))
        response = client.post("/api/v1/safety/check", json={
            "user_id": str(uuid4()),
            "content": "I'm having a good day",
            "check_type": "pre_check",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["is_safe"] is True
        assert data["crisis_level"] == "NONE"

    @pytest.mark.asyncio
    async def test_safety_check_crisis_content(self, client: TestClient, mock_safety_service: MagicMock) -> None:
        """Test safety check with crisis content."""
        mock_safety_service.check_safety = AsyncMock(return_value=SafetyCheckResult(
            is_safe=False,
            crisis_level=DomainCrisisLevel.CRITICAL,
            risk_score=Decimal("0.95"),
            risk_factors=[RiskFactor(
                factor_type="keyword", severity=Decimal("0.95"),
                evidence="Critical keyword", confidence=Decimal("0.9"), detection_layer=1,
            )],
            protective_factors=[],
            recommended_action="escalate_immediately",
            requires_escalation=True,
            requires_human_review=True,
            detection_time_ms=3,
            detection_layer=1,
        ))
        response = client.post("/api/v1/safety/check", json={
            "user_id": str(uuid4()),
            "content": "I want to end my life",
            "check_type": "pre_check",
            "include_resources": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["is_safe"] is False
        assert data["crisis_level"] == "CRITICAL"
        assert data["requires_escalation"] is True
        assert len(data["crisis_resources"]) > 0

    def test_safety_check_validation_error(self, client: TestClient) -> None:
        """Test safety check with invalid request."""
        response = client.post("/api/v1/safety/check", json={
            "user_id": "invalid-uuid",
            "content": "",
        })
        assert response.status_code == 422


class TestCrisisDetectionEndpoint:
    """Tests for /detect-crisis endpoint."""

    @pytest.mark.asyncio
    async def test_detect_crisis(self, client: TestClient, mock_safety_service: MagicMock) -> None:
        """Test crisis detection endpoint."""
        from services.safety_service.src.domain.service import CrisisDetectionResult
        mock_safety_service.detect_crisis = AsyncMock(return_value=CrisisDetectionResult(
            crisis_detected=True,
            crisis_level=DomainCrisisLevel.HIGH,
            trigger_indicators=["KEYWORD:suicide"],
            confidence=Decimal("0.85"),
            detection_layers_triggered=[1],
            detection_time_ms=4,
        ))
        response = client.post("/api/v1/safety/detect-crisis", json={
            "user_id": str(uuid4()),
            "content": "I'm thinking about hurting myself",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["crisis_detected"] is True
        assert data["crisis_level"] == "HIGH"


class TestEscalationEndpoint:
    """Tests for /escalate endpoint."""

    @pytest.mark.asyncio
    async def test_trigger_escalation(self, client: TestClient, mock_safety_service: MagicMock) -> None:
        """Test escalation triggering."""
        mock_safety_service.escalate = AsyncMock(return_value=EscalationResult(
            escalation_id=uuid4(),
            status="IN_PROGRESS",
            priority="CRITICAL",
            assigned_clinician_id=uuid4(),
            notification_sent=True,
            actions_taken=["CRITICAL workflow initiated"],
            estimated_response_minutes=5,
            resources_provided=True,
        ))
        response = client.post("/api/v1/safety/escalate", json={
            "user_id": str(uuid4()),
            "crisis_level": "CRITICAL",
            "reason": "Suicidal ideation detected",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["priority"] == "CRITICAL"
        assert data["notification_sent"] is True


class TestAssessmentEndpoint:
    """Tests for /assess endpoint."""

    @pytest.mark.asyncio
    async def test_safety_assessment(self, client: TestClient, mock_safety_service: MagicMock) -> None:
        """Test comprehensive safety assessment."""
        from services.safety_service.src.domain.service import SafetyAssessmentResult
        mock_safety_service.assess_safety = AsyncMock(return_value=SafetyAssessmentResult(
            assessment_id=uuid4(),
            overall_risk_level=DomainCrisisLevel.ELEVATED,
            overall_risk_score=Decimal("0.55"),
            message_assessments=[{"message_index": 0, "crisis_level": "ELEVATED"}],
            trajectory_analysis={"trend": "stable"},
            recommendations=["Monitor closely"],
            requires_intervention=False,
        ))
        response = client.post("/api/v1/safety/assess", json={
            "user_id": str(uuid4()),
            "messages": ["I'm feeling anxious"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["overall_risk_level"] == "ELEVATED"


class TestOutputFilterEndpoint:
    """Tests for /filter-output endpoint."""

    @pytest.mark.asyncio
    async def test_filter_output(self, client: TestClient, mock_safety_service: MagicMock) -> None:
        """Test output filtering."""
        from services.safety_service.src.domain.service import OutputFilterResult
        mock_safety_service.filter_output = AsyncMock(return_value=OutputFilterResult(
            filter_id=uuid4(),
            filtered_response="Safe response",
            modifications_made=[],
            resources_appended=False,
            is_safe=True,
            filter_time_ms=2,
        ))
        response = client.post("/api/v1/safety/filter-output", json={
            "user_id": str(uuid4()),
            "original_response": "Safe response",
            "user_crisis_level": "NONE",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["is_safe"] is True


class TestResourcesEndpoint:
    """Tests for /resources endpoint."""

    def test_get_crisis_resources(self, client: TestClient, mock_safety_service: MagicMock) -> None:
        """Test getting crisis resources."""
        response = client.get("/api/v1/safety/resources?level=HIGH")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert any("988" in r["contact"] for r in data)


class TestStatusEndpoint:
    """Tests for /status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status(self, client: TestClient, mock_safety_service: MagicMock) -> None:
        """Test getting service status."""
        mock_safety_service.get_status = AsyncMock(return_value={
            "status": "operational",
            "initialized": True,
            "statistics": {"total_checks": 100},
        })
        response = client.get("/api/v1/safety/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"


class TestRequestModels:
    """Tests for request models."""

    def test_safety_check_request_valid(self) -> None:
        """Test valid safety check request."""
        request = SafetyCheckRequest(
            user_id=uuid4(),
            content="Test content",
            check_type=SafetyCheckType.PRE_CHECK,
        )
        assert request.content == "Test content"
        assert request.check_type == SafetyCheckType.PRE_CHECK

    def test_crisis_detection_request(self) -> None:
        """Test crisis detection request."""
        request = CrisisDetectionRequest(
            user_id=uuid4(),
            content="Test content",
            conversation_history=["msg1", "msg2"],
        )
        assert len(request.conversation_history) == 2

    def test_escalation_request(self) -> None:
        """Test escalation request."""
        request = EscalationRequest(
            user_id=uuid4(),
            crisis_level=CrisisLevel.HIGH,
            reason="Test reason",
        )
        assert request.crisis_level == CrisisLevel.HIGH

    def test_assessment_request(self) -> None:
        """Test assessment request."""
        request = SafetyAssessmentRequest(
            user_id=uuid4(),
            messages=["msg1", "msg2"],
        )
        assert len(request.messages) == 2


class TestResponseModels:
    """Tests for response models."""

    def test_safety_check_response(self) -> None:
        """Test safety check response."""
        response = SafetyCheckResponse(
            user_id=uuid4(),
            is_safe=True,
            crisis_level=CrisisLevel.NONE,
            risk_score=Decimal("0.1"),
            recommended_action="continue",
            detection_time_ms=5,
            detection_layer=1,
        )
        assert response.is_safe is True

    def test_risk_factor_dto(self) -> None:
        """Test risk factor DTO."""
        rf = RiskFactorDTO(
            factor_type="keyword",
            severity=Decimal("0.8"),
            evidence="Test evidence",
            confidence=Decimal("0.9"),
            detection_layer=1,
        )
        assert rf.factor_type == "keyword"
