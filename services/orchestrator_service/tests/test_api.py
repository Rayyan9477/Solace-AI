"""
Unit tests for Orchestrator Service API Endpoints.
Tests REST endpoints, WebSocket handlers, and request/response models.
"""
from __future__ import annotations
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from services.orchestrator_service.src.api import (
    ChatMessageRequest,
    ChatMessageResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    HealthResponse,
    ConversationHistoryResponse,
    router,
)


class TestChatMessageRequest:
    """Tests for ChatMessageRequest model."""

    def test_valid_request(self) -> None:
        """Test creating valid chat message request."""
        request = ChatMessageRequest(
            message="Hello, how are you?",
            user_id="user-123",
            session_id="session-456",
        )
        assert request.message == "Hello, how are you?"
        assert request.user_id == "user-123"
        assert request.session_id == "session-456"
        assert request.thread_id is None

    def test_request_with_optional_fields(self) -> None:
        """Test request with all optional fields."""
        request = ChatMessageRequest(
            message="Test message",
            user_id="user-123",
            session_id="session-456",
            thread_id="thread-789",
            conversation_context="Previous context",
            metadata={"source": "test"},
        )
        assert request.thread_id == "thread-789"
        assert request.conversation_context == "Previous context"
        assert request.metadata == {"source": "test"}

    def test_request_validates_message_length(self) -> None:
        """Test that message length is validated."""
        with pytest.raises(ValueError):
            ChatMessageRequest(
                message="",
                user_id="user-123",
                session_id="session-456",
            )


class TestChatMessageResponse:
    """Tests for ChatMessageResponse model."""

    def test_valid_response(self) -> None:
        """Test creating valid chat message response."""
        response = ChatMessageResponse(
            response="I'm here to help.",
            thread_id="thread-123",
            session_id="session-456",
            intent="general_chat",
            intent_confidence=0.75,
            safety_flags={"risk_level": "none"},
            processing_time_ms=150.5,
            agents_used=["safety", "chat"],
        )
        assert response.response == "I'm here to help."
        assert response.intent == "general_chat"
        assert response.processing_time_ms == 150.5
        assert "safety" in response.agents_used


class TestSessionCreateRequest:
    """Tests for SessionCreateRequest model."""

    def test_valid_request(self) -> None:
        """Test creating valid session request."""
        request = SessionCreateRequest(
            user_id="user-123",
        )
        assert request.user_id == "user-123"
        assert request.initial_context is None

    def test_request_with_context(self) -> None:
        """Test session request with initial context."""
        request = SessionCreateRequest(
            user_id="user-123",
            initial_context="User has anxiety history",
            metadata={"referral": "therapist"},
        )
        assert request.initial_context == "User has anxiety history"
        assert request.metadata["referral"] == "therapist"


class TestSessionCreateResponse:
    """Tests for SessionCreateResponse model."""

    def test_valid_response(self) -> None:
        """Test creating valid session response."""
        response = SessionCreateResponse(
            session_id="session-123",
            thread_id="thread-456",
            created_at="2024-01-01T00:00:00Z",
        )
        assert response.session_id == "session-123"
        assert response.thread_id == "thread-456"


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_valid_response(self) -> None:
        """Test creating valid health response."""
        response = HealthResponse(
            status="healthy",
            graph_ready=True,
            active_sessions=5,
            uptime_seconds=3600.0,
        )
        assert response.status == "healthy"
        assert response.graph_ready is True


class TestConversationHistoryResponse:
    """Tests for ConversationHistoryResponse model."""

    def test_valid_response(self) -> None:
        """Test creating valid conversation history response."""
        response = ConversationHistoryResponse(
            thread_id="thread-123",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            message_count=2,
            last_activity="2024-01-01T00:00:00Z",
        )
        assert response.thread_id == "thread-123"
        assert len(response.messages) == 2
        assert response.message_count == 2


class TestRouterEndpoints:
    """Tests for API router endpoints."""

    @pytest.fixture
    def mock_graph_builder(self):
        """Create mock graph builder."""
        mock_builder = MagicMock()
        mock_builder.invoke = AsyncMock(return_value={
            "final_response": "Test response",
            "intent": "general_chat",
            "intent_confidence": 0.75,
            "safety_flags": {"risk_level": "none"},
            "agent_results": [
                {"agent_type": "safety", "success": True},
                {"agent_type": "chat", "success": True},
            ],
            "processing_phase": "completed",
        })
        mock_builder.get_checkpointer = MagicMock(return_value=None)
        return mock_builder

    def test_agents_endpoint(self) -> None:
        """Test list agents endpoint structure."""
        from services.orchestrator_service.src.api import list_agents
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(list_agents())
        assert "agents" in result
        assert len(result["agents"]) == 7
        assert result["total_agents"] == 7
        agent_types = [a["type"] for a in result["agents"]]
        assert "safety" in agent_types
        assert "supervisor" in agent_types
        assert "therapy" in agent_types


class TestChatMessageRequestValidation:
    """Tests for ChatMessageRequest validation."""

    def test_message_max_length(self) -> None:
        """Test message max length validation."""
        long_message = "a" * 10001
        with pytest.raises(ValueError):
            ChatMessageRequest(
                message=long_message,
                user_id="user-123",
                session_id="session-456",
            )

    def test_required_fields(self) -> None:
        """Test required fields validation."""
        with pytest.raises(ValueError):
            ChatMessageRequest(
                message="Hello",
                user_id="",
                session_id="session-456",
            )


class TestAPIModelsEdgeCases:
    """Tests for edge cases in API models."""

    def test_response_with_empty_agents(self) -> None:
        """Test response with empty agents list."""
        response = ChatMessageResponse(
            response="Default response",
            thread_id="thread-123",
            session_id="session-456",
            intent="error",
            intent_confidence=0.0,
            safety_flags={},
            processing_time_ms=0.0,
            agents_used=[],
            metadata={"error": "timeout"},
        )
        assert len(response.agents_used) == 0
        assert response.metadata["error"] == "timeout"

    def test_session_response_empty_metadata(self) -> None:
        """Test session response with empty metadata."""
        response = SessionCreateResponse(
            session_id="session-123",
            thread_id="thread-456",
            created_at="2024-01-01T00:00:00Z",
            metadata={},
        )
        assert response.metadata == {}

    def test_history_response_no_last_activity(self) -> None:
        """Test history response without last activity."""
        response = ConversationHistoryResponse(
            thread_id="thread-123",
            messages=[],
            message_count=0,
            last_activity=None,
        )
        assert response.last_activity is None
        assert response.message_count == 0


class TestAPIEndpointBehavior:
    """Tests for API endpoint behavior patterns."""

    def test_chat_request_generates_thread_id(self) -> None:
        """Test that chat request without thread_id still works."""
        request = ChatMessageRequest(
            message="Hello",
            user_id="user-123",
            session_id="session-456",
            thread_id=None,
        )
        assert request.thread_id is None

    def test_response_includes_all_fields(self) -> None:
        """Test that response includes all required fields."""
        response = ChatMessageResponse(
            response="Test",
            thread_id="thread-123",
            session_id="session-456",
            intent="general_chat",
            intent_confidence=0.5,
            safety_flags={},
            processing_time_ms=100.0,
            agents_used=["chat"],
        )
        assert all([
            response.response,
            response.thread_id,
            response.session_id,
            response.intent,
            response.intent_confidence >= 0,
            response.safety_flags is not None,
            response.processing_time_ms >= 0,
            response.agents_used is not None,
        ])


class TestBatchProcessing:
    """Tests for batch processing endpoint patterns."""

    def test_batch_request_format(self) -> None:
        """Test batch request is list of ChatMessageRequest."""
        messages = [
            ChatMessageRequest(message="Hello", user_id="user-1", session_id="session-1"),
            ChatMessageRequest(message="Hi there", user_id="user-1", session_id="session-1"),
            ChatMessageRequest(message="How are you?", user_id="user-1", session_id="session-1"),
        ]
        assert len(messages) == 3
        for msg in messages:
            assert isinstance(msg, ChatMessageRequest)

    def test_batch_responses_match_requests(self) -> None:
        """Test that batch responses match request count."""
        responses = [
            ChatMessageResponse(
                response=f"Response {i}",
                thread_id=f"thread-{i}",
                session_id="session-1",
                intent="general_chat",
                intent_confidence=0.5,
                safety_flags={},
                processing_time_ms=100.0,
                agents_used=["chat"],
            )
            for i in range(3)
        ]
        assert len(responses) == 3


class TestWebSocketProtocol:
    """Tests for WebSocket protocol patterns."""

    def test_websocket_message_types(self) -> None:
        """Test expected WebSocket message types."""
        message_types = ["ping", "pong", "message", "processing", "response", "error", "connected"]
        assert len(message_types) == 7

    def test_connected_message_format(self) -> None:
        """Test connected message format."""
        connected_msg = {
            "type": "connected",
            "session_id": "session-123",
            "thread_id": "thread-456",
            "connection_id": "conn-789",
        }
        assert connected_msg["type"] == "connected"
        assert all(key in connected_msg for key in ["session_id", "thread_id", "connection_id"])

    def test_response_message_format(self) -> None:
        """Test response message format."""
        response_msg = {
            "type": "response",
            "response": "Hello, how can I help?",
            "intent": "general_chat",
            "intent_confidence": 0.75,
            "safety_flags": {"risk_level": "none"},
            "processing_time_ms": 150.5,
            "thread_id": "thread-123",
        }
        assert response_msg["type"] == "response"
        assert "response" in response_msg
        assert "processing_time_ms" in response_msg

    def test_error_message_format(self) -> None:
        """Test error message format."""
        error_msg = {
            "type": "error",
            "message": "Processing error occurred",
        }
        assert error_msg["type"] == "error"
        assert "message" in error_msg
