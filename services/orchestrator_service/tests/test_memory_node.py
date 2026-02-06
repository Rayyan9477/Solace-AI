"""
Tests for Memory Retrieval Node integration.
"""
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, Mock, patch

from src.langgraph.memory_node import MemoryRetrievalNode, MemoryNodeSettings, memory_retrieval_node
from src.langgraph.state_schema import create_initial_state, OrchestratorState, ProcessingPhase


@pytest.fixture
def mock_memory_client():
    """Mock Memory Service Client."""
    with patch("src.langgraph.memory_node.MemoryServiceClient") as mock:
        client_instance = AsyncMock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def sample_state() -> OrchestratorState:
    """Create a sample orchestrator state."""
    user_id = uuid4()
    session_id = uuid4()
    return create_initial_state(
        user_id=user_id,
        session_id=session_id,
        message="I'm feeling anxious today",
        metadata={"test": True},
    )


@pytest.fixture
def sample_memory_response():
    """Sample response from Memory Service."""
    return {
        "context_id": str(uuid4()),
        "user_id": str(uuid4()),
        "assembled_context": "User has previously discussed anxiety and coping mechanisms. Last session focused on breathing exercises.",
        "total_tokens": 1500,
        "token_breakdown": {
            "working_memory": 200,
            "session_memory": 500,
            "episodic_memory": 800,
        },
        "sources_used": ["tier_2_working", "tier_3_session", "tier_4_episodic"],
        "assembly_time_ms": 45,
        "retrieval_count": 5,
    }


class TestMemoryRetrievalNode:
    """Test suite for Memory Retrieval Node."""

    @pytest.mark.asyncio
    async def test_memory_retrieval_success(
        self, mock_memory_client, sample_state, sample_memory_response
    ):
        """Test successful memory retrieval."""
        # Setup mock response
        mock_memory_client.post.return_value = AsyncMock(
            success=True,
            data=sample_memory_response,
            status_code=200,
            response_time_ms=45.0,
        )

        # Create node and process
        settings = MemoryNodeSettings(enable_retrieval=True)
        node = MemoryRetrievalNode(settings=settings)
        
        result = await node.process(sample_state)

        # Verify state updates
        assert "conversation_context" in result
        assert "assembled_context" in result
        assert "memory_sources" in result
        assert "memory_context" in result
        assert "processing_phase" in result
        assert "agent_results" in result

        # Verify content
        assert result["assembled_context"] == sample_memory_response["assembled_context"]
        assert result["conversation_context"] == sample_memory_response["assembled_context"]
        assert len(result["memory_sources"]) == 3
        assert result["memory_context"]["total_tokens"] == 1500
        assert result["memory_context"]["retrieval_count"] == 5
        assert result["processing_phase"] == ProcessingPhase.CONTEXT_LOADING.value

        # Verify agent results
        agent_results = result["agent_results"]
        assert len(agent_results) == 1
        assert agent_results[0]["success"] is True
        assert agent_results[0]["metadata"]["phase"] == "memory_retrieval"
        assert agent_results[0]["metadata"]["total_tokens"] == 1500

    @pytest.mark.asyncio
    async def test_memory_retrieval_disabled(self, sample_state):
        """Test memory retrieval when disabled."""
        settings = MemoryNodeSettings(enable_retrieval=False)
        node = MemoryRetrievalNode(settings=settings)
        
        result = await node.process(sample_state)

        # Verify empty context
        assert result["conversation_context"] == ""
        assert result["assembled_context"] == ""
        assert result["memory_sources"] == []
        assert result["memory_context"]["fallback_mode"] is True
        assert result["processing_phase"] == ProcessingPhase.CONTEXT_LOADING.value

    @pytest.mark.asyncio
    async def test_memory_retrieval_service_error_with_fallback(
        self, mock_memory_client, sample_state
    ):
        """Test memory retrieval with service error and fallback enabled."""
        # Setup mock to return error
        mock_memory_client.post.return_value = AsyncMock(
            success=False,
            data=None,
            error="Service unavailable",
            status_code=503,
        )

        settings = MemoryNodeSettings(enable_retrieval=True, fallback_on_error=True)
        node = MemoryRetrievalNode(settings=settings)
        
        result = await node.process(sample_state)

        # Verify fallback behavior
        assert result["conversation_context"] == ""
        assert result["assembled_context"] == ""
        assert result["memory_context"]["fallback_mode"] is True
        assert result["agent_results"][0]["success"] is False

    @pytest.mark.asyncio
    async def test_memory_retrieval_service_error_without_fallback(
        self, mock_memory_client, sample_state
    ):
        """Test memory retrieval with service error and fallback disabled."""
        # Setup mock to return error
        mock_memory_client.post.return_value = AsyncMock(
            success=False,
            data=None,
            error="Service unavailable",
            status_code=503,
        )

        settings = MemoryNodeSettings(enable_retrieval=True, fallback_on_error=False)
        node = MemoryRetrievalNode(settings=settings)
        
        # Verify exception is raised
        with pytest.raises(RuntimeError, match="Memory service request failed"):
            await node.process(sample_state)

    @pytest.mark.asyncio
    async def test_memory_node_function(self, mock_memory_client, sample_state, sample_memory_response):
        """Test the standalone memory_retrieval_node function."""
        # Setup mock response
        mock_memory_client.post.return_value = AsyncMock(
            success=True,
            data=sample_memory_response,
            status_code=200,
        )

        result = await memory_retrieval_node(sample_state)

        # Verify it returns proper state update
        assert "conversation_context" in result
        assert "assembled_context" in result
        assert result["assembled_context"] == sample_memory_response["assembled_context"]

    def test_node_statistics(self):
        """Test node statistics tracking."""
        settings = MemoryNodeSettings(token_budget=5000)
        node = MemoryRetrievalNode(settings=settings)

        stats = node.get_statistics()

        assert stats["total_retrievals"] == 0
        assert stats["token_budget"] == 5000
        assert stats["retrieval_enabled"] is True

    @pytest.mark.asyncio
    async def test_invalid_uuid_format(self, mock_memory_client):
        """Test handling of invalid UUID format."""
        # Create state with invalid UUIDs
        invalid_state = OrchestratorState(
            user_id="invalid-uuid",
            session_id="also-invalid",
            current_message="Test message",
        )

        settings = MemoryNodeSettings(enable_retrieval=True, fallback_on_error=True)
        node = MemoryRetrievalNode(settings=settings)

        result = await node.process(invalid_state)

        # Should fallback due to error
        assert result["memory_context"]["fallback_mode"] is True
        assert result["agent_results"][0]["success"] is False
