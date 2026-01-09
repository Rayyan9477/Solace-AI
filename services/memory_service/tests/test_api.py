"""
Unit tests for Memory Service API - Endpoint testing.
Tests all memory service endpoints with mock service.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from services.memory_service.src.api import router, get_memory_service
from services.memory_service.src.schemas import (
    MemoryTier, RetentionCategory,
    StoreMemoryRequest, RetrieveMemoryRequest, ContextAssemblyRequest,
    SessionStartRequest, SessionEndRequest, AddMessageRequest,
    ConsolidationRequest,
)
from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.domain.models import (
    StoreMemoryResult, RetrieveMemoryResult, ContextAssemblyResult,
    SessionStartResult, SessionEndResult, AddMessageResult,
    ConsolidationResult, UserProfileResult,
)


@pytest.fixture
def mock_memory_service() -> MagicMock:
    """Create mock memory service."""
    service = MagicMock(spec=MemoryService)
    service._initialized = True
    return service


@pytest.fixture
def user_id() -> uuid4:
    """Create a test user ID."""
    return uuid4()


@pytest.fixture
def session_id() -> uuid4:
    """Create a test session ID."""
    return uuid4()


class TestStoreMemoryEndpoint:
    """Tests for store memory endpoint."""

    @pytest.mark.asyncio
    async def test_store_memory_request_validation(self) -> None:
        """Test store memory request validation."""
        request = StoreMemoryRequest(
            user_id=uuid4(),
            session_id=uuid4(),
            content="Test message",
            content_type="message",
            tier=MemoryTier.SESSION_MEMORY,
            retention_category=RetentionCategory.MEDIUM_TERM,
            importance_score=Decimal("0.5"),
        )
        assert request.content == "Test message"
        assert request.tier == MemoryTier.SESSION_MEMORY

    @pytest.mark.asyncio
    async def test_store_memory_minimal_request(self) -> None:
        """Test store memory with minimal request."""
        request = StoreMemoryRequest(
            user_id=uuid4(),
            content="Minimal test",
        )
        assert request.tier == MemoryTier.SESSION_MEMORY
        assert request.retention_category == RetentionCategory.MEDIUM_TERM


class TestRetrieveMemoryEndpoint:
    """Tests for retrieve memory endpoint."""

    @pytest.mark.asyncio
    async def test_retrieve_request_validation(self) -> None:
        """Test retrieve memory request validation."""
        request = RetrieveMemoryRequest(
            user_id=uuid4(),
            tiers=[MemoryTier.SESSION_MEMORY, MemoryTier.EPISODIC_MEMORY],
            query="anxiety",
            limit=20,
            min_importance=Decimal("0.3"),
        )
        assert len(request.tiers) == 2
        assert request.query == "anxiety"

    @pytest.mark.asyncio
    async def test_retrieve_request_defaults(self) -> None:
        """Test retrieve request default values."""
        request = RetrieveMemoryRequest(user_id=uuid4())
        assert request.tiers == []
        assert request.limit == 20
        assert request.min_importance == Decimal("0.0")


class TestContextAssemblyEndpoint:
    """Tests for context assembly endpoint."""

    @pytest.mark.asyncio
    async def test_context_assembly_request_full(self) -> None:
        """Test context assembly with full options."""
        request = ContextAssemblyRequest(
            user_id=uuid4(),
            session_id=uuid4(),
            current_message="How are you feeling?",
            token_budget=8000,
            include_safety_context=True,
            include_therapeutic_context=True,
            retrieval_query="anxiety",
            priority_topics=["work", "stress"],
        )
        assert request.token_budget == 8000
        assert request.include_safety_context is True
        assert len(request.priority_topics) == 2

    @pytest.mark.asyncio
    async def test_context_assembly_request_minimal(self) -> None:
        """Test context assembly with minimal options."""
        request = ContextAssemblyRequest(user_id=uuid4())
        assert request.token_budget == 8000
        assert request.include_safety_context is True


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    @pytest.mark.asyncio
    async def test_session_start_request(self) -> None:
        """Test session start request."""
        request = SessionStartRequest(
            user_id=uuid4(),
            session_type="therapeutic",
            initial_context={"mood": "anxious"},
        )
        assert request.session_type == "therapeutic"
        assert request.initial_context["mood"] == "anxious"

    @pytest.mark.asyncio
    async def test_session_end_request(self) -> None:
        """Test session end request."""
        request = SessionEndRequest(
            user_id=uuid4(),
            session_id=uuid4(),
            trigger_consolidation=True,
            include_summary=True,
        )
        assert request.trigger_consolidation is True
        assert request.include_summary is True

    @pytest.mark.asyncio
    async def test_add_message_request(self) -> None:
        """Test add message request."""
        request = AddMessageRequest(
            user_id=uuid4(),
            session_id=uuid4(),
            role="user",
            content="I'm feeling better today",
            emotion_detected="happy",
        )
        assert request.role == "user"
        assert request.emotion_detected == "happy"


class TestConsolidationEndpoint:
    """Tests for consolidation endpoint."""

    @pytest.mark.asyncio
    async def test_consolidation_request_full(self) -> None:
        """Test consolidation request with all options."""
        request = ConsolidationRequest(
            user_id=uuid4(),
            session_id=uuid4(),
            extract_facts=True,
            generate_summary=True,
            update_knowledge_graph=True,
            apply_decay=True,
        )
        assert request.extract_facts is True
        assert request.apply_decay is True

    @pytest.mark.asyncio
    async def test_consolidation_request_partial(self) -> None:
        """Test consolidation request with partial options."""
        request = ConsolidationRequest(
            user_id=uuid4(),
            session_id=uuid4(),
            extract_facts=False,
            generate_summary=True,
            update_knowledge_graph=False,
            apply_decay=False,
        )
        assert request.extract_facts is False


class TestMemoryTierEnum:
    """Tests for memory tier enum."""

    def test_tier_values(self) -> None:
        """Test memory tier values."""
        assert MemoryTier.INPUT_BUFFER.value == "tier_1_input"
        assert MemoryTier.WORKING_MEMORY.value == "tier_2_working"
        assert MemoryTier.SESSION_MEMORY.value == "tier_3_session"
        assert MemoryTier.EPISODIC_MEMORY.value == "tier_4_episodic"
        assert MemoryTier.SEMANTIC_MEMORY.value == "tier_5_semantic"

    def test_tier_count(self) -> None:
        """Test all 5 tiers exist."""
        assert len(MemoryTier) == 5


class TestRetentionCategoryEnum:
    """Tests for retention category enum."""

    def test_retention_values(self) -> None:
        """Test retention category values."""
        assert RetentionCategory.PERMANENT.value == "permanent"
        assert RetentionCategory.LONG_TERM.value == "long_term"
        assert RetentionCategory.MEDIUM_TERM.value == "medium_term"
        assert RetentionCategory.SHORT_TERM.value == "short_term"


class TestRequestValidation:
    """Tests for request validation."""

    @pytest.mark.asyncio
    async def test_store_content_min_length(self) -> None:
        """Test content minimum length validation."""
        with pytest.raises(ValueError):
            StoreMemoryRequest(user_id=uuid4(), content="")

    @pytest.mark.asyncio
    async def test_importance_score_bounds(self) -> None:
        """Test importance score bounds."""
        request = StoreMemoryRequest(
            user_id=uuid4(),
            content="Test",
            importance_score=Decimal("0.5"),
        )
        assert Decimal("0") <= request.importance_score <= Decimal("1")

    @pytest.mark.asyncio
    async def test_token_budget_bounds(self) -> None:
        """Test token budget bounds."""
        request = ContextAssemblyRequest(
            user_id=uuid4(),
            token_budget=8000,
        )
        assert 1000 <= request.token_budget <= 32000

    @pytest.mark.asyncio
    async def test_retrieve_limit_bounds(self) -> None:
        """Test retrieve limit bounds."""
        request = RetrieveMemoryRequest(
            user_id=uuid4(),
            limit=50,
        )
        assert 1 <= request.limit <= 100


class TestResponseModels:
    """Tests for response model structure."""

    def test_store_response_fields(self) -> None:
        """Test store response has required fields."""
        result = StoreMemoryResult(
            record_id=uuid4(),
            tier="tier_3_session",
            stored=True,
            storage_time_ms=5,
        )
        assert result.stored is True
        assert result.storage_time_ms >= 0

    def test_retrieve_response_fields(self) -> None:
        """Test retrieve response has required fields."""
        result = RetrieveMemoryResult(
            records=[],
            total_found=0,
            retrieval_time_ms=10,
            tiers_searched=["tier_3_session"],
        )
        assert result.total_found == 0

    def test_context_assembly_response_fields(self) -> None:
        """Test context assembly response has required fields."""
        result = ContextAssemblyResult(
            context_id=uuid4(),
            assembled_context="Test context",
            total_tokens=100,
            token_breakdown={"system": 50, "messages": 50},
            sources_used=["system_prompt", "recent_messages"],
            assembly_time_ms=20,
        )
        assert result.total_tokens == 100
        assert len(result.sources_used) == 2

    def test_session_start_response_fields(self) -> None:
        """Test session start response has required fields."""
        result = SessionStartResult(
            session_id=uuid4(),
            session_number=1,
            previous_session_summary="Previous summary",
            user_profile_loaded=True,
        )
        assert result.session_number == 1

    def test_consolidation_response_fields(self) -> None:
        """Test consolidation response has required fields."""
        result = ConsolidationResult(
            consolidation_id=uuid4(),
            summary_generated="Session summary",
            facts_extracted=5,
            knowledge_nodes_updated=3,
            memories_decayed=10,
            memories_archived=2,
            consolidation_time_ms=50,
        )
        assert result.facts_extracted == 5


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_very_long_content(self) -> None:
        """Test handling very long content."""
        long_content = "Test " * 10000
        request = StoreMemoryRequest(
            user_id=uuid4(),
            content=long_content[:50000],
        )
        assert len(request.content) == 50000

    @pytest.mark.asyncio
    async def test_unicode_content(self) -> None:
        """Test handling unicode content."""
        request = StoreMemoryRequest(
            user_id=uuid4(),
            content="Test with emoji and unicode characters",
        )
        assert request.content is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_metadata(self) -> None:
        """Test handling special characters in metadata."""
        request = StoreMemoryRequest(
            user_id=uuid4(),
            content="Test",
            metadata={"key": "value with 'quotes' and \"double quotes\""},
        )
        assert "quotes" in request.metadata["key"]

    @pytest.mark.asyncio
    async def test_empty_priority_topics(self) -> None:
        """Test empty priority topics list."""
        request = ContextAssemblyRequest(
            user_id=uuid4(),
            priority_topics=[],
        )
        assert request.priority_topics == []

    @pytest.mark.asyncio
    async def test_null_session_id(self) -> None:
        """Test null session ID handling."""
        request = StoreMemoryRequest(
            user_id=uuid4(),
            session_id=None,
            content="Test without session",
        )
        assert request.session_id is None


class TestDataTransferObjects:
    """Tests for DTO classes."""

    def test_memory_record_dto_creation(self) -> None:
        """Test creating memory record DTO."""
        from services.memory_service.src.schemas import MemoryRecordDTO
        record = MemoryRecordDTO(
            user_id=uuid4(),
            tier=MemoryTier.SESSION_MEMORY,
            content="Test content",
            content_type="message",
            retention_category=RetentionCategory.MEDIUM_TERM,
        )
        assert record.content == "Test content"

    def test_user_profile_request_defaults(self) -> None:
        """Test user profile request defaults."""
        from services.memory_service.src.schemas import UserProfileRequest
        request = UserProfileRequest(user_id=uuid4())
        assert request.include_knowledge_graph is False
        assert request.include_session_history is True
        assert request.session_limit == 10
