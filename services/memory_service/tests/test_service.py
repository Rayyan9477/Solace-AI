"""
Unit tests for Memory Service - Main orchestration service.
Tests 5-tier memory hierarchy, session management, and CRUD operations.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.domain.models import (
    MemoryServiceSettings, MemoryRecord, SessionState,
    StoreMemoryResult, RetrieveMemoryResult, SessionStartResult,
    SessionEndResult, AddMessageResult, ConsolidationResult, UserProfileResult,
)


@pytest.fixture
def memory_service_settings() -> MemoryServiceSettings:
    """Create test settings."""
    return MemoryServiceSettings(
        working_memory_max_tokens=4000,
        session_memory_ttl_hours=24,
        enable_auto_consolidation=False,
        enable_decay=True,
        max_history_per_user=50,
    )


@pytest.fixture
def memory_service(memory_service_settings: MemoryServiceSettings) -> MemoryService:
    """Create memory service instance."""
    return MemoryService(settings=memory_service_settings)


@pytest.fixture
def user_id() -> uuid4:
    """Create a test user ID."""
    return uuid4()


@pytest.fixture
def session_id() -> uuid4:
    """Create a test session ID."""
    return uuid4()


class TestMemoryServiceInitialization:
    """Tests for memory service initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_service(self, memory_service: MemoryService) -> None:
        """Test service initialization."""
        assert memory_service._initialized is False
        await memory_service.initialize()
        assert memory_service._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown_service(self, memory_service: MemoryService) -> None:
        """Test service shutdown."""
        await memory_service.initialize()
        await memory_service.shutdown()
        assert memory_service._initialized is False

    @pytest.mark.asyncio
    async def test_get_status_initialized(self, memory_service: MemoryService) -> None:
        """Test status when initialized."""
        await memory_service.initialize()
        status = await memory_service.get_status()
        assert status["status"] == "operational"
        assert status["initialized"] is True
        assert "statistics" in status
        assert "tier_counts" in status


class TestMemoryStorage:
    """Tests for memory storage operations."""

    @pytest.mark.asyncio
    async def test_store_memory_session_tier(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test storing memory to session tier."""
        await memory_service.initialize()
        result = await memory_service.store_memory(
            user_id=user_id, session_id=uuid4(), content="Test message content",
            content_type="message", tier="tier_3_session",
            retention_category="medium_term", importance_score=Decimal("0.5"),
            metadata={"role": "user"},
        )
        assert isinstance(result, StoreMemoryResult)
        assert result.stored is True
        assert result.tier == "tier_3_session"
        assert result.storage_time_ms >= 0

    @pytest.mark.asyncio
    async def test_store_memory_semantic_tier(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test storing memory to semantic tier."""
        await memory_service.initialize()
        result = await memory_service.store_memory(
            user_id=user_id, session_id=None, content="User prefers morning sessions",
            content_type="fact", tier="tier_5_semantic",
            retention_category="long_term", importance_score=Decimal("0.8"),
            metadata={"fact_type": "preference"},
        )
        assert result.stored is True
        assert result.tier == "tier_5_semantic"

    @pytest.mark.asyncio
    async def test_store_multiple_memories(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test storing multiple memories."""
        await memory_service.initialize()
        for i in range(5):
            result = await memory_service.store_memory(
                user_id=user_id, session_id=uuid4(),
                content=f"Message {i}", content_type="message",
                tier="tier_3_session", retention_category="medium_term",
                importance_score=Decimal("0.5"), metadata={},
            )
            assert result.stored is True


class TestMemoryRetrieval:
    """Tests for memory retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_memories_empty(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test retrieving memories when none exist."""
        await memory_service.initialize()
        result = await memory_service.retrieve_memories(
            user_id=user_id, session_id=None, tiers=None,
            query=None, limit=10, min_importance=Decimal("0.0"),
            time_range_hours=None,
        )
        assert isinstance(result, RetrieveMemoryResult)
        assert result.total_found == 0
        assert result.records == []

    @pytest.mark.asyncio
    async def test_retrieve_memories_after_store(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test retrieving memories after storing."""
        await memory_service.initialize()
        await memory_service.store_memory(
            user_id=user_id, session_id=uuid4(), content="Important message",
            content_type="message", tier="tier_3_session",
            retention_category="medium_term", importance_score=Decimal("0.7"),
            metadata={},
        )
        result = await memory_service.retrieve_memories(
            user_id=user_id, session_id=None, tiers=["tier_3_session"],
            query=None, limit=10, min_importance=Decimal("0.0"),
            time_range_hours=None,
        )
        assert result.total_found == 1
        assert len(result.records) == 1

    @pytest.mark.asyncio
    async def test_retrieve_with_importance_filter(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test retrieving with importance filter."""
        await memory_service.initialize()
        await memory_service.store_memory(
            user_id=user_id, session_id=uuid4(), content="Low importance",
            content_type="message", tier="tier_3_session",
            retention_category="short_term", importance_score=Decimal("0.2"),
            metadata={},
        )
        await memory_service.store_memory(
            user_id=user_id, session_id=uuid4(), content="High importance",
            content_type="message", tier="tier_3_session",
            retention_category="long_term", importance_score=Decimal("0.9"),
            metadata={},
        )
        result = await memory_service.retrieve_memories(
            user_id=user_id, session_id=None, tiers=None,
            query=None, limit=10, min_importance=Decimal("0.5"),
            time_range_hours=None,
        )
        assert result.total_found == 1
        assert result.records[0].importance_score >= Decimal("0.5")

    @pytest.mark.asyncio
    async def test_retrieve_with_semantic_query(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test retrieving with semantic query."""
        await memory_service.initialize()
        await memory_service.store_memory(
            user_id=user_id, session_id=uuid4(), content="I feel anxious today",
            content_type="message", tier="tier_3_session",
            retention_category="medium_term", importance_score=Decimal("0.6"),
            metadata={},
        )
        await memory_service.store_memory(
            user_id=user_id, session_id=uuid4(), content="Weather is nice",
            content_type="message", tier="tier_3_session",
            retention_category="short_term", importance_score=Decimal("0.3"),
            metadata={},
        )
        result = await memory_service.retrieve_memories(
            user_id=user_id, session_id=None, tiers=None,
            query="anxious", limit=10, min_importance=Decimal("0.0"),
            time_range_hours=None,
        )
        assert result.total_found == 1
        assert "anxious" in result.records[0].content


class TestSessionManagement:
    """Tests for session management."""

    @pytest.mark.asyncio
    async def test_start_session(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test starting a new session."""
        await memory_service.initialize()
        result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic",
            initial_context={"mood": "neutral"},
        )
        assert isinstance(result, SessionStartResult)
        assert result.session_number == 1
        assert result.user_profile_loaded is True

    @pytest.mark.asyncio
    async def test_start_multiple_sessions(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test starting multiple sessions increments count."""
        await memory_service.initialize()
        result1 = await memory_service.start_session(user_id=user_id, session_type="therapeutic", initial_context={})
        await memory_service.end_session(user_id, result1.session_id, False, False)
        result2 = await memory_service.start_session(user_id=user_id, session_type="therapeutic", initial_context={})
        assert result2.session_number == 2

    @pytest.mark.asyncio
    async def test_end_session(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test ending a session."""
        await memory_service.initialize()
        start_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        end_result = await memory_service.end_session(
            user_id=user_id, session_id=start_result.session_id,
            trigger_consolidation=False, include_summary=False,
        )
        assert isinstance(end_result, SessionEndResult)
        assert end_result.message_count == 0

    @pytest.mark.asyncio
    async def test_add_message_to_session(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test adding a message to session."""
        await memory_service.initialize()
        start_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        message_result = await memory_service.add_message(
            user_id=user_id, session_id=start_result.session_id,
            role="user", content="I'm feeling stressed about work",
            emotion_detected="stress", importance_override=None, metadata={},
        )
        assert isinstance(message_result, AddMessageResult)
        assert message_result.stored_to_tier == "tier_3_session"
        assert message_result.working_memory_updated is True


class TestContextAssembly:
    """Tests for context assembly."""

    @pytest.mark.asyncio
    async def test_assemble_context_basic(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test basic context assembly without external assembler."""
        await memory_service.initialize()
        start_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        await memory_service.add_message(
            user_id=user_id, session_id=start_result.session_id,
            role="user", content="Hello, I need help",
            emotion_detected=None, importance_override=None, metadata={},
        )
        result = await memory_service.assemble_context(
            user_id=user_id, session_id=start_result.session_id,
            current_message="How are you?", token_budget=4000,
            include_safety=False, include_therapeutic=False,
            retrieval_query=None, priority_topics=[],
        )
        assert result.assembled_context != ""
        assert result.total_tokens > 0


class TestUserProfile:
    """Tests for user profile operations."""

    @pytest.mark.asyncio
    async def test_get_user_profile_empty(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test getting profile for new user."""
        await memory_service.initialize()
        result = await memory_service.get_user_profile(
            user_id=user_id, include_knowledge_graph=False,
            include_session_history=True, session_limit=10,
        )
        assert isinstance(result, UserProfileResult)
        assert result.total_sessions == 0

    @pytest.mark.asyncio
    async def test_get_user_profile_after_sessions(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test getting profile after sessions."""
        await memory_service.initialize()
        start_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        await memory_service.end_session(user_id, start_result.session_id, False, False)
        result = await memory_service.get_user_profile(
            user_id=user_id, include_knowledge_graph=False,
            include_session_history=True, session_limit=10,
        )
        assert result.total_sessions == 1


class TestDataDeletion:
    """Tests for GDPR-compliant data deletion."""

    @pytest.mark.asyncio
    async def test_delete_user_data(self, memory_service: MemoryService, user_id: uuid4) -> None:
        """Test deleting all user data."""
        await memory_service.initialize()
        await memory_service.store_memory(
            user_id=user_id, session_id=uuid4(), content="Test",
            content_type="message", tier="tier_3_session",
            retention_category="medium_term", importance_score=Decimal("0.5"),
            metadata={},
        )
        await memory_service.delete_user_data(user_id=user_id)
        result = await memory_service.retrieve_memories(
            user_id=user_id, session_id=None, tiers=None,
            query=None, limit=10, min_importance=Decimal("0.0"),
            time_range_hours=None,
        )
        assert result.total_found == 0


class TestImportanceCalculation:
    """Tests for importance score calculation."""

    def test_calculate_importance_user_message(self, memory_service: MemoryService) -> None:
        """Test importance calculation for user messages."""
        importance = memory_service._calculate_importance("Regular message", "user")
        assert importance >= Decimal("0.5")
        assert importance <= Decimal("1.0")

    def test_calculate_importance_crisis_keywords(self, memory_service: MemoryService) -> None:
        """Test importance calculation with crisis keywords."""
        importance = memory_service._calculate_importance("I'm in crisis and need help", "user")
        assert importance >= Decimal("0.8")

    def test_calculate_importance_assistant_message(self, memory_service: MemoryService) -> None:
        """Test importance calculation for assistant messages."""
        importance = memory_service._calculate_importance("I understand how you feel.", "assistant")
        assert importance == Decimal("0.5")
