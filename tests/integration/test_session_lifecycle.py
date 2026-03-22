"""
Integration tests for session lifecycle: create -> message -> end -> consolidation.

Verifies the MemoryService session management methods work correctly in
isolation (in-memory only, no external infrastructure dependencies).
"""
from __future__ import annotations

import pytest
from decimal import Decimal
from uuid import uuid4

from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.domain.models import (
    MemoryServiceSettings,
    SessionStartResult,
    SessionEndResult,
    AddMessageResult,
)


@pytest.fixture
def memory_settings() -> MemoryServiceSettings:
    """Settings tuned for fast, in-memory-only testing."""
    return MemoryServiceSettings(
        enable_auto_consolidation=False,
        enable_decay=False,
        working_memory_max_tokens=4000,
        max_tier_records_per_user=100,
    )


@pytest.fixture
def memory_service(memory_settings: MemoryServiceSettings) -> MemoryService:
    """Create MemoryService with no external repos (pure in-memory)."""
    return MemoryService(settings=memory_settings)


class TestSessionLifecycle:
    """Integration test for session create -> message -> end flow."""

    @pytest.mark.asyncio
    async def test_initialize_service(self, memory_service: MemoryService) -> None:
        """Service initialization should succeed without external repos."""
        await memory_service.initialize()
        assert memory_service._initialized is True

    @pytest.mark.asyncio
    async def test_session_start(self, memory_service: MemoryService) -> None:
        """Starting a session should return a valid SessionStartResult."""
        await memory_service.initialize()

        user_id = uuid4()
        result = await memory_service.start_session(
            user_id=user_id,
            session_type="therapeutic",
            initial_context={"source": "test"},
        )

        assert isinstance(result, SessionStartResult)
        assert result.session_number >= 1
        assert result.session_id is not None
        assert result.user_profile_loaded is True

    @pytest.mark.asyncio
    async def test_session_start_increments_session_number(
        self, memory_service: MemoryService
    ) -> None:
        """Subsequent sessions for the same user should have increasing numbers."""
        await memory_service.initialize()

        user_id = uuid4()
        result1 = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        # End first session before starting next
        await memory_service.end_session(
            user_id=user_id,
            session_id=result1.session_id,
            trigger_consolidation=False,
            include_summary=False,
        )
        result2 = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )

        assert result2.session_number == result1.session_number + 1

    @pytest.mark.asyncio
    async def test_add_message_to_session(self, memory_service: MemoryService) -> None:
        """Adding a message should store it in session memory tier."""
        await memory_service.initialize()

        user_id = uuid4()
        session_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        session_id = session_result.session_id

        msg_result = await memory_service.add_message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content="I feel anxious today",
            emotion_detected="anxiety",
            importance_override=None,
            metadata={},
        )

        assert isinstance(msg_result, AddMessageResult)
        assert msg_result.stored_to_tier == "tier_3_session"
        assert msg_result.working_memory_updated is True

    @pytest.mark.asyncio
    async def test_retrieve_memories_after_message(
        self, memory_service: MemoryService
    ) -> None:
        """After adding a message, retrieving memories should return it."""
        await memory_service.initialize()

        user_id = uuid4()
        session_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        session_id = session_result.session_id

        await memory_service.add_message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content="I feel anxious today",
            emotion_detected=None,
            importance_override=None,
            metadata={},
        )

        memories = await memory_service.retrieve_memories(
            user_id=user_id,
            session_id=session_id,
            tiers=["tier_3_session"],
            query=None,
            limit=10,
            min_importance=Decimal("0"),
            time_range_hours=None,
        )

        assert memories.total_found > 0
        assert len(memories.records) > 0
        # Verify the stored content
        contents = [r.content for r in memories.records]
        assert any("anxious" in c for c in contents)

    @pytest.mark.asyncio
    async def test_session_end_returns_result(
        self, memory_service: MemoryService
    ) -> None:
        """Ending a session should return a valid SessionEndResult."""
        await memory_service.initialize()

        user_id = uuid4()
        session_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        session_id = session_result.session_id

        await memory_service.add_message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content="Test message for session",
            emotion_detected=None,
            importance_override=None,
            metadata={},
        )

        end_result = await memory_service.end_session(
            user_id=user_id,
            session_id=session_id,
            trigger_consolidation=False,
            include_summary=False,
        )

        assert isinstance(end_result, SessionEndResult)
        assert end_result.message_count == 1
        assert end_result.duration_minutes >= 0

    @pytest.mark.asyncio
    async def test_crisis_content_forces_permanent_retention(
        self, memory_service: MemoryService
    ) -> None:
        """Messages containing crisis keywords should be marked permanent."""
        await memory_service.initialize()

        user_id = uuid4()
        session_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        session_id = session_result.session_id

        await memory_service.add_message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content="I want to hurt myself",
            emotion_detected=None,
            importance_override=None,
            metadata={},
        )

        # Retrieve and verify the record has permanent retention
        memories = await memory_service.retrieve_memories(
            user_id=user_id,
            session_id=session_id,
            tiers=["tier_3_session"],
            query=None,
            limit=10,
            min_importance=Decimal("0"),
            time_range_hours=None,
        )
        crisis_records = [
            r for r in memories.records if "hurt myself" in r.content
        ]
        assert len(crisis_records) > 0
        for record in crisis_records:
            assert record.retention_category == "permanent"
            assert record.importance_score == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_multiple_messages_in_session(
        self, memory_service: MemoryService
    ) -> None:
        """Multiple messages should all be retrievable within the session."""
        await memory_service.initialize()

        user_id = uuid4()
        session_result = await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )
        session_id = session_result.session_id

        messages = [
            ("user", "I've been feeling stressed"),
            ("assistant", "I hear you. Can you tell me more?"),
            ("user", "Work has been overwhelming"),
        ]

        for role, content in messages:
            await memory_service.add_message(
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                emotion_detected=None,
                importance_override=None,
                metadata={},
            )

        end_result = await memory_service.end_session(
            user_id=user_id,
            session_id=session_id,
            trigger_consolidation=False,
            include_summary=False,
        )

        assert end_result.message_count == 3

    @pytest.mark.asyncio
    async def test_shutdown_cleans_active_sessions(
        self, memory_service: MemoryService
    ) -> None:
        """Shutdown should end all active sessions and mark service uninitialized."""
        await memory_service.initialize()

        user_id = uuid4()
        await memory_service.start_session(
            user_id=user_id, session_type="therapeutic", initial_context={},
        )

        assert len(memory_service._active_sessions) == 1

        await memory_service.shutdown()

        assert len(memory_service._active_sessions) == 0
        assert memory_service._initialized is False
