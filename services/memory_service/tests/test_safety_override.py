"""
Unit tests for Memory Service - Safety Override.
Verifies crisis content is forced to permanent retention with maximum importance,
and normal content retains its original retention category.
"""
from __future__ import annotations

import pytest
from decimal import Decimal
from uuid import uuid4

from services.memory_service.src.domain.service import MemoryService
from services.memory_service.src.domain.models import (
    MemoryServiceSettings,
    StoreMemoryResult,
)


@pytest.fixture
def memory_service() -> MemoryService:
    """Create memory service with default settings (no external repos)."""
    settings = MemoryServiceSettings(
        enable_auto_consolidation=False,
        enable_decay=False,
    )
    return MemoryService(settings=settings)


class TestSafetyOverride:
    """Verify crisis content is forced to permanent retention."""

    @pytest.mark.asyncio
    async def test_crisis_keyword_kill_myself_forces_permanent(
        self, memory_service: MemoryService
    ) -> None:
        """Content containing 'kill myself' should be overridden to permanent retention."""
        await memory_service.initialize()
        user_id = uuid4()
        session_id = uuid4()

        result = await memory_service.store_memory(
            user_id=user_id,
            session_id=session_id,
            content="I want to kill myself",
            content_type="message",
            tier="tier_3_session",
            retention_category="short_term",
            importance_score=Decimal("0.3"),
            metadata={},
        )

        assert isinstance(result, StoreMemoryResult)
        assert result.stored is True

        # Verify the stored record has permanent retention
        records = memory_service._tier_3_session.get(user_id, [])
        assert len(records) == 1
        assert records[0].retention_category == "permanent"
        assert records[0].importance_score == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_crisis_keyword_suicide_forces_permanent(
        self, memory_service: MemoryService
    ) -> None:
        """Content containing 'suicide' should be overridden to permanent retention."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I have been thinking about suicide",
            content_type="message",
            tier="tier_3_session",
            retention_category="short_term",
            importance_score=Decimal("0.2"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert len(records) == 1
        assert records[0].retention_category == "permanent"
        assert records[0].importance_score == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_crisis_keyword_end_my_life_forces_permanent(
        self, memory_service: MemoryService
    ) -> None:
        """Content containing 'end my life' should be overridden to permanent."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I want to end my life right now",
            content_type="message",
            tier="tier_3_session",
            retention_category="medium_term",
            importance_score=Decimal("0.4"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert records[0].retention_category == "permanent"
        assert records[0].importance_score == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_crisis_keyword_self_harm_forces_permanent(
        self, memory_service: MemoryService
    ) -> None:
        """Content containing 'self-harm' should be overridden to permanent."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I have been engaging in self-harm",
            content_type="message",
            tier="tier_3_session",
            retention_category="short_term",
            importance_score=Decimal("0.3"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert records[0].retention_category == "permanent"
        assert records[0].importance_score == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_crisis_keyword_want_to_die_forces_permanent(
        self, memory_service: MemoryService
    ) -> None:
        """Content containing 'want to die' should be overridden to permanent."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="Sometimes I just want to die",
            content_type="message",
            tier="tier_3_session",
            retention_category="short_term",
            importance_score=Decimal("0.5"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert records[0].retention_category == "permanent"
        assert records[0].importance_score == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_crisis_keyword_hurt_myself_forces_permanent(
        self, memory_service: MemoryService
    ) -> None:
        """Content containing 'hurt myself' should be overridden to permanent."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I feel like I want to hurt myself",
            content_type="message",
            tier="tier_3_session",
            retention_category="short_term",
            importance_score=Decimal("0.3"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert records[0].retention_category == "permanent"
        assert records[0].importance_score == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_crisis_override_is_case_insensitive(
        self, memory_service: MemoryService
    ) -> None:
        """Crisis keyword detection should be case-insensitive."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I want to KILL MYSELF",
            content_type="message",
            tier="tier_3_session",
            retention_category="short_term",
            importance_score=Decimal("0.3"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert records[0].retention_category == "permanent"
        assert records[0].importance_score == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_crisis_override_works_on_all_tiers(
        self, memory_service: MemoryService
    ) -> None:
        """Safety override should work regardless of target tier."""
        await memory_service.initialize()
        user_id = uuid4()

        for tier in ["tier_3_session", "tier_4_episodic", "tier_5_semantic"]:
            await memory_service.store_memory(
                user_id=user_id,
                session_id=uuid4(),
                content="I want to kill myself",
                content_type="message",
                tier=tier,
                retention_category="short_term",
                importance_score=Decimal("0.3"),
                metadata={},
            )

        for tier_storage in [
            memory_service._tier_3_session,
            memory_service._tier_4_episodic,
            memory_service._tier_5_semantic,
        ]:
            records = tier_storage.get(user_id, [])
            for record in records:
                assert record.retention_category == "permanent", (
                    f"Crisis record in {record.tier} should be permanent"
                )
                assert record.importance_score == Decimal("1.0")


class TestNormalContentKeepsOriginalCategory:
    """Verify that non-crisis content retains its specified retention category."""

    @pytest.mark.asyncio
    async def test_normal_content_keeps_short_term(
        self, memory_service: MemoryService
    ) -> None:
        """Normal content should keep its short_term retention category."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I had a good day today",
            content_type="message",
            tier="tier_3_session",
            retention_category="short_term",
            importance_score=Decimal("0.3"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert len(records) == 1
        assert records[0].retention_category == "short_term"
        assert records[0].importance_score == Decimal("0.3")

    @pytest.mark.asyncio
    async def test_normal_content_keeps_medium_term(
        self, memory_service: MemoryService
    ) -> None:
        """Normal content should keep its medium_term retention category."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I started a new exercise routine",
            content_type="message",
            tier="tier_3_session",
            retention_category="medium_term",
            importance_score=Decimal("0.5"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert records[0].retention_category == "medium_term"
        assert records[0].importance_score == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_normal_content_keeps_long_term(
        self, memory_service: MemoryService
    ) -> None:
        """Normal content should keep its long_term retention category."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I enjoy painting as a hobby",
            content_type="fact",
            tier="tier_5_semantic",
            retention_category="long_term",
            importance_score=Decimal("0.7"),
            metadata={},
        )

        records = memory_service._tier_5_semantic.get(user_id, [])
        assert records[0].retention_category == "long_term"
        assert records[0].importance_score == Decimal("0.7")

    @pytest.mark.asyncio
    async def test_normal_permanent_keeps_permanent(
        self, memory_service: MemoryService
    ) -> None:
        """Content explicitly marked permanent should stay permanent."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="Patient has diabetes diagnosis",
            content_type="fact",
            tier="tier_5_semantic",
            retention_category="permanent",
            importance_score=Decimal("0.9"),
            metadata={},
        )

        records = memory_service._tier_5_semantic.get(user_id, [])
        assert records[0].retention_category == "permanent"
        assert records[0].importance_score == Decimal("0.9")

    @pytest.mark.asyncio
    async def test_similar_but_not_crisis_keeps_original(
        self, memory_service: MemoryService
    ) -> None:
        """Content with words similar to crisis keywords but not matching
        should keep original category."""
        await memory_service.initialize()
        user_id = uuid4()

        await memory_service.store_memory(
            user_id=user_id,
            session_id=uuid4(),
            content="I watched a documentary about survival",
            content_type="message",
            tier="tier_3_session",
            retention_category="short_term",
            importance_score=Decimal("0.3"),
            metadata={},
        )

        records = memory_service._tier_3_session.get(user_id, [])
        assert records[0].retention_category == "short_term"
        assert records[0].importance_score == Decimal("0.3")


class TestCrisisKeywordsConstant:
    """Verify the CRISIS_KEYWORDS constant is properly configured."""

    def test_crisis_keywords_is_frozenset(self) -> None:
        """CRISIS_KEYWORDS should be an immutable frozenset."""
        assert isinstance(MemoryService.CRISIS_KEYWORDS, frozenset)

    def test_crisis_keywords_contains_expected_entries(self) -> None:
        """CRISIS_KEYWORDS should contain all critical safety phrases."""
        expected = {"suicide", "kill myself", "end my life", "self-harm",
                    "hurt myself", "want to die"}
        assert expected.issubset(MemoryService.CRISIS_KEYWORDS), (
            f"Missing keywords: {expected - MemoryService.CRISIS_KEYWORDS}"
        )

    def test_crisis_keywords_not_empty(self) -> None:
        """CRISIS_KEYWORDS must never be empty."""
        assert len(MemoryService.CRISIS_KEYWORDS) > 0
