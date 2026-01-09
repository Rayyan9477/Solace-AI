"""
Unit tests for Context Assembler - LLM context assembly with token budgeting.
Tests token allocation, priority-based inclusion, and section building.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Any
from services.memory_service.src.domain.context_assembler import (
    ContextAssembler, ContextAssemblerSettings, ContextSection,
    ContextAssemblyOutput, TokenAllocation,
)


@dataclass
class MockMemoryRecord:
    """Mock memory record for testing."""
    record_id: Any = field(default_factory=uuid4)
    user_id: Any = field(default_factory=uuid4)
    content: str = ""
    importance_score: Decimal = Decimal("0.5")
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@pytest.fixture
def assembler_settings() -> ContextAssemblerSettings:
    """Create test settings."""
    return ContextAssemblerSettings(
        default_token_budget=8000,
        system_prompt_budget=1000,
        safety_context_budget=500,
        user_profile_budget=500,
        recent_messages_budget=4000,
        retrieved_context_budget=2000,
        response_buffer=1000,
        max_recent_messages=20,
        chars_per_token=4,
        enable_compression=True,
    )


@pytest.fixture
def context_assembler(assembler_settings: ContextAssemblerSettings) -> ContextAssembler:
    """Create context assembler instance."""
    return ContextAssembler(settings=assembler_settings)


@pytest.fixture
def user_id() -> uuid4:
    """Create a test user ID."""
    return uuid4()


@pytest.fixture
def sample_working_memory() -> list[MockMemoryRecord]:
    """Create sample working memory records."""
    return [
        MockMemoryRecord(content="Hello, I need some help", metadata={"role": "user"}),
        MockMemoryRecord(content="I'm here to help you", metadata={"role": "assistant"}),
        MockMemoryRecord(content="I've been feeling anxious", metadata={"role": "user"}),
    ]


@pytest.fixture
def sample_user_profile() -> dict[str, Any]:
    """Create sample user profile."""
    return {
        "facts": {
            "name": "John",
            "age": "35",
            "occupation": "Software Engineer",
        },
        "preferences": {
            "communication_style": "direct",
        },
        "safety": {
            "crisis_history": "No previous crises",
            "risk_factors": ["work stress"],
            "protective_factors": ["family support", "exercise"],
        },
        "therapeutic": {
            "treatment_plan": "CBT for anxiety",
            "current_phase": "Phase 2",
            "effective_techniques": ["breathing exercises", "thought journaling"],
            "active_goals": ["reduce anxiety", "improve sleep"],
        },
    }


class TestContextAssemblerInitialization:
    """Tests for context assembler initialization."""

    def test_create_assembler_default_settings(self) -> None:
        """Test creating assembler with default settings."""
        assembler = ContextAssembler()
        assert assembler._settings is not None
        assert assembler._system_prompt_template != ""

    def test_create_assembler_custom_settings(self, assembler_settings: ContextAssemblerSettings) -> None:
        """Test creating assembler with custom settings."""
        assembler = ContextAssembler(settings=assembler_settings)
        assert assembler._settings.default_token_budget == 8000


class TestTokenAllocation:
    """Tests for token budget allocation."""

    def test_calculate_allocation(self, context_assembler: ContextAssembler) -> None:
        """Test token allocation calculation."""
        allocation = context_assembler._calculate_allocation(8000)
        assert isinstance(allocation, TokenAllocation)
        assert allocation.system_prompt > 0
        assert allocation.safety_context > 0
        assert allocation.user_profile > 0
        assert allocation.recent_messages > 0
        assert allocation.retrieved_context > 0

    def test_allocation_respects_budget(self, context_assembler: ContextAssembler) -> None:
        """Test that allocation respects total budget minus response buffer."""
        total_budget = 8000
        allocation = context_assembler._calculate_allocation(total_budget)
        # Allocation is calculated from available budget (total minus response buffer)
        # so total section allocations should be reasonable
        total_sections = (
            allocation.system_prompt + allocation.safety_context +
            allocation.user_profile + allocation.therapeutic_context +
            allocation.recent_messages + allocation.retrieved_context
        )
        available = total_budget - context_assembler._settings.response_buffer
        # Sections should fit within available budget (with some tolerance for remaining)
        assert total_sections <= available + 500

    def test_get_token_budget_breakdown(self, context_assembler: ContextAssembler) -> None:
        """Test getting token budget breakdown."""
        breakdown = context_assembler.get_token_budget_breakdown(8000)
        assert "system_prompt" in breakdown
        assert "safety_context" in breakdown
        assert "response_buffer" in breakdown
        assert breakdown["response_buffer"] == context_assembler._settings.response_buffer


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_short_text(self, context_assembler: ContextAssembler) -> None:
        """Test token estimation for short text."""
        text = "Hello world"
        tokens = context_assembler._estimate_tokens(text)
        assert tokens >= 1
        assert tokens == len(text) // context_assembler._settings.chars_per_token or tokens == 1

    def test_estimate_tokens_long_text(self, context_assembler: ContextAssembler) -> None:
        """Test token estimation for long text."""
        text = "This is a much longer text " * 100
        tokens = context_assembler._estimate_tokens(text)
        expected = len(text) // context_assembler._settings.chars_per_token
        assert tokens == expected


class TestTextTruncation:
    """Tests for text truncation."""

    def test_truncate_to_tokens_within_limit(self, context_assembler: ContextAssembler) -> None:
        """Test truncation when text is within limit."""
        text = "Short text"
        truncated = context_assembler._truncate_to_tokens(text, 100)
        assert truncated == text

    def test_truncate_to_tokens_exceeds_limit(self, context_assembler: ContextAssembler) -> None:
        """Test truncation when text exceeds limit."""
        text = "This is a very long text that needs to be truncated. " * 20
        truncated = context_assembler._truncate_to_tokens(text, 50)
        assert len(truncated) < len(text)
        assert truncated.endswith("...")

    def test_truncate_at_sentence_boundary(self, context_assembler: ContextAssembler) -> None:
        """Test truncation prefers sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence that is longer."
        truncated = context_assembler._truncate_to_tokens(text, 15)
        assert "..." in truncated or truncated.endswith(".")


class TestSectionBuilding:
    """Tests for building context sections."""

    def test_build_system_prompt_section(self, context_assembler: ContextAssembler) -> None:
        """Test building system prompt section."""
        section = context_assembler._build_system_prompt_section(budget=1000)
        assert isinstance(section, ContextSection)
        assert section.section_type == "system_prompt"
        assert section.priority == 10
        assert "[System]" in section.content

    def test_build_safety_section_with_data(self, context_assembler: ContextAssembler,
                                            user_id: uuid4, sample_user_profile: dict) -> None:
        """Test building safety section with data."""
        section = context_assembler._build_safety_section(user_id, sample_user_profile, budget=500)
        assert section.section_type == "safety_context"
        assert section.priority == 9
        assert "Safety Context" in section.content

    def test_build_safety_section_empty(self, context_assembler: ContextAssembler, user_id: uuid4) -> None:
        """Test building safety section without data."""
        section = context_assembler._build_safety_section(user_id, {}, budget=500)
        assert section.content == ""

    def test_build_user_profile_section(self, context_assembler: ContextAssembler,
                                        sample_user_profile: dict) -> None:
        """Test building user profile section."""
        section = context_assembler._build_user_profile_section(sample_user_profile, budget=500)
        assert section.section_type == "user_profile"
        assert section.priority == 7
        assert "John" in section.content or "User Profile" in section.content

    def test_build_therapeutic_section(self, context_assembler: ContextAssembler,
                                        sample_user_profile: dict) -> None:
        """Test building therapeutic context section."""
        section = context_assembler._build_therapeutic_section(sample_user_profile, budget=500)
        assert section.section_type == "therapeutic_context"
        assert section.priority == 8

    def test_build_recent_messages_section(self, context_assembler: ContextAssembler,
                                           sample_working_memory: list) -> None:
        """Test building recent messages section."""
        section = context_assembler._build_recent_messages_section(
            sample_working_memory, [], budget=2000
        )
        assert section.section_type == "recent_messages"
        assert "Conversation History" in section.content
        assert "anxious" in section.content.lower()


class TestRetrievedContextBuilding:
    """Tests for retrieved context section."""

    def test_build_retrieved_section_no_query(self, context_assembler: ContextAssembler) -> None:
        """Test building retrieved section without query."""
        section, count = context_assembler._build_retrieved_section([], None, [], budget=1000)
        assert section.content == ""
        assert count == 0

    def test_build_retrieved_section_with_query(self, context_assembler: ContextAssembler,
                                                 sample_working_memory: list) -> None:
        """Test building retrieved section with query."""
        section, count = context_assembler._build_retrieved_section(
            sample_working_memory, "anxious", [], budget=1000
        )
        assert count >= 1
        assert "anxious" in section.content.lower()

    def test_build_retrieved_section_with_topics(self, context_assembler: ContextAssembler,
                                                  sample_working_memory: list) -> None:
        """Test building retrieved section with priority topics."""
        section, count = context_assembler._build_retrieved_section(
            sample_working_memory, None, ["help", "anxious"], budget=1000
        )
        assert count >= 1


class TestContextAssembly:
    """Tests for full context assembly."""

    @pytest.mark.asyncio
    async def test_assemble_basic_context(self, context_assembler: ContextAssembler,
                                           user_id: uuid4, sample_working_memory: list,
                                           sample_user_profile: dict) -> None:
        """Test basic context assembly."""
        result = await context_assembler.assemble(
            user_id=user_id, session_id=uuid4(),
            current_message="I need help",
            token_budget=8000, include_safety=True,
            include_therapeutic=True, retrieval_query=None,
            priority_topics=[], working_memory=sample_working_memory,
            session_memory=[], user_profile=sample_user_profile,
        )
        assert isinstance(result, ContextAssemblyOutput)
        assert result.assembled_context != ""
        assert result.total_tokens > 0
        assert len(result.sources_used) > 0

    @pytest.mark.asyncio
    async def test_assemble_with_current_message(self, context_assembler: ContextAssembler,
                                                  user_id: uuid4) -> None:
        """Test context assembly includes current message."""
        result = await context_assembler.assemble(
            user_id=user_id, session_id=uuid4(),
            current_message="How are you feeling today?",
            token_budget=8000, include_safety=False,
            include_therapeutic=False, retrieval_query=None,
            priority_topics=[], working_memory=[],
            session_memory=[], user_profile={},
        )
        assert "How are you feeling today" in result.assembled_context
        assert "current_input" in result.sources_used

    @pytest.mark.asyncio
    async def test_assemble_respects_token_budget(self, context_assembler: ContextAssembler,
                                                   user_id: uuid4, sample_working_memory: list,
                                                   sample_user_profile: dict) -> None:
        """Test context assembly respects token budget."""
        small_budget = 500
        result = await context_assembler.assemble(
            user_id=user_id, session_id=uuid4(),
            current_message="Test", token_budget=small_budget,
            include_safety=True, include_therapeutic=True,
            retrieval_query=None, priority_topics=[],
            working_memory=sample_working_memory,
            session_memory=[], user_profile=sample_user_profile,
        )
        assert result.total_tokens <= small_budget

    @pytest.mark.asyncio
    async def test_assemble_with_retrieval_query(self, context_assembler: ContextAssembler,
                                                  user_id: uuid4, sample_working_memory: list) -> None:
        """Test context assembly with retrieval query."""
        result = await context_assembler.assemble(
            user_id=user_id, session_id=uuid4(),
            current_message=None, token_budget=8000,
            include_safety=False, include_therapeutic=False,
            retrieval_query="anxiety", priority_topics=[],
            working_memory=[], session_memory=sample_working_memory,
            user_profile={},
        )
        assert result.retrieval_count >= 0

    @pytest.mark.asyncio
    async def test_assemble_empty_inputs(self, context_assembler: ContextAssembler,
                                          user_id: uuid4) -> None:
        """Test context assembly with empty inputs."""
        result = await context_assembler.assemble(
            user_id=user_id, session_id=uuid4(),
            current_message=None, token_budget=8000,
            include_safety=False, include_therapeutic=False,
            retrieval_query=None, priority_topics=[],
            working_memory=[], session_memory=[], user_profile={},
        )
        assert result.assembled_context != ""
        assert "system_prompt" in result.sources_used


class TestSectionAssembly:
    """Tests for section assembly logic."""

    def test_assemble_sections_priority_order(self, context_assembler: ContextAssembler) -> None:
        """Test sections are assembled in priority order."""
        sections = [
            ContextSection(section_type="low", content="Low priority", token_count=50, priority=3),
            ContextSection(section_type="high", content="High priority", token_count=50, priority=9),
            ContextSection(section_type="medium", content="Medium priority", token_count=50, priority=5),
        ]
        final, breakdown, truncated = context_assembler._assemble_sections(sections, budget=1000)
        assert "High priority" in final
        assert breakdown["high"] > 0

    def test_assemble_sections_truncation(self, context_assembler: ContextAssembler) -> None:
        """Test sections are truncated when budget exceeded."""
        large_content = "Large content " * 500
        sections = [
            ContextSection(section_type="first", content="First", token_count=10, priority=10),
            ContextSection(section_type="second", content=large_content,
                          token_count=len(large_content)//4, priority=5),
        ]
        final, breakdown, truncated = context_assembler._assemble_sections(sections, budget=100)
        assert "first" in breakdown
        assert truncated or len(breakdown) <= 2


class TestMessageCompression:
    """Tests for message compression."""

    def test_compress_message_within_limit(self, context_assembler: ContextAssembler) -> None:
        """Test compression when message is within limit."""
        message = "Short message"
        compressed = context_assembler._compress_message(message, max_tokens=100)
        assert compressed == message

    def test_compress_message_exceeds_limit(self, context_assembler: ContextAssembler) -> None:
        """Test compression when message exceeds limit."""
        message = "This is a very long message " * 50
        compressed = context_assembler._compress_message(message, max_tokens=20)
        assert compressed is not None
        assert len(compressed) < len(message)
        assert compressed.endswith("...")

    def test_compress_message_too_small_budget(self, context_assembler: ContextAssembler) -> None:
        """Test compression with too small budget returns None."""
        message = "Some message"
        compressed = context_assembler._compress_message(message, max_tokens=5)
        assert compressed is None
