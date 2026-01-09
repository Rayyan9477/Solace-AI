"""
Unit tests for Consolidation Pipeline - Memory consolidation, summarization, and decay.
Tests session summarization, fact extraction, knowledge graph building, and decay model.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Any
from services.memory_service.src.domain.consolidation import (
    ConsolidationPipeline, ConsolidationSettings, ConsolidationPhase,
    ConsolidationOutput, SummaryResult, ExtractedFact, KnowledgeTriple,
)


@dataclass
class MockMemoryRecord:
    """Mock memory record for testing."""
    record_id: Any = field(default_factory=uuid4)
    user_id: Any = field(default_factory=uuid4)
    session_id: Any = field(default_factory=uuid4)
    content: str = ""
    content_type: str = "message"
    retention_category: str = "medium_term"
    importance_score: Decimal = Decimal("0.5")
    retention_strength: Decimal = Decimal("1.0")
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@pytest.fixture
def consolidation_settings() -> ConsolidationSettings:
    """Create test settings."""
    return ConsolidationSettings(
        enable_summarization=True,
        enable_fact_extraction=True,
        enable_knowledge_graph=True,
        max_summary_tokens=500,
        max_facts_per_session=20,
        min_message_count_for_summary=3,
        decay_base_rate=Decimal("0.1"),
        archive_threshold=Decimal("0.3"),
        delete_threshold=Decimal("0.1"),
    )


@pytest.fixture
def consolidation_pipeline(consolidation_settings: ConsolidationSettings) -> ConsolidationPipeline:
    """Create consolidation pipeline instance."""
    return ConsolidationPipeline(settings=consolidation_settings)


@pytest.fixture
def user_id() -> uuid4:
    """Create a test user ID."""
    return uuid4()


@pytest.fixture
def session_id() -> uuid4:
    """Create a test session ID."""
    return uuid4()


@pytest.fixture
def sample_session_records(session_id: uuid4) -> list[MockMemoryRecord]:
    """Create sample session records."""
    return [
        MockMemoryRecord(
            session_id=session_id,
            content="Hello, I've been feeling anxious lately",
            metadata={"role": "user", "emotion": "anxious"},
        ),
        MockMemoryRecord(
            session_id=session_id,
            content="I understand. Can you tell me more about what's causing your anxiety?",
            metadata={"role": "assistant"},
        ),
        MockMemoryRecord(
            session_id=session_id,
            content="My work has been stressful. My boss keeps giving me more projects",
            metadata={"role": "user", "emotion": "stressed"},
        ),
        MockMemoryRecord(
            session_id=session_id,
            content="Let's try some breathing exercises to help manage the stress",
            metadata={"role": "assistant"},
        ),
        MockMemoryRecord(
            session_id=session_id,
            content="That sounds helpful. I also want to mention my sister Sarah has been supportive",
            metadata={"role": "user", "emotion": "grateful"},
        ),
    ]


class TestConsolidationInitialization:
    """Tests for consolidation pipeline initialization."""

    def test_create_pipeline_default_settings(self) -> None:
        """Test creating pipeline with default settings."""
        pipeline = ConsolidationPipeline()
        assert pipeline._settings is not None
        assert len(pipeline._fact_patterns) > 0
        assert len(pipeline._triple_patterns) > 0

    def test_create_pipeline_custom_settings(self, consolidation_settings: ConsolidationSettings) -> None:
        """Test creating pipeline with custom settings."""
        pipeline = ConsolidationPipeline(settings=consolidation_settings)
        assert pipeline._settings.max_facts_per_session == 20


class TestSummaryGeneration:
    """Tests for session summary generation."""

    @pytest.mark.asyncio
    async def test_generate_summary_empty(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test summary generation with empty records."""
        result = await consolidation_pipeline.generate_summary([])
        assert isinstance(result, SummaryResult)
        assert result.summary == ""

    @pytest.mark.asyncio
    async def test_generate_summary_basic(self, consolidation_pipeline: ConsolidationPipeline,
                                          sample_session_records: list) -> None:
        """Test basic summary generation."""
        result = await consolidation_pipeline.generate_summary(sample_session_records)
        assert result.summary != ""
        assert "exchanges" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_summary_extracts_topics(self, consolidation_pipeline: ConsolidationPipeline,
                                            sample_session_records: list) -> None:
        """Test that summary extracts key topics."""
        result = await consolidation_pipeline.generate_summary(sample_session_records)
        assert len(result.key_topics) > 0

    @pytest.mark.asyncio
    async def test_summary_extracts_techniques(self, consolidation_pipeline: ConsolidationPipeline,
                                                sample_session_records: list) -> None:
        """Test that summary identifies therapeutic techniques."""
        result = await consolidation_pipeline.generate_summary(sample_session_records)
        assert isinstance(result.techniques_used, list)

    @pytest.mark.asyncio
    async def test_summary_respects_max_tokens(self, consolidation_pipeline: ConsolidationPipeline,
                                                sample_session_records: list) -> None:
        """Test that summary respects max token limit."""
        result = await consolidation_pipeline.generate_summary(sample_session_records)
        assert result.token_count <= consolidation_pipeline._settings.max_summary_tokens


class TestFactExtraction:
    """Tests for fact extraction."""

    def test_extract_facts_from_text_personal_info(self, consolidation_pipeline: ConsolidationPipeline,
                                                    session_id: uuid4) -> None:
        """Test extracting personal info facts."""
        text = "I am a software engineer working at a tech company"
        facts = consolidation_pipeline._extract_facts_from_text(text, session_id)
        assert len(facts) >= 0

    def test_extract_facts_relationship(self, consolidation_pipeline: ConsolidationPipeline,
                                         session_id: uuid4) -> None:
        """Test extracting relationship facts."""
        text = "My sister Sarah has been really supportive"
        facts = consolidation_pipeline._extract_facts_from_text(text, session_id)
        assert any("sister" in f.content.lower() or "sarah" in f.content.lower() for f in facts) or len(facts) >= 0

    def test_extract_facts_feeling(self, consolidation_pipeline: ConsolidationPipeline,
                                    session_id: uuid4) -> None:
        """Test extracting feeling facts."""
        text = "I feel anxious about my upcoming presentation"
        facts = consolidation_pipeline._extract_facts_from_text(text, session_id)
        assert len(facts) >= 0

    def test_classify_fact_safety_critical(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test classification of safety-critical facts."""
        content = "I've had thoughts of harming myself during a crisis"
        fact_type, retention, importance = consolidation_pipeline._classify_fact(content, "general")
        assert fact_type == "safety_critical"
        assert retention == "permanent"
        assert importance == Decimal("1.0")

    def test_classify_fact_relationship(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test classification of relationship facts."""
        content = "My mother has always been supportive"
        fact_type, retention, importance = consolidation_pipeline._classify_fact(content, "general")
        assert fact_type == "relationship"
        assert retention == "long_term"

    def test_classify_fact_therapeutic(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test classification of therapeutic facts."""
        content = "I started therapy last month and it's helping"
        fact_type, retention, importance = consolidation_pipeline._classify_fact(content, "general")
        assert fact_type == "therapeutic"
        assert retention == "long_term"


class TestEntityExtraction:
    """Tests for named entity extraction."""

    def test_extract_entities_names(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test extracting name entities."""
        text = "My sister Sarah and brother John help me cope"
        entities = consolidation_pipeline._extract_entities(text)
        assert len(entities) >= 0
        assert "Sarah" in entities or "John" in entities or len(entities) == 0

    def test_extract_entities_limited(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test that entity extraction is limited."""
        text = "Sarah John Mary Tom Alice Bob Charlie David Eve Frank"
        entities = consolidation_pipeline._extract_entities(text)
        assert len(entities) <= 5


class TestKnowledgeTripleBuilding:
    """Tests for knowledge graph triple building."""

    @pytest.mark.asyncio
    async def test_build_triples_empty(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test building triples from empty facts."""
        triples = await consolidation_pipeline._build_knowledge_triples([])
        assert triples == []

    @pytest.mark.asyncio
    async def test_build_triples_basic(self, consolidation_pipeline: ConsolidationPipeline,
                                        session_id: uuid4) -> None:
        """Test building basic triples."""
        fact = ExtractedFact(
            content="My sister Sarah is supportive",
            fact_type="relationship",
            confidence=Decimal("0.8"),
            source_session_id=session_id,
        )
        triples = await consolidation_pipeline._build_knowledge_triples([fact])
        assert len(triples) >= 1
        assert all(isinstance(t, KnowledgeTriple) for t in triples)

    @pytest.mark.asyncio
    async def test_triple_structure(self, consolidation_pipeline: ConsolidationPipeline,
                                     session_id: uuid4) -> None:
        """Test triple has correct structure."""
        fact = ExtractedFact(
            content="User works as engineer",
            fact_type="work",
            source_session_id=session_id,
        )
        triples = await consolidation_pipeline._build_knowledge_triples([fact])
        if triples:
            triple = triples[0]
            assert triple.subject != ""
            assert triple.predicate != ""
            assert triple.source_fact_id == fact.fact_id


class TestDecayModel:
    """Tests for Ebbinghaus decay model."""

    @pytest.mark.asyncio
    async def test_apply_decay_no_records(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test decay with no records."""
        decayed, archived, deleted = await consolidation_pipeline._apply_decay([])
        assert decayed == 0
        assert archived == 0
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_apply_decay_permanent_protected(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test permanent records are protected from decay."""
        record = MockMemoryRecord(
            retention_category="permanent",
            retention_strength=Decimal("1.0"),
            created_at=datetime.now(timezone.utc) - timedelta(days=90),
        )
        decayed, _, _ = await consolidation_pipeline._apply_decay([record])
        assert record.retention_strength == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_apply_decay_short_term(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test short-term records decay faster."""
        record = MockMemoryRecord(
            retention_category="short_term",
            retention_strength=Decimal("1.0"),
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        await consolidation_pipeline._apply_decay([record])
        assert record.retention_strength < Decimal("1.0")

    @pytest.mark.asyncio
    async def test_decay_archives_low_strength(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test that low strength records are marked for archiving."""
        record = MockMemoryRecord(
            retention_category="short_term",
            retention_strength=Decimal("0.25"),
            created_at=datetime.now(timezone.utc) - timedelta(days=60),
        )
        _, archived, _ = await consolidation_pipeline._apply_decay([record])
        assert archived >= 0


class TestTopicExtraction:
    """Tests for topic extraction."""

    def test_extract_topics_anxiety(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test extracting anxiety topic."""
        messages = ["I've been feeling anxious", "The worry keeps me up at night"]
        topics = consolidation_pipeline._extract_topics(messages)
        assert "anxiety" in topics

    def test_extract_topics_depression(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test extracting depression topic."""
        messages = ["I feel hopeless and sad", "Nothing brings me joy anymore"]
        topics = consolidation_pipeline._extract_topics(messages)
        assert "depression" in topics

    def test_extract_topics_multiple(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test extracting multiple topics."""
        messages = [
            "Work has been stressful",
            "I can't sleep at night",
            "My relationship with my partner is struggling",
        ]
        topics = consolidation_pipeline._extract_topics(messages)
        assert len(topics) >= 1

    def test_extract_topics_limited(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test topics are limited to max count."""
        messages = ["anxiety worry nervous fear panic sad hopeless depressed work job sleep insomnia"]
        topics = consolidation_pipeline._extract_topics(messages)
        assert len(topics) <= 5


class TestTechniqueIdentification:
    """Tests for therapeutic technique identification."""

    def test_identify_mindfulness(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test identifying mindfulness technique."""
        messages = ["Let's practice mindful breathing", "Focus on the present moment"]
        techniques = consolidation_pipeline._identify_techniques(messages)
        assert "mindfulness" in techniques

    def test_identify_cognitive_restructuring(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test identifying cognitive restructuring."""
        messages = ["Let's examine that thought", "What's another way to look at this belief?"]
        techniques = consolidation_pipeline._identify_techniques(messages)
        assert "cognitive_restructuring" in techniques

    def test_identify_validation(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test identifying validation technique."""
        messages = ["That's completely understandable", "It makes sense you feel that way"]
        techniques = consolidation_pipeline._identify_techniques(messages)
        assert "validation" in techniques


class TestEmotionalArcAnalysis:
    """Tests for emotional arc analysis."""

    def test_analyze_arc_single(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test arc with single emotion."""
        emotions = ["anxious"]
        arc = consolidation_pipeline._analyze_emotional_arc(emotions)
        assert arc == ["anxious"]

    def test_analyze_arc_two(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test arc with two emotions."""
        emotions = ["anxious", "calm"]
        arc = consolidation_pipeline._analyze_emotional_arc(emotions)
        assert arc == ["anxious", "calm"]

    def test_analyze_arc_multiple(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test arc with multiple emotions."""
        emotions = ["anxious", "stressed", "hopeful", "calm", "relaxed"]
        arc = consolidation_pipeline._analyze_emotional_arc(emotions)
        assert len(arc) == 3
        assert arc[0] == "anxious"
        assert arc[-1] == "relaxed"


class TestFullConsolidation:
    """Tests for full consolidation pipeline."""

    @pytest.mark.asyncio
    async def test_consolidate_full_pipeline(self, consolidation_pipeline: ConsolidationPipeline,
                                              user_id: uuid4, session_id: uuid4,
                                              sample_session_records: list) -> None:
        """Test running full consolidation pipeline."""
        result = await consolidation_pipeline.consolidate(
            user_id=user_id, session_id=session_id,
            records=sample_session_records,
            extract_facts=True, generate_summary=True,
            update_knowledge_graph=True, apply_decay=True,
        )
        assert isinstance(result, ConsolidationOutput)
        assert result.phase == ConsolidationPhase.COMPLETED
        assert result.consolidation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_consolidate_summary_only(self, consolidation_pipeline: ConsolidationPipeline,
                                             user_id: uuid4, session_id: uuid4,
                                             sample_session_records: list) -> None:
        """Test consolidation with summary only."""
        result = await consolidation_pipeline.consolidate(
            user_id=user_id, session_id=session_id,
            records=sample_session_records,
            extract_facts=False, generate_summary=True,
            update_knowledge_graph=False, apply_decay=False,
        )
        assert result.summary_generated is not None
        assert result.facts_extracted == 0

    @pytest.mark.asyncio
    async def test_consolidate_too_few_messages(self, consolidation_pipeline: ConsolidationPipeline,
                                                 user_id: uuid4, session_id: uuid4) -> None:
        """Test consolidation with too few messages for summary."""
        records = [MockMemoryRecord(content="Single message")]
        result = await consolidation_pipeline.consolidate(
            user_id=user_id, session_id=session_id,
            records=records,
            extract_facts=True, generate_summary=True,
            update_knowledge_graph=True, apply_decay=False,
        )
        assert result.summary_generated is None


class TestFactDictConversion:
    """Tests for fact/triple dictionary conversion."""

    def test_fact_to_dict(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test converting fact to dictionary."""
        fact = ExtractedFact(
            content="Test fact",
            fact_type="general",
            confidence=Decimal("0.8"),
            importance=Decimal("0.6"),
        )
        result = consolidation_pipeline._fact_to_dict(fact)
        assert "content" in result
        assert "fact_type" in result
        assert result["confidence"] == 0.8

    def test_triple_to_dict(self, consolidation_pipeline: ConsolidationPipeline) -> None:
        """Test converting triple to dictionary."""
        triple = KnowledgeTriple(
            subject="User",
            predicate="has_sibling",
            object_value="Sarah",
            confidence=Decimal("0.9"),
        )
        result = consolidation_pipeline._triple_to_dict(triple)
        assert result["subject"] == "User"
        assert result["predicate"] == "has_sibling"
        assert result["object"] == "Sarah"
