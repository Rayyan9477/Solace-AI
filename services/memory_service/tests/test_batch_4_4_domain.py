"""
Solace-AI Memory Service - Batch 4.4 Domain Tests.

Comprehensive unit tests for entities, value objects, events, config, and knowledge graph.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4

from services.memory_service.src.domain.entities import (
    MemoryTier, RetentionCategory, ContentType, MemoryRecordId,
    MemoryRecordEntity, UserProfileEntity, SessionSummaryEntity, TherapeuticEventEntity,
)
from services.memory_service.src.domain.value_objects import (
    MemoryTierSpec, RetentionPolicyType, MemoryTierConfig, RetentionPolicy,
    TokenBudget, RetrievalQuery, RetrievalResult, EmotionalState,
    ConsolidationRequest, ConsolidationOutcome, ContextWindow,
)
from services.memory_service.src.events import (
    MemoryStoredEvent, MemoryRetrievedEvent, MemoryConsolidatedEvent,
    MemoryDecayedEvent, ContextAssembledEvent, SafetyMemoryCreatedEvent,
    MemoryEventFactory, MemoryEventPublisher, MemoryEventConsumer,
)
from services.memory_service.src.config import (
    PostgresConfig, RedisConfig, WeaviateConfig, KafkaConfig,
    DecayConfig, ConsolidationConfig, ContextAssemblyConfig,
    MemoryServiceConfig, Settings, get_settings,
)
from services.memory_service.src.domain.knowledge_graph import (
    RelationType, EntityType, KnowledgeTriple, GraphEntity,
    TripleExtractor, KnowledgeGraph,
)


class TestMemoryRecordEntity:
    """Tests for MemoryRecordEntity."""

    def test_create_memory_record(self) -> None:
        user_id = uuid4()
        record = MemoryRecordEntity(user_id=user_id, content="Test content")
        assert record.user_id == user_id
        assert record.content == "Test content"
        assert record.tier == MemoryTier.TIER_3_SESSION
        assert record.retention_strength == Decimal("1.0")

    def test_record_access_boosts_strength(self) -> None:
        record = MemoryRecordEntity(user_id=uuid4(), content="Test", retention_strength=Decimal("0.5"))
        updated = record.record_access()
        assert updated.access_count == 1
        assert updated.retention_strength > Decimal("0.5")

    def test_apply_decay(self) -> None:
        record = MemoryRecordEntity(user_id=uuid4(), content="Test", retention_strength=Decimal("0.8"))
        decayed = record.apply_decay(Decimal("0.1"))
        assert decayed.retention_strength == Decimal("0.7")

    def test_safety_critical_no_decay(self) -> None:
        record = MemoryRecordEntity(user_id=uuid4(), content="Crisis info", is_safety_critical=True)
        decayed = record.apply_decay(Decimal("0.5"))
        assert decayed.retention_strength == Decimal("1.0")

    def test_should_archive(self) -> None:
        record = MemoryRecordEntity(user_id=uuid4(), content="Test", retention_strength=Decimal("0.05"))
        assert record.should_archive(threshold=Decimal("0.1")) is True

    def test_mark_safety_critical(self) -> None:
        record = MemoryRecordEntity(user_id=uuid4(), content="Test")
        marked = record.mark_safety_critical()
        assert marked.is_safety_critical is True
        assert marked.retention_category == RetentionCategory.PERMANENT

    def test_to_storage_dict(self) -> None:
        record = MemoryRecordEntity(user_id=uuid4(), content="Test")
        data = record.to_storage_dict()
        assert "record_id" in data
        assert data["content"] == "Test"
        assert data["tier"] == "tier_3_session"


class TestUserProfileEntity:
    """Tests for UserProfileEntity."""

    def test_create_profile(self) -> None:
        user_id = uuid4()
        profile = UserProfileEntity(user_id=user_id)
        assert profile.user_id == user_id
        assert profile.total_sessions == 0
        assert profile.crisis_count == 0

    def test_add_personal_fact(self) -> None:
        profile = UserProfileEntity(user_id=uuid4())
        updated = profile.add_personal_fact("work", "Software engineer")
        assert "work" in updated.personal_facts
        assert len(updated.personal_facts["work"]) == 1

    def test_record_session(self) -> None:
        profile = UserProfileEntity(user_id=uuid4())
        updated = profile.record_session()
        assert updated.total_sessions == 1
        assert updated.first_session_date is not None
        assert updated.last_session_date is not None

    def test_record_crisis(self) -> None:
        profile = UserProfileEntity(user_id=uuid4())
        updated = profile.record_crisis()
        assert updated.crisis_count == 1
        assert updated.last_crisis_date is not None

    def test_add_trigger(self) -> None:
        profile = UserProfileEntity(user_id=uuid4())
        updated = profile.add_trigger("loud noises")
        assert "loud noises" in updated.triggers

    def test_add_coping_strategy(self) -> None:
        profile = UserProfileEntity(user_id=uuid4())
        updated = profile.add_coping_strategy("deep breathing")
        assert "deep breathing" in updated.coping_strategies

    def test_get_safety_summary(self) -> None:
        profile = UserProfileEntity(user_id=uuid4(), crisis_count=2, triggers=["stress"])
        summary = profile.get_safety_summary()
        assert summary["crisis_count"] == 2
        assert "stress" in summary["triggers"]


class TestSessionSummaryEntity:
    """Tests for SessionSummaryEntity."""

    def test_create_summary(self) -> None:
        summary = SessionSummaryEntity(session_id=uuid4(), user_id=uuid4(), session_number=5, summary_text="Good progress today")
        assert summary.session_number == 5
        assert summary.summary_text == "Good progress today"

    def test_get_therapeutic_value(self) -> None:
        summary = SessionSummaryEntity(session_id=uuid4(), user_id=uuid4(), session_number=1, summary_text="Session",
                                        key_insights=["insight1"], techniques_used=["CBT"], homework_assigned=["task1"])
        value = summary.get_therapeutic_value()
        assert value > Decimal("0.3")

    def test_apply_decay(self) -> None:
        summary = SessionSummaryEntity(session_id=uuid4(), user_id=uuid4(), session_number=1, summary_text="Test")
        decayed = summary.apply_decay(Decimal("0.2"))
        assert decayed.retention_strength == Decimal("0.8")

    def test_to_context_string(self) -> None:
        summary = SessionSummaryEntity(session_id=uuid4(), user_id=uuid4(), session_number=3, summary_text="Discussion about anxiety",
                                        key_topics=["anxiety", "work"])
        context = summary.to_context_string()
        assert "Session 3" in context
        assert "anxiety" in context


class TestTherapeuticEventEntity:
    """Tests for TherapeuticEventEntity."""

    def test_create_crisis_event(self) -> None:
        event = TherapeuticEventEntity.create_crisis_event(user_id=uuid4(), session_id=uuid4(), title="Crisis detected",
                                                            description="User expressed suicidal ideation")
        assert event.event_type == "crisis"
        assert event.is_safety_critical is True
        assert event.severity == "critical"

    def test_create_milestone(self) -> None:
        event = TherapeuticEventEntity.create_milestone(user_id=uuid4(), session_id=uuid4(), title="Completed exposure therapy",
                                                         description="Successfully faced fear")
        assert event.event_type == "milestone"
        assert event.severity == "low"

    def test_link_events(self) -> None:
        event1 = TherapeuticEventEntity(user_id=uuid4(), event_type="progress", title="Progress")
        event2_id = uuid4()
        linked = event1.link_event(event2_id)
        assert event2_id in linked.related_events


class TestRetentionPolicy:
    """Tests for RetentionPolicy value object."""

    def test_permanent_policy(self) -> None:
        policy = RetentionPolicy.permanent()
        assert policy.base_decay_rate == Decimal("0")
        action = policy.get_action(Decimal("0.01"))
        assert action == "retain"

    def test_medium_term_policy(self) -> None:
        policy = RetentionPolicy.medium_term()
        assert policy.base_decay_rate == Decimal("0.01")

    def test_calculate_decay(self) -> None:
        policy = RetentionPolicy.medium_term()
        new_strength = policy.calculate_decay(Decimal("1.0"), hours_elapsed=10)
        assert new_strength < Decimal("1.0")

    def test_get_action_archive(self) -> None:
        policy = RetentionPolicy.medium_term()
        action = policy.get_action(Decimal("0.08"))
        assert action == "archive"

    def test_get_action_delete(self) -> None:
        policy = RetentionPolicy.short_term()
        action = policy.get_action(Decimal("0.005"))
        assert action == "delete"


class TestTokenBudget:
    """Tests for TokenBudget value object."""

    def test_create_budget(self) -> None:
        budget = TokenBudget(total_tokens=8000)
        assert budget.total_tokens == 8000
        assert budget.available_tokens > 0

    def test_allocation(self) -> None:
        budget = TokenBudget(total_tokens=8000, system_prompt_tokens=500, recent_messages_tokens=2000)
        assert budget.allocated_tokens == 2500

    def test_can_fit(self) -> None:
        budget = TokenBudget(total_tokens=8000, reserved_tokens=500)
        assert budget.can_fit(7000) is True
        assert budget.can_fit(8000) is False

    def test_to_breakdown(self) -> None:
        budget = TokenBudget(total_tokens=8000)
        breakdown = budget.to_breakdown()
        assert "total" in breakdown
        assert "available" in breakdown


class TestEmotionalState:
    """Tests for EmotionalState value object."""

    def test_neutral_state(self) -> None:
        state = EmotionalState.neutral()
        assert state.primary_emotion == "neutral"
        assert state.valence == Decimal("0")

    def test_positive_state(self) -> None:
        state = EmotionalState.positive("joyful")
        assert state.is_positive() is True
        assert state.is_negative() is False

    def test_negative_state(self) -> None:
        state = EmotionalState.negative("anxious")
        assert state.is_negative() is True
        assert state.is_positive() is False

    def test_intensity(self) -> None:
        state = EmotionalState(valence=Decimal("0.8"), arousal=Decimal("0.9"), dominance=Decimal("0.5"))
        intensity = state.intensity()
        assert intensity > Decimal("0")


class TestMemoryEvents:
    """Tests for memory event classes."""

    def test_memory_stored_event(self) -> None:
        event = MemoryEventFactory.memory_stored(user_id=uuid4(), session_id=uuid4(), record_id=uuid4(),
                                                  tier="tier_3_session", content_type="user_message",
                                                  importance_score=Decimal("0.7"), storage_backend="postgres", storage_time_ms=15)
        assert event.event_type == "memory.stored"
        assert event.storage_time_ms == 15

    def test_memory_consolidated_event(self) -> None:
        event = MemoryEventFactory.memory_consolidated(user_id=uuid4(), session_id=uuid4(), consolidation_id=uuid4(),
                                                        summary_generated=True, facts_extracted=5, triples_created=3,
                                                        memories_archived=10, consolidation_time_ms=250)
        assert event.event_type == "memory.consolidated"
        assert event.facts_extracted == 5

    def test_safety_memory_event(self) -> None:
        event = MemoryEventFactory.safety_memory_created(user_id=uuid4(), session_id=uuid4(), record_id=uuid4(),
                                                          safety_type="crisis", priority="CRITICAL")
        assert event.event_type == "memory.safety.created"
        assert event.priority == "CRITICAL"


class TestMemoryEventPublisher:
    """Tests for MemoryEventPublisher."""

    def test_publisher_disabled(self) -> None:
        publisher = MemoryEventPublisher(publisher=None)
        stats = publisher.get_stats()
        assert stats["enabled"] is False

    @pytest.mark.asyncio
    async def test_publish_when_disabled(self) -> None:
        publisher = MemoryEventPublisher(publisher=None)
        event = MemoryEventFactory.memory_stored(user_id=uuid4(), session_id=None, record_id=uuid4(),
                                                  tier="tier_3_session", content_type="user_message",
                                                  importance_score=Decimal("0.5"), storage_backend="postgres", storage_time_ms=10)
        result = await publisher.publish(event)
        assert result is False


class TestMemoryEventConsumer:
    """Tests for MemoryEventConsumer."""

    def test_consumer_initialization(self) -> None:
        consumer = MemoryEventConsumer(consumer=None)
        stats = consumer.get_stats()
        assert stats["enabled"] is False

    def test_register_handler(self) -> None:
        consumer = MemoryEventConsumer(consumer=None)
        async def handler(event): pass
        consumer.register_handler("memory.stored", handler)
        assert "memory.stored" in consumer._handlers


class TestConfigSettings:
    """Tests for configuration settings."""

    def test_postgres_config(self) -> None:
        config = PostgresConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert "postgresql+asyncpg" in config.connection_url

    def test_redis_config(self) -> None:
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.working_memory_ttl == 3600
        assert "redis://" in config.connection_url

    def test_weaviate_config(self) -> None:
        config = WeaviateConfig()
        assert config.embedding_dimension == 1536
        assert "http://" in config.http_url

    def test_decay_config(self) -> None:
        config = DecayConfig()
        assert config.enabled is True
        assert config.permanent_decay_rate == Decimal("0")

    def test_memory_service_config(self) -> None:
        config = MemoryServiceConfig()
        assert config.service_name == "memory-service"
        assert config.port == 8003

    def test_settings_singleton(self) -> None:
        Settings.reset()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_settings_to_dict(self) -> None:
        Settings.reset()
        settings = get_settings()
        data = settings.to_dict()
        assert "service" in data
        assert "postgres" in data
        assert data["service"]["name"] == "memory-service"


class TestKnowledgeTriple:
    """Tests for KnowledgeTriple."""

    def test_create_triple(self) -> None:
        triple = KnowledgeTriple(user_id=uuid4(), subject="user", subject_type=EntityType.USER,
                                 predicate=RelationType.FEELS, object="anxious", object_type=EntityType.EMOTION)
        assert triple.subject == "user"
        assert triple.predicate == RelationType.FEELS
        assert triple.object == "anxious"

    def test_to_natural_language(self) -> None:
        triple = KnowledgeTriple(user_id=uuid4(), subject="user", subject_type=EntityType.USER,
                                 predicate=RelationType.WORKS_AT, object="Google", object_type=EntityType.ORGANIZATION)
        text = triple.to_natural_language()
        assert "user works at Google" == text

    def test_matches_pattern(self) -> None:
        triple = KnowledgeTriple(user_id=uuid4(), subject="user", subject_type=EntityType.USER,
                                 predicate=RelationType.HAS, object="dog", object_type=EntityType.PERSON)
        assert triple.matches_pattern(subject="user") is True
        assert triple.matches_pattern(predicate=RelationType.HAS) is True
        assert triple.matches_pattern(obj="cat") is False

    def test_invalidate(self) -> None:
        triple = KnowledgeTriple(user_id=uuid4(), subject="user", subject_type=EntityType.USER,
                                 predicate=RelationType.FEELS, object="happy", object_type=EntityType.EMOTION)
        invalidated = triple.invalidate()
        assert invalidated.is_active is False
        assert invalidated.valid_to is not None


class TestTripleExtractor:
    """Tests for TripleExtractor."""

    def test_extract_name(self) -> None:
        extractor = TripleExtractor(user_id=uuid4())
        triples = extractor.extract("My name is John")
        names = [t for t in triples if t.predicate == RelationType.IS]
        assert len(names) >= 1

    def test_extract_emotion(self) -> None:
        extractor = TripleExtractor(user_id=uuid4())
        triples = extractor.extract("I'm feeling anxious today")
        emotions = [t for t in triples if t.predicate == RelationType.FEELS]
        assert len(emotions) >= 1

    def test_extract_work(self) -> None:
        extractor = TripleExtractor(user_id=uuid4())
        triples = extractor.extract("I work at Microsoft")
        work = [t for t in triples if t.predicate == RelationType.WORKS_AT]
        assert len(work) >= 1

    def test_extract_condition(self) -> None:
        extractor = TripleExtractor(user_id=uuid4())
        triples = extractor.extract("I have anxiety and depression")
        conditions = [t for t in triples if t.predicate == RelationType.DIAGNOSED_WITH]
        assert len(conditions) >= 1


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph."""

    def test_add_triple(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        triple = KnowledgeTriple(user_id=graph.user_id, subject="user", subject_type=EntityType.USER,
                                 predicate=RelationType.FEELS, object="happy", object_type=EntityType.EMOTION)
        triple_id = graph.add_triple(triple)
        assert triple_id is not None

    def test_add_from_text(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        ids = graph.add_from_text("I'm feeling anxious and I work at Google")
        assert len(ids) >= 1

    def test_query_by_subject(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        graph.add_from_text("I feel happy")
        results = graph.query(subject="user")
        assert len(results) >= 1

    def test_query_by_predicate(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        graph.add_from_text("I feel anxious")
        results = graph.query(predicate=RelationType.FEELS)
        assert len(results) >= 1

    def test_get_user_facts(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        graph.add_from_text("I work at Amazon. I feel stressed.")
        facts = graph.get_user_facts()
        assert len(facts) >= 1

    def test_get_entity_relationships(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        graph.add_from_text("I feel happy. I feel sad.")
        relationships = graph.get_entity_relationships("user")
        assert len(relationships) >= 1

    def test_invalidate_triple(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        triple = KnowledgeTriple(user_id=graph.user_id, subject="user", subject_type=EntityType.USER,
                                 predicate=RelationType.FEELS, object="happy", object_type=EntityType.EMOTION)
        triple_id = graph.add_triple(triple)
        result = graph.invalidate_triple(triple_id)
        assert result is True
        active = graph.query(active_only=True)
        assert all(t.triple_id != triple_id for t in active)

    def test_to_summary(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        graph.add_from_text("I feel happy. I work at Google.")
        summary = graph.to_summary()
        assert "total_triples" in summary
        assert "total_entities" in summary

    def test_clear(self) -> None:
        graph = KnowledgeGraph(user_id=uuid4())
        graph.add_from_text("I feel happy")
        graph.clear()
        summary = graph.to_summary()
        assert summary["total_triples"] == 0


class TestMemoryTierConfig:
    """Tests for MemoryTierConfig."""

    def test_input_buffer_config(self) -> None:
        config = MemoryTierConfig.input_buffer()
        assert config.tier == MemoryTierSpec.TIER_1_INPUT
        assert config.max_latency_ms == 1

    def test_working_memory_config(self) -> None:
        config = MemoryTierConfig.working_memory()
        assert config.tier == MemoryTierSpec.TIER_2_WORKING
        assert config.storage_backend == "redis"

    def test_semantic_memory_config(self) -> None:
        config = MemoryTierConfig.semantic_memory()
        assert config.tier == MemoryTierSpec.TIER_5_SEMANTIC
        assert config.supports_decay is True


class TestConsolidationOutcome:
    """Tests for ConsolidationOutcome."""

    def test_successful_outcome(self) -> None:
        outcome = ConsolidationOutcome(session_id=uuid4(), summary_generated="Good session",
                                        facts_extracted=5, triples_created=3)
        assert outcome.success is True

    def test_failed_outcome(self) -> None:
        outcome = ConsolidationOutcome(session_id=uuid4(), errors=["Failed to generate summary"])
        assert outcome.success is False


class TestContextWindow:
    """Tests for ContextWindow."""

    def test_create_window(self) -> None:
        window = ContextWindow(user_id=uuid4(), session_id=uuid4(), max_tokens=8000, total_tokens=2000)
        assert window.available_tokens == 6000

    def test_utilization(self) -> None:
        window = ContextWindow(user_id=uuid4(), session_id=uuid4(), max_tokens=8000, total_tokens=4000)
        assert window.utilization == Decimal("0.5")

    def test_can_add(self) -> None:
        window = ContextWindow(user_id=uuid4(), session_id=uuid4(), max_tokens=8000, total_tokens=7000)
        assert window.can_add(500) is True
        assert window.can_add(2000) is False
