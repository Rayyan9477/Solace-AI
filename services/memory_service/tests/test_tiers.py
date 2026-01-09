"""
Solace-AI Memory Service - Unit Tests for Tier Implementations.
Tests for working_memory, session_memory, episodic_memory, semantic_memory, and decay_manager.
"""
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
from services.memory_service.src.domain.working_memory import (
    WorkingMemoryManager, WorkingMemorySettings, InputBufferItem, WorkingMemoryItem,
)
from services.memory_service.src.domain.session_memory import (
    SessionMemoryManager, SessionMemorySettings, SessionStatus, SessionMessage,
)
from services.memory_service.src.domain.episodic_memory import (
    EpisodicMemoryManager, EpisodicMemorySettings, EventType, EventSeverity,
)
from services.memory_service.src.domain.semantic_memory import (
    SemanticMemoryManager, SemanticMemorySettings, FactCategory, FactStatus, KnowledgeGraphQuery,
)
from services.memory_service.src.domain.decay_manager import (
    DecayManager, DecaySettings, RetentionCategory, DecayAction,
)


class TestWorkingMemoryManager:
    """Tests for WorkingMemoryManager (Tier 1-2)."""

    @pytest.fixture
    def manager(self) -> WorkingMemoryManager:
        settings = WorkingMemorySettings(max_tokens=1000, max_messages=20)
        return WorkingMemoryManager(settings)

    @pytest.fixture
    def user_id(self) -> uuid4:
        return uuid4()

    @pytest.fixture
    def session_id(self) -> uuid4:
        return uuid4()

    def test_set_input_buffer(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test setting input buffer (Tier 1)."""
        item = manager.set_input(user_id, session_id, "Hello", "user")
        assert item.user_id == user_id
        assert item.content == "Hello"
        assert item.role == "user"

    def test_get_input_buffer(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test retrieving input buffer."""
        manager.set_input(user_id, session_id, "Test message", "user")
        item = manager.get_input(user_id)
        assert item is not None
        assert item.content == "Test message"

    def test_clear_input_buffer(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test clearing input buffer."""
        manager.set_input(user_id, session_id, "Test", "user")
        assert manager.clear_input(user_id) is True
        assert manager.get_input(user_id) is None

    def test_add_to_working_memory(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test adding to working memory (Tier 2)."""
        item = manager.add_to_working_memory(user_id, session_id, "Test content", "user")
        assert item.content == "Test content"
        assert item.token_count > 0

    def test_get_working_memory(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test retrieving working memory."""
        manager.add_to_working_memory(user_id, session_id, "Message 1", "user")
        manager.add_to_working_memory(user_id, session_id, "Message 2", "assistant")
        items = manager.get_working_memory(user_id)
        assert len(items) == 2

    def test_working_memory_token_budget(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test working memory respects token budget."""
        for i in range(10):
            manager.add_to_working_memory(user_id, session_id, f"Message {i}" * 10, "user")
        items = manager.get_working_memory(user_id, max_tokens=50)
        total_tokens = sum(item.token_count for item in items)
        assert total_tokens <= 50

    def test_context_window_state(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test context window state calculation."""
        manager.add_to_working_memory(user_id, session_id, "Test", "user")
        state = manager.get_context_window_state(user_id)
        assert state.user_id == user_id
        assert state.message_count == 1
        assert state.total_tokens > 0

    def test_clear_working_memory(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test clearing working memory."""
        manager.add_to_working_memory(user_id, session_id, "Test", "user")
        cleared = manager.clear_working_memory(user_id)
        assert cleared == 1
        assert len(manager.get_working_memory(user_id)) == 0

    def test_summarize_old_messages(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test summarization of old messages."""
        for i in range(15):
            manager.add_to_working_memory(user_id, session_id, f"Message {i}", "user")
        summarized, tokens_saved = manager.summarize_old_messages(user_id)
        assert summarized > 0

    def test_get_for_llm_context(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test LLM context formatting."""
        manager.add_to_working_memory(user_id, session_id, "Hello", "user")
        manager.add_to_working_memory(user_id, session_id, "Hi there", "assistant")
        context, breakdown = manager.get_for_llm_context(user_id, 1000)
        assert "User:" in context
        assert "Assistant:" in context

    def test_priority_boost_safety(self, manager: WorkingMemoryManager, user_id, session_id):
        """Test safety keywords boost priority."""
        item = manager.add_to_working_memory(user_id, session_id, "I feel suicidal", "user")
        assert item.priority_score > 1.0


class TestSessionMemoryManager:
    """Tests for SessionMemoryManager (Tier 3)."""

    @pytest.fixture
    def manager(self) -> SessionMemoryManager:
        return SessionMemoryManager(SessionMemorySettings())

    @pytest.fixture
    def user_id(self) -> uuid4:
        return uuid4()

    def test_create_session(self, manager: SessionMemoryManager, user_id):
        """Test session creation."""
        session = manager.create_session(user_id, "therapeutic")
        assert session.user_id == user_id
        assert session.session_number == 1
        assert session.status == SessionStatus.ACTIVE

    def test_get_session(self, manager: SessionMemoryManager, user_id):
        """Test session retrieval."""
        created = manager.create_session(user_id)
        retrieved = manager.get_session(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_active_session(self, manager: SessionMemoryManager, user_id):
        """Test getting active session."""
        manager.create_session(user_id)
        active = manager.get_active_session(user_id)
        assert active is not None
        assert active.status == SessionStatus.ACTIVE

    def test_add_message(self, manager: SessionMemoryManager, user_id):
        """Test adding message to session."""
        session = manager.create_session(user_id)
        message = manager.add_message(session.session_id, "user", "Hello")
        assert message is not None
        assert message.content == "Hello"

    def test_get_messages(self, manager: SessionMemoryManager, user_id):
        """Test retrieving messages."""
        session = manager.create_session(user_id)
        manager.add_message(session.session_id, "user", "Hello")
        manager.add_message(session.session_id, "assistant", "Hi")
        messages = manager.get_messages(session.session_id)
        assert len(messages) == 2

    def test_get_messages_with_filter(self, manager: SessionMemoryManager, user_id):
        """Test message filtering by role."""
        session = manager.create_session(user_id)
        manager.add_message(session.session_id, "user", "Hello")
        manager.add_message(session.session_id, "assistant", "Hi")
        user_msgs = manager.get_messages(session.session_id, role_filter="user")
        assert len(user_msgs) == 1

    def test_end_session(self, manager: SessionMemoryManager, user_id):
        """Test ending session."""
        session = manager.create_session(user_id)
        ended = manager.end_session(session.session_id)
        assert ended is not None
        assert ended.status == SessionStatus.ENDED

    def test_pause_resume_session(self, manager: SessionMemoryManager, user_id):
        """Test pausing and resuming session."""
        session = manager.create_session(user_id)
        assert manager.pause_session(session.session_id) is True
        assert manager.get_session(session.session_id).status == SessionStatus.PAUSED
        assert manager.resume_session(session.session_id) is True
        assert manager.get_session(session.session_id).status == SessionStatus.ACTIVE

    def test_session_statistics(self, manager: SessionMemoryManager, user_id):
        """Test session statistics."""
        session = manager.create_session(user_id)
        manager.add_message(session.session_id, "user", "Hello")
        stats = manager.get_session_statistics(session.session_id)
        assert stats is not None
        assert stats.message_count == 1
        assert stats.user_message_count == 1

    def test_session_history(self, manager: SessionMemoryManager, user_id):
        """Test user session history."""
        manager.create_session(user_id)
        manager.end_session(manager.get_active_session(user_id).session_id)
        manager.create_session(user_id)
        history = manager.get_user_session_history(user_id)
        assert len(history) >= 1

    def test_topic_detection(self, manager: SessionMemoryManager, user_id):
        """Test automatic topic detection."""
        session = manager.create_session(user_id)
        manager.add_message(session.session_id, "user", "I've been feeling anxious about work")
        retrieved = manager.get_session(session.session_id)
        assert "anxiety" in retrieved.topics_detected or "work" in retrieved.topics_detected


class TestEpisodicMemoryManager:
    """Tests for EpisodicMemoryManager (Tier 4)."""

    @pytest.fixture
    def manager(self) -> EpisodicMemoryManager:
        return EpisodicMemoryManager(EpisodicMemorySettings())

    @pytest.fixture
    def user_id(self) -> uuid4:
        return uuid4()

    @pytest.fixture
    def session_id(self) -> uuid4:
        return uuid4()

    def test_store_session_summary(self, manager: EpisodicMemoryManager, user_id, session_id):
        """Test storing session summary."""
        summary = manager.store_session_summary(
            user_id, session_id, 1, "Session about anxiety",
            ["anxiety", "work"], ["anxious", "calm"], ["CBT"], ["reframe thoughts"],
            [], 10, 30,
        )
        assert summary.user_id == user_id
        assert summary.summary_text == "Session about anxiety"

    def test_get_recent_summaries(self, manager: EpisodicMemoryManager, user_id):
        """Test retrieving recent summaries."""
        for i in range(3):
            manager.store_session_summary(user_id, uuid4(), i + 1, f"Summary {i}",
                                          [], [], [], [], [], 5, 15)
        summaries = manager.get_recent_summaries(user_id, limit=2)
        assert len(summaries) == 2

    def test_store_event(self, manager: EpisodicMemoryManager, user_id):
        """Test storing therapeutic event."""
        event = manager.store_event(user_id, EventType.MILESTONE, "First breakthrough",
                                    "Patient had first major insight")
        assert event.user_id == user_id
        assert event.event_type == EventType.MILESTONE

    def test_get_timeline(self, manager: EpisodicMemoryManager, user_id):
        """Test timeline retrieval."""
        manager.store_event(user_id, EventType.SESSION, "Session 1", "Regular session")
        manager.store_event(user_id, EventType.MILESTONE, "Progress", "Made progress")
        from services.memory_service.src.domain.episodic_memory import TimelineQuery
        query = TimelineQuery(user_id=user_id)
        events = manager.get_timeline(query)
        assert len(events) == 2

    def test_get_crisis_history(self, manager: EpisodicMemoryManager, user_id):
        """Test crisis history retrieval."""
        manager.store_event(user_id, EventType.CRISIS, "Crisis event",
                            "Patient expressed suicidal ideation", EventSeverity.CRITICAL)
        manager.store_event(user_id, EventType.SESSION, "Normal session", "Regular")
        crisis = manager.get_crisis_history(user_id)
        assert len(crisis) == 1
        assert crisis[0].event_type == EventType.CRISIS

    def test_therapeutic_context(self, manager: EpisodicMemoryManager, user_id, session_id):
        """Test therapeutic context assembly."""
        manager.store_session_summary(user_id, session_id, 1, "First session",
                                      ["anxiety"], ["worried"], ["CBT"], ["insight"], [], 10, 30)
        context = manager.get_therapeutic_context(user_id)
        assert "recent_sessions" in context
        assert context["total_sessions"] == 1

    def test_link_events(self, manager: EpisodicMemoryManager, user_id):
        """Test linking related events."""
        event1 = manager.store_event(user_id, EventType.CRISIS, "Crisis", "Initial crisis")
        event2 = manager.store_event(user_id, EventType.TREATMENT, "Treatment", "Crisis intervention")
        assert manager.link_events(event1.event_id, event2.event_id) is True


class TestSemanticMemoryManager:
    """Tests for SemanticMemoryManager (Tier 5)."""

    @pytest.fixture
    def manager(self) -> SemanticMemoryManager:
        return SemanticMemoryManager(SemanticMemorySettings())

    @pytest.fixture
    def user_id(self) -> uuid4:
        return uuid4()

    def test_store_fact(self, manager: SemanticMemoryManager, user_id):
        """Test storing a fact."""
        fact = manager.store_fact(user_id, "Patient is a teacher", FactCategory.PERSONAL,
                                  Decimal("0.8"), Decimal("0.6"))
        assert fact is not None
        assert fact.content == "Patient is a teacher"

    def test_store_fact_low_confidence_rejected(self, manager: SemanticMemoryManager, user_id):
        """Test low confidence facts are rejected."""
        fact = manager.store_fact(user_id, "Maybe a fact", FactCategory.GENERAL,
                                  Decimal("0.3"), Decimal("0.5"))
        assert fact is None

    def test_get_facts_by_category(self, manager: SemanticMemoryManager, user_id):
        """Test getting facts by category."""
        manager.store_fact(user_id, "Has sister Sarah", FactCategory.RELATIONSHIP,
                           Decimal("0.8"), Decimal("0.7"))
        manager.store_fact(user_id, "Works as engineer", FactCategory.PERSONAL,
                           Decimal("0.9"), Decimal("0.6"))
        relationships = manager.get_facts_by_category(user_id, FactCategory.RELATIONSHIP)
        assert len(relationships) == 1

    def test_get_safety_facts(self, manager: SemanticMemoryManager, user_id):
        """Test retrieving safety-critical facts."""
        manager.store_fact(user_id, "History of self-harm", FactCategory.SAFETY,
                           Decimal("1.0"), Decimal("1.0"))
        manager.store_fact(user_id, "Likes coffee", FactCategory.PREFERENCE,
                           Decimal("0.7"), Decimal("0.3"))
        safety = manager.get_safety_facts(user_id)
        assert len(safety) == 1

    def test_update_fact(self, manager: SemanticMemoryManager, user_id):
        """Test updating a fact."""
        fact = manager.store_fact(user_id, "Old content", FactCategory.GENERAL,
                                  Decimal("0.7"), Decimal("0.5"))
        updated = manager.update_fact(fact.fact_id, new_content="New content")
        assert updated is not None
        assert "New" in updated.content

    def test_store_triple(self, manager: SemanticMemoryManager, user_id):
        """Test storing knowledge triple."""
        triple = manager.store_triple(user_id, "User", "has_sister", "Sarah", Decimal("0.8"))
        assert triple.subject == "User"
        assert triple.predicate == "has_sister"
        assert triple.object_value == "Sarah"

    def test_query_knowledge_graph(self, manager: SemanticMemoryManager, user_id):
        """Test knowledge graph query."""
        manager.store_triple(user_id, "User", "has_sister", "Sarah", Decimal("0.8"))
        manager.store_triple(user_id, "User", "works_at", "Hospital", Decimal("0.9"))
        query = KnowledgeGraphQuery(user_id=user_id, subject="User")
        results = manager.query_knowledge_graph(query)
        assert len(results) == 2

    def test_get_entity_relationships(self, manager: SemanticMemoryManager, user_id):
        """Test getting entity relationships."""
        manager.store_triple(user_id, "Sarah", "is_sister_of", "User", Decimal("0.8"))
        manager.store_triple(user_id, "Sarah", "lives_in", "NYC", Decimal("0.7"))
        relationships = manager.get_entity_relationships(user_id, "Sarah")
        assert len(relationships) == 2

    def test_user_profile_facts(self, manager: SemanticMemoryManager, user_id):
        """Test user profile generation."""
        manager.store_fact(user_id, "Age 35", FactCategory.PERSONAL, Decimal("0.9"), Decimal("0.5"))
        profile = manager.get_user_profile_facts(user_id)
        assert "personal" in profile
        assert len(profile["personal"]) == 1

    def test_delete_user_data(self, manager: SemanticMemoryManager, user_id):
        """Test GDPR deletion."""
        manager.store_fact(user_id, "Fact 1", FactCategory.GENERAL, Decimal("0.7"), Decimal("0.5"))
        manager.store_triple(user_id, "S", "P", "O", Decimal("0.8"))
        facts, triples, entities = manager.delete_user_data(user_id)
        assert facts == 1
        assert triples == 1


class TestDecayManager:
    """Tests for DecayManager (Ebbinghaus decay)."""

    @pytest.fixture
    def manager(self) -> DecayManager:
        return DecayManager(DecaySettings())

    @pytest.fixture
    def item_id(self) -> uuid4:
        return uuid4()

    def test_calculate_retention(self, manager: DecayManager):
        """Test retention calculation."""
        retention = manager.calculate_retention(0, Decimal("1.0"), Decimal("0.1"))
        assert retention == Decimal("1.0")
        retention_later = manager.calculate_retention(30, Decimal("1.0"), Decimal("0.1"))
        assert retention_later < Decimal("1.0")

    def test_get_decay_rate(self, manager: DecayManager):
        """Test decay rate by category."""
        permanent = manager.get_decay_rate(RetentionCategory.PERMANENT)
        assert permanent == Decimal("0")
        short_term = manager.get_decay_rate(RetentionCategory.SHORT_TERM)
        long_term = manager.get_decay_rate(RetentionCategory.LONG_TERM)
        assert short_term > long_term

    def test_apply_decay(self, manager: DecayManager, item_id):
        """Test applying decay to item."""
        created = datetime.now(timezone.utc) - timedelta(days=30)
        result = manager.apply_decay(item_id, Decimal("1.0"), "medium_term", created)
        assert result.new_strength < Decimal("1.0")
        assert result.decay_applied > Decimal("0")

    def test_permanent_no_decay(self, manager: DecayManager, item_id):
        """Test permanent items don't decay."""
        created = datetime.now(timezone.utc) - timedelta(days=365)
        result = manager.apply_decay(item_id, Decimal("1.0"), "permanent", created)
        assert result.new_strength == Decimal("1.0")
        assert result.action == DecayAction.KEEP

    def test_reinforce(self, manager: DecayManager, item_id):
        """Test reinforcement boosts stability."""
        initial = manager.get_stability(item_id)
        manager.reinforce(item_id)
        after = manager.get_stability(item_id)
        assert after > initial

    def test_mark_permanent(self, manager: DecayManager, item_id):
        """Test marking items as permanent."""
        manager.mark_permanent(item_id)
        assert manager.is_permanent(item_id) is True
        created = datetime.now(timezone.utc) - timedelta(days=365)
        result = manager.apply_decay(item_id, Decimal("1.0"), "short_term", created)
        assert result.new_strength == Decimal("1.0")

    def test_process_batch(self, manager: DecayManager):
        """Test batch processing."""
        items = []
        for i in range(5):
            created = datetime.now(timezone.utc) - timedelta(days=i * 10)
            items.append((uuid4(), Decimal("1.0"), "medium_term", created, None))
        result = manager.process_batch(items)
        assert result.items_processed == 5

    def test_retention_forecast(self, manager: DecayManager, item_id):
        """Test retention forecasting."""
        created = datetime.now(timezone.utc)
        forecast = manager.get_retention_forecast(item_id, Decimal("1.0"), "medium_term", created, 30)
        assert len(forecast) > 0
        assert forecast[0][1] > forecast[-1][1]

    def test_decay_action_thresholds(self, manager: DecayManager, item_id):
        """Test decay action determination."""
        created = datetime.now(timezone.utc) - timedelta(days=365)
        result = manager.apply_decay(item_id, Decimal("1.0"), "short_term", created)
        assert result.action in [DecayAction.KEEP, DecayAction.ARCHIVE, DecayAction.DELETE]

    def test_emotional_content_slower_decay(self, manager: DecayManager, item_id):
        """Test emotional content decays slower."""
        created = datetime.now(timezone.utc) - timedelta(days=30)
        normal = manager.apply_decay(item_id, Decimal("1.0"), "medium_term", created)
        emotional = manager.apply_decay(uuid4(), Decimal("1.0"), "medium_term", created, is_emotional=True)
        assert emotional.new_strength >= normal.new_strength
