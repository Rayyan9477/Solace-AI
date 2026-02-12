"""Unit tests for Solace-AI Testing Library - Factories module."""

from __future__ import annotations

import pytest

from solace_testing.factories import (
    BaseFactory,
    DiagnosisFactory,
    EntityFactory,
    EntityIdFactory,
    EventFactory,
    FactoryConfig,
    FactoryRegistry,
    FactorySequence,
    LLMResponseFactory,
    MessageFactory,
    SafetyAssessmentFactory,
    SessionFactory,
    ToolCallFactory,
    UserFactory,
    VectorFactory,
    get_factory_registry,
)


class TestFactorySequence:
    """Tests for FactorySequence."""

    def test_next_increments(self) -> None:
        seq = FactorySequence(start=1, prefix="id-")
        assert seq.next() == "id-1"
        assert seq.next() == "id-2"
        assert seq.next() == "id-3"

    def test_reset(self) -> None:
        seq = FactorySequence(start=1)
        seq.next()
        seq.next()
        seq.reset()
        assert seq.next() == "1"

    def test_custom_start(self) -> None:
        seq = FactorySequence(start=100)
        assert seq.next() == "100"


class TestFactoryConfig:
    """Tests for FactoryConfig."""

    def test_default_values(self) -> None:
        config = FactoryConfig()
        assert config.seed is None
        assert config.locale == "en_US"
        assert config.default_batch_size == 10

    def test_seed_setting(self) -> None:
        config = FactoryConfig(seed=42)
        assert config.seed == 42


class TestEntityIdFactory:
    """Tests for EntityIdFactory."""

    def test_create_id(self) -> None:
        factory = EntityIdFactory(prefix="usr-")
        entity_id = factory.create()
        assert entity_id.startswith("usr-")
        assert len(entity_id) > 4

    def test_create_with_override(self) -> None:
        factory = EntityIdFactory()
        entity_id = factory.create(value="custom-id")
        assert entity_id == "custom-id"


class TestEntityFactory:
    """Tests for EntityFactory."""

    def test_create_entity(self) -> None:
        factory = EntityFactory(entity_type="User")
        entity = factory.create()
        assert "id" in entity
        assert "name" in entity
        assert "metadata" in entity
        assert entity["name"].startswith("User-")

    def test_create_with_overrides(self) -> None:
        factory = EntityFactory()
        entity = factory.create(name="Custom Name", extra_field="value")
        assert entity["name"] == "Custom Name"
        assert entity["extra_field"] == "value"

    def test_create_batch(self) -> None:
        factory = EntityFactory()
        entities = factory.create_batch(5)
        assert len(entities) == 5
        ids = [e["id"] for e in entities]
        assert len(set(ids)) == 5


class TestUserFactory:
    """Tests for UserFactory."""

    def test_create_user(self) -> None:
        factory = UserFactory()
        user = factory.create()
        assert "id" in user
        assert "name" in user
        assert "email" in user
        assert user["status"] == "active"

    def test_create_with_custom_email(self) -> None:
        factory = UserFactory()
        user = factory.create(email="test@example.com")
        assert user["email"] == "test@example.com"


class TestSessionFactory:
    """Tests for SessionFactory."""

    def test_create_session(self) -> None:
        factory = SessionFactory()
        session = factory.create()
        assert "id" in session
        assert "user_id" in session
        assert session["status"] == "active"
        assert "correlation_id" in session

    def test_create_with_user_id(self) -> None:
        factory = SessionFactory()
        session = factory.create(user_id="user-123")
        assert session["user_id"] == "user-123"


class TestMessageFactory:
    """Tests for MessageFactory."""

    def test_create_user_message(self) -> None:
        factory = MessageFactory()
        msg = factory.create(role="user")
        assert msg["role"] == "user"
        assert "content" in msg
        assert "session_id" in msg

    def test_create_assistant_message(self) -> None:
        factory = MessageFactory()
        msg = factory.create(role="assistant")
        assert msg["role"] == "assistant"

    def test_create_conversation(self) -> None:
        factory = MessageFactory()
        convo = factory.create_conversation(turns=3)
        assert len(convo) == 6
        assert convo[0]["role"] == "user"
        assert convo[1]["role"] == "assistant"
        session_ids = set(m["session_id"] for m in convo)
        assert len(session_ids) == 1


class TestEventFactory:
    """Tests for EventFactory."""

    def test_create_event(self) -> None:
        factory = EventFactory(event_type="UserCreated")
        event = factory.create()
        assert event["event_type"] == "UserCreated"
        assert "event_id" in event
        assert "timestamp" in event
        assert "correlation_id" in event

    def test_create_with_payload(self) -> None:
        factory = EventFactory()
        event = factory.create(payload={"user_id": "123"})
        assert event["payload"]["user_id"] == "123"


class TestDiagnosisFactory:
    """Tests for DiagnosisFactory."""

    def test_create_diagnosis(self) -> None:
        factory = DiagnosisFactory()
        diag = factory.create()
        assert "id" in diag
        assert "condition_code" in diag
        assert "condition_name" in diag
        assert "icd_code" in diag
        assert 0 <= diag["confidence"] <= 1
        assert diag["severity"] in ["mild", "moderate", "severe"]

    def test_create_with_custom_condition(self) -> None:
        factory = DiagnosisFactory()
        diag = factory.create(condition=("TEST", "Test Condition", "F00.0"))
        assert diag["condition_code"] == "TEST"
        assert diag["condition_name"] == "Test Condition"


class TestSafetyAssessmentFactory:
    """Tests for SafetyAssessmentFactory."""

    def test_create_assessment(self) -> None:
        factory = SafetyAssessmentFactory()
        assessment = factory.create()
        assert "id" in assessment
        assert "risk_level" in assessment
        assert assessment["risk_level"] in ["LOW", "ELEVATED", "HIGH"]

    def test_high_risk_has_crisis(self) -> None:
        factory = SafetyAssessmentFactory()
        assessment = factory.create(risk_level="HIGH")
        assert assessment["crisis_detected"] is True


class TestVectorFactory:
    """Tests for VectorFactory."""

    def test_create_vector(self) -> None:
        factory = VectorFactory(dimensions=128)
        vector = factory.create()
        assert len(vector) == 128
        norm = sum(x * x for x in vector) ** 0.5
        assert abs(norm - 1.0) < 0.0001

    def test_custom_dimensions(self) -> None:
        factory = VectorFactory(dimensions=256)
        vector = factory.create(dimensions=512)
        assert len(vector) == 512


class TestToolCallFactory:
    """Tests for ToolCallFactory."""

    def test_create_tool_call(self) -> None:
        factory = ToolCallFactory()
        call = factory.create(name="get_weather", arguments={"city": "NYC"})
        assert call["name"] == "get_weather"
        assert call["arguments"]["city"] == "NYC"
        assert "id" in call


class TestLLMResponseFactory:
    """Tests for LLMResponseFactory."""

    def test_create_response(self) -> None:
        factory = LLMResponseFactory()
        response = factory.create()
        assert "content" in response
        assert response["finish_reason"] == "stop"
        assert "usage" in response

    def test_create_with_tool_calls(self) -> None:
        factory = LLMResponseFactory()
        response = factory.create(
            tool_calls=[{"id": "tc1", "name": "test", "arguments": {}}]
        )
        assert len(response["tool_calls"]) == 1


class TestFactoryRegistry:
    """Tests for FactoryRegistry."""

    def test_default_factories_registered(self) -> None:
        registry = FactoryRegistry()
        assert registry.get("entity") is not None
        assert registry.get("user") is not None
        assert registry.get("message") is not None

    def test_create_via_registry(self) -> None:
        registry = FactoryRegistry()
        user = registry.create("user", name="Test User")
        assert user["name"] == "Test User"

    def test_create_batch_via_registry(self) -> None:
        registry = FactoryRegistry()
        users = registry.create_batch("user", 3)
        assert len(users) == 3

    def test_register_custom_factory(self) -> None:
        registry = FactoryRegistry()
        custom = EntityFactory(entity_type="Custom")
        registry.register("custom", custom)
        result = registry.create("custom")
        assert "Custom-" in result["name"]

    def test_get_nonexistent_factory(self) -> None:
        registry = FactoryRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_reset_all_sequences(self) -> None:
        registry = FactoryRegistry()
        registry.create("entity")
        registry.create("entity")
        registry.reset_all()


class TestGetFactoryRegistry:
    """Tests for get_factory_registry function."""

    def test_returns_singleton(self) -> None:
        reg1 = get_factory_registry()
        reg2 = get_factory_registry()
        assert reg1 is reg2

    def test_has_default_factories(self) -> None:
        registry = get_factory_registry()
        assert registry.get("user") is not None
        assert registry.get("session") is not None


class TestBaseFactoryMethods:
    """Tests for BaseFactory static methods."""

    def test_random_string(self) -> None:
        s = BaseFactory.random_string(10)
        assert len(s) == 10
        assert s.isalpha()

    def test_random_email(self) -> None:
        email = BaseFactory.random_email("example.com")
        assert "@example.com" in email

    def test_random_uuid(self) -> None:
        uid = BaseFactory.random_uuid()
        assert len(uid) == 36
        assert "-" in uid

    def test_random_datetime(self) -> None:
        dt = BaseFactory.random_datetime()
        assert dt is not None


class TestDeterministicBehavior:
    """Tests for deterministic behavior with seed."""

    def test_seeded_factory_produces_same_results(self) -> None:
        import random
        random.seed(42)
        factory1 = UserFactory()
        user1 = factory1.create()
        random.seed(42)
        factory2 = UserFactory()
        user2 = factory2.create()
        assert user1["name"] == user2["name"]
