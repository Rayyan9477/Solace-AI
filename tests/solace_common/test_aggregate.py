"""
Unit tests for Solace-AI Aggregate Module.
"""

import pytest
from datetime import datetime, timezone
from typing import Any

from solace_common.domain.aggregate import (
    AggregateEvent,
    AggregateRoot,
    DomainEvent,
    EntityCreatedEvent,
    EntityDeletedEvent,
    EntityUpdatedEvent,
    EventEnvelope,
    InMemoryEventStore,
)
from solace_common.domain.entity import EntityId


class ConcreteEntityId(EntityId):
    """Concrete EntityId for testing."""
    pass


class UserCreatedEvent(AggregateEvent):
    """Test event for user creation."""
    name: str
    email: str


class UserUpdatedEvent(AggregateEvent):
    """Test event for user update."""
    changed_fields: list[str]


class UserAggregate(AggregateRoot[ConcreteEntityId]):
    """Test aggregate for user domain."""

    _entity_type = "User"
    name: str
    email: str
    status: str = "active"

    def validate_invariants(self) -> None:
        """Validate user invariants."""
        if not self.name:
            from solace_common.exceptions import InvariantViolationError
            raise InvariantViolationError("User name cannot be empty")
        if not self.email or "@" not in self.email:
            from solace_common.exceptions import InvariantViolationError
            raise InvariantViolationError("User must have valid email")

    def update_name(self, new_name: str) -> None:
        """Update user name and raise event."""
        old_name = self.name
        object.__setattr__(self, "name", new_name)
        self.touch()

        self._raise_event(
            UserUpdatedEvent.create(
                self,
                event_type="UserUpdated",
                changed_fields=["name"],
            )
        )

    def deactivate(self) -> None:
        """Deactivate user and raise event."""
        object.__setattr__(self, "status", "inactive")
        self.touch()


class TestDomainEvent:
    """Tests for DomainEvent base class."""

    def test_event_creation(self) -> None:
        """Test basic event creation."""
        event = DomainEvent(
            event_type="TestEvent",
            aggregate_id="agg-123",
            aggregate_type="TestAggregate",
        )

        assert event.event_type == "TestEvent"
        assert event.aggregate_id == "agg-123"
        assert event.aggregate_type == "TestAggregate"
        assert event.event_id is not None
        assert event.occurred_at is not None
        assert event.version == 1

    def test_with_correlation(self) -> None:
        """Test adding correlation context."""
        event = DomainEvent(
            event_type="TestEvent",
            aggregate_id="agg-123",
            aggregate_type="TestAggregate",
        )

        correlated = event.with_correlation("corr-456", "cause-789")

        assert correlated.correlation_id == "corr-456"
        assert correlated.causation_id == "cause-789"
        assert correlated.event_type == event.event_type

    def test_immutability(self) -> None:
        """Test event is immutable."""
        event = DomainEvent(
            event_type="TestEvent",
            aggregate_id="agg-123",
            aggregate_type="TestAggregate",
        )

        with pytest.raises(Exception):
            event.event_type = "Modified"  # type: ignore[misc]


class TestAggregateRoot:
    """Tests for AggregateRoot."""

    def test_aggregate_creation(self) -> None:
        """Test basic aggregate creation."""
        entity_id = ConcreteEntityId.generate()
        user = UserAggregate(
            id=entity_id,
            name="John Doe",
            email="john@example.com",
        )

        assert user.id == entity_id
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.status == "active"

    def test_raise_event(self) -> None:
        """Test raising domain events."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        user.update_name("Jane Doe")

        assert user.has_pending_events() is True
        events = user.collect_events()
        assert len(events) == 1
        assert events[0].event_type == "UserUpdated"

    def test_collect_clears_events(self) -> None:
        """Test collect_events clears the event list."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        user.update_name("Jane Doe")
        user.collect_events()

        assert user.has_pending_events() is False
        assert len(user.collect_events()) == 0

    def test_multiple_events(self) -> None:
        """Test multiple events are collected."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        user.update_name("Jane Doe")
        user.update_name("Janet Doe")

        events = user.collect_events()
        assert len(events) == 2

    def test_validate_invariants_success(self) -> None:
        """Test invariant validation passes for valid state."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        user.validate_invariants()  # Should not raise

    def test_validate_invariants_failure(self) -> None:
        """Test invariant validation fails for invalid state."""
        from solace_common.exceptions import InvariantViolationError

        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="",  # Invalid: empty name
            email="john@example.com",
        )

        with pytest.raises(InvariantViolationError):
            user.validate_invariants()

    def test_entity_type(self) -> None:
        """Test entity type accessor."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        assert user.get_entity_type() == "User"


class TestAggregateEvent:
    """Tests for AggregateEvent factory method."""

    def test_create_from_aggregate(self) -> None:
        """Test creating event with aggregate context."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        event = UserCreatedEvent.create(
            user,
            event_type="UserCreated",
            name="John Doe",
            email="john@example.com",
        )

        assert event.aggregate_id == str(user.id)
        assert event.aggregate_type == "User"
        assert event.name == "John Doe"
        assert event.email == "john@example.com"


class TestBuiltInEvents:
    """Tests for built-in event types."""

    def test_entity_created_event(self) -> None:
        """Test EntityCreatedEvent."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        event = EntityCreatedEvent.create(
            user,
            event_type="UserCreated",
            entity_data={"name": "John Doe", "email": "john@example.com"},
        )

        assert event.entity_data["name"] == "John Doe"

    def test_entity_updated_event(self) -> None:
        """Test EntityUpdatedEvent."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        event = EntityUpdatedEvent.create(
            user,
            event_type="UserUpdated",
            changed_fields=["name"],
            previous_values={"name": "John Doe"},
            new_values={"name": "Jane Doe"},
        )

        assert "name" in event.changed_fields
        assert event.previous_values["name"] == "John Doe"
        assert event.new_values["name"] == "Jane Doe"

    def test_entity_deleted_event(self) -> None:
        """Test EntityDeletedEvent."""
        user = UserAggregate(
            id=ConcreteEntityId.generate(),
            name="John Doe",
            email="john@example.com",
        )

        event = EntityDeletedEvent.create(
            user,
            event_type="UserDeleted",
            reason="User requested deletion",
        )

        assert event.reason == "User requested deletion"


class TestEventEnvelope:
    """Tests for EventEnvelope."""

    def test_wrap_event(self) -> None:
        """Test wrapping event in envelope."""
        event = DomainEvent(
            event_type="TestEvent",
            aggregate_id="agg-123",
            aggregate_type="TestAggregate",
        )

        envelope = EventEnvelope.wrap(event, destination="events.test")

        assert envelope.event == event
        assert envelope.destination == "events.test"
        assert envelope.partition_key == "agg-123"
        assert envelope.headers["event_type"] == "TestEvent"

    def test_envelope_defaults(self) -> None:
        """Test envelope default values."""
        event = DomainEvent(
            event_type="TestEvent",
            aggregate_id="agg-123",
            aggregate_type="TestAggregate",
        )

        envelope = EventEnvelope.wrap(event)

        assert envelope.retry_count == 0
        assert envelope.max_retries == 3


class TestInMemoryEventStore:
    """Tests for InMemoryEventStore."""

    @pytest.fixture
    def event_store(self) -> InMemoryEventStore:
        """Create fresh event store for each test."""
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_append_and_get_events(self, event_store: InMemoryEventStore) -> None:
        """Test appending and retrieving events."""
        events = [
            DomainEvent(
                event_type="Event1",
                aggregate_id="agg-1",
                aggregate_type="Test",
            ),
            DomainEvent(
                event_type="Event2",
                aggregate_id="agg-1",
                aggregate_type="Test",
            ),
        ]

        await event_store.append("agg-1", events)
        retrieved = await event_store.get_events("agg-1")

        assert len(retrieved) == 2
        assert retrieved[0].event_type == "Event1"
        assert retrieved[1].event_type == "Event2"

    @pytest.mark.asyncio
    async def test_get_events_from_version(self, event_store: InMemoryEventStore) -> None:
        """Test retrieving events from specific version."""
        events = [
            DomainEvent(
                event_type=f"Event{i}",
                aggregate_id="agg-1",
                aggregate_type="Test",
            )
            for i in range(5)
        ]

        await event_store.append("agg-1", events)
        retrieved = await event_store.get_events("agg-1", from_version=3)

        assert len(retrieved) == 2
        assert retrieved[0].event_type == "Event3"
        assert retrieved[1].event_type == "Event4"

    @pytest.mark.asyncio
    async def test_get_all_events(self, event_store: InMemoryEventStore) -> None:
        """Test retrieving all events across aggregates."""
        await event_store.append("agg-1", [
            DomainEvent(event_type="E1", aggregate_id="agg-1", aggregate_type="T"),
        ])
        await event_store.append("agg-2", [
            DomainEvent(event_type="E2", aggregate_id="agg-2", aggregate_type="T"),
        ])

        all_events = await event_store.get_all_events()

        assert len(all_events) == 2

    @pytest.mark.asyncio
    async def test_clear(self, event_store: InMemoryEventStore) -> None:
        """Test clearing event store."""
        await event_store.append("agg-1", [
            DomainEvent(event_type="E1", aggregate_id="agg-1", aggregate_type="T"),
        ])

        event_store.clear()
        events = await event_store.get_events("agg-1")

        assert len(events) == 0
