"""
Unit tests for Domain Events.
Tests event definitions, EventBus, and EventStore.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import pytest

from services.therapy_service.src.events import (
    EventType, DomainEvent, EventBus, EventStore,
    SessionStartedEvent, SessionPhaseChangedEvent, SessionEndedEvent,
    InterventionCompletedEvent, TreatmentPlanCreatedEvent,
    TreatmentPhaseAdvancedEvent, GoalAchievedEvent,
    OutcomeRecordedEvent, RiskLevelElevatedEvent, SteppedCareChangedEvent,
)


class TestDomainEvent:
    """Tests for base DomainEvent."""

    def test_create_event(self) -> None:
        """Test event creation."""
        event = DomainEvent(
            event_type=EventType.SESSION_STARTED,
            aggregate_id=uuid4(),
            user_id=uuid4(),
            payload={"test": "data"},
        )
        assert event.event_type == EventType.SESSION_STARTED
        assert event.payload["test"] == "data"
        assert event.version == 1

    def test_to_dict(self) -> None:
        """Test event serialization."""
        aggregate_id = uuid4()
        event = DomainEvent(
            event_type=EventType.SESSION_STARTED,
            aggregate_id=aggregate_id,
        )
        data = event.to_dict()
        assert data["event_type"] == "session.started"
        assert data["aggregate_id"] == str(aggregate_id)
        assert "timestamp" in data


class TestSessionStartedEvent:
    """Tests for SessionStartedEvent."""

    def test_create_event(self) -> None:
        """Test creating session started event."""
        session_id = uuid4()
        user_id = uuid4()
        plan_id = uuid4()
        event = SessionStartedEvent.create(
            session_id=session_id,
            user_id=user_id,
            treatment_plan_id=plan_id,
            session_number=5,
        )
        assert event.event_type == EventType.SESSION_STARTED
        assert event.aggregate_id == session_id
        assert event.user_id == user_id
        assert event.payload["session_number"] == 5


class TestSessionPhaseChangedEvent:
    """Tests for SessionPhaseChangedEvent."""

    def test_create_event(self) -> None:
        """Test creating phase changed event."""
        session_id = uuid4()
        user_id = uuid4()
        event = SessionPhaseChangedEvent.create(
            session_id=session_id,
            user_id=user_id,
            from_phase="opening",
            to_phase="working",
            trigger="user_ready",
        )
        assert event.event_type == EventType.SESSION_PHASE_CHANGED
        assert event.payload["from_phase"] == "opening"
        assert event.payload["to_phase"] == "working"


class TestSessionEndedEvent:
    """Tests for SessionEndedEvent."""

    def test_create_event(self) -> None:
        """Test creating session ended event."""
        session_id = uuid4()
        user_id = uuid4()
        event = SessionEndedEvent.create(
            session_id=session_id,
            user_id=user_id,
            duration_minutes=45,
            techniques_count=3,
            skills_practiced=["grounding", "breathing"],
        )
        assert event.event_type == EventType.SESSION_ENDED
        assert event.payload["duration_minutes"] == 45
        assert len(event.payload["skills_practiced"]) == 2


class TestInterventionCompletedEvent:
    """Tests for InterventionCompletedEvent."""

    def test_create_event(self) -> None:
        """Test creating intervention completed event."""
        intervention_id = uuid4()
        session_id = uuid4()
        user_id = uuid4()
        event = InterventionCompletedEvent.create(
            intervention_id=intervention_id,
            session_id=session_id,
            user_id=user_id,
            technique_name="Thought Record",
            engagement_score=0.85,
        )
        assert event.event_type == EventType.INTERVENTION_COMPLETED
        assert event.payload["technique_name"] == "Thought Record"
        assert event.payload["engagement_score"] == 0.85


class TestTreatmentPlanCreatedEvent:
    """Tests for TreatmentPlanCreatedEvent."""

    def test_create_event(self) -> None:
        """Test creating treatment plan created event."""
        plan_id = uuid4()
        user_id = uuid4()
        event = TreatmentPlanCreatedEvent.create(
            plan_id=plan_id,
            user_id=user_id,
            diagnosis="Depression",
            modality="cbt",
            stepped_care_level=2,
        )
        assert event.event_type == EventType.TREATMENT_PLAN_CREATED
        assert event.payload["diagnosis"] == "Depression"
        assert event.payload["stepped_care_level"] == 2


class TestTreatmentPhaseAdvancedEvent:
    """Tests for TreatmentPhaseAdvancedEvent."""

    def test_create_event(self) -> None:
        """Test creating phase advanced event."""
        plan_id = uuid4()
        user_id = uuid4()
        event = TreatmentPhaseAdvancedEvent.create(
            plan_id=plan_id,
            user_id=user_id,
            from_phase="foundation",
            to_phase="active_treatment",
            sessions_completed=4,
        )
        assert event.event_type == EventType.TREATMENT_PLAN_PHASE_ADVANCED
        assert event.payload["from_phase"] == "foundation"
        assert event.payload["sessions_completed"] == 4


class TestGoalAchievedEvent:
    """Tests for GoalAchievedEvent."""

    def test_create_event(self) -> None:
        """Test creating goal achieved event."""
        plan_id = uuid4()
        goal_id = uuid4()
        user_id = uuid4()
        event = GoalAchievedEvent.create(
            plan_id=plan_id,
            goal_id=goal_id,
            user_id=user_id,
            description="Reduce anxiety",
        )
        assert event.event_type == EventType.TREATMENT_PLAN_GOAL_ACHIEVED
        assert event.payload["description"] == "Reduce anxiety"


class TestOutcomeRecordedEvent:
    """Tests for OutcomeRecordedEvent."""

    def test_create_event(self) -> None:
        """Test creating outcome recorded event."""
        measure_id = uuid4()
        user_id = uuid4()
        event = OutcomeRecordedEvent.create(
            measure_id=measure_id,
            user_id=user_id,
            instrument="phq9",
            raw_score=14,
            is_clinical=True,
        )
        assert event.event_type == EventType.OUTCOME_RECORDED
        assert event.payload["raw_score"] == 14
        assert event.payload["is_clinical"] is True


class TestRiskLevelElevatedEvent:
    """Tests for RiskLevelElevatedEvent."""

    def test_create_event(self) -> None:
        """Test creating risk elevated event."""
        session_id = uuid4()
        user_id = uuid4()
        event = RiskLevelElevatedEvent.create(
            session_id=session_id,
            user_id=user_id,
            previous_level="low",
            current_level="high",
            flags=["suicidal_ideation"],
        )
        assert event.event_type == EventType.RISK_LEVEL_ELEVATED
        assert event.payload["current_level"] == "high"
        assert "suicidal_ideation" in event.payload["flags"]


class TestSteppedCareChangedEvent:
    """Tests for SteppedCareChangedEvent."""

    def test_create_event(self) -> None:
        """Test creating stepped care changed event."""
        plan_id = uuid4()
        user_id = uuid4()
        event = SteppedCareChangedEvent.create(
            plan_id=plan_id,
            user_id=user_id,
            from_level=3,
            to_level=2,
            reason="symptom_improvement",
        )
        assert event.event_type == EventType.STEPPED_CARE_CHANGED
        assert event.payload["from_level"] == 3
        assert event.payload["to_level"] == 2


class TestEventBus:
    """Tests for EventBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self) -> None:
        """Test subscribing and publishing events."""
        bus = EventBus()
        received_events: list[DomainEvent] = []

        async def handler(event: DomainEvent) -> None:
            received_events.append(event)

        bus.subscribe(EventType.SESSION_STARTED, handler)
        event = SessionStartedEvent.create(
            session_id=uuid4(),
            user_id=uuid4(),
            treatment_plan_id=uuid4(),
            session_number=1,
        )
        await bus.publish(event)
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.SESSION_STARTED

    @pytest.mark.asyncio
    async def test_subscribe_all(self) -> None:
        """Test subscribing to all events."""
        bus = EventBus()
        received_events: list[DomainEvent] = []

        async def handler(event: DomainEvent) -> None:
            received_events.append(event)

        bus.subscribe_all(handler)
        event1 = SessionStartedEvent.create(uuid4(), uuid4(), uuid4(), 1)
        event2 = SessionEndedEvent.create(uuid4(), uuid4(), 30, 2, [])
        await bus.publish(event1)
        await bus.publish(event2)
        assert len(received_events) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        """Test unsubscribing from events."""
        bus = EventBus()
        received_events: list[DomainEvent] = []

        async def handler(event: DomainEvent) -> None:
            received_events.append(event)

        bus.subscribe(EventType.SESSION_STARTED, handler)
        result = bus.unsubscribe(EventType.SESSION_STARTED, handler)
        assert result is True
        event = SessionStartedEvent.create(uuid4(), uuid4(), uuid4(), 1)
        await bus.publish(event)
        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_handler_error_continues(self) -> None:
        """Test that handler errors don't stop other handlers."""
        bus = EventBus()
        call_count = 0

        async def error_handler(event: DomainEvent) -> None:
            raise ValueError("Test error")

        async def good_handler(event: DomainEvent) -> None:
            nonlocal call_count
            call_count += 1

        bus.subscribe(EventType.SESSION_STARTED, error_handler)
        bus.subscribe(EventType.SESSION_STARTED, good_handler)
        event = SessionStartedEvent.create(uuid4(), uuid4(), uuid4(), 1)
        await bus.publish(event)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_publish_batch(self) -> None:
        """Test publishing multiple events."""
        bus = EventBus()
        received_events: list[DomainEvent] = []

        async def handler(event: DomainEvent) -> None:
            received_events.append(event)

        bus.subscribe_all(handler)
        events = [
            SessionStartedEvent.create(uuid4(), uuid4(), uuid4(), 1),
            SessionEndedEvent.create(uuid4(), uuid4(), 30, 2, []),
        ]
        await bus.publish_batch(events)
        assert len(received_events) == 2

    def test_get_published_events(self) -> None:
        """Test getting published events history."""
        bus = EventBus()
        event = DomainEvent(event_type=EventType.SESSION_STARTED)
        import asyncio
        asyncio.run(bus.publish(event))
        published = bus.get_published_events()
        assert len(published) == 1

    def test_clear_published_events(self) -> None:
        """Test clearing published events."""
        bus = EventBus()
        event = DomainEvent(event_type=EventType.SESSION_STARTED)
        import asyncio
        asyncio.run(bus.publish(event))
        bus.clear_published_events()
        assert len(bus.get_published_events()) == 0


class TestEventStore:
    """Tests for EventStore."""

    @pytest.mark.asyncio
    async def test_append_and_get(self) -> None:
        """Test appending and getting events."""
        store = EventStore()
        aggregate_id = uuid4()
        event = DomainEvent(
            event_type=EventType.SESSION_STARTED,
            aggregate_id=aggregate_id,
        )
        await store.append(event)
        events = await store.get_events(aggregate_id)
        assert len(events) == 1
        assert events[0].aggregate_id == aggregate_id

    @pytest.mark.asyncio
    async def test_get_events_by_type(self) -> None:
        """Test getting events by type."""
        store = EventStore()
        event1 = DomainEvent(event_type=EventType.SESSION_STARTED, aggregate_id=uuid4())
        event2 = DomainEvent(event_type=EventType.SESSION_ENDED, aggregate_id=uuid4())
        await store.append(event1)
        await store.append(event2)
        started_events = await store.get_events_by_type(EventType.SESSION_STARTED)
        assert len(started_events) == 1

    @pytest.mark.asyncio
    async def test_get_events_since(self) -> None:
        """Test getting events since timestamp."""
        store = EventStore()
        old_event = DomainEvent(event_type=EventType.SESSION_STARTED, aggregate_id=uuid4())
        old_event.timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        new_event = DomainEvent(event_type=EventType.SESSION_STARTED, aggregate_id=uuid4())
        await store.append(old_event)
        await store.append(new_event)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_events = await store.get_events_since(cutoff)
        assert len(recent_events) == 1

    @pytest.mark.asyncio
    async def test_get_events_for_user(self) -> None:
        """Test getting events for user."""
        store = EventStore()
        user_id = uuid4()
        event1 = DomainEvent(
            event_type=EventType.SESSION_STARTED,
            aggregate_id=uuid4(),
            user_id=user_id,
        )
        event2 = DomainEvent(
            event_type=EventType.SESSION_STARTED,
            aggregate_id=uuid4(),
            user_id=uuid4(),
        )
        await store.append(event1)
        await store.append(event2)
        user_events = await store.get_events_for_user(user_id)
        assert len(user_events) == 1

    @pytest.mark.asyncio
    async def test_count(self) -> None:
        """Test counting events."""
        store = EventStore()
        await store.append(DomainEvent(event_type=EventType.SESSION_STARTED, aggregate_id=uuid4()))
        await store.append(DomainEvent(event_type=EventType.SESSION_ENDED, aggregate_id=uuid4()))
        count = await store.count()
        assert count == 2
