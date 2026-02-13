"""
Unit tests for Personality Service Infrastructure Components.
Tests events, config, and repository implementations.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
import pytest

from services.personality_service.src.events import (
    EventType, DomainEvent, ProfileCreatedEvent, ProfileUpdatedEvent,
    AssessmentCompletedEvent, TraitChangedEvent, StyleChangedEvent,
    StabilityAchievedEvent, EventBus, EventStore, EventPublisher,
)
from services.personality_service.src.config import (
    DatabaseSettings, RedisSettings, AISettings, TraitDetectionSettings,
    StyleAdaptationSettings, ProfileSettings, ObservabilitySettings,
    PersonalityServiceConfig, get_config, reload_config,
)
from services.personality_service.src.infrastructure.repository import (
    RepositoryFactory, UnitOfWork,
)
from services.personality_service.tests.fixtures import (
    InMemoryPersonalityRepository, ProfileQueryBuilder, AssessmentQueryBuilder,
)
from services.personality_service.src.domain.entities import (
    PersonalityProfile, TraitAssessment, ProfileSnapshot,
)
from services.personality_service.src.domain.value_objects import OceanScores
from services.personality_service.src.schemas import (
    PersonalityTrait, AssessmentSource, CommunicationStyleType,
)


class TestEventTypes:
    """Tests for event type enumeration."""

    def test_event_types_defined(self) -> None:
        """Test all event types are defined."""
        assert EventType.PROFILE_CREATED.value == "profile.created"
        assert EventType.PROFILE_UPDATED.value == "profile.updated"
        assert EventType.ASSESSMENT_COMPLETED.value == "assessment.completed"
        assert EventType.TRAIT_CHANGED.value == "trait.changed"
        assert EventType.STYLE_CHANGED.value == "style.changed"
        assert EventType.STABILITY_ACHIEVED.value == "stability.achieved"


class TestDomainEvent:
    """Tests for base DomainEvent."""

    def test_create_domain_event(self) -> None:
        """Test domain event creation."""
        event = DomainEvent(
            event_type=EventType.PROFILE_UPDATED,
            aggregate_id=uuid4(),
            user_id=uuid4(),
            payload={"key": "value"},
        )
        assert event.event_type == EventType.PROFILE_UPDATED
        assert event.aggregate_type == "personality_profile"
        assert event.payload == {"key": "value"}

    def test_event_to_dict(self) -> None:
        """Test event serialization."""
        event = DomainEvent()
        data = event.to_dict()
        assert "event_id" in data
        assert "event_type" in data
        assert "timestamp" in data

    def test_event_from_dict(self) -> None:
        """Test event deserialization."""
        original = DomainEvent(
            event_type=EventType.PROFILE_CREATED,
            aggregate_id=uuid4(),
            payload={"test": 123},
        )
        data = original.to_dict()
        restored = DomainEvent.from_dict(data)
        assert restored.event_type == original.event_type
        assert restored.payload == original.payload


class TestSpecificEvents:
    """Tests for specific event types."""

    def test_profile_created_event(self) -> None:
        """Test ProfileCreatedEvent creation."""
        profile_id = uuid4()
        user_id = uuid4()
        event = ProfileCreatedEvent.create(profile_id, user_id)
        assert event.event_type == EventType.PROFILE_CREATED
        assert event.aggregate_id == profile_id
        assert event.user_id == user_id

    def test_assessment_completed_event(self) -> None:
        """Test AssessmentCompletedEvent creation."""
        assessment_id = uuid4()
        user_id = uuid4()
        event = AssessmentCompletedEvent.create(
            assessment_id, user_id,
            AssessmentSource.ENSEMBLE, 0.75, 150.5,
        )
        assert event.event_type == EventType.ASSESSMENT_COMPLETED
        assert event.payload["source"] == "ensemble"
        assert event.payload["confidence"] == 0.75

    def test_trait_changed_event(self) -> None:
        """Test TraitChangedEvent creation."""
        event = TraitChangedEvent.create(
            uuid4(), uuid4(),
            PersonalityTrait.OPENNESS,
            0.5, 0.8, 0.3,
        )
        assert event.event_type == EventType.TRAIT_CHANGED
        assert event.payload["trait"] == "openness"
        assert event.payload["change_magnitude"] == 0.3

    def test_style_changed_event(self) -> None:
        """Test StyleChangedEvent creation."""
        event = StyleChangedEvent.create(
            uuid4(), uuid4(),
            CommunicationStyleType.BALANCED,
            CommunicationStyleType.ANALYTICAL,
        )
        assert event.event_type == EventType.STYLE_CHANGED
        assert event.payload["previous_style"] == "balanced"
        assert event.payload["new_style"] == "analytical"


class TestEventBus:
    """Tests for EventBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self) -> None:
        """Test subscribing and publishing events."""
        bus = EventBus()
        received_events = []

        async def handler(event: DomainEvent) -> None:
            received_events.append(event)

        bus.subscribe(EventType.PROFILE_CREATED, handler)
        event = ProfileCreatedEvent.create(uuid4(), uuid4())
        await bus.publish(event)
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.PROFILE_CREATED

    @pytest.mark.asyncio
    async def test_subscribe_all(self) -> None:
        """Test subscribing to all events."""
        bus = EventBus()
        received_events = []

        async def handler(event: DomainEvent) -> None:
            received_events.append(event)

        bus.subscribe_all(handler)
        await bus.publish(ProfileCreatedEvent.create(uuid4(), uuid4()))
        await bus.publish(ProfileUpdatedEvent.create(uuid4(), uuid4(), 1, 1))
        assert len(received_events) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        """Test unsubscribing handler."""
        bus = EventBus()
        received = []

        async def handler(event: DomainEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.PROFILE_CREATED, handler)
        result = bus.unsubscribe(EventType.PROFILE_CREATED, handler)
        assert result is True
        await bus.publish(ProfileCreatedEvent.create(uuid4(), uuid4()))
        assert len(received) == 0

    def test_get_published_events(self) -> None:
        """Test getting published events history."""
        bus = EventBus()
        assert len(bus.get_published_events()) == 0

    def test_get_events_by_type(self) -> None:
        """Test filtering events by type."""
        bus = EventBus()
        bus._published_events.append(ProfileCreatedEvent.create(uuid4(), uuid4()))
        bus._published_events.append(ProfileUpdatedEvent.create(uuid4(), uuid4(), 1, 1))
        created = bus.get_events_by_type(EventType.PROFILE_CREATED)
        assert len(created) == 1


class TestEventStore:
    """Tests for EventStore."""

    @pytest.mark.asyncio
    async def test_append_and_get(self) -> None:
        """Test appending and retrieving events."""
        store = EventStore()
        aggregate_id = uuid4()
        event = ProfileCreatedEvent.create(aggregate_id, uuid4())
        event.aggregate_id = aggregate_id
        await store.append(event)
        events = await store.get_events(aggregate_id)
        assert len(events) == 1
        assert events[0].aggregate_id == aggregate_id

    @pytest.mark.asyncio
    async def test_get_events_by_type(self) -> None:
        """Test getting events by type."""
        store = EventStore()
        await store.append(ProfileCreatedEvent.create(uuid4(), uuid4()))
        await store.append(ProfileUpdatedEvent.create(uuid4(), uuid4(), 1, 1))
        created = await store.get_events_by_type(EventType.PROFILE_CREATED)
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_get_events_for_user(self) -> None:
        """Test getting events for user."""
        store = EventStore()
        user_id = uuid4()
        await store.append(ProfileCreatedEvent.create(uuid4(), user_id))
        await store.append(ProfileCreatedEvent.create(uuid4(), uuid4()))
        user_events = await store.get_events_for_user(user_id)
        assert len(user_events) == 1


class TestConfigSettings:
    """Tests for configuration settings."""

    def test_database_settings(self) -> None:
        """Test database settings defaults."""
        settings = DatabaseSettings()
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.name == "personality_db"
        assert "postgresql" in settings.connection_string

    def test_redis_settings(self) -> None:
        """Test Redis settings defaults."""
        settings = RedisSettings()
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert "redis://" in settings.url

    def test_ai_settings(self) -> None:
        """Test AI settings defaults."""
        settings = AISettings()
        assert settings.model_provider == "anthropic"
        assert settings.temperature == 0.3
        assert settings.max_tokens == 2048

    def test_trait_detection_settings(self) -> None:
        """Test trait detection settings defaults."""
        settings = TraitDetectionSettings()
        assert settings.min_text_length == 50
        assert settings.enable_llm_detection is True
        assert settings.ensemble_weight_text == 0.4

    def test_style_adaptation_settings(self) -> None:
        """Test style adaptation settings defaults."""
        settings = StyleAdaptationSettings()
        assert settings.high_trait_threshold == 0.7
        assert settings.low_trait_threshold == 0.3
        assert settings.default_warmth == 0.6

    def test_profile_settings(self) -> None:
        """Test profile settings defaults."""
        settings = ProfileSettings()
        assert settings.enable_caching is True
        assert settings.max_history_size == 100
        assert settings.stability_threshold == 0.7

    def test_main_config(self) -> None:
        """Test main service configuration."""
        config = PersonalityServiceConfig()
        assert config.service_name == "personality-service"
        assert config.port == 8004
        assert config.database is not None
        assert config.ai is not None

    def test_config_to_dict(self) -> None:
        """Test configuration serialization."""
        config = PersonalityServiceConfig()
        data = config.to_dict()
        assert "service_name" in data
        assert "ai" in data
        assert "detection" in data

    def test_validate_ensemble_weights(self) -> None:
        """Test ensemble weights validation."""
        config = PersonalityServiceConfig()
        assert config.validate_ensemble_weights() is True


class TestInMemoryRepository:
    """Tests for InMemoryPersonalityRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get_profile(self) -> None:
        """Test saving and retrieving profile."""
        repo = InMemoryPersonalityRepository()
        profile = PersonalityProfile.create_for_user(uuid4())
        await repo.save_profile(profile)
        retrieved = await repo.get_profile(profile.profile_id)
        assert retrieved is not None
        assert retrieved.profile_id == profile.profile_id

    @pytest.mark.asyncio
    async def test_get_profile_by_user(self) -> None:
        """Test getting profile by user ID."""
        repo = InMemoryPersonalityRepository()
        user_id = uuid4()
        profile = PersonalityProfile.create_for_user(user_id)
        await repo.save_profile(profile)
        retrieved = await repo.get_profile_by_user(user_id)
        assert retrieved is not None
        assert retrieved.user_id == user_id

    @pytest.mark.asyncio
    async def test_list_profiles(self) -> None:
        """Test listing profiles with pagination."""
        repo = InMemoryPersonalityRepository()
        for _ in range(5):
            profile = PersonalityProfile.create_for_user(uuid4())
            await repo.save_profile(profile)
        profiles = await repo.list_profiles(limit=3)
        assert len(profiles) == 3

    @pytest.mark.asyncio
    async def test_delete_profile(self) -> None:
        """Test deleting profile."""
        repo = InMemoryPersonalityRepository()
        profile = PersonalityProfile.create_for_user(uuid4())
        await repo.save_profile(profile)
        result = await repo.delete_profile(profile.profile_id)
        assert result is True
        retrieved = await repo.get_profile(profile.profile_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_save_and_get_assessment(self) -> None:
        """Test saving and retrieving assessment."""
        repo = InMemoryPersonalityRepository()
        assessment = TraitAssessment(
            user_id=uuid4(),
            ocean_scores=OceanScores.neutral(),
        )
        await repo.save_assessment(assessment)
        retrieved = await repo.get_assessment(assessment.assessment_id)
        assert retrieved is not None
        assert retrieved.assessment_id == assessment.assessment_id

    @pytest.mark.asyncio
    async def test_list_user_assessments(self) -> None:
        """Test listing user assessments."""
        repo = InMemoryPersonalityRepository()
        user_id = uuid4()
        for _ in range(5):
            assessment = TraitAssessment(
                user_id=user_id,
                ocean_scores=OceanScores.neutral(),
            )
            await repo.save_assessment(assessment)
        assessments = await repo.list_user_assessments(user_id, limit=3)
        assert len(assessments) == 3

    @pytest.mark.asyncio
    async def test_save_and_get_snapshot(self) -> None:
        """Test saving and retrieving snapshot."""
        repo = InMemoryPersonalityRepository()
        profile = PersonalityProfile.create_for_user(uuid4())
        profile.add_assessment(TraitAssessment(ocean_scores=OceanScores.neutral()))
        snapshot = ProfileSnapshot.from_profile(profile)
        await repo.save_snapshot(snapshot)
        retrieved = await repo.get_snapshot(snapshot.snapshot_id)
        assert retrieved is not None
        assert retrieved.snapshot_id == snapshot.snapshot_id

    @pytest.mark.asyncio
    async def test_delete_user_data(self) -> None:
        """Test GDPR user data deletion."""
        repo = InMemoryPersonalityRepository()
        user_id = uuid4()
        profile = PersonalityProfile.create_for_user(user_id)
        await repo.save_profile(profile)
        for _ in range(3):
            assessment = TraitAssessment(user_id=user_id, ocean_scores=OceanScores.neutral())
            await repo.save_assessment(assessment)
        deleted = await repo.delete_user_data(user_id)
        assert deleted >= 4
        assert await repo.get_profile_by_user(user_id) is None

    @pytest.mark.asyncio
    async def test_get_statistics(self) -> None:
        """Test getting repository statistics."""
        repo = InMemoryPersonalityRepository()
        await repo.save_profile(PersonalityProfile.create_for_user(uuid4()))
        stats = await repo.get_statistics()
        assert stats["total_profiles"] == 1
        assert stats["profiles_saved"] == 1


class TestProfileQueryBuilder:
    """Tests for ProfileQueryBuilder."""

    @pytest.mark.asyncio
    async def test_query_by_style_type(self) -> None:
        """Test querying by communication style type."""
        repo = InMemoryPersonalityRepository()
        profile = PersonalityProfile.create_for_user(uuid4())
        scores = OceanScores(
            openness=Decimal("0.8"),
            conscientiousness=Decimal("0.8"),
            extraversion=Decimal("0.3"),
            agreeableness=Decimal("0.5"),
            neuroticism=Decimal("0.4"),
        )
        profile.add_assessment(TraitAssessment(ocean_scores=scores))
        await repo.save_profile(profile)
        query = ProfileQueryBuilder(repo)
        results = await query.with_style_type(CommunicationStyleType.ANALYTICAL).execute()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_with_limit(self) -> None:
        """Test query with limit."""
        repo = InMemoryPersonalityRepository()
        for _ in range(5):
            profile = PersonalityProfile.create_for_user(uuid4())
            await repo.save_profile(profile)
        query = ProfileQueryBuilder(repo)
        results = await query.limit(2).execute()
        assert len(results) == 2


class TestAssessmentQueryBuilder:
    """Tests for AssessmentQueryBuilder."""

    @pytest.mark.asyncio
    async def test_query_by_user(self) -> None:
        """Test querying by user ID."""
        repo = InMemoryPersonalityRepository()
        user_id = uuid4()
        for _ in range(3):
            assessment = TraitAssessment(user_id=user_id, ocean_scores=OceanScores.neutral())
            await repo.save_assessment(assessment)
        for _ in range(2):
            assessment = TraitAssessment(user_id=uuid4(), ocean_scores=OceanScores.neutral())
            await repo.save_assessment(assessment)
        query = AssessmentQueryBuilder(repo)
        results = await query.for_user(user_id).execute()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_by_source(self) -> None:
        """Test querying by assessment source."""
        repo = InMemoryPersonalityRepository()
        await repo.save_assessment(TraitAssessment(
            ocean_scores=OceanScores.neutral(),
            source=AssessmentSource.TEXT_ANALYSIS,
        ))
        await repo.save_assessment(TraitAssessment(
            ocean_scores=OceanScores.neutral(),
            source=AssessmentSource.ENSEMBLE,
        ))
        query = AssessmentQueryBuilder(repo)
        results = await query.from_source(AssessmentSource.TEXT_ANALYSIS).execute()
        assert len(results) == 1


class TestUnitOfWork:
    """Tests for UnitOfWork pattern."""

    @pytest.mark.asyncio
    async def test_commit_changes(self) -> None:
        """Test committing changes."""
        repo = InMemoryPersonalityRepository()
        async with UnitOfWork(repo) as uow:
            profile = PersonalityProfile.create_for_user(uuid4())
            uow.add_profile(profile)
        assert await repo.get_profile(profile.profile_id) is not None

    @pytest.mark.asyncio
    async def test_rollback_on_error(self) -> None:
        """Test rollback on error."""
        repo = InMemoryPersonalityRepository()
        profile = PersonalityProfile.create_for_user(uuid4())
        try:
            async with UnitOfWork(repo) as uow:
                uow.add_profile(profile)
                raise ValueError("Test error")
        except ValueError:
            pass
        assert await repo.get_profile(profile.profile_id) is None


class TestRepositoryFactory:
    """Tests for RepositoryFactory."""

    def test_get_default_raises_in_test_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_default raises RuntimeError when ENVIRONMENT=test."""
        monkeypatch.setenv("ENVIRONMENT", "test")
        RepositoryFactory.reset()
        with pytest.raises(RuntimeError, match="tests/fixtures.py"):
            RepositoryFactory.get_default()

    def test_get_default_raises_without_postgres(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_default raises RuntimeError without PostgreSQL config."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        RepositoryFactory.reset()
        with pytest.raises(RuntimeError, match="PostgreSQL is required"):
            RepositoryFactory.get_default()

    def test_create_unit_of_work_with_explicit_repo(self) -> None:
        """Test creating unit of work with an explicit repository."""
        repo = InMemoryPersonalityRepository()
        uow = RepositoryFactory.create_unit_of_work(repository=repo)
        assert isinstance(uow, UnitOfWork)
        assert uow.repository is repo
