"""
Integration tests for event schema consistency across services.

Verifies that Phases 6-7 cross-service fixes maintain consistent enum
definitions, field names, topic mappings, and Kafka event structures
between local service schemas and the canonical solace_events schemas.
"""
from __future__ import annotations

import pytest
from uuid import uuid4

from solace_events.schemas import (
    TherapyModality as KafkaModality,
    MemoryStoredEvent as KafkaMemoryStoredEvent,
    EVENT_REGISTRY,
    _TOPIC_MAP,
    get_topic_for_event,
    BaseEvent,
    SessionStartedEvent,
    UserCreatedKafkaEvent,
    NotificationSentKafkaEvent,
)
from services.therapy_service.src.schemas import TherapyModality as LocalModality
from services.memory_service.src.events import (
    MemoryStoredEvent as MemoryServiceStoredEvent,
)


class TestEventSchemaConsistency:
    """Verify event schemas are consistent across services."""

    def test_therapy_modality_uppercase_matches_kafka(self) -> None:
        """Local therapy modalities uppercased should match Kafka enum.

        The therapy service uses lowercase enum values (e.g. 'cbt') while the
        canonical Kafka schemas use uppercase (e.g. 'CBT'). Every local modality
        must have a corresponding uppercase entry in the Kafka enum.
        """
        kafka_values = {m.value for m in KafkaModality}
        for local in LocalModality:
            assert local.value.upper() in kafka_values, (
                f"Local modality '{local.value}' (uppercased: '{local.value.upper()}') "
                f"not found in Kafka TherapyModality values: {sorted(kafka_values)}"
            )

    def test_kafka_modality_lowercase_matches_local(self) -> None:
        """Every Kafka modality should have a corresponding local modality."""
        local_values = {m.value.upper() for m in LocalModality}
        kafka_values = {m.value for m in KafkaModality}
        assert kafka_values == local_values, (
            f"Modality sets differ.\n"
            f"  Kafka:              {sorted(kafka_values)}\n"
            f"  Local (uppercased): {sorted(local_values)}"
        )

    def test_sfbt_in_kafka_modality(self) -> None:
        """SFBT (Solution-Focused Brief Therapy) must exist in Kafka enum."""
        assert hasattr(KafkaModality, "SFBT"), "KafkaModality missing SFBT member"
        assert KafkaModality.SFBT.value == "SFBT"

    def test_sfbt_in_local_modality(self) -> None:
        """SFBT must exist in the local therapy service enum."""
        assert hasattr(LocalModality, "SFBT"), "LocalModality missing SFBT member"
        assert LocalModality.SFBT.value == "sfbt"

    def test_mindfulness_in_both_enums(self) -> None:
        """MINDFULNESS modality must exist in both enums."""
        assert hasattr(KafkaModality, "MINDFULNESS")
        assert hasattr(LocalModality, "MINDFULNESS")
        assert KafkaModality.MINDFULNESS.value == "MINDFULNESS"
        assert LocalModality.MINDFULNESS.value == "mindfulness"

    def test_topic_map_has_user_prefix(self) -> None:
        """User events should route to solace.users topic."""
        user_event = UserCreatedKafkaEvent(
            user_id=uuid4(),
            email="test@example.com",
            display_name="Test",
            role="user",
        )
        topic = get_topic_for_event(user_event)
        assert topic == "solace.users"

    def test_topic_map_has_notification_prefix(self) -> None:
        """Notification events should route to solace.notifications topic."""
        notification_event = NotificationSentKafkaEvent(
            user_id=uuid4(),
            notification_id=uuid4(),
            channel="email",
        )
        topic = get_topic_for_event(notification_event)
        assert topic == "solace.notifications"

    def test_topic_map_has_session_prefix(self) -> None:
        """Session events should route to solace.sessions topic."""
        session_event = SessionStartedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            session_number=1,
        )
        topic = get_topic_for_event(session_event)
        assert topic == "solace.sessions"

    def test_topic_map_covers_all_core_prefixes(self) -> None:
        """All core event prefixes must be present in _TOPIC_MAP."""
        required_prefixes = {
            "session.", "safety.", "diagnosis.", "therapy.",
            "memory.", "personality.", "user.", "notification.",
        }
        topic_prefixes = set(_TOPIC_MAP.keys())
        for prefix in required_prefixes:
            assert prefix in topic_prefixes, (
                f"_TOPIC_MAP missing required prefix '{prefix}'. "
                f"Current prefixes: {sorted(topic_prefixes)}"
            )

    def test_memory_event_canonical_field_names(self) -> None:
        """Memory service MemoryStoredEvent must use canonical field names.

        The fields were renamed from 'record_id'/'tier' to 'memory_id'/'memory_tier'
        to match the canonical Kafka schema.
        """
        local_fields = set(MemoryServiceStoredEvent.model_fields.keys())
        assert "memory_id" in local_fields, (
            "memory service MemoryStoredEvent missing 'memory_id' field"
        )
        assert "memory_tier" in local_fields, (
            "memory service MemoryStoredEvent missing 'memory_tier' field"
        )

    def test_memory_event_canonical_fields_match_kafka(self) -> None:
        """Canonical and local MemoryStoredEvent should share core field names."""
        kafka_fields = set(KafkaMemoryStoredEvent.model_fields.keys())
        local_fields = set(MemoryServiceStoredEvent.model_fields.keys())
        # Both must have the canonical identifiers
        for field_name in ("memory_id", "memory_tier", "content_type"):
            assert field_name in kafka_fields, f"Kafka MemoryStoredEvent missing '{field_name}'"
            assert field_name in local_fields, f"Local MemoryStoredEvent missing '{field_name}'"

    def test_memory_stored_event_instantiation(self) -> None:
        """MemoryStoredEvent should be constructable with canonical field names."""
        event = MemoryServiceStoredEvent(
            user_id=uuid4(),
            memory_id=uuid4(),
            memory_tier="tier_3_session",
            content_type="message",
            importance_score="0.5",
            storage_backend="postgres",
            storage_time_ms=10,
        )
        data = event.model_dump()
        assert "memory_id" in data
        assert "memory_tier" in data
        assert data["memory_tier"] == "tier_3_session"
        assert data["content_type"] == "message"

    def test_event_registry_completeness(self) -> None:
        """EVENT_REGISTRY must include all major event type strings."""
        required_event_types = [
            "session.started", "session.ended",
            "safety.crisis.detected", "safety.escalation.triggered",
            "diagnosis.completed",
            "therapy.session.started", "therapy.intervention.delivered",
            "memory.stored", "memory.consolidated",
            "personality.assessed",
            "notification.sent",
            "user.created", "user.deleted",
        ]
        for event_type in required_event_types:
            assert event_type in EVENT_REGISTRY, (
                f"EVENT_REGISTRY missing '{event_type}'. "
                f"Available: {sorted(EVENT_REGISTRY.keys())}"
            )

    def test_event_registry_values_are_base_event_subclasses(self) -> None:
        """All EVENT_REGISTRY values must be BaseEvent subclasses."""
        for event_type, event_class in EVENT_REGISTRY.items():
            assert issubclass(event_class, BaseEvent), (
                f"EVENT_REGISTRY['{event_type}'] = {event_class} "
                f"is not a BaseEvent subclass"
            )
