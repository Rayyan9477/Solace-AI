"""Alignment tests for event schema compatibility between services.

Verifies that enum definitions, field names, and topic mappings are
consistent across the therapy service, memory service, and the
canonical Kafka event schemas defined in solace_events.
"""

import pytest

from solace_events.schemas import (
    MemoryStoredEvent as KafkaMemoryStoredEvent,
    TherapyModality as KafkaTherapyModality,
    _TOPIC_MAP,
)
from services.therapy_service.src.schemas import (
    TherapyModality as LocalTherapyModality,
)
from services.memory_service.src.events import (
    MemoryStoredEvent as MemoryServiceStoredEvent,
)


class TestEventSchemaAlignment:
    """Tests that event schemas remain consistent across service boundaries."""

    def test_therapy_modality_enum_consistent(self) -> None:
        """Verify therapy service TherapyModality values (lowercase) can be
        uppercased to match the Kafka TherapyModality enum values.

        The therapy service uses lowercase enum values (e.g. 'cbt') while the
        canonical Kafka schemas use uppercase (e.g. 'CBT'). Both enums must
        cover the same set of modalities.
        """
        local_values = {m.value.upper() for m in LocalTherapyModality}
        kafka_values = {m.value for m in KafkaTherapyModality}

        assert local_values == kafka_values, (
            f"Therapy modality mismatch.\n"
            f"  Local (uppercased): {sorted(local_values)}\n"
            f"  Kafka:              {sorted(kafka_values)}"
        )

    def test_memory_event_field_names_match_canonical(self) -> None:
        """Verify memory service MemoryStoredEvent has 'memory_id' and
        'memory_tier' fields matching the canonical schema.

        These field names were previously 'record_id' and 'tier'; this test
        guards against regressions.
        """
        memory_fields = set(MemoryServiceStoredEvent.model_fields.keys())
        canonical_fields = set(KafkaMemoryStoredEvent.model_fields.keys())

        assert "memory_id" in memory_fields, (
            "memory service MemoryStoredEvent is missing 'memory_id' field"
        )
        assert "memory_tier" in memory_fields, (
            "memory service MemoryStoredEvent is missing 'memory_tier' field"
        )
        assert "memory_id" in canonical_fields, (
            "canonical MemoryStoredEvent is missing 'memory_id' field"
        )
        assert "memory_tier" in canonical_fields, (
            "canonical MemoryStoredEvent is missing 'memory_tier' field"
        )

    def test_sfbt_in_both_enums(self) -> None:
        """Verify SFBT (Solution-Focused Brief Therapy) exists in both
        the local therapy service enum and the canonical Kafka enum.
        """
        local_names = {m.name for m in LocalTherapyModality}
        kafka_names = {m.name for m in KafkaTherapyModality}

        assert "SFBT" in local_names, (
            "SFBT missing from therapy service TherapyModality"
        )
        assert "SFBT" in kafka_names, (
            "SFBT missing from Kafka TherapyModality"
        )

    def test_topic_map_covers_all_event_prefixes(self) -> None:
        """Verify _TOPIC_MAP has entries for all core event prefixes.

        Every domain area must have a routing entry so events are published
        to the correct Kafka topic.
        """
        required_prefixes = [
            "session.",
            "safety.",
            "diagnosis.",
            "therapy.",
            "memory.",
            "personality.",
            "user.",
            "notification.",
        ]

        topic_prefixes = set(_TOPIC_MAP.keys())

        for prefix in required_prefixes:
            assert prefix in topic_prefixes, (
                f"_TOPIC_MAP is missing required prefix '{prefix}'. "
                f"Current prefixes: {sorted(topic_prefixes)}"
            )
