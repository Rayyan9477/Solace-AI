"""Unit tests for Solace-AI Event Schemas."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from solace_events.schemas import (
    AssessmentCompletedEvent,
    BaseEvent,
    ClinicalHypothesis,
    Confidence,
    CrisisDetectedEvent,
    CrisisLevel,
    DiagnosisCompletedEvent,
    EVENT_REGISTRY,
    EventMetadata,
    InterventionDeliveredEvent,
    MemoryConsolidatedEvent,
    MemoryStoredEvent,
    MemoryTier,
    MessageReceivedEvent,
    OceanScores,
    PersonalityAssessedEvent,
    ResponseGeneratedEvent,
    RetentionCategory,
    RiskFactor,
    SafetyAssessmentEvent,
    SessionEndedEvent,
    SessionStartedEvent,
    StyleGeneratedEvent,
    TherapyModality,
    TherapySessionStartedEvent,
    deserialize_event,
    get_topic_for_event,
)


class TestEventMetadata:
    """Tests for EventMetadata."""

    def test_default_values(self) -> None:
        """Test default metadata values."""
        metadata = EventMetadata()

        assert metadata.event_id is not None
        assert metadata.timestamp is not None
        assert metadata.correlation_id is not None
        assert metadata.version == 1
        assert metadata.source_service == "solace-ai"

    def test_custom_values(self) -> None:
        """Test custom metadata values."""
        event_id = uuid4()
        corr_id = uuid4()
        metadata = EventMetadata(
            event_id=event_id,
            correlation_id=corr_id,
            version=2,
        )

        assert metadata.event_id == event_id
        assert metadata.correlation_id == corr_id
        assert metadata.version == 2

    def test_immutability(self) -> None:
        """Test metadata is frozen."""
        metadata = EventMetadata()
        with pytest.raises(Exception):
            metadata.version = 5  # type: ignore[misc]


class TestBaseEvent:
    """Tests for BaseEvent."""

    def test_event_creation(self) -> None:
        """Test basic event creation."""
        user_id = uuid4()
        event = SessionStartedEvent(
            user_id=user_id,
            session_number=1,
        )

        assert event.event_type == "session.started"
        assert event.user_id == user_id
        assert event.metadata.event_id is not None

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        user_id = uuid4()
        event = SessionStartedEvent(
            user_id=user_id,
            session_number=1,
        )

        data = event.to_dict()

        assert data["event_type"] == "session.started"
        assert data["user_id"] == str(user_id)
        assert "metadata" in data

    def test_with_correlation(self) -> None:
        """Test adding correlation context."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        corr_id = uuid4()
        cause_id = uuid4()

        correlated = event.with_correlation(corr_id, cause_id)

        assert correlated.metadata.correlation_id == corr_id
        assert correlated.metadata.causation_id == cause_id


class TestSessionEvents:
    """Tests for session events."""

    def test_session_started_event(self) -> None:
        """Test SessionStartedEvent."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=5,
            channel="mobile",
        )

        assert event.event_type == "session.started"
        assert event.session_number == 5
        assert event.channel == "mobile"

    def test_session_ended_event(self) -> None:
        """Test SessionEndedEvent."""
        event = SessionEndedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            duration_seconds=3600,
            message_count=25,
            end_reason="user_initiated",
        )

        assert event.event_type == "session.ended"
        assert event.duration_seconds == 3600
        assert event.message_count == 25

    def test_message_received_event(self) -> None:
        """Test MessageReceivedEvent."""
        event = MessageReceivedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            message_id=uuid4(),
            content_length=150,
            content_hash="abc123",
        )

        assert event.event_type == "session.message.received"
        assert event.content_length == 150

    def test_response_generated_event(self) -> None:
        """Test ResponseGeneratedEvent."""
        event = ResponseGeneratedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            response_id=uuid4(),
            response_length=500,
            generation_time_ms=250,
            model_used="claude-3",
            tokens_used=150,
        )

        assert event.event_type == "session.response.generated"
        assert event.generation_time_ms == 250


class TestSafetyEvents:
    """Tests for safety events."""

    def test_safety_assessment_event(self) -> None:
        """Test SafetyAssessmentEvent."""
        risk_factor = RiskFactor(
            factor_type="suicidal_ideation",
            severity=Decimal("0.7"),
            evidence="Expression of hopelessness",
            confidence=Decimal("0.85"),
        )

        event = SafetyAssessmentEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            risk_level=CrisisLevel.ELEVATED,
            risk_score=Decimal("0.6"),
            risk_factors=[risk_factor],
            detection_layer=2,
            recommended_action="Enhanced monitoring",
        )

        assert event.event_type == "safety.assessment.completed"
        assert event.risk_level == CrisisLevel.ELEVATED
        assert len(event.risk_factors) == 1

    def test_crisis_detected_event(self) -> None:
        """Test CrisisDetectedEvent."""
        event = CrisisDetectedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            crisis_level=CrisisLevel.CRITICAL,
            trigger_indicators=["explicit_threat", "immediate_danger"],
            detection_layer=1,
            confidence=Decimal("0.95"),
            escalation_action="immediate_intervention",
            requires_human_review=True,
        )

        assert event.event_type == "safety.crisis.detected"
        assert event.crisis_level == CrisisLevel.CRITICAL
        assert event.requires_human_review is True


class TestDiagnosisEvents:
    """Tests for diagnosis events."""

    def test_diagnosis_completed_event(self) -> None:
        """Test DiagnosisCompletedEvent."""
        hypothesis = ClinicalHypothesis(
            condition_code="F32.1",
            condition_name="Major Depressive Disorder",
            confidence=Confidence.HIGH,
            evidence_summary="Multiple symptoms observed",
            severity="MODERATE",
        )

        event = DiagnosisCompletedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            assessment_id=uuid4(),
            primary_hypothesis=hypothesis,
            stepped_care_level=2,
        )

        assert event.event_type == "diagnosis.completed"
        assert event.primary_hypothesis is not None
        assert event.primary_hypothesis.condition_code == "F32.1"

    def test_assessment_completed_event(self) -> None:
        """Test AssessmentCompletedEvent."""
        event = AssessmentCompletedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            instrument_code="PHQ-9",
            total_score=15,
            severity_category="Moderately Severe",
            subscale_scores={"somatic": 5, "cognitive": 10},
        )

        assert event.event_type == "diagnosis.assessment.completed"
        assert event.instrument_code == "PHQ-9"
        assert event.total_score == 15


class TestTherapyEvents:
    """Tests for therapy events."""

    def test_therapy_session_started(self) -> None:
        """Test TherapySessionStartedEvent."""
        event = TherapySessionStartedEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            treatment_plan_id=uuid4(),
            session_number=3,
            planned_focus=["anxiety_management", "cognitive_restructuring"],
            active_modalities=[TherapyModality.CBT, TherapyModality.MINDFULNESS],
        )

        assert event.event_type == "therapy.session.started"
        assert event.session_number == 3
        assert TherapyModality.CBT in event.active_modalities

    def test_intervention_delivered_event(self) -> None:
        """Test InterventionDeliveredEvent."""
        event = InterventionDeliveredEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            intervention_id=uuid4(),
            technique="thought_challenging",
            modality=TherapyModality.CBT,
            selection_rationale={"relevance": Decimal("0.9")},
            user_engagement_score=Decimal("0.75"),
        )

        assert event.event_type == "therapy.intervention.delivered"
        assert event.modality == TherapyModality.CBT


class TestMemoryEvents:
    """Tests for memory events."""

    def test_memory_stored_event(self) -> None:
        """Test MemoryStoredEvent."""
        event = MemoryStoredEvent(
            user_id=uuid4(),
            session_id=uuid4(),
            memory_id=uuid4(),
            memory_tier=MemoryTier.EPISODIC,
            content_type="conversation_summary",
            retention_category=RetentionCategory.LONG_TERM,
            embedding_generated=True,
        )

        assert event.event_type == "memory.stored"
        assert event.memory_tier == MemoryTier.EPISODIC
        assert event.embedding_generated is True

    def test_memory_consolidated_event(self) -> None:
        """Test MemoryConsolidatedEvent."""
        event = MemoryConsolidatedEvent(
            user_id=uuid4(),
            consolidation_id=uuid4(),
            session_ids=[uuid4(), uuid4()],
            facts_extracted=15,
            embeddings_created=10,
            summary_generated=True,
        )

        assert event.event_type == "memory.consolidated"
        assert event.facts_extracted == 15


class TestPersonalityEvents:
    """Tests for personality events."""

    def test_personality_assessed_event(self) -> None:
        """Test PersonalityAssessedEvent."""
        scores = OceanScores(
            openness=Decimal("0.7"),
            conscientiousness=Decimal("0.6"),
            extraversion=Decimal("0.4"),
            agreeableness=Decimal("0.8"),
            neuroticism=Decimal("0.5"),
        )

        event = PersonalityAssessedEvent(
            user_id=uuid4(),
            assessment_id=uuid4(),
            ocean_scores=scores,
            assessment_source="ENSEMBLE",
            confidence=Decimal("0.85"),
            sample_size=50,
        )

        assert event.event_type == "personality.assessed"
        assert event.ocean_scores.openness == Decimal("0.7")

    def test_style_generated_event(self) -> None:
        """Test StyleGeneratedEvent."""
        event = StyleGeneratedEvent(
            user_id=uuid4(),
            style_id=uuid4(),
            target_module="therapy",
            formality_level=Decimal("0.3"),
            warmth_level=Decimal("0.8"),
            directness_level=Decimal("0.5"),
            vocabulary_complexity=Decimal("0.6"),
        )

        assert event.event_type == "personality.style.generated"
        assert event.warmth_level == Decimal("0.8")


class TestEventDeserialization:
    """Tests for event deserialization."""

    def test_deserialize_session_event(self) -> None:
        """Test deserializing session event."""
        data = {
            "event_type": "session.started",
            "user_id": str(uuid4()),
            "session_number": 1,
            "channel": "web",
            "client_info": {},
            "metadata": {
                "event_id": str(uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correlation_id": str(uuid4()),
                "version": 1,
                "source_service": "solace-ai",
            },
        }

        event = deserialize_event(data)

        assert isinstance(event, SessionStartedEvent)
        assert event.session_number == 1

    def test_deserialize_unknown_type(self) -> None:
        """Test deserializing unknown event type."""
        data = {
            "event_type": "unknown.event",
            "user_id": str(uuid4()),
            "metadata": {
                "event_id": str(uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correlation_id": str(uuid4()),
                "version": 1,
                "source_service": "solace-ai",
            },
        }

        event = deserialize_event(data)

        assert isinstance(event, BaseEvent)

    def test_deserialize_missing_type(self) -> None:
        """Test error on missing event type."""
        with pytest.raises(ValueError):
            deserialize_event({"user_id": str(uuid4())})


class TestEventRegistry:
    """Tests for event registry."""

    def test_all_events_registered(self) -> None:
        """Test all event types are in registry."""
        expected_types = [
            "session.started",
            "session.ended",
            "session.message.received",
            "session.response.generated",
            "safety.assessment.completed",
            "safety.crisis.detected",
            "diagnosis.completed",
            "therapy.session.started",
            "therapy.intervention.delivered",
            "memory.stored",
            "memory.consolidated",
            "personality.assessed",
            "personality.style.generated",
        ]

        for event_type in expected_types:
            assert event_type in EVENT_REGISTRY


class TestTopicRouting:
    """Tests for event topic routing."""

    def test_session_events_route(self) -> None:
        """Test session events route to sessions topic."""
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        assert get_topic_for_event(event) == "solace.sessions"

    def test_safety_events_route(self) -> None:
        """Test safety events route to safety topic."""
        event = CrisisDetectedEvent(
            user_id=uuid4(),
            crisis_level=CrisisLevel.HIGH,
            trigger_indicators=["test"],
            detection_layer=1,
            confidence=Decimal("0.9"),
            escalation_action="test",
        )
        assert get_topic_for_event(event) == "solace.safety"

    def test_diagnosis_events_route(self) -> None:
        """Test diagnosis events route to assessments topic."""
        event = DiagnosisCompletedEvent(
            user_id=uuid4(),
            assessment_id=uuid4(),
            stepped_care_level=1,
        )
        assert get_topic_for_event(event) == "solace.assessments"

    def test_therapy_events_route(self) -> None:
        """Test therapy events route to therapy topic."""
        event = TherapySessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        assert get_topic_for_event(event) == "solace.therapy"

    def test_memory_events_route(self) -> None:
        """Test memory events route to memory topic."""
        event = MemoryStoredEvent(
            user_id=uuid4(),
            memory_id=uuid4(),
            memory_tier=MemoryTier.SESSION,
            content_type="test",
            retention_category=RetentionCategory.SHORT_TERM,
        )
        assert get_topic_for_event(event) == "solace.memory"

    def test_personality_events_route(self) -> None:
        """Test personality events route to personality topic."""
        event = StyleGeneratedEvent(
            user_id=uuid4(),
            style_id=uuid4(),
            target_module="test",
            formality_level=Decimal("0.5"),
            warmth_level=Decimal("0.5"),
            directness_level=Decimal("0.5"),
            vocabulary_complexity=Decimal("0.5"),
        )
        assert get_topic_for_event(event) == "solace.personality"
