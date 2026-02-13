"""
Solace-AI Diagnosis Service - Batch 5.3 Unit Tests.
Tests for entities, value objects, repository, events, and config.
"""
from __future__ import annotations
from decimal import Decimal
from uuid import uuid4
import pytest
import pytest_asyncio

from services.diagnosis_service.src.schemas import (
    DiagnosisPhase, SeverityLevel, SymptomType, ConfidenceLevel,
)
from services.diagnosis_service.src.domain.entities import (
    EntityBase, SymptomEntity, HypothesisEntity, DiagnosisSessionEntity, DiagnosisRecordEntity,
)
from services.diagnosis_service.src.domain.value_objects import (
    SeverityScore, ConfidenceScore, TemporalInfo, DiagnosisCriteria,
    HiTOPDimension, ClinicalHypothesis, SessionProgress,
)
from services.diagnosis_service.src.infrastructure.repository import RepositoryFactory
from services.diagnosis_service.tests.fixtures import (
    InMemoryDiagnosisRepository, SessionQueryBuilder, RecordQueryBuilder,
)
from services.diagnosis_service.src.events import (
    DiagnosisEvent, SessionStartedEvent, SessionEndedEvent, PhaseTransitionEvent,
    SymptomExtractedEvent, HypothesisGeneratedEvent, HypothesisChallengedEvent,
    SafetyFlagRaisedEvent, EventDispatcher, EventFactory,
)
from services.diagnosis_service.src.config import (
    DatabaseSettings, RedisSettings, AISettings, ReasoningSettings,
    SafetySettings, ObservabilitySettings, DiagnosisServiceConfig, get_config, reload_config,
)


class TestEntityBase:
    """Tests for EntityBase."""

    def test_entity_creation(self) -> None:
        """Test entity base creation."""
        entity = EntityBase()
        assert entity.id is not None
        assert entity.version == 1
        assert entity.created_at is not None

    def test_entity_touch(self) -> None:
        """Test entity touch updates timestamp and version."""
        entity = EntityBase()
        original_version = entity.version
        original_updated = entity.updated_at
        entity.touch()
        assert entity.version == original_version + 1
        assert entity.updated_at >= original_updated


class TestSymptomEntity:
    """Tests for SymptomEntity."""

    def test_symptom_creation(self) -> None:
        """Test symptom entity creation."""
        symptom = SymptomEntity(name="anxiety", description="Feeling anxious")
        assert symptom.name == "anxiety"
        assert symptom.is_active is True
        assert symptom.validated is False

    def test_symptom_validate(self) -> None:
        """Test symptom validation."""
        symptom = SymptomEntity(name="anxiety")
        symptom.validate("clinician_review")
        assert symptom.validated is True
        assert symptom.validation_source == "clinician_review"

    def test_symptom_deactivate(self) -> None:
        """Test symptom deactivation."""
        symptom = SymptomEntity(name="anxiety")
        symptom.deactivate()
        assert symptom.is_active is False

    def test_symptom_add_trigger(self) -> None:
        """Test adding triggers."""
        symptom = SymptomEntity(name="anxiety")
        symptom.add_trigger("stress")
        symptom.add_trigger("stress")
        assert symptom.triggers == ["stress"]

    def test_symptom_to_dict(self) -> None:
        """Test symptom serialization."""
        symptom = SymptomEntity(name="anxiety", severity=SeverityLevel.MODERATE)
        data = symptom.to_dict()
        assert data["name"] == "anxiety"
        assert data["severity"] == "moderate"


class TestHypothesisEntity:
    """Tests for HypothesisEntity."""

    def test_hypothesis_creation(self) -> None:
        """Test hypothesis entity creation."""
        hypothesis = HypothesisEntity(name="GAD", confidence=Decimal("0.7"))
        assert hypothesis.name == "GAD"
        assert hypothesis.challenged is False

    def test_hypothesis_apply_challenge(self) -> None:
        """Test applying challenge to hypothesis."""
        hypothesis = HypothesisEntity(name="GAD", confidence=Decimal("0.8"))
        hypothesis.apply_challenge(["bias detected"], Decimal("-0.1"))
        assert hypothesis.challenged is True
        assert hypothesis.confidence == Decimal("0.7")
        assert hypothesis.original_confidence == Decimal("0.8")

    def test_hypothesis_calibrate(self) -> None:
        """Test hypothesis calibration."""
        hypothesis = HypothesisEntity(name="MDD", confidence=Decimal("0.6"))
        hypothesis.calibrate(Decimal("0.55"))
        assert hypothesis.calibrated is True
        assert hypothesis.confidence == Decimal("0.55")

    def test_hypothesis_add_evidence(self) -> None:
        """Test adding evidence."""
        hypothesis = HypothesisEntity(name="GAD")
        hypothesis.add_evidence("depressed mood", supporting=True)
        hypothesis.add_evidence("manic episode", supporting=False)
        assert "depressed mood" in hypothesis.supporting_evidence
        assert "manic episode" in hypothesis.contra_evidence

    def test_hypothesis_confidence_level_update(self) -> None:
        """Test confidence level updates correctly."""
        hypothesis = HypothesisEntity(name="GAD", confidence=Decimal("0.9"))
        hypothesis.calibrate(Decimal("0.85"))
        assert hypothesis.confidence_level == ConfidenceLevel.VERY_HIGH


class TestDiagnosisSessionEntity:
    """Tests for DiagnosisSessionEntity."""

    def test_session_creation(self) -> None:
        """Test session entity creation."""
        session = DiagnosisSessionEntity()
        assert session.is_active is True
        assert session.phase == DiagnosisPhase.RAPPORT

    def test_session_add_symptom(self) -> None:
        """Test adding symptom to session."""
        session = DiagnosisSessionEntity()
        symptom = SymptomEntity(name="anxiety")
        session.add_symptom(symptom)
        assert len(session.symptoms) == 1
        assert symptom.session_id == session.id

    def test_session_phase_transition(self) -> None:
        """Test phase transition."""
        session = DiagnosisSessionEntity()
        session.transition_phase(DiagnosisPhase.HISTORY)
        assert session.phase == DiagnosisPhase.HISTORY

    def test_session_add_message(self) -> None:
        """Test adding message."""
        session = DiagnosisSessionEntity()
        session.add_message("user", "I feel anxious")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"

    def test_session_end(self) -> None:
        """Test ending session."""
        session = DiagnosisSessionEntity()
        session.end_session(summary="Test summary", recommendations=["Follow up"])
        assert session.is_active is False
        assert session.ended_at is not None
        assert session.summary == "Test summary"

    def test_session_active_symptoms(self) -> None:
        """Test active symptoms property."""
        session = DiagnosisSessionEntity()
        s1 = SymptomEntity(name="anxiety")
        s2 = SymptomEntity(name="depression")
        s2.deactivate()
        session.add_symptom(s1)
        session.add_symptom(s2)
        assert len(session.active_symptoms) == 1

    def test_session_primary_hypothesis(self) -> None:
        """Test primary hypothesis management."""
        session = DiagnosisSessionEntity()
        h1 = HypothesisEntity(name="GAD")
        session.add_hypothesis(h1)
        session.set_primary_hypothesis(h1.id)
        assert session.get_primary_hypothesis() == h1


class TestDiagnosisRecordEntity:
    """Tests for DiagnosisRecordEntity."""

    def test_record_creation(self) -> None:
        """Test record creation."""
        record = DiagnosisRecordEntity(primary_diagnosis="GAD", confidence=Decimal("0.75"))
        assert record.primary_diagnosis == "GAD"
        assert record.reviewed is False

    def test_record_mark_reviewed(self) -> None:
        """Test marking record as reviewed."""
        record = DiagnosisRecordEntity(primary_diagnosis="GAD")
        record.mark_reviewed("Dr. Smith")
        assert record.reviewed is True
        assert record.reviewed_by == "Dr. Smith"


class TestSeverityScore:
    """Tests for SeverityScore value object."""

    def test_phq9_severe(self) -> None:
        """Test PHQ-9 severe score."""
        score = SeverityScore.from_phq9(22)
        assert score.level == SeverityLevel.SEVERE
        assert score.is_clinical is True

    def test_phq9_moderate(self) -> None:
        """Test PHQ-9 moderate score."""
        score = SeverityScore.from_phq9(12)
        assert score.level == SeverityLevel.MODERATE

    def test_gad7_mild(self) -> None:
        """Test GAD-7 mild score."""
        score = SeverityScore.from_gad7(6)
        assert score.level == SeverityLevel.MILD

    def test_pcl5_threshold(self) -> None:
        """Test PCL-5 clinical threshold."""
        score = SeverityScore.from_pcl5(35)
        assert score.level == SeverityLevel.MILD
        assert "meets clinical threshold" in score.interpretation.lower()

    def test_percentage_calculation(self) -> None:
        """Test percentage calculation."""
        score = SeverityScore.from_phq9(27)
        assert score.percentage == 100.0


class TestConfidenceScore:
    """Tests for ConfidenceScore value object."""

    def test_confidence_creation(self) -> None:
        """Test confidence score creation."""
        confidence = ConfidenceScore.create(Decimal("0.75"), evidence_count=5)
        assert confidence.level == ConfidenceLevel.HIGH
        assert confidence.is_reliable is True

    def test_confidence_clamping(self) -> None:
        """Test confidence clamping to 0-1 range."""
        confidence = ConfidenceScore.create(Decimal("1.5"))
        assert confidence.value == Decimal("1")

    def test_confidence_adjustment(self) -> None:
        """Test confidence adjustment."""
        confidence = ConfidenceScore.create(Decimal("0.7"))
        adjusted = confidence.with_adjustment(Decimal("-0.1"))
        assert adjusted.value == Decimal("0.6")


class TestTemporalInfo:
    """Tests for TemporalInfo value object."""

    def test_temporal_creation(self) -> None:
        """Test temporal info creation."""
        temporal = TemporalInfo.create(duration="3 months")
        assert temporal.duration_days == 90
        assert temporal.is_chronic is False

    def test_chronic_detection(self) -> None:
        """Test chronic condition detection."""
        temporal = TemporalInfo.create(duration="8 months")
        assert temporal.is_chronic is True

    def test_duration_criterion(self) -> None:
        """Test duration criterion check."""
        temporal = TemporalInfo.create(duration="3 weeks")
        assert temporal.meets_duration_criterion is True


class TestClinicalHypothesis:
    """Tests for ClinicalHypothesis value object."""

    def test_hypothesis_creation(self) -> None:
        """Test clinical hypothesis creation."""
        hypothesis = ClinicalHypothesis.create(
            name="Major Depressive Disorder",
            confidence_value=Decimal("0.7"),
            dsm5_code="F32.1",
            criteria_met=["depressed mood", "anhedonia"],
        )
        assert hypothesis.name == "Major Depressive Disorder"
        assert hypothesis.dsm5_code == "F32.1"

    def test_hypothesis_with_challenge(self) -> None:
        """Test applying challenge to hypothesis."""
        hypothesis = ClinicalHypothesis.create(name="GAD", confidence_value=Decimal("0.8"))
        challenged = hypothesis.with_challenge(Decimal("-0.1"), ["anchoring bias"])
        assert challenged.confidence.value < hypothesis.confidence.value
        assert "anchoring bias" in challenged.contra_evidence


class TestInMemoryRepository:
    """Tests for InMemoryDiagnosisRepository."""

    @pytest_asyncio.fixture
    async def repository(self) -> InMemoryDiagnosisRepository:
        """Create repository fixture."""
        return InMemoryDiagnosisRepository()

    @pytest.mark.asyncio
    async def test_save_and_get_session(self, repository: InMemoryDiagnosisRepository) -> None:
        """Test saving and retrieving session."""
        session = DiagnosisSessionEntity()
        await repository.save_session(session)
        retrieved = await repository.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    @pytest.mark.asyncio
    async def test_get_active_session(self, repository: InMemoryDiagnosisRepository) -> None:
        """Test getting active session."""
        user_id = uuid4()
        session = DiagnosisSessionEntity(user_id=user_id)
        await repository.save_session(session)
        active = await repository.get_active_session(user_id)
        assert active is not None
        assert active.user_id == user_id

    @pytest.mark.asyncio
    async def test_list_user_sessions(self, repository: InMemoryDiagnosisRepository) -> None:
        """Test listing user sessions."""
        user_id = uuid4()
        for _ in range(3):
            await repository.save_session(DiagnosisSessionEntity(user_id=user_id))
        sessions = await repository.list_user_sessions(user_id)
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_delete_session(self, repository: InMemoryDiagnosisRepository) -> None:
        """Test deleting session."""
        session = DiagnosisSessionEntity()
        await repository.save_session(session)
        result = await repository.delete_session(session.id)
        assert result is True
        retrieved = await repository.get_session(session.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_save_and_get_record(self, repository: InMemoryDiagnosisRepository) -> None:
        """Test saving and retrieving record."""
        record = DiagnosisRecordEntity(primary_diagnosis="GAD")
        await repository.save_record(record)
        retrieved = await repository.get_record(record.id)
        assert retrieved is not None
        assert retrieved.primary_diagnosis == "GAD"

    @pytest.mark.asyncio
    async def test_delete_user_data(self, repository: InMemoryDiagnosisRepository) -> None:
        """Test GDPR deletion."""
        user_id = uuid4()
        await repository.save_session(DiagnosisSessionEntity(user_id=user_id))
        await repository.save_record(DiagnosisRecordEntity(user_id=user_id))
        deleted = await repository.delete_user_data(user_id)
        assert deleted == 2

    @pytest.mark.asyncio
    async def test_get_statistics(self, repository: InMemoryDiagnosisRepository) -> None:
        """Test repository statistics."""
        await repository.save_session(DiagnosisSessionEntity())
        stats = await repository.get_statistics()
        assert stats["total_sessions"] == 1


class TestSessionQueryBuilder:
    """Tests for SessionQueryBuilder."""

    @pytest.mark.asyncio
    async def test_query_by_user(self) -> None:
        """Test querying by user."""
        repo = InMemoryDiagnosisRepository()
        user_id = uuid4()
        await repo.save_session(DiagnosisSessionEntity(user_id=user_id))
        await repo.save_session(DiagnosisSessionEntity(user_id=uuid4()))
        results = await SessionQueryBuilder(repo).for_user(user_id).execute()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_active_only(self) -> None:
        """Test querying active sessions only."""
        repo = InMemoryDiagnosisRepository()
        s1 = DiagnosisSessionEntity()
        s2 = DiagnosisSessionEntity()
        s2.end_session()
        await repo.save_session(s1)
        await repo.save_session(s2)
        results = await SessionQueryBuilder(repo).active_only().execute()
        assert len(results) == 1


class TestEventDispatcher:
    """Tests for EventDispatcher."""

    @pytest.mark.asyncio
    async def test_dispatch_event(self) -> None:
        """Test dispatching event."""
        dispatcher = EventDispatcher()
        received: list[DiagnosisEvent] = []
        async def handler(event: DiagnosisEvent) -> None:
            received.append(event)
        dispatcher.subscribe("session_started", handler)
        event = SessionStartedEvent(user_id=uuid4(), session_id=uuid4())
        await dispatcher.dispatch(event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_global_handler(self) -> None:
        """Test global event handler."""
        dispatcher = EventDispatcher()
        received: list[DiagnosisEvent] = []
        async def handler(event: DiagnosisEvent) -> None:
            received.append(event)
        dispatcher.subscribe_all(handler)
        await dispatcher.dispatch(SessionStartedEvent())
        await dispatcher.dispatch(SessionEndedEvent())
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_event_log(self) -> None:
        """Test event logging."""
        dispatcher = EventDispatcher()
        await dispatcher.dispatch(SessionStartedEvent())
        await dispatcher.dispatch(SessionEndedEvent())
        log = dispatcher.get_event_log()
        assert len(log) == 2


class TestEventFactory:
    """Tests for EventFactory."""

    def test_session_started_event(self) -> None:
        """Test creating session started event."""
        user_id, session_id = uuid4(), uuid4()
        event = EventFactory.session_started(user_id, session_id, 1)
        assert event.event_type == "session_started"
        assert event.session_number == 1

    def test_phase_transition_event(self) -> None:
        """Test creating phase transition event."""
        event = EventFactory.phase_transition(uuid4(), uuid4(), DiagnosisPhase.RAPPORT,
                                               DiagnosisPhase.HISTORY, Decimal("0.6"))
        assert event.from_phase == DiagnosisPhase.RAPPORT
        assert event.to_phase == DiagnosisPhase.HISTORY

    def test_safety_flag_event(self) -> None:
        """Test creating safety flag event."""
        event = EventFactory.safety_flag_raised(uuid4(), uuid4(), "crisis", "high",
                                                 "mentioned self-harm", "escalate")
        assert event.flag_type == "crisis"
        assert event.severity == "high"


class TestConfiguration:
    """Tests for configuration classes."""

    def test_database_settings(self) -> None:
        """Test database settings."""
        settings = DatabaseSettings()
        assert settings.port == 5432
        assert "postgresql" in settings.connection_string

    def test_redis_settings(self) -> None:
        """Test Redis settings."""
        settings = RedisSettings()
        assert settings.port == 6379
        assert "redis://" in settings.url

    def test_ai_settings(self) -> None:
        """Test AI settings."""
        settings = AISettings()
        assert settings.model_provider == "anthropic"
        assert settings.temperature == 0.3

    def test_reasoning_settings(self) -> None:
        """Test reasoning settings."""
        settings = ReasoningSettings()
        assert settings.enable_anti_sycophancy is True
        assert settings.max_hypotheses == 5

    def test_safety_settings(self) -> None:
        """Test safety settings."""
        settings = SafetySettings()
        assert settings.enable_safety_checks is True
        assert len(settings.crisis_keywords) > 0

    def test_main_config(self) -> None:
        """Test main configuration."""
        config = DiagnosisServiceConfig()
        assert config.service_name == "diagnosis-service"
        assert config.reasoning.enable_devil_advocate is True

    def test_config_to_dict(self) -> None:
        """Test configuration serialization."""
        config = DiagnosisServiceConfig()
        data = config.to_dict()
        assert "service_name" in data
        assert "reasoning" in data

    def test_get_config_cached(self) -> None:
        """Test cached config retrieval."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reload_config(self) -> None:
        """Test config reload."""
        config1 = get_config()
        config2 = reload_config()
        assert config1.service_name == config2.service_name


class TestRepositoryFactory:
    """Tests for RepositoryFactory."""

    def test_create_in_memory(self) -> None:
        """Test creating in-memory repository directly from fixtures."""
        repo = InMemoryDiagnosisRepository()
        assert isinstance(repo, InMemoryDiagnosisRepository)

    def test_get_default_singleton(self) -> None:
        """Test default repository is singleton when instance is set."""
        from unittest.mock import MagicMock
        RepositoryFactory.reset()
        mock_repo = MagicMock()
        RepositoryFactory._instance = mock_repo
        try:
            repo1 = RepositoryFactory.get_default()
            repo2 = RepositoryFactory.get_default()
            assert repo1 is repo2
        finally:
            RepositoryFactory.reset()
