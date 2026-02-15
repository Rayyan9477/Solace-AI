"""
Data persistence round-trip tests for Solace-AI platform.

Verifies entity serialization/deserialization, schema registry integrity,
PHI field annotations, repository factory patterns, and domain entity
state management without requiring a running database.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# Domain entity round-trip tests
# ---------------------------------------------------------------------------


class TestOrchestratorStateSerialization:
    """Round-trip tests for orchestrator state schema entities."""

    def test_message_entry_round_trip(self):
        """MessageEntry survives to_dict/from_dict round-trip."""
        from services.orchestrator_service.src.langgraph.state_schema import MessageEntry

        original = MessageEntry.user_message("Hello, I need help", metadata={"source": "web"})
        d = original.to_dict()
        restored = MessageEntry.from_dict(d)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.metadata == original.metadata
        assert str(restored.message_id) == str(original.message_id)

    def test_safety_flags_round_trip(self):
        """SafetyFlags survives to_dict/from_dict round-trip."""
        from services.orchestrator_service.src.langgraph.state_schema import SafetyFlags
        from solace_common.enums import CrisisLevel as RiskLevel

        original = SafetyFlags(
            risk_level=RiskLevel.HIGH,
            crisis_detected=True,
            crisis_type="suicidal_ideation",
            requires_escalation=True,
            escalation_reason="High severity risk",
            monitoring_level="intensive",
            triggered_keywords=["suicide", "end my life"],
            contraindications=["antidepressant"],
        )
        d = original.to_dict()
        restored = SafetyFlags.from_dict(d)

        assert restored.risk_level == RiskLevel.HIGH
        assert restored.crisis_detected is True
        assert restored.crisis_type == "suicidal_ideation"
        assert restored.requires_escalation is True
        assert restored.monitoring_level == "intensive"
        assert set(restored.triggered_keywords) == {"suicide", "end my life"}

    def test_safety_flags_safe_factory(self):
        """SafetyFlags.safe() creates non-crisis flags."""
        from services.orchestrator_service.src.langgraph.state_schema import SafetyFlags

        flags = SafetyFlags.safe()
        assert flags.is_safe() is True
        assert flags.crisis_detected is False

    def test_agent_result_round_trip(self):
        """AgentResult survives to_dict/from_dict round-trip."""
        from services.orchestrator_service.src.langgraph.state_schema import (
            AgentResult,
            AgentType,
        )

        original = AgentResult(
            agent_type=AgentType.THERAPY,
            success=True,
            response_content="Here is a CBT technique for you.",
            confidence=0.92,
            processing_time_ms=145.3,
            metadata={"technique": "cognitive_restructuring"},
        )
        d = original.to_dict()
        restored = AgentResult.from_dict(d)

        assert restored.agent_type == AgentType.THERAPY
        assert restored.success is True
        assert restored.response_content == original.response_content
        assert restored.confidence == pytest.approx(0.92)

    def test_processing_metadata_round_trip(self):
        """ProcessingMetadata survives to_dict/from_dict round-trip."""
        from services.orchestrator_service.src.langgraph.state_schema import (
            ProcessingMetadata,
            AgentType,
        )

        original = ProcessingMetadata(
            session_id=uuid4(),
            user_id=uuid4(),
            active_agents=[AgentType.THERAPY, AgentType.SAFETY],
            completed_agents=[AgentType.MEMORY],
            retry_count=1,
        )
        d = original.to_dict()
        restored = ProcessingMetadata.from_dict(d)

        assert str(restored.session_id) == str(original.session_id)
        assert str(restored.user_id) == str(original.user_id)
        assert len(restored.active_agents) == 2
        assert restored.retry_count == 1

    def test_create_initial_state_has_all_required_fields(self):
        """create_initial_state produces a complete OrchestratorState."""
        from services.orchestrator_service.src.langgraph.state_schema import (
            create_initial_state,
            StateValidator,
        )

        state = create_initial_state(
            user_id=uuid4(),
            session_id=uuid4(),
            message="I've been feeling anxious lately",
        )

        is_valid, errors = StateValidator.validate_state(state)
        assert is_valid, f"State validation errors: {errors}"
        assert state["current_message"] == "I've been feeling anxious lately"
        assert len(state["messages"]) == 1
        assert state["processing_phase"] == "initialized"

    def test_safety_flags_reducer_escalates_risk(self):
        """update_safety_flags reducer always keeps the higher risk level."""
        from services.orchestrator_service.src.langgraph.state_schema import (
            update_safety_flags,
        )

        left = {"risk_level": "LOW", "crisis_detected": False}
        right = {"risk_level": "HIGH", "crisis_detected": True}
        merged = update_safety_flags(left, right)

        assert merged["risk_level"] == "HIGH"
        assert merged["crisis_detected"] is True

    def test_safety_flags_reducer_preserves_high_from_left(self):
        """Reducer doesn't downgrade risk from left side."""
        from services.orchestrator_service.src.langgraph.state_schema import (
            update_safety_flags,
        )

        left = {"risk_level": "CRITICAL", "crisis_detected": True}
        right = {"risk_level": "LOW", "crisis_detected": False}
        merged = update_safety_flags(left, right)

        assert merged["risk_level"] == "CRITICAL"
        assert merged["crisis_detected"] is True


# ---------------------------------------------------------------------------
# Diagnosis domain entity round-trips
# ---------------------------------------------------------------------------


class TestDiagnosisEntityRoundTrip:
    """Round-trip tests for diagnosis service domain entities."""

    def test_symptom_entity_to_dict(self):
        """SymptomEntity serializes all fields."""
        from services.diagnosis_service.src.domain.entities import SymptomEntity
        from services.diagnosis_service.src.schemas import SeverityLevel, SymptomType

        symptom = SymptomEntity(
            name="persistent_sadness",
            description="Feeling sad most of the day",
            symptom_type=SymptomType.EMOTIONAL,
            severity=SeverityLevel.MODERATE,
            onset="2 weeks ago",
            duration="continuous",
            frequency="daily",
            triggers=["morning", "isolation"],
            confidence=Decimal("0.85"),
        )
        d = symptom.to_dict()

        assert d["name"] == "persistent_sadness"
        assert d["severity"] == "MODERATE"
        assert d["symptom_type"] == "emotional"
        assert d["confidence"] == "0.85"
        assert "morning" in d["triggers"]
        assert UUID(d["id"])  # Valid UUID

    def test_diagnosis_session_lifecycle(self):
        """DiagnosisSession supports full lifecycle: add symptoms, hypotheses, end."""
        from services.diagnosis_service.src.domain.entities import (
            DiagnosisSessionEntity,
            SymptomEntity,
            HypothesisEntity,
        )
        from services.diagnosis_service.src.schemas import (
            SeverityLevel,
            SymptomType,
            ConfidenceLevel,
            DiagnosisPhase,
        )

        session = DiagnosisSessionEntity(user_id=uuid4())
        assert session.is_active is True

        # Add symptom
        symptom = SymptomEntity(
            name="insomnia",
            symptom_type=SymptomType.BEHAVIORAL,
            severity=SeverityLevel.MILD,
        )
        session.add_symptom(symptom)
        assert len(session.symptoms) == 1

        # Add hypothesis
        hyp = HypothesisEntity(
            name="Major Depressive Disorder",
            dsm5_code="F32.1",
            confidence_level=ConfidenceLevel.MEDIUM,
        )
        session.add_hypothesis(hyp)
        session.set_primary_hypothesis(hyp.id)
        assert session.get_primary_hypothesis().name == "Major Depressive Disorder"

        # End session
        session.end_session(summary="Patient shows signs of mild depression")
        assert session.is_active is False
        assert session.ended_at is not None

        d = session.to_dict()
        assert d["symptom_count"] == 1
        assert d["hypothesis_count"] == 1
        assert d["is_active"] is False

    def test_diagnosis_record_entity_to_dict(self):
        """DiagnosisRecordEntity serializes correctly."""
        from services.diagnosis_service.src.domain.entities import DiagnosisRecordEntity
        from services.diagnosis_service.src.schemas import SeverityLevel

        record = DiagnosisRecordEntity(
            primary_diagnosis="Major Depressive Disorder",
            dsm5_code="F32.1",
            icd11_code="6A70",
            confidence=Decimal("0.82"),
            severity=SeverityLevel.MODERATE,
            symptom_summary=["persistent_sadness", "insomnia", "fatigue"],
            recommendations=["CBT", "medication evaluation"],
        )
        d = record.to_dict()

        assert d["primary_diagnosis"] == "Major Depressive Disorder"
        assert d["dsm5_code"] == "F32.1"
        assert d["severity"] == "MODERATE"
        assert len(d["symptom_summary"]) == 3


# ---------------------------------------------------------------------------
# Therapy DTO round-trips
# ---------------------------------------------------------------------------


class TestTherapyDTORoundTrip:
    """Round-trip tests for therapy service Pydantic DTOs."""

    def test_treatment_plan_dto_serialization(self):
        """TreatmentPlanDTO round-trips through JSON."""
        from services.therapy_service.src.schemas import (
            TreatmentPlanDTO,
            TherapyModality,
        )
        from solace_common.enums import SeverityLevel

        plan = TreatmentPlanDTO(
            plan_id=uuid4(),
            user_id=uuid4(),
            primary_diagnosis="Generalized Anxiety Disorder",
            severity=SeverityLevel.MODERATE,
            primary_modality=TherapyModality.CBT,
            adjunct_modalities=[TherapyModality.MINDFULNESS],
            current_phase=2,
            sessions_completed=4,
            skills_acquired=["thought_records", "breathing_techniques"],
        )
        d = plan.model_dump(mode="json")
        restored = TreatmentPlanDTO.model_validate(d)

        assert restored.primary_diagnosis == plan.primary_diagnosis
        assert restored.severity == SeverityLevel.MODERATE
        assert restored.primary_modality == TherapyModality.CBT
        assert len(restored.skills_acquired) == 2

    def test_session_state_dto_serialization(self):
        """SessionStateDTO round-trips through JSON."""
        from services.therapy_service.src.schemas import (
            SessionStateDTO,
            SessionPhase,
        )
        from solace_common.enums import CrisisLevel as RiskLevel

        session = SessionStateDTO(
            session_id=uuid4(),
            user_id=uuid4(),
            treatment_plan_id=uuid4(),
            session_number=3,
            current_phase=SessionPhase.WORKING,
            mood_rating=6,
            agenda_items=["Review homework", "Practice CBT"],
            current_risk=RiskLevel.NONE,
            engagement_score=0.78,
        )
        d = session.model_dump(mode="json")
        restored = SessionStateDTO.model_validate(d)

        assert restored.session_number == 3
        assert restored.current_phase == SessionPhase.WORKING
        assert restored.mood_rating == 6
        assert restored.current_risk == RiskLevel.NONE

    def test_outcome_score_dto_serialization(self):
        """OutcomeScoreDTO round-trips with PHQ-9 scores."""
        from services.therapy_service.src.schemas import (
            OutcomeScoreDTO,
            OutcomeInstrument,
        )
        from solace_common.enums import SeverityLevel

        score = OutcomeScoreDTO(
            score_id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            total_score=14,
            subscale_scores={"cognitive": 6, "somatic": 8},
            severity_category=SeverityLevel.MODERATE,
            clinically_significant=True,
            reliable_change=False,
            recorded_at=datetime.now(timezone.utc),
        )
        d = score.model_dump(mode="json")
        restored = OutcomeScoreDTO.model_validate(d)

        assert restored.instrument == OutcomeInstrument.PHQ9
        assert restored.total_score == 14
        assert restored.severity_category == SeverityLevel.MODERATE
        assert restored.clinically_significant is True


# ---------------------------------------------------------------------------
# Schema Registry integrity
# ---------------------------------------------------------------------------


class TestSchemaRegistryIntegrity:
    """Tests that the schema registry has safety entities properly registered."""

    def test_safety_entities_registered(self):
        """All safety entities are in the schema registry."""
        # Force entity module import to trigger @SchemaRegistry.register
        import services.orchestrator_service.src.langgraph.state_schema  # noqa: F401
        from src.solace_infrastructure.database.entities.safety_entities import (
            SafetyAssessment,
            SafetyPlan,
            RiskFactor,
            ContraindicationCheck,
        )
        from src.solace_infrastructure.database.schema_registry import SchemaRegistry

        assert SchemaRegistry.is_registered("safety_assessments")
        assert SchemaRegistry.is_registered("safety_plans")
        assert SchemaRegistry.is_registered("risk_factors")
        assert SchemaRegistry.is_registered("contraindication_checks")

    def test_safety_assessment_has_phi_fields(self):
        """SafetyAssessment declares PHI fields for encryption."""
        from src.solace_infrastructure.database.entities.safety_entities import (
            SafetyAssessment,
        )

        phi_fields = SafetyAssessment.__phi_fields__
        assert "content_assessed" in phi_fields
        assert "assessment_notes" in phi_fields
        assert "review_notes" in phi_fields

    def test_safety_plan_has_phi_fields(self):
        """SafetyPlan declares PHI fields for encryption."""
        from src.solace_infrastructure.database.entities.safety_entities import SafetyPlan

        phi_fields = SafetyPlan.__phi_fields__
        assert "clinician_notes" in phi_fields

    def test_registry_statistics(self):
        """Schema registry tracks registered entities."""
        from src.solace_infrastructure.database.schema_registry import SchemaRegistry

        stats = SchemaRegistry.get_statistics()
        assert stats["total_entities"] >= 4  # At least the 4 safety entities


# ---------------------------------------------------------------------------
# Repository factory patterns
# ---------------------------------------------------------------------------


class TestRepositoryFactoryPatterns:
    """Tests that repository factories create correct instances."""

    def test_diagnosis_repository_factory_singleton(self):
        """DiagnosisRepositoryFactory returns singleton."""
        from services.diagnosis_service.src.infrastructure.repository import (
            RepositoryFactory,
        )

        RepositoryFactory.reset()
        f1 = RepositoryFactory.get_default()
        f2 = RepositoryFactory.get_default()
        assert f1 is f2
        RepositoryFactory.reset()

    def test_personality_repository_factory_singleton(self):
        """PersonalityRepositoryFactory returns singleton."""
        from services.personality_service.src.infrastructure.repository import (
            RepositoryFactory,
        )

        RepositoryFactory.reset()
        f1 = RepositoryFactory.get_default()
        f2 = RepositoryFactory.get_default()
        assert f1 is f2
        RepositoryFactory.reset()


# ---------------------------------------------------------------------------
# Canonical enum integration with entities
# ---------------------------------------------------------------------------


class TestCanonicalEnumEntityIntegration:
    """Verify canonical enums are used consistently across domain entities."""

    def test_safety_flags_uses_canonical_risk_level(self):
        """SafetyFlags uses CrisisLevel from solace_common."""
        from services.orchestrator_service.src.langgraph.state_schema import SafetyFlags
        from solace_common.enums import CrisisLevel

        flags = SafetyFlags(risk_level=CrisisLevel.HIGH)
        assert isinstance(flags.risk_level, CrisisLevel)
        assert flags.risk_level.value == "HIGH"

    def test_therapy_session_uses_canonical_risk_level(self):
        """Therapy SessionStateDTO uses CrisisLevel."""
        from services.therapy_service.src.schemas import SessionStateDTO, SessionPhase
        from solace_common.enums import CrisisLevel

        session = SessionStateDTO(
            session_id=uuid4(),
            user_id=uuid4(),
            treatment_plan_id=uuid4(),
            session_number=1,
            current_phase=SessionPhase.OPENING,
            current_risk=CrisisLevel.ELEVATED,
        )
        assert session.current_risk == CrisisLevel.ELEVATED

    def test_therapy_plan_uses_canonical_severity(self):
        """TreatmentPlanDTO uses SeverityLevel from solace_common."""
        from services.therapy_service.src.schemas import (
            TreatmentPlanDTO,
            TherapyModality,
        )
        from solace_common.enums import SeverityLevel

        plan = TreatmentPlanDTO(
            plan_id=uuid4(),
            user_id=uuid4(),
            primary_diagnosis="GAD",
            severity=SeverityLevel.MILD,
            primary_modality=TherapyModality.CBT,
        )
        assert isinstance(plan.severity, SeverityLevel)

    def test_diagnosis_record_uses_canonical_severity(self):
        """DiagnosisRecordEntity uses SeverityLevel from solace_common."""
        from services.diagnosis_service.src.domain.entities import DiagnosisRecordEntity
        from solace_common.enums import SeverityLevel

        record = DiagnosisRecordEntity(
            primary_diagnosis="MDD",
            severity=SeverityLevel.SEVERE,
        )
        assert record.severity == SeverityLevel.SEVERE

    def test_severity_from_string_handles_legacy_lowercase(self):
        """SeverityLevel.from_string handles lowercase values from old data."""
        from solace_common.enums import SeverityLevel

        assert SeverityLevel.from_string("mild") == SeverityLevel.MILD
        assert SeverityLevel.from_string("MODERATE") == SeverityLevel.MODERATE
        assert SeverityLevel.from_string("moderately_severe") == SeverityLevel.MODERATELY_SEVERE

    def test_crisis_level_from_string_handles_aliases(self):
        """CrisisLevel.from_string maps service-specific aliases."""
        from solace_common.enums import CrisisLevel

        assert CrisisLevel.from_string("moderate") == CrisisLevel.ELEVATED
        assert CrisisLevel.from_string("imminent") == CrisisLevel.CRITICAL
        assert CrisisLevel.from_string("severe") == CrisisLevel.CRITICAL
        assert CrisisLevel.from_string("minimal") == CrisisLevel.NONE
