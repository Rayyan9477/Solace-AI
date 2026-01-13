"""
Unit tests for Diagnosis Service Batch 5.1 - Core Components.
Tests 4-step Chain-of-Reasoning, symptom extraction, and differential generation.
"""
from __future__ import annotations
import pytest
from decimal import Decimal
from uuid import uuid4
from services.diagnosis_service.src.schemas import (
    DiagnosisPhase, SeverityLevel, SymptomType, ReasoningStep,
    SymptomDTO, HypothesisDTO, DifferentialDTO,
    AssessmentRequest, SessionStartRequest, SessionEndRequest,
    SymptomExtractionRequest, DifferentialRequest,
)
from services.diagnosis_service.src.domain.service import (
    DiagnosisService, DiagnosisServiceSettings,
)
from services.diagnosis_service.src.domain.models import SessionState, AssessmentResult
from services.diagnosis_service.src.domain.symptom_extractor import (
    SymptomExtractor, SymptomExtractorSettings, ExtractionResult,
)
from services.diagnosis_service.src.domain.differential import (
    DifferentialGenerator, DifferentialSettings, DifferentialResult,
)


class TestSymptomExtractor:
    """Tests for SymptomExtractor component."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.extractor = SymptomExtractor(SymptomExtractorSettings())

    @pytest.mark.asyncio
    async def test_extract_depressed_mood(self) -> None:
        """Test extraction of depressed mood symptoms."""
        message = "I've been feeling really sad and hopeless lately"
        result = await self.extractor.extract(message, [], [])
        assert isinstance(result, ExtractionResult)
        symptom_names = {s.name for s in result.symptoms}
        assert "depressed_mood" in symptom_names

    @pytest.mark.asyncio
    async def test_extract_anxiety_symptoms(self) -> None:
        """Test extraction of anxiety symptoms."""
        message = "I feel anxious and worried all the time, very nervous"
        result = await self.extractor.extract(message, [], [])
        symptom_names = {s.name for s in result.symptoms}
        assert "anxiety" in symptom_names

    @pytest.mark.asyncio
    async def test_extract_sleep_disturbance(self) -> None:
        """Test extraction of sleep disturbance."""
        message = "I can't sleep at night, I have insomnia"
        result = await self.extractor.extract(message, [], [])
        symptom_names = {s.name for s in result.symptoms}
        assert "sleep_disturbance" in symptom_names

    @pytest.mark.asyncio
    async def test_extract_multiple_symptoms(self) -> None:
        """Test extraction of multiple symptoms from one message."""
        message = "I'm sad, can't sleep, and feel exhausted with no energy"
        result = await self.extractor.extract(message, [], [])
        assert len(result.symptoms) >= 2

    @pytest.mark.asyncio
    async def test_severity_detection_severe(self) -> None:
        """Test detection of severe severity indicators."""
        message = "I'm extremely depressed and it's unbearable"
        result = await self.extractor.extract(message, [], [])
        severe_symptoms = [s for s in result.symptoms if s.severity == SeverityLevel.SEVERE]
        assert len(severe_symptoms) > 0

    @pytest.mark.asyncio
    async def test_severity_detection_mild(self) -> None:
        """Test detection of mild severity indicators."""
        message = "I feel a little sad sometimes"
        result = await self.extractor.extract(message, [], [])
        if result.symptoms:
            assert result.symptoms[0].severity in [SeverityLevel.MINIMAL, SeverityLevel.MILD]

    @pytest.mark.asyncio
    async def test_risk_indicator_detection(self) -> None:
        """Test detection of risk indicators."""
        message = "I've been thinking about hurting myself"
        result = await self.extractor.extract(message, [], [])
        assert "self_harm" in result.risk_indicators

    @pytest.mark.asyncio
    async def test_temporal_extraction(self) -> None:
        """Test extraction of temporal information."""
        message = "I've been feeling sad for 3 weeks now"
        result = await self.extractor.extract(message, [], [])
        assert "duration" in result.temporal_info

    @pytest.mark.asyncio
    async def test_contextual_factors_work(self) -> None:
        """Test extraction of work-related contextual factors."""
        message = "My job is causing me a lot of stress and anxiety"
        result = await self.extractor.extract(message, [], [])
        assert "work_related" in result.contextual_factors or "stress" in result.contextual_factors

    @pytest.mark.asyncio
    async def test_no_duplicate_symptoms(self) -> None:
        """Test that existing symptoms are not duplicated."""
        existing = [SymptomDTO(
            symptom_id=uuid4(), name="depressed_mood",
            description="test", symptom_type=SymptomType.EMOTIONAL
        )]
        message = "I feel very sad and depressed"
        result = await self.extractor.extract(message, [], existing)
        new_depressed = [s for s in result.symptoms if s.name == "depressed_mood"]
        assert len(new_depressed) == 0

    def test_merge_symptoms_upgrade_severity(self) -> None:
        """Test that symptom merge upgrades severity when appropriate."""
        existing = [SymptomDTO(
            symptom_id=uuid4(), name="anxiety",
            description="mild anxiety", symptom_type=SymptomType.EMOTIONAL,
            severity=SeverityLevel.MILD
        )]
        new_symptoms = [SymptomDTO(
            symptom_id=uuid4(), name="anxiety",
            description="severe anxiety", symptom_type=SymptomType.EMOTIONAL,
            severity=SeverityLevel.SEVERE
        )]
        merged = self.extractor.merge_symptoms(existing, new_symptoms)
        anxiety_symptom = next(s for s in merged if s.name == "anxiety")
        assert anxiety_symptom.severity == SeverityLevel.SEVERE

    def test_symptom_burden_calculation(self) -> None:
        """Test symptom burden calculation."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="anxiety",
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="fatigue",
                      description="test", symptom_type=SymptomType.SOMATIC,
                      severity=SeverityLevel.MILD),
        ]
        burden = self.extractor.calculate_symptom_burden(symptoms)
        assert burden["symptom_count"] == 2
        assert burden["categories"] == 2
        assert burden["average_severity"] > 0

    def test_get_statistics(self) -> None:
        """Test statistics retrieval."""
        stats = self.extractor.get_statistics()
        assert "extractions" in stats
        assert "symptoms_found" in stats


class TestDifferentialGenerator:
    """Tests for DifferentialGenerator component."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.generator = DifferentialGenerator(DifferentialSettings())

    @pytest.mark.asyncio
    async def test_generate_depression_hypothesis(self) -> None:
        """Test generation of depression hypothesis from symptoms."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia",
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="sleep_disturbance",
                      description="test", symptom_type=SymptomType.SOMATIC),
            SymptomDTO(symptom_id=uuid4(), name="fatigue",
                      description="test", symptom_type=SymptomType.SOMATIC),
        ]
        result = await self.generator.generate(symptoms, {})
        assert isinstance(result, DifferentialResult)
        assert len(result.hypotheses) > 0
        hypothesis_names = [h.name.lower() for h in result.hypotheses]
        assert any("depress" in name for name in hypothesis_names)

    @pytest.mark.asyncio
    async def test_generate_anxiety_hypothesis(self) -> None:
        """Test generation of anxiety hypothesis from symptoms."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="anxiety",
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="physical_tension",
                      description="test", symptom_type=SymptomType.SOMATIC),
            SymptomDTO(symptom_id=uuid4(), name="sleep_disturbance",
                      description="test", symptom_type=SymptomType.SOMATIC),
        ]
        result = await self.generator.generate(symptoms, {})
        assert len(result.hypotheses) > 0

    @pytest.mark.asyncio
    async def test_hypotheses_have_dsm5_codes(self) -> None:
        """Test that hypotheses include DSM-5 codes."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
        ]
        result = await self.generator.generate(symptoms, {})
        for hyp in result.hypotheses:
            assert hyp.dsm5_code is not None

    @pytest.mark.asyncio
    async def test_hypotheses_have_confidence_intervals(self) -> None:
        """Test that hypotheses include confidence intervals."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
        ]
        result = await self.generator.generate(symptoms, {})
        for hyp in result.hypotheses:
            assert hyp.confidence_interval is not None
            assert hyp.confidence_interval[0] <= hyp.confidence <= hyp.confidence_interval[1]

    @pytest.mark.asyncio
    async def test_hitop_scores_generated(self) -> None:
        """Test that HiTOP dimensional scores are generated."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
            SymptomDTO(symptom_id=uuid4(), name="anxiety",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
        ]
        result = await self.generator.generate(symptoms, {})
        assert "internalizing" in result.hitop_scores
        assert result.hitop_scores["internalizing"] > Decimal("0")

    @pytest.mark.asyncio
    async def test_missing_info_identified(self) -> None:
        """Test that missing information is identified."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
        ]
        result = await self.generator.generate(symptoms, {})
        assert len(result.missing_info) > 0

    @pytest.mark.asyncio
    async def test_recommended_questions_generated(self) -> None:
        """Test that recommended questions are generated."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
        ]
        result = await self.generator.generate(symptoms, {})
        assert len(result.recommended_questions) > 0

    @pytest.mark.asyncio
    async def test_max_hypotheses_limit(self) -> None:
        """Test that hypothesis count respects max limit."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
            SymptomDTO(symptom_id=uuid4(), name="anxiety",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
            SymptomDTO(symptom_id=uuid4(), name="sleep_disturbance",
                      description="test", symptom_type=SymptomType.SOMATIC),
            SymptomDTO(symptom_id=uuid4(), name="fatigue",
                      description="test", symptom_type=SymptomType.SOMATIC),
            SymptomDTO(symptom_id=uuid4(), name="concentration_difficulty",
                      description="test", symptom_type=SymptomType.COGNITIVE),
        ]
        result = await self.generator.generate(symptoms, {})
        assert len(result.hypotheses) <= self.generator._settings.max_hypotheses

    def test_get_dsm5_criteria(self) -> None:
        """Test retrieval of DSM-5 criteria."""
        criteria = self.generator.get_dsm5_criteria("major_depressive_disorder")
        assert criteria is not None
        assert "dsm5_code" in criteria
        assert "required_symptoms" in criteria

    def test_get_hitop_dimension(self) -> None:
        """Test retrieval of HiTOP dimension."""
        dimension = self.generator.get_hitop_dimension("internalizing")
        assert dimension is not None
        assert "symptoms" in dimension


class TestDiagnosisService:
    """Tests for DiagnosisService component."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.extractor = SymptomExtractor(SymptomExtractorSettings())
        self.generator = DifferentialGenerator(DifferentialSettings())
        self.service = DiagnosisService(
            settings=DiagnosisServiceSettings(),
            symptom_extractor=self.extractor,
            differential_generator=self.generator,
        )

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self) -> None:
        """Test service initialization and shutdown."""
        await self.service.initialize()
        assert self.service._initialized is True
        await self.service.shutdown()
        assert self.service._initialized is False

    @pytest.mark.asyncio
    async def test_start_session(self) -> None:
        """Test starting a diagnosis session."""
        await self.service.initialize()
        result = await self.service.start_session(
            user_id=uuid4(),
            session_type="assessment",
            initial_context={},
            previous_session_id=None,
        )
        assert result.session_id is not None
        assert result.session_number == 1
        assert result.initial_phase == DiagnosisPhase.RAPPORT
        assert len(result.greeting) > 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_start_session_returning_user(self) -> None:
        """Test starting session for returning user."""
        await self.service.initialize()
        user_id = uuid4()
        result1 = await self.service.start_session(user_id, "assessment", {}, None)
        await self.service.end_session(user_id, result1.session_id, False)
        result2 = await self.service.start_session(user_id, "assessment", {}, result1.session_id)
        assert result2.session_number == 2
        assert result2.loaded_context is True
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_assess_full_pipeline(self) -> None:
        """Test full 4-step assessment pipeline."""
        await self.service.initialize()
        user_id = uuid4()
        session_result = await self.service.start_session(user_id, "assessment", {}, None)
        result = await self.service.assess(
            user_id=user_id,
            session_id=session_result.session_id,
            message="I've been feeling very sad and hopeless for weeks",
            conversation_history=[],
            existing_symptoms=[],
            current_phase=DiagnosisPhase.RAPPORT,
            current_differential=None,
            user_context={},
        )
        assert result.assessment_id is not None
        assert len(result.reasoning_chain) == 4
        step_names = [r.step for r in result.reasoning_chain]
        assert ReasoningStep.ANALYZE in step_names
        assert ReasoningStep.HYPOTHESIZE in step_names
        assert ReasoningStep.CHALLENGE in step_names
        assert ReasoningStep.SYNTHESIZE in step_names
        assert result.processing_time_ms >= 0  # Can be 0 if sub-millisecond
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_assess_extracts_symptoms(self) -> None:
        """Test that assessment extracts symptoms."""
        await self.service.initialize()
        user_id = uuid4()
        session_result = await self.service.start_session(user_id, "assessment", {}, None)
        result = await self.service.assess(
            user_id=user_id,
            session_id=session_result.session_id,
            message="I feel depressed and can't sleep",
            conversation_history=[],
            existing_symptoms=[],
            current_phase=DiagnosisPhase.HISTORY,
            current_differential=None,
            user_context={},
        )
        assert len(result.extracted_symptoms) > 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_assess_generates_differential(self) -> None:
        """Test that assessment generates differential."""
        await self.service.initialize()
        user_id = uuid4()
        session_result = await self.service.start_session(user_id, "assessment", {}, None)
        result = await self.service.assess(
            user_id=user_id,
            session_id=session_result.session_id,
            message="I've been feeling sad, hopeless, can't enjoy anything",
            conversation_history=[],
            existing_symptoms=[],
            current_phase=DiagnosisPhase.ASSESSMENT,
            current_differential=None,
            user_context={},
        )
        assert result.differential is not None
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_assess_response_text_generated(self) -> None:
        """Test that assessment generates response text."""
        await self.service.initialize()
        user_id = uuid4()
        session_result = await self.service.start_session(user_id, "assessment", {}, None)
        result = await self.service.assess(
            user_id=user_id,
            session_id=session_result.session_id,
            message="I'm feeling anxious",
            conversation_history=[],
            existing_symptoms=[],
            current_phase=DiagnosisPhase.RAPPORT,
            current_differential=None,
            user_context={},
        )
        assert len(result.response_text) > 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_end_session(self) -> None:
        """Test ending a diagnosis session."""
        await self.service.initialize()
        user_id = uuid4()
        session_result = await self.service.start_session(user_id, "assessment", {}, None)
        end_result = await self.service.end_session(
            user_id=user_id,
            session_id=session_result.session_id,
            generate_summary=True,
        )
        assert end_result.duration_minutes >= 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_get_session_state(self) -> None:
        """Test getting session state."""
        await self.service.initialize()
        user_id = uuid4()
        session_result = await self.service.start_session(user_id, "assessment", {}, None)
        state = await self.service.get_session_state(session_result.session_id)
        assert state is not None
        assert state["phase"] == DiagnosisPhase.RAPPORT.value
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_get_session_state_not_found(self) -> None:
        """Test getting non-existent session state."""
        await self.service.initialize()
        state = await self.service.get_session_state(uuid4())
        assert state is None
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_extract_symptoms_only(self) -> None:
        """Test symptom extraction without full assessment."""
        await self.service.initialize()
        result = await self.service.extract_symptoms(
            user_id=uuid4(),
            session_id=uuid4(),
            message="I'm feeling anxious and worried",
            conversation_history=[],
            existing_symptoms=[],
        )
        assert result.extraction_id is not None
        assert len(result.extracted_symptoms) > 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_generate_differential_only(self) -> None:
        """Test differential generation without full assessment."""
        await self.service.initialize()
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
        ]
        result = await self.service.generate_differential(
            user_id=uuid4(),
            session_id=uuid4(),
            symptoms=symptoms,
            user_history={},
            current_differential=None,
        )
        assert result.differential_id is not None
        assert result.differential is not None
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_challenge_hypothesis(self) -> None:
        """Test Devil's Advocate challenge."""
        await self.service.initialize()
        user_id = uuid4()
        session_result = await self.service.start_session(user_id, "assessment", {}, None)
        await self.service.assess(
            user_id=user_id,
            session_id=session_result.session_id,
            message="I've been feeling sad and hopeless",
            conversation_history=[],
            existing_symptoms=[],
            current_phase=DiagnosisPhase.ASSESSMENT,
            current_differential=None,
            user_context={},
        )
        challenge_result = await self.service.challenge_hypothesis(
            session_id=session_result.session_id,
            hypothesis_id=uuid4(),
        )
        assert challenge_result is not None
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_get_history(self) -> None:
        """Test getting diagnosis history."""
        await self.service.initialize()
        user_id = uuid4()
        session_result = await self.service.start_session(user_id, "assessment", {}, None)
        await self.service.end_session(user_id, session_result.session_id, True)
        history = await self.service.get_history(user_id, 10, True, True)
        assert len(history.sessions) == 1
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_delete_user_data(self) -> None:
        """Test GDPR-compliant user data deletion."""
        await self.service.initialize()
        user_id = uuid4()
        await self.service.start_session(user_id, "assessment", {}, None)
        await self.service.delete_user_data(user_id)
        history = await self.service.get_history(user_id, 10, True, True)
        assert len(history.sessions) == 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        """Test service status retrieval."""
        await self.service.initialize()
        status = await self.service.get_status()
        assert status["status"] == "operational"
        assert status["initialized"] is True
        assert "statistics" in status
        await self.service.shutdown()


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_symptom_dto_validation(self) -> None:
        """Test SymptomDTO validation."""
        symptom = SymptomDTO(
            symptom_id=uuid4(),
            name="anxiety",
            description="Test symptom",
            symptom_type=SymptomType.EMOTIONAL,
            severity=SeverityLevel.MODERATE,
            confidence=Decimal("0.8"),
        )
        assert symptom.confidence == Decimal("0.8")
        assert symptom.symptom_type == SymptomType.EMOTIONAL

    def test_symptom_dto_confidence_bounds(self) -> None:
        """Test SymptomDTO confidence bounds validation."""
        with pytest.raises(ValueError):
            SymptomDTO(
                symptom_id=uuid4(),
                name="test",
                description="test",
                symptom_type=SymptomType.EMOTIONAL,
                confidence=Decimal("1.5"),
            )

    def test_hypothesis_dto_validation(self) -> None:
        """Test HypothesisDTO validation."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(),
            name="Major Depressive Disorder",
            dsm5_code="F32",
            confidence=Decimal("0.75"),
            severity=SeverityLevel.MODERATE,
        )
        assert hypothesis.confidence == Decimal("0.75")

    def test_differential_dto_structure(self) -> None:
        """Test DifferentialDTO structure."""
        differential = DifferentialDTO(
            primary=HypothesisDTO(
                hypothesis_id=uuid4(),
                name="Primary",
                confidence=Decimal("0.8"),
            ),
            alternatives=[
                HypothesisDTO(
                    hypothesis_id=uuid4(),
                    name="Alternative",
                    confidence=Decimal("0.5"),
                )
            ],
            missing_info=["duration"],
        )
        assert differential.primary is not None
        assert len(differential.alternatives) == 1

    def test_assessment_request_defaults(self) -> None:
        """Test AssessmentRequest default values."""
        request = AssessmentRequest(
            user_id=uuid4(),
            session_id=uuid4(),
            message="Test message",
        )
        assert request.current_phase == DiagnosisPhase.RAPPORT
        assert request.conversation_history == []

    def test_diagnosis_phase_enum(self) -> None:
        """Test DiagnosisPhase enum values."""
        assert DiagnosisPhase.RAPPORT.value == "rapport"
        assert DiagnosisPhase.HISTORY.value == "history"
        assert DiagnosisPhase.ASSESSMENT.value == "assessment"
        assert DiagnosisPhase.DIAGNOSIS.value == "diagnosis"
        assert DiagnosisPhase.CLOSURE.value == "closure"

    def test_reasoning_step_enum(self) -> None:
        """Test ReasoningStep enum values."""
        assert ReasoningStep.ANALYZE.value == "analyze"
        assert ReasoningStep.HYPOTHESIZE.value == "hypothesize"
        assert ReasoningStep.CHALLENGE.value == "challenge"
        assert ReasoningStep.SYNTHESIZE.value == "synthesize"
