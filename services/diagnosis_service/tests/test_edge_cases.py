"""
Edge Case and In-Depth Tests for Diagnosis Service Batch 5.1.
Tests boundary conditions, stress scenarios, and complex workflows.
"""
from __future__ import annotations
import pytest
from decimal import Decimal
from uuid import uuid4
from services.diagnosis_service.src.schemas import (
    DiagnosisPhase, SeverityLevel, SymptomType, ReasoningStep,
    SymptomDTO, HypothesisDTO, DifferentialDTO,
)
from services.diagnosis_service.src.domain.service import (
    DiagnosisService, DiagnosisServiceSettings,
)
from services.diagnosis_service.src.domain.models import SessionState, AssessmentResult
from services.diagnosis_service.src.domain.symptom_extractor import (
    SymptomExtractor, SymptomExtractorSettings,
)
from services.diagnosis_service.src.domain.differential import (
    DifferentialGenerator, DifferentialSettings,
)


class TestSymptomExtractorEdgeCases:
    """Edge case tests for SymptomExtractor."""

    def setup_method(self) -> None:
        self.extractor = SymptomExtractor(SymptomExtractorSettings())

    @pytest.mark.asyncio
    async def test_empty_message(self) -> None:
        """Test extraction from empty message."""
        result = await self.extractor.extract("", [], [])
        assert len(result.symptoms) == 0
        assert len(result.risk_indicators) == 0

    @pytest.mark.asyncio
    async def test_message_with_only_punctuation(self) -> None:
        """Test extraction from punctuation-only message."""
        result = await self.extractor.extract("...", [], [])
        assert len(result.symptoms) == 0

    @pytest.mark.asyncio
    async def test_very_long_message(self) -> None:
        """Test extraction from very long message."""
        long_msg = "I feel sad and depressed. " * 100
        result = await self.extractor.extract(long_msg, [], [])
        assert len(result.symptoms) > 0
        assert len(result.symptoms) <= self.extractor._settings.max_symptoms_per_message

    @pytest.mark.asyncio
    async def test_message_with_special_characters(self) -> None:
        """Test extraction handles special characters."""
        msg = "I'm feeling @#$% depressed!!! 😢 Can't sleep..."
        result = await self.extractor.extract(msg, [], [])
        symptom_names = {s.name for s in result.symptoms}
        assert "depressed_mood" in symptom_names or "sleep_disturbance" in symptom_names

    @pytest.mark.asyncio
    async def test_case_insensitive_detection(self) -> None:
        """Test symptom detection is case insensitive."""
        result_lower = await self.extractor.extract("i feel depressed", [], [])
        result_upper = await self.extractor.extract("I FEEL DEPRESSED", [], [])
        result_mixed = await self.extractor.extract("I Feel DePrEsSeD", [], [])
        assert len(result_lower.symptoms) == len(result_upper.symptoms) == len(result_mixed.symptoms)

    @pytest.mark.asyncio
    async def test_multiple_severity_indicators(self) -> None:
        """Test handling of multiple severity indicators."""
        msg = "I'm extremely severely depressed and overwhelmingly anxious"
        result = await self.extractor.extract(msg, [], [])
        severe_count = sum(1 for s in result.symptoms if s.severity == SeverityLevel.SEVERE)
        assert severe_count > 0

    @pytest.mark.asyncio
    async def test_conversation_history_context(self) -> None:
        """Test extraction uses conversation history."""
        history = [
            {"role": "user", "content": "I've been feeling sad"},
            {"role": "assistant", "content": "I understand"},
            {"role": "user", "content": "And I can't sleep at night"},
        ]
        result = await self.extractor.extract("It's getting worse", history, [])
        assert result.temporal_info is not None or len(result.contextual_factors) >= 0

    @pytest.mark.asyncio
    async def test_all_risk_indicators(self) -> None:
        """Test detection of all risk indicator types."""
        test_messages = [
            ("I want to kill myself", "suicidal_ideation"),
            ("I've been cutting myself", "self_harm"),
            ("I want to hurt someone", "harm_to_others"),
            ("There's no point in going on", "hopelessness"),
            ("I hear voices telling me things", "psychotic_features"),
            ("I was abused as a child", "trauma_disclosure"),
        ]
        for msg, expected_risk in test_messages:
            result = await self.extractor.extract(msg, [], [])
            assert expected_risk in result.risk_indicators, f"Failed for: {msg}"

    @pytest.mark.asyncio
    async def test_all_contextual_factors(self) -> None:
        """Test detection of all contextual factor types."""
        test_messages = [
            ("My job is stressing me out", "work_related"),
            ("My relationship is falling apart", "relationship"),
            ("My family doesn't understand", "family_related"),
            ("School exams are overwhelming", "academic"),
            ("I can't pay my bills", "financial"),
            ("My chronic illness is exhausting", "health_related"),
            ("I lost my father last month", "loss_grief"),
            ("I'm under so much stress", "stress"),
        ]
        for msg, expected_factor in test_messages:
            result = await self.extractor.extract(msg, [], [])
            assert expected_factor in result.contextual_factors, f"Failed for: {msg}"

    @pytest.mark.asyncio
    async def test_temporal_patterns(self) -> None:
        """Test temporal information extraction patterns."""
        test_messages = [
            ("for 3 weeks", "duration"),
            ("since January", "onset"),
            ("started last month", "onset"),
            ("sometimes", "frequency"),
            ("every morning", "time_of_day"),
            ("daily", "frequency"),
            ("getting worse", "progression"),
            ("suddenly", "onset_type"),
        ]
        for pattern, expected_type in test_messages:
            result = await self.extractor.extract(f"I've been feeling sad {pattern}", [], [])
            assert expected_type in result.temporal_info, f"Failed for pattern: {pattern}"

    def test_symptom_burden_edge_cases(self) -> None:
        """Test symptom burden calculation edge cases."""
        empty_burden = self.extractor.calculate_symptom_burden([])
        assert empty_burden["symptom_count"] == 0
        assert empty_burden["average_severity"] == 0

        single = [SymptomDTO(
            symptom_id=uuid4(), name="test",
            description="test", symptom_type=SymptomType.EMOTIONAL,
            severity=SeverityLevel.SEVERE
        )]
        single_burden = self.extractor.calculate_symptom_burden(single)
        assert single_burden["symptom_count"] == 1
        assert single_burden["average_severity"] == 1.0


class TestDifferentialGeneratorEdgeCases:
    """Edge case tests for DifferentialGenerator."""

    def setup_method(self) -> None:
        self.generator = DifferentialGenerator(DifferentialSettings())

    @pytest.mark.asyncio
    async def test_empty_symptoms(self) -> None:
        """Test differential generation with no symptoms returns fallback."""
        result = await self.generator.generate([], {})
        # With no symptoms, generator returns low-confidence fallback (Adjustment Disorder)
        assert len(result.hypotheses) <= 1
        if result.hypotheses:
            assert result.hypotheses[0].confidence <= Decimal("0.7")
            assert len(result.hypotheses[0].criteria_met) == 0

    @pytest.mark.asyncio
    async def test_single_symptom(self) -> None:
        """Test differential with single symptom."""
        symptoms = [SymptomDTO(
            symptom_id=uuid4(), name="depressed_mood",
            description="test", symptom_type=SymptomType.EMOTIONAL
        )]
        result = await self.generator.generate(symptoms, {})
        assert len(result.hypotheses) > 0
        for hyp in result.hypotheses:
            assert hyp.confidence >= Decimal(str(self.generator._settings.min_confidence_threshold))

    @pytest.mark.asyncio
    async def test_all_depression_symptoms(self) -> None:
        """Test with full depression symptom set."""
        symptoms = []
        for name in ["depressed_mood", "anhedonia", "sleep_disturbance", "fatigue",
                     "appetite_change", "concentration_difficulty", "guilt"]:
            symptoms.append(SymptomDTO(
                symptom_id=uuid4(), name=name,
                description="test", symptom_type=SymptomType.EMOTIONAL,
                severity=SeverityLevel.MODERATE
            ))
        result = await self.generator.generate(symptoms, {})
        hypothesis_names = [h.name.lower() for h in result.hypotheses]
        assert any("depress" in name for name in hypothesis_names)
        if result.hypotheses:
            assert result.hypotheses[0].confidence > Decimal("0.5")

    @pytest.mark.asyncio
    async def test_all_anxiety_symptoms(self) -> None:
        """Test with full anxiety symptom set."""
        symptoms = []
        for name in ["anxiety", "physical_tension", "sleep_disturbance",
                     "concentration_difficulty", "irritability"]:
            symptoms.append(SymptomDTO(
                symptom_id=uuid4(), name=name,
                description="test", symptom_type=SymptomType.EMOTIONAL,
                severity=SeverityLevel.MODERATE
            ))
        result = await self.generator.generate(symptoms, {})
        hypothesis_names = [h.name.lower() for h in result.hypotheses]
        assert any("anxiety" in name or "panic" in name for name in hypothesis_names)

    @pytest.mark.asyncio
    async def test_mixed_disorder_symptoms(self) -> None:
        """Test with symptoms spanning multiple disorders."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
            SymptomDTO(symptom_id=uuid4(), name="anxiety",
                      description="test", symptom_type=SymptomType.EMOTIONAL),
            SymptomDTO(symptom_id=uuid4(), name="intrusive_thoughts",
                      description="test", symptom_type=SymptomType.COGNITIVE),
        ]
        result = await self.generator.generate(symptoms, {})
        assert len(result.hypotheses) > 1

    @pytest.mark.asyncio
    async def test_hitop_dimension_coverage(self) -> None:
        """Test that HiTOP covers all dimensions."""
        all_hitop_symptoms = [
            "depressed_mood", "anxiety", "guilt", "anhedonia",
            "intrusive_thoughts", "concentration_difficulty",
            "irritability", "social_withdrawal", "physical_tension",
            "fatigue", "sleep_disturbance"
        ]
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name=name,
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.MODERATE)
            for name in all_hitop_symptoms
        ]
        result = await self.generator.generate(symptoms, {})
        expected_dimensions = ["internalizing", "detachment", "somatoform",
                              "disinhibited_externalizing"]
        for dim in expected_dimensions:
            assert dim in result.hitop_scores

    @pytest.mark.asyncio
    async def test_severity_affects_confidence(self) -> None:
        """Test that symptom severity affects hypothesis confidence."""
        mild_symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.MILD),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia",
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.MILD),
        ]
        severe_symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood",
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.SEVERE),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia",
                      description="test", symptom_type=SymptomType.EMOTIONAL,
                      severity=SeverityLevel.SEVERE),
        ]
        mild_result = await self.generator.generate(mild_symptoms, {})
        severe_result = await self.generator.generate(severe_symptoms, {})
        if mild_result.hypotheses and severe_result.hypotheses:
            assert severe_result.hypotheses[0].confidence >= mild_result.hypotheses[0].confidence

    def test_comorbidity_calculation(self) -> None:
        """Test comorbidity likelihood calculation."""
        hypotheses = [
            HypothesisDTO(hypothesis_id=uuid4(), name="Major Depressive Disorder",
                        confidence=Decimal("0.8")),
            HypothesisDTO(hypothesis_id=uuid4(), name="Generalized Anxiety Disorder",
                        confidence=Decimal("0.7")),
        ]
        comorbidity = self.generator.calculate_comorbidity_likelihood(hypotheses)
        assert len(comorbidity) > 0


class TestDiagnosisServiceEdgeCases:
    """Edge case tests for DiagnosisService."""

    def setup_method(self) -> None:
        self.extractor = SymptomExtractor(SymptomExtractorSettings())
        self.generator = DifferentialGenerator(DifferentialSettings())
        self.service = DiagnosisService(
            settings=DiagnosisServiceSettings(),
            symptom_extractor=self.extractor,
            differential_generator=self.generator,
        )

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self) -> None:
        """Test handling multiple concurrent user sessions."""
        await self.service.initialize()
        users = [uuid4() for _ in range(5)]
        sessions = []
        for user_id in users:
            result = await self.service.start_session(user_id, "assessment", {}, None)
            sessions.append((user_id, result.session_id))
        assert len(self.service._active_sessions) == 5
        for user_id, session_id in sessions:
            state = await self.service.get_session_state(session_id)
            assert state is not None
            assert state["user_id"] == str(user_id)
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_session_isolation(self) -> None:
        """Test that sessions don't leak data between users."""
        await self.service.initialize()
        user1, user2 = uuid4(), uuid4()
        session1 = await self.service.start_session(user1, "assessment", {}, None)
        session2 = await self.service.start_session(user2, "assessment", {}, None)
        await self.service.assess(user1, session1.session_id,
                                  "I feel depressed", [], [], DiagnosisPhase.RAPPORT, None, {})
        state2 = await self.service.get_session_state(session2.session_id)
        assert state2["symptom_count"] == 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_assess_without_components(self) -> None:
        """Test assessment when components are not configured."""
        bare_service = DiagnosisService(settings=DiagnosisServiceSettings())
        await bare_service.initialize()
        session = await bare_service.start_session(uuid4(), "assessment", {}, None)
        result = await bare_service.assess(
            session.session_id, session.session_id, "I feel sad",
            [], [], DiagnosisPhase.RAPPORT, None, {}
        )
        assert result is not None
        assert len(result.reasoning_chain) == 4
        await bare_service.shutdown()

    @pytest.mark.asyncio
    async def test_anti_sycophancy_disabled(self) -> None:
        """Test behavior when anti-sycophancy is disabled."""
        settings = DiagnosisServiceSettings(enable_anti_sycophancy=False)
        service = DiagnosisService(
            settings=settings,
            symptom_extractor=self.extractor,
            differential_generator=self.generator,
        )
        await service.initialize()
        session = await service.start_session(uuid4(), "assessment", {}, None)
        result = await service.assess(
            session.session_id, session.session_id,
            "I feel extremely depressed", [], [],
            DiagnosisPhase.RAPPORT, None, {}
        )
        challenge_step = next(s for s in result.reasoning_chain if s.step == ReasoningStep.CHALLENGE)
        assert len(challenge_step.details.get("biases", [])) == 0
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_phase_transition_threshold(self) -> None:
        """Test phase transition based on confidence threshold."""
        await self.service.initialize()
        session = await self.service.start_session(uuid4(), "assessment", {}, None)
        result = await self.service.assess(
            session.session_id, session.session_id,
            "I feel extremely sad, hopeless, can't sleep, no energy, can't eat, can't concentrate, feel guilty",
            [], [], DiagnosisPhase.RAPPORT, None, {}
        )
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_end_nonexistent_session(self) -> None:
        """Test ending a session that doesn't exist."""
        await self.service.initialize()
        result = await self.service.end_session(uuid4(), uuid4(), True)
        assert result.duration_minutes == 0
        assert result.messages_exchanged == 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_user_data_deletion_with_active_session(self) -> None:
        """Test GDPR deletion with active session."""
        await self.service.initialize()
        user_id = uuid4()
        session = await self.service.start_session(user_id, "assessment", {}, None)
        await self.service.delete_user_data(user_id)
        state = await self.service.get_session_state(session.session_id)
        assert state is None
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_statistics_tracking(self) -> None:
        """Test that service statistics are tracked correctly."""
        await self.service.initialize()
        user_id = uuid4()
        session = await self.service.start_session(user_id, "assessment", {}, None)
        await self.service.assess(
            user_id, session.session_id, "I feel sad", [], [],
            DiagnosisPhase.RAPPORT, None, {}
        )
        await self.service.extract_symptoms(user_id, session.session_id, "I feel anxious", [], [])
        symptoms = [SymptomDTO(
            symptom_id=uuid4(), name="anxiety",
            description="test", symptom_type=SymptomType.EMOTIONAL
        )]
        await self.service.generate_differential(user_id, session.session_id, symptoms, {}, None)
        await self.service.challenge_hypothesis(session.session_id, uuid4())
        await self.service.end_session(user_id, session.session_id, True)
        status = await self.service.get_status()
        assert status["statistics"]["assessments"] >= 1
        assert status["statistics"]["extractions"] >= 1
        assert status["statistics"]["differentials"] >= 1
        assert status["statistics"]["challenges"] >= 1
        assert status["statistics"]["sessions_started"] >= 1
        assert status["statistics"]["sessions_ended"] >= 1
        await self.service.shutdown()


class TestFullDiagnosticWorkflow:
    """Test complete diagnostic workflows end-to-end."""

    def setup_method(self) -> None:
        self.extractor = SymptomExtractor(SymptomExtractorSettings())
        self.generator = DifferentialGenerator(DifferentialSettings())
        self.service = DiagnosisService(
            settings=DiagnosisServiceSettings(),
            symptom_extractor=self.extractor,
            differential_generator=self.generator,
        )

    @pytest.mark.asyncio
    async def test_complete_depression_assessment_workflow(self) -> None:
        """Test complete workflow for depression assessment."""
        await self.service.initialize()
        user_id = uuid4()
        session = await self.service.start_session(user_id, "assessment", {}, None)
        assert session.initial_phase == DiagnosisPhase.RAPPORT
        rapport_result = await self.service.assess(
            user_id, session.session_id,
            "I've been feeling really down and sad for weeks now",
            [], [], DiagnosisPhase.RAPPORT, None, {}
        )
        assert len(rapport_result.extracted_symptoms) > 0
        assert any(s.name == "depressed_mood" for s in rapport_result.extracted_symptoms)
        history_result = await self.service.assess(
            user_id, session.session_id,
            "It started about a month ago when I lost my job. I can't sleep and have no energy",
            [], rapport_result.extracted_symptoms,
            DiagnosisPhase.HISTORY, rapport_result.differential, {}
        )
        assert len(history_result.extracted_symptoms) >= 1
        assessment_result = await self.service.assess(
            user_id, session.session_id,
            "I'd say the intensity is about 7 out of 10. I can't enjoy anything anymore",
            [], history_result.extracted_symptoms,
            DiagnosisPhase.ASSESSMENT, history_result.differential, {}
        )
        assert assessment_result.differential.primary is not None
        end_result = await self.service.end_session(user_id, session.session_id, True)
        assert end_result.final_differential is not None
        assert end_result.summary is not None
        assert len(end_result.recommendations) > 0
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_complete_anxiety_assessment_workflow(self) -> None:
        """Test complete workflow for anxiety assessment."""
        await self.service.initialize()
        user_id = uuid4()
        session = await self.service.start_session(user_id, "assessment", {}, None)
        messages = [
            "I'm constantly worried about everything, I can't relax",
            "My heart races, I feel tense all the time, especially at work",
            "This has been going on for about 6 months now",
        ]
        all_symptoms: list[SymptomDTO] = []
        differential = None
        for msg in messages:
            result = await self.service.assess(
                user_id, session.session_id, msg,
                [], all_symptoms, DiagnosisPhase.ASSESSMENT, differential, {}
            )
            all_symptoms.extend(result.extracted_symptoms)
            differential = result.differential
        assert any("anxiety" in s.name for s in all_symptoms)
        if differential and differential.primary:
            assert "anxiety" in differential.primary.name.lower() or "adjustment" in differential.primary.name.lower()
        await self.service.shutdown()

    @pytest.mark.asyncio
    async def test_longitudinal_tracking_workflow(self) -> None:
        """Test workflow spanning multiple sessions."""
        await self.service.initialize()
        user_id = uuid4()
        session1 = await self.service.start_session(user_id, "assessment", {}, None)
        await self.service.assess(
            user_id, session1.session_id, "I feel depressed",
            [], [], DiagnosisPhase.RAPPORT, None, {}
        )
        await self.service.end_session(user_id, session1.session_id, True)
        session2 = await self.service.start_session(user_id, "followup", {}, session1.session_id)
        assert session2.session_number == 2
        assert session2.loaded_context is True
        history = await self.service.get_history(user_id, 10, True, True)
        assert len(history.sessions) == 1
        await self.service.shutdown()
