"""
Comprehensive Unit Tests for Diagnosis Service Batch 5.2.
Tests advocate, confidence, clinical_codes, severity, and evidence components.
"""
from __future__ import annotations
import pytest
from decimal import Decimal
from uuid import uuid4
from services.diagnosis_service.src.schemas import (
    HypothesisDTO, SymptomDTO, SeverityLevel, SymptomType, ConfidenceLevel,
)
from services.diagnosis_service.src.domain.advocate import (
    DevilsAdvocate, AdvocateSettings, ChallengeResult, BiasAnalysisResult,
)
from services.diagnosis_service.src.domain.confidence import (
    ConfidenceCalibrator, ConfidenceSettings, CalibrationResult, ConsistencyResult,
)
from services.diagnosis_service.src.domain.clinical_codes import (
    ClinicalCodeMapper, ClinicalCodesSettings, ClinicalCode, CodeLookupResult, ValidationResult,
)
from services.diagnosis_service.src.domain.severity import (
    SeverityAssessor, SeveritySettings, QuestionnaireResult, SeverityAssessmentResult,
)
from services.diagnosis_service.src.domain.evidence import (
    EvidenceEvaluator, EvidenceSettings, EvidenceItem, EvidenceEvaluationResult, EvidenceSummary,
)


class TestDevilsAdvocate:
    """Tests for DevilsAdvocate challenger."""

    def setup_method(self) -> None:
        self.advocate = DevilsAdvocate(AdvocateSettings())

    @pytest.mark.asyncio
    async def test_challenge_depression_hypothesis(self) -> None:
        """Test challenging a depression hypothesis."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Major Depressive Disorder",
            confidence=Decimal("0.8"), dsm5_code="F32",
            criteria_met=["depressed_mood", "anhedonia", "fatigue"],
            criteria_missing=["sleep_disturbance"],
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="Low mood",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
        ]
        result = await self.advocate.challenge_hypothesis(hypothesis, symptoms, {})
        assert isinstance(result, ChallengeResult)
        assert result.hypothesis_id == hypothesis.hypothesis_id
        assert len(result.challenges) >= 2
        assert result.confidence_adjustment <= 0

    @pytest.mark.asyncio
    async def test_challenge_anxiety_hypothesis(self) -> None:
        """Test challenging an anxiety hypothesis."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Generalized Anxiety Disorder",
            confidence=Decimal("0.75"), dsm5_code="F41.1",
            criteria_met=["anxiety", "worry"], criteria_missing=["restlessness"],
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="anxiety", description="Excessive worry",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
        ]
        result = await self.advocate.challenge_hypothesis(hypothesis, symptoms, {})
        assert len(result.challenges) >= 2
        assert "anxiety" in result.challenges[0].lower() or "stress" in result.challenges[0].lower()

    @pytest.mark.asyncio
    async def test_generate_alternatives(self) -> None:
        """Test alternative hypothesis generation."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Major Depressive Disorder",
            confidence=Decimal("0.7"), criteria_met=["depressed_mood"],
        )
        result = await self.advocate.challenge_hypothesis(hypothesis, [], {})
        assert len(result.alternative_explanations) > 0

    @pytest.mark.asyncio
    async def test_bias_analysis_premature_closure(self) -> None:
        """Test bias detection for premature closure."""
        hypotheses = [HypothesisDTO(
            hypothesis_id=uuid4(), name="MDD", confidence=Decimal("0.85"),
            criteria_met=["depressed_mood"], criteria_missing=["anhedonia", "fatigue", "sleep"],
        )]
        result = await self.advocate.analyze_bias(hypotheses, [], [])
        assert isinstance(result, BiasAnalysisResult)
        assert "premature_closure" in result.detected_biases

    @pytest.mark.asyncio
    async def test_bias_analysis_confirmation_bias(self) -> None:
        """Test confirmation bias detection."""
        hypotheses = [HypothesisDTO(
            hypothesis_id=uuid4(), name="Depression", confidence=Decimal("0.9"),
            criteria_met=["depressed_mood"], criteria_missing=["a", "b", "c"],
        )]
        result = await self.advocate.analyze_bias(hypotheses, [], [])
        assert "confirmation_bias" in result.detected_biases

    @pytest.mark.asyncio
    async def test_counter_arguments(self) -> None:
        """Test counter-argument generation."""
        hypothesis = HypothesisDTO(hypothesis_id=uuid4(), name="Depression", confidence=Decimal("0.7"))
        evidence = ["Patient reports feeling sad", "Sleep disturbance noted"]
        counters = await self.advocate.generate_counter_arguments(hypothesis, evidence)
        assert len(counters) > 0

    def test_get_bias_description(self) -> None:
        """Test bias description retrieval."""
        desc = self.advocate.get_bias_description("confirmation_bias")
        assert desc is not None
        assert "confirm" in desc.lower()

    def test_statistics(self) -> None:
        """Test statistics tracking."""
        stats = self.advocate.get_statistics()
        assert "challenges_generated" in stats
        assert "biases_detected" in stats


class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator."""

    def setup_method(self) -> None:
        self.calibrator = ConfidenceCalibrator(ConfidenceSettings())

    @pytest.mark.asyncio
    async def test_calibrate_hypothesis(self) -> None:
        """Test confidence calibration."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Major Depressive Disorder",
            confidence=Decimal("0.8"),
            criteria_met=["depressed_mood", "anhedonia", "fatigue"],
            criteria_missing=["sleep_disturbance"],
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
        ]
        result = await self.calibrator.calibrate(hypothesis, symptoms, {})
        assert isinstance(result, CalibrationResult)
        assert result.calibrated_confidence >= Decimal("0")
        assert result.calibrated_confidence <= Decimal("1")
        assert result.confidence_interval[0] < result.confidence_interval[1]

    @pytest.mark.asyncio
    async def test_calibrate_low_evidence(self) -> None:
        """Test calibration with low evidence."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Panic Disorder",
            confidence=Decimal("0.9"),
            criteria_met=[], criteria_missing=["panic", "anxiety"],
        )
        result = await self.calibrator.calibrate(hypothesis, [], {})
        assert result.calibrated_confidence < Decimal("0.9")

    @pytest.mark.asyncio
    async def test_consistency_analysis(self) -> None:
        """Test consistency analysis."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="GAD", confidence=Decimal("0.7"),
            criteria_met=["anxiety"], criteria_missing=[],
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="anxiety", description="test",
                      symptom_type=SymptomType.EMOTIONAL, duration="3 months"),
        ]
        result = await self.calibrator.analyze_consistency(hypothesis, symptoms)
        assert isinstance(result, ConsistencyResult)
        assert 0 <= result.consistency_score <= 1

    @pytest.mark.asyncio
    async def test_calibrate_multiple(self) -> None:
        """Test calibrating multiple hypotheses."""
        hypotheses = [
            HypothesisDTO(hypothesis_id=uuid4(), name="MDD", confidence=Decimal("0.7"),
                         criteria_met=["depressed_mood"]),
            HypothesisDTO(hypothesis_id=uuid4(), name="GAD", confidence=Decimal("0.6"),
                         criteria_met=["anxiety"]),
        ]
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="test",
                      symptom_type=SymptomType.EMOTIONAL),
        ]
        results = await self.calibrator.calibrate_multiple(hypotheses, symptoms, {})
        assert len(results) == 2

    def test_confidence_level_determination(self) -> None:
        """Test confidence level categorical determination."""
        hypothesis = HypothesisDTO(hypothesis_id=uuid4(), name="Test", confidence=Decimal("0.85"))

    def test_base_rate_lookup(self) -> None:
        """Test base rate retrieval."""
        rate = self.calibrator.get_base_rate("major_depressive_disorder")
        assert rate is not None
        assert 0 < rate < 1

    def test_statistics(self) -> None:
        """Test statistics tracking."""
        stats = self.calibrator.get_statistics()
        assert "calibrations" in stats


class TestClinicalCodeMapper:
    """Tests for ClinicalCodeMapper."""

    def setup_method(self) -> None:
        self.mapper = ClinicalCodeMapper(ClinicalCodesSettings())

    def test_lookup_dsm5_code(self) -> None:
        """Test DSM-5 code lookup."""
        result = self.mapper.lookup("F32")
        assert result.found is True
        assert result.code is not None
        assert result.code.system == "DSM-5-TR"
        assert "depressive" in result.code.name.lower()

    def test_lookup_icd11_code(self) -> None:
        """Test ICD-11 code lookup."""
        result = self.mapper.lookup("6A70")
        assert result.found is True
        assert result.code is not None
        assert result.code.system == "ICD-11"

    def test_lookup_nonexistent_code(self) -> None:
        """Test lookup of non-existent code."""
        result = self.mapper.lookup("INVALID")
        assert result.found is False
        assert result.code is None

    def test_validate_dsm5_code(self) -> None:
        """Test DSM-5 code validation."""
        result = self.mapper.validate("F41.1", "DSM-5-TR")
        assert result.valid is True
        assert result.system == "DSM-5-TR"

    def test_validate_invalid_code(self) -> None:
        """Test validation of invalid code."""
        result = self.mapper.validate("INVALID", "DSM-5")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_crosswalk_dsm5_to_icd11(self) -> None:
        """Test crosswalk from DSM-5 to ICD-11."""
        result = self.mapper.crosswalk_code("F32", "ICD-11")
        assert result.found is True
        assert result.code is not None
        assert result.code.code == "6A70"

    def test_crosswalk_icd11_to_dsm5(self) -> None:
        """Test crosswalk from ICD-11 to DSM-5."""
        result = self.mapper.crosswalk_code("6B00", "DSM-5")
        assert result.found is True
        assert result.code.code == "F41.1"

    def test_get_codes_by_category(self) -> None:
        """Test getting codes by category."""
        codes = self.mapper.get_codes_by_category("depressive")
        assert len(codes) > 0
        assert all("depress" in c.category.lower() for c in codes)

    def test_get_severity_specifiers(self) -> None:
        """Test getting severity specifiers."""
        specifiers = self.mapper.get_severity_specifiers("F32")
        assert len(specifiers) > 0
        assert "mild" in specifiers

    def test_get_related_codes(self) -> None:
        """Test getting related codes."""
        related = self.mapper.get_related_codes("F32")
        assert len(related) > 0
        assert "F33" in related

    def test_statistics(self) -> None:
        """Test statistics tracking."""
        self.mapper.lookup("F32")
        stats = self.mapper.get_statistics()
        assert stats["lookups"] >= 1


class TestSeverityAssessor:
    """Tests for SeverityAssessor."""

    def setup_method(self) -> None:
        self.assessor = SeverityAssessor(SeveritySettings())

    @pytest.mark.asyncio
    async def test_assess_from_symptoms(self) -> None:
        """Test severity assessment from symptoms."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="fatigue", description="test",
                      symptom_type=SymptomType.SOMATIC, severity=SeverityLevel.MILD),
        ]
        result = await self.assessor.assess(symptoms)
        assert isinstance(result, SeverityAssessmentResult)
        assert result.depression_severity is not None

    @pytest.mark.asyncio
    async def test_phq9_scoring(self) -> None:
        """Test PHQ-9 scoring with explicit responses."""
        responses = {
            "phq9_1": 2, "phq9_2": 2, "phq9_3": 1,
            "phq9_4": 2, "phq9_5": 1, "phq9_6": 1,
            "phq9_7": 2, "phq9_8": 1, "phq9_9": 0,
        }
        result = await self.assessor.assess([], responses)
        assert result.depression_severity is not None
        assert result.depression_severity.total_score == 12
        assert result.depression_severity.severity_level == SeverityLevel.MODERATE

    @pytest.mark.asyncio
    async def test_gad7_scoring(self) -> None:
        """Test GAD-7 scoring."""
        responses = {
            "gad7_1": 2, "gad7_2": 2, "gad7_3": 2,
            "gad7_4": 1, "gad7_5": 1, "gad7_6": 2, "gad7_7": 1,
        }
        result = await self.assessor.assess([], responses)
        assert result.anxiety_severity is not None
        assert result.anxiety_severity.total_score == 11
        assert result.anxiety_severity.severity_level == SeverityLevel.MODERATE

    @pytest.mark.asyncio
    async def test_severe_depression_score(self) -> None:
        """Test severe depression scoring."""
        responses = {f"phq9_{i}": 3 for i in range(1, 10)}
        result = await self.assessor.assess([], responses)
        assert result.depression_severity.total_score == 27
        assert result.depression_severity.severity_level == SeverityLevel.SEVERE

    @pytest.mark.asyncio
    async def test_minimal_symptoms(self) -> None:
        """Test minimal symptom assessment."""
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="mild_worry", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MINIMAL),
        ]
        result = await self.assessor.assess(symptoms)
        assert result.overall_severity in [SeverityLevel.MINIMAL, SeverityLevel.MILD]

    @pytest.mark.asyncio
    async def test_composite_score(self) -> None:
        """Test composite score calculation."""
        responses = {
            "phq9_1": 2, "phq9_2": 2,
            "gad7_1": 2, "gad7_2": 2,
        }
        result = await self.assessor.assess([], responses)
        assert 0 <= result.composite_score <= 1

    @pytest.mark.asyncio
    async def test_functional_impairment(self) -> None:
        """Test functional impairment assessment."""
        responses = {f"phq9_{i}": 3 for i in range(1, 10)}
        result = await self.assessor.assess([], responses)
        assert result.functional_impairment == "severe"

    def test_statistics(self) -> None:
        """Test statistics tracking."""
        stats = self.assessor.get_statistics()
        assert "assessments" in stats
        assert "phq9_scored" in stats


class TestEvidenceEvaluator:
    """Tests for EvidenceEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = EvidenceEvaluator(EvidenceSettings())

    @pytest.mark.asyncio
    async def test_evaluate_depression_hypothesis(self) -> None:
        """Test evidence evaluation for depression."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Major Depressive Disorder",
            confidence=Decimal("0.7"),
            criteria_met=["depressed_mood", "anhedonia"],
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE,
                      duration="3 weeks"),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
        ]
        result = await self.evaluator.evaluate(hypothesis, symptoms, {})
        assert isinstance(result, EvidenceEvaluationResult)
        assert result.hypothesis_id == hypothesis.hypothesis_id
        assert len(result.supporting_evidence) > 0
        assert 0 <= result.total_evidence_score <= 1

    @pytest.mark.asyncio
    async def test_evaluate_with_contradicting_evidence(self) -> None:
        """Test evaluation with contradicting evidence."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Major Depressive Disorder",
            confidence=Decimal("0.7"),
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="manic_episode", description="test",
                      symptom_type=SymptomType.BEHAVIORAL, severity=SeverityLevel.MODERATE),
        ]
        result = await self.evaluator.evaluate(hypothesis, symptoms, {"manic_episode": True})
        assert len(result.contradicting_evidence) > 0

    @pytest.mark.asyncio
    async def test_missing_evidence_identification(self) -> None:
        """Test missing evidence identification."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Generalized Anxiety Disorder",
            confidence=Decimal("0.6"),
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="worry", description="test",
                      symptom_type=SymptomType.COGNITIVE),
        ]
        result = await self.evaluator.evaluate(hypothesis, symptoms, {})
        assert len(result.missing_evidence) > 0

    @pytest.mark.asyncio
    async def test_contextual_evidence(self) -> None:
        """Test contextual evidence gathering."""
        hypothesis = HypothesisDTO(hypothesis_id=uuid4(), name="MDD", confidence=Decimal("0.7"))
        context = {"family_history": True, "previous_episodes": True}
        result = await self.evaluator.evaluate(hypothesis, [], context)
        contextual = [e for e in result.supporting_evidence if e.category == "contextual"]
        assert len(contextual) > 0

    @pytest.mark.asyncio
    async def test_evidence_quality_assessment(self) -> None:
        """Test evidence quality assessment."""
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Major Depressive Disorder",
            confidence=Decimal("0.8"),
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.SEVERE,
                      duration="4 weeks"),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.SEVERE),
            SymptomDTO(symptom_id=uuid4(), name="fatigue", description="test",
                      symptom_type=SymptomType.SOMATIC, severity=SeverityLevel.MODERATE),
        ]
        result = await self.evaluator.evaluate(hypothesis, symptoms, {})
        assert result.evidence_quality in ["high", "moderate", "low", "insufficient"]

    @pytest.mark.asyncio
    async def test_evaluate_multiple_hypotheses(self) -> None:
        """Test evaluating multiple hypotheses."""
        hypotheses = [
            HypothesisDTO(hypothesis_id=uuid4(), name="MDD", confidence=Decimal("0.7")),
            HypothesisDTO(hypothesis_id=uuid4(), name="GAD", confidence=Decimal("0.6")),
        ]
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="test",
                      symptom_type=SymptomType.EMOTIONAL),
        ]
        results = await self.evaluator.evaluate_multiple(hypotheses, symptoms, {})
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_evidence_summary(self) -> None:
        """Test evidence summary generation."""
        hypotheses = [
            HypothesisDTO(hypothesis_id=uuid4(), name="MDD", confidence=Decimal("0.7")),
        ]
        evaluations = [EvidenceEvaluationResult(
            supporting_evidence=[EvidenceItem(description="test", strength=0.7)],
            missing_evidence=["duration"],
            evidence_quality="moderate",
        )]
        summary = await self.evaluator.summarize_evidence(hypotheses, evaluations)
        assert isinstance(summary, EvidenceSummary)
        assert summary.hypotheses_evaluated == 1

    def test_statistics(self) -> None:
        """Test statistics tracking."""
        stats = self.evaluator.get_statistics()
        assert "evaluations" in stats
        assert "evidence_items_processed" in stats


class TestIntegrationScenarios:
    """Integration tests combining multiple Batch 5.2 components."""

    @pytest.mark.asyncio
    async def test_full_anti_sycophancy_workflow(self) -> None:
        """Test complete anti-sycophancy workflow."""
        advocate = DevilsAdvocate()
        calibrator = ConfidenceCalibrator()
        evaluator = EvidenceEvaluator()
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Major Depressive Disorder",
            confidence=Decimal("0.85"), dsm5_code="F32",
            criteria_met=["depressed_mood", "anhedonia", "fatigue"],
            criteria_missing=["sleep_disturbance"],
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="Low mood",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="anhedonia", description="Loss of interest",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
        ]
        challenge_result = await advocate.challenge_hypothesis(hypothesis, symptoms, {})
        assert len(challenge_result.challenges) >= 2
        calibration_result = await calibrator.calibrate(hypothesis, symptoms, {})
        assert calibration_result.calibrated_confidence <= hypothesis.confidence
        evidence_result = await evaluator.evaluate(hypothesis, symptoms, {})
        assert len(evidence_result.supporting_evidence) > 0
        bias_result = await advocate.analyze_bias([hypothesis], symptoms, [])
        assert isinstance(bias_result, BiasAnalysisResult)

    @pytest.mark.asyncio
    async def test_severity_with_clinical_codes(self) -> None:
        """Test severity assessment with clinical code validation."""
        assessor = SeverityAssessor()
        mapper = ClinicalCodeMapper()
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="depressed_mood", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
            SymptomDTO(symptom_id=uuid4(), name="anxiety", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MODERATE),
        ]
        severity_result = await assessor.assess(symptoms)
        depression_code = mapper.lookup("F32")
        anxiety_code = mapper.lookup("F41.1")
        assert depression_code.found is True
        assert anxiety_code.found is True
        assert severity_result.depression_severity is not None
        assert severity_result.anxiety_severity is not None

    @pytest.mark.asyncio
    async def test_evidence_confidence_integration(self) -> None:
        """Test evidence evaluation affecting confidence calibration."""
        evaluator = EvidenceEvaluator()
        calibrator = ConfidenceCalibrator()
        hypothesis = HypothesisDTO(
            hypothesis_id=uuid4(), name="Generalized Anxiety Disorder",
            confidence=Decimal("0.9"),
            criteria_met=["anxiety"],
            criteria_missing=["worry", "restlessness", "fatigue"],
        )
        symptoms = [
            SymptomDTO(symptom_id=uuid4(), name="anxiety", description="test",
                      symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MILD),
        ]
        evidence_result = await evaluator.evaluate(hypothesis, symptoms, {})
        calibration_result = await calibrator.calibrate(hypothesis, symptoms, {})
        assert evidence_result.evidence_quality in ["low", "insufficient"]
        assert calibration_result.calibrated_confidence < hypothesis.confidence
