"""
Solace-AI Diagnosis Service - Devil's Advocate Challenger.
Implements anti-sycophancy mechanisms to prevent diagnostic confirmation bias.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import HypothesisDTO, SymptomDTO, SeverityLevel, SymptomType

logger = structlog.get_logger(__name__)


class AdvocateSettings(BaseSettings):
    """Devil's Advocate configuration."""
    challenge_threshold: float = Field(default=0.6)
    min_challenges_per_hypothesis: int = Field(default=2)
    max_challenges_per_hypothesis: int = Field(default=5)
    counter_hypothesis_threshold: float = Field(default=0.3)
    bias_detection_sensitivity: float = Field(default=0.7)
    enable_alternative_generation: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="ADVOCATE_", env_file=".env", extra="ignore")


@dataclass
class ChallengeResult:
    """Result from hypothesis challenge."""
    challenge_id: UUID = field(default_factory=uuid4)
    hypothesis_id: UUID = field(default_factory=uuid4)
    challenges: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)
    bias_flags: list[str] = field(default_factory=list)
    confidence_adjustment: Decimal = Decimal("0.0")
    counter_questions: list[str] = field(default_factory=list)


@dataclass
class BiasAnalysisResult:
    """Result from cognitive bias analysis."""
    analysis_id: UUID = field(default_factory=uuid4)
    detected_biases: list[str] = field(default_factory=list)
    bias_scores: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    overall_risk: str = "low"


class DevilsAdvocate:
    """Challenges diagnostic hypotheses to prevent confirmation bias."""

    def __init__(self, settings: AdvocateSettings | None = None) -> None:
        self._settings = settings or AdvocateSettings()
        self._challenge_templates = self._build_challenge_templates()
        self._bias_patterns = self._build_bias_patterns()
        self._alternative_mappings = self._build_alternative_mappings()
        self._counter_questions = self._build_counter_questions()
        self._stats = {"challenges_generated": 0, "biases_detected": 0, "alternatives_proposed": 0}

    def _build_challenge_templates(self) -> dict[str, list[str]]:
        """Build challenge templates for different diagnosis categories."""
        return {
            "depression": [
                "Consider whether symptoms could reflect adjustment to life stressors rather than MDD",
                "Evaluate if anhedonia represents social withdrawal from anxiety versus depression",
                "Assess whether fatigue could have medical origins (thyroid, anemia, sleep apnea)",
                "Question if concentration issues stem from anxiety rather than depression",
                "Consider substance use as potential cause of depressive symptoms",
            ],
            "anxiety": [
                "Evaluate whether anxiety symptoms could represent normal stress response",
                "Consider if physical symptoms suggest medical condition (cardiac, endocrine)",
                "Assess whether fear response is proportional to actual threat level",
                "Question if avoidance behavior preceded or followed anxiety symptoms",
                "Consider caffeine, medication side effects, or withdrawal as causes",
            ],
            "panic": [
                "Assess whether panic attacks could be symptoms of medical condition",
                "Consider if anticipatory anxiety maintains panic cycle artificially",
                "Evaluate whether situational triggers suggest phobia rather than panic disorder",
                "Question accuracy of catastrophic interpretations of physical sensations",
                "Consider comorbid conditions that may better explain symptom pattern",
            ],
            "trauma": [
                "Evaluate if symptoms meet threshold for PTSD versus adjustment disorder",
                "Consider whether hypervigilance is adaptive given current circumstances",
                "Assess if avoidance targets trauma reminders specifically or is generalized",
                "Question whether intrusive symptoms are trauma-specific memories",
                "Consider cultural factors in trauma expression and coping",
            ],
            "general": [
                "What alternative diagnoses could explain this symptom pattern?",
                "What evidence would contradict this hypothesis?",
                "Are there symptoms that don't fit this diagnosis?",
                "Could functional impairment have other causes?",
                "Is the symptom timeline consistent with this diagnosis?",
            ],
        }

    def _build_bias_patterns(self) -> dict[str, dict[str, Any]]:
        """Build cognitive bias detection patterns."""
        return {
            "confirmation_bias": {
                "description": "Favoring information that confirms existing beliefs",
                "indicators": ["selective_evidence", "ignored_contradictions", "premature_closure"],
                "risk_weight": 0.9,
            },
            "anchoring_bias": {
                "description": "Over-relying on first piece of information encountered",
                "indicators": ["early_hypothesis_fixation", "insufficient_revision", "initial_symptom_weight"],
                "risk_weight": 0.8,
            },
            "availability_heuristic": {
                "description": "Overweighting easily recalled diagnoses",
                "indicators": ["common_diagnosis_preference", "recent_case_influence", "salient_feature_focus"],
                "risk_weight": 0.7,
            },
            "base_rate_neglect": {
                "description": "Ignoring prevalence rates in diagnosis",
                "indicators": ["rare_diagnosis_without_strong_evidence", "prevalence_ignored"],
                "risk_weight": 0.75,
            },
            "premature_closure": {
                "description": "Accepting diagnosis before fully evaluating alternatives",
                "indicators": ["insufficient_differential", "missing_rule_outs", "incomplete_history"],
                "risk_weight": 0.85,
            },
            "attribution_error": {
                "description": "Attributing symptoms to single cause when multiple exist",
                "indicators": ["single_diagnosis_preference", "comorbidity_underestimation"],
                "risk_weight": 0.6,
            },
        }

    def _build_alternative_mappings(self) -> dict[str, list[str]]:
        """Build alternative diagnosis mappings."""
        return {
            "major_depressive_disorder": [
                "adjustment_disorder_with_depressed_mood",
                "persistent_depressive_disorder",
                "bipolar_disorder_depressive_episode",
                "substance_induced_depressive_disorder",
                "medical_condition_induced_depression",
            ],
            "generalized_anxiety_disorder": [
                "adjustment_disorder_with_anxiety",
                "social_anxiety_disorder",
                "panic_disorder",
                "medical_condition_induced_anxiety",
                "substance_induced_anxiety",
            ],
            "panic_disorder": [
                "generalized_anxiety_disorder",
                "social_anxiety_disorder",
                "specific_phobia",
                "medical_condition_induced_panic",
                "substance_induced_anxiety",
            ],
            "ptsd": [
                "acute_stress_disorder",
                "adjustment_disorder",
                "generalized_anxiety_disorder",
                "major_depressive_disorder",
                "complex_ptsd",
            ],
            "social_anxiety_disorder": [
                "generalized_anxiety_disorder",
                "avoidant_personality_disorder",
                "autism_spectrum_disorder",
                "depression_with_social_withdrawal",
                "agoraphobia",
            ],
        }

    def _build_counter_questions(self) -> dict[str, list[str]]:
        """Build counter-diagnostic questions."""
        return {
            "symptom_onset": [
                "When exactly did these symptoms begin?",
                "What was happening in your life when this started?",
            ],
            "symptom_course": [
                "Have the symptoms been constant or do they fluctuate?",
                "Has anything made the symptoms better or worse?",
            ],
            "exclusion_criteria": [
                "Have you experienced any manic or hypomanic episodes?",
                "Are you taking any medications or substances?",
                "Have you had any recent medical tests?",
            ],
            "functional_impact": [
                "How are these symptoms affecting your daily functioning?",
                "Are you able to maintain your responsibilities?",
            ],
            "context_factors": [
                "Are there current life stressors that might explain these symptoms?",
                "How would you describe your support system?",
            ],
        }

    async def challenge_hypothesis(self, hypothesis: HypothesisDTO, symptoms: list[SymptomDTO],
                                    context: dict[str, Any]) -> ChallengeResult:
        """Generate challenges for a diagnostic hypothesis."""
        self._stats["challenges_generated"] += 1
        result = ChallengeResult(hypothesis_id=hypothesis.hypothesis_id)
        category = self._categorize_hypothesis(hypothesis.name)
        challenges = self._generate_challenges(category, hypothesis, symptoms)
        result.challenges = challenges[:self._settings.max_challenges_per_hypothesis]
        counter_evidence = self._find_counter_evidence(hypothesis, symptoms)
        result.counter_evidence = counter_evidence
        if self._settings.enable_alternative_generation:
            alternatives = self._generate_alternatives(hypothesis.name)
            result.alternative_explanations = alternatives
            self._stats["alternatives_proposed"] += len(alternatives)
        adjustment = self._calculate_confidence_adjustment(hypothesis, counter_evidence, symptoms)
        result.confidence_adjustment = Decimal(str(round(adjustment, 2)))
        result.counter_questions = self._generate_counter_questions(hypothesis, symptoms)
        logger.debug("hypothesis_challenged", hypothesis=hypothesis.name,
                    challenges=len(result.challenges), adjustment=float(result.confidence_adjustment))
        return result

    def _categorize_hypothesis(self, name: str) -> str:
        """Categorize hypothesis for challenge template selection."""
        name_lower = name.lower()
        if "depress" in name_lower:
            return "depression"
        if "anxiety" in name_lower or "gad" in name_lower:
            return "anxiety"
        if "panic" in name_lower:
            return "panic"
        if "ptsd" in name_lower or "trauma" in name_lower or "stress" in name_lower:
            return "trauma"
        return "general"

    def _generate_challenges(self, category: str, hypothesis: HypothesisDTO,
                              symptoms: list[SymptomDTO]) -> list[str]:
        """Generate specific challenges for hypothesis."""
        challenges: list[str] = []
        templates = self._challenge_templates.get(category, self._challenge_templates["general"])
        challenges.extend(templates[:self._settings.min_challenges_per_hypothesis])
        if hypothesis.criteria_missing:
            for missing in hypothesis.criteria_missing[:2]:
                challenges.append(f"Missing criterion '{missing}' - consider if diagnosis is appropriate")
        if hypothesis.confidence > Decimal("0.8"):
            challenges.append("High confidence diagnosis - ensure not overlooking alternatives")
        if len(symptoms) < 4:
            challenges.append("Limited symptom presentation - more information may be needed")
        return challenges

    def _find_counter_evidence(self, hypothesis: HypothesisDTO,
                                symptoms: list[SymptomDTO]) -> list[str]:
        """Find evidence that contradicts the hypothesis."""
        counter: list[str] = []
        name_lower = hypothesis.name.lower()
        symptom_names = {s.name.lower() for s in symptoms}
        if "depress" in name_lower:
            if "elevated_mood" in symptom_names or "euphoria" in symptom_names:
                counter.append("Presence of elevated mood suggests bipolar rather than unipolar depression")
            if "preserved_reactivity" in symptom_names:
                counter.append("Mood reactivity preserved - consider atypical features or adjustment")
        if "anxiety" in name_lower:
            if "relaxation" in symptom_names or "calm" in symptom_names:
                counter.append("Ability to relax suggests episodic rather than generalized anxiety")
        if "panic" in name_lower:
            if all(s.severity in [SeverityLevel.MINIMAL, SeverityLevel.MILD] for s in symptoms):
                counter.append("Mild severity across symptoms - panic typically presents with intense episodes")
        if hypothesis.criteria_missing and len(hypothesis.criteria_missing) > len(hypothesis.criteria_met):
            counter.append("More criteria missing than met - diagnosis may be premature")
        return counter

    def _generate_alternatives(self, hypothesis_name: str) -> list[str]:
        """Generate alternative diagnostic explanations."""
        key = hypothesis_name.lower().replace(" ", "_")
        alternatives = self._alternative_mappings.get(key, [])
        if not alternatives:
            for mapping_key in self._alternative_mappings:
                if mapping_key in key or key in mapping_key:
                    alternatives = self._alternative_mappings[mapping_key]
                    break
        return alternatives[:3]

    def _calculate_confidence_adjustment(self, hypothesis: HypothesisDTO, counter_evidence: list[str],
                                          symptoms: list[SymptomDTO]) -> float:
        """Calculate confidence adjustment based on challenges."""
        adjustment = 0.0
        adjustment -= len(counter_evidence) * 0.05
        if hypothesis.criteria_missing:
            adjustment -= len(hypothesis.criteria_missing) * 0.03
        severe_count = sum(1 for s in symptoms if s.severity in [SeverityLevel.SEVERE, SeverityLevel.MODERATELY_SEVERE])
        mild_count = sum(1 for s in symptoms if s.severity in [SeverityLevel.MINIMAL, SeverityLevel.MILD])
        if mild_count > severe_count * 2:
            adjustment -= 0.05
        return max(min(adjustment, 0.1), -0.3)

    def _generate_counter_questions(self, hypothesis: HypothesisDTO,
                                     symptoms: list[SymptomDTO]) -> list[str]:
        """Generate questions to explore counter-evidence."""
        questions: list[str] = []
        has_duration = any(s.duration for s in symptoms)
        has_onset = any(s.onset for s in symptoms)
        if not has_duration:
            questions.extend(self._counter_questions["symptom_course"][:1])
        if not has_onset:
            questions.extend(self._counter_questions["symptom_onset"][:1])
        questions.extend(self._counter_questions["exclusion_criteria"][:1])
        return questions[:3]

    async def analyze_bias(self, hypotheses: list[HypothesisDTO], symptoms: list[SymptomDTO],
                           reasoning_history: list[dict[str, Any]]) -> BiasAnalysisResult:
        """Analyze reasoning for cognitive biases."""
        result = BiasAnalysisResult()
        if len(hypotheses) == 1 and hypotheses[0].confidence > Decimal("0.7"):
            result.detected_biases.append("premature_closure")
            result.bias_scores["premature_closure"] = 0.7
            result.recommendations.append("Consider generating additional differential diagnoses")
        if hypotheses:
            top = hypotheses[0]
            if top.confidence > Decimal("0.85") and len(top.criteria_missing) > 2:
                result.detected_biases.append("confirmation_bias")
                result.bias_scores["confirmation_bias"] = 0.6
                result.recommendations.append("High confidence despite missing criteria - review evidence")
        if len(hypotheses) >= 2:
            conf_diff = float(hypotheses[0].confidence - hypotheses[1].confidence)
            if conf_diff > 0.4:
                result.detected_biases.append("anchoring_bias")
                result.bias_scores["anchoring_bias"] = 0.5
                result.recommendations.append("Large confidence gap - ensure alternatives adequately considered")
        common_disorders = ["major_depressive_disorder", "generalized_anxiety_disorder"]
        if hypotheses and hypotheses[0].name.lower().replace(" ", "_") in common_disorders:
            if hypotheses[0].confidence > Decimal("0.8"):
                result.detected_biases.append("availability_heuristic")
                result.bias_scores["availability_heuristic"] = 0.4
                result.recommendations.append("Common diagnosis - verify not overlooking rarer conditions")
        if result.detected_biases:
            self._stats["biases_detected"] += len(result.detected_biases)
            avg_score = sum(result.bias_scores.values()) / len(result.bias_scores)
            result.overall_risk = "high" if avg_score > 0.6 else "medium" if avg_score > 0.4 else "low"
        logger.debug("bias_analysis_complete", biases=len(result.detected_biases),
                    risk=result.overall_risk)
        return result

    async def generate_counter_arguments(self, hypothesis: HypothesisDTO,
                                          supporting_evidence: list[str]) -> list[str]:
        """Generate counter-arguments for supporting evidence."""
        counters: list[str] = []
        for evidence in supporting_evidence:
            evidence_lower = evidence.lower()
            if "sad" in evidence_lower or "depressed" in evidence_lower:
                counters.append("Sadness could reflect normal grief or situational response")
            if "anxious" in evidence_lower or "worried" in evidence_lower:
                counters.append("Worry may be proportional to real-life stressors")
            if "sleep" in evidence_lower:
                counters.append("Sleep issues could have medical, environmental, or behavioral causes")
            if "fatigue" in evidence_lower or "tired" in evidence_lower:
                counters.append("Fatigue could indicate medical condition, sleep disorder, or lifestyle factors")
            if "concentration" in evidence_lower:
                counters.append("Concentration difficulties may relate to stress, ADHD, or sleep deprivation")
        return counters[:5]

    def get_bias_description(self, bias_name: str) -> str | None:
        """Get description of a cognitive bias."""
        pattern = self._bias_patterns.get(bias_name)
        return pattern["description"] if pattern else None

    def get_statistics(self) -> dict[str, int]:
        """Get advocate statistics."""
        return self._stats.copy()
