"""
Solace-AI Diagnosis Service - Evidence-Based Hypothesis Support.
Evaluates and scores clinical evidence supporting diagnostic hypotheses.
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


class EvidenceSettings(BaseSettings):
    """Evidence evaluation configuration."""
    min_evidence_threshold: float = Field(default=0.3)
    required_evidence_weight: float = Field(default=0.6)
    supporting_evidence_weight: float = Field(default=0.3)
    contextual_evidence_weight: float = Field(default=0.1)
    temporal_evidence_boost: float = Field(default=0.1)
    severity_evidence_boost: float = Field(default=0.05)
    model_config = SettingsConfigDict(env_prefix="EVIDENCE_", env_file=".env", extra="ignore")


@dataclass
class EvidenceItem:
    """Single piece of clinical evidence."""
    evidence_id: UUID = field(default_factory=uuid4)
    source: str = ""
    category: str = ""
    description: str = ""
    strength: float = 0.5
    supports_hypothesis: bool = True
    symptom_id: UUID | None = None
    relevance_score: float = 0.5


@dataclass
class EvidenceEvaluationResult:
    """Result from evidence evaluation."""
    evaluation_id: UUID = field(default_factory=uuid4)
    hypothesis_id: UUID = field(default_factory=uuid4)
    total_evidence_score: float = 0.0
    supporting_evidence: list[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    evidence_quality: str = "moderate"
    confidence_adjustment: float = 0.0


@dataclass
class EvidenceSummary:
    """Summary of evidence across hypotheses."""
    summary_id: UUID = field(default_factory=uuid4)
    hypotheses_evaluated: int = 0
    strongest_hypothesis: str = ""
    evidence_gap_score: float = 0.0
    recommended_assessments: list[str] = field(default_factory=list)
    overall_evidence_quality: str = "moderate"


class EvidenceEvaluator:
    """Evaluates clinical evidence for diagnostic hypotheses."""

    def __init__(self, settings: EvidenceSettings | None = None) -> None:
        self._settings = settings or EvidenceSettings()
        self._evidence_criteria = self._build_evidence_criteria()
        self._required_evidence = self._build_required_evidence()
        self._contextual_factors = self._build_contextual_factors()
        self._stats = {"evaluations": 0, "evidence_items_processed": 0}

    def _build_evidence_criteria(self) -> dict[str, dict[str, Any]]:
        """Build evidence criteria for disorders."""
        return {
            "major_depressive_disorder": {
                "required": ["depressed_mood_or_anhedonia", "duration_2_weeks"],
                "supporting": ["sleep_disturbance", "fatigue", "appetite_change", "concentration_difficulty",
                              "guilt", "psychomotor_changes", "suicidal_ideation"],
                "exclusionary": ["manic_episode", "substance_induced", "medical_condition"],
                "threshold": 5,
            },
            "generalized_anxiety_disorder": {
                "required": ["excessive_worry", "duration_6_months"],
                "supporting": ["restlessness", "fatigue", "concentration_difficulty", "irritability",
                              "muscle_tension", "sleep_disturbance"],
                "exclusionary": ["panic_disorder", "social_anxiety", "substance_induced"],
                "threshold": 3,
            },
            "panic_disorder": {
                "required": ["recurrent_panic_attacks", "persistent_concern"],
                "supporting": ["anticipatory_anxiety", "avoidance_behavior", "physical_symptoms"],
                "exclusionary": ["medical_condition", "substance_induced"],
                "threshold": 4,
            },
            "ptsd": {
                "required": ["trauma_exposure", "intrusion_symptoms", "avoidance"],
                "supporting": ["negative_cognitions", "hyperarousal", "duration_1_month"],
                "exclusionary": ["substance_induced", "psychotic_disorder"],
                "threshold": 4,
            },
            "social_anxiety_disorder": {
                "required": ["social_fear", "duration_6_months"],
                "supporting": ["avoidance", "fear_scrutiny", "anticipatory_anxiety"],
                "exclusionary": ["autism_spectrum", "medical_condition"],
                "threshold": 2,
            },
        }

    def _build_required_evidence(self) -> dict[str, list[str]]:
        """Build required evidence mapping to symptoms."""
        return {
            "depressed_mood_or_anhedonia": ["depressed_mood", "anhedonia", "sadness", "loss_of_interest"],
            "duration_2_weeks": ["chronic", "persistent", "ongoing", "continuous"],
            "excessive_worry": ["anxiety", "worry", "nervousness", "apprehension"],
            "duration_6_months": ["chronic", "long_term", "persistent"],
            "recurrent_panic_attacks": ["panic", "panic_attack", "sudden_fear"],
            "persistent_concern": ["worry_about_attacks", "anticipatory_anxiety"],
            "trauma_exposure": ["trauma", "traumatic_event", "abuse", "assault", "accident"],
            "intrusion_symptoms": ["intrusive_thoughts", "flashbacks", "nightmares"],
            "avoidance": ["avoidance", "avoiding", "withdrawal"],
            "social_fear": ["social_anxiety", "fear_of_judgment", "embarrassment_fear"],
        }

    def _build_contextual_factors(self) -> dict[str, float]:
        """Build contextual evidence factors."""
        return {
            "family_history": 0.15,
            "previous_episodes": 0.20,
            "recent_stressor": 0.10,
            "treatment_response": 0.15,
            "functional_impairment": 0.20,
            "duration_confirmed": 0.10,
            "onset_identified": 0.10,
        }

    async def evaluate(self, hypothesis: HypothesisDTO, symptoms: list[SymptomDTO],
                        context: dict[str, Any]) -> EvidenceEvaluationResult:
        """Evaluate evidence for a hypothesis."""
        self._stats["evaluations"] += 1
        result = EvidenceEvaluationResult(hypothesis_id=hypothesis.hypothesis_id)
        hypothesis_key = hypothesis.name.lower().replace(" ", "_")
        criteria = self._evidence_criteria.get(hypothesis_key)
        if criteria is None:
            for key in self._evidence_criteria:
                if key in hypothesis_key or hypothesis_key in key:
                    criteria = self._evidence_criteria[key]
                    break
        if criteria is None:
            criteria = {"required": [], "supporting": [], "exclusionary": [], "threshold": 2}
        supporting = self._gather_supporting_evidence(symptoms, criteria, context)
        result.supporting_evidence = supporting
        self._stats["evidence_items_processed"] += len(supporting)
        contradicting = self._gather_contradicting_evidence(symptoms, criteria, context)
        result.contradicting_evidence = contradicting
        result.missing_evidence = self._identify_missing_evidence(symptoms, criteria)
        result.total_evidence_score = self._calculate_evidence_score(result, criteria)
        result.evidence_quality = self._assess_evidence_quality(result)
        result.confidence_adjustment = self._calculate_confidence_adjustment(result)
        logger.debug("evidence_evaluated", hypothesis=hypothesis.name,
                    supporting=len(supporting), contradicting=len(contradicting),
                    score=result.total_evidence_score)
        return result

    def _gather_supporting_evidence(self, symptoms: list[SymptomDTO], criteria: dict[str, Any],
                                     context: dict[str, Any]) -> list[EvidenceItem]:
        """Gather supporting evidence from symptoms and context."""
        evidence: list[EvidenceItem] = []
        required = set(criteria.get("required", []))
        supporting = set(criteria.get("supporting", []))
        symptom_names = {s.name.lower() for s in symptoms}
        for req in required:
            mapped_symptoms = self._required_evidence.get(req, [req])
            matched = symptom_names & set(mapped_symptoms)
            if matched:
                for match in matched:
                    symptom = next((s for s in symptoms if s.name.lower() == match), None)
                    evidence.append(EvidenceItem(
                        source="symptom",
                        category="required",
                        description=f"Required criterion met: {match}",
                        strength=0.8 if symptom and symptom.severity in [SeverityLevel.MODERATE, SeverityLevel.SEVERE, SeverityLevel.MODERATELY_SEVERE] else 0.6,
                        symptom_id=symptom.symptom_id if symptom else None,
                        relevance_score=0.9,
                    ))
        for supp in supporting:
            if supp.lower() in symptom_names:
                symptom = next((s for s in symptoms if s.name.lower() == supp.lower()), None)
                strength = self._calculate_symptom_strength(symptom) if symptom else 0.5
                evidence.append(EvidenceItem(
                    source="symptom",
                    category="supporting",
                    description=f"Supporting symptom present: {supp}",
                    strength=strength,
                    symptom_id=symptom.symptom_id if symptom else None,
                    relevance_score=0.7,
                ))
        for factor, weight in self._contextual_factors.items():
            if context.get(factor):
                evidence.append(EvidenceItem(
                    source="context",
                    category="contextual",
                    description=f"Contextual factor present: {factor}",
                    strength=weight,
                    relevance_score=0.5,
                ))
        return evidence

    def _gather_contradicting_evidence(self, symptoms: list[SymptomDTO], criteria: dict[str, Any],
                                        context: dict[str, Any]) -> list[EvidenceItem]:
        """Gather evidence contradicting the hypothesis."""
        evidence: list[EvidenceItem] = []
        exclusionary = set(criteria.get("exclusionary", []))
        symptom_names = {s.name.lower() for s in symptoms}
        for excl in exclusionary:
            if excl.lower() in symptom_names or context.get(excl):
                evidence.append(EvidenceItem(
                    source="exclusion",
                    category="exclusionary",
                    description=f"Exclusionary criterion present: {excl}",
                    strength=0.8,
                    supports_hypothesis=False,
                    relevance_score=0.9,
                ))
        mild_symptoms = [s for s in symptoms if s.severity in [SeverityLevel.MINIMAL, SeverityLevel.MILD]]
        if len(mild_symptoms) > len(symptoms) * 0.8 and len(symptoms) > 2:
            evidence.append(EvidenceItem(
                source="severity",
                category="severity_inconsistency",
                description="Most symptoms are mild - may not meet clinical threshold",
                strength=0.4,
                supports_hypothesis=False,
                relevance_score=0.6,
            ))
        return evidence

    def _identify_missing_evidence(self, symptoms: list[SymptomDTO],
                                     criteria: dict[str, Any]) -> list[str]:
        """Identify missing evidence for the hypothesis."""
        missing: list[str] = []
        symptom_names = {s.name.lower() for s in symptoms}
        required = criteria.get("required", [])
        for req in required:
            mapped = self._required_evidence.get(req, [req])
            if not (symptom_names & set(mapped)):
                missing.append(f"Missing required evidence: {req}")
        has_duration = any(s.duration for s in symptoms)
        has_onset = any(s.onset for s in symptoms)
        if not has_duration:
            missing.append("Duration information not established")
        if not has_onset:
            missing.append("Onset timing not established")
        return missing

    def _calculate_symptom_strength(self, symptom: SymptomDTO) -> float:
        """Calculate evidence strength from symptom."""
        base_strength = 0.5
        severity_boost = {
            SeverityLevel.MINIMAL: 0.0,
            SeverityLevel.MILD: 0.1,
            SeverityLevel.MODERATE: 0.2,
            SeverityLevel.MODERATELY_SEVERE: 0.3,
            SeverityLevel.SEVERE: 0.4,
        }
        base_strength += severity_boost.get(symptom.severity, 0.1)
        if symptom.duration:
            base_strength += self._settings.temporal_evidence_boost
        if float(symptom.confidence) > 0.8:
            base_strength += 0.1
        return min(base_strength, 1.0)

    def _calculate_evidence_score(self, result: EvidenceEvaluationResult,
                                    criteria: dict[str, Any]) -> float:
        """Calculate total evidence score."""
        supporting_score = sum(e.strength * e.relevance_score for e in result.supporting_evidence)
        contradicting_score = sum(e.strength * e.relevance_score for e in result.contradicting_evidence)
        required_count = sum(1 for e in result.supporting_evidence if e.category == "required")
        supporting_count = sum(1 for e in result.supporting_evidence if e.category == "supporting")
        threshold = criteria.get("threshold", 2)
        criteria_ratio = min((required_count * 2 + supporting_count) / max(threshold, 1), 1.0)
        evidence_score = (
            criteria_ratio * self._settings.required_evidence_weight +
            min(supporting_score / 5, 1.0) * self._settings.supporting_evidence_weight
        )
        evidence_score -= contradicting_score * 0.3
        missing_penalty = len(result.missing_evidence) * 0.05
        evidence_score -= missing_penalty
        return max(0.0, min(1.0, evidence_score))

    def _assess_evidence_quality(self, result: EvidenceEvaluationResult) -> str:
        """Assess overall evidence quality."""
        supporting_count = len(result.supporting_evidence)
        contradicting_count = len(result.contradicting_evidence)
        missing_count = len(result.missing_evidence)
        if supporting_count >= 5 and contradicting_count == 0 and missing_count <= 1:
            return "high"
        if supporting_count >= 3 and contradicting_count <= 1 and missing_count <= 2:
            return "moderate"
        if supporting_count >= 2:
            return "low"
        return "insufficient"

    def _calculate_confidence_adjustment(self, result: EvidenceEvaluationResult) -> float:
        """Calculate confidence adjustment based on evidence."""
        adjustment = 0.0
        if result.evidence_quality == "high":
            adjustment += 0.1
        elif result.evidence_quality == "moderate":
            adjustment += 0.0
        elif result.evidence_quality == "low":
            adjustment -= 0.1
        else:
            adjustment -= 0.2
        adjustment -= len(result.contradicting_evidence) * 0.05
        adjustment -= len(result.missing_evidence) * 0.03
        return max(-0.3, min(0.2, adjustment))

    async def summarize_evidence(self, hypotheses: list[HypothesisDTO],
                                  evaluations: list[EvidenceEvaluationResult]) -> EvidenceSummary:
        """Summarize evidence across all hypotheses."""
        summary = EvidenceSummary(hypotheses_evaluated=len(hypotheses))
        if evaluations:
            best_idx = max(range(len(evaluations)), key=lambda i: evaluations[i].total_evidence_score)
            if hypotheses:
                summary.strongest_hypothesis = hypotheses[min(best_idx, len(hypotheses) - 1)].name
        total_missing = sum(len(e.missing_evidence) for e in evaluations)
        total_items = sum(len(e.supporting_evidence) + len(e.missing_evidence) for e in evaluations)
        summary.evidence_gap_score = total_missing / max(total_items, 1)
        summary.recommended_assessments = self._generate_assessment_recommendations(evaluations)
        qualities = [e.evidence_quality for e in evaluations]
        if all(q == "high" for q in qualities):
            summary.overall_evidence_quality = "high"
        elif any(q == "high" for q in qualities) or all(q in ["high", "moderate"] for q in qualities):
            summary.overall_evidence_quality = "moderate"
        else:
            summary.overall_evidence_quality = "low"
        return summary

    def _generate_assessment_recommendations(self,
                                               evaluations: list[EvidenceEvaluationResult]) -> list[str]:
        """Generate recommended assessments based on evidence gaps."""
        recommendations: list[str] = []
        all_missing: list[str] = []
        for evaluation in evaluations:
            all_missing.extend(evaluation.missing_evidence)
        if any("duration" in m.lower() for m in all_missing):
            recommendations.append("Establish symptom duration and timeline")
        if any("onset" in m.lower() for m in all_missing):
            recommendations.append("Determine symptom onset circumstances")
        if any("required" in m.lower() for m in all_missing):
            recommendations.append("Assess core diagnostic criteria more thoroughly")
        return recommendations[:3]

    async def evaluate_multiple(self, hypotheses: list[HypothesisDTO],
                                 symptoms: list[SymptomDTO],
                                 context: dict[str, Any]) -> list[EvidenceEvaluationResult]:
        """Evaluate evidence for multiple hypotheses."""
        results: list[EvidenceEvaluationResult] = []
        for hypothesis in hypotheses:
            result = await self.evaluate(hypothesis, symptoms, context)
            results.append(result)
        return results

    def get_statistics(self) -> dict[str, int]:
        """Get evaluator statistics."""
        return self._stats.copy()
