"""
Solace-AI Diagnosis Service - Sample Consistency Calibration.
Implements Bayesian confidence calibration and consistency scoring.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import HypothesisDTO, SymptomDTO, SeverityLevel, ConfidenceLevel

logger = structlog.get_logger(__name__)


class ConfidenceSettings(BaseSettings):
    """Confidence calibration configuration."""
    prior_weight: float = Field(default=0.3)
    evidence_weight: float = Field(default=0.5)
    consistency_weight: float = Field(default=0.2)
    min_confidence: float = Field(default=0.1)
    max_confidence: float = Field(default=0.95)
    calibration_samples: int = Field(default=5)
    uncertainty_penalty: float = Field(default=0.1)
    model_config = SettingsConfigDict(env_prefix="CONFIDENCE_", env_file=".env", extra="ignore")


@dataclass
class CalibrationResult:
    """Result from confidence calibration."""
    calibration_id: UUID = field(default_factory=uuid4)
    original_confidence: Decimal = Decimal("0.5")
    calibrated_confidence: Decimal = Decimal("0.5")
    confidence_interval: tuple[Decimal, Decimal] = field(default_factory=lambda: (Decimal("0.4"), Decimal("0.6")))
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    calibration_factors: dict[str, float] = field(default_factory=dict)
    uncertainty_score: float = 0.0


@dataclass
class ConsistencyResult:
    """Result from consistency analysis."""
    consistency_id: UUID = field(default_factory=uuid4)
    consistency_score: float = 0.0
    inconsistencies: list[str] = field(default_factory=list)
    temporal_consistency: float = 0.0
    symptom_consistency: float = 0.0
    severity_consistency: float = 0.0


class ConfidenceCalibrator:
    """Calibrates diagnostic confidence using Bayesian methods."""

    def __init__(self, settings: ConfidenceSettings | None = None) -> None:
        self._settings = settings or ConfidenceSettings()
        self._base_rates = self._build_base_rates()
        self._severity_weights = self._build_severity_weights()
        self._consistency_rules = self._build_consistency_rules()
        self._stats = {"calibrations": 0, "adjustments_made": 0}

    def _build_base_rates(self) -> dict[str, float]:
        """Build population base rates for disorders."""
        return {
            "major_depressive_disorder": 0.07,
            "generalized_anxiety_disorder": 0.03,
            "panic_disorder": 0.03,
            "social_anxiety_disorder": 0.07,
            "ptsd": 0.04,
            "adjustment_disorder": 0.10,
            "persistent_depressive_disorder": 0.02,
            "bipolar_disorder": 0.03,
            "ocd": 0.01,
            "specific_phobia": 0.08,
        }

    def _build_severity_weights(self) -> dict[SeverityLevel, float]:
        """Build severity level weights for calibration."""
        return {
            SeverityLevel.MINIMAL: 0.2,
            SeverityLevel.MILD: 0.4,
            SeverityLevel.MODERATE: 0.6,
            SeverityLevel.MODERATELY_SEVERE: 0.8,
            SeverityLevel.SEVERE: 1.0,
        }

    def _build_consistency_rules(self) -> list[dict[str, Any]]:
        """Build symptom consistency rules."""
        return [
            {
                "rule": "depression_requires_mood_or_anhedonia",
                "diagnosis": "major_depressive_disorder",
                "required_any": ["depressed_mood", "anhedonia"],
            },
            {
                "rule": "anxiety_requires_worry",
                "diagnosis": "generalized_anxiety_disorder",
                "required_any": ["anxiety", "worry", "nervousness"],
            },
            {
                "rule": "panic_requires_attacks",
                "diagnosis": "panic_disorder",
                "required_any": ["panic", "panic_attack"],
            },
            {
                "rule": "ptsd_requires_trauma",
                "diagnosis": "ptsd",
                "required_any": ["intrusive_thoughts", "flashbacks", "nightmares"],
            },
            {
                "rule": "social_anxiety_requires_social_fear",
                "diagnosis": "social_anxiety_disorder",
                "required_any": ["social_withdrawal", "social_fear", "performance_anxiety"],
            },
        ]

    async def calibrate(self, hypothesis: HypothesisDTO,
                        symptoms: list[SymptomDTO],
                        context: dict[str, Any]) -> CalibrationResult:
        """Calibrate confidence score using Bayesian methods."""
        self._stats["calibrations"] += 1
        result = CalibrationResult(original_confidence=hypothesis.confidence)
        prior = self._calculate_prior(hypothesis.name)
        likelihood = self._calculate_likelihood(hypothesis, symptoms)
        evidence_strength = self._calculate_evidence_strength(symptoms)
        posterior = self._bayesian_update(prior, likelihood, evidence_strength)
        consistency = await self.analyze_consistency(hypothesis, symptoms)
        consistency_factor = consistency.consistency_score
        calibrated = (
            posterior * self._settings.evidence_weight +
            float(hypothesis.confidence) * self._settings.prior_weight +
            consistency_factor * self._settings.consistency_weight
        )
        calibrated = max(self._settings.min_confidence, min(self._settings.max_confidence, calibrated))
        uncertainty = self._calculate_uncertainty(hypothesis, symptoms, consistency)
        interval_width = 0.15 + (uncertainty * 0.1)
        result.calibrated_confidence = Decimal(str(round(calibrated, 2)))
        result.confidence_interval = (
            Decimal(str(max(0, round(calibrated - interval_width, 2)))),
            Decimal(str(min(1, round(calibrated + interval_width, 2))))
        )
        result.confidence_level = self._determine_confidence_level(calibrated)
        result.uncertainty_score = uncertainty
        result.calibration_factors = {
            "prior": prior,
            "likelihood": likelihood,
            "evidence_strength": evidence_strength,
            "consistency": consistency_factor,
            "uncertainty": uncertainty,
        }
        if abs(float(hypothesis.confidence) - calibrated) > 0.05:
            self._stats["adjustments_made"] += 1
        logger.debug("confidence_calibrated", original=float(hypothesis.confidence),
                    calibrated=calibrated, uncertainty=uncertainty)
        return result

    def _calculate_prior(self, hypothesis_name: str) -> float:
        """Calculate prior probability from base rates."""
        key = hypothesis_name.lower().replace(" ", "_")
        base_rate = self._base_rates.get(key)
        if base_rate is None:
            for rate_key, rate in self._base_rates.items():
                if rate_key in key or key in rate_key:
                    return rate
            return 0.05
        return base_rate

    def _calculate_likelihood(self, hypothesis: HypothesisDTO,
                               symptoms: list[SymptomDTO]) -> float:
        """Calculate likelihood of symptoms given hypothesis."""
        if not symptoms:
            return 0.3
        met_count = len(hypothesis.criteria_met)
        missing_count = len(hypothesis.criteria_missing)
        total = met_count + missing_count if (met_count + missing_count) > 0 else 1
        criteria_ratio = met_count / total
        severity_total = sum(self._severity_weights.get(s.severity, 0.5) for s in symptoms)
        avg_severity = severity_total / len(symptoms) if symptoms else 0.5
        likelihood = (criteria_ratio * 0.7) + (avg_severity * 0.3)
        return min(0.95, likelihood)

    def _calculate_evidence_strength(self, symptoms: list[SymptomDTO]) -> float:
        """Calculate overall evidence strength from symptoms."""
        if not symptoms:
            return 0.3
        total_confidence = sum(float(s.confidence) for s in symptoms)
        avg_confidence = total_confidence / len(symptoms)
        symptom_count_factor = min(len(symptoms) / 5, 1.0)
        severity_count = sum(1 for s in symptoms if s.severity in [SeverityLevel.MODERATE, SeverityLevel.MODERATELY_SEVERE, SeverityLevel.SEVERE])
        severity_factor = severity_count / len(symptoms) if symptoms else 0
        strength = (avg_confidence * 0.5) + (symptom_count_factor * 0.3) + (severity_factor * 0.2)
        return min(0.95, strength)

    def _bayesian_update(self, prior: float, likelihood: float,
                          evidence_strength: float) -> float:
        """Perform Bayesian update on confidence."""
        prior = max(0.01, min(0.99, prior))
        likelihood = max(0.01, min(0.99, likelihood))
        marginal = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        if marginal < 0.01:
            marginal = 0.01
        posterior = (likelihood * prior) / marginal
        weighted_posterior = (posterior * evidence_strength) + (prior * (1 - evidence_strength))
        return weighted_posterior

    async def analyze_consistency(self, hypothesis: HypothesisDTO,
                                   symptoms: list[SymptomDTO]) -> ConsistencyResult:
        """Analyze consistency of hypothesis with symptoms."""
        result = ConsistencyResult()
        symptom_consistency = self._check_symptom_consistency(hypothesis, symptoms)
        result.symptom_consistency = symptom_consistency
        temporal_consistency = self._check_temporal_consistency(symptoms)
        result.temporal_consistency = temporal_consistency
        severity_consistency = self._check_severity_consistency(hypothesis, symptoms)
        result.severity_consistency = severity_consistency
        result.consistency_score = (
            symptom_consistency * 0.5 +
            temporal_consistency * 0.25 +
            severity_consistency * 0.25
        )
        result.inconsistencies = self._identify_inconsistencies(hypothesis, symptoms)
        return result

    def _check_symptom_consistency(self, hypothesis: HypothesisDTO,
                                     symptoms: list[SymptomDTO]) -> float:
        """Check if symptoms are consistent with hypothesis."""
        hypothesis_key = hypothesis.name.lower().replace(" ", "_")
        symptom_names = {s.name.lower() for s in symptoms}
        for rule in self._consistency_rules:
            if rule["diagnosis"] in hypothesis_key or hypothesis_key in rule["diagnosis"]:
                required = set(rule["required_any"])
                if required & symptom_names:
                    return 1.0
                return 0.5
        met_ratio = len(hypothesis.criteria_met) / (len(hypothesis.criteria_met) + len(hypothesis.criteria_missing) + 1)
        return met_ratio

    def _check_temporal_consistency(self, symptoms: list[SymptomDTO]) -> float:
        """Check temporal consistency of symptoms."""
        with_duration = [s for s in symptoms if s.duration]
        with_onset = [s for s in symptoms if s.onset]
        if not with_duration and not with_onset:
            return 0.5
        consistency = 0.5
        if with_duration:
            consistency += 0.25
        if with_onset:
            consistency += 0.25
        return consistency

    def _check_severity_consistency(self, hypothesis: HypothesisDTO,
                                      symptoms: list[SymptomDTO]) -> float:
        """Check if severity levels are consistent across symptoms."""
        if not symptoms:
            return 0.5
        severities = [s.severity for s in symptoms]
        severity_values = [self._severity_weights.get(s, 0.5) for s in severities]
        if len(severity_values) < 2:
            return 0.8
        mean_severity = sum(severity_values) / len(severity_values)
        variance = sum((v - mean_severity) ** 2 for v in severity_values) / len(severity_values)
        consistency = 1.0 - min(variance * 2, 0.5)
        return consistency

    def _identify_inconsistencies(self, hypothesis: HypothesisDTO,
                                    symptoms: list[SymptomDTO]) -> list[str]:
        """Identify specific inconsistencies."""
        inconsistencies: list[str] = []
        if hypothesis.criteria_missing and len(hypothesis.criteria_missing) > len(hypothesis.criteria_met):
            inconsistencies.append("More criteria missing than met for this diagnosis")
        severities = [s.severity for s in symptoms]
        if severities:
            severe = sum(1 for s in severities if s in [SeverityLevel.SEVERE, SeverityLevel.MODERATELY_SEVERE])
            mild = sum(1 for s in severities if s in [SeverityLevel.MINIMAL, SeverityLevel.MILD])
            if severe > 0 and mild > severe * 2:
                inconsistencies.append("Mixed severity levels may indicate different conditions")
        if hypothesis.confidence > Decimal("0.8") and not hypothesis.supporting_evidence:
            inconsistencies.append("High confidence without documented supporting evidence")
        return inconsistencies

    def _calculate_uncertainty(self, hypothesis: HypothesisDTO, symptoms: list[SymptomDTO],
                                 consistency: ConsistencyResult) -> float:
        """Calculate uncertainty score."""
        uncertainty = 0.0
        if len(symptoms) < 3:
            uncertainty += 0.2
        missing_ratio = len(hypothesis.criteria_missing) / (len(hypothesis.criteria_met) + len(hypothesis.criteria_missing) + 1)
        uncertainty += missing_ratio * 0.3
        uncertainty += (1 - consistency.consistency_score) * 0.3
        temporal_missing = sum(1 for s in symptoms if not s.duration and not s.onset)
        if symptoms:
            uncertainty += (temporal_missing / len(symptoms)) * 0.2
        return min(1.0, uncertainty)

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine categorical confidence level."""
        if confidence >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        if confidence >= 0.7:
            return ConfidenceLevel.HIGH
        if confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    async def calibrate_multiple(self, hypotheses: list[HypothesisDTO],
                                  symptoms: list[SymptomDTO],
                                  context: dict[str, Any]) -> list[CalibrationResult]:
        """Calibrate multiple hypotheses."""
        results: list[CalibrationResult] = []
        for hypothesis in hypotheses:
            result = await self.calibrate(hypothesis, symptoms, context)
            results.append(result)
        return results

    def get_base_rate(self, disorder: str) -> float | None:
        """Get base rate for a disorder."""
        key = disorder.lower().replace(" ", "_")
        return self._base_rates.get(key)

    def get_statistics(self) -> dict[str, int]:
        """Get calibration statistics."""
        return self._stats.copy()
