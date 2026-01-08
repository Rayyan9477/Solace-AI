"""
Solace-AI Contraindication Checker - Technique-condition contraindication validation.
Implements clinical rules engine for safe therapeutic technique selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class ContraindicationType(str, Enum):
    """Type of contraindication."""
    ABSOLUTE = "ABSOLUTE"  # Never use - dangerous
    RELATIVE = "RELATIVE"  # Use with caution - monitor closely
    TECHNIQUE_SPECIFIC = "TECHNIQUE_SPECIFIC"  # Requires prerequisites
    TIMING = "TIMING"  # Inappropriate timing (e.g., during crisis)
    SEVERITY = "SEVERITY"  # Inappropriate for severity level


class TherapyTechnique(str, Enum):
    """Therapeutic techniques to validate."""
    EXPOSURE_THERAPY = "EXPOSURE_THERAPY"
    COGNITIVE_RESTRUCTURING = "COGNITIVE_RESTRUCTURING"
    BEHAVIORAL_ACTIVATION = "BEHAVIORAL_ACTIVATION"
    MINDFULNESS_MEDITATION = "MINDFULNESS_MEDITATION"
    DBT_DIARY_CARD = "DBT_DIARY_CARD"
    DBT_DISTRESS_TOLERANCE = "DBT_DISTRESS_TOLERANCE"
    EMOTION_REGULATION = "EMOTION_REGULATION"
    INTERPERSONAL_EFFECTIVENESS = "INTERPERSONAL_EFFECTIVENESS"
    SOMATIC_EXPERIENCING = "SOMATIC_EXPERIENCING"
    EMDR = "EMDR"
    ACCEPTANCE_COMMITMENT = "ACCEPTANCE_COMMITMENT"
    PROGRESSIVE_MUSCLE_RELAXATION = "PROGRESSIVE_MUSCLE_RELAXATION"
    GROUNDING_TECHNIQUES = "GROUNDING_TECHNIQUES"


class MentalHealthCondition(str, Enum):
    """Mental health conditions for contraindication checking."""
    ACTIVE_PSYCHOSIS = "ACTIVE_PSYCHOSIS"
    SEVERE_DEPRESSION = "SEVERE_DEPRESSION"
    ACTIVE_SUBSTANCE_USE = "ACTIVE_SUBSTANCE_USE"
    DISSOCIATIVE_DISORDER = "DISSOCIATIVE_DISORDER"
    ACUTE_MANIA = "ACUTE_MANIA"
    SEVERE_PTSD = "SEVERE_PTSD"
    PERSONALITY_DISORDER = "PERSONALITY_DISORDER"
    EATING_DISORDER = "EATING_DISORDER"
    ACUTE_CRISIS = "ACUTE_CRISIS"
    SUICIDAL_IDEATION = "SUICIDAL_IDEATION"


class ContraindicationMatch(BaseModel):
    """Represents a detected contraindication."""
    technique: TherapyTechnique = Field(..., description="Therapeutic technique")
    condition: MentalHealthCondition = Field(..., description="Contraindicating condition")
    contraindication_type: ContraindicationType = Field(..., description="Type of contraindication")
    severity: Decimal = Field(..., ge=0, le=1, description="Severity of contraindication")
    rationale: str = Field(..., description="Clinical rationale")
    alternative_techniques: list[TherapyTechnique] = Field(default_factory=list, description="Safer alternatives")
    prerequisites: list[str] = Field(default_factory=list, description="Required prerequisites if applicable")


class ContraindicationResult(BaseModel):
    """Result of contraindication check."""
    check_id: UUID = Field(default_factory=uuid4, description="Unique check ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    technique: TherapyTechnique = Field(..., description="Technique being validated")
    is_safe: bool = Field(..., description="Whether technique is safe to use")
    contraindications: list[ContraindicationMatch] = Field(default_factory=list, description="Found contraindications")
    risk_score: Decimal = Field(..., ge=0, le=1, description="Overall contraindication risk")
    safety_level: str = Field(..., description="SAFE, CAUTION, or UNSAFE")
    clinical_notes: str = Field(..., description="Clinical guidance")


class ContraindicationConfig(BaseSettings):
    """Configuration for contraindication checker."""
    enable_absolute_checks: bool = Field(default=True, description="Check absolute contraindications")
    enable_relative_checks: bool = Field(default=True, description="Check relative contraindications")
    enable_timing_checks: bool = Field(default=True, description="Check timing appropriateness")
    enable_severity_checks: bool = Field(default=True, description="Check severity matching")
    enable_prerequisite_checks: bool = Field(default=True, description="Check prerequisites")
    absolute_block_threshold: Decimal = Field(default=Decimal("0.9"), description="Threshold for absolute blocking")
    relative_caution_threshold: Decimal = Field(default=Decimal("0.6"), description="Threshold for caution")

    model_config = SettingsConfigDict(env_prefix="CONTRAINDICATION_", env_file=".env", extra="ignore")


@dataclass
class ContraindicationRule:
    """Rule for technique-condition contraindication."""
    technique: TherapyTechnique
    condition: MentalHealthCondition
    contraindication_type: ContraindicationType
    severity: Decimal
    rationale: str
    alternatives: list[TherapyTechnique]
    prerequisites: list[str]


class ContraindicationChecker:
    """
    Clinical rules engine for therapeutic technique contraindication checking.
    Validates technique safety based on user conditions, severity, and timing.
    """

    def __init__(self, config: ContraindicationConfig | None = None) -> None:
        """Initialize contraindication checker with clinical rules."""
        self._config = config or ContraindicationConfig()
        self._rules = self._load_contraindication_rules()
        logger.info("contraindication_checker_initialized", rules_count=len(self._rules))

    def _load_contraindication_rules(self) -> list[ContraindicationRule]:
        """Load clinical contraindication rules."""
        rules: list[ContraindicationRule] = []

        # ABSOLUTE CONTRAINDICATIONS (Never use)
        rules.extend([
            ContraindicationRule(
                technique=TherapyTechnique.EXPOSURE_THERAPY,
                condition=MentalHealthCondition.ACTIVE_PSYCHOSIS,
                contraindication_type=ContraindicationType.ABSOLUTE,
                severity=Decimal("1.0"),
                rationale="Exposure therapy during active psychosis can worsen symptoms and cause severe distress",
                alternatives=[TherapyTechnique.GROUNDING_TECHNIQUES, TherapyTechnique.PROGRESSIVE_MUSCLE_RELAXATION],
                prerequisites=["Stabilization of psychotic symptoms", "Medication compliance"]
            ),
            ContraindicationRule(
                technique=TherapyTechnique.EXPOSURE_THERAPY,
                condition=MentalHealthCondition.ACUTE_CRISIS,
                contraindication_type=ContraindicationType.ABSOLUTE,
                severity=Decimal("0.95"),
                rationale="Exposure therapy during crisis can overwhelm coping capacity and escalate risk",
                alternatives=[TherapyTechnique.GROUNDING_TECHNIQUES, TherapyTechnique.DBT_DISTRESS_TOLERANCE],
                prerequisites=["Crisis resolution", "Safety plan in place"]
            ),
            ContraindicationRule(
                technique=TherapyTechnique.EMDR,
                condition=MentalHealthCondition.DISSOCIATIVE_DISORDER,
                contraindication_type=ContraindicationType.ABSOLUTE,
                severity=Decimal("0.95"),
                rationale="EMDR can trigger severe dissociation in patients with dissociative disorders",
                alternatives=[TherapyTechnique.SOMATIC_EXPERIENCING, TherapyTechnique.GROUNDING_TECHNIQUES],
                prerequisites=["Dissociation management skills", "Specialized therapist"]
            ),
        ])

        # RELATIVE CONTRAINDICATIONS (Use with caution)
        rules.extend([
            ContraindicationRule(
                technique=TherapyTechnique.COGNITIVE_RESTRUCTURING,
                condition=MentalHealthCondition.SEVERE_DEPRESSION,
                contraindication_type=ContraindicationType.RELATIVE,
                severity=Decimal("0.7"),
                rationale="Cognitive restructuring requires cognitive capacity that may be impaired in severe depression",
                alternatives=[TherapyTechnique.BEHAVIORAL_ACTIVATION, TherapyTechnique.GROUNDING_TECHNIQUES],
                prerequisites=["Baseline cognitive functioning", "Moderate energy levels"]
            ),
            ContraindicationRule(
                technique=TherapyTechnique.MINDFULNESS_MEDITATION,
                condition=MentalHealthCondition.SEVERE_PTSD,
                contraindication_type=ContraindicationType.RELATIVE,
                severity=Decimal("0.65"),
                rationale="Mindfulness can trigger flashbacks in severe PTSD without proper grounding skills",
                alternatives=[TherapyTechnique.GROUNDING_TECHNIQUES, TherapyTechnique.PROGRESSIVE_MUSCLE_RELAXATION],
                prerequisites=["Grounding skills mastery", "Safety cues established"]
            ),
            ContraindicationRule(
                technique=TherapyTechnique.EMOTION_REGULATION,
                condition=MentalHealthCondition.ACTIVE_SUBSTANCE_USE,
                contraindication_type=ContraindicationType.RELATIVE,
                severity=Decimal("0.65"),
                rationale="Substance use impairs emotional regulation capacity and skill application",
                alternatives=[TherapyTechnique.DBT_DISTRESS_TOLERANCE, TherapyTechnique.GROUNDING_TECHNIQUES],
                prerequisites=["Sobriety or stable substance use", "Addiction treatment engagement"]
            ),
        ])

        # TECHNIQUE-SPECIFIC PREREQUISITES
        rules.extend([
            ContraindicationRule(
                technique=TherapyTechnique.DBT_DIARY_CARD,
                condition=MentalHealthCondition.ACUTE_CRISIS,
                contraindication_type=ContraindicationType.TECHNIQUE_SPECIFIC,
                severity=Decimal("0.6"),
                rationale="Diary cards require emotional stability and are typically introduced after crisis skills",
                alternatives=[TherapyTechnique.DBT_DISTRESS_TOLERANCE, TherapyTechnique.GROUNDING_TECHNIQUES],
                prerequisites=["Distress tolerance skills", "Crisis skills mastery", "Stable emotional baseline"]
            ),
            ContraindicationRule(
                technique=TherapyTechnique.INTERPERSONAL_EFFECTIVENESS,
                condition=MentalHealthCondition.ACUTE_MANIA,
                contraindication_type=ContraindicationType.TIMING,
                severity=Decimal("0.7"),
                rationale="Interpersonal effectiveness training requires stable mood and impulse control",
                alternatives=[TherapyTechnique.GROUNDING_TECHNIQUES, TherapyTechnique.PROGRESSIVE_MUSCLE_RELAXATION],
                prerequisites=["Mood stabilization", "Medication adherence"]
            ),
        ])

        # SEVERITY MISMATCHES
        rules.extend([
            ContraindicationRule(
                technique=TherapyTechnique.BEHAVIORAL_ACTIVATION,
                condition=MentalHealthCondition.SUICIDAL_IDEATION,
                contraindication_type=ContraindicationType.SEVERITY,
                severity=Decimal("0.8"),
                rationale="Behavioral activation alone is insufficient for active suicidal ideation",
                alternatives=[TherapyTechnique.DBT_DISTRESS_TOLERANCE, TherapyTechnique.GROUNDING_TECHNIQUES],
                prerequisites=["Safety plan established", "Crisis intervention in place"]
            ),
        ])

        return rules

    def check(self, technique: TherapyTechnique, conditions: list[MentalHealthCondition],
             user_id: UUID | None = None, context: dict[str, Any] | None = None) -> ContraindicationResult:
        """
        Check for contraindications of a technique given user conditions.

        Args:
            technique: Therapeutic technique to validate
            conditions: Active mental health conditions
            user_id: Optional user ID for logging
            context: Optional context (severity scores, session number, etc.)

        Returns:
            Contraindication check result with safety assessment
        """
        contraindications: list[ContraindicationMatch] = []

        # Check all rules for this technique and conditions
        for rule in self._rules:
            if rule.technique != technique:
                continue

            if rule.condition not in conditions:
                continue

            # Apply configuration filters
            if not self._should_check_rule(rule):
                continue

            # Create contraindication match
            match = ContraindicationMatch(
                technique=technique,
                condition=rule.condition,
                contraindication_type=rule.contraindication_type,
                severity=rule.severity,
                rationale=rule.rationale,
                alternative_techniques=rule.alternatives,
                prerequisites=rule.prerequisites
            )
            contraindications.append(match)

        # Calculate risk score and safety level
        risk_score = self._calculate_risk_score(contraindications)
        safety_level = self._determine_safety_level(risk_score, contraindications)
        is_safe = safety_level == "SAFE"

        # Generate clinical notes
        clinical_notes = self._generate_clinical_notes(technique, contraindications, safety_level)

        result = ContraindicationResult(
            technique=technique,
            is_safe=is_safe,
            contraindications=contraindications,
            risk_score=risk_score,
            safety_level=safety_level,
            clinical_notes=clinical_notes
        )

        if user_id and not is_safe:
            logger.warning("contraindication_detected", user_id=str(user_id),
                          technique=technique.value, safety_level=safety_level,
                          contraindication_count=len(contraindications))

        return result

    def _should_check_rule(self, rule: ContraindicationRule) -> bool:
        """Determine if rule should be checked based on configuration."""
        if rule.contraindication_type == ContraindicationType.ABSOLUTE:
            return self._config.enable_absolute_checks
        if rule.contraindication_type == ContraindicationType.RELATIVE:
            return self._config.enable_relative_checks
        if rule.contraindication_type == ContraindicationType.TIMING:
            return self._config.enable_timing_checks
        if rule.contraindication_type == ContraindicationType.SEVERITY:
            return self._config.enable_severity_checks
        if rule.contraindication_type == ContraindicationType.TECHNIQUE_SPECIFIC:
            return self._config.enable_prerequisite_checks
        return True

    def _calculate_risk_score(self, contraindications: list[ContraindicationMatch]) -> Decimal:
        """Calculate overall contraindication risk score."""
        if not contraindications:
            return Decimal("0.0")

        # Use maximum severity
        max_severity = max(c.severity for c in contraindications)

        # Count absolute contraindications
        absolute_count = sum(1 for c in contraindications
                            if c.contraindication_type == ContraindicationType.ABSOLUTE)

        # Boost for multiple contraindications
        multiplicity_boost = min(Decimal(str(len(contraindications) - 1)) * Decimal("0.1"), Decimal("0.3"))

        total_risk = max_severity + multiplicity_boost
        return min(total_risk, Decimal("1.0"))

    def _determine_safety_level(self, risk_score: Decimal,
                                contraindications: list[ContraindicationMatch]) -> str:
        """Determine safety level from risk score and contraindications."""
        # Check for absolute contraindications
        has_absolute = any(c.contraindication_type == ContraindicationType.ABSOLUTE
                          for c in contraindications)

        if has_absolute or risk_score >= self._config.absolute_block_threshold:
            return "UNSAFE"

        if risk_score >= self._config.relative_caution_threshold:
            return "CAUTION"

        return "SAFE"

    def _generate_clinical_notes(self, technique: TherapyTechnique,
                                 contraindications: list[ContraindicationMatch],
                                 safety_level: str) -> str:
        """Generate clinical guidance notes."""
        if safety_level == "SAFE":
            return f"{technique.value} is appropriate for the current conditions."

        if safety_level == "UNSAFE":
            absolute = [c for c in contraindications
                       if c.contraindication_type == ContraindicationType.ABSOLUTE]
            if absolute:
                notes = f"ABSOLUTE CONTRAINDICATION: {technique.value} is unsafe. "
                notes += f"Reason: {absolute[0].rationale}. "
                if absolute[0].alternative_techniques:
                    alts = ", ".join(t.value for t in absolute[0].alternative_techniques[:2])
                    notes += f"Consider alternatives: {alts}."
                return notes

        # CAUTION
        notes = f"CAUTION: {technique.value} requires careful monitoring. "
        if contraindications:
            notes += f"Primary concern: {contraindications[0].rationale}. "
            if contraindications[0].prerequisites:
                prereqs = ", ".join(contraindications[0].prerequisites[:2])
                notes += f"Prerequisites: {prereqs}."

        return notes

    def get_safe_alternatives(self, technique: TherapyTechnique,
                             conditions: list[MentalHealthCondition]) -> list[TherapyTechnique]:
        """Get safe alternative techniques for given conditions."""
        result = self.check(technique, conditions)
        if result.is_safe:
            return [technique]

        # Collect alternatives from contraindications
        alternatives: set[TherapyTechnique] = set()
        for contra in result.contraindications:
            alternatives.update(contra.alternative_techniques)

        return list(alternatives)
