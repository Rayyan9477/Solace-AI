"""
Solace-AI Diagnosis Service - DSM-5-TR/HiTOP Differential Generation.
Generates clinical hypotheses with confidence scores and dimensional mapping.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import SymptomDTO, HypothesisDTO, SymptomType, SeverityLevel

if TYPE_CHECKING:
    from services.shared.infrastructure.llm_client import UnifiedLLMClient

logger = structlog.get_logger(__name__)

DIFFERENTIAL_PROMPT = (
    "You are a clinical psychologist generating differential diagnoses. "
    "Given the symptoms below, provide DSM-5-TR aligned diagnostic hypotheses.\n\n"
    "For each hypothesis, provide:\n"
    "- name: Human-readable disorder name\n"
    "- dsm5_code: ICD-10/DSM-5 code (e.g. F32, F41.1)\n"
    "- confidence: 0.0-1.0\n"
    "- criteria_met: list of symptom names that support this diagnosis\n"
    "- reasoning: brief clinical rationale\n\n"
    "Important: Never provide definitive diagnoses. These are clinical hypotheses only.\n\n"
    'Respond with ONLY valid JSON: {"hypotheses": [{"name": "...", "dsm5_code": "...", '
    '"confidence": 0.0, "criteria_met": ["..."], "reasoning": "..."}]}'
)


class DifferentialSettings(BaseSettings):
    """Differential generator configuration."""
    max_hypotheses: int = Field(default=5)
    min_confidence_threshold: float = Field(default=0.3)
    enable_hitop_mapping: bool = Field(default=True)
    enable_dsm5_mapping: bool = Field(default=True)
    confidence_decay_rate: float = Field(default=0.1)
    comorbidity_boost: float = Field(default=0.1)
    model_config = SettingsConfigDict(env_prefix="DIFFERENTIAL_", env_file=".env", extra="ignore")


@dataclass
class DifferentialResult:
    """Result from differential generation."""
    hypotheses: list[HypothesisDTO] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    hitop_scores: dict[str, Decimal] = field(default_factory=dict)
    recommended_questions: list[str] = field(default_factory=list)


class DifferentialGenerator:
    """Generates differential diagnoses using DSM-5-TR criteria and HiTOP dimensions."""

    def __init__(
        self,
        settings: DifferentialSettings | None = None,
        llm_client: UnifiedLLMClient | None = None,
    ) -> None:
        self._settings = settings or DifferentialSettings()
        self._llm_client = llm_client
        self._dsm5_criteria = self._build_dsm5_criteria()
        self._hitop_dimensions = self._build_hitop_dimensions()
        self._question_bank = self._build_question_bank()
        self._stats = {"generations": 0, "hypotheses_generated": 0}

    def _build_dsm5_criteria(self) -> dict[str, dict[str, Any]]:
        """Build DSM-5-TR diagnostic criteria mapping."""
        return {
            "major_depressive_disorder": {
                "dsm5_code": "F32",
                "icd11_code": "6A70",
                "required_symptoms": ["depressed_mood", "anhedonia"],
                "supporting_symptoms": ["sleep_disturbance", "fatigue", "appetite_change",
                                       "concentration_difficulty", "guilt", "psychomotor"],
                "min_criteria": 5,
                "duration_requirement": "2_weeks",
                "exclusions": ["bipolar_history", "substance_induced"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 5,
                    SeverityLevel.MODERATE: 6,
                    SeverityLevel.SEVERE: 8,
                },
            },
            "generalized_anxiety_disorder": {
                "dsm5_code": "F41.1",
                "icd11_code": "6B00",
                "required_symptoms": ["anxiety"],
                "supporting_symptoms": ["restlessness", "fatigue", "concentration_difficulty",
                                       "irritability", "physical_tension", "sleep_disturbance"],
                "min_criteria": 3,
                "duration_requirement": "6_months",
                "exclusions": ["panic_disorder", "social_anxiety"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 3,
                    SeverityLevel.MODERATE: 4,
                    SeverityLevel.SEVERE: 6,
                },
            },
            "panic_disorder": {
                "dsm5_code": "F41.0",
                "icd11_code": "6B01",
                "required_symptoms": ["panic", "physical_tension"],
                "supporting_symptoms": ["anxiety", "avoidance", "fear"],
                "min_criteria": 4,
                "duration_requirement": "1_month",
                "exclusions": ["medical_condition"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 4,
                    SeverityLevel.MODERATE: 6,
                    SeverityLevel.SEVERE: 8,
                },
            },
            "social_anxiety_disorder": {
                "dsm5_code": "F40.10",
                "icd11_code": "6B04",
                "required_symptoms": ["anxiety", "social_withdrawal"],
                "supporting_symptoms": ["fear", "avoidance", "physical_tension"],
                "min_criteria": 2,
                "duration_requirement": "6_months",
                "exclusions": ["autism_spectrum"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 2,
                    SeverityLevel.MODERATE: 3,
                    SeverityLevel.SEVERE: 5,
                },
            },
            "adjustment_disorder": {
                "dsm5_code": "F43.2",
                "icd11_code": "6B43",
                "required_symptoms": [],
                "supporting_symptoms": ["depressed_mood", "anxiety", "irritability",
                                       "concentration_difficulty", "sleep_disturbance"],
                "min_criteria": 1,
                "duration_requirement": "3_months",
                "exclusions": ["major_depression", "gad"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 1,
                    SeverityLevel.MODERATE: 2,
                    SeverityLevel.SEVERE: 4,
                },
            },
            "persistent_depressive_disorder": {
                "dsm5_code": "F34.1",
                "icd11_code": "6A72",
                "required_symptoms": ["depressed_mood"],
                "supporting_symptoms": ["appetite_change", "sleep_disturbance", "fatigue",
                                       "concentration_difficulty", "hopelessness"],
                "min_criteria": 2,
                "duration_requirement": "2_years",
                "exclusions": ["bipolar", "psychotic"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 2,
                    SeverityLevel.MODERATE: 3,
                    SeverityLevel.SEVERE: 5,
                },
            },
            "ptsd": {
                "dsm5_code": "F43.10",
                "icd11_code": "6B40",
                "required_symptoms": ["intrusive_thoughts", "avoidance"],
                "supporting_symptoms": ["anxiety", "sleep_disturbance", "irritability",
                                       "concentration_difficulty", "hypervigilance"],
                "min_criteria": 4,
                "duration_requirement": "1_month",
                "exclusions": ["substance_induced"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 4,
                    SeverityLevel.MODERATE: 6,
                    SeverityLevel.SEVERE: 8,
                },
            },
            "bipolar_i_disorder": {
                "dsm5_code": "F31",
                "icd11_code": "6A60",
                "required_symptoms": ["elevated_mood"],
                "supporting_symptoms": ["grandiosity", "decreased_need_for_sleep", "pressured_speech",
                                       "racing_thoughts", "impulsivity", "risk_taking", "irritability"],
                "min_criteria": 3,
                "duration_requirement": "1_week",
                "exclusions": ["substance_induced", "medical_condition"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 3,
                    SeverityLevel.MODERATE: 5,
                    SeverityLevel.SEVERE: 7,
                },
            },
            "bipolar_ii_disorder": {
                "dsm5_code": "F31.81",
                "icd11_code": "6A61",
                "required_symptoms": ["elevated_mood", "depressed_mood"],
                "supporting_symptoms": ["grandiosity", "decreased_need_for_sleep", "pressured_speech",
                                       "racing_thoughts", "impulsivity", "anhedonia", "fatigue"],
                "min_criteria": 3,
                "duration_requirement": "4_days",
                "exclusions": ["substance_induced"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 3,
                    SeverityLevel.MODERATE: 4,
                    SeverityLevel.SEVERE: 6,
                },
            },
            "obsessive_compulsive_disorder": {
                "dsm5_code": "F42",
                "icd11_code": "6B20",
                "required_symptoms": ["intrusive_thoughts"],
                "supporting_symptoms": ["compulsive_behavior", "anxiety", "avoidance",
                                       "concentration_difficulty", "guilt"],
                "min_criteria": 2,
                "duration_requirement": "ongoing",
                "exclusions": ["substance_induced", "medical_condition"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 2,
                    SeverityLevel.MODERATE: 3,
                    SeverityLevel.SEVERE: 5,
                },
            },
            "adhd_combined": {
                "dsm5_code": "F90.2",
                "icd11_code": "6A05",
                "required_symptoms": ["concentration_difficulty"],
                "supporting_symptoms": ["impulsivity", "hyperactivity", "inattention",
                                       "restlessness", "irritability", "sleep_disturbance"],
                "min_criteria": 6,
                "duration_requirement": "6_months",
                "exclusions": ["autism_spectrum", "psychotic"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 6,
                    SeverityLevel.MODERATE: 8,
                    SeverityLevel.SEVERE: 10,
                },
            },
            "anorexia_nervosa": {
                "dsm5_code": "F50.0",
                "icd11_code": "6B80",
                "required_symptoms": ["food_restriction", "body_image_distortion"],
                "supporting_symptoms": ["appetite_change", "anxiety", "social_withdrawal",
                                       "fatigue", "physical_tension"],
                "min_criteria": 2,
                "duration_requirement": "3_months",
                "exclusions": ["medical_condition"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 2,
                    SeverityLevel.MODERATE: 3,
                    SeverityLevel.SEVERE: 5,
                },
            },
            "bulimia_nervosa": {
                "dsm5_code": "F50.2",
                "icd11_code": "6B81",
                "required_symptoms": ["binge_eating", "purging"],
                "supporting_symptoms": ["body_image_distortion", "guilt", "anxiety",
                                       "appetite_change", "depressed_mood"],
                "min_criteria": 2,
                "duration_requirement": "3_months",
                "exclusions": ["anorexia_nervosa"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 2,
                    SeverityLevel.MODERATE: 3,
                    SeverityLevel.SEVERE: 5,
                },
            },
            "binge_eating_disorder": {
                "dsm5_code": "F50.81",
                "icd11_code": "6B82",
                "required_symptoms": ["binge_eating"],
                "supporting_symptoms": ["guilt", "depressed_mood", "anxiety",
                                       "appetite_change", "social_withdrawal"],
                "min_criteria": 3,
                "duration_requirement": "3_months",
                "exclusions": ["bulimia_nervosa"],
                "severity_thresholds": {
                    SeverityLevel.MILD: 3,
                    SeverityLevel.MODERATE: 4,
                    SeverityLevel.SEVERE: 5,
                },
            },
            "alcohol_use_disorder": {
                "dsm5_code": "F10.2",
                "icd11_code": "6C40.2",
                "required_symptoms": ["substance_use"],
                "supporting_symptoms": ["tolerance", "withdrawal", "craving",
                                       "concentration_difficulty", "social_withdrawal",
                                       "sleep_disturbance", "irritability"],
                "min_criteria": 2,
                "duration_requirement": "12_months",
                "exclusions": [],
                "severity_thresholds": {
                    SeverityLevel.MILD: 2,
                    SeverityLevel.MODERATE: 4,
                    SeverityLevel.SEVERE: 6,
                },
            },
            "substance_use_disorder": {
                "dsm5_code": "F19.2",
                "icd11_code": "6C4Z",
                "required_symptoms": ["substance_use"],
                "supporting_symptoms": ["tolerance", "withdrawal", "craving",
                                       "impulsivity", "social_withdrawal",
                                       "concentration_difficulty", "risk_taking"],
                "min_criteria": 2,
                "duration_requirement": "12_months",
                "exclusions": [],
                "severity_thresholds": {
                    SeverityLevel.MILD: 2,
                    SeverityLevel.MODERATE: 4,
                    SeverityLevel.SEVERE: 6,
                },
            },
            "borderline_personality_disorder": {
                "dsm5_code": "F60.3",
                "icd11_code": "6D10",
                "required_symptoms": ["emotional_instability"],
                "supporting_symptoms": ["abandonment_fear", "identity_disturbance",
                                       "impulsivity", "self_harm", "irritability",
                                       "social_withdrawal", "depressed_mood"],
                "min_criteria": 5,
                "duration_requirement": "pervasive",
                "exclusions": [],
                "severity_thresholds": {
                    SeverityLevel.MILD: 5,
                    SeverityLevel.MODERATE: 6,
                    SeverityLevel.SEVERE: 8,
                },
            },
        }

    def _build_hitop_dimensions(self) -> dict[str, dict[str, Any]]:
        """Build HiTOP dimensional model mapping."""
        return {
            "internalizing": {
                "subfactors": ["distress", "fear"],
                "symptoms": ["depressed_mood", "anxiety", "guilt", "hopelessness", "anhedonia"],
                "description": "Tendency toward negative emotionality and inward focus",
            },
            "thought_disorder": {
                "subfactors": ["psychoticism"],
                "symptoms": ["intrusive_thoughts", "disorganization", "hallucinations"],
                "description": "Disruption in thinking and perception",
            },
            "disinhibited_externalizing": {
                "subfactors": ["impulsivity", "distractibility"],
                "symptoms": ["impulsivity", "concentration_difficulty", "risk_taking"],
                "description": "Difficulty with impulse control and attention",
            },
            "antagonistic_externalizing": {
                "subfactors": ["hostility", "manipulativeness"],
                "symptoms": ["irritability", "aggression", "interpersonal_conflict"],
                "description": "Interpersonal hostility and callousness",
            },
            "detachment": {
                "subfactors": ["withdrawal", "anhedonia"],
                "symptoms": ["social_withdrawal", "anhedonia", "emotional_flatness"],
                "description": "Social withdrawal and emotional detachment",
            },
            "somatoform": {
                "subfactors": ["somatic_symptoms"],
                "symptoms": ["physical_tension", "fatigue", "appetite_change", "sleep_disturbance"],
                "description": "Physical symptom manifestation",
            },
        }

    def _build_question_bank(self) -> dict[str, list[str]]:
        """Build recommended questions by symptom gap."""
        return {
            "duration": ["How long have you been experiencing these feelings?",
                        "When did you first notice these changes?"],
            "severity": ["On a scale of 0-10, how intense are these feelings?",
                        "How much do these symptoms affect your daily life?"],
            "frequency": ["How often do you experience this?",
                         "Is this something you feel constantly or does it come and go?"],
            "triggers": ["Have you noticed anything that makes it better or worse?",
                        "Are there specific situations that bring this on?"],
            "function": ["How is this affecting your work/school/relationships?",
                        "Are you still able to do the things you need to do?"],
            "history": ["Have you experienced anything like this before?",
                       "Is there a history of mental health concerns in your family?"],
            "coping": ["What have you tried to help with this?",
                      "Is there anything that provides relief?"],
            "support": ["Do you have people you can talk to about this?",
                       "Who do you turn to for support?"],
        }

    async def generate(self, symptoms: list[SymptomDTO],
                       user_context: dict[str, Any]) -> DifferentialResult:
        """Generate differential diagnosis from symptoms."""
        self._stats["generations"] += 1
        result = DifferentialResult()
        symptom_names = {s.name for s in symptoms}
        hypotheses: list[tuple[str, float, dict[str, Any]]] = []
        for disorder, criteria in self._dsm5_criteria.items():
            confidence, details = self._calculate_match_confidence(symptom_names, symptoms, criteria)
            if confidence >= self._settings.min_confidence_threshold:
                hypotheses.append((disorder, confidence, details))
        hypotheses.sort(key=lambda x: x[1], reverse=True)
        for disorder, confidence, details in hypotheses[:self._settings.max_hypotheses]:
            criteria = self._dsm5_criteria[disorder]
            severity = self._determine_severity(details["criteria_met_count"], criteria)
            hypothesis = HypothesisDTO(
                hypothesis_id=uuid4(),
                name=self._format_disorder_name(disorder),
                dsm5_code=criteria["dsm5_code"],
                icd11_code=criteria["icd11_code"],
                confidence=Decimal(str(round(confidence, 2))),
                confidence_interval=(Decimal(str(max(0, confidence - 0.1))),
                                   Decimal(str(min(1, confidence + 0.1)))),
                criteria_met=details["criteria_met"],
                criteria_missing=details["criteria_missing"],
                supporting_evidence=[s.description for s in symptoms if s.name in details["criteria_met"]],
                severity=severity,
                hitop_dimensions=self._calculate_hitop_scores(symptoms),
            )
            result.hypotheses.append(hypothesis)

        # Enhance with LLM differential when available
        if self._llm_client is not None and self._llm_client.is_available:
            llm_hypotheses = await self._llm_generate_differential(symptoms, result.hypotheses)
            if llm_hypotheses:
                existing_codes = {h.dsm5_code for h in result.hypotheses}
                for h in llm_hypotheses:
                    if h.dsm5_code not in existing_codes and len(result.hypotheses) < self._settings.max_hypotheses:
                        result.hypotheses.append(h)

        self._stats["hypotheses_generated"] += len(result.hypotheses)
        if self._settings.enable_hitop_mapping:
            result.hitop_scores = self._calculate_hitop_scores(symptoms)
        result.missing_info = self._identify_missing_info(symptoms, hypotheses)
        result.recommended_questions = self._generate_recommended_questions(result.missing_info)
        logger.debug("differential_generated", hypotheses=len(result.hypotheses),
                    missing_info=len(result.missing_info))
        return result

    async def _llm_generate_differential(
        self,
        symptoms: list[SymptomDTO],
        existing_hypotheses: list[HypothesisDTO],
    ) -> list[HypothesisDTO]:
        """Use LLM to generate additional differential hypotheses."""
        try:
            symptom_summary = ", ".join(
                f"{s.name} ({s.severity.value}, confidence={s.confidence})" for s in symptoms
            )
            user_msg = f"Symptoms: {symptom_summary}"
            if existing_hypotheses:
                existing_names = ", ".join(h.name for h in existing_hypotheses)
                user_msg += f"\n\nRule-based hypotheses already identified: {existing_names}"
                user_msg += "\nPlease suggest additional diagnoses not already listed."

            response = await self._llm_client.generate(
                system_prompt=DIFFERENTIAL_PROMPT,
                user_message=user_msg,
                service_name="diagnosis_differential",
                task_type="diagnosis",
                max_tokens=600,
            )
            if not response:
                return []

            parsed = json.loads(response.strip())
            llm_hypotheses: list[HypothesisDTO] = []
            for item in parsed.get("hypotheses", []):
                confidence = float(item.get("confidence", 0.5))
                if confidence < self._settings.min_confidence_threshold:
                    continue
                llm_hypotheses.append(HypothesisDTO(
                    hypothesis_id=uuid4(),
                    name=item.get("name", "Unknown"),
                    dsm5_code=item.get("dsm5_code", ""),
                    icd11_code="",
                    confidence=Decimal(str(round(confidence, 2))),
                    confidence_interval=(
                        Decimal(str(max(0, confidence - 0.15))),
                        Decimal(str(min(1, confidence + 0.15))),
                    ),
                    criteria_met=item.get("criteria_met", []),
                    criteria_missing=[],
                    supporting_evidence=[item.get("reasoning", "")],
                    severity=SeverityLevel.MODERATE,
                    hitop_dimensions=self._calculate_hitop_scores(symptoms),
                ))
            logger.info("llm_differential_generated", count=len(llm_hypotheses))
            return llm_hypotheses
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("llm_differential_parse_failed", error=str(e))
            return []
        except Exception as e:
            logger.warning("llm_differential_failed", error=str(e))
            return []

    def _calculate_match_confidence(self, symptom_names: set[str], symptoms: list[SymptomDTO],
                                     criteria: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Calculate confidence score for diagnosis match."""
        required = set(criteria["required_symptoms"])
        supporting = set(criteria["supporting_symptoms"])
        all_criteria = required | supporting
        met = symptom_names & all_criteria
        missing = all_criteria - symptom_names
        required_met = required & symptom_names
        required_met_ratio = len(required_met) / len(required) if required else 1.0
        total_met_ratio = len(met) / len(all_criteria) if all_criteria else 0.0
        base_confidence = (required_met_ratio * 0.6) + (total_met_ratio * 0.4)
        severity_scores = {s.name: s.severity for s in symptoms if s.name in met}
        severity_boost = sum(
            0.05 if sev in [SeverityLevel.MODERATE, SeverityLevel.MODERATELY_SEVERE, SeverityLevel.SEVERE] else 0
            for sev in severity_scores.values()
        )
        confidence = min(base_confidence + severity_boost, 0.95)
        if len(required_met) < len(required):
            confidence *= 0.7
        details = {
            "criteria_met": list(met),
            "criteria_missing": list(missing),
            "criteria_met_count": len(met),
            "required_met": list(required_met),
        }
        return confidence, details

    def _determine_severity(self, criteria_met: int, criteria: dict[str, Any]) -> SeverityLevel:
        """Determine severity based on criteria met."""
        thresholds = criteria.get("severity_thresholds", {})
        if criteria_met >= thresholds.get(SeverityLevel.SEVERE, 99):
            return SeverityLevel.SEVERE
        if criteria_met >= thresholds.get(SeverityLevel.MODERATE, 99):
            return SeverityLevel.MODERATE
        if criteria_met >= thresholds.get(SeverityLevel.MILD, 99):
            return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    def _calculate_hitop_scores(self, symptoms: list[SymptomDTO]) -> dict[str, Decimal]:
        """Calculate HiTOP dimensional scores."""
        scores: dict[str, float] = {dim: 0.0 for dim in self._hitop_dimensions}
        symptom_names = {s.name for s in symptoms}
        for dimension, config in self._hitop_dimensions.items():
            dim_symptoms = set(config["symptoms"])
            overlap = symptom_names & dim_symptoms
            if overlap:
                score = len(overlap) / len(dim_symptoms)
                for symptom in symptoms:
                    if symptom.name in overlap:
                        severity_multiplier = {
                            SeverityLevel.MINIMAL: 0.2,
                            SeverityLevel.MILD: 0.4,
                            SeverityLevel.MODERATE: 0.6,
                            SeverityLevel.MODERATELY_SEVERE: 0.8,
                            SeverityLevel.SEVERE: 1.0,
                        }
                        score += severity_multiplier.get(symptom.severity, 0.5) * 0.2
                scores[dimension] = min(score, 1.0)
        return {k: Decimal(str(round(v, 2))) for k, v in scores.items()}

    def _identify_missing_info(self, symptoms: list[SymptomDTO],
                                hypotheses: list[tuple[str, float, dict[str, Any]]]) -> list[str]:
        """Identify missing information for diagnosis."""
        missing: list[str] = []
        has_duration = any(s.duration for s in symptoms)
        has_onset = any(s.onset for s in symptoms)
        has_triggers = any(s.triggers for s in symptoms)
        if not has_duration:
            missing.append("duration")
        if not has_onset:
            missing.append("onset")
        if not has_triggers:
            missing.append("triggers")
        if hypotheses:
            top_hypothesis = hypotheses[0]
            details = top_hypothesis[2]
            if details["criteria_missing"]:
                missing.append(f"symptoms:{details['criteria_missing'][0]}")
        return missing

    def _generate_recommended_questions(self, missing_info: list[str]) -> list[str]:
        """Generate recommended questions based on missing info."""
        questions: list[str] = []
        for info in missing_info[:3]:
            category = info.split(":")[0] if ":" in info else info
            if category in self._question_bank:
                questions.extend(self._question_bank[category][:1])
        return questions

    def _format_disorder_name(self, key: str) -> str:
        """Format disorder key to readable name."""
        return key.replace("_", " ").title()

    def get_dsm5_criteria(self, disorder: str) -> dict[str, Any] | None:
        """Get DSM-5 criteria for a specific disorder."""
        return self._dsm5_criteria.get(disorder)

    def get_hitop_dimension(self, dimension: str) -> dict[str, Any] | None:
        """Get HiTOP dimension details."""
        return self._hitop_dimensions.get(dimension)

    def calculate_comorbidity_likelihood(self, hypotheses: list[HypothesisDTO]) -> dict[str, float]:
        """Calculate likelihood of comorbidity between hypotheses."""
        comorbidity_patterns = {
            ("major_depressive_disorder", "generalized_anxiety_disorder"): 0.7,
            ("major_depressive_disorder", "social_anxiety_disorder"): 0.5,
            ("panic_disorder", "generalized_anxiety_disorder"): 0.4,
            ("ptsd", "major_depressive_disorder"): 0.6,
            ("bipolar_i_disorder", "substance_use_disorder"): 0.5,
            ("bipolar_i_disorder", "generalized_anxiety_disorder"): 0.5,
            ("adhd_combined", "major_depressive_disorder"): 0.4,
            ("adhd_combined", "substance_use_disorder"): 0.4,
            ("obsessive_compulsive_disorder", "major_depressive_disorder"): 0.6,
            ("borderline_personality_disorder", "major_depressive_disorder"): 0.7,
            ("borderline_personality_disorder", "substance_use_disorder"): 0.5,
            ("borderline_personality_disorder", "ptsd"): 0.5,
            ("anorexia_nervosa", "major_depressive_disorder"): 0.5,
            ("bulimia_nervosa", "major_depressive_disorder"): 0.5,
            ("alcohol_use_disorder", "major_depressive_disorder"): 0.6,
        }
        result: dict[str, float] = {}
        names = [h.name.lower().replace(" ", "_") for h in hypotheses]
        for (d1, d2), likelihood in comorbidity_patterns.items():
            if d1 in names and d2 in names:
                result[f"{d1}+{d2}"] = likelihood
        return result

    def get_statistics(self) -> dict[str, int]:
        """Get generation statistics."""
        return self._stats.copy()
