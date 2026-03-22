"""
Solace-AI Diagnosis Service - PHQ-9/GAD-7 Severity Assessment.
Implements standardized clinical questionnaire scoring and severity mapping.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import SeverityLevel, SymptomDTO

logger = structlog.get_logger(__name__)


class SeveritySettings(BaseSettings):
    """Severity assessment configuration."""
    enable_phq9: bool = Field(default=True)
    enable_gad7: bool = Field(default=True)
    enable_phq15: bool = Field(default=True)
    enable_pcl5: bool = Field(default=True)
    enable_cssrs: bool = Field(default=True)
    enable_audit: bool = Field(default=True)
    enable_mdq: bool = Field(default=True)
    enable_asrs: bool = Field(default=True)
    min_items_for_assessment: int = Field(default=3)
    imputation_method: str = Field(default="mean")
    model_config = SettingsConfigDict(env_prefix="SEVERITY_", env_file=".env", extra="ignore")


@dataclass
class QuestionnaireResult:
    """Result from questionnaire scoring."""
    result_id: UUID = field(default_factory=uuid4)
    questionnaire: str = ""
    total_score: int = 0
    max_score: int = 0
    severity_level: SeverityLevel = SeverityLevel.MINIMAL
    item_scores: dict[str, int] = field(default_factory=dict)
    interpretation: str = ""
    recommendations: list[str] = field(default_factory=list)
    items_answered: int = 0
    items_imputed: int = 0


@dataclass
class SeverityAssessmentResult:
    """Comprehensive severity assessment result."""
    assessment_id: UUID = field(default_factory=uuid4)
    overall_severity: SeverityLevel = SeverityLevel.MINIMAL
    depression_severity: QuestionnaireResult | None = None
    anxiety_severity: QuestionnaireResult | None = None
    somatic_severity: QuestionnaireResult | None = None
    trauma_severity: QuestionnaireResult | None = None
    suicidality_severity: QuestionnaireResult | None = None
    alcohol_severity: QuestionnaireResult | None = None
    bipolar_screen: QuestionnaireResult | None = None
    adhd_screen: QuestionnaireResult | None = None
    composite_score: float = 0.0
    functional_impairment: str = "none"


class SeverityAssessor:
    """Assesses clinical severity using standardized questionnaires."""

    def __init__(self, settings: SeveritySettings | None = None) -> None:
        self._settings = settings or SeveritySettings()
        self._phq9_items = self._build_phq9_items()
        self._gad7_items = self._build_gad7_items()
        self._phq15_items = self._build_phq15_items()
        self._pcl5_items = self._build_pcl5_items()
        self._cssrs_items = self._build_cssrs_items()
        self._audit_items = self._build_audit_items()
        self._mdq_items = self._build_mdq_items()
        self._asrs_items = self._build_asrs_items()
        self._symptom_mappings = self._build_symptom_mappings()
        self._stats = {"assessments": 0, "phq9_scored": 0, "gad7_scored": 0}

    def _build_phq9_items(self) -> dict[str, dict[str, Any]]:
        """Build PHQ-9 questionnaire items."""
        return {
            "phq9_1": {"text": "Little interest or pleasure in doing things", "symptom": "anhedonia"},
            "phq9_2": {"text": "Feeling down, depressed, or hopeless", "symptom": "depressed_mood"},
            "phq9_3": {"text": "Trouble falling/staying asleep or sleeping too much", "symptom": "sleep_disturbance"},
            "phq9_4": {"text": "Feeling tired or having little energy", "symptom": "fatigue"},
            "phq9_5": {"text": "Poor appetite or overeating", "symptom": "appetite_change"},
            "phq9_6": {"text": "Feeling bad about yourself", "symptom": "guilt"},
            "phq9_7": {"text": "Trouble concentrating on things", "symptom": "concentration_difficulty"},
            "phq9_8": {"text": "Moving/speaking slowly or being fidgety/restless", "symptom": "psychomotor"},
            "phq9_9": {"text": "Thoughts of self-harm or suicide", "symptom": "suicidal_ideation"},
        }

    def _build_gad7_items(self) -> dict[str, dict[str, Any]]:
        """Build GAD-7 questionnaire items."""
        return {
            "gad7_1": {"text": "Feeling nervous, anxious, or on edge", "symptom": "anxiety"},
            "gad7_2": {"text": "Not being able to stop or control worrying", "symptom": "worry"},
            "gad7_3": {"text": "Worrying too much about different things", "symptom": "excessive_worry"},
            "gad7_4": {"text": "Trouble relaxing", "symptom": "restlessness"},
            "gad7_5": {"text": "Being so restless that it's hard to sit still", "symptom": "motor_restlessness"},
            "gad7_6": {"text": "Becoming easily annoyed or irritable", "symptom": "irritability"},
            "gad7_7": {"text": "Feeling afraid as if something awful might happen", "symptom": "fear"},
        }

    def _build_phq15_items(self) -> dict[str, dict[str, Any]]:
        """Build PHQ-15 somatic symptom items."""
        return {
            "phq15_1": {"text": "Stomach pain", "symptom": "stomach_pain"},
            "phq15_2": {"text": "Back pain", "symptom": "back_pain"},
            "phq15_3": {"text": "Pain in arms, legs, or joints", "symptom": "pain"},
            "phq15_4": {"text": "Headaches", "symptom": "headache"},
            "phq15_5": {"text": "Chest pain", "symptom": "chest_pain"},
            "phq15_6": {"text": "Dizziness", "symptom": "dizziness"},
            "phq15_7": {"text": "Fainting spells", "symptom": "fainting"},
            "phq15_8": {"text": "Heart pounding or racing", "symptom": "palpitations"},
            "phq15_9": {"text": "Shortness of breath", "symptom": "dyspnea"},
            "phq15_10": {"text": "Constipation, loose bowels, or diarrhea", "symptom": "gi_symptoms"},
            "phq15_11": {"text": "Nausea, gas, or indigestion", "symptom": "nausea"},
            "phq15_12": {"text": "Feeling tired or having low energy", "symptom": "fatigue"},
            "phq15_13": {"text": "Trouble sleeping", "symptom": "sleep_disturbance"},
            "phq15_14": {"text": "Menstrual cramps or other problems with your periods", "symptom": "menstrual_problems"},
            "phq15_15": {"text": "Pain or problems during sexual intercourse", "symptom": "sexual_pain"},
        }

    def _build_pcl5_items(self) -> dict[str, dict[str, Any]]:
        """Build PCL-5 PTSD checklist items (abbreviated)."""
        return {
            "pcl5_1": {"text": "Repeated disturbing memories, thoughts, or images", "symptom": "intrusive_thoughts"},
            "pcl5_2": {"text": "Repeated disturbing dreams", "symptom": "nightmares"},
            "pcl5_3": {"text": "Suddenly feeling as if stressful experience were happening again", "symptom": "flashbacks"},
            "pcl5_4": {"text": "Feeling upset when reminded of stressful experience", "symptom": "emotional_reactivity"},
            "pcl5_5": {"text": "Physical reactions when reminded", "symptom": "physiological_reactivity"},
            "pcl5_6": {"text": "Avoiding memories, thoughts, or feelings", "symptom": "avoidance"},
            "pcl5_7": {"text": "Avoiding external reminders", "symptom": "situational_avoidance"},
            "pcl5_8": {"text": "Trouble remembering important parts", "symptom": "amnesia"},
            "pcl5_9": {"text": "Strong negative beliefs about yourself or world", "symptom": "negative_cognitions"},
            "pcl5_10": {"text": "Blaming yourself or others", "symptom": "blame"},
        }

    def _build_cssrs_items(self) -> dict[str, dict[str, Any]]:
        """Build C-SSRS suicidality assessment items."""
        return {
            "cssrs_1": {"text": "Wish to be dead", "symptom": "death_wish", "level": 1},
            "cssrs_2": {"text": "Non-specific active suicidal thoughts", "symptom": "suicidal_thoughts", "level": 2},
            "cssrs_3": {"text": "Active suicidal ideation with any methods (not plan) without intent", "symptom": "suicidal_ideation_methods", "level": 3},
            "cssrs_4": {"text": "Active suicidal ideation with some intent to act, without specific plan", "symptom": "suicidal_intent", "level": 4},
            "cssrs_5": {"text": "Active suicidal ideation with specific plan and intent", "symptom": "suicidal_plan", "level": 5},
            "cssrs_6": {"text": "Preparatory acts or behavior", "symptom": "preparatory_behavior", "level": 5},
        }

    def _build_audit_items(self) -> dict[str, dict[str, Any]]:
        """Build AUDIT alcohol screening items."""
        return {
            "audit_1": {"text": "How often do you have a drink containing alcohol?", "symptom": "alcohol_frequency"},
            "audit_2": {"text": "How many drinks containing alcohol on a typical drinking day?", "symptom": "alcohol_quantity"},
            "audit_3": {"text": "How often have you had 6+ drinks on one occasion?", "symptom": "binge_drinking"},
            "audit_4": {"text": "Unable to stop drinking once started", "symptom": "impaired_control"},
            "audit_5": {"text": "Failed to do what was normally expected due to drinking", "symptom": "role_failure"},
            "audit_6": {"text": "Needed a first drink in the morning", "symptom": "morning_drinking"},
            "audit_7": {"text": "Guilt or remorse after drinking", "symptom": "drinking_guilt"},
            "audit_8": {"text": "Unable to remember events due to drinking", "symptom": "blackouts"},
            "audit_9": {"text": "Injury related to drinking", "symptom": "alcohol_injury"},
            "audit_10": {"text": "Others concerned about your drinking", "symptom": "others_concerned"},
        }

    def _build_mdq_items(self) -> dict[str, dict[str, Any]]:
        """Build MDQ bipolar screening items."""
        return {
            "mdq_1": {"text": "Felt so good or hyper that others thought you were not your normal self", "symptom": "elevated_mood"},
            "mdq_2": {"text": "Were so irritable that you shouted at people", "symptom": "irritability"},
            "mdq_3": {"text": "Felt much more self-confident than usual", "symptom": "grandiosity"},
            "mdq_4": {"text": "Got much less sleep than usual and still didn't feel tired", "symptom": "decreased_sleep"},
            "mdq_5": {"text": "Were much more talkative or spoke faster than usual", "symptom": "pressured_speech"},
            "mdq_6": {"text": "Thoughts raced through your head", "symptom": "racing_thoughts"},
            "mdq_7": {"text": "Were so easily distracted that you had trouble concentrating", "symptom": "distractibility"},
            "mdq_8": {"text": "Had much more energy than usual", "symptom": "increased_energy"},
            "mdq_9": {"text": "Were much more active or did many more things than usual", "symptom": "increased_activity"},
            "mdq_10": {"text": "Were much more social or outgoing than usual", "symptom": "increased_sociability"},
            "mdq_11": {"text": "Were much more interested in sex than usual", "symptom": "hypersexuality"},
            "mdq_12": {"text": "Did things that were unusual for you or that others might think excessive", "symptom": "excessive_behavior"},
            "mdq_13": {"text": "Spending money got you or your family in trouble", "symptom": "reckless_spending"},
            "mdq_14": {"text": "Several symptoms occurred during the same period", "symptom": "concurrent_symptoms"},
            "mdq_15": {"text": "Caused moderate or serious problems", "symptom": "functional_impairment"},
        }

    def _build_asrs_items(self) -> dict[str, dict[str, Any]]:
        """Build ASRS ADHD screening items."""
        return {
            "asrs_1": {"text": "Trouble wrapping up final details of a project", "symptom": "inattention_detail"},
            "asrs_2": {"text": "Difficulty getting things in order for a task requiring organization", "symptom": "organization_difficulty"},
            "asrs_3": {"text": "Problems remembering appointments or obligations", "symptom": "forgetfulness"},
            "asrs_4": {"text": "Avoid or delay starting tasks requiring a lot of thought", "symptom": "task_avoidance"},
            "asrs_5": {"text": "Fidget or squirm with hands or feet in long meetings", "symptom": "hyperactivity"},
            "asrs_6": {"text": "Feel overly active and compelled to do things", "symptom": "restlessness"},
        }

    def _build_symptom_mappings(self) -> dict[str, list[str]]:
        """Build symptom to questionnaire item mappings."""
        mappings: dict[str, list[str]] = {}
        for item_id, item in self._phq9_items.items():
            symptom = item["symptom"]
            if symptom not in mappings:
                mappings[symptom] = []
            mappings[symptom].append(item_id)
        for item_id, item in self._gad7_items.items():
            symptom = item["symptom"]
            if symptom not in mappings:
                mappings[symptom] = []
            mappings[symptom].append(item_id)
        return mappings

    async def assess(self, symptoms: list[SymptomDTO],
                     responses: dict[str, int] | None = None) -> SeverityAssessmentResult:
        """Perform comprehensive severity assessment."""
        self._stats["assessments"] += 1
        result = SeverityAssessmentResult()
        inferred_responses = self._infer_responses_from_symptoms(symptoms)
        all_responses = {**inferred_responses, **(responses or {})}
        if self._settings.enable_phq9:
            result.depression_severity = self._score_phq9(all_responses)
        if self._settings.enable_gad7:
            result.anxiety_severity = self._score_gad7(all_responses)
        if self._settings.enable_phq15:
            result.somatic_severity = self._score_phq15(all_responses)
        if self._settings.enable_pcl5:
            result.trauma_severity = self._score_pcl5(all_responses)
        if self._settings.enable_cssrs:
            result.suicidality_severity = self._score_cssrs(all_responses)
        if self._settings.enable_audit:
            result.alcohol_severity = self._score_audit(all_responses)
        if self._settings.enable_mdq:
            result.bipolar_screen = self._score_mdq(all_responses)
        if self._settings.enable_asrs:
            result.adhd_screen = self._score_asrs(all_responses)
        result.overall_severity = self._calculate_overall_severity(result)
        result.composite_score = self._calculate_composite_score(result)
        result.functional_impairment = self._assess_functional_impairment(result)
        logger.debug("severity_assessed", overall=result.overall_severity.value,
                    composite=result.composite_score)
        return result

    def _infer_responses_from_symptoms(self, symptoms: list[SymptomDTO]) -> dict[str, int]:
        """Infer questionnaire responses from symptoms."""
        responses: dict[str, int] = {}
        severity_to_score = {
            SeverityLevel.MINIMAL: 0,
            SeverityLevel.MILD: 1,
            SeverityLevel.MODERATE: 2,
            SeverityLevel.MODERATELY_SEVERE: 3,
            SeverityLevel.SEVERE: 3,
        }
        for symptom in symptoms:
            symptom_name = symptom.name.lower()
            score = severity_to_score.get(symptom.severity, 1)
            if symptom_name in self._symptom_mappings:
                for item_id in self._symptom_mappings[symptom_name]:
                    responses[item_id] = max(responses.get(item_id, 0), score)
            for item_id, item in {**self._phq9_items, **self._gad7_items, **self._phq15_items, **self._pcl5_items}.items():
                if item["symptom"].lower() == symptom_name:
                    responses[item_id] = max(responses.get(item_id, 0), score)
        return responses

    def _score_phq9(self, responses: dict[str, int]) -> QuestionnaireResult:
        """Score PHQ-9 depression questionnaire."""
        self._stats["phq9_scored"] += 1
        result = QuestionnaireResult(questionnaire="PHQ-9", max_score=27)
        item_scores: dict[str, int] = {}
        for item_id in self._phq9_items:
            if item_id in responses:
                score = min(max(responses[item_id], 0), 3)
                item_scores[item_id] = score
                result.items_answered += 1
        if result.items_answered >= self._settings.min_items_for_assessment:
            if result.items_answered < len(self._phq9_items):
                mean_score = sum(item_scores.values()) / result.items_answered
                for item_id in self._phq9_items:
                    if item_id not in item_scores:
                        item_scores[item_id] = int(Decimal(str(mean_score)).quantize(Decimal("1"), rounding="ROUND_HALF_UP"))
                        result.items_imputed += 1
        result.item_scores = item_scores
        result.total_score = sum(item_scores.values())
        result.severity_level = self._interpret_phq9(result.total_score)
        result.interpretation = self._get_phq9_interpretation(result.total_score)
        result.recommendations = self._get_phq9_recommendations(result.severity_level)
        return result

    def _score_gad7(self, responses: dict[str, int]) -> QuestionnaireResult:
        """Score GAD-7 anxiety questionnaire."""
        self._stats["gad7_scored"] += 1
        result = QuestionnaireResult(questionnaire="GAD-7", max_score=21)
        item_scores: dict[str, int] = {}
        for item_id in self._gad7_items:
            if item_id in responses:
                score = min(max(responses[item_id], 0), 3)
                item_scores[item_id] = score
                result.items_answered += 1
        if result.items_answered >= self._settings.min_items_for_assessment:
            if result.items_answered < len(self._gad7_items):
                mean_score = sum(item_scores.values()) / result.items_answered
                for item_id in self._gad7_items:
                    if item_id not in item_scores:
                        item_scores[item_id] = int(Decimal(str(mean_score)).quantize(Decimal("1"), rounding="ROUND_HALF_UP"))
                        result.items_imputed += 1
        result.item_scores = item_scores
        result.total_score = sum(item_scores.values())
        result.severity_level = self._interpret_gad7(result.total_score)
        result.interpretation = self._get_gad7_interpretation(result.total_score)
        result.recommendations = self._get_gad7_recommendations(result.severity_level)
        return result

    def _score_phq15(self, responses: dict[str, int]) -> QuestionnaireResult:
        """Score PHQ-15 somatic symptom questionnaire."""
        result = QuestionnaireResult(questionnaire="PHQ-15", max_score=30)
        item_scores: dict[str, int] = {}
        for item_id in self._phq15_items:
            if item_id in responses:
                score = min(max(responses[item_id], 0), 2)
                item_scores[item_id] = score
                result.items_answered += 1
        result.item_scores = item_scores
        result.total_score = sum(item_scores.values())
        result.severity_level = self._interpret_phq15(result.total_score)
        result.interpretation = self._get_phq15_interpretation(result.total_score)
        return result

    def _score_pcl5(self, responses: dict[str, int]) -> QuestionnaireResult:
        """Score PCL-5 PTSD checklist."""
        result = QuestionnaireResult(questionnaire="PCL-5", max_score=40)
        item_scores: dict[str, int] = {}
        for item_id in self._pcl5_items:
            if item_id in responses:
                score = min(max(responses[item_id], 0), 4)
                item_scores[item_id] = score
                result.items_answered += 1
        result.item_scores = item_scores
        result.total_score = sum(item_scores.values())
        result.severity_level = self._interpret_pcl5(result.total_score)
        result.interpretation = self._get_pcl5_interpretation(result.total_score)
        return result

    def _score_cssrs(self, responses: dict[str, int]) -> QuestionnaireResult:
        """Score C-SSRS suicidality assessment."""
        result = QuestionnaireResult(questionnaire="C-SSRS", max_score=5)
        max_level = 0
        item_scores: dict[str, int] = {}
        for item_id, item in self._cssrs_items.items():
            if item_id in responses and responses[item_id] > 0:
                item_scores[item_id] = responses[item_id]
                max_level = max(max_level, item["level"])
                result.items_answered += 1
        result.item_scores = item_scores
        result.total_score = max_level
        result.severity_level = self._interpret_cssrs(max_level)
        result.interpretation = self._get_cssrs_interpretation(max_level)
        result.recommendations = self._get_cssrs_recommendations(max_level)
        return result

    def _score_audit(self, responses: dict[str, int]) -> QuestionnaireResult:
        """Score AUDIT alcohol screening."""
        result = QuestionnaireResult(questionnaire="AUDIT", max_score=40)
        item_scores: dict[str, int] = {}
        for item_id in self._audit_items:
            if item_id in responses:
                score = min(max(responses[item_id], 0), 4)
                item_scores[item_id] = score
                result.items_answered += 1
        result.item_scores = item_scores
        result.total_score = sum(item_scores.values())
        result.severity_level = self._interpret_audit(result.total_score)
        result.interpretation = self._get_audit_interpretation(result.total_score)
        return result

    def _score_mdq(self, responses: dict[str, int]) -> QuestionnaireResult:
        """Score MDQ bipolar screening."""
        result = QuestionnaireResult(questionnaire="MDQ", max_score=15)
        item_scores: dict[str, int] = {}
        for item_id in self._mdq_items:
            if item_id in responses:
                score = min(max(responses[item_id], 0), 1)
                item_scores[item_id] = score
                result.items_answered += 1
        result.item_scores = item_scores
        result.total_score = sum(item_scores.values())
        yes_count = sum(1 for v in list(item_scores.values())[:13] if v > 0)
        concurrent = item_scores.get("mdq_14", 0) > 0
        impairment = item_scores.get("mdq_15", 0) > 0
        positive_screen = yes_count >= 7 and concurrent and impairment
        result.severity_level = SeverityLevel.MODERATE if positive_screen else SeverityLevel.MINIMAL
        result.interpretation = "Positive bipolar screen - clinical evaluation recommended" if positive_screen else "Negative bipolar screen"
        return result

    def _score_asrs(self, responses: dict[str, int]) -> QuestionnaireResult:
        """Score ASRS ADHD screening."""
        result = QuestionnaireResult(questionnaire="ASRS", max_score=24)
        item_scores: dict[str, int] = {}
        for item_id in self._asrs_items:
            if item_id in responses:
                score = min(max(responses[item_id], 0), 4)
                item_scores[item_id] = score
                result.items_answered += 1
        result.item_scores = item_scores
        result.total_score = sum(item_scores.values())
        result.severity_level = self._interpret_asrs(result.total_score)
        result.interpretation = self._get_asrs_interpretation(result.total_score)
        return result

    def _interpret_phq9(self, score: int) -> SeverityLevel:
        """Interpret PHQ-9 score to severity level."""
        if score >= 20:
            return SeverityLevel.SEVERE
        if score >= 15:
            return SeverityLevel.MODERATELY_SEVERE
        if score >= 10:
            return SeverityLevel.MODERATE
        if score >= 5:
            return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    def _interpret_gad7(self, score: int) -> SeverityLevel:
        """Interpret GAD-7 score to severity level."""
        if score >= 15:
            return SeverityLevel.SEVERE
        if score >= 10:
            return SeverityLevel.MODERATE
        if score >= 5:
            return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    def _interpret_phq15(self, score: int) -> SeverityLevel:
        """Interpret PHQ-15 score to severity level."""
        if score >= 15: return SeverityLevel.SEVERE
        if score >= 10: return SeverityLevel.MODERATE
        if score >= 5: return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    def _interpret_pcl5(self, score: int) -> SeverityLevel:
        """Interpret PCL-5 score to severity level (10-item version, max_score=40)."""
        if score >= 31: return SeverityLevel.SEVERE
        if score >= 22: return SeverityLevel.MODERATE
        if score >= 17: return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    def _get_phq9_interpretation(self, score: int) -> str:
        """Get PHQ-9 score interpretation."""
        if score >= 20:
            return "Severe depression - immediate clinical attention recommended"
        if score >= 15:
            return "Moderately severe depression - active treatment warranted"
        if score >= 10:
            return "Moderate depression - treatment plan should be considered"
        if score >= 5:
            return "Mild depression - watchful waiting, repeat assessment"
        return "Minimal symptoms - monitor if clinically indicated"

    def _get_gad7_interpretation(self, score: int) -> str:
        """Get GAD-7 score interpretation."""
        if score >= 15:
            return "Severe anxiety - clinical intervention recommended"
        if score >= 10:
            return "Moderate anxiety - further evaluation and treatment consideration"
        if score >= 5:
            return "Mild anxiety - monitoring recommended"
        return "Minimal anxiety symptoms"

    def _get_phq15_interpretation(self, score: int) -> str:
        """Get PHQ-15 interpretation."""
        if score >= 15:
            return "High somatic symptom severity"
        if score >= 10:
            return "Medium somatic symptom severity"
        if score >= 5:
            return "Low somatic symptom severity"
        return "Minimal somatic symptoms"

    def _get_pcl5_interpretation(self, score: int) -> str:
        """Get PCL-5 interpretation."""
        return "Probable PTSD diagnosis - clinical evaluation recommended" if score >= 17 else "Below clinical threshold for PTSD"

    def _get_phq9_recommendations(self, severity: SeverityLevel) -> list[str]:
        """Get PHQ-9 based recommendations."""
        if severity == SeverityLevel.SEVERE:
            return ["Immediate psychiatric evaluation", "Consider medication and psychotherapy", "Safety assessment"]
        if severity == SeverityLevel.MODERATELY_SEVERE:
            return ["Active treatment with psychotherapy and/or medication", "Regular monitoring"]
        if severity == SeverityLevel.MODERATE:
            return ["Treatment plan consideration", "Psychoeducation", "Follow-up assessment"]
        if severity == SeverityLevel.MILD:
            return ["Watchful waiting", "Support and psychoeducation", "Reassess in 2-4 weeks"]
        return ["Continue monitoring if clinically indicated"]

    def _get_gad7_recommendations(self, severity: SeverityLevel) -> list[str]:
        """Get GAD-7 based recommendations."""
        if severity == SeverityLevel.SEVERE:
            return ["Clinical intervention recommended", "Consider CBT and/or medication"]
        if severity == SeverityLevel.MODERATE:
            return ["Further evaluation warranted", "Consider psychotherapy"]
        if severity == SeverityLevel.MILD:
            return ["Monitoring recommended", "Psychoeducation about anxiety management"]
        return ["Continue routine care"]

    def _interpret_cssrs(self, level: int) -> SeverityLevel:
        if level >= 4: return SeverityLevel.SEVERE
        if level >= 3: return SeverityLevel.MODERATELY_SEVERE
        if level >= 2: return SeverityLevel.MODERATE
        if level >= 1: return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    def _get_cssrs_interpretation(self, level: int) -> str:
        interpretations = {
            0: "No suicidal ideation",
            1: "Wish to be dead - passive ideation",
            2: "Non-specific active suicidal thoughts",
            3: "Active ideation with methods, no intent",
            4: "Active ideation with intent, no plan",
            5: "Active ideation with specific plan and intent - IMMEDIATE RISK",
        }
        return interpretations.get(level, "Unknown level")

    def _get_cssrs_recommendations(self, level: int) -> list[str]:
        if level >= 4: return ["Immediate safety assessment", "Consider hospitalization", "Remove access to means", "Contact emergency services if needed"]
        if level >= 3: return ["Urgent clinical evaluation", "Safety planning", "Increase monitoring frequency"]
        if level >= 2: return ["Clinical evaluation recommended", "Develop safety plan", "Regular follow-up"]
        if level >= 1: return ["Monitor and reassess", "Explore protective factors"]
        return ["Continue routine assessment"]

    def _interpret_audit(self, score: int) -> SeverityLevel:
        if score >= 20: return SeverityLevel.SEVERE
        if score >= 16: return SeverityLevel.MODERATELY_SEVERE
        if score >= 8: return SeverityLevel.MODERATE
        if score >= 1: return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    def _get_audit_interpretation(self, score: int) -> str:
        if score >= 20: return "Zone IV - Possible dependence, referral for evaluation"
        if score >= 16: return "Zone III - Harmful/hazardous drinking, brief counseling + monitoring"
        if score >= 8: return "Zone II - Hazardous drinking, simple advice recommended"
        if score >= 1: return "Zone I - Low risk drinking"
        return "No alcohol use indicated"

    def _interpret_asrs(self, score: int) -> SeverityLevel:
        if score >= 17: return SeverityLevel.SEVERE
        if score >= 14: return SeverityLevel.MODERATE
        if score >= 9: return SeverityLevel.MILD
        return SeverityLevel.MINIMAL

    def _get_asrs_interpretation(self, score: int) -> str:
        if score >= 14: return "Highly consistent with ADHD - clinical evaluation recommended"
        if score >= 9: return "Consistent with ADHD - further screening recommended"
        return "Below ADHD screening threshold"

    def _calculate_overall_severity(self, result: SeverityAssessmentResult) -> SeverityLevel:
        """Calculate overall severity from individual assessments."""
        severities: list[SeverityLevel] = []
        if result.depression_severity:
            severities.append(result.depression_severity.severity_level)
        if result.anxiety_severity:
            severities.append(result.anxiety_severity.severity_level)
        if result.somatic_severity:
            severities.append(result.somatic_severity.severity_level)
        if result.trauma_severity:
            severities.append(result.trauma_severity.severity_level)
        if result.suicidality_severity:
            severities.append(result.suicidality_severity.severity_level)
        if result.alcohol_severity:
            severities.append(result.alcohol_severity.severity_level)
        if result.bipolar_screen:
            severities.append(result.bipolar_screen.severity_level)
        if result.adhd_screen:
            severities.append(result.adhd_screen.severity_level)
        if not severities:
            return SeverityLevel.MINIMAL
        severity_order = [SeverityLevel.MINIMAL, SeverityLevel.MILD, SeverityLevel.MODERATE,
                         SeverityLevel.MODERATELY_SEVERE, SeverityLevel.SEVERE]
        return max(severities, key=lambda s: severity_order.index(s))

    def _calculate_composite_score(self, result: SeverityAssessmentResult) -> float:
        """Calculate composite severity score."""
        scores: list[float] = []
        if result.depression_severity:
            scores.append(result.depression_severity.total_score / result.depression_severity.max_score)
        if result.anxiety_severity:
            scores.append(result.anxiety_severity.total_score / result.anxiety_severity.max_score)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _assess_functional_impairment(self, result: SeverityAssessmentResult) -> str:
        """Assess functional impairment level."""
        if result.overall_severity == SeverityLevel.SEVERE:
            return "severe"
        if result.overall_severity in [SeverityLevel.MODERATE, SeverityLevel.MODERATELY_SEVERE]:
            return "moderate"
        if result.overall_severity == SeverityLevel.MILD:
            return "mild"
        return "none"

    def get_statistics(self) -> dict[str, int]:
        """Get assessor statistics."""
        return self._stats.copy()
