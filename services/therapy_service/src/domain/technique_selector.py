"""
Solace-AI Therapy Service - Therapeutic Technique Selection.
Multi-stage evidence-based technique selection with personalization and contraindication checking.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import TherapyModality, SeverityLevel, RiskLevel, TechniqueDTO, SessionPhase

logger = structlog.get_logger(__name__)


class TechniqueSelectorSettings(BaseSettings):
    """Technique selector configuration."""
    enable_personalization: bool = Field(default=True)
    recency_penalty_weight: float = Field(default=0.15)
    min_confidence_threshold: float = Field(default=0.6)
    max_techniques_per_session: int = Field(default=3)
    enable_strict_contraindications: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="TECHNIQUE_SELECTOR_", env_file=".env", extra="ignore")


class TechniqueSelector:
    """
    Selects appropriate therapeutic techniques using 4-stage pipeline.

    Pipeline stages:
    1. Clinical Filter: Diagnosis and severity-based candidate selection
    2. Personalization: User preferences, personality, and history scoring
    3. Context Ranking: Session phase, treatment plan progress
    4. Final Selection: Combined scoring with contraindication validation
    """

    def __init__(self, settings: TechniqueSelectorSettings | None = None) -> None:
        self._settings = settings or TechniqueSelectorSettings()
        self._technique_library = self._initialize_technique_library()
        self._technique_usage_history: dict[UUID, list[dict[str, Any]]] = {}
        logger.info("technique_selector_initialized", technique_count=len(self._technique_library))

    def _initialize_technique_library(self) -> dict[UUID, TechniqueDTO]:
        """Initialize library of therapeutic techniques."""
        specs = [
            ("Thought Record", TherapyModality.CBT, "cognitive_restructuring", "Identify and challenge automatic negative thoughts", 15, True, ["severe_cognitive_impairment"]),
            ("Behavioral Activation", TherapyModality.CBT, "behavioral", "Schedule and engage in mood-boosting activities", 12, True, ["severe_depression_immobile"]),
            ("Exposure Hierarchy", TherapyModality.CBT, "exposure", "Gradual exposure to feared situations", 20, True, ["high_suicide_risk", "acute_psychosis"]),
            ("Socratic Questioning", TherapyModality.CBT, "cognitive_restructuring", "Guided questioning to examine thought validity", 10, False, []),
            ("Mindfulness of Breath", TherapyModality.MINDFULNESS, "grounding", "Focus attention on breath to anchor present awareness", 8, True, ["trauma_dissociation"]),
            ("STOP Skill", TherapyModality.DBT, "distress_tolerance", "Stop, Take a step back, Observe, Proceed mindfully", 5, True, []),
            ("DEAR MAN", TherapyModality.DBT, "interpersonal_effectiveness", "Assertive communication framework", 15, True, ["acute_agitation"]),
            ("Radical Acceptance", TherapyModality.DBT, "distress_tolerance", "Accept reality without judgment", 12, False, ["severe_avoidance"]),
            ("Values Clarification", TherapyModality.ACT, "values", "Identify core personal values and life directions", 18, True, []),
            ("Cognitive Defusion", TherapyModality.ACT, "cognitive", "Separate self from thoughts to reduce their impact", 10, True, ["severe_cognitive_impairment"]),
            ("Committed Action", TherapyModality.ACT, "behavioral", "Take values-consistent action despite discomfort", 15, True, ["severe_depression_immobile"]),
            ("Motivational Interviewing", TherapyModality.MI, "engagement", "Explore and resolve ambivalence about change", 20, False, []),
            ("Change Talk Elicitation", TherapyModality.MI, "engagement", "Evoke client's own arguments for change", 12, False, []),
            ("Body Scan", TherapyModality.MINDFULNESS, "awareness", "Systematic attention to body sensations", 15, True, ["trauma_dissociation", "acute_panic"]),
            ("5-4-3-2-1 Grounding", TherapyModality.MINDFULNESS, "grounding", "Sensory grounding technique for anxiety", 5, False, []),
        ]
        techniques = [TechniqueDTO(technique_id=uuid4(), name=s[0], modality=s[1], category=s[2],
                                    description=s[3], duration_minutes=s[4], requires_homework=s[5],
                                    contraindications=s[6]) for s in specs]
        return {t.technique_id: t for t in techniques}

    async def select_technique(
        self,
        user_id: UUID,
        diagnosis: str,
        severity: SeverityLevel,
        modality: TherapyModality,
        session_phase: SessionPhase,
        user_context: dict[str, Any],
        treatment_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Select optimal technique using 4-stage pipeline.

        Args:
            user_id: User identifier
            diagnosis: Primary diagnosis
            severity: Symptom severity level
            modality: Primary therapy modality
            session_phase: Current session phase
            user_context: User preferences and history
            treatment_plan: Treatment plan details

        Returns:
            Dictionary with selected technique, alternatives, and reasoning
        """
        logger.debug("technique_selection_started", user_id=str(user_id), modality=modality.value)

        stage1_candidates = self._stage1_clinical_filter(diagnosis, severity, modality)
        if not stage1_candidates:
            logger.warning("no_clinical_candidates", diagnosis=diagnosis, severity=severity.value)
            return {"selected": None, "alternatives": [], "reasoning": "No suitable techniques for criteria"}

        stage2_scores = self._stage2_personalization(stage1_candidates, user_id, user_context)

        stage3_ranked = self._stage3_context_ranking(stage2_scores, session_phase, treatment_plan)

        stage4_result = self._stage4_final_selection(stage3_ranked, user_context)

        if stage4_result["selected"]:
            self._record_technique_usage(user_id, stage4_result["selected"])

        logger.info(
            "technique_selected",
            user_id=str(user_id),
            technique=stage4_result["selected"].name if stage4_result["selected"] else None,
            alternatives_count=len(stage4_result["alternatives"])
        )

        return stage4_result

    def _stage1_clinical_filter(
        self,
        diagnosis: str,
        severity: SeverityLevel,
        modality: TherapyModality,
    ) -> list[TechniqueDTO]:
        """Stage 1: Filter techniques by diagnosis, severity, and modality."""
        candidates = []
        condition_lower = diagnosis.lower()
        condition_categories = {
            "depression": ["behavioral", "cognitive_restructuring", "values"],
            "anxiety": ["exposure", "grounding", "cognitive_restructuring", "distress_tolerance"],
            "stress": ["distress_tolerance", "grounding", "awareness"],
            "trauma": ["grounding", "distress_tolerance", "cognitive_restructuring"],
        }
        for technique in self._technique_library.values():
            if technique.modality != modality and modality != TherapyModality.MINDFULNESS:
                continue
            if modality == TherapyModality.MINDFULNESS and technique.modality == TherapyModality.MINDFULNESS:
                candidates.append(technique)
                continue
            for key, cats in condition_categories.items():
                if key in condition_lower and technique.category in cats:
                    if key == "trauma" and "trauma_dissociation" in technique.contraindications:
                        continue
                    candidates.append(technique)
                    break
            else:
                if modality == technique.modality:
                    candidates.append(technique)
        if severity in [SeverityLevel.SEVERE, SeverityLevel.MODERATELY_SEVERE]:
            candidates = [t for t in candidates if t.duration_minutes <= 12]
        return candidates

    def _stage2_personalization(
        self,
        candidates: list[TechniqueDTO],
        user_id: UUID,
        user_context: dict[str, Any],
    ) -> dict[UUID, float]:
        """Stage 2: Score techniques based on user personalization."""
        scores = {}
        preferences = user_context.get("preferences", {})
        personality = user_context.get("personality", {})
        history = self._technique_usage_history.get(user_id, [])
        for technique in candidates:
            score = 0.5
            if preferences.get("preferred_modalities") and technique.modality.value in preferences["preferred_modalities"]:
                score += 0.15
            if personality.get("openness", 0) > 0.7 and technique.category in ["values", "cognitive"]:
                score += 0.1
            if personality.get("conscientiousness", 0) > 0.7 and technique.requires_homework:
                score += 0.1
            recent_uses = [h for h in history if h["technique_id"] == technique.technique_id]
            if recent_uses:
                days_since = (datetime.now(timezone.utc) - recent_uses[-1]["timestamp"]).days
                if days_since < 7:
                    score -= self._settings.recency_penalty_weight * (7 - days_since) / 7
            if user_context.get("current_risk", RiskLevel.NONE) in [RiskLevel.HIGH, RiskLevel.IMMINENT]:
                if technique.category in ["grounding", "distress_tolerance"]:
                    score += 0.2
            scores[technique.technique_id] = max(0.0, min(1.0, score))
        return scores

    def _stage3_context_ranking(
        self,
        scored_techniques: dict[UUID, float],
        session_phase: SessionPhase,
        treatment_plan: dict[str, Any],
    ) -> list[tuple[TechniqueDTO, float]]:
        """Stage 3: Rank techniques by session context."""
        ranked = []
        current_phase_int = treatment_plan.get("current_phase", 1)
        skills_acquired = treatment_plan.get("skills_acquired", [])
        for technique_id, base_score in scored_techniques.items():
            technique = self._technique_library[technique_id]
            context_score = base_score
            if session_phase == SessionPhase.OPENING and technique.duration_minutes <= 10:
                context_score += 0.1
            elif session_phase == SessionPhase.WORKING and 10 <= technique.duration_minutes <= 20:
                context_score += 0.15
            elif session_phase == SessionPhase.CLOSING and technique.duration_minutes <= 8:
                context_score += 0.1
            phase_cats = {1: ["engagement", "grounding", "awareness"], 2: ["cognitive_restructuring", "behavioral", "exposure"], 3: ["values", "behavioral"]}
            if technique.category in phase_cats.get(current_phase_int, []):
                context_score += 0.15 if current_phase_int <= 2 else 0.1
            if technique.name in skills_acquired:
                context_score += 0.05
            ranked.append((technique, max(0.0, min(1.0, context_score))))
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def _stage4_final_selection(
        self,
        ranked_techniques: list[tuple[TechniqueDTO, float]],
        user_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Stage 4: Final selection with contraindication validation."""
        if not ranked_techniques:
            return {"selected": None, "alternatives": [], "reasoning": "No techniques available"}
        user_contraindications = user_context.get("contraindications", [])
        risk_level = user_context.get("current_risk", RiskLevel.NONE)
        selected = None
        alternatives = []
        reasoning_parts = []
        for technique, score in ranked_techniques:
            if score < self._settings.min_confidence_threshold:
                continue
            if self._settings.enable_strict_contraindications:
                if any(ci in technique.contraindications for ci in user_contraindications):
                    reasoning_parts.append(f"{technique.name} excluded: contraindication")
                    continue
            if risk_level in [RiskLevel.HIGH, RiskLevel.IMMINENT] and technique.category not in ["grounding", "distress_tolerance"]:
                reasoning_parts.append(f"{technique.name} skipped: requires crisis techniques")
                continue
            if selected is None:
                selected = technique
                reasoning_parts.append(f"Selected {technique.name} (score: {score:.2f})")
            elif len(alternatives) < 2:
                alternatives.append(technique)
                reasoning_parts.append(f"Alternative: {technique.name} (score: {score:.2f})")
            if selected and len(alternatives) >= 2:
                break
        return {
            "selected": selected, "alternatives": alternatives,
            "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "No suitable techniques found",
            "contraindications_checked": self._settings.enable_strict_contraindications,
        }

    def validate_contraindications(
        self,
        technique: TechniqueDTO,
        user_state: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Validate technique against user contraindications.

        Args:
            technique: Technique to validate
            user_state: Current user state

        Returns:
            Tuple of (is_safe, contraindications_found)
        """
        user_contraindications = user_state.get("contraindications", [])
        found = [ci for ci in technique.contraindications if ci in user_contraindications]

        is_safe = len(found) == 0
        return is_safe, found

    def _record_technique_usage(self, user_id: UUID, technique: TechniqueDTO) -> None:
        """Record technique usage for recency tracking."""
        if user_id not in self._technique_usage_history:
            self._technique_usage_history[user_id] = []

        self._technique_usage_history[user_id].append({
            "technique_id": technique.technique_id,
            "technique_name": technique.name,
            "timestamp": datetime.now(timezone.utc),
        })

        if len(self._technique_usage_history[user_id]) > 50:
            self._technique_usage_history[user_id] = self._technique_usage_history[user_id][-50:]

    def get_techniques_by_modality(self, modality: TherapyModality | None = None) -> list[TechniqueDTO]:
        """
        Get techniques filtered by modality.

        Args:
            modality: Modality to filter by, or None for all

        Returns:
            List of techniques
        """
        if modality is None:
            return list(self._technique_library.values())
        return [t for t in self._technique_library.values() if t.modality == modality]

    def get_technique_by_id(self, technique_id: UUID) -> TechniqueDTO | None:
        """Get technique by ID."""
        return self._technique_library.get(technique_id)
