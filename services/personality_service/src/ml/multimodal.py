"""
Solace-AI Personality Service - Multimodal Fusion.
Late fusion multimodal analysis combining text, LIWC, and LLM personality detection.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog
from ..schemas import PersonalityTrait, AssessmentSource, OceanScoresDTO, TraitScoreDTO

logger = structlog.get_logger(__name__)


class FusionStrategy(str, Enum):
    """Multimodal fusion strategies."""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE = "adaptive"
    MAX_CONFIDENCE = "max_confidence"
    BAYESIAN = "bayesian"


class ModalityType(str, Enum):
    """Types of input modalities."""
    TEXT_ROBERTA = "text_roberta"
    TEXT_LLM = "text_llm"
    LIWC = "liwc"
    VOICE = "voice"
    BEHAVIORAL = "behavioral"


class MultimodalFusionSettings(BaseSettings):
    """Multimodal fusion configuration."""
    fusion_strategy: FusionStrategy = Field(default=FusionStrategy.CONFIDENCE_WEIGHTED)
    roberta_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    llm_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    liwc_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    min_modalities: int = Field(default=1, ge=1, le=5)
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    disagreement_threshold: float = Field(default=0.3, ge=0.0, le=0.5)
    temporal_decay_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    enable_disagreement_handling: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_FUSION_", env_file=".env", extra="ignore")


@dataclass
class ModalityResult:
    """Result from a single modality."""
    result_id: UUID = field(default_factory=uuid4)
    modality: ModalityType = ModalityType.TEXT_ROBERTA
    source: AssessmentSource = AssessmentSource.TEXT_ANALYSIS
    scores: dict[PersonalityTrait, float] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self, threshold: float) -> bool:
        return self.confidence >= threshold and len(self.scores) == 5


@dataclass
class FusionResult:
    """Result of multimodal fusion."""
    fusion_id: UUID = field(default_factory=uuid4)
    fused_scores: dict[PersonalityTrait, float] = field(default_factory=dict)
    overall_confidence: float = 0.5
    modalities_used: list[ModalityType] = field(default_factory=list)
    disagreement_flags: list[str] = field(default_factory=list)
    fusion_strategy: FusionStrategy = FusionStrategy.CONFIDENCE_WEIGHTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WeightCalculator:
    """Calculates weights for multimodal fusion."""
    _BAYESIAN_PRIORS = {ModalityType.TEXT_ROBERTA: 0.7, ModalityType.TEXT_LLM: 0.65, ModalityType.LIWC: 0.6, ModalityType.VOICE: 0.5, ModalityType.BEHAVIORAL: 0.55}

    def __init__(self, settings: MultimodalFusionSettings) -> None:
        self._settings = settings
        self._base_weights = {ModalityType.TEXT_ROBERTA: settings.roberta_weight, ModalityType.TEXT_LLM: settings.llm_weight, ModalityType.LIWC: settings.liwc_weight, ModalityType.VOICE: 0.0, ModalityType.BEHAVIORAL: 0.0}

    def compute_weights(self, results: list[ModalityResult], strategy: FusionStrategy) -> dict[ModalityType, float]:
        strategies = {FusionStrategy.WEIGHTED_AVERAGE: self._weighted_average_weights, FusionStrategy.CONFIDENCE_WEIGHTED: self._confidence_weighted,
                      FusionStrategy.ADAPTIVE: self._adaptive_weights, FusionStrategy.MAX_CONFIDENCE: self._max_confidence_weights, FusionStrategy.BAYESIAN: self._bayesian_weights}
        return strategies.get(strategy, self._weighted_average_weights)(results)

    def _weighted_average_weights(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        weights = {r.modality: self._base_weights.get(r.modality, 0.3) for r in results}
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()} if total > 0 else weights

    def _confidence_weighted(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        weights = {r.modality: self._base_weights.get(r.modality, 0.3) * r.confidence for r in results}
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()} if total > 0 else weights

    def _adaptive_weights(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        if len(results) < 2: return self._confidence_weighted(results)
        weights = {r.modality: self._base_weights.get(r.modality, 0.3) * r.confidence * self._compute_consistency(r, results) for r in results}
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()} if total > 0 else weights

    def _compute_consistency(self, target: ModalityResult, all_results: list[ModalityResult]) -> float:
        if len(all_results) < 2: return 1.0
        deviations = []
        for other in all_results:
            if other.result_id != target.result_id:
                trait_diffs = [abs(target.scores.get(t, 0.5) - other.scores.get(t, 0.5)) for t in PersonalityTrait]
                deviations.append(sum(trait_diffs) / len(trait_diffs))
        return max(0.2, 1.0 - (sum(deviations) / len(deviations) if deviations else 0.0))

    def _max_confidence_weights(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        if not results: return {}
        best = max(results, key=lambda r: r.confidence)
        return {r.modality: 1.0 if r.result_id == best.result_id else 0.0 for r in results}

    def _bayesian_weights(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        weights = {r.modality: self._BAYESIAN_PRIORS.get(r.modality, 0.5) * r.confidence for r in results}
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()} if total > 0 else weights


class DisagreementHandler:
    """Handles disagreement between modalities."""
    def __init__(self, settings: MultimodalFusionSettings) -> None:
        self._settings = settings

    def detect_disagreements(self, results: list[ModalityResult]) -> list[str]:
        if len(results) < 2: return []
        disagreements = []
        for trait in PersonalityTrait:
            scores = [r.scores.get(trait, 0.5) for r in results]
            score_range = max(scores) - min(scores)
            if score_range > self._settings.disagreement_threshold:
                disagreements.append(f"{trait.value}_disagreement:{score_range:.2f}:{','.join(r.modality.value for r in results)}")
        return disagreements

    def resolve_disagreement(self, results: list[ModalityResult], trait: PersonalityTrait) -> float:
        scores_with_confidence = [(r.scores.get(trait, 0.5), r.confidence) for r in results]
        total_confidence = sum(c for _, c in scores_with_confidence)
        return sum(s * c for s, c in scores_with_confidence) / total_confidence if total_confidence > 0 else 0.5


class FusionEngine:
    """Core fusion engine for combining modality results."""
    def __init__(self, weight_calculator: WeightCalculator, disagreement_handler: DisagreementHandler) -> None:
        self._weight_calculator = weight_calculator
        self._disagreement_handler = disagreement_handler

    def fuse(self, results: list[ModalityResult], strategy: FusionStrategy) -> FusionResult:
        if not results:
            return FusionResult(fused_scores={trait: 0.5 for trait in PersonalityTrait}, overall_confidence=0.2, modalities_used=[], fusion_strategy=strategy)
        weights = self._weight_calculator.compute_weights(results, strategy)
        return FusionResult(fused_scores=self._compute_fused_scores(results, weights), overall_confidence=self._compute_overall_confidence(results, weights),
                           modalities_used=[r.modality for r in results], disagreement_flags=self._disagreement_handler.detect_disagreements(results), fusion_strategy=strategy)

    def _compute_fused_scores(self, results: list[ModalityResult], weights: dict[ModalityType, float]) -> dict[PersonalityTrait, float]:
        fused = {}
        for trait in PersonalityTrait:
            weighted_sum = sum(r.scores.get(trait, 0.5) * weights.get(r.modality, 0.0) for r in results)
            total_weight = sum(weights.get(r.modality, 0.0) for r in results)
            fused[trait] = weighted_sum / total_weight if total_weight > 0 else 0.5
        return fused

    def _compute_overall_confidence(self, results: list[ModalityResult], weights: dict[ModalityType, float]) -> float:
        weighted_conf = sum(r.confidence * weights.get(r.modality, 0.0) for r in results)
        total_weight = sum(weights.get(r.modality, 0.0) for r in results)
        base_conf = weighted_conf / total_weight if total_weight > 0 else 0.3
        return min(0.95, base_conf + min(0.2, len(results) * 0.05))


class MultimodalFusion:
    """Main multimodal fusion orchestrator."""

    def __init__(self, settings: MultimodalFusionSettings | None = None) -> None:
        self._settings = settings or MultimodalFusionSettings()
        self._weight_calculator = WeightCalculator(self._settings)
        self._disagreement_handler = DisagreementHandler(self._settings)
        self._fusion_engine = FusionEngine(self._weight_calculator, self._disagreement_handler)
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("multimodal_fusion_initialized", strategy=self._settings.fusion_strategy.value, min_modalities=self._settings.min_modalities)

    async def fuse(self, roberta_scores: OceanScoresDTO | None = None, llm_scores: OceanScoresDTO | None = None, liwc_scores: OceanScoresDTO | None = None, strategy: FusionStrategy | None = None) -> OceanScoresDTO:
        results: list[ModalityResult] = []
        if roberta_scores: results.append(self._convert_to_modality_result(roberta_scores, ModalityType.TEXT_ROBERTA, AssessmentSource.TEXT_ANALYSIS))
        if llm_scores: results.append(self._convert_to_modality_result(llm_scores, ModalityType.TEXT_LLM, AssessmentSource.LLM_ZERO_SHOT))
        if liwc_scores: results.append(self._convert_to_modality_result(liwc_scores, ModalityType.LIWC, AssessmentSource.LIWC_FEATURES))
        valid_results = [r for r in results if r.is_valid(self._settings.confidence_threshold)]
        if len(valid_results) < self._settings.min_modalities:
            valid_results = results[:self._settings.min_modalities] if results else []
        if not valid_results:
            logger.warning("no_valid_modality_results")
            return self._neutral_scores()
        fusion_result = self._fusion_engine.fuse(valid_results, strategy or self._settings.fusion_strategy)
        if fusion_result.disagreement_flags:
            logger.info("modality_disagreements_detected", flags=fusion_result.disagreement_flags)
        return self._fusion_result_to_scores(fusion_result)

    async def fuse_with_metadata(self, modality_results: list[ModalityResult], strategy: FusionStrategy | None = None) -> tuple[OceanScoresDTO, FusionResult]:
        fusion_result = self._fusion_engine.fuse(modality_results, strategy or self._settings.fusion_strategy)
        return self._fusion_result_to_scores(fusion_result), fusion_result

    def _convert_to_modality_result(self, scores: OceanScoresDTO, modality: ModalityType, source: AssessmentSource) -> ModalityResult:
        return ModalityResult(modality=modality, source=source, confidence=scores.overall_confidence, timestamp=scores.assessed_at,
                             scores={PersonalityTrait.OPENNESS: scores.openness, PersonalityTrait.CONSCIENTIOUSNESS: scores.conscientiousness,
                                     PersonalityTrait.EXTRAVERSION: scores.extraversion, PersonalityTrait.AGREEABLENESS: scores.agreeableness, PersonalityTrait.NEUROTICISM: scores.neuroticism})

    def _fusion_result_to_scores(self, result: FusionResult) -> OceanScoresDTO:
        return OceanScoresDTO(openness=result.fused_scores.get(PersonalityTrait.OPENNESS, 0.5), conscientiousness=result.fused_scores.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5),
                             extraversion=result.fused_scores.get(PersonalityTrait.EXTRAVERSION, 0.5), agreeableness=result.fused_scores.get(PersonalityTrait.AGREEABLENESS, 0.5),
                             neuroticism=result.fused_scores.get(PersonalityTrait.NEUROTICISM, 0.5), overall_confidence=result.overall_confidence, trait_scores=self._build_trait_scores(result))

    def _build_trait_scores(self, result: FusionResult) -> list[TraitScoreDTO]:
        margin = 0.1 * (1 - result.overall_confidence)
        evidence = [f"fusion_{m.value}" for m in result.modalities_used][:3]
        return [TraitScoreDTO(trait=trait, value=value, confidence_lower=max(0.0, value - margin), confidence_upper=min(1.0, value + margin),
                              sample_count=len(result.modalities_used), evidence_markers=evidence) for trait, value in result.fused_scores.items()]

    def _neutral_scores(self) -> OceanScoresDTO:
        return OceanScoresDTO(openness=0.5, conscientiousness=0.5, extraversion=0.5, agreeableness=0.5, neuroticism=0.5, overall_confidence=0.2, trait_scores=[])

    async def shutdown(self) -> None:
        self._initialized = False
        logger.info("multimodal_fusion_shutdown")
