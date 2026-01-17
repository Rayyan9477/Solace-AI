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
        """Check if result meets confidence threshold."""
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

    def __init__(self, settings: MultimodalFusionSettings) -> None:
        self._settings = settings
        self._base_weights = {
            ModalityType.TEXT_ROBERTA: settings.roberta_weight,
            ModalityType.TEXT_LLM: settings.llm_weight,
            ModalityType.LIWC: settings.liwc_weight,
            ModalityType.VOICE: 0.0,
            ModalityType.BEHAVIORAL: 0.0,
        }

    def compute_weights(
        self, results: list[ModalityResult], strategy: FusionStrategy
    ) -> dict[ModalityType, float]:
        """Compute fusion weights based on strategy."""
        if strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_weights(results)
        if strategy == FusionStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted(results)
        if strategy == FusionStrategy.ADAPTIVE:
            return self._adaptive_weights(results)
        if strategy == FusionStrategy.MAX_CONFIDENCE:
            return self._max_confidence_weights(results)
        if strategy == FusionStrategy.BAYESIAN:
            return self._bayesian_weights(results)
        return self._weighted_average_weights(results)

    def _weighted_average_weights(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        """Use predefined base weights."""
        weights = {}
        total = 0.0
        for r in results:
            w = self._base_weights.get(r.modality, 0.3)
            weights[r.modality] = w
            total += w
        if total > 0:
            for m in weights:
                weights[m] /= total
        return weights

    def _confidence_weighted(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        """Weight by confidence scores."""
        weights = {}
        total = 0.0
        for r in results:
            base_weight = self._base_weights.get(r.modality, 0.3)
            confidence_adjusted = base_weight * r.confidence
            weights[r.modality] = confidence_adjusted
            total += confidence_adjusted
        if total > 0:
            for m in weights:
                weights[m] /= total
        return weights

    def _adaptive_weights(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        """Adaptively weight based on consistency."""
        if len(results) < 2:
            return self._confidence_weighted(results)
        weights = {}
        for r in results:
            consistency_score = self._compute_consistency(r, results)
            base_weight = self._base_weights.get(r.modality, 0.3)
            weights[r.modality] = base_weight * r.confidence * consistency_score
        total = sum(weights.values())
        if total > 0:
            for m in weights:
                weights[m] /= total
        return weights

    def _compute_consistency(self, target: ModalityResult, all_results: list[ModalityResult]) -> float:
        """Compute consistency score for a modality."""
        if len(all_results) < 2:
            return 1.0
        deviations = []
        for other in all_results:
            if other.result_id == target.result_id:
                continue
            trait_diffs = []
            for trait in PersonalityTrait:
                t_score = target.scores.get(trait, 0.5)
                o_score = other.scores.get(trait, 0.5)
                trait_diffs.append(abs(t_score - o_score))
            deviations.append(sum(trait_diffs) / len(trait_diffs))
        avg_deviation = sum(deviations) / len(deviations) if deviations else 0.0
        return max(0.2, 1.0 - avg_deviation)

    def _max_confidence_weights(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        """Give full weight to highest confidence modality."""
        if not results:
            return {}
        best = max(results, key=lambda r: r.confidence)
        return {r.modality: 1.0 if r.result_id == best.result_id else 0.0 for r in results}

    def _bayesian_weights(self, results: list[ModalityResult]) -> dict[ModalityType, float]:
        """Bayesian-inspired weighting using prior reliability."""
        weights = {}
        priors = {
            ModalityType.TEXT_ROBERTA: 0.7,
            ModalityType.TEXT_LLM: 0.65,
            ModalityType.LIWC: 0.6,
            ModalityType.VOICE: 0.5,
            ModalityType.BEHAVIORAL: 0.55,
        }
        total = 0.0
        for r in results:
            prior = priors.get(r.modality, 0.5)
            posterior = prior * r.confidence
            weights[r.modality] = posterior
            total += posterior
        if total > 0:
            for m in weights:
                weights[m] /= total
        return weights


class DisagreementHandler:
    """Handles disagreement between modalities."""

    def __init__(self, settings: MultimodalFusionSettings) -> None:
        self._settings = settings

    def detect_disagreements(self, results: list[ModalityResult]) -> list[str]:
        """Detect disagreements between modalities."""
        if len(results) < 2:
            return []
        disagreements = []
        for trait in PersonalityTrait:
            scores = [r.scores.get(trait, 0.5) for r in results]
            score_range = max(scores) - min(scores)
            if score_range > self._settings.disagreement_threshold:
                modality_names = [r.modality.value for r in results]
                disagreements.append(f"{trait.value}_disagreement:{score_range:.2f}:{','.join(modality_names)}")
        return disagreements

    def resolve_disagreement(
        self, results: list[ModalityResult], trait: PersonalityTrait
    ) -> float:
        """Resolve disagreement for a specific trait."""
        scores_with_confidence = [(r.scores.get(trait, 0.5), r.confidence) for r in results]
        total_confidence = sum(c for _, c in scores_with_confidence)
        if total_confidence == 0:
            return 0.5
        weighted_sum = sum(s * c for s, c in scores_with_confidence)
        return weighted_sum / total_confidence


class FusionEngine:
    """Core fusion engine for combining modality results."""

    def __init__(
        self, weight_calculator: WeightCalculator, disagreement_handler: DisagreementHandler
    ) -> None:
        self._weight_calculator = weight_calculator
        self._disagreement_handler = disagreement_handler

    def fuse(
        self, results: list[ModalityResult], strategy: FusionStrategy
    ) -> FusionResult:
        """Fuse multiple modality results into a single assessment."""
        if not results:
            return FusionResult(
                fused_scores={trait: 0.5 for trait in PersonalityTrait},
                overall_confidence=0.2,
                modalities_used=[],
                fusion_strategy=strategy,
            )
        weights = self._weight_calculator.compute_weights(results, strategy)
        disagreements = self._disagreement_handler.detect_disagreements(results)
        fused_scores = self._compute_fused_scores(results, weights)
        overall_confidence = self._compute_overall_confidence(results, weights)
        return FusionResult(
            fused_scores=fused_scores,
            overall_confidence=overall_confidence,
            modalities_used=[r.modality for r in results],
            disagreement_flags=disagreements,
            fusion_strategy=strategy,
        )

    def _compute_fused_scores(
        self, results: list[ModalityResult], weights: dict[ModalityType, float]
    ) -> dict[PersonalityTrait, float]:
        """Compute fused trait scores."""
        fused = {trait: 0.0 for trait in PersonalityTrait}
        for trait in PersonalityTrait:
            weighted_sum = 0.0
            total_weight = 0.0
            for r in results:
                w = weights.get(r.modality, 0.0)
                score = r.scores.get(trait, 0.5)
                weighted_sum += score * w
                total_weight += w
            fused[trait] = weighted_sum / total_weight if total_weight > 0 else 0.5
        return fused

    def _compute_overall_confidence(
        self, results: list[ModalityResult], weights: dict[ModalityType, float]
    ) -> float:
        """Compute overall confidence from modality confidences."""
        weighted_conf = 0.0
        total_weight = 0.0
        for r in results:
            w = weights.get(r.modality, 0.0)
            weighted_conf += r.confidence * w
            total_weight += w
        base_conf = weighted_conf / total_weight if total_weight > 0 else 0.3
        modality_bonus = min(0.2, len(results) * 0.05)
        return min(0.95, base_conf + modality_bonus)


class MultimodalFusion:
    """Main multimodal fusion orchestrator."""

    def __init__(self, settings: MultimodalFusionSettings | None = None) -> None:
        self._settings = settings or MultimodalFusionSettings()
        self._weight_calculator = WeightCalculator(self._settings)
        self._disagreement_handler = DisagreementHandler(self._settings)
        self._fusion_engine = FusionEngine(self._weight_calculator, self._disagreement_handler)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the multimodal fusion system."""
        self._initialized = True
        logger.info(
            "multimodal_fusion_initialized",
            strategy=self._settings.fusion_strategy.value,
            min_modalities=self._settings.min_modalities,
        )

    async def fuse(
        self,
        roberta_scores: OceanScoresDTO | None = None,
        llm_scores: OceanScoresDTO | None = None,
        liwc_scores: OceanScoresDTO | None = None,
        strategy: FusionStrategy | None = None,
    ) -> OceanScoresDTO:
        """Fuse scores from multiple detection methods."""
        results: list[ModalityResult] = []
        if roberta_scores:
            results.append(self._convert_to_modality_result(
                roberta_scores, ModalityType.TEXT_ROBERTA, AssessmentSource.TEXT_ANALYSIS
            ))
        if llm_scores:
            results.append(self._convert_to_modality_result(
                llm_scores, ModalityType.TEXT_LLM, AssessmentSource.LLM_ZERO_SHOT
            ))
        if liwc_scores:
            results.append(self._convert_to_modality_result(
                liwc_scores, ModalityType.LIWC, AssessmentSource.LIWC_FEATURES
            ))
        valid_results = [
            r for r in results if r.is_valid(self._settings.confidence_threshold)
        ]
        if len(valid_results) < self._settings.min_modalities:
            valid_results = results[:self._settings.min_modalities] if results else []
        if not valid_results:
            logger.warning("no_valid_modality_results")
            return self._neutral_scores()
        fusion_strategy = strategy or self._settings.fusion_strategy
        fusion_result = self._fusion_engine.fuse(valid_results, fusion_strategy)
        if fusion_result.disagreement_flags:
            logger.info(
                "modality_disagreements_detected",
                flags=fusion_result.disagreement_flags,
            )
        return self._fusion_result_to_scores(fusion_result)

    async def fuse_with_metadata(
        self,
        modality_results: list[ModalityResult],
        strategy: FusionStrategy | None = None,
    ) -> tuple[OceanScoresDTO, FusionResult]:
        """Fuse with full metadata return."""
        fusion_strategy = strategy or self._settings.fusion_strategy
        fusion_result = self._fusion_engine.fuse(modality_results, fusion_strategy)
        scores = self._fusion_result_to_scores(fusion_result)
        return scores, fusion_result

    def _convert_to_modality_result(
        self, scores: OceanScoresDTO, modality: ModalityType, source: AssessmentSource
    ) -> ModalityResult:
        """Convert OceanScoresDTO to ModalityResult."""
        trait_scores = {
            PersonalityTrait.OPENNESS: scores.openness,
            PersonalityTrait.CONSCIENTIOUSNESS: scores.conscientiousness,
            PersonalityTrait.EXTRAVERSION: scores.extraversion,
            PersonalityTrait.AGREEABLENESS: scores.agreeableness,
            PersonalityTrait.NEUROTICISM: scores.neuroticism,
        }
        return ModalityResult(
            modality=modality,
            source=source,
            scores=trait_scores,
            confidence=scores.overall_confidence,
            timestamp=scores.assessed_at,
        )

    def _fusion_result_to_scores(self, result: FusionResult) -> OceanScoresDTO:
        """Convert FusionResult to OceanScoresDTO."""
        trait_scores = self._build_trait_scores(result)
        return OceanScoresDTO(
            openness=result.fused_scores.get(PersonalityTrait.OPENNESS, 0.5),
            conscientiousness=result.fused_scores.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5),
            extraversion=result.fused_scores.get(PersonalityTrait.EXTRAVERSION, 0.5),
            agreeableness=result.fused_scores.get(PersonalityTrait.AGREEABLENESS, 0.5),
            neuroticism=result.fused_scores.get(PersonalityTrait.NEUROTICISM, 0.5),
            overall_confidence=result.overall_confidence,
            trait_scores=trait_scores,
        )

    def _build_trait_scores(self, result: FusionResult) -> list[TraitScoreDTO]:
        """Build detailed trait scores from fusion result."""
        scores = []
        margin = 0.1 * (1 - result.overall_confidence)
        for trait, value in result.fused_scores.items():
            lower = max(0.0, value - margin)
            upper = min(1.0, value + margin)
            evidence = [f"fusion_{m.value}" for m in result.modalities_used]
            scores.append(TraitScoreDTO(
                trait=trait,
                value=value,
                confidence_lower=lower,
                confidence_upper=upper,
                sample_count=len(result.modalities_used),
                evidence_markers=evidence[:3],
            ))
        return scores

    def _neutral_scores(self) -> OceanScoresDTO:
        """Return neutral scores when fusion fails."""
        return OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            overall_confidence=0.2,
            trait_scores=[],
        )

    async def shutdown(self) -> None:
        """Shutdown the fusion system."""
        self._initialized = False
        logger.info("multimodal_fusion_shutdown")
