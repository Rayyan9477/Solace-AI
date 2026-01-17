"""Tests for multimodal fusion."""
from __future__ import annotations
import pytest
from datetime import datetime, timezone

from services.personality_service.src.schemas import PersonalityTrait, OceanScoresDTO
from services.personality_service.src.ml.multimodal import (
    MultimodalFusionSettings, MultimodalFusion, FusionStrategy,
    ModalityType, ModalityResult, FusionResult,
    WeightCalculator, DisagreementHandler, FusionEngine,
)


def create_ocean_scores(
    openness: float = 0.5,
    conscientiousness: float = 0.5,
    extraversion: float = 0.5,
    agreeableness: float = 0.5,
    neuroticism: float = 0.5,
    confidence: float = 0.6,
) -> OceanScoresDTO:
    """Helper to create OceanScoresDTO."""
    return OceanScoresDTO(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=neuroticism,
        overall_confidence=confidence,
    )


def create_modality_result(
    modality: ModalityType,
    openness: float = 0.5,
    confidence: float = 0.6,
) -> ModalityResult:
    """Helper to create ModalityResult."""
    return ModalityResult(
        modality=modality,
        scores={
            PersonalityTrait.OPENNESS: openness,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.5,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.5,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        confidence=confidence,
    )


class TestMultimodalFusionSettings:
    """Tests for MultimodalFusionSettings."""

    def test_default_settings(self) -> None:
        settings = MultimodalFusionSettings()
        assert settings.fusion_strategy == FusionStrategy.CONFIDENCE_WEIGHTED
        assert settings.roberta_weight == 0.35
        assert settings.llm_weight == 0.35
        assert settings.liwc_weight == 0.30
        assert settings.min_modalities == 1

    def test_custom_settings(self) -> None:
        settings = MultimodalFusionSettings(
            fusion_strategy=FusionStrategy.ADAPTIVE,
            roberta_weight=0.5,
        )
        assert settings.fusion_strategy == FusionStrategy.ADAPTIVE
        assert settings.roberta_weight == 0.5


class TestModalityResult:
    """Tests for ModalityResult."""

    def test_is_valid_above_threshold(self) -> None:
        result = create_modality_result(ModalityType.TEXT_ROBERTA, confidence=0.7)
        assert result.is_valid(0.5) is True

    def test_is_valid_below_threshold(self) -> None:
        result = create_modality_result(ModalityType.TEXT_ROBERTA, confidence=0.3)
        assert result.is_valid(0.5) is False

    def test_is_valid_missing_scores(self) -> None:
        result = ModalityResult(modality=ModalityType.TEXT_ROBERTA, confidence=0.7)
        assert result.is_valid(0.5) is False


class TestWeightCalculator:
    """Tests for WeightCalculator."""

    def test_weighted_average_weights(self) -> None:
        settings = MultimodalFusionSettings()
        calc = WeightCalculator(settings)
        results = [
            create_modality_result(ModalityType.TEXT_ROBERTA),
            create_modality_result(ModalityType.TEXT_LLM),
        ]
        weights = calc.compute_weights(results, FusionStrategy.WEIGHTED_AVERAGE)
        assert len(weights) == 2
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

    def test_confidence_weighted(self) -> None:
        settings = MultimodalFusionSettings()
        calc = WeightCalculator(settings)
        results = [
            create_modality_result(ModalityType.TEXT_ROBERTA, confidence=0.9),
            create_modality_result(ModalityType.TEXT_LLM, confidence=0.3),
        ]
        weights = calc.compute_weights(results, FusionStrategy.CONFIDENCE_WEIGHTED)
        assert weights[ModalityType.TEXT_ROBERTA] > weights[ModalityType.TEXT_LLM]

    def test_max_confidence_weights(self) -> None:
        settings = MultimodalFusionSettings()
        calc = WeightCalculator(settings)
        results = [
            create_modality_result(ModalityType.TEXT_ROBERTA, confidence=0.9),
            create_modality_result(ModalityType.TEXT_LLM, confidence=0.3),
        ]
        weights = calc.compute_weights(results, FusionStrategy.MAX_CONFIDENCE)
        assert weights[ModalityType.TEXT_ROBERTA] == 1.0
        assert weights[ModalityType.TEXT_LLM] == 0.0


class TestDisagreementHandler:
    """Tests for DisagreementHandler."""

    def test_detect_no_disagreement(self) -> None:
        settings = MultimodalFusionSettings()
        handler = DisagreementHandler(settings)
        results = [
            create_modality_result(ModalityType.TEXT_ROBERTA, openness=0.6),
            create_modality_result(ModalityType.TEXT_LLM, openness=0.65),
        ]
        disagreements = handler.detect_disagreements(results)
        assert len(disagreements) == 0

    def test_detect_disagreement(self) -> None:
        settings = MultimodalFusionSettings(disagreement_threshold=0.2)
        handler = DisagreementHandler(settings)
        results = [
            create_modality_result(ModalityType.TEXT_ROBERTA, openness=0.2),
            create_modality_result(ModalityType.TEXT_LLM, openness=0.8),
        ]
        disagreements = handler.detect_disagreements(results)
        assert len(disagreements) > 0
        assert "openness" in disagreements[0].lower()

    def test_resolve_disagreement(self) -> None:
        settings = MultimodalFusionSettings()
        handler = DisagreementHandler(settings)
        results = [
            create_modality_result(ModalityType.TEXT_ROBERTA, openness=0.3, confidence=0.8),
            create_modality_result(ModalityType.TEXT_LLM, openness=0.7, confidence=0.4),
        ]
        resolved = handler.resolve_disagreement(results, PersonalityTrait.OPENNESS)
        assert 0.3 < resolved < 0.7


class TestFusionEngine:
    """Tests for FusionEngine."""

    def test_fuse_single_modality(self) -> None:
        settings = MultimodalFusionSettings()
        weight_calc = WeightCalculator(settings)
        disagree_handler = DisagreementHandler(settings)
        engine = FusionEngine(weight_calc, disagree_handler)
        results = [create_modality_result(ModalityType.TEXT_ROBERTA, openness=0.7)]
        fusion_result = engine.fuse(results, FusionStrategy.CONFIDENCE_WEIGHTED)
        assert fusion_result.fused_scores[PersonalityTrait.OPENNESS] == 0.7

    def test_fuse_multiple_modalities(self) -> None:
        settings = MultimodalFusionSettings()
        weight_calc = WeightCalculator(settings)
        disagree_handler = DisagreementHandler(settings)
        engine = FusionEngine(weight_calc, disagree_handler)
        results = [
            create_modality_result(ModalityType.TEXT_ROBERTA, openness=0.8, confidence=0.7),
            create_modality_result(ModalityType.TEXT_LLM, openness=0.6, confidence=0.7),
        ]
        fusion_result = engine.fuse(results, FusionStrategy.CONFIDENCE_WEIGHTED)
        assert 0.6 < fusion_result.fused_scores[PersonalityTrait.OPENNESS] < 0.8

    def test_fuse_empty_results(self) -> None:
        settings = MultimodalFusionSettings()
        weight_calc = WeightCalculator(settings)
        disagree_handler = DisagreementHandler(settings)
        engine = FusionEngine(weight_calc, disagree_handler)
        fusion_result = engine.fuse([], FusionStrategy.CONFIDENCE_WEIGHTED)
        assert fusion_result.fused_scores[PersonalityTrait.OPENNESS] == 0.5
        assert fusion_result.overall_confidence == 0.2


class TestMultimodalFusion:
    """Tests for MultimodalFusion."""

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        assert fusion._initialized is True

    @pytest.mark.asyncio
    async def test_fuse_single_roberta(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        roberta_scores = create_ocean_scores(openness=0.8, confidence=0.7)
        result = await fusion.fuse(roberta_scores=roberta_scores)
        assert result.openness == 0.8

    @pytest.mark.asyncio
    async def test_fuse_multiple_modalities(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        roberta_scores = create_ocean_scores(openness=0.8, confidence=0.7)
        llm_scores = create_ocean_scores(openness=0.6, confidence=0.7)
        result = await fusion.fuse(roberta_scores=roberta_scores, llm_scores=llm_scores)
        assert 0.6 < result.openness < 0.8

    @pytest.mark.asyncio
    async def test_fuse_all_modalities(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        roberta_scores = create_ocean_scores(openness=0.8, confidence=0.7)
        llm_scores = create_ocean_scores(openness=0.6, confidence=0.7)
        liwc_scores = create_ocean_scores(openness=0.7, confidence=0.6)
        result = await fusion.fuse(
            roberta_scores=roberta_scores,
            llm_scores=llm_scores,
            liwc_scores=liwc_scores,
        )
        assert 0.6 <= result.openness <= 0.8
        assert result.overall_confidence > 0.5

    @pytest.mark.asyncio
    async def test_fuse_no_modalities_returns_neutral(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        result = await fusion.fuse()
        assert result.openness == 0.5
        assert result.overall_confidence == 0.2

    @pytest.mark.asyncio
    async def test_fuse_with_custom_strategy(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        roberta_scores = create_ocean_scores(openness=0.9, confidence=0.9)
        llm_scores = create_ocean_scores(openness=0.3, confidence=0.3)
        result = await fusion.fuse(
            roberta_scores=roberta_scores,
            llm_scores=llm_scores,
            strategy=FusionStrategy.MAX_CONFIDENCE,
        )
        assert result.openness == 0.9

    @pytest.mark.asyncio
    async def test_fuse_with_metadata(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        results = [
            create_modality_result(ModalityType.TEXT_ROBERTA, openness=0.7, confidence=0.8),
            create_modality_result(ModalityType.TEXT_LLM, openness=0.6, confidence=0.7),
        ]
        scores, fusion_result = await fusion.fuse_with_metadata(results)
        assert isinstance(scores, OceanScoresDTO)
        assert isinstance(fusion_result, FusionResult)
        assert len(fusion_result.modalities_used) == 2

    @pytest.mark.asyncio
    async def test_fuse_low_confidence_filtered(self) -> None:
        settings = MultimodalFusionSettings(confidence_threshold=0.7)
        fusion = MultimodalFusion(settings=settings)
        await fusion.initialize()
        roberta_scores = create_ocean_scores(openness=0.9, confidence=0.8)
        llm_scores = create_ocean_scores(openness=0.3, confidence=0.4)
        result = await fusion.fuse(
            roberta_scores=roberta_scores,
            llm_scores=llm_scores,
        )
        assert result.openness > 0.7

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        await fusion.shutdown()
        assert fusion._initialized is False

    @pytest.mark.asyncio
    async def test_fuse_produces_trait_scores(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        roberta_scores = create_ocean_scores(openness=0.8, confidence=0.7)
        result = await fusion.fuse(roberta_scores=roberta_scores)
        assert len(result.trait_scores) == 5

    @pytest.mark.asyncio
    async def test_different_fusion_strategies(self) -> None:
        fusion = MultimodalFusion()
        await fusion.initialize()
        roberta_scores = create_ocean_scores(openness=0.8, confidence=0.6)
        llm_scores = create_ocean_scores(openness=0.4, confidence=0.6)
        for strategy in FusionStrategy:
            result = await fusion.fuse(
                roberta_scores=roberta_scores,
                llm_scores=llm_scores,
                strategy=strategy,
            )
            assert 0.0 <= result.openness <= 1.0
