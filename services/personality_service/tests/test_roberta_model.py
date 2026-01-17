"""Tests for RoBERTa personality detector."""
from __future__ import annotations
import pytest
from uuid import uuid4
from datetime import datetime, timezone

from services.personality_service.src.schemas import PersonalityTrait, OceanScoresDTO
from services.personality_service.src.ml.roberta_model import (
    RoBERTaSettings, RoBERTaPersonalityDetector, RoBERTaPrediction,
    TextPreprocessor, SigmoidActivation, PersonalityClassificationHead,
    BatchPredictionResult,
)


class TestRoBERTaSettings:
    """Tests for RoBERTaSettings."""

    def test_default_settings(self) -> None:
        settings = RoBERTaSettings()
        assert settings.model_name == "roberta-base"
        assert settings.max_sequence_length == 512
        assert settings.batch_size == 8
        assert settings.device == "cpu"
        assert settings.num_labels == 5
        assert settings.confidence_threshold == 0.6

    def test_custom_settings(self) -> None:
        settings = RoBERTaSettings(
            model_name="roberta-large",
            max_sequence_length=256,
            batch_size=16,
        )
        assert settings.model_name == "roberta-large"
        assert settings.max_sequence_length == 256
        assert settings.batch_size == 16


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_preprocess_removes_urls(self) -> None:
        preprocessor = TextPreprocessor()
        text = "Check this link https://example.com for more info"
        result = preprocessor.preprocess(text)
        assert "[URL]" in result
        assert "https://example.com" not in result

    def test_preprocess_removes_mentions(self) -> None:
        preprocessor = TextPreprocessor()
        text = "Hey @username check this out"
        result = preprocessor.preprocess(text)
        assert "[MENTION]" in result
        assert "@username" not in result

    def test_preprocess_handles_hashtags(self) -> None:
        preprocessor = TextPreprocessor()
        text = "Feeling #happy today"
        result = preprocessor.preprocess(text)
        assert "#happy" not in result
        assert "happy" in result

    def test_preprocess_normalizes_whitespace(self) -> None:
        preprocessor = TextPreprocessor()
        text = "Multiple   spaces\n\nand newlines"
        result = preprocessor.preprocess(text)
        assert "  " not in result

    def test_preprocess_truncates_long_text(self) -> None:
        preprocessor = TextPreprocessor()
        text = "a" * 3000
        result = preprocessor.preprocess(text)
        assert len(result) <= 2048

    def test_preprocess_batch(self) -> None:
        preprocessor = TextPreprocessor()
        texts = ["Hello @user", "Visit https://example.com"]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == 2
        assert "[MENTION]" in results[0]
        assert "[URL]" in results[1]


class TestSigmoidActivation:
    """Tests for SigmoidActivation."""

    def test_sigmoid_zero(self) -> None:
        sigmoid = SigmoidActivation()
        result = sigmoid([0.0])
        assert 0.49 < result[0] < 0.51

    def test_sigmoid_positive(self) -> None:
        sigmoid = SigmoidActivation()
        result = sigmoid([5.0])
        assert result[0] > 0.99

    def test_sigmoid_negative(self) -> None:
        sigmoid = SigmoidActivation()
        result = sigmoid([-5.0])
        assert result[0] < 0.01

    def test_sigmoid_multiple_values(self) -> None:
        sigmoid = SigmoidActivation()
        result = sigmoid([-2.0, 0.0, 2.0])
        assert len(result) == 3
        assert result[0] < result[1] < result[2]


class TestPersonalityClassificationHead:
    """Tests for PersonalityClassificationHead."""

    def test_forward_returns_prediction(self) -> None:
        settings = RoBERTaSettings()
        head = PersonalityClassificationHead(settings)
        pooled = [0.1] * 768
        result = head.forward(pooled)
        assert isinstance(result, RoBERTaPrediction)
        assert len(result.trait_probabilities) == 5

    def test_forward_probabilities_in_range(self) -> None:
        settings = RoBERTaSettings()
        head = PersonalityClassificationHead(settings)
        pooled = [0.5] * 768
        result = head.forward(pooled)
        for trait, prob in result.trait_probabilities.items():
            assert 0.0 <= prob <= 1.0

    def test_forward_confidence_in_range(self) -> None:
        settings = RoBERTaSettings()
        head = PersonalityClassificationHead(settings)
        pooled = [0.3] * 768
        result = head.forward(pooled)
        assert 0.0 <= result.confidence <= 1.0


class TestRoBERTaPersonalityDetector:
    """Tests for RoBERTaPersonalityDetector."""

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        assert detector._initialized is True

    @pytest.mark.asyncio
    async def test_detect_returns_ocean_scores(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        text = "I love meeting new people and exploring creative ideas. Learning new things excites me!"
        result = await detector.detect(text)
        assert isinstance(result, OceanScoresDTO)
        assert 0.0 <= result.openness <= 1.0
        assert 0.0 <= result.conscientiousness <= 1.0
        assert 0.0 <= result.extraversion <= 1.0
        assert 0.0 <= result.agreeableness <= 1.0
        assert 0.0 <= result.neuroticism <= 1.0

    @pytest.mark.asyncio
    async def test_detect_short_text_returns_neutral(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        result = await detector.detect("Hi")
        assert result.openness == 0.5
        assert result.overall_confidence < 0.3

    @pytest.mark.asyncio
    async def test_detect_empty_text_returns_neutral(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        result = await detector.detect("")
        assert result.overall_confidence == 0.2

    @pytest.mark.asyncio
    async def test_detect_batch(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        texts = [
            "I enjoy working on challenging projects and achieving my goals.",
            "Meeting friends and having fun conversations makes me happy!",
        ]
        results = await detector.detect_batch(texts)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, OceanScoresDTO)

    @pytest.mark.asyncio
    async def test_detect_batch_empty(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        results = await detector.detect_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_embeddings(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        text = "This is a test sentence for embedding extraction."
        embeddings = await detector.get_embeddings(text)
        assert len(embeddings) == 768
        assert all(isinstance(e, float) for e in embeddings)

    @pytest.mark.asyncio
    async def test_embedding_cache(self) -> None:
        settings = RoBERTaSettings(cache_embeddings=True)
        detector = RoBERTaPersonalityDetector(settings=settings)
        await detector.initialize()
        text = "Test sentence for caching"
        emb1 = await detector.get_embeddings(text)
        emb2 = await detector.get_embeddings(text)
        assert emb1 == emb2

    def test_clear_cache(self) -> None:
        detector = RoBERTaPersonalityDetector()
        detector._embedding_cache["test"] = [0.1] * 768
        detector.clear_cache()
        assert len(detector._embedding_cache) == 0

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        await detector.shutdown()
        assert detector._initialized is False
        assert len(detector._embedding_cache) == 0

    @pytest.mark.asyncio
    async def test_detect_positive_text(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        text = "I am so happy and grateful! Love spending time with friends and family."
        result = await detector.detect(text)
        assert result.agreeableness >= 0.45

    @pytest.mark.asyncio
    async def test_detect_achievement_text(self) -> None:
        detector = RoBERTaPersonalityDetector()
        await detector.initialize()
        text = "Working hard to achieve my goals. Success requires effort and dedication."
        result = await detector.detect(text)
        assert result.conscientiousness >= 0.45
