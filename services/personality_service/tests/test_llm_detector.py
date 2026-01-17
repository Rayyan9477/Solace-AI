"""Tests for LLM personality detector."""
from __future__ import annotations
import pytest
from typing import Any

from services.personality_service.src.schemas import PersonalityTrait, OceanScoresDTO
from services.personality_service.src.ml.llm_detector import (
    LLMDetectorSettings, LLMPersonalityDetector, LLMAnalysisResult,
    PromptBuilder, ResponseParser,
)


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "") -> None:
        self._response = response
        self._call_count = 0

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        service_name: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        self._call_count += 1
        return self._response


class TestLLMDetectorSettings:
    """Tests for LLMDetectorSettings."""

    def test_default_settings(self) -> None:
        settings = LLMDetectorSettings()
        assert settings.model_name == "gpt-4"
        assert settings.temperature == 0.3
        assert settings.max_tokens == 500
        assert settings.retry_attempts == 2
        assert settings.fallback_on_error is True

    def test_custom_settings(self) -> None:
        settings = LLMDetectorSettings(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=300,
        )
        assert settings.model_name == "gpt-3.5-turbo"
        assert settings.temperature == 0.5
        assert settings.max_tokens == 300


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_build_system_prompt(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()
        assert "Big Five" in prompt
        assert "OCEAN" in prompt
        assert "psychologist" in prompt.lower()

    def test_build_analysis_prompt_with_reasoning(self) -> None:
        builder = PromptBuilder()
        text = "Sample text for analysis"
        prompt = builder.build_analysis_prompt(text, include_reasoning=True)
        assert text in prompt
        assert "reasoning" in prompt.lower()
        assert "JSON" in prompt

    def test_build_analysis_prompt_without_reasoning(self) -> None:
        builder = PromptBuilder()
        text = "Sample text"
        prompt = builder.build_analysis_prompt(text, include_reasoning=False)
        assert text in prompt
        assert "JSON" in prompt


class TestResponseParser:
    """Tests for ResponseParser."""

    def test_parse_valid_json(self) -> None:
        parser = ResponseParser()
        response = '''{"openness": 0.7, "conscientiousness": 0.6, "extraversion": 0.8, "agreeableness": 0.75, "neuroticism": 0.3}'''
        result = parser.parse(response, include_reasoning=False)
        assert result.scores[PersonalityTrait.OPENNESS] == 0.7
        assert result.scores[PersonalityTrait.EXTRAVERSION] == 0.8
        assert result.scores[PersonalityTrait.NEUROTICISM] == 0.3

    def test_parse_json_with_reasoning(self) -> None:
        parser = ResponseParser()
        response = '''{
            "openness": {"score": 0.8, "reasoning": "Creative language"},
            "conscientiousness": {"score": 0.6, "reasoning": "Organized"},
            "extraversion": {"score": 0.7, "reasoning": "Social"},
            "agreeableness": {"score": 0.75, "reasoning": "Kind"},
            "neuroticism": {"score": 0.2, "reasoning": "Stable"},
            "overall_confidence": 0.85,
            "evidence": ["positive_language", "social_words"]
        }'''
        result = parser.parse(response, include_reasoning=True)
        assert result.scores[PersonalityTrait.OPENNESS] == 0.8
        assert result.confidence == 0.85
        assert "positive_language" in result.evidence
        assert result.reasoning[PersonalityTrait.OPENNESS] == "Creative language"

    def test_parse_invalid_json_returns_default(self) -> None:
        parser = ResponseParser()
        response = "This is not JSON"
        result = parser.parse(response, include_reasoning=False)
        for trait in PersonalityTrait:
            assert result.scores[trait] == 0.5
        assert result.confidence == 0.3

    def test_parse_clamps_scores(self) -> None:
        parser = ResponseParser()
        response = '''{"openness": 1.5, "conscientiousness": -0.3, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5}'''
        result = parser.parse(response, include_reasoning=False)
        assert result.scores[PersonalityTrait.OPENNESS] == 1.0
        assert result.scores[PersonalityTrait.CONSCIENTIOUSNESS] == 0.0


class TestLLMPersonalityDetector:
    """Tests for LLMPersonalityDetector."""

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        detector = LLMPersonalityDetector()
        await detector.initialize()
        assert detector._initialized is True

    @pytest.mark.asyncio
    async def test_detect_short_text_returns_neutral(self) -> None:
        detector = LLMPersonalityDetector()
        await detector.initialize()
        result = await detector.detect("Hi there")
        assert result.overall_confidence == 0.2

    @pytest.mark.asyncio
    async def test_detect_empty_text_returns_neutral(self) -> None:
        detector = LLMPersonalityDetector()
        await detector.initialize()
        result = await detector.detect("")
        assert result.overall_confidence == 0.2

    @pytest.mark.asyncio
    async def test_detect_with_mock_client(self) -> None:
        mock_response = '''{"openness": 0.75, "conscientiousness": 0.65, "extraversion": 0.8, "agreeableness": 0.7, "neuroticism": 0.25, "overall_confidence": 0.85, "evidence": ["social"]}'''
        client = MockLLMClient(mock_response)
        detector = LLMPersonalityDetector(llm_client=client)
        await detector.initialize()
        text = "I love exploring new ideas and meeting interesting people!"
        result = await detector.detect(text)
        assert result.openness == 0.75
        assert result.extraversion == 0.8
        assert result.overall_confidence == 0.85
        assert client._call_count == 1

    @pytest.mark.asyncio
    async def test_detect_caches_response(self) -> None:
        mock_response = '''{"openness": 0.6, "conscientiousness": 0.6, "extraversion": 0.6, "agreeableness": 0.6, "neuroticism": 0.4, "overall_confidence": 0.7}'''
        client = MockLLMClient(mock_response)
        settings = LLMDetectorSettings(cache_responses=True)
        detector = LLMPersonalityDetector(settings=settings, llm_client=client)
        await detector.initialize()
        text = "This is a test text for caching purposes that is long enough."
        await detector.detect(text)
        await detector.detect(text)
        assert client._call_count == 1

    @pytest.mark.asyncio
    async def test_detect_heuristic_fallback(self) -> None:
        detector = LLMPersonalityDetector()
        await detector.initialize()
        text = "I am happy and love my friends. We work hard to achieve success together."
        result = await detector.detect(text)
        assert isinstance(result, OceanScoresDTO)
        assert 0.0 <= result.extraversion <= 1.0

    @pytest.mark.asyncio
    async def test_detect_with_reasoning(self) -> None:
        mock_response = '''{
            "openness": {"score": 0.8, "reasoning": "Creative"},
            "conscientiousness": {"score": 0.6, "reasoning": "Organized"},
            "extraversion": {"score": 0.7, "reasoning": "Social"},
            "agreeableness": {"score": 0.75, "reasoning": "Kind"},
            "neuroticism": {"score": 0.2, "reasoning": "Calm"},
            "overall_confidence": 0.8
        }'''
        client = MockLLMClient(mock_response)
        detector = LLMPersonalityDetector(llm_client=client)
        await detector.initialize()
        text = "I enjoy creative activities and spending time with friends."
        scores, reasoning = await detector.detect_with_reasoning(text)
        assert scores.openness == 0.8
        assert reasoning[PersonalityTrait.OPENNESS] == "Creative"

    def test_clear_cache(self) -> None:
        detector = LLMPersonalityDetector()
        detector._response_cache["test"] = LLMAnalysisResult()
        detector.clear_cache()
        assert len(detector._response_cache) == 0

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        detector = LLMPersonalityDetector()
        await detector.initialize()
        await detector.shutdown()
        assert detector._initialized is False
        assert len(detector._response_cache) == 0

    @pytest.mark.asyncio
    async def test_heuristic_positive_emotion(self) -> None:
        detector = LLMPersonalityDetector()
        await detector.initialize()
        text = "I am so happy and grateful! Everything is wonderful and amazing today!"
        result = await detector.detect(text)
        assert result.agreeableness >= 0.5

    @pytest.mark.asyncio
    async def test_heuristic_social_text(self) -> None:
        detector = LLMPersonalityDetector()
        await detector.initialize()
        text = "I love spending time with friends and family. We always have fun together!"
        result = await detector.detect(text)
        assert result.extraversion >= 0.5

    @pytest.mark.asyncio
    async def test_heuristic_negative_emotion(self) -> None:
        detector = LLMPersonalityDetector()
        await detector.initialize()
        text = "I am feeling sad and worried about everything. Life seems terrible and anxious."
        result = await detector.detect(text)
        assert result.neuroticism >= 0.5
