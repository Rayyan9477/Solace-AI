"""
Solace-AI Personality Service - Zero-Shot LLM Personality Detector.
Uses large language models for personality trait detection via zero-shot analysis.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import PersonalityTrait, AssessmentSource, OceanScoresDTO, TraitScoreDTO

logger = structlog.get_logger(__name__)


class LLMDetectorSettings(BaseSettings):
    """LLM detector configuration."""
    model_name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=100, le=2000)
    max_input_length: int = Field(default=3000, ge=100, le=10000)
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    retry_attempts: int = Field(default=2, ge=0, le=5)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_reasoning: bool = Field(default=True)
    cache_responses: bool = Field(default=True)
    fallback_on_error: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_LLM_", env_file=".env", extra="ignore")


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients."""
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        service_name: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str: ...


@dataclass
class LLMAnalysisResult:
    """Result of LLM personality analysis."""
    result_id: UUID = field(default_factory=uuid4)
    scores: dict[PersonalityTrait, float] = field(default_factory=dict)
    reasoning: dict[PersonalityTrait, str] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_response: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PromptBuilder:
    """Builds prompts for LLM personality analysis."""
    _SYSTEM_PROMPT = """You are an expert psychologist specializing in personality assessment using the Big Five (OCEAN) model.
Your task is to analyze text and estimate personality trait scores.
Be objective, nuanced, and consider multiple indicators.
Always provide brief reasoning for each score."""

    _ANALYSIS_PROMPT = """Analyze the following text and estimate Big Five (OCEAN) personality trait scores.
Return a JSON object with scores from 0.0 to 1.0 for each trait, along with brief reasoning.

Trait definitions:
- openness: Intellectual curiosity, creativity, preference for novelty and variety
- conscientiousness: Organization, dependability, self-discipline, achievement-striving
- extraversion: Sociability, assertiveness, positive emotions, energy from social interaction
- agreeableness: Cooperation, trust, empathy, concern for social harmony
- neuroticism: Emotional instability, anxiety, moodiness, tendency toward negative emotions

Text to analyze:
---
{text}
---

Respond ONLY with valid JSON in this exact format:
{{
  "openness": {{"score": 0.0, "reasoning": "brief explanation"}},
  "conscientiousness": {{"score": 0.0, "reasoning": "brief explanation"}},
  "extraversion": {{"score": 0.0, "reasoning": "brief explanation"}},
  "agreeableness": {{"score": 0.0, "reasoning": "brief explanation"}},
  "neuroticism": {{"score": 0.0, "reasoning": "brief explanation"}},
  "overall_confidence": 0.0,
  "evidence": ["marker1", "marker2"]
}}"""

    _SIMPLIFIED_PROMPT = """Analyze this text for Big Five personality traits.
Return JSON with scores (0.0-1.0) for: openness, conscientiousness, extraversion, agreeableness, neuroticism.

Text: {text}

JSON format: {{"openness": 0.0, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 0.0}}"""

    def build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        return self._SYSTEM_PROMPT

    def build_analysis_prompt(self, text: str, include_reasoning: bool = True) -> str:
        """Build analysis prompt with text."""
        template = self._ANALYSIS_PROMPT if include_reasoning else self._SIMPLIFIED_PROMPT
        return template.format(text=text)


class ResponseParser:
    """Parses LLM responses into structured results."""
    _JSON_PATTERN = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)

    def parse(self, response: str, include_reasoning: bool = True) -> LLMAnalysisResult:
        """Parse LLM response into analysis result."""
        json_match = self._JSON_PATTERN.search(response)
        if not json_match:
            logger.warning("no_json_found_in_llm_response")
            return self._default_result(response)
        try:
            data = json.loads(json_match.group())
            return self._parse_structured_response(data, response, include_reasoning)
        except json.JSONDecodeError as e:
            logger.warning("json_parse_error", error=str(e))
            return self._default_result(response)

    def _parse_structured_response(
        self, data: dict[str, Any], raw_response: str, include_reasoning: bool
    ) -> LLMAnalysisResult:
        """Parse structured JSON response."""
        scores: dict[PersonalityTrait, float] = {}
        reasoning: dict[PersonalityTrait, str] = {}
        for trait in PersonalityTrait:
            trait_data = data.get(trait.value)
            if isinstance(trait_data, dict):
                scores[trait] = self._safe_score(trait_data.get("score", 0.5))
                if include_reasoning:
                    reasoning[trait] = str(trait_data.get("reasoning", ""))
            elif isinstance(trait_data, (int, float)):
                scores[trait] = self._safe_score(trait_data)
            else:
                scores[trait] = 0.5
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        confidence = self._safe_score(data.get("overall_confidence", 0.6))
        return LLMAnalysisResult(
            scores=scores,
            reasoning=reasoning,
            evidence=[str(e) for e in evidence[:10]],
            confidence=confidence,
            raw_response=raw_response,
        )

    def _safe_score(self, value: Any) -> float:
        """Safely convert value to score in [0, 1]."""
        try:
            score = float(value)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5

    def _default_result(self, raw_response: str) -> LLMAnalysisResult:
        """Return default result on parse failure."""
        return LLMAnalysisResult(
            scores={trait: 0.5 for trait in PersonalityTrait},
            reasoning={},
            evidence=[],
            confidence=0.3,
            raw_response=raw_response,
        )


class LLMPersonalityDetector:
    """Zero-shot LLM-based personality detector."""

    def __init__(
        self,
        settings: LLMDetectorSettings | None = None,
        llm_client: LLMClientProtocol | None = None,
    ) -> None:
        self._settings = settings or LLMDetectorSettings()
        self._llm_client = llm_client
        self._prompt_builder = PromptBuilder()
        self._response_parser = ResponseParser()
        self._response_cache: dict[str, LLMAnalysisResult] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the LLM detector."""
        self._initialized = True
        logger.info(
            "llm_detector_initialized",
            model=self._settings.model_name,
            has_client=self._llm_client is not None,
        )

    async def detect(self, text: str) -> OceanScoresDTO:
        """Detect personality traits using zero-shot LLM analysis."""
        if not text or len(text.strip()) < 20:
            logger.warning("text_too_short_for_llm", length=len(text))
            return self._neutral_scores(confidence=0.2)
        truncated_text = text[:self._settings.max_input_length]
        cache_key = truncated_text[:256]
        if self._settings.cache_responses and cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            logger.debug("llm_cache_hit", cache_key_prefix=cache_key[:50])
            return self._result_to_scores(cached)
        result = await self._run_analysis(truncated_text)
        if self._settings.cache_responses and len(self._response_cache) < 500:
            self._response_cache[cache_key] = result
        return self._result_to_scores(result)

    async def detect_with_reasoning(self, text: str) -> tuple[OceanScoresDTO, dict[PersonalityTrait, str]]:
        """Detect personality with reasoning explanations."""
        if not text or len(text.strip()) < 20:
            return self._neutral_scores(confidence=0.2), {}
        truncated_text = text[:self._settings.max_input_length]
        result = await self._run_analysis(truncated_text, include_reasoning=True)
        return self._result_to_scores(result), result.reasoning

    async def _run_analysis(self, text: str, include_reasoning: bool = True) -> LLMAnalysisResult:
        """Run LLM analysis with retry logic."""
        if self._llm_client is None:
            logger.warning("no_llm_client_using_heuristic_fallback")
            return self._heuristic_fallback(text)
        system_prompt = self._prompt_builder.build_system_prompt()
        user_prompt = self._prompt_builder.build_analysis_prompt(text, include_reasoning)
        last_error: Exception | None = None
        import asyncio
        for attempt in range(self._settings.retry_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    self._llm_client.generate(
                        system_prompt=system_prompt,
                        user_message=user_prompt,
                        service_name="personality_llm_detector",
                        temperature=self._settings.temperature,
                        max_tokens=self._settings.max_tokens,
                    ),
                    timeout=self._settings.timeout_seconds,
                )
                result = self._response_parser.parse(response, include_reasoning)
                if result.confidence >= self._settings.confidence_threshold:
                    return result
                logger.warning("low_confidence_result", confidence=result.confidence, attempt=attempt)
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"LLM call timed out after {self._settings.timeout_seconds}s")
                logger.warning("llm_analysis_timeout", attempt=attempt, timeout_s=self._settings.timeout_seconds)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning("llm_analysis_parse_failed", attempt=attempt, error=str(e))
            except Exception as e:
                last_error = e
                logger.warning("llm_analysis_attempt_failed", attempt=attempt, error_type=type(e).__name__, error=str(e))
        logger.error("llm_analysis_failed_all_attempts", error=str(last_error))
        if self._settings.fallback_on_error:
            return self._heuristic_fallback(text)
        return LLMAnalysisResult(
            scores={trait: 0.5 for trait in PersonalityTrait},
            confidence=0.2,
        )

    def _heuristic_fallback(self, text: str) -> LLMAnalysisResult:
        """Heuristic-based fallback when LLM is unavailable."""
        words = text.lower().split()
        word_count = len(words) or 1
        positive_words = {'happy', 'love', 'great', 'good', 'wonderful', 'joy', 'excited', 'amazing'}
        negative_words = {'sad', 'angry', 'hate', 'bad', 'terrible', 'worried', 'anxious', 'fear'}
        social_words = {'friend', 'family', 'people', 'together', 'we', 'us', 'team', 'group'}
        cognitive_words = {'think', 'believe', 'understand', 'analyze', 'consider', 'realize'}
        achievement_words = {'achieve', 'goal', 'success', 'work', 'accomplish', 'complete'}
        pos_ratio = sum(1 for w in words if w in positive_words) / word_count
        neg_ratio = sum(1 for w in words if w in negative_words) / word_count
        social_ratio = sum(1 for w in words if w in social_words) / word_count
        cog_ratio = sum(1 for w in words if w in cognitive_words) / word_count
        ach_ratio = sum(1 for w in words if w in achievement_words) / word_count
        exclaim_ratio = text.count('!') / word_count
        question_ratio = text.count('?') / word_count
        openness = self._clamp(0.5 + cog_ratio * 10 + question_ratio * 3)
        conscientiousness = self._clamp(0.5 + ach_ratio * 8)
        extraversion = self._clamp(0.5 + social_ratio * 8 + exclaim_ratio * 5)
        agreeableness = self._clamp(0.5 + pos_ratio * 6 + social_ratio * 4 - neg_ratio * 3)
        neuroticism = self._clamp(0.5 + neg_ratio * 8 - pos_ratio * 3)
        scores = {
            PersonalityTrait.OPENNESS: openness,
            PersonalityTrait.CONSCIENTIOUSNESS: conscientiousness,
            PersonalityTrait.EXTRAVERSION: extraversion,
            PersonalityTrait.AGREEABLENESS: agreeableness,
            PersonalityTrait.NEUROTICISM: neuroticism,
        }
        evidence = []
        if pos_ratio > 0.02:
            evidence.append("positive_language")
        if neg_ratio > 0.02:
            evidence.append("negative_language")
        if social_ratio > 0.02:
            evidence.append("social_orientation")
        return LLMAnalysisResult(
            scores=scores,
            reasoning={},
            evidence=evidence,
            confidence=0.5,
            raw_response="heuristic_fallback",
        )

    def _clamp(self, value: float) -> float:
        """Clamp value between 0 and 1."""
        return max(0.0, min(1.0, value))

    def _result_to_scores(self, result: LLMAnalysisResult) -> OceanScoresDTO:
        """Convert analysis result to OceanScoresDTO."""
        trait_scores = self._build_trait_scores(result)
        return OceanScoresDTO(
            openness=result.scores.get(PersonalityTrait.OPENNESS, 0.5),
            conscientiousness=result.scores.get(PersonalityTrait.CONSCIENTIOUSNESS, 0.5),
            extraversion=result.scores.get(PersonalityTrait.EXTRAVERSION, 0.5),
            agreeableness=result.scores.get(PersonalityTrait.AGREEABLENESS, 0.5),
            neuroticism=result.scores.get(PersonalityTrait.NEUROTICISM, 0.5),
            overall_confidence=result.confidence,
            trait_scores=trait_scores,
        )

    def _build_trait_scores(self, result: LLMAnalysisResult) -> list[TraitScoreDTO]:
        """Build detailed trait scores."""
        scores = []
        margin = 0.12 * (1 - result.confidence)
        for trait, value in result.scores.items():
            lower = max(0.0, value - margin)
            upper = min(1.0, value + margin)
            evidence = [e for e in result.evidence if trait.value[:4] in e.lower()]
            scores.append(TraitScoreDTO(
                trait=trait,
                value=value,
                confidence_lower=lower,
                confidence_upper=upper,
                sample_count=1,
                evidence_markers=evidence[:3] or [f"llm_{trait.value}"],
            ))
        return scores

    def _neutral_scores(self, confidence: float = 0.3) -> OceanScoresDTO:
        """Return neutral scores."""
        return OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            overall_confidence=confidence,
            trait_scores=[],
        )

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()
        logger.info("llm_response_cache_cleared")

    async def shutdown(self) -> None:
        """Shutdown the detector."""
        self._response_cache.clear()
        self._initialized = False
        logger.info("llm_detector_shutdown")
