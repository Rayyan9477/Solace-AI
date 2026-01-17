"""
Solace-AI Personality Service - OCEAN Trait Ensemble Detection.
Detects Big Five personality traits using text analysis, LIWC features, and LLM zero-shot.
"""
from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import (
    PersonalityTrait, AssessmentSource, OceanScoresDTO, TraitScoreDTO,
)

logger = structlog.get_logger(__name__)


class TraitDetectorSettings(BaseSettings):
    """Trait detector configuration."""
    min_text_length: int = Field(default=50)
    max_text_length: int = Field(default=10000)
    ensemble_weights_text: float = Field(default=0.4, ge=0.0, le=1.0)
    ensemble_weights_liwc: float = Field(default=0.3, ge=0.0, le=1.0)
    ensemble_weights_llm: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_base: float = Field(default=0.5)
    confidence_sample_factor: float = Field(default=0.1)
    llm_temperature: float = Field(default=0.3)
    enable_llm_detection: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_TRAIT_", env_file=".env", extra="ignore")


@dataclass
class LIWCFeatures:
    """LIWC-style linguistic feature extraction results."""
    word_count: int = 0
    i_words_ratio: float = 0.0
    we_words_ratio: float = 0.0
    social_words_ratio: float = 0.0
    positive_emotion_ratio: float = 0.0
    negative_emotion_ratio: float = 0.0
    cognitive_process_ratio: float = 0.0
    achievement_words_ratio: float = 0.0
    tentative_words_ratio: float = 0.0
    insight_words_ratio: float = 0.0
    question_marks_ratio: float = 0.0
    exclamation_ratio: float = 0.0


@dataclass
class TraitDetectionResult:
    """Result of a single trait detection method."""
    detection_id: UUID = field(default_factory=uuid4)
    source: AssessmentSource = AssessmentSource.TEXT_ANALYSIS
    scores: dict[PersonalityTrait, float] = field(default_factory=dict)
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LIWCFeatureExtractor:
    """Extracts LIWC-style linguistic features from text."""
    _I_WORDS = {"i", "me", "my", "mine", "myself"}
    _WE_WORDS = {"we", "us", "our", "ours", "ourselves"}
    _SOCIAL_WORDS = {"friend", "family", "talk", "share", "together", "meet", "call", "group", "team", "people"}
    _POSITIVE_WORDS = {"happy", "love", "good", "great", "wonderful", "joy", "exciting", "beautiful", "hope", "kind"}
    _NEGATIVE_WORDS = {"sad", "angry", "hate", "bad", "terrible", "awful", "hurt", "pain", "fear", "worry", "anxious"}
    _COGNITIVE_WORDS = {"think", "know", "believe", "understand", "consider", "realize", "reason", "decide", "analyze"}
    _ACHIEVEMENT_WORDS = {"work", "achieve", "success", "goal", "complete", "accomplish", "win", "effort", "try", "improve"}
    _TENTATIVE_WORDS = {"maybe", "perhaps", "might", "possibly", "could", "guess", "seem", "appear", "probably", "unsure"}
    _INSIGHT_WORDS = {"realize", "understand", "discover", "learn", "insight", "aware", "recognize", "comprehend", "notice"}

    def extract(self, text: str) -> LIWCFeatures:
        """Extract LIWC features from text."""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        word_count = len(words) if words else 1
        word_set = set(words)
        features = LIWCFeatures(
            word_count=len(words),
            i_words_ratio=self._count_matches(words, self._I_WORDS) / word_count,
            we_words_ratio=self._count_matches(words, self._WE_WORDS) / word_count,
            social_words_ratio=self._count_matches(words, self._SOCIAL_WORDS) / word_count,
            positive_emotion_ratio=self._count_matches(words, self._POSITIVE_WORDS) / word_count,
            negative_emotion_ratio=self._count_matches(words, self._NEGATIVE_WORDS) / word_count,
            cognitive_process_ratio=self._count_matches(words, self._COGNITIVE_WORDS) / word_count,
            achievement_words_ratio=self._count_matches(words, self._ACHIEVEMENT_WORDS) / word_count,
            tentative_words_ratio=self._count_matches(words, self._TENTATIVE_WORDS) / word_count,
            insight_words_ratio=self._count_matches(words, self._INSIGHT_WORDS) / word_count,
            question_marks_ratio=text.count("?") / word_count,
            exclamation_ratio=text.count("!") / word_count,
        )
        return features

    def _count_matches(self, words: list[str], word_set: set[str]) -> int:
        """Count how many words match the target set."""
        return sum(1 for w in words if w in word_set)


class TextBasedDetector:
    """Detects OCEAN traits using text pattern analysis."""

    def __init__(self) -> None:
        self._liwc_extractor = LIWCFeatureExtractor()

    def detect(self, text: str) -> TraitDetectionResult:
        """Detect traits using text-based heuristics."""
        features = self._liwc_extractor.extract(text)
        scores = self._compute_trait_scores(features)
        evidence = self._generate_evidence(features)
        confidence = min(0.7, 0.3 + (features.word_count / 500) * 0.4)
        return TraitDetectionResult(
            source=AssessmentSource.TEXT_ANALYSIS,
            scores=scores,
            confidence=confidence,
            evidence=evidence,
        )

    def _compute_trait_scores(self, features: LIWCFeatures) -> dict[PersonalityTrait, float]:
        """Compute OCEAN scores from LIWC features."""
        openness = self._clamp(0.5 + features.insight_words_ratio * 10 + features.question_marks_ratio * 5)
        conscientiousness = self._clamp(0.5 + features.achievement_words_ratio * 8 - features.tentative_words_ratio * 3)
        extraversion = self._clamp(0.5 + features.social_words_ratio * 8 + features.we_words_ratio * 5 + features.exclamation_ratio * 3)
        agreeableness = self._clamp(0.5 + features.positive_emotion_ratio * 6 + features.social_words_ratio * 4 - features.negative_emotion_ratio * 3)
        neuroticism = self._clamp(0.5 + features.negative_emotion_ratio * 8 + features.tentative_words_ratio * 4 - features.positive_emotion_ratio * 3)
        return {
            PersonalityTrait.OPENNESS: openness,
            PersonalityTrait.CONSCIENTIOUSNESS: conscientiousness,
            PersonalityTrait.EXTRAVERSION: extraversion,
            PersonalityTrait.AGREEABLENESS: agreeableness,
            PersonalityTrait.NEUROTICISM: neuroticism,
        }

    def _clamp(self, value: float) -> float:
        """Clamp value between 0 and 1."""
        return max(0.0, min(1.0, value))

    def _generate_evidence(self, features: LIWCFeatures) -> list[str]:
        """Generate evidence markers from features."""
        evidence = []
        if features.positive_emotion_ratio > 0.02:
            evidence.append("high_positive_emotion")
        if features.negative_emotion_ratio > 0.02:
            evidence.append("elevated_negative_emotion")
        if features.social_words_ratio > 0.02:
            evidence.append("social_orientation")
        if features.achievement_words_ratio > 0.015:
            evidence.append("achievement_focus")
        if features.insight_words_ratio > 0.01:
            evidence.append("reflective_thinking")
        return evidence


class LLMBasedDetector:
    """Detects OCEAN traits using LLM zero-shot analysis."""
    _DETECTION_PROMPT = """Analyze the following text and estimate Big Five (OCEAN) personality traits.
Return a JSON object with scores from 0.0 to 1.0 for each trait:
- openness: intellectual curiosity, creativity, preference for novelty
- conscientiousness: organization, dependability, self-discipline
- extraversion: sociability, assertiveness, positive emotions
- agreeableness: cooperation, trust, empathy toward others
- neuroticism: emotional instability, anxiety, moodiness

Also provide brief evidence for each score.

Text to analyze:
{text}

Respond ONLY with valid JSON in this format:
{"openness": 0.0, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 0.0, "evidence": ["marker1", "marker2"]}"""

    def __init__(self, llm_client: Any) -> None:
        self._llm_client = llm_client

    async def detect(self, text: str) -> TraitDetectionResult:
        """Detect traits using LLM zero-shot analysis."""
        prompt = self._DETECTION_PROMPT.format(text=text[:2000])
        try:
            response = await self._llm_client.generate(
                system_prompt="You are a psychology expert analyzing personality traits from text.",
                user_message=prompt,
                service_name="personality_detection",
                temperature=0.3,
            )
            scores, evidence = self._parse_response(response)
            return TraitDetectionResult(source=AssessmentSource.LLM_ZERO_SHOT, scores=scores, confidence=0.65, evidence=evidence)
        except Exception as e:
            logger.warning("llm_detection_failed", error=str(e))
            return self._default_result()

    def _parse_response(self, response: str) -> tuple[dict[PersonalityTrait, float], list[str]]:
        """Parse LLM JSON response into scores."""
        try:
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if not json_match:
                return self._default_scores(), []
            data = json.loads(json_match.group())
            scores = {
                PersonalityTrait.OPENNESS: self._safe_float(data.get("openness", 0.5)),
                PersonalityTrait.CONSCIENTIOUSNESS: self._safe_float(data.get("conscientiousness", 0.5)),
                PersonalityTrait.EXTRAVERSION: self._safe_float(data.get("extraversion", 0.5)),
                PersonalityTrait.AGREEABLENESS: self._safe_float(data.get("agreeableness", 0.5)),
                PersonalityTrait.NEUROTICISM: self._safe_float(data.get("neuroticism", 0.5)),
            }
            evidence = data.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            return scores, evidence
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("llm_response_parse_failed", error=str(e))
            return self._default_scores(), []

    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float in [0,1]."""
        try:
            f = float(value)
            return max(0.0, min(1.0, f))
        except (ValueError, TypeError):
            return 0.5

    def _default_scores(self) -> dict[PersonalityTrait, float]:
        """Return neutral default scores."""
        return {trait: 0.5 for trait in PersonalityTrait}

    def _default_result(self) -> TraitDetectionResult:
        """Return default result on failure."""
        return TraitDetectionResult(source=AssessmentSource.LLM_ZERO_SHOT, scores=self._default_scores(), confidence=0.3, evidence=[])


class TraitDetector:
    """Ensemble trait detector combining multiple detection methods."""

    def __init__(self, settings: TraitDetectorSettings | None = None, llm_client: Any = None) -> None:
        self._settings = settings or TraitDetectorSettings()
        self._text_detector = TextBasedDetector()
        self._llm_detector = LLMBasedDetector(llm_client) if llm_client else None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the trait detector."""
        self._initialized = True
        logger.info("trait_detector_initialized", llm_enabled=self._llm_detector is not None)

    async def detect(self, text: str, sources: list[AssessmentSource] | None = None) -> OceanScoresDTO:
        """Detect OCEAN traits using ensemble of detection methods."""
        if len(text) < self._settings.min_text_length:
            logger.warning("text_too_short", length=len(text), min_required=self._settings.min_text_length)
            return self._neutral_scores()
        text = text[:self._settings.max_text_length]
        sources = sources or [AssessmentSource.ENSEMBLE]
        results: list[TraitDetectionResult] = []
        text_result = self._text_detector.detect(text)
        results.append(text_result)
        if self._settings.enable_llm_detection and self._llm_detector and AssessmentSource.ENSEMBLE in sources:
            llm_result = await self._llm_detector.detect(text)
            results.append(llm_result)
        return self._ensemble_scores(results)

    def _ensemble_scores(self, results: list[TraitDetectionResult]) -> OceanScoresDTO:
        """Combine multiple detection results into ensemble scores."""
        if not results:
            return self._neutral_scores()
        weights = {AssessmentSource.TEXT_ANALYSIS: self._settings.ensemble_weights_text, AssessmentSource.LIWC_FEATURES: self._settings.ensemble_weights_liwc, AssessmentSource.LLM_ZERO_SHOT: self._settings.ensemble_weights_llm}
        weighted_scores: dict[PersonalityTrait, float] = {trait: 0.0 for trait in PersonalityTrait}
        total_weight = 0.0
        all_evidence: list[str] = []
        for result in results:
            weight = weights.get(result.source, 0.3) * result.confidence
            total_weight += weight
            for trait, score in result.scores.items():
                weighted_scores[trait] += score * weight
            all_evidence.extend(result.evidence)
        if total_weight > 0:
            for trait in PersonalityTrait:
                weighted_scores[trait] /= total_weight
        overall_confidence = sum(r.confidence for r in results) / len(results)
        trait_scores = self._build_trait_scores(weighted_scores, overall_confidence, all_evidence)
        return OceanScoresDTO(
            openness=weighted_scores[PersonalityTrait.OPENNESS],
            conscientiousness=weighted_scores[PersonalityTrait.CONSCIENTIOUSNESS],
            extraversion=weighted_scores[PersonalityTrait.EXTRAVERSION],
            agreeableness=weighted_scores[PersonalityTrait.AGREEABLENESS],
            neuroticism=weighted_scores[PersonalityTrait.NEUROTICISM],
            overall_confidence=overall_confidence,
            trait_scores=trait_scores,
        )

    def _build_trait_scores(self, scores: dict[PersonalityTrait, float], confidence: float, evidence: list[str]) -> list[TraitScoreDTO]:
        """Build detailed trait score objects."""
        result = []
        margin = 0.15 * (1 - confidence)
        for trait, value in scores.items():
            lower = max(0.0, value - margin)
            upper = min(1.0, value + margin)
            trait_evidence = [e for e in evidence if trait.value[:4] in e.lower()] or evidence[:2]
            result.append(TraitScoreDTO(trait=trait, value=value, confidence_lower=lower, confidence_upper=upper, sample_count=1, evidence_markers=trait_evidence[:3]))
        return result

    def _neutral_scores(self) -> OceanScoresDTO:
        """Return neutral scores when detection fails."""
        return OceanScoresDTO(openness=0.5, conscientiousness=0.5, extraversion=0.5, agreeableness=0.5, neuroticism=0.5, overall_confidence=0.3, trait_scores=[])

    async def shutdown(self) -> None:
        """Shutdown the trait detector."""
        self._initialized = False
        logger.info("trait_detector_shutdown")
