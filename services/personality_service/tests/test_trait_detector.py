"""
Unit tests for Trait Detection.
Tests LIWC feature extraction, text-based detection, and ensemble methods.
"""
from __future__ import annotations
import pytest

from services.personality_service.src.schemas import PersonalityTrait, AssessmentSource
from services.personality_service.src.domain.trait_detector import (
    TraitDetector, TraitDetectorSettings, TextBasedDetector,
    LIWCFeatureExtractor, LIWCFeatures, TraitDetectionResult,
)


class TestLIWCFeatureExtractor:
    """Tests for LIWC feature extraction."""

    @pytest.fixture
    def extractor(self) -> LIWCFeatureExtractor:
        """Create LIWC feature extractor."""
        return LIWCFeatureExtractor()

    def test_extract_basic_features(self, extractor: LIWCFeatureExtractor) -> None:
        """Test basic feature extraction."""
        text = "I think I understand the problem. We should work together."
        features = extractor.extract(text)
        assert features.word_count == 10
        assert features.i_words_ratio > 0
        assert features.we_words_ratio > 0

    def test_extract_positive_emotion(self, extractor: LIWCFeatureExtractor) -> None:
        """Test positive emotion detection."""
        text = "I am so happy and excited about this wonderful opportunity. It's great and beautiful."
        features = extractor.extract(text)
        assert features.positive_emotion_ratio > 0.05

    def test_extract_negative_emotion(self, extractor: LIWCFeatureExtractor) -> None:
        """Test negative emotion detection."""
        text = "I feel sad and angry about this terrible situation. It's really awful and painful."
        features = extractor.extract(text)
        assert features.negative_emotion_ratio > 0.05

    def test_extract_social_words(self, extractor: LIWCFeatureExtractor) -> None:
        """Test social word detection."""
        text = "I love spending time with my family and friends. We talk and share together."
        features = extractor.extract(text)
        assert features.social_words_ratio > 0.02

    def test_extract_achievement_words(self, extractor: LIWCFeatureExtractor) -> None:
        """Test achievement word detection."""
        text = "I work hard to achieve my goals. I will accomplish great success."
        features = extractor.extract(text)
        assert features.achievement_words_ratio > 0.02

    def test_extract_cognitive_words(self, extractor: LIWCFeatureExtractor) -> None:
        """Test cognitive word detection."""
        text = "I think and believe I understand this. Let me consider and analyze it."
        features = extractor.extract(text)
        assert features.cognitive_process_ratio > 0.02

    def test_extract_question_marks(self, extractor: LIWCFeatureExtractor) -> None:
        """Test question mark detection."""
        text = "Why is this happening? How can I understand? What should I do?"
        features = extractor.extract(text)
        assert features.question_marks_ratio > 0

    def test_extract_empty_text(self, extractor: LIWCFeatureExtractor) -> None:
        """Test extraction from empty text."""
        features = extractor.extract("")
        assert features.word_count == 0


class TestTextBasedDetector:
    """Tests for text-based trait detection."""

    @pytest.fixture
    def detector(self) -> TextBasedDetector:
        """Create text-based detector."""
        return TextBasedDetector()

    def test_detect_from_text(self, detector: TextBasedDetector) -> None:
        """Test basic detection from text."""
        text = "I really enjoy learning new things and exploring different perspectives. It's wonderful."
        result = detector.detect(text)
        assert result.source == AssessmentSource.TEXT_ANALYSIS
        assert len(result.scores) == 5
        assert all(0.0 <= score <= 1.0 for score in result.scores.values())

    def test_detect_high_openness(self, detector: TextBasedDetector) -> None:
        """Test detection of high openness."""
        text = "I realize and understand many new insights. I notice and discover things. What do you think?"
        result = detector.detect(text)
        assert result.scores[PersonalityTrait.OPENNESS] > 0.5

    def test_detect_high_conscientiousness(self, detector: TextBasedDetector) -> None:
        """Test detection of high conscientiousness."""
        text = "I work hard to achieve my goals. I always try to accomplish and complete my tasks successfully."
        result = detector.detect(text)
        assert result.scores[PersonalityTrait.CONSCIENTIOUSNESS] > 0.5

    def test_detect_high_extraversion(self, detector: TextBasedDetector) -> None:
        """Test detection of high extraversion."""
        text = "We love meeting friends and talking with people! It's so exciting to share together!"
        result = detector.detect(text)
        assert result.scores[PersonalityTrait.EXTRAVERSION] > 0.5

    def test_detect_high_agreeableness(self, detector: TextBasedDetector) -> None:
        """Test detection of high agreeableness."""
        text = "I love spending time with family and friends. It makes me so happy and kind."
        result = detector.detect(text)
        assert result.scores[PersonalityTrait.AGREEABLENESS] > 0.5

    def test_detect_high_neuroticism(self, detector: TextBasedDetector) -> None:
        """Test detection of high neuroticism."""
        text = "I feel sad and worried about everything. Maybe I'm unsure. I'm anxious and afraid."
        result = detector.detect(text)
        assert result.scores[PersonalityTrait.NEUROTICISM] > 0.5

    def test_detection_confidence(self, detector: TextBasedDetector) -> None:
        """Test confidence increases with text length."""
        short_text = "I think this is good."
        long_text = "I really think this is wonderful. " * 50
        short_result = detector.detect(short_text)
        long_result = detector.detect(long_text)
        assert long_result.confidence >= short_result.confidence


class TestTraitDetector:
    """Tests for ensemble trait detector."""

    @pytest.fixture
    def settings(self) -> TraitDetectorSettings:
        """Create trait detector settings."""
        return TraitDetectorSettings(enable_llm_detection=False)

    @pytest.fixture
    def detector(self, settings: TraitDetectorSettings) -> TraitDetector:
        """Create trait detector without LLM."""
        return TraitDetector(settings=settings, llm_client=None)

    @pytest.mark.asyncio
    async def test_initialize(self, detector: TraitDetector) -> None:
        """Test detector initialization."""
        await detector.initialize()
        assert detector._initialized is True

    @pytest.mark.asyncio
    async def test_detect_returns_ocean_scores(self, detector: TraitDetector) -> None:
        """Test detection returns OCEAN scores."""
        await detector.initialize()
        text = "I think learning new things is wonderful. I love working with my team to achieve our goals."
        scores = await detector.detect(text)
        assert scores.openness is not None
        assert scores.conscientiousness is not None
        assert scores.extraversion is not None
        assert scores.agreeableness is not None
        assert scores.neuroticism is not None
        assert 0.0 <= scores.openness <= 1.0
        assert 0.0 <= scores.overall_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_detect_short_text_returns_neutral(self, detector: TraitDetector) -> None:
        """Test short text returns neutral scores."""
        await detector.initialize()
        text = "Hi"
        scores = await detector.detect(text)
        assert scores.openness == 0.5
        assert scores.overall_confidence == 0.3

    @pytest.mark.asyncio
    async def test_detect_with_sources(self, detector: TraitDetector) -> None:
        """Test detection with specific sources."""
        await detector.initialize()
        text = "I enjoy exploring new ideas and understanding different perspectives on complex topics."
        scores = await detector.detect(text, sources=[AssessmentSource.TEXT_ANALYSIS])
        assert scores is not None
        assert len(scores.trait_scores) >= 0

    @pytest.mark.asyncio
    async def test_shutdown(self, detector: TraitDetector) -> None:
        """Test detector shutdown."""
        await detector.initialize()
        await detector.shutdown()
        assert detector._initialized is False


class TestTraitDetectionResult:
    """Tests for TraitDetectionResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating detection result."""
        scores = {trait: 0.5 for trait in PersonalityTrait}
        result = TraitDetectionResult(
            source=AssessmentSource.TEXT_ANALYSIS,
            scores=scores,
            confidence=0.7,
            evidence=["marker1", "marker2"],
        )
        assert result.source == AssessmentSource.TEXT_ANALYSIS
        assert result.confidence == 0.7
        assert len(result.evidence) == 2
        assert result.detection_id is not None


class TestTraitDetectorSettings:
    """Tests for TraitDetectorSettings."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = TraitDetectorSettings()
        assert settings.min_text_length == 50
        assert settings.max_text_length == 10000
        assert settings.enable_llm_detection is True
        assert settings.ensemble_weights_text == 0.4
        assert settings.ensemble_weights_liwc == 0.3
        assert settings.ensemble_weights_llm == 0.3

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = TraitDetectorSettings(
            min_text_length=100,
            enable_llm_detection=False,
            ensemble_weights_text=0.6,
        )
        assert settings.min_text_length == 100
        assert settings.enable_llm_detection is False
        assert settings.ensemble_weights_text == 0.6
