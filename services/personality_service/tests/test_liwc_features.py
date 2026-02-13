"""Tests for LIWC feature extraction."""
from __future__ import annotations
import pytest

from services.personality_service.src.schemas import PersonalityTrait, OceanScoresDTO
from services.personality_service.src.ml.liwc_features import (
    LIWCProcessorSettings, LIWCProcessor, LIWCFeatureVector,
    LIWCDictionary, FeatureExtractor, PersonalityMapper,
)


class TestLIWCProcessorSettings:
    """Tests for LIWCProcessorSettings."""

    def test_default_settings(self) -> None:
        settings = LIWCProcessorSettings()
        assert settings.min_word_count == 20
        assert settings.max_text_length == 10000
        assert settings.normalize_scores is True
        assert settings.confidence_base == 0.4
        assert settings.max_confidence == 0.8

    def test_custom_settings(self) -> None:
        settings = LIWCProcessorSettings(
            min_word_count=50,
            confidence_base=0.5,
        )
        assert settings.min_word_count == 50
        assert settings.confidence_base == 0.5


class TestLIWCFeatureVector:
    """Tests for LIWCFeatureVector."""

    def test_default_values(self) -> None:
        vector = LIWCFeatureVector()
        assert vector.word_count == 0
        assert vector.positive_emotion == 0.0
        assert vector.negative_emotion == 0.0

    def test_to_dict(self) -> None:
        vector = LIWCFeatureVector(word_count=100, positive_emotion=2.5)
        d = vector.to_dict()
        assert d['word_count'] == 100.0
        assert d['positive_emotion'] == 2.5
        assert 'cognitive' in d


class TestLIWCDictionary:
    """Tests for LIWCDictionary."""

    def test_get_categories(self) -> None:
        dictionary = LIWCDictionary()
        categories = dictionary.get_categories()
        assert 'i_words' in categories
        assert 'positive_emotion' in categories
        assert 'cognitive' in categories
        assert 'achievement' in categories

    def test_i_words_category(self) -> None:
        dictionary = LIWCDictionary()
        categories = dictionary.get_categories()
        assert 'i' in categories['i_words']
        assert 'me' in categories['i_words']
        assert 'my' in categories['i_words']

    def test_positive_emotion_category(self) -> None:
        dictionary = LIWCDictionary()
        categories = dictionary.get_categories()
        assert 'happy' in categories['positive_emotion']
        assert 'love' in categories['positive_emotion']
        assert 'wonderful' in categories['positive_emotion']


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_extract_basic(self) -> None:
        extractor = FeatureExtractor()
        text = "I am happy today. We went to see my friends."
        features = extractor.extract(text)
        assert features.word_count > 0
        assert features.i_words > 0
        assert features.we_words > 0

    def test_extract_positive_emotion(self) -> None:
        extractor = FeatureExtractor()
        text = "I am so happy and grateful! Everything is wonderful and love fills the air!"
        features = extractor.extract(text)
        assert features.positive_emotion > 0

    def test_extract_negative_emotion(self) -> None:
        extractor = FeatureExtractor()
        text = "I am sad and worried. Everything seems terrible and I feel hurt."
        features = extractor.extract(text)
        assert features.negative_emotion > 0

    def test_extract_cognitive_words(self) -> None:
        extractor = FeatureExtractor()
        text = "I think and believe that we should understand and analyze this problem."
        features = extractor.extract(text)
        assert features.cognitive > 0

    def test_extract_achievement_words(self) -> None:
        extractor = FeatureExtractor()
        text = "We must work hard to achieve success and accomplish our goals."
        features = extractor.extract(text)
        assert features.achievement > 0

    def test_extract_punctuation(self) -> None:
        extractor = FeatureExtractor()
        text = "What do you think? Is this exciting! Really?"
        features = extractor.extract(text)
        assert features.question_marks > 0
        assert features.exclamation_marks > 0

    def test_extract_social_words(self) -> None:
        extractor = FeatureExtractor()
        text = "I love to talk with my friend and share stories with the group."
        features = extractor.extract(text)
        assert features.social > 0
        assert features.friends > 0

    def test_extract_empty_text(self) -> None:
        extractor = FeatureExtractor()
        features = extractor.extract("")
        assert features.word_count == 0


class TestPersonalityMapper:
    """Tests for PersonalityMapper."""

    def test_map_neutral_features(self) -> None:
        mapper = PersonalityMapper()
        features = LIWCFeatureVector(word_count=100)
        scores = mapper.map_to_ocean(features)
        assert len(scores) == 5
        for trait in PersonalityTrait:
            assert trait in scores
            assert 0.0 <= scores[trait] <= 1.0

    def test_map_high_insight_increases_openness(self) -> None:
        mapper = PersonalityMapper()
        features = LIWCFeatureVector(word_count=100, insight=5.0, cognitive=3.0)
        scores = mapper.map_to_ocean(features)
        assert scores[PersonalityTrait.OPENNESS] > 0.5

    def test_map_high_achievement_increases_conscientiousness(self) -> None:
        mapper = PersonalityMapper()
        features = LIWCFeatureVector(word_count=100, achievement=5.0, work=3.0)
        scores = mapper.map_to_ocean(features)
        assert scores[PersonalityTrait.CONSCIENTIOUSNESS] > 0.5

    def test_map_high_social_increases_extraversion(self) -> None:
        mapper = PersonalityMapper()
        features = LIWCFeatureVector(word_count=100, social=5.0, we_words=3.0)
        scores = mapper.map_to_ocean(features)
        assert scores[PersonalityTrait.EXTRAVERSION] > 0.5

    def test_map_high_positive_increases_agreeableness(self) -> None:
        mapper = PersonalityMapper()
        features = LIWCFeatureVector(word_count=100, positive_emotion=5.0, family=2.0)
        scores = mapper.map_to_ocean(features)
        assert scores[PersonalityTrait.AGREEABLENESS] > 0.5

    def test_map_high_negative_increases_neuroticism(self) -> None:
        mapper = PersonalityMapper()
        features = LIWCFeatureVector(word_count=100, negative_emotion=5.0, anxiety=3.0)
        scores = mapper.map_to_ocean(features)
        assert scores[PersonalityTrait.NEUROTICISM] > 0.5


class TestLIWCProcessor:
    """Tests for LIWCProcessor."""

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        assert processor._initialized is True

    @pytest.mark.asyncio
    async def test_process_returns_ocean_scores(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        text = "I think we should work hard to achieve our goals. My friends and family support me. I am happy and grateful for everything. The future looks wonderful."
        result = await processor.process(text)
        assert isinstance(result, OceanScoresDTO)
        assert 0.0 <= result.openness <= 1.0
        assert 0.0 <= result.conscientiousness <= 1.0

    @pytest.mark.asyncio
    async def test_process_short_text_returns_neutral(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        result = await processor.process("Hello")
        assert result.overall_confidence < 0.3

    @pytest.mark.asyncio
    async def test_process_empty_text_returns_neutral(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        result = await processor.process("")
        assert result.overall_confidence == 0.2

    @pytest.mark.asyncio
    async def test_process_confidence_increases_with_length(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        short_text = "I am happy today. My friends are great. We have fun together."
        long_text = short_text * 10
        short_result = await processor.process(short_text)
        long_result = await processor.process(long_text)
        assert long_result.overall_confidence >= short_result.overall_confidence

    @pytest.mark.asyncio
    async def test_extract_features(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        text = "I am thinking about my achievements and goals for the future."
        features = await processor.extract_features(text)
        assert isinstance(features, LIWCFeatureVector)
        assert features.word_count > 0

    @pytest.mark.asyncio
    async def test_process_positive_text(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        text = "I am so happy and grateful today! Everything is wonderful and amazing. I love my friends and family so much. Joy fills my heart with warmth."
        result = await processor.process(text)
        assert result.agreeableness >= 0.5

    @pytest.mark.asyncio
    async def test_process_achievement_text(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        text = "I work hard every day to achieve my goals. Success comes from effort and dedication. I accomplish tasks efficiently and improve constantly."
        result = await processor.process(text)
        assert result.conscientiousness >= 0.5

    @pytest.mark.asyncio
    async def test_process_social_text(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        text = "We love spending time together with our friends. Meeting people and sharing stories is so exciting! Our family gatherings are always fun."
        result = await processor.process(text)
        assert result.extraversion >= 0.5

    @pytest.mark.asyncio
    async def test_process_anxious_text(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        text = "I am worried and anxious about everything. Fear and stress dominate my thoughts. I feel sad and nervous most of the time."
        result = await processor.process(text)
        assert result.neuroticism >= 0.5

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        await processor.shutdown()
        assert processor._initialized is False

    @pytest.mark.asyncio
    async def test_trait_scores_have_evidence(self) -> None:
        processor = LIWCProcessor()
        await processor.initialize()
        text = "I think deeply about complex problems. My insight helps me understand and realize new things. I discover and learn constantly."
        result = await processor.process(text)
        openness_score = next((s for s in result.trait_scores if s.trait == PersonalityTrait.OPENNESS), None)
        assert openness_score is not None, "Expected openness trait score for text about deep thinking"
        assert len(openness_score.evidence_markers) > 0, "Expected evidence markers for openness"
