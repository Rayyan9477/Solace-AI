"""
Batch 3.3 Integration Test - End-to-end functional validation of all ML components.
Tests the full pipeline from keyword detection through LLM assessment and contraindication checking.
"""
import pytest
from decimal import Decimal
from uuid import uuid4
from services.safety_service.src.ml.keyword_detector import KeywordDetector
from services.safety_service.src.ml.sentiment_analyzer import SentimentAnalyzer
from services.safety_service.src.ml.pattern_matcher import PatternMatcher
from services.safety_service.src.ml.llm_assessor import LLMAssessor
from services.safety_service.src.ml.contraindication import (
    ContraindicationChecker,
    TherapyTechnique,
    MentalHealthCondition,
)


class TestBatch33Integration:
    """Integration tests for Batch 3.3 ML components."""

    @pytest.fixture
    def keyword_detector(self) -> KeywordDetector:
        """Initialize keyword detector."""
        return KeywordDetector()

    @pytest.fixture
    def sentiment_analyzer(self) -> SentimentAnalyzer:
        """Initialize sentiment analyzer."""
        return SentimentAnalyzer()

    @pytest.fixture
    def pattern_matcher(self) -> PatternMatcher:
        """Initialize pattern matcher."""
        return PatternMatcher()

    @pytest.fixture
    def llm_assessor(self) -> LLMAssessor:
        """Initialize LLM assessor."""
        return LLMAssessor()

    @pytest.fixture
    def contraindication_checker(self) -> ContraindicationChecker:
        """Initialize contraindication checker."""
        return ContraindicationChecker()

    def test_crisis_detection_pipeline_critical(
        self,
        keyword_detector: KeywordDetector,
        sentiment_analyzer: SentimentAnalyzer,
        pattern_matcher: PatternMatcher,
    ) -> None:
        """Test full crisis detection pipeline for critical case."""
        text = "I want to kill myself tonight, I have a plan and feel hopeless"
        user_id = uuid4()

        # Step 1: Keyword detection
        keyword_matches = keyword_detector.detect(text, user_id)
        assert len(keyword_matches) > 0
        keyword_risk = keyword_detector.calculate_risk_score(keyword_matches)
        assert keyword_risk >= Decimal("0.9")

        # Step 2: Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze(text, user_id)
        assert sentiment_result.risk_score >= Decimal("0.7")
        assert len(sentiment_result.distress_indicators) > 0

        # Step 3: Pattern matching
        pattern_matches = pattern_matcher.detect(text, user_id)
        assert len(pattern_matches) > 0
        pattern_risk = pattern_matcher.calculate_risk_score(pattern_matches)
        assert pattern_risk >= Decimal("0.9")
        assert pattern_matcher.has_critical_patterns(pattern_matches)

        # Aggregate assessment
        avg_risk = (keyword_risk + sentiment_result.risk_score + pattern_risk) / Decimal("3")
        assert avg_risk >= Decimal("0.8")
        print(f"[OK] Critical case detected - Avg risk: {avg_risk}")

    def test_crisis_detection_pipeline_moderate(
        self,
        keyword_detector: KeywordDetector,
        sentiment_analyzer: SentimentAnalyzer,
        pattern_matcher: PatternMatcher,
    ) -> None:
        """Test full crisis detection pipeline for moderate case."""
        text = "I'm feeling very stressed and anxious about everything"
        user_id = uuid4()

        # Step 1: Keyword detection
        keyword_matches = keyword_detector.detect(text, user_id)
        keyword_risk = keyword_detector.calculate_risk_score(keyword_matches)

        # Step 2: Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze(text, user_id)

        # Step 3: Pattern matching
        pattern_matches = pattern_matcher.detect(text, user_id)
        pattern_risk = pattern_matcher.calculate_risk_score(pattern_matches)

        # Aggregate assessment (moderate case)
        if keyword_risk > 0 or sentiment_result.risk_score > 0 or pattern_risk > 0:
            avg_risk = (keyword_risk + sentiment_result.risk_score + pattern_risk) / Decimal("3")
            assert avg_risk < Decimal("0.9")  # Not critical
        print(f"[OK] Moderate case processed correctly")

    def test_crisis_detection_pipeline_safe(
        self,
        keyword_detector: KeywordDetector,
        sentiment_analyzer: SentimentAnalyzer,
        pattern_matcher: PatternMatcher,
    ) -> None:
        """Test full crisis detection pipeline for safe case."""
        text = "I had a wonderful day and I'm feeling great"
        user_id = uuid4()

        # Step 1: Keyword detection
        keyword_matches = keyword_detector.detect(text, user_id)
        keyword_risk = keyword_detector.calculate_risk_score(keyword_matches)
        assert keyword_risk < Decimal("0.3")

        # Step 2: Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze(text, user_id)
        assert sentiment_result.compound_score > Decimal("0.0")  # Positive

        # Step 3: Pattern matching
        pattern_matches = pattern_matcher.detect(text, user_id)
        assert len(pattern_matches) == 0
        print("[OK] Safe case processed correctly")

    @pytest.mark.asyncio
    async def test_llm_assessment_integration(self, llm_assessor: LLMAssessor) -> None:
        """Test LLM assessor integration."""
        user_id = uuid4()

        # Test critical text
        critical_text = "I want to die"
        result = await llm_assessor.assess(critical_text, user_id)
        assert result is not None
        assert result.risk_score > Decimal("0.0")
        print(f"[OK] LLM assessment: risk_level={result.risk_level.value}, score={result.risk_score}")

        # Test safe text
        safe_text = "I'm feeling good today"
        result = await llm_assessor.assess(safe_text, user_id)
        assert result is not None
        print("[OK] LLM assessment completed for safe text")

    def test_contraindication_integration(
        self, contraindication_checker: ContraindicationChecker
    ) -> None:
        """Test contraindication checker integration."""
        user_id = uuid4()

        # Test absolute contraindication
        result = contraindication_checker.check(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACTIVE_PSYCHOSIS],
            user_id=user_id,
        )
        assert result.is_safe is False
        assert result.safety_level == "UNSAFE"
        assert len(result.contraindications) > 0
        print(f"[OK] Contraindication detected: {result.clinical_notes}")

        # Test safe technique
        result = contraindication_checker.check(
            technique=TherapyTechnique.GROUNDING_TECHNIQUES,
            conditions=[],
            user_id=user_id,
        )
        assert result.is_safe is True
        assert result.safety_level == "SAFE"
        print("[OK] Safe technique validated")

        # Test alternatives
        alternatives = contraindication_checker.get_safe_alternatives(
            technique=TherapyTechnique.EXPOSURE_THERAPY,
            conditions=[MentalHealthCondition.ACUTE_CRISIS],
        )
        assert len(alternatives) > 0
        assert TherapyTechnique.EXPOSURE_THERAPY not in alternatives
        print(f"[OK] Safe alternatives provided: {[t.value for t in alternatives[:2]]}")

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(
        self,
        keyword_detector: KeywordDetector,
        sentiment_analyzer: SentimentAnalyzer,
        pattern_matcher: PatternMatcher,
        llm_assessor: LLMAssessor,
        contraindication_checker: ContraindicationChecker,
    ) -> None:
        """Test complete end-to-end pipeline with all components."""
        text = "I'm feeling depressed and hopeless, struggling with anxiety"
        user_id = uuid4()

        # Step 1: Multi-layer detection
        keyword_matches = keyword_detector.detect(text, user_id)
        sentiment_result = sentiment_analyzer.analyze(text, user_id)
        pattern_matches = pattern_matcher.detect(text, user_id)

        keyword_risk = keyword_detector.calculate_risk_score(keyword_matches)
        pattern_risk = pattern_matcher.calculate_risk_score(pattern_matches)

        # Step 2: Aggregate risk assessment
        total_detections = len(keyword_matches) + len(pattern_matches)
        avg_risk = (
            keyword_risk + sentiment_result.risk_score + pattern_risk
        ) / Decimal("3")

        print(f"[OK] Detection summary:")
        print(f"    Keyword matches: {len(keyword_matches)}")
        print(f"    Pattern matches: {len(pattern_matches)}")
        print(f"    Sentiment: {sentiment_result.emotional_state.value}")
        print(f"    Average risk: {avg_risk}")

        # Step 3: LLM deep assessment
        llm_result = await llm_assessor.assess(text, user_id)
        print(f"    LLM risk level: {llm_result.risk_level.value}")
        print(f"    LLM confidence: {llm_result.confidence}")

        # Step 4: Determine conditions from assessment
        conditions = []
        if "depressed" in text.lower():
            conditions.append(MentalHealthCondition.SEVERE_DEPRESSION)
        if "anxiety" in text.lower() or "anxious" in text.lower():
            conditions.append(MentalHealthCondition.SEVERE_PTSD)

        # Step 5: Check contraindications for potential techniques
        cognitive_check = contraindication_checker.check(
            technique=TherapyTechnique.COGNITIVE_RESTRUCTURING,
            conditions=conditions,
            user_id=user_id,
        )
        print(f"    Cognitive restructuring: {cognitive_check.safety_level}")

        behavioral_check = contraindication_checker.check(
            technique=TherapyTechnique.BEHAVIORAL_ACTIVATION,
            conditions=conditions,
            user_id=user_id,
        )
        print(f"    Behavioral activation: {behavioral_check.safety_level}")

        # Validate pipeline worked
        assert total_detections > 0 or sentiment_result.risk_score > 0
        assert llm_result is not None
        assert cognitive_check is not None
        assert behavioral_check is not None

        print("[OK] Full pipeline integration test passed")

    def test_component_initialization_performance(
        self,
        keyword_detector: KeywordDetector,
        sentiment_analyzer: SentimentAnalyzer,
        pattern_matcher: PatternMatcher,
        contraindication_checker: ContraindicationChecker,
    ) -> None:
        """Test that all components initialize correctly."""
        assert keyword_detector is not None
        assert len(keyword_detector._keyword_database) > 0
        print(f"[OK] Keyword detector: {len(keyword_detector._keyword_database)} keywords loaded")

        assert sentiment_analyzer is not None
        assert len(sentiment_analyzer._lexicon) > 0
        print(f"[OK] Sentiment analyzer: {len(sentiment_analyzer._lexicon)} terms loaded")

        assert pattern_matcher is not None
        assert len(pattern_matcher._patterns) > 0
        print(f"[OK] Pattern matcher: {len(pattern_matcher._patterns)} patterns loaded")

        assert contraindication_checker is not None
        assert len(contraindication_checker._rules) > 0
        print(
            f"[OK] Contraindication checker: {len(contraindication_checker._rules)} rules loaded"
        )

    def test_edge_cases_and_error_handling(
        self,
        keyword_detector: KeywordDetector,
        sentiment_analyzer: SentimentAnalyzer,
        pattern_matcher: PatternMatcher,
    ) -> None:
        """Test edge cases and error handling."""
        # Empty text
        assert len(keyword_detector.detect("")) == 0
        assert sentiment_analyzer.analyze("").risk_score == Decimal("0.0")
        assert len(pattern_matcher.detect("")) == 0
        print("[OK] Empty text handled correctly")

        # Very long text
        long_text = "word " * 1000
        keyword_detector.detect(long_text)
        sentiment_analyzer.analyze(long_text)
        pattern_matcher.detect(long_text)
        print("[OK] Long text handled correctly")

        # Special characters
        special_text = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        keyword_detector.detect(special_text)
        sentiment_analyzer.analyze(special_text)
        pattern_matcher.detect(special_text)
        print("[OK] Special characters handled correctly")
