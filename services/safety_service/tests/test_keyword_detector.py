"""
Tests for keyword_detector.py - Fast crisis keyword detection with trie-based matching.
"""
import pytest
from decimal import Decimal
from uuid import uuid4
from services.safety_service.src.ml.keyword_detector import (
    KeywordDetector,
    KeywordDetectorConfig,
    KeywordMatch,
    KeywordSeverity,
    KeywordCategory,
)


class TestKeywordDetector:
    """Test suite for KeywordDetector."""

    @pytest.fixture
    def detector(self) -> KeywordDetector:
        """Create keyword detector with default config."""
        config = KeywordDetectorConfig(context_window_chars=30)
        return KeywordDetector(config)

    def test_initialization(self, detector: KeywordDetector) -> None:
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector._config is not None
        assert len(detector._keyword_database) > 0

    def test_detect_critical_keywords(self, detector: KeywordDetector) -> None:
        """Test detection of critical suicide-related keywords."""
        text = "I want to kill myself tonight"
        matches = detector.detect(text)

        assert len(matches) > 0
        assert any(m.severity == KeywordSeverity.CRITICAL for m in matches)
        assert any(m.category == KeywordCategory.SUICIDAL_IDEATION for m in matches)

    def test_detect_high_keywords(self, detector: KeywordDetector) -> None:
        """Test detection of high-risk keywords."""
        text = "I'm feeling hopeless and want to hurt myself"
        matches = detector.detect(text)

        assert len(matches) >= 2
        assert any(m.keyword in ["hopeless", "hurt myself"] for m in matches)

    def test_detect_elevated_keywords(self, detector: KeywordDetector) -> None:
        """Test detection of elevated keywords."""
        text = "I'm feeling very depressed and overwhelmed"
        matches = detector.detect(text)

        assert len(matches) >= 2
        assert any(m.severity == KeywordSeverity.ELEVATED for m in matches)

    def test_case_insensitive_matching(self, detector: KeywordDetector) -> None:
        """Test case-insensitive keyword detection."""
        text1 = "I want to DIE"
        text2 = "i want to die"

        matches1 = detector.detect(text1)
        matches2 = detector.detect(text2)

        assert len(matches1) > 0
        assert len(matches2) > 0
        assert matches1[0].keyword == matches2[0].keyword

    def test_whole_word_matching(self, detector: KeywordDetector) -> None:
        """Test whole word matching to avoid false positives."""
        # "sadness" should not trigger "sad" keyword
        text = "I don't feel sadness anymore"
        matches = detector.detect(text)

        # Should not match 'sad' as a separate word within 'sadness'
        sad_matches = [m for m in matches if m.keyword == "sad"]
        assert len(sad_matches) == 0

    def test_context_extraction(self, detector: KeywordDetector) -> None:
        """Test context window extraction around matches."""
        text = "This is some prefix text. I want to kill myself. This is suffix text."
        matches = detector.detect(text)

        assert len(matches) > 0
        assert len(matches[0].context_snippet) > len(matches[0].keyword)
        assert matches[0].keyword in matches[0].context_snippet

    def test_position_tracking(self, detector: KeywordDetector) -> None:
        """Test accurate position tracking of matches."""
        text = "start hopeless middle depressed end"
        matches = detector.detect(text)

        assert len(matches) >= 2
        # First match should have lower position
        assert matches[0].position < matches[1].position

    def test_calculate_risk_score(self, detector: KeywordDetector) -> None:
        """Test risk score calculation from matches."""
        text = "I want to kill myself I'm hopeless"
        matches = detector.detect(text)

        risk_score = detector.calculate_risk_score(matches)
        assert Decimal("0.0") <= risk_score <= Decimal("1.0")
        assert risk_score >= Decimal("0.9")  # Should be very high risk

    def test_calculate_risk_score_multiple_matches(self, detector: KeywordDetector) -> None:
        """Test risk score with multiple matches uses diminishing returns."""
        text1 = "I'm sad"
        text2 = "I'm sad, depressed, anxious, and overwhelmed"

        matches1 = detector.detect(text1)
        matches2 = detector.detect(text2)

        score1 = detector.calculate_risk_score(matches1)
        score2 = detector.calculate_risk_score(matches2)

        # More matches should increase score but with diminishing returns
        assert score2 > score1
        assert score2 <= Decimal("1.0")

    def test_get_highest_severity(self, detector: KeywordDetector) -> None:
        """Test getting highest severity from matches."""
        text = "I'm sad and want to kill myself"
        matches = detector.detect(text)

        highest = detector.get_highest_severity(matches)
        assert highest == KeywordSeverity.CRITICAL

    def test_get_highest_severity_empty(self, detector: KeywordDetector) -> None:
        """Test highest severity with no matches."""
        text = "I'm feeling okay today"
        matches = detector.detect(text)

        highest = detector.get_highest_severity(matches)
        # Either no matches or only very mild matches
        if highest:
            assert highest in [KeywordSeverity.LOW, KeywordSeverity.INFO]

    def test_get_categories(self, detector: KeywordDetector) -> None:
        """Test category extraction from matches."""
        text = "I'm hopeless and want to hurt myself"
        matches = detector.detect(text)

        categories = detector.get_categories(matches)
        assert KeywordCategory.HOPELESSNESS in categories
        assert KeywordCategory.SELF_HARM in categories

    def test_empty_text(self, detector: KeywordDetector) -> None:
        """Test handling of empty text."""
        matches = detector.detect("")
        assert len(matches) == 0

        risk_score = detector.calculate_risk_score(matches)
        assert risk_score == Decimal("0.0")

    def test_no_keywords_text(self, detector: KeywordDetector) -> None:
        """Test text with no crisis keywords."""
        text = "The weather is nice today and I went for a walk"
        matches = detector.detect(text)

        assert len(matches) == 0
        risk_score = detector.calculate_risk_score(matches)
        assert risk_score == Decimal("0.0")

    def test_max_matches_limit(self) -> None:
        """Test maximum matches limit is respected."""
        config = KeywordDetectorConfig(max_matches_per_text=3)
        detector = KeywordDetector(config)

        # Text with many keywords
        text = "sad stressed worried depressed anxious overwhelmed scared upset frustrated tired"
        matches = detector.detect(text)

        assert len(matches) <= 3

    def test_user_id_logging(self, detector: KeywordDetector) -> None:
        """Test detection with user ID for logging."""
        user_id = uuid4()
        text = "I want to die"

        matches = detector.detect(text, user_id=user_id)
        assert len(matches) > 0

    def test_match_attributes(self, detector: KeywordDetector) -> None:
        """Test all match attributes are populated correctly."""
        text = "I'm feeling hopeless"
        matches = detector.detect(text)

        assert len(matches) > 0
        match = matches[0]

        assert match.keyword is not None
        assert match.position >= 0
        assert match.severity in KeywordSeverity
        assert match.category in KeywordCategory
        assert Decimal("0.0") <= match.confidence <= Decimal("1.0")
        assert len(match.context_snippet) > 0
        assert Decimal("0.0") <= match.weight <= Decimal("1.0")

    def test_sorting_by_severity(self, detector: KeywordDetector) -> None:
        """Test matches are sorted by severity (critical first)."""
        text = "I'm sad and want to kill myself"
        matches = detector.detect(text)

        assert len(matches) >= 2
        # First match should be most severe
        assert matches[0].severity.value in ["CRITICAL", "HIGH"]

    def test_plan_intent_detection(self, detector: KeywordDetector) -> None:
        """Test detection of plan/intent keywords."""
        text = "I have a plan to end it all"
        matches = detector.detect(text)

        assert len(matches) > 0
        assert any(m.category == KeywordCategory.PLAN_INTENT for m in matches)

    def test_means_access_detection(self, detector: KeywordDetector) -> None:
        """Test detection of means access keywords."""
        text = "I have pills at home"
        matches = detector.detect(text)

        assert len(matches) > 0
        assert any(m.category == KeywordCategory.MEANS_ACCESS for m in matches)
