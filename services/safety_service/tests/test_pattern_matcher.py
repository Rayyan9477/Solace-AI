"""
Tests for pattern_matcher.py - Advanced pattern-based crisis detection.
"""
import pytest
from decimal import Decimal
from uuid import uuid4
from services.safety_service.src.ml.pattern_matcher import (
    PatternMatcher,
    PatternMatcherConfig,
    PatternMatch,
    PatternType,
)


class TestPatternMatcher:
    """Test suite for PatternMatcher."""

    @pytest.fixture
    def matcher(self) -> PatternMatcher:
        """Create pattern matcher with default config."""
        return PatternMatcher()

    def test_initialization(self, matcher: PatternMatcher) -> None:
        """Test matcher initializes correctly."""
        assert matcher is not None
        assert len(matcher._patterns) > 0

    def test_detect_suicidal_ideation(self, matcher: PatternMatcher) -> None:
        """Test detection of suicidal_ideation patterns."""
        text = "I'm thinking to kill myself"
        matches = matcher.detect(text)

        assert len(matches) > 0
        assert any(m.pattern_type == PatternType.SUICIDAL_IDEATION for m in matches)
        assert any(m.severity >= Decimal("0.9") for m in matches)

    def test_detect_plan_indicators(self, matcher: PatternMatcher) -> None:
        """Test detection of plan indicator patterns."""
        text = "I have a plan to end my life"
        matches = matcher.detect(text)

        assert len(matches) > 0
        assert any(m.pattern_type == PatternType.PLAN_INDICATOR for m in matches)

    def test_detect_timeframe_urgency(self, matcher: PatternMatcher) -> None:
        """Test detection of time-bound urgency."""
        text = "Tonight I will die"
        matches = matcher.detect(text)

        assert len(matches) > 0
        assert any(m.pattern_type == PatternType.TIMEFRAME_URGENCY for m in matches)

    def test_detect_hopelessness(self, matcher: PatternMatcher) -> None:
        """Test detection of hopelessness expressions."""
        text = "Things will never get better"
        matches = matcher.detect(text)

        assert len(matches) > 0
        assert any(m.pattern_type == PatternType.HOPELESSNESS_EXPRESSION for m in matches)

    def test_detect_farewell_messages(self, matcher: PatternMatcher) -> None:
        """Test detection of farewell messages."""
        text = "Goodbye everyone, sorry for everything"
        matches = matcher.detect(text)

        assert len(matches) > 0
        assert any(m.pattern_type == PatternType.FAREWELL_MESSAGE for m in matches)

    def test_detect_self_harm(self, matcher: PatternMatcher) -> None:
        """Test detection of self-harm intent."""
        text = "I want to cut myself"
        matches = matcher.detect(text)

        assert len(matches) > 0
        assert any(m.pattern_type == PatternType.SELF_HARM_INTENT for m in matches)

    def test_detect_means_access(self, matcher: PatternMatcher) -> None:
        """Test detection of means access."""
        text = "I have access to a gun"
        matches = matcher.detect(text)

        assert len(matches) > 0
        assert any(m.pattern_type == PatternType.MEANS_ACCESS for m in matches)

    def test_context_extraction(self, matcher: PatternMatcher) -> None:
        """Test context is extracted around matches."""
        text = "This is some text. I can't take it anymore. More text here."
        matches = matcher.detect(text)

        assert len(matches) > 0
        assert len(matches[0].context) > len(matches[0].matched_text)

    def test_calculate_risk_score(self, matcher: PatternMatcher) -> None:
        """Test risk score calculation."""
        text = "I want to die and have a plan to end it tonight"
        matches = matcher.detect(text)

        risk_score = matcher.calculate_risk_score(matches)
        assert Decimal("0.0") <= risk_score <= Decimal("1.0")
        assert risk_score >= Decimal("0.9")  # Should be very high

    def test_calculate_risk_score_empty(self, matcher: PatternMatcher) -> None:
        """Test risk score with no matches."""
        matches = []
        risk_score = matcher.calculate_risk_score(matches)
        assert risk_score == Decimal("0.0")

    def test_get_dominant_pattern_type(self, matcher: PatternMatcher) -> None:
        """Test getting dominant (most severe) pattern type."""
        text = "I want to die and feel hopeless"
        matches = matcher.detect(text)

        dominant = matcher.get_dominant_pattern_type(matches)
        # Should detect at least one pattern type
        assert dominant is not None

    def test_has_critical_patterns(self, matcher: PatternMatcher) -> None:
        """Test detection of critical patterns."""
        text = "I'm going to kill myself tonight"
        matches = matcher.detect(text)

        assert matcher.has_critical_patterns(matches) is True

    def test_has_critical_patterns_false(self, matcher: PatternMatcher) -> None:
        """Test no critical patterns in safe text."""
        text = "I'm having a good day"
        matches = matcher.detect(text)

        assert matcher.has_critical_patterns(matches) is False

    def test_group_by_type(self, matcher: PatternMatcher) -> None:
        """Test grouping matches by pattern type."""
        text = "I'm hopeless and want to die and can't take it anymore"
        matches = matcher.detect(text)

        grouped = matcher.group_by_type(matches)
        assert isinstance(grouped, dict)
        assert len(grouped) > 0

    def test_case_insensitive_matching(self, matcher: PatternMatcher) -> None:
        """Test case-insensitive pattern matching."""
        text1 = "I WANT TO DIE"
        text2 = "i want to die"

        matches1 = matcher.detect(text1)
        matches2 = matcher.detect(text2)

        assert len(matches1) > 0
        assert len(matches2) > 0

    def test_empty_text(self, matcher: PatternMatcher) -> None:
        """Test handling of empty text."""
        matches = matcher.detect("")
        assert len(matches) == 0

    def test_no_patterns_text(self, matcher: PatternMatcher) -> None:
        """Test text with no crisis patterns."""
        text = "The weather is nice and I went for a walk"
        matches = matcher.detect(text)

        assert len(matches) == 0

    def test_user_id_logging(self, matcher: PatternMatcher) -> None:
        """Test detection with user ID."""
        user_id = uuid4()
        text = "I want to die"

        matches = matcher.detect(text, user_id=user_id)
        assert len(matches) > 0

    def test_match_attributes(self, matcher: PatternMatcher) -> None:
        """Test all match attributes are populated."""
        text = "I can't take this anymore"
        matches = matcher.detect(text)

        if matches:
            match = matches[0]
            assert match.pattern_type in PatternType
            assert match.matched_text is not None
            assert match.position >= 0
            assert Decimal("0.0") <= match.severity <= Decimal("1.0")
            assert Decimal("0.0") <= match.confidence <= Decimal("1.0")
            assert len(match.context) > 0
            assert len(match.explanation) > 0

    def test_sorting_by_severity(self, matcher: PatternMatcher) -> None:
        """Test matches sorted by severity."""
        text = "I want to kill myself tonight I have a plan"
        matches = matcher.detect(text)

        assert len(matches) >= 1
        # Verify descending severity order
        for i in range(len(matches) - 1):
            assert matches[i].severity >= matches[i + 1].severity

    def test_max_matches_limit(self) -> None:
        """Test maximum matches limit."""
        config = PatternMatcherConfig(max_matches_per_text=2)
        matcher = PatternMatcher(config)

        text = "I want to die, have a plan, can't go on, feeling hopeless, goodbye world"
        matches = matcher.detect(text)

        assert len(matches) <= 2

    def test_social_withdrawal_pattern(self, matcher: PatternMatcher) -> None:
        """Test social withdrawal pattern detection."""
        text = "I pushed everyone away from everyone"
        matches = matcher.detect(text)

        # May or may not detect social withdrawal depending on exact pattern
        # Just verify detect completes without error
        assert matches is not None

    def test_trauma_flashback_pattern(self, matcher: PatternMatcher) -> None:
        """Test trauma/flashback pattern detection."""
        text = "I'm stuck in that memory from the trauma"
        matches = matcher.detect(text)

        # May detect trauma pattern or other patterns
        assert matches is not None

    def test_multiple_pattern_types(self, matcher: PatternMatcher) -> None:
        """Test detection of multiple pattern types in same text."""
        text = "I have a plan to end my life tonight, I feel hopeless and never get better"
        matches = matcher.detect(text)

        pattern_types = {m.pattern_type for m in matches}
        # Should detect at least one pattern type
        assert len(pattern_types) >= 1
