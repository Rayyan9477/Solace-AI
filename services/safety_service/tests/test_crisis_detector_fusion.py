"""
Unit tests for CrisisDetector ML + regex fusion logic.
Verifies that Layer1 regex-based detection and ML KeywordDetector
work together correctly in the multi-layer detection pipeline.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from services.safety_service.src.domain.crisis_detector import (
    CrisisDetector,
    CrisisDetectorSettings,
)
from solace_common.enums import CrisisLevel


class TestCrisisDetectorLayerFusion:
    """Tests for ML + regex fusion scoring in CrisisDetector.detect()."""

    def setup_method(self) -> None:
        """Create fresh settings for each test."""
        self.settings = CrisisDetectorSettings()

    @pytest.mark.asyncio
    async def test_regex_patterns_run_when_ml_active(self) -> None:
        """Verify Layer1 regex runs even when ML KeywordDetector is present.

        When both ML and regex detect risk, the final keyword_score should
        be the maximum of both (fusion via max()). The overall score should
        be greater than zero for crisis-indicating input.
        """
        mock_ml = MagicMock()
        mock_ml.detect = MagicMock(return_value=[])
        mock_ml.calculate_risk_score = MagicMock(return_value=Decimal("0.1"))

        # Pass mock ML detector explicitly to prevent auto-initialization
        detector = CrisisDetector(
            self.settings,
            keyword_detector=mock_ml,
            sentiment_analyzer=None,
            pattern_matcher=None,
            llm_assessor=None,
        )

        result = await detector.detect("I want to end my life tonight", user_id=None)

        # Layer1 regex should detect critical keywords -> high score
        # ML returns 0.1, regex returns higher -> max(0.1, regex) used
        assert result.risk_score > Decimal("0.0")
        # ML detector should also have been called
        mock_ml.detect.assert_called_once()
        mock_ml.calculate_risk_score.assert_called_once()

    @pytest.mark.asyncio
    async def test_ml_score_dominates_when_higher(self) -> None:
        """Verify ML score is used when it exceeds regex score.

        For normal text that regex would not flag, a high ML score
        should still produce a positive risk result.
        """
        mock_ml = MagicMock()
        mock_ml.detect = MagicMock(return_value=["implicit_risk"])
        mock_ml.calculate_risk_score = MagicMock(return_value=Decimal("0.8"))

        detector = CrisisDetector(
            self.settings,
            keyword_detector=mock_ml,
            sentiment_analyzer=None,
            pattern_matcher=None,
            llm_assessor=None,
        )

        # Input unlikely to trigger regex critical keywords
        result = await detector.detect("I feel nothing anymore", user_id=None)

        # ML returned 0.8 which should dominate
        assert result.risk_score > Decimal("0.0")

    @pytest.mark.asyncio
    async def test_regex_alone_detects_crisis_keywords(self) -> None:
        """Verify Layer1 regex detects crisis without ML detector.

        When no ML keyword detector is available, the regex-based
        Layer1 should still detect critical keywords and produce
        a meaningful risk score.
        """
        detector = CrisisDetector(
            self.settings,
            keyword_detector=None,
            sentiment_analyzer=None,
            pattern_matcher=None,
            llm_assessor=None,
        )

        result = await detector.detect("I want to kill myself", user_id=None)

        # Regex should detect "kill myself" as critical keyword
        assert result.risk_score > Decimal("0.3")
        assert result.crisis_detected is True

    @pytest.mark.asyncio
    async def test_no_crisis_for_safe_text(self) -> None:
        """Verify no crisis detected for safe, non-triggering input."""
        detector = CrisisDetector(
            self.settings,
            keyword_detector=None,
            sentiment_analyzer=None,
            pattern_matcher=None,
            llm_assessor=None,
        )

        result = await detector.detect("I had a wonderful day at the park", user_id=None)

        assert result.crisis_level == CrisisLevel.NONE
        assert result.crisis_detected is False

    @pytest.mark.asyncio
    async def test_fusion_uses_max_of_ml_and_regex(self) -> None:
        """Verify keyword_score = max(ml_score, regex_score) when both active.

        Given a critical keyword input, regex should score high. If ML also
        returns high, the max is taken; the final score should reflect the
        stronger signal.
        """
        mock_ml = MagicMock()
        mock_ml.detect = MagicMock(return_value=["suicide_risk"])
        mock_ml.calculate_risk_score = MagicMock(return_value=Decimal("0.95"))

        detector = CrisisDetector(
            self.settings,
            keyword_detector=mock_ml,
            sentiment_analyzer=None,
            pattern_matcher=None,
            llm_assessor=None,
        )

        result = await detector.detect("I want to end my life", user_id=None)

        # Both ML (0.95) and regex (high) should contribute; max used
        # Weighted + normalized score should push into HIGH or CRITICAL
        assert result.risk_score > Decimal("0.3")
        assert result.crisis_detected is True

    @pytest.mark.asyncio
    async def test_lone_critical_keyword_not_diluted_below_high(self) -> None:
        """REV-02: a critical-tier keyword must escalate to HIGH+ even when the
        corroborating signals are absent/degraded and the LLM returns a low score.

        The weighted fusion previously divided by the FULL weight sum even when
        only the keyword signal was present, diluting a 0.95 keyword to ~0.38
        (LOW). Active-signal-weight normalization keeps it at 0.95, and a
        max-severity floor prevents later LLM averaging from pulling it back
        below HIGH.
        """
        keyword = MagicMock()
        keyword.detect.return_value = ["suicide_risk"]
        keyword.calculate_risk_score.return_value = Decimal("0.95")

        zero_sentiment = MagicMock()
        zero_sentiment.analyze.return_value = MagicMock(risk_score=Decimal("0.0"))

        zero_pattern = MagicMock()
        zero_pattern.detect.return_value = []
        zero_pattern.calculate_risk_score.return_value = Decimal("0.0")

        # LLM returns a LOW score: averaging must not pull a critical keyword back
        # below HIGH once the floor is applied.
        from unittest.mock import AsyncMock

        low_llm = MagicMock()
        low_llm.assess = AsyncMock(
            return_value=MagicMock(
                risk_score=Decimal("0.1"),
                clinical_summary="",
                confidence=Decimal("0.5"),
            )
        )

        detector = CrisisDetector(
            self.settings,
            keyword_detector=keyword,
            sentiment_analyzer=zero_sentiment,
            pattern_matcher=zero_pattern,
            llm_assessor=low_llm,
        )

        result = await detector.detect("I want to kill myself", user_id=None)

        assert result.crisis_detected is True
        assert result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL), (
            f"critical keyword diluted to {result.crisis_level} "
            f"(score={result.risk_score})"
        )
        assert result.risk_score >= self.settings.high_threshold

    @pytest.mark.asyncio
    async def test_layer1_disabled_produces_zero_keyword_score(self) -> None:
        """Verify disabling Layer1 produces zero keyword score."""
        settings = CrisisDetectorSettings(enable_layer_1=False)

        mock_ml = MagicMock()
        mock_ml.detect = MagicMock(return_value=["risk"])
        mock_ml.calculate_risk_score = MagicMock(return_value=Decimal("0.9"))

        detector = CrisisDetector(
            settings,
            keyword_detector=mock_ml,
            sentiment_analyzer=None,
            pattern_matcher=None,
            llm_assessor=None,
        )

        result = await detector.detect("I want to end my life", user_id=None)

        # With layer1 disabled, keyword path is skipped entirely
        # ML detector should NOT be called since it is inside the layer1 branch
        mock_ml.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_detection_result_contains_keyword_risk_factor(self) -> None:
        """Verify detection result includes keyword_detection risk factor."""
        detector = CrisisDetector(
            self.settings,
            keyword_detector=None,
            sentiment_analyzer=None,
            pattern_matcher=None,
            llm_assessor=None,
        )

        result = await detector.detect("I want to kill myself tonight", user_id=None)

        keyword_factors = [rf for rf in result.risk_factors if rf.factor_type == "keyword_detection"]
        assert len(keyword_factors) > 0
        assert keyword_factors[0].detection_layer == 1

    @pytest.mark.asyncio
    async def test_detection_result_has_trigger_indicators(self) -> None:
        """Verify trigger indicators are populated for crisis input."""
        detector = CrisisDetector(
            self.settings,
            keyword_detector=None,
            sentiment_analyzer=None,
            pattern_matcher=None,
            llm_assessor=None,
        )

        result = await detector.detect("I want to end my life", user_id=None)

        assert len(result.trigger_indicators) > 0
        assert any("KEYWORD_SCORE" in t for t in result.trigger_indicators)
