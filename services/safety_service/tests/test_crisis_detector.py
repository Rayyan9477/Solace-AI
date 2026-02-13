"""
Unit tests for Solace-AI Crisis Detector.
Tests multi-layer crisis detection with various risk scenarios.
"""
from __future__ import annotations
import pytest
from decimal import Decimal
from services.safety_service.src.domain.crisis_detector import (
    CrisisDetector, CrisisDetectorSettings, CrisisLevel, RiskFactor,
    DetectionResult, Layer1InputGate, Layer2ProcessingGuard,
    Layer3OutputFilter, Layer4ContinuousMonitor, KeywordSet,
)


class TestCrisisLevel:
    """Tests for CrisisLevel enum."""

    def test_from_score_critical(self) -> None:
        """Test CRITICAL level from high score."""
        assert CrisisLevel.from_score(Decimal("0.95")) == CrisisLevel.CRITICAL
        assert CrisisLevel.from_score(Decimal("0.9")) == CrisisLevel.CRITICAL

    def test_from_score_high(self) -> None:
        """Test HIGH level from score."""
        assert CrisisLevel.from_score(Decimal("0.85")) == CrisisLevel.HIGH
        assert CrisisLevel.from_score(Decimal("0.7")) == CrisisLevel.HIGH

    def test_from_score_elevated(self) -> None:
        """Test ELEVATED level from score."""
        assert CrisisLevel.from_score(Decimal("0.6")) == CrisisLevel.ELEVATED
        assert CrisisLevel.from_score(Decimal("0.5")) == CrisisLevel.ELEVATED

    def test_from_score_low(self) -> None:
        """Test LOW level from score."""
        assert CrisisLevel.from_score(Decimal("0.4")) == CrisisLevel.LOW
        assert CrisisLevel.from_score(Decimal("0.3")) == CrisisLevel.LOW

    def test_from_score_none(self) -> None:
        """Test NONE level from low score."""
        assert CrisisLevel.from_score(Decimal("0.2")) == CrisisLevel.NONE
        assert CrisisLevel.from_score(Decimal("0.0")) == CrisisLevel.NONE


class TestRiskFactor:
    """Tests for RiskFactor model."""

    def test_create_valid_risk_factor(self) -> None:
        """Test creating valid risk factor."""
        rf = RiskFactor(
            factor_type="keyword",
            severity=Decimal("0.8"),
            evidence="Test evidence",
            confidence=Decimal("0.9"),
            detection_layer=1,
        )
        assert rf.factor_type == "keyword"
        assert rf.severity == Decimal("0.8")
        assert rf.detection_layer == 1

    def test_severity_bounds(self) -> None:
        """Test severity must be between 0 and 1."""
        with pytest.raises(ValueError):
            RiskFactor(factor_type="test", severity=Decimal("1.5"),
                       evidence="", confidence=Decimal("0.5"), detection_layer=1)


class TestLayer1InputGate:
    """Tests for Layer 1 input gate detection."""

    @pytest.fixture
    def layer1(self) -> Layer1InputGate:
        """Create Layer 1 detector."""
        return Layer1InputGate(CrisisDetectorSettings())

    def test_detect_critical_keyword(self, layer1: Layer1InputGate) -> None:
        """Test detection of critical keywords."""
        result = layer1.detect("I want to end my life")
        assert result.crisis_detected is True
        assert result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
        assert any("CRITICAL_KEYWORD" in t or "PATTERN" in t for t in result.trigger_indicators)

    def test_detect_high_keyword(self, layer1: Layer1InputGate) -> None:
        """Test detection of high-risk keywords."""
        result = layer1.detect("I've been thinking about self-harm")
        assert result.crisis_detected is True
        assert result.crisis_level in (CrisisLevel.LOW, CrisisLevel.ELEVATED, CrisisLevel.HIGH, CrisisLevel.CRITICAL)

    def test_detect_elevated_keyword(self, layer1: Layer1InputGate) -> None:
        """Test detection of elevated keywords."""
        result = layer1.detect("I'm feeling very depressed and overwhelmed")
        assert result.crisis_detected is True
        assert result.crisis_level in (CrisisLevel.NONE, CrisisLevel.LOW, CrisisLevel.ELEVATED, CrisisLevel.HIGH)

    def test_detect_pattern_suicidal_ideation(self, layer1: Layer1InputGate) -> None:
        """Test pattern detection for suicidal ideation."""
        result = layer1.detect("I'm planning to end it all soon")
        assert result.crisis_detected is True
        assert any("PATTERN" in t for t in result.trigger_indicators)

    def test_detect_pattern_farewell(self, layer1: Layer1InputGate) -> None:
        """Test pattern detection for farewell messages."""
        result = layer1.detect("Tell my family I love them, goodbye")
        assert result.crisis_detected is True
        assert result.crisis_level in (CrisisLevel.LOW, CrisisLevel.ELEVATED, CrisisLevel.HIGH, CrisisLevel.CRITICAL)

    def test_no_crisis_normal_message(self, layer1: Layer1InputGate) -> None:
        """Test no crisis detected for normal message."""
        result = layer1.detect("I had a good day at work today")
        assert result.crisis_level == CrisisLevel.NONE
        assert result.crisis_detected is False

    def test_risk_history_increases_score(self, layer1: Layer1InputGate) -> None:
        """Test risk history increases detection score."""
        history = {"previous_crisis_events": 2, "high_risk_flag": True}
        result = layer1.detect("I'm feeling a bit down", history)
        assert result.risk_score > Decimal("0.0")

    def test_recommended_action_for_levels(self, layer1: Layer1InputGate) -> None:
        """Test recommended actions match crisis levels."""
        critical = layer1.detect("I want to kill myself tonight")
        assert critical.recommended_action == "escalate_immediately"
        normal = layer1.detect("I'm doing well")
        assert normal.recommended_action == "continue"


class TestLayer2ProcessingGuard:
    """Tests for Layer 2 processing guard."""

    @pytest.fixture
    def layer2(self) -> Layer2ProcessingGuard:
        """Create Layer 2 detector."""
        return Layer2ProcessingGuard(CrisisDetectorSettings())

    @pytest.fixture
    def base_result(self) -> DetectionResult:
        """Create base detection result."""
        return DetectionResult(
            crisis_detected=True,
            crisis_level=CrisisLevel.HIGH,
            risk_score=Decimal("0.7"),
            risk_factors=[],
            trigger_indicators=[],
            detection_layers_triggered=[1],
        )

    def test_contraindication_detection(self, layer2: Layer2ProcessingGuard, base_result: DetectionResult) -> None:
        """Test contraindication detection for high risk."""
        context = {"active_technique": "exposure_therapy"}
        result = layer2.validate_context("test content", context, base_result)
        assert 2 in result.detection_layers_triggered
        assert any(rf.factor_type == "contraindication" for rf in result.risk_factors)

    def test_no_contraindication_safe_technique(self, layer2: Layer2ProcessingGuard, base_result: DetectionResult) -> None:
        """Test no contraindication for safe technique."""
        context = {"active_technique": "mindfulness"}
        result = layer2.validate_context("test content", context, base_result)
        contraindications = [rf for rf in result.risk_factors if rf.factor_type == "contraindication"]
        assert len(contraindications) == 0


class TestLayer3OutputFilter:
    """Tests for Layer 3 output filter."""

    @pytest.fixture
    def layer3(self) -> Layer3OutputFilter:
        """Create Layer 3 filter."""
        return Layer3OutputFilter(CrisisDetectorSettings())

    def test_filter_unsafe_phrase(self, layer3: Layer3OutputFilter) -> None:
        """Test filtering of unsafe phrases."""
        response = "Sometimes people feel like they should give up on life"
        filtered, modifications, is_safe = layer3.filter_output(response, CrisisLevel.ELEVATED)
        assert "give up" not in filtered.lower() or len(modifications) > 0

    def test_safe_response_passes(self, layer3: Layer3OutputFilter) -> None:
        """Test safe response passes through."""
        response = "It's completely normal to feel stressed sometimes."
        filtered, modifications, is_safe = layer3.filter_output(response, CrisisLevel.LOW)
        assert filtered == response
        assert is_safe is True

    def test_critical_missing_elements(self, layer3: Layer3OutputFilter) -> None:
        """Test detection of missing required elements for CRITICAL."""
        response = "I understand you're feeling down."
        _, modifications, _ = layer3.filter_output(response, CrisisLevel.CRITICAL)
        assert any("missing required elements" in m.lower() for m in modifications)


class TestLayer4ContinuousMonitor:
    """Tests for Layer 4 continuous monitoring."""

    @pytest.fixture
    def layer4(self) -> Layer4ContinuousMonitor:
        """Create Layer 4 monitor."""
        return Layer4ContinuousMonitor(CrisisDetectorSettings())

    def test_trajectory_insufficient_data(self, layer4: Layer4ContinuousMonitor) -> None:
        """Test trajectory analysis with insufficient data."""
        result = layer4.analyze_trajectory(["single message"], CrisisLevel.LOW)
        assert result["trend"] == "insufficient_data"

    def test_trajectory_deteriorating(self, layer4: Layer4ContinuousMonitor) -> None:
        """Test detection of deteriorating trajectory."""
        messages = [
            "I'm feeling worse today",
            "Things are getting bad",
            "I can't handle this anymore",
            "I hate everything",
            "I'll never get better",
        ]
        result = layer4.analyze_trajectory(messages, CrisisLevel.ELEVATED)
        assert result["deteriorating"] is True
        assert result["trend"] == "deteriorating"

    def test_trajectory_stable(self, layer4: Layer4ContinuousMonitor) -> None:
        """Test detection of stable trajectory."""
        messages = [
            "I'm doing okay today",
            "Had a nice walk",
            "Feeling alright",
            "Pretty good afternoon",
            "Things are manageable",
        ]
        result = layer4.analyze_trajectory(messages, CrisisLevel.LOW)
        assert result["deteriorating"] is False


class TestCrisisDetector:
    """Tests for main CrisisDetector class."""

    @pytest.fixture
    def detector(self) -> CrisisDetector:
        """Create crisis detector."""
        return CrisisDetector(CrisisDetectorSettings())

    @pytest.mark.asyncio
    async def test_detect_critical_crisis(self, detector: CrisisDetector) -> None:
        """Test detection of critical crisis."""
        result = await detector.detect("I'm going to kill myself tonight")
        assert result.crisis_detected is True
        assert result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
        assert result.risk_score >= Decimal("0.7")

    @pytest.mark.asyncio
    async def test_detect_no_crisis(self, detector: CrisisDetector) -> None:
        """Test no crisis in safe content."""
        result = await detector.detect("I'm feeling great today!")
        assert result.crisis_detected is False
        assert result.crisis_level == CrisisLevel.NONE

    @pytest.mark.asyncio
    async def test_detect_with_context(self, detector: CrisisDetector) -> None:
        """Test detection with context."""
        context = {"active_technique": "exposure_therapy"}
        result = await detector.detect("I'm feeling anxious", context=context)
        assert result is not None

    @pytest.mark.asyncio
    async def test_detect_with_history(self, detector: CrisisDetector) -> None:
        """Test detection with conversation history."""
        history = ["I feel worse", "Everything is terrible", "I can't cope"]
        result = await detector.detect("I'm not doing well", conversation_history=history)
        assert result is not None

    def test_filter_output(self, detector: CrisisDetector) -> None:
        """Test output filtering."""
        response = "You should just give up on that."
        filtered, mods, is_safe = detector.filter_output(response, CrisisLevel.ELEVATED)
        assert len(mods) > 0 or "give up" not in filtered.lower()

    def test_analyze_trajectory(self, detector: CrisisDetector) -> None:
        """Test trajectory analysis."""
        history = ["message 1", "message 2", "message 3"]
        result = detector.analyze_trajectory(history, CrisisLevel.LOW)
        assert "trend" in result


class TestCrisisDetectorSettings:
    """Tests for CrisisDetectorSettings."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = CrisisDetectorSettings()
        assert settings.enable_layer_1 is True
        assert settings.enable_layer_2 is True
        assert settings.enable_layer_3 is True
        assert settings.enable_layer_4 is True
        assert settings.critical_threshold == Decimal("0.9")

    def test_custom_thresholds(self) -> None:
        """Test custom threshold settings."""
        settings = CrisisDetectorSettings(
            low_threshold=Decimal("0.25"),
            critical_threshold=Decimal("0.85"),
        )
        assert settings.low_threshold == Decimal("0.25")
        assert settings.critical_threshold == Decimal("0.85")
