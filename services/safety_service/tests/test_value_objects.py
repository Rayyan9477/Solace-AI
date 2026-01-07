"""
Unit tests for Solace-AI Safety Service Value Objects.
Tests immutable value objects for safety domain.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from services.safety_service.src.domain.value_objects import (
    RiskScore, RiskSeverity, TriggerIndicator, TriggerCategory,
    ProtectiveFactor, ProtectiveFactorType, DetectionLayer,
    ContraindicationRule, SafetyThresholds, ResponseModification,
    TrajectoryAnalysis,
)


class TestRiskScore:
    """Tests for RiskScore value object."""

    def test_create_risk_score(self) -> None:
        """Test creating a risk score."""
        score = RiskScore(value=Decimal("0.5"), confidence=Decimal("0.8"))
        assert score.value == Decimal("0.5")
        assert score.confidence == Decimal("0.8")

    def test_risk_score_immutable(self) -> None:
        """Test that risk score is immutable."""
        score = RiskScore(value=Decimal("0.5"))
        with pytest.raises(Exception):
            score.value = Decimal("0.6")

    def test_severity_extreme(self) -> None:
        """Test extreme severity detection."""
        score = RiskScore(value=Decimal("0.95"))
        assert score.severity == RiskSeverity.EXTREME

    def test_severity_severe(self) -> None:
        """Test severe severity detection."""
        score = RiskScore(value=Decimal("0.8"))
        assert score.severity == RiskSeverity.SEVERE

    def test_severity_high(self) -> None:
        """Test high severity detection."""
        score = RiskScore(value=Decimal("0.65"))
        assert score.severity == RiskSeverity.HIGH

    def test_severity_moderate(self) -> None:
        """Test moderate severity detection."""
        score = RiskScore(value=Decimal("0.45"))
        assert score.severity == RiskSeverity.MODERATE

    def test_severity_low(self) -> None:
        """Test low severity detection."""
        score = RiskScore(value=Decimal("0.25"))
        assert score.severity == RiskSeverity.LOW

    def test_severity_minimal(self) -> None:
        """Test minimal severity detection."""
        score = RiskScore(value=Decimal("0.1"))
        assert score.severity == RiskSeverity.MINIMAL

    def test_is_critical(self) -> None:
        """Test critical flag."""
        critical = RiskScore(value=Decimal("0.95"))
        non_critical = RiskScore(value=Decimal("0.5"))
        assert critical.is_critical is True
        assert non_critical.is_critical is False

    def test_is_elevated(self) -> None:
        """Test elevated flag."""
        elevated = RiskScore(value=Decimal("0.6"))
        low = RiskScore(value=Decimal("0.3"))
        assert elevated.is_elevated is True
        assert low.is_elevated is False

    def test_zero_factory(self) -> None:
        """Test zero factory method."""
        score = RiskScore.zero()
        assert score.value == Decimal("0")
        assert score.confidence == Decimal("1.0")

    def test_maximum_factory(self) -> None:
        """Test maximum factory method."""
        score = RiskScore.maximum()
        assert score.value == Decimal("1.0")

    def test_comparison_with_risk_score(self) -> None:
        """Test comparison between risk scores."""
        score1 = RiskScore(value=Decimal("0.5"))
        score2 = RiskScore(value=Decimal("0.7"))
        assert score1 < score2
        assert score2 > score1
        assert score1 != score2

    def test_comparison_with_decimal(self) -> None:
        """Test comparison with Decimal."""
        score = RiskScore(value=Decimal("0.5"))
        assert score == Decimal("0.5")
        assert score < Decimal("0.7")
        assert score > Decimal("0.3")

    def test_comparison_with_float(self) -> None:
        """Test comparison with float."""
        score = RiskScore(value=Decimal("0.5"))
        assert score == 0.5
        assert score < 0.7

    def test_hash(self) -> None:
        """Test hashability."""
        score1 = RiskScore(value=Decimal("0.5"), confidence=Decimal("0.8"))
        score2 = RiskScore(value=Decimal("0.5"), confidence=Decimal("0.8"))
        assert hash(score1) == hash(score2)

    def test_weighted_average(self) -> None:
        """Test weighted average calculation."""
        score1 = RiskScore(value=Decimal("0.4"))
        score2 = RiskScore(value=Decimal("0.8"))
        avg = score1.weighted_average(score2, Decimal("0.5"))
        assert avg.value == Decimal("0.6")

    def test_coerce_from_float(self) -> None:
        """Test value coercion from float."""
        score = RiskScore(value=0.5, confidence=0.8)
        assert isinstance(score.value, Decimal)


class TestTriggerIndicator:
    """Tests for TriggerIndicator value object."""

    def test_create_trigger(self) -> None:
        """Test creating a trigger indicator."""
        trigger = TriggerIndicator(
            category=TriggerCategory.KEYWORD,
            indicator="KEYWORD:suicide",
            severity=RiskSeverity.SEVERE,
        )
        assert trigger.category == TriggerCategory.KEYWORD
        assert trigger.indicator == "KEYWORD:suicide"

    def test_trigger_immutable(self) -> None:
        """Test that trigger is immutable."""
        trigger = TriggerIndicator(
            category=TriggerCategory.KEYWORD,
            indicator="test",
        )
        with pytest.raises(Exception):
            trigger.indicator = "changed"

    def test_keyword_factory(self) -> None:
        """Test keyword factory method."""
        trigger = TriggerIndicator.keyword("dangerous", RiskSeverity.HIGH)
        assert trigger.category == TriggerCategory.KEYWORD
        assert "dangerous" in trigger.indicator
        assert trigger.evidence is not None

    def test_pattern_factory(self) -> None:
        """Test pattern factory method."""
        trigger = TriggerIndicator.pattern("suicidal_ideation")
        assert trigger.category == TriggerCategory.PATTERN
        assert "suicidal_ideation" in trigger.indicator

    def test_equality(self) -> None:
        """Test equality based on category and indicator."""
        t1 = TriggerIndicator(category=TriggerCategory.KEYWORD, indicator="test")
        t2 = TriggerIndicator(category=TriggerCategory.KEYWORD, indicator="test")
        assert t1 == t2

    def test_hash(self) -> None:
        """Test hashability."""
        t1 = TriggerIndicator(category=TriggerCategory.KEYWORD, indicator="test")
        t2 = TriggerIndicator(category=TriggerCategory.KEYWORD, indicator="test")
        assert hash(t1) == hash(t2)


class TestProtectiveFactor:
    """Tests for ProtectiveFactor value object."""

    def test_create_factor(self) -> None:
        """Test creating a protective factor."""
        factor = ProtectiveFactor(
            factor_type=ProtectiveFactorType.SOCIAL_SUPPORT,
            strength=Decimal("0.7"),
            description="Strong family connections",
        )
        assert factor.factor_type == ProtectiveFactorType.SOCIAL_SUPPORT
        assert factor.strength == Decimal("0.7")

    def test_is_strong(self) -> None:
        """Test strong factor detection."""
        strong = ProtectiveFactor(
            factor_type=ProtectiveFactorType.SOCIAL_SUPPORT,
            strength=Decimal("0.8"),
            description="Test",
        )
        weak = ProtectiveFactor(
            factor_type=ProtectiveFactorType.SOCIAL_SUPPORT,
            strength=Decimal("0.4"),
            description="Test",
        )
        assert strong.is_strong is True
        assert weak.is_strong is False

    def test_social_support_factory(self) -> None:
        """Test social support factory method."""
        factor = ProtectiveFactor.social_support("Family support", Decimal("0.6"))
        assert factor.factor_type == ProtectiveFactorType.SOCIAL_SUPPORT

    def test_treatment_engagement_factory(self) -> None:
        """Test treatment engagement factory method."""
        factor = ProtectiveFactor.treatment_engagement("Regular therapy")
        assert factor.factor_type == ProtectiveFactorType.TREATMENT_ENGAGEMENT
        assert factor.strength == Decimal("0.7")


class TestContraindicationRule:
    """Tests for ContraindicationRule value object."""

    def test_create_rule(self) -> None:
        """Test creating a contraindication rule."""
        rule = ContraindicationRule(
            technique="exposure_therapy",
            crisis_levels=["HIGH", "CRITICAL"],
            reason="Can worsen acute crisis",
            alternative_techniques=["grounding", "safety_planning"],
        )
        assert rule.technique == "exposure_therapy"
        assert len(rule.crisis_levels) == 2

    def test_is_contraindicated(self) -> None:
        """Test contraindication checking."""
        rule = ContraindicationRule(
            technique="exposure_therapy",
            crisis_levels=["HIGH", "CRITICAL"],
            reason="Test reason",
        )
        assert rule.is_contraindicated("HIGH") is True
        assert rule.is_contraindicated("high") is True
        assert rule.is_contraindicated("LOW") is False


class TestSafetyThresholds:
    """Tests for SafetyThresholds value object."""

    def test_create_thresholds(self) -> None:
        """Test creating safety thresholds."""
        thresholds = SafetyThresholds()
        assert thresholds.low_threshold == Decimal("0.3")
        assert thresholds.critical_threshold == Decimal("0.9")

    def test_get_level_critical(self) -> None:
        """Test getting critical level."""
        thresholds = SafetyThresholds()
        assert thresholds.get_level_for_score(Decimal("0.95")) == "CRITICAL"

    def test_get_level_high(self) -> None:
        """Test getting high level."""
        thresholds = SafetyThresholds()
        assert thresholds.get_level_for_score(Decimal("0.75")) == "HIGH"

    def test_get_level_elevated(self) -> None:
        """Test getting elevated level."""
        thresholds = SafetyThresholds()
        assert thresholds.get_level_for_score(Decimal("0.55")) == "ELEVATED"

    def test_get_level_low(self) -> None:
        """Test getting low level."""
        thresholds = SafetyThresholds()
        assert thresholds.get_level_for_score(Decimal("0.35")) == "LOW"

    def test_get_level_none(self) -> None:
        """Test getting none level."""
        thresholds = SafetyThresholds()
        assert thresholds.get_level_for_score(Decimal("0.1")) == "NONE"

    def test_get_level_from_risk_score(self) -> None:
        """Test getting level from RiskScore object."""
        thresholds = SafetyThresholds()
        score = RiskScore(value=Decimal("0.8"))
        assert thresholds.get_level_for_score(score) == "HIGH"


class TestResponseModification:
    """Tests for ResponseModification value object."""

    def test_create_modification(self) -> None:
        """Test creating a response modification."""
        mod = ResponseModification(
            modification_type="filter",
            original_content="give up",
            modified_content="[supportive message]",
            reason="Unsafe phrase detected",
        )
        assert mod.modification_type == "filter"
        assert mod.layer == DetectionLayer.OUTPUT_FILTER


class TestTrajectoryAnalysis:
    """Tests for TrajectoryAnalysis value object."""

    def test_create_analysis(self) -> None:
        """Test creating trajectory analysis."""
        analysis = TrajectoryAnalysis(
            trend="deteriorating",
            deteriorating=True,
            risk_delta=Decimal("0.2"),
            negative_ratio=Decimal("0.7"),
            messages_analyzed=5,
        )
        assert analysis.trend == "deteriorating"
        assert analysis.deteriorating is True

    def test_requires_attention_deteriorating(self) -> None:
        """Test attention required when deteriorating."""
        analysis = TrajectoryAnalysis(deteriorating=True)
        assert analysis.requires_attention is True

    def test_requires_attention_high_negative_ratio(self) -> None:
        """Test attention required with high negative ratio."""
        analysis = TrajectoryAnalysis(
            deteriorating=False,
            negative_ratio=Decimal("0.7"),
        )
        assert analysis.requires_attention is True

    def test_no_attention_stable(self) -> None:
        """Test no attention for stable trajectory."""
        analysis = TrajectoryAnalysis(
            trend="stable",
            deteriorating=False,
            negative_ratio=Decimal("0.3"),
        )
        assert analysis.requires_attention is False
