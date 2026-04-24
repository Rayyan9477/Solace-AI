"""
Tests for llm_assessor.py - LLM-based deep risk assessment.
"""
from decimal import Decimal
from uuid import uuid4

import pytest

from services.safety_service.src.ml.llm_assessor import (
    LLMAssessor,
    LLMAssessorConfig,
    ProtectiveFactor,
    RiskAssessment,
    RiskDimension,
    RiskFactor,
    RiskLevel,
)


class TestLLMAssessor:
    """Test suite for LLMAssessor."""

    @pytest.fixture
    def assessor(self) -> LLMAssessor:
        """Create LLM assessor with default config."""
        return LLMAssessor()

    def test_initialization(self, assessor: LLMAssessor) -> None:
        """Test assessor initializes correctly."""
        assert assessor is not None
        assert assessor._config is not None

    @pytest.mark.asyncio
    async def test_assess_empty_text(self, assessor: LLMAssessor) -> None:
        """Test assessment of empty text."""
        result = await assessor.assess("")

        assert result.risk_level == RiskLevel.NONE
        assert result.risk_score == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_assess_returns_result(self, assessor: LLMAssessor) -> None:
        """Test assess returns valid RiskAssessment."""
        text = "I'm feeling sad today"
        result = await assessor.assess(text)

        assert isinstance(result, RiskAssessment)
        assert result.assessment_id is not None
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_assess_with_user_id(self, assessor: LLMAssessor) -> None:
        """Test assessment with user ID."""
        user_id = uuid4()
        text = "I'm feeling okay"

        result = await assessor.assess(text, user_id=user_id)
        assert result is not None

    @pytest.mark.asyncio
    async def test_assess_with_context(self, assessor: LLMAssessor) -> None:
        """Test assessment with additional context."""
        text = "I'm struggling"
        context = {"prior_risk_score": 0.7, "session_number": 5}

        result = await assessor.assess(text, context=context)
        assert result is not None

    @pytest.mark.asyncio
    async def test_fallback_assessment_crisis(self, assessor: LLMAssessor) -> None:
        """Test fallback creates appropriate assessment for crisis keywords."""
        text = "I want to kill myself"

        # This will trigger fallback since mock LLM is used
        result = await assessor.assess(text)

        assert result.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.ELEVATED]
        assert result.risk_score > Decimal("0.0")

    @pytest.mark.asyncio
    async def test_caching_enabled(self) -> None:
        """Test assessment caching when enabled."""
        config = LLMAssessorConfig(enable_caching=True, cache_ttl_minutes=60)
        assessor = LLMAssessor(config)

        text = "I'm feeling sad"

        result1 = await assessor.assess(text)
        result2 = await assessor.assess(text)

        # Second call should return cached result
        assert result1.assessment_id == result2.assessment_id

    @pytest.mark.asyncio
    async def test_caching_disabled(self) -> None:
        """Test no caching when disabled."""
        config = LLMAssessorConfig(enable_caching=False)
        assessor = LLMAssessor(config)

        text = "I'm feeling sad"

        result1 = await assessor.assess(text)
        result2 = await assessor.assess(text)

        # Different assessment IDs
        assert result1.assessment_id != result2.assessment_id

    def test_parse_llm_response(self, assessor: LLMAssessor) -> None:
        """Test parsing of LLM JSON response."""
        response = {
            "risk_level": "HIGH",
            "risk_score": 0.8,
            "confidence": 0.9,
            "risk_factors": [
                {
                    "dimension": "SUICIDAL_IDEATION",
                    "severity": 0.9,
                    "evidence": "wants to die",
                    "rationale": "explicit suicidal thoughts"
                }
            ],
            "protective_factors": [
                {
                    "factor": "social support",
                    "strength": 0.6,
                    "evidence": "has supportive family"
                }
            ],
            "clinical_summary": "High risk assessment",
            "recommended_actions": ["Safety plan", "Crisis resources"],
            "immediate_risk": True,
            "contextual_notes": "First session"
        }

        assessment = assessor._parse_llm_response(response)

        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.risk_score == Decimal("0.8")
        assert len(assessment.risk_factors) == 1
        assert len(assessment.protective_factors) == 1
        assert assessment.immediate_risk is True

    def test_create_minimal_assessment(self, assessor: LLMAssessor) -> None:
        """Test creation of minimal assessment."""
        assessment = assessor._create_minimal_assessment("test reason")

        assert assessment.risk_level == RiskLevel.NONE
        assert assessment.risk_score == Decimal("0.0")
        assert "test reason" in assessment.clinical_summary

    def test_create_fallback_assessment_critical(self, assessor: LLMAssessor) -> None:
        """Test fallback assessment for critical keywords."""
        text = "I want to commit suicide"
        assessment = assessor._create_fallback_assessment(text, "error")

        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.immediate_risk is True

    def test_create_fallback_assessment_moderate(self, assessor: LLMAssessor) -> None:
        """Test fallback assessment for non-critical text."""
        text = "I'm feeling stressed"
        assessment = assessor._create_fallback_assessment(text, "error")

        assert assessment.risk_level == RiskLevel.ELEVATED
        assert assessment.immediate_risk is False

    def test_generate_cache_key(self, assessor: LLMAssessor) -> None:
        """Test cache key generation."""
        text = "test text"
        context = {"key": "value"}

        key1 = assessor._generate_cache_key(text, context)
        key2 = assessor._generate_cache_key(text, context)
        key3 = assessor._generate_cache_key("different", context)

        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key

    def test_cache_key_includes_user_id(self, assessor: LLMAssessor) -> None:
        """H-05 regression: cache key must differ between users for identical text.

        Before the fix, two users with the same message would collide on the
        same cache key, and user A could receive user B's risk assessment.
        This is a privacy + safety bug: a correctly non-crisis assessment
        for one user could be served for another user whose identical phrase
        actually indicates crisis given their context.
        """
        text = "same utterance across users"
        context = {"intent": "discuss"}

        user_a = uuid4()
        user_b = uuid4()

        key_a = assessor._generate_cache_key(text, context, user_a)
        key_b = assessor._generate_cache_key(text, context, user_b)
        key_none = assessor._generate_cache_key(text, context)
        key_a_again = assessor._generate_cache_key(text, context, user_a)

        assert key_a != key_b, (
            "H-05 regression: identical text for different users must not "
            "produce the same cache key. This risks cross-user PHI leakage."
        )
        assert key_a != key_none, "user-scoped key must differ from anonymous key"
        assert key_a == key_a_again, "same user + same input must be reproducible"

    @pytest.mark.asyncio
    async def test_crisis_assessments_are_not_cached(self, assessor: LLMAssessor) -> None:
        """H-05 regression: HIGH/CRITICAL results must never be cached.

        Caching a transient crisis state could mask a real-time intervention
        signal on a subsequent call. The assess() method skips caching when
        the assessment resolves to HIGH or CRITICAL.
        """
        # Stub the LLM call to return a HIGH-risk assessment every time.
        # We return the JSON shape the real parser expects rather than a
        # RiskAssessment object — that way the full parse path runs.
        async def _fake_call_llm(system_prompt: str, user_prompt: str) -> str:
            # Round-trip the assessment through parse_llm_response by returning
            # a JSON object the real parser understands.
            return (
                '{"risk_level":"HIGH","risk_score":0.75,"confidence":0.9,'
                '"clinical_summary":"simulated high risk",'
                '"risk_factors":[],"protective_factors":[],"immediate_risk":false,'
                '"recommended_actions":["monitor"],"warning_signs":[],'
                '"contextual_notes":""}'
            )

        assessor._call_llm = _fake_call_llm  # type: ignore[assignment,method-assign]

        user_id = uuid4()
        text = "I have been thinking about ending it all"

        # Two calls with identical inputs — if caching were active on HIGH,
        # we'd hit the cache and _call_llm would not run the second time.
        call_count = {"n": 0}
        original = _fake_call_llm

        async def _counting_call(sp: str, up: str) -> str:
            call_count["n"] += 1
            return await original(sp, up)

        assessor._call_llm = _counting_call  # type: ignore[assignment,method-assign]

        await assessor.assess(text, user_id=user_id)
        await assessor.assess(text, user_id=user_id)

        assert call_count["n"] == 2, (
            "H-05 regression: HIGH-risk assessment must not be cached. "
            "The LLM should be called again on the second identical request."
        )

    def test_get_cached_assessment_miss(self, assessor: LLMAssessor) -> None:
        """Test cache miss returns None."""
        cached = assessor._get_cached_assessment("nonexistent_key")
        assert cached is None

    def test_cache_assessment(self, assessor: LLMAssessor) -> None:
        """Test caching an assessment."""
        assessment = RiskAssessment(
            risk_level=RiskLevel.LOW,
            risk_score=Decimal("0.2"),
            confidence=Decimal("0.8"),
            clinical_summary="Test"
        )

        cache_key = "test_key"
        assessor._cache_assessment(cache_key, assessment)

        cached = assessor._get_cached_assessment(cache_key)
        assert cached is not None
        assert cached.risk_level == RiskLevel.LOW

    def test_get_highest_risk_dimension(self, assessor: LLMAssessor) -> None:
        """Test getting highest risk dimension."""
        assessment = RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_score=Decimal("0.8"),
            confidence=Decimal("0.9"),
            clinical_summary="Test",
            risk_factors=[
                RiskFactor(
                    dimension=RiskDimension.HOPELESSNESS,
                    severity=Decimal("0.6"),
                    evidence="text",
                    rationale="reason"
                ),
                RiskFactor(
                    dimension=RiskDimension.SUICIDAL_IDEATION,
                    severity=Decimal("0.9"),
                    evidence="text",
                    rationale="reason"
                )
            ]
        )

        highest = assessor.get_highest_risk_dimension(assessment)
        assert highest == RiskDimension.SUICIDAL_IDEATION

    def test_get_highest_risk_dimension_none(self, assessor: LLMAssessor) -> None:
        """Test getting highest dimension with no risk factors."""
        assessment = RiskAssessment(
            risk_level=RiskLevel.LOW,
            risk_score=Decimal("0.1"),
            confidence=Decimal("0.8"),
            clinical_summary="Test"
        )

        highest = assessor.get_highest_risk_dimension(assessment)
        assert highest is None

    def test_build_system_prompt(self, assessor: LLMAssessor) -> None:
        """Test system prompt is built."""
        prompt = assessor._build_system_prompt()

        assert len(prompt) > 0
        assert "risk assessment" in prompt.lower()
        assert "JSON" in prompt

    def test_build_user_prompt(self, assessor: LLMAssessor) -> None:
        """Test user prompt is built with text."""
        text = "test input"
        prompt = assessor._build_user_prompt(text)

        assert text in prompt
        assert "analyze" in prompt.lower()

    def test_build_user_prompt_with_context(self, assessor: LLMAssessor) -> None:
        """Test user prompt includes context."""
        text = "test input"
        context = {"session": 5}

        prompt = assessor._build_user_prompt(text, context)

        assert text in prompt
        assert "context" in prompt.lower()

    def test_risk_assessment_attributes(self) -> None:
        """Test RiskAssessment model attributes."""
        assessment = RiskAssessment(
            risk_level=RiskLevel.ELEVATED,
            risk_score=Decimal("0.5"),
            confidence=Decimal("0.7"),
            clinical_summary="Test summary",
            recommended_actions=["action1", "action2"]
        )

        assert assessment.assessment_id is not None
        assert assessment.timestamp is not None
        assert assessment.risk_level == RiskLevel.ELEVATED
        assert len(assessment.recommended_actions) == 2

    def test_risk_factor_model(self) -> None:
        """Test RiskFactor model."""
        factor = RiskFactor(
            dimension=RiskDimension.EMOTIONAL_DISTRESS,
            severity=Decimal("0.7"),
            evidence="feeling very anxious",
            rationale="High anxiety levels"
        )

        assert factor.dimension == RiskDimension.EMOTIONAL_DISTRESS
        assert factor.severity == Decimal("0.7")

    def test_protective_factor_model(self) -> None:
        """Test ProtectiveFactor model."""
        factor = ProtectiveFactor(
            factor="strong social support",
            strength=Decimal("0.8"),
            evidence="has supportive family and friends"
        )

        assert factor.strength == Decimal("0.8")
        assert "social support" in factor.factor
