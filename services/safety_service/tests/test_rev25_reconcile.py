"""REV-25 clinical-accuracy reconciliation (Milestone-2 Phase C.5).

Prove-It tests encoding the corrected behavior after reconciling the safety
thresholds/latency documentation with the code:

  (1) The CRITICAL crisis floor is 0.85 (matches docs/CLINICAL-VALIDATION.md),
      NOT 0.9. A lone explicit CRITICAL keyword now escalates to crisis_level
      CRITICAL at the 0.85 floor (previously it capped at HIGH / 0.7).
  (2) DetectionResult.detection_time_ms is a real measurement (> 0), not a
      hard-coded 0, so end-to-end latency is observable.
  (3) The Layer-1 keyword-gate meets a <10ms latency SLO (measured median is
      well under 1ms on this codebase).

These changes lower a patient-safety threshold and must be surfaced for
clinician sign-off (Phase D2).
"""
from __future__ import annotations

import statistics
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.safety_service.src.domain.crisis_detector import (
    CrisisDetector,
    CrisisDetectorSettings,
    Layer1InputGate,
)
from services.safety_service.src.domain.value_objects import SafetyThresholds
from solace_common.enums import CrisisLevel


def _isolated_detector(settings: CrisisDetectorSettings | None = None) -> CrisisDetector:
    """Build a CrisisDetector whose ML/LLM signals contribute nothing, so the
    test exercises only the deterministic Layer-1 regex path + the critical
    keyword floor. Mirrors the isolation approach in test_crisis_detector_fusion.
    """
    settings = settings or CrisisDetectorSettings()
    keyword = MagicMock()
    keyword.detect.return_value = []
    keyword.calculate_risk_score.return_value = Decimal("0")
    sentiment = MagicMock()
    sentiment.analyze.return_value = MagicMock(risk_score=Decimal("0"))
    pattern = MagicMock()
    pattern.detect.return_value = []
    pattern.calculate_risk_score.return_value = Decimal("0")
    llm = MagicMock()
    llm.assess = AsyncMock(
        return_value=MagicMock(
            risk_score=Decimal("0"), clinical_summary="", confidence=Decimal("0.5")
        )
    )
    return CrisisDetector(
        settings,
        keyword_detector=keyword,
        sentiment_analyzer=sentiment,
        pattern_matcher=pattern,
        llm_assessor=llm,
    )


class TestReconciledCriticalThreshold:
    """Drift (1): the CRITICAL crisis floor defaults to 0.85, not 0.9."""

    def test_runtime_settings_critical_threshold_is_085(self) -> None:
        assert CrisisDetectorSettings().critical_threshold == Decimal("0.85")

    def test_value_object_thresholds_critical_is_085(self) -> None:
        assert SafetyThresholds().critical_threshold == Decimal("0.85")

    def test_085_boundary_now_classifies_critical(self) -> None:
        """A score sitting exactly on the reconciled 0.85 floor is CRITICAL.
        Under the previous 0.9 default this same score classified as HIGH.
        """
        thresholds = SafetyThresholds()
        assert thresholds.get_level_for_score(Decimal("0.85")) == "CRITICAL"
        # Just below the floor stays HIGH (guards against an over-broad change).
        assert thresholds.get_level_for_score(Decimal("0.849")) == "HIGH"


class TestLoneCriticalKeywordFloorsToCritical:
    """Prove-It core: a lone explicit CRITICAL keyword yields crisis_level
    CRITICAL at the 0.85 floor.

    These utterances are pure critical-tier keywords with NO corroborating
    regex pattern; the Layer-1 gate dilutes each to ~0.475 on its own, so only
    the reconciled critical-keyword floor lifts them to CRITICAL (0.85).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "utterance",
        [
            "better off dead",
            "take my life",
            "I can't go on",
            "I feel like I want to end it all",
        ],
    )
    async def test_lone_critical_keyword_is_critical_at_085(self, utterance: str) -> None:
        detector = _isolated_detector()
        result = await detector.detect(utterance)

        assert result.crisis_detected is True
        assert result.crisis_level == CrisisLevel.CRITICAL, (
            f"{utterance!r} scored {result.crisis_level.value} "
            f"(risk {float(result.risk_score):.3f}); a lone critical keyword must "
            f"reach CRITICAL at the 0.85 floor."
        )
        # Floored exactly to the reconciled CRITICAL threshold (0.85).
        assert result.risk_score >= Decimal("0.85")
        assert result.risk_score == detector._settings.critical_threshold
        assert result.recommended_action == "escalate_immediately"


class TestDetectionTimeObservable:
    """Drift (2): detection_time_ms is a real measurement, not a hard-coded 0."""

    @pytest.mark.asyncio
    async def test_detection_time_ms_is_positive(self) -> None:
        detector = _isolated_detector()
        result = await detector.detect("better off dead")
        assert result.detection_time_ms > 0, (
            "detection_time_ms must be a real measurement so latency is "
            "observable; it was hard-coded to 0 on the main detect() path."
        )

    @pytest.mark.asyncio
    async def test_detection_time_ms_populated_for_safe_input(self) -> None:
        detector = _isolated_detector()
        result = await detector.detect("I had a lovely walk in the park today")
        assert result.detection_time_ms > 0


class TestLayer1KeywordGateLatencySLO:
    """Drift (3): the Layer-1 keyword gate meets a <10ms latency SLO.

    Measured median on this codebase is well under 1ms (~0.07ms); asserting the
    10ms SLO leaves a ~100x margin, so this is not flaky. See the REV-25 delta
    for the true measured numbers.
    """

    KEYWORD_GATE_SLO_MS = 10.0

    def test_keyword_gate_latency_under_slo(self) -> None:
        gate = Layer1InputGate(CrisisDetectorSettings())
        samples = [
            "I want to die",
            "better off dead",
            "I had a good day at work",
            "I feel hopeless and worthless",
            "no reason to live tonight",
        ]
        # Warm up (regex compilation / branch prediction) before measuring.
        for _ in range(50):
            for msg in samples:
                gate.detect(msg)

        latencies_ms: list[float] = []
        for i in range(1000):
            msg = samples[i % len(samples)]
            start = time.perf_counter()
            gate.detect(msg)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        median_ms = statistics.median(latencies_ms)
        latencies_ms.sort()
        p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]

        assert median_ms < self.KEYWORD_GATE_SLO_MS, (
            f"Layer-1 keyword-gate median latency {median_ms:.4f}ms exceeds the "
            f"{self.KEYWORD_GATE_SLO_MS}ms SLO."
        )
        # p95 also comfortably under the SLO on this codebase.
        assert p95_ms < self.KEYWORD_GATE_SLO_MS, (
            f"Layer-1 keyword-gate p95 latency {p95_ms:.4f}ms exceeds the "
            f"{self.KEYWORD_GATE_SLO_MS}ms SLO."
        )
