"""
REV-30 Prove-It: current-turn symptoms must inform THIS message's step-4 calibration.

The 4-step Chain-of-Reasoning merges the current message's freshly-extracted
symptoms into ``session.symptoms`` only AFTER the chain completes (``_update_session``).
Before the fix, step-4 Bayesian calibration saw only PRIOR-turn symptoms, so a symptom
mentioned only in the current utterance was ignored for this message's calibration.

These tests construct cases where prior-turn vs current-turn symptoms produce a
different Bayesian calibration and assert the current-turn symptom is used.

The lever is ``ConfidenceCalibrator._check_symptom_consistency``: for a
``major depressive disorder`` hypothesis, symptom_consistency = 1.0 when a required
symptom (``anhedonia``/``depressed_mood``) is present, else 0.5. That 0.5 gap flows
into the calibrated confidence, so its presence in the working symptom set is
directly observable in ``differential.primary.confidence``.
"""
from __future__ import annotations
import pytest
from decimal import Decimal
from uuid import uuid4

from services.diagnosis_service.src.schemas import (
    DiagnosisPhase, SeverityLevel, SymptomType,
    SymptomDTO, HypothesisDTO,
)
from services.diagnosis_service.src.domain.service import (
    DiagnosisService, DiagnosisServiceSettings,
)


def _sym(name: str) -> SymptomDTO:
    """Build a MILD symptom with identical severity/confidence so the only
    calibration-relevant difference between two symptoms is their name."""
    return SymptomDTO(
        symptom_id=uuid4(), name=name, description=name,
        symptom_type=SymptomType.EMOTIONAL, severity=SeverityLevel.MILD,
        confidence=Decimal("0.7"),
    )


# Message markers the fake extractor keys off of.
PRIOR_FATIGUE = "prior turn: I am tired [PRIOR_FATIGUE]"       # -> [fatigue] (MDD consistency 0.5)
CUR_ANHEDONIA = "this turn: nothing brings joy [CUR_ANHEDONIA]"  # -> [anhedonia] (MDD consistency 1.0)
CUR_NONE = "this turn: hello there [CUR_NONE]"                 # -> [] (no new symptoms)


class _FakeExtraction:
    def __init__(self, symptoms: list[SymptomDTO]) -> None:
        self.symptoms = symptoms
        self.temporal_info: dict[str, str] = {}
        self.contextual_factors: list[str] = []
        self.risk_indicators: list[str] = []


class _FakeExtractor:
    """Deterministic extractor: returns a fixed symptom set based on a message marker."""

    async def extract(self, message: str, history, existing) -> _FakeExtraction:
        if "[PRIOR_FATIGUE]" in message:
            return _FakeExtraction([_sym("fatigue")])
        if "[CUR_ANHEDONIA]" in message:
            return _FakeExtraction([_sym("anhedonia")])
        return _FakeExtraction([])


class _FakeDifferential:
    def __init__(self, hypotheses: list[HypothesisDTO]) -> None:
        self.hypotheses = hypotheses
        self.missing_info: list[str] = []
        self.hitop_scores: dict = {}
        self.recommended_questions: list[str] = []


class _FakeGenerator:
    """Always returns the same MDD hypothesis regardless of input symptoms, so the
    ONLY thing that varies calibration across runs is the symptom set fed to it."""

    async def generate(self, symptoms, user_context) -> _FakeDifferential:
        hyp = HypothesisDTO(
            hypothesis_id=uuid4(), name="major depressive disorder",
            confidence=Decimal("0.6"),
            criteria_met=["low_mood", "sleep_disturbance"],
            criteria_missing=["weight_change"],
        )
        return _FakeDifferential([hyp])


def _make_service() -> DiagnosisService:
    # anti-sycophancy OFF removes the Devil's-Advocate confidence adjustment so the
    # only calibration variable is the symptom set reaching step-4.
    settings = DiagnosisServiceSettings(enable_anti_sycophancy=False)
    return DiagnosisService(
        settings=settings,
        symptom_extractor=_FakeExtractor(),
        differential_generator=_FakeGenerator(),
    )


async def _assess(svc: DiagnosisService, user_id, sid, message):
    return await svc.assess(
        user_id=user_id, session_id=sid, message=message,
        conversation_history=[], existing_symptoms=[],
        current_phase=DiagnosisPhase.RAPPORT, current_differential=None,
        user_context={},
    )


async def _run_turns(messages: list[str]):
    svc = _make_service()
    await svc.initialize()
    user_id = uuid4()
    start = await svc.start_session(
        user_id=user_id, session_type="assessment", initial_context={}, previous_session_id=None
    )
    sid = start.session_id
    result = None
    for msg in messages:
        result = await _assess(svc, user_id, sid, msg)
    return result


@pytest.mark.asyncio
async def test_current_turn_symptom_affects_this_message_calibration() -> None:
    """A relevant symptom mentioned ONLY in the current message must raise THIS
    message's calibrated confidence beyond what the prior-turn symptom alone gives.

    Both runs share an identical prior turn (session symptoms = [fatigue]).
    - MAIN turn-2 adds 'anhedonia' (a required MDD symptom) => MDD consistency 1.0.
    - CONTROL turn-2 adds nothing               => calibration sees [fatigue] only => 0.5.

    Before the fix, MAIN's turn-2 calibration also saw only the prior [fatigue]
    (current 'anhedonia' not yet merged), so the two confidences were EQUAL.
    """
    main = await _run_turns([PRIOR_FATIGUE, CUR_ANHEDONIA])
    control = await _run_turns([PRIOR_FATIGUE, CUR_NONE])

    assert main.differential.primary is not None
    assert control.differential.primary is not None
    conf_main = main.differential.primary.confidence
    conf_control = control.differential.primary.confidence

    assert conf_main > conf_control, (
        f"current-turn symptom ignored by step-4 calibration: "
        f"main={conf_main} control={conf_control}"
    )


@pytest.mark.asyncio
async def test_symptom_only_in_current_message_on_fresh_session_is_used() -> None:
    """On a FRESH session (no prior symptoms), a symptom present ONLY in the current
    message must still drive calibration for that same message."""
    relevant = await _run_turns([CUR_ANHEDONIA])     # working set = [anhedonia] -> 1.0
    irrelevant = await _run_turns([PRIOR_FATIGUE])   # working set = [fatigue]   -> 0.5

    assert relevant.differential.primary is not None
    assert irrelevant.differential.primary is not None
    conf_relevant = relevant.differential.primary.confidence
    conf_irrelevant = irrelevant.differential.primary.confidence

    assert conf_relevant > conf_irrelevant, (
        f"current-only symptom on fresh session ignored: "
        f"relevant={conf_relevant} irrelevant={conf_irrelevant}"
    )
