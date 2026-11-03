"""
REV-26 reconciliation Prove-It: Devil's-Advocate declared bias set must equal
the biases the detector can actually emit.

Before this reconciliation, ``_build_bias_patterns`` declared 6 cognitive biases
but ``analyze_bias`` only ever emits 4 (premature_closure, confirmation_bias,
anchoring_bias, availability_heuristic). ``base_rate_neglect`` and
``attribution_error`` were dead metadata: advertised as a capability but never
producible at runtime. This test locks the advertised set to reality.
"""
from __future__ import annotations
import inspect
import re
from decimal import Decimal
from uuid import uuid4

from services.diagnosis_service.src.schemas import HypothesisDTO
from services.diagnosis_service.src.domain.advocate import (
    DevilsAdvocate, AdvocateSettings,
)


def _emittable_from_source(advocate: DevilsAdvocate) -> set[str]:
    """Biases the detector CAN produce, derived from analyze_bias source."""
    source = inspect.getsource(advocate.analyze_bias)
    return set(re.findall(r'detected_biases\.append\(\s*"([^"]+)"\s*\)', source))


class TestDeclaredBiasesMatchEmittable:
    def setup_method(self) -> None:
        self.advocate = DevilsAdvocate(AdvocateSettings())

    def test_no_declared_bias_is_never_emitted(self) -> None:
        """Every declared bias must be one the detector can actually emit."""
        declared = set(self.advocate._bias_patterns.keys())
        emittable = _emittable_from_source(self.advocate)
        # The advertised capability equals reality: no orphan declarations.
        assert declared == emittable, (
            f"declared-but-never-emitted: {declared - emittable}; "
            f"emitted-but-undeclared: {emittable - declared}"
        )

    def test_dead_biases_removed(self) -> None:
        """The two never-emitted biases are gone from the declared set."""
        declared = set(self.advocate._bias_patterns.keys())
        assert "base_rate_neglect" not in declared
        assert "attribution_error" not in declared

    def test_get_bias_description_none_for_removed(self) -> None:
        """Removed biases no longer advertise a description."""
        assert self.advocate.get_bias_description("base_rate_neglect") is None
        assert self.advocate.get_bias_description("attribution_error") is None

    async def test_every_declared_bias_is_runtime_reachable(self) -> None:
        """Each declared bias is emitted by at least one concrete scenario."""
        emitted: set[str] = set()

        # premature_closure: single hypothesis, confidence > 0.7
        emitted |= set((await self.advocate.analyze_bias(
            [HypothesisDTO(hypothesis_id=uuid4(), name="MDD",
                           confidence=Decimal("0.75"), criteria_met=["m"])],
            [], [],
        )).detected_biases)

        # confirmation_bias: confidence > 0.85 with >2 missing criteria
        emitted |= set((await self.advocate.analyze_bias(
            [HypothesisDTO(hypothesis_id=uuid4(), name="Depression",
                           confidence=Decimal("0.9"), criteria_met=["m"],
                           criteria_missing=["a", "b", "c"])],
            [], [],
        )).detected_biases)

        # anchoring_bias: two hypotheses with confidence gap > 0.4
        emitted |= set((await self.advocate.analyze_bias(
            [HypothesisDTO(hypothesis_id=uuid4(), name="A", confidence=Decimal("0.9"),
                           criteria_met=["m"]),
             HypothesisDTO(hypothesis_id=uuid4(), name="B", confidence=Decimal("0.4"),
                           criteria_met=["m"])],
            [], [],
        )).detected_biases)

        # availability_heuristic: common disorder with confidence > 0.8
        emitted |= set((await self.advocate.analyze_bias(
            [HypothesisDTO(hypothesis_id=uuid4(), name="Major Depressive Disorder",
                           confidence=Decimal("0.85"), criteria_met=["m"])],
            [], [],
        )).detected_biases)

        declared = set(self.advocate._bias_patterns.keys())
        assert emitted == declared, (
            f"declared not reachable at runtime: {declared - emitted}; "
            f"reachable but undeclared: {emitted - declared}"
        )
