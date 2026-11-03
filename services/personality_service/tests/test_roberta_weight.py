"""Phase B / B-4 (P2): the seeded-random RoBERTa stub must not carry ensemble weight.

roberta_model has no fine-tuned checkpoint wired (load_weights is never called), so
its "scores" are a fixed seeded-RNG linear projection of heuristic features that
sigmoid toward ~0.5. At the default weight 0.5 (the HIGHEST of the three sources) it
diluted the real LIWC/LLM signal toward neutral. Until a real checkpoint exists it
must be weight 0 — the ensemble renormalizes by total weight, so LIWC+LLM carry it.
"""
from __future__ import annotations

from services.personality_service.src.domain.trait_detector import TraitDetectorSettings


def test_roberta_stub_has_zero_default_ensemble_weight() -> None:
    s = TraitDetectorSettings()
    assert s.ensemble_weights_roberta == 0.0, (
        "the seeded-random RoBERTa stub must be weight 0 until a real checkpoint is loaded"
    )
    # LLM + LIWC still carry the ensemble.
    assert s.ensemble_weights_llm > 0.0
    assert s.ensemble_weights_liwc > 0.0
