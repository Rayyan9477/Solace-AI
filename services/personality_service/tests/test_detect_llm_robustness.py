"""Phase B / B-1 (P1): personality /detect must not 500 on every call.

The REV-15 brace-escape fix was applied to the *dead* class ml/llm_detector.py.
The ACTIVE detector is domain/trait_detector.py::LLMBasedDetector, whose
_DETECTION_PROMPT still had an unescaped JSON example ({"openness": 0.0, ...}) and
whose .format(text=...) ran OUTSIDE the try — so it raised KeyError('"openness"')
before any LLM call, propagated through TraitDetector.detect (unguarded), and
surfaced as HTTP 500 on every /detect whenever the shared LLM client is importable
(the default production config).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.personality_service.src.domain.trait_detector import (
    LLMBasedDetector,
    TraitDetector,
    TraitDetectorSettings,
)


def test_detection_prompt_formats_without_keyerror() -> None:
    """The active detector's prompt must .format(text=...) without a KeyError."""
    out = LLMBasedDetector._DETECTION_PROMPT.format(text="I love exploring new ideas")
    assert "I love exploring new ideas" in out
    # The literal JSON example survives (escaped braces render as single braces).
    assert '"openness"' in out


@pytest.mark.asyncio
async def test_detect_returns_scores_when_llm_client_fails() -> None:
    """A failing LLM client must degrade to a fallback, never crash /detect."""
    client = MagicMock()
    client.generate = AsyncMock(side_effect=RuntimeError("LLM provider down"))
    detector = TraitDetector(settings=TraitDetectorSettings(), llm_client=client)
    await detector.initialize()

    scores = await detector.detect(
        "I really enjoy meeting new people and starting creative projects every single week."
    )
    assert scores is not None  # must not raise / 500
