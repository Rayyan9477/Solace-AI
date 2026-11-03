"""Phase B / B-3 (P1): POST /profile/update must honor submitted OCEAN scores.

The handler ignored request.ocean_scores and synthesized a 39-char placeholder text
that fell below min_text_length=50, so detection returned neutral 0.5 scores that
EMA-dragged every trait toward 0.5 on each update. A submitted external assessment
must actually move the profile toward the submitted values.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from services.personality_service.src.domain.service import PersonalityOrchestrator
from services.personality_service.src.schemas import AssessmentSource, OceanScoresDTO


@pytest.mark.asyncio
async def test_update_profile_from_scores_honors_submitted_values() -> None:
    orch = PersonalityOrchestrator()  # no repo / no llm — pure in-memory
    await orch.initialize()
    uid = uuid4()

    high_openness = OceanScoresDTO(
        openness=0.95, conscientiousness=0.5, extraversion=0.5,
        agreeableness=0.5, neuroticism=0.5,
    )
    await orch.update_profile_from_scores(uid, high_openness, AssessmentSource.SELF_REPORT)

    prof = await orch.get_profile(uid)
    assert prof is not None
    # Must move toward the submitted 0.95 — was stuck at neutral 0.5 (placeholder text).
    assert prof.ocean_scores.openness > 0.55, (
        f"submitted openness ignored: profile openness={prof.ocean_scores.openness}"
    )
