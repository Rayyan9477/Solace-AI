"""Phase B / B-5 (P2): assessment_history must survive the persist round-trip.

_profile_to_entity built the domain entity WITHOUT assessment_history and
_entity_to_profile never restored it, so a profile reloaded from the repository lost
its entire assessment history (the DB column round-trips fine — the loss was purely in
the service-layer mapping). This breaks stability/trend computation, which reads
profile.assessment_history. Also the FIRST assessment was never appended to history,
leaving len(history) == assessment_count - 1 (off-by-one).
"""
from __future__ import annotations

from uuid import uuid4
import pytest

from services.personality_service.src.schemas import OceanScoresDTO, AssessmentSource
from services.personality_service.src.domain.service import (
    PersonalityOrchestrator,
    PersonalityServiceSettings,
    PersonalityProfile,
)
from services.personality_service.tests.fixtures import InMemoryPersonalityRepository


def _scores(o, c, e, a, n, conf=0.7) -> OceanScoresDTO:
    return OceanScoresDTO(
        openness=o, conscientiousness=c, extraversion=e,
        agreeableness=a, neuroticism=n, overall_confidence=conf,
    )


@pytest.fixture
def settings() -> PersonalityServiceSettings:
    return PersonalityServiceSettings(enable_llm_detection=False)


@pytest.fixture
def orchestrator_with_repo(settings: PersonalityServiceSettings) -> PersonalityOrchestrator:
    return PersonalityOrchestrator(
        settings=settings, llm_client=None, repository=InMemoryPersonalityRepository()
    )


def test_profile_to_entity_preserves_history(orchestrator_with_repo: PersonalityOrchestrator) -> None:
    profile = PersonalityProfile(user_id=uuid4())
    profile.ocean_scores = _scores(0.7, 0.6, 0.5, 0.8, 0.4)
    profile.assessment_history = [
        _scores(0.6, 0.6, 0.5, 0.7, 0.4),
        _scores(0.8, 0.6, 0.5, 0.9, 0.4),
    ]

    entity = orchestrator_with_repo._profile_to_entity(profile)
    assert len(entity.assessment_history) == 2


def test_entity_to_profile_restores_history_values(orchestrator_with_repo: PersonalityOrchestrator) -> None:
    profile = PersonalityProfile(user_id=uuid4())
    profile.ocean_scores = _scores(0.7, 0.6, 0.5, 0.8, 0.4)
    profile.assessment_history = [_scores(0.61, 0.62, 0.53, 0.74, 0.45, conf=0.83)]

    entity = orchestrator_with_repo._profile_to_entity(profile)
    restored = orchestrator_with_repo._entity_to_profile(entity)

    assert len(restored.assessment_history) == 1
    h = restored.assessment_history[0]
    assert h.openness == pytest.approx(0.61)
    assert h.agreeableness == pytest.approx(0.74)
    assert h.neuroticism == pytest.approx(0.45)
    assert h.overall_confidence == pytest.approx(0.83)


@pytest.mark.asyncio
async def test_history_survives_repo_reload(orchestrator_with_repo: PersonalityOrchestrator) -> None:
    await orchestrator_with_repo.initialize()
    user_id = uuid4()
    await orchestrator_with_repo.update_profile_from_scores(
        user_id, _scores(0.6, 0.5, 0.5, 0.7, 0.4), AssessmentSource.SELF_REPORT
    )
    await orchestrator_with_repo.update_profile_from_scores(
        user_id, _scores(0.8, 0.5, 0.5, 0.9, 0.4), AssessmentSource.SELF_REPORT
    )

    # Evict the in-memory ProfileStore cache to force a reload from the repository.
    await orchestrator_with_repo._profile_store.delete(user_id)
    reloaded = await orchestrator_with_repo.get_profile(user_id)  # triggers _entity_to_profile
    assert reloaded is not None

    cached = orchestrator_with_repo._profile_store.get(user_id)
    assert cached is not None
    assert len(cached.assessment_history) >= 1, "history must survive persist + reload"


@pytest.mark.asyncio
async def test_first_assessment_recorded_in_history(orchestrator_with_repo: PersonalityOrchestrator) -> None:
    await orchestrator_with_repo.initialize()
    user_id = uuid4()
    profile = await orchestrator_with_repo._get_or_create_profile(user_id)
    profile = orchestrator_with_repo._update_profile_with_assessment(
        profile, _scores(0.7, 0.6, 0.5, 0.8, 0.4)
    )
    assert profile.assessment_count == 1
    assert len(profile.assessment_history) == 1  # off-by-one fix: first assessment recorded
