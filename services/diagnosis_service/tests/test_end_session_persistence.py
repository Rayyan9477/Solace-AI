"""M1 gate P0: end_session must not silently swallow a clinical-record persist failure.

REV-04 fixed the specific NOT-NULL trigger that used to guarantee the save failed,
but the broad ``try/except`` around ``save_record`` in ``end_session`` still catches
*any* persistence failure (pool exhaustion, transient outage, a future constraint),
logs a warning, and returns a normal ``SessionEndResult``. The caller sees a
successful session end with a summary while the diagnosis record never lands in
Postgres — clinical data silently lost.

The record write must fail loud so the caller/API surfaces it and can retry.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from services.diagnosis_service.src.domain.differential import (
    DifferentialGenerator,
    DifferentialSettings,
)
from services.diagnosis_service.src.domain.service import (
    DiagnosisService,
    DiagnosisServiceSettings,
)
from services.diagnosis_service.src.domain.symptom_extractor import (
    SymptomExtractor,
    SymptomExtractorSettings,
)


class _FailingRepo:
    """Repository whose save_record fails (e.g. transient DB outage)."""

    async def save_record(self, record):  # noqa: ANN001
        raise RuntimeError("DB pool exhausted")


def _service_with_repo(repo) -> DiagnosisService:  # noqa: ANN001
    return DiagnosisService(
        settings=DiagnosisServiceSettings(),
        symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
        differential_generator=DifferentialGenerator(DifferentialSettings()),
        repository=repo,
    )


@pytest.mark.asyncio
async def test_end_session_propagates_persistence_failure() -> None:
    service = _service_with_repo(_FailingRepo())
    await service.initialize()
    owner = uuid4()
    started = await service.start_session(
        user_id=owner,
        session_type="assessment",
        initial_context={},
        previous_session_id=None,
    )

    # A persistence failure must NOT be swallowed into a successful SessionEndResult.
    with pytest.raises(Exception):
        await service.end_session(
            user_id=owner, session_id=started.session_id, generate_summary=False
        )
