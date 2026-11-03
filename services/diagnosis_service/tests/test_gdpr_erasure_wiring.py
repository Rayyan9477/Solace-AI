"""REV-17: diagnosis-service GDPR right-to-erasure wiring.

The diagnosis service already has a working durable delete
(``DiagnosisRepositoryPort.delete_user_data`` wipes the user's sessions AND
records). REV-17 routes that delete through the shared, fail-loud
``UserDataErasure`` orchestrator so a failed store delete is NEVER reported as a
completed erasure.

Acceptance criteria encoded here:
  (a) the factory registers EXACTLY the diagnosis Postgres store;
  (b) ``erase`` deletes all of the user's rows via the repository and returns
      the rows-deleted count;
  (c) if the repository delete raises, ``erase`` raises ``ErasureIncomplete``
      (fail-loud) — a partial/failed erasure is not swallowed;
  (d) the domain ``DiagnosisService.delete_user_data`` drives the delete through
      the orchestrator (durable rows gone on success; failure propagates).
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from solace_infrastructure.gdpr import ErasureIncomplete, UserDataErasure

from services.diagnosis_service.src.domain.entities import (
    DiagnosisRecordEntity,
    DiagnosisSessionEntity,
)
from services.diagnosis_service.src.domain.erasure import (
    DIAGNOSIS_POSTGRES_STORE,
    build_user_data_erasure,
)
from services.diagnosis_service.src.domain.service import (
    DiagnosisService,
    DiagnosisServiceSettings,
)
from services.diagnosis_service.tests.fixtures import InMemoryDiagnosisRepository


# ---------------------------------------------------------------------------
# (a) factory registers exactly the expected store name(s)
# ---------------------------------------------------------------------------
def test_factory_registers_exactly_diagnosis_postgres_store() -> None:
    repo = InMemoryDiagnosisRepository()
    erasure = build_user_data_erasure(repo)

    assert isinstance(erasure, UserDataErasure)
    assert erasure.store_names == (DIAGNOSIS_POSTGRES_STORE,)
    assert DIAGNOSIS_POSTGRES_STORE == "diagnosis_postgres"


# ---------------------------------------------------------------------------
# (b) erase() deletes all user rows via the repo and returns the count
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_erase_deletes_all_user_rows_and_returns_count() -> None:
    repo = InMemoryDiagnosisRepository()
    user_id = uuid4()
    other_user = uuid4()

    # Seed 2 sessions + 1 record for the target user, plus one other user's row
    # that must be untouched.
    await repo.save_session(DiagnosisSessionEntity(user_id=user_id))
    await repo.save_session(DiagnosisSessionEntity(user_id=user_id))
    await repo.save_record(DiagnosisRecordEntity(user_id=user_id, primary_diagnosis="GAD"))
    await repo.save_session(DiagnosisSessionEntity(user_id=other_user))

    erasure = build_user_data_erasure(repo)
    report = await erasure.erase(user_id)

    assert report.complete is True
    assert report.failed == {}
    # 2 sessions + 1 record = 3 rows removed for this user, all from one store.
    assert report.deleted[DIAGNOSIS_POSTGRES_STORE] == 3
    assert report.total_deleted == 3

    # Target user's rows are gone; the other user's row survives.
    assert await repo.list_user_sessions(user_id) == []
    assert await repo.list_user_records(user_id) == []
    assert len(await repo.list_user_sessions(other_user)) == 1


@pytest.mark.asyncio
async def test_erase_fires_audit_only_on_full_success() -> None:
    repo = InMemoryDiagnosisRepository()
    user_id = uuid4()
    await repo.save_session(DiagnosisSessionEntity(user_id=user_id))

    emitted: list = []
    erasure = build_user_data_erasure(repo)
    report = await erasure.erase(user_id, audit_emit=lambda r: emitted.append(r))

    assert report.audit_emitted is True
    assert emitted and emitted[0] is report


# ---------------------------------------------------------------------------
# (c) repo delete raises -> erase() raises ErasureIncomplete (fail-loud)
# ---------------------------------------------------------------------------
class _FailingRepo:
    """Repository whose delete_user_data fails (e.g. transient DB outage)."""

    async def delete_user_data(self, user_id):  # noqa: ANN001
        raise RuntimeError("DB pool exhausted")


@pytest.mark.asyncio
async def test_erase_fails_loud_when_repo_delete_raises() -> None:
    erasure = build_user_data_erasure(_FailingRepo())

    with pytest.raises(ErasureIncomplete) as exc_info:
        await erasure.erase(uuid4())

    report = exc_info.value.report
    assert report.complete is False
    assert DIAGNOSIS_POSTGRES_STORE in report.failed
    assert report.audit_emitted is False


# ---------------------------------------------------------------------------
# (d) domain delete_user_data drives the durable delete through the orchestrator
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_domain_delete_user_data_routes_through_erasure() -> None:
    repo = InMemoryDiagnosisRepository()
    user_id = uuid4()
    await repo.save_session(DiagnosisSessionEntity(user_id=user_id))
    await repo.save_record(DiagnosisRecordEntity(user_id=user_id, primary_diagnosis="MDD"))

    service = DiagnosisService(settings=DiagnosisServiceSettings(), repository=repo)
    await service.initialize()

    # Existing HTTP contract: returns None (endpoint maps to 204).
    result = await service.delete_user_data(user_id)
    assert result is None

    assert await repo.list_user_sessions(user_id) == []
    assert await repo.list_user_records(user_id) == []


@pytest.mark.asyncio
async def test_domain_delete_user_data_propagates_erasure_failure() -> None:
    service = DiagnosisService(
        settings=DiagnosisServiceSettings(), repository=_FailingRepo()
    )
    await service.initialize()

    # A failed durable erase must NOT be swallowed into a "successful" delete.
    with pytest.raises(ErasureIncomplete):
        await service.delete_user_data(uuid4())
