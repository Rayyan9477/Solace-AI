"""C.2 / REV-17: shared GDPR right-to-erasure orchestrator.

The orchestrator runs a set of store-scoped deleters for a single user and
guarantees the fail-loud contract the codebase adopted for REV-36/44:
  - EVERY registered deleter is attempted, even if an earlier one raises;
  - per-store counts and per-store failures are both collected;
  - if ANY deleter failed the caller gets an ``ErasureIncomplete`` carrying the
    partial report — a partial erasure is NEVER reported as complete;
  - the completion audit event fires only on FULL success (injectable, so the
    core has no hard dependency on the audit service).
"""
from __future__ import annotations

import pytest

from solace_infrastructure.gdpr import (
    ErasureIncomplete,
    ErasureReport,
    UserDataErasure,
)


def _deleter(count: int):
    """A StoreDeleter that reports it deleted ``count`` rows."""

    async def _run(user_id: str) -> int:
        return count

    return _run


def _raising_deleter(exc: Exception):
    """A StoreDeleter that fails."""

    async def _run(user_id: str) -> int:
        raise exc

    return _run


class TestAllDeletersSucceed:
    async def test_report_complete_counts_and_audit_emitted_once(self) -> None:
        erasure = UserDataErasure()
        erasure.register("postgres_users", _deleter(3))
        erasure.register("weaviate_vectors", _deleter(7))

        emitted: list[ErasureReport] = []

        async def audit_emit(report: ErasureReport) -> None:
            emitted.append(report)

        report = await erasure.erase("user-123", audit_emit=audit_emit)

        assert report.complete is True
        assert report.deleted == {"postgres_users": 3, "weaviate_vectors": 7}
        assert report.failed == {}
        # audit fires exactly once, with the very report that was returned.
        assert emitted == [report]
        assert report.audit_emitted is True


class TestOneDeleterRaises:
    async def test_incomplete_raised_but_other_deleters_still_ran(self) -> None:
        erasure = UserDataErasure()
        erasure.register("postgres_users", _deleter(3))
        # The failing store is registered BETWEEN two good ones, so a passing
        # test proves the deleter AFTER it still ran (no short-circuit).
        erasure.register("safety_assessments", _raising_deleter(RuntimeError("boom")))
        erasure.register("weaviate_vectors", _deleter(7))

        audit_calls: list[ErasureReport] = []

        async def audit_emit(report: ErasureReport) -> None:
            audit_calls.append(report)

        with pytest.raises(ErasureIncomplete) as excinfo:
            await erasure.erase("user-123", audit_emit=audit_emit)

        report = excinfo.value.report
        # The failing store is recorded as a failure (with its error text).
        assert "safety_assessments" in report.failed
        assert "boom" in report.failed["safety_assessments"]
        # The OTHER two deleters STILL ran and recorded their counts.
        assert report.deleted == {"postgres_users": 3, "weaviate_vectors": 7}
        # A partial erasure is NEVER complete...
        assert report.complete is False
        # ...and NEVER emits the completion audit event.
        assert audit_calls == []
        # The exception also surfaces the per-store failures directly.
        assert "safety_assessments" in excinfo.value.failed


class TestEmptyRegistry:
    async def test_empty_registry_refuses_to_attest(self) -> None:
        # A GDPR run with NO stores registered is a misconfiguration, not a success —
        # it must NOT return complete=True (which would be a false erasure attestation).
        erasure = UserDataErasure()
        with pytest.raises(RuntimeError):
            await erasure.erase("user-123")


class TestAuditEmitFailure:
    async def test_audit_sink_failure_does_not_fail_a_successful_erasure(self) -> None:
        # Data is erased; a broken audit sink is logged distinctly, NOT raised as an
        # erasure failure (re-driving would be wrong — the data is already gone).
        erasure = UserDataErasure()
        erasure.register("postgres_users", _deleter(3))

        async def audit_emit(report: ErasureReport) -> None:
            raise RuntimeError("audit sink down")

        report = await erasure.erase("user-123", audit_emit=audit_emit)

        assert report.complete is True
        assert report.deleted == {"postgres_users": 3}
        assert report.audit_emitted is False  # caller can see the audit didn't land


class TestBadDeleterReturn:
    async def test_non_int_return_is_not_marked_failed(self) -> None:
        # A deleter that succeeded but returned a non-int (contract slip) must not be
        # marked failed (its rows are gone) — record 0 and keep the erasure complete.
        erasure = UserDataErasure()

        async def none_returning(user_id: str):
            return None  # forgot to return the count

        erasure.register("sloppy_store", none_returning)
        erasure.register("good_store", _deleter(5))

        report = await erasure.erase("user-123")

        assert report.complete is True
        assert report.failed == {}
        assert report.deleted == {"sloppy_store": 0, "good_store": 5}
