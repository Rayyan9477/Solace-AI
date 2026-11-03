"""C.2 / REV-17: shared GDPR right-to-erasure orchestrator.

A user's right to erasure (GDPR Art. 17) spans every store that holds their PHI:
Postgres tables across a dozen service databases, Weaviate vector collections,
and Redis caches. Each service knows how to delete its own rows; what is missing
is a single component that drives them all with a strict, auditable contract.

``UserDataErasure`` is that store-agnostic core. Services register a named
:data:`StoreDeleter` (an ``async (user_id) -> deleted_count``) for each store
they own, and :meth:`UserDataErasure.erase` runs them all for one user under the
fail-loud rules the codebase adopted for REV-36/44:

  - EVERY registered deleter is attempted, even if an earlier one raises — one
    store's outage must not leave the others un-erased;
  - per-store counts and per-store failures are BOTH collected;
  - if ANY deleter failed the caller gets an :class:`ErasureIncomplete` carrying
    the partial :class:`ErasureReport` — a partial erasure is NEVER reported as
    complete, and a failure is NEVER swallowed;
  - the completion audit event fires ONLY on full success, via an injected
    ``audit_emit`` callback so this core has no hard dependency on the audit
    service.

The actual per-service store deleters (wiring each service's
``delete_user_data`` / repository deletes into this registry) are a follow-up;
this module ships the orchestrator and the registration API.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# An async callable that deletes all of one user's rows from a single store and
# returns the number of rows it removed.
StoreDeleter = Callable[[str], Awaitable[int]]


@dataclass
class ErasureReport:
    """Outcome of a right-to-erasure run for a single user.

    ``deleted`` maps store name -> rows removed for the stores that succeeded;
    ``failed`` maps store name -> error string for the stores that raised.
    ``complete`` is True only when every registered deleter succeeded.
    """

    user_id: str
    deleted: dict[str, int] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    complete: bool = False
    # True only when a completion audit was requested AND successfully emitted. A failed
    # audit sink does not make the erasure itself incomplete (the data is already gone).
    audit_emitted: bool = False

    @property
    def total_deleted(self) -> int:
        """Total rows removed across all stores that succeeded."""
        return sum(self.deleted.values())

    def to_dict(self) -> dict[str, object]:
        """Serialize for the audit event / structured logging."""
        return {
            "user_id": self.user_id,
            "deleted": dict(self.deleted),
            "failed": dict(self.failed),
            "complete": self.complete,
            "audit_emitted": self.audit_emitted,
            "total_deleted": self.total_deleted,
        }


class ErasureIncomplete(Exception):
    """Raised when one or more store deleters failed during erasure.

    Carries the partial :class:`ErasureReport` (with the per-store counts that
    DID succeed) and the per-store failures, so the caller can surface, retry,
    and — critically — NOT report the erasure as complete.
    """

    def __init__(self, report: ErasureReport) -> None:
        self.report = report
        self.failed = report.failed
        stores = ", ".join(sorted(report.failed)) or "unknown"
        super().__init__(
            f"GDPR erasure incomplete for user {report.user_id}: "
            f"{len(report.failed)} store(s) failed ({stores})"
        )


class UserDataErasure:
    """Store-agnostic orchestrator for GDPR right-to-erasure.

    Register a :data:`StoreDeleter` per store, then call :meth:`erase` to run
    them all for one user under the fail-loud contract described in the module
    docstring.
    """

    def __init__(self) -> None:
        self._deleters: dict[str, StoreDeleter] = {}

    def register(self, name: str, deleter: StoreDeleter) -> None:
        """Register a named store deleter.

        Duplicate names are rejected fail-loud: silently overwriting a deleter
        would mean a store is never erased, which is exactly the class of bug
        this component exists to prevent.
        """
        if not name:
            raise ValueError("store deleter name must be a non-empty string")
        if name in self._deleters:
            raise ValueError(f"store deleter already registered: {name!r}")
        self._deleters[name] = deleter

    @property
    def store_names(self) -> tuple[str, ...]:
        """Names of the registered stores, in registration order."""
        return tuple(self._deleters)

    async def erase(
        self,
        user_id: str,
        *,
        audit_emit: Optional[Callable[[ErasureReport], object]] = None,
    ) -> ErasureReport:
        """Erase one user's data from every registered store.

        Runs ALL deleters even if some fail, collecting per-store counts and
        per-store failures. On full success, invokes ``audit_emit(report)`` (if
        provided) and returns the report. If ANY deleter raised, raises
        :class:`ErasureIncomplete` with the partial report AFTER attempting them
        all — never swallowing a failure, never reporting a partial as complete.

        ``audit_emit`` may be a sync or async callable; it is only called on a
        fully successful erasure, and a failing audit sink is logged distinctly (it does
        NOT turn a successful erasure into a failure). Note: a deleter raising
        ``asyncio.CancelledError`` (BaseException) aborts the run immediately — remaining
        stores are un-attempted — so callers must treat a cancelled erase() as incomplete.
        """
        if not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not self._deleters:
            # A GDPR run with zero registered stores is a misconfiguration, not a
            # success — refuse to attest an erasure that could not have happened.
            raise RuntimeError(
                "UserDataErasure.erase called with no registered store deleters"
            )
        uid = str(user_id)

        report = ErasureReport(user_id=uid)
        # Snapshot so a concurrent register() can't mutate the dict mid-iteration.
        for name, deleter in list(self._deleters.items()):
            try:
                raw = await deleter(uid)
            except Exception as exc:  # noqa: BLE001 - collect, then fail loud below
                report.failed[name] = str(exc)
                logger.error("erasure_store_failed", store=name, user_id=uid, error=str(exc))
                continue
            # A deleter that succeeded but returned a non-int count is a benign contract
            # slip — record 0 with a warning rather than marking the (completed) delete failed.
            if not isinstance(raw, int) or isinstance(raw, bool):
                logger.warning(
                    "erasure_store_bad_count", store=name, user_id=uid,
                    returned=type(raw).__name__,
                )
                raw = 0
            report.deleted[name] = raw
            logger.info("erasure_store_deleted", store=name, user_id=uid, deleted=raw)

        report.complete = not report.failed

        if not report.complete:
            logger.error(
                "erasure_incomplete",
                user_id=uid,
                failed_stores=sorted(report.failed),
                deleted_stores=sorted(report.deleted),
            )
            raise ErasureIncomplete(report)

        # Data is erased. Emit the completion audit, isolating a failing audit sink so it
        # is not mistaken for a failed erasure (the data is already gone).
        if audit_emit is not None:
            try:
                result = audit_emit(report)
                if inspect.isawaitable(result):
                    await result
                report.audit_emitted = True
            except Exception as exc:  # noqa: BLE001
                report.audit_emitted = False
                logger.error("erasure_audit_emit_failed", user_id=uid, error=str(exc))

        logger.info(
            "erasure_complete",
            user_id=uid,
            stores=len(report.deleted),
            total_deleted=report.total_deleted,
            audit_emitted=report.audit_emitted,
        )
        return report
