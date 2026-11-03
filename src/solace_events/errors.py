"""Solace-AI event plane errors."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class EventStoreSchemaError(RuntimeError):
    """The event tables that Alembic slot ``010_event_plane`` creates are absent or incomplete (MVP-266).

    Services never create these tables (D-01: Alembic is the only schema of record; D-02: ``solace_app`` has no
    DDL), so a store's ``ensure_table()`` raises this at startup instead of running DDL. ``missing`` maps each
    table to the columns it lacks; a table that is absent, or not visible to the connected role, lists all of its
    columns. The message names tables and columns only, never a DSN or a value.
    """

    def __init__(self, missing: Mapping[str, Iterable[str]]) -> None:
        self.missing: dict[str, list[str]] = {table: sorted(columns) for table, columns in sorted(missing.items())}
        detail = "; ".join(f"{table}: {', '.join(columns)}" for table, columns in self.missing.items())
        super().__init__(
            "event store schema is not at slot 010_event_plane; apply the migrations as solace_migrator "
            f"(python -m scripts.db.migrate upgrade head). Missing, or not visible to this role: {detail}"
        )

    def __reduce__(self) -> tuple[type[EventStoreSchemaError], tuple[dict[str, list[str]]]]:
        return (type(self), (self.missing,))


class OutboxOwnershipError(ValueError):
    """An outbox record owned by another service, or by a blank owner, was given to a store (MVP-133).

    A store writes only its own service's ``event_outbox`` rows, the same per-service ownership its claim
    enforces: a foreign-owned row would be published by the other service's poller, and a blank-owned one could
    never be claimed by any poller. Nothing is written. The message names only the two service names.
    """

    def __init__(self, *, store_service: str, record_service: str) -> None:
        self.store_service = store_service
        self.record_service = record_service
        super().__init__(
            f"outbox record owned by source_service {record_service!r} refused by the {store_service!r} store: "
            "a store writes only its own service's rows (MVP-133)"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild_outbox_ownership_error, (self.store_service, self.record_service))


def _rebuild_outbox_ownership_error(store_service: str, record_service: str) -> OutboxOwnershipError:
    return OutboxOwnershipError(store_service=store_service, record_service=record_service)
