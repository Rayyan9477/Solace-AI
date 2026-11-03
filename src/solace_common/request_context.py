"""Request-scoped context: the current user (C.2 / RLS) and the correlation ids.

A neutral contextvar shared by two layers that must not depend on each other:
- the auth layer (``solace_security``) SETS it after validating a request's token;
- the DB layer (``solace_infrastructure``) READS it to set the Postgres RLS GUC
  ``app.current_user_id`` on the connection the request's queries run on.

When unset (service-to-service calls, background jobs, unauthenticated requests) the DB
layer leaves the GUC unset, so RLS returns no user-scoped rows — the safe default.

Work that has no request (registration side-writes, event consumers, CLIs, erasure jobs)
uses ``acting_as_user`` to set the RLS user explicitly for a bounded block (MVP-034).

The correlation ids (``X-Request-ID`` / ``X-Correlation-ID`` / ``X-Turn-ID``) live here too,
so the request-context middleware can set them once per request and every outbound hop
can read them back as headers (MVP-356).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from uuid import UUID

_current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_turn_id: ContextVar[str | None] = ContextVar("turn_id", default=None)

_CORRELATION_HEADER_VARS: tuple[tuple[str, ContextVar[str | None]], ...] = (
    ("X-Request-ID", _request_id),
    ("X-Correlation-ID", _correlation_id),
    ("X-Turn-ID", _turn_id),
)


def set_current_user_id(user_id: str | None) -> Token:
    """Set the current request's user id. Returns a token for ``reset_current_user_id``."""
    return _current_user_id.set(str(user_id) if user_id is not None else None)


def get_current_user_id() -> str | None:
    """Return the current request's user id, or None if unset."""
    return _current_user_id.get()


def reset_current_user_id(token: Token | None = None) -> None:
    """Reset the contextvar — to the pre-set value if a token is given, else to None."""
    if token is not None:
        _current_user_id.reset(token)
    else:
        _current_user_id.set(None)


def _canonical_user_id(user_id: str | UUID | None) -> str | None:
    """Return the canonical lower-case UUID string for ``user_id`` (None passes through)."""
    if user_id is None:
        return None
    if not isinstance(user_id, (str, UUID)):
        # str() of an int would be re-read as hex and silently become a different identity.
        raise ValueError("acting_as_user requires a UUID user id")
    try:
        return str(UUID(str(user_id)))
    except ValueError:
        # The rejected value is deliberately not echoed: it may be an email or other PII.
        raise ValueError("acting_as_user requires a UUID user id") from None


@contextmanager
def acting_as_user(user_id: str | UUID | None) -> Iterator[str | None]:
    """Act as ``user_id`` for RLS inside the block, restoring the previous user on exit.

    The value is normalised to the canonical lower-case UUID string (or None) and yielded.
    A value that is not a UUID raises ``ValueError`` before anything is set. The previous
    value is restored on exit, including when the block raises.
    """
    value = _canonical_user_id(user_id)
    token = set_current_user_id(value)
    try:
        yield value
    finally:
        reset_current_user_id(token)


def set_correlation_ids(*, request_id: str, correlation_id: str, turn_id: str) -> None:
    """Set the correlation ids for the current context. Empty values are stored as unset."""
    _request_id.set(request_id or None)
    _correlation_id.set(correlation_id or None)
    _turn_id.set(turn_id or None)


def get_correlation_headers() -> dict[str, str]:
    """Return the outbound headers for the correlation ids that are set and non-empty."""
    return {name: value for name, var in _CORRELATION_HEADER_VARS if (value := var.get())}


def clear_correlation_ids() -> None:
    """Unset all correlation ids in the current context."""
    for _, var in _CORRELATION_HEADER_VARS:
        var.set(None)
