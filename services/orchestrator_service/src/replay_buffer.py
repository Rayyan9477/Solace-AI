"""REV-23: bounded per-session replay buffer for the outbound WS message tail.

WS thread-continuity already resumes conversation context by (user_id, session_id)
on reconnect. What was missing is server-side replay of the in-flight OUTBOUND
message tail a client missed while briefly disconnected: messages the server
emitted after the client dropped but before it reconnected.

This module provides that seam. Each outbound message is tagged with a monotonic
per-session sequence number as it is appended; on reconnect a client presents the
last seq it saw and the buffer replays only the messages after it (seq > last_seq),
exactly once, in order, BEFORE live streaming resumes. The buffer is bounded (cap
N per session, oldest evicted); if the requested cursor is older than the retained
window a gap is signalled (resync) rather than silently dropping messages.

The store is intentionally behind a small interface so a durable (Redis/DB) backing
can be substituted later — durable replay is infra-gated; the in-memory store keeps
this feature usable now without a database. New behaviour is inert unless a store is
wired onto ``app.state.replay_buffer`` (OFF by default).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayResult:
    """Outcome of a replay query for a given cursor.

    messages: the retained messages with seq > cursor, in ascending seq order,
              each a copy that already carries its ``seq`` field.
    gap:      True when the message immediately after the cursor has already been
              evicted (cursor older than the retained window) — the client should
              resync rather than assume it received a contiguous tail.
    oldest_seq / latest_seq: bounds of the retained window (None if empty).
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    gap: bool = False
    oldest_seq: int | None = None
    latest_seq: int | None = None


class InMemoryReplayBufferStore:
    """In-memory bounded ring buffer of outbound messages, keyed per session.

    The sequence counter is monotonic per key for the lifetime of the process and
    is NOT reset by eviction — evicting the oldest retained message never re-uses a
    seq, so cursors stay meaningful across the retained window.
    """

    def __init__(self, max_per_session: int = 100) -> None:
        if max_per_session < 1:
            raise ValueError("max_per_session must be >= 1")
        self._max = max_per_session
        self._buffers: dict[str, deque[dict[str, Any]]] = {}
        self._seq: dict[str, int] = {}

    def append(self, key: str, message: dict[str, Any]) -> dict[str, Any]:
        """Assign the next per-session seq, buffer a copy, and return it.

        The returned dict is the stored copy with its ``seq`` set, so the caller can
        send exactly what was buffered (replay emits identical frames).
        """
        seq = self._seq.get(key, 0) + 1
        self._seq[key] = seq
        stored = dict(message)
        stored["seq"] = seq
        buf = self._buffers.get(key)
        if buf is None:
            buf = deque(maxlen=self._max)
            self._buffers[key] = buf
        buf.append(stored)  # deque(maxlen=...) auto-evicts the oldest
        return dict(stored)

    def get_replay(self, key: str, last_seq: int) -> ReplayResult:
        """Return the retained messages after ``last_seq`` and whether a gap exists."""
        buf = self._buffers.get(key)
        if not buf:
            return ReplayResult(messages=[], gap=False, oldest_seq=None, latest_seq=None)
        oldest = buf[0]["seq"]
        latest = buf[-1]["seq"]
        # A gap exists when the next message the client needs (last_seq + 1) has
        # already been evicted, i.e. last_seq + 1 < oldest  <=>  last_seq < oldest - 1.
        gap = last_seq < oldest - 1
        messages = [dict(m) for m in buf if m["seq"] > last_seq]
        return ReplayResult(
            messages=messages,
            gap=gap,
            oldest_seq=oldest,
            latest_seq=latest,
        )

    def latest_seq(self, key: str) -> int:
        """Highest seq assigned for a key (0 if none) — for diagnostics/tests."""
        return self._seq.get(key, 0)

    def clear(self, key: str) -> None:
        """Drop a session's buffer and reset its counter (e.g. on session end)."""
        self._buffers.pop(key, None)
        self._seq.pop(key, None)
