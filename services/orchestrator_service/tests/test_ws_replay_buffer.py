"""REV-23: server-side replay of the in-flight WS message tail on reconnect.

Core unit tests for the bounded per-session ring buffer that backs replay.
Each outbound message is tagged with a monotonic per-session sequence number;
on reconnect a client presents its last-seen seq and the buffer replays only
the messages it missed (seq > last_seq), exactly once, in order. When the
requested cursor is older than the retained window a gap is signalled (resync)
rather than silently dropping messages.
"""
from __future__ import annotations

from services.orchestrator_service.src.replay_buffer import InMemoryReplayBufferStore


def test_append_assigns_monotonic_seq_starting_at_one() -> None:
    store = InMemoryReplayBufferStore(max_per_session=50)
    s1 = store.append("k", {"type": "response", "response": "a"})
    s2 = store.append("k", {"type": "response", "response": "b"})
    s3 = store.append("k", {"type": "response", "response": "c"})
    assert [s1["seq"], s2["seq"], s3["seq"]] == [1, 2, 3]
    # returned message preserves original fields plus seq
    assert s1["response"] == "a"


def test_replay_returns_only_messages_after_cursor_in_order() -> None:
    store = InMemoryReplayBufferStore(max_per_session=50)
    for i in range(1, 6):  # seqs 1..5
        store.append("k", {"type": "response", "response": f"m{i}"})
    result = store.get_replay("k", last_seq=2)
    seqs = [m["seq"] for m in result.messages]
    assert seqs == [3, 4, 5]
    assert result.gap is False
    # exactly once, ordered, no dupes
    assert [m["response"] for m in result.messages] == ["m3", "m4", "m5"]


def test_replay_caught_up_cursor_returns_nothing_no_gap() -> None:
    store = InMemoryReplayBufferStore(max_per_session=50)
    for i in range(1, 4):
        store.append("k", {"type": "response"})
    result = store.get_replay("k", last_seq=3)
    assert result.messages == []
    assert result.gap is False


def test_too_old_cursor_signals_gap() -> None:
    store = InMemoryReplayBufferStore(max_per_session=3)
    for i in range(1, 6):  # seqs 1..5, buffer keeps only 3,4,5
        store.append("k", {"type": "response", "response": f"m{i}"})
    result = store.get_replay("k", last_seq=1)  # msg 2 was evicted -> gap
    assert result.gap is True
    assert result.oldest_seq == 3
    # still surfaces what survives, in order
    assert [m["seq"] for m in result.messages] == [3, 4, 5]


def test_boundary_cursor_at_oldest_minus_one_is_not_a_gap() -> None:
    store = InMemoryReplayBufferStore(max_per_session=3)
    for i in range(1, 6):  # buffer keeps 3,4,5 ; oldest=3
        store.append("k", {"type": "response"})
    result = store.get_replay("k", last_seq=2)  # 2 == oldest-1, next (3) present
    assert result.gap is False
    assert [m["seq"] for m in result.messages] == [3, 4, 5]


def test_eviction_bounds_buffer_size() -> None:
    store = InMemoryReplayBufferStore(max_per_session=3)
    for i in range(1, 11):
        store.append("k", {"type": "response"})
    result = store.get_replay("k", last_seq=0)
    assert len(result.messages) == 3
    assert [m["seq"] for m in result.messages] == [8, 9, 10]


def test_unknown_session_returns_empty_no_gap() -> None:
    store = InMemoryReplayBufferStore(max_per_session=3)
    result = store.get_replay("never-seen", last_seq=5)
    assert result.messages == []
    assert result.gap is False
    assert result.oldest_seq is None


def test_sessions_are_isolated() -> None:
    store = InMemoryReplayBufferStore(max_per_session=50)
    store.append("a", {"type": "response", "response": "a1"})
    store.append("b", {"type": "response", "response": "b1"})
    store.append("a", {"type": "response", "response": "a2"})
    ra = store.get_replay("a", last_seq=0)
    rb = store.get_replay("b", last_seq=0)
    assert [m["seq"] for m in ra.messages] == [1, 2]
    assert [m["seq"] for m in rb.messages] == [1]
    assert rb.messages[0]["response"] == "b1"


def test_replay_returns_copies_not_live_refs() -> None:
    store = InMemoryReplayBufferStore(max_per_session=50)
    store.append("k", {"type": "response", "response": "a"})
    result = store.get_replay("k", last_seq=0)
    result.messages[0]["response"] = "mutated"
    again = store.get_replay("k", last_seq=0)
    assert again.messages[0]["response"] == "a"
