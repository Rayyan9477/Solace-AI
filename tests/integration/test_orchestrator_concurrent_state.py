"""REV-31 / A.7: LangGraph concurrent-write crash on the parallel-agent path.

When the supervisor selects two or more agents that both write ``processing_phase``
(assessment + emotion both set it to ``PARALLEL_PROCESSING``), LangGraph fans them
out in one step. ``processing_phase`` had no reducer in ``OrchestratorState``, so the
second concurrent write raised:

    InvalidUpdateError: At key 'processing_phase': Can receive only one value per
    step. Use an Annotated key to handle multiple values.

This crashed the crisis path whenever the safety agent was selected alongside
other agents. The fix adds a reducer so the channel accepts concurrent writes.
These tests build the real ``OrchestratorState`` schema and fan two nodes into it
in parallel — reproducing the exact channel collision.
"""
from __future__ import annotations

import asyncio

from uuid import uuid4

from langgraph.graph import START, END, StateGraph

from services.orchestrator_service.src.langgraph.state_schema import (
    OrchestratorState,
    ProcessingPhase,
    create_initial_state,
    merge_final_response,
    merge_processing_phase,
)


def _phase_node_a(state: OrchestratorState) -> dict:
    return {"processing_phase": ProcessingPhase.PARALLEL_PROCESSING.value}


def _phase_node_b(state: OrchestratorState) -> dict:
    return {"processing_phase": ProcessingPhase.PARALLEL_PROCESSING.value}


def _build_parallel_phase_graph():
    builder = StateGraph(OrchestratorState)
    builder.add_node("a", _phase_node_a)
    builder.add_node("b", _phase_node_b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    return builder.compile()


class TestConcurrentProcessingPhase:
    def test_parallel_writes_to_processing_phase_do_not_crash(self) -> None:
        """Two nodes writing processing_phase in one step must not raise."""
        graph = _build_parallel_phase_graph()
        state = create_initial_state(uuid4(), uuid4(), "I feel anxious")
        result = graph.invoke(state)  # was: InvalidUpdateError
        assert result["processing_phase"] == ProcessingPhase.PARALLEL_PROCESSING.value


class TestProcessingPhaseReducer:
    """The reducer preserves last-write-wins for sequential updates."""

    def test_sequential_update_takes_latest(self) -> None:
        assert (
            merge_processing_phase(
                ProcessingPhase.PARALLEL_PROCESSING.value,
                ProcessingPhase.AGGREGATION.value,
            )
            == ProcessingPhase.AGGREGATION.value
        )

    def test_concurrent_same_value_is_stable(self) -> None:
        assert (
            merge_processing_phase(
                ProcessingPhase.PARALLEL_PROCESSING.value,
                ProcessingPhase.PARALLEL_PROCESSING.value,
            )
            == ProcessingPhase.PARALLEL_PROCESSING.value
        )

    def test_last_write_wins_preserves_sequential_progression(self) -> None:
        """The reducer must not alter normal sequential phase progression.

        A crisis session still advances crisis_handling -> completed exactly as
        before (safety state is carried by safety_flags, not this field).
        """
        assert (
            merge_processing_phase(
                ProcessingPhase.CRISIS_HANDLING.value,
                ProcessingPhase.COMPLETED.value,
            )
            == ProcessingPhase.COMPLETED.value
        )


class TestFinalResponseReducer:
    """final_response now has a defensive reducer (was a latent crisis-path crash)."""

    def test_non_empty_incoming_wins(self) -> None:
        assert merge_final_response("old", "new crisis response") == "new crisis response"

    def test_empty_incoming_does_not_clobber(self) -> None:
        assert merge_final_response("real response", "") == "real response"

    def test_parallel_writes_do_not_crash(self) -> None:
        builder = StateGraph(OrchestratorState)
        builder.add_node("a", lambda s: {"final_response": "from a"})
        builder.add_node("b", lambda s: {"final_response": "from b"})
        builder.add_edge(START, "a")
        builder.add_edge(START, "b")
        builder.add_edge("a", END)
        builder.add_edge("b", END)
        graph = builder.compile()
        result = graph.invoke(create_initial_state(uuid4(), uuid4(), "hi"))
        assert result["final_response"] in ("from a", "from b")
