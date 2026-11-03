"""Supports MVP-357 (WS-05.10): memory publishes the canonical therapy__memory /context examples (IF-11 provider shapes)."""
from __future__ import annotations

from uuid import UUID

from services.memory_service.src.schemas import (
    THERAPY_CONTEXT_REQUEST_EXAMPLE,
    THERAPY_CONTEXT_RESPONSE_EXAMPLE,
    ContextAssemblyRequest,
    ContextAssemblyResponse,
)

USER = UUID("00000000-0000-4000-a000-000000000001")


def test_mvp_357_request_example_validates_and_names_the_user() -> None:
    request = ContextAssemblyRequest.model_validate(THERAPY_CONTEXT_REQUEST_EXAMPLE)
    assert request.user_id == USER and request.session_id == UUID("00000000-0000-4000-b000-000000000001")
    assert set(THERAPY_CONTEXT_REQUEST_EXAMPLE) == {"user_id", "session_id", "current_message"}
    schema = ContextAssemblyRequest.model_json_schema()
    assert schema["required"] == ["user_id"] and schema["examples"] == [THERAPY_CONTEXT_REQUEST_EXAMPLE]


def test_mvp_357_response_example_validates_and_carries_the_contract_fields() -> None:
    response = ContextAssemblyResponse.model_validate(THERAPY_CONTEXT_RESPONSE_EXAMPLE)
    assert response.user_id == USER
    assert set(THERAPY_CONTEXT_RESPONSE_EXAMPLE) == {
        "user_id", "assembled_context", "total_tokens", "sources_used", "assembly_time_ms", "retrieval_count"}
    assert ContextAssemblyResponse.model_json_schema()["examples"] == [THERAPY_CONTEXT_RESPONSE_EXAMPLE]


def test_mvp_357_examples_are_synthetic() -> None:
    assert THERAPY_CONTEXT_REQUEST_EXAMPLE["current_message"] == "example message"
    assert THERAPY_CONTEXT_RESPONSE_EXAMPLE["assembled_context"] == "example context"
