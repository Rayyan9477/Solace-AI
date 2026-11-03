"""REV-31: downstream agent HTTP paths must include the service router prefix.

therapy_agent and diagnosis_agent built bare paths (``/sessions/.../message``,
``/assess``, ``/extract-symptoms``, ``/techniques``) while the services mount their
routers under ``/api/v1/therapy`` and ``/api/v1/diagnosis`` — so those calls 404'd
in production and the orchestrator silently degraded to a canned fallback. These
tests pin the corrected, prefixed URLs so the regression can't reappear.
"""
from __future__ import annotations

import httpx
import pytest

from services.orchestrator_service.src.agents.therapy_agent import (
    TherapyAgentSettings,
    TherapyServiceClient,
)
from services.orchestrator_service.src.agents.diagnosis_agent import (
    DiagnosisAgentSettings,
    DiagnosisServiceClient,
)


class _StopAfterRecord(Exception):
    """Raised by the fake client once it has captured the request URL."""


class _RecordingClient:
    """Fake httpx.AsyncClient that records the request URL then short-circuits
    (before any response parsing). Not an httpx error type, so the agents' retry
    loops don't catch it — it propagates out cleanly for the test to assert on."""

    last_url: str | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_RecordingClient":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def post(self, url, json=None):  # noqa: ANN001
        _RecordingClient.last_url = url
        raise _StopAfterRecord

    async def get(self, url, params=None):  # noqa: ANN001
        _RecordingClient.last_url = url
        raise _StopAfterRecord


@pytest.fixture
def record_urls(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _RecordingClient())
    _RecordingClient.last_url = None
    return _RecordingClient


@pytest.mark.asyncio
async def test_therapy_process_message_uses_api_v1_prefix(record_urls) -> None:
    client = TherapyServiceClient(TherapyAgentSettings(service_url="http://therapy:8006"))
    with pytest.raises(_StopAfterRecord):
        await client.process_message("s1", "u1", "hi", [])
    assert record_urls.last_url == "http://therapy:8006/api/v1/therapy/sessions/s1/message"


@pytest.mark.asyncio
async def test_therapy_get_techniques_uses_api_v1_prefix(record_urls) -> None:
    client = TherapyServiceClient(TherapyAgentSettings(service_url="http://therapy:8006"))
    with pytest.raises(_StopAfterRecord):
        await client.get_techniques()
    assert record_urls.last_url == "http://therapy:8006/api/v1/therapy/techniques"


@pytest.mark.asyncio
async def test_diagnosis_assess_uses_api_v1_prefix(record_urls) -> None:
    client = DiagnosisServiceClient(DiagnosisAgentSettings(service_url="http://diag:8004"))
    with pytest.raises(_StopAfterRecord):
        await client.assess("u1", "s1", "hi", [])
    assert record_urls.last_url == "http://diag:8004/api/v1/diagnosis/assess"


@pytest.mark.asyncio
async def test_diagnosis_extract_symptoms_uses_api_v1_prefix(record_urls) -> None:
    client = DiagnosisServiceClient(DiagnosisAgentSettings(service_url="http://diag:8004"))
    with pytest.raises(_StopAfterRecord):
        await client.extract_symptoms("hi", [])
    assert record_urls.last_url == "http://diag:8004/api/v1/diagnosis/extract-symptoms"
