"""Shared test fixtures for safety service tests."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from services.safety_service.src.domain.escalation import NotificationServiceClient


@pytest.fixture(autouse=True)
def _mock_notification_http():
    """Mock HTTP client for notification service to avoid real network calls.

    NotificationServiceClient makes real HTTP calls via httpx to the
    notification microservice. In tests, we mock _ensure_client to return
    a mock async client that always returns successful responses.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "successful_deliveries": 1,
        "request_id": "test-mock-123",
    }
    mock_response.text = '{"successful_deliveries": 1}'

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()

    original = NotificationServiceClient._ensure_client

    async def _mock_ensure_client(self):
        return mock_client

    NotificationServiceClient._ensure_client = _mock_ensure_client
    yield mock_client
    NotificationServiceClient._ensure_client = original
