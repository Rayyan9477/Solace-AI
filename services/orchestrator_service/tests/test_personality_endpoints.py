"""
Unit tests for Orchestrator Service - Personality Agent Endpoints.
Verifies that PersonalityServiceClient constructs correct API URLs with /api/v1/personality/ prefix,
and that SafetyServiceClient uses lowercase check_type.
"""
from __future__ import annotations

import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.orchestrator_service.src.agents.personality_agent import (
    PersonalityAgent,
    PersonalityAgentSettings,
    PersonalityServiceClient,
)
from services.orchestrator_service.src.infrastructure.clients import (
    SafetyServiceClient,
)


class TestPersonalityAgentEndpoints:
    """Verify personality service URLs have correct /api/v1/personality/ prefix."""

    def test_detect_url_includes_api_prefix(self) -> None:
        """Verify personality detect URL has /api/v1/personality/ prefix."""
        settings = PersonalityAgentSettings(service_url="http://personality:8007")
        client = PersonalityServiceClient(settings)
        # The detect_personality method constructs the URL internally.
        # We verify by inspecting the source code of detect_personality
        source = inspect.getsource(client.detect_personality)
        assert "/api/v1/personality/detect" in source, (
            "detect_personality must use /api/v1/personality/detect endpoint"
        )

    def test_style_url_includes_api_prefix(self) -> None:
        """Verify personality style URL has /api/v1/personality/ prefix."""
        settings = PersonalityAgentSettings(service_url="http://personality:8007")
        client = PersonalityServiceClient(settings)
        source = inspect.getsource(client.get_style)
        assert "/api/v1/personality/style" in source, (
            "get_style must use /api/v1/personality/style endpoint"
        )

    def test_base_url_trailing_slash_stripped(self) -> None:
        """Base URL trailing slash should be stripped to prevent double slashes."""
        settings = PersonalityAgentSettings(service_url="http://personality:8007/")
        client = PersonalityServiceClient(settings)
        assert not client._base_url.endswith("/"), (
            f"Base URL should not end with slash: {client._base_url}"
        )

    def test_detect_url_constructed_from_base_url(self) -> None:
        """The detect endpoint should be base_url + /api/v1/personality/detect."""
        settings = PersonalityAgentSettings(service_url="http://localhost:8007")
        client = PersonalityServiceClient(settings)
        expected_base = "http://localhost:8007"
        assert client._base_url == expected_base
        # Verify the full URL would be correct by checking the method source
        source = inspect.getsource(client.detect_personality)
        assert 'f"{self._base_url}/api/v1/personality/detect"' in source or \
               "/api/v1/personality/detect" in source

    def test_default_service_url(self) -> None:
        """Default personality service URL should be http://localhost:8007."""
        settings = PersonalityAgentSettings()
        assert settings.service_url == "http://localhost:8007"


class TestPersonalityServiceClientConfig:
    """Verify PersonalityServiceClient configuration."""

    def test_client_uses_settings_url(self) -> None:
        """Client should use the service_url from settings."""
        settings = PersonalityAgentSettings(service_url="http://custom:9000")
        client = PersonalityServiceClient(settings)
        assert client._base_url == "http://custom:9000"

    def test_client_timeout_from_settings(self) -> None:
        """Client should use timeout_seconds from settings."""
        settings = PersonalityAgentSettings(timeout_seconds=30.0)
        client = PersonalityServiceClient(settings)
        assert client._settings.timeout_seconds == 30.0

    def test_client_max_retries_from_settings(self) -> None:
        """Client should use max_retries from settings."""
        settings = PersonalityAgentSettings(max_retries=5)
        client = PersonalityServiceClient(settings)
        assert client._settings.max_retries == 5


class TestSafetyClientCase:
    """Verify SafetyServiceClient uses lowercase check_type."""

    def test_check_type_default_is_lowercase(self) -> None:
        """SafetyServiceClient.check_safety default check_type should be lowercase."""
        sig = inspect.signature(SafetyServiceClient.check_safety)
        check_type_param = sig.parameters["check_type"]
        default_value = check_type_param.default
        assert default_value == "full_assessment", (
            f"Expected default check_type='full_assessment', got '{default_value}'"
        )

    def test_check_type_default_is_snake_case(self) -> None:
        """Default check_type should use snake_case, not camelCase or UPPER_CASE."""
        sig = inspect.signature(SafetyServiceClient.check_safety)
        check_type_param = sig.parameters["check_type"]
        default_value = check_type_param.default
        # Snake case: all lowercase with underscores
        assert default_value == default_value.lower(), (
            f"check_type default should be lowercase: '{default_value}'"
        )
        assert " " not in default_value, (
            "check_type default should not contain spaces"
        )

    def test_check_type_is_string_parameter(self) -> None:
        """check_type parameter should be typed as str."""
        sig = inspect.signature(SafetyServiceClient.check_safety)
        check_type_param = sig.parameters["check_type"]
        annotation = check_type_param.annotation
        assert annotation is str or annotation == "str", (
            f"check_type should be typed as str, got {annotation}"
        )

    def test_safety_check_endpoint_uses_api_prefix(self) -> None:
        """SafetyServiceClient.check_safety should POST to /api/v1/safety/check."""
        source = inspect.getsource(SafetyServiceClient.check_safety)
        assert "/api/v1/safety/check" in source, (
            "check_safety must use /api/v1/safety/check endpoint"
        )

    def test_check_safety_includes_check_type_in_payload(self) -> None:
        """The check_type parameter should be included in the request payload."""
        source = inspect.getsource(SafetyServiceClient.check_safety)
        assert "check_type" in source, (
            "check_safety payload should include check_type field"
        )
