"""
Unit tests for Shared LLM Client.
Tests Portkey AI Gateway integration, fallbacks, and response generation.
"""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from services.shared.infrastructure import (
    LLMClientSettings,
    UnifiedLLMClient,
    LLM_SYSTEM_PROMPTS,
    get_llm_prompt,
)
from services.therapy_service.src.infrastructure import (
    get_therapy_prompt,
    THERAPY_MODALITY_PROMPTS,
)


class TestLLMClientSettings:
    """Tests for LLMClientSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings are properly initialized."""
        settings = LLMClientSettings()
        assert settings.portkey_gateway_url == "https://api.portkey.ai/v1"
        assert settings.primary_provider == "anthropic"
        assert settings.primary_model == "claude-sonnet-4-20250514"
        assert settings.fallback_provider == "openai"
        assert settings.fallback_model == "gpt-4o"
        assert settings.max_tokens == 1024
        assert settings.temperature == 0.7
        assert settings.retry_attempts == 3
        assert settings.enable_caching is True
        assert settings.cache_mode == "semantic"
        assert settings.enable_fallback is True
        assert settings.enable_load_balancing is False

    def test_custom_settings(self) -> None:
        """Test custom settings override defaults."""
        settings = LLMClientSettings(
            primary_provider="openai",
            primary_model="gpt-4o",
            max_tokens=2048,
            temperature=0.5,
            enable_caching=False,
            enable_load_balancing=True,
            load_balance_weight_primary=0.6,
        )
        assert settings.primary_provider == "openai"
        assert settings.primary_model == "gpt-4o"
        assert settings.max_tokens == 2048
        assert settings.temperature == 0.5
        assert settings.enable_caching is False
        assert settings.enable_load_balancing is True
        assert settings.load_balance_weight_primary == 0.6


class TestUnifiedLLMClient:
    """Tests for UnifiedLLMClient functionality."""

    def test_client_initialization_state(self) -> None:
        """Test client initializes with correct state."""
        client = UnifiedLLMClient()
        assert client._initialized is False
        assert client._client is None
        assert client._request_count == 0
        assert client._error_count == 0
        assert client._cache_hits == 0

    @pytest.mark.asyncio
    async def test_initialize_without_portkey(self) -> None:
        """Test initialization handles missing portkey_ai gracefully."""
        client = UnifiedLLMClient()
        with patch.dict("sys.modules", {"portkey_ai": None}):
            with patch("services.shared.infrastructure.llm_client.logger"):
                await client.initialize()
        assert client._initialized is True
        assert client._client is None

    @pytest.mark.asyncio
    async def test_initialize_with_mock_portkey(self) -> None:
        """Test initialization with mocked Portkey."""
        mock_portkey_class = MagicMock()
        mock_client = MagicMock()
        mock_portkey_class.return_value = mock_client

        client = UnifiedLLMClient()
        with patch("services.shared.infrastructure.llm_client.logger"):
            with patch.object(
                client, "_build_portkey_config", return_value={"targets": []}
            ):
                import sys
                mock_module = MagicMock()
                mock_module.AsyncPortkey = mock_portkey_class
                sys.modules["portkey_ai"] = mock_module
                try:
                    await client.initialize()
                    assert client._initialized is True
                finally:
                    del sys.modules["portkey_ai"]

    def test_build_portkey_config_fallback_mode(self) -> None:
        """Test Portkey config generation with fallback strategy."""
        settings = LLMClientSettings(
            enable_fallback=True,
            enable_load_balancing=False,
            anthropic_api_key="test-anthropic-key",
            openai_api_key="test-openai-key",
        )
        client = UnifiedLLMClient(settings)
        config = client._build_portkey_config()

        assert "strategy" in config
        assert config["strategy"]["mode"] == "fallback"
        assert "targets" in config
        assert len(config["targets"]) == 2
        assert config["targets"][0]["provider"] == "anthropic"
        assert config["targets"][1]["provider"] == "openai"
        assert "retry" in config
        assert config["retry"]["attempts"] == 3

    def test_build_portkey_config_loadbalance_mode(self) -> None:
        """Test Portkey config generation with load balancing strategy."""
        settings = LLMClientSettings(
            enable_fallback=True,
            enable_load_balancing=True,
            load_balance_weight_primary=0.7,
        )
        client = UnifiedLLMClient(settings)
        config = client._build_portkey_config()

        assert config["strategy"]["mode"] == "loadbalance"
        assert config["targets"][0]["weight"] == 0.7
        assert abs(config["targets"][1]["weight"] - 0.3) < 0.001  # Float comparison

    def test_build_portkey_config_with_caching(self) -> None:
        """Test Portkey config includes caching when enabled."""
        settings = LLMClientSettings(
            enable_caching=True,
            cache_mode="semantic",
        )
        client = UnifiedLLMClient(settings)
        config = client._build_portkey_config()

        assert "cache" in config
        assert config["cache"]["mode"] == "semantic"

    def test_build_portkey_config_without_caching(self) -> None:
        """Test Portkey config excludes caching when disabled."""
        settings = LLMClientSettings(enable_caching=False)
        client = UnifiedLLMClient(settings)
        config = client._build_portkey_config()

        assert "cache" not in config

    def test_build_messages_basic(self) -> None:
        """Test message building with basic input."""
        client = UnifiedLLMClient()
        messages = client._build_messages(
            system_prompt="You are helpful.",
            user_message="Hello",
            conversation_history=None,
            context=None,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_build_messages_with_history(self) -> None:
        """Test message building with conversation history."""
        client = UnifiedLLMClient()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        messages = client._build_messages(
            system_prompt="System",
            user_message="How are you?",
            conversation_history=history,
            context=None,
        )

        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"
        assert messages[2]["role"] == "assistant"

    def test_build_messages_with_context(self) -> None:
        """Test message building with context."""
        client = UnifiedLLMClient()
        messages = client._build_messages(
            system_prompt="Base prompt",
            user_message="User input",
            conversation_history=None,
            context="Additional context here",
        )

        assert "Additional context here" in messages[0]["content"]

    def test_build_messages_limits_history(self) -> None:
        """Test message building limits conversation history to 10 messages."""
        client = UnifiedLLMClient()
        history = [{"role": "user", "content": f"Message {i}"} for i in range(15)]
        messages = client._build_messages(
            system_prompt="System",
            user_message="Final",
            conversation_history=history,
            context=None,
        )

        # 1 system + 10 history + 1 user = 12
        assert len(messages) == 12

    @pytest.mark.asyncio
    async def test_generate_no_client(self) -> None:
        """Test generation returns empty when client not initialized."""
        client = UnifiedLLMClient()
        client._initialized = True
        client._client = None

        response = await client.generate(
            system_prompt="Test",
            user_message="Hello",
            service_name="test_service",
        )

        assert response == ""

    @pytest.mark.asyncio
    async def test_generate_with_mock_client(self) -> None:
        """Test generation with mocked LLM client."""
        client = UnifiedLLMClient()
        client._initialized = True

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated response"

        mock_portkey = MagicMock()
        mock_portkey.chat.completions.create = AsyncMock(return_value=mock_response)
        client._client = mock_portkey

        response = await client.generate(
            system_prompt="You are helpful",
            user_message="Hello",
            service_name="test_service",
        )

        assert response == "Generated response"
        assert client._request_count == 1

    @pytest.mark.asyncio
    async def test_generate_error_handling(self) -> None:
        """Test generation handles errors gracefully."""
        client = UnifiedLLMClient()
        client._initialized = True

        mock_portkey = MagicMock()
        mock_portkey.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        client._client = mock_portkey

        response = await client.generate(
            system_prompt="Test",
            user_message="Hello",
            service_name="test_service",
        )

        assert response == ""
        assert client._error_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_fallback(self) -> None:
        """Test generate_with_fallback returns fallback on failure."""
        client = UnifiedLLMClient()
        client._initialized = True
        client._client = None

        response = await client.generate_with_fallback(
            system_prompt="Test",
            user_message="Hello",
            fallback_response="Fallback response here",
            service_name="test_service",
        )

        assert response == "Fallback response here"

    def test_extract_response_text_string(self) -> None:
        """Test extracting text from string content response."""
        client = UnifiedLLMClient()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response text"

        result = client._extract_response_text(mock_response)
        assert result == "Response text"

    def test_extract_response_text_list(self) -> None:
        """Test extracting text from list content response."""
        client = UnifiedLLMClient()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = [
            {"text": "Part 1"},
            {"text": "Part 2"},
        ]

        result = client._extract_response_text(mock_response)
        assert result == "Part 1 Part 2"

    def test_extract_response_text_empty(self) -> None:
        """Test extracting text from empty response."""
        client = UnifiedLLMClient()
        mock_response = MagicMock()
        mock_response.choices = []

        result = client._extract_response_text(mock_response)
        assert result == ""

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test client shutdown."""
        client = UnifiedLLMClient()
        client._initialized = True
        client._client = MagicMock()
        client._request_count = 10
        client._error_count = 2

        with patch("services.shared.infrastructure.llm_client.logger"):
            await client.shutdown()

        assert client._initialized is False
        assert client._client is None

    def test_get_statistics(self) -> None:
        """Test getting client statistics."""
        settings = LLMClientSettings(
            primary_provider="anthropic",
            primary_model="claude-sonnet-4-20250514",
            enable_fallback=True,
            enable_caching=True,
            cache_mode="semantic",
        )
        client = UnifiedLLMClient(settings)
        client._initialized = True
        client._request_count = 100
        client._error_count = 5
        client._cache_hits = 20

        stats = client.get_statistics()

        assert stats["initialized"] is True
        assert stats["total_requests"] == 100
        assert stats["total_errors"] == 5
        assert stats["cache_hits"] == 20
        assert stats["error_rate"] == 0.05
        assert stats["primary_provider"] == "anthropic"
        assert stats["caching_enabled"] is True
        assert stats["cache_mode"] == "semantic"

    def test_is_available_property(self) -> None:
        """Test is_available property."""
        client = UnifiedLLMClient()
        assert client.is_available is False

        client._initialized = True
        assert client.is_available is False

        client._client = MagicMock()
        assert client.is_available is True


class TestLLMSystemPrompts:
    """Tests for shared LLM system prompts."""

    def test_all_prompts_exist(self) -> None:
        """Test all expected prompts exist."""
        expected = [
            "therapy_general", "therapy_cbt", "therapy_dbt", "therapy_act",
            "therapy_mi", "therapy_mindfulness", "therapy_crisis",
            "diagnosis_assessment", "diagnosis_screening",
            "safety_assessment", "safety_intervention",
            "memory_summarization",
        ]
        for prompt_key in expected:
            assert prompt_key in LLM_SYSTEM_PROMPTS
            assert len(LLM_SYSTEM_PROMPTS[prompt_key]) > 50

    def test_therapy_prompts_content(self) -> None:
        """Test therapy prompts contain key elements."""
        assert "Cognitive Behavioral" in LLM_SYSTEM_PROMPTS["therapy_cbt"]
        assert "Dialectical" in LLM_SYSTEM_PROMPTS["therapy_dbt"]
        assert "Acceptance and Commitment" in LLM_SYSTEM_PROMPTS["therapy_act"]
        assert "Motivational Interviewing" in LLM_SYSTEM_PROMPTS["therapy_mi"]

    def test_crisis_prompt_content(self) -> None:
        """Test crisis prompt contains safety elements."""
        prompt = LLM_SYSTEM_PROMPTS["therapy_crisis"]
        assert "SAFETY" in prompt
        assert "988" in prompt

    def test_diagnosis_prompts_content(self) -> None:
        """Test diagnosis prompts exist and have content."""
        assert "assessment" in LLM_SYSTEM_PROMPTS["diagnosis_assessment"].lower()
        assert "screening" in LLM_SYSTEM_PROMPTS["diagnosis_screening"].lower()

    def test_safety_prompts_content(self) -> None:
        """Test safety prompts exist and have content."""
        assert "safety" in LLM_SYSTEM_PROMPTS["safety_assessment"].lower()
        assert "SAFETY" in LLM_SYSTEM_PROMPTS["safety_intervention"]


class TestGetLLMPrompt:
    """Tests for get_llm_prompt function."""

    def test_get_existing_prompt(self) -> None:
        """Test getting existing prompt."""
        prompt = get_llm_prompt("therapy_cbt")
        assert "Cognitive Behavioral" in prompt

    def test_get_unknown_prompt_with_fallback(self) -> None:
        """Test unknown prompt returns fallback."""
        prompt = get_llm_prompt("unknown_key", fallback_key="therapy_general")
        assert "compassionate" in prompt.lower()

    def test_get_unknown_prompt_without_fallback(self) -> None:
        """Test unknown prompt with invalid fallback returns empty."""
        prompt = get_llm_prompt("unknown", "also_unknown")
        assert prompt == ""


class TestTherapyPromptHelper:
    """Tests for therapy-specific prompt helper."""

    def test_get_therapy_prompt_cbt(self) -> None:
        """Test getting CBT prompt."""
        prompt = get_therapy_prompt("cbt")
        assert "Cognitive Behavioral" in prompt

    def test_get_therapy_prompt_dbt(self) -> None:
        """Test getting DBT prompt."""
        prompt = get_therapy_prompt("dbt")
        assert "Dialectical" in prompt

    def test_get_therapy_prompt_act(self) -> None:
        """Test getting ACT prompt."""
        prompt = get_therapy_prompt("act")
        assert "Acceptance and Commitment" in prompt

    def test_get_therapy_prompt_mi(self) -> None:
        """Test getting MI prompt."""
        prompt = get_therapy_prompt("mi")
        assert "Motivational Interviewing" in prompt

    def test_get_therapy_prompt_mindfulness(self) -> None:
        """Test getting mindfulness prompt."""
        prompt = get_therapy_prompt("mindfulness")
        assert "mindfulness" in prompt.lower()

    def test_crisis_overrides_modality(self) -> None:
        """Test crisis flag overrides modality."""
        prompt = get_therapy_prompt("cbt", is_crisis=True)
        assert "SAFETY" in prompt
        assert "Cognitive Behavioral" not in prompt

    def test_unknown_modality_returns_general(self) -> None:
        """Test unknown modality returns general prompt."""
        prompt = get_therapy_prompt("unknown_modality")
        assert "compassionate" in prompt.lower()

    def test_case_insensitive(self) -> None:
        """Test modality matching is case insensitive."""
        prompt_lower = get_therapy_prompt("cbt")
        prompt_upper = get_therapy_prompt("CBT")
        assert prompt_lower == prompt_upper

    def test_all_modality_mappings_valid(self) -> None:
        """Test all modality mappings point to valid prompts."""
        for modality, prompt_key in THERAPY_MODALITY_PROMPTS.items():
            assert prompt_key in LLM_SYSTEM_PROMPTS
