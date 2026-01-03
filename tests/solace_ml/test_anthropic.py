"""Unit tests for Anthropic client module."""
from __future__ import annotations
import pytest
from solace_ml.anthropic import (
    AnthropicSettings, AnthropicClient, create_anthropic_client,
)
from solace_ml.llm_client import (
    LLMProvider, MessageRole, FinishReason, Message, ToolDefinition,
    ToolParameter, ToolCall, TokenUsage, LLMResponse,
)


class TestAnthropicSettings:
    """Tests for AnthropicSettings model."""

    def test_default_settings(self):
        settings = AnthropicSettings()
        assert settings.model == "claude-sonnet-4-20250514"
        assert settings.max_tokens == 4096
        assert settings.thinking_budget is None

    def test_custom_settings(self):
        settings = AnthropicSettings(model="claude-3-opus-20240229", max_tokens=8192)
        assert settings.model == "claude-3-opus-20240229"
        assert settings.max_tokens == 8192

    def test_thinking_budget(self):
        settings = AnthropicSettings(thinking_budget=10000)
        assert settings.thinking_budget == 10000

    def test_beta_features(self):
        settings = AnthropicSettings(beta_features=["prompt-caching-2024-07-31"])
        assert "prompt-caching-2024-07-31" in settings.beta_features


class TestAnthropicClient:
    """Tests for AnthropicClient class."""

    @pytest.fixture
    def client(self):
        return AnthropicClient()

    def test_client_creation(self, client):
        assert client.provider == LLMProvider.ANTHROPIC
        assert client.model == "claude-sonnet-4-20250514"

    def test_client_with_settings(self):
        settings = AnthropicSettings(model="claude-3-opus-20240229")
        client = AnthropicClient(settings)
        assert client.model == "claude-3-opus-20240229"

    def test_build_headers(self, client):
        headers = client._build_headers()
        assert "x-api-key" in headers
        assert "anthropic-version" in headers
        assert headers["content-type"] == "application/json"

    def test_build_payload_basic(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        payload = client._build_payload(messages, None, None)
        assert payload["model"] == client.model
        assert payload["max_tokens"] == 4096
        assert len(payload["messages"]) == 1

    def test_build_payload_with_system(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        payload = client._build_payload(messages, None, "You are helpful.")
        assert payload["system"] == "You are helpful."

    def test_build_payload_with_tools(self, client):
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]
        tools = [ToolDefinition(name="get_weather", description="Get weather")]
        payload = client._build_payload(messages, tools, None)
        assert "tools" in payload
        assert len(payload["tools"]) == 1

    def test_build_payload_with_stream(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        payload = client._build_payload(messages, None, None, stream=True)
        assert payload["stream"] is True

    def test_convert_messages(self, client):
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        converted = client._convert_messages(messages)
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"

    def test_convert_messages_with_tool_calls(self, client):
        tool_calls = [ToolCall(id="tc_1", name="get_weather", arguments={"city": "NYC"})]
        messages = [Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls)]
        converted = client._convert_messages(messages)
        assert converted[0]["content"][0]["type"] == "tool_use"

    def test_convert_messages_with_tool_result(self, client):
        messages = [Message(role=MessageRole.TOOL, content='{"temp": 72}', tool_call_id="tc_1")]
        converted = client._convert_messages(messages)
        assert converted[0]["role"] == "user"
        assert converted[0]["content"][0]["type"] == "tool_result"

    def test_convert_tool(self, client):
        params = [ToolParameter(name="city", type="string", description="City")]
        tool = ToolDefinition(name="get_weather", description="Get weather", parameters=params)
        converted = client._convert_tool(tool)
        assert converted["name"] == "get_weather"
        assert "input_schema" in converted

    def test_map_stop_reason(self, client):
        assert client._map_stop_reason("end_turn") == FinishReason.STOP
        assert client._map_stop_reason("stop_sequence") == FinishReason.STOP
        assert client._map_stop_reason("max_tokens") == FinishReason.LENGTH
        assert client._map_stop_reason("tool_use") == FinishReason.TOOL_CALL
        assert client._map_stop_reason(None) == FinishReason.STOP

    def test_parse_response(self, client):
        data = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
            "model": "claude-3-opus"
        }
        response = client._parse_response(data, 100.0, "req_123")
        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.latency_ms == 100.0

    def test_parse_response_with_tool_use(self, client):
        data = {
            "content": [
                {"type": "tool_use", "id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}}
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "stop_reason": "tool_use"
        }
        response = client._parse_response(data, 150.0, "req_456")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.finish_reason == FinishReason.TOOL_CALL

    def test_parse_stream_event_text_delta(self, client):
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}
        chunk = client._parse_stream_event(event)
        assert chunk is not None
        assert chunk.content == "Hello"

    def test_parse_stream_event_message_delta(self, client):
        event = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 50}}
        chunk = client._parse_stream_event(event)
        assert chunk is not None
        assert chunk.is_final is True
        assert chunk.usage.completion_tokens == 50

    def test_parse_stream_event_tool_use_start(self, client):
        event = {"type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "tc_1", "name": "get_weather"}}
        chunk = client._parse_stream_event(event)
        assert chunk is not None
        assert len(chunk.tool_calls) == 1


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_anthropic_client(self):
        client = create_anthropic_client()
        assert isinstance(client, AnthropicClient)
        assert client.provider == LLMProvider.ANTHROPIC

    def test_create_anthropic_client_with_settings(self):
        settings = AnthropicSettings(model="claude-3-haiku-20240307")
        client = create_anthropic_client(settings)
        assert client.model == "claude-3-haiku-20240307"
