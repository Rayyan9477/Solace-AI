"""Unit tests for OpenAI client module."""
from __future__ import annotations
import pytest
from solace_ml.openai import (
    OpenAISettings, OpenAIClient, ToolChoice, create_openai_client,
)
from solace_ml.llm_client import (
    LLMProvider, MessageRole, FinishReason, Message, ToolDefinition,
    ToolParameter, ToolCall, TokenUsage, LLMResponse,
)


class TestOpenAISettings:
    """Tests for OpenAISettings model."""

    def test_default_settings(self):
        settings = OpenAISettings()
        assert settings.model == "gpt-4o"
        assert settings.max_tokens == 4096
        assert settings.frequency_penalty == 0.0

    def test_custom_settings(self):
        settings = OpenAISettings(model="gpt-4o-mini", max_tokens=8192)
        assert settings.model == "gpt-4o-mini"
        assert settings.max_tokens == 8192

    def test_organization_id(self):
        settings = OpenAISettings(organization_id="org-123")
        assert settings.organization_id == "org-123"

    def test_response_format(self):
        settings = OpenAISettings(response_format="json_object")
        assert settings.response_format == "json_object"

    def test_seed(self):
        settings = OpenAISettings(seed=12345)
        assert settings.seed == 12345


class TestToolChoice:
    """Tests for ToolChoice constants."""

    def test_tool_choice_values(self):
        assert ToolChoice.AUTO == "auto"
        assert ToolChoice.NONE == "none"
        assert ToolChoice.REQUIRED == "required"


class TestOpenAIClient:
    """Tests for OpenAIClient class."""

    @pytest.fixture
    def client(self):
        return OpenAIClient()

    def test_client_creation(self, client):
        assert client.provider == LLMProvider.OPENAI
        assert client.model == "gpt-4o"

    def test_client_with_settings(self):
        settings = OpenAISettings(model="gpt-4o-mini")
        client = OpenAIClient(settings)
        assert client.model == "gpt-4o-mini"

    def test_build_headers(self, client):
        headers = client._build_headers()
        assert "Authorization" in headers
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_with_org(self):
        settings = OpenAISettings(organization_id="org-123")
        client = OpenAIClient(settings)
        headers = client._build_headers()
        assert headers["OpenAI-Organization"] == "org-123"

    def test_build_payload_basic(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        payload = client._build_payload(messages, None, None)
        assert payload["model"] == "gpt-4o"
        assert len(payload["messages"]) == 1

    def test_build_payload_with_system(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        payload = client._build_payload(messages, None, "You are helpful.")
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."

    def test_build_payload_with_tools(self, client):
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]
        tools = [ToolDefinition(name="get_weather", description="Get weather")]
        payload = client._build_payload(messages, tools, None)
        assert "tools" in payload
        assert payload["tool_choice"] == "auto"

    def test_build_payload_with_stream(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        payload = client._build_payload(messages, None, None, stream=True)
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True

    def test_build_payload_with_response_format(self):
        settings = OpenAISettings(response_format="json_object")
        client = OpenAIClient(settings)
        messages = [Message(role=MessageRole.USER, content="Return JSON")]
        payload = client._build_payload(messages, None, None)
        assert payload["response_format"]["type"] == "json_object"

    def test_convert_messages(self, client):
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        converted = client._convert_messages(messages, None)
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"

    def test_convert_messages_with_system(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        converted = client._convert_messages(messages, "You are helpful.")
        assert len(converted) == 2
        assert converted[0]["role"] == "system"

    def test_convert_messages_with_tool_calls(self, client):
        tool_calls = [ToolCall(id="tc_1", name="get_weather", arguments={"city": "NYC"})]
        messages = [Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls)]
        converted = client._convert_messages(messages, None)
        assert "tool_calls" in converted[0]
        assert converted[0]["tool_calls"][0]["type"] == "function"

    def test_convert_messages_with_tool_result(self, client):
        messages = [Message(role=MessageRole.TOOL, content='{"temp": 72}', tool_call_id="tc_1")]
        converted = client._convert_messages(messages, None)
        assert converted[0]["role"] == "tool"
        assert converted[0]["tool_call_id"] == "tc_1"

    def test_convert_tool(self, client):
        params = [ToolParameter(name="city", type="string", description="City")]
        tool = ToolDefinition(name="get_weather", description="Get weather", parameters=params)
        converted = client._convert_tool(tool)
        assert converted["type"] == "function"
        assert converted["function"]["name"] == "get_weather"

    def test_map_finish_reason(self, client):
        assert client._map_finish_reason("stop") == FinishReason.STOP
        assert client._map_finish_reason("length") == FinishReason.LENGTH
        assert client._map_finish_reason("tool_calls") == FinishReason.TOOL_CALL
        assert client._map_finish_reason("content_filter") == FinishReason.CONTENT_FILTER
        assert client._map_finish_reason(None) == FinishReason.STOP

    def test_parse_response(self, client):
        data = {
            "choices": [{"message": {"content": "Hello!", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o",
            "id": "chatcmpl-123"
        }
        response = client._parse_response(data, 100.0, "req_123")
        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.total_tokens == 15
        assert response.request_id == "chatcmpl-123"

    def test_parse_response_with_tool_calls(self, client):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }
        response = client._parse_response(data, 150.0, "req_456")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments["city"] == "NYC"
        assert response.finish_reason == FinishReason.TOOL_CALL

    def test_parse_stream_chunk_content(self, client):
        data = {"choices": [{"delta": {"content": "Hello"}}]}
        chunk = client._parse_stream_chunk(data, {})
        assert chunk.content == "Hello"

    def test_parse_stream_chunk_finish(self, client):
        data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        chunk = client._parse_stream_chunk(data, {})
        assert chunk.is_final is True
        assert chunk.finish_reason == FinishReason.STOP

    def test_parse_stream_chunk_tool_call(self, client):
        accumulated = {}
        data1 = {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "tc_1", "function": {"name": "get_weather", "arguments": '{"ci'}}
        ]}}]}
        client._parse_stream_chunk(data1, accumulated)
        data2 = {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ty": "NYC"}'}}
        ]}}]}
        client._parse_stream_chunk(data2, accumulated)
        data3 = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        chunk = client._parse_stream_chunk(data3, accumulated)
        assert chunk.is_final is True
        assert len(chunk.tool_calls) == 1
        assert chunk.tool_calls[0].arguments["city"] == "NYC"

    def test_parse_stream_chunk_with_usage(self, client):
        data = {"usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}}
        chunk = client._parse_stream_chunk(data, {})
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 150


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_openai_client(self):
        client = create_openai_client()
        assert isinstance(client, OpenAIClient)
        assert client.provider == LLMProvider.OPENAI

    def test_create_openai_client_with_settings(self):
        settings = OpenAISettings(model="gpt-4o-mini")
        client = create_openai_client(settings)
        assert client.model == "gpt-4o-mini"
