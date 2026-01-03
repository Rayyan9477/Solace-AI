"""Unit tests for Gemini client module."""
from __future__ import annotations
import pytest
from solace_ml.gemini import (
    GeminiSettings, GeminiClient, create_gemini_client,
)
from solace_ml.llm_client import (
    LLMProvider, MessageRole, FinishReason, Message, ToolDefinition,
    ToolParameter, ToolCall, TokenUsage, LLMResponse,
)


class TestGeminiSettings:
    """Tests for GeminiSettings model."""

    def test_default_settings(self):
        settings = GeminiSettings()
        assert settings.model == "gemini-2.0-flash"
        assert settings.max_tokens == 8192

    def test_custom_settings(self):
        settings = GeminiSettings(model="gemini-1.5-pro", max_tokens=16384)
        assert settings.model == "gemini-1.5-pro"
        assert settings.max_tokens == 16384

    def test_top_k_setting(self):
        settings = GeminiSettings(top_k=40)
        assert settings.top_k == 40

    def test_safety_settings(self):
        safety = [{"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}]
        settings = GeminiSettings(safety_settings=safety)
        assert len(settings.safety_settings) == 1


class TestGeminiClient:
    """Tests for GeminiClient class."""

    @pytest.fixture
    def client(self):
        return GeminiClient()

    def test_client_creation(self, client):
        assert client.provider == LLMProvider.GEMINI
        assert client.model == "gemini-2.0-flash"

    def test_client_with_settings(self):
        settings = GeminiSettings(model="gemini-1.5-flash")
        client = GeminiClient(settings)
        assert client.model == "gemini-1.5-flash"

    def test_build_url(self, client):
        url = client._build_url(stream=False)
        assert "generateContent" in url
        assert "gemini-2.0-flash" in url

    def test_build_url_stream(self, client):
        url = client._build_url(stream=True)
        assert "streamGenerateContent" in url

    def test_build_headers(self, client):
        headers = client._build_headers()
        assert headers["Content-Type"] == "application/json"

    def test_build_payload_basic(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        payload = client._build_payload(messages, None, None)
        assert "contents" in payload
        assert "generationConfig" in payload

    def test_build_payload_with_system(self, client):
        messages = [Message(role=MessageRole.USER, content="Hello")]
        payload = client._build_payload(messages, None, "You are helpful.")
        assert "systemInstruction" in payload
        assert payload["systemInstruction"]["parts"][0]["text"] == "You are helpful."

    def test_build_payload_with_tools(self, client):
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]
        tools = [ToolDefinition(name="get_weather", description="Get weather")]
        payload = client._build_payload(messages, tools, None)
        assert "tools" in payload
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"

    def test_convert_messages(self, client):
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        converted = client._convert_messages(messages)
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "model"

    def test_convert_messages_with_tool_calls(self, client):
        tool_calls = [ToolCall(id="tc_1", name="get_weather", arguments={"city": "NYC"})]
        messages = [Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls)]
        converted = client._convert_messages(messages)
        assert "functionCall" in converted[0]["parts"][0]

    def test_convert_messages_with_tool_result(self, client):
        messages = [Message(role=MessageRole.TOOL, content='{"temp": 72}',
                           tool_call_id="tc_1", name="get_weather")]
        converted = client._convert_messages(messages)
        assert "functionResponse" in converted[0]["parts"][0]

    def test_convert_tool(self, client):
        params = [ToolParameter(name="city", type="string", description="City")]
        tool = ToolDefinition(name="get_weather", description="Get weather", parameters=params)
        converted = client._convert_tool(tool)
        assert converted["name"] == "get_weather"
        assert "parameters" in converted

    def test_map_finish_reason(self, client):
        assert client._map_finish_reason("STOP") == FinishReason.STOP
        assert client._map_finish_reason("MAX_TOKENS") == FinishReason.LENGTH
        assert client._map_finish_reason("SAFETY") == FinishReason.CONTENT_FILTER
        assert client._map_finish_reason(None) == FinishReason.STOP

    def test_parse_response(self, client):
        data = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello!"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        }
        response = client._parse_response(data, 100.0, "req_123")
        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.total_tokens == 15

    def test_parse_response_with_function_call(self, client):
        data = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {"name": "get_weather", "args": {"city": "NYC"}}
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10, "totalTokenCount": 30}
        }
        response = client._parse_response(data, 150.0, "req_456")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments["city"] == "NYC"

    def test_parse_stream_chunk_content(self, client):
        data = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
        chunk = client._parse_stream_chunk(data)
        assert chunk.content == "Hello"

    def test_parse_stream_chunk_usage(self, client):
        data = {"usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 100, "totalTokenCount": 150}}
        chunk = client._parse_stream_chunk(data)
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 150
        assert chunk.is_final is True


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_gemini_client(self):
        client = create_gemini_client()
        assert isinstance(client, GeminiClient)
        assert client.provider == LLMProvider.GEMINI

    def test_create_gemini_client_with_settings(self):
        settings = GeminiSettings(model="gemini-1.5-pro")
        client = create_gemini_client(settings)
        assert client.model == "gemini-1.5-pro"
