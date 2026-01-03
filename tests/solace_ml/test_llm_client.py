"""Unit tests for LLM client module."""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from solace_ml.llm_client import (
    LLMProvider, MessageRole, FinishReason, Message, ToolParameter, ToolDefinition,
    ToolCall, TokenUsage, LLMResponse, StreamChunk, LLMSettings, RetryPolicy,
    LLMError, RateLimitError, MultiProviderClient, build_messages, extract_tool_calls,
)


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_provider_values(self):
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.OLLAMA.value == "ollama"


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_role_values(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"


class TestFinishReason:
    """Tests for FinishReason enum."""

    def test_finish_reason_values(self):
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.TOOL_CALL.value == "tool_call"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.name is None

    def test_message_with_name(self):
        msg = Message(role=MessageRole.USER, content="Hello", name="User1")
        assert msg.name == "User1"

    def test_message_with_tool_call_id(self):
        msg = Message(role=MessageRole.TOOL, content="result", tool_call_id="call_123")
        assert msg.tool_call_id == "call_123"

    def test_message_with_tool_calls(self):
        tool_calls = [ToolCall(id="tc_1", name="get_weather", arguments={"city": "NYC"})]
        msg = Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "get_weather"


class TestToolParameter:
    """Tests for ToolParameter model."""

    def test_create_parameter(self):
        param = ToolParameter(name="city", type="string", description="City name")
        assert param.name == "city"
        assert param.param_type == "string"
        assert param.required is True

    def test_parameter_with_enum(self):
        param = ToolParameter(name="unit", type="string", description="Unit",
                             enum=["celsius", "fahrenheit"], required=False)
        assert param.enum == ["celsius", "fahrenheit"]
        assert param.required is False


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_create_tool(self):
        tool = ToolDefinition(name="get_weather", description="Get weather for city")
        assert tool.name == "get_weather"
        assert len(tool.parameters) == 0

    def test_tool_with_parameters(self):
        params = [
            ToolParameter(name="city", type="string", description="City name"),
            ToolParameter(name="unit", type="string", description="Unit", required=False)
        ]
        tool = ToolDefinition(name="get_weather", description="Get weather", parameters=params)
        assert len(tool.parameters) == 2

    def test_to_json_schema(self):
        params = [ToolParameter(name="city", type="string", description="City name")]
        tool = ToolDefinition(name="get_weather", description="Get weather", parameters=params)
        schema = tool.to_json_schema()
        assert schema["type"] == "object"
        assert "city" in schema["properties"]
        assert "city" in schema["required"]


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self):
        tc = ToolCall(id="call_123", name="get_weather", arguments={"city": "NYC"})
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments["city"] == "NYC"


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_create_usage(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_default_values(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.cached_tokens == 0

    def test_cost_estimate(self):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        assert usage.cost_estimate > 0


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_create_response(self):
        response = LLMResponse(content="Hello!", model="claude-3-opus")
        assert response.content == "Hello!"
        assert response.finish_reason == FinishReason.STOP
        assert response.model == "claude-3-opus"

    def test_response_with_tool_calls(self):
        tool_calls = [ToolCall(id="tc_1", name="get_weather", arguments={})]
        response = LLMResponse(content="", tool_calls=tool_calls, finish_reason=FinishReason.TOOL_CALL)
        assert response.has_tool_calls is True
        assert len(response.tool_calls) == 1

    def test_response_no_tool_calls(self):
        response = LLMResponse(content="Hello")
        assert response.has_tool_calls is False


class TestStreamChunk:
    """Tests for StreamChunk model."""

    def test_create_chunk(self):
        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.is_final is False

    def test_final_chunk(self):
        chunk = StreamChunk(content="", finish_reason=FinishReason.STOP, is_final=True)
        assert chunk.is_final is True


class TestLLMSettings:
    """Tests for LLMSettings model."""

    def test_default_settings(self):
        settings = LLMSettings()
        assert settings.max_tokens == 4096
        assert settings.temperature == 0.7
        assert settings.max_retries == 3

    def test_custom_settings(self):
        settings = LLMSettings(max_tokens=8192, temperature=0.5)
        assert settings.max_tokens == 8192
        assert settings.temperature == 0.5


class TestRetryPolicy:
    """Tests for RetryPolicy model."""

    def test_default_policy(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0

    def test_get_delay(self):
        policy = RetryPolicy(base_delay=1.0, multiplier=2.0)
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0

    def test_max_delay(self):
        policy = RetryPolicy(base_delay=10.0, multiplier=5.0, max_delay=30.0)
        assert policy.get_delay(3) == 30.0


class TestLLMError:
    """Tests for LLMError exception."""

    def test_create_error(self):
        error = LLMError("Test error", provider=LLMProvider.ANTHROPIC)
        assert str(error) == "Test error"
        assert error.provider == LLMProvider.ANTHROPIC
        assert error.retryable is False

    def test_retryable_error(self):
        error = LLMError("Timeout", provider=LLMProvider.OPENAI,
                        error_type="timeout", retryable=True)
        assert error.retryable is True
        assert error.error_type == "timeout"


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_create_rate_limit_error(self):
        error = RateLimitError("Rate limited", LLMProvider.OPENAI, retry_after=60.0)
        assert error.retryable is True
        assert error.retry_after == 60.0
        assert error.status_code == 429


class TestBuildMessages:
    """Tests for build_messages helper."""

    def test_simple_message(self):
        messages = build_messages("Hello")
        assert len(messages) == 1
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hello"

    def test_with_system_prompt(self):
        messages = build_messages("Hello", system="You are helpful.")
        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER

    def test_with_history(self):
        history = [
            Message(role=MessageRole.USER, content="Hi"),
            Message(role=MessageRole.ASSISTANT, content="Hello!")
        ]
        messages = build_messages("How are you?", history=history)
        assert len(messages) == 3
        assert messages[2].content == "How are you?"


class TestExtractToolCalls:
    """Tests for extract_tool_calls helper."""

    def test_extract_calls(self):
        tool_calls = [
            ToolCall(id="tc_1", name="get_weather", arguments={"city": "NYC"}),
            ToolCall(id="tc_2", name="get_time", arguments={"timezone": "EST"})
        ]
        response = LLMResponse(content="", tool_calls=tool_calls)
        extracted = extract_tool_calls(response)
        assert len(extracted) == 2
        assert extracted[0]["name"] == "get_weather"
        assert extracted[1]["arguments"]["timezone"] == "EST"

    def test_extract_empty(self):
        response = LLMResponse(content="Hello")
        extracted = extract_tool_calls(response)
        assert len(extracted) == 0
