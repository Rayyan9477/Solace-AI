"""Solace-AI Anthropic Client - Claude adapter with streaming support."""
from __future__ import annotations
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
import httpx
from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict
import structlog
from solace_ml.llm_client import (
    LLMClient, LLMSettings, LLMProvider, LLMResponse, LLMError, RateLimitError,
    Message, MessageRole, ToolDefinition, ToolCall, TokenUsage, StreamChunk, FinishReason,
)

logger = structlog.get_logger(__name__)
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicSettings(LLMSettings):
    """Anthropic-specific settings."""
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default="claude-sonnet-4-20250514")
    max_tokens: int = Field(default=4096, ge=1, le=200000)
    thinking_budget: int | None = Field(default=None, description="Extended thinking budget")
    beta_features: list[str] = Field(default_factory=list)
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_", env_file=".env", extra="ignore")


class AnthropicClient(LLMClient):
    """Anthropic Claude API client with streaming."""

    def __init__(self, settings: AnthropicSettings | None = None) -> None:
        settings = settings or AnthropicSettings()
        super().__init__(settings, LLMProvider.ANTHROPIC)
        self._settings: AnthropicSettings = settings
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_seconds))

    async def complete(self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
                       system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        """Generate completion using Claude."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        payload = self._build_payload(messages, tools, system_prompt, **kwargs)
        try:
            response = await self._http.post(
                ANTHROPIC_API_URL, json=payload,
                headers=self._build_headers(stream=False)
            )
            self._check_response(response)
            data = response.json()
            latency = (time.perf_counter() - start_time) * 1000
            return self._parse_response(data, latency, request_id)
        except httpx.TimeoutException as e:
            raise LLMError(f"Request timeout: {e}", provider=self._provider,
                          error_type="timeout", retryable=True) from e
        except httpx.RequestError as e:
            raise LLMError(f"Connection error: {e}", provider=self._provider,
                          error_type="connection_error", retryable=True) from e

    async def stream(self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
                     system_prompt: str | None = None, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream completion using Claude."""
        payload = self._build_payload(messages, tools, system_prompt, stream=True, **kwargs)
        try:
            async with self._http.stream(
                "POST", ANTHROPIC_API_URL, json=payload,
                headers=self._build_headers(stream=True)
            ) as response:
                self._check_response(response)
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        chunk = self._parse_stream_event(data)
                        if chunk:
                            yield chunk
                            if chunk.is_final:
                                break
                    except json.JSONDecodeError:
                        continue
        except httpx.TimeoutException as e:
            raise LLMError(f"Stream timeout: {e}", provider=self._provider,
                          error_type="timeout", retryable=True) from e

    async def health_check(self) -> bool:
        """Check Anthropic API health."""
        try:
            response = await self._http.post(
                ANTHROPIC_API_URL, json={"model": self._settings.model,
                                          "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]},
                headers=self._build_headers(stream=False)
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("health_check_failed", provider="anthropic", error=str(e))
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http.aclose()

    def _build_headers(self, stream: bool = False) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "x-api-key": self._settings.api_key.get_secret_value(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if self._settings.beta_features:
            headers["anthropic-beta"] = ",".join(self._settings.beta_features)
        return headers

    def _build_payload(self, messages: list[Message], tools: list[ToolDefinition] | None,
                       system_prompt: str | None, stream: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Build API request payload."""
        anthropic_messages = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._settings.model),
            "max_tokens": kwargs.get("max_tokens", self._settings.max_tokens),
            "messages": anthropic_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if self._settings.temperature is not None:
            payload["temperature"] = kwargs.get("temperature", self._settings.temperature)
        if self._settings.top_p != 1.0:
            payload["top_p"] = self._settings.top_p
        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]
        if stream:
            payload["stream"] = True
        if self._settings.thinking_budget:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self._settings.thinking_budget}
        return payload

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format."""
        result = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            converted: dict[str, Any] = {"role": msg.role.value}
            if msg.tool_calls:
                converted["content"] = [
                    {"type": "tool_use", "id": tc.id, "name": tc.name,
                     "input": tc.arguments} for tc in msg.tool_calls
                ]
            elif msg.tool_call_id:
                converted["role"] = "user"
                converted["content"] = [
                    {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}
                ]
            else:
                converted["content"] = msg.content
            result.append(converted)
        return result

    def _convert_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert tool definition to Anthropic format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.to_json_schema()
        }

    def _check_response(self, response: httpx.Response) -> None:
        """Check response for errors."""
        if response.status_code == 200:
            return
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError("Rate limit exceeded", self._provider,
                                float(retry_after) if retry_after else None)
        if response.status_code >= 500:
            raise LLMError(f"Server error: {response.status_code}", provider=self._provider,
                          error_type="server_error", status_code=response.status_code, retryable=True)
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text
        raise LLMError(f"API error: {error_msg}", provider=self._provider,
                      error_type="api_error", status_code=response.status_code)

    def _parse_response(self, data: dict[str, Any], latency: float, request_id: str) -> LLMResponse:
        """Parse API response."""
        content = ""
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"], name=block["name"], arguments=block.get("input", {})
                ))
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            cached_tokens=usage_data.get("cache_read_input_tokens", 0)
        )
        finish_reason = self._map_stop_reason(data.get("stop_reason"))
        return LLMResponse(
            content=content, finish_reason=finish_reason, tool_calls=tool_calls,
            usage=usage, model=data.get("model", self._settings.model),
            provider=self._provider, latency_ms=latency, request_id=request_id
        )

    def _parse_stream_event(self, event: dict[str, Any]) -> StreamChunk | None:
        """Parse streaming event."""
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                return StreamChunk(content=delta.get("text", ""))
            if delta.get("type") == "input_json_delta":
                return StreamChunk(content=delta.get("partial_json", ""))
        elif event_type == "message_delta":
            delta = event.get("delta", {})
            usage_data = event.get("usage", {})
            usage = TokenUsage(completion_tokens=usage_data.get("output_tokens", 0)) if usage_data else None
            return StreamChunk(
                finish_reason=self._map_stop_reason(delta.get("stop_reason")),
                usage=usage, is_final=True
            )
        elif event_type == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                return StreamChunk(tool_calls=[ToolCall(
                    id=block.get("id", ""), name=block.get("name", ""), arguments={}
                )])
        return None

    def _map_stop_reason(self, reason: str | None) -> FinishReason:
        """Map Anthropic stop reason to standard enum."""
        mapping = {
            "end_turn": FinishReason.STOP, "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH, "tool_use": FinishReason.TOOL_CALL,
        }
        return mapping.get(reason or "", FinishReason.STOP)


def create_anthropic_client(settings: AnthropicSettings | None = None) -> AnthropicClient:
    """Factory function to create Anthropic client."""
    return AnthropicClient(settings)


async def quick_complete(prompt: str, *, model: str | None = None,
                         system: str | None = None) -> str:
    """Quick completion helper."""
    client = create_anthropic_client()
    try:
        settings = AnthropicSettings(model=model) if model else None
        if settings:
            client = AnthropicClient(settings)
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await client.complete(messages, system_prompt=system)
        return response.content
    finally:
        await client.close()
