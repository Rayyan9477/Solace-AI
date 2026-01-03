"""Solace-AI xAI Client - Grok adapter with OpenAI-compatible API."""
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
XAI_API_URL = "https://api.x.ai/v1/chat/completions"


class XAISettings(LLMSettings):
    """xAI Grok-specific settings."""
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default="grok-3")
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    reasoning_effort: str | None = Field(default=None, description="Reasoning effort (low/medium/high)")
    model_config = SettingsConfigDict(env_prefix="XAI_", env_file=".env", extra="ignore")


class XAIClient(LLMClient):
    """xAI Grok API client with streaming (OpenAI-compatible)."""

    def __init__(self, settings: XAISettings | None = None) -> None:
        settings = settings or XAISettings()
        super().__init__(settings, LLMProvider.XAI)
        self._settings: XAISettings = settings
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_seconds))

    async def complete(self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
                       system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        """Generate completion using xAI Grok."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        payload = self._build_payload(messages, tools, system_prompt, **kwargs)
        try:
            response = await self._http.post(XAI_API_URL, json=payload, headers=self._build_headers())
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
        """Stream completion using xAI Grok."""
        payload = self._build_payload(messages, tools, system_prompt, stream=True, **kwargs)
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        try:
            async with self._http.stream(
                "POST", XAI_API_URL, json=payload, headers=self._build_headers()
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
                        chunk = self._parse_stream_chunk(data, accumulated_tool_calls)
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
        """Check xAI API health."""
        try:
            response = await self._http.post(
                XAI_API_URL,
                json={"model": self._settings.model, "max_tokens": 5,
                      "messages": [{"role": "user", "content": "hi"}]},
                headers=self._build_headers()
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("health_check_failed", provider="xai", error=str(e))
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http.aclose()

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
            "Content-Type": "application/json"
        }

    def _build_payload(self, messages: list[Message], tools: list[ToolDefinition] | None,
                       system_prompt: str | None, stream: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Build API request payload."""
        converted_msgs = self._convert_messages(messages, system_prompt)
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._settings.model),
            "messages": converted_msgs,
            "max_tokens": kwargs.get("max_tokens", self._settings.max_tokens),
            "temperature": kwargs.get("temperature", self._settings.temperature),
        }
        if self._settings.top_p != 1.0:
            payload["top_p"] = self._settings.top_p
        if self._settings.frequency_penalty != 0.0:
            payload["frequency_penalty"] = self._settings.frequency_penalty
        if self._settings.presence_penalty != 0.0:
            payload["presence_penalty"] = self._settings.presence_penalty
        if self._settings.reasoning_effort:
            payload["reasoning_effort"] = self._settings.reasoning_effort
        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
        return payload

    def _convert_messages(self, messages: list[Message], system_prompt: str | None) -> list[dict[str, Any]]:
        """Convert messages to xAI format (OpenAI-compatible)."""
        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                result.append({"role": "system", "content": msg.content})
            elif msg.role == MessageRole.USER:
                result.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                converted: dict[str, Any] = {"role": "assistant"}
                if msg.tool_calls:
                    converted["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in msg.tool_calls
                    ]
                    converted["content"] = msg.content or ""
                else:
                    converted["content"] = msg.content
                result.append(converted)
            elif msg.role == MessageRole.TOOL:
                result.append({
                    "role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content
                })
        return result

    def _convert_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert tool definition to xAI format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.to_json_schema()
            }
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
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=func.get("name", ""), arguments=arguments))
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        finish_reason = self._map_finish_reason(choice.get("finish_reason"))
        return LLMResponse(
            content=content, finish_reason=finish_reason, tool_calls=tool_calls,
            usage=usage, model=data.get("model", self._settings.model),
            provider=self._provider, latency_ms=latency,
            request_id=data.get("id", request_id)
        )

    def _parse_stream_chunk(self, data: dict[str, Any],
                           accumulated: dict[int, dict[str, Any]]) -> StreamChunk | None:
        """Parse streaming chunk."""
        choices = data.get("choices", [])
        if not choices:
            usage_data = data.get("usage")
            if usage_data:
                return StreamChunk(
                    usage=TokenUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0)
                    )
                )
            return None
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        content = delta.get("content", "")
        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            if idx not in accumulated:
                accumulated[idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
            if tc.get("id"):
                accumulated[idx]["id"] = tc["id"]
            func = tc.get("function", {})
            if func.get("name"):
                accumulated[idx]["name"] = func["name"]
            if func.get("arguments"):
                accumulated[idx]["arguments"] += func["arguments"]
        if finish_reason:
            tool_calls: list[ToolCall] = []
            for tc_data in accumulated.values():
                try:
                    arguments = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=arguments))
            return StreamChunk(
                content=content, finish_reason=self._map_finish_reason(finish_reason),
                tool_calls=tool_calls, is_final=True
            )
        return StreamChunk(content=content) if content else None

    def _map_finish_reason(self, reason: str | None) -> FinishReason:
        """Map xAI finish reason to standard enum."""
        mapping = {
            "stop": FinishReason.STOP, "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALL, "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason or "", FinishReason.STOP)


def create_xai_client(settings: XAISettings | None = None) -> XAIClient:
    """Factory function to create xAI client."""
    return XAIClient(settings)


async def quick_complete(prompt: str, *, model: str | None = None, system: str | None = None) -> str:
    """Quick completion helper."""
    client = create_xai_client()
    try:
        settings = XAISettings(model=model) if model else None
        if settings:
            client = XAIClient(settings)
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await client.complete(messages, system_prompt=system)
        return response.content
    finally:
        await client.close()
