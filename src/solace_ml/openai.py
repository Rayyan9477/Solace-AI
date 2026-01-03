"""Solace-AI OpenAI Client - OpenAI adapter with function calling support."""
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
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings."""
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default="gpt-4o")
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    organization_id: str | None = Field(default=None)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    response_format: str | None = Field(default=None, description="json_object or text")
    seed: int | None = Field(default=None, description="Deterministic sampling seed")
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore")


class ToolChoice(str):
    """Tool choice options."""
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class OpenAIClient(LLMClient):
    """OpenAI API client with function calling."""

    def __init__(self, settings: OpenAISettings | None = None) -> None:
        settings = settings or OpenAISettings()
        super().__init__(settings, LLMProvider.OPENAI)
        self._settings: OpenAISettings = settings
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_seconds))

    async def complete(self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
                       system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        """Generate completion using OpenAI."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        payload = self._build_payload(messages, tools, system_prompt, **kwargs)
        try:
            response = await self._http.post(
                OPENAI_API_URL, json=payload, headers=self._build_headers()
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
        """Stream completion using OpenAI."""
        payload = self._build_payload(messages, tools, system_prompt, stream=True, **kwargs)
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        try:
            async with self._http.stream(
                "POST", OPENAI_API_URL, json=payload, headers=self._build_headers()
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
                    except json.JSONDecodeError:
                        continue
        except httpx.TimeoutException as e:
            raise LLMError(f"Stream timeout: {e}", provider=self._provider,
                          error_type="timeout", retryable=True) from e

    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            response = await self._http.post(
                OPENAI_API_URL, json={"model": self._settings.model,
                                       "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]},
                headers=self._build_headers()
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("health_check_failed", provider="openai", error=str(e))
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http.aclose()

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        if self._settings.organization_id:
            headers["OpenAI-Organization"] = self._settings.organization_id
        return headers

    def _build_payload(self, messages: list[Message], tools: list[ToolDefinition] | None,
                       system_prompt: str | None, stream: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Build API request payload."""
        openai_messages = self._convert_messages(messages, system_prompt)
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._settings.model),
            "messages": openai_messages,
        }
        max_tokens = kwargs.get("max_tokens", self._settings.max_tokens)
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if self._settings.temperature is not None:
            payload["temperature"] = kwargs.get("temperature", self._settings.temperature)
        if self._settings.top_p != 1.0:
            payload["top_p"] = self._settings.top_p
        if self._settings.frequency_penalty != 0.0:
            payload["frequency_penalty"] = self._settings.frequency_penalty
        if self._settings.presence_penalty != 0.0:
            payload["presence_penalty"] = self._settings.presence_penalty
        if self._settings.response_format:
            payload["response_format"] = {"type": self._settings.response_format}
        if self._settings.seed is not None:
            payload["seed"] = self._settings.seed
        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _convert_messages(self, messages: list[Message], system_prompt: str | None) -> list[dict[str, Any]]:
        """Convert messages to OpenAI format."""
        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for msg in messages:
            converted: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if msg.name:
                converted["name"] = msg.name
            if msg.tool_calls:
                converted["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in msg.tool_calls
                ]
                converted.pop("content", None)
            if msg.tool_call_id:
                converted["role"] = "tool"
                converted["tool_call_id"] = msg.tool_call_id
            result.append(converted)
        return result

    def _convert_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert tool definition to OpenAI format."""
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
        content = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=func.get("name", ""), arguments=args))
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
        choice = data.get("choices", [{}])[0] if data.get("choices") else {}
        delta = choice.get("delta", {})
        if not choice and not data.get("usage"):
            return None
        content = delta.get("content", "")
        finish_reason = None
        tool_calls: list[ToolCall] = []
        if choice.get("finish_reason"):
            finish_reason = self._map_finish_reason(choice["finish_reason"])
        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            if idx not in accumulated:
                accumulated[idx] = {"id": "", "name": "", "arguments": ""}
            if tc_delta.get("id"):
                accumulated[idx]["id"] = tc_delta["id"]
            func = tc_delta.get("function", {})
            if func.get("name"):
                accumulated[idx]["name"] = func["name"]
            if func.get("arguments"):
                accumulated[idx]["arguments"] += func["arguments"]
        if finish_reason == FinishReason.TOOL_CALL:
            for tc_data in accumulated.values():
                try:
                    args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args))
        usage = None
        if data.get("usage"):
            usage_data = data["usage"]
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        return StreamChunk(
            content=content, finish_reason=finish_reason, tool_calls=tool_calls,
            usage=usage, is_final=finish_reason is not None
        )

    def _map_finish_reason(self, reason: str | None) -> FinishReason:
        """Map OpenAI finish reason to standard enum."""
        mapping = {
            "stop": FinishReason.STOP, "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALL, "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason or "", FinishReason.STOP)


def create_openai_client(settings: OpenAISettings | None = None) -> OpenAIClient:
    """Factory function to create OpenAI client."""
    return OpenAIClient(settings)


async def quick_complete(prompt: str, *, model: str | None = None,
                         system: str | None = None) -> str:
    """Quick completion helper."""
    client = create_openai_client()
    try:
        settings = OpenAISettings(model=model) if model else None
        if settings:
            client = OpenAIClient(settings)
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await client.complete(messages, system_prompt=system)
        return response.content
    finally:
        await client.close()
