"""Solace-AI Google Gemini Client - Gemini adapter with streaming support."""
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
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiSettings(LLMSettings):
    """Google Gemini-specific settings."""
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default="gemini-2.0-flash")
    max_tokens: int = Field(default=8192, ge=1, le=1000000)
    top_k: int | None = Field(default=None, description="Top-k sampling parameter")
    safety_settings: list[dict[str, str]] = Field(default_factory=list)
    model_config = SettingsConfigDict(env_prefix="GEMINI_", env_file=".env", extra="ignore")


class GeminiClient(LLMClient):
    """Google Gemini API client with streaming."""

    def __init__(self, settings: GeminiSettings | None = None) -> None:
        settings = settings or GeminiSettings()
        super().__init__(settings, LLMProvider.GEMINI)
        self._settings: GeminiSettings = settings
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_seconds))

    async def complete(self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
                       system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        """Generate completion using Gemini."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        url = self._build_url(stream=False)
        payload = self._build_payload(messages, tools, system_prompt, **kwargs)
        try:
            response = await self._http.post(url, json=payload, headers=self._build_headers())
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
        """Stream completion using Gemini."""
        url = self._build_url(stream=True)
        payload = self._build_payload(messages, tools, system_prompt, **kwargs)
        try:
            async with self._http.stream("POST", url, json=payload, headers=self._build_headers()) as response:
                self._check_response(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        if line.startswith("data: "):
                            line = line[6:]
                        data = json.loads(line)
                        chunk = self._parse_stream_chunk(data)
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
        """Check Gemini API health."""
        try:
            url = f"{GEMINI_API_URL}/{self._settings.model}"
            headers = {"x-goog-api-key": self._settings.api_key.get_secret_value()}
            response = await self._http.get(url, headers=headers)
            return response.status_code == 200
        except Exception as e:
            logger.warning("health_check_failed", provider="gemini", error=str(e))
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http.aclose()

    def _build_url(self, stream: bool = False) -> str:
        """Build API URL."""
        action = "streamGenerateContent" if stream else "generateContent"
        return f"{GEMINI_API_URL}/{self._settings.model}:{action}"

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with API key in header instead of URL query."""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self._settings.api_key.get_secret_value(),
        }

    def _build_payload(self, messages: list[Message], tools: list[ToolDefinition] | None,
                       system_prompt: str | None, **kwargs: Any) -> dict[str, Any]:
        """Build API request payload."""
        contents = self._convert_messages(messages)
        payload: dict[str, Any] = {"contents": contents}
        generation_config: dict[str, Any] = {
            "maxOutputTokens": kwargs.get("max_tokens", self._settings.max_tokens),
            "temperature": kwargs.get("temperature", self._settings.temperature),
        }
        if self._settings.top_p != 1.0:
            generation_config["topP"] = self._settings.top_p
        if self._settings.top_k is not None:
            generation_config["topK"] = self._settings.top_k
        payload["generationConfig"] = generation_config
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": [self._convert_tool(t) for t in tools]}]
        if self._settings.safety_settings:
            payload["safetySettings"] = self._settings.safety_settings
        return payload

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to Gemini format."""
        result = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            role = "model" if msg.role == MessageRole.ASSISTANT else "user"
            parts: list[dict[str, Any]] = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
            elif msg.tool_call_id:
                parts.append({
                    "functionResponse": {
                        "name": msg.name or "function",
                        "response": {"result": msg.content}
                    }
                })
            else:
                parts.append({"text": msg.content})
            result.append({"role": role, "parts": parts})
        return result

    def _convert_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert tool definition to Gemini format."""
        schema = tool.to_json_schema()
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema
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
        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    content += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=fc.get("name", ""),
                        arguments=fc.get("args", {})
                    ))
            finish_reason = self._map_finish_reason(candidate.get("finishReason"))
        else:
            finish_reason = FinishReason.ERROR
        usage_data = data.get("usageMetadata", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
            cached_tokens=usage_data.get("cachedContentTokenCount", 0)
        )
        return LLMResponse(
            content=content, finish_reason=finish_reason, tool_calls=tool_calls,
            usage=usage, model=self._settings.model, provider=self._provider,
            latency_ms=latency, request_id=request_id
        )

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk | None:
        """Parse streaming chunk."""
        candidates = data.get("candidates", [])
        if not candidates:
            usage_data = data.get("usageMetadata")
            if usage_data:
                return StreamChunk(
                    usage=TokenUsage(
                        prompt_tokens=usage_data.get("promptTokenCount", 0),
                        completion_tokens=usage_data.get("candidatesTokenCount", 0),
                        total_tokens=usage_data.get("totalTokenCount", 0)
                    ),
                    is_final=True
                )
            return None
        candidate = candidates[0]
        content = ""
        tool_calls: list[ToolCall] = []
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=fc.get("name", ""),
                    arguments=fc.get("args", {})
                ))
        finish_reason = self._map_finish_reason(candidate.get("finishReason"))
        is_final = finish_reason != FinishReason.STOP or candidate.get("finishReason") is not None
        return StreamChunk(
            content=content, tool_calls=tool_calls,
            finish_reason=finish_reason if is_final else None,
            is_final=candidate.get("finishReason") is not None
        )

    def _map_finish_reason(self, reason: str | None) -> FinishReason:
        """Map Gemini finish reason to standard enum."""
        mapping = {
            "STOP": FinishReason.STOP, "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER, "RECITATION": FinishReason.CONTENT_FILTER,
            "OTHER": FinishReason.STOP,
        }
        return mapping.get(reason or "", FinishReason.STOP)


def create_gemini_client(settings: GeminiSettings | None = None) -> GeminiClient:
    """Factory function to create Gemini client."""
    return GeminiClient(settings)


async def quick_complete(prompt: str, *, model: str | None = None, system: str | None = None) -> str:
    """Quick completion helper."""
    client = create_gemini_client()
    try:
        settings = GeminiSettings(model=model) if model else None
        if settings:
            client = GeminiClient(settings)
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await client.complete(messages, system_prompt=system)
        return response.content
    finally:
        await client.close()
