"""Solace-AI LLM Client - Abstract interface for multi-provider LLM access."""
from __future__ import annotations
import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeVar
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)
T = TypeVar("T", bound="LLMClient")


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    MINIMAX = "minimax"
    OLLAMA = "ollama"


class MessageRole(str, Enum):
    """Standard message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Completion finish reasons."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class Message(BaseModel):
    """Chat message structure."""
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolParameter(BaseModel):
    """Tool parameter definition."""
    name: str
    param_type: str = Field(alias="type")
    description: str
    required: bool = True
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    """Tool/function definition for LLM."""
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON schema format."""
        props, required = {}, []
        for p in self.parameters:
            props[p.name] = {"type": p.param_type, "description": p.description}
            if p.enum:
                props[p.name]["enum"] = p.enum
            if p.required:
                required.append(p.name)
        return {"type": "object", "properties": props, "required": required}


class ToolCall(BaseModel):
    """Tool call from LLM response."""
    id: str
    name: str
    arguments: dict[str, Any]


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    @property
    def cost_estimate(self) -> float:
        """Rough cost estimate (varies by model)."""
        return (self.prompt_tokens * 0.003 + self.completion_tokens * 0.015) / 1000


class LLMResponse(BaseModel):
    """Standard LLM response structure."""
    content: str
    finish_reason: FinishReason = FinishReason.STOP
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    latency_ms: float = 0.0
    request_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    content: str = ""
    finish_reason: FinishReason | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None
    is_final: bool = False


class LLMSettings(BaseSettings):
    """Base settings for LLM clients."""
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default="claude-sonnet-4-20250514")
    max_tokens: int = Field(default=4096, ge=1, le=200000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1)
    retry_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    model_config = SettingsConfigDict(extra="ignore")


class RetryPolicy(BaseModel):
    """Retry policy configuration."""
    max_retries: int = 3
    base_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 60.0
    retryable_errors: set[str] = Field(default_factory=lambda: {
        "rate_limit", "timeout", "server_error", "connection_error"
    })

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.multiplier ** attempt)
        return min(delay, self.max_delay)


class LLMError(Exception):
    """Base LLM error."""
    def __init__(self, message: str, *, provider: LLMProvider, error_type: str = "unknown",
                 status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    def __init__(self, message: str, provider: LLMProvider, retry_after: float | None = None) -> None:
        super().__init__(message, provider=provider, error_type="rate_limit",
                        status_code=429, retryable=True)
        self.retry_after = retry_after


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, settings: LLMSettings, provider: LLMProvider) -> None:
        self._settings = settings
        self._provider = provider
        self._retry_policy = RetryPolicy(
            max_retries=settings.max_retries,
            base_delay=settings.retry_delay_seconds,
            multiplier=settings.retry_multiplier,
        )

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def model(self) -> str:
        return self._settings.model

    @abstractmethod
    async def complete(self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
                       system_prompt: str | None = None, **kwargs: Any) -> LLMResponse:
        """Generate completion for messages."""

    @abstractmethod
    async def stream(self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
                     system_prompt: str | None = None, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream completion for messages."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""

    async def complete_with_retry(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Complete with automatic retry on transient errors."""
        last_error: Exception | None = None
        for attempt in range(self._retry_policy.max_retries + 1):
            try:
                return await self.complete(messages, **kwargs)
            except LLMError as e:
                last_error = e
                if not e.retryable or attempt >= self._retry_policy.max_retries:
                    raise
                delay = self._retry_policy.get_delay(attempt)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = max(delay, e.retry_after)
                logger.warning("llm_retry", provider=self._provider.value, attempt=attempt + 1,
                              delay=delay, error_type=e.error_type)
                await asyncio.sleep(delay)
        raise last_error or LLMError("Max retries exceeded", provider=self._provider)


class MultiProviderClient:
    """Client supporting multiple LLM providers with fallback."""

    def __init__(self, primary: LLMClient, fallbacks: list[LLMClient] | None = None) -> None:
        self._primary = primary
        self._fallbacks = fallbacks or []
        self._clients = [primary] + self._fallbacks

    @property
    def primary(self) -> LLMClient:
        return self._primary

    @property
    def available_providers(self) -> list[LLMProvider]:
        return [c.provider for c in self._clients]

    async def complete(self, messages: list[Message], *, provider: LLMProvider | None = None,
                       **kwargs: Any) -> LLMResponse:
        """Complete using specified or primary provider."""
        if provider:
            client = self._get_client(provider)
            return await client.complete_with_retry(messages, **kwargs)
        return await self._primary.complete_with_retry(messages, **kwargs)

    async def complete_with_fallback(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Complete with fallback to other providers on failure."""
        errors: list[tuple[LLMProvider, Exception]] = []
        for client in self._clients:
            try:
                return await client.complete_with_retry(messages, **kwargs)
            except LLMError as e:
                errors.append((client.provider, e))
                logger.warning("provider_failed", provider=client.provider.value, error=str(e))
        error_msg = "; ".join(f"{p.value}: {e}" for p, e in errors)
        raise LLMError(f"All providers failed: {error_msg}", provider=self._primary.provider)

    async def stream(self, messages: list[Message], *, provider: LLMProvider | None = None,
                     **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream using specified or primary provider."""
        client = self._get_client(provider) if provider else self._primary
        async for chunk in client.stream(messages, **kwargs):
            yield chunk

    async def health_check_all(self) -> dict[LLMProvider, bool]:
        """Check health of all providers."""
        results = {}
        for client in self._clients:
            try:
                results[client.provider] = await client.health_check()
            except Exception:
                results[client.provider] = False
        return results

    def _get_client(self, provider: LLMProvider) -> LLMClient:
        """Get client for provider."""
        for client in self._clients:
            if client.provider == provider:
                return client
        raise ValueError(f"Provider {provider.value} not configured")


def build_messages(user_content: str, *, history: list[Message] | None = None,
                   system: str | None = None) -> list[Message]:
    """Build message list from user content and history."""
    messages: list[Message] = []
    if system:
        messages.append(Message(role=MessageRole.SYSTEM, content=system))
    if history:
        messages.extend(history)
    messages.append(Message(role=MessageRole.USER, content=user_content))
    return messages


def extract_tool_calls(response: LLMResponse) -> list[dict[str, Any]]:
    """Extract tool calls as dictionaries."""
    return [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls]


async def measure_latency(coro: Any) -> tuple[Any, float]:
    """Measure coroutine execution time."""
    start = time.perf_counter()
    result = await coro
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed
