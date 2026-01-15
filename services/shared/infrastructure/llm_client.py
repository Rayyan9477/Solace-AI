"""
Solace-AI Shared Infrastructure - Unified LLM Client.
Unified LLM gateway using Portkey AI for multi-provider support with fallbacks,
load balancing, caching, and observability across all services.
"""
from __future__ import annotations
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class LLMClientSettings(BaseSettings):
    """
    Unified LLM client configuration with Portkey Gateway.

    Environment variables use the LLM_ prefix for shared configuration.
    Services can override with service-specific prefixes.
    """
    portkey_api_key: str = Field(default="")
    portkey_gateway_url: str = Field(default="https://api.portkey.ai/v1")
    primary_provider: str = Field(default="anthropic")
    primary_model: str = Field(default="claude-sonnet-4-20250514")
    fallback_provider: str = Field(default="openai")
    fallback_model: str = Field(default="gpt-4o")
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    azure_api_key: str = Field(default="")
    azure_endpoint: str = Field(default="")
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)
    retry_attempts: int = Field(default=3)
    timeout_seconds: int = Field(default=30)
    enable_caching: bool = Field(default=True)
    cache_mode: str = Field(default="semantic")
    enable_fallback: bool = Field(default=True)
    enable_load_balancing: bool = Field(default=False)
    load_balance_weight_primary: float = Field(default=0.7)
    trace_id_prefix: str = Field(default="solace")
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")


class UnifiedLLMClient:
    """
    Unified LLM client for all Solace-AI services using Portkey AI Gateway.

    Provides resilient LLM access with automatic fallbacks, load balancing,
    semantic caching, and comprehensive observability.

    Uses AsyncPortkey for non-blocking async operations following Portkey SDK patterns.

    Example usage:
        ```python
        from services.shared import UnifiedLLMClient, LLMClientSettings

        settings = LLMClientSettings()
        client = UnifiedLLMClient(settings)
        await client.initialize()

        response = await client.generate(
            system_prompt="You are a helpful assistant.",
            user_message="Hello!",
            service_name="my_service",
        )
        ```
    """

    def __init__(self, settings: LLMClientSettings | None = None) -> None:
        self._settings = settings or LLMClientSettings()
        self._client = None
        self._initialized = False
        self._request_count = 0
        self._error_count = 0
        self._cache_hits = 0

    async def initialize(self) -> None:
        """Initialize the LLM client with Portkey configuration."""
        try:
            from portkey_ai import AsyncPortkey
            config = self._build_portkey_config()
            self._client = AsyncPortkey(
                api_key=self._settings.portkey_api_key,
                base_url=self._settings.portkey_gateway_url,
                config=config,
            )
            self._initialized = True
            logger.info(
                "unified_llm_client_initialized",
                provider=self._settings.primary_provider,
                model=self._settings.primary_model,
                fallback_enabled=self._settings.enable_fallback,
                caching_enabled=self._settings.enable_caching,
                cache_mode=self._settings.cache_mode,
            )
        except ImportError:
            logger.warning("portkey_ai_not_installed", fallback="using_mock_client")
            self._client = None
            self._initialized = True
        except Exception as e:
            logger.error("unified_llm_client_initialization_failed", error=str(e))
            self._initialized = False
            raise

    def _build_portkey_config(self) -> dict[str, Any]:
        """
        Build Portkey configuration with fallbacks, load balancing, and caching.

        Configuration structure follows Portkey SDK patterns:
        - strategy.mode: 'fallback' or 'loadbalance'
        - targets: List of provider configurations
        - cache: Caching configuration
        - retry: Retry configuration
        """
        config: dict[str, Any] = {
            "retry": {"attempts": self._settings.retry_attempts}
        }

        if self._settings.enable_caching:
            config["cache"] = {"mode": self._settings.cache_mode}

        targets = []

        primary_target: dict[str, Any] = {
            "provider": self._settings.primary_provider,
            "override_params": {"model": self._settings.primary_model},
        }
        if self._settings.primary_provider == "anthropic" and self._settings.anthropic_api_key:
            primary_target["api_key"] = self._settings.anthropic_api_key
        elif self._settings.primary_provider == "openai" and self._settings.openai_api_key:
            primary_target["api_key"] = self._settings.openai_api_key

        if self._settings.enable_load_balancing:
            primary_target["weight"] = self._settings.load_balance_weight_primary

        targets.append(primary_target)

        if self._settings.enable_fallback:
            fallback_target: dict[str, Any] = {
                "provider": self._settings.fallback_provider,
                "override_params": {"model": self._settings.fallback_model},
            }
            if self._settings.fallback_provider == "openai" and self._settings.openai_api_key:
                fallback_target["api_key"] = self._settings.openai_api_key
            elif self._settings.fallback_provider == "anthropic" and self._settings.anthropic_api_key:
                fallback_target["api_key"] = self._settings.anthropic_api_key

            if self._settings.enable_load_balancing:
                fallback_target["weight"] = 1.0 - self._settings.load_balance_weight_primary

            targets.append(fallback_target)

        if self._settings.enable_load_balancing:
            config["strategy"] = {"mode": "loadbalance"}
        elif self._settings.enable_fallback:
            config["strategy"] = {"mode": "fallback"}

        config["targets"] = targets
        return config

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
        service_name: str = "unknown",
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a response using the configured LLM.

        Args:
            system_prompt: System prompt with instructions
            user_message: User's message to respond to
            conversation_history: Previous conversation messages
            context: Additional context to append to system prompt
            service_name: Name of the calling service for tracing
            metadata: Additional metadata for tracing
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            Generated response text
        """
        if not self._initialized or self._client is None:
            logger.warning("llm_client_not_initialized", service=service_name)
            return ""

        self._request_count += 1
        messages = self._build_messages(system_prompt, user_message, conversation_history, context)

        try:
            response = await self._execute_completion(
                messages=messages,
                service_name=service_name,
                metadata=metadata,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return self._extract_response_text(response)
        except Exception as e:
            self._error_count += 1
            logger.error("llm_generation_failed", error=str(e), service=service_name)
            return ""

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[dict[str, str]] | None,
        context: str | None,
    ) -> list[dict[str, Any]]:
        """Build message array for LLM completion."""
        full_system = system_prompt
        if context:
            full_system += f"\n\n{context}"

        messages: list[dict[str, Any]] = [{"role": "system", "content": full_system}]

        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        messages.append({"role": "user", "content": user_message})
        return messages

    async def _execute_completion(
        self,
        messages: list[dict[str, Any]],
        service_name: str,
        metadata: dict[str, Any] | None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Execute LLM completion request using AsyncPortkey."""
        request_params = {
            "model": self._settings.primary_model,
            "messages": messages,
            "max_tokens": max_tokens or self._settings.max_tokens,
            "temperature": temperature or self._settings.temperature,
        }

        trace_id = f"{self._settings.trace_id_prefix}-{service_name}"
        if metadata:
            trace_id = f"{trace_id}-{metadata.get('request_id', 'unknown')}"

        response = await self._client.chat.completions.create(
            **request_params,
            metadata={"trace_id": trace_id, "service": service_name, **(metadata or {})},
        )
        return response

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from LLM response."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
        return ""

    async def generate_with_fallback(
        self,
        system_prompt: str,
        user_message: str,
        fallback_response: str,
        **kwargs: Any,
    ) -> str:
        """
        Generate response with a fallback if LLM fails.

        Args:
            system_prompt: System prompt with instructions
            user_message: User's message to respond to
            fallback_response: Response to return if LLM fails
            **kwargs: Additional arguments passed to generate()

        Returns:
            Generated response or fallback
        """
        response = await self.generate(system_prompt, user_message, **kwargs)
        return response if response else fallback_response

    async def shutdown(self) -> None:
        """Shutdown the LLM client."""
        logger.info(
            "unified_llm_client_shutting_down",
            total_requests=self._request_count,
            total_errors=self._error_count,
            cache_hits=self._cache_hits,
        )
        self._client = None
        self._initialized = False

    def get_statistics(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "initialized": self._initialized,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "cache_hits": self._cache_hits,
            "error_rate": self._error_count / max(self._request_count, 1),
            "primary_provider": self._settings.primary_provider,
            "primary_model": self._settings.primary_model,
            "fallback_enabled": self._settings.enable_fallback,
            "caching_enabled": self._settings.enable_caching,
            "cache_mode": self._settings.cache_mode,
        }

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @property
    def is_available(self) -> bool:
        """Check if LLM is available for requests."""
        return self._initialized and self._client is not None


# Service-specific system prompts
LLM_SYSTEM_PROMPTS = {
    # Therapy Service Prompts
    "therapy_general": """You are a compassionate, evidence-based AI therapy assistant trained in
CBT, DBT, ACT, and Motivational Interviewing. Your responses should:
- Demonstrate empathy and validation
- Use Socratic questioning when appropriate
- Maintain appropriate therapeutic boundaries
- Never diagnose or prescribe medication
- Encourage professional help for serious concerns
- Focus on the client's strengths and capabilities""",
    "therapy_cbt": """You are applying Cognitive Behavioral Therapy principles. Focus on:
- Identifying automatic negative thoughts
- Examining evidence for and against thoughts
- Developing balanced alternative perspectives
- Connecting thoughts, feelings, and behaviors
- Assigning appropriate behavioral experiments or homework""",
    "therapy_dbt": """You are applying Dialectical Behavior Therapy principles. Focus on:
- Balancing acceptance and change
- Teaching distress tolerance skills
- Practicing mindfulness techniques
- Developing emotional regulation strategies
- Building interpersonal effectiveness""",
    "therapy_act": """You are applying Acceptance and Commitment Therapy principles. Focus on:
- Promoting psychological flexibility
- Practicing acceptance of difficult thoughts/feelings
- Clarifying personal values
- Encouraging committed action aligned with values
- Using cognitive defusion techniques""",
    "therapy_mi": """You are applying Motivational Interviewing principles. Focus on:
- Expressing empathy through reflective listening
- Developing discrepancy between current behavior and goals
- Rolling with resistance rather than confronting
- Supporting self-efficacy and autonomy
- Eliciting change talk from the client""",
    "therapy_mindfulness": """You are guiding mindfulness practice. Focus on:
- Present-moment awareness
- Non-judgmental observation
- Breath awareness and body scanning
- Gentle redirection when mind wanders
- Cultivating self-compassion""",
    "therapy_crisis": """SAFETY PRIORITY: The client may be in distress. Your response should:
- Validate their feelings with warmth
- Assess immediate safety concerns
- Provide crisis resources (988 Suicide & Crisis Lifeline)
- Encourage connection with support systems
- Avoid leaving them alone in distress""",

    # Diagnosis Service Prompts
    "diagnosis_assessment": """You are a clinical assessment assistant. Your role is to:
- Analyze reported symptoms systematically
- Consider differential diagnoses
- Identify severity indicators
- Note relevant contextual factors
- Recommend appropriate assessment tools
- Never provide definitive diagnoses - suggest clinical evaluation""",
    "diagnosis_screening": """You are conducting a mental health screening. Focus on:
- Gathering relevant symptom information
- Asking clarifying questions sensitively
- Assessing duration and severity
- Identifying risk factors
- Recommending appropriate next steps""",

    # Safety Service Prompts
    "safety_assessment": """You are a safety assessment specialist. Your priorities are:
- Identifying immediate safety concerns
- Assessing risk levels accurately
- Providing appropriate crisis resources
- Ensuring user well-being
- Escalating when necessary""",
    "safety_intervention": """IMMEDIATE SAFETY RESPONSE: Focus on:
- De-escalation techniques
- Grounding and stabilization
- Crisis resource provision (988, 911)
- Warm handoff to appropriate services
- Continuous safety monitoring""",

    # Memory Service Prompts
    "memory_summarization": """You are a clinical documentation assistant. Generate concise,
professional summaries that include:
- Key topics and themes discussed
- Important insights or breakthroughs
- Techniques or interventions used
- Progress indicators
- Recommendations for follow-up""",
}


def get_llm_prompt(prompt_key: str, fallback_key: str = "therapy_general") -> str:
    """
    Get a system prompt by key.

    Args:
        prompt_key: The key for the desired prompt
        fallback_key: Key to use if prompt_key not found

    Returns:
        The system prompt string
    """
    return LLM_SYSTEM_PROMPTS.get(prompt_key, LLM_SYSTEM_PROMPTS.get(fallback_key, ""))
