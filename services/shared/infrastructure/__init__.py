"""
Solace-AI Shared Infrastructure Layer.
Provides LLM clients, external service integrations, and infrastructure components
shared across all microservices.
"""
from .llm_client import (
    LLMClientSettings,
    LLMConfigurationError,
    UnifiedLLMClient,
    LLM_SYSTEM_PROMPTS,
    get_llm_prompt,
)

__all__ = [
    "LLMClientSettings",
    "LLMConfigurationError",
    "UnifiedLLMClient",
    "LLM_SYSTEM_PROMPTS",
    "get_llm_prompt",
]
