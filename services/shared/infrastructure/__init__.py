"""
Solace-AI Shared Infrastructure Layer.
Provides LLM clients, external service integrations, and infrastructure components
shared across all microservices.
"""
from .llm_client import (
    LLMClientSettings,
    UnifiedLLMClient,
    LLM_SYSTEM_PROMPTS,
    get_llm_prompt,
)

__all__ = [
    "LLMClientSettings",
    "UnifiedLLMClient",
    "LLM_SYSTEM_PROMPTS",
    "get_llm_prompt",
]
