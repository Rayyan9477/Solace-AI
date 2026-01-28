"""
Solace-AI Therapy Service - Infrastructure Layer.
Re-exports shared LLM client and provides therapy-specific utilities.
"""
from services.shared.infrastructure import (
    LLMClientSettings,
    UnifiedLLMClient,
    LLM_SYSTEM_PROMPTS,
    get_llm_prompt,
)
from .context_assembler import (
    ContextAssembler,
    ContextAssemblerSettings,
    MemoryContext,
    get_context_assembler,
    initialize_context_assembler,
)


# Therapy-specific aliases for backward compatibility
TherapyLLMClient = UnifiedLLMClient


# Therapy-specific prompt keys
THERAPY_MODALITY_PROMPTS = {
    "general": "therapy_general",
    "cbt": "therapy_cbt",
    "dbt": "therapy_dbt",
    "act": "therapy_act",
    "mi": "therapy_mi",
    "mindfulness": "therapy_mindfulness",
    "crisis": "therapy_crisis",
}


def get_therapy_prompt(modality: str, is_crisis: bool = False) -> str:
    """
    Get the appropriate therapy system prompt.

    Args:
        modality: Therapy modality (cbt, dbt, act, mi, mindfulness)
        is_crisis: Whether this is a crisis situation

    Returns:
        System prompt string for the specified modality
    """
    if is_crisis:
        return get_llm_prompt("therapy_crisis")
    prompt_key = THERAPY_MODALITY_PROMPTS.get(modality.lower(), "therapy_general")
    return get_llm_prompt(prompt_key)


__all__ = [
    "LLMClientSettings",
    "UnifiedLLMClient",
    "TherapyLLMClient",
    "LLM_SYSTEM_PROMPTS",
    "get_llm_prompt",
    "get_therapy_prompt",
    "THERAPY_MODALITY_PROMPTS",
    # Context assembly for Memory Service integration
    "ContextAssembler",
    "ContextAssemblerSettings",
    "MemoryContext",
    "get_context_assembler",
    "initialize_context_assembler",
]
