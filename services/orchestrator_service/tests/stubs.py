"""
Test stub agent node functions for orchestrator graph testing.

These are simple local implementations that return hardcoded responses,
used by tests when downstream services are not available.
Moved from graph_builder.py to keep production code clean.
"""
from __future__ import annotations
from typing import Any

from services.orchestrator_service.src.langgraph.state_schema import (
    OrchestratorState,
    AgentType,
    AgentResult,
)


def chat_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Chat agent stub - returns a hardcoded response for testing."""
    message = state.get("current_message", "")
    personality_style = state.get("personality_style", {})
    warmth = personality_style.get("warmth", 0.7)
    response = (
        "Thank you for sharing that with me. I'm here to listen and support you. "
        if warmth > 0.7
        else "I understand. "
    ) + "How are you feeling about this?"
    return {
        "agent_results": [
            AgentResult(
                agent_type=AgentType.CHAT,
                success=True,
                response_content=response,
                confidence=0.7,
                metadata={"warmth": warmth},
            ).to_dict()
        ]
    }


def diagnosis_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Diagnosis agent stub - returns a hardcoded response for testing."""
    return {
        "agent_results": [
            AgentResult(
                agent_type=AgentType.DIAGNOSIS,
                success=True,
                response_content=(
                    "Based on what you've shared, it might be helpful to explore "
                    "these feelings further. Would you like to talk more about "
                    "when these symptoms started?"
                ),
                confidence=0.75,
                metadata={"assessment_type": "symptom_exploration"},
            ).to_dict()
        ]
    }


def therapy_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Therapy agent stub - returns a hardcoded response for testing."""
    return {
        "agent_results": [
            AgentResult(
                agent_type=AgentType.THERAPY,
                success=True,
                response_content=(
                    "It sounds like you're going through a difficult time. "
                    "One thing that might help is to take a moment to notice "
                    "how you're feeling right now, without trying to change it. "
                    "Just observe your thoughts and feelings with curiosity."
                ),
                confidence=0.80,
                metadata={"technique": "mindfulness_observation", "modality": "ACT"},
            ).to_dict()
        ]
    }


def personality_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Personality agent stub - returns hardcoded style parameters for testing."""
    style = {
        "warmth": 0.7,
        "structure": 0.5,
        "complexity": 0.5,
        "directness": 0.5,
        "energy": 0.5,
        "validation_level": 0.6,
        "style_type": "balanced",
    }
    return {
        "personality_style": style,
        "agent_results": [
            AgentResult(
                agent_type=AgentType.PERSONALITY,
                success=True,
                confidence=0.70,
                metadata={"style_params": style},
            ).to_dict()
        ],
    }
