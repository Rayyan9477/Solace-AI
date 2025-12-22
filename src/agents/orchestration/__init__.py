"""Orchestration module - Agent coordination and supervision.

This module provides the core orchestration components for multi-agent coordination.
The AgentOrchestrator is the primary class for managing agent interactions.

NOTE: OptimizedAgentOrchestrator is available in src.optimization if performance
tuning is needed, but should be used explicitly - not as a silent replacement.
"""

from .agent_orchestrator import AgentOrchestrator
from .supervisor_agent import SupervisorAgent

__all__ = ['AgentOrchestrator', 'SupervisorAgent']