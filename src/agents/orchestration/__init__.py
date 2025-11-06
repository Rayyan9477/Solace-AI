"""Orchestration module - Agent coordination and supervision"""

from .agent_orchestrator import AgentOrchestrator
from .supervisor_agent import SupervisorAgent

# Import optimized orchestrator if available
try:
    from src.optimization.optimized_orchestrator import OptimizedAgentOrchestrator
    # Use optimized version by default
    AgentOrchestrator = OptimizedAgentOrchestrator
    OPTIMIZATION_ENABLED = True
except ImportError:
    # Fallback to standard orchestrator
    OPTIMIZATION_ENABLED = False

__all__ = ['AgentOrchestrator', 'SupervisorAgent', 'OPTIMIZATION_ENABLED']