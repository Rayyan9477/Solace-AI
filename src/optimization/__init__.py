"""
Multi-Agent System Optimization Module

Provides comprehensive performance optimization for the mental health chatbot's
multi-agent system, including parallel execution, caching, context compression,
and cost optimization.
"""

from .performance_profiler import (
    AgentPerformanceProfiler,
    MultiAgentOrchestrationOptimizer,
    CostOptimizationAnalyzer,
    PerformanceMetrics
)

from .context_optimizer import (
    SemanticContextCompressor,
    AgentResultCache,
    ContextWindowManager,
    ContextItem
)

from .optimized_orchestrator import OptimizedAgentOrchestrator

__all__ = [
    # Performance Profiling
    'AgentPerformanceProfiler',
    'MultiAgentOrchestrationOptimizer',
    'CostOptimizationAnalyzer',
    'PerformanceMetrics',

    # Context Optimization
    'SemanticContextCompressor',
    'AgentResultCache',
    'ContextWindowManager',
    'ContextItem',

    # Orchestration
    'OptimizedAgentOrchestrator'
]

__version__ = '1.0.0'