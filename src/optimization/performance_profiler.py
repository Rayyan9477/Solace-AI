"""
Multi-Agent Performance Profiler for Contextual Chatbot System

This module provides comprehensive performance profiling for all agents,
identifying bottlenecks and optimization opportunities.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for agent execution."""
    agent_name: str
    execution_time: float
    memory_usage: float
    token_count: int
    confidence_score: float
    cache_hits: int = 0
    cache_misses: int = 0

class AgentPerformanceProfiler:
    """Profile performance of individual agents."""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.bottlenecks: Dict[str, List[str]] = {}

    async def profile_agent(self, agent, input_data: str, context: Dict[str, Any]) -> PerformanceMetrics:
        """Profile a single agent's performance."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Execute agent
        result = await agent.process(input_data, context)

        # Calculate metrics
        execution_time = time.perf_counter() - start_time
        memory_delta = self._get_memory_usage() - start_memory

        metrics = PerformanceMetrics(
            agent_name=agent.name,
            execution_time=execution_time,
            memory_usage=memory_delta,
            token_count=self._estimate_tokens(input_data, result),
            confidence_score=result.get('metadata', {}).get('confidence', 0.0)
        )

        self.metrics_history.append(metrics)
        self._identify_bottlenecks(metrics)

        return metrics

    def _identify_bottlenecks(self, metrics: PerformanceMetrics):
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Slow execution (>2 seconds)
        if metrics.execution_time > 2.0:
            bottlenecks.append(f"Slow execution: {metrics.execution_time:.2f}s")

        # High memory usage (>100MB)
        if metrics.memory_usage > 100_000_000:
            bottlenecks.append(f"High memory: {metrics.memory_usage / 1_000_000:.1f}MB")

        # High token usage (>1000 tokens)
        if metrics.token_count > 1000:
            bottlenecks.append(f"High token usage: {metrics.token_count}")

        # Low confidence (<0.5)
        if metrics.confidence_score < 0.5:
            bottlenecks.append(f"Low confidence: {metrics.confidence_score:.2f}")

        if bottlenecks:
            self.bottlenecks[metrics.agent_name] = bottlenecks

    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0.0

    def _estimate_tokens(self, input_data: Any, output_data: Any) -> int:
        """Estimate token count for LLM usage."""
        # Rough estimation: 1 token per 4 characters
        input_str = str(input_data)
        output_str = str(output_data)
        return (len(input_str) + len(output_str)) // 4

    def get_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Generate optimization recommendations based on profiling."""
        recommendations = {}

        for agent_name, issues in self.bottlenecks.items():
            agent_recommendations = []

            for issue in issues:
                if "Slow execution" in issue:
                    agent_recommendations.append("Consider caching frequent queries")
                    agent_recommendations.append("Implement async operations")
                    agent_recommendations.append("Use lighter model for simple tasks")

                elif "High memory" in issue:
                    agent_recommendations.append("Implement memory pooling")
                    agent_recommendations.append("Clear unused references")
                    agent_recommendations.append("Use streaming for large data")

                elif "High token usage" in issue:
                    agent_recommendations.append("Implement context compression")
                    agent_recommendations.append("Use semantic truncation")
                    agent_recommendations.append("Cache common responses")

                elif "Low confidence" in issue:
                    agent_recommendations.append("Improve fallback strategies")
                    agent_recommendations.append("Add validation layers")
                    agent_recommendations.append("Enhance training data")

            if agent_recommendations:
                recommendations[agent_name] = list(set(agent_recommendations))

        return recommendations


class MultiAgentOrchestrationOptimizer:
    """Optimize multi-agent orchestration and coordination."""

    def __init__(self):
        self.parallel_execution_opportunities: List[Dict[str, Any]] = []
        self.redundant_operations: List[Dict[str, Any]] = []
        self.optimization_metrics = {
            'original_latency': 0.0,
            'optimized_latency': 0.0,
            'parallel_speedup': 0.0,
            'cache_hit_rate': 0.0
        }

    def analyze_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow for optimization opportunities."""
        agent_sequence = workflow.get('agent_sequence', [])

        # Identify parallel execution opportunities
        parallel_groups = self._identify_parallel_groups(agent_sequence)

        # Identify redundant operations
        redundancies = self._find_redundant_operations(agent_sequence)

        # Calculate potential speedup
        speedup = self._calculate_speedup(parallel_groups, agent_sequence)

        return {
            'parallel_groups': parallel_groups,
            'redundancies': redundancies,
            'potential_speedup': speedup,
            'optimizations': self._generate_optimizations(parallel_groups, redundancies)
        }

    def _identify_parallel_groups(self, agent_sequence: List[str]) -> List[List[str]]:
        """Identify agents that can run in parallel."""
        # Agents with no dependencies can run in parallel
        parallel_groups = []

        # Group 1: Initial analysis agents (can run in parallel)
        initial_agents = ['emotion_agent', 'personality_agent', 'safety_agent']
        if all(agent in agent_sequence for agent in initial_agents):
            parallel_groups.append(initial_agents)

        # Group 2: Support agents (can run in parallel)
        support_agents = ['search_agent', 'crawler_agent']
        if any(agent in agent_sequence for agent in support_agents):
            parallel_groups.append([a for a in support_agents if a in agent_sequence])

        return parallel_groups

    def _find_redundant_operations(self, agent_sequence: List[str]) -> List[Dict[str, Any]]:
        """Find redundant or duplicate operations."""
        redundancies = []

        # Check for duplicate vector DB queries
        vector_db_agents = ['search_agent', 'crawler_agent', 'therapy_agent']
        active_vdb_agents = [a for a in vector_db_agents if a in agent_sequence]

        if len(active_vdb_agents) > 1:
            redundancies.append({
                'type': 'vector_db_queries',
                'agents': active_vdb_agents,
                'recommendation': 'Implement shared vector DB cache'
            })

        # Check for duplicate sentiment analysis
        sentiment_agents = ['emotion_agent', 'safety_agent']
        active_sentiment = [a for a in sentiment_agents if a in agent_sequence]

        if len(active_sentiment) > 1:
            redundancies.append({
                'type': 'sentiment_analysis',
                'agents': active_sentiment,
                'recommendation': 'Share sentiment analysis results'
            })

        return redundancies

    def _calculate_speedup(self, parallel_groups: List[List[str]],
                          agent_sequence: List[str]) -> float:
        """Calculate potential speedup from parallelization."""
        if not parallel_groups:
            return 1.0

        # Assume each agent takes 1 unit of time
        sequential_time = len(agent_sequence)

        # Calculate parallel time
        parallel_time = sequential_time
        for group in parallel_groups:
            # Parallel group takes time of slowest agent (1 unit)
            time_saved = len(group) - 1
            parallel_time -= time_saved

        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        return speedup

    def _generate_optimizations(self, parallel_groups: List[List[str]],
                               redundancies: List[Dict[str, Any]]) -> List[str]:
        """Generate specific optimization recommendations."""
        optimizations = []

        if parallel_groups:
            optimizations.append(f"Parallelize {len(parallel_groups)} agent groups for {len(sum(parallel_groups, []))-len(parallel_groups)}x speedup")

        if redundancies:
            optimizations.append(f"Eliminate {len(redundancies)} redundant operations")

        optimizations.extend([
            "Implement result caching with 5-minute TTL",
            "Use lighter models for low-complexity tasks",
            "Pre-warm agent contexts for faster startup",
            "Batch vector DB queries for efficiency"
        ])

        return optimizations


class CostOptimizationAnalyzer:
    """Analyze and optimize LLM usage costs."""

    # Model costs per 1K tokens (approximate)
    MODEL_COSTS = {
        'gpt-4': 0.03,
        'gpt-3.5-turbo': 0.002,
        'claude-3-opus': 0.015,
        'claude-3-sonnet': 0.003,
        'claude-3-haiku': 0.00025,
        'gemini-pro': 0.001
    }

    def __init__(self):
        self.token_usage: Dict[str, int] = {}
        self.cost_by_agent: Dict[str, float] = {}

    def analyze_cost(self, agent_name: str, model: str, tokens: int) -> float:
        """Analyze cost for an agent's LLM usage."""
        cost_per_1k = self.MODEL_COSTS.get(model, 0.01)
        cost = (tokens / 1000) * cost_per_1k

        # Track usage
        self.token_usage[agent_name] = self.token_usage.get(agent_name, 0) + tokens
        self.cost_by_agent[agent_name] = self.cost_by_agent.get(agent_name, 0.0) + cost

        return cost

    def recommend_model_optimization(self) -> Dict[str, str]:
        """Recommend optimal models for each agent based on complexity."""
        recommendations = {}

        # High complexity agents need powerful models
        high_complexity = ['therapy_agent', 'diagnosis_agent', 'chat_agent']
        medium_complexity = ['emotion_agent', 'personality_agent', 'safety_agent']
        low_complexity = ['search_agent', 'crawler_agent']

        for agent in high_complexity:
            recommendations[agent] = 'claude-3-sonnet'  # Balance of quality and cost

        for agent in medium_complexity:
            recommendations[agent] = 'claude-3-haiku'  # Fast and cheap

        for agent in low_complexity:
            recommendations[agent] = 'gemini-pro'  # Very cheap for simple tasks

        return recommendations

    def calculate_savings(self, current_model: str, recommended_model: str,
                         monthly_tokens: int) -> float:
        """Calculate potential monthly savings from model optimization."""
        current_cost = (monthly_tokens / 1000) * self.MODEL_COSTS.get(current_model, 0.01)
        optimized_cost = (monthly_tokens / 1000) * self.MODEL_COSTS.get(recommended_model, 0.01)
        savings = current_cost - optimized_cost
        return max(0, savings)