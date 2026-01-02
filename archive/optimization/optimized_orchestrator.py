"""
Optimized Agent Orchestrator with Performance Enhancements

Implements parallel execution, caching, and intelligent resource management
for the multi-agent mental health system.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

from src.optimization.performance_profiler import (
    AgentPerformanceProfiler,
    MultiAgentOrchestrationOptimizer,
    CostOptimizationAnalyzer
)
from src.optimization.context_optimizer import (
    ContextWindowManager,
    AgentResultCache
)

logger = logging.getLogger(__name__)


class OptimizedAgentOrchestrator:
    """
    Enhanced orchestrator with performance optimizations for mental health chatbot.

    Key Optimizations:
    1. Parallel agent execution
    2. Result caching
    3. Context compression
    4. Dynamic model selection
    5. Performance monitoring
    """

    def __init__(self, agent_modules: Dict[str, Any], config: Dict[str, Any] = None):
        self.agent_modules = agent_modules
        self.config = config or {}
        self.profiler = AgentPerformanceProfiler()
        self.optimizer = MultiAgentOrchestrationOptimizer()
        self.cost_analyzer = CostOptimizationAnalyzer()
        self.context_manager = ContextWindowManager()

        # Load optimization config if available
        try:
            from src.config import OptimizationConfig
            self.opt_config = OptimizationConfig
            max_workers = self.opt_config.MAX_PARALLEL_WORKERS
            self.parallel_groups = self.opt_config.get_parallel_groups()
        except ImportError:
            self.opt_config = None
            max_workers = 5
            # Default parallel execution groups
            self.parallel_groups = {
                'initial_assessment': ['emotion_agent', 'personality_agent', 'safety_agent'],
                'information_gathering': ['search_agent', 'crawler_agent'],
                'clinical_assessment': ['diagnosis_agent', 'therapy_agent']
            }

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Model selection strategy
        self.model_strategy = self.cost_analyzer.recommend_model_optimization()

        # Performance metrics
        self.metrics = {
            'total_executions': 0,
            'cache_hits': 0,
            'parallel_executions': 0,
            'average_latency': 0.0,
            'total_cost': 0.0
        }

    async def execute_optimized_workflow(
        self,
        workflow_id: str,
        input_data: str,
        context: Dict[str, Any] = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute workflow with performance optimizations.

        Args:
            workflow_id: ID of workflow to execute
            input_data: User input
            context: Execution context
            session_id: Session identifier

        Returns:
            Optimized execution results with performance metrics
        """
        start_time = time.perf_counter()
        self.metrics['total_executions'] += 1

        # Get workflow definition
        workflow = self._get_workflow(workflow_id)
        agent_sequence = workflow['agent_sequence']

        # Analyze workflow for optimization
        optimization_analysis = self.optimizer.analyze_workflow(workflow)

        # Initialize results
        results = {}
        full_context = context or {}

        # Execute with optimizations
        if optimization_analysis['parallel_groups']:
            results = await self._execute_with_parallelization(
                agent_sequence,
                input_data,
                full_context,
                optimization_analysis['parallel_groups']
            )
        else:
            results = await self._execute_sequential(
                agent_sequence,
                input_data,
                full_context
            )

        # Calculate metrics
        execution_time = time.perf_counter() - start_time
        self._update_metrics(execution_time, results)

        # Generate optimization report
        optimization_report = self._generate_optimization_report(
            execution_time,
            optimization_analysis,
            results
        )

        return {
            'status': 'success',
            'results': results,
            'execution_time': execution_time,
            'optimization_report': optimization_report,
            'session_id': session_id or f'optimized_{int(time.time())}'
        }

    async def _execute_with_parallelization(
        self,
        agent_sequence: List[str],
        input_data: str,
        context: Dict[str, Any],
        parallel_groups: List[List[str]]
    ) -> Dict[str, Any]:
        """Execute workflow with parallel agent execution."""
        results = {}
        executed_agents = set()

        # Execute parallel groups
        for group in parallel_groups:
            # Check if all agents in group are in sequence and not yet executed
            group_agents = [a for a in group if a in agent_sequence and a not in executed_agents]

            if group_agents:
                # Execute group in parallel
                group_results = await self._execute_parallel_group(
                    group_agents,
                    input_data,
                    context
                )

                # Update results and context
                results.update(group_results)
                context.update(self._extract_context_from_results(group_results))
                executed_agents.update(group_agents)

                self.metrics['parallel_executions'] += 1

        # Execute remaining agents sequentially
        remaining_agents = [a for a in agent_sequence if a not in executed_agents]
        for agent_name in remaining_agents:
            # Check cache first
            use_cache, cached_result = self.context_manager.should_use_cache(
                agent_name, input_data, context
            )

            if use_cache:
                results[agent_name] = cached_result
                self.metrics['cache_hits'] += 1
                logger.info(f"Using cached result for {agent_name}")
            else:
                # Execute agent
                result = await self._execute_single_agent(agent_name, input_data, context)
                results[agent_name] = result

                # Cache result
                self.context_manager.cache_result(agent_name, input_data, context, result)

            # Update context for next agent
            context.update(self._extract_context_from_results({agent_name: results[agent_name]}))

        return results

    async def _execute_parallel_group(
        self,
        agents: List[str],
        input_data: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a group of agents in parallel."""
        tasks = []
        results = {}

        logger.info(f"Executing agents in parallel: {agents}")

        # Create tasks for each agent
        for agent_name in agents:
            # Check cache first
            use_cache, cached_result = self.context_manager.should_use_cache(
                agent_name, input_data, context
            )

            if use_cache:
                results[agent_name] = cached_result
                self.metrics['cache_hits'] += 1
                logger.info(f"Using cached result for {agent_name}")
            else:
                # Create async task
                task = asyncio.create_task(
                    self._execute_single_agent(agent_name, input_data, context.copy())
                )
                tasks.append((agent_name, task))

        # Wait for all tasks to complete
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result

                # Cache result
                self.context_manager.cache_result(agent_name, input_data, context, result)

            except Exception as e:
                logger.error(f"Error executing {agent_name}: {str(e)}")
                results[agent_name] = {
                    'error': str(e),
                    'status': 'failed'
                }

        return results

    async def _execute_sequential(
        self,
        agent_sequence: List[str],
        input_data: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute agents sequentially (fallback mode)."""
        results = {}

        for agent_name in agent_sequence:
            # Check cache
            use_cache, cached_result = self.context_manager.should_use_cache(
                agent_name, input_data, context
            )

            if use_cache:
                results[agent_name] = cached_result
                self.metrics['cache_hits'] += 1
            else:
                # Execute agent
                result = await self._execute_single_agent(agent_name, input_data, context)
                results[agent_name] = result

                # Cache result
                self.context_manager.cache_result(agent_name, input_data, context, result)

            # Update context for next agent
            context.update(self._extract_context_from_results({agent_name: results[agent_name]}))

        return results

    async def _execute_single_agent(
        self,
        agent_name: str,
        input_data: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single agent with optimizations."""
        agent = self.agent_modules.get(agent_name)

        if not agent:
            return {'error': f'Agent {agent_name} not found', 'status': 'failed'}

        # Optimize context for this agent
        optimized_context = self.context_manager.get_optimized_context(
            agent_name, context, input_data
        )

        # Profile agent execution
        start_time = time.perf_counter()

        try:
            # Execute agent
            result = await agent.process(input_data, optimized_context)

            # Track performance
            execution_time = time.perf_counter() - start_time
            token_count = self._estimate_tokens(input_data, result)

            # Analyze cost
            model = self.model_strategy.get(agent_name, 'claude-3-haiku')
            cost = self.cost_analyzer.analyze_cost(agent_name, model, token_count)

            # Add metrics to result
            if 'metadata' not in result:
                result['metadata'] = {}

            result['metadata'].update({
                'execution_time': execution_time,
                'token_count': token_count,
                'model_used': model,
                'cost': cost,
                'context_compressed': True
            })

            return result

        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed',
                'agent_name': agent_name
            }

    def _extract_context_from_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context from agent results."""
        context_updates = {}

        for agent_name, result in results.items():
            if isinstance(result, dict) and 'error' not in result:
                # Extract agent-specific context
                if agent_name == 'emotion_agent':
                    if 'emotion_data' in result:
                        context_updates['emotion'] = result['emotion_data']
                elif agent_name == 'safety_agent':
                    if 'safety_assessment' in result:
                        context_updates['safety'] = result['safety_assessment']
                elif agent_name == 'personality_agent':
                    if 'personality_profile' in result:
                        context_updates['personality'] = result['personality_profile']
                elif agent_name == 'diagnosis_agent':
                    if 'diagnosis' in result:
                        context_updates['diagnosis'] = result['diagnosis']
                elif agent_name == 'therapy_agent':
                    if 'therapeutic_recommendations' in result:
                        context_updates['therapy'] = result['therapeutic_recommendations']

        return context_updates

    def _get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow definition."""
        workflows = {
            'mental_health_assessment': {
                'agent_sequence': [
                    'emotion_agent', 'personality_agent', 'safety_agent',
                    'diagnosis_agent', 'therapy_agent', 'chat_agent'
                ]
            },
            'crisis_response': {
                'agent_sequence': [
                    'safety_agent', 'emotion_agent', 'therapy_agent', 'chat_agent'
                ]
            },
            'information_search': {
                'agent_sequence': [
                    'search_agent', 'crawler_agent', 'chat_agent'
                ]
            }
        }

        return workflows.get(workflow_id, workflows['mental_health_assessment'])

    def _estimate_tokens(self, input_data: Any, output: Any) -> int:
        """Estimate token count for LLM usage."""
        input_str = str(input_data)
        output_str = str(output)
        return (len(input_str) + len(output_str)) // 4

    def _update_metrics(self, execution_time: float, results: Dict[str, Any]):
        """Update performance metrics."""
        # Update average latency
        total_execs = self.metrics['total_executions']
        current_avg = self.metrics['average_latency']
        self.metrics['average_latency'] = (
            (current_avg * (total_execs - 1) + execution_time) / total_execs
        )

        # Update total cost
        for result in results.values():
            if isinstance(result, dict):
                cost = result.get('metadata', {}).get('cost', 0)
                self.metrics['total_cost'] += cost

    def _generate_optimization_report(
        self,
        execution_time: float,
        optimization_analysis: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        # Calculate actual speedup
        baseline_time = len(results) * 1.5  # Assume 1.5s per agent baseline
        actual_speedup = baseline_time / execution_time if execution_time > 0 else 1.0

        # Get cache statistics
        cache_stats = self.context_manager.cache.get_cache_stats()

        report = {
            'execution_metrics': {
                'execution_time': execution_time,
                'baseline_time': baseline_time,
                'actual_speedup': actual_speedup,
                'potential_speedup': optimization_analysis.get('potential_speedup', 1.0)
            },
            'cache_performance': {
                'hit_rate': cache_stats['hit_rate'],
                'hits': cache_stats['hit_count'],
                'misses': cache_stats['miss_count'],
                'cached_items': cache_stats['cached_items']
            },
            'cost_analysis': {
                'total_cost': self.metrics['total_cost'],
                'average_cost_per_execution': self.metrics['total_cost'] / max(1, self.metrics['total_executions']),
                'cost_by_agent': self.cost_analyzer.cost_by_agent
            },
            'optimization_opportunities': optimization_analysis.get('optimizations', []),
            'parallel_groups_executed': self.metrics['parallel_executions'],
            'recommendations': self._generate_recommendations(results)
        }

        return report

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on execution results."""
        recommendations = []

        # Check for slow agents
        slow_agents = []
        for agent_name, result in results.items():
            if isinstance(result, dict):
                exec_time = result.get('metadata', {}).get('execution_time', 0)
                if exec_time > 2.0:
                    slow_agents.append(agent_name)

        if slow_agents:
            recommendations.append(f"Optimize slow agents: {', '.join(slow_agents)}")

        # Check cache effectiveness
        cache_stats = self.context_manager.cache.get_cache_stats()
        if cache_stats['hit_rate'] < 0.3:
            recommendations.append("Increase cache TTL or improve cache key strategy")

        # Check for high token usage
        high_token_agents = []
        for agent_name, result in results.items():
            if isinstance(result, dict):
                tokens = result.get('metadata', {}).get('token_count', 0)
                if tokens > 1000:
                    high_token_agents.append(agent_name)

        if high_token_agents:
            recommendations.append(f"Implement context compression for: {', '.join(high_token_agents)}")

        # Model optimization
        recommendations.append("Consider using lighter models for simple classification tasks")

        return recommendations

    async def shutdown(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("Optimized orchestrator shutdown complete")