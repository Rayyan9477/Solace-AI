"""
Agent Performance Analyzer - Advanced Analysis and Optimization Framework

Implements comprehensive performance analysis, failure classification,
and improvement tracking for all agents in the mental health system.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Classification of agent failure modes."""
    INSTRUCTION_MISUNDERSTANDING = "instruction_misunderstanding"
    OUTPUT_FORMAT_ERROR = "output_format_error"
    CONTEXT_LOSS = "context_loss"
    TOOL_MISUSE = "tool_misuse"
    CONSTRAINT_VIOLATION = "constraint_violation"
    EDGE_CASE = "edge_case"
    TIMEOUT = "timeout"
    HALLUCINATION = "hallucination"
    INCOMPLETE_RESPONSE = "incomplete_response"


class PerformanceMetric(Enum):
    """Key performance metrics to track."""
    TASK_SUCCESS_RATE = "task_success_rate"
    AVERAGE_CORRECTIONS = "average_corrections"
    TOOL_EFFICIENCY = "tool_efficiency"
    USER_SATISFACTION = "user_satisfaction"
    RESPONSE_LATENCY = "response_latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    HALLUCINATION_RATE = "hallucination_rate"
    CONSISTENCY_SCORE = "consistency_score"
    SAFETY_SCORE = "safety_score"


@dataclass
class AgentInteraction:
    """Record of a single agent interaction."""
    interaction_id: str
    agent_name: str
    timestamp: datetime
    input_data: str
    output_data: Any
    context: Dict[str, Any]
    success: bool
    execution_time: float
    token_count: int
    corrections: int = 0
    user_feedback: Optional[str] = None
    failure_mode: Optional[FailureMode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline metrics for an agent."""
    agent_name: str
    task_success_rate: float
    average_corrections: float
    tool_efficiency: float
    user_satisfaction: float
    response_latency: float
    token_efficiency: float
    hallucination_rate: float
    consistency_score: float
    safety_score: float
    measurement_period: timedelta
    sample_size: int


class AgentPerformanceAnalyzer:
    """
    Comprehensive performance analysis for agent optimization.

    Tracks interactions, identifies patterns, and provides improvement recommendations.
    """

    def __init__(self):
        self.interactions: Dict[str, List[AgentInteraction]] = defaultdict(list)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.failure_patterns: Dict[str, Dict[FailureMode, int]] = defaultdict(lambda: defaultdict(int))
        self.feedback_patterns: Dict[str, List[str]] = defaultdict(list)
        self.improvement_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record_interaction(self, interaction: AgentInteraction):
        """Record a new agent interaction."""
        self.interactions[interaction.agent_name].append(interaction)

        # Track failure patterns
        if not interaction.success and interaction.failure_mode:
            self.failure_patterns[interaction.agent_name][interaction.failure_mode] += 1

        # Track feedback
        if interaction.user_feedback:
            self.feedback_patterns[interaction.agent_name].append(interaction.user_feedback)

    def analyze_performance(self, agent_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Comprehensive performance analysis for an agent.

        Args:
            agent_name: Name of the agent to analyze
            days: Number of days to include in analysis

        Returns:
            Detailed performance analysis with metrics and patterns
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        agent_interactions = [
            i for i in self.interactions[agent_name]
            if i.timestamp > cutoff_date
        ]

        if not agent_interactions:
            return {"error": f"No interactions found for {agent_name} in the last {days} days"}

        # Calculate metrics
        metrics = self._calculate_metrics(agent_interactions)

        # Identify patterns
        patterns = self._identify_patterns(agent_interactions)

        # Classify failures
        failure_analysis = self._analyze_failures(agent_interactions)

        # Analyze feedback
        feedback_analysis = self._analyze_feedback(agent_interactions)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, patterns, failure_analysis, feedback_analysis
        )

        return {
            "agent_name": agent_name,
            "analysis_period": f"{days} days",
            "total_interactions": len(agent_interactions),
            "metrics": metrics,
            "patterns": patterns,
            "failure_analysis": failure_analysis,
            "feedback_analysis": feedback_analysis,
            "recommendations": recommendations,
            "baseline_comparison": self._compare_to_baseline(agent_name, metrics)
        }

    def _calculate_metrics(self, interactions: List[AgentInteraction]) -> Dict[str, float]:
        """Calculate performance metrics from interactions."""
        if not interactions:
            return {}

        successful = [i for i in interactions if i.success]

        # Task success rate
        task_success_rate = len(successful) / len(interactions)

        # Average corrections
        corrections = [i.corrections for i in interactions]
        avg_corrections = sum(corrections) / len(corrections)

        # Response latency
        latencies = [i.execution_time for i in interactions]
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = np.percentile(latencies, 95) if latencies else 0

        # Token efficiency (output tokens per successful task)
        token_counts = [i.token_count for i in successful] if successful else [0]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        # Hallucination rate (based on corrections and failures)
        hallucination_interactions = [
            i for i in interactions
            if i.failure_mode == FailureMode.HALLUCINATION
        ]
        hallucination_rate = len(hallucination_interactions) / len(interactions)

        # Consistency score (based on similar inputs)
        consistency_score = self._calculate_consistency(interactions)

        # Safety score (no violations)
        safety_violations = [
            i for i in interactions
            if i.failure_mode == FailureMode.CONSTRAINT_VIOLATION
        ]
        safety_score = 1 - (len(safety_violations) / len(interactions))

        # User satisfaction (from feedback)
        satisfaction_score = self._calculate_satisfaction(interactions)

        return {
            "task_success_rate": task_success_rate,
            "average_corrections": avg_corrections,
            "average_latency": avg_latency,
            "p95_latency": p95_latency,
            "average_tokens": avg_tokens,
            "hallucination_rate": hallucination_rate,
            "consistency_score": consistency_score,
            "safety_score": safety_score,
            "user_satisfaction": satisfaction_score,
            "tool_efficiency": 0.85  # Placeholder - would calculate from tool usage
        }

    def _identify_patterns(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Identify patterns in agent interactions."""
        patterns = {
            "common_failures": [],
            "correction_patterns": [],
            "time_based_patterns": [],
            "context_patterns": []
        }

        # Common failure patterns
        failures = [i for i in interactions if not i.success]
        if failures:
            failure_modes = [f.failure_mode for f in failures if f.failure_mode]
            if failure_modes:
                mode_counts = Counter(failure_modes)
                patterns["common_failures"] = [
                    {"mode": mode.value, "count": count}
                    for mode, count in mode_counts.most_common(3)
                ]

        # Correction patterns
        high_correction = [i for i in interactions if i.corrections > 2]
        if high_correction:
            # Analyze what types of inputs lead to corrections
            patterns["correction_patterns"] = self._analyze_correction_patterns(high_correction)

        # Time-based patterns (performance by hour of day)
        hourly_success = defaultdict(list)
        for interaction in interactions:
            hour = interaction.timestamp.hour
            hourly_success[hour].append(1 if interaction.success else 0)

        patterns["time_based_patterns"] = {
            hour: sum(successes) / len(successes) if successes else 0
            for hour, successes in hourly_success.items()
        }

        return patterns

    def _analyze_failures(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Detailed analysis of failure modes."""
        failures = [i for i in interactions if not i.success]

        if not failures:
            return {"failure_rate": 0, "failures_by_mode": {}}

        failure_rate = len(failures) / len(interactions)

        # Group by failure mode
        failures_by_mode = defaultdict(list)
        for failure in failures:
            if failure.failure_mode:
                failures_by_mode[failure.failure_mode.value].append({
                    "input": failure.input_data[:100],  # First 100 chars
                    "timestamp": failure.timestamp.isoformat(),
                    "execution_time": failure.execution_time
                })

        # Find root causes
        root_causes = self._identify_root_causes(failures)

        return {
            "failure_rate": failure_rate,
            "total_failures": len(failures),
            "failures_by_mode": dict(failures_by_mode),
            "root_causes": root_causes,
            "recovery_suggestions": self._generate_recovery_suggestions(failures_by_mode)
        }

    def _analyze_feedback(self, interactions: List[AgentInteraction]) -> Dict[str, Any]:
        """Analyze user feedback patterns."""
        feedback_items = [i for i in interactions if i.user_feedback]

        if not feedback_items:
            return {"feedback_count": 0, "patterns": []}

        # Simple sentiment analysis (would use NLP in production)
        positive_keywords = ['good', 'great', 'helpful', 'perfect', 'thanks']
        negative_keywords = ['bad', 'wrong', 'incorrect', 'useless', 'terrible']
        improvement_keywords = ['could', 'should', 'better', 'improve', 'suggest']

        positive_count = 0
        negative_count = 0
        improvement_count = 0

        for item in feedback_items:
            feedback_lower = item.user_feedback.lower()
            if any(kw in feedback_lower for kw in positive_keywords):
                positive_count += 1
            if any(kw in feedback_lower for kw in negative_keywords):
                negative_count += 1
            if any(kw in feedback_lower for kw in improvement_keywords):
                improvement_count += 1

        return {
            "feedback_count": len(feedback_items),
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "improvement_suggestions": improvement_count,
            "feedback_rate": len(feedback_items) / len(interactions),
            "sample_feedback": [f.user_feedback[:200] for f in feedback_items[:5]]
        }

    def _generate_recommendations(self, metrics: Dict[str, float],
                                 patterns: Dict[str, Any],
                                 failure_analysis: Dict[str, Any],
                                 feedback_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific improvement recommendations."""
        recommendations = []

        # Success rate recommendations
        if metrics.get("task_success_rate", 0) < 0.8:
            recommendations.append({
                "priority": "HIGH",
                "category": "Success Rate",
                "issue": f"Task success rate is {metrics['task_success_rate']:.1%}",
                "recommendation": "Enhance prompt clarity and add more few-shot examples",
                "expected_impact": "15-20% improvement in success rate"
            })

        # Latency recommendations
        if metrics.get("average_latency", 0) > 2.0:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Performance",
                "issue": f"Average latency is {metrics['average_latency']:.1f}s",
                "recommendation": "Implement response streaming and optimize context size",
                "expected_impact": "30-40% reduction in latency"
            })

        # Hallucination recommendations
        if metrics.get("hallucination_rate", 0) > 0.05:
            recommendations.append({
                "priority": "HIGH",
                "category": "Accuracy",
                "issue": f"Hallucination rate is {metrics['hallucination_rate']:.1%}",
                "recommendation": "Add fact-checking validators and constitutional AI principles",
                "expected_impact": "70% reduction in hallucinations"
            })

        # Safety recommendations
        if metrics.get("safety_score", 1) < 0.95:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "Safety",
                "issue": f"Safety score is {metrics['safety_score']:.1%}",
                "recommendation": "Strengthen safety constraints and add validation layers",
                "expected_impact": "Achieve 99%+ safety compliance"
            })

        # Pattern-based recommendations
        if patterns.get("common_failures"):
            top_failure = patterns["common_failures"][0]
            recommendations.append({
                "priority": "HIGH",
                "category": "Failure Patterns",
                "issue": f"Most common failure: {top_failure['mode']} ({top_failure['count']} occurrences)",
                "recommendation": f"Add specific handling for {top_failure['mode']} scenarios",
                "expected_impact": "50% reduction in this failure type"
            })

        return recommendations

    def _calculate_consistency(self, interactions: List[AgentInteraction]) -> float:
        """Calculate consistency score based on similar inputs."""
        # Group interactions by similar inputs (simplified - would use embeddings in production)
        input_groups = defaultdict(list)
        for interaction in interactions:
            # Simple grouping by first 50 chars
            key = interaction.input_data[:50]
            input_groups[key].append(interaction)

        # Calculate consistency within groups
        consistency_scores = []
        for group in input_groups.values():
            if len(group) > 1:
                # Check if outputs are similar (simplified)
                success_rate = sum(1 for i in group if i.success) / len(group)
                consistency_scores.append(success_rate)

        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0

    def _calculate_satisfaction(self, interactions: List[AgentInteraction]) -> float:
        """Calculate user satisfaction from various signals."""
        signals = []

        for interaction in interactions:
            score = 0.5  # Neutral baseline

            # Success adds to satisfaction
            if interaction.success:
                score += 0.3

            # Corrections reduce satisfaction
            score -= min(0.3, interaction.corrections * 0.1)

            # Quick responses improve satisfaction
            if interaction.execution_time < 1.0:
                score += 0.1
            elif interaction.execution_time > 3.0:
                score -= 0.1

            # Explicit feedback
            if interaction.user_feedback:
                if 'good' in interaction.user_feedback.lower():
                    score = min(1.0, score + 0.2)
                elif 'bad' in interaction.user_feedback.lower():
                    score = max(0.0, score - 0.2)

            signals.append(max(0.0, min(1.0, score)))

        return sum(signals) / len(signals) if signals else 0.5

    def _analyze_correction_patterns(self, high_correction_interactions: List[AgentInteraction]) -> List[str]:
        """Identify patterns in interactions requiring corrections."""
        patterns = []

        # Check for common input patterns
        input_lengths = [len(i.input_data) for i in high_correction_interactions]
        avg_length = sum(input_lengths) / len(input_lengths)

        if avg_length > 500:
            patterns.append("Long inputs tend to require more corrections")

        # Check for complex queries (multiple questions)
        multi_question = [i for i in high_correction_interactions if '?' in i.input_data and i.input_data.count('?') > 1]
        if len(multi_question) > len(high_correction_interactions) * 0.3:
            patterns.append("Multi-question queries often need corrections")

        # Check for specific topics (would use NLP in production)
        technical_terms = ['technical', 'complex', 'detailed', 'specific']
        technical_queries = [
            i for i in high_correction_interactions
            if any(term in i.input_data.lower() for term in technical_terms)
        ]
        if len(technical_queries) > len(high_correction_interactions) * 0.4:
            patterns.append("Technical queries require more refinement")

        return patterns

    def _identify_root_causes(self, failures: List[AgentInteraction]) -> List[str]:
        """Identify root causes of failures."""
        root_causes = []

        # Check for timeout issues
        timeout_failures = [f for f in failures if f.execution_time > 10.0]
        if timeout_failures:
            root_causes.append(f"Timeout issues affecting {len(timeout_failures)} failures")

        # Check for context issues
        high_context = [f for f in failures if f.context and len(str(f.context)) > 10000]
        if high_context:
            root_causes.append(f"Context overflow in {len(high_context)} failures")

        # Check for specific failure modes
        for mode in FailureMode:
            mode_failures = [f for f in failures if f.failure_mode == mode]
            if len(mode_failures) > len(failures) * 0.2:  # More than 20% of failures
                root_causes.append(f"{mode.value}: {len(mode_failures)} occurrences")

        return root_causes

    def _generate_recovery_suggestions(self, failures_by_mode: Dict[str, List]) -> List[str]:
        """Generate suggestions for recovering from failures."""
        suggestions = []

        for mode, failures in failures_by_mode.items():
            if mode == FailureMode.INSTRUCTION_MISUNDERSTANDING.value:
                suggestions.append("Clarify role definition and add instruction examples")
            elif mode == FailureMode.OUTPUT_FORMAT_ERROR.value:
                suggestions.append("Add output format validators and templates")
            elif mode == FailureMode.CONTEXT_LOSS.value:
                suggestions.append("Implement context compression and prioritization")
            elif mode == FailureMode.TOOL_MISUSE.value:
                suggestions.append("Enhance tool usage examples and constraints")
            elif mode == FailureMode.CONSTRAINT_VIOLATION.value:
                suggestions.append("Strengthen safety checks and add pre-validation")
            elif mode == FailureMode.HALLUCINATION.value:
                suggestions.append("Add fact-checking and source verification")

        return suggestions

    def _compare_to_baseline(self, agent_name: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics to baseline."""
        if agent_name not in self.baselines:
            return {"status": "No baseline established"}

        baseline = self.baselines[agent_name]
        comparison = {}

        # Compare each metric
        metrics_map = {
            "task_success_rate": baseline.task_success_rate,
            "average_corrections": baseline.average_corrections,
            "average_latency": baseline.response_latency,
            "hallucination_rate": baseline.hallucination_rate,
            "safety_score": baseline.safety_score,
            "user_satisfaction": baseline.user_satisfaction
        }

        for metric_name, baseline_value in metrics_map.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                change = ((current_value - baseline_value) / baseline_value * 100) if baseline_value else 0

                # Determine if improvement or regression
                is_improvement = True
                if metric_name in ["average_corrections", "average_latency", "hallucination_rate"]:
                    is_improvement = change < 0  # Lower is better
                else:
                    is_improvement = change > 0  # Higher is better

                comparison[metric_name] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "change_percent": change,
                    "status": "IMPROVED" if is_improvement else "REGRESSED"
                }

        return comparison

    def establish_baseline(self, agent_name: str, interactions: List[AgentInteraction]):
        """Establish performance baseline for an agent."""
        metrics = self._calculate_metrics(interactions)

        baseline = PerformanceBaseline(
            agent_name=agent_name,
            task_success_rate=metrics.get("task_success_rate", 0),
            average_corrections=metrics.get("average_corrections", 0),
            tool_efficiency=metrics.get("tool_efficiency", 0),
            user_satisfaction=metrics.get("user_satisfaction", 0.5),
            response_latency=metrics.get("average_latency", 0),
            token_efficiency=metrics.get("average_tokens", 0),
            hallucination_rate=metrics.get("hallucination_rate", 0),
            consistency_score=metrics.get("consistency_score", 0),
            safety_score=metrics.get("safety_score", 1.0),
            measurement_period=timedelta(days=30),
            sample_size=len(interactions)
        )

        self.baselines[agent_name] = baseline
        return baseline

    def generate_improvement_report(self, agent_name: str) -> str:
        """Generate comprehensive improvement report for an agent."""
        analysis = self.analyze_performance(agent_name)

        if "error" in analysis:
            return f"Error: {analysis['error']}"

        report = f"""
# Performance Improvement Report: {agent_name}

## Executive Summary
- **Analysis Period**: {analysis['analysis_period']}
- **Total Interactions**: {analysis['total_interactions']}
- **Overall Success Rate**: {analysis['metrics']['task_success_rate']:.1%}
- **User Satisfaction**: {analysis['metrics']['user_satisfaction']:.1%}

## Key Metrics
"""

        # Add metrics table
        metrics = analysis['metrics']
        report += f"""
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Success Rate | {metrics['task_success_rate']:.1%} | 85% | {'✅' if metrics['task_success_rate'] > 0.85 else '⚠️'} |
| Avg Corrections | {metrics['average_corrections']:.1f} | <1.0 | {'✅' if metrics['average_corrections'] < 1.0 else '⚠️'} |
| Avg Latency | {metrics['average_latency']:.1f}s | <2.0s | {'✅' if metrics['average_latency'] < 2.0 else '⚠️'} |
| Safety Score | {metrics['safety_score']:.1%} | 99% | {'✅' if metrics['safety_score'] > 0.99 else '⚠️'} |

## Failure Analysis
"""

        # Add failure analysis
        failure_analysis = analysis['failure_analysis']
        report += f"""
- **Failure Rate**: {failure_analysis['failure_rate']:.1%}
- **Total Failures**: {failure_analysis['total_failures']}

### Root Causes:
"""
        for cause in failure_analysis.get('root_causes', []):
            report += f"- {cause}\n"

        # Add recommendations
        report += "\n## Recommendations\n\n"
        for rec in analysis['recommendations']:
            report += f"""
### {rec['priority']}: {rec['category']}
- **Issue**: {rec['issue']}
- **Recommendation**: {rec['recommendation']}
- **Expected Impact**: {rec['expected_impact']}
"""

        # Add baseline comparison if available
        if 'baseline_comparison' in analysis and analysis['baseline_comparison'].get('status') != "No baseline established":
            report += "\n## Baseline Comparison\n\n"
            report += "| Metric | Baseline | Current | Change | Status |\n"
            report += "|--------|----------|---------|--------|--------|\n"

            for metric, data in analysis['baseline_comparison'].items():
                if isinstance(data, dict):
                    report += f"| {metric} | {data['baseline']:.2f} | {data['current']:.2f} | "
                    report += f"{data['change_percent']:+.1f}% | {data['status']} |\n"

        return report