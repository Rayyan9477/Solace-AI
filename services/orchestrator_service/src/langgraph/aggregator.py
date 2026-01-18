"""
Solace-AI Orchestrator Service - Aggregator.
Aggregates and synthesizes responses from multiple agents.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .state_schema import (
    OrchestratorState,
    AgentType,
    ProcessingPhase,
    AgentResult,
    MessageEntry,
)

logger = structlog.get_logger(__name__)


class AggregationStrategy(str, Enum):
    """Strategies for response aggregation."""
    PRIORITY_BASED = "priority_based"
    WEIGHTED_MERGE = "weighted_merge"
    CONSENSUS = "consensus"
    FIRST_SUCCESS = "first_success"


class AggregatorSettings(BaseSettings):
    """Configuration for the aggregator."""
    strategy: str = Field(default="priority_based")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_response_length: int = Field(default=1500, ge=100, le=5000)
    enable_response_merging: bool = Field(default=True)
    safety_priority_boost: float = Field(default=0.3, ge=0.0, le=1.0)
    therapy_priority_boost: float = Field(default=0.2, ge=0.0, le=1.0)
    fallback_response: str = Field(default="I'm here to support you. How can I help?")
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_AGGREGATOR_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class AgentContribution:
    """Contribution from a single agent."""
    agent_type: AgentType
    content: str
    confidence: float
    priority_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_type": self.agent_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "priority_score": self.priority_score,
            "metadata": self.metadata,
        }


@dataclass
class AggregationResult:
    """Result of response aggregation."""
    final_content: str
    primary_source: AgentType
    contributing_agents: list[AgentType]
    overall_confidence: float
    strategy_used: AggregationStrategy
    contributions: list[AgentContribution]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_content": self.final_content,
            "primary_source": self.primary_source.value,
            "contributing_agents": [a.value for a in self.contributing_agents],
            "overall_confidence": self.overall_confidence,
            "strategy_used": self.strategy_used.value,
            "contributions": [c.to_dict() for c in self.contributions],
            "metadata": self.metadata,
        }


class ResponseRanker:
    """Ranks agent responses by priority and quality."""

    AGENT_BASE_PRIORITIES: dict[AgentType, float] = {
        AgentType.SAFETY: 1.0,
        AgentType.THERAPY: 0.85,
        AgentType.DIAGNOSIS: 0.80,
        AgentType.CHAT: 0.60,
        AgentType.PERSONALITY: 0.50,
        AgentType.SUPERVISOR: 0.40,
        AgentType.AGGREGATOR: 0.30,
    }

    def __init__(self, settings: AggregatorSettings) -> None:
        self._settings = settings

    def rank(self, results: list[dict[str, Any]]) -> list[AgentContribution]:
        """Rank agent results by priority."""
        contributions: list[AgentContribution] = []
        for result in results:
            content = result.get("response_content")
            if not content:
                continue
            agent_type = AgentType(result.get("agent_type", "chat"))
            confidence = result.get("confidence", 0.5)
            base_priority = self.AGENT_BASE_PRIORITIES.get(agent_type, 0.5)
            priority_score = self._calculate_priority(
                agent_type, base_priority, confidence
            )
            contributions.append(AgentContribution(
                agent_type=agent_type,
                content=content,
                confidence=confidence,
                priority_score=priority_score,
                metadata=result.get("metadata", {}),
            ))
        contributions.sort(key=lambda c: c.priority_score, reverse=True)
        return contributions

    def _calculate_priority(
        self,
        agent_type: AgentType,
        base_priority: float,
        confidence: float,
    ) -> float:
        """Calculate final priority score."""
        priority = base_priority * 0.6 + confidence * 0.4
        if agent_type == AgentType.SAFETY:
            priority += self._settings.safety_priority_boost
        elif agent_type == AgentType.THERAPY:
            priority += self._settings.therapy_priority_boost
        return min(priority, 1.0)


class ResponseMerger:
    """Merges multiple agent responses into coherent output."""

    def __init__(self, settings: AggregatorSettings) -> None:
        self._settings = settings

    def merge(
        self,
        contributions: list[AgentContribution],
        strategy: AggregationStrategy,
    ) -> str:
        """Merge contributions based on strategy."""
        if not contributions:
            return self._settings.fallback_response
        if strategy == AggregationStrategy.FIRST_SUCCESS:
            return self._first_success_merge(contributions)
        if strategy == AggregationStrategy.PRIORITY_BASED:
            return self._priority_based_merge(contributions)
        if strategy == AggregationStrategy.WEIGHTED_MERGE:
            return self._weighted_merge(contributions)
        return self._priority_based_merge(contributions)

    def _first_success_merge(self, contributions: list[AgentContribution]) -> str:
        """Return first successful response."""
        for contrib in contributions:
            if contrib.confidence >= self._settings.confidence_threshold:
                return contrib.content
        return contributions[0].content if contributions else self._settings.fallback_response

    def _priority_based_merge(self, contributions: list[AgentContribution]) -> str:
        """Merge based on priority ordering."""
        if len(contributions) == 1:
            return contributions[0].content
        primary = contributions[0]
        result = primary.content
        if len(contributions) > 1 and self._settings.enable_response_merging:
            secondary = contributions[1]
            if self._should_append_secondary(primary, secondary):
                supplement = self._extract_supplement(secondary.content, primary.content)
                if supplement:
                    result = f"{result}\n\n{supplement}"
        return self._truncate(result)

    def _weighted_merge(self, contributions: list[AgentContribution]) -> str:
        """Merge with weighted combination."""
        if len(contributions) == 1:
            return contributions[0].content
        high_priority = [c for c in contributions if c.priority_score >= 0.7]
        if high_priority:
            return self._priority_based_merge(high_priority)
        return self._priority_based_merge(contributions)

    def _should_append_secondary(
        self,
        primary: AgentContribution,
        secondary: AgentContribution,
    ) -> bool:
        """Determine if secondary content should be appended."""
        if primary.agent_type == AgentType.SAFETY:
            return False
        if secondary.agent_type == AgentType.PERSONALITY:
            return False
        if secondary.confidence < self._settings.confidence_threshold:
            return False
        return True

    def _extract_supplement(self, secondary: str, primary: str) -> str:
        """Extract supplementary content from secondary response."""
        secondary_sentences = secondary.split(". ")
        primary_lower = primary.lower()
        unique_parts: list[str] = []
        for sentence in secondary_sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) > 20 and sentence_lower not in primary_lower:
                unique_parts.append(sentence.strip())
        if unique_parts:
            supplement = ". ".join(unique_parts[:2])
            if not supplement.endswith("."):
                supplement += "."
            return supplement
        return ""

    def _truncate(self, content: str) -> str:
        """Truncate content to max length."""
        if len(content) <= self._settings.max_response_length:
            return content
        truncated = content[:self._settings.max_response_length]
        last_sentence = truncated.rfind(". ")
        if last_sentence > self._settings.max_response_length // 2:
            truncated = truncated[:last_sentence + 1]
        return truncated


class Aggregator:
    """Aggregates responses from multiple agents into final output."""

    def __init__(self, settings: AggregatorSettings | None = None) -> None:
        self._settings = settings or AggregatorSettings()
        self._ranker = ResponseRanker(self._settings)
        self._merger = ResponseMerger(self._settings)
        self._aggregation_count = 0

    def aggregate(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Aggregate agent results into final response.
        Main LangGraph node function.
        """
        self._aggregation_count += 1
        agent_results = state.get("agent_results", [])
        personality_style = state.get("personality_style", {})
        safety_flags = state.get("safety_flags", {})
        logger.info(
            "aggregator_processing",
            result_count=len(agent_results),
            has_personality=bool(personality_style),
        )
        result = self._perform_aggregation(
            agent_results, safety_flags, personality_style
        )
        response_msg = MessageEntry.assistant_message(
            content=result.final_content,
            metadata={
                "aggregated": True,
                "primary_source": result.primary_source.value,
                "contributing_agents": [a.value for a in result.contributing_agents],
            },
        )
        agent_result = AgentResult(
            agent_type=AgentType.AGGREGATOR,
            success=True,
            response_content=result.final_content,
            confidence=result.overall_confidence,
            metadata={
                "strategy": result.strategy_used.value,
                "contributor_count": len(result.contributing_agents),
            },
        )
        logger.info(
            "aggregator_complete",
            primary_source=result.primary_source.value,
            contributor_count=len(result.contributing_agents),
            strategy=result.strategy_used.value,
            response_length=len(result.final_content),
        )
        return {
            "final_response": result.final_content,
            "messages": [response_msg.to_dict()],
            "processing_phase": ProcessingPhase.AGGREGATION.value,
            "agent_results": [agent_result.to_dict()],
            "metadata": {
                **state.get("metadata", {}),
                "aggregation": result.to_dict(),
            },
        }

    def _perform_aggregation(
        self,
        results: list[dict[str, Any]],
        safety_flags: dict[str, Any],
        personality_style: dict[str, Any],
    ) -> AggregationResult:
        """Perform the aggregation process."""
        contributions = self._ranker.rank(results)
        if not contributions:
            return AggregationResult(
                final_content=self._settings.fallback_response,
                primary_source=AgentType.CHAT,
                contributing_agents=[],
                overall_confidence=0.5,
                strategy_used=AggregationStrategy.FIRST_SUCCESS,
                contributions=[],
                metadata={"fallback_used": True},
            )
        strategy = self._select_strategy(contributions, safety_flags)
        final_content = self._merger.merge(contributions, strategy)
        primary_source = contributions[0].agent_type
        contributing_agents = list({c.agent_type for c in contributions})
        overall_confidence = self._calculate_overall_confidence(contributions)
        return AggregationResult(
            final_content=final_content,
            primary_source=primary_source,
            contributing_agents=contributing_agents,
            overall_confidence=overall_confidence,
            strategy_used=strategy,
            contributions=contributions,
            metadata={"result_count": len(results)},
        )

    def _select_strategy(
        self,
        contributions: list[AgentContribution],
        safety_flags: dict[str, Any],
    ) -> AggregationStrategy:
        """Select aggregation strategy based on context."""
        if safety_flags.get("crisis_detected"):
            return AggregationStrategy.FIRST_SUCCESS
        if len(contributions) == 1:
            return AggregationStrategy.FIRST_SUCCESS
        has_safety = any(c.agent_type == AgentType.SAFETY for c in contributions)
        if has_safety:
            return AggregationStrategy.PRIORITY_BASED
        return AggregationStrategy.PRIORITY_BASED

    def _calculate_overall_confidence(
        self,
        contributions: list[AgentContribution],
    ) -> float:
        """Calculate overall confidence from contributions."""
        if not contributions:
            return 0.5
        total_weight = sum(c.priority_score for c in contributions)
        weighted_confidence = sum(
            c.confidence * c.priority_score for c in contributions
        )
        if total_weight > 0:
            return weighted_confidence / total_weight
        return contributions[0].confidence

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "total_aggregations": self._aggregation_count,
            "settings": {
                "strategy": self._settings.strategy,
                "confidence_threshold": self._settings.confidence_threshold,
                "max_response_length": self._settings.max_response_length,
            },
        }


def aggregator_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for aggregation.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    aggregator = Aggregator()
    return aggregator.aggregate(state)
