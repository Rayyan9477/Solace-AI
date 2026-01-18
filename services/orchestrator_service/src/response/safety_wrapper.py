"""
Solace-AI Orchestrator Service - Safety Wrapper.
Wraps responses with safety resources and crisis information when needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..langgraph.state_schema import (
    OrchestratorState,
    AgentType,
    ProcessingPhase,
    AgentResult,
    RiskLevel,
)

logger = structlog.get_logger(__name__)


class ResourceType(str, Enum):
    """Types of crisis resources."""
    HOTLINE = "hotline"
    TEXT_LINE = "text_line"
    CHAT = "chat"
    WEBSITE = "website"
    LOCAL = "local"


class SafetyWrapperSettings(BaseSettings):
    """Configuration for the safety wrapper."""
    enable_resource_injection: bool = Field(default=True)
    enable_content_filtering: bool = Field(default=True)
    enable_disclaimer_injection: bool = Field(default=True)
    resource_threshold: str = Field(default="moderate")
    always_include_988: bool = Field(default=True)
    max_resources_shown: int = Field(default=3, ge=1, le=5)
    filter_harmful_phrases: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_SAFETY_WRAPPER_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class CrisisResource:
    """A crisis resource for user safety."""
    name: str
    resource_type: ResourceType
    contact: str
    description: str
    available_24_7: bool = True
    region: str = "US"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "resource_type": self.resource_type.value,
            "contact": self.contact,
            "description": self.description,
            "available_24_7": self.available_24_7,
            "region": self.region,
        }

    def format_display(self) -> str:
        """Format for display in response."""
        availability = " (24/7)" if self.available_24_7 else ""
        return f"- **{self.name}**: {self.contact}{availability}"


@dataclass
class SafetyWrapResult:
    """Result of safety wrapping."""
    content: str
    original_content: str
    resources_added: list[CrisisResource]
    content_filtered: bool
    disclaimer_added: bool
    risk_level: RiskLevel
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "original_content": self.original_content,
            "resources_added": [r.to_dict() for r in self.resources_added],
            "content_filtered": self.content_filtered,
            "disclaimer_added": self.disclaimer_added,
            "risk_level": self.risk_level.value,
            "metadata": self.metadata,
        }


class ResourceProvider:
    """Provides crisis resources based on context."""

    CRISIS_RESOURCES: list[CrisisResource] = [
        CrisisResource(
            name="988 Suicide & Crisis Lifeline",
            resource_type=ResourceType.HOTLINE,
            contact="Call or text 988",
            description="Free, confidential support 24/7",
            available_24_7=True,
        ),
        CrisisResource(
            name="Crisis Text Line",
            resource_type=ResourceType.TEXT_LINE,
            contact="Text HOME to 741741",
            description="Free crisis support via text",
            available_24_7=True,
        ),
        CrisisResource(
            name="National Domestic Violence Hotline",
            resource_type=ResourceType.HOTLINE,
            contact="1-800-799-7233",
            description="Support for domestic violence",
            available_24_7=True,
        ),
        CrisisResource(
            name="SAMHSA National Helpline",
            resource_type=ResourceType.HOTLINE,
            contact="1-800-662-4357",
            description="Mental health and substance abuse",
            available_24_7=True,
        ),
        CrisisResource(
            name="Trevor Project",
            resource_type=ResourceType.HOTLINE,
            contact="1-866-488-7386",
            description="LGBTQ+ youth crisis support",
            available_24_7=True,
        ),
    ]

    def __init__(self, settings: SafetyWrapperSettings) -> None:
        self._settings = settings

    def get_resources(self, risk_level: RiskLevel) -> list[CrisisResource]:
        """Get appropriate resources for risk level."""
        resources: list[CrisisResource] = []
        if self._settings.always_include_988:
            resources.append(self.CRISIS_RESOURCES[0])
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            resources.append(self.CRISIS_RESOURCES[1])
            if len(resources) < self._settings.max_resources_shown:
                resources.append(self.CRISIS_RESOURCES[3])
        elif risk_level == RiskLevel.MODERATE:
            if self.CRISIS_RESOURCES[1] not in resources:
                resources.append(self.CRISIS_RESOURCES[1])
        return resources[:self._settings.max_resources_shown]

    def format_resources_block(self, resources: list[CrisisResource]) -> str:
        """Format resources into display block."""
        if not resources:
            return ""
        lines = ["\n\n**If you need immediate support:**"]
        for resource in resources:
            lines.append(resource.format_display())
        return "\n".join(lines)


class ContentFilter:
    """Filters potentially harmful content from responses."""

    HARMFUL_PHRASES: list[str] = [
        "you should just",
        "give up",
        "no point",
        "end it",
        "not worth",
        "just do it",
        "easy way out",
    ]
    REPLACEMENT_PHRASES: dict[str, str] = {
        "give up": "take a break",
        "no point": "it may feel difficult",
        "not worth": "challenging",
    }

    def __init__(self, settings: SafetyWrapperSettings) -> None:
        self._settings = settings

    def filter(self, content: str) -> tuple[str, bool]:
        """Filter harmful content from response."""
        if not self._settings.filter_harmful_phrases:
            return content, False
        filtered = content
        was_filtered = False
        content_lower = filtered.lower()
        for harmful in self.HARMFUL_PHRASES:
            if harmful in content_lower:
                for original, replacement in self.REPLACEMENT_PHRASES.items():
                    if original in content_lower:
                        filtered = filtered.replace(original, replacement)
                        was_filtered = True
                if harmful in filtered.lower():
                    filtered = self._remove_harmful_sentence(filtered, harmful)
                    was_filtered = True
        return filtered, was_filtered

    def _remove_harmful_sentence(self, content: str, harmful: str) -> str:
        """Remove sentence containing harmful phrase."""
        sentences = content.split(". ")
        filtered_sentences = [
            s for s in sentences
            if harmful.lower() not in s.lower()
        ]
        return ". ".join(filtered_sentences)


class DisclaimerInjector:
    """Injects appropriate disclaimers into responses."""

    CRISIS_DISCLAIMER: str = (
        "Remember, I'm here to support you, but I'm not a substitute for "
        "professional crisis intervention. If you're in immediate danger, "
        "please contact emergency services or a crisis hotline."
    )
    THERAPEUTIC_DISCLAIMER: str = (
        "While I can provide support and coping strategies, please consider "
        "working with a licensed mental health professional for ongoing care."
    )

    def __init__(self, settings: SafetyWrapperSettings) -> None:
        self._settings = settings

    def inject(
        self,
        content: str,
        risk_level: RiskLevel,
        include_resources: bool,
    ) -> tuple[str, bool]:
        """Inject appropriate disclaimer."""
        if not self._settings.enable_disclaimer_injection:
            return content, False
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            if not self._has_disclaimer(content):
                return f"{content}\n\n{self.CRISIS_DISCLAIMER}", True
        return content, False

    def _has_disclaimer(self, content: str) -> bool:
        """Check if content already has a disclaimer."""
        disclaimer_indicators = [
            "not a substitute",
            "professional help",
            "emergency services",
            "crisis hotline",
            "licensed professional",
        ]
        content_lower = content.lower()
        return any(ind in content_lower for ind in disclaimer_indicators)


class SafetyWrapper:
    """Wraps responses with safety content when needed."""

    def __init__(self, settings: SafetyWrapperSettings | None = None) -> None:
        self._settings = settings or SafetyWrapperSettings()
        self._resource_provider = ResourceProvider(self._settings)
        self._content_filter = ContentFilter(self._settings)
        self._disclaimer_injector = DisclaimerInjector(self._settings)
        self._wrap_count = 0

    def wrap(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Wrap response with safety content.
        Main LangGraph node function.
        """
        self._wrap_count += 1
        content = state.get("final_response", "")
        safety_flags = state.get("safety_flags", {})
        logger.info(
            "safety_wrapper_processing",
            content_length=len(content),
            crisis_detected=safety_flags.get("crisis_detected", False),
        )
        risk_level = RiskLevel(safety_flags.get("risk_level", "none"))
        result = self._apply_safety_wrap(content, risk_level, safety_flags)
        updated_flags = {
            **safety_flags,
            "safety_resources_shown": len(result.resources_added) > 0,
        }
        agent_result = AgentResult(
            agent_type=AgentType.SAFETY,
            success=True,
            response_content=result.content,
            confidence=0.95,
            metadata={
                "resources_count": len(result.resources_added),
                "content_filtered": result.content_filtered,
                "disclaimer_added": result.disclaimer_added,
            },
        )
        logger.info(
            "safety_wrapper_complete",
            resources_added=len(result.resources_added),
            content_filtered=result.content_filtered,
            risk_level=risk_level.value,
        )
        return {
            "final_response": result.content,
            "safety_flags": updated_flags,
            "processing_phase": ProcessingPhase.SAFETY_POSTCHECK.value,
            "agent_results": [agent_result.to_dict()],
            "metadata": {
                **state.get("metadata", {}),
                "safety_wrap": result.to_dict(),
            },
        }

    def _apply_safety_wrap(
        self,
        content: str,
        risk_level: RiskLevel,
        safety_flags: dict[str, Any],
    ) -> SafetyWrapResult:
        """Apply safety wrapping to content."""
        wrapped = content
        filtered_content, was_filtered = self._content_filter.filter(wrapped)
        wrapped = filtered_content
        resources: list[CrisisResource] = []
        if self._should_add_resources(risk_level, safety_flags):
            resources = self._resource_provider.get_resources(risk_level)
            resource_block = self._resource_provider.format_resources_block(resources)
            if resource_block:
                wrapped = f"{wrapped}{resource_block}"
        disclaimer_added = False
        wrapped, disclaimer_added = self._disclaimer_injector.inject(
            wrapped, risk_level, len(resources) > 0
        )
        return SafetyWrapResult(
            content=wrapped,
            original_content=content,
            resources_added=resources,
            content_filtered=was_filtered,
            disclaimer_added=disclaimer_added,
            risk_level=risk_level,
            metadata={"flags": safety_flags},
        )

    def _should_add_resources(
        self,
        risk_level: RiskLevel,
        safety_flags: dict[str, Any],
    ) -> bool:
        """Determine if resources should be added."""
        if not self._settings.enable_resource_injection:
            return False
        if safety_flags.get("safety_resources_shown"):
            return False
        threshold = self._settings.resource_threshold
        threshold_map = {
            "none": [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL],
            "low": [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL],
            "moderate": [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL],
            "high": [RiskLevel.HIGH, RiskLevel.CRITICAL],
            "critical": [RiskLevel.CRITICAL],
        }
        required_levels = threshold_map.get(threshold, [RiskLevel.HIGH, RiskLevel.CRITICAL])
        return risk_level in required_levels

    def get_statistics(self) -> dict[str, Any]:
        """Get wrapper statistics."""
        return {
            "total_wraps": self._wrap_count,
            "settings": {
                "resource_injection": self._settings.enable_resource_injection,
                "content_filtering": self._settings.enable_content_filtering,
                "resource_threshold": self._settings.resource_threshold,
            },
        }


def safety_wrapper_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for safety wrapping.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    wrapper = SafetyWrapper()
    return wrapper.wrap(state)
