"""
Solace-AI Memory Service - LLM Context Assembly.
Assembles context for LLM within token budget with priority-based inclusion.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

if TYPE_CHECKING:
    from .service import MemoryRecord

logger = structlog.get_logger(__name__)


class ContextAssemblerSettings(BaseSettings):
    """Configuration for context assembly."""
    default_token_budget: int = Field(default=8000, description="Default token budget")
    system_prompt_budget: int = Field(default=1000, description="System prompt token allocation")
    safety_context_budget: int = Field(default=500, description="Safety context token allocation")
    user_profile_budget: int = Field(default=500, description="User profile token allocation")
    recent_messages_budget: int = Field(default=4000, description="Recent messages token allocation")
    retrieved_context_budget: int = Field(default=2000, description="Retrieved context allocation")
    response_buffer: int = Field(default=1000, description="Reserved for response")
    max_recent_messages: int = Field(default=20, description="Max recent messages to include")
    relevance_threshold: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    chars_per_token: int = Field(default=4, description="Approximate chars per token")
    enable_compression: bool = Field(default=True, description="Enable context compression")
    model_config = SettingsConfigDict(env_prefix="CONTEXT_", env_file=".env", extra="ignore")


@dataclass
class TokenAllocation:
    """Token allocation for context sections."""
    system_prompt: int = 0
    safety_context: int = 0
    user_profile: int = 0
    recent_messages: int = 0
    retrieved_context: int = 0
    therapeutic_context: int = 0
    remaining: int = 0


class ContextSection(BaseModel):
    """A section of assembled context."""
    section_type: str = Field(...)
    content: str = Field(default="")
    token_count: int = Field(default=0)
    priority: int = Field(default=5, ge=1, le=10)
    source: str = Field(default="unknown")


class ContextAssemblyOutput(BaseModel):
    """Output from context assembly."""
    context_id: UUID = Field(default_factory=uuid4)
    assembled_context: str = Field(default="")
    total_tokens: int = Field(default=0)
    token_breakdown: dict[str, int] = Field(default_factory=dict)
    sources_used: list[str] = Field(default_factory=list)
    retrieval_count: int = Field(default=0)
    sections_included: int = Field(default=0)
    truncated: bool = Field(default=False)


class ContextAssembler:
    """Assembles LLM context from multiple memory sources within token budget."""

    def __init__(self, settings: ContextAssemblerSettings | None = None) -> None:
        self._settings = settings or ContextAssemblerSettings()
        self._system_prompt_template = self._load_system_prompt_template()
        self._safety_context_cache: dict[UUID, str] = {}

    async def assemble(self, user_id: UUID, session_id: UUID | None, current_message: str | None,
                       token_budget: int, include_safety: bool, include_therapeutic: bool,
                       retrieval_query: str | None, priority_topics: list[str],
                       working_memory: list[Any], session_memory: list[Any],
                       user_profile: dict[str, Any]) -> ContextAssemblyOutput:
        """Assemble context for LLM within token budget."""
        start_time = time.perf_counter()
        allocation = self._calculate_allocation(token_budget)
        sections: list[ContextSection] = []
        sources_used: list[str] = []
        retrieval_count = 0
        system_section = self._build_system_prompt_section(allocation.system_prompt)
        if system_section.content:
            sections.append(system_section)
            sources_used.append("system_prompt")
        if include_safety:
            safety_section = self._build_safety_section(user_id, user_profile, allocation.safety_context)
            if safety_section.content:
                sections.append(safety_section)
                sources_used.append("safety_context")
        profile_section = self._build_user_profile_section(user_profile, allocation.user_profile)
        if profile_section.content:
            sections.append(profile_section)
            sources_used.append("user_profile")
        if include_therapeutic:
            therapeutic_section = self._build_therapeutic_section(user_profile, allocation.therapeutic_context)
            if therapeutic_section.content:
                sections.append(therapeutic_section)
                sources_used.append("therapeutic_context")
        messages_section = self._build_recent_messages_section(
            working_memory, session_memory, allocation.recent_messages
        )
        if messages_section.content:
            sections.append(messages_section)
            sources_used.append("recent_messages")
        if retrieval_query or priority_topics:
            retrieved_section, count = self._build_retrieved_section(
                session_memory, retrieval_query, priority_topics, allocation.retrieved_context
            )
            if retrieved_section.content:
                sections.append(retrieved_section)
                sources_used.append("retrieved_context")
                retrieval_count = count
        if current_message:
            current_section = ContextSection(
                section_type="current_input", content=f"\n[Current Message]\nUser: {current_message}",
                token_count=self._estimate_tokens(current_message) + 10, priority=10, source="current_input",
            )
            sections.append(current_section)
            sources_used.append("current_input")
        sections.sort(key=lambda s: s.priority, reverse=True)
        final_context, token_breakdown, truncated = self._assemble_sections(sections, token_budget)
        total_tokens = sum(token_breakdown.values())
        assembly_time_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug("context_assembled", user_id=str(user_id), total_tokens=total_tokens,
                     sections=len(sections), time_ms=assembly_time_ms)
        return ContextAssemblyOutput(
            assembled_context=final_context, total_tokens=total_tokens,
            token_breakdown=token_breakdown, sources_used=sources_used,
            retrieval_count=retrieval_count, sections_included=len(sections),
            truncated=truncated,
        )

    def _calculate_allocation(self, total_budget: int) -> TokenAllocation:
        """Calculate token allocation for each section."""
        available = total_budget - self._settings.response_buffer
        system = min(self._settings.system_prompt_budget, available // 8)
        safety = min(self._settings.safety_context_budget, available // 16)
        profile = min(self._settings.user_profile_budget, available // 16)
        therapeutic = available // 16
        recent = min(self._settings.recent_messages_budget, available // 2)
        retrieved = min(self._settings.retrieved_context_budget, available // 4)
        remaining = available - (system + safety + profile + therapeutic + recent + retrieved)
        return TokenAllocation(
            system_prompt=system, safety_context=safety, user_profile=profile,
            recent_messages=recent, retrieved_context=retrieved,
            therapeutic_context=therapeutic, remaining=max(0, remaining),
        )

    def _build_system_prompt_section(self, budget: int) -> ContextSection:
        """Build system prompt section."""
        prompt = self._system_prompt_template
        if self._estimate_tokens(prompt) > budget:
            prompt = self._truncate_to_tokens(prompt, budget)
        return ContextSection(
            section_type="system_prompt", content=f"[System]\n{prompt}",
            token_count=self._estimate_tokens(prompt), priority=10, source="system_prompt",
        )

    def _build_safety_section(self, user_id: UUID, profile: dict[str, Any], budget: int) -> ContextSection:
        """Build safety context section."""
        safety_info = profile.get("safety", {})
        if not safety_info:
            return ContextSection(section_type="safety_context", priority=9, source="safety_context")
        parts = ["[Safety Context - CRITICAL INFORMATION]"]
        if crisis_history := safety_info.get("crisis_history"):
            parts.append(f"Crisis History: {crisis_history}")
        if safety_plan := safety_info.get("safety_plan"):
            parts.append(f"Safety Plan Active: {safety_plan}")
        if risk_factors := safety_info.get("risk_factors"):
            parts.append(f"Known Risk Factors: {', '.join(risk_factors)}")
        if emergency_contacts := safety_info.get("emergency_contacts"):
            parts.append(f"Emergency Contacts: Available")
        if protective_factors := safety_info.get("protective_factors"):
            parts.append(f"Protective Factors: {', '.join(protective_factors)}")
        content = "\n".join(parts)
        if self._estimate_tokens(content) > budget:
            content = self._truncate_to_tokens(content, budget)
        return ContextSection(
            section_type="safety_context", content=content,
            token_count=self._estimate_tokens(content), priority=9, source="safety_context",
        )

    def _build_user_profile_section(self, profile: dict[str, Any], budget: int) -> ContextSection:
        """Build user profile section."""
        facts = profile.get("facts", {})
        if not facts:
            return ContextSection(section_type="user_profile", priority=7, source="user_profile")
        parts = ["[User Profile]"]
        for key, value in list(facts.items())[:10]:
            parts.append(f"- {key}: {value}")
        if preferences := profile.get("preferences", {}):
            parts.append("\nPreferences:")
            for key, value in list(preferences.items())[:5]:
                parts.append(f"- {key}: {value}")
        content = "\n".join(parts)
        if self._estimate_tokens(content) > budget:
            content = self._truncate_to_tokens(content, budget)
        return ContextSection(
            section_type="user_profile", content=content,
            token_count=self._estimate_tokens(content), priority=7, source="user_profile",
        )

    def _build_therapeutic_section(self, profile: dict[str, Any], budget: int) -> ContextSection:
        """Build therapeutic context section."""
        therapeutic = profile.get("therapeutic", {})
        if not therapeutic:
            return ContextSection(section_type="therapeutic_context", priority=8, source="therapeutic")
        parts = ["[Therapeutic Context]"]
        if treatment_plan := therapeutic.get("treatment_plan"):
            parts.append(f"Active Treatment: {treatment_plan}")
        if current_phase := therapeutic.get("current_phase"):
            parts.append(f"Treatment Phase: {current_phase}")
        if techniques := therapeutic.get("effective_techniques"):
            parts.append(f"Effective Techniques: {', '.join(techniques[:5])}")
        if homework := therapeutic.get("pending_homework"):
            parts.append(f"Pending Homework: {homework}")
        if goals := therapeutic.get("active_goals"):
            parts.append(f"Active Goals: {', '.join(goals[:3])}")
        content = "\n".join(parts)
        if self._estimate_tokens(content) > budget:
            content = self._truncate_to_tokens(content, budget)
        return ContextSection(
            section_type="therapeutic_context", content=content,
            token_count=self._estimate_tokens(content), priority=8, source="therapeutic",
        )

    def _build_recent_messages_section(self, working_memory: list[Any],
                                        session_memory: list[Any], budget: int) -> ContextSection:
        """Build recent messages section from working memory."""
        all_messages = list(working_memory) + list(session_memory)
        all_messages = sorted(all_messages, key=lambda m: getattr(m, 'created_at', datetime.min))
        recent = all_messages[-self._settings.max_recent_messages:]
        parts = ["[Conversation History]"]
        total_tokens = 5
        for msg in recent:
            role = msg.metadata.get("role", "user") if hasattr(msg, 'metadata') else "user"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            line = f"{role.capitalize()}: {content}"
            line_tokens = self._estimate_tokens(line)
            if total_tokens + line_tokens > budget:
                if self._settings.enable_compression:
                    compressed = self._compress_message(line, budget - total_tokens - 10)
                    if compressed:
                        parts.append(compressed)
                        total_tokens += self._estimate_tokens(compressed)
                break
            parts.append(line)
            total_tokens += line_tokens
        content = "\n".join(parts)
        return ContextSection(
            section_type="recent_messages", content=content,
            token_count=total_tokens, priority=8, source="recent_messages",
        )

    def _build_retrieved_section(self, memory_records: list[Any], query: str | None,
                                  priority_topics: list[str], budget: int) -> tuple[ContextSection, int]:
        """Build retrieved context section using RAG-style retrieval."""
        if not query and not priority_topics:
            return ContextSection(section_type="retrieved_context", priority=6, source="retrieved"), 0
        search_terms = [query] if query else []
        search_terms.extend(priority_topics)
        relevant_records = []
        for record in memory_records:
            content = record.content if hasattr(record, 'content') else str(record)
            content_lower = content.lower()
            relevance = sum(1 for term in search_terms if term.lower() in content_lower)
            if relevance > 0:
                importance = float(record.importance_score) if hasattr(record, 'importance_score') else 0.5
                score = relevance * importance
                relevant_records.append((record, score))
        relevant_records.sort(key=lambda x: x[1], reverse=True)
        parts = ["[Retrieved Context]"]
        total_tokens = 5
        count = 0
        for record, score in relevant_records[:10]:
            content = record.content if hasattr(record, 'content') else str(record)
            line_tokens = self._estimate_tokens(content)
            if total_tokens + line_tokens > budget:
                break
            parts.append(f"- {content}")
            total_tokens += line_tokens
            count += 1
        content = "\n".join(parts) if count > 0 else ""
        return ContextSection(
            section_type="retrieved_context", content=content,
            token_count=total_tokens if count > 0 else 0, priority=6, source="retrieved",
        ), count

    def _assemble_sections(self, sections: list[ContextSection],
                           budget: int) -> tuple[str, dict[str, int], bool]:
        """Assemble all sections into final context."""
        parts: list[str] = []
        breakdown: dict[str, int] = {}
        total_tokens = 0
        truncated = False
        for section in sections:
            if not section.content:
                continue
            if total_tokens + section.token_count <= budget:
                parts.append(section.content)
                breakdown[section.section_type] = section.token_count
                total_tokens += section.token_count
            else:
                remaining = budget - total_tokens
                if remaining > 50:
                    truncated_content = self._truncate_to_tokens(section.content, remaining)
                    parts.append(truncated_content)
                    breakdown[section.section_type] = self._estimate_tokens(truncated_content)
                    total_tokens += breakdown[section.section_type]
                    truncated = True
                break
        return "\n\n".join(parts), breakdown, truncated

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return max(1, len(text) // self._settings.chars_per_token)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        max_chars = max_tokens * self._settings.chars_per_token
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars - 20]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        break_point = max(last_period, last_newline)
        if break_point > max_chars // 2:
            return truncated[:break_point + 1] + "..."
        return truncated + "..."

    def _compress_message(self, message: str, max_tokens: int) -> str | None:
        """Compress a message to fit within token budget."""
        if max_tokens < 10:
            return None
        max_chars = max_tokens * self._settings.chars_per_token
        if len(message) <= max_chars:
            return message
        return message[:max_chars - 3] + "..."

    def _load_system_prompt_template(self) -> str:
        """Load the default system prompt template."""
        return (
            "You are Solace-AI, a compassionate and professional AI therapeutic assistant. "
            "Your role is to provide supportive, evidence-based mental health guidance while "
            "maintaining appropriate boundaries. Always prioritize user safety. If you detect "
            "signs of crisis or immediate danger, provide crisis resources and encourage "
            "professional help. Use therapeutic techniques appropriately based on the user's "
            "treatment context. Be warm, empathetic, and non-judgmental in all interactions."
        )

    def get_token_budget_breakdown(self, total_budget: int) -> dict[str, int]:
        """Get a breakdown of how tokens would be allocated."""
        allocation = self._calculate_allocation(total_budget)
        return {
            "system_prompt": allocation.system_prompt,
            "safety_context": allocation.safety_context,
            "user_profile": allocation.user_profile,
            "therapeutic_context": allocation.therapeutic_context,
            "recent_messages": allocation.recent_messages,
            "retrieved_context": allocation.retrieved_context,
            "remaining": allocation.remaining,
            "response_buffer": self._settings.response_buffer,
        }
