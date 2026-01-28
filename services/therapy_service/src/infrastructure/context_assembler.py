"""
Solace-AI Therapy Service - Context Assembly.
Fetches and assembles context from Memory Service for enriched therapeutic responses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID
import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class ContextAssemblerSettings(BaseSettings):
    """Configuration for context assembly."""
    memory_service_url: str = Field(default="http://localhost:8005")
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=30.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    default_token_budget: int = Field(default=4000, ge=1000, le=16000)
    include_safety_context: bool = Field(default=True)
    include_therapeutic_context: bool = Field(default=True)
    fallback_on_error: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=60)
    model_config = SettingsConfigDict(
        env_prefix="THERAPY_CONTEXT_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class MemoryContext:
    """Assembled memory context for therapy session."""
    user_profile: dict[str, Any] = field(default_factory=dict)
    conversation_summary: str = ""
    relevant_memories: list[dict[str, Any]] = field(default_factory=list)
    therapeutic_history: dict[str, Any] = field(default_factory=dict)
    safety_context: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    assembly_time_ms: float = 0.0
    from_cache: bool = False

    def to_prompt_context(self) -> str:
        """Convert to a formatted context string for LLM prompt."""
        parts = []

        if self.user_profile:
            parts.append("## User Profile")
            if self.user_profile.get("name"):
                parts.append(f"Name: {self.user_profile.get('name')}")
            if self.user_profile.get("preferences"):
                parts.append(f"Preferences: {self.user_profile.get('preferences')}")

        if self.therapeutic_history:
            parts.append("\n## Therapeutic History")
            if self.therapeutic_history.get("sessions_completed"):
                parts.append(f"Sessions completed: {self.therapeutic_history.get('sessions_completed')}")
            if self.therapeutic_history.get("techniques_used"):
                parts.append(f"Techniques practiced: {', '.join(self.therapeutic_history.get('techniques_used', []))}")
            if self.therapeutic_history.get("key_themes"):
                parts.append(f"Key themes: {', '.join(self.therapeutic_history.get('key_themes', []))}")
            if self.therapeutic_history.get("progress_notes"):
                parts.append(f"Progress: {self.therapeutic_history.get('progress_notes')}")

        if self.conversation_summary:
            parts.append(f"\n## Recent Conversation Summary\n{self.conversation_summary}")

        if self.relevant_memories:
            parts.append("\n## Relevant Past Interactions")
            for i, memory in enumerate(self.relevant_memories[:5], 1):
                content = memory.get("content", "")[:200]  # Truncate long memories
                parts.append(f"{i}. {content}")

        if self.safety_context and self.safety_context.get("alerts"):
            parts.append("\n## Safety Alerts")
            for alert in self.safety_context.get("alerts", []):
                parts.append(f"- {alert}")

        return "\n".join(parts) if parts else ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_profile": self.user_profile,
            "conversation_summary": self.conversation_summary,
            "relevant_memories": self.relevant_memories,
            "therapeutic_history": self.therapeutic_history,
            "safety_context": self.safety_context,
            "total_tokens": self.total_tokens,
            "assembly_time_ms": self.assembly_time_ms,
            "from_cache": self.from_cache,
        }


class MemoryServiceClient:
    """HTTP client for Memory Service communication."""

    def __init__(self, settings: ContextAssemblerSettings) -> None:
        self._settings = settings
        self._base_url = settings.memory_service_url.rstrip("/")

    async def assemble_context(
        self,
        user_id: UUID,
        session_id: UUID,
        current_message: str,
        token_budget: int | None = None,
        retrieval_query: str | None = None,
        priority_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assemble context from Memory Service."""
        url = f"{self._base_url}/api/v1/memory/context"
        payload = {
            "user_id": str(user_id),
            "session_id": str(session_id),
            "current_message": current_message,
            "token_budget": token_budget or self._settings.default_token_budget,
            "include_safety_context": self._settings.include_safety_context,
            "include_therapeutic_context": self._settings.include_therapeutic_context,
            "retrieval_query": retrieval_query,
            "priority_topics": priority_topics or [],
        }

        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "memory_service_http_error",
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
                except httpx.RequestError as e:
                    logger.warning(
                        "memory_service_request_error",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
        raise RuntimeError("Memory service request failed after retries")

    async def get_user_profile(self, user_id: UUID) -> dict[str, Any]:
        """Get user profile from Memory Service."""
        url = f"{self._base_url}/api/v1/memory/profile/{user_id}"

        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("get_user_profile_failed", error=str(e))
                return {}

    async def retrieve_memories(
        self,
        user_id: UUID,
        query: str,
        limit: int = 5,
        tiers: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories from Memory Service."""
        url = f"{self._base_url}/api/v1/memory/retrieve"
        payload = {
            "user_id": str(user_id),
            "query": query,
            "limit": limit,
            "tiers": tiers or ["EPISODIC", "SEMANTIC"],
        }

        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("records", [])
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("retrieve_memories_failed", error=str(e))
                return []


class ContextAssembler:
    """
    Assembles rich context for therapy sessions from Memory Service.

    Retrieves user profiles, conversation history, therapeutic context,
    and relevant memories to provide comprehensive context for response generation.
    """

    def __init__(self, settings: ContextAssemblerSettings | None = None) -> None:
        self._settings = settings or ContextAssemblerSettings()
        self._client = MemoryServiceClient(self._settings)
        self._cache: dict[str, tuple[MemoryContext, datetime]] = {}

    async def assemble(
        self,
        user_id: UUID,
        session_id: UUID,
        current_message: str,
        token_budget: int | None = None,
        priority_topics: list[str] | None = None,
    ) -> MemoryContext:
        """
        Assemble context for therapy session.

        Args:
            user_id: User identifier
            session_id: Current session ID
            current_message: Current user message
            token_budget: Token limit for context
            priority_topics: Topics to prioritize in retrieval

        Returns:
            MemoryContext with assembled data
        """
        start_time = datetime.now(timezone.utc)

        # Check cache first
        cache_key = f"{user_id}:{session_id}"
        if cache_key in self._cache:
            cached_context, cached_at = self._cache[cache_key]
            cache_age = (start_time - cached_at).total_seconds()
            if cache_age < self._settings.cache_ttl_seconds:
                cached_context.from_cache = True
                logger.debug("context_from_cache", user_id=str(user_id), age_seconds=cache_age)
                return cached_context

        try:
            # Fetch full assembled context from Memory Service
            context_data = await self._client.assemble_context(
                user_id=user_id,
                session_id=session_id,
                current_message=current_message,
                token_budget=token_budget,
                retrieval_query=current_message,
                priority_topics=priority_topics,
            )

            memory_context = MemoryContext(
                user_profile=context_data.get("user_profile", {}),
                conversation_summary=context_data.get("assembled_context", ""),
                relevant_memories=context_data.get("sources_used", []),
                therapeutic_history=context_data.get("therapeutic_context", {}),
                safety_context=context_data.get("safety_context", {}),
                total_tokens=context_data.get("total_tokens", 0),
            )

        except Exception as e:
            logger.warning("context_assembly_failed", error=str(e))
            if self._settings.fallback_on_error:
                memory_context = await self._fallback_assembly(user_id, session_id, current_message)
            else:
                raise

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        memory_context.assembly_time_ms = elapsed_ms

        # Cache the result
        self._cache[cache_key] = (memory_context, start_time)

        logger.info(
            "context_assembled",
            user_id=str(user_id),
            total_tokens=memory_context.total_tokens,
            assembly_time_ms=elapsed_ms,
        )

        return memory_context

    async def _fallback_assembly(
        self,
        user_id: UUID,
        session_id: UUID,
        current_message: str,
    ) -> MemoryContext:
        """
        Fallback context assembly when Memory Service is unavailable.

        Provides minimal context to allow therapy session to continue.
        """
        logger.warning("using_fallback_context_assembly", user_id=str(user_id))

        # Try to get user profile separately
        profile = await self._client.get_user_profile(user_id)

        # Try to retrieve some relevant memories
        memories = await self._client.retrieve_memories(
            user_id=user_id,
            query=current_message,
            limit=3,
        )

        return MemoryContext(
            user_profile=profile if profile else {},
            conversation_summary="",
            relevant_memories=memories,
            therapeutic_history={},
            safety_context={},
            total_tokens=0,
        )

    async def enrich_conversation_history(
        self,
        user_id: UUID,
        session_id: UUID,
        conversation_history: list[dict[str, str]],
        current_message: str,
    ) -> list[dict[str, str]]:
        """
        Enrich conversation history with relevant memory context.

        Prepends relevant context to conversation history for better
        therapeutic responses.

        Args:
            user_id: User identifier
            session_id: Session identifier
            conversation_history: Existing conversation
            current_message: Current user message

        Returns:
            Enriched conversation history with context prepended
        """
        context = await self.assemble(
            user_id=user_id,
            session_id=session_id,
            current_message=current_message,
        )

        prompt_context = context.to_prompt_context()
        if not prompt_context:
            return conversation_history

        # Prepend context as a system message
        enriched = [
            {"role": "system", "content": f"[User Context]\n{prompt_context}"}
        ]
        enriched.extend(conversation_history)

        return enriched

    def clear_cache(self, user_id: UUID | None = None, session_id: UUID | None = None) -> None:
        """Clear context cache."""
        if user_id and session_id:
            cache_key = f"{user_id}:{session_id}"
            self._cache.pop(cache_key, None)
        elif user_id:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


# Module-level singleton
_context_assembler: ContextAssembler | None = None


def get_context_assembler() -> ContextAssembler:
    """Get or create context assembler singleton."""
    global _context_assembler
    if _context_assembler is None:
        _context_assembler = ContextAssembler()
    return _context_assembler


async def initialize_context_assembler(
    settings: ContextAssemblerSettings | None = None,
) -> ContextAssembler:
    """Initialize the context assembler with optional settings."""
    global _context_assembler
    _context_assembler = ContextAssembler(settings)
    return _context_assembler
