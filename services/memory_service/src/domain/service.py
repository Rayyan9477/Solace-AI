"""
Solace-AI Memory Service - Main memory orchestration service.
Coordinates 5-tier memory hierarchy, context assembly, and consolidation.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, TYPE_CHECKING
from uuid import UUID
import structlog

from .models import (
    MemoryServiceSettings, MemoryRecord, SessionState,
    StoreMemoryResult, RetrieveMemoryResult, ContextAssemblyResult,
    SessionStartResult, SessionEndResult, AddMessageResult,
    ConsolidationResult, UserProfileResult,
)

if TYPE_CHECKING:
    from .context_assembler import ContextAssembler
    from .consolidation import ConsolidationPipeline

logger = structlog.get_logger(__name__)


class MemoryService:
    """Main memory service orchestrating 5-tier memory hierarchy."""

    def __init__(self, settings: MemoryServiceSettings | None = None,
                 context_assembler: ContextAssembler | None = None,
                 consolidation_pipeline: ConsolidationPipeline | None = None) -> None:
        self._settings = settings or MemoryServiceSettings()
        self._context_assembler = context_assembler
        self._consolidation_pipeline = consolidation_pipeline
        self._tier1_input: dict[UUID, MemoryRecord] = {}
        self._tier2_working: dict[UUID, list[MemoryRecord]] = {}
        self._tier3_session: dict[UUID, list[MemoryRecord]] = {}
        self._tier4_episodic: dict[UUID, list[MemoryRecord]] = {}
        self._tier5_semantic: dict[UUID, list[MemoryRecord]] = {}
        self._active_sessions: dict[UUID, SessionState] = {}
        self._user_session_counts: dict[UUID, int] = {}
        self._user_profiles: dict[UUID, dict[str, Any]] = {}
        self._initialized = False
        self._stats = {"total_stores": 0, "total_retrieves": 0, "total_assemblies": 0,
                      "sessions_started": 0, "sessions_ended": 0, "consolidations": 0}

    async def initialize(self) -> None:
        """Initialize the memory service."""
        logger.info("memory_service_initializing")
        self._initialized = True
        logger.info("memory_service_initialized", settings={
            "working_memory_max": self._settings.working_memory_max_tokens,
            "auto_consolidation": self._settings.enable_auto_consolidation,
            "decay_enabled": self._settings.enable_decay,
        })

    async def shutdown(self) -> None:
        """Shutdown the memory service."""
        logger.info("memory_service_shutting_down", stats=self._stats)
        for session in list(self._active_sessions.values()):
            await self.end_session(session.user_id, session.session_id, False, False)
        self._initialized = False

    async def store_memory(self, user_id: UUID, session_id: UUID | None, content: str,
                           content_type: str, tier: str, retention_category: str,
                           importance_score: Decimal, metadata: dict[str, Any]) -> StoreMemoryResult:
        """Store a memory record to the specified tier."""
        start_time = time.perf_counter()
        self._stats["total_stores"] += 1
        record = MemoryRecord(
            user_id=user_id, session_id=session_id, tier=tier, content=content,
            content_type=content_type, retention_category=retention_category,
            importance_score=importance_score, metadata=metadata,
        )
        self._store_to_tier(record, tier)
        storage_time_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug("memory_stored", user_id=str(user_id), tier=tier, time_ms=storage_time_ms)
        return StoreMemoryResult(record_id=record.record_id, tier=tier, stored=True,
                                 storage_time_ms=storage_time_ms)

    async def retrieve_memories(self, user_id: UUID, session_id: UUID | None,
                                tiers: list[str] | None, query: str | None,
                                limit: int, min_importance: Decimal,
                                time_range_hours: int | None) -> RetrieveMemoryResult:
        """Retrieve memories based on query and filters."""
        start_time = time.perf_counter()
        self._stats["total_retrieves"] += 1
        search_tiers = tiers or ["tier_2_working", "tier_3_session", "tier_4_episodic", "tier_5_semantic"]
        all_records: list[MemoryRecord] = []
        for tier in search_tiers:
            tier_records = self._get_tier_records(user_id, tier)
            for record in tier_records:
                if record.importance_score >= min_importance:
                    if session_id is None or record.session_id == session_id:
                        if time_range_hours is None or self._within_time_range(record, time_range_hours):
                            all_records.append(record)
        if query:
            all_records = self._semantic_filter(all_records, query)
        all_records = sorted(all_records, key=lambda r: (r.importance_score, r.created_at), reverse=True)[:limit]
        retrieval_time_ms = int((time.perf_counter() - start_time) * 1000)
        return RetrieveMemoryResult(
            records=all_records, total_found=len(all_records),
            retrieval_time_ms=retrieval_time_ms, tiers_searched=search_tiers,
        )

    async def assemble_context(self, user_id: UUID, session_id: UUID | None,
                               current_message: str | None, token_budget: int,
                               include_safety: bool, include_therapeutic: bool,
                               retrieval_query: str | None,
                               priority_topics: list[str]) -> ContextAssemblyResult:
        """Assemble context for LLM within token budget."""
        start_time = time.perf_counter()
        self._stats["total_assemblies"] += 1
        if self._context_assembler:
            result = await self._context_assembler.assemble(
                user_id=user_id, session_id=session_id, current_message=current_message,
                token_budget=token_budget, include_safety=include_safety,
                include_therapeutic=include_therapeutic, retrieval_query=retrieval_query,
                priority_topics=priority_topics, working_memory=self._tier2_working.get(user_id, []),
                session_memory=self._tier3_session.get(user_id, []),
                user_profile=self._user_profiles.get(user_id, {}),
            )
            assembly_time_ms = int((time.perf_counter() - start_time) * 1000)
            return ContextAssemblyResult(
                context_id=result.context_id, assembled_context=result.assembled_context,
                total_tokens=result.total_tokens, token_breakdown=result.token_breakdown,
                sources_used=result.sources_used, assembly_time_ms=assembly_time_ms,
                retrieval_count=result.retrieval_count,
            )
        context = self._build_basic_context(user_id, session_id, current_message, token_budget)
        assembly_time_ms = int((time.perf_counter() - start_time) * 1000)
        return ContextAssemblyResult(
            assembled_context=context, total_tokens=len(context.split()),
            token_breakdown={"basic": len(context.split())}, sources_used=["working_memory"],
            assembly_time_ms=assembly_time_ms,
        )

    async def start_session(self, user_id: UUID, session_type: str,
                            initial_context: dict[str, Any]) -> SessionStartResult:
        """Start a new session for user."""
        self._stats["sessions_started"] += 1
        session_number = self._user_session_counts.get(user_id, 0) + 1
        self._user_session_counts[user_id] = session_number
        session = SessionState(
            user_id=user_id, session_number=session_number,
            session_type=session_type, metadata=initial_context,
        )
        self._active_sessions[session.session_id] = session
        self._tier2_working[user_id] = []
        self._tier3_session.setdefault(user_id, [])
        previous_summary = self._get_previous_session_summary(user_id)
        self._load_user_profile(user_id)
        logger.info("session_started", user_id=str(user_id), session_id=str(session.session_id),
                    session_number=session_number)
        return SessionStartResult(
            session_id=session.session_id, session_number=session_number,
            previous_session_summary=previous_summary, user_profile_loaded=True,
        )

    async def end_session(self, user_id: UUID, session_id: UUID,
                          trigger_consolidation: bool, include_summary: bool) -> SessionEndResult:
        """End a session and optionally trigger consolidation."""
        self._stats["sessions_ended"] += 1
        session = self._active_sessions.get(session_id)
        if not session:
            return SessionEndResult(message_count=0, duration_minutes=0)
        message_count = len(session.messages)
        duration = datetime.now(timezone.utc) - session.started_at
        duration_minutes = int(duration.total_seconds() / 60)
        summary = None
        key_topics: list[str] = []
        consolidation_triggered = False
        if include_summary and self._consolidation_pipeline:
            summary_result = await self._consolidation_pipeline.generate_summary(session.messages)
            summary = summary_result.summary
            key_topics = summary_result.key_topics
        if trigger_consolidation and self._settings.enable_auto_consolidation:
            await self.consolidate(user_id, session_id, True, True, True, self._settings.enable_decay)
            consolidation_triggered = True
        del self._active_sessions[session_id]
        self._tier2_working.pop(user_id, None)
        logger.info("session_ended", user_id=str(user_id), session_id=str(session_id),
                    message_count=message_count, duration_minutes=duration_minutes)
        return SessionEndResult(
            message_count=message_count, duration_minutes=duration_minutes,
            summary=summary, consolidation_triggered=consolidation_triggered, key_topics=key_topics,
        )

    async def add_message(self, user_id: UUID, session_id: UUID, role: str, content: str,
                          emotion_detected: str | None, importance_override: Decimal | None,
                          metadata: dict[str, Any]) -> AddMessageResult:
        """Add a message to the current session."""
        start_time = time.perf_counter()
        importance = importance_override if importance_override is not None else self._calculate_importance(content, role)
        record = MemoryRecord(
            user_id=user_id, session_id=session_id, tier="tier_3_session",
            content=content, content_type="message", importance_score=importance,
            metadata={"role": role, "emotion": emotion_detected, **metadata},
        )
        self._tier3_session.setdefault(user_id, []).append(record)
        self._update_working_memory(user_id, record)
        session = self._active_sessions.get(session_id)
        if session:
            session.messages.append(record)
        storage_time_ms = int((time.perf_counter() - start_time) * 1000)
        return AddMessageResult(
            message_id=record.record_id, stored_to_tier="tier_3_session",
            working_memory_updated=True, storage_time_ms=storage_time_ms,
        )

    async def consolidate(self, user_id: UUID, session_id: UUID, extract_facts: bool,
                          generate_summary: bool, update_knowledge_graph: bool,
                          apply_decay: bool) -> ConsolidationResult:
        """Trigger memory consolidation pipeline."""
        start_time = time.perf_counter()
        self._stats["consolidations"] += 1
        if not self._consolidation_pipeline:
            return ConsolidationResult(consolidation_time_ms=0)
        session_records = [r for r in self._tier3_session.get(user_id, []) if r.session_id == session_id]
        result = await self._consolidation_pipeline.consolidate(
            user_id=user_id, session_id=session_id, records=session_records,
            extract_facts=extract_facts, generate_summary=generate_summary,
            update_knowledge_graph=update_knowledge_graph, apply_decay=apply_decay,
        )
        if result.summary_generated:
            summary_record = MemoryRecord(
                user_id=user_id, session_id=session_id, tier="tier_4_episodic",
                content=result.summary_generated, content_type="session_summary",
                retention_category="long_term", importance_score=Decimal("0.8"),
            )
            self._tier4_episodic.setdefault(user_id, []).append(summary_record)
        for fact in result.extracted_facts:
            fact_record = MemoryRecord(
                user_id=user_id, tier="tier_5_semantic", content=fact["content"],
                content_type="fact", retention_category=fact.get("retention", "long_term"),
                importance_score=Decimal(str(fact.get("importance", 0.7))), metadata=fact.get("metadata", {}),
            )
            self._tier5_semantic.setdefault(user_id, []).append(fact_record)
        if apply_decay:
            decayed, archived = self._apply_decay_model(user_id)
            result.memories_decayed = decayed
            result.memories_archived = archived
        consolidation_time_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("consolidation_completed", user_id=str(user_id), session_id=str(session_id),
                    facts=result.facts_extracted, time_ms=consolidation_time_ms)
        return ConsolidationResult(
            consolidation_id=result.consolidation_id, summary_generated=result.summary_generated,
            facts_extracted=result.facts_extracted, knowledge_nodes_updated=result.knowledge_nodes_updated,
            memories_decayed=result.memories_decayed, memories_archived=result.memories_archived,
            consolidation_time_ms=consolidation_time_ms,
        )

    async def get_user_profile(self, user_id: UUID, include_knowledge_graph: bool,
                               include_session_history: bool, session_limit: int) -> UserProfileResult:
        """Get user profile from memory."""
        profile = self._user_profiles.get(user_id, {})
        sessions = self._tier4_episodic.get(user_id, [])
        total_sessions = self._user_session_counts.get(user_id, 0)
        first_date = min((s.created_at for s in sessions), default=None) if sessions else None
        last_date = max((s.created_at for s in sessions), default=None) if sessions else None
        recent_sessions = []
        if include_session_history:
            for record in sorted(sessions, key=lambda r: r.created_at, reverse=True)[:session_limit]:
                recent_sessions.append({"session_id": str(record.session_id), "date": record.created_at.isoformat(),
                                        "summary": record.content[:200] if record.content else None})
        knowledge_graph = None
        if include_knowledge_graph:
            semantic_records = self._tier5_semantic.get(user_id, [])
            knowledge_graph = {"nodes": len(semantic_records), "facts": [r.content for r in semantic_records[:20]]}
        return UserProfileResult(
            total_sessions=total_sessions, first_session_date=first_date, last_session_date=last_date,
            profile_facts=profile.get("facts", {}), knowledge_graph=knowledge_graph,
            recent_sessions=recent_sessions, therapeutic_context=profile.get("therapeutic", {}),
        )

    async def delete_user_data(self, user_id: UUID) -> None:
        """Delete all user data (GDPR compliance)."""
        self._tier1_input.pop(user_id, None)
        self._tier2_working.pop(user_id, None)
        self._tier3_session.pop(user_id, None)
        self._tier4_episodic.pop(user_id, None)
        self._tier5_semantic.pop(user_id, None)
        self._user_profiles.pop(user_id, None)
        self._user_session_counts.pop(user_id, None)
        for sid, session in list(self._active_sessions.items()):
            if session.user_id == user_id:
                del self._active_sessions[sid]
        logger.info("user_data_deleted", user_id=str(user_id))

    async def get_status(self) -> dict[str, Any]:
        """Get service status and statistics."""
        return {
            "status": "operational" if self._initialized else "initializing",
            "initialized": self._initialized,
            "statistics": self._stats,
            "active_sessions": len(self._active_sessions),
            "users_tracked": len(self._user_session_counts),
            "tier_counts": {
                "tier_2_working": sum(len(v) for v in self._tier2_working.values()),
                "tier_3_session": sum(len(v) for v in self._tier3_session.values()),
                "tier_4_episodic": sum(len(v) for v in self._tier4_episodic.values()),
                "tier_5_semantic": sum(len(v) for v in self._tier5_semantic.values()),
            },
        }

    def _store_to_tier(self, record: MemoryRecord, tier: str) -> None:
        """Store record to appropriate tier."""
        tier_map = {"tier_1_input": self._tier1_input, "tier_2_working": self._tier2_working,
                    "tier_3_session": self._tier3_session, "tier_4_episodic": self._tier4_episodic,
                    "tier_5_semantic": self._tier5_semantic}
        storage = tier_map.get(tier)
        if storage is not None:
            if tier == "tier_1_input":
                storage[record.user_id] = record
            else:
                storage.setdefault(record.user_id, []).append(record)

    def _get_tier_records(self, user_id: UUID, tier: str) -> list[MemoryRecord]:
        """Get records from a tier for user."""
        tier_map = {"tier_2_working": self._tier2_working, "tier_3_session": self._tier3_session,
                    "tier_4_episodic": self._tier4_episodic, "tier_5_semantic": self._tier5_semantic}
        return tier_map.get(tier, {}).get(user_id, [])

    def _within_time_range(self, record: MemoryRecord, hours: int) -> bool:
        """Check if record is within time range."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return record.created_at >= cutoff

    def _semantic_filter(self, records: list[MemoryRecord], query: str) -> list[MemoryRecord]:
        """Basic semantic filtering (placeholder for vector search)."""
        return [r for r in records if query.lower() in r.content.lower()]

    def _get_previous_session_summary(self, user_id: UUID) -> str | None:
        """Get summary from previous session."""
        episodic = self._tier4_episodic.get(user_id, [])
        summaries = [r for r in episodic if r.content_type == "session_summary"]
        if summaries:
            return sorted(summaries, key=lambda r: r.created_at, reverse=True)[0].content
        return None

    def _load_user_profile(self, user_id: UUID) -> None:
        """Load or initialize user profile."""
        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = {"facts": {}, "therapeutic": {}, "preferences": {}}

    def _calculate_importance(self, content: str, role: str) -> Decimal:
        """Calculate importance score for content."""
        base_score = Decimal("0.5")
        if role == "user":
            base_score += Decimal("0.1")
        important_keywords = ["crisis", "emergency", "help", "suicide", "harm", "danger", "medication", "therapy"]
        if any(kw in content.lower() for kw in important_keywords):
            base_score += Decimal("0.3")
        return min(base_score, Decimal("1.0"))

    def _update_working_memory(self, user_id: UUID, record: MemoryRecord) -> None:
        """Update working memory with new record."""
        working = self._tier2_working.setdefault(user_id, [])
        working.append(record)
        while len(working) > 20:
            working.pop(0)

    def _build_basic_context(self, user_id: UUID, session_id: UUID | None,
                             current_message: str | None, token_budget: int) -> str:
        """Build basic context without assembler."""
        parts = []
        working = self._tier2_working.get(user_id, [])
        for record in working[-10:]:
            role = record.metadata.get("role", "user")
            parts.append(f"{role}: {record.content}")
        if current_message:
            parts.append(f"user: {current_message}")
        return "\n".join(parts)

    def _apply_decay_model(self, user_id: UUID) -> tuple[int, int]:
        """Apply Ebbinghaus decay model to memories."""
        decayed = 0
        archived = 0
        for tier_storage in [self._tier3_session, self._tier4_episodic]:
            records = tier_storage.get(user_id, [])
            for record in records:
                if record.retention_category not in ("permanent", "long_term"):
                    age_days = (datetime.now(timezone.utc) - record.created_at).days
                    decay_rate = Decimal("0.1") if record.retention_category == "medium_term" else Decimal("0.15")
                    record.retention_strength = max(Decimal("0.1"), record.retention_strength - (decay_rate * age_days / 30))
                    decayed += 1
                    if record.retention_strength < Decimal("0.3"):
                        archived += 1
        return decayed, archived
