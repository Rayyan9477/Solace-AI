"""
Solace-AI Memory Service - Memory Consolidation Pipeline.
Handles session summarization, fact extraction, knowledge graph updates, and decay.
"""
from __future__ import annotations
import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

if TYPE_CHECKING:
    from .service import MemoryRecord

logger = structlog.get_logger(__name__)


class ConsolidationSettings(BaseSettings):
    """Configuration for consolidation pipeline."""
    enable_summarization: bool = Field(default=True, description="Enable session summarization")
    enable_fact_extraction: bool = Field(default=True, description="Enable fact extraction")
    enable_knowledge_graph: bool = Field(default=True, description="Enable KG updates")
    max_summary_tokens: int = Field(default=500, description="Max summary length")
    max_facts_per_session: int = Field(default=20, description="Max facts to extract")
    min_message_count_for_summary: int = Field(default=3, description="Min messages for summary")
    decay_base_rate: Decimal = Field(default=Decimal("0.1"), description="Base decay rate per day")
    decay_medium_term_rate: Decimal = Field(default=Decimal("0.05"), description="Medium term decay")
    decay_short_term_rate: Decimal = Field(default=Decimal("0.15"), description="Short term decay")
    archive_threshold: Decimal = Field(default=Decimal("0.3"), description="Archive below this")
    delete_threshold: Decimal = Field(default=Decimal("0.1"), description="Delete below this")
    reinforcement_multiplier: Decimal = Field(default=Decimal("1.5"), description="Reinforcement boost")
    model_config = SettingsConfigDict(env_prefix="CONSOLIDATION_", env_file=".env", extra="ignore")


class ConsolidationPhase(str, Enum):
    """Phases of the consolidation pipeline."""
    PENDING = "pending"
    SUMMARIZING = "summarizing"
    EXTRACTING_FACTS = "extracting_facts"
    UPDATING_GRAPH = "updating_graph"
    APPLYING_DECAY = "applying_decay"
    ARCHIVING = "archiving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractedFact:
    """A fact extracted from session memory."""
    fact_id: UUID = field(default_factory=uuid4)
    content: str = ""
    fact_type: str = "general"
    confidence: Decimal = Decimal("0.7")
    importance: Decimal = Decimal("0.5")
    retention_category: str = "medium_term"
    source_session_id: UUID | None = None
    related_entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# KnowledgeTriple imported from semantic_memory to avoid duplicate definitions (M37).
# The semantic_memory version is a superset (adds user_id, metadata with defaults).
from .semantic_memory import KnowledgeTriple  # noqa: E402


class SummaryResult(BaseModel):
    """Result from summary generation."""
    summary: str = Field(default="")
    key_topics: list[str] = Field(default_factory=list)
    emotional_arc: list[str] = Field(default_factory=list)
    techniques_used: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0)


class ConsolidationOutput(BaseModel):
    """Output from consolidation pipeline."""
    consolidation_id: UUID = Field(default_factory=uuid4)
    session_id: UUID | None = Field(default=None)
    phase: ConsolidationPhase = Field(default=ConsolidationPhase.COMPLETED)
    summary_generated: str | None = Field(default=None)
    facts_extracted: int = Field(default=0)
    extracted_facts: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_nodes_updated: int = Field(default=0)
    triples_created: list[dict[str, Any]] = Field(default_factory=list)
    memories_decayed: int = Field(default=0)
    memories_archived: int = Field(default=0)
    memories_deleted: int = Field(default=0)
    consolidation_time_ms: int = Field(default=0)
    error: str | None = Field(default=None)


class ConsolidationPipeline:
    """Memory consolidation pipeline for session end processing."""

    def __init__(self, settings: ConsolidationSettings | None = None) -> None:
        self._settings = settings or ConsolidationSettings()
        self._fact_patterns = self._compile_fact_patterns()
        self._triple_patterns = self._compile_triple_patterns()
        self._therapeutic_techniques = self._load_therapeutic_techniques()
        self._topic_keywords = self._load_topic_keywords()

    async def consolidate(self, user_id: UUID, session_id: UUID, records: list[Any],
                          extract_facts: bool, generate_summary: bool,
                          update_knowledge_graph: bool, apply_decay: bool) -> ConsolidationOutput:
        """Run the full consolidation pipeline."""
        start_time = time.perf_counter()
        output = ConsolidationOutput(session_id=session_id, phase=ConsolidationPhase.PENDING)
        try:
            if generate_summary and len(records) >= self._settings.min_message_count_for_summary:
                output.phase = ConsolidationPhase.SUMMARIZING
                summary_result = await self.generate_summary(records)
                output.summary_generated = summary_result.summary
            facts: list[ExtractedFact] = []
            if extract_facts:
                output.phase = ConsolidationPhase.EXTRACTING_FACTS
                facts = await self._extract_facts(records, session_id)
                output.facts_extracted = len(facts)
                output.extracted_facts = [self._fact_to_dict(f) for f in facts]
            if update_knowledge_graph and facts:
                output.phase = ConsolidationPhase.UPDATING_GRAPH
                triples = await self._build_knowledge_triples(facts)
                output.knowledge_nodes_updated = len(triples)
                output.triples_created = [self._triple_to_dict(t) for t in triples]
            if apply_decay:
                output.phase = ConsolidationPhase.APPLYING_DECAY
                decayed, archived, deleted = await self._apply_decay(records)
                output.memories_decayed = decayed
                output.memories_archived = archived
                output.memories_deleted = deleted
            output.phase = ConsolidationPhase.COMPLETED
        except Exception as e:
            logger.error("consolidation_failed", session_id=str(session_id), error=str(e))
            output.phase = ConsolidationPhase.FAILED
            output.error = str(e)
        output.consolidation_time_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("consolidation_completed", session_id=str(session_id),
                    facts=output.facts_extracted, phase=output.phase.value,
                    time_ms=output.consolidation_time_ms)
        return output

    async def generate_summary(self, records: list[Any]) -> SummaryResult:
        """Generate a summary of the session."""
        if not records:
            return SummaryResult()
        messages = []
        emotions = []
        for record in records:
            content = record.content if hasattr(record, 'content') else str(record)
            messages.append(content)
            if hasattr(record, 'metadata'):
                if emotion := record.metadata.get('emotion'):
                    emotions.append(emotion)
        key_topics = self._extract_topics(messages)
        techniques = self._identify_techniques(messages)
        emotional_arc = self._analyze_emotional_arc(emotions)
        summary_parts = []
        summary_parts.append(f"Session with {len(messages)} exchanges.")
        if key_topics:
            summary_parts.append(f"Main topics: {', '.join(key_topics[:5])}.")
        if techniques:
            summary_parts.append(f"Therapeutic approaches: {', '.join(techniques[:3])}.")
        if emotional_arc:
            summary_parts.append(f"Emotional progression: {' -> '.join(emotional_arc)}.")
        user_messages = [m for r, m in zip(records, messages) if hasattr(r, 'metadata') and r.metadata.get('role') == 'user']
        if user_messages:
            key_concerns = self._extract_key_concerns(user_messages)
            if key_concerns:
                summary_parts.append(f"Key concerns: {', '.join(key_concerns[:3])}.")
        summary = " ".join(summary_parts)
        if len(summary) > self._settings.max_summary_tokens * 4:
            summary = summary[:self._settings.max_summary_tokens * 4 - 3] + "..."
        return SummaryResult(
            summary=summary, key_topics=key_topics,
            emotional_arc=emotional_arc, techniques_used=techniques,
            token_count=len(summary) // 4,
        )

    async def _extract_facts(self, records: list[Any], session_id: UUID) -> list[ExtractedFact]:
        """Extract facts from session records."""
        facts: list[ExtractedFact] = []
        for record in records:
            content = record.content if hasattr(record, 'content') else str(record)
            if hasattr(record, 'metadata') and record.metadata.get('role') == 'user':
                extracted = self._extract_facts_from_text(content, session_id)
                facts.extend(extracted)
                if len(facts) >= self._settings.max_facts_per_session:
                    break
        seen_contents = set()
        unique_facts = []
        for fact in facts:
            normalized = fact.content.lower().strip()
            if normalized not in seen_contents:
                seen_contents.add(normalized)
                unique_facts.append(fact)
        return unique_facts[:self._settings.max_facts_per_session]

    def _extract_facts_from_text(self, text: str, session_id: UUID) -> list[ExtractedFact]:
        """Extract facts from a single text."""
        facts: list[ExtractedFact] = []
        for pattern_name, pattern in self._fact_patterns.items():
            for match in pattern.finditer(text):
                content = match.group(0).strip()
                if len(content) > 10:
                    fact_type, retention, importance = self._classify_fact(content, pattern_name)
                    facts.append(ExtractedFact(
                        content=content, fact_type=fact_type,
                        importance=importance, retention_category=retention,
                        source_session_id=session_id,
                        related_entities=self._extract_entities(content),
                    ))
        return facts

    def _classify_fact(self, content: str, pattern_name: str) -> tuple[str, str, Decimal]:
        """Classify a fact's type, retention, and importance."""
        content_lower = content.lower()
        safety_keywords = ["crisis", "suicide", "harm", "emergency", "danger"]
        if any(kw in content_lower for kw in safety_keywords):
            return "safety_critical", "permanent", Decimal("1.0")
        relationship_keywords = ["family", "friend", "partner", "wife", "husband", "mother", "father"]
        if any(kw in content_lower for kw in relationship_keywords):
            return "relationship", "long_term", Decimal("0.8")
        therapeutic_keywords = ["therapy", "medication", "treatment", "diagnosis"]
        if any(kw in content_lower for kw in therapeutic_keywords):
            return "therapeutic", "long_term", Decimal("0.9")
        preference_keywords = ["prefer", "like", "enjoy", "hate", "love"]
        if any(kw in content_lower for kw in preference_keywords):
            return "preference", "medium_term", Decimal("0.6")
        return "general", "medium_term", Decimal("0.5")

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text (simplified)."""
        entities = []
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 1:
                if i == 0 or words[i-1].endswith(('.', '!', '?', ':')):
                    continue
                entities.append(word.strip('.,!?'))
        return list(set(entities))[:5]

    async def _build_knowledge_triples(self, facts: list[ExtractedFact]) -> list[KnowledgeTriple]:
        """Build knowledge graph triples from extracted facts."""
        triples: list[KnowledgeTriple] = []
        for fact in facts:
            for pattern_name, pattern in self._triple_patterns.items():
                match = pattern.search(fact.content)
                if match:
                    groups = match.groups()
                    if len(groups) >= 3:
                        triple = KnowledgeTriple(
                            subject=groups[0].strip() if groups[0] else "User",
                            predicate=groups[1].strip() if groups[1] else pattern_name,
                            object_value=groups[2].strip() if groups[2] else "",
                            confidence=fact.confidence,
                            source_fact_id=fact.fact_id,
                        )
                        if triple.object_value:
                            triples.append(triple)
                        break
            else:
                triple = KnowledgeTriple(
                    subject="User", predicate=f"has_{fact.fact_type}",
                    object_value=fact.content[:100], confidence=fact.confidence,
                    source_fact_id=fact.fact_id,
                )
                triples.append(triple)
        return triples

    async def _apply_decay(self, records: list[Any]) -> tuple[int, int, int]:
        """Apply Ebbinghaus decay model to records."""
        decayed = 0
        archived = 0
        deleted = 0
        for record in records:
            if not hasattr(record, 'retention_category'):
                continue
            if record.retention_category == "permanent":
                continue
            age_days = (datetime.now(timezone.utc) - record.created_at).days if hasattr(record, 'created_at') else 0
            if record.retention_category == "short_term":
                decay_rate = self._settings.decay_short_term_rate
            elif record.retention_category == "medium_term":
                decay_rate = self._settings.decay_medium_term_rate
            else:
                decay_rate = self._settings.decay_base_rate
            current_strength = getattr(record, 'retention_strength', Decimal("1.0"))
            decay_factor = decay_rate * Decimal(str(age_days)) / Decimal("30")
            new_strength = max(Decimal("0.0"), current_strength - decay_factor)
            if hasattr(record, 'retention_strength'):
                record.retention_strength = new_strength
            decayed += 1
            if new_strength < self._settings.delete_threshold:
                deleted += 1
            elif new_strength < self._settings.archive_threshold:
                archived += 1
        return decayed, archived, deleted

    def _extract_topics(self, messages: list[str]) -> list[str]:
        """Extract key topics from messages."""
        topics: dict[str, int] = {}
        all_text = " ".join(messages).lower()
        for topic, keywords in self._topic_keywords.items():
            count = sum(1 for kw in keywords if kw in all_text)
            if count > 0:
                topics[topic] = count
        return sorted(topics.keys(), key=lambda t: topics[t], reverse=True)[:5]

    def _identify_techniques(self, messages: list[str]) -> list[str]:
        """Identify therapeutic techniques used in session."""
        techniques = []
        all_text = " ".join(messages).lower()
        for technique, indicators in self._therapeutic_techniques.items():
            if any(ind in all_text for ind in indicators):
                techniques.append(technique)
        return techniques[:5]

    def _analyze_emotional_arc(self, emotions: list[str]) -> list[str]:
        """Analyze the emotional arc of the session."""
        if not emotions: return []
        if len(emotions) <= 2: return [emotions[0], emotions[-1]] if len(emotions) == 2 else emotions
        return [emotions[0], emotions[len(emotions)//2], emotions[-1]]

    def _extract_key_concerns(self, user_messages: list[str]) -> list[str]:
        """Extract key concerns from user messages."""
        concerns = []
        concern_indicators = ["worried about", "struggling with", "having trouble", "feeling",
                             "can't", "hard to", "difficult", "anxious about", "stressed about"]
        for msg in user_messages:
            msg_lower = msg.lower()
            for indicator in concern_indicators:
                if indicator in msg_lower:
                    idx = msg_lower.find(indicator)
                    concern = msg[idx:idx+50].strip()
                    if len(concern) > 15:
                        concerns.append(concern.split('.')[0])
                    break
        return list(set(concerns))[:3]

    def _compile_fact_patterns(self) -> dict[str, re.Pattern]:
        """Compile regex patterns for fact extraction."""
        I = re.IGNORECASE
        return {"personal_info": re.compile(r"(?:my|i have|i am|i\'m)\s+(?:a\s+)?([A-Za-z\s]{3,30})", I),
                "relationship": re.compile(r"(?:my\s+)?(mother|father|sister|brother|friend|partner|wife|husband|child)\s+([A-Za-z]+)", I),
                "feeling": re.compile(r"(?:i feel|feeling|i\'m feeling)\s+([a-z\s]{3,20})", I),
                "preference": re.compile(r"(?:i\s+(?:really\s+)?(?:like|love|hate|prefer|enjoy))\s+([a-z\s]{3,30})", I),
                "work": re.compile(r"(?:i work|working|my job|my work)\s+(?:as|at|in)?\s*([A-Za-z\s]{3,30})", I)}

    def _compile_triple_patterns(self) -> dict[str, re.Pattern]:
        """Compile patterns for knowledge triple extraction."""
        I = re.IGNORECASE
        return {"has_relationship": re.compile(r"(?:my\s+)?(\w+)\s+(is|are|was)\s+(?:my\s+)?(\w+)", I),
                "works_as": re.compile(r"(I|my\s+\w+)\s+(?:work|works)\s+(?:as|at)\s+(.+)", I),
                "feels": re.compile(r"(I|they|he|she)\s+(?:feel|feels)\s+(\w+)\s+(?:about|when)\s+(.+)", I)}

    def _load_therapeutic_techniques(self) -> dict[str, list[str]]:
        """Load therapeutic technique indicators."""
        return {"cognitive_restructuring": ["thought", "thinking", "belief", "perspective", "reframe"],
                "mindfulness": ["mindful", "present moment", "breathing", "meditation", "aware"],
                "behavioral_activation": ["activity", "schedule", "do something", "engagement"],
                "emotion_regulation": ["emotion", "feeling", "regulate", "manage", "cope"],
                "exposure": ["face", "confront", "approach", "gradually", "exposure"],
                "validation": ["understand", "valid", "makes sense", "natural", "normal"],
                "problem_solving": ["solution", "options", "solve", "plan", "strategy"]}

    def _load_topic_keywords(self) -> dict[str, list[str]]:
        """Load topic detection keywords."""
        return {"anxiety": ["anxious", "worry", "nervous", "panic", "fear"],
                "depression": ["sad", "hopeless", "depressed", "empty", "worthless"],
                "relationships": ["relationship", "partner", "family", "friend", "conflict"],
                "work_stress": ["work", "job", "boss", "career", "workplace"],
                "sleep": ["sleep", "insomnia", "tired", "rest", "fatigue"],
                "self_esteem": ["confidence", "self-esteem", "self-worth", "inadequate"],
                "grief": ["loss", "grief", "death", "mourning", "passed away"],
                "trauma": ["trauma", "abuse", "ptsd", "flashback", "nightmare"]}

    def _fact_to_dict(self, fact: ExtractedFact) -> dict[str, Any]:
        """Convert fact to dictionary."""
        return {"fact_id": str(fact.fact_id), "content": fact.content, "fact_type": fact.fact_type,
                "confidence": float(fact.confidence), "importance": float(fact.importance),
                "retention": fact.retention_category, "related_entities": fact.related_entities}

    def _triple_to_dict(self, triple: KnowledgeTriple) -> dict[str, Any]:
        """Convert triple to dictionary."""
        return {"triple_id": str(triple.triple_id), "subject": triple.subject,
                "predicate": triple.predicate, "object": triple.object_value, "confidence": float(triple.confidence)}
