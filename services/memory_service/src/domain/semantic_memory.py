"""
Solace-AI Memory Service - Semantic Memory (Tier 5).
Manages persistent user knowledge, facts, and knowledge graph.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class SemanticMemorySettings(BaseSettings):
    """Configuration for semantic memory behavior."""
    max_facts_per_user: int = Field(default=500, description="Max facts per user")
    max_triples_per_user: int = Field(default=1000, description="Max knowledge triples")
    min_confidence_threshold: float = Field(default=0.5, description="Min confidence to store")
    enable_versioning: bool = Field(default=True, description="Enable fact versioning")
    conflict_resolution_mode: str = Field(default="latest", description="latest|highest_confidence|manual")
    model_config = SettingsConfigDict(env_prefix="SEMANTIC_MEMORY_", env_file=".env", extra="ignore")


class FactCategory(str, Enum):
    """Categories of stored facts."""
    PERMANENT = "permanent"
    PERSONAL = "personal"
    RELATIONSHIP = "relationship"
    THERAPEUTIC = "therapeutic"
    PREFERENCE = "preference"
    SAFETY = "safety"
    GENERAL = "general"


class FactStatus(str, Enum):
    """Status of a fact."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    ARCHIVED = "archived"


@dataclass
class UserFact:
    """A fact about the user (Tier 5 primary storage)."""
    fact_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    category: FactCategory = FactCategory.GENERAL
    content: str = ""
    confidence: Decimal = Decimal("0.7")
    importance: Decimal = Decimal("0.5")
    status: FactStatus = FactStatus.ACTIVE
    source_session_id: UUID | None = None
    version: int = 1
    supersedes: UUID | None = None
    verified_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    related_entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeTriple:
    """A knowledge graph triple (subject, predicate, object)."""
    triple_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    subject: str = ""
    predicate: str = ""
    object_value: str = ""
    confidence: Decimal = Decimal("0.7")
    source_fact_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    entity_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    name: str = ""
    entity_type: str = "unknown"
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeGraphQuery(BaseModel):
    """Query for knowledge graph traversal."""
    user_id: UUID
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    min_confidence: float = Field(default=0.5)
    limit: int = Field(default=50, le=200)


class SemanticMemoryManager:
    """Manages Tier 5 Semantic Memory - facts and knowledge graph."""

    def __init__(self, settings: SemanticMemorySettings | None = None) -> None:
        self._settings = settings or SemanticMemorySettings()
        self._facts: dict[UUID, list[UserFact]] = {}
        self._triples: dict[UUID, list[KnowledgeTriple]] = {}
        self._entities: dict[UUID, dict[str, Entity]] = {}
        self._stats = {"facts_stored": 0, "facts_updated": 0, "triples_stored": 0,
                       "queries": 0, "conflicts_resolved": 0}

    def store_fact(self, user_id: UUID, content: str, category: FactCategory,
                   confidence: Decimal, importance: Decimal,
                   source_session_id: UUID | None = None,
                   related_entities: list[str] | None = None,
                   metadata: dict[str, Any] | None = None) -> UserFact | None:
        """Store a new fact about the user."""
        if float(confidence) < self._settings.min_confidence_threshold:
            logger.debug("fact_rejected_low_confidence", confidence=float(confidence))
            return None
        existing = self._find_conflicting_fact(user_id, content, category)
        if existing:
            return self._resolve_conflict(existing, content, confidence, importance, source_session_id)
        self._stats["facts_stored"] += 1
        fact = UserFact(
            user_id=user_id, category=category, content=content,
            confidence=confidence, importance=importance,
            source_session_id=source_session_id,
            related_entities=related_entities or [], metadata=metadata or {},
        )
        facts = self._facts.setdefault(user_id, [])
        facts.append(fact)
        self._enforce_fact_limit(user_id)
        for entity_name in fact.related_entities:
            self._ensure_entity(user_id, entity_name)
        logger.info("fact_stored", user_id=str(user_id), category=category.value,
                    confidence=float(confidence))
        return fact

    def get_fact(self, fact_id: UUID) -> UserFact | None:
        """Get a specific fact by ID."""
        for facts in self._facts.values():
            for fact in facts:
                if fact.fact_id == fact_id:
                    return fact
        return None

    def get_facts_by_category(self, user_id: UUID, category: FactCategory,
                              active_only: bool = True) -> list[UserFact]:
        """Get facts in a specific category."""
        self._stats["queries"] += 1
        facts = self._facts.get(user_id, [])
        filtered = [f for f in facts if f.category == category]
        if active_only:
            filtered = [f for f in filtered if f.status == FactStatus.ACTIVE]
        return sorted(filtered, key=lambda f: f.importance, reverse=True)

    def get_all_active_facts(self, user_id: UUID) -> list[UserFact]:
        """Get all active facts for user."""
        self._stats["queries"] += 1
        facts = self._facts.get(user_id, [])
        return [f for f in facts if f.status == FactStatus.ACTIVE]

    def get_safety_facts(self, user_id: UUID) -> list[UserFact]:
        """Get safety-critical facts (never decay)."""
        facts = self._facts.get(user_id, [])
        return [f for f in facts if f.category == FactCategory.SAFETY and f.status == FactStatus.ACTIVE]

    def update_fact(self, fact_id: UUID, new_content: str | None = None,
                    new_confidence: Decimal | None = None,
                    new_importance: Decimal | None = None) -> UserFact | None:
        """Update an existing fact."""
        fact = self.get_fact(fact_id)
        if not fact or fact.status != FactStatus.ACTIVE:
            return None
        self._stats["facts_updated"] += 1
        if self._settings.enable_versioning:
            old_fact = fact
            old_fact.status = FactStatus.SUPERSEDED
            new_fact = UserFact(
                user_id=fact.user_id, category=fact.category,
                content=new_content or fact.content,
                confidence=new_confidence or fact.confidence,
                importance=new_importance or fact.importance,
                source_session_id=fact.source_session_id,
                version=fact.version + 1, supersedes=fact.fact_id,
                related_entities=fact.related_entities, metadata=fact.metadata,
            )
            self._facts[fact.user_id].append(new_fact)
            return new_fact
        if new_content:
            fact.content = new_content
        if new_confidence:
            fact.confidence = new_confidence
        if new_importance:
            fact.importance = new_importance
        fact.updated_at = datetime.now(timezone.utc)
        return fact

    def verify_fact(self, fact_id: UUID) -> bool:
        """Mark a fact as verified."""
        fact = self.get_fact(fact_id)
        if fact:
            fact.verified_at = datetime.now(timezone.utc)
            fact.confidence = min(Decimal("1.0"), fact.confidence + Decimal("0.1"))
            return True
        return False

    def dispute_fact(self, fact_id: UUID) -> bool:
        """Mark a fact as disputed."""
        fact = self.get_fact(fact_id)
        if fact:
            fact.status = FactStatus.DISPUTED
            return True
        return False

    def store_triple(self, user_id: UUID, subject: str, predicate: str, object_value: str,
                     confidence: Decimal, source_fact_id: UUID | None = None,
                     metadata: dict[str, Any] | None = None) -> KnowledgeTriple:
        """Store a knowledge graph triple."""
        self._stats["triples_stored"] += 1
        triple = KnowledgeTriple(
            user_id=user_id, subject=subject, predicate=predicate,
            object_value=object_value, confidence=confidence,
            source_fact_id=source_fact_id, metadata=metadata or {},
        )
        triples = self._triples.setdefault(user_id, [])
        triples.append(triple)
        self._enforce_triple_limit(user_id)
        self._ensure_entity(user_id, subject)
        self._ensure_entity(user_id, object_value)
        logger.debug("triple_stored", user_id=str(user_id), subject=subject, predicate=predicate)
        return triple

    def query_knowledge_graph(self, query: KnowledgeGraphQuery) -> list[KnowledgeTriple]:
        """Query the knowledge graph."""
        self._stats["queries"] += 1
        triples = self._triples.get(query.user_id, [])
        if query.subject:
            triples = [t for t in triples if query.subject.lower() in t.subject.lower()]
        if query.predicate:
            triples = [t for t in triples if query.predicate.lower() in t.predicate.lower()]
        if query.object_value:
            triples = [t for t in triples if query.object_value.lower() in t.object_value.lower()]
        triples = [t for t in triples if float(t.confidence) >= query.min_confidence]
        return sorted(triples, key=lambda t: t.confidence, reverse=True)[:query.limit]

    def get_entity_relationships(self, user_id: UUID, entity_name: str) -> list[KnowledgeTriple]:
        """Get all relationships for an entity."""
        triples = self._triples.get(user_id, [])
        name_lower = entity_name.lower()
        return [t for t in triples if name_lower in t.subject.lower() or name_lower in t.object_value.lower()]

    def get_user_profile_facts(self, user_id: UUID) -> dict[str, Any]:
        """Get structured user profile from facts."""
        facts = self.get_all_active_facts(user_id)
        profile: dict[str, Any] = {"personal": [], "relationships": [], "therapeutic": [],
                                   "preferences": [], "safety": []}
        for fact in facts:
            key = fact.category.value if fact.category.value in profile else "personal"
            profile[key].append({"content": fact.content, "confidence": float(fact.confidence),
                                 "verified": fact.verified_at is not None})
        return profile

    def get_knowledge_graph_summary(self, user_id: UUID) -> dict[str, Any]:
        """Get summary of knowledge graph for context."""
        triples = self._triples.get(user_id, [])
        entities = self._entities.get(user_id, {})
        predicates: dict[str, int] = {}
        for triple in triples:
            predicates[triple.predicate] = predicates.get(triple.predicate, 0) + 1
        return {
            "total_triples": len(triples),
            "total_entities": len(entities),
            "entity_names": list(entities.keys())[:20],
            "relationship_types": sorted(predicates.keys(), key=lambda p: predicates[p], reverse=True)[:10],
            "avg_confidence": sum(float(t.confidence) for t in triples) / len(triples) if triples else 0,
        }

    def search_facts(self, user_id: UUID, query: str, limit: int = 10) -> list[UserFact]:
        """Search facts by content."""
        self._stats["queries"] += 1
        facts = self._facts.get(user_id, [])
        query_lower = query.lower()
        matching = [f for f in facts if f.status == FactStatus.ACTIVE and query_lower in f.content.lower()]
        return sorted(matching, key=lambda f: (f.importance, f.confidence), reverse=True)[:limit]

    def get_therapeutic_context(self, user_id: UUID) -> dict[str, Any]:
        """Get therapeutic-relevant facts for LLM context."""
        safety = self.get_safety_facts(user_id)
        therapeutic = self.get_facts_by_category(user_id, FactCategory.THERAPEUTIC)
        relationships = self.get_facts_by_category(user_id, FactCategory.RELATIONSHIP)
        return {
            "safety_critical": [f.content for f in safety],
            "therapeutic_facts": [f.content for f in therapeutic[:10]],
            "key_relationships": [f.content for f in relationships[:10]],
            "has_safety_concerns": len(safety) > 0,
        }

    def delete_user_data(self, user_id: UUID) -> tuple[int, int, int]:
        """Delete all semantic memory for user (GDPR)."""
        facts_deleted = len(self._facts.pop(user_id, []))
        triples_deleted = len(self._triples.pop(user_id, []))
        entities_deleted = len(self._entities.pop(user_id, {}))
        logger.info("semantic_memory_deleted", user_id=str(user_id),
                    facts=facts_deleted, triples=triples_deleted, entities=entities_deleted)
        return facts_deleted, triples_deleted, entities_deleted

    def get_statistics(self) -> dict[str, Any]:
        """Get semantic memory statistics."""
        total_facts = sum(len(f) for f in self._facts.values())
        total_triples = sum(len(t) for t in self._triples.values())
        total_entities = sum(len(e) for e in self._entities.values())
        return {**self._stats, "total_facts": total_facts, "total_triples": total_triples,
                "total_entities": total_entities, "users_tracked": len(self._facts)}

    def _find_conflicting_fact(self, user_id: UUID, content: str,
                               category: FactCategory) -> UserFact | None:
        """Find existing fact that conflicts with new content."""
        facts = self._facts.get(user_id, [])
        content_lower = content.lower()
        for fact in facts:
            if fact.status != FactStatus.ACTIVE:
                continue
            if fact.category != category:
                continue
            existing_lower = fact.content.lower()
            if content_lower in existing_lower or existing_lower in content_lower:
                return fact
        return None

    def _resolve_conflict(self, existing: UserFact, new_content: str,
                          new_confidence: Decimal, new_importance: Decimal,
                          source_session_id: UUID | None) -> UserFact:
        """Resolve conflict between existing and new fact."""
        self._stats["conflicts_resolved"] += 1
        if self._settings.conflict_resolution_mode == "highest_confidence":
            if new_confidence > existing.confidence:
                return self.update_fact(existing.fact_id, new_content, new_confidence, new_importance) or existing
            return existing
        return self.update_fact(existing.fact_id, new_content, new_confidence, new_importance) or existing

    def _ensure_entity(self, user_id: UUID, name: str) -> None:
        """Ensure entity exists in graph."""
        if not name or len(name) < 2:
            return
        entities = self._entities.setdefault(user_id, {})
        name_key = name.lower().strip()
        if name_key not in entities:
            entities[name_key] = Entity(user_id=user_id, name=name)

    def _enforce_fact_limit(self, user_id: UUID) -> None:
        """Enforce max facts per user."""
        facts = self._facts.get(user_id, [])
        if len(facts) > self._settings.max_facts_per_user:
            safety = [f for f in facts if f.category == FactCategory.SAFETY]
            others = [f for f in facts if f.category != FactCategory.SAFETY]
            others.sort(key=lambda f: (f.status == FactStatus.ACTIVE, f.importance, f.confidence))
            keep = self._settings.max_facts_per_user - len(safety)
            self._facts[user_id] = safety + others[-keep:] if keep > 0 else safety

    def _enforce_triple_limit(self, user_id: UUID) -> None:
        """Enforce max triples per user."""
        triples = self._triples.get(user_id, [])
        if len(triples) > self._settings.max_triples_per_user:
            triples.sort(key=lambda t: (t.confidence, t.created_at))
            self._triples[user_id] = triples[-self._settings.max_triples_per_user:]
