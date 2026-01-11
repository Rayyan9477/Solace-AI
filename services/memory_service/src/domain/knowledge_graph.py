"""
Solace-AI Memory Service - Knowledge Graph.

Implements temporal knowledge graph for semantic memory storage.
Supports triple extraction, graph queries, and relationship tracking.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Iterator
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
import structlog

logger = structlog.get_logger(__name__)


class RelationType(str, Enum):
    """Types of relationships in the knowledge graph."""
    HAS = "has"
    IS = "is"
    FEELS = "feels"
    EXPERIENCED = "experienced"
    KNOWS = "knows"
    WORKS_AT = "works_at"
    LIVES_IN = "lives_in"
    RELATED_TO = "related_to"
    DIAGNOSED_WITH = "diagnosed_with"
    TAKES = "takes"
    PRACTICES = "practices"
    TRIGGERS = "triggers"
    HELPS_WITH = "helps_with"
    SUPPORTS = "supports"


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    USER = "user"
    PERSON = "person"
    EMOTION = "emotion"
    CONDITION = "condition"
    MEDICATION = "medication"
    LOCATION = "location"
    ORGANIZATION = "organization"
    ACTIVITY = "activity"
    SYMPTOM = "symptom"
    TECHNIQUE = "technique"
    TRIGGER = "trigger"
    GOAL = "goal"
    EVENT = "event"


class KnowledgeTriple(BaseModel):
    """A subject-predicate-object triple in the knowledge graph."""
    triple_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    subject: str = Field(min_length=1, max_length=500)
    subject_type: EntityType
    predicate: RelationType
    object: str = Field(min_length=1, max_length=500)
    object_type: EntityType
    confidence: Decimal = Field(default=Decimal("0.7"), ge=Decimal("0"), le=Decimal("1"))
    source_session_id: UUID | None = None
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: datetime | None = None
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(frozen=True)

    def to_natural_language(self) -> str:
        """Convert triple to natural language sentence."""
        predicate_text = self.predicate.value.replace("_", " ")
        return f"{self.subject} {predicate_text} {self.object}"

    def matches_pattern(self, subject: str | None = None, predicate: RelationType | None = None, obj: str | None = None) -> bool:
        """Check if triple matches a pattern (None = wildcard)."""
        if subject and self.subject.lower() != subject.lower():
            return False
        if predicate and self.predicate != predicate:
            return False
        if obj and self.object.lower() != obj.lower():
            return False
        return True

    def invalidate(self) -> KnowledgeTriple:
        """Mark triple as no longer valid."""
        return self.model_copy(update={"is_active": False, "valid_to": datetime.now(timezone.utc)})


class GraphEntity(BaseModel):
    """An entity node in the knowledge graph."""
    entity_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    name: str = Field(min_length=1, max_length=500)
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    importance: Decimal = Field(default=Decimal("0.5"), ge=Decimal("0"), le=Decimal("1"))
    mention_count: int = Field(default=1, ge=1)
    first_mentioned: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_mentioned: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(frozen=True)

    def record_mention(self) -> GraphEntity:
        """Record a new mention of this entity."""
        return self.model_copy(update={"mention_count": self.mention_count + 1, "last_mentioned": datetime.now(timezone.utc)})

    def add_alias(self, alias: str) -> GraphEntity:
        """Add an alias for this entity."""
        if alias.lower() not in [a.lower() for a in self.aliases]:
            return self.model_copy(update={"aliases": [*self.aliases, alias]})
        return self


class TripleExtractor:
    """Extracts knowledge triples from conversation text."""

    PATTERNS = [
        (r"my name is (\w+)", EntityType.USER, RelationType.IS, EntityType.PERSON, lambda m: ("user", m.group(1))),
        (r"I(?:'m| am) (\w+)", EntityType.USER, RelationType.IS, EntityType.PERSON, lambda m: ("user", m.group(1))),
        (r"I work at (.+?)(?:\.|,|$)", EntityType.USER, RelationType.WORKS_AT, EntityType.ORGANIZATION, lambda m: ("user", m.group(1).strip())),
        (r"I live in (.+?)(?:\.|,|$)", EntityType.USER, RelationType.LIVES_IN, EntityType.LOCATION, lambda m: ("user", m.group(1).strip())),
        (r"I(?:'m| am) feeling (\w+)", EntityType.USER, RelationType.FEELS, EntityType.EMOTION, lambda m: ("user", m.group(1))),
        (r"I feel (\w+)", EntityType.USER, RelationType.FEELS, EntityType.EMOTION, lambda m: ("user", m.group(1))),
        (r"I have (\w+)", EntityType.USER, RelationType.HAS, EntityType.CONDITION, lambda m: ("user", m.group(1))),
        (r"I(?:'m| am) diagnosed with (.+?)(?:\.|,|$)", EntityType.USER, RelationType.DIAGNOSED_WITH, EntityType.CONDITION, lambda m: ("user", m.group(1).strip())),
        (r"I take (.+?)(?:\.|,| for|$)", EntityType.USER, RelationType.TAKES, EntityType.MEDICATION, lambda m: ("user", m.group(1).strip())),
        (r"(.+?) helps me", EntityType.TECHNIQUE, RelationType.HELPS_WITH, EntityType.USER, lambda m: (m.group(1).strip(), "user")),
        (r"(.+?) triggers (?:my )?(.+)", EntityType.TRIGGER, RelationType.TRIGGERS, EntityType.SYMPTOM, lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"my (\w+) (\w+)", EntityType.USER, RelationType.HAS, EntityType.PERSON, lambda m: ("user", f"{m.group(1)} {m.group(2)}")),
    ]

    EMOTION_KEYWORDS = {"happy", "sad", "anxious", "angry", "scared", "worried", "hopeful", "frustrated", "calm", "stressed", "depressed", "overwhelmed", "peaceful", "nervous"}
    CONDITION_KEYWORDS = {"anxiety", "depression", "ptsd", "ocd", "adhd", "bipolar", "panic", "phobia", "insomnia", "trauma"}

    def __init__(self, user_id: UUID) -> None:
        self.user_id = user_id
        self._stats = {"texts_processed": 0, "triples_extracted": 0}

    def extract(self, text: str, session_id: UUID | None = None) -> list[KnowledgeTriple]:
        """Extract knowledge triples from text."""
        self._stats["texts_processed"] += 1
        triples: list[KnowledgeTriple] = []
        text_lower = text.lower()
        for pattern, subj_type, predicate, obj_type, extractor in self.PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                try:
                    subject, obj = extractor(match)
                    if len(subject) >= 2 and len(obj) >= 2:
                        triple = KnowledgeTriple(user_id=self.user_id, subject=subject, subject_type=subj_type,
                                                 predicate=predicate, object=obj, object_type=obj_type,
                                                 source_session_id=session_id, confidence=Decimal("0.7"))
                        triples.append(triple)
                except Exception as e:
                    logger.debug("triple_extraction_error", pattern=pattern, error=str(e))
        triples.extend(self._extract_emotions(text, session_id))
        triples.extend(self._extract_conditions(text, session_id))
        self._stats["triples_extracted"] += len(triples)
        return triples

    def _extract_emotions(self, text: str, session_id: UUID | None) -> list[KnowledgeTriple]:
        """Extract emotion-related triples."""
        triples = []
        text_lower = text.lower()
        for emotion in self.EMOTION_KEYWORDS:
            if emotion in text_lower:
                triple = KnowledgeTriple(user_id=self.user_id, subject="user", subject_type=EntityType.USER,
                                         predicate=RelationType.FEELS, object=emotion, object_type=EntityType.EMOTION,
                                         source_session_id=session_id, confidence=Decimal("0.6"))
                triples.append(triple)
        return triples

    def _extract_conditions(self, text: str, session_id: UUID | None) -> list[KnowledgeTriple]:
        """Extract condition-related triples."""
        triples = []
        text_lower = text.lower()
        for condition in self.CONDITION_KEYWORDS:
            if condition in text_lower and ("my " + condition in text_lower or "have " + condition in text_lower or "with " + condition in text_lower):
                triple = KnowledgeTriple(user_id=self.user_id, subject="user", subject_type=EntityType.USER,
                                         predicate=RelationType.DIAGNOSED_WITH, object=condition, object_type=EntityType.CONDITION,
                                         source_session_id=session_id, confidence=Decimal("0.65"))
                triples.append(triple)
        return triples

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)


class KnowledgeGraph:
    """In-memory knowledge graph with query capabilities."""

    def __init__(self, user_id: UUID) -> None:
        self.user_id = user_id
        self._triples: dict[UUID, KnowledgeTriple] = {}
        self._entities: dict[UUID, GraphEntity] = {}
        self._subject_index: dict[str, set[UUID]] = {}
        self._predicate_index: dict[RelationType, set[UUID]] = {}
        self._object_index: dict[str, set[UUID]] = {}
        self._extractor = TripleExtractor(user_id)

    def add_triple(self, triple: KnowledgeTriple) -> UUID:
        """Add a triple to the graph."""
        self._triples[triple.triple_id] = triple
        subj_key = triple.subject.lower()
        if subj_key not in self._subject_index:
            self._subject_index[subj_key] = set()
        self._subject_index[subj_key].add(triple.triple_id)
        if triple.predicate not in self._predicate_index:
            self._predicate_index[triple.predicate] = set()
        self._predicate_index[triple.predicate].add(triple.triple_id)
        obj_key = triple.object.lower()
        if obj_key not in self._object_index:
            self._object_index[obj_key] = set()
        self._object_index[obj_key].add(triple.triple_id)
        self._ensure_entity(triple.subject, triple.subject_type)
        self._ensure_entity(triple.object, triple.object_type)
        logger.debug("triple_added", triple_id=str(triple.triple_id), subject=triple.subject, predicate=triple.predicate.value)
        return triple.triple_id

    def _ensure_entity(self, name: str, entity_type: EntityType) -> None:
        """Ensure entity exists in the graph."""
        name_lower = name.lower()
        for entity in self._entities.values():
            if entity.name.lower() == name_lower or name_lower in [a.lower() for a in entity.aliases]:
                self._entities[entity.entity_id] = entity.record_mention()
                return
        entity = GraphEntity(user_id=self.user_id, name=name, entity_type=entity_type)
        self._entities[entity.entity_id] = entity

    def add_from_text(self, text: str, session_id: UUID | None = None) -> list[UUID]:
        """Extract and add triples from text."""
        triples = self._extractor.extract(text, session_id)
        return [self.add_triple(t) for t in triples]

    def query(self, subject: str | None = None, predicate: RelationType | None = None, obj: str | None = None,
              active_only: bool = True) -> list[KnowledgeTriple]:
        """Query triples by pattern."""
        candidate_ids: set[UUID] | None = None
        if subject:
            subj_key = subject.lower()
            candidate_ids = self._subject_index.get(subj_key, set())
        if predicate:
            pred_ids = self._predicate_index.get(predicate, set())
            candidate_ids = pred_ids if candidate_ids is None else candidate_ids & pred_ids
        if obj:
            obj_key = obj.lower()
            obj_ids = self._object_index.get(obj_key, set())
            candidate_ids = obj_ids if candidate_ids is None else candidate_ids & obj_ids
        if candidate_ids is None:
            candidate_ids = set(self._triples.keys())
        results = []
        for tid in candidate_ids:
            triple = self._triples.get(tid)
            if triple and (not active_only or triple.is_active):
                results.append(triple)
        return results

    def get_entity_relationships(self, entity_name: str, include_incoming: bool = True, include_outgoing: bool = True) -> list[KnowledgeTriple]:
        """Get all relationships for an entity."""
        results = []
        entity_lower = entity_name.lower()
        if include_outgoing:
            results.extend(self.query(subject=entity_name))
        if include_incoming:
            results.extend(self.query(obj=entity_name))
        return results

    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> set[str]:
        """Get entities related to a given entity (breadth-first)."""
        visited: set[str] = {entity_name.lower()}
        current_level = {entity_name.lower()}
        for _ in range(max_depth):
            next_level: set[str] = set()
            for entity in current_level:
                for triple in self.query(subject=entity):
                    obj_lower = triple.object.lower()
                    if obj_lower not in visited:
                        next_level.add(obj_lower)
                        visited.add(obj_lower)
                for triple in self.query(obj=entity):
                    subj_lower = triple.subject.lower()
                    if subj_lower not in visited:
                        next_level.add(subj_lower)
                        visited.add(subj_lower)
            current_level = next_level
            if not current_level:
                break
        return visited - {entity_name.lower()}

    def get_user_facts(self) -> list[str]:
        """Get all facts about the user as natural language."""
        user_triples = self.query(subject="user")
        return [t.to_natural_language() for t in user_triples if t.is_active]

    def invalidate_triple(self, triple_id: UUID) -> bool:
        """Invalidate a triple."""
        if triple_id in self._triples:
            self._triples[triple_id] = self._triples[triple_id].invalidate()
            return True
        return False

    def get_entities(self, entity_type: EntityType | None = None) -> list[GraphEntity]:
        """Get all entities, optionally filtered by type."""
        entities = list(self._entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return sorted(entities, key=lambda e: e.mention_count, reverse=True)

    def to_summary(self) -> dict[str, Any]:
        """Get graph summary."""
        active = sum(1 for t in self._triples.values() if t.is_active)
        return {"total_triples": len(self._triples), "active_triples": active, "total_entities": len(self._entities),
                "predicate_distribution": {p.value: len(ids) for p, ids in self._predicate_index.items()}}

    def clear(self) -> None:
        """Clear the graph."""
        self._triples.clear()
        self._entities.clear()
        self._subject_index.clear()
        self._predicate_index.clear()
        self._object_index.clear()
