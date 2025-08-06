"""
Advanced Semantic Memory Network System
Implements sophisticated memory consolidation, retrieval, and learning mechanisms
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import uuid
from collections import defaultdict
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import asyncpg
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    SEMANTIC = "semantic"          # Long-term factual knowledge
    EPISODIC = "episodic"         # Session-specific experiences
    PROCEDURAL = "procedural"     # Therapeutic techniques and procedures
    METACOGNITIVE = "metacognitive" # Self-awareness and learning


class MemoryImportance(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


@dataclass
class MemoryNode:
    """Individual memory node in the semantic network"""
    node_id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    embedding: np.ndarray = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    confidence: float = 1.0
    tags: Set[str] = field(default_factory=set)
    source: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConnection:
    """Connection between memory nodes"""
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_reinforced: datetime = field(default_factory=datetime.utcnow)
    reinforcement_count: int = 1


@dataclass
class MemoryQuery:
    """Memory query with context"""
    query: str
    memory_types: List[MemoryType] = field(default_factory=lambda: list(MemoryType))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    importance_threshold: MemoryImportance = MemoryImportance.LOW
    max_results: int = 10
    include_connections: bool = True


@dataclass
class MemoryResult:
    """Memory retrieval result"""
    node: MemoryNode
    similarity_score: float
    relevance_score: float
    connected_nodes: List[Tuple[MemoryNode, float]] = field(default_factory=list)


class MemoryStore(ABC):
    """Abstract memory storage interface"""
    
    @abstractmethod
    async def store_node(self, node: MemoryNode) -> bool:
        pass
    
    @abstractmethod
    async def retrieve_node(self, node_id: str) -> Optional[MemoryNode]:
        pass
    
    @abstractmethod
    async def search_nodes(self, query: MemoryQuery) -> List[MemoryResult]:
        pass
    
    @abstractmethod
    async def store_connection(self, connection: MemoryConnection) -> bool:
        pass
    
    @abstractmethod
    async def get_connections(self, node_id: str) -> List[MemoryConnection]:
        pass


class PostgreSQLMemoryStore(MemoryStore):
    """PostgreSQL-based memory storage with vector support"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize database connection and tables"""
        self.pool = await asyncpg.create_pool(self.connection_string)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    node_id VARCHAR PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type VARCHAR NOT NULL,
                    importance INTEGER NOT NULL,
                    embedding vector(768),
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    confidence FLOAT DEFAULT 1.0,
                    tags TEXT[],
                    source VARCHAR,
                    user_id VARCHAR,
                    session_id VARCHAR,
                    metadata JSONB
                );
                
                CREATE TABLE IF NOT EXISTS memory_connections (
                    source_id VARCHAR NOT NULL,
                    target_id VARCHAR NOT NULL,
                    relationship_type VARCHAR NOT NULL,
                    strength FLOAT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_reinforced TIMESTAMP NOT NULL,
                    reinforcement_count INTEGER DEFAULT 1,
                    PRIMARY KEY (source_id, target_id, relationship_type)
                );
                
                CREATE INDEX IF NOT EXISTS idx_memory_nodes_embedding ON memory_nodes 
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                
                CREATE INDEX IF NOT EXISTS idx_memory_nodes_user ON memory_nodes (user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_nodes_session ON memory_nodes (session_id);
                CREATE INDEX IF NOT EXISTS idx_memory_nodes_type ON memory_nodes (memory_type);
                CREATE INDEX IF NOT EXISTS idx_memory_connections_source ON memory_connections (source_id);
            """)
            
    async def store_node(self, node: MemoryNode) -> bool:
        """Store memory node"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO memory_nodes 
                    (node_id, content, memory_type, importance, embedding, created_at, 
                     last_accessed, access_count, confidence, tags, source, user_id, 
                     session_id, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (node_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        last_accessed = EXCLUDED.last_accessed,
                        access_count = memory_nodes.access_count + 1,
                        confidence = EXCLUDED.confidence,
                        tags = EXCLUDED.tags,
                        metadata = EXCLUDED.metadata
                """, 
                node.node_id, node.content, node.memory_type.value, node.importance.value,
                node.embedding.tolist() if node.embedding is not None else None,
                node.created_at, node.last_accessed, node.access_count, node.confidence,
                list(node.tags), node.source, node.user_id, node.session_id, 
                json.dumps(node.metadata))
                return True
        except Exception as e:
            logger.error(f"Failed to store memory node {node.node_id}: {e}")
            return False
            
    async def retrieve_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory node by ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM memory_nodes WHERE node_id = $1", node_id)
                
                if row:
                    return self._row_to_node(row)
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory node {node_id}: {e}")
            return None
            
    async def search_nodes(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search memory nodes with vector similarity"""
        try:
            results = []
            
            async with self.pool.acquire() as conn:
                # Build query conditions
                conditions = []
                params = []
                param_count = 0
                
                if query.memory_types:
                    param_count += 1
                    conditions.append(f"memory_type = ANY(${param_count})")
                    params.append([mt.value for mt in query.memory_types])
                    
                if query.user_id:
                    param_count += 1
                    conditions.append(f"user_id = ${param_count}")
                    params.append(query.user_id)
                    
                if query.session_id:
                    param_count += 1
                    conditions.append(f"session_id = ${param_count}")
                    params.append(query.session_id)
                    
                if query.time_range:
                    param_count += 1
                    conditions.append(f"created_at >= ${param_count}")
                    params.append(query.time_range[0])
                    param_count += 1
                    conditions.append(f"created_at <= ${param_count}")
                    params.append(query.time_range[1])
                    
                param_count += 1
                conditions.append(f"importance >= ${param_count}")
                params.append(query.importance_threshold.value)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # Vector search if embedding available
                param_count += 1
                limit_clause = f"LIMIT ${param_count}"
                params.append(query.max_results)
                
                sql = f"""
                    SELECT *, 1 - (embedding <=> $1) as similarity_score
                    FROM memory_nodes 
                    WHERE {where_clause}
                    ORDER BY similarity_score DESC
                    {limit_clause}
                """
                
                # For now, use text search as fallback
                # In production, you'd generate embedding for query
                rows = await conn.fetch(sql.replace("$1", "'[0]'"), *params)
                
                for row in rows:
                    node = self._row_to_node(row)
                    similarity_score = float(row.get('similarity_score', 0.5))
                    
                    # Calculate relevance score
                    relevance_score = self._calculate_relevance(node, query, similarity_score)
                    
                    result = MemoryResult(
                        node=node,
                        similarity_score=similarity_score,
                        relevance_score=relevance_score
                    )
                    
                    # Get connected nodes if requested
                    if query.include_connections:
                        connections = await self.get_connections(node.node_id)
                        connected_nodes = []
                        for conn in connections[:5]:  # Limit connections
                            connected_node = await self.retrieve_node(conn.target_id)
                            if connected_node:
                                connected_nodes.append((connected_node, conn.strength))
                        result.connected_nodes = connected_nodes
                        
                    results.append(result)
                    
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to search memory nodes: {e}")
            return []
            
    async def store_connection(self, connection: MemoryConnection) -> bool:
        """Store memory connection"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO memory_connections 
                    (source_id, target_id, relationship_type, strength, created_at, 
                     last_reinforced, reinforcement_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (source_id, target_id, relationship_type) DO UPDATE SET
                        strength = EXCLUDED.strength,
                        last_reinforced = EXCLUDED.last_reinforced,
                        reinforcement_count = memory_connections.reinforcement_count + 1
                """, 
                connection.source_id, connection.target_id, connection.relationship_type,
                connection.strength, connection.created_at, connection.last_reinforced,
                connection.reinforcement_count)
                return True
        except Exception as e:
            logger.error(f"Failed to store memory connection: {e}")
            return False
            
    async def get_connections(self, node_id: str) -> List[MemoryConnection]:
        """Get all connections for a node"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM memory_connections 
                    WHERE source_id = $1 OR target_id = $1
                    ORDER BY strength DESC
                """, node_id)
                
                connections = []
                for row in rows:
                    connection = MemoryConnection(
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        relationship_type=row['relationship_type'],
                        strength=row['strength'],
                        created_at=row['created_at'],
                        last_reinforced=row['last_reinforced'],
                        reinforcement_count=row['reinforcement_count']
                    )
                    connections.append(connection)
                    
                return connections
        except Exception as e:
            logger.error(f"Failed to get connections for {node_id}: {e}")
            return []
            
    def _row_to_node(self, row) -> MemoryNode:
        """Convert database row to MemoryNode"""
        embedding = None
        if row['embedding']:
            embedding = np.array(row['embedding'])
            
        return MemoryNode(
            node_id=row['node_id'],
            content=row['content'],
            memory_type=MemoryType(row['memory_type']),
            importance=MemoryImportance(row['importance']),
            embedding=embedding,
            created_at=row['created_at'],
            last_accessed=row['last_accessed'],
            access_count=row['access_count'],
            confidence=row['confidence'],
            tags=set(row['tags']) if row['tags'] else set(),
            source=row['source'] or "",
            user_id=row['user_id'],
            session_id=row['session_id'],
            metadata=row['metadata'] or {}
        )
        
    def _calculate_relevance(self, node: MemoryNode, query: MemoryQuery, 
                           similarity_score: float) -> float:
        """Calculate relevance score for a memory node"""
        relevance = similarity_score * 0.4  # Base similarity
        
        # Importance boost
        relevance += (node.importance.value / 5.0) * 0.2
        
        # Recency boost
        age_days = (datetime.utcnow() - node.created_at).days
        recency_score = max(0, 1 - (age_days / 365))
        relevance += recency_score * 0.1
        
        # Access frequency boost
        access_score = min(1.0, node.access_count / 100)
        relevance += access_score * 0.1
        
        # Confidence boost
        relevance += node.confidence * 0.1
        
        # Session relevance
        if query.session_id and node.session_id == query.session_id:
            relevance += 0.1
            
        return min(1.0, relevance)


class SemanticMemoryNetwork:
    """
    Advanced semantic memory network with learning and consolidation
    """
    
    def __init__(self, memory_store: MemoryStore, embedding_model: str = "all-MiniLM-L6-v2"):
        self.memory_store = memory_store
        self.embedding_model = SentenceTransformer(embedding_model)
        self.graph = nx.DiGraph()
        self.consolidation_threshold = 0.8
        self.decay_factor = 0.95
        
    async def initialize(self):
        """Initialize the memory network"""
        if hasattr(self.memory_store, 'initialize'):
            await self.memory_store.initialize()
            
    async def store_memory(self, content: str, memory_type: MemoryType,
                          importance: MemoryImportance = MemoryImportance.MEDIUM,
                          user_id: Optional[str] = None, session_id: Optional[str] = None,
                          tags: Set[str] = None, source: str = "",
                          metadata: Dict[str, Any] = None) -> str:
        """Store a new memory in the network"""
        
        # Generate embedding
        embedding = self.embedding_model.encode(content)
        
        # Create memory node
        node = MemoryNode(
            node_id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            tags=tags or set(),
            source=source,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # Store in database
        success = await self.memory_store.store_node(node)
        
        if success:
            # Add to graph
            self.graph.add_node(node.node_id, node=node)
            
            # Find and create connections
            await self._create_connections(node)
            
            logger.info(f"Stored memory: {node.node_id}")
            return node.node_id
        else:
            raise Exception("Failed to store memory node")
            
    async def retrieve_memories(self, query: str, memory_types: List[MemoryType] = None,
                              user_id: Optional[str] = None, session_id: Optional[str] = None,
                              max_results: int = 10) -> List[MemoryResult]:
        """Retrieve memories based on query"""
        
        memory_query = MemoryQuery(
            query=query,
            memory_types=memory_types or list(MemoryType),
            user_id=user_id,
            session_id=session_id,
            max_results=max_results
        )
        
        results = await self.memory_store.search_nodes(memory_query)
        
        # Update access counts
        for result in results:
            result.node.last_accessed = datetime.utcnow()
            result.node.access_count += 1
            await self.memory_store.store_node(result.node)
            
        return results
        
    async def consolidate_memories(self, user_id: Optional[str] = None):
        """Consolidate related memories and strengthen connections"""
        
        # Get recent memories for consolidation
        query = MemoryQuery(
            query="",  # Empty query to get all
            user_id=user_id,
            time_range=(datetime.utcnow() - timedelta(days=7), datetime.utcnow()),
            max_results=100
        )
        
        recent_memories = await self.memory_store.search_nodes(query)
        
        # Find similar memories for consolidation
        for i, memory1 in enumerate(recent_memories):
            for memory2 in recent_memories[i+1:]:
                similarity = self._calculate_similarity(memory1.node, memory2.node)
                
                if similarity > self.consolidation_threshold:
                    # Create or strengthen connection
                    connection = MemoryConnection(
                        source_id=memory1.node.node_id,
                        target_id=memory2.node.node_id,
                        relationship_type="similar",
                        strength=similarity
                    )
                    
                    await self.memory_store.store_connection(connection)
                    
        logger.info(f"Consolidated memories for user: {user_id}")
        
    async def decay_memories(self, user_id: Optional[str] = None):
        """Apply memory decay to reduce importance of old memories"""
        
        # Get old memories
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        query = MemoryQuery(
            query="",
            user_id=user_id,
            time_range=(datetime.min, cutoff_date),
            max_results=1000
        )
        
        old_memories = await self.memory_store.search_nodes(query)
        
        for memory_result in old_memories:
            node = memory_result.node
            
            # Apply decay based on access frequency
            if node.access_count < 5:  # Rarely accessed memories decay more
                node.confidence *= self.decay_factor
                node.importance = MemoryImportance(max(1, node.importance.value - 1))
                
                await self.memory_store.store_node(node)
                
        logger.info(f"Applied memory decay for user: {user_id}")
        
    async def get_memory_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user's memory patterns"""
        
        query = MemoryQuery(query="", user_id=user_id, max_results=1000)
        memories = await self.memory_store.search_nodes(query)
        
        if not memories:
            return {"total_memories": 0}
            
        # Calculate insights
        memory_by_type = defaultdict(int)
        memory_by_importance = defaultdict(int)
        recent_memories = 0
        total_confidence = 0
        
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        for memory_result in memories:
            node = memory_result.node
            memory_by_type[node.memory_type.value] += 1
            memory_by_importance[node.importance.value] += 1
            total_confidence += node.confidence
            
            if node.created_at > cutoff_date:
                recent_memories += 1
                
        avg_confidence = total_confidence / len(memories)
        
        return {
            "total_memories": len(memories),
            "recent_memories": recent_memories,
            "average_confidence": avg_confidence,
            "memory_by_type": dict(memory_by_type),
            "memory_by_importance": dict(memory_by_importance),
            "most_common_tags": self._get_common_tags(memories)
        }
        
    def _calculate_similarity(self, node1: MemoryNode, node2: MemoryNode) -> float:
        """Calculate similarity between two memory nodes"""
        if node1.embedding is None or node2.embedding is None:
            return 0.0
            
        # Cosine similarity
        cos_sim = np.dot(node1.embedding, node2.embedding) / (
            np.linalg.norm(node1.embedding) * np.linalg.norm(node2.embedding)
        )
        
        # Tag similarity
        tag_similarity = len(node1.tags & node2.tags) / max(1, len(node1.tags | node2.tags))
        
        # Combine similarities
        return (cos_sim * 0.8) + (tag_similarity * 0.2)
        
    async def _create_connections(self, node: MemoryNode):
        """Create connections with similar existing memories"""
        
        # Find similar memories
        query = MemoryQuery(
            query=node.content,
            memory_types=[node.memory_type],
            user_id=node.user_id,
            max_results=10
        )
        
        similar_memories = await self.memory_store.search_nodes(query)
        
        for memory_result in similar_memories:
            if memory_result.node.node_id != node.node_id:
                similarity = self._calculate_similarity(node, memory_result.node)
                
                if similarity > 0.7:  # High similarity threshold
                    connection = MemoryConnection(
                        source_id=node.node_id,
                        target_id=memory_result.node.node_id,
                        relationship_type="similar",
                        strength=similarity
                    )
                    
                    await self.memory_store.store_connection(connection)
                    
    def _get_common_tags(self, memories: List[MemoryResult], top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most common tags from memories"""
        tag_counts = defaultdict(int)
        
        for memory_result in memories:
            for tag in memory_result.node.tags:
                tag_counts[tag] += 1
                
        return sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        

class MetaCognitiveMemory:
    """
    Meta-cognitive memory for system self-improvement and learning
    """
    
    def __init__(self, memory_network: SemanticMemoryNetwork):
        self.memory_network = memory_network
        self.learning_patterns: Dict[str, Any] = {}
        
    async def record_interaction_outcome(self, interaction_id: str, user_id: str,
                                        therapy_technique: str, user_response: str,
                                        effectiveness_score: float):
        """Record the outcome of a therapeutic interaction"""
        
        memory_content = f"Interaction {interaction_id}: Used {therapy_technique}. " \
                        f"User response: {user_response}. Effectiveness: {effectiveness_score}"
                        
        await self.memory_network.store_memory(
            content=memory_content,
            memory_type=MemoryType.METACOGNITIVE,
            importance=MemoryImportance.HIGH if effectiveness_score > 0.7 else MemoryImportance.MEDIUM,
            user_id=user_id,
            tags={"interaction_outcome", therapy_technique},
            metadata={
                "interaction_id": interaction_id,
                "therapy_technique": therapy_technique,
                "effectiveness_score": effectiveness_score,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    async def learn_from_patterns(self, user_id: str) -> Dict[str, Any]:
        """Learn patterns from user interactions to improve therapy"""
        
        # Retrieve meta-cognitive memories
        memories = await self.memory_network.retrieve_memories(
            query="interaction outcome effectiveness",
            memory_types=[MemoryType.METACOGNITIVE],
            user_id=user_id,
            max_results=100
        )
        
        if not memories:
            return {"learning": "Insufficient data for pattern learning"}
            
        # Analyze patterns
        technique_effectiveness = defaultdict(list)
        
        for memory_result in memories:
            metadata = memory_result.node.metadata
            if 'therapy_technique' in metadata and 'effectiveness_score' in metadata:
                technique = metadata['therapy_technique']
                score = metadata['effectiveness_score']
                technique_effectiveness[technique].append(score)
                
        # Calculate average effectiveness for each technique
        technique_recommendations = {}
        for technique, scores in technique_effectiveness.items():
            avg_score = sum(scores) / len(scores)
            technique_recommendations[technique] = {
                "average_effectiveness": avg_score,
                "sample_size": len(scores),
                "recommendation": "high" if avg_score > 0.7 else "medium" if avg_score > 0.5 else "low"
            }
            
        # Store learning as new memory
        learning_content = f"Learned therapy technique effectiveness for user: " \
                          f"{json.dumps(technique_recommendations)}"
                          
        await self.memory_network.store_memory(
            content=learning_content,
            memory_type=MemoryType.METACOGNITIVE,
            importance=MemoryImportance.HIGH,
            user_id=user_id,
            tags={"pattern_learning", "therapy_effectiveness"},
            metadata={"technique_recommendations": technique_recommendations}
        )
        
        return {
            "learning": "Pattern analysis completed",
            "technique_recommendations": technique_recommendations,
            "total_interactions_analyzed": len(memories)
        }