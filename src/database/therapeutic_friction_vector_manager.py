"""
Enhanced Vector Database Manager for Therapeutic Friction Sub-Agents.

This module provides specialized vector database operations for therapeutic friction
sub-agents, including domain-specific knowledge storage, retrieval, and cross-domain
semantic search capabilities.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import json
import numpy as np
from dataclasses import dataclass, asdict

from src.database.vector_store import VectorStore
from src.utils.logger import get_logger


class TherapeuticFrictionDocumentType(Enum):
    """Document types for therapeutic friction knowledge domains."""
    READINESS_PATTERN = "readiness_pattern"
    BREAKTHROUGH_INDICATOR = "breakthrough_indicator"
    RELATIONSHIP_DYNAMIC = "relationship_dynamic"
    INTERVENTION_OUTCOME = "intervention_outcome"
    PROGRESS_TRAJECTORY = "progress_trajectory"
    THERAPEUTIC_TECHNIQUE = "therapeutic_technique"
    CLINICAL_EVIDENCE = "clinical_evidence"
    CASE_STUDY = "case_study"


@dataclass
class TherapeuticDocument:
    """Structured document for therapeutic friction knowledge."""
    document_id: str
    document_type: TherapeuticFrictionDocumentType
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class TherapeuticFrictionVectorManager:
    """
    Enhanced vector database manager for therapeutic friction sub-agents.
    
    Provides domain-specific storage, retrieval, and cross-domain semantic search
    capabilities for therapeutic friction knowledge.
    """
    
    def __init__(self, model_provider=None, config: Dict[str, Any] = None):
        """Initialize the therapeutic friction vector manager."""
        self.config = config or {}
        self.model_provider = model_provider
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize vector stores for each domain
        self.vector_stores: Dict[TherapeuticFrictionDocumentType, VectorStore] = {}
        self.collection_configs = {
            TherapeuticFrictionDocumentType.READINESS_PATTERN: {
                "dimensions": 768,
                "description": "User readiness patterns and indicators",
                "index_params": {"metric": "cosine", "index_type": "IVF_FLAT"}
            },
            TherapeuticFrictionDocumentType.BREAKTHROUGH_INDICATOR: {
                "dimensions": 768,
                "description": "Breakthrough moments and insight patterns",
                "index_params": {"metric": "cosine", "index_type": "IVF_FLAT"}
            },
            TherapeuticFrictionDocumentType.RELATIONSHIP_DYNAMIC: {
                "dimensions": 768,
                "description": "Therapeutic relationship dynamics and alliance patterns",
                "index_params": {"metric": "cosine", "index_type": "IVF_FLAT"}
            },
            TherapeuticFrictionDocumentType.INTERVENTION_OUTCOME: {
                "dimensions": 768,
                "description": "Intervention effectiveness and outcome data",
                "index_params": {"metric": "cosine", "index_type": "IVF_FLAT"}
            },
            TherapeuticFrictionDocumentType.PROGRESS_TRAJECTORY: {
                "dimensions": 768,
                "description": "Therapeutic progress patterns and trajectories",
                "index_params": {"metric": "cosine", "index_type": "IVF_FLAT"}
            }
        }
        
        # Document storage and indexing
        self.documents: Dict[str, TherapeuticDocument] = {}
        self.document_index: Dict[TherapeuticFrictionDocumentType, List[str]] = {
            doc_type: [] for doc_type in TherapeuticFrictionDocumentType
        }
        
        # Cross-domain relationship tracking
        self.domain_relationships: Dict[str, List[str]] = {}
        
        # Performance metrics
        self.retrieval_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "average_retrieval_time": 0.0
        }
        
        # Initialize vector stores
        self._initialize_vector_stores()
    
    def _initialize_vector_stores(self):
        """Initialize vector stores for each therapeutic domain."""
        try:
            import os
            base_path = self.config.get("storage_path", "src/data/vector_store/therapeutic_friction")
            
            for doc_type, config in self.collection_configs.items():
                collection_path = os.path.join(base_path, doc_type.value)
                os.makedirs(collection_path, exist_ok=True)
                
                self.vector_stores[doc_type] = VectorStore(
                    collection_name=f"therapeutic_friction_{doc_type.value}",
                    vector_dimensions=config["dimensions"],
                    storage_path=collection_path
                )
                
                self.logger.debug(f"Initialized vector store for {doc_type.value}")
            
            self.logger.info(f"Initialized {len(self.vector_stores)} therapeutic friction vector stores")
            
        except Exception as e:
            self.logger.error(f"Error initializing vector stores: {str(e)}")
            raise
    
    async def add_document(self, document: TherapeuticDocument) -> bool:
        """Add a therapeutic document to the appropriate vector store."""
        try:
            # Generate embedding if not provided
            if document.embedding is None:
                document.embedding = await self._get_embedding(document.content)
            
            # Get the appropriate vector store
            vector_store = self.vector_stores.get(document.document_type)
            if not vector_store:
                self.logger.error(f"No vector store found for document type: {document.document_type}")
                return False
            
            # Prepare metadata for vector storage
            vector_metadata = {
                "document_id": document.document_id,
                "title": document.title,
                "document_type": document.document_type.value,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                **document.metadata
            }
            
            # Add to vector store
            vector_store.add_item(
                vector=document.embedding,
                metadata=vector_metadata
            )
            
            # Store document locally
            self.documents[document.document_id] = document
            self.document_index[document.document_type].append(document.document_id)
            
            # Update cross-domain relationships if specified
            await self._update_cross_domain_relationships(document)
            
            self.logger.debug(f"Added document {document.document_id} to {document.document_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document {document.document_id}: {str(e)}")
            return False
    
    async def search_documents(self, query: str, document_type: TherapeuticFrictionDocumentType,
                             top_k: int = 5, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for documents within a specific therapeutic domain."""
        try:
            start_time = datetime.now()
            
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Get appropriate vector store
            vector_store = self.vector_stores.get(document_type)
            if not vector_store:
                self.logger.error(f"No vector store found for document type: {document_type}")
                return []
            
            # Perform vector search
            search_results = vector_store.search(
                query_vector=query_embedding,
                top_k=top_k * 2  # Get more results for filtering
            )
            
            # Filter and enrich results
            filtered_results = []
            for result in search_results:
                # Apply filter criteria if provided
                if filter_criteria and not self._matches_filter_criteria(result, filter_criteria):
                    continue
                
                # Get full document information
                document_id = result.get("metadata", {}).get("document_id")
                if document_id and document_id in self.documents:
                    full_document = self.documents[document_id]
                    
                    enriched_result = {
                        "document": asdict(full_document),
                        "similarity_score": result.get("score", 0.0),
                        "match_reason": self._generate_match_reason(query, full_document),
                        "relevance_metadata": self._calculate_relevance_metadata(query, full_document)
                    }
                    filtered_results.append(enriched_result)
                
                if len(filtered_results) >= top_k:
                    break
            
            # Update performance metrics
            retrieval_time = (datetime.now() - start_time).total_seconds()
            self._update_retrieval_metrics(retrieval_time)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error searching documents for {document_type.value}: {str(e)}")
            return []
    
    async def cross_domain_search(self, query: str, primary_domain: TherapeuticFrictionDocumentType,
                                secondary_domains: List[TherapeuticFrictionDocumentType] = None,
                                top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Perform cross-domain semantic search across multiple therapeutic domains."""
        try:
            if secondary_domains is None:
                secondary_domains = [dt for dt in TherapeuticFrictionDocumentType if dt != primary_domain]
            
            # Perform concurrent searches across domains
            search_tasks = []
            
            # Primary domain search
            primary_task = asyncio.create_task(
                self.search_documents(query, primary_domain, top_k),
                name=f"primary_{primary_domain.value}"
            )
            search_tasks.append((primary_domain, primary_task))
            
            # Secondary domain searches
            for domain in secondary_domains:
                task = asyncio.create_task(
                    self.search_documents(query, domain, max(1, top_k // 2)),
                    name=f"secondary_{domain.value}"
                )
                search_tasks.append((domain, task))
            
            # Wait for all searches to complete
            results = {}
            for domain, task in search_tasks:
                try:
                    domain_results = await task
                    results[domain.value] = domain_results
                except Exception as e:
                    self.logger.error(f"Error in cross-domain search for {domain.value}: {str(e)}")
                    results[domain.value] = []
            
            # Analyze cross-domain relationships
            cross_domain_insights = await self._analyze_cross_domain_relationships(query, results)
            results["cross_domain_insights"] = cross_domain_insights
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in cross-domain search: {str(e)}")
            return {}
    
    async def get_domain_statistics(self, document_type: TherapeuticFrictionDocumentType = None) -> Dict[str, Any]:
        """Get statistics for a specific domain or all domains."""
        try:
            if document_type:
                # Statistics for specific domain
                vector_store = self.vector_stores.get(document_type)
                document_count = len(self.document_index.get(document_type, []))
                
                return {
                    "document_type": document_type.value,
                    "total_documents": document_count,
                    "vector_store_size": vector_store.count() if vector_store else 0,
                    "last_updated": max(
                        [self.documents[doc_id].updated_at for doc_id in self.document_index.get(document_type, [])],
                        default=datetime.min
                    ).isoformat() if document_count > 0 else None
                }
            else:
                # Statistics for all domains
                all_stats = {}
                total_documents = 0
                
                for doc_type in TherapeuticFrictionDocumentType:
                    stats = await self.get_domain_statistics(doc_type)
                    all_stats[doc_type.value] = stats
                    total_documents += stats["total_documents"]
                
                all_stats["summary"] = {
                    "total_documents": total_documents,
                    "total_domains": len(TherapeuticFrictionDocumentType),
                    "retrieval_metrics": self.retrieval_metrics
                }
                
                return all_stats
                
        except Exception as e:
            self.logger.error(f"Error getting domain statistics: {str(e)}")
            return {}
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing therapeutic document."""
        try:
            if document_id not in self.documents:
                self.logger.error(f"Document {document_id} not found")
                return False
            
            document = self.documents[document_id]
            
            # Update document fields
            if "content" in updates:
                document.content = updates["content"]
                # Regenerate embedding for content changes
                document.embedding = await self._get_embedding(document.content)
            
            if "title" in updates:
                document.title = updates["title"]
            
            if "metadata" in updates:
                document.metadata.update(updates["metadata"])
            
            document.updated_at = datetime.now()
            
            # Update in vector store
            vector_store = self.vector_stores.get(document.document_type)
            if vector_store and document.embedding:
                # Remove old entry and add updated one
                # Note: This is a simplified approach; real implementation might need more sophisticated updating
                vector_metadata = {
                    "document_id": document.document_id,
                    "title": document.title,
                    "document_type": document.document_type.value,
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat(),
                    **document.metadata
                }
                
                vector_store.add_item(
                    vector=document.embedding,
                    metadata=vector_metadata
                )
            
            self.logger.debug(f"Updated document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document {document_id}: {str(e)}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a therapeutic document."""
        try:
            if document_id not in self.documents:
                self.logger.error(f"Document {document_id} not found")
                return False
            
            document = self.documents[document_id]
            
            # Remove from document index
            if document_id in self.document_index[document.document_type]:
                self.document_index[document.document_type].remove(document_id)
            
            # Remove from local storage
            del self.documents[document_id]
            
            # Remove cross-domain relationships
            if document_id in self.domain_relationships:
                del self.domain_relationships[document_id]
            
            self.logger.debug(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    async def batch_add_documents(self, documents: List[TherapeuticDocument]) -> Dict[str, Any]:
        """Add multiple documents in batch for efficiency."""
        try:
            start_time = datetime.now()
            successful_adds = 0
            failed_adds = []
            
            # Process documents in batches
            batch_size = self.config.get("batch_size", 10)
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [self.add_document(doc) for doc in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_adds.append({
                            "document_id": batch[j].document_id,
                            "error": str(result)
                        })
                    elif result:
                        successful_adds += 1
                    else:
                        failed_adds.append({
                            "document_id": batch[j].document_id,
                            "error": "Unknown error"
                        })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "total_documents": len(documents),
                "successful_adds": successful_adds,
                "failed_adds": len(failed_adds),
                "failed_documents": failed_adds,
                "processing_time": processing_time,
                "documents_per_second": len(documents) / processing_time if processing_time > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch document addition: {str(e)}")
            return {"error": str(e)}
    
    async def get_recommendations_for_agent(self, agent_type: str, user_context: Dict[str, Any],
                                          top_k: int = 3) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a specific therapeutic friction sub-agent."""
        try:
            # Map agent types to document types
            agent_domain_mapping = {
                "readiness_assessment": TherapeuticFrictionDocumentType.READINESS_PATTERN,
                "breakthrough_detection": TherapeuticFrictionDocumentType.BREAKTHROUGH_INDICATOR,
                "relationship_monitoring": TherapeuticFrictionDocumentType.RELATIONSHIP_DYNAMIC,
                "intervention_strategy": TherapeuticFrictionDocumentType.INTERVENTION_OUTCOME,
                "progress_tracking": TherapeuticFrictionDocumentType.PROGRESS_TRAJECTORY
            }
            
            primary_domain = agent_domain_mapping.get(agent_type)
            if not primary_domain:
                self.logger.error(f"Unknown agent type: {agent_type}")
                return []
            
            # Generate search query from user context
            search_query = self._generate_contextual_query(user_context, agent_type)
            
            # Get domain-specific recommendations
            primary_results = await self.search_documents(search_query, primary_domain, top_k)
            
            # Get cross-domain insights
            secondary_domains = [dt for dt in TherapeuticFrictionDocumentType if dt != primary_domain]
            cross_domain_results = await self.cross_domain_search(
                search_query, primary_domain, secondary_domains[:2], top_k=2
            )
            
            # Combine and rank recommendations
            recommendations = self._combine_and_rank_recommendations(
                primary_results, cross_domain_results, user_context, agent_type
            )
            
            return recommendations[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations for {agent_type}: {str(e)}")
            return []
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using model provider or fallback."""
        try:
            if self.model_provider and hasattr(self.model_provider, "get_embedding"):
                return await self.model_provider.get_embedding(text)
            else:
                # Fallback to random embeddings for testing
                import numpy as np
                return list(np.random.rand(768).astype(float))
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            return list(np.random.rand(768).astype(float))
    
    def _matches_filter_criteria(self, result: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if a search result matches the filter criteria."""
        metadata = result.get("metadata", {})
        
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def _generate_match_reason(self, query: str, document: TherapeuticDocument) -> str:
        """Generate explanation for why a document matches the query."""
        # Simplified match reason generation
        query_words = set(query.lower().split())
        content_words = set(document.content.lower().split())
        title_words = set(document.title.lower().split())
        
        content_matches = len(query_words.intersection(content_words))
        title_matches = len(query_words.intersection(title_words))
        
        if title_matches > 0:
            return f"Strong match in title ({title_matches} keywords)"
        elif content_matches > 2:
            return f"Good content match ({content_matches} keywords)"
        else:
            return "Semantic similarity match"
    
    def _calculate_relevance_metadata(self, query: str, document: TherapeuticDocument) -> Dict[str, Any]:
        """Calculate relevance metadata for a document match."""
        return {
            "document_age_days": (datetime.now() - document.created_at).days,
            "last_updated_days": (datetime.now() - document.updated_at).days,
            "document_type": document.document_type.value,
            "metadata_richness": len(document.metadata),
            "content_length": len(document.content)
        }
    
    async def _update_cross_domain_relationships(self, document: TherapeuticDocument):
        """Update cross-domain relationships for a document."""
        try:
            # Find related documents in other domains
            related_documents = []
            
            for doc_type, doc_ids in self.document_index.items():
                if doc_type == document.document_type:
                    continue
                
                # Simple relationship detection (can be enhanced with ML)
                for doc_id in doc_ids[:10]:  # Limit to prevent performance issues
                    other_doc = self.documents.get(doc_id)
                    if other_doc and self._documents_are_related(document, other_doc):
                        related_documents.append(doc_id)
            
            if related_documents:
                self.domain_relationships[document.document_id] = related_documents
                
        except Exception as e:
            self.logger.error(f"Error updating cross-domain relationships: {str(e)}")
    
    def _documents_are_related(self, doc1: TherapeuticDocument, doc2: TherapeuticDocument) -> bool:
        """Simple heuristic to determine if two documents are related."""
        # Check for common keywords in metadata
        common_keywords = set(doc1.metadata.get("keywords", [])).intersection(
            set(doc2.metadata.get("keywords", []))
        )
        
        if len(common_keywords) > 0:
            return True
        
        # Check for common therapeutic modalities
        common_modalities = set(doc1.metadata.get("therapeutic_modalities", [])).intersection(
            set(doc2.metadata.get("therapeutic_modalities", []))
        )
        
        return len(common_modalities) > 0
    
    async def _analyze_cross_domain_relationships(self, query: str, 
                                                results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze relationships across domain search results."""
        try:
            # Extract key themes across domains
            all_documents = []
            for domain_results in results.values():
                if isinstance(domain_results, list):
                    all_documents.extend(domain_results)
            
            if not all_documents:
                return {"themes": [], "relationships": [], "insights": []}
            
            # Analyze common themes
            common_themes = self._extract_common_themes(all_documents)
            
            # Identify cross-domain patterns
            cross_patterns = self._identify_cross_domain_patterns(results)
            
            # Generate insights
            insights = self._generate_cross_domain_insights(common_themes, cross_patterns)
            
            return {
                "themes": common_themes,
                "cross_patterns": cross_patterns,
                "insights": insights,
                "total_documents_analyzed": len(all_documents)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-domain relationships: {str(e)}")
            return {"error": str(e)}
    
    def _extract_common_themes(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from cross-domain documents."""
        # Simplified theme extraction
        theme_counts = {}
        
        for doc_result in documents:
            doc = doc_result.get("document", {})
            metadata = doc.get("metadata", {})
            
            # Count keywords
            for keyword in metadata.get("keywords", []):
                theme_counts[keyword] = theme_counts.get(keyword, 0) + 1
            
            # Count therapeutic modalities
            for modality in metadata.get("therapeutic_modalities", []):
                theme_counts[modality] = theme_counts.get(modality, 0) + 1
        
        # Return top themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5] if count > 1]
    
    def _identify_cross_domain_patterns(self, results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Identify patterns across domain results."""
        patterns = []
        
        # Check for consistent high-confidence matches across domains
        high_confidence_domains = []
        for domain, domain_results in results.items():
            if isinstance(domain_results, list) and domain_results:
                avg_confidence = sum(r.get("similarity_score", 0) for r in domain_results) / len(domain_results)
                if avg_confidence > 0.7:
                    high_confidence_domains.append(domain)
        
        if len(high_confidence_domains) > 1:
            patterns.append(f"Strong cross-domain alignment across {', '.join(high_confidence_domains)}")
        
        return patterns
    
    def _generate_cross_domain_insights(self, themes: List[str], patterns: List[str]) -> List[str]:
        """Generate insights from cross-domain analysis."""
        insights = []
        
        if themes:
            insights.append(f"Key therapeutic themes: {', '.join(themes[:3])}")
        
        if patterns:
            insights.extend(patterns)
        
        if not themes and not patterns:
            insights.append("Limited cross-domain relationships detected")
        
        return insights
    
    def _generate_contextual_query(self, user_context: Dict[str, Any], agent_type: str) -> str:
        """Generate contextual search query from user context."""
        query_parts = []
        
        # Add emotional context
        emotion = user_context.get("emotion_analysis", {}).get("primary_emotion")
        if emotion:
            query_parts.append(f"{emotion} emotion")
        
        # Add readiness context
        readiness = user_context.get("user_readiness")
        if readiness:
            query_parts.append(f"{readiness} readiness")
        
        # Add agent-specific context
        agent_contexts = {
            "readiness_assessment": "therapeutic readiness motivation",
            "breakthrough_detection": "insight breakthrough transformation",
            "relationship_monitoring": "therapeutic alliance trust",
            "intervention_strategy": "therapeutic intervention technique",
            "progress_tracking": "therapeutic progress outcome"
        }
        
        if agent_type in agent_contexts:
            query_parts.append(agent_contexts[agent_type])
        
        return " ".join(query_parts) if query_parts else "therapeutic guidance"
    
    def _combine_and_rank_recommendations(self, primary_results: List[Dict[str, Any]],
                                        cross_domain_results: Dict[str, List[Dict[str, Any]]],
                                        user_context: Dict[str, Any], agent_type: str) -> List[Dict[str, Any]]:
        """Combine and rank recommendations from multiple sources."""
        all_recommendations = []
        
        # Add primary results with highest weight
        for result in primary_results:
            result["recommendation_score"] = result.get("similarity_score", 0.0) * 1.0
            result["source"] = "primary_domain"
            all_recommendations.append(result)
        
        # Add cross-domain results with lower weight
        for domain, domain_results in cross_domain_results.items():
            if isinstance(domain_results, list):
                for result in domain_results[:2]:  # Limit cross-domain results
                    result["recommendation_score"] = result.get("similarity_score", 0.0) * 0.7
                    result["source"] = f"cross_domain_{domain}"
                    all_recommendations.append(result)
        
        # Sort by recommendation score
        all_recommendations.sort(key=lambda x: x.get("recommendation_score", 0.0), reverse=True)
        
        return all_recommendations
    
    def _update_retrieval_metrics(self, retrieval_time: float):
        """Update performance metrics for retrieval operations."""
        self.retrieval_metrics["total_queries"] += 1
        
        # Update average retrieval time
        total_time = (self.retrieval_metrics["average_retrieval_time"] * 
                     (self.retrieval_metrics["total_queries"] - 1) + retrieval_time)
        self.retrieval_metrics["average_retrieval_time"] = total_time / self.retrieval_metrics["total_queries"]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the vector manager."""
        try:
            status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "vector_stores": {},
                "document_counts": {},
                "performance_metrics": self.retrieval_metrics
            }
            
            # Check each vector store
            for doc_type, vector_store in self.vector_stores.items():
                try:
                    store_count = vector_store.count()
                    status["vector_stores"][doc_type.value] = {
                        "status": "healthy",
                        "document_count": store_count
                    }
                    status["document_counts"][doc_type.value] = len(self.document_index.get(doc_type, []))
                except Exception as e:
                    status["vector_stores"][doc_type.value] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # Overall health assessment
            unhealthy_stores = [s for s in status["vector_stores"].values() if s["status"] != "healthy"]
            if unhealthy_stores:
                status["status"] = "degraded"
                status["issues"] = f"{len(unhealthy_stores)} vector stores have issues"
            
            return status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }