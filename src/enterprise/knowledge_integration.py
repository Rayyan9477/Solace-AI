"""
Real-Time Research and Knowledge Update Integration System for Solace-AI

This module provides comprehensive knowledge management capabilities including:
- Real-time research paper monitoring and integration
- Automated knowledge base updates
- Expert knowledge validation and curation
- Clinical guideline synchronization
- Evidence-based recommendation updates
- Knowledge graph maintenance
- Research trend analysis
- Peer-reviewed content integration
- Continuous learning pipeline
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
import threading
from abc import ABC, abstractmethod
import aiohttp
import xml.etree.ElementTree as ET
import feedparser
from urllib.parse import urljoin, urlparse
import re

from src.utils.logger import get_logger
from src.integration.event_bus import EventBus, Event, EventType, EventPriority
from src.utils.vector_db_integration import get_conversation_tracker

logger = get_logger(__name__)


class KnowledgeSource(Enum):
    """Types of knowledge sources."""
    RESEARCH_PAPER = "research_paper"
    CLINICAL_GUIDELINE = "clinical_guideline"
    EXPERT_OPINION = "expert_opinion"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    CASE_STUDY = "case_study"
    CLINICAL_TRIAL = "clinical_trial"
    BEST_PRACTICE = "best_practice"
    REGULATORY_UPDATE = "regulatory_update"


class ContentStatus(Enum):
    """Status of knowledge content."""
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    VALIDATED = "validated"
    INTEGRATED = "integrated"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


class ValidationLevel(Enum):
    """Validation levels for knowledge content."""
    AUTOMATIC = "automatic"
    PEER_REVIEW = "peer_review"
    EXPERT_VALIDATION = "expert_validation"
    CLINICAL_COMMITTEE = "clinical_committee"


@dataclass
class KnowledgeItem:
    """Knowledge item structure."""
    
    item_id: str
    title: str
    source_type: KnowledgeSource
    content: str
    abstract: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    source_url: Optional[str] = None
    doi: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    status: ContentStatus = ContentStatus.PENDING
    validation_level: ValidationLevel = ValidationLevel.AUTOMATIC
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'item_id': self.item_id,
            'title': self.title,
            'source_type': self.source_type.value,
            'content': self.content,
            'abstract': self.abstract,
            'authors': self.authors,
            'publication_date': self.publication_date.isoformat() if self.publication_date else None,
            'source_url': self.source_url,
            'doi': self.doi,
            'keywords': self.keywords,
            'categories': self.categories,
            'confidence_score': self.confidence_score,
            'relevance_score': self.relevance_score,
            'status': self.status.value,
            'validation_level': self.validation_level.value,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class ValidationResult:
    """Result of knowledge validation."""
    
    item_id: str
    is_valid: bool
    confidence: float
    validation_notes: str
    validator_id: str
    validation_criteria: Dict[str, bool]
    recommendations: List[str] = field(default_factory=list)
    requires_review: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'item_id': self.item_id,
            'is_valid': self.is_valid,
            'confidence': self.confidence,
            'validation_notes': self.validation_notes,
            'validator_id': self.validator_id,
            'validation_criteria': self.validation_criteria,
            'recommendations': self.recommendations,
            'requires_review': self.requires_review,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass 
class ResearchTrend:
    """Research trend analysis result."""
    
    topic: str
    trend_direction: str  # 'increasing', 'stable', 'declining'
    confidence: float
    supporting_papers: List[str]
    key_findings: List[str]
    implications: List[str]
    time_period: str
    analysis_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'topic': self.topic,
            'trend_direction': self.trend_direction,
            'confidence': self.confidence,
            'supporting_papers': self.supporting_papers,
            'key_findings': self.key_findings,
            'implications': self.implications,
            'time_period': self.time_period,
            'analysis_date': self.analysis_date.isoformat()
        }


class KnowledgeValidator(ABC):
    """Abstract base class for knowledge validators."""
    
    def __init__(self, validator_id: str):
        self.validator_id = validator_id
    
    @abstractmethod
    async def validate(self, item: KnowledgeItem) -> ValidationResult:
        """Validate a knowledge item."""
        pass


class AutomaticContentValidator(KnowledgeValidator):
    """Automatic content validation using heuristics and NLP."""
    
    def __init__(self):
        super().__init__("automatic_validator")
        self.quality_keywords = [
            'peer-reviewed', 'randomized', 'controlled', 'systematic',
            'meta-analysis', 'evidence-based', 'clinical trial'
        ]
        self.warning_keywords = [
            'preliminary', 'unverified', 'anecdotal', 'opinion',
            'case report', 'small sample', 'limited evidence'
        ]
    
    async def validate(self, item: KnowledgeItem) -> ValidationResult:
        """Perform automatic validation."""
        
        validation_criteria = {
            'has_abstract': item.abstract is not None and len(item.abstract) > 50,
            'has_authors': len(item.authors) > 0,
            'has_publication_date': item.publication_date is not None,
            'recent_publication': self._is_recent_publication(item),
            'quality_indicators': self._check_quality_indicators(item),
            'no_warning_flags': not self._has_warning_flags(item),
            'appropriate_length': self._check_content_length(item)
        }
        
        # Calculate confidence based on criteria
        passed_criteria = sum(1 for passed in validation_criteria.values() if passed)
        confidence = passed_criteria / len(validation_criteria)
        
        # Determine if valid
        is_valid = confidence >= 0.6 and validation_criteria['no_warning_flags']
        
        # Generate validation notes
        notes = self._generate_validation_notes(validation_criteria, item)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_criteria, item)
        
        # Determine if manual review is needed
        requires_review = (
            confidence < 0.8 or 
            not validation_criteria['quality_indicators'] or
            item.source_type in [KnowledgeSource.EXPERT_OPINION, KnowledgeSource.CASE_STUDY]
        )
        
        return ValidationResult(
            item_id=item.item_id,
            is_valid=is_valid,
            confidence=confidence,
            validation_notes=notes,
            validator_id=self.validator_id,
            validation_criteria=validation_criteria,
            recommendations=recommendations,
            requires_review=requires_review
        )
    
    def _is_recent_publication(self, item: KnowledgeItem) -> bool:
        """Check if publication is recent (within 5 years)."""
        if not item.publication_date:
            return False
        
        cutoff_date = datetime.now() - timedelta(days=5*365)
        return item.publication_date >= cutoff_date
    
    def _check_quality_indicators(self, item: KnowledgeItem) -> bool:
        """Check for quality indicators in content."""
        text = (item.title + ' ' + (item.abstract or '') + ' ' + item.content).lower()
        
        quality_count = sum(1 for keyword in self.quality_keywords if keyword in text)
        return quality_count >= 2
    
    def _has_warning_flags(self, item: KnowledgeItem) -> bool:
        """Check for warning flags in content."""
        text = (item.title + ' ' + (item.abstract or '') + ' ' + item.content).lower()
        
        return any(keyword in text for keyword in self.warning_keywords)
    
    def _check_content_length(self, item: KnowledgeItem) -> bool:
        """Check if content has appropriate length."""
        total_length = len(item.content) + len(item.abstract or '')
        return 100 <= total_length <= 50000  # Reasonable bounds
    
    def _generate_validation_notes(self, criteria: Dict[str, bool], item: KnowledgeItem) -> str:
        """Generate human-readable validation notes."""
        notes = []
        
        if not criteria['has_abstract']:
            notes.append("Missing or insufficient abstract")
        if not criteria['has_authors']:
            notes.append("No authors specified")
        if not criteria['has_publication_date']:
            notes.append("Missing publication date")
        if not criteria['recent_publication']:
            notes.append("Publication is older than 5 years")
        if not criteria['quality_indicators']:
            notes.append("Lacks quality indicators (peer-review, controlled study, etc.)")
        if not criteria['no_warning_flags']:
            notes.append("Contains warning flags (preliminary, unverified, etc.)")
        if not criteria['appropriate_length']:
            notes.append("Content length is inappropriate")
        
        if not notes:
            notes.append("All automatic validation criteria passed")
        
        return '; '.join(notes)
    
    def _generate_recommendations(self, criteria: Dict[str, bool], item: KnowledgeItem) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if not criteria['quality_indicators']:
            recommendations.append("Verify peer-review status and study methodology")
        if not criteria['recent_publication']:
            recommendations.append("Consider if older research is still relevant")
        if not criteria['has_abstract']:
            recommendations.append("Add comprehensive abstract")
        
        return recommendations


class ClinicalGuidelineValidator(KnowledgeValidator):
    """Validator for clinical guidelines and best practices."""
    
    def __init__(self):
        super().__init__("clinical_guideline_validator")
        self.authoritative_sources = [
            'who.int', 'nih.gov', 'apa.org', 'nice.org.uk',
            'cdc.gov', 'cochrane.org', 'nejm.org', 'bmj.com'
        ]
        self.guideline_keywords = [
            'guideline', 'recommendation', 'best practice', 'standard of care',
            'clinical practice', 'evidence-based', 'consensus'
        ]
    
    async def validate(self, item: KnowledgeItem) -> ValidationResult:
        """Validate clinical guideline content."""
        
        validation_criteria = {
            'authoritative_source': self._is_authoritative_source(item),
            'guideline_content': self._contains_guideline_content(item),
            'structured_recommendations': self._has_structured_recommendations(item),
            'evidence_levels': self._references_evidence_levels(item),
            'recent_update': self._is_recently_updated(item),
            'clear_scope': self._has_clear_scope(item)
        }
        
        # Higher standards for clinical guidelines
        passed_criteria = sum(1 for passed in validation_criteria.values() if passed)
        confidence = passed_criteria / len(validation_criteria)
        
        # Clinical guidelines require higher confidence threshold
        is_valid = confidence >= 0.8 and validation_criteria['authoritative_source']
        
        notes = self._generate_clinical_notes(validation_criteria, item)
        recommendations = self._generate_clinical_recommendations(validation_criteria)
        
        # Clinical guidelines always require review
        requires_review = True
        
        return ValidationResult(
            item_id=item.item_id,
            is_valid=is_valid,
            confidence=confidence,
            validation_notes=notes,
            validator_id=self.validator_id,
            validation_criteria=validation_criteria,
            recommendations=recommendations,
            requires_review=requires_review
        )
    
    def _is_authoritative_source(self, item: KnowledgeItem) -> bool:
        """Check if source is from authoritative organization."""
        if not item.source_url:
            return False
        
        return any(source in item.source_url.lower() for source in self.authoritative_sources)
    
    def _contains_guideline_content(self, item: KnowledgeItem) -> bool:
        """Check if content contains guideline-specific language."""
        text = (item.title + ' ' + (item.abstract or '') + ' ' + item.content).lower()
        
        return any(keyword in text for keyword in self.guideline_keywords)
    
    def _has_structured_recommendations(self, item: KnowledgeItem) -> bool:
        """Check if content has structured recommendations."""
        text = item.content.lower()
        
        # Look for structured recommendation patterns
        patterns = [
            r'recommendation\s+\d+', r'grade\s+[a-d]', r'level\s+[iv]+',
            r'should\s+be', r'is\s+recommended', r'consider'
        ]
        
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _references_evidence_levels(self, item: KnowledgeItem) -> bool:
        """Check if content references evidence levels."""
        text = item.content.lower()
        
        evidence_indicators = [
            'level i', 'level ii', 'level iii', 'level iv',
            'grade a', 'grade b', 'grade c', 'grade d',
            'strong evidence', 'moderate evidence', 'weak evidence',
            'systematic review', 'randomized controlled trial'
        ]
        
        return any(indicator in text for indicator in evidence_indicators)
    
    def _is_recently_updated(self, item: KnowledgeItem) -> bool:
        """Check if guideline is recently updated (within 3 years)."""
        if not item.publication_date:
            return False
        
        cutoff_date = datetime.now() - timedelta(days=3*365)
        return item.publication_date >= cutoff_date
    
    def _has_clear_scope(self, item: KnowledgeItem) -> bool:
        """Check if guideline has clear scope definition."""
        text = (item.title + ' ' + (item.abstract or '')).lower()
        
        scope_indicators = [
            'scope', 'applies to', 'intended for', 'target population',
            'clinical setting', 'patient population', 'indication'
        ]
        
        return any(indicator in text for indicator in scope_indicators)
    
    def _generate_clinical_notes(self, criteria: Dict[str, bool], item: KnowledgeItem) -> str:
        """Generate clinical validation notes."""
        notes = []
        
        if not criteria['authoritative_source']:
            notes.append("Source is not from recognized authoritative organization")
        if not criteria['guideline_content']:
            notes.append("Content does not clearly represent clinical guidelines")
        if not criteria['structured_recommendations']:
            notes.append("Lacks structured clinical recommendations")
        if not criteria['evidence_levels']:
            notes.append("Does not reference evidence levels or quality")
        if not criteria['recent_update']:
            notes.append("Guideline may be outdated (>3 years old)")
        if not criteria['clear_scope']:
            notes.append("Scope and applicability not clearly defined")
        
        if not notes:
            notes.append("Clinical guideline validation criteria met")
        
        return '; '.join(notes)
    
    def _generate_clinical_recommendations(self, criteria: Dict[str, bool]) -> List[str]:
        """Generate clinical validation recommendations."""
        recommendations = []
        
        if not criteria['authoritative_source']:
            recommendations.append("Verify source credibility and authority")
        if not criteria['structured_recommendations']:
            recommendations.append("Ensure recommendations are clearly structured and actionable")
        if not criteria['evidence_levels']:
            recommendations.append("Include evidence quality ratings for recommendations")
        
        return recommendations


class KnowledgeSourceMonitor(ABC):
    """Abstract base class for knowledge source monitors."""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.last_update = None
        self.monitored_items = set()
    
    @abstractmethod
    async def fetch_updates(self) -> List[KnowledgeItem]:
        """Fetch new updates from the knowledge source."""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[KnowledgeItem]:
        """Search the knowledge source."""
        pass


class PubMedMonitor(KnowledgeSourceMonitor):
    """Monitor PubMed for new research papers."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("pubmed")
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_terms = [
            "mental health", "psychology", "psychiatric", "depression",
            "anxiety", "therapy", "counseling", "cognitive behavioral therapy"
        ]
        self.session = None
    
    async def initialize(self):
        """Initialize the HTTP session."""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
    
    async def fetch_updates(self) -> List[KnowledgeItem]:
        """Fetch new updates from PubMed."""
        if not self.session:
            await self.initialize()
        
        all_items = []
        
        for search_term in self.search_terms:
            try:
                # Search for recent papers (last 30 days)
                search_query = f"{search_term} AND (\"last 30 days\"[PDat])"
                items = await self.search(search_query, limit=20)
                all_items.extend(items)
                
                # Small delay to respect API limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching updates for term '{search_term}': {e}")
        
        # Remove duplicates based on item_id
        unique_items = {}
        for item in all_items:
            if item.item_id not in unique_items:
                unique_items[item.item_id] = item
        
        return list(unique_items.values())
    
    async def search(self, query: str, limit: int = 10) -> List[KnowledgeItem]:
        """Search PubMed."""
        if not self.session:
            await self.initialize()
        
        try:
            # Step 1: Search for PMIDs
            search_url = f"{self.base_url}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': limit,
                'retmode': 'xml'
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
            
            async with self.session.get(search_url, params=search_params) as response:
                search_xml = await response.text()
            
            # Parse search results to get PMIDs
            pmids = self._parse_search_results(search_xml)
            
            if not pmids:
                return []
            
            # Step 2: Fetch details for PMIDs
            fetch_url = f"{self.base_url}efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            if self.api_key:
                fetch_params['api_key'] = self.api_key
            
            async with self.session.get(fetch_url, params=fetch_params) as response:
                fetch_xml = await response.text()
            
            # Parse article details
            items = self._parse_article_details(fetch_xml)
            
            return items
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def _parse_search_results(self, xml_content: str) -> List[str]:
        """Parse search results XML to extract PMIDs."""
        try:
            root = ET.fromstring(xml_content)
            pmids = []
            
            for id_elem in root.findall('.//Id'):
                pmids.append(id_elem.text)
            
            return pmids
            
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return []
    
    def _parse_article_details(self, xml_content: str) -> List[KnowledgeItem]:
        """Parse article details XML."""
        try:
            root = ET.fromstring(xml_content)
            items = []
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract basic information
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else None
                    
                    if not pmid:
                        continue
                    
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else None
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        lastname = author.find('.//LastName')
                        firstname = author.find('.//ForeName')
                        if lastname is not None and firstname is not None:
                            authors.append(f"{firstname.text} {lastname.text}")
                    
                    # Extract publication date
                    pub_date = None
                    date_elem = article.find('.//PubDate')
                    if date_elem is not None:
                        year_elem = date_elem.find('.//Year')
                        month_elem = date_elem.find('.//Month')
                        day_elem = date_elem.find('.//Day')
                        
                        if year_elem is not None:
                            year = int(year_elem.text)
                            month = int(month_elem.text) if month_elem is not None else 1
                            day = int(day_elem.text) if day_elem is not None else 1
                            pub_date = datetime(year, month, day)
                    
                    # Extract keywords
                    keywords = []
                    for keyword in article.findall('.//Keyword'):
                        if keyword.text:
                            keywords.append(keyword.text)
                    
                    # Create knowledge item
                    item = KnowledgeItem(
                        item_id=f"pubmed_{pmid}",
                        title=title,
                        source_type=KnowledgeSource.RESEARCH_PAPER,
                        content=abstract or title,  # Use abstract as content
                        abstract=abstract,
                        authors=authors,
                        publication_date=pub_date,
                        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        keywords=keywords,
                        categories=["research", "pubmed"],
                        metadata={
                            'pmid': pmid,
                            'source': 'pubmed'
                        }
                    )
                    
                    items.append(item)
                    
                except Exception as e:
                    logger.error(f"Error parsing individual article: {e}")
                    continue
            
            return items
            
        except Exception as e:
            logger.error(f"Error parsing article details: {e}")
            return []


class ArXivMonitor(KnowledgeSourceMonitor):
    """Monitor ArXiv for preprints in relevant categories."""
    
    def __init__(self):
        super().__init__("arxiv")
        self.base_url = "http://export.arxiv.org/api/query"
        self.categories = ["q-bio.NC", "cs.AI", "cs.CL", "stat.ML"]  # Neuroscience, AI, etc.
        self.session = None
    
    async def initialize(self):
        """Initialize the HTTP session.""" 
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
    
    async def fetch_updates(self) -> List[KnowledgeItem]:
        """Fetch new updates from ArXiv."""
        if not self.session:
            await self.initialize()
        
        all_items = []
        
        for category in self.categories:
            try:
                # Search for recent papers in category
                search_query = f"cat:{category}"
                items = await self.search(search_query, limit=10)
                all_items.extend(items)
                
                # Small delay
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching updates for category '{category}': {e}")
        
        return all_items
    
    async def search(self, query: str, limit: int = 10) -> List[KnowledgeItem]:
        """Search ArXiv."""
        if not self.session:
            await self.initialize()
        
        try:
            params = {
                'search_query': query,
                'start': 0,
                'max_results': limit,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                xml_content = await response.text()
            
            items = self._parse_arxiv_results(xml_content)
            return items
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def _parse_arxiv_results(self, xml_content: str) -> List[KnowledgeItem]:
        """Parse ArXiv search results."""
        try:
            # Parse using feedparser which handles ArXiv's Atom format well
            feed = feedparser.parse(xml_content)
            items = []
            
            for entry in feed.entries:
                try:
                    # Extract ArXiv ID from URL
                    arxiv_id = entry.id.split('/')[-1]
                    
                    # Extract authors
                    authors = []
                    if hasattr(entry, 'authors'):
                        authors = [author.name for author in entry.authors]
                    
                    # Extract categories
                    categories = []
                    if hasattr(entry, 'tags'):
                        categories = [tag.term for tag in entry.tags]
                    
                    # Parse publication date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # Create knowledge item
                    item = KnowledgeItem(
                        item_id=f"arxiv_{arxiv_id}",
                        title=entry.title,
                        source_type=KnowledgeSource.RESEARCH_PAPER,
                        content=entry.summary,
                        abstract=entry.summary,
                        authors=authors,
                        publication_date=pub_date,
                        source_url=entry.link,
                        categories=categories + ["preprint", "arxiv"],
                        metadata={
                            'arxiv_id': arxiv_id,
                            'source': 'arxiv'
                        }
                    )
                    
                    items.append(item)
                    
                except Exception as e:
                    logger.error(f"Error parsing ArXiv entry: {e}")
                    continue
            
            return items
            
        except Exception as e:
            logger.error(f"Error parsing ArXiv results: {e}")
            return []


class KnowledgeGraphManager:
    """Manages knowledge graph updates and relationships."""
    
    def __init__(self):
        self.knowledge_graph = defaultdict(set)  # Simple adjacency list
        self.entity_metadata = {}
        self.relationships = {}
        self.lock = threading.RLock()
    
    def add_knowledge_item(self, item: KnowledgeItem) -> None:
        """Add knowledge item to the graph."""
        with self.lock:
            # Add entity
            self.entity_metadata[item.item_id] = {
                'title': item.title,
                'type': item.source_type.value,
                'keywords': item.keywords,
                'categories': item.categories,
                'confidence': item.confidence_score
            }
            
            # Create relationships based on keywords and categories
            self._create_relationships(item)
    
    def _create_relationships(self, item: KnowledgeItem) -> None:
        """Create relationships between knowledge items."""
        
        # Find related items based on keywords
        for other_id, metadata in self.entity_metadata.items():
            if other_id == item.item_id:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(item, metadata)
            
            if similarity > 0.3:  # Threshold for relationship
                self.knowledge_graph[item.item_id].add(other_id)
                self.knowledge_graph[other_id].add(item.item_id)
                
                # Store relationship metadata
                rel_key = tuple(sorted([item.item_id, other_id]))
                self.relationships[rel_key] = {
                    'similarity': similarity,
                    'type': 'conceptual_similarity',
                    'created_at': datetime.now()
                }
    
    def _calculate_similarity(self, item: KnowledgeItem, metadata: Dict[str, Any]) -> float:
        """Calculate similarity between knowledge items."""
        
        # Keywords similarity
        item_keywords = set(k.lower() for k in item.keywords)
        other_keywords = set(k.lower() for k in metadata.get('keywords', []))
        
        if not item_keywords and not other_keywords:
            keyword_sim = 0.0
        elif not item_keywords or not other_keywords:
            keyword_sim = 0.0
        else:
            keyword_sim = len(item_keywords.intersection(other_keywords)) / len(item_keywords.union(other_keywords))
        
        # Categories similarity
        item_categories = set(c.lower() for c in item.categories)
        other_categories = set(c.lower() for c in metadata.get('categories', []))
        
        if not item_categories and not other_categories:
            category_sim = 0.0
        elif not item_categories or not other_categories:
            category_sim = 0.0
        else:
            category_sim = len(item_categories.intersection(other_categories)) / len(item_categories.union(other_categories))
        
        # Combined similarity
        return (keyword_sim + category_sim) / 2
    
    def get_related_items(self, item_id: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """Get items related to the given item."""
        with self.lock:
            if item_id not in self.knowledge_graph:
                return []
            
            related_ids = list(self.knowledge_graph[item_id])
            
            # Sort by relationship strength
            related_with_scores = []
            for related_id in related_ids:
                rel_key = tuple(sorted([item_id, related_id]))
                similarity = self.relationships.get(rel_key, {}).get('similarity', 0.0)
                
                related_with_scores.append({
                    'item_id': related_id,
                    'similarity': similarity,
                    'metadata': self.entity_metadata.get(related_id, {})
                })
            
            # Sort by similarity and return top items
            related_with_scores.sort(key=lambda x: x['similarity'], reverse=True)
            return related_with_scores[:max_items]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        with self.lock:
            total_nodes = len(self.entity_metadata)
            total_edges = sum(len(neighbors) for neighbors in self.knowledge_graph.values()) // 2
            
            # Node degree statistics
            degrees = [len(neighbors) for neighbors in self.knowledge_graph.values()]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            
            # Source type distribution
            source_types = defaultdict(int)
            for metadata in self.entity_metadata.values():
                source_types[metadata.get('type', 'unknown')] += 1
            
            return {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'average_degree': avg_degree,
                'source_type_distribution': dict(source_types),
                'density': (2 * total_edges) / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
            }


class ResearchTrendAnalyzer:
    """Analyzes research trends and emerging topics."""
    
    def __init__(self):
        self.knowledge_items: List[KnowledgeItem] = []
        self.trend_cache = {}
        self.cache_timeout = timedelta(hours=6)
    
    def add_knowledge_items(self, items: List[KnowledgeItem]) -> None:
        """Add knowledge items for trend analysis."""
        self.knowledge_items.extend(items)
        
        # Keep only recent items for trend analysis
        cutoff_date = datetime.now() - timedelta(days=365)
        self.knowledge_items = [
            item for item in self.knowledge_items
            if item.publication_date and item.publication_date >= cutoff_date
        ]
    
    async def analyze_trends(self, topic: str, time_period_days: int = 180) -> ResearchTrend:
        """Analyze trends for a specific topic."""
        
        # Check cache
        cache_key = f"{topic}_{time_period_days}"
        if cache_key in self.trend_cache:
            cached_result, cache_time = self.trend_cache[cache_key]
            if datetime.now() - cache_time < self.cache_timeout:
                return cached_result
        
        # Filter items by topic and time period
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        relevant_items = []
        
        for item in self.knowledge_items:
            if not item.publication_date or item.publication_date < cutoff_date:
                continue
            
            # Check if item is relevant to topic
            if self._is_relevant_to_topic(item, topic):
                relevant_items.append(item)
        
        if len(relevant_items) < 3:  # Not enough data for trend analysis
            return ResearchTrend(
                topic=topic,
                trend_direction='insufficient_data',
                confidence=0.0,
                supporting_papers=[],
                key_findings=[],
                implications=[],
                time_period=f"{time_period_days} days"
            )
        
        # Analyze trend direction
        trend_direction = self._calculate_trend_direction(relevant_items, time_period_days)
        confidence = min(1.0, len(relevant_items) / 20)  # Higher confidence with more papers
        
        # Extract key findings
        key_findings = self._extract_key_findings(relevant_items, topic)
        
        # Generate implications
        implications = self._generate_implications(trend_direction, key_findings, topic)
        
        result = ResearchTrend(
            topic=topic,
            trend_direction=trend_direction,
            confidence=confidence,
            supporting_papers=[item.item_id for item in relevant_items[:10]],  # Limit to top 10
            key_findings=key_findings,
            implications=implications,
            time_period=f"{time_period_days} days"
        )
        
        # Cache result
        self.trend_cache[cache_key] = (result, datetime.now())
        
        return result
    
    def _is_relevant_to_topic(self, item: KnowledgeItem, topic: str) -> bool:
        """Check if knowledge item is relevant to the topic."""
        topic_lower = topic.lower()
        
        # Check in title
        if topic_lower in item.title.lower():
            return True
        
        # Check in keywords
        for keyword in item.keywords:
            if topic_lower in keyword.lower() or keyword.lower() in topic_lower:
                return True
        
        # Check in abstract/content
        content = (item.abstract or item.content).lower()
        if topic_lower in content:
            return True
        
        return False
    
    def _calculate_trend_direction(self, items: List[KnowledgeItem], time_period_days: int) -> str:
        """Calculate trend direction based on publication frequency."""
        
        # Group by month
        monthly_counts = defaultdict(int)
        
        for item in items:
            if item.publication_date:
                month_key = item.publication_date.strftime('%Y-%m')
                monthly_counts[month_key] += 1
        
        if len(monthly_counts) < 2:
            return 'stable'
        
        # Calculate trend
        sorted_months = sorted(monthly_counts.keys())
        counts = [monthly_counts[month] for month in sorted_months]
        
        # Simple linear trend calculation
        n = len(counts)
        if n < 2:
            return 'stable'
        
        x_mean = (n - 1) / 2
        y_mean = sum(counts) / n
        
        slope_numerator = sum((i - x_mean) * (counts[i] - y_mean) for i in range(n))
        slope_denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if slope_denominator == 0:
            return 'stable'
        
        slope = slope_numerator / slope_denominator
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _extract_key_findings(self, items: List[KnowledgeItem], topic: str) -> List[str]:
        """Extract key findings from research items."""
        
        # This is a simplified implementation
        # In practice, would use NLP to extract key findings
        
        findings = []
        
        # Look for common patterns in abstracts
        common_terms = defaultdict(int)
        
        for item in items:
            content = (item.abstract or item.content).lower()
            words = re.findall(r'\w+', content)
            
            for word in words:
                if len(word) > 4 and word not in ['study', 'research', 'analysis', 'method', 'result']:
                    common_terms[word] += 1
        
        # Get most frequent terms
        sorted_terms = sorted(common_terms.items(), key=lambda x: x[1], reverse=True)
        
        # Generate findings based on frequent terms
        for term, count in sorted_terms[:3]:
            if count >= len(items) * 0.3:  # Appears in at least 30% of papers
                findings.append(f"Frequent mention of '{term}' in recent research on {topic}")
        
        if not findings:
            findings.append(f"Emerging research activity in {topic}")
        
        return findings
    
    def _generate_implications(self, trend_direction: str, key_findings: List[str], topic: str) -> List[str]:
        """Generate implications based on trend analysis."""
        
        implications = []
        
        if trend_direction == 'increasing':
            implications.append(f"Growing research interest in {topic} indicates emerging importance")
            implications.append("Consider updating knowledge base with latest findings")
        elif trend_direction == 'declining':
            implications.append(f"Declining research in {topic} may indicate maturity or shifting focus")
            implications.append("Review established knowledge for continued relevance")
        else:
            implications.append(f"Stable research activity in {topic} indicates consistent interest")
        
        # Add implications based on findings
        for finding in key_findings:
            if 'frequent mention' in finding.lower():
                implications.append("Key concepts are being reinforced across multiple studies")
        
        return implications


class KnowledgeIntegrationSystem:
    """
    Comprehensive real-time research and knowledge integration system.
    Monitors multiple sources, validates content, and maintains knowledge graph.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        
        # Knowledge storage
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.pending_validation: Dict[str, KnowledgeItem] = {}
        self.validated_items: Dict[str, KnowledgeItem] = {}
        
        # Components
        self.source_monitors: Dict[str, KnowledgeSourceMonitor] = {}
        self.validators: Dict[str, KnowledgeValidator] = {}
        self.knowledge_graph = KnowledgeGraphManager()
        self.trend_analyzer = ResearchTrendAnalyzer()
        
        # Configuration
        self.monitoring_enabled = True
        self.auto_validation_enabled = True
        self.update_interval_minutes = 60
        self.max_items_per_source = 100
        
        # Background tasks
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._validation_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize components
        self._initialize_components()
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("KnowledgeIntegrationSystem initialized")
    
    async def start(self) -> None:
        """Start the knowledge integration system."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize source monitors
        for monitor in self.source_monitors.values():
            if hasattr(monitor, 'initialize'):
                await monitor.initialize()
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._validation_task = asyncio.create_task(self._validation_loop())
        
        # Emit startup event
        await self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source_agent="knowledge_integration",
            data={'component': 'knowledge_system', 'status': 'started'}
        ))
        
        logger.info("KnowledgeIntegrationSystem started")
    
    async def stop(self) -> None:
        """Stop the knowledge integration system."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._validation_task:
            self._validation_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self._monitoring_task, self._validation_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cleanup source monitors
        for monitor in self.source_monitors.values():
            if hasattr(monitor, 'cleanup'):
                await monitor.cleanup()
        
        logger.info("KnowledgeIntegrationSystem stopped")
    
    async def add_knowledge_item(self, item: KnowledgeItem) -> bool:
        """Add a knowledge item to the system."""
        
        try:
            with self.lock:
                # Check for duplicates
                if item.item_id in self.knowledge_items:
                    logger.info(f"Knowledge item {item.item_id} already exists")
                    return False
                
                # Add to pending validation
                self.pending_validation[item.item_id] = item
                self.knowledge_items[item.item_id] = item
            
            # Publish new knowledge event
            await self.event_bus.publish(Event(
                event_type="knowledge_item_added",
                source_agent="knowledge_integration",
                priority=EventPriority.NORMAL,
                data={
                    'item_id': item.item_id,
                    'title': item.title,
                    'source_type': item.source_type.value,
                    'requires_validation': True
                }
            ))
            
            logger.info(f"Added knowledge item: {item.item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge item: {e}")
            return False
    
    async def validate_knowledge_item(self, item_id: str) -> Optional[ValidationResult]:
        """Validate a specific knowledge item."""
        
        with self.lock:
            if item_id not in self.knowledge_items:
                return None
            
            item = self.knowledge_items[item_id]
        
        # Choose appropriate validator
        validator = self._select_validator(item)
        
        if not validator:
            logger.warning(f"No suitable validator found for item {item_id}")
            return None
        
        try:
            validation_result = await validator.validate(item)
            
            # Update item status based on validation
            with self.lock:
                if validation_result.is_valid:
                    item.status = ContentStatus.VALIDATED
                    item.confidence_score = validation_result.confidence
                    self.validated_items[item_id] = item
                    
                    # Add to knowledge graph
                    self.knowledge_graph.add_knowledge_item(item)
                    
                    # Add to trend analyzer
                    self.trend_analyzer.add_knowledge_items([item])
                    
                else:
                    if validation_result.requires_review:
                        item.status = ContentStatus.UNDER_REVIEW
                    else:
                        item.status = ContentStatus.REJECTED
                
                item.updated_at = datetime.now()
                
                # Remove from pending validation
                self.pending_validation.pop(item_id, None)
            
            # Publish validation event
            await self.event_bus.publish(Event(
                event_type="knowledge_validation_completed",
                source_agent="knowledge_integration",
                priority=EventPriority.HIGH if not validation_result.is_valid else EventPriority.NORMAL,
                data={
                    'item_id': item_id,
                    'validation_result': validation_result.to_dict()
                }
            ))
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating knowledge item {item_id}: {e}")
            return None
    
    async def search_knowledge(self, query: str, 
                             source_types: List[KnowledgeSource] = None,
                             limit: int = 20) -> List[KnowledgeItem]:
        """Search knowledge items."""
        
        with self.lock:
            items = list(self.validated_items.values())
        
        # Filter by source types if specified
        if source_types:
            items = [item for item in items if item.source_type in source_types]
        
        # Simple text-based search
        query_lower = query.lower()
        matching_items = []
        
        for item in items:
            relevance_score = 0.0
            
            # Check title
            if query_lower in item.title.lower():
                relevance_score += 2.0
            
            # Check keywords
            for keyword in item.keywords:
                if query_lower in keyword.lower():
                    relevance_score += 1.5
            
            # Check abstract/content
            content = (item.abstract or item.content).lower()
            if query_lower in content:
                relevance_score += 1.0
            
            if relevance_score > 0:
                item.relevance_score = relevance_score
                matching_items.append(item)
        
        # Sort by relevance and return top results
        matching_items.sort(key=lambda x: x.relevance_score, reverse=True)
        return matching_items[:limit]
    
    async def get_research_trends(self, topic: str) -> ResearchTrend:
        """Get research trends for a topic."""
        return await self.trend_analyzer.analyze_trends(topic)
    
    async def get_related_knowledge(self, item_id: str) -> List[Dict[str, Any]]:
        """Get knowledge items related to the specified item."""
        return self.knowledge_graph.get_related_items(item_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        
        with self.lock:
            total_items = len(self.knowledge_items)
            validated_items = len(self.validated_items)
            pending_items = len(self.pending_validation)
            
            # Status by source type
            source_stats = defaultdict(int)
            for item in self.knowledge_items.values():
                source_stats[item.source_type.value] += 1
        
        graph_stats = self.knowledge_graph.get_graph_statistics()
        
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'total_knowledge_items': total_items,
            'validated_items': validated_items,
            'pending_validation': pending_items,
            'validation_rate': (validated_items / total_items * 100) if total_items > 0 else 0,
            'source_distribution': dict(source_stats),
            'knowledge_graph_stats': graph_stats,
            'active_monitors': len(self.source_monitors),
            'active_validators': len(self.validators),
            'last_update': datetime.now().isoformat()
        }
    
    def _initialize_components(self) -> None:
        """Initialize system components."""
        
        # Initialize source monitors
        self.source_monitors['pubmed'] = PubMedMonitor()
        self.source_monitors['arxiv'] = ArXivMonitor()
        
        # Initialize validators
        self.validators['automatic'] = AutomaticContentValidator()
        self.validators['clinical'] = ClinicalGuidelineValidator()
        
        logger.info("Knowledge integration components initialized")
    
    def _select_validator(self, item: KnowledgeItem) -> Optional[KnowledgeValidator]:
        """Select appropriate validator for a knowledge item."""
        
        if item.source_type == KnowledgeSource.CLINICAL_GUIDELINE:
            return self.validators.get('clinical')
        elif item.source_type in [KnowledgeSource.RESEARCH_PAPER, KnowledgeSource.SYSTEMATIC_REVIEW]:
            return self.validators.get('automatic')
        else:
            return self.validators.get('automatic')  # Default validator
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions."""
        
        # Subscribe to knowledge requests
        self.event_bus.subscribe(
            "knowledge_search_request",
            self._handle_knowledge_search,
            agent_id="knowledge_integration"
        )
        
        logger.info("Knowledge integration event subscriptions configured")
    
    async def _handle_knowledge_search(self, event: Event) -> None:
        """Handle knowledge search requests."""
        try:
            search_data = event.data
            query = search_data.get('query', '')
            limit = search_data.get('limit', 10)
            
            results = await self.search_knowledge(query, limit=limit)
            
            # Send response if reply_to is specified
            if event.reply_to:
                await self.event_bus.publish(Event(
                    event_type="knowledge_search_response",
                    source_agent="knowledge_integration",
                    target_agent=event.reply_to,
                    correlation_id=event.correlation_id,
                    data={
                        'query': query,
                        'results': [item.to_dict() for item in results],
                        'total_results': len(results)
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error handling knowledge search: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Background loop for monitoring knowledge sources."""
        
        while self._running:
            try:
                if not self.monitoring_enabled:
                    await asyncio.sleep(60)
                    continue
                
                # Fetch updates from all monitors
                all_new_items = []
                
                for monitor_name, monitor in self.source_monitors.items():
                    try:
                        new_items = await monitor.fetch_updates()
                        
                        # Limit items per source
                        if len(new_items) > self.max_items_per_source:
                            new_items = new_items[:self.max_items_per_source]
                        
                        all_new_items.extend(new_items)
                        
                        logger.info(f"Fetched {len(new_items)} items from {monitor_name}")
                        
                    except Exception as e:
                        logger.error(f"Error fetching from {monitor_name}: {e}")
                
                # Add new items to the system
                for item in all_new_items:
                    await self.add_knowledge_item(item)
                
                if all_new_items:
                    # Publish batch update event
                    await self.event_bus.publish(Event(
                        event_type="knowledge_batch_update",
                        source_agent="knowledge_integration",
                        priority=EventPriority.NORMAL,
                        data={
                            'new_items_count': len(all_new_items),
                            'sources_updated': list(self.source_monitors.keys())
                        }
                    ))
                
                # Wait before next monitoring cycle
                await asyncio.sleep(self.update_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _validation_loop(self) -> None:
        """Background loop for validating knowledge items."""
        
        while self._running:
            try:
                if not self.auto_validation_enabled:
                    await asyncio.sleep(60)
                    continue
                
                # Get items pending validation
                with self.lock:
                    pending_items = list(self.pending_validation.keys())
                
                if not pending_items:
                    await asyncio.sleep(30)
                    continue
                
                # Validate items (limit batch size to avoid overload)
                batch_size = 5
                for i in range(0, len(pending_items), batch_size):
                    batch = pending_items[i:i + batch_size]
                    
                    # Validate batch concurrently
                    validation_tasks = [
                        self.validate_knowledge_item(item_id) 
                        for item_id in batch
                    ]
                    
                    results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                    
                    # Process results
                    successful_validations = sum(
                        1 for result in results 
                        if isinstance(result, ValidationResult)
                    )
                    
                    logger.info(f"Validated batch of {len(batch)} items, {successful_validations} successful")
                    
                    # Small delay between batches
                    await asyncio.sleep(5)
                
                # Wait before next validation cycle
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in validation loop: {e}")
                await asyncio.sleep(60)


# Factory function
def create_knowledge_integration_system(event_bus: EventBus) -> KnowledgeIntegrationSystem:
    """Create a knowledge integration system instance."""
    return KnowledgeIntegrationSystem(event_bus)