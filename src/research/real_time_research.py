"""
Real-Time Research Integration for Evidence-Based Practice

This module integrates the latest mental health research into the diagnostic process,
providing access to current clinical guidelines, treatment efficacy data, and
evidence-based recommendations that align with professional standards.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import json
import numpy as np
from collections import defaultdict
import aiohttp
import hashlib
import os

from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger
from ..models.llm import get_llm

logger = get_logger(__name__)


def _json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return {"__datetime__": True, "value": obj.isoformat()}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "value": obj.tolist()}
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def _json_deserial(obj: Dict[str, Any]) -> Any:
    """JSON deserializer for custom types."""
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["value"])
    if "__ndarray__" in obj:
        return np.array(obj["value"])
    return obj

@dataclass
class ResearchArticle:
    """Research article or study"""
    article_id: str
    title: str
    authors: List[str]
    journal: str
    publication_date: datetime
    doi: str
    abstract: str
    keywords: List[str]
    study_type: str  # RCT, meta-analysis, systematic_review, case_study
    evidence_level: str  # high, medium, low
    clinical_relevance: float  # 0.0 to 1.0
    mental_health_domains: List[str]
    treatment_approaches: List[str]
    population_studied: str
    sample_size: int
    key_findings: List[str]
    clinical_implications: List[str]
    limitations: List[str]

@dataclass
class ClinicalGuideline:
    """Clinical practice guideline"""
    guideline_id: str
    organization: str  # APA, WHO, NICE, etc.
    title: str
    version: str
    last_updated: datetime
    scope: str
    target_conditions: List[str]
    recommendations: List[Dict[str, Any]]
    evidence_grades: Dict[str, str]
    implementation_notes: List[str]
    cultural_considerations: List[str]
    contraindications: List[str]

@dataclass
class TreatmentEfficacy:
    """Treatment efficacy data from research"""
    treatment_id: str
    treatment_name: str
    condition: str
    population: str
    efficacy_score: float  # 0.0 to 1.0
    effect_size: float
    number_needed_to_treat: Optional[int]
    response_rate: float
    remission_rate: float
    dropout_rate: float
    adverse_effects: List[str]
    duration_weeks: int
    follow_up_data: Dict[str, float]
    studies_count: int
    total_participants: int
    confidence_interval: Tuple[float, float]
    last_updated: datetime

@dataclass
class EvidenceBasedRecommendation:
    """Evidence-based treatment recommendation"""
    recommendation_id: str
    condition: str
    severity: str
    recommendation_text: str
    evidence_level: str
    strength_of_recommendation: str  # strong, moderate, weak
    supporting_studies: List[str]
    effect_size: float
    confidence_score: float
    alternatives: List[str]
    contraindications: List[str]
    monitoring_requirements: List[str]
    expected_timeline: str
    cultural_adaptations: List[str]

class RealTimeResearchEngine:
    """
    Engine for integrating real-time mental health research into
    diagnostic and treatment processes with evidence-based recommendations.
    """
    
    def __init__(self, vector_db: Optional[CentralVectorDB] = None):
        """Initialize the real-time research engine"""
        self.vector_db = vector_db
        self.llm = get_llm()
        self.logger = get_logger(__name__)
        
        # Research databases
        self.research_articles = {}  # article_id -> ResearchArticle
        self.clinical_guidelines = {}  # guideline_id -> ClinicalGuideline
        self.treatment_efficacies = defaultdict(list)  # condition -> [TreatmentEfficacy]
        self.evidence_cache = {}  # For caching search results
        
        # API configurations
        self.research_apis = {
            "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            "cochrane": "https://www.cochranelibrary.com/api/",
            "nice": "https://www.nice.org.uk/api/",
            "apa": "https://www.apa.org/api/"
        }
        
        # Update frequencies
        self.guideline_update_frequency = 30  # days
        self.research_update_frequency = 7    # days
        self.efficacy_update_frequency = 14   # days
        
        # Cache settings
        self.cache_duration = 24 * 60 * 60  # 24 hours in seconds
        
        # Load existing data
        self._load_research_data()
    
    async def get_evidence_based_recommendations(self,
                                               condition: str,
                                               severity: str,
                                               patient_characteristics: Dict[str, Any],
                                               cultural_context: str = None) -> List[EvidenceBasedRecommendation]:
        """
        Get evidence-based treatment recommendations
        
        Args:
            condition: Mental health condition
            severity: Severity level (mild, moderate, severe)
            patient_characteristics: Patient demographics and characteristics
            cultural_context: Cultural background for adaptation
            
        Returns:
            List of evidence-based recommendations
        """
        try:
            self.logger.info(f"Getting evidence-based recommendations for {condition} ({severity})")
            
            # Get current clinical guidelines
            guidelines = await self._get_relevant_guidelines(condition)
            
            # Get treatment efficacy data
            efficacy_data = await self._get_treatment_efficacy(condition, severity)
            
            # Get recent research
            recent_research = await self._search_recent_research(
                condition, days_back=90
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                condition, severity, patient_characteristics, 
                guidelines, efficacy_data, recent_research, cultural_context
            )
            
            # Rank recommendations by evidence strength
            recommendations.sort(
                key=lambda x: (x.confidence_score, x.effect_size), reverse=True
            )
            
            # Store recommendations for future reference
            for rec in recommendations:
                await self._store_recommendation(rec)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting evidence-based recommendations: {str(e)}")
            return []
    
    async def search_current_research(self,
                                    query: str,
                                    study_types: List[str] = None,
                                    date_range: int = 365) -> List[ResearchArticle]:
        """
        Search current research literature
        
        Args:
            query: Search query
            study_types: Types of studies to include
            date_range: Days back to search
            
        Returns:
            List of relevant research articles
        """
        try:
            self.logger.info(f"Searching current research for: {query}")
            
            # Check cache first
            cache_key = self._generate_cache_key(query, study_types, date_range)
            if cache_key in self.evidence_cache:
                cached_result = self.evidence_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    return cached_result["articles"]
            
            # Search multiple databases
            all_articles = []
            
            # Search PubMed
            pubmed_articles = await self._search_pubmed(query, date_range)
            all_articles.extend(pubmed_articles)
            
            # Search Cochrane (simulated)
            cochrane_articles = await self._search_cochrane(query, date_range)
            all_articles.extend(cochrane_articles)
            
            # Filter by study types if specified
            if study_types:
                all_articles = [
                    article for article in all_articles
                    if article.study_type in study_types
                ]
            
            # Remove duplicates and rank by relevance
            unique_articles = self._deduplicate_articles(all_articles)
            ranked_articles = self._rank_articles_by_relevance(unique_articles, query)
            
            # Cache results
            self.evidence_cache[cache_key] = {
                "articles": ranked_articles,
                "timestamp": datetime.now(),
                "query": query
            }
            
            # Store in vector database
            for article in ranked_articles[:10]:  # Store top 10
                await self._store_article_in_db(article)
            
            return ranked_articles
            
        except Exception as e:
            self.logger.error(f"Error searching current research: {str(e)}")
            return []
    
    async def get_clinical_guidelines(self,
                                    condition: str,
                                    organization: str = None) -> List[ClinicalGuideline]:
        """
        Get current clinical guidelines for a condition
        
        Args:
            condition: Mental health condition
            organization: Specific organization (optional)
            
        Returns:
            List of relevant clinical guidelines
        """
        try:
            self.logger.info(f"Getting clinical guidelines for {condition}")
            
            # Check if guidelines need updating
            await self._update_guidelines_if_needed()
            
            # Filter guidelines by condition
            relevant_guidelines = []
            for guideline in self.clinical_guidelines.values():
                if condition.lower() in [c.lower() for c in guideline.target_conditions]:
                    if not organization or guideline.organization.lower() == organization.lower():
                        relevant_guidelines.append(guideline)
            
            # Sort by organization priority and recency
            org_priority = {"WHO": 1, "APA": 2, "NICE": 3, "AMA": 4}
            relevant_guidelines.sort(
                key=lambda x: (
                    org_priority.get(x.organization, 99),
                    -x.last_updated.timestamp()
                )
            )
            
            return relevant_guidelines
            
        except Exception as e:
            self.logger.error(f"Error getting clinical guidelines: {str(e)}")
            return []
    
    async def get_treatment_efficacy_data(self,
                                        treatment: str,
                                        condition: str,
                                        population: str = "general") -> Optional[TreatmentEfficacy]:
        """
        Get treatment efficacy data from research
        
        Args:
            treatment: Treatment approach
            condition: Mental health condition
            population: Target population
            
        Returns:
            Treatment efficacy data if available
        """
        try:
            # Check existing efficacy data
            condition_efficacies = self.treatment_efficacies.get(condition, [])
            
            for efficacy in condition_efficacies:
                if (treatment.lower() in efficacy.treatment_name.lower() and
                    population.lower() in efficacy.population.lower()):
                    # Check if data is current
                    if (datetime.now() - efficacy.last_updated).days < self.efficacy_update_frequency:
                        return efficacy
            
            # Search for new efficacy data
            new_efficacy = await self._search_treatment_efficacy(treatment, condition, population)
            
            if new_efficacy:
                self.treatment_efficacies[condition].append(new_efficacy)
                await self._store_efficacy_in_db(new_efficacy)
                return new_efficacy
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting treatment efficacy data: {str(e)}")
            return None
    
    async def update_research_database(self) -> Dict[str, int]:
        """
        Update research database with latest information
        
        Returns:
            Update statistics
        """
        try:
            self.logger.info("Updating research database")
            
            stats = {
                "articles_added": 0,
                "guidelines_updated": 0,
                "efficacy_data_updated": 0
            }
            
            # Update research articles for key mental health topics
            key_topics = [
                "depression treatment", "anxiety therapy", "CBT effectiveness",
                "PTSD treatment", "bipolar therapy", "schizophrenia treatment",
                "mindfulness therapy", "DBT effectiveness"
            ]
            
            for topic in key_topics:
                new_articles = await self.search_current_research(topic, date_range=30)
                stats["articles_added"] += len(new_articles)
            
            # Update clinical guidelines
            guidelines_updated = await self._update_all_guidelines()
            stats["guidelines_updated"] = guidelines_updated
            
            # Update treatment efficacy data
            efficacy_updated = await self._update_efficacy_data()
            stats["efficacy_data_updated"] = efficacy_updated
            
            # Persist updated data
            self._persist_research_data()
            
            self.logger.info(f"Research database update completed: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error updating research database: {str(e)}")
            return {"error": str(e)}
    
    async def validate_treatment_approach(self,
                                        treatment: str,
                                        condition: str,
                                        patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate treatment approach against current evidence
        
        Args:
            treatment: Proposed treatment
            condition: Mental health condition
            patient_profile: Patient characteristics
            
        Returns:
            Validation results with evidence support
        """
        try:
            # Get evidence for treatment
            efficacy_data = await self.get_treatment_efficacy_data(treatment, condition)
            guidelines = await self.get_clinical_guidelines(condition)
            recent_research = await self.search_current_research(
                f"{treatment} {condition}", date_range=180
            )
            
            # Analyze evidence support
            evidence_support = self._analyze_evidence_support(
                treatment, condition, efficacy_data, guidelines, recent_research
            )
            
            # Check for contraindications
            contraindications = self._check_contraindications(
                treatment, patient_profile, guidelines
            )
            
            # Generate validation summary
            validation_summary = await self._generate_validation_summary(
                treatment, condition, evidence_support, contraindications
            )
            
            return {
                "treatment": treatment,
                "condition": condition,
                "evidence_support": evidence_support,
                "contraindications": contraindications,
                "validation_summary": validation_summary,
                "recommendation": self._generate_treatment_recommendation(evidence_support),
                "confidence_level": evidence_support.get("confidence", 0.5),
                "alternative_treatments": await self._suggest_alternatives(
                    treatment, condition, evidence_support
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error validating treatment approach: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _search_pubmed(self, query: str, date_range: int) -> List[ResearchArticle]:
        """Search PubMed database (simulated)"""
        # In a real implementation, this would use the actual PubMed API
        # For now, return simulated results
        
        sample_articles = [
            ResearchArticle(
                article_id=f"pubmed_{hashlib.md5(query.encode()).hexdigest()[:8]}",
                title=f"Effectiveness of {query} in Clinical Practice",
                authors=["Smith, J.", "Johnson, M.", "Williams, K."],
                journal="Journal of Clinical Psychology",
                publication_date=datetime.now() - timedelta(days=30),
                doi="10.1234/jcp.2024.001",
                abstract=f"This study examines the effectiveness of {query} in clinical settings...",
                keywords=query.split(),
                study_type="RCT",
                evidence_level="high",
                clinical_relevance=0.8,
                mental_health_domains=["mood_disorders", "anxiety_disorders"],
                treatment_approaches=[query],
                population_studied="adults",
                sample_size=150,
                key_findings=[f"{query} showed significant improvement", "Effect size: d=0.7"],
                clinical_implications=["Consider as first-line treatment", "Monitor for side effects"],
                limitations=["Small sample size", "Short follow-up period"]
            )
        ]
        
        return sample_articles
    
    async def _search_cochrane(self, query: str, date_range: int) -> List[ResearchArticle]:
        """Search Cochrane database (simulated)"""
        # Simulated Cochrane results
        return []
    
    async def _get_relevant_guidelines(self, condition: str) -> List[ClinicalGuideline]:
        """Get relevant clinical guidelines for condition"""
        return [g for g in self.clinical_guidelines.values() 
                if condition.lower() in [c.lower() for c in g.target_conditions]]
    
    async def _get_treatment_efficacy(self, condition: str, severity: str) -> List[TreatmentEfficacy]:
        """Get treatment efficacy data for condition and severity"""
        return self.treatment_efficacies.get(condition, [])
    
    async def _search_recent_research(self, condition: str, days_back: int) -> List[ResearchArticle]:
        """Search for recent research on condition"""
        return await self.search_current_research(condition, date_range=days_back)
    
    async def _generate_recommendations(self,
                                      condition: str,
                                      severity: str,
                                      patient_characteristics: Dict[str, Any],
                                      guidelines: List[ClinicalGuideline],
                                      efficacy_data: List[TreatmentEfficacy],
                                      recent_research: List[ResearchArticle],
                                      cultural_context: str = None) -> List[EvidenceBasedRecommendation]:
        """Generate evidence-based recommendations"""
        
        recommendations = []
        
        # Generate recommendations from guidelines
        for guideline in guidelines:
            for rec_data in guideline.recommendations:
                if self._matches_severity(rec_data, severity):
                    recommendation = EvidenceBasedRecommendation(
                        recommendation_id=f"rec_{guideline.guideline_id}_{len(recommendations)}",
                        condition=condition,
                        severity=severity,
                        recommendation_text=rec_data.get("text", ""),
                        evidence_level=rec_data.get("evidence_level", "medium"),
                        strength_of_recommendation=rec_data.get("strength", "moderate"),
                        supporting_studies=rec_data.get("studies", []),
                        effect_size=rec_data.get("effect_size", 0.5),
                        confidence_score=0.8,  # High confidence for guidelines
                        alternatives=rec_data.get("alternatives", []),
                        contraindications=rec_data.get("contraindications", []),
                        monitoring_requirements=rec_data.get("monitoring", []),
                        expected_timeline=rec_data.get("timeline", "8-12 weeks"),
                        cultural_adaptations=self._get_cultural_adaptations(
                            rec_data.get("text", ""), cultural_context
                        )
                    )
                    recommendations.append(recommendation)
        
        # Generate recommendations from efficacy data
        for efficacy in efficacy_data:
            if efficacy.efficacy_score > 0.6:  # Only include effective treatments
                recommendation = EvidenceBasedRecommendation(
                    recommendation_id=f"rec_efficacy_{efficacy.treatment_id}",
                    condition=condition,
                    severity=severity,
                    recommendation_text=f"Consider {efficacy.treatment_name} based on research evidence",
                    evidence_level="high" if efficacy.studies_count > 5 else "medium",
                    strength_of_recommendation="strong" if efficacy.efficacy_score > 0.8 else "moderate",
                    supporting_studies=[f"{efficacy.studies_count} studies, {efficacy.total_participants} participants"],
                    effect_size=efficacy.effect_size,
                    confidence_score=min(1.0, efficacy.efficacy_score + 0.1),
                    alternatives=[],
                    contraindications=efficacy.adverse_effects,
                    monitoring_requirements=[f"Monitor for {', '.join(efficacy.adverse_effects[:3])}"],
                    expected_timeline=f"{efficacy.duration_weeks} weeks",
                    cultural_adaptations=[]
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _matches_severity(self, rec_data: Dict[str, Any], severity: str) -> bool:
        """Check if recommendation matches severity level"""
        rec_severity = rec_data.get("severity", ["mild", "moderate", "severe"])
        return severity in rec_severity
    
    def _get_cultural_adaptations(self, recommendation: str, cultural_context: str) -> List[str]:
        """Get cultural adaptations for recommendation"""
        if not cultural_context:
            return []
        
        # Simple cultural adaptation logic
        adaptations = []
        if cultural_context in ["asian", "collectivist"]:
            adaptations.append("Consider family involvement in treatment")
        if cultural_context in ["hispanic", "latino"]:
            adaptations.append("Integrate cultural concepts of family and spirituality")
        
        return adaptations
    
    def _deduplicate_articles(self, articles: List[ResearchArticle]) -> List[ResearchArticle]:
        """Remove duplicate articles"""
        seen_dois = set()
        unique_articles = []
        
        for article in articles:
            if article.doi not in seen_dois:
                seen_dois.add(article.doi)
                unique_articles.append(article)
        
        return unique_articles
    
    def _rank_articles_by_relevance(self, articles: List[ResearchArticle], query: str) -> List[ResearchArticle]:
        """Rank articles by relevance to query"""
        query_words = set(query.lower().split())
        
        def relevance_score(article):
            title_words = set(article.title.lower().split())
            keyword_words = set([kw.lower() for kw in article.keywords])
            
            title_overlap = len(query_words.intersection(title_words))
            keyword_overlap = len(query_words.intersection(keyword_words))
            
            # Weight by evidence level and clinical relevance
            evidence_weight = {"high": 1.0, "medium": 0.7, "low": 0.5}.get(article.evidence_level, 0.5)
            
            return (title_overlap * 2 + keyword_overlap) * evidence_weight * article.clinical_relevance
        
        articles.sort(key=relevance_score, reverse=True)
        return articles
    
    def _generate_cache_key(self, query: str, study_types: List[str], date_range: int) -> str:
        """Generate cache key for search"""
        key_data = f"{query}_{study_types}_{date_range}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid"""
        timestamp = cached_result.get("timestamp")
        if not timestamp:
            return False
        
        age_seconds = (datetime.now() - timestamp).total_seconds()
        return age_seconds < self.cache_duration
    
    async def _update_guidelines_if_needed(self):
        """Update guidelines if they're outdated"""
        # Check if guidelines need updating
        for guideline in self.clinical_guidelines.values():
            days_old = (datetime.now() - guideline.last_updated).days
            if days_old > self.guideline_update_frequency:
                # Update guideline (simulated)
                guideline.last_updated = datetime.now()
    
    async def _search_treatment_efficacy(self, treatment: str, condition: str, population: str) -> Optional[TreatmentEfficacy]:
        """Search for treatment efficacy data"""
        # Simulated efficacy data search
        return TreatmentEfficacy(
            treatment_id=f"efficacy_{treatment}_{condition}",
            treatment_name=treatment,
            condition=condition,
            population=population,
            efficacy_score=0.75,
            effect_size=0.6,
            number_needed_to_treat=4,
            response_rate=0.70,
            remission_rate=0.45,
            dropout_rate=0.15,
            adverse_effects=["mild side effects", "transient symptoms"],
            duration_weeks=12,
            follow_up_data={"6_months": 0.65, "12_months": 0.60},
            studies_count=8,
            total_participants=1200,
            confidence_interval=(0.65, 0.85),
            last_updated=datetime.now()
        )
    
    def _analyze_evidence_support(self,
                                treatment: str,
                                condition: str,
                                efficacy_data: Optional[TreatmentEfficacy],
                                guidelines: List[ClinicalGuideline],
                                research: List[ResearchArticle]) -> Dict[str, Any]:
        """Analyze evidence support for treatment"""
        
        support = {
            "confidence": 0.5,
            "evidence_level": "medium",
            "supporting_factors": [],
            "concerns": []
        }
        
        # Analyze efficacy data
        if efficacy_data:
            if efficacy_data.efficacy_score > 0.7:
                support["supporting_factors"].append(f"High efficacy score: {efficacy_data.efficacy_score:.2f}")
                support["confidence"] += 0.2
            if efficacy_data.studies_count > 5:
                support["supporting_factors"].append(f"Multiple studies: {efficacy_data.studies_count}")
                support["confidence"] += 0.1
        
        # Analyze guidelines
        guideline_support = sum(1 for g in guidelines 
                              if any(treatment.lower() in rec.get("text", "").lower() 
                                   for rec in g.recommendations))
        if guideline_support > 0:
            support["supporting_factors"].append(f"Supported by {guideline_support} clinical guidelines")
            support["confidence"] += 0.2
        
        # Analyze research
        relevant_research = len([r for r in research if treatment.lower() in r.title.lower()])
        if relevant_research > 0:
            support["supporting_factors"].append(f"Recent research: {relevant_research} relevant studies")
            support["confidence"] += 0.1
        
        # Determine evidence level
        if support["confidence"] > 0.8:
            support["evidence_level"] = "high"
        elif support["confidence"] > 0.6:
            support["evidence_level"] = "medium"
        else:
            support["evidence_level"] = "low"
        
        support["confidence"] = min(1.0, support["confidence"])
        
        return support
    
    def _check_contraindications(self,
                               treatment: str,
                               patient_profile: Dict[str, Any],
                               guidelines: List[ClinicalGuideline]) -> List[str]:
        """Check for contraindications"""
        contraindications = []
        
        # Check age-related contraindications
        age = patient_profile.get("age", 30)
        if age < 18 and "child" not in treatment.lower():
            contraindications.append("Treatment not validated for pediatric population")
        
        # Check medical conditions
        medical_conditions = patient_profile.get("medical_conditions", [])
        if "heart_condition" in medical_conditions and "stimulant" in treatment.lower():
            contraindications.append("Caution with stimulants due to cardiac condition")
        
        # Check guideline contraindications
        for guideline in guidelines:
            contraindications.extend(guideline.contraindications)
        
        return contraindications
    
    async def _generate_validation_summary(self,
                                         treatment: str,
                                         condition: str,
                                         evidence_support: Dict[str, Any],
                                         contraindications: List[str]) -> str:
        """Generate validation summary"""
        try:
            prompt = f"""
            Generate a clinical validation summary for:
            Treatment: {treatment}
            Condition: {condition}
            Evidence Level: {evidence_support.get('evidence_level')}
            Confidence: {evidence_support.get('confidence'):.2f}
            Supporting Factors: {evidence_support.get('supporting_factors')}
            Contraindications: {contraindications}
            
            Provide a brief, professional summary of the evidence support.
            """
            
            summary = await self.llm.generate_response(prompt)
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating validation summary: {str(e)}")
            return f"Evidence support for {treatment} in {condition}: {evidence_support.get('evidence_level')} level"
    
    def _generate_treatment_recommendation(self, evidence_support: Dict[str, Any]) -> str:
        """Generate treatment recommendation based on evidence"""
        confidence = evidence_support.get("confidence", 0.5)
        
        if confidence > 0.8:
            return "Strongly recommended based on robust evidence"
        elif confidence > 0.6:
            return "Recommended with moderate confidence"
        elif confidence > 0.4:
            return "May be considered with caution"
        else:
            return "Insufficient evidence for recommendation"
    
    async def _suggest_alternatives(self,
                                  treatment: str,
                                  condition: str,
                                  evidence_support: Dict[str, Any]) -> List[str]:
        """Suggest alternative treatments"""
        # Get efficacy data for condition
        condition_efficacies = self.treatment_efficacies.get(condition, [])
        
        # Sort by efficacy and exclude current treatment
        alternatives = [
            e.treatment_name for e in condition_efficacies
            if e.treatment_name.lower() != treatment.lower() and e.efficacy_score > 0.6
        ]
        
        return alternatives[:3]  # Return top 3 alternatives
    
    # Storage methods
    
    async def _store_article_in_db(self, article: ResearchArticle):
        """Store research article in vector database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "research_article",
                "title": article.title,
                "authors": article.authors,
                "journal": article.journal,
                "abstract": article.abstract,
                "study_type": article.study_type,
                "evidence_level": article.evidence_level,
                "clinical_relevance": article.clinical_relevance,
                "publication_date": article.publication_date.isoformat()
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=article.article_id
            )
            
        except Exception as e:
            self.logger.error(f"Error storing article in database: {str(e)}")
    
    async def _store_recommendation(self, recommendation: EvidenceBasedRecommendation):
        """Store recommendation in database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "evidence_based_recommendation",
                "condition": recommendation.condition,
                "severity": recommendation.severity,
                "recommendation_text": recommendation.recommendation_text,
                "evidence_level": recommendation.evidence_level,
                "strength_of_recommendation": recommendation.strength_of_recommendation,
                "confidence_score": recommendation.confidence_score,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=recommendation.recommendation_id
            )
            
        except Exception as e:
            self.logger.error(f"Error storing recommendation: {str(e)}")
    
    async def _store_efficacy_in_db(self, efficacy: TreatmentEfficacy):
        """Store treatment efficacy data in database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "treatment_efficacy",
                "treatment_name": efficacy.treatment_name,
                "condition": efficacy.condition,
                "efficacy_score": efficacy.efficacy_score,
                "effect_size": efficacy.effect_size,
                "response_rate": efficacy.response_rate,
                "studies_count": efficacy.studies_count,
                "total_participants": efficacy.total_participants,
                "last_updated": efficacy.last_updated.isoformat()
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=efficacy.treatment_id
            )
            
        except Exception as e:
            self.logger.error(f"Error storing efficacy data: {str(e)}")
    
    def _load_research_data(self):
        """Load persisted research data using JSON (CWE-502 fix: replaced pickle)"""
        try:
            data_dir = "src/data/research"

            # Load articles (JSON format for security)
            articles_file = f"{data_dir}/articles.json"
            if os.path.exists(articles_file):
                with open(articles_file, "r", encoding="utf-8") as f:
                    self.research_articles = json.load(f, object_hook=_json_deserial)

            # Load guidelines (JSON format for security)
            guidelines_file = f"{data_dir}/guidelines.json"
            if os.path.exists(guidelines_file):
                with open(guidelines_file, "r", encoding="utf-8") as f:
                    self.clinical_guidelines = json.load(f, object_hook=_json_deserial)

            # Load efficacy data (JSON format for security)
            efficacy_file = f"{data_dir}/efficacy.json"
            if os.path.exists(efficacy_file):
                with open(efficacy_file, "r", encoding="utf-8") as f:
                    self.treatment_efficacies = json.load(f, object_hook=_json_deserial)

            self.logger.info("Successfully loaded research data")

        except Exception as e:
            self.logger.warning(f"Could not load research data, starting fresh: {str(e)}")
    
    def _persist_research_data(self):
        """Persist research data to disk using JSON (CWE-502 fix: replaced pickle)"""
        try:
            data_dir = "src/data/research"
            os.makedirs(data_dir, exist_ok=True)

            # Save articles as JSON
            with open(f"{data_dir}/articles.json", "w", encoding="utf-8") as f:
                json.dump(self.research_articles, f, default=_json_serial, indent=2)

            # Save guidelines as JSON
            with open(f"{data_dir}/guidelines.json", "w", encoding="utf-8") as f:
                json.dump(self.clinical_guidelines, f, default=_json_serial, indent=2)

            # Save efficacy data as JSON
            with open(f"{data_dir}/efficacy.json", "w", encoding="utf-8") as f:
                json.dump(dict(self.treatment_efficacies), f, default=_json_serial, indent=2)

        except Exception as e:
            self.logger.error(f"Error persisting research data: {str(e)}")
    
    async def _update_all_guidelines(self) -> int:
        """Update all clinical guidelines"""
        # Simulated guideline update
        return len(self.clinical_guidelines)
    
    async def _update_efficacy_data(self) -> int:
        """Update treatment efficacy data"""
        # Simulated efficacy data update
        return sum(len(efficacies) for efficacies in self.treatment_efficacies.values())
    
    def __del__(self):
        """Ensure data is persisted when object is destroyed"""
        try:
            self._persist_research_data()
        except (OSError, IOError, RuntimeError, TypeError, AttributeError):
            # Silently ignore persistence errors during destruction
            # Object may be in partially destroyed state
            pass