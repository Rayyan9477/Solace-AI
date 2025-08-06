"""
Real-time Medical Literature Monitoring and Integration System
Continuously monitors and integrates latest mental health research
"""

import asyncio
import aiohttp
import feedparser
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import re
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class ResearchSource(Enum):
    PUBMED = "pubmed"
    PSYCINFO = "psycinfo"
    COCHRANE = "cochrane"
    ARXIV = "arxiv"
    CLINICAL_TRIALS = "clinicaltrials"
    NATURE = "nature"
    SCIENCE = "science"
    JAMA = "jama"
    NEJM = "nejm"


class StudyType(Enum):
    RCT = "randomized_controlled_trial"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL = "case_control"
    CLINICAL_TRIAL = "clinical_trial"
    OBSERVATIONAL = "observational"
    THEORETICAL = "theoretical"


class EvidenceLevel(Enum):
    LEVEL_1 = 1  # Systematic reviews and meta-analyses
    LEVEL_2 = 2  # Individual RCTs
    LEVEL_3 = 3  # Controlled trials without randomization
    LEVEL_4 = 4  # Case-control and cohort studies
    LEVEL_5 = 5  # Case series and case reports
    LEVEL_6 = 6  # Expert opinion


@dataclass
class ResearchPaper:
    """Research paper metadata and content"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    journal: str
    publication_date: datetime
    doi: Optional[str]
    pmid: Optional[str]
    url: str
    study_type: StudyType
    evidence_level: EvidenceLevel
    keywords: List[str]
    mental_health_conditions: List[str]
    therapeutic_interventions: List[str]
    outcome_measures: List[str]
    sample_size: Optional[int]
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    p_value: Optional[float]
    source: ResearchSource
    relevance_score: float = 0.0
    embedding: Optional[np.ndarray] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    clinical_recommendations: List[str] = field(default_factory=list)
    methodology_quality: Optional[float] = None  # 0-1 score
    bias_assessment: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResearchAlert:
    """Alert for significant research findings"""
    alert_id: str
    title: str
    description: str
    papers: List[ResearchPaper]
    alert_type: str  # breakthrough, contradiction, new_treatment, etc.
    confidence_level: float
    clinical_impact: str  # high, medium, low
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed: bool = False


class PubMedMonitor:
    """Monitor PubMed for new mental health research"""
    
    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_terms = [
            "mental health therapy",
            "cognitive behavioral therapy",
            "depression treatment",
            "anxiety treatment",
            "psychotherapy effectiveness",
            "digital mental health",
            "teletherapy",
            "mental health outcomes",
            "psychological intervention",
            "mindfulness therapy"
        ]
        
    async def search_recent_papers(self, days_back: int = 7) -> List[ResearchPaper]:
        """Search for recent papers in mental health"""
        papers = []
        
        for search_term in self.search_terms:
            try:
                # Search for paper IDs
                search_params = {
                    'db': 'pubmed',
                    'term': f'{search_term} AND ("last {days_back} days"[PDat])',
                    'retmax': 50,
                    'sort': 'relevance',
                    'tool': 'solace_ai',
                    'email': self.email
                }
                
                search_url = f"{self.base_url}esearch.fcgi?" + urlencode(search_params)
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url) as response:
                        if response.status == 200:
                            search_results = await response.text()
                            pmids = self._extract_pmids(search_results)
                            
                            # Fetch detailed information for each paper
                            for pmid in pmids[:10]:  # Limit to 10 per search term
                                paper = await self._fetch_paper_details(pmid)
                                if paper:
                                    papers.append(paper)
                                    
                await asyncio.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error searching PubMed for '{search_term}': {e}")
                
        return papers
        
    async def _fetch_paper_details(self, pmid: str) -> Optional[ResearchPaper]:
        """Fetch detailed information for a specific PMID"""
        try:
            fetch_params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml',
                'tool': 'solace_ai',
                'email': self.email
            }
            
            fetch_url = f"{self.base_url}efetch.fcgi?" + urlencode(fetch_params)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        return self._parse_pubmed_xml(xml_content, pmid)
                        
        except Exception as e:
            logger.error(f"Error fetching details for PMID {pmid}: {e}")
            
        return None
        
    def _extract_pmids(self, search_results: str) -> List[str]:
        """Extract PMIDs from search results XML"""
        try:
            root = ET.fromstring(search_results)
            pmids = []
            
            for id_elem in root.findall('.//Id'):
                pmids.append(id_elem.text)
                
            return pmids
        except Exception as e:
            logger.error(f"Error extracting PMIDs: {e}")
            return []
            
    def _parse_pubmed_xml(self, xml_content: str, pmid: str) -> Optional[ResearchPaper]:
        """Parse PubMed XML response into ResearchPaper object"""
        try:
            root = ET.fromstring(xml_content)
            article = root.find('.//PubmedArticle')
            
            if article is None:
                return None
                
            # Extract basic information
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            abstract_elem = article.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Authors
            authors = []
            for author in article.findall('.//Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
                    
            # Journal
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown"
            
            # Publication date
            pub_date = self._extract_publication_date(article)
            
            # DOI
            doi_elem = article.find('.//ArticleId[@IdType="doi"]')
            doi = doi_elem.text if doi_elem is not None else None
            
            # Keywords and MeSH terms
            keywords = self._extract_keywords(article)
            
            # Determine study type and evidence level
            study_type, evidence_level = self._classify_study(title, abstract, keywords)
            
            # Extract mental health conditions and interventions
            conditions = self._extract_conditions(title, abstract, keywords)
            interventions = self._extract_interventions(title, abstract, keywords)
            
            return ResearchPaper(
                paper_id=f"pmid_{pmid}",
                title=title,
                authors=authors,
                abstract=abstract,
                journal=journal,
                publication_date=pub_date,
                doi=doi,
                pmid=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                study_type=study_type,
                evidence_level=evidence_level,
                keywords=keywords,
                mental_health_conditions=conditions,
                therapeutic_interventions=interventions,
                outcome_measures=[],
                source=ResearchSource.PUBMED
            )
            
        except Exception as e:
            logger.error(f"Error parsing PubMed XML for PMID {pmid}: {e}")
            return None
            
    def _extract_publication_date(self, article) -> datetime:
        """Extract publication date from XML"""
        try:
            pub_date = article.find('.//PubDate')
            if pub_date is not None:
                year = pub_date.find('Year')
                month = pub_date.find('Month')
                day = pub_date.find('Day')
                
                year_val = int(year.text) if year is not None else datetime.now().year
                month_val = self._month_to_int(month.text) if month is not None else 1
                day_val = int(day.text) if day is not None else 1
                
                return datetime(year_val, month_val, day_val)
        except:
            pass
            
        return datetime.now()
        
    def _month_to_int(self, month_str: str) -> int:
        """Convert month string to integer"""
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        return month_map.get(month_str, 1)
        
    def _extract_keywords(self, article) -> List[str]:
        """Extract keywords and MeSH terms"""
        keywords = []
        
        # MeSH terms
        for mesh in article.findall('.//MeshHeading/DescriptorName'):
            if mesh.text:
                keywords.append(mesh.text.lower())
                
        # Author keywords
        for keyword in article.findall('.//Keyword'):
            if keyword.text:
                keywords.append(keyword.text.lower())
                
        return list(set(keywords))
        
    def _classify_study(self, title: str, abstract: str, keywords: List[str]) -> Tuple[StudyType, EvidenceLevel]:
        """Classify study type and evidence level"""
        text = (title + " " + abstract).lower()
        
        # Check for systematic review or meta-analysis
        if any(term in text for term in ['systematic review', 'meta-analysis', 'meta analysis']):
            return StudyType.SYSTEMATIC_REVIEW, EvidenceLevel.LEVEL_1
            
        # Check for RCT
        if any(term in text for term in ['randomized controlled trial', 'randomized trial', 'rct']):
            return StudyType.RCT, EvidenceLevel.LEVEL_2
            
        # Check for clinical trial
        if 'clinical trial' in text:
            return StudyType.CLINICAL_TRIAL, EvidenceLevel.LEVEL_3
            
        # Check for cohort study
        if any(term in text for term in ['cohort study', 'longitudinal study', 'prospective study']):
            return StudyType.COHORT_STUDY, EvidenceLevel.LEVEL_4
            
        # Check for case-control
        if 'case-control' in text or 'case control' in text:
            return StudyType.CASE_CONTROL, EvidenceLevel.LEVEL_4
            
        # Default to observational
        return StudyType.OBSERVATIONAL, EvidenceLevel.LEVEL_5
        
    def _extract_conditions(self, title: str, abstract: str, keywords: List[str]) -> List[str]:
        """Extract mental health conditions mentioned"""
        text = (title + " " + abstract).lower()
        
        conditions = []
        condition_terms = {
            'depression': ['depression', 'depressive', 'major depressive disorder', 'mdd'],
            'anxiety': ['anxiety', 'anxiety disorder', 'generalized anxiety', 'gad', 'panic'],
            'ptsd': ['ptsd', 'post-traumatic stress', 'trauma'],
            'bipolar': ['bipolar', 'manic', 'mania'],
            'schizophrenia': ['schizophrenia', 'psychosis', 'psychotic'],
            'adhd': ['adhd', 'attention deficit', 'hyperactivity'],
            'ocd': ['ocd', 'obsessive-compulsive'],
            'eating_disorders': ['anorexia', 'bulimia', 'eating disorder'],
            'substance_abuse': ['substance abuse', 'addiction', 'substance use disorder']
        }
        
        for condition, terms in condition_terms.items():
            if any(term in text for term in terms):
                conditions.append(condition)
                
        return conditions
        
    def _extract_interventions(self, title: str, abstract: str, keywords: List[str]) -> List[str]:
        """Extract therapeutic interventions mentioned"""
        text = (title + " " + abstract).lower()
        
        interventions = []
        intervention_terms = {
            'cbt': ['cognitive behavioral therapy', 'cbt', 'cognitive behaviour therapy'],
            'dbt': ['dialectical behavior therapy', 'dbt', 'dialectical behaviour therapy'],
            'act': ['acceptance and commitment therapy', 'act'],
            'mindfulness': ['mindfulness', 'meditation', 'mindfulness-based'],
            'psychotherapy': ['psychotherapy', 'therapy', 'counseling', 'counselling'],
            'medication': ['medication', 'pharmacotherapy', 'antidepressant', 'antipsychotic'],
            'emdr': ['emdr', 'eye movement desensitization'],
            'group_therapy': ['group therapy', 'group treatment'],
            'telehealth': ['telehealth', 'teletherapy', 'digital therapy', 'online therapy']
        }
        
        for intervention, terms in intervention_terms.items():
            if any(term in text for term in terms):
                interventions.append(intervention)
                
        return interventions


class ClinicalTrialsMonitor:
    """Monitor ClinicalTrials.gov for new mental health trials"""
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/query/"
        
    async def search_recent_trials(self, days_back: int = 7) -> List[ResearchPaper]:
        """Search for recent clinical trials"""
        trials = []
        
        search_terms = [
            "mental health",
            "depression",
            "anxiety",
            "psychotherapy",
            "cognitive behavioral therapy"
        ]
        
        for term in search_terms:
            try:
                params = {
                    'expr': term,
                    'min_rnk': 1,
                    'max_rnk': 20,
                    'fmt': 'json'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url + "study_fields", params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for study in data.get('StudyFieldsResponse', {}).get('StudyFields', []):
                                trial = self._parse_clinical_trial(study)
                                if trial:
                                    trials.append(trial)
                                    
                await asyncio.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error searching clinical trials for '{term}': {e}")
                
        return trials
        
    def _parse_clinical_trial(self, study_data: Dict) -> Optional[ResearchPaper]:
        """Parse clinical trial data into ResearchPaper object"""
        try:
            nct_id = study_data.get('NCTId', [''])[0]
            title = study_data.get('BriefTitle', [''])[0]
            summary = study_data.get('BriefSummary', [''])[0]
            
            # Extract conditions and interventions
            conditions = study_data.get('Condition', [])
            interventions = study_data.get('InterventionName', [])
            
            # Create paper object
            return ResearchPaper(
                paper_id=f"nct_{nct_id}",
                title=title,
                authors=[],  # Not typically available in clinical trials API
                abstract=summary,
                journal="ClinicalTrials.gov",
                publication_date=datetime.now(),  # Use current date as placeholder
                doi=None,
                pmid=None,
                url=f"https://clinicaltrials.gov/ct2/show/{nct_id}",
                study_type=StudyType.CLINICAL_TRIAL,
                evidence_level=EvidenceLevel.LEVEL_2,
                keywords=[],
                mental_health_conditions=[c.lower() for c in conditions],
                therapeutic_interventions=[i.lower() for i in interventions],
                outcome_measures=[],
                source=ResearchSource.CLINICAL_TRIALS
            )
            
        except Exception as e:
            logger.error(f"Error parsing clinical trial data: {e}")
            return None


class ResearchSynthesizer:
    """Synthesize and analyze research findings"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def analyze_papers(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze a collection of research papers"""
        if not papers:
            return {"analysis": "No papers to analyze"}
            
        # Generate embeddings for papers
        for paper in papers:
            if paper.embedding is None:
                text = f"{paper.title} {paper.abstract}"
                paper.embedding = self.embedding_model.encode(text)
                
        # Cluster similar papers
        clusters = await self._cluster_papers(papers)
        
        # Analyze trends
        trends = await self._analyze_trends(papers)
        
        # Identify contradictions
        contradictions = await self._identify_contradictions(papers)
        
        # Extract key findings
        key_findings = await self._extract_key_findings(papers)
        
        return {
            "total_papers": len(papers),
            "paper_clusters": clusters,
            "research_trends": trends,
            "contradictions": contradictions,
            "key_findings": key_findings,
            "evidence_distribution": self._analyze_evidence_levels(papers),
            "condition_coverage": self._analyze_condition_coverage(papers),
            "intervention_effectiveness": await self._analyze_intervention_effectiveness(papers)
        }
        
    async def _cluster_papers(self, papers: List[ResearchPaper]) -> Dict[str, List[str]]:
        """Cluster papers by similarity"""
        if len(papers) < 2:
            return {"single_cluster": [p.paper_id for p in papers]}
            
        # Simple clustering based on conditions and interventions
        clusters = {}
        
        for paper in papers:
            # Create cluster key based on primary condition and intervention
            primary_condition = paper.mental_health_conditions[0] if paper.mental_health_conditions else "general"
            primary_intervention = paper.therapeutic_interventions[0] if paper.therapeutic_interventions else "general"
            
            cluster_key = f"{primary_condition}_{primary_intervention}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(paper.paper_id)
            
        return clusters
        
    async def _analyze_trends(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze research trends"""
        # Analyze by publication date
        recent_papers = [p for p in papers if (datetime.now() - p.publication_date).days <= 365]
        
        # Count conditions and interventions
        condition_counts = {}
        intervention_counts = {}
        
        for paper in recent_papers:
            for condition in paper.mental_health_conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            for intervention in paper.therapeutic_interventions:
                intervention_counts[intervention] = intervention_counts.get(intervention, 0) + 1
                
        return {
            "emerging_conditions": sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "trending_interventions": sorted(intervention_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "research_velocity": len(recent_papers),
            "high_evidence_studies": len([p for p in recent_papers if p.evidence_level.value <= 2])
        }
        
    async def _identify_contradictions(self, papers: List[ResearchPaper]) -> List[Dict[str, Any]]:
        """Identify contradictory findings"""
        contradictions = []
        
        # Group papers by condition and intervention
        groups = {}
        for paper in papers:
            for condition in paper.mental_health_conditions:
                for intervention in paper.therapeutic_interventions:
                    key = f"{condition}_{intervention}"
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(paper)
                    
        # Look for contradictory outcomes within groups
        for key, group_papers in groups.items():
            if len(group_papers) >= 2:
                # Simple contradiction detection based on keywords
                positive_papers = []
                negative_papers = []
                
                for paper in group_papers:
                    text = (paper.title + " " + paper.abstract).lower()
                    if any(word in text for word in ['effective', 'significant improvement', 'reduced symptoms']):
                        positive_papers.append(paper)
                    elif any(word in text for word in ['ineffective', 'no significant', 'no improvement']):
                        negative_papers.append(paper)
                        
                if positive_papers and negative_papers:
                    contradictions.append({
                        "topic": key,
                        "positive_studies": [p.paper_id for p in positive_papers],
                        "negative_studies": [p.paper_id for p in negative_papers],
                        "confidence": 0.7  # Simple confidence score
                    })
                    
        return contradictions
        
    async def _extract_key_findings(self, papers: List[ResearchPaper]) -> List[Dict[str, Any]]:
        """Extract key findings from high-quality papers"""
        high_quality_papers = [p for p in papers if p.evidence_level.value <= 2]
        
        findings = []
        for paper in high_quality_papers[:10]:  # Top 10 high-quality papers
            # Simple finding extraction (in production, use NLP)
            abstract = paper.abstract.lower()
            
            finding = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "key_finding": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                "evidence_level": paper.evidence_level.value,
                "conditions": paper.mental_health_conditions,
                "interventions": paper.therapeutic_interventions,
                "clinical_relevance": "high" if paper.evidence_level.value <= 2 else "medium"
            }
            findings.append(finding)
            
        return findings
        
    def _analyze_evidence_levels(self, papers: List[ResearchPaper]) -> Dict[str, int]:
        """Analyze distribution of evidence levels"""
        level_counts = {}
        for paper in papers:
            level = f"Level_{paper.evidence_level.value}"
            level_counts[level] = level_counts.get(level, 0) + 1
        return level_counts
        
    def _analyze_condition_coverage(self, papers: List[ResearchPaper]) -> Dict[str, int]:
        """Analyze coverage of different mental health conditions"""
        condition_counts = {}
        for paper in papers:
            for condition in paper.mental_health_conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
        return condition_counts
        
    async def _analyze_intervention_effectiveness(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze effectiveness of different interventions"""
        intervention_analysis = {}
        
        for paper in papers:
            for intervention in paper.therapeutic_interventions:
                if intervention not in intervention_analysis:
                    intervention_analysis[intervention] = {
                        "total_studies": 0,
                        "high_quality_studies": 0,
                        "positive_outcomes": 0,
                        "mixed_outcomes": 0,
                        "negative_outcomes": 0
                    }
                    
                analysis = intervention_analysis[intervention]
                analysis["total_studies"] += 1
                
                if paper.evidence_level.value <= 2:
                    analysis["high_quality_studies"] += 1
                    
                # Simple outcome classification based on abstract
                abstract = paper.abstract.lower()
                if any(word in abstract for word in ['effective', 'significant improvement', 'reduced']):
                    analysis["positive_outcomes"] += 1
                elif any(word in abstract for word in ['mixed', 'partial', 'modest']):
                    analysis["mixed_outcomes"] += 1
                elif any(word in abstract for word in ['ineffective', 'no significant', 'no difference']):
                    analysis["negative_outcomes"] += 1
                    
        return intervention_analysis


class LiteratureMonitor:
    """Main literature monitoring system"""
    
    def __init__(self, email: str, update_interval: int = 3600):  # 1 hour default
        self.email = email
        self.update_interval = update_interval
        self.pubmed_monitor = PubMedMonitor(email)
        self.clinical_trials_monitor = ClinicalTrialsMonitor()
        self.research_synthesizer = ResearchSynthesizer()
        self.papers_cache: List[ResearchPaper] = []
        self.last_update: Optional[datetime] = None
        self.alerts: List[ResearchAlert] = []
        self._running = False
        
    async def start_monitoring(self):
        """Start continuous literature monitoring"""
        if self._running:
            return
            
        self._running = True
        logger.info("Started literature monitoring")
        
        while self._running:
            try:
                await self.update_literature()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in literature monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
                
    async def stop_monitoring(self):
        """Stop literature monitoring"""
        self._running = False
        logger.info("Stopped literature monitoring")
        
    async def update_literature(self) -> Dict[str, Any]:
        """Update literature database with new papers"""
        logger.info("Updating literature database...")
        
        new_papers = []
        
        # Get papers from PubMed
        try:
            pubmed_papers = await self.pubmed_monitor.search_recent_papers(days_back=1)
            new_papers.extend(pubmed_papers)
            logger.info(f"Found {len(pubmed_papers)} new PubMed papers")
        except Exception as e:
            logger.error(f"Error fetching PubMed papers: {e}")
            
        # Get clinical trials
        try:
            clinical_trials = await self.clinical_trials_monitor.search_recent_trials(days_back=7)
            new_papers.extend(clinical_trials)
            logger.info(f"Found {len(clinical_trials)} new clinical trials")
        except Exception as e:
            logger.error(f"Error fetching clinical trials: {e}")
            
        # Filter out duplicates
        existing_ids = {paper.paper_id for paper in self.papers_cache}
        unique_papers = [paper for paper in new_papers if paper.paper_id not in existing_ids]
        
        # Add to cache
        self.papers_cache.extend(unique_papers)
        
        # Keep only recent papers (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.papers_cache = [p for p in self.papers_cache if p.publication_date > cutoff_date]
        
        # Analyze papers and generate alerts
        if unique_papers:
            analysis = await self.research_synthesizer.analyze_papers(unique_papers)
            await self._generate_alerts(unique_papers, analysis)
            
        self.last_update = datetime.now()
        
        return {
            "new_papers_found": len(unique_papers),
            "total_papers_cached": len(self.papers_cache),
            "new_alerts_generated": len([a for a in self.alerts if not a.reviewed]),
            "last_update": self.last_update.isoformat()
        }
        
    async def get_latest_research(self, condition: str = None, 
                                intervention: str = None, 
                                evidence_level: int = None,
                                limit: int = 20) -> List[ResearchPaper]:
        """Get latest research with optional filters"""
        filtered_papers = self.papers_cache
        
        if condition:
            filtered_papers = [p for p in filtered_papers if condition.lower() in p.mental_health_conditions]
            
        if intervention:
            filtered_papers = [p for p in filtered_papers if intervention.lower() in p.therapeutic_interventions]
            
        if evidence_level:
            filtered_papers = [p for p in filtered_papers if p.evidence_level.value <= evidence_level]
            
        # Sort by publication date
        filtered_papers.sort(key=lambda p: p.publication_date, reverse=True)
        
        return filtered_papers[:limit]
        
    async def get_research_synthesis(self, topic: str) -> Dict[str, Any]:
        """Get comprehensive synthesis on a specific topic"""
        relevant_papers = []
        
        for paper in self.papers_cache:
            text = (paper.title + " " + paper.abstract).lower()
            if topic.lower() in text:
                relevant_papers.append(paper)
                
        if not relevant_papers:
            return {"error": f"No research found for topic: {topic}"}
            
        analysis = await self.research_synthesizer.analyze_papers(relevant_papers)
        
        return {
            "topic": topic,
            "analysis": analysis,
            "paper_count": len(relevant_papers),
            "latest_papers": [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "journal": p.journal,
                    "publication_date": p.publication_date.isoformat(),
                    "url": p.url,
                    "evidence_level": p.evidence_level.value
                }
                for p in sorted(relevant_papers, key=lambda x: x.publication_date, reverse=True)[:5]
            ]
        }
        
    async def _generate_alerts(self, papers: List[ResearchPaper], analysis: Dict[str, Any]):
        """Generate research alerts based on new findings"""
        # Generate breakthrough alert
        high_quality_papers = [p for p in papers if p.evidence_level.value <= 2]
        
        if high_quality_papers:
            alert = ResearchAlert(
                alert_id=str(hash(str([p.paper_id for p in high_quality_papers]))),
                title="New High-Quality Research Available",
                description=f"Found {len(high_quality_papers)} new high-quality studies",
                papers=high_quality_papers,
                alert_type="high_quality_research",
                confidence_level=0.9,
                clinical_impact="high"
            )
            self.alerts.append(alert)
            
        # Generate contradiction alert
        contradictions = analysis.get("contradictions", [])
        if contradictions:
            for contradiction in contradictions:
                alert = ResearchAlert(
                    alert_id=str(hash(str(contradiction))),
                    title=f"Contradictory Findings: {contradiction['topic']}",
                    description=f"Found contradictory results for {contradiction['topic']}",
                    papers=[p for p in papers if p.paper_id in contradiction['positive_studies'] + contradiction['negative_studies']],
                    alert_type="contradiction",
                    confidence_level=contradiction['confidence'],
                    clinical_impact="medium"
                )
                self.alerts.append(alert)
                
    async def get_unreviewed_alerts(self) -> List[ResearchAlert]:
        """Get alerts that haven't been reviewed"""
        return [alert for alert in self.alerts if not alert.reviewed]
        
    async def mark_alert_reviewed(self, alert_id: str):
        """Mark an alert as reviewed"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.reviewed = True
                break