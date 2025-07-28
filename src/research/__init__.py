"""
Research Integration Module

This module provides real-time research integration for evidence-based
mental health practice, including clinical guidelines and treatment efficacy data.
"""

try:
    from .real_time_research import (
        RealTimeResearchEngine,
        ResearchArticle,
        ClinicalGuideline,
        TreatmentEfficacy,
        EvidenceBasedRecommendation
    )
except ImportError:
    RealTimeResearchEngine = None
    ResearchArticle = None
    ClinicalGuideline = None
    TreatmentEfficacy = None
    EvidenceBasedRecommendation = None

__all__ = [
    'RealTimeResearchEngine',
    'ResearchArticle',
    'ClinicalGuideline',
    'TreatmentEfficacy',
    'EvidenceBasedRecommendation'
]