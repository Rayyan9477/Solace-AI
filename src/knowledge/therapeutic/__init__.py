"""
Therapeutic knowledge base for mental health support chatbot.

This module provides a knowledge base of evidence-based therapeutic techniques
from various approaches including CBT, DBT, ACT, and mindfulness practices.
"""

from src.knowledge.therapeutic.knowledge_base import TherapeuticKnowledgeBase
from src.knowledge.therapeutic.technique_service import TherapeuticTechniqueService

__all__ = ['TherapeuticKnowledgeBase', 'TherapeuticTechniqueService']