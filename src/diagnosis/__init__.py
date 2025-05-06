"""
Diagnosis module initialization.

This package provides tools for mental health assessment by integrating multiple data sources:
1. Voice emotion analysis
2. Conversational AI
3. Personality test results
"""

from .integrated_diagnosis import DiagnosisModule
from .enhanced_diagnosis import EnhancedDiagnosisModule
from .comprehensive_diagnosis import ComprehensiveDiagnosisModule, create_diagnosis_module

__all__ = ['DiagnosisModule', 'EnhancedDiagnosisModule', 'ComprehensiveDiagnosisModule', 'create_diagnosis_module']