"""Clinical agents module - Diagnosis and therapy agents"""

from .diagnosis_agent import EnhancedDiagnosisAgent
from .therapy_agent import TherapyAgent

# Alias for backward compatibility
DiagnosisAgent = EnhancedDiagnosisAgent

__all__ = ['DiagnosisAgent', 'EnhancedDiagnosisAgent', 'TherapyAgent']
