"""
Enhanced Diagnosis Module

This module provides comprehensive diagnostic capabilities including:
- Temporal analysis and symptom tracking
- Differential diagnosis with evidence-based reasoning
- Therapeutic friction and growth-oriented responses
- Cultural sensitivity and adaptation
- Adaptive learning and personalization
- Enhanced memory and insight storage
- Real-time research integration

Legacy compatibility maintained for existing integrations.
"""

# New enhanced systems (with error handling for missing modules)
try:
    from .temporal_analysis import TemporalAnalysisEngine, SymptomEntry, BehavioralPattern
except ImportError:
    TemporalAnalysisEngine = None
    SymptomEntry = None
    BehavioralPattern = None

try:
    from .differential_diagnosis import (
        DifferentialDiagnosisEngine, 
        DifferentialDiagnosis, 
        DiagnosticCriterion,
        ComorbidityAssessment
    )
except ImportError:
    DifferentialDiagnosisEngine = None
    DifferentialDiagnosis = None
    DiagnosticCriterion = None
    ComorbidityAssessment = None

try:
    from .therapeutic_friction import (
        TherapeuticFrictionEngine, 
        TherapeuticResponse, 
        GrowthMoment,
        UserReadinessProfile
    )
except ImportError:
    TherapeuticFrictionEngine = None
    TherapeuticResponse = None
    GrowthMoment = None
    UserReadinessProfile = None

try:
    from .cultural_sensitivity import (
        CulturalSensitivityEngine, 
        CulturalProfile, 
        CulturalAdaptation,
        CulturalIntervention
    )
except ImportError:
    CulturalSensitivityEngine = None
    CulturalProfile = None
    CulturalAdaptation = None
    CulturalIntervention = None

try:
    from .adaptive_learning import (
        AdaptiveLearningEngine, 
        InterventionOutcome, 
        UserProfile,
        LearningInsight
    )
except ImportError:
    AdaptiveLearningEngine = None
    InterventionOutcome = None
    UserProfile = None
    LearningInsight = None

try:
    from .enhanced_integrated_system import (
        EnhancedIntegratedDiagnosticSystem,
        ComprehensiveDiagnosticResult
    )
except ImportError:
    EnhancedIntegratedDiagnosticSystem = None
    ComprehensiveDiagnosticResult = None

# Legacy compatibility imports (existing modules)
try:
    from .integrated_diagnosis import DiagnosisModule
except ImportError:
    DiagnosisModule = None

try:
    from .enhanced_diagnosis import EnhancedDiagnosisModule
except ImportError:
    EnhancedDiagnosisModule = None

try:
    from .comprehensive_diagnosis import ComprehensiveDiagnosisModule
except ImportError:
    ComprehensiveDiagnosisModule = None

# Enhanced factory function with fallback
def create_diagnosis_module(use_agentic_rag=True, use_cache=True, **kwargs):
    """
    Factory function to create diagnosis module with enhanced capabilities
    
    Args:
        use_agentic_rag: Enable agentic RAG functionality
        use_cache: Enable vector caching
        **kwargs: Additional configuration options
        
    Returns:
        Best available diagnostic system
    """
    try:
        # Try new enhanced integrated system first
        if EnhancedIntegratedDiagnosticSystem is not None:
            return EnhancedIntegratedDiagnosticSystem(**kwargs)
    except Exception:
        pass
    
    try:
        # Fallback to comprehensive diagnosis module
        if ComprehensiveDiagnosisModule is not None:
            return ComprehensiveDiagnosisModule()
    except Exception:
        pass
    
    try:
        # Fallback to enhanced diagnosis module
        if EnhancedDiagnosisModule is not None:
            return EnhancedDiagnosisModule()
    except Exception:
        pass
    
    try:
        # Ultimate fallback to basic diagnosis module
        if DiagnosisModule is not None:
            return DiagnosisModule()
    except Exception:
        pass
    
    # If all else fails, return None and let caller handle
    raise ImportError("No diagnosis modules could be loaded")

__all__ = [
    # Factory function (always available)
    'create_diagnosis_module',
    
    # Enhanced systems (may be None if import failed)
    'EnhancedIntegratedDiagnosticSystem',
    'ComprehensiveDiagnosticResult',
    'TemporalAnalysisEngine',
    'DifferentialDiagnosisEngine', 
    'TherapeuticFrictionEngine',
    'CulturalSensitivityEngine',
    'AdaptiveLearningEngine',
    
    # Data structures (may be None if import failed)
    'SymptomEntry',
    'BehavioralPattern',
    'DifferentialDiagnosis',
    'DiagnosticCriterion',
    'ComorbidityAssessment',
    'TherapeuticResponse',
    'GrowthMoment',
    'UserReadinessProfile',
    'CulturalProfile',
    'CulturalAdaptation',
    'CulturalIntervention',
    'InterventionOutcome',
    'UserProfile',
    'LearningInsight',
    
    # Legacy modules (may be None if import failed)
    'DiagnosisModule',
    'EnhancedDiagnosisModule',
    'ComprehensiveDiagnosisModule'
]