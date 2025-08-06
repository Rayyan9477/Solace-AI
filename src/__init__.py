"""
Solace-AI: Advanced Mental Health AI Platform

This module provides the main entry points and imports for the Solace-AI platform,
organized into modular components for better maintainability and scalability.
"""

# Core modules
try:
    from .agents import therapy_agent, diagnosis_agent
except ImportError:
    therapy_agent = None
    diagnosis_agent = None

try:
    from .analysis import emotion_analysis, conversation_analysis
except ImportError:
    emotion_analysis = None
    conversation_analysis = None

try:
    from .database import vector_store, conversation_tracker
except ImportError:
    vector_store = None
    conversation_tracker = None

try:
    from .utils import logger, helpers, error_handling
except ImportError:
    logger = None
    helpers = None
    error_handling = None

# Machine Learning models
try:
    from .ml_models import (
        BaseModel, ModelRegistry, 
        BayesianDiagnosticLayer, UncertaintyQuantifier,
        MultiModalAttention, AdaptiveFusion
    )
except ImportError:
    BaseModel = None
    ModelRegistry = None
    BayesianDiagnosticLayer = None
    UncertaintyQuantifier = None
    MultiModalAttention = None
    AdaptiveFusion = None

# Feature extractors
try:
    from .feature_extractors import (
        TextFeatureExtractor, SemanticAnalyzer, SentimentExtractor,
        VoiceFeatureExtractor, BehavioralAnalyzer,
        TemporalAnalyzer, ContextualAnalyzer,
        MultiModalFeatureFusion
    )
except ImportError:
    TextFeatureExtractor = None
    SemanticAnalyzer = None
    SentimentExtractor = None
    VoiceFeatureExtractor = None
    BehavioralAnalyzer = None
    TemporalAnalyzer = None
    ContextualAnalyzer = None
    MultiModalFeatureFusion = None

# Clinical decision support
try:
    from .clinical_decision_support import (
        ClinicalRuleEngine, DiagnosticAlgorithm,
        TreatmentRecommendationEngine, RiskAssessmentEngine,
        ClinicalGuidelinesManager, ClinicalAlertSystem
    )
except ImportError:
    ClinicalRuleEngine = None
    DiagnosticAlgorithm = None
    TreatmentRecommendationEngine = None
    RiskAssessmentEngine = None
    ClinicalGuidelinesManager = None
    ClinicalAlertSystem = None

# Diagnosis system
try:
    from .diagnosis import (
        comprehensive_diagnosis,
        enterprise_multimodal_pipeline,
        model_management
    )
except ImportError:
    comprehensive_diagnosis = None
    enterprise_multimodal_pipeline = None
    model_management = None

# Configuration and settings
try:
    from .config import settings
except ImportError:
    settings = None

__version__ = "2.0.0"
__author__ = "Solace-AI Development Team"
__description__ = "Advanced AI-powered mental health platform with enterprise-grade features"

def get_version_info():
    """Get detailed version information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "ml_models": "Neural network models for diagnosis and prediction",
            "feature_extractors": "Multi-modal feature extraction pipeline", 
            "clinical_decision_support": "Evidence-based clinical decision tools",
            "diagnosis": "Comprehensive diagnostic assessment system",
            "agents": "Conversational AI agents for therapy and diagnosis",
            "analysis": "Emotion and conversation analysis tools",
            "database": "Vector storage and conversation tracking",
            "utils": "Utility functions and error handling"
        }
    }

def get_system_status():
    """Get system status and health check"""
    try:
        status = {
            "status": "healthy",
            "version": __version__,
            "components": {}
        }
        
        # Check core components
        status["components"]["logger"] = "available" if logger else "unavailable"
        status["components"]["ml_models"] = "available" if BaseModel else "unavailable"  
        status["components"]["vector_store"] = "available" if vector_store else "unavailable"
        status["components"]["feature_extractors"] = "available" if TextFeatureExtractor else "unavailable"
        status["components"]["clinical_decision_support"] = "available" if ClinicalRuleEngine else "unavailable"
        
        return status
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": __version__
        }