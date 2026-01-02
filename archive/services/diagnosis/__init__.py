"""
Diagnosis Services Package

This package provides the unified diagnosis service architecture for Solace-AI,
including service interfaces, implementations, orchestration, and agent adapters.
"""

from .interfaces import (
    IDiagnosisService,
    IEnhancedDiagnosisService,
    IMemoryIntegrationService,
    IVectorDatabaseIntegrationService,
    IDiagnosisOrchestrator,
    IDiagnosisAgentAdapter,
    DiagnosisRequest,
    DiagnosisResult,
    DiagnosisType,
    ConfidenceLevel
)

from .unified_service import UnifiedDiagnosisService
from .orchestrator import DiagnosisOrchestrator
from .agent_adapter import DiagnosisAgentAdapter

__all__ = [
    # Interfaces
    "IDiagnosisService",
    "IEnhancedDiagnosisService", 
    "IMemoryIntegrationService",
    "IVectorDatabaseIntegrationService",
    "IDiagnosisOrchestrator",
    "IDiagnosisAgentAdapter",
    
    # Data classes
    "DiagnosisRequest",
    "DiagnosisResult",
    "DiagnosisType",
    "ConfidenceLevel",
    
    # Implementations
    "UnifiedDiagnosisService",
    "DiagnosisOrchestrator", 
    "DiagnosisAgentAdapter"
]