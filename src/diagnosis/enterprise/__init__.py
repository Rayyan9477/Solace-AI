"""
Enterprise Multi-Modal Diagnostic Pipeline

A comprehensive, modular enterprise-level diagnostic system for mental health assessment.

This package provides:
- Multi-modal fusion with transformer attention
- Bayesian uncertainty quantification  
- Clinical decision support (DSM-5/ICD-11 compliant)
- Real-time adaptation and personalization
- Temporal sequence modeling
- HIPAA compliance and audit logging
- A/B testing and performance monitoring

Author: Solace-AI Development Team
Version: 1.0.0
"""

from .core import EnterpriseMultiModalPipeline, IntegratedDiagnosticSystem
from .models import *
from .clinical import *
from .feature_extraction import *
from .config import *
from .utils import *

__version__ = "1.0.0"
__author__ = "Solace-AI Development Team"

# Export main classes and functions
__all__ = [
    'EnterpriseMultiModalPipeline',
    'IntegratedDiagnosticSystem',
    'create_enterprise_pipeline',
    'ModalityType',
    'ClinicalSeverity',
    'ConfidenceLevel'
]

def create_enterprise_pipeline(config=None, **kwargs):
    """Factory function to create enterprise pipeline"""
    from .core import EnterpriseMultiModalPipeline
    return EnterpriseMultiModalPipeline(config=config, **kwargs)