"""
Diagnosis Constants Module

Shared constants for mental health condition definitions, response templates,
and diagnostic thresholds used across all diagnosis modules.

This module provides a single source of truth for:
- CONDITION_DEFINITIONS: Mental health conditions with symptoms, indicators, correlations
- RESPONSE_TEMPLATES: Severity-based response templates for each condition
- SEVERITY_LEVELS: Standard severity level definitions
"""

from .condition_definitions import (
    CONDITION_DEFINITIONS,
    RESPONSE_TEMPLATES,
    SEVERITY_LEVELS,
    get_condition_names,
    get_symptoms_for_condition,
    get_severity_threshold,
)

__all__ = [
    'CONDITION_DEFINITIONS',
    'RESPONSE_TEMPLATES',
    'SEVERITY_LEVELS',
    'get_condition_names',
    'get_symptoms_for_condition',
    'get_severity_threshold',
]
