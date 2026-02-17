"""Solace-AI Diagnosis Service - AMIE-inspired 4-step Chain-of-Reasoning diagnostic assessment."""
from __future__ import annotations

__all__ = [
    "DiagnosisPhase", "SeverityLevel", "ConfidenceLevel", "ReasoningStep", "SymptomType",
    "SymptomDTO", "HypothesisDTO", "DifferentialDTO", "ReasoningStepResultDTO",
    "AssessmentRequest", "AssessmentResponse",
    "SymptomExtractionRequest", "SymptomExtractionResponse",
    "DifferentialRequest", "DifferentialResponse",
    "SessionStartRequest", "SessionStartResponse",
    "SessionEndRequest", "SessionEndResponse",
]


def __getattr__(name: str):
    """Lazy imports to avoid triggering schema loading during test collection."""
    if name in __all__:
        from .schemas import (  # noqa: F811
            DiagnosisPhase, SeverityLevel, ConfidenceLevel, ReasoningStep, SymptomType,
            SymptomDTO, HypothesisDTO, DifferentialDTO, ReasoningStepResultDTO,
            AssessmentRequest, AssessmentResponse,
            SymptomExtractionRequest, SymptomExtractionResponse,
            DifferentialRequest, DifferentialResponse,
            SessionStartRequest, SessionStartResponse,
            SessionEndRequest, SessionEndResponse,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
