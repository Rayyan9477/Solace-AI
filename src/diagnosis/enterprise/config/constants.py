"""
Constants and enums for Enterprise Diagnostic Pipeline
"""

from enum import Enum
from typing import Dict, Any

class ModalityType(Enum):
    """Supported modality types for multi-modal fusion"""
    TEXT = "text"
    VOICE = "voice"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    PHYSIOLOGICAL = "physiological"
    CONTEXTUAL = "contextual"

class ConfidenceLevel(Enum):
    """Confidence levels for diagnostic results"""
    VERY_LOW = 0.2
    LOW = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

class ClinicalSeverity(Enum):
    """Clinical severity levels following DSM-5/ICD-11 standards"""
    REMISSION = "remission"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class ProcessingStage(Enum):
    """Processing stages for pipeline monitoring"""
    INPUT_VALIDATION = "input_validation"
    FEATURE_EXTRACTION = "feature_extraction"
    MODAL_FUSION = "modal_fusion"
    DIAGNOSTIC_INFERENCE = "diagnostic_inference"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    RECOMMENDATION_GENERATION = "recommendation_generation"
    OUTPUT_FORMATTING = "output_formatting"

# Mental health condition definitions
CONDITION_DEFINITIONS = {
    "depression": {
        "name": "Depression",
        "dsm5_code": "296.xx",
        "icd11_code": "6A70",
        "symptoms": [
            "persistent sadness", "loss of interest", "fatigue", "sleep problems",
            "appetite changes", "feelings of worthlessness", "difficulty concentrating", 
            "negative thoughts", "low energy", "social withdrawal", "low self-esteem",
            "suicidal thoughts", "feeling empty", "guilt", "self-blame", "helplessness",
            "hopelessness", "psychomotor retardation", "crying spells"
        ],
        "voice_indicators": [
            "monotone voice", "slower speech", "reduced pitch variation", "quieter volume",
            "flat affect", "audible sighs", "long pauses", "reduced speech rate"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high",
                "extraversion": "low",
                "openness": "variable",
                "agreeableness": "variable",
                "conscientiousness": "low"
            }
        },
        "severity_thresholds": {
            "mild": 3,
            "moderate": 5,
            "severe": 7
        }
    },
    "anxiety": {
        "name": "Anxiety",
        "dsm5_code": "300.xx",
        "icd11_code": "6B00",
        "symptoms": [
            "excessive worry", "restlessness", "fatigue", "difficulty concentrating", 
            "irritability", "muscle tension", "sleep problems", "racing thoughts",
            "feeling on edge", "anticipating worst outcomes", "avoiding situations",
            "panic attacks", "sweating", "trembling", "heart palpitations",
            "shortness of breath", "fear of losing control", "difficulty making decisions"
        ],
        "voice_indicators": [
            "faster speech", "higher pitch", "trembling voice", "rapid breathing",
            "voice cracks", "stuttering", "frequent clearing of throat", "halting speech"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high", 
                "extraversion": "variable",
                "openness": "variable",
                "agreeableness": "variable", 
                "conscientiousness": "high"
            }
        },
        "severity_thresholds": {
            "mild": 3,
            "moderate": 5,
            "severe": 7
        }
    },
    "stress": {
        "name": "Stress",
        "dsm5_code": "309.xx",
        "icd11_code": "6B43",
        "symptoms": [
            "feeling overwhelmed", "racing thoughts", "difficulty relaxing", 
            "irritability", "muscle tension", "headaches", "fatigue",
            "sleep problems", "difficulty concentrating", "mood changes",
            "increased heart rate", "digestive issues", "feeling pressured",
            "inability to switch off", "worrying about the future"
        ],
        "voice_indicators": [
            "faster speech", "tense tone", "higher pitch", "louder volume",
            "rapid breathing", "voice strain", "clipped sentences"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high",
                "extraversion": "variable",
                "openness": "variable",
                "agreeableness": "low when stressed",
                "conscientiousness": "variable"
            }
        },
        "severity_thresholds": {
            "mild": 3,
            "moderate": 5,
            "severe": 7
        }
    },
    "ptsd": {
        "name": "PTSD",
        "dsm5_code": "309.81",
        "icd11_code": "6B40",
        "symptoms": [
            "flashbacks", "nightmares", "intrusive memories", "distress at reminders",
            "avoiding trauma reminders", "negative mood", "feeling detached",
            "hypervigilance", "exaggerated startle response", "difficulty sleeping",
            "irritability", "concentration problems", "memory gaps", "self-destructive behavior"
        ],
        "voice_indicators": [
            "emotional numbing in voice", "sudden vocal shifts", "voice trembling",
            "hesitation when discussing triggers", "heightened vocal tension"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high",
                "extraversion": "low",
                "openness": "variable",
                "agreeableness": "low",
                "conscientiousness": "variable"
            }
        },
        "severity_thresholds": {
            "mild": 3,     
            "moderate": 6,  
            "severe": 9     
        }
    },
    "bipolar": {
        "name": "Bipolar Disorder",
        "dsm5_code": "296.xx",
        "icd11_code": "6A60",
        "symptoms": [
            "mood swings", "periods of depression", "periods of elation", 
            "increased energy", "decreased need for sleep", "grandiose ideas",
            "racing thoughts", "rapid speech", "impulsive behavior", "irritability",
            "risky behavior", "poor judgment", "inflated self-esteem"
        ],
        "voice_indicators": [
            "rapid speech during mania", "pressured speech", "flight of ideas in speech",
            "loud volume during mania", "monotone during depression", "dramatic tone shifts"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "variable",
                "extraversion": "variable/cyclical",
                "openness": "high",
                "agreeableness": "variable",
                "conscientiousness": "variable/cyclical"
            }
        },
        "severity_thresholds": {
            "mild": 3,     
            "moderate": 6, 
            "severe": 9    
        }
    }
}

# Default configuration values
DEFAULT_MODEL_CONFIG = {
    "fusion_dim": 512,
    "attention_heads": 8,
    "dropout": 0.1,
    "temporal_hidden_dim": 256,
    "temporal_layers": 2,
    "uncertainty_samples": 100
}

DEFAULT_CLINICAL_CONFIG = {
    "confidence_threshold": 0.6,
    "severity_thresholds": {
        "mild": 0.3,
        "moderate": 0.6,
        "severe": 0.8
    },
    "dsm5_compliance": True,
    "icd11_compliance": True
}

DEFAULT_PRIVACY_CONFIG = {
    "enable_encryption": True,
    "audit_logging": True,
    "data_retention_days": 90,
    "anonymization": True
}

DEFAULT_PERFORMANCE_CONFIG = {
    "batch_size": 32,
    "max_sequence_length": 512,
    "cache_size": 1000,
    "model_update_frequency": 24  # hours
}