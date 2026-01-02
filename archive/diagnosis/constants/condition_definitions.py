"""
Mental Health Condition Definitions

Centralized definitions for mental health conditions including:
- Symptom lists
- Voice indicators
- Personality correlations (Big Five model)
- Severity thresholds

These definitions are based on clinical guidelines and should be used
consistently across all diagnosis modules to ensure standardized assessments.
"""

from typing import Dict, Any, List, Optional

# Standard severity levels used across all conditions
SEVERITY_LEVELS = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "critical": 4,
}

# Mental health condition definitions with associated symptoms and patterns
CONDITION_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "depression": {
        "name": "Depression",
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

# Response templates for different diagnosis severities
RESPONSE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "severe": {
        "depression": (
            "I'm noticing several indicators in our conversation that align with symptoms of depression "
            "at a significant level. These include {symptoms}. Based on these patterns, it may be beneficial "
            "to speak with a mental health professional soon. They can provide proper evaluation and support."
        ),
        "anxiety": (
            "Our conversation suggests you may be experiencing several symptoms associated with anxiety "
            "at a concerning level, such as {symptoms}. Speaking with a mental health professional "
            "could provide you with effective strategies and support for managing these feelings."
        ),
        "stress": (
            "I'm detecting multiple signs of significant stress in our conversation, including {symptoms}. "
            "This level of stress can impact your wellbeing if sustained. Connecting with a healthcare "
            "provider could help you develop effective stress management techniques."
        ),
        "ptsd": (
            "I've noticed several patterns in our conversation that align with post-traumatic stress, "
            "including {symptoms}. These symptoms appear to be at a significant level. Speaking with a "
            "trauma-informed mental health professional would be beneficial for proper assessment and care."
        ),
        "bipolar": (
            "Our conversation contains several indicators that align with bipolar patterns, including {symptoms}. "
            "The nature of these symptoms suggests it would be beneficial to consult with a psychiatrist "
            "who can provide proper evaluation and discuss management strategies."
        )
    },
    "moderate": {
        "depression": (
            "I'm noticing some patterns in our conversation that have similarities to depression symptoms, "
            "like {symptoms}. These indicators are at a moderate level. If these feelings are affecting "
            "your daily life, consider speaking with a healthcare provider."
        ),
        "anxiety": (
            "Our conversation suggests some moderate anxiety-related patterns, including {symptoms}. "
            "If you find these feelings are interfering with your daily activities, speaking with a "
            "mental health professional could be helpful."
        ),
        "stress": (
            "I'm recognizing moderate stress indicators in our conversation, such as {symptoms}. "
            "Finding effective ways to manage stress can prevent it from becoming more severe. "
            "Consider activities like mindfulness, exercise, or speaking with a professional."
        ),
        "ptsd": (
            "Some patterns in our conversation align with moderate post-traumatic stress responses, "
            "including {symptoms}. Speaking with a mental health professional who specializes in "
            "trauma could provide helpful insights and support."
        ),
        "bipolar": (
            "I'm noticing some patterns in our interaction that have similarities to mood cycling, "
            "including {symptoms}. These patterns are at a moderate level. A mental health professional "
            "could help determine if these experiences warrant further attention."
        )
    },
    "mild": {
        "depression": (
            "I'm noticing some mild indicators in our conversation that sometimes accompany low mood, "
            "including {symptoms}. While these are mild, monitoring how they affect you over time "
            "is important."
        ),
        "anxiety": (
            "There are some mild patterns in our conversation that can sometimes be associated with "
            "anxiety, such as {symptoms}. These appear to be at a mild level, but it's good to be "
            "aware of them."
        ),
        "stress": (
            "I'm detecting some mild stress indicators in our conversation, like {symptoms}. "
            "While mild stress is a normal part of life, having good coping strategies is important."
        ),
        "ptsd": (
            "I've noticed some mild stress response patterns in our conversation, including {symptoms}. "
            "While these are mild, paying attention to how they affect you is important, especially "
            "if they're connected to difficult experiences."
        ),
        "bipolar": (
            "There are some mild mood variation patterns in our conversation, including {symptoms}. "
            "These appear to be at a mild level. Monitoring how your mood patterns affect your life "
            "can be helpful."
        )
    }
}


def get_condition_names() -> List[str]:
    """Get list of all defined condition names."""
    return list(CONDITION_DEFINITIONS.keys())


def get_symptoms_for_condition(condition: str) -> List[str]:
    """Get symptoms list for a specific condition.

    Args:
        condition: Condition key (e.g., 'depression', 'anxiety')

    Returns:
        List of symptom strings, or empty list if condition not found
    """
    if condition in CONDITION_DEFINITIONS:
        return CONDITION_DEFINITIONS[condition].get("symptoms", [])
    return []


def get_severity_threshold(condition: str, severity: str) -> Optional[int]:
    """Get the symptom count threshold for a severity level.

    Args:
        condition: Condition key (e.g., 'depression', 'anxiety')
        severity: Severity level ('mild', 'moderate', 'severe')

    Returns:
        Threshold count, or None if not found
    """
    if condition in CONDITION_DEFINITIONS:
        thresholds = CONDITION_DEFINITIONS[condition].get("severity_thresholds", {})
        return thresholds.get(severity)
    return None
