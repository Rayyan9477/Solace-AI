"""
Solace-AI Orchestrator Service - Emotion Agent.
Handles emotional support routing and empathetic response generation.
"""
from __future__ import annotations
import re
import time
from typing import Any
import structlog

from ..langgraph.state_schema import (
    OrchestratorState, AgentType, AgentResult, ProcessingPhase,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Keyword-based emotion detector
# ---------------------------------------------------------------------------

# Maps emotion categories to trigger words (lowercase).
_EMOTION_KEYWORDS: dict[str, set[str]] = {
    "sadness": {
        "sad", "depressed", "down", "unhappy", "miserable", "hopeless",
        "empty", "lonely", "grief", "crying", "tearful", "heartbroken",
        "worthless", "despair", "gloomy", "melancholy",
    },
    "anxiety": {
        "anxious", "worried", "nervous", "scared", "afraid", "panic",
        "stressed", "tense", "overwhelmed", "dread", "uneasy", "restless",
        "fearful", "apprehensive", "on edge",
    },
    "anger": {
        "angry", "furious", "mad", "frustrated", "irritated", "annoyed",
        "rage", "resentful", "hostile", "bitter", "livid", "outraged",
    },
    "shame": {
        "ashamed", "embarrassed", "guilty", "humiliated", "inadequate",
        "regret", "blame", "disgrace", "mortified",
    },
    "joy": {
        "happy", "grateful", "thankful", "excited", "proud", "relieved",
        "hopeful", "content", "cheerful", "optimistic", "delighted",
    },
    "confusion": {
        "confused", "lost", "uncertain", "unsure", "don't know",
        "overwhelmed", "stuck", "torn", "conflicted",
    },
}

# Empathic response templates per detected emotion.
_EMPATHIC_RESPONSES: dict[str, str] = {
    "sadness": (
        "I can hear that you're carrying a lot of sadness right now, "
        "and I want you to know that it's okay to feel this way. "
        "Sadness is a natural response, and you don't have to go through it alone."
    ),
    "anxiety": (
        "It sounds like you're feeling really anxious, and that can be incredibly "
        "overwhelming. I want you to know that what you're feeling is valid, "
        "and there are ways we can work through this together."
    ),
    "anger": (
        "I can sense that you're feeling frustrated or angry right now. "
        "Those feelings are completely understandable given what you're going through. "
        "It's important to acknowledge them rather than push them aside."
    ),
    "shame": (
        "It takes real courage to share feelings of shame or guilt. "
        "I want you to know that everyone struggles sometimes, "
        "and these feelings don't define who you are as a person."
    ),
    "joy": (
        "It's wonderful to hear some positive feelings coming through. "
        "Recognizing and savoring these moments is really important. "
        "What do you think has been contributing to this feeling?"
    ),
    "confusion": (
        "Feeling confused or uncertain can be really disorienting. "
        "Let's try to sort through what you're experiencing together. "
        "Sometimes talking it out can help bring a bit more clarity."
    ),
}

_DEFAULT_RESPONSE = (
    "I hear you, and I want you to know that your feelings are valid. "
    "It takes courage to share what you're going through. "
    "Can you tell me a little more about how you're feeling right now?"
)


def _detect_emotions(text: str) -> list[tuple[str, float]]:
    """Detect emotions from text using keyword matching.

    Returns a list of (emotion, score) tuples sorted by score descending.
    The score is the proportion of keywords matched relative to the category
    size, giving a rough confidence estimate.
    """
    text_lower = text.lower()
    results: list[tuple[str, float]] = []
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        matched = sum(
            1
            for kw in keywords
            if re.search(rf'\b{re.escape(kw)}\b', text_lower)
        )
        if matched > 0:
            score = min(0.95, 0.4 + (matched / len(keywords)) * 0.55)
            results.append((emotion, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _build_empathic_response(
    detected: list[tuple[str, float]],
) -> tuple[str, str, float]:
    """Build an empathic response based on detected emotions.

    Returns (response_text, primary_emotion, confidence).
    """
    if not detected:
        return _DEFAULT_RESPONSE, "neutral", 0.5

    primary_emotion, confidence = detected[0]
    response = _EMPATHIC_RESPONSES.get(primary_emotion, _DEFAULT_RESPONSE)

    # If multiple strong emotions detected, acknowledge complexity
    if len(detected) >= 2 and detected[1][1] >= 0.5:
        secondary = detected[1][0]
        response += (
            f" I also notice some {secondary} in what you're sharing. "
            "It's completely normal to experience a mix of emotions."
        )

    return response, primary_emotion, confidence


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


async def emotion_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Emotion agent - detects emotions and provides empathetic responses."""
    start_time = time.perf_counter()
    message = state.get("current_message", "")
    user_id = state.get("user_id", "")
    logger.info("emotion_agent_processing", user_id=user_id, message_length=len(message))

    detected_emotions = _detect_emotions(message)
    response, primary_emotion, confidence = _build_empathic_response(detected_emotions)

    processing_time = int((time.perf_counter() - start_time) * 1000)
    agent_result = AgentResult(
        agent_type=AgentType.EMOTION,
        success=True,
        response_content=response,
        confidence=confidence,
        processing_time_ms=processing_time,
        metadata={
            "intent": "emotional_support",
            "primary_emotion": primary_emotion,
            "detected_emotions": [
                {"emotion": e, "score": round(s, 3)} for e, s in detected_emotions
            ],
        },
    )
    logger.info(
        "emotion_agent_complete",
        processing_time_ms=processing_time,
        primary_emotion=primary_emotion,
        emotion_count=len(detected_emotions),
    )
    return {
        "agent_results": [agent_result.to_dict()],
        "processing_phase": ProcessingPhase.PARALLEL_PROCESSING.value,
        "emotion_output": {
            "empathy_applied": True,
            "primary_emotion": primary_emotion,
            "detected_emotions": [e for e, _ in detected_emotions],
        },
    }
