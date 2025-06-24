"""
Example script demonstrating the integrated diagnosis module functionality.

This script shows how to combine voice emotion data, conversation data, 
and personality assessments to generate a comprehensive mental health assessment.
"""

import asyncio
import json
from pathlib import Path
import logging

# Project imports
from src.diagnosis import DiagnosisModule
from src.utils.agentic_rag import AgenticRAG

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example data
SAMPLE_CONVERSATION_DATA = {
    "text": "I've been feeling really tired lately and I'm having trouble getting out of bed. "
            "I don't enjoy activities I used to like, and I find myself worrying a lot about the future. "
            "Sometimes I feel worthless and have trouble concentrating on my work.",
    "extracted_symptoms": [
        "fatigue", 
        "loss of interest", 
        "excessive worry",
        "feelings of worthlessness",
        "difficulty concentrating"
    ],
    "sentiment": -0.65,
    "topics": ["mood", "energy", "work", "self-perception"]
}

SAMPLE_VOICE_DATA = {
    "emotions": {
        "sad": 0.72,
        "neutral": 0.15,
        "anxious": 0.65,
        "angry": 0.12
    },
    "characteristics": {
        "speech_rate": 0.35,  # slower than average
        "pitch_variation": 0.28,  # low variation
        "volume": 0.40,  # quieter than average
        "pauses": 0.68  # more pauses than average
    },
    "confidence": 0.85
}

SAMPLE_PERSONALITY_DATA = {
    "big_five": {
        "neuroticism": 0.76,
        "extraversion": 0.32,
        "openness": 0.61,
        "agreeableness": 0.58,
        "conscientiousness": 0.45
    },
    "mbti": "INFP",
    "assessment_date": "2025-05-01"
}

async def main():
    """Run the example diagnosis integration"""
    logger.info("Starting integrated diagnosis example")
    
    # Create the diagnosis module instance
    # In a real application, you might pass an actual AgenticRAG instance
    diagnosis_module = DiagnosisModule(agentic_rag=None, use_vector_cache=False)
    
    # Generate a diagnosis using all data sources
    result = await diagnosis_module.generate_diagnosis(
        conversation_data=SAMPLE_CONVERSATION_DATA,
        voice_emotion_data=SAMPLE_VOICE_DATA,
        personality_data=SAMPLE_PERSONALITY_DATA,
        user_id="example_user_123"
    )
    
    # Pretty print the results
    print("\n=== INTEGRATED DIAGNOSIS RESULTS ===\n")
    
    if result["success"]:
        print(f"Detected Conditions:")
        for condition in result["conditions"]:
            print(f"  - {condition['name']} (Severity: {condition['severity']}, "
                  f"Confidence: {condition['confidence']:.2f})")
        
        print("\nOverall Severity:", result["severity"])
        print("Overall Confidence:", f"{result['confidence']:.2f}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print("\nInsights:")
        print(f"  Symptoms Identified: {', '.join(result['insights']['symptoms_identified'])}")
        print(f"  Emotional Indicators: {', '.join(result['insights']['emotional_indicators'])}")
        
        # Print evidence summary
        evidence = result["insights"]["evidence"]
        print(f"\nEvidence Summary:")
        print(f"  Total Indicators: {evidence['total_indicators']}")
        print(f"  Indicators by Source:")
        for source, count in evidence["indicators_by_source"].items():
            print(f"    - {source}: {count}")
        
        print("\nKey Symptoms by Condition:")
        for condition_symptoms in evidence["key_symptoms"]:
            print(f"  {condition_symptoms['condition']}: {', '.join(condition_symptoms['symptoms'])}")
    else:
        print("No significant conditions detected.")
        if "message" in result:
            print(f"Message: {result['message']}")
        if "error" in result:
            print(f"Error: {result['error']}")
            
    print(f"\nProcessing Time: {result['processing_time_seconds']:.2f} seconds")
    print("\n=====================================\n")
    
    # Example of using just conversation data
    print("Generating diagnosis with conversation data only...")
    result_conv_only = await diagnosis_module.generate_diagnosis(
        conversation_data=SAMPLE_CONVERSATION_DATA
    )
    print(f"Success: {result_conv_only['success']}")
    if result_conv_only['success']:
        print(f"Detected conditions: {[c['name'] for c in result_conv_only['conditions']]}")
    
    # Example of using just voice emotion data
    print("\nGenerating diagnosis with voice emotion data only...")
    result_voice_only = await diagnosis_module.generate_diagnosis(
        voice_emotion_data=SAMPLE_VOICE_DATA
    )
    print(f"Success: {result_voice_only['success']}")
    if result_voice_only['success']:
        print(f"Detected conditions: {[c['name'] for c in result_voice_only['conditions']]}")
    
    # Example of using just personality data
    print("\nGenerating diagnosis with personality data only...")
    result_personality_only = await diagnosis_module.generate_diagnosis(
        personality_data=SAMPLE_PERSONALITY_DATA
    )
    print(f"Success: {result_personality_only['success']}")
    if result_personality_only['success']:
        print(f"Detected conditions: {[c['name'] for c in result_personality_only['conditions']]}")

if __name__ == "__main__":
    asyncio.run(main())