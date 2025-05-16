#!/usr/bin/env python
"""
Central Vector Database Test Script

This script tests the central vector database functionality by performing 
various operations like storing and retrieving different types of data.
"""

import sys
import logging
from pathlib import Path
import asyncio
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.database.central_vector_db import CentralVectorDB
from src.utils.vector_db_integration import (
    add_user_data, 
    get_user_data, 
    search_relevant_data,
    search_knowledge,
    find_therapy_resources
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_central_vector_db():
    """Test all central vector database functionality"""
    
    print("\n===== CENTRAL VECTOR DATABASE TEST =====\n")
    
    # Initialize DB directly for testing
    db = CentralVectorDB(user_id="test_user")
    print(f"Initialized central vector DB for test_user")
    
    # Test storing user profile
    profile_data = {
        "name": "Test User",
        "age": 30,
        "preferences": {
            "theme": "dark",
            "notification_enabled": True
        },
        "interests": ["anxiety management", "stress reduction", "meditation"],
        "timestamp": datetime.now().isoformat()
    }
    
    print("\n1. Adding user profile...")
    profile_id = db.add_user_profile(profile_data)
    print(f"Profile ID: {profile_id}")
    
    # Test retrieving user profile
    print("\n2. Retrieving user profile...")
    retrieved_profile = db.get_user_profile()
    print(f"Retrieved profile: {json.dumps(retrieved_profile, indent=2)}")
    
    # Test storing diagnostic data
    diagnostic_data = {
        "assessment_date": datetime.now().isoformat(),
        "primary_concerns": ["anxiety", "sleep issues"],
        "severity_level": "moderate",
        "potential_conditions": [
            {"condition": "Generalized Anxiety Disorder", "confidence": 0.85},
            {"condition": "Insomnia", "confidence": 0.72}
        ],
        "recommendations": [
            "Consider speaking with a therapist about anxiety management",
            "Establish a regular sleep schedule",
            "Practice relaxation techniques before bedtime"
        ]
    }
    
    print("\n3. Adding diagnostic data...")
    diagnosis_id = db.add_diagnostic_data(diagnostic_data)
    print(f"Diagnosis ID: {diagnosis_id}")
    
    # Test storing personality assessment
    personality_data = {
        "assessment_type": "big_five",
        "assessment_date": datetime.now().isoformat(),
        "scores": {
            "openness": 0.75,
            "conscientiousness": 0.68,
            "extraversion": 0.45,
            "agreeableness": 0.80,
            "neuroticism": 0.62
        },
        "interpretation": "You score high in openness and agreeableness, suggesting you are creative, curious, and empathetic.",
        "recommendations": [
            "Your high neuroticism score suggests practicing mindfulness may help manage stress",
            "Your moderate extraversion indicates balance between social activity and alone time"
        ]
    }
    
    print("\n4. Adding personality assessment...")
    personality_id = db.add_personality_assessment(personality_data)
    print(f"Personality ID: {personality_id}")
    
    # Test storing conversations
    conversation_tracker = db.get_conversation_tracker()
    
    print("\n5. Adding conversation...")
    conversation_id = conversation_tracker.add_conversation(
        user_message="I've been feeling anxious lately, especially at night when trying to sleep.",
        assistant_response="I'm sorry to hear you're experiencing anxiety, particularly at night. This is common and there are several techniques that might help. Have you tried any relaxation methods before bedtime?",
        emotion_data={"primary_emotion": "anxiety", "intensity": 0.75},
        metadata={"session_id": "test_session", "timestamp": datetime.now().isoformat()}
    )
    print(f"Conversation ID: {conversation_id}")
    
    # Test storing knowledge items
    knowledge_item = {
        "title": "Grounding Techniques for Anxiety",
        "content": "Grounding techniques help manage anxiety by focusing on the present moment. The 5-4-3-2-1 technique involves identifying 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste.",
        "category": "anxiety_management",
        "keywords": ["anxiety", "grounding", "coping strategies", "mindfulness"],
        "source": "Clinical practice guidelines",
        "timestamp": datetime.now().isoformat()
    }
    
    print("\n6. Adding knowledge item...")
    knowledge_id = db.add_knowledge_item(knowledge_item)
    print(f"Knowledge ID: {knowledge_id}")
    
    # Test storing therapy resource
    therapy_resource = {
        "title": "Progressive Muscle Relaxation Script",
        "content": "Progressive muscle relaxation involves tensing and then releasing each muscle group. Start with your toes and work up to your head. Tense each muscle group for 5 seconds, then relax for 30 seconds before moving to the next group.",
        "resource_type": "technique",
        "condition": "anxiety",
        "format": "text",
        "keywords": ["relaxation", "anxiety", "stress", "tension"],
        "timestamp": datetime.now().isoformat()
    }
    
    print("\n7. Adding therapy resource...")
    resource_id = db.add_therapy_resource(therapy_resource)
    print(f"Resource ID: {resource_id}")
    
    # Test semantic search
    print("\n8. Testing semantic search for 'anxiety at night'...")
    search_results = db.search_documents("anxiety at night", limit=3)
    print(f"Found {len(search_results)} relevant documents")
    for i, result in enumerate(search_results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result.get('score', 'N/A')}")
        print(f"  Type: {result.get('type', 'Unknown')}")
        title = result.get('title', result.get('content', '')[:50] + '...')
        print(f"  Title/Content: {title}")
    
    # Test utility functions
    print("\n9. Testing vector_db_integration utilities...")
    
    # Test adding data via utilities
    print("\nAdding data via utilities...")
    mood_data = {
        "mood": "contemplative",
        "intensity": 0.6,
        "timestamp": datetime.now().isoformat()
    }
    mood_id = add_user_data("mood", mood_data)
    print(f"Added mood data with ID: {mood_id}")
    
    # Test getting data via utilities
    print("\nGetting latest diagnosis via utilities...")
    latest_diagnosis = get_user_data("diagnosis")
    if latest_diagnosis:
        print(f"Retrieved diagnosis with severity: {latest_diagnosis.get('severity_level', 'unknown')}")
    else:
        print("No diagnosis found")
    
    # Test searching knowledge
    print("\nSearching knowledge about sleep issues...")
    knowledge_results = search_knowledge("difficulty sleeping")
    print(f"Found {len(knowledge_results)} knowledge items")
    
    # Test finding therapy resources
    print("\nFinding therapy resources for anxiety...")
    therapy_results = find_therapy_resources("anxiety")
    print(f"Found {len(therapy_results)} therapy resources")
    
    print("\n===== TEST COMPLETE =====\n")
    
    return {
        "success": True,
        "profile_id": profile_id,
        "diagnosis_id": diagnosis_id,
        "personality_id": personality_id,
        "conversation_id": conversation_id,
        "knowledge_id": knowledge_id,
        "resource_id": resource_id
    }

if __name__ == "__main__":
    try:
        results = asyncio.run(test_central_vector_db())
        
        print("\nSUMMARY:")
        print("--------")
        for key, value in results.items():
            if key != "success":
                print(f"{key}: {value}")
                
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
