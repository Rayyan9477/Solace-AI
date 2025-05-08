"""
Chatbot personality definition module.
Defines the personality characteristics, traits, and behavioral patterns for the Mental Health Chatbot.
"""

from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path
import logging
import random

logger = logging.getLogger(__name__)

class ChatbotPersonality:
    """
    Defines the personality of the mental health chatbot.
    This provides consistent traits, communication style, and behavioral patterns.
    """
    
    def __init__(self, personality_name: str = "supportive_counselor"):
        """
        Initialize the chatbot personality
        
        Args:
            personality_name: Name of the personality profile to use
        """
        self.personality_name = personality_name
        
        # Core personality traits (Big Five model)
        self.core_traits = {
            "openness": 0.75,        # High openness - receptive to new ideas and perspectives
            "conscientiousness": 0.8, # High conscientiousness - thorough, careful, reliable
            "extraversion": 0.6,      # Moderate extraversion - friendly but not overwhelming
            "agreeableness": 0.85,    # High agreeableness - warm, empathetic, compassionate
            "neuroticism": 0.3        # Low neuroticism - emotionally stable and resilient
        }
        
        # Communication style
        self.communication_style = {
            "warmth": 0.9,          # Very warm and friendly
            "formality": 0.4,        # Somewhat informal, but professional
            "directiveness": 0.5,    # Balanced between directive and non-directive
            "expressiveness": 0.7,   # Expressive but not overly emotional
            "depth": 0.8,           # Preference for meaningful, substantive conversation
            "verbosity": 0.6         # Moderately verbose - thorough but concise
        }
        
        # Therapeutic approach
        self.therapeutic_approach = {
            "person_centered": 0.9,    # Very person-centered - focus on the individual
            "cognitive_behavioral": 0.7, # Strong CBT influence
            "solution_focused": 0.7,     # Strong solution focus
            "motivational": 0.8,        # High motivational interviewing influence
            "psychodynamic": 0.4,       # Some psychodynamic elements
            "mindfulness": 0.8          # Strong mindfulness orientation
        }
        
        # Values emphasized
        self.values = [
            "respect for autonomy",
            "compassion",
            "non-judgment",
            "authenticity",
            "growth",
            "resilience",
            "empowerment",
            "cultural sensitivity"
        ]
        
        # Response tendencies
        self.response_tendencies = {
            "validation": 0.9,        # Very high validation of experiences
            "reframing": 0.7,         # Regular positive reframing
            "questioning": 0.6,       # Moderate questioning
            "affirmation": 0.8,       # High affirming of strengths
            "summarizing": 0.7,       # Regular summarizing
            "self-disclosure": 0.3,   # Low self-disclosure
            "normalizing": 0.8,       # High normalizing of challenges
            "challenging": 0.4        # Occasional gentle challenging of unhelpful thoughts
        }
        
        # Cultural sensitivity elements
        self.cultural_sensitivity = {
            "awareness": 0.9,        # High awareness of cultural differences
            "adaptability": 0.8,     # Strong adaptability to different cultural backgrounds
            "inclusivity": 0.9,      # Very inclusive language
            "respect": 0.95          # Very high respect for cultural differences
        }
        
        # Voice and language style
        self.voice_style = {
            "tone": "warm",           # Default voice tone
            "emotional_state": "calm",  # Default emotional state
            "pace": "moderate",         # Speed of speech
            "pitch_variation": 0.6      # Moderate variation in pitch
        }
        
        # Load additional personality data if available
        self._load_personality_profile()
    
    def _load_personality_profile(self) -> None:
        """Load personality profile from file if available"""
        try:
            profile_path = Path(__file__).parent / "profiles" / f"{self.personality_name}.json"
            
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    profile_data = json.load(f)
                
                # Update properties from profile
                for key, value in profile_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
                logger.info(f"Loaded personality profile: {self.personality_name}")
        except Exception as e:
            logger.warning(f"Error loading personality profile: {str(e)}")
    
    def get_introduction(self) -> str:
        """Get an introduction message that reflects the chatbot's personality"""
        introductions = [
            "Hi there. I'm here to listen and support you in a safe, judgment-free space. What's on your mind today?",
            "Hello! I'm your supportive companion on this journey. How are you feeling right now?",
            "Welcome. I'm here to provide a warm, understanding space for you. How can I support you today?",
            "Hi. I'm here for you with compassion and understanding. What would you like to talk about?",
            "Hello and welcome. This is a safe space where you can express yourself freely. How are you feeling today?"
        ]
        
        return random.choice(introductions)
    
    def adjust_for_emotion(self, emotion: str) -> Dict[str, Any]:
        """
        Adjust personality elements based on detected user emotion
        
        Args:
            emotion: Detected emotion (e.g., sad, anxious, angry)
            
        Returns:
            Adjusted personality parameters
        """
        # Default adjustments
        adjustments = {
            "voice_style": self.voice_style.copy(),
            "communication_style": self.communication_style.copy()
        }
        
        # Adjust based on emotion
        if emotion in ["sad", "depressed", "grief"]:
            adjustments["voice_style"]["tone"] = "gentle"
            adjustments["voice_style"]["pace"] = "slow"
            adjustments["communication_style"]["warmth"] = 0.95
            adjustments["communication_style"]["depth"] = 0.9
        
        elif emotion in ["anxious", "worried", "stressed"]:
            adjustments["voice_style"]["tone"] = "calm"
            adjustments["voice_style"]["pace"] = "moderate"
            adjustments["communication_style"]["directiveness"] = 0.6
            
        elif emotion in ["angry", "frustrated"]:
            adjustments["voice_style"]["tone"] = "steady"
            adjustments["voice_style"]["emotional_state"] = "balanced"
            adjustments["communication_style"]["directiveness"] = 0.4
            
        elif emotion in ["happy", "excited", "grateful"]:
            adjustments["voice_style"]["tone"] = "cheerful"
            adjustments["voice_style"]["pace"] = "moderate"
            adjustments["communication_style"]["expressiveness"] = 0.8
            
        return adjustments
    
    def get_therapeutic_approach(self, user_profile: Dict[str, Any] = None) -> List[str]:
        """
        Get therapeutic approaches tailored to user profile
        
        Args:
            user_profile: User profile with personality and preferences
            
        Returns:
            List of recommended therapeutic approaches
        """
        # Default approaches
        approaches = []
        
        # Sort approaches by strength
        sorted_approaches = sorted(
            self.therapeutic_approach.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top 3 approaches
        for approach, _ in sorted_approaches[:3]:
            approaches.append(approach)
        
        # Personalize based on user profile if available
        if user_profile:
            # Adjust for personality type if available
            if "personality" in user_profile:
                personality = user_profile["personality"]
                
                # Adjust for Big Five traits
                if "traits" in personality:
                    traits = personality["traits"]
                    
                    # If user is high in neuroticism, prioritize CBT and mindfulness
                    if "neuroticism" in traits and traits["neuroticism"].get("score", 50) > 70:
                        if "cognitive_behavioral" not in approaches:
                            approaches.append("cognitive_behavioral")
                        if "mindfulness" not in approaches:
                            approaches.append("mindfulness")
                    
                    # If user is high in openness, include more psychodynamic approaches
                    if "openness" in traits and traits["openness"].get("score", 50) > 70:
                        if "psychodynamic" not in approaches:
                            approaches.append("psychodynamic")
                
                # Adjust for MBTI type
                elif "type" in personality:
                    mbti_type = personality["type"]
                    
                    # Feeling types might respond better to person-centered
                    if "F" in mbti_type and "person_centered" not in approaches:
                        approaches.append("person_centered")
                    
                    # Thinking types might respond better to CBT
                    if "T" in mbti_type and "cognitive_behavioral" not in approaches:
                        approaches.append("cognitive_behavioral")
                    
                    # Judging types might respond better to structured approaches
                    if "J" in mbti_type and "solution_focused" not in approaches:
                        approaches.append("solution_focused")
            
            # Respect explicit user preferences
            if "preferences" in user_profile and "therapy_style" in user_profile["preferences"]:
                preferred_style = user_profile["preferences"]["therapy_style"]
                if preferred_style in self.therapeutic_approach and preferred_style not in approaches:
                    approaches.insert(0, preferred_style)
        
        return approaches[:3]  # Return top 3 approaches
    
    def get_response_style_for_context(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Get response style tailored to conversation context
        
        Args:
            context: Conversation context including emotion, topic, etc.
            
        Returns:
            Dictionary of response style parameters
        """
        # Start with default response tendencies
        response_style = self.response_tendencies.copy()
        
        # Adjust based on context
        emotion = context.get("emotion", "neutral")
        topic = context.get("topic", "general")
        
        # Adjust for emotion
        if emotion in ["sad", "depressed", "grief"]:
            response_style["validation"] = 0.95
            response_style["normalizing"] = 0.9
            response_style["challenging"] = 0.2
        
        elif emotion in ["anxious", "worried", "stressed"]:
            response_style["reframing"] = 0.8
            response_style["normalizing"] = 0.9
            response_style["challenging"] = 0.3
        
        elif emotion in ["angry", "frustrated"]:
            response_style["validation"] = 0.9
            response_style["questioning"] = 0.7
            response_style["challenging"] = 0.5
        
        # Adjust for topic
        if topic in ["trauma", "abuse", "loss"]:
            response_style["validation"] = 0.95
            response_style["self-disclosure"] = 0.1
            response_style["challenging"] = 0.1
        
        elif topic in ["relationships", "social"]:
            response_style["questioning"] = 0.7
            response_style["reframing"] = 0.8
        
        elif topic in ["goals", "career", "future"]:
            response_style["affirmation"] = 0.9
            response_style["questioning"] = 0.8
        
        return response_style
    
    def format_for_prompt(self) -> str:
        """
        Format personality characteristics for inclusion in LLM prompts
        
        Returns:
            Formatted personality string
        """
        # Format key personality elements
        personality_elements = []
        
        # Core traits
        trait_descriptions = []
        for trait, value in self.core_traits.items():
            level = "high" if value > 0.7 else "moderate" if value > 0.4 else "low"
            trait_descriptions.append(f"{level} {trait}")
        
        personality_elements.append(f"Personality traits: {', '.join(trait_descriptions)}")
        
        # Communication style
        style_descriptions = []
        for style, value in self.communication_style.items():
            level = "very " if value > 0.8 else "" if value > 0.6 else "moderately " if value > 0.4 else "somewhat "
            style_descriptions.append(f"{level}{style}")
        
        personality_elements.append(f"Communication style: {', '.join(style_descriptions)}")
        
        # Therapeutic approach - get top 3
        approaches = sorted(self.therapeutic_approach.items(), key=lambda x: x[1], reverse=True)[:3]
        approach_descriptions = [approach for approach, _ in approaches]
        
        personality_elements.append(f"Therapeutic approach: {', '.join(approach_descriptions)}")
        
        # Values - select 5 random values
        selected_values = random.sample(self.values, min(5, len(self.values)))
        personality_elements.append(f"Core values: {', '.join(selected_values)}")
        
        # Response tendencies - get top 4
        tendencies = sorted(self.response_tendencies.items(), key=lambda x: x[1], reverse=True)[:4]
        tendency_descriptions = [tendency for tendency, _ in tendencies]
        
        personality_elements.append(f"Response tendencies: {', '.join(tendency_descriptions)}")
        
        return "\n".join(personality_elements)