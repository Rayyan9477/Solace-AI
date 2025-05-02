from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
import logging
from datetime import datetime
import json
import os
import sys

# Add the project root to the path to import the personality modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from personality.big_five import BigFiveAssessment
from personality.mbti import MBTIAssessment

# Initialize logger
logger = logging.getLogger(__name__)

@tool(name="personality_assessment", description="Conducts personality assessments using Big Five or MBTI models")
async def assess_personality(assessment_type: str, responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conducts a personality assessment based on user responses
    
    Args:
        assessment_type: The type of assessment to conduct ('big_five' or 'mbti')
        responses: Dictionary containing user responses to assessment questions
        
    Returns:
        Dictionary containing assessment results and interpretation
    """
    try:
        if assessment_type.lower() == 'big_five':
            assessment = BigFiveAssessment()
            return assessment.compute_results(responses)
        elif assessment_type.lower() == 'mbti':
            assessment = MBTIAssessment()
            return assessment.compute_results(responses)
        else:
            return {
                "error": f"Unknown assessment type: {assessment_type}",
                "valid_types": ["big_five", "mbti"]
            }
    except Exception as e:
        logger.error(f"Error in personality assessment: {str(e)}")
        return {
            "error": str(e),
            "assessment_type": assessment_type
        }

class PersonalityAgent(BaseAgent):
    def __init__(self, model: BaseLanguageModel):
        # Create a memory instance
        memory = Memory(
            memory="personality_memory",
            storage="local_storage"
        )
        
        super().__init__(
            model=model,
            name="personality_analyzer",
            description="Expert system for personality assessment and interpretation",
            tools=[assess_personality],
            memory=memory,
            knowledge=AgentKnowledge()
        )
        
        # Create assessment instances
        self.big_five_assessment = BigFiveAssessment()
        self.mbti_assessment = MBTIAssessment()
        
        # Add dynamic question generation capability
        self.question_adaptation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert psychologist specializing in personalized assessment.
Your task is to adapt personality assessment questions based on the user's context, previous responses, and emotional state.
Make the questions relevant to their specific situation while maintaining the scientific validity of the assessment.
Do not change the core psychological construct being measured, only the context or framing of the question."""),
            HumanMessage(content="""Assessment Type: {assessment_type}
Original Question: {original_question}
User Context: {user_context}
Previous Responses: {previous_responses}
Emotional State: {emotional_state}

Generate a more personalized version of this question that:
1. Measures the same psychological construct
2. Adapts to the user's specific context or emotional state
3. Feels more conversational and less like a formal assessment
4. Maintains scientific validity""")
        ])
        
        # Add conversational assessment prompt
        self.conversational_assessment_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in personality psychology conducting a natural, conversational assessment.
Instead of asking standard questionnaire items, engage the user in a conversation that will reveal their personality traits.
Extract meaningful personality insights from their responses while making the interaction feel like a friendly conversation.
Your goal is to assess the user's personality in a way that feels organic rather than clinical."""),
            HumanMessage(content="""Assessment Type: {assessment_type}
Assessment Progress: {progress}
User Context: {user_context}
Previous Conversation: {previous_conversation}
Emotional State: {emotional_state}

Generate the next natural question or response that will help assess the user's personality traits.
Focus on assessing: {traits_to_assess}""")
        ])
        
        # Initialize TTS/STT integration components
        self.voice_enabled = False
        self.voice_component = None
        self.whisper_component = None
        
        # Add voice style preferences mapping
        self.voice_style_mapping = {
            "extraversion": {"high": "energetic", "low": "calm"},
            "agreeableness": {"high": "warm", "low": "neutral"},
            "conscientiousness": {"high": "precise", "low": "relaxed"},
            "neuroticism": {"high": "empathetic", "low": "confident"},
            "openness": {"high": "creative", "low": "straightforward"},
            # MBTI mappings
            "E": "energetic", "I": "thoughtful",
            "S": "clear", "N": "expressive",
            "T": "logical", "F": "empathetic",
            "J": "structured", "P": "conversational"
        }
        
        self.interpretation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in personality psychology and assessment.
Your role is to interpret personality assessment results and provide insights on:
1. Key personality traits and their implications
2. Potential strengths and growth areas
3. Communication preferences and learning styles
4. Stress responses and coping mechanisms
5. Relationship dynamics and teamwork preferences
6. Emotional tendencies and patterns

Provide a balanced, nuanced interpretation that avoids stereotyping or overgeneralizing.
Focus on how understanding personality can help with personal growth and self-awareness."""),
            HumanMessage(content="""Assessment Type: {assessment_type}
Assessment Results: {assessment_results}
User Context: {user_context}
Emotional Context: {emotional_context}

Provide an interpretation of these results, focusing on:
- Key insights about the personality profile
- Potential strengths and growth areas
- Communication and learning preferences
- Stress responses and coping strategies
- Emotional patterns and tendencies
- How this information might help with the user's mental health journey""")
        ])
        
        # Define personality trait to emotion mappings
        self.trait_emotion_mappings = {
            # Big Five mappings
            "openness": {
                "high": ["curiosity", "creativity", "excitement"],
                "low": ["caution", "pragmatism", "contentment"]
            },
            "conscientiousness": {
                "high": ["determination", "focus", "satisfaction"],
                "low": ["spontaneity", "adaptability", "relaxation"]
            },
            "extraversion": {
                "high": ["enthusiasm", "energy", "joy"],
                "low": ["contemplation", "calm", "contentment"]
            },
            "agreeableness": {
                "high": ["empathy", "warmth", "compassion"],
                "low": ["assertiveness", "skepticism", "independence"]
            },
            "neuroticism": {
                "high": ["anxiety", "sensitivity", "caution"],
                "low": ["resilience", "confidence", "calm"]
            },
            # MBTI mappings
            "introversion": ["reflection", "depth", "independence"],
            "extraversion_mbti": ["enthusiasm", "engagement", "expression"],
            "sensing": ["practical", "present-focused", "realistic"],
            "intuition": ["possibility", "future-focused", "imagination"],
            "thinking": ["logical", "analytical", "objective"],
            "feeling": ["empathetic", "harmonious", "values-based"],
            "judging": ["structured", "decisive", "planned"],
            "perceiving": ["flexible", "adaptable", "spontaneous"]
        }
        
        # Define emotion indicator emojis
        self.emotion_indicators = {
            # Basic emotions
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😮",
            "disgust": "🤢",
            "neutral": "😐",
            
            # Personality-related emotions
            "curiosity": "🤔",
            "creativity": "🎨",
            "excitement": "🤩",
            "caution": "⚠️",
            "pragmatism": "🔧",
            "contentment": "😌",
            "determination": "💪",
            "focus": "🔍",
            "satisfaction": "😊",
            "spontaneity": "🎭",
            "adaptability": "🦎",
            "relaxation": "😌",
            "enthusiasm": "🎉",
            "energy": "⚡",
            "contemplation": "🧠",
            "calm": "😌",
            "empathy": "💕",
            "warmth": "🌞",
            "compassion": "🤲",
            "assertiveness": "👊",
            "skepticism": "🤨",
            "independence": "🦅",
            "anxiety": "😰",
            "sensitivity": "🌸",
            "resilience": "🛡️",
            "confidence": "🦁",
            
            # MBTI-specific indicators
            "reflection": "🪞",
            "depth": "🌊",
            "engagement": "🤝",
            "expression": "🗣️",
            "practical": "🛠️",
            "present-focused": "⏰",
            "realistic": "🏢",
            "possibility": "💫",
            "future-focused": "🔮",
            "imagination": "🧙‍♂️",
            "logical": "🧮",
            "analytical": "📊",
            "objective": "⚖️",
            "empathetic": "❤️",
            "harmonious": "☯️",
            "values-based": "🧭",
            "structured": "📐",
            "decisive": "✅",
            "planned": "📅",
            "flexible": "🤸",
            "spontaneous": "🎲"
        }

    async def conduct_assessment(self, assessment_type: str, responses: Dict[str, Any], emotional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Conduct a personality assessment and provide interpretation
        
        Args:
            assessment_type: The type of assessment ('big_five' or 'mbti')
            responses: User responses to assessment questions
            emotional_context: Optional emotional context to incorporate in interpretation
            
        Returns:
            Dictionary containing assessment results and interpretation
        """
        try:
            # Get assessment results
            results = await assess_personality(assessment_type, responses)
            
            # Analyze emotional indicators based on personality traits
            emotion_indicators = self._generate_emotion_indicators(assessment_type, results)
            
            # Store results in memory
            try:
                await self.memory.add("assessment_results", {
                    "type": assessment_type,
                    "results": results,
                    "emotion_indicators": emotion_indicators,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
            
            # Generate interpretation
            interpretation = await self._generate_interpretation(assessment_type, results, emotional_context)
            
            # Combine results and interpretation
            return {
                "assessment_type": assessment_type,
                "results": results,
                "emotion_indicators": emotion_indicators,
                "interpretation": interpretation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error conducting personality assessment: {str(e)}")
            return {
                "error": str(e),
                "assessment_type": assessment_type
            }
    
    def _generate_emotion_indicators(self, assessment_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate emotion indicators based on personality assessment results
        
        Args:
            assessment_type: The type of assessment ('big_five' or 'mbti')
            results: Assessment results
            
        Returns:
            Dictionary with emotion indicators and scores
        """
        indicators = {
            "primary_emotions": [],
            "secondary_emotions": [],
            "emotional_tendencies": {},
            "visual_indicators": []
        }
        
        try:
            if assessment_type.lower() == 'big_five':
                # Process Big Five traits
                traits = results.get("scores", {})
                
                for trait, score in traits.items():
                    trait_lower = trait.lower()
                    if trait_lower in self.trait_emotion_mappings:
                        # Determine if the trait is high or low
                        level = "high" if score > 0.6 else "low"
                        
                        # Get associated emotions
                        emotions = self.trait_emotion_mappings[trait_lower][level]
                        
                        # Add to emotional tendencies
                        for emotion in emotions:
                            indicators["emotional_tendencies"][emotion] = score if level == "high" else 1 - score
                
            elif assessment_type.lower() == 'mbti':
                # Process MBTI type
                mbti_type = results.get("type", "")
                
                if len(mbti_type) == 4:
                    # Add emotions based on each MBTI dimension
                    dimensions = [
                        "introversion" if mbti_type[0] == "I" else "extraversion_mbti",
                        "sensing" if mbti_type[1] == "S" else "intuition",
                        "thinking" if mbti_type[2] == "T" else "feeling",
                        "judging" if mbti_type[3] == "J" else "perceiving"
                    ]
                    
                    # Add emotions for each dimension
                    for dimension in dimensions:
                        if dimension in self.trait_emotion_mappings:
                            emotions = self.trait_emotion_mappings[dimension]
                            for emotion in emotions:
                                indicators["emotional_tendencies"][emotion] = 0.75  # Default strength
            
            # Sort emotions by strength
            sorted_emotions = sorted(
                indicators["emotional_tendencies"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get primary and secondary emotions
            if sorted_emotions:
                indicators["primary_emotions"] = [sorted_emotions[0][0]]
                indicators["secondary_emotions"] = [e[0] for e in sorted_emotions[1:4]]
                
                # Add visual indicators (emojis)
                for emotion, _ in sorted_emotions[:5]:
                    if emotion in self.emotion_indicators:
                        indicators["visual_indicators"].append({
                            "emotion": emotion,
                            "indicator": self.emotion_indicators[emotion]
                        })
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error generating emotion indicators: {str(e)}")
            return {
                "error": str(e),
                "primary_emotions": ["neutral"],
                "secondary_emotions": [],
                "emotional_tendencies": {},
                "visual_indicators": [{"emotion": "neutral", "indicator": "😐"}]
            }
    
    async def _generate_interpretation(self, assessment_type: str, results: Dict[str, Any], emotional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate an interpretation of assessment results"""
        try:
            # Get user context from memory
            user_context = await self.memory.get("user_context", {})
            
            # Format emotional context
            emotional_context_str = "{}"
            if emotional_context:
                emotional_context_str = json.dumps(emotional_context, indent=2)
            
            # Generate interpretation using LLM
            llm_response = await self.llm.agenerate_messages([
                self.interpretation_prompt.format_messages(
                    assessment_type=assessment_type,
                    assessment_results=json.dumps(results, indent=2),
                    user_context=json.dumps(user_context, indent=2),
                    emotional_context=emotional_context_str
                )[0]
            ])
            
            # Parse response
            interpretation = self._parse_interpretation(llm_response.generations[0][0].text)
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error generating interpretation: {str(e)}")
            return {
                "error": str(e),
                "fallback_interpretation": "Unable to generate detailed interpretation at this time."
            }
    
    def _parse_interpretation(self, text: str) -> Dict[str, Any]:
        """Parse the structured interpretation from LLM response"""
        sections = {
            "key_insights": [],
            "strengths": [],
            "growth_areas": [],
            "communication_preferences": [],
            "stress_responses": [],
            "emotional_patterns": [],
            "mental_health_implications": []
        }
        
        current_section = None
        
        try:
            lines = text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                lower_line = line.lower()
                if "key insight" in lower_line or "overview" in lower_line:
                    current_section = "key_insights"
                elif "strength" in lower_line:
                    current_section = "strengths"
                elif "growth" in lower_line or "challenge" in lower_line or "development" in lower_line:
                    current_section = "growth_areas"
                elif "communication" in lower_line or "learning" in lower_line:
                    current_section = "communication_preferences"
                elif "stress" in lower_line or "coping" in lower_line:
                    current_section = "stress_responses"
                elif "emotion" in lower_line or "feeling" in lower_line:
                    current_section = "emotional_patterns"
                elif "mental health" in lower_line or "wellbeing" in lower_line or "well-being" in lower_line:
                    current_section = "mental_health_implications"
                elif current_section and line.startswith('-'):
                    # Add bullet points to current section
                    sections[current_section].append(line[1:].strip())
                elif current_section and not any(header in lower_line for header in ["key", "strength", "growth", "communication", "stress", "emotion", "mental"]):
                    # Add non-header text to current section
                    sections[current_section].append(line)
        except Exception:
            # If parsing fails, return the raw text
            return {
                "raw_interpretation": text,
                "parsing_error": True
            }
            
        return sections
    
    async def get_previous_assessment(self) -> Dict[str, Any]:
        """Retrieve the most recent assessment results from memory"""
        try:
            return await self.memory.get("assessment_results", {})
        except Exception as e:
            logger.warning(f"Failed to retrieve assessment from memory: {str(e)}")
            return {}
    
    async def update_user_context(self, context: Dict[str, Any]) -> None:
        """Update the user context stored in memory"""
        try:
            await self.memory.add("user_context", context)
        except Exception as e:
            logger.warning(f"Failed to update user context: {str(e)}")
            
    async def integrate_emotion_data(self, emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate emotion data with personality assessment
        
        Args:
            emotion_data: Emotion analysis data from the EmotionAgent
            
        Returns:
            Dictionary with integrated insights
        """
        try:
            # Get the most recent assessment results
            assessment = await self.get_previous_assessment()
            
            if not assessment or "results" not in assessment:
                return {
                    "error": "No personality assessment found to integrate with emotions",
                    "success": False
                }
            
            assessment_type = assessment.get("type", "unknown")
            results = assessment.get("results", {})
            
            # Generate an integrated interpretation
            interpretation = await self._generate_interpretation(
                assessment_type=assessment_type,
                results=results,
                emotional_context=emotion_data
            )
            
            # Find personality-emotion correlations
            correlations = self._find_emotion_personality_correlations(assessment_type, results, emotion_data)
            
            return {
                "success": True,
                "integrated_interpretation": interpretation,
                "personality_emotion_correlations": correlations,
                "assessment_data": assessment,
                "emotion_data": emotion_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error integrating emotion data: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _find_emotion_personality_correlations(self, assessment_type: str, personality_data: Dict[str, Any], emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find correlations between personality traits and emotional patterns
        
        Args:
            assessment_type: Type of personality assessment
            personality_data: Personality assessment results
            emotion_data: Emotion analysis data
            
        Returns:
            Dictionary with correlation insights
        """
        correlations = {
            "high_correlations": [],
            "moderate_correlations": [],
            "low_correlations": [],
            "interesting_patterns": []
        }
        
        try:
            # Get primary emotion from emotion data
            primary_emotion = emotion_data.get("primary_emotion", "neutral")
            intensity = emotion_data.get("intensity", 5)
            secondary_emotions = emotion_data.get("secondary_emotions", [])
            
            if assessment_type.lower() == 'big_five':
                traits = personality_data.get("scores", {})
                
                # Check for specific correlations based on the Big Five model
                
                # Neuroticism correlations with anxiety/negative emotions
                if "neuroticism" in traits:
                    neuroticism = traits.get("neuroticism", 0.5)
                    if neuroticism > 0.6 and primary_emotion in ["anxiety", "fear", "sadness", "stress"]:
                        correlations["high_correlations"].append({
                            "trait": "neuroticism",
                            "emotion": primary_emotion,
                            "description": f"High neuroticism often correlates with {primary_emotion} responses"
                        })
                    elif neuroticism < 0.4 and primary_emotion in ["calm", "relaxed", "content"]:
                        correlations["high_correlations"].append({
                            "trait": "emotional stability",
                            "emotion": primary_emotion,
                            "description": f"Low neuroticism (high emotional stability) often correlates with {primary_emotion} states"
                        })
                
                # Extraversion correlations with social emotions
                if "extraversion" in traits:
                    extraversion = traits.get("extraversion", 0.5)
                    if extraversion > 0.6 and primary_emotion in ["excitement", "joy", "enthusiasm"]:
                        correlations["high_correlations"].append({
                            "trait": "extraversion",
                            "emotion": primary_emotion,
                            "description": f"High extraversion often correlates with {primary_emotion} in social settings"
                        })
                
                # Openness correlations with curiosity/wonder
                if "openness" in traits:
                    openness = traits.get("openness", 0.5)
                    if openness > 0.6 and primary_emotion in ["curiosity", "wonder", "interest"]:
                        correlations["high_correlations"].append({
                            "trait": "openness",
                            "emotion": primary_emotion,
                            "description": f"High openness often correlates with {primary_emotion} toward new experiences"
                        })
                
                # Agreeableness correlations with empathy/compassion
                if "agreeableness" in traits:
                    agreeableness = traits.get("agreeableness", 0.5)
                    if agreeableness > 0.6 and primary_emotion in ["empathy", "compassion", "warmth"]:
                        correlations["high_correlations"].append({
                            "trait": "agreeableness",
                            "emotion": primary_emotion,
                            "description": f"High agreeableness often correlates with {primary_emotion} toward others"
                        })
                
            elif assessment_type.lower() == 'mbti':
                mbti_type = personality_data.get("type", "")
                
                if len(mbti_type) == 4:
                    # Check for specific correlations based on MBTI dimensions
                    
                    # Introversion/Extraversion correlations
                    if mbti_type[0] == "E" and primary_emotion in ["enthusiasm", "excitement", "joy"]:
                        correlations["high_correlations"].append({
                            "trait": "extraversion (E)",
                            "emotion": primary_emotion,
                            "description": f"Extraverted types often experience {primary_emotion} in social interactions"
                        })
                    elif mbti_type[0] == "I" and primary_emotion in ["contentment", "calm", "reflection"]:
                        correlations["high_correlations"].append({
                            "trait": "introversion (I)",
                            "emotion": primary_emotion,
                            "description": f"Introverted types often experience {primary_emotion} during solitude"
                        })
                    
                    # Thinking/Feeling correlations
                    if mbti_type[2] == "F" and primary_emotion in ["empathy", "compassion", "connection"]:
                        correlations["high_correlations"].append({
                            "trait": "feeling (F)",
                            "emotion": primary_emotion,
                            "description": f"Feeling types often experience {primary_emotion} in interpersonal situations"
                        })
                    
                    # Judging/Perceiving correlations with stress
                    if mbti_type[3] == "J" and primary_emotion in ["anxiety", "stress"] and "unpredictability" in emotion_data.get("triggers", []):
                        correlations["high_correlations"].append({
                            "trait": "judging (J)",
                            "emotion": primary_emotion,
                            "description": f"Judging types may experience {primary_emotion} in unpredictable situations"
                        })
            
            # Add an interesting pattern if no high correlations were found
            if not correlations["high_correlations"]:
                correlations["interesting_patterns"].append({
                    "description": f"Your current emotional state ({primary_emotion}) doesn't show obvious correlations with your personality profile, suggesting it may be more situational than trait-based."
                })
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error finding emotion-personality correlations: {str(e)}")
            return {
                "error": str(e),
                "high_correlations": [],
                "moderate_correlations": [],
                "low_correlations": [],
                "interesting_patterns": []
            }
