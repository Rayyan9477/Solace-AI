"""
Dynamic personality assessment component for API and CLI interfaces.
Supports personality assessment through API and CLI.
"""

import time
import random
import sys
import os
import logging
from typing import Dict, Any, List, Optional, Callable

# Add path to import personality assessment modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from personality.big_five import BigFiveAssessment
from personality.mbti import MBTIAssessment

logger = logging.getLogger(__name__)

class DynamicPersonalityAssessment:
    """
    Component for conducting dynamic, conversational personality assessments
    with voice and emotion integration - refactored for API/CLI use
    """
    
    def __init__(
        self, 
        personality_agent,
        voice_component=None,
        emotion_agent=None,
        on_complete=None
    ):
        """
        Initialize the dynamic personality assessment component
        
        Args:
            personality_agent: PersonalityAgent instance for assessment logic
            voice_component: Optional VoiceComponent instance
            emotion_agent: Optional EmotionAgent for emotional analysis
            on_complete: Callback function to call when assessment is complete
        """
        self.personality_agent = personality_agent
        self.voice_component = voice_component
        self.emotion_agent = emotion_agent
        self.on_complete_callback = on_complete
        
        # Set up assessment state
        self.state = {
            "assessment_type": None,  # 'big_five' or 'mbti'
            "conversation_mode": False,  # True for conversational, False for traditional
            "current_question_index": 0,
            "responses": {},
            "conversation_history": [],
            "emotion_data": {},
            "adaptations": {},
            "voice_mode": False,
            "current_traits_focus": [],  # Traits currently being assessed
            "assessment_progress": 0.0,  # Progress from 0.0 to 1.0
            "user_context": {},  # Additional context about the user
            "last_emotion": "neutral"
        }
        
        # Initialize the assessment managers
        self.big_five_assessment = BigFiveAssessment()
        self.mbti_assessment = MBTIAssessment()
        
        # Set up adapters for conversational flow
        self.trait_conversation_starters = {
            # Big Five conversation starters
            "extraversion": [
                "Tell me about the last time you were at a large social gathering. How did you feel?",
                "Do you find yourself energized after spending time with others, or do you need time alone to recharge?",
                "When faced with a group activity, do you prefer to take the lead or observe first?"
            ],
            "agreeableness": [
                "How do you typically handle disagreements with friends or colleagues?",
                "When someone asks for your help with something inconvenient, what's your usual response?",
                "What's your approach to giving feedback when someone has made a mistake?"
            ],
            "conscientiousness": [
                "How do you approach deadlines and planning for important tasks?",
                "Describe your ideal workspace or environment for getting things done.",
                "When starting a new project, what's your usual process?"
            ],
            "neuroticism": [
                "How do you typically respond to unexpected changes or disruptions to your plans?",
                "Tell me about how you handle stressful situations.",
                "What helps you feel calm when you're worried about something?"
            ],
            "openness": [
                "What's your approach to trying new experiences or unfamiliar activities?",
                "How important is it for you to have variety in your day-to-day life?",
                "When you encounter ideas that challenge your worldview, how do you typically respond?"
            ],
            
            # MBTI conversation starters
            "E/I": [
                "How do you prefer to spend your free time - with others or by yourself?",
                "When solving a problem, do you think better by talking it through with someone or reflecting on your own?",
                "What feels more natural to you - having many shorter conversations or a few deeper ones?"
            ],
            "S/N": [
                "When learning something new, do you prefer practical examples or understanding the theory behind it?",
                "When planning a trip, what aspects do you focus on the most?",
                "Do you find yourself more interested in what is actually happening now or what could happen in the future?"
            ],
            "T/F": [
                "How do you typically make important decisions in your life?",
                "When a friend comes to you with a problem, what's your first instinct - to offer solutions or emotional support?",
                "In a debate, what matters more to you - the logical consistency of your argument or maintaining harmony in the group?"
            ],
            "J/P": [
                "How far in advance do you usually plan your activities?",
                "How do you feel about last-minute changes to your plans?",
                "Do you prefer having a structured routine or keeping your options open?"
            ]
        }

    def set_assessment_type(self, assessment_type: str) -> Dict[str, Any]:
        """
        Set the assessment type (big_five or mbti)
        
        Args:
            assessment_type: Type of assessment to conduct
            
        Returns:
            Updated state information
        """
        if assessment_type not in ["big_five", "mbti"]:
            return {"success": False, "error": "Invalid assessment type"}
        
        self.state["assessment_type"] = assessment_type
        
        # Set initial traits focus based on assessment type
        if assessment_type == "big_five":
            self.state["current_traits_focus"] = ["extraversion", "openness"]
        else:  # mbti
            self.state["current_traits_focus"] = ["E/I", "S/N"]
        
        return {
            "success": True, 
            "state": self.get_state(),
            "message": f"Assessment type set to {assessment_type}"
        }
    
    def set_conversation_mode(self, conversation_mode: bool) -> Dict[str, Any]:
        """
        Set whether to use conversational mode
        
        Args:
            conversation_mode: Whether to use conversational mode
            
        Returns:
            Updated state information
        """
        self.state["conversation_mode"] = conversation_mode
        
        return {
            "success": True,
            "state": self.get_state(),
            "message": f"Conversation mode set to {conversation_mode}"
        }
    
    def set_voice_mode(self, voice_mode: bool) -> Dict[str, Any]:
        """
        Set whether to use voice mode
        
        Args:
            voice_mode: Whether to use voice mode
            
        Returns:
            Updated state information
        """
        # Check if voice component is available
        if voice_mode and not self.voice_component:
            return {"success": False, "error": "Voice component not available"}
        
        self.state["voice_mode"] = voice_mode
        
        return {
            "success": True,
            "state": self.get_state(),
            "message": f"Voice mode set to {voice_mode}"
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current assessment state
        
        Returns:
            Current assessment state
        """
        return self.state.copy()
    
    def get_next_question(self) -> Dict[str, Any]:
        """
        Get the next question based on the current state
        
        Returns:
            Question data with text, options, etc.
        """
        try:
            # If assessment type not set, return error
            if not self.state["assessment_type"]:
                return {"success": False, "error": "Assessment type not set"}
            
            # Get the appropriate assessment manager
            assessment = self.big_five_assessment if self.state["assessment_type"] == "big_five" else self.mbti_assessment
            
            # Get questions from the assessment
            if self.state["conversation_mode"]:
                # Get a conversation starter for the current trait focus
                trait = random.choice(self.state["current_traits_focus"])
                starters = self.trait_conversation_starters.get(trait, [])
                
                if not starters:
                    return {"success": False, "error": f"No conversation starters for trait {trait}"}
                
                question = {
                    "id": f"{trait}_{self.state['current_question_index']}",
                    "text": random.choice(starters),
                    "trait": trait,
                    "type": "open_ended"
                }
            else:
                # Get a structured question
                questions = assessment.get_questions()
                
                if self.state["current_question_index"] >= len(questions):
                    return {"success": False, "error": "No more questions available", "assessment_complete": True}
                
                question = questions[self.state["current_question_index"]]
            
            self.state["current_question_index"] += 1
            self.state["assessment_progress"] = min(1.0, self.state["current_question_index"] / 20)  # Assuming 20 questions
            
            return {
                "success": True,
                "question": question,
                "progress": self.state["assessment_progress"],
                "traits_focus": self.state["current_traits_focus"]
            }
            
        except Exception as e:
            logger.error(f"Error getting next question: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def submit_response(self, question_id: str, response: Any) -> Dict[str, Any]:
        """
        Submit a response to a question
        
        Args:
            question_id: ID of the question
            response: Response value or text
            
        Returns:
            Status and next steps
        """
        try:
            # Store the response
            self.state["responses"][question_id] = response
            
            # If conversational mode, add to conversation history
            if self.state["conversation_mode"]:
                self.state["conversation_history"].append({
                    "role": "user",
                    "content": response if isinstance(response, str) else str(response)
                })
            
            # Check if assessment is complete
            assessment_complete = self.state["assessment_progress"] >= 1.0
            
            # If using emotion agent, analyze response
            emotion_data = {}
            if self.emotion_agent and isinstance(response, str):
                try:
                    emotion_data = self.emotion_agent.analyze_text(response)
                    self.state["emotion_data"][question_id] = emotion_data
                    self.state["last_emotion"] = emotion_data.get("primary_emotion", "neutral")
                except Exception as e:
                    logger.error(f"Error analyzing emotion: {str(e)}")
            
            # Update traits focus based on responses
            if len(self.state["responses"]) % 5 == 0:
                self._update_traits_focus()
            
            # If assessment is complete, process results
            if assessment_complete:
                return self._process_assessment_results()
            
            return {
                "success": True,
                "state": self.get_state(),
                "emotion_data": emotion_data,
                "assessment_complete": assessment_complete,
                "next_question": self.get_next_question() if not assessment_complete else None
            }
            
        except Exception as e:
            logger.error(f"Error submitting response: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _update_traits_focus(self):
        """Update which traits to focus on based on responses so far"""
        if self.state["assessment_type"] == "big_five":
            all_traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
            
            # Choose 2 traits that have the fewest responses
            trait_counts = {trait: 0 for trait in all_traits}
            
            for question_id in self.state["responses"]:
                if "_" in question_id:
                    trait = question_id.split("_")[0]
                    if trait in trait_counts:
                        trait_counts[trait] += 1
            
            # Sort traits by count
            sorted_traits = sorted(trait_counts.items(), key=lambda x: x[1])
            self.state["current_traits_focus"] = [trait for trait, count in sorted_traits[:2]]
        else:  # mbti
            all_dimensions = ["E/I", "S/N", "T/F", "J/P"]
            
            # Choose 2 dimensions that have the fewest responses
            dimension_counts = {dim: 0 for dim in all_dimensions}
            
            for question_id in self.state["responses"]:
                if "_" in question_id:
                    dim = question_id.split("_")[0]
                    if dim in dimension_counts:
                        dimension_counts[dim] += 1
            
            # Sort dimensions by count
            sorted_dims = sorted(dimension_counts.items(), key=lambda x: x[1])
            self.state["current_traits_focus"] = [dim for dim, count in sorted_dims[:2]]
    
    def _process_assessment_results(self) -> Dict[str, Any]:
        """Process the assessment results"""
        try:
            # Use the personality agent to process the results
            if self.personality_agent:
                results = self.personality_agent.process_assessment_results(
                    self.state["assessment_type"],
                    self.state["responses"],
                    self.state["conversation_history"] if self.state["conversation_mode"] else [],
                    self.state["emotion_data"]
                )
            else:
                # If no personality agent, use the assessment manager to process
                assessment = self.big_five_assessment if self.state["assessment_type"] == "big_five" else self.mbti_assessment
                results = assessment.calculate_results(self.state["responses"])
            
            # Call the on_complete callback if provided
            if self.on_complete_callback:
                self.on_complete_callback(results)
            
            return {
                "success": True,
                "assessment_complete": True,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing assessment results: {str(e)}")
            return {"success": False, "error": str(e), "assessment_complete": True}
    
    def reset_assessment(self) -> Dict[str, Any]:
        """
        Reset the assessment to start over
        
        Returns:
            Status of reset operation
        """
        try:
            # Reset the state
            self.state = {
                "assessment_type": None,
                "conversation_mode": False,
                "current_question_index": 0,
                "responses": {},
                "conversation_history": [],
                "emotion_data": {},
                "adaptations": {},
                "voice_mode": False,
                "current_traits_focus": [],
                "assessment_progress": 0.0,
                "user_context": {},
                "last_emotion": "neutral"
            }
            
            return {"success": True, "message": "Assessment reset successfully"}
            
        except Exception as e:
            logger.error(f"Error resetting assessment: {str(e)}")
            return {"success": False, "error": str(e)}
