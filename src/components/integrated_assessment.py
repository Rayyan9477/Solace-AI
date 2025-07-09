"""
Integrated assessment component for the mental health app.
Combines mental health and personality assessment in a single interface.
"""

import json
import os
import logging
from typing import Dict, Any, Callable, Optional, List
import asyncio

logger = logging.getLogger(__name__)

class IntegratedAssessmentComponent:
    """Component for integrated assessment interaction with clients"""
    
    def __init__(self, on_complete: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the integrated assessment component
        
        Args:
            on_complete: Callback function to call when assessment is complete
        """
        self.on_complete = on_complete
        self.questions = self._load_questions()
        self.assessment_state = {
            "step": 1,  # 1: Mental Health, 2: Personality, 3: Results
            "mental_health_responses": {},
            "personality_responses": {}
        }
        
    def _load_questions(self) -> Dict[str, Any]:
        """Load assessment questions from file"""
        try:
            # Try to load questions from the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'data', 'personality')
            
            questions_path = os.path.join(data_dir, 'diagnosis_questions.json')
            
            with open(questions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading assessment questions: {str(e)}")
            # Return empty questions as fallback
            return {"mental_health": [], "personality": []}
    
    def get_mental_health_questions(self) -> List[Dict[str, Any]]:
        """Get mental health assessment questions"""
        return self.questions.get("mental_health", [])
    
    def get_personality_questions(self) -> List[Dict[str, Any]]:
        """Get personality assessment questions"""
        return self.questions.get("personality", [])
    
    def process_mental_health_responses(self, responses: Dict[str, Any]) -> bool:
        """
        Process mental health assessment responses
        
        Args:
            responses: Dict of question IDs to response values
            
        Returns:
            Success status
        """
        try:
            # Store responses
            self.assessment_state["mental_health_responses"] = responses
            
            # Move to next step
            self.assessment_state["step"] = 2
            
            return True
        except Exception as e:
            logger.error(f"Error processing mental health responses: {str(e)}")
            return False
    
    def process_personality_responses(self, responses: Dict[str, Any]) -> bool:
        """
        Process personality assessment responses
        
        Args:
            responses: Dict of question IDs to response values
            
        Returns:
            Success status
        """
        try:
            # Store responses
            self.assessment_state["personality_responses"] = responses
            
            # Move to next step
            self.assessment_state["step"] = 3
            
            # Call the callback if provided
            if self.on_complete:
                assessment_data = {
                    "mental_health_responses": self.assessment_state["mental_health_responses"],
                    "personality_responses": self.assessment_state["personality_responses"]
                }
                self.on_complete(assessment_data)
            
            return True
        except Exception as e:
            logger.error(f"Error processing personality responses: {str(e)}")
            return False
    
    def reset_assessment(self) -> bool:
        """
        Reset the assessment to the beginning
        
        Returns:
            Success status
        """
        try:
            # Reset responses
            self.assessment_state = {
                "step": 1,
                "mental_health_responses": {},
                "personality_responses": {}
            }
            
            return True
        except Exception as e:
            logger.error(f"Error resetting assessment: {str(e)}")
            return False
    
    def get_assessment_state(self) -> Dict[str, Any]:
        """
        Get the current assessment state
        
        Returns:
            Current assessment state
        """
        return self.assessment_state.copy()

class IntegratedAssessment:
    """Class for managing integrated mental health assessments"""
    
    def __init__(self, diagnosis_agent=None):
        """
        Initialize the integrated assessment
        
        Args:
            diagnosis_agent: Agent for processing assessment results
        """
        self.diagnosis_agent = diagnosis_agent
        
    async def conduct_assessment(self, mental_health_responses, personality_responses):
        """
        Process the assessment responses
        
        Args:
            mental_health_responses: Dict of mental health assessment responses
            personality_responses: Dict of personality assessment responses
            
        Returns:
            Assessment results
        """
        try:
            # If no diagnosis agent is available, return basic results
            if self.diagnosis_agent is None:
                return {
                    "success": True,
                    "assessment_results": {
                        "mental_health": self._basic_mental_health_analysis(mental_health_responses),
                        "personality": self._basic_personality_analysis(personality_responses)
                    }
                }
            
            # Use the diagnosis agent to process responses
            results = await self.diagnosis_agent.process_integrated_assessment(
                mental_health_responses,
                personality_responses
            )
            
            return {
                "success": True,
                "assessment_results": results
            }
        except Exception as e:
            logger.error(f"Error conducting assessment: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_empathy_response(self, assessment_results):
        """
        Generate an empathetic response based on assessment results
        
        Args:
            assessment_results: Dict containing assessment results
            
        Returns:
            String containing empathetic response
        """
        try:
            if self.diagnosis_agent is None:
                return self._default_empathy_response(assessment_results)
            
            return self.diagnosis_agent.generate_empathy_response(assessment_results)
        except Exception as e:
            logger.error(f"Error generating empathy response: {str(e)}")
            return "I appreciate you sharing your experiences with me. How are you feeling right now?"
    
    def generate_immediate_actions(self, assessment_results):
        """
        Generate immediate actions based on assessment results
        
        Args:
            assessment_results: Dict containing assessment results
            
        Returns:
            List of immediate action suggestions
        """
        try:
            if self.diagnosis_agent is None:
                return self._default_immediate_actions(assessment_results)
            
            return self.diagnosis_agent.generate_immediate_actions(assessment_results)
        except Exception as e:
            logger.error(f"Error generating immediate actions: {str(e)}")
            return [
                "Consider speaking with a mental health professional",
                "Practice deep breathing for 5 minutes",
                "Make sure you're getting adequate rest"
            ]
    
    def _basic_mental_health_analysis(self, responses):
        """Basic analysis of mental health responses"""
        if not responses:
            return {"risk_level": "unknown"}
        
        # Simple scoring mechanism
        total_score = sum(int(value) for value in responses.values())
        avg_score = total_score / len(responses) if responses else 0
        
        if avg_score >= 3:
            risk_level = "high"
        elif avg_score >= 2:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "avg_score": avg_score,
            "total_score": total_score,
            "num_questions": len(responses)
        }
    
    def _basic_personality_analysis(self, responses):
        """Basic analysis of personality responses"""
        if not responses:
            return {}
        
        # Group by personality traits (assuming key format: "trait_name_number")
        traits = {}
        for key, value in responses.items():
            # Extract trait name if available in key
            trait = "general"
            if "_" in key:
                trait = key.split("_")[0]
            
            if trait not in traits:
                traits[trait] = []
            
            traits[trait].append(int(value))
        
        # Calculate averages
        trait_scores = {}
        for trait, values in traits.items():
            trait_scores[trait] = sum(values) / len(values) if values else 0
        
        return {
            "trait_scores": trait_scores
        }
    
    def _default_empathy_response(self, assessment_results):
        """Generate default empathy response"""
        if not assessment_results:
            return "Thank you for completing the assessment. I'm here to support you."
        
        mental_health = assessment_results.get("mental_health", {})
        risk_level = mental_health.get("risk_level", "unknown")
        
        if risk_level == "high":
            return "I can see you're going through a difficult time. It takes courage to share these feelings, and I appreciate your honesty. Remember that seeking help is a sign of strength, not weakness."
        elif risk_level == "moderate":
            return "Thank you for sharing. It sounds like you've been experiencing some challenges lately. It's important to acknowledge these feelings and know that support is available."
        else:
            return "Thank you for completing the assessment. I'm here to listen and support you in whatever you're experiencing."
    
    def _default_immediate_actions(self, assessment_results):
        """Generate default immediate actions"""
        if not assessment_results:
            return [
                "Consider speaking with a mental health professional",
                "Practice mindfulness meditation daily",
                "Maintain a consistent sleep schedule"
            ]
        
        mental_health = assessment_results.get("mental_health", {})
        risk_level = mental_health.get("risk_level", "unknown")
        
        if risk_level == "high":
            return [
                "Please consider speaking with a mental health professional as soon as possible",
                "Contact a crisis support line if you're feeling overwhelmed",
                "Practice deep breathing exercises when feeling anxious",
                "Reach out to a trusted friend or family member",
                "Prioritize basic self-care: sleep, nutrition, and rest"
            ]
        elif risk_level == "moderate":
            return [
                "Consider scheduling an appointment with a mental health professional",
                "Practice daily mindfulness or meditation for 10 minutes",
                "Maintain a regular sleep schedule",
                "Engage in moderate physical activity",
                "Identify and limit exposure to stress triggers"
            ]
        else:
            return [
                "Continue practicing self-care routines",
                "Engage in regular physical activity",
                "Maintain social connections",
                "Practice mindfulness techniques",
                "Monitor your mood and energy levels"
            ]
