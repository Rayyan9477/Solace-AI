"""
Integrated assessment component for the mental health app.
Combines mental health and personality assessment in a single interface.
"""

import streamlit as st
import json
import os
import logging
from typing import Dict, Any, Callable, Optional
import asyncio

logger = logging.getLogger(__name__)

class IntegratedAssessmentComponent:
    """Component for rendering the integrated assessment interface"""
    
    def __init__(self, on_complete: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the integrated assessment component
        
        Args:
            on_complete: Callback function to call when assessment is complete
        """
        self.on_complete = on_complete
        self.questions = self._load_questions()
        
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
    
    def render(self):
        """Render the integrated assessment interface"""
        st.header("Comprehensive Mental Health Assessment")
        
        # Initialize session state for assessment
        if "assessment_step" not in st.session_state:
            st.session_state["assessment_step"] = 1  # 1: Mental Health, 2: Personality, 3: Results
            
        if "mental_health_responses" not in st.session_state:
            st.session_state["mental_health_responses"] = {}
            
        if "personality_responses" not in st.session_state:
            st.session_state["personality_responses"] = {}
        
        # Render the appropriate step
        if st.session_state["assessment_step"] == 1:
            self._render_mental_health_assessment()
        elif st.session_state["assessment_step"] == 2:
            self._render_personality_assessment()
        elif st.session_state["assessment_step"] == 3:
            self._render_assessment_complete()
    
    def _render_mental_health_assessment(self):
        """Render the mental health assessment step"""
        st.subheader("Mental Health Check-In")
        st.markdown("""
        Please answer the following questions about how you've been feeling recently.
        Your responses will help us understand your current mental health status.
        """)
        
        with st.form("mental_health_form"):
            responses = {}
            
            for question in self.questions.get("mental_health", []):
                q_id = str(question["id"])
                q_text = question["text"]
                options = question.get("options", [])
                
                # Create a radio button for each question
                option_texts = [opt["text"] for opt in options]
                option_values = [opt["value"] for opt in options]
                
                # Use previous response if available
                default_idx = 0
                if q_id in st.session_state.get("mental_health_responses", {}):
                    prev_value = st.session_state["mental_health_responses"][q_id]
                    if prev_value in option_values:
                        default_idx = option_values.index(prev_value)
                
                response = st.radio(
                    f"{q_text}",
                    options=option_texts,
                    index=default_idx,
                    key=f"mh_{q_id}"
                )
                
                # Store the selected value
                selected_idx = option_texts.index(response)
                responses[q_id] = option_values[selected_idx]
            
            # Submit button
            if st.form_submit_button("Continue to Personality Assessment"):
                # Store responses in session state
                st.session_state["mental_health_responses"] = responses
                
                # Move to next step
                st.session_state["assessment_step"] = 2
                st.rerun()
    
    def _render_personality_assessment(self):
        """Render the personality assessment step"""
        st.subheader("Personality Assessment")
        st.markdown("""
        Please indicate how accurately each statement describes you.
        Your personality traits can help us provide more personalized support.
        """)
        
        with st.form("personality_form"):
            responses = {}
            
            for question in self.questions.get("personality", []):
                q_id = str(question["id"])
                q_text = question["text"]
                options = question.get("options", [])
                
                # Create a radio button for each question
                option_texts = [opt["text"] for opt in options]
                option_values = [opt["value"] for opt in options]
                
                # Use previous response if available
                default_idx = 2  # Default to neutral
                if q_id in st.session_state.get("personality_responses", {}):
                    prev_value = st.session_state["personality_responses"][q_id]
                    if prev_value in option_values:
                        default_idx = option_values.index(prev_value)
                
                response = st.radio(
                    f"{q_text}",
                    options=option_texts,
                    index=default_idx,
                    key=f"p_{q_id}"
                )
                
                # Store the selected value
                selected_idx = option_texts.index(response)
                responses[q_id] = option_values[selected_idx]
            
            # Submit button
            if st.form_submit_button("Complete Assessment"):
                # Store responses in session state
                st.session_state["personality_responses"] = responses
                
                # Move to next step
                st.session_state["assessment_step"] = 3
                
                # Call the callback if provided
                if self.on_complete:
                    assessment_data = {
                        "mental_health_responses": st.session_state["mental_health_responses"],
                        "personality_responses": st.session_state["personality_responses"]
                    }
                    self.on_complete(assessment_data)
                
                st.rerun()
    
    def _render_assessment_complete(self):
        """Render the assessment complete step"""
        st.success("Assessment completed successfully!")
        st.markdown("""
        Thank you for completing the assessment. Your responses will help us provide
        more personalized support tailored to your needs.
        """)
        
        # Show a spinner while processing results
        with st.spinner("Processing your results..."):
            # This would normally be handled by the callback
            pass
        
        # Option to retake assessment
        if st.button("Retake Assessment"):
            # Reset responses
            st.session_state["mental_health_responses"] = {}
            st.session_state["personality_responses"] = {}
            st.session_state["assessment_step"] = 1
            st.rerun()

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
