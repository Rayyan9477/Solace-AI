"""
Integrated Mental Health Support Chatbot Application
This serves as the main entry point for the application, integrating all components.
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, Any, Optional, List
import os
import time

# Import UI components
from src.ui.ui_manager import UIComponentManager
from src.ui.landing_page import LandingPageComponent
from src.ui.conversational_assessment import ConversationalAssessmentComponent
from src.ui.results_display import ResultsDisplayComponent
from src.ui.chat_interface import ChatInterfaceComponent

# Import application components
from src.config.settings import AppConfig
from src.agents.agent_orchestrator import AgentOrchestrator
from src.utils.voice_ai import VoiceManager
from src.components.voice_component import VoiceComponent
from src.agents.integrated_diagnosis_agent import IntegratedDiagnosisAgent
from src.agents.chat_agent import ChatAgent
from src.agents.emotion_agent import EmotionAgent
from src.agents.safety_agent import SafetyAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MentalHealthApp:
    """Main application class that orchestrates the UI and business logic"""
    
    def __init__(self):
        """Initialize the application"""
        self.ui_manager = UIComponentManager()
        self.components = {}
        self.voice_enabled = False
        self.voice_component = None
        
        # Set page config
        st.set_page_config(
            page_title="Mental Health Support Assistant",
            page_icon="🧠",
            layout="wide"
        )
        
        # Initialize application state
        if "app_initialized" not in st.session_state:
            st.session_state.app_initialized = False
            st.session_state.assessment_results = None
    
    async def initialize(self):
        """Initialize application components"""
        with st.spinner("Initializing application..."):
            # Initialize application components
            try:
                self.components = await self._initialize_components()
                st.session_state.app_initialized = True
                logger.info("Application initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing application: {str(e)}")
                st.error(f"Error initializing application: {str(e)}")
    
    async def _initialize_components(self) -> Dict[str, Any]:
        """Initialize all application components"""
        components = {}
        
        # Check for API key
        api_key = AppConfig.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY is missing from your .env file or environment variables. Please set it and restart the app.")
            st.stop()

        try:
            # Initialize voice features
            voice_manager = VoiceManager()
            voice_result = await voice_manager.initialize()
            
            if voice_result["success"]:
                self.voice_enabled = True
                self.voice_component = VoiceComponent(voice_ai=voice_result["voice_ai"])
                components["voice_ai"] = voice_result["voice_ai"]
                logger.info("Voice features initialized successfully")
            else:
                self.voice_enabled = False
                logger.warning(f"Voice features not available: {voice_result.get('error', 'Unknown error')}")
                
                # Show warning for missing dependencies
                if "missing_dependencies" in voice_result:
                    missing = voice_result["missing_dependencies"]
                    st.warning(
                        f"Voice features are disabled due to missing dependencies: {', '.join(missing)}. "
                        "To enable voice features, install the required packages."
                    )
            
            # Initialize core agents
            # Use our new Gemini LLM implementation instead of the generic LLMProvider
            from src.models.gemini_llm import GeminiLLM
            
            # Create a shared Gemini LLM instance for consistency across agents
            gemini_llm = GeminiLLM(api_key=api_key)
            
            # Initialize the diagnosis agent with Gemini
            diagnosis_agent = IntegratedDiagnosisAgent(
                model=gemini_llm,
                temperature=0.2
            )
            components["integrated_diagnosis"] = diagnosis_agent
            
            # Initialize the chat agent with Gemini
            chat_agent = ChatAgent(
                model=gemini_llm,
                temperature=0.7
            )
            components["chat"] = chat_agent
            
            # Initialize the emotion agent with Gemini
            emotion_agent = EmotionAgent(
                model=gemini_llm,
                temperature=0.2
            )
            components["emotion"] = emotion_agent
            
            # Initialize the safety agent with Gemini
            safety_agent = SafetyAgent(
                model=gemini_llm,
                temperature=0.1
            )
            components["safety"] = safety_agent
            
            # Initialize the agent orchestrator
            orchestrator = AgentOrchestrator(
                chat_agent=chat_agent,
                emotion_agent=emotion_agent,
                safety_agent=safety_agent
            )
            components["orchestrator"] = orchestrator
            
            # Register UI components
            self._register_ui_components(components)
            
            return components
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def _register_ui_components(self, components: Dict[str, Any]):
        """Register UI components with the UI manager"""
        # Landing page
        landing_page = LandingPageComponent(
            on_start_voice=lambda: self._handle_start_mode("voice"),
            on_start_text=lambda: self._handle_start_mode("text"),
            voice_enabled=self.voice_enabled
        )
        self.ui_manager.register_component("landing", landing_page)
        
        # Conversational assessment
        conversational_assessment = ConversationalAssessmentComponent(
            assessment_agent=components.get("integrated_diagnosis"),
            on_complete=self._handle_assessment_complete,
            voice_enabled=self.voice_enabled,
            voice_component=self.voice_component
        )
        self.ui_manager.register_component("assessment", conversational_assessment)
        
        # Results display
        results_display = ResultsDisplayComponent(
            on_continue=lambda: self.ui_manager.navigate_to("chat"),
            voice_enabled=self.voice_enabled
        )
        self.ui_manager.register_component("results", results_display)
        
        # Chat interface
        chat_interface = ChatInterfaceComponent(
            process_message_callback=self._process_message,
            on_back=lambda: self.ui_manager.navigate_to("results"),
            voice_enabled=self.voice_enabled,
            voice_component=self.voice_component
        )
        self.ui_manager.register_component("chat", chat_interface)
    
    def _handle_start_mode(self, mode: str):
        """Handle starting in a specific mode (voice or text)"""
        self.ui_manager.set_interaction_mode(mode)
        self.ui_manager.navigate_to("assessment")
    
    def _handle_assessment_complete(self, assessment_results: Dict[str, Any]):
        """Handle completion of the assessment"""
        st.session_state.assessment_results = assessment_results
        
        # For demo purposes, generate some placeholder results if none provided
        if not assessment_results or (
            not assessment_results.get("mental_health_responses") and 
            not assessment_results.get("personality_responses")
        ):
            st.session_state.assessment_results = self._generate_placeholder_results()
            
        # Navigate to results page
        self.ui_manager.navigate_to("results")
    
    async def _process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message using the agent orchestrator"""
        if "orchestrator" in self.components:
            try:
                orchestrator = self.components["orchestrator"]
                result = await orchestrator.process_message(message)
                return result
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                return {"error": str(e)}
        else:
            return {
                "response": "I'm unable to process your message at the moment. The system is still initializing.",
                "error": "System not fully initialized"
            }
    
    def _generate_placeholder_results(self) -> Dict[str, Any]:
        """Generate placeholder results for demo purposes"""
        return {
            "personality": {
                "traits": {
                    "extraversion": {"score": 65, "category": "high"},
                    "agreeableness": {"score": 78, "category": "high"},
                    "conscientiousness": {"score": 58, "category": "average"},
                    "neuroticism": {"score": 45, "category": "average"},
                    "openness": {"score": 82, "category": "high"}
                }
            },
            "mental_health": {
                "overall_status": "mild",
                "scores": {
                    "depression": 25,
                    "anxiety": 30,
                    "stress": 40,
                    "sleep": 35,
                    "social": 20,
                    "cognitive": 15,
                    "physical": 25
                },
                "severity_levels": {
                    "depression": "mild",
                    "anxiety": "mild",
                    "stress": "moderate",
                    "sleep": "moderate",
                    "social": "mild",
                    "cognitive": "mild",
                    "physical": "mild"
                }
            }
        }
    
    def _generate_recommended_actions(self, assessment_results: Dict[str, Any]) -> List[str]:
        """
        Generate personalized recommended actions based on assessment results
        
        Args:
            assessment_results: Dictionary containing assessment results
            
        Returns:
            List of recommended actions
        """
        actions = []
        
        # Get mental health data
        mental_health = assessment_results.get("mental_health", {})
        severity_levels = mental_health.get("severity_levels", {})
        
        # Add general action for everyone
        actions.append("Continue the conversation to explore specific areas of concern")
        
        # Add journaling recommendation for most people
        actions.append("Consider trying a daily journaling practice to track your mood and thoughts")
        
        # Add specific recommendations based on assessment
        if severity_levels.get("stress", "mild") in ["moderate", "severe"]:
            actions.append("Explore stress management techniques like deep breathing or progressive muscle relaxation")
        
        if severity_levels.get("sleep", "mild") in ["moderate", "severe"]:
            actions.append("Set a consistent sleep schedule to help address sleep difficulties")
        
        if severity_levels.get("anxiety", "mild") in ["moderate", "severe"]:
            actions.append("Practice mindfulness or meditation to help manage anxiety")
        
        if severity_levels.get("depression", "mild") in ["moderate", "severe"]:
            actions.append("Create a daily structure with small, achievable goals")
        
        if severity_levels.get("social", "mild") in ["moderate", "severe"]:
            actions.append("Reach out to a friend or family member for a brief conversation today")
        
        # Add self-care recommendation for everyone
        actions.append("Make time for a small self-care activity that you enjoy today")
        
        # If many severe indicators, suggest professional support
        severe_count = sum(1 for severity in severity_levels.values() if severity == "severe")
        if severe_count >= 2:
            actions.append("Consider speaking with a mental health professional for additional support")
        
        # Return maximum 5 actions
        return actions[:5]
    
    async def run(self):
        """Run the application"""
        # Run async initialization if not already done
        if not st.session_state.app_initialized:
            asyncio.run(self.initialize())
            
            # Exit if initialization failed
            if not st.session_state.app_initialized:
                return
        
        # Check if we're on the results page and have assessment results
        if (self.ui_manager.current_route == "results" or 
            st.session_state.ui_manager_state.get("current_route") == "results"):
            
            if st.session_state.assessment_results:
                # Get the results component
                results_component = self.ui_manager.components.get("results")
                if results_component:
                    # Use Gemini to generate empathetic response
                    empathy_response = await results_component.generate_empathetic_response(
                        st.session_state.assessment_results
                    )
                    
                    # Generate recommended actions based on assessment results
                    immediate_actions = self._generate_recommended_actions(
                        st.session_state.assessment_results
                    )
                    
                    # Render results page with personalized Gemini-generated response
                    results_component.render(
                        assessment_results=st.session_state.assessment_results,
                        empathy_response=empathy_response,
                        immediate_actions=immediate_actions
                    )
                return
        
        # For other routes, use the UI manager to render the current component
        self.ui_manager.render_current()

# Main entry point
if __name__ == "__main__":
    app = MentalHealthApp()
    asyncio.run(app.run())