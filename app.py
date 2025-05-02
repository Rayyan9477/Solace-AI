"""
Mental Health Support Chatbot Application
Main entry point for the application
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, Any, Optional
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
from src.personality import PersonalityManager
from src.utils.metrics import metrics_manager
# Import Whisper ASR components
from src.utils.whisper_asr import WhisperASR
from src.utils.voice_input_manager import VoiceInputManager

# Setup logging
logging.basicConfig(
    level=getattr(logging, AppConfig.LOG_LEVEL),
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
            page_title=AppConfig.APP_NAME,
            page_icon="🧠",
            layout="wide"
        )
        
        # Initialize application state
        if "app_initialized" not in st.session_state:
            st.session_state.app_initialized = False
            st.session_state.assessment_results = None
            st.session_state.personality_manager = None
    
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
        from src.main import initialize_components
        
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
                
                # Initialize Whisper V3 Turbo ASR
                try:
                    whisper_model = "turbo"  # Use turbo by default
                    logger.info(f"Initializing Whisper {whisper_model} ASR...")
                    
                    # Initialize VoiceInputManager with Whisper ASR
                    voice_input_manager = VoiceInputManager(model_name=whisper_model)
                    components["voice_input_manager"] = voice_input_manager
                    
                    # Enhanced voice capability message
                    st.session_state["whisper_asr_enabled"] = True
                    logger.info(f"Whisper {whisper_model} ASR initialized successfully")
                except Exception as whisper_error:
                    logger.warning(f"Whisper ASR initialization failed: {str(whisper_error)}")
                    st.session_state["whisper_asr_enabled"] = False
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
            
            # Initialize all other components from main.py
            main_components = initialize_components() 
            
            # Add the agent components to our components dictionary
            components.update(main_components)
            
            # Initialize personality manager
            personality_manager = PersonalityManager()
            components["personality_manager"] = personality_manager
            st.session_state.personality_manager = personality_manager
            
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
            voice_component=self.voice_component,
            whisper_voice_input=components.get("voice_input_manager") if st.session_state.get("whisper_asr_enabled", False) else None
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
            voice_component=self.voice_component,
            # Add Whisper V3 Turbo voice input manager if available
            whisper_voice_input=components.get("voice_input_manager") if st.session_state.get("whisper_asr_enabled", False) else None
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
            
        # Track assessment completed
        metrics_manager.track_interaction("assessment_completed")
            
        # Navigate to results page
        self.ui_manager.navigate_to("results")
    
    async def _process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message using the agent orchestrator"""
        if "orchestrator" in self.components:
            try:
                # Track the interaction
                metrics_manager.track_interaction("chat_message")
                
                # Start timing for response
                start_time = time.time()
                
                # Process the message
                orchestrator = self.components["orchestrator"]
                result = await orchestrator.process_message(message)
                
                # Track response time
                end_time = time.time()
                metrics_manager.track_response_time(end_time - start_time)
                
                # Track any emotions detected
                if "emotion_analysis" in result and "primary_emotion" in result["emotion_analysis"]:
                    emotion = result["emotion_analysis"]["primary_emotion"]
                    intensity = result["emotion_analysis"].get("intensity", 5)
                    metrics_manager.track_emotion(emotion, intensity)
                
                # Track safety flags if any
                if result.get("requires_escalation", False):
                    metrics_manager.track_safety_flag("critical")
                
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
    
    def run(self):
        """Run the application"""
        # Run async initialization if not already done
        if not st.session_state.app_initialized:
            asyncio.run(self.initialize())
            
            # Exit if initialization failed
            if not st.session_state.app_initialized:
                return
        
        # Display a status indicator for Whisper ASR if enabled
        if st.session_state.get("whisper_asr_enabled", False):
            st.sidebar.success("🚀 Whisper V3 Turbo ASR Enabled")
            st.sidebar.markdown("""
            <div style='background-color: #f0f7ff; padding: 10px; border-radius: 10px; margin-top: 10px;'>
                <p style='margin: 0; font-size: 0.85rem;'>
                    <b>Enhanced Voice Recognition:</b> Whisper V3 Turbo provides state-of-the-art speech recognition accuracy.
                    <br><br>
                    Available in both chat interface and assessment components.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Check if we're on the results page and have assessment results
        if (self.ui_manager.current_route == "results" or 
            st.session_state.ui_manager_state.get("current_route") == "results"):
            
            if st.session_state.assessment_results:
                # Generate empathy response and immediate actions
                empathy_response = (
                    "Based on your responses, I can see that you're experiencing some moderate levels of "
                    "stress and sleep challenges, while showing resilience in other areas. Your personality "
                    "profile indicates someone who is generally sociable, compassionate, and open to new "
                    "experiences, which can be valuable strengths in managing your well-being."
                )
                
                immediate_actions = [
                    "Continue the conversation to explore specific areas of concern",
                    "Consider trying a simple daily journaling practice to track your mood",
                    "Explore stress management techniques like deep breathing or progressive muscle relaxation",
                    "Set a consistent sleep schedule to help address sleep difficulties"
                ]
                
                # Render results page with the data
                results_component = self.ui_manager.components.get("results")
                if results_component:
                    results_component.render(
                        assessment_results=st.session_state.assessment_results,
                        empathy_response=empathy_response,
                        immediate_actions=immediate_actions
                    )
                return
        
        # For other routes, use the UI manager to render the current component
        self.ui_manager.render_current()

# Application entry point
if __name__ == "__main__":
    app = MentalHealthApp()
    app.run()