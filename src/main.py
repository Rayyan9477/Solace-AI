import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from config.settings import AppConfig
from langchain_community.docstore.document import Document
from database.vector_store import FaissVectorStore
from agents.chat_agent import ChatAgent
from agents.crawler_agent import CrawlerAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.emotion_agent import EmotionAgent
from agents.safety_agent import SafetyAgent
from agents.search_agent import SearchAgent
from agents.personality_agent import PersonalityAgent
from agents.integrated_diagnosis_agent import IntegratedDiagnosisAgent
from utils.metrics import track_metric
import sentry_sdk
from prometheus_client import start_http_server
import time
import asyncio
from models.llm import AgnoLLM as SafeLLM
import os
from typing import Dict, Any, List
from agents.agent_orchestrator import AgentOrchestrator
import logging
from datetime import datetime
import google.generativeai as genai
import torch
# Voice AI imports
from utils.voice_ai import VoiceAI, VoiceManager
from utils.voice_input_manager import VoiceInputManager
from utils.whisper_asr import WhisperASR
from components.voice_component import VoiceComponent
from components.integrated_assessment import IntegratedAssessmentComponent
from components.diagnosis_results import DiagnosisResultsComponent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name=s) - %(levelname=s) - %(message=s'
)
logger = logging.getLogger(__name__)

# Configure CUDA/CPU device
if torch.cuda.is_available():
    try:
        # Test CUDA initialization
        torch.cuda.init()
        device = "cuda"
        logger.info("CUDA is available and initialized successfully")
    except Exception as e:
        device = "cpu"
        logger.warning(f"CUDA initialization failed, falling back to CPU: {str(e)}")
else:
    device = "cpu"
    logger.info("CUDA is not available, using CPU")

# Ensure required configuration is present
if not hasattr(AppConfig, 'MAX_RESPONSE_TOKENS'):
    setattr(AppConfig, 'MAX_RESPONSE_TOKENS', 2000)

if not AppConfig.GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing from your .env file. Please set it and restart the app.")
    st.stop()

# Initialize Gemini 2.0 Flash
try:
    genai.configure(api_key=AppConfig.GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Gemini: {e}")
    st.stop()

# Initialize monitoring
if AppConfig.SENTRY_DSN:
    sentry_sdk.init(dsn=AppConfig.SENTRY_DSN, traces_sample_rate=1.0)

if AppConfig.PROMETHEUS_ENABLED:
    start_http_server(8000)

# Initialize Streamlit
st.set_page_config(page_title=AppConfig.APP_NAME, layout="wide")

APP_NAME = "Mental Health Support Bot"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

def generate_empathy(diagnosis_text: str) -> str:
    """Generate empathetic response based on diagnosis"""
    if not diagnosis_text:
        return "Thank you for sharing your feelings. I'm here to support you."

    base_responses = {
        "depression": "It takes courage to acknowledge these feelings.",
        "anxiety": "Uncertainty can feel overwhelming, but you're not alone.",
        "general": "What you're experiencing sounds challenging."
    }

    for key in base_responses:
        if key in diagnosis_text.lower():
            return base_responses[key]

    return f"Living with {diagnosis_text.lower()} can be difficult. Let's work through this together."

def generate_guidance(diagnosis_text: str, crawler_agent: CrawlerAgent) -> str:
    """Generate actionable guidance with resources"""
    base_guidance = """1. Consider reaching out to a mental health professional
2. Practice grounding techniques daily
3. Maintain a regular sleep schedule"""

    resources = crawler_agent.safe_crawl(f"evidence-based treatments for {diagnosis_text}")[:1000]
    return f"{base_guidance}\n\n**Resources:**\n{resources}"

def initialize_components() -> Dict[str, Any]:
    """Initialize all components for the application"""
    try:
        # Show initialization progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize the LLM (20% of progress)
        try:
            status_text.text("Initializing language model...")
            llm_config = {
                "model": AppConfig.MODEL_NAME,
                "api_key": AppConfig.GEMINI_API_KEY,
                "temperature": float(os.getenv("TEMPERATURE", "0.7")),
                "top_p": float(os.getenv("TOP_P", "0.9")),
                "top_k": int(os.getenv("TOP_K", "50")),
                "max_tokens": AppConfig.MAX_RESPONSE_TOKENS
            }
            llm = SafeLLM(model_config=llm_config)
            if llm.model is None:
                raise ValueError("LLM model instance not properly initialized")
            progress_bar.progress(0.2)
        except Exception as llm_error:
            logger.error(f"Failed to initialize LLM: {str(llm_error)}")
            st.error("Failed to initialize the language model. The application may not function correctly.")
            progress_bar.progress(1.0)
            return {}

        # Initialize Voice AI with progress updates (40% of progress)
        status_text.text("Initializing Voice AI models...")
        voice_manager = VoiceManager()
        voice_result = asyncio.run(voice_manager.initialize())

        # Initialize Whisper ASR for speech recognition
        status_text.text("Initializing Whisper speech recognition...")
        try:
            whisper_voice_manager = VoiceInputManager(model_name="turbo")
            whisper_available = True
            logger.info("Whisper ASR initialized successfully")
        except Exception as whisper_error:
            whisper_voice_manager = None
            whisper_available = False
            logger.warning(f"Whisper ASR initialization failed: {str(whisper_error)}")
            st.warning("Whisper speech recognition is not available. Falling back to standard voice features.")

        progress_bar.progress(0.4)
            
        if voice_result["success"]:
            voice_ai = voice_result["voice_ai"]
            logger.info("Voice AI initialized successfully")
        else:
            error_msg = voice_result.get("error", "Unknown error")
            if "missing_dependencies" in voice_result:
                missing = voice_result["missing_dependencies"]
                st.warning(
                    f"Voice features are disabled due to missing dependencies: {', '.join(missing)}. "
                    "To enable voice features, install the required packages:\n"
                    f"pip install {' '.join(missing)}"
                )
            else:
                st.warning(f"Voice features will not be available: {error_msg}. The application will continue in text-only mode.")
            voice_ai = None

        progress_bar.progress(0.6)

        # Initialize agents (40% of progress)
        try:
            status_text.text("Initializing AI agents...")
            safety_agent = SafetyAgent(model=llm)
            emotion_agent = EmotionAgent(model=llm)
            chat_agent = ChatAgent(model=llm)
            diagnosis_agent = DiagnosisAgent(model=llm)
            crawler_agent = CrawlerAgent(model=llm)
            personality_agent = PersonalityAgent(model=llm)
            integrated_diagnosis_agent = IntegratedDiagnosisAgent(model=llm)

            # Initialize integrated assessment component
            from components.integrated_assessment import IntegratedAssessment
            integrated_diagnosis = IntegratedAssessment(diagnosis_agent=diagnosis_agent)

            # Initialize orchestrator
            orchestrator = AgentOrchestrator(
                agents={
                    "safety": safety_agent,
                    "emotion": emotion_agent,
                    "chat": chat_agent,
                    "diagnosis": diagnosis_agent,
                    "crawler": crawler_agent,
                    "personality": personality_agent,
                    "integrated_diagnosis": integrated_diagnosis_agent
                }
            )
            progress_bar.progress(1.0)
            status_text.text("Initialization complete!")
            logger.info("Successfully initialized all agents and orchestrator")

            # Clear progress indicators
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            return {
                "safety": safety_agent,
                "emotion": emotion_agent,
                "chat_agent": chat_agent,
                "diagnosis": diagnosis_agent,
                "crawler": crawler_agent,
                "personality": personality_agent,
                "integrated_diagnosis": integrated_diagnosis_agent,
                "orchestrator": orchestrator,
                "llm": llm,
                "voice_ai": voice_ai,
                "whisper_voice_manager": whisper_voice_manager if whisper_available else None,
                "integrated_diagnosis": integrated_diagnosis
            }
        except Exception as agent_error:
            logger.error(f"Error initializing agents: {str(agent_error)}")
            st.error(f"Error initializing agent components: {str(agent_error)}")
            progress_bar.progress(1.0)
            status_text.empty()
            # Return partial initialization with just the LLM and voice
            return {
                "llm": llm,
                "voice_ai": voice_ai,
                "safety": None,
                "emotion": None,
                "chat_agent": None,
                "diagnosis": None,
                "crawler": None,
                "personality": None,
                "integrated_diagnosis": None,
                "orchestrator": None
            }
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing application: {str(e)}")
        return {}

def reset_session():
    """Reset the application session state"""
    st.session_state.clear()
    st.session_state.update({
        "step": 1,
        "symptoms": [],
        "diagnosis": "",
        "personality": {},
        "history": [],
        "start_time": time.time(),
        "assessment_component": None,
        "assessment_complete": False,
        "assessment_data": {},
        "integrated_assessment_results": {},
        "empathy_response": "",
        "immediate_actions": [],
        "metrics": {
            "interactions": 0,
            "response_times": [],
            "safety_flags": 0
        }
    })

def render_assessment(diagnosis_agent):
    import asyncio
    from agents.diagnosis_agent import phq9_assessment, gad7_assessment
    st.header("Mental Health Check-In")
    with st.form("assessment_form"):
        phq9_resps = []
        st.subheader("Depression Screening (PHQ-9)")
        for i, q in enumerate(AppConfig.PHQ9_QUESTIONS):
            choice = st.radio(f"{i+1}. {q}",
                               options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                               key=f"phq9_{i}")
            phq9_resps.append(["Not at all","Several days","More than half the days","Nearly every day"].index(choice))
        st.subheader("Anxiety Screening (GAD-7)")
        gad7_resps = []
        for i, q in enumerate(AppConfig.GAD7_QUESTIONS):
            choice = st.radio(f"{i+1}. {q}",
                               options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                               key=f"gad7_{i}")
            gad7_resps.append(["Not at all","Several days","More than half the days","Nearly every day"].index(choice))
        if st.form_submit_button("Continue"):
            # Compute PHQ-9 score and severity inline
            phq9_score = sum(phq9_resps)
            if phq9_score >= 20:
                phq9_severity = "severe"
            elif phq9_score >= 15:
                phq9_severity = "moderately severe"
            elif phq9_score >= 10:
                phq9_severity = "moderate"
            elif phq9_score >= 5:
                phq9_severity = "mild"
            else:
                phq9_severity = "minimal"
            phq9_result = {"score": phq9_score, "severity": phq9_severity}
            # Compute GAD-7 score and severity inline
            gad7_score = sum(gad7_resps)
            if gad7_score >= 15:
                gad7_severity = "severe"
            elif gad7_score >= 10:
                gad7_severity = "moderate"
            elif gad7_score >= 5:
                gad7_severity = "mild"
            else:
                gad7_severity = "minimal"
            gad7_result = {"score": gad7_score, "severity": gad7_severity}
            # Store diagnosis summary
            diag_text = (
                f"PHQ-9: {phq9_severity.capitalize()} (score {phq9_score}), "
                f"GAD-7: {gad7_severity.capitalize()} (score {gad7_score})"
            )
            st.session_state.update({
                "phq9": phq9_result,
                "gad7": gad7_result,
                "diagnosis": diag_text,
                "step": 3
            })
            track_metric("assessment_completed", 1)
            st.rerun()

def render_diagnosis(crawler_agent):
    st.header("Your Support Plan")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Understanding Your Experience")
        st.markdown(f"""
        <div style='background-color:#f8f9fa; padding:20px; border-radius:10px; margin-bottom:20px;'>
        {generate_empathy(st.session_state["diagnosis"])}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Recommended Next Steps")
        # Check if crawler_agent is available
        if crawler_agent is None:
            guidance = """1. Consider reaching out to a mental health professional
2. Practice grounding techniques daily
3. Maintain a regular sleep schedule

**Resources:**
- National Crisis Hotline: 988
- Emergency Services: 911
- Crisis Text Line: Text HOME to 741741"""
        else:
            guidance = generate_guidance(
                st.session_state["diagnosis"],
                crawler_agent
            )
        st.markdown(f"""
        <div style='background-color:#e9ecef; padding:20px; border-radius:10px;'>
        {guidance}
        </div>
        """, unsafe_allow_html=True)

    if st.button("Continue to Personality Assessment"):
        st.session_state["step"] = 3
        st.rerun()

async def process_user_message(user_input: str, components: Dict[str, Any]) -> Dict[str, Any]:
    """Process a user message and generate a response"""
    try:
        # Check if components are available
        if not components:
            logger.error("No components available for processing message")
            return {
                "response": "I apologize, but the chat functionality is currently unavailable. Please try again later.",
                "error": "No components available"
            }

        # Check for essential components
        for component_name in ["safety", "emotion", "chat_agent"]:
            if component_name not in components or components[component_name] is None:
                logger.error(f"Missing required component: {component_name}")
                return {
                    "response": "I apologize, but the chat functionality is currently unavailable. Please try again later.",
                    "error": f"Missing {component_name} component"
                }

        # Ensure metrics key exists
        if "metrics" not in st.session_state:
            st.session_state["metrics"] = {
                "interactions": 0,
                "response_times": [],
                "safety_flags": 0
            }

        # Update metrics
        st.session_state["metrics"]["interactions"] += 1

        # Safety check
        try:
            # Properly await the coroutine
            safety_result = await components["safety"].check_message(user_input)
        except Exception as safety_error:
            logger.error(f"Safety check failed: {str(safety_error)}")
            safety_result = {"safe": True}  # Default to safe if check fails

        if not safety_result.get("safe", True):
            st.session_state["metrics"]["safety_flags"] += 1
            return {
                "response": "I'm concerned about your safety. Please consider reaching out to a mental health professional or crisis hotline.",
                "safety_alert": True
            }

        # Emotion analysis
        try:
            # Properly await the coroutine
            emotion_result = await components["emotion"].analyze_emotion(user_input)
        except Exception as emotion_error:
            logger.error(f"Emotion analysis failed: {str(emotion_error)}")
            emotion_result = {"primary_emotion": "unknown"}  # Default emotion

        # Generate response
        start_time = time.time()
        try:
            # Get diagnosis data from integrated assessment if available
            diagnosis_data = {}
            if "integrated_assessment_results" in st.session_state:
                diagnosis_data = st.session_state["integrated_assessment_results"].get("assessment_results", {})

            # Properly await the coroutine
            response = await components["chat_agent"].generate_response(user_input, {
                "emotion": emotion_result,
                "safety": safety_result,
                "diagnosis": diagnosis_data
            })
        except Exception as chat_error:
            logger.error(f"Chat generation failed: {str(chat_error)}")
            return {
                "response": "I'm having difficulty generating a response right now. Could you try rephrasing your message?",
                "error": str(chat_error),
                "emotion": emotion_result
            }

        end_time = time.time()

        # Update metrics
        st.session_state["metrics"]["response_times"].append(end_time - start_time)

        return {
            "response": response.get("response", "I'm having trouble generating a response right now."),
            "emotion": emotion_result,
            "safety": safety_result
        }
    except Exception as e:
        logger.error(f"Error processing user message: {str(e)}")
        return {
            "response": "I'm having trouble processing your message right now. Please try again later.",
            "error": str(e)
        }

def main():
    components = initialize_components()

    st.title(AppConfig.APP_NAME)
    st.markdown("### A Safe Space for Mental Health Support")

    # Show system status in expander
    with st.expander("System Status", expanded=False):
        st.markdown("### Component Status")

        # Check component status
        component_status = {
            "Language Model": components.get("llm") is not None,
            "Safety Agent": components.get("safety") is not None,
            "Emotion Agent": components.get("emotion") is not None,
            "Chat Agent": components.get("chat_agent") is not None,
            "Diagnosis Agent": components.get("diagnosis") is not None,
            "Personality Agent": components.get("personality") is not None
        }

        # Display status
        for component, status in component_status.items():
            if status:
                st.success(f"✅ {component}: Available")
            else:
                st.error(f"❌ {component}: Unavailable")

        # Show environment info
        st.markdown("### Environment")
        st.write(f"App Version: {AppConfig.APP_VERSION}")
        st.write(f"Debug Mode: {'Enabled' if AppConfig.DEBUG else 'Disabled'}")
        st.write(f"Model: {AppConfig.MODEL_NAME}")

    # Initialize session state
    if "step" not in st.session_state:
        reset_session()

    # Ensure metrics key exists
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {
            "interactions": 0,
            "response_times": [],
            "safety_flags": 0
        }

    # Application routing
    if st.session_state["step"] == 1:
        render_integrated_assessment(components["integrated_diagnosis"])
    elif st.session_state["step"] == 2:
        render_diagnosis(components["crawler"])
    elif st.session_state["step"] == 3:
        render_personality_assessment(components["personality"])
    elif st.session_state["step"] == 4:
        render_chat_interface(components)
    elif st.session_state["step"] == 5:
        render_crisis_protocol()

def render_integrated_assessment(integrated_diagnosis_agent):
    """Render the integrated assessment interface"""
    import asyncio

    # Check if integrated_diagnosis_agent is available
    if integrated_diagnosis_agent is None:
        st.warning("The integrated assessment functionality is currently unavailable. Falling back to standard assessment.")
        render_assessment(None)  # Fallback to standard assessment
        return

    # Initialize assessment component if not already done
    if "assessment_component" not in st.session_state or st.session_state["assessment_component"] is None:
        def on_assessment_complete(assessment_data):
            # Store assessment data in session state
            st.session_state["assessment_data"] = assessment_data

            # Process the assessment data
            with st.spinner("Analyzing your responses..."):
                # Run the assessment
                assessment_results = asyncio.run(integrated_diagnosis_agent.conduct_assessment(
                    assessment_data["mental_health_responses"],
                    assessment_data["personality_responses"]
                ))

                # Store results in session state
                st.session_state["integrated_assessment_results"] = assessment_results

                # Generate empathy response
                empathy_response = integrated_diagnosis_agent.generate_empathy_response(
                    assessment_results.get("assessment_results", {})
                )

                # Generate immediate actions
                immediate_actions = integrated_diagnosis_agent.generate_immediate_actions(
                    assessment_results.get("assessment_results", {})
                )

                # Store in session state
                st.session_state["empathy_response"] = empathy_response
                st.session_state["immediate_actions"] = immediate_actions

                # Move to results display
                st.session_state["assessment_complete"] = True

        # Create the assessment component
        from components.integrated_assessment import IntegratedAssessmentComponent
        st.session_state["assessment_component"] = IntegratedAssessmentComponent(
            on_complete=on_assessment_complete
        )

    # Check if assessment is complete
    if st.session_state.get("assessment_complete", False):
        # Render the results
        from components.diagnosis_results import DiagnosisResultsComponent
        results_component = DiagnosisResultsComponent(
            on_continue=lambda: st.session_state.update({"step": 4})  # Move to chat interface
        )

        results_component.render(
            st.session_state.get("integrated_assessment_results", {}).get("assessment_results", {}),
            st.session_state.get("empathy_response", ""),
            st.session_state.get("immediate_actions", [])
        )
    else:
        # Check that the assessment component exists before rendering
        if st.session_state["assessment_component"] is not None:
            st.session_state["assessment_component"].render()
        else:
            st.error("Could not initialize assessment component. Please try again or contact support.")
            if st.button("Try Standard Assessment Instead"):
                render_assessment(None)

def render_personality_assessment(personality_agent):
    """Render the personality assessment interface with voice and emotion integration"""
    import asyncio
    import json
    from components.dynamic_personality_assessment import DynamicPersonalityAssessmentComponent
    
    # Check if voice components are available in session state
    voice_ai = st.session_state.get("voice_ai")
    whisper_voice_manager = st.session_state.get("whisper_voice_manager")
    emotion_agent = st.session_state.get("emotion_agent")
    
    # Initialize the dynamic assessment component if not already in session state
    if "dynamic_personality_component" not in st.session_state:
        # Define callback for assessment completion
        def on_assessment_complete(results):
            st.session_state["personality"] = results
            st.session_state["personality_assessment_complete"] = True
            # Store completed assessment in session state
            if "completed_assessments" not in st.session_state:
                st.session_state["completed_assessments"] = []
            # Add a timestamp to the assessment
            results["timestamp"] = datetime.now().isoformat()
            st.session_state["completed_assessments"].append(results)
        
        # Create component instance with available components
        st.session_state["dynamic_personality_component"] = DynamicPersonalityAssessmentComponent(
            personality_agent=personality_agent,
            voice_ai=voice_ai,
            whisper_voice_manager=whisper_voice_manager,
            emotion_agent=emotion_agent,
            on_complete=on_assessment_complete
        )
    
    # Header and description
    st.header("Personality Assessment")
    
    # If assessment is complete, show results or offer to retake
    if st.session_state.get("personality_assessment_complete", False):
        personality_data = st.session_state.get("personality", {})
        
        if not personality_data:
            st.warning("No assessment results found. Please take an assessment first.")
            if st.button("Take New Assessment"):
                st.session_state["personality_assessment_complete"] = False
                # Reset dynamic component state
                if "dynamic_assessment_state" in st.session_state:
                    del st.session_state["dynamic_assessment_state"]
                st.rerun()
        else:
            # Display assessment results
            assessment_type = personality_data.get("assessment_type", "")
            if not assessment_type and "traits" in personality_data:
                assessment_type = "big_five"
            elif not assessment_type and "type" in personality_data:
                assessment_type = "mbti"
            
            # Show results based on assessment type
            if assessment_type == "big_five" or "traits" in personality_data:
                _render_big_five_results(personality_data, personality_agent, voice_ai)
            elif assessment_type == "mbti" or "type" in personality_data:
                _render_mbti_results(personality_data, personality_agent, voice_ai)
            else:
                st.error("Unknown assessment type in results")
            
            # Show button to retake assessment
            if st.button("Take Another Assessment"):
                # Reset assessment completion flag
                st.session_state["personality_assessment_complete"] = False
                # Reset dynamic component state
                if "dynamic_assessment_state" in st.session_state:
                    del st.session_state["dynamic_assessment_state"]
                st.rerun()
            
            # Button to continue to chat
            if st.button("Continue to Chat"):
                st.session_state["step"] = 4
                st.rerun()
    else:
        # Render the dynamic assessment component
        st.session_state["dynamic_personality_component"].render()

def _render_big_five_results(results, personality_agent, voice_ai=None):
    """Render Big Five assessment results with voice capabilities"""
    import asyncio
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get trait data
    traits = results.get("traits", {})
    
    if not traits:
        st.warning("No trait data found in results")
        return
    
    st.markdown("### Your Big Five (OCEAN) Profile")
    
    # Play audio summary if voice is enabled
    if voice_ai and "voice_summary_played" not in st.session_state:
        # Create a summary of the results
        trait_highlights = []
        for trait_name, trait_data in traits.items():
            score = trait_data.get("score", 50)
            category = trait_data.get("category", "average")
            trait_highlights.append(f"Your {trait_name} score is {int(score)}%, which is {category}.")
        
        summary = "I've analyzed your personality profile. " + " ".join(trait_highlights[:3])
        
        # Play the summary
        voice_ai.text_to_speech(summary, voice_style="warm")
        st.session_state["voice_summary_played"] = True
    
    # Create a bar chart for the main traits
    if traits:
        # Extract trait scores and names
        trait_names = list(traits.keys())
        trait_scores = [traits[name].get("score", 50) for name in trait_names]
        
        # Capitalize trait names for display
        trait_names = [name.capitalize() for name in trait_names]
        
        # Create the chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(trait_names, trait_scores, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        
        # Add labels and title
        ax.set_ylabel('Percentile Score')
        ax.set_title('Your Big Five Personality Profile')
        ax.set_ylim(0, 100)
        
        # Add a horizontal line at 50%
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}%', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    # Display trait descriptions
    st.markdown("### Trait Descriptions")
    
    for trait_name, trait_data in traits.items():
        score = trait_data.get("score", 50)
        category = trait_data.get("category", "average")
        facets = trait_data.get("facets", [])
        
        # Get description based on category
        description = _get_big_five_trait_description(trait_name, category)
        
        # Create expander for each trait
        with st.expander(f"**{trait_name.capitalize()} ({int(score)}%)**", expanded=score > 80 or score < 20):
            st.markdown(description)
            
            # Show facets if available
            if facets:
                st.markdown("#### Facet Breakdown")
                for facet in facets:
                    facet_name = facet.get("name", "")
                    facet_score = facet.get("score", 0)
                    facet_category = facet.get("category", "average")
                    
                    if facet_name:
                        st.markdown(f"**{facet_name.capitalize()}**: {facet_score} ({facet_category})")
    
    # Get and display interpretation if available
    if "interpretation" in results and isinstance(results["interpretation"], dict):
        _render_personality_interpretation(results["interpretation"])
    else:
        # Generate interpretation using personality agent
        try:
            interpretation_result = asyncio.run(personality_agent.conduct_assessment("big_five", results))
            interpretation = interpretation_result.get("interpretation", {})
            
            if interpretation and not isinstance(interpretation, str) and "error" not in interpretation:
                _render_personality_interpretation(interpretation)
        except Exception as e:
            st.warning(f"Could not generate personalized insights: {str(e)}")

def _render_mbti_results(results, personality_agent, voice_ai=None):
    """Render MBTI assessment results with voice capabilities"""
    import asyncio
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get MBTI type data
    personality_type = results.get("type", "")
    type_name = results.get("type_name", "")
    
    if not personality_type:
        st.warning("No personality type found in results")
        return
    
    st.markdown(f"### Your MBTI Type: {personality_type} - {type_name}")
    
    # Play audio summary if voice is enabled
    if voice_ai and "voice_summary_played" not in st.session_state:
        # Create a summary of the results
        description = results.get("description", "")
        summary_text = f"Your personality type is {personality_type}, known as {type_name}. {description[:100]}..."
        
        # Play the summary
        voice_ai.text_to_speech(summary_text, voice_style="warm")
        st.session_state["voice_summary_played"] = True
    
    # Display type description
    description = results.get("description", "No description available.")
    st.markdown(f"**Description**: {description}")
    
    # Display dimension scores
    dimensions = results.get("dimensions", {})
    if dimensions:
        st.markdown("### Dimension Preferences")
        
        # Create a horizontal bar chart for each dimension
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        dimension_names = ["E/I", "S/N", "T/F", "J/P"]
        dimension_labels = [
            ["Extraversion", "Introversion"],
            ["Sensing", "Intuition"],
            ["Thinking", "Feeling"],
            ["Judging", "Perceiving"]
        ]
        dimension_colors = [
            ["#3498db", "#2980b9"],
            ["#2ecc71", "#27ae60"],
            ["#e74c3c", "#c0392b"],
            ["#f39c12", "#d35400"]
        ]
        
        for i, dim_name in enumerate(dimension_names):
            dim_data = dimensions.get(dim_name, {})
            if not dim_data:
                continue
            
            percentages = dim_data.get("percentages", {})
            if not percentages:
                continue
            
            # Get the two dimension values (e.g., E and I)
            dim_keys = list(percentages.keys())
            if len(dim_keys) < 2:
                continue
            
            # Get scores
            scores = [percentages.get(dim_keys[0], 50), percentages.get(dim_keys[1], 50)]
            
            # Create the horizontal bar chart
            ax = axes[i]
            bars = ax.barh([dimension_labels[i][0], dimension_labels[i][1]], scores, color=dimension_colors[i])
            
            # Add labels
            ax.set_xlim(0, 100)
            ax.set_title(f"{dimension_labels[i][0]} vs {dimension_labels[i][1]}")
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}%', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Display strengths and weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        strengths = results.get("strengths", [])
        if strengths:
            st.markdown("### Strengths")
            for strength in strengths:
                st.markdown(f"- {strength}")
    
    with col2:
        weaknesses = results.get("weaknesses", [])
        if weaknesses:
            st.markdown("### Potential Challenges")
            for weakness in weaknesses:
                st.markdown(f"- {weakness}")
    
    # Get and display interpretation if available
    if "interpretation" in results and isinstance(results["interpretation"], dict):
        _render_personality_interpretation(results["interpretation"])
    else:
        # Generate interpretation using personality agent
        try:
            interpretation_result = asyncio.run(personality_agent.conduct_assessment("mbti", results))
            interpretation = interpretation_result.get("interpretation", {})
            
            if interpretation and not isinstance(interpretation, str) and "error" not in interpretation:
                _render_personality_interpretation(interpretation)
        except Exception as e:
            st.warning(f"Could not generate personalized insights: {str(e)}")

def _get_big_five_trait_description(trait_name, category):
    """Get description for a Big Five trait based on category"""
    if trait_name == "openness":
        if category == "high":
            return "You are curious, imaginative, and open to new experiences. You likely have a broad range of interests and appreciate art, creativity, and intellectual pursuits."
        elif category == "low":
            return "You prefer routine, practicality, and tradition. You may focus more on concrete facts than abstract theories and might be more conventional in your approach."
        else:
            return "You balance curiosity with practicality. You can appreciate new ideas while maintaining a grounded perspective."
    elif trait_name == "conscientiousness":
        if category == "high":
            return "You are organized, disciplined, and detail-oriented. You likely plan ahead, follow through on commitments, and strive for achievement."
        elif category == "low":
            return "You tend to be more flexible, spontaneous, and relaxed about deadlines or organization. You may prefer to go with the flow rather than stick to rigid plans."
        else:
            return "You balance organization with flexibility. You can be structured when needed but also adapt to changing circumstances."
    elif trait_name == "extraversion":
        if category == "high":
            return "You are outgoing, energetic, and draw energy from social interactions. You likely enjoy being around others and may seek excitement and stimulation."
        elif category == "low":
            return "You tend to be more reserved and may prefer solitary activities. You might find social interactions draining and need time alone to recharge."
        else:
            return "You balance sociability with independence. You can enjoy social situations but also value your alone time."
    elif trait_name == "agreeableness":
        if category == "high":
            return "You are compassionate, cooperative, and considerate of others' feelings. You likely value harmony and may prioritize others' needs."
        elif category == "low":
            return "You tend to be more direct, competitive, or skeptical. You might prioritize truth over tact and may be more willing to challenge others."
        else:
            return "You balance cooperation with healthy skepticism. You can be kind while maintaining appropriate boundaries."
    elif trait_name == "neuroticism":
        if category == "high":
            return "You may experience emotions more intensely and be more sensitive to stress. You might worry more than others and be more aware of potential problems."
        elif category == "low":
            return "You tend to be emotionally stable and resilient to stress. You likely remain calm under pressure and recover quickly from setbacks."
        else:
            return "You have a balanced emotional response. You can feel appropriate emotions without being overwhelmed by them."
    else:
        return "No detailed description available for this trait."

def _render_personality_interpretation(interpretation):
    """Render the personality interpretation sections"""
    # Display key insights
    if interpretation.get("key_insights"):
        st.markdown("### Personalized Insights")
        for insight in interpretation["key_insights"]:
            st.markdown(f"- {insight}")
    
    # Display strengths and growth areas in columns
    if interpretation.get("strengths") or interpretation.get("growth_areas"):
        col1, col2 = st.columns(2)
        
        with col1:
            if interpretation.get("strengths"):
                st.markdown("#### Strengths")
                for strength in interpretation["strengths"]:
                    st.markdown(f"- {strength}")
        
        with col2:
            if interpretation.get("growth_areas"):
                st.markdown("#### Growth Areas")
                for area in interpretation["growth_areas"]:
                    st.markdown(f"- {area}")
    
    # Display communication preferences
    if interpretation.get("communication_preferences"):
        st.markdown("#### Communication Style")
        for pref in interpretation["communication_preferences"]:
            st.markdown(f"- {pref}")
    
    # Display stress responses
    if interpretation.get("stress_responses"):
        st.markdown("#### Stress Response")
        for response in interpretation["stress_responses"]:
            st.markdown(f"- {response}")
    
    # Display mental health implications
    if interpretation.get("mental_health_implications"):
        st.markdown("#### Mental Health Insights")
        for implication in interpretation["mental_health_implications"]:
            st.markdown(f"- {implication}")
    
    # Display emotional patterns
    if interpretation.get("emotional_patterns"):
        st.markdown("#### Emotional Tendencies")
        for pattern in interpretation["emotional_patterns"]:
            st.markdown(f"- {pattern}")

def render_chat_interface(components: Dict[str, Any]):
    """Render the chat interface with voice capabilities"""
    import asyncio

    st.markdown("### Chat with Your Mental Health Assistant")

    # Check if components are available
    if not components or not components.get("chat_agent"):
        st.warning("The chat functionality is currently unavailable. Please try again later.")
        if st.button("Start New Assessment"):
            reset_session()
            st.rerun()
        return

    # Initialize voice component if available
    voice_enabled = components.get("voice_ai") is not None
    whisper_enabled = components.get("whisper_voice_manager") is not None

    if voice_enabled:
        # Initialize voice component with callback for speech-to-text
        if "voice_component" not in st.session_state:
            voice_ai = components["voice_ai"]

            def on_voice_input(text):
                if text:  # Process voice input like a regular text input
                    st.session_state["voice_input"] = text
                    st.rerun()

            st.session_state["voice_component"] = VoiceComponent(
                voice_ai=voice_ai,
                on_transcription=on_voice_input
            )

            # Initialize voice settings
            if "voice_style" not in st.session_state:
                st.session_state["voice_style"] = "warm"

    # Add voice controls in an expander
    if voice_enabled or whisper_enabled:
        with st.expander("🎤 Voice Interaction Settings", expanded=False):
            if whisper_enabled:
                st.checkbox("Enable Whisper V3 Turbo voice recognition", value=True, key="use_whisper")
            
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎙️ Speech Input")
                
                # Standard voice input
                if voice_enabled:
                    st.session_state["voice_component"].render_voice_input()
                
                # Whisper voice input
                if whisper_enabled and st.session_state.get("use_whisper", True):
                    if st.button("🎙️ Whisper Voice Input"):
                        with st.spinner("Listening..."):
                            result = components["whisper_voice_manager"].transcribe_once()
                            if result["success"] and result["text"]:
                                st.session_state["voice_input"] = result["text"]
                                st.rerun()
                            else:
                                st.error(f"Could not understand audio: {result.get('error', 'Unknown error')}")

            with col2:
                st.markdown("### 🔊 Voice Settings")
                if voice_enabled:
                    st.session_state["voice_style"] = st.session_state["voice_component"].render_voice_selector()
                st.checkbox("Automatically speak responses", value=True, key="auto_speak_responses")

                # Test voice button
                if voice_enabled and st.button("Test Voice"):
                    test_text = "Hello! I'm your mental health assistant. I'm here to listen and support you with empathy and compassion."
                    st.session_state["voice_component"].render_voice_output(
                        test_text,
                        autoplay=True
                    )

        # Horizontal separator
        st.markdown("---")

    # Display chat history
    for message in st.session_state["history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "emotion" in message:
                st.caption(f"Emotion: {message['emotion'].get('primary_emotion', 'unknown')}")

        # Play voice response for assistant messages if enabled
        if voice_enabled and message["role"] == "assistant" and st.session_state.get("auto_speak_responses", False):
            # Check if this message hasn't been spoken yet (using a simple tracking system)
            message_id = hash(message["content"])
            if "spoken_messages" not in st.session_state:
                st.session_state["spoken_messages"] = set()

            if message_id not in st.session_state["spoken_messages"]:
                # Speak the message
                st.session_state["voice_component"].render_voice_output(
                    message["content"],
                    autoplay=True
                )
                # Mark as spoken
                st.session_state["spoken_messages"].add(message_id)

    # Check if we have a pending voice input
    if voice_enabled and "voice_input" in st.session_state:
        user_input = st.session_state["voice_input"]
        # Clear the input so it doesn't get processed again
        del st.session_state["voice_input"]
    else:
        # Regular text input
        user_input = st.chat_input("Type your message here, or use the voice input above...")

    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)

        # Process message asynchronously
        with st.spinner("Thinking..."):
            try:
                # Use asyncio to run the async function
                result = asyncio.run(process_user_message(user_input, components))

                if result:
                    # Check for specific errors returned by process_user_message
                    if "error" in result:
                        st.error(f"Assistant Error: {result['error']}")
                        logger.error(f"Error processed from process_user_message: {result['error']}")
                    else:
                        # Update session state with the new messages
                        st.session_state["history"].extend([
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": result["response"], "emotion": result.get("emotion")}
                        ])

                        # Display assistant response
                        with st.chat_message("assistant"):
                            st.write(result["response"])
                            if "emotion" in result and result["emotion"]:
                                st.caption(f"Emotion: {result['emotion'].get('primary_emotion', 'unknown')}")

                        # Speak the assistant's response if voice is enabled
                        if voice_enabled and st.session_state.get("auto_speak_responses", False):
                            st.session_state["voice_component"].render_voice_output(
                                result["response"],
                                autoplay=True
                            )

                        # Handle safety alerts
                        if result.get("safety_alert", False):
                            st.warning("⚠️ Safety Alert: Please consider reaching out to a mental health professional or crisis hotline.")
                            st.session_state["step"] = 5  # Move to crisis protocol
                            st.rerun()
                else:
                    st.error("Failed to generate a response. Please try again.")
            except Exception as e:
                st.error(f"An error occurred while processing your message: {str(e)}")
                logger.error(f"Error in chat interface during processing: {str(e)}")

    # Add a button to go back to assessment
    if st.button("Start New Assessment"):
        reset_session()
        st.rerun()

def render_crisis_protocol():
    st.error("Immediate Support Needed")
    st.markdown(AppConfig.CRISIS_RESOURCES)
    if st.button("Restart Session"):
        reset_session()
        st.rerun()

if __name__ == "__main__":
    main()