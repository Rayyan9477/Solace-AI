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
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure required configuration is present
if not hasattr(AppConfig, 'MAX_RESPONSE_TOKENS'):
    setattr(AppConfig, 'MAX_RESPONSE_TOKENS', 2000)

if not AppConfig.ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY must be set in the configuration")

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
        # Initialize the LLM
        try:
            llm = SafeLLM()
        except Exception as llm_error:
            logger.error(f"Failed to initialize LLM: {str(llm_error)}")
            st.error("Failed to initialize the language model. The application may not function correctly.")
            # Return empty components to allow the app to start
            return {
                "safety": None,
                "emotion": None,
                "chat_agent": None,
                "diagnosis": None,
                "crawler": None,
                "orchestrator": None,
                "llm": None
            }
        
        # Initialize agents
        safety_agent = SafetyAgent(model=llm)
        emotion_agent = EmotionAgent(model=llm)
        chat_agent = ChatAgent(model=llm)
        diagnosis_agent = DiagnosisAgent(model=llm)
        crawler_agent = CrawlerAgent(model=llm)
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(
            agents={
                "safety": safety_agent,
                "emotion": emotion_agent,
                "chat": chat_agent,
                "diagnosis": diagnosis_agent
            }
        )
        
        return {
            "safety": safety_agent,
            "emotion": emotion_agent,
            "chat_agent": chat_agent,
            "diagnosis": diagnosis_agent,
            "crawler": crawler_agent,
            "orchestrator": orchestrator,
            "llm": llm
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
        "history": [],
        "start_time": time.time(),
        "metrics": {
            "interactions": 0,
            "response_times": [],
            "safety_flags": 0
        }
    })

def render_assessment(diagnosis_agent):
    st.header("Mental Health Check-In")
    with st.form("assessment_form"):
        responses = {}
        for idx, question in enumerate(AppConfig.ASSESSMENT_QUESTIONS):
            responses[question] = st.radio(
                f"{idx+1}. {question}",
                options=("No", "Yes"),
                key=f"q{idx}"
            )
        
        if st.form_submit_button("Continue"):
            symptoms = [q for q, a in responses.items() if a == "Yes"]
            diagnosis = diagnosis_agent.diagnose(symptoms)
            st.session_state.update({
                "symptoms": symptoms,
                "diagnosis": diagnosis,
                "step": 2
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
    
    if st.button("Start Chat Session"):
        st.session_state["step"] = 3
        st.rerun()

async def process_user_message(user_input: str, components: Dict[str, Any]) -> Dict[str, Any]:
    """Process a user message and generate a response"""
    try:
        # Check if components are available
        if not components or not components.get("safety") or not components.get("emotion") or not components.get("chat_agent"):
            return {
                "response": "I apologize, but the chat functionality is currently unavailable. Please try again later.",
                "error": "Components not available"
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
        safety_result = await components["safety"].check_message(user_input)
        if not safety_result.get("safe", True):
            st.session_state["metrics"]["safety_flags"] += 1
            return {
                "response": "I'm concerned about your safety. Please consider reaching out to a mental health professional or crisis hotline.",
                "safety_alert": True
            }
        
        # Emotion analysis
        emotion_result = await components["emotion"].analyze_emotion(user_input)
        
        # Generate response
        start_time = time.time()
        response = await components["chat_agent"].generate_response(user_input, {
            "emotion": emotion_result,
            "safety": safety_result
        })
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
        render_assessment(components["diagnosis"])
    elif st.session_state["step"] == 2:
        render_diagnosis(components["crawler"])
    elif st.session_state["step"] == 3:
        render_chat_interface(components)
    elif st.session_state["step"] == 4:
        render_crisis_protocol()

def render_chat_interface(components: Dict[str, Any]):
    """Render the chat interface"""
    st.markdown("### Chat with Your Mental Health Assistant")
    
    # Check if components are available
    if not components or not components.get("chat_agent"):
        st.warning("The chat functionality is currently unavailable. Please try again later.")
        if st.button("Start New Assessment"):
            reset_session()
            st.rerun()
        return
    
    # Display chat history
    for message in st.session_state["history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "emotion" in message:
                st.caption(f"Emotion: {message['emotion'].get('primary_emotion', 'unknown')}")
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process message in a separate thread to avoid blocking the UI
        with st.spinner("Thinking..."):
            try:
                # Get the current event loop or create a new one
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Create a new event loop if none exists
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async process in a separate thread
                result = None
                
                def run_async_process():
                    nonlocal result
                    result = loop.run_until_complete(process_user_message(user_input, components))
                
                # Run in a separate thread to avoid blocking the UI
                thread = threading.Thread(target=run_async_process)
                thread.start()
                thread.join()  # Wait for the thread to complete
                
                if result:
                    # Update session state with the new messages
                    st.session_state["history"].extend([
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": result["response"]}
                    ])
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(result["response"])
                        if "emotion" in result:
                            st.caption(f"Emotion: {result['emotion'].get('primary_emotion', 'unknown')}")
                    
                    # Handle safety alerts
                    if result.get("safety_alert", False):
                        st.warning("⚠️ Safety Alert: Please consider reaching out to a mental health professional or crisis hotline.")
                        st.session_state["step"] = 4  # Move to crisis protocol
                        st.rerun()
                else:
                    st.error("Failed to generate a response. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error in chat interface: {str(e)}")
    
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