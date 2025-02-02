import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import Document  # Import Document
from config.settings import AppConfig
from database.vector_store import FAISSVectorStore
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

# Initialize monitoring
if AppConfig.SENTRY_DSN:
    sentry_sdk.init(dsn=AppConfig.SENTRY_DSN, traces_sample_rate=1.0)

if AppConfig.PROMETHEUS_ENABLED:
    start_http_server(8000)

# Initialize Streamlit
st.set_page_config(page_title=AppConfig.APP_NAME, layout="wide")

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

@st.cache_resource
def initialize_components():
    vector_store = FAISSVectorStore(**AppConfig.get_vector_store_config(), allow_dangerous_deserialization=True)
    vector_store.connect()
    
    chat_agent = ChatAgent(
        model_name=AppConfig.MODEL_NAME,
        use_cpu=AppConfig.USE_CPU
    )
    
    crawler_agent = CrawlerAgent(AppConfig.get_crawler_config())
    diagnosis_agent = DiagnosisAgent(chat_agent.llm)
    emotion_agent = EmotionAgent(chat_agent.llm)
    safety_agent = SafetyAgent(chat_agent.llm)
    search_agent = SearchAgent(vector_store)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    return {
        "vector_store": vector_store,
        "chat_agent": chat_agent,
        "crawler": crawler_agent,
        "diagnosis": diagnosis_agent,
        "emotion": emotion_agent,
        "safety": safety_agent,
        "search": search_agent,
        "memory": memory
    }

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

def process_user_message(message: str, components: dict):
    start_time = time.time()
    session = st.session_state
    
    # Safety check
    safety = components["safety"].check_message(message)
    if not safety["safe"]:
        session["step"] = 4
        session["metrics"]["safety_flags"] += 1
        track_metric("safety_flag_raised", 1)
        st.rerun()
    
    # Emotion analysis
    emotion = components["emotion"].analyze(message)
    
    # Generate response
    context = components["search"].retrieve_context(message)
    response = components["chat_agent"].generate_response(
        context=context,
        question=message,
        emotion=emotion,
        safety=safety
    )
    
    # Update session
    session["history"].extend([
        {"role": "human", "content": message, "emotion": emotion},
        {"role": "ai", "content": response}
    ])
    
    # Update metrics
    session["metrics"]["interactions"] += 1
    session["metrics"]["response_times"].append(time.time() - start_time)
    track_metric("response_time", session["metrics"]["response_times"][-1])
    
    # Save interaction to vector store
    components["vector_store"].upsert([
        Document(page_content=message, metadata={"role": "human", "emotion": emotion}),
        Document(page_content=response, metadata={"role": "ai", "emotion": emotion})
    ])
    
    st.rerun()

def main():
    components = initialize_components()
    
    st.title(AppConfig.APP_NAME)
    st.markdown("### A Safe Space for Mental Health Support")
    
    # Initialize session state
    if "step" not in st.session_state:
        reset_session()
    
    # Application routing
    if st.session_state["step"] == 1:
        render_assessment(components["diagnosis"])
    elif st.session_state["step"] == 2:
        render_diagnosis(components["crawler"])
    elif st.session_state["step"] == 3:
        render_chat_interface(components)
    elif st.session_state["step"] == 4:
        render_crisis_protocol()

def render_chat_interface(components):
    st.header("Supportive Chat")
    
    # Display chat history
    for msg in st.session_state["history"]:
        role = "user" if msg["role"] == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])
            if msg.get("emotion"):
                st.caption(f"Detected emotion: {msg['emotion']['primary_emotion']} ({msg['emotion']['intensity']}/10)")
    
    # User input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        process_user_message(user_input, components)
    
    # Safety check
    if st.session_state["metrics"]["safety_flags"] > 0:
        st.error(AppConfig.CRISIS_RESOURCES)

def render_crisis_protocol():
    st.error("Immediate Support Needed")
    st.markdown(AppConfig.CRISIS_RESOURCES)
    if st.button("Restart Session"):
        reset_session()
        st.rerun()

if __name__ == "__main__":
    main()