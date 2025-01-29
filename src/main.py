import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from config.settings import AppConfig
from database.vector_store import FAISSVectorStore
from agents.chat_agent import ChatAgent
from agents.crawler_agent import CrawlerAgent
from agents.diagnosis_agent import DiagnosisAgent
from utils.helpers import sanitize_input
import sentry_sdk
from prometheus_client import start_http_server

# Initialize Sentry for error tracking
if AppConfig.SENTRY_DSN:
    sentry_sdk.init(dsn=AppConfig.SENTRY_DSN)

# Start Prometheus server for metrics
if AppConfig.PROMETHEUS_ENABLED:
    start_http_server(8000)

# Initialize Streamlit app
st.set_page_config(page_title=AppConfig.APP_NAME, page_icon="🤖", layout="wide")

@st.cache_resource
def load_vector_store():
    vector_store_config = AppConfig.get_vector_store_config()
    vector_store = FAISSVectorStore(
        path=vector_store_config['path'],
        collection=vector_store_config['collection'],
        allow_dangerous_deserialization=AppConfig.ALLOW_DANGEROUS_DESERIALIZATION
    )
    vector_store.connect()
    return vector_store

@st.cache_resource
def load_chat_agent():
    return ChatAgent(model_name=AppConfig.MODEL_NAME, use_cpu=AppConfig.USE_CPU)

@st.cache_resource
def load_crawler_agent():
    return CrawlerAgent(config=AppConfig.get_crawler_config())

@st.cache_resource
def load_diagnosis_agent(_llm):
    return DiagnosisAgent(_llm)

def load_agents(vector_store):
    chat_agent = load_chat_agent()
    crawler_agent = load_crawler_agent()
    diagnosis_agent = load_diagnosis_agent(chat_agent.llm)
    return chat_agent, crawler_agent, diagnosis_agent

def main():
    vector_store = load_vector_store()
    chat_agent, crawler_agent, diagnosis_agent = load_agents(vector_store)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a mental health chatbot. Use the following context to answer the question:\nContext: {context}\nQuestion: {question}\nAnswer:"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_agent.llm,
        retriever=vector_store.as_retriever(search_kwargs=AppConfig.get_rag_config()),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    st.title(f"🤖 {AppConfig.APP_NAME}")
    st.write("Welcome to the Mental Health Chatbot. Let's get started.")

    if "step" not in st.session_state:
        st.session_state.step = 1
        st.session_state.symptoms = []
        st.session_state.diagnosis = ""
        st.session_state.chat_history = []

    if st.session_state.step == 1:
        handle_self_assessment(diagnosis_agent)
    elif st.session_state.step == 2:
        handle_diagnosis()
    elif st.session_state.step == 3:
        handle_chat(qa_chain, crawler_agent)

def handle_self_assessment(diagnosis_agent):
    st.header("Step 1: Self-Assessment")
    st.write("Please answer the following questions to help us understand how you're feeling.")

    questions = [
        "Have you been feeling sad or down frequently?",
        "Have you lost interest in activities you once enjoyed?",
        "Are you experiencing excessive worry or fear?",
        "Have you noticed changes in your sleep patterns?",
        "Do you feel fatigued or lack energy?",
        "Are you having difficulty concentrating?",
        "Have you experienced feelings of hopelessness?",
        "Do you have thoughts of self-harm or suicide?"
    ]

    responses = {}
    for question in questions:
        responses[question] = st.radio(question, ("No", "Yes"), key=question)

    if st.button("Submit"):
        symptoms = collect_symptoms_responses(responses)
        st.session_state.symptoms = symptoms
        diagnosis = diagnosis_agent.diagnose(symptoms)
        st.session_state.diagnosis = diagnosis
        st.session_state.step = 2
        st.rerun()

def handle_diagnosis():
    st.header("Step 2: Diagnosis")
    st.write(f"**Diagnosis:** {st.session_state.diagnosis}")

    if any(condition in st.session_state.diagnosis.lower() for condition in ["depression", "anxiety"]):
        st.info("I'm here to help you. Let's discuss how you're feeling.")
        if st.button("Continue to Chat"):
            st.session_state.step = 3
            st.rerun()
    else:
        st.success("It's good to hear that you're doing well. If you ever need someone to talk to, I'm here.")
        if st.button("Restart"):
            reset_session()
            st.rerun()

def handle_chat(qa_chain, crawler_agent):
    st.header("Step 3: Chat with the Bot")
    st.write("Type your messages below and the chatbot will respond with empathy and support.")

    for message in st.session_state.chat_history:
        st.write(f"**{'You' if message['role'] == 'human' else 'Chatbot'}:** {message['content']}")

    user_input = st.text_input("You:", key="user_input")

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            sanitized_input = sanitize_input(user_input)
            
            st_callback = StreamlitCallbackHandler(st.container())
            
            with st.spinner("Generating response..."):
                response = qa_chain({"question": sanitized_input}, callbacks=[st_callback])
            
            if "I don't have enough information" in response['answer']:
                crawled_info = crawler_agent.crawl(sanitized_input)
                response['answer'] += f"\n\nAdditional information:\n{crawled_info}"

            st.session_state.chat_history.append({"role": "human", "content": sanitized_input})
            st.session_state.chat_history.append({"role": "ai", "content": response['answer']})

            st.rerun()

    if st.button("End Chat"):
        st.success("Chatbot: Take care. Remember, seeking professional help is important.")
        if st.button("Restart"):
            reset_session()
            st.rerun()

def collect_symptoms_responses(responses: dict) -> list:
    return [extract_symptom_from_question(question) for question, answer in responses.items() if answer.lower() == "yes"]

def extract_symptom_from_question(question: str) -> str:
    symptom_map = {
        "sad or down": "sadness",
        "lost interest": "loss of interest",
        "excessive worry": "worry",
        "changes in your sleep": "changes in sleep patterns",
        "fatigued": "fatigue",
        "difficulty concentrating": "difficulty concentrating",
        "feelings of hopelessness": "hopelessness",
        "thoughts of self-harm": "thoughts of self-harm"
    }
    
    for key, value in symptom_map.items():
        if key in question.lower():
            return value
    return "unknown_symptom"

def reset_session():
    st.session_state.step = 1
    st.session_state.symptoms = []
    st.session_state.diagnosis = ""
    st.session_state.chat_history = []

if __name__ == "__main__":
    main()

