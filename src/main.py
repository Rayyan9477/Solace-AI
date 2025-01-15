import streamlit as st
from config.settings import AppConfig
from database.vector_store import ChromaVectorStore
from agents.search_agent import SearchAgent
from agents.chat_agent import ChatAgent
from utils.helpers import sanitize_input, get_basic_embedding

# Initialize Streamlit app
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🤖", layout="wide")

def main():
    # Load configuration
    config = AppConfig()

    # Initialize vector store
    vector_store = ChromaVectorStore(collection_name=config.VECTOR_DB_COLLECTION)
    vector_store.connect()

    # Initialize search agent with ChromaDB vector store
    search_agent = SearchAgent(
        vector_store=vector_store,
        search_api_key="",  # No API key needed for local ChromaDB
        fallback_engine=""   # Not used since we're using ChromaDB
    )

    # Initialize chat agent
    chat_agent = ChatAgent(
        model_name=config.MODEL_NAME,
        use_cpu=config.USE_CPU
    )

    st.title("🤖 Mental Health Chatbot")
    st.write("Welcome to the Mental Health Chatbot. Let's get started.")

    if "step" not in st.session_state:
        st.session_state.step = 1
        st.session_state.symptoms = []
        st.session_state.diagnosis = ""
        st.session_state.chat_history = []

    if st.session_state.step == 1:
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
            diagnosis = chat_agent.diagnose_condition(symptoms)
            st.session_state.diagnosis = diagnosis
            st.session_state.step = 2

    elif st.session_state.step == 2:
        st.header("Step 2: Diagnosis")
        st.write(f"**Diagnosis:** {st.session_state.diagnosis}")

        if "depression" in st.session_state.diagnosis.lower() or "anxiety" in st.session_state.diagnosis.lower():
            st.success("I'm here to help you. Let's discuss how you're feeling.")
            st.session_state.step = 3
        else:
            st.balloons()
            st.info("It's good to hear that you're doing well. If you ever need someone to talk to, I'm here.")
            if st.button("Restart"):
                st.session_state.step = 1
                st.session_state.symptoms = []
                st.session_state.diagnosis = ""
                st.session_state.chat_history = []
                st.experimental_rerun()

    elif st.session_state.step == 3:
        st.header("Step 3: Chat with the Bot")

        st.write("Type your messages below and the chatbot will respond with empathy and support.")

        # Display chat history
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**Chatbot:** {chat['content']}")

        # Input area for user message
        user_input = st.text_input("You:", key="user_input")

        if st.button("Send"):
            if user_input.strip() == "":
                st.warning("Please enter a message.")
            else:
                sanitized_input = sanitize_input(user_input)
                st.session_state.chat_history.append({"role": "user", "content": sanitized_input})

                # Retrieve context using RAG
                embedding = get_basic_embedding(sanitized_input)
                context = search_agent.retrieve_context(sanitized_input)

                # Generate response with context
                response = chat_agent.generate_response(sanitized_input, context)
                st.session_state.chat_history.append({"role": "chatbot", "content": response})

                # Clear the input box and rerun to update chat history
                st.experimental_rerun()

        if st.button("End Chat"):
            st.success("Chatbot: Take care. Remember, seeking professional help is important.")
            if st.button("Restart"):
                st.session_state.step = 1
                st.session_state.symptoms = []
                st.session_state.diagnosis = ""
                st.session_state.chat_history = []
                st.experimental_rerun()

def collect_symptoms_responses(responses: dict) -> list:
    """
    Collects user symptoms based on their responses.
    Returns a list of symptoms reported by the user.
    """
    symptoms = []
    for question, answer in responses.items():
        if answer.lower() == "yes":
            symptom = extract_symptom_from_question(question)
            symptoms.append(symptom)
    return symptoms

def extract_symptom_from_question(question: str) -> str:
    """
    Extracts a key symptom from a question string.
    """
    if "sad or down" in question.lower():
        return "sadness"
    elif "lost interest" in question.lower():
        return "loss of interest"
    elif "excessive worry" in question.lower():
        return "worry"
    elif "changes in your sleep" in question.lower():
        return "changes in sleep patterns"
    elif "fatigued" in question.lower():
        return "fatigue"
    elif "difficulty concentrating" in question.lower():
        return "difficulty concentrating"
    elif "feelings of hopelessness" in question.lower():
        return "hopelessness"
    elif "thoughts of self-harm" in question.lower():
        return "thoughts of self-harm"
    else:
        return "unknown_symptom"

if __name__ == "__main__":
    main()