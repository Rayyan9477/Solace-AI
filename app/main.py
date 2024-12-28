import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from model import HealthChatModel
from vector_store import VectorStore
from config.config import Config
import torch
import time

# Custom CSS for modern chat interface
def apply_custom_css():
    st.markdown("""
        <style>
        /* Chat container */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Message input */
        .stTextInput {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 80%;
            max-width: 800px;
            background: white;
            padding: 10px;
            border-radius: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }
        
        /* Chat messages */
        .stChatMessage {
            padding: 10px 20px;
            border-radius: 15px;
            margin: 5px 0;
            max-width: 80%;
        }
        
        .user-message {
            background-color: #e3f2fd;
            margin-left: auto;
        }
        
        .bot-message {
            background-color: #f5f5f5;
            margin-right: auto;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = HealthChatModel()
        st.session_state.model.load_model()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore(dimension=384)

def display_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

def main():
    st.set_page_config(
        page_title="Health Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    apply_custom_css()
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Options")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        st.header("About")
        st.markdown("""
        This AI healthcare assistant can help you with:
        - Health-related questions
        - Medical information
        - General wellness advice
        
        Please note: This is not a replacement for professional medical advice.
        """)

    # Main chat interface
    st.title("Healthcare Assistant 🏥")
    
    # Chat messages container with bottom padding for input
    chat_container = st.container()
    padding_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            display_message(
                message["role"],
                message["content"]
            )
    
    # Add padding to prevent messages from being hidden behind input
    with padding_container:
        st.markdown("<div style='padding-bottom: 100px;'></div>", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Type your message here...")
    if prompt:
        # Display user message
        display_message("user", prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Process the response
                    response = st.session_state.model.generate_response(prompt)
                    
                    # Display the response with typing effect
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)  # Adjust speed as needed
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    # Save to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": full_response}
                    )
                    
                except Exception as e:
                    st.error("I apologize, but I'm having trouble generating a response. Please try again.")
                    print(f"Error details: {str(e)}")  # For debugging

if __name__ == "__main__":
    main()