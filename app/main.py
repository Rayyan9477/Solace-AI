import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import yaml
import os
from utils.model_handler import ModelHandler
from app.chat_handler import ChatHandler

# Load configuration
@st.cache_resource
def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    config = load_config()
    model_path = config["model"]["path"]
    handler = ModelHandler(model_path)
    return handler.load_model_and_tokenizer()

def initialize_chat():
    if "chat_handler" not in st.session_state:
        try:
            config = load_config()
            model_handler = ModelHandler(config["model"]["path"])
            generator = model_handler.load_model_and_tokenizer()
            st.session_state.chat_handler = ChatHandler(generator, config)
        except Exception as e:
            st.error(f"Error initializing chat: {str(e)}")
            return False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return True

def main():
    st.title("Health Chatbot")
    
    if not initialize_chat():
        st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you with your health-related questions?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_handler.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()