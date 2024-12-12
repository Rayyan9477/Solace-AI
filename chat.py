import random
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as transformers_logging
from context_manager import ContextManager
import logging
import streamlit as st  
import os
import train

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("Using CPU")

# Load intents
with open('data.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained model and tokenizer
model_name = 'fine_tuned_model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

bot_name = "Roy"
context_manager = ContextManager()
logging.basicConfig(level=logging.DEBUG, filename='chatbot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_interaction(user_id, user_input, bot_response):
    logging.info(f"User ID: {user_id}, User Input: {user_input}, Bot Response: {bot_response}")

def get_response(sentence, user_id):
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    tag = intents['intents'][predicted_class_id]['tag']
    probs = torch.softmax(logits, dim=1)
    prob = probs[0][predicted_class_id]

    logging.debug(f"Input Sentence: {sentence}")
    logging.debug(f"Predicted class ID: {predicted_class_id}")
    logging.debug(f"Tag: {tag}")
    logging.debug(f"Probability: {prob.item()}")

    if prob.item() > 0.01:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                logging.debug(f"Matched Intent: {intent['tag']}")
                if 'context_set' in intent:
                    context_manager.add_to_context(user_id, 'context', intent['context_set'])
                if 'context_filter' not in intent or context_manager.get_from_context(user_id, 'context') == intent['context_filter']:
                    response = random.choice(intent['responses'])
                    logging.debug(f"Response: {response}")
                    return response

    logging.debug("No matching intent found or probability too low.")
    return "I'm not sure how to respond to that. Can you please rephrase or ask something else?"

# Streamlit app
st.set_page_config(page_title="Chatbot", page_icon=":speech_balloon:")

# Load custom CSS from file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🤖 Chat with Roy")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# User input
def get_text():
    input_text = st.text_input("You:", "", key="input")
    return input_text

user_input = get_text()

if user_input:
    user_id = "user"
    response = get_response(user_input, user_id)
    st.session_state.conversation.append(("You", user_input))
    st.session_state.conversation.append((bot_name, response))

# Display conversation
for i in range(len(st.session_state.conversation)):
    speaker, message = st.session_state.conversation[i]
    if speaker == "You":
        st.markdown(f"<div class='chat-container' style='align-items: flex-end;'><div class='message user-message'>{message}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-container' style='align-items: flex-start;'><div class='message bot-message'>{message}</div></div>", unsafe_allow_html=True)

# Footer with contact details
st.markdown("""
    <div class="footer">
        <p>Contact:
        <a href="mailto:rayyanahmed265@yahoo.com">rayyanahmed265@yahoo.com</a> |
        <a href="https://github.com/Rayyan9477">GitHub</a> |
        <a href="https://www.linkedin.com/in/rayyan-ahmed9477/">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)