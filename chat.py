import random
import json
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from context_manager import ContextManager
import logging
import streamlit as st
from safetensors import safe_open
import os

# Setup logging and device
logging.basicConfig(level=logging.DEBUG, filename='chatbot.log', filemode='a',
                   format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cpu')
logging.info("Using CPU")

# Load intents
with open('data.json', 'r') as json_data:
    intents = json.load(json_data)

try:
    # Model configuration
    model_name = "RayyanAhmed9477/Health-Chatbot"
    
    # Load tokenizer and configuration
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                            local_files_only=False,
                                            trust_remote_code=True)
    
    config = AutoConfig.from_pretrained(model_name)
    
    # Initialize model with configuration
    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        local_files_only=False,
        trust_remote_code=True
    )
    
    # Load adapter weights if available
    adapter_path = os.path.join(model_name, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        with safe_open(adapter_path, framework="pt", device="cpu") as f:
            adapter_state_dict = {key: f.get_tensor(key) for key in f.keys()}
            model.load_state_dict(adapter_state_dict, strict=False)
    
    model.to(device)
    model.eval()

except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    st.error("Failed to load the Health-Chatbot model. Please check your internet connection.")
    raise

# Rest of the chat.py code remains the same
...

bot_name = "Roy"
context_manager = ContextManager()
logging.basicConfig(
    level=logging.DEBUG,
    filename='chatbot.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_interaction(user_id, user_input, bot_response):
    logging.info(
        f"User ID: {user_id}, User Input: {user_input}, Bot Response: {bot_response}"
    )

def get_response(sentence, user_id):
    # Tokenize and encode the input
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get embeddings from the last hidden state
    embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
    
    # Simple classification layer
    num_labels = len(intents['intents'])
    classifier = torch.nn.Linear(model.config.hidden_size, num_labels).to(device)
    logits = classifier(embeddings)
    
    predicted_class_id = logits.argmax(dim=-1).item()
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

    return "I'm not sure how to respond to that. Can you please rephrase or ask something else?"

# Streamlit UI
st.set_page_config(page_title="Chatbot", page_icon=":speech_balloon:")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🤖 Chat with Roy")

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

def get_text():
    input_text = st.text_input("You:", "", key="input")
    return input_text

user_input = get_text()

if user_input:
    user_id = "user"
    response = get_response(user_input, user_id)
    st.session_state.conversation.append(("You", user_input))
    st.session_state.conversation.append((bot_name, response))

for i in range(len(st.session_state.conversation)):
    speaker, message = st.session_state.conversation[i]
    if speaker == "You":
        st.markdown(
            f"<div class='chat-container' style='align-items: flex-end;'>"
            f"<div class='message user-message'>{message}</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-container' style='align-items: flex-start;'>"
            f"<div class='message bot-message'>{message}</div></div>",
            unsafe_allow_html=True
        )

st.markdown("""
    <div class="footer">
        <p>Contact:
        <a href="mailto:rayyanahmed265@yahoo.com">rayyanahmed265@yahoo.com</a> |
        <a href="https://github.com/Rayyan9477">GitHub</a> |
        <a href="https://www.linkedin.com/in/rayyan-ahmed9477/">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)