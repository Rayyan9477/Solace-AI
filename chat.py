import random
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as transformers_logging
from context_manager import ContextManager
import logging

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

def chat():
    print("Let's chat! (type 'quit' to exit)")
    user_id = "default_user"
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break
        resp = get_response(sentence, user_id)
        print(f"{bot_name}: {resp}")
        log_interaction(user_id, sentence, resp)

def test_model_accuracy():
    test_questions = [
        ("Can you help me with something?", ["Sure, how can I assist you?"]),
        ("Can you help me with something?", ["Sure, how can I assist you?"]),
        ("Can you help me with something?", ["Sure, how can I assist you?"]),
        ("I am feeling lonely", ["I'm sorry to hear that. I'm here for you. Talking about it might help. So, tell me why do you think you're feeling this way?", "I'm here for you. Could you tell me why you're feeling this way?", "Why do you think you feel this way?", "How long have you been feeling this way?"]),
        ("I am feeling lonely", ["I'm sorry to hear that. I'm here for you. Talking about it might help. So, tell me why do you think you're feeling this way?", "I'm here for you. Could you tell me why you're feeling this way?", "Why do you think you feel this way?", "How long have you been feeling this way?"]),
        ("I am feeling lonely", ["I'm sorry to hear that. I'm here for you. Talking about it might help. So, tell me why do you think you're feeling this way?", "I'm here for you. Could you tell me why you're feeling this way?", "Why do you think you feel this way?", "How long have you been feeling this way?"]),
        ("I feel so worthless.", ["It's only natural to feel this way. Tell me more. What else is on your mind?", "Let's discuss further why you're feeling this way.", "I first want to let you know that you are not alone in your feelings and there is always someone there to help . you can always change your feelings and change your way of thinking by being open to trying to change.", "i first want to let you know that you are not alone in your feelings and there is always someone there to help . you can always change your feelings and change your way of thinking by being open to trying to change."]),
        ("I feel so worthless.", ["It's only natural to feel this way. Tell me more. What else is on your mind?", "Let's discuss further why you're feeling this way.", "I first want to let you know that you are not alone in your feelings and there is always someone there to help . you can always change your feelings and change your way of thinking by being open to trying to change.", "i first want to let you know that you are not alone in your feelings and there is always someone there to help . you can always change your feelings and change your way of thinking by being open to trying to change."]),
        ("I feel so worthless.", ["It's only natural to feel this way. Tell me more. What else is on your mind?", "Let's discuss further why you're feeling this way.", "I first want to let you know that you are not alone in your feelings and there is always someone there to help . you can always change your feelings and change your way of thinking by being open to trying to change.", "i first want to let you know that you are not alone in your feelings and there is always someone there to help . you can always change your feelings and change your way of thinking by being open to trying to change."]),
        ("I feel so anxius.", ["Don't be hard on yourself. What's the reason behind this?", "Can you tell me more about this feeling?", "I understand that it can be scary. Tell me more about it.", "Don't let the little worries bring you down. What's the worse that can happen?"]),
        ("I feel so anxius.", ["Don't be hard on yourself. What's the reason behind this?", "Can you tell me more about this feeling?", "I understand that it can be scary. Tell me more about it.", "Don't let the little worries bring you down. What's the worse that can happen?"]),
        ("I feel so anxius.", ["Don't be hard on yourself. What's the reason behind this?", "Can you tell me more about this feeling?", "I understand that it can be scary. Tell me more about it.", "Don't let the little worries bring you down. What's the worse that can happen?"]),
        ("I can't take it anymore", ["It helps to talk about what's happening. You're going to be okay", "Talk to me. Tell me more. It helps if you open up yourself to someone else.", "Sometimes when we are depressed, it is hard to care about anything. It can be hard to do the simplest of things. Give yourself time to heal."]),
        ("I can't take it anymore", ["It helps to talk about what's happening. You're going to be okay", "Talk to me. Tell me more. It helps if you open up yourself to someone else.", "Sometimes when we are depressed, it is hard to care about anything. It can be hard to do the simplest of things. Give yourself time to heal."]),
        ("I can't take it anymore", ["It helps to talk about what's happening. You're going to be okay", "Talk to me. Tell me more. It helps if you open up yourself to someone else.", "Sometimes when we are depressed, it is hard to care about anything. It can be hard to do the simplest of things. Give yourself time to heal."]),
        ("I am so stressed out", ["What do you think is causing this?", "Take a breath and gather your thoughts. Go take a walk if possible. Stay hydrated", "Give yourself a break. Go easy on yourself.", "I am sorry to hear that. What is the reason behind this?"]),
        ("I am so stressed out", ["What do you think is causing this?", "Take a breath and gather your thoughts. Go take a walk if possible. Stay hydrated", "Give yourself a break. Go easy on yourself.", "I am sorry to hear that. What is the reason behind this?"]),
        ("I am so stressed out", ["What do you think is causing this?", "Take a breath and gather your thoughts. Go take a walk if possible. Stay hydrated", "Give yourself a break. Go easy on yourself.", "I am sorry to hear that. What is the reason behind this?"]),
        ("I have insominia", ["What do you think is the reason behind this?", "That seem awful. What do you think is behind this?"]),
        ("I have insominia", ["What do you think is the reason behind this?", "That seem awful. What do you think is behind this?"]),
        ("I have insominia", ["What do you think is the reason behind this?", "That seem awful. What do you think is behind this?"]),
        ("I'm scared", ["It's only natural to feel this way. I'm here for you.", "It'll all be okay. This feeling is only momentary.", "I understand how you feel. Don't put yourself down because of it."]),
        ("I'm scared", ["It's only natural to feel this way. I'm here for you.", "It'll all be okay. This feeling is only momentary.", "I understand how you feel. Don't put yourself down because of it."]),
        ("I'm scared", ["It's only natural to feel this way. I'm here for you.", "It'll all be okay. This feeling is only momentary.", "I understand how you feel. Don't put yourself down because of it."]),
        ("My mom died", ["I'm sorry to hear that. If you want to talk about it. I'm here.", "I am really sorry to hear that. I am here to help you with grief, anxiety and anything else you may feel at this time.", "My condolences. I'm here if you need to talk."]),
        ("My mom died", ["I'm sorry to hear that. If you want to talk about it. I'm here.", "I am really sorry to hear that. I am here to help you with grief, anxiety and anything else you may feel at this time.", "My condolences. I'm here if you need to talk."]),
        ("My mom died", ["I'm sorry to hear that. If you want to talk about it. I'm here.", "I am really sorry to hear that. I am here to help you with grief, anxiety and anything else you may feel at this time.", "My condolences. I'm here if you need to talk."]),
        ("You don't understand me.", ["It sound like i'm not being very helpful right now.", "I'm sorry to hear that. I'm doing my best to help", "I'm trying my best to help you. So please talk to me"]),
        ("You don't understand me.", ["It sound like i'm not being very helpful right now.", "I'm sorry to hear that. I'm doing my best to help", "I'm trying my best to help you. So please talk to me"]),
        ("You don't understand me.", ["It sound like i'm not being very helpful right now.", "I'm sorry to hear that. I'm doing my best to help", "I'm trying my best to help you. So please talk to me"]),
        ("That's all.", ["I heard you & noted it all. See you later.", "Oh okay we're done for today then. See you later", "I hope you have a great day. See you soon", "Okay we're done. Have a great day", "Okay I see. Enjoy the rest of your day then"]),
        ("That's all.", ["I heard you & noted it all. See you later.", "Oh okay we're done for today then. See you later", "I hope you have a great day. See you soon", "Okay we're done. Have a great day", "Okay I see. Enjoy the rest of your day then"]),
        ("That's all.", ["I heard you & noted it all. See you later.", "Oh okay we're done for today then. See you later", "I hope you have a great day. See you soon", "Okay we're done. Have a great day", "Okay I see. Enjoy the rest of your day then"])
    ]

    correct_responses = 0
    user_id = "test_user"
    for question, expected_responses in test_questions:
        response = get_response(question, user_id)
        print(f"Question: {question}")
        print(f"Response: {response}")
        print("-" * 50)
        if response in expected_responses:
            correct_responses += 1


def main_menu():
    print("Welcome to the Chatbot!")
    print("1. Use automated prompts")
    print("2. Enter prompts manually")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        test_model_accuracy()
    elif choice == "2":
        chat()
    else:
        print("Invalid choice. Please enter 1 or 2.")
        main_menu()

if __name__ == "__main__":
    main_menu()