"""
Conversational assessment component for the mental health chatbot.
Provides a chat-based personality and mental health assessment.
"""

import streamlit as st
from typing import Dict, Any, Callable, Optional, List
import json
import os
import time
import asyncio
from .base_component import BaseComponent

class ConversationalAssessmentComponent(BaseComponent):
    """Component for rendering a conversational assessment interface"""
    
    def __init__(self, 
                 assessment_agent,
                 on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
                 voice_enabled: bool = False):
        """
        Initialize the conversational assessment component
        
        Args:
            assessment_agent: Agent to process assessment responses
            on_complete: Callback for when assessment is complete
            voice_enabled: Whether voice interaction is enabled
        """
        super().__init__(key="conversational_assessment")
        self.assessment_agent = assessment_agent
        self.on_complete = on_complete
        self.voice_enabled = voice_enabled
        self.questions = self._load_questions()
        
        # Apply custom CSS
        self.apply_assessment_css()
    
    def render(self):
        """Render the conversational assessment interface"""
        # Initialize conversation state if needed
        if not self.has_state("conversation_started"):
            self.set_state("conversation_started", True)
            self.set_state("current_step", "intro")
            self.set_state("messages", [])
            self.set_state("responses", {})
            self.set_state("typing_delay", 0.03)  # Seconds per character for typing effect
        
        # Get current state
        current_step = self.get_state("current_step")
        messages = self.get_state("messages", [])
        
        # Container for the conversation
        chat_container = st.container()
        
        # Container for user input
        input_container = st.container()
        
        # Process the current step
        if current_step == "intro":
            self._handle_intro()
        elif current_step == "mental_health":
            self._handle_mental_health_assessment()
        elif current_step == "personality":
            self._handle_personality_assessment()
        elif current_step == "processing":
            self._handle_processing()
        elif current_step == "complete":
            self._handle_completion()
        
        # Render conversation
        with chat_container:
            for msg in messages:
                self._render_message(msg)
            
            # Add typing indicator for the last assistant message
            if self.has_state("typing") and self.get_state("typing"):
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown("⏳ *typing...*")
        
        # Handle user input
        with input_container:
            if current_step not in ["intro", "processing", "complete"]:
                self._handle_user_input()
                
                # Voice input (if enabled)
                if self.voice_enabled and self.has_state("voice_component"):
                    voice_component = self.get_state("voice_component")
                    if hasattr(voice_component, "render_voice_input"):
                        voice_component.render_voice_input()
    
    def _render_message(self, message):
        """Render a single message in the conversation"""
        role = message.get("role", "assistant")
        content = message.get("content", "")
        
        # Select avatar based on role
        avatar = "🧠" if role == "assistant" else None
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
    
    def _handle_intro(self):
        """Handle the introduction step"""
        # Add intro message if not already present
        if not self.get_state("messages", []):
            intro_message = {
                "role": "assistant",
                "content": (
                    "# Welcome to Your Personalized Assessment\n\n"
                    "I'm going to have a conversation with you to understand your mental health "
                    "needs and personality traits. This will help me provide more personalized support.\n\n"
                    "The assessment takes about 5-10 minutes and is completely confidential. "
                    "I'll ask about your feelings, experiences, and preferences.\n\n"
                    "Ready to get started?"
                )
            }
            
            # Add with typing effect
            self._add_assistant_message_with_typing(intro_message["content"])
        
        # Add buttons to start or skip
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Assessment", key="btn_start_assessment", use_container_width=True):
                self.set_state("current_step", "mental_health")
                self.set_state("mental_health_index", 0)
                
                # Add transition message
                self._add_assistant_message_with_typing(
                    "Great! Let's start with a few questions about how you've been feeling recently."
                )
        
        with col2:
            if st.button("Skip Assessment", key="btn_skip_assessment", use_container_width=True):
                # Skip to completion with empty responses
                self.set_state("current_step", "complete")
                
                if self.on_complete:
                    self.on_complete({
                        "mental_health_responses": {},
                        "personality_responses": {}
                    })
    
    def _handle_mental_health_assessment(self):
        """Handle the mental health assessment step"""
        mental_health_index = self.get_state("mental_health_index", 0)
        
        if mental_health_index < len(self.questions.get("mental_health", [])):
            # Get current question
            question = self.questions["mental_health"][mental_health_index]
            
            # Add question if not already asked
            question_key = f"mental_health_{question['id']}_asked"
            if not self.has_state(question_key):
                self.set_state(question_key, True)
                
                # Add question to conversation
                question_text = question["text"]
                self._add_assistant_message_with_typing(question_text)
        else:
            # Mental health assessment complete, move to personality assessment
            self.set_state("current_step", "personality")
            self.set_state("personality_index", 0)
            
            # Add transition message
            self._add_assistant_message_with_typing(
                "Thanks for sharing. Now I'd like to understand more about your personality and preferences."
            )
    
    def _handle_personality_assessment(self):
        """Handle the personality assessment step"""
        personality_index = self.get_state("personality_index", 0)
        
        if personality_index < len(self.questions.get("personality", [])):
            # Get current question
            question = self.questions["personality"][personality_index]
            
            # Add question if not already asked
            question_key = f"personality_{question['id']}_asked"
            if not self.has_state(question_key):
                self.set_state(question_key, True)
                
                # Add question to conversation
                question_text = question["text"]
                self._add_assistant_message_with_typing(question_text)
        else:
            # Personality assessment complete, move to processing
            self.set_state("current_step", "processing")
            
            # Add transition message
            self._add_assistant_message_with_typing(
                "Thank you for sharing all this information. I'm now analyzing your responses to create your personalized profile..."
            )
            
            # Process responses
            self._process_responses()
    
    def _handle_processing(self):
        """Handle the processing step"""
        # This step is handled by _process_responses
        pass
    
    def _handle_completion(self):
        """Handle the completion step"""
        # Add completion message if not already added
        if not self.has_state("completion_message_added"):
            self.set_state("completion_message_added", True)
            
            # Add completion message
            self._add_assistant_message_with_typing(
                "✅ Your assessment is complete! I've analyzed your responses and created a personalized profile."
            )
    
    def _handle_user_input(self):
        """Handle user input for the current question"""
        # Get current question
        current_step = self.get_state("current_step")
        
        if current_step == "mental_health":
            index = self.get_state("mental_health_index", 0)
            questions = self.questions.get("mental_health", [])
        elif current_step == "personality":
            index = self.get_state("personality_index", 0)
            questions = self.questions.get("personality", [])
        else:
            return
        
        # Check if we have questions and the index is valid
        if not questions or index >= len(questions):
            return
            
        question = questions[index]
        options = question.get("options", [])
        
        # Create a unique key for the current question
        input_key = f"{current_step}_{question['id']}_input"
        
        # Display options as buttons if available, otherwise use text input
        if options:
            # Create horizontal button layout
            cols = st.columns(len(options))
            
            for i, option in enumerate(options):
                with cols[i]:
                    button_key = f"{input_key}_option_{i}"
                    if st.button(option["text"], key=button_key, use_container_width=True):
                        self._process_answer(question, option["text"], option["value"])
        else:
            # Use standard text input
            user_input = st.chat_input("Type your response...", key=input_key)
            
            if user_input:
                self._process_answer(question, user_input)
    
    def _process_answer(self, question, text_response, value=None):
        """Process an answer from the user"""
        # Add user message to conversation
        self._add_user_message(text_response)
        
        # Store the response
        responses = self.get_state("responses", {})
        current_step = self.get_state("current_step")
        
        # Initialize section if not exists
        if current_step not in responses:
            responses[current_step] = {}
        
        # Store response
        question_id = str(question["id"])
        responses[current_step][question_id] = {
            "text": text_response,
            "value": value if value is not None else text_response
        }
        
        # Update responses in state
        self.set_state("responses", responses)
        
        # Add an acknowledgment
        acknowledgments = [
            "I understand.",
            "Thank you for sharing that.",
            "I appreciate your honesty.",
            "Got it.",
            "Thank you for your response.",
            "I hear you."
        ]
        import random
        ack = random.choice(acknowledgments)
        
        # Add with typing effect
        self._add_assistant_message_with_typing(ack)
        
        # Move to next question
        if current_step == "mental_health":
            index = self.get_state("mental_health_index", 0) + 1
            self.set_state("mental_health_index", index)
        elif current_step == "personality":
            index = self.get_state("personality_index", 0) + 1
            self.set_state("personality_index", index)
        
        # Force a rerun to show the next question
        st.rerun()
    
    def _process_responses(self):
        """Process all responses and complete the assessment"""
        # Format responses for the assessment agent
        mental_health_responses = {}
        personality_responses = {}
        
        responses = self.get_state("responses", {})
        
        # Extract mental health responses
        if "mental_health" in responses:
            for question_id, response in responses["mental_health"].items():
                if isinstance(response["value"], (int, float, str)):
                    mental_health_responses[question_id] = response["value"]
        
        # Extract personality responses
        if "personality" in responses:
            for question_id, response in responses["personality"].items():
                if isinstance(response["value"], (int, float, str)):
                    personality_responses[question_id] = response["value"]
        
        # Now we simulate processing
        with st.spinner("Analyzing your responses..."):
            time.sleep(2)  # Simulate processing time
            
            # Move to completion
            self.set_state("current_step", "complete")
            
            # Call the callback if provided
            if self.on_complete:
                self.on_complete({
                    "mental_health_responses": mental_health_responses,
                    "personality_responses": personality_responses
                })
    
    def _add_assistant_message_with_typing(self, content):
        """Add an assistant message with a typing effect"""
        # Set typing state
        self.set_state("typing", True)
        
        # Force rerun to show typing indicator
        st.rerun()
        
        # Wait for typing delay
        time.sleep(min(len(content) * self.get_state("typing_delay"), 3))
        
        # Add message to conversation
        messages = self.get_state("messages", [])
        messages.append({"role": "assistant", "content": content})
        self.set_state("messages", messages)
        
        # Turn off typing indicator
        self.set_state("typing", False)
    
    def _add_user_message(self, content):
        """Add a user message to the conversation"""
        messages = self.get_state("messages", [])
        messages.append({"role": "user", "content": content})
        self.set_state("messages", messages)
    
    def _load_questions(self) -> Dict[str, Any]:
        """Load assessment questions from file"""
        try:
            # Try to load questions from the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'data', 'personality')
            
            questions_path = os.path.join(data_dir, 'diagnosis_questions.json')
            
            with open(questions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading assessment questions: {str(e)}")
            # Return empty questions as fallback
            return {"mental_health": [], "personality": []}
    
    def apply_assessment_css(self):
        """Apply custom CSS styling for the assessment UI"""
        assessment_css = """
        /* Chat styling */
        div.stChatMessage {
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        div.stChatMessage [data-testid="chatAvatarIcon"] {
            background: linear-gradient(135deg, #9896f0 0%, #FBC8D4 100%);
        }
        
        /* Option buttons styling */
        div.stButton button {
            background: white;
            color: #333;
            border: 1px solid #ddd;
            font-weight: 500;
            border-radius: 12px;
            padding: 0.6rem 0.3rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 0.5rem;
        }
        
        div.stButton button:hover {
            transform: translateY(-2px);
            border-color: #9896f0;
            box-shadow: 0 4px 10px rgba(152, 150, 240, 0.2);
        }
        
        /* Chat input styling */
        div.stChatInputContainer {
            border-top: 1px solid #eee;
            padding-top: 1rem;
        }
        
        div.stChatInputContainer textarea {
            border-radius: 25px;
            border: 1px solid #eee;
            padding: 0.8rem 1.2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        div.stChatInputContainer textarea:focus {
            border-color: #9896f0;
            box-shadow: 0 2px 15px rgba(152, 150, 240, 0.2);
        }
        
        /* Typography */
        .stMarkdown h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #333;
        }
        
        .stMarkdown h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            color: #333;
        }
        
        .stMarkdown p {
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 0.8rem;
            color: #444;
        }
        """
        
        self.apply_custom_css(assessment_css)