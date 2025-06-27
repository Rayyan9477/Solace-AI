"""
Chat interface component for the mental health chatbot.
Supports both voice and text-based interactions with an empathetic UI.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Callable
import time
import asyncio
import logging
from .base_component import BaseUIComponent

logger = logging.getLogger(__name__)

class ChatInterfaceComponent(BaseUIComponent):
    """Component for rendering a modern, empathetic chat interface"""
    
    def __init__(self, 
                 process_message_callback: Callable[[str], Dict[str, Any]],
                 on_back: Optional[Callable] = None,
                 voice_enabled: bool = False,
                 voice_component = None,
                 whisper_voice_input = None):
        """
        Initialize the chat interface component
        
        Args:
            process_message_callback: Callback to process user messages
            on_back: Callback when user clicks back button
            voice_enabled: Whether voice interaction is enabled
            voice_component: Voice component for speech input/output (if enabled)
            whisper_voice_input: Enhanced Whisper V3 Turbo voice input manager
        """
        super().__init__(name="ChatInterface", description="Empathetic chat interface for mental health support")
        self.process_message = process_message_callback
        self.on_back = on_back
        self.voice_enabled = voice_enabled
        self.voice_component = voice_component
        self.whisper_voice_input = whisper_voice_input
        
        # Apply custom styling
        self.apply_chat_css()
        
        # Initialize session state for this component
        if "chat_state" not in st.session_state:
            st.session_state.chat_state = {
                "messages": [],
                "typing": False,
                "last_user_input": "",
                "spoken_messages": set(),
                "auto_speak_responses": True if voice_enabled else False,
                "show_emotions": True,
                "use_whisper": whisper_voice_input is not None
            }
    
    def render(self, **kwargs):
        """Render the chat interface"""
        # Header with back button
        col1, col2 = st.columns([1, 9])
        with col1:
            if st.button("← Back", key="btn_chat_back"):
                if self.on_back:
                    self.on_back()
        
        with col2:
            st.markdown("<h2 class='chat-header'>Your Support Conversation</h2>", unsafe_allow_html=True)
        
        # Settings expander
        with st.expander("⚙️ Chat Settings", expanded=False):
            col_settings1, col_settings2 = st.columns(2)
            
            with col_settings1:
                st.checkbox("Show emotion indicators", 
                           value=st.session_state.chat_state["show_emotions"],
                           key="cb_show_emotions",
                           on_change=self._on_show_emotions_changed)
            
            with col_settings2:
                if self.voice_enabled:
                    st.checkbox("Auto-speak responses", 
                               value=st.session_state.chat_state["auto_speak_responses"],
                               key="cb_auto_speak",
                               on_change=self._on_auto_speak_changed)
            
            # Reset chat button
            if st.button("Reset Conversation", key="btn_reset_chat"):
                self._reset_chat()
        
        # Display voice controls if enabled
        if self.voice_enabled and (self.voice_component or self.whisper_voice_input):
            with st.expander("🎤 Voice Controls", expanded=False):
                col_voice1, col_voice2 = st.columns(2)
                
                with col_voice1:
                    st.markdown("### 🎙️ Speak to Assistant")
                    
                    # Render standard voice input component
                    if hasattr(self.voice_component, "render_voice_input"):
                        self.voice_component.render_voice_input(
                            on_input=self._handle_voice_input
                        )
                    
                    # Add Whisper V3 Turbo speech recognition
                    if self.whisper_voice_input and st.session_state.chat_state.get("use_whisper", False):
                        st.markdown("#### 🚀 Enhanced Speech Recognition")
                        if st.button("🎙️ Speak with Whisper V3 Turbo", key="btn_whisper_speak"):
                            with st.spinner("Listening..."):
                                try:
                                    result = self.whisper_voice_input.transcribe_once()
                                    if result.get("success", False) and result.get("text"):
                                        # Process the transcribed text
                                        self._handle_voice_input(result["text"])
                                        st.success(f"Transcribed: {result['text']}")
                                    else:
                                        st.error(f"Could not transcribe audio: {result.get('error', 'Unknown error')}")
                                except Exception as e:
                                    logger.error(f"Error with Whisper transcription: {str(e)}")
                                    st.error(f"Error: {str(e)}")
                
                with col_voice2:
                    st.markdown("### 🔊 Voice Settings")
                    
                    # Regular voice selector for TTS
                    if hasattr(self.voice_component, "render_voice_selector"):
                        self.voice_component.render_voice_selector()
                    
                    # Test voice button
                    if st.button("Test Voice", key="btn_test_voice"):
                        self._test_voice()
                    
                    # Whisper ASR settings
                    if self.whisper_voice_input:
                        st.checkbox("Use Whisper V3 Turbo for speech recognition", 
                                   value=st.session_state.chat_state.get("use_whisper", False),
                                   key="cb_use_whisper",
                                   on_change=self._on_whisper_changed)
                        
                        if st.session_state.chat_state.get("use_whisper", False):
                            st.markdown("""
                            <div style='background-color: #f0f7ff; padding: 10px; border-radius: 10px; margin-top: 10px;'>
                                <p style='margin: 0; font-size: 0.9rem;'>
                                    <b>📝 Whisper V3 Turbo</b>: Enhanced speech recognition for more accurate transcription
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Message container
        message_container = st.container()
        
        # User input area - below messages but will be processed first
        user_input = self._render_user_input_area()
        
        # Process any pending voice input
        voice_input = kwargs.get("voice_input")
        if voice_input:
            user_input = voice_input
        
        # Process user input if available
        if user_input:
            # Add user message to chat
            self._add_user_message(user_input)
            
            # Process message and get response
            with st.spinner("Thinking..."):
                try:
                    result = self.process_message(user_input)
                    
                    if result:
                        # Check for errors
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            # Add assistant message to chat
                            self._add_assistant_message(
                                result.get("response", "I'm not sure how to respond to that."),
                                emotion_data=result.get("emotion_analysis")
                            )
                            
                            # Handle safety alerts or escalation
                            if result.get("requires_escalation", False):
                                st.warning("⚠️ This situation may require professional support. Please consider reaching out to a mental health professional.")
                                
                            # Speak response if auto-speak is enabled
                            if self.voice_enabled and st.session_state.chat_state["auto_speak_responses"]:
                                if hasattr(self.voice_component, "speak_text"):
                                    self.voice_component.speak_text(result.get("response", ""))
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")
        
        # Display messages in the message container
        with message_container:
            self._render_messages()
            
            # Show typing indicator
            if st.session_state.chat_state["typing"]:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown("⏳ *typing...*")
    
    def _render_user_input_area(self):
        """Render the user input area and return any input provided"""
        user_input = st.chat_input("Type your message here...", key="chat_user_input")
        return user_input
    
    def _render_messages(self):
        """Render all messages in the chat"""
        messages = st.session_state.chat_state["messages"]
        
        # If no messages yet, show a welcome message
        if not messages:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown("# Welcome to your supportive conversation\n\nI'm here to listen, understand, and provide empathetic support for your mental well-being. How are you feeling today?")
            
            # Add this welcome message to the chat history
            st.session_state.chat_state["messages"].append({
                "role": "assistant",
                "content": "# Welcome to your supportive conversation\n\nI'm here to listen, understand, and provide empathetic support for your mental well-being. How are you feeling today?"
            })
            
            # Speak welcome message if auto-speak is enabled
            if self.voice_enabled and st.session_state.chat_state["auto_speak_responses"]:
                if hasattr(self.voice_component, "speak_text"):
                    self.voice_component.speak_text("Welcome to your supportive conversation. I'm here to listen, understand, and provide empathetic support for your mental well-being. How are you feeling today?")
        else:
            # Display all messages
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                # Select avatar based on role
                avatar = "🧠" if role == "assistant" else None
                
                with st.chat_message(role, avatar=avatar):
                    st.markdown(content)
                    
                    # Show emotion if available and enabled
                    if role == "assistant" and st.session_state.chat_state["show_emotions"]:
                        emotion_data = message.get("emotion_data", {})
                        if emotion_data:
                            primary_emotion = emotion_data.get("primary_emotion", "neutral")
                            self._render_emotion_indicator(primary_emotion)
    
    def _render_emotion_indicator(self, emotion: str):
        """Render a visual indicator for the detected emotion"""
        emotion_emojis = {
            "joy": "😊",
            "sadness": "😔",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😮",
            "disgust": "😖",
            "neutral": "😐",
            "empathy": "🤗",
            "concern": "🙁",
            "encouragement": "💪",
            "curiosity": "🤔"
        }
        
        emoji = emotion_emojis.get(emotion.lower(), "😐")
        
        st.caption(f"Emotional tone: {emoji} {emotion.capitalize()}")
    
    def _add_user_message(self, content: str):
        """Add a user message to the chat"""
        st.session_state.chat_state["messages"].append({
            "role": "user",
            "content": content
        })
        
        # Store last user input
        st.session_state.chat_state["last_user_input"] = content
    
    def _add_assistant_message(self, content: str, emotion_data: Optional[Dict[str, Any]] = None):
        """Add an assistant message to the chat"""
        st.session_state.chat_state["messages"].append({
            "role": "assistant",
            "content": content,
            "emotion_data": emotion_data
        })
    
    def _handle_voice_input(self, text: str):
        """Handle voice input from the voice component"""
        if text:
            # Force a rerun with the voice input
            st.rerun(voice_input=text)
    
    def _test_voice(self):
        """Test the voice output with a sample message"""
        if self.voice_enabled and hasattr(self.voice_component, "speak_text"):
            test_text = "Hello! I'm your mental health assistant. I'm here to listen and support you with empathy and compassion."
            self.voice_component.speak_text(test_text)
    
    def _reset_chat(self):
        """Reset the chat state"""
        st.session_state.chat_state["messages"] = []
        st.session_state.chat_state["last_user_input"] = ""
        st.session_state.chat_state["spoken_messages"] = set()
        st.rerun()
    
    def _on_auto_speak_changed(self):
        """Handle change in auto-speak setting"""
        st.session_state.chat_state["auto_speak_responses"] = st.session_state.cb_auto_speak
    
    def _on_show_emotions_changed(self):
        """Handle change in show emotions setting"""
        st.session_state.chat_state["show_emotions"] = st.session_state.cb_show_emotions
    
    def _on_whisper_changed(self):
        """Handle change in Whisper V3 Turbo setting"""
        st.session_state.chat_state["use_whisper"] = st.session_state.cb_use_whisper
    
    def apply_chat_css(self):
        """Apply custom CSS for the chat interface"""
        chat_css = """
        /* Global styles */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        
        .stApp {
            font-family: 'DM Sans', sans-serif;
        }
        
        /* Chat header */
        .chat-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        /* Message styling */
        [data-testid="stChatMessage"] {
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* User message */
        [data-testid="stChatMessage"][data-testid="user"] {
            background-color: #f0f7ff;
        }
        
        /* Assistant message */
        [data-testid="stChatMessage"]:not([data-testid="user"]) {
            background-color: #f8f9fa;
        }
        
        /* Emotion indicator */
        .stChatMessage .stCaption {
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: #666;
            display: inline-block;
            padding: 0.3rem 0.7rem;
            background: #f0f0f0;
            border-radius: 20px;
        }
        
        /* Input area */
        .stChatInputContainer {
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
        
        .stChatInputContainer textarea {
            border-radius: 20px;
            border: 1px solid #e0e0e0;
            padding: 0.8rem 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .stChatInputContainer textarea:focus {
            border-color: #9896f0;
            box-shadow: 0 2px 15px rgba(152, 150, 240, 0.2);
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 20px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Back button */
        [data-testid="baseButton-secondary"] {
            background-color: #f0f0f0;
            color: #333;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #555;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 0.6rem 1rem;
            margin-bottom: 1rem;
        }
        
        .streamlit-expanderContent {
            background-color: #fff;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 0.5rem;
            border: 1px solid #f0f0f0;
        }
        """
        
        st.markdown(f"""
        <style>
        {chat_css}
        </style>
        """, unsafe_allow_html=True)