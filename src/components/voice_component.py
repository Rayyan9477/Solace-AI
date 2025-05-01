"""
Voice component for Streamlit interface.
Provides UI elements for voice interaction with the chatbot.
"""

import streamlit as st
import numpy as np
import time
import asyncio
import base64
import os
from typing import Dict, Any, Callable, Optional
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class VoiceComponent:
    """Component for handling voice interaction in Streamlit"""
    
    def __init__(self, voice_ai, on_transcription: Optional[Callable] = None):
        """
        Initialize voice component
        
        Args:
            voice_ai: VoiceAI instance for speech processing
            on_transcription: Callback function to call when transcription is ready
        """
        self.voice_ai = voice_ai
        self.on_transcription = on_transcription
        self.audio_queue = queue.Queue()
        self.recording = False
        self.model_initialized = False
        
        # Create event loop for async operations
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # Set RTC configuration for WebRTC (using Google's STUN servers)
        self.rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    def render_voice_input(self):
        """Render voice input component"""
        # Initialize models if not done yet
        if not self.model_initialized:
            with st.spinner("Loading speech recognition model..."):
                try:
                    self.loop.run_until_complete(self.voice_ai.initialize_stt())
                    self.model_initialized = True
                except Exception as e:
                    st.error(f"Failed to initialize speech recognition: {str(e)}")
                    return
        
        # Create a unique key for the WebRTC component
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=self.rtc_configuration,
            media_stream_constraints={"audio": True},
            video_processor_factory=None,
            audio_processor_factory=self._get_audio_processor_factory(),
            async_processing=True,
        )
        
        # Status indicator
        status_indicator = st.empty()
        
        if webrtc_ctx.state.playing:
            status_indicator.info("🎤 Listening... (speak now)")
            self.recording = True
        else:
            status_indicator.info("Click 'START' to begin speaking")
            self.recording = False
        
        # Check the audio queue for new transcriptions
        if not self.audio_queue.empty() and not self.recording:
            audio_data = self.audio_queue.get()
            
            # Show transcribing status
            status_indicator.info("🔄 Transcribing...")
            
            # Process the audio data asynchronously
            result = self.loop.run_until_complete(self.process_audio(audio_data))
            
            if result["success"] and result["text"]:
                # Clear the status and show the transcribed text
                status_indicator.success(f"✅ Transcribed: {result['text']}")
                
                # Call the callback if provided
                if self.on_transcription:
                    self.on_transcription(result["text"])
            else:
                status_indicator.error("❌ Failed to transcribe speech")
    
    async def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process audio data asynchronously"""
        try:
            result = await self.voice_ai.speech_to_text(audio_data)
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "text": ""}

    def render_voice_output(self, text: str, autoplay: bool = True):
        """Render voice output component"""
        try:
            # Initialize TTS model if needed
            if not hasattr(self, 'tts_initialized'):
                with st.spinner("Loading text-to-speech model..."):
                    self.loop.run_until_complete(self.voice_ai.initialize_tts())
                    self.tts_initialized = True
            
            # Generate speech
            result = self.loop.run_until_complete(self.voice_ai.text_to_speech(
                text,
                voice_style=st.session_state.get("voice_style", "warm")
            ))
            
            if result["success"] and result["audio_bytes"]:
                # Convert audio bytes to base64 for HTML audio element
                audio_base64 = base64.b64encode(result["audio_bytes"]).decode()
                
                # Create audio element with autoplay if enabled
                audio_html = f"""
                <audio {' autoplay' if autoplay else ''} controls>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.error("Failed to generate speech")
                
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
    
    def render_voice_selector(self):
        """Render voice selector component"""
        voices = {
            "default": "Default Voice",
            "male": "Male Voice",
            "female": "Female Voice", 
            "child": "Child Voice",
            "elder": "Elder Voice",
            "warm": "Warm & Compassionate"
        }
        
        selected_voice = st.selectbox(
            "Select Voice Style",
            options=list(voices.keys()),
            format_func=lambda x: voices[x],
            index=5  # Default to warm voice
        )
        
        # Save the selected voice style
        if selected_voice != st.session_state.get("voice_style"):
            self.voice_ai.set_voice_style(selected_voice)
        
        return selected_voice
    
    def _get_audio_processor_factory(self):
        """Create an audio processor factory for WebRTC"""
        audio_buffer = bytearray()
        
        class AudioProcessor:
            def __init__(self):
                self.sample_rate = 16000
                self.recording_started = False
                self.silent_frames = 0
                self.max_silence_frames = 30  # About 3 seconds of silence
            
            def recv(self, frame):
                nonlocal audio_buffer
                
                # Mark recording as started
                self.recording_started = True
                
                # Process audio frame
                sound = frame.to_ndarray()
                
                # Check for silence
                if np.abs(sound).mean() < 0.01:
                    self.silent_frames += 1
                else:
                    self.silent_frames = 0
                
                # Add frame to buffer
                audio_buffer.extend(sound.tobytes())
                
                # If we've detected enough silence after some speech, consider the recording complete
                if self.recording_started and self.silent_frames >= self.max_silence_frames and len(audio_buffer) > 0:
                    # Add audio buffer to queue
                    self.audio_queue.put(bytes(audio_buffer))
                    
                    # Reset buffer
                    audio_buffer.clear()
                    
                    # Reset silence counter
                    self.silent_frames = 0
                    
                # Return the frame for visualization/debugging
                return frame
        
        # Use current audio queue
        AudioProcessor.audio_queue = self.audio_queue
        
        return AudioProcessor