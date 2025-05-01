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
        
        # Set RTC configuration for WebRTC (using Google's STUN servers)
        self.rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
    def render_voice_input(self):
        """Render voice input component"""
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
            result = asyncio.run(self.voice_ai.speech_to_text(audio_data))
            
            if result["success"] and result["text"]:
                # Clear the status and show the transcribed text
                status_indicator.success(f"✅ Transcribed: {result['text']}")
                
                # Call the callback if provided
                if self.on_transcription:
                    self.on_transcription(result["text"])
            else:
                status_indicator.error("❌ Failed to transcribe speech")
    
    def render_voice_output(self, text: str, autoplay: bool = True):
        """
        Render voice output component
        
        Args:
            text: Text to convert to speech
            autoplay: Whether to autoplay the audio
        """
        # Create placeholder for audio player
        audio_player = st.empty()
        
        # Convert text to speech
        result = asyncio.run(self.voice_ai.text_to_speech(text))
        
        if result["success"] and result["audio_bytes"]:
            # Create audio player with base64 encoded audio
            audio_base64 = base64.b64encode(result["audio_bytes"]).decode("utf-8")
            audio_html = f"""
                <audio id="audio-player" {"autoplay" if autoplay else ""} controls>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            """
            audio_player.markdown(audio_html, unsafe_allow_html=True)
            return True
        else:
            st.error("Failed to generate speech")
            return False

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
        
        return selected_voice