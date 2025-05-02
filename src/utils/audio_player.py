"""
Audio playback module for playing generated speech.
Handles playing audio data through the device's speakers.
"""

import sounddevice as sd
import soundfile as sf
import io
import numpy as np
import logging
from typing import Dict, Any, Optional
import time

# Configure logger
logger = logging.getLogger(__name__)

class AudioPlayer:
    """Audio player for TTS output"""
    
    def __init__(self):
        """Initialize the audio player"""
        self.is_playing = False
        self.current_audio = None
        logger.info("AudioPlayer initialized")
        
    def play_audio(self, audio_bytes: bytes, sample_rate: int = 24000) -> Dict[str, Any]:
        """
        Play audio data through speakers
        
        Args:
            audio_bytes: WAV audio data as bytes
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with playback status
        """
        if not audio_bytes:
            logger.warning("No audio data provided for playback")
            return {
                "success": False,
                "error": "No audio data provided"
            }
            
        try:
            # If already playing, stop current playback
            if self.is_playing:
                self.stop_playback()
            
            # Convert bytes to numpy array using soundfile
            with io.BytesIO(audio_bytes) as audio_buffer:
                audio_data, file_sample_rate = sf.read(audio_buffer)
                
            # Use the file's sample rate if provided in the file
            if file_sample_rate:
                sample_rate = file_sample_rate
                
            # Store current audio data
            self.current_audio = {
                "data": audio_data,
                "sample_rate": sample_rate
            }
            
            # Start playback
            self.is_playing = True
            
            # Play the audio (blocking)
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait until audio is done playing
            
            # Reset playback state
            self.is_playing = False
            
            return {
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}", exc_info=True)
            self.is_playing = False
            return {
                "success": False,
                "error": str(e)
            }
    
    def play_audio_nonblocking(self, audio_bytes: bytes, sample_rate: int = 24000) -> Dict[str, Any]:
        """
        Play audio data through speakers without blocking
        
        Args:
            audio_bytes: WAV audio data as bytes
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with playback status
        """
        if not audio_bytes:
            logger.warning("No audio data provided for playback")
            return {
                "success": False,
                "error": "No audio data provided"
            }
            
        try:
            # If already playing, stop current playback
            if self.is_playing:
                self.stop_playback()
            
            # Convert bytes to numpy array using soundfile
            with io.BytesIO(audio_bytes) as audio_buffer:
                audio_data, file_sample_rate = sf.read(audio_buffer)
                
            # Use the file's sample rate if provided in the file
            if file_sample_rate:
                sample_rate = file_sample_rate
                
            # Store current audio data
            self.current_audio = {
                "data": audio_data,
                "sample_rate": sample_rate
            }
            
            # Set callback for when audio finishes
            def callback(outdata, frames, time, status):
                if status:
                    logger.warning(f"Audio playback status: {status}")
                
            # Start playback without blocking
            self.is_playing = True
            sd.play(audio_data, sample_rate, callback=callback)
            
            return {
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}", exc_info=True)
            self.is_playing = False
            return {
                "success": False,
                "error": str(e)
            }
            
    def stop_playback(self) -> None:
        """Stop current audio playback"""
        if self.is_playing:
            try:
                sd.stop()
                self.is_playing = False
                logger.info("Audio playback stopped")
            except Exception as e:
                logger.error(f"Error stopping audio playback: {str(e)}")

    def is_device_available(self) -> bool:
        """Check if audio output device is available"""
        try:
            devices = sd.query_devices()
            return any(device['max_output_channels'] > 0 for device in devices)
        except Exception as e:
            logger.error(f"Error checking audio devices: {str(e)}")
            return False