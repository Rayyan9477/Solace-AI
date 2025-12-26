"""
Audio playback module for playing generated speech.
Handles playing audio data through the device's speakers.
"""

import sounddevice as sd
import soundfile as sf
import io
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import time

# Configure logger
logger = logging.getLogger(__name__)

# Audio validation constants
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB max audio file
MIN_AUDIO_SIZE_BYTES = 44  # Minimum WAV header size
MAX_DURATION_SECONDS = 600  # 10 minutes max
MIN_SAMPLE_RATE = 8000  # 8 kHz minimum
MAX_SAMPLE_RATE = 192000  # 192 kHz maximum
SUPPORTED_FORMATS = {'WAV', 'FLAC', 'OGG', 'RAW'}

# WAV file magic bytes
WAV_MAGIC_RIFF = b'RIFF'
WAV_MAGIC_WAVE = b'WAVE'

class AudioPlayer:
    """Audio player for TTS output"""

    def __init__(self):
        """Initialize the audio player"""
        self.is_playing = False
        self.current_audio = None
        logger.info("AudioPlayer initialized")

    def _validate_audio_bytes(self, audio_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validate audio bytes for format and size constraints.

        Args:
            audio_bytes: Raw audio data as bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Size validation
        audio_size = len(audio_bytes)

        if audio_size < MIN_AUDIO_SIZE_BYTES:
            return False, f"Audio data too small ({audio_size} bytes). Minimum is {MIN_AUDIO_SIZE_BYTES} bytes."

        if audio_size > MAX_AUDIO_SIZE_BYTES:
            return False, f"Audio data too large ({audio_size / (1024*1024):.2f} MB). Maximum is {MAX_AUDIO_SIZE_BYTES / (1024*1024):.0f} MB."

        # Format validation - check for WAV magic bytes
        if len(audio_bytes) >= 12:
            riff_header = audio_bytes[:4]
            wave_format = audio_bytes[8:12]

            if riff_header == WAV_MAGIC_RIFF and wave_format == WAV_MAGIC_WAVE:
                # Valid WAV format
                logger.debug("Audio format validated: WAV")
                return True, None

            # Try to detect other formats via soundfile
            try:
                with io.BytesIO(audio_bytes) as buffer:
                    info = sf.info(buffer)
                    format_name = info.format.upper()

                    if format_name not in SUPPORTED_FORMATS:
                        return False, f"Unsupported audio format: {format_name}. Supported: {', '.join(SUPPORTED_FORMATS)}"

                    # Validate sample rate
                    if not (MIN_SAMPLE_RATE <= info.samplerate <= MAX_SAMPLE_RATE):
                        return False, f"Sample rate {info.samplerate} Hz out of range ({MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz)"

                    # Validate duration
                    duration = info.duration
                    if duration > MAX_DURATION_SECONDS:
                        return False, f"Audio duration {duration:.1f}s exceeds maximum {MAX_DURATION_SECONDS}s"

                    logger.debug(f"Audio format validated: {format_name}, {info.samplerate}Hz, {duration:.1f}s")
                    return True, None

            except Exception as e:
                return False, f"Could not validate audio format: {str(e)}"

        return False, "Audio data too short to determine format"
        
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

        # Validate audio format and size
        is_valid, validation_error = self._validate_audio_bytes(audio_bytes)
        if not is_valid:
            logger.warning(f"Audio validation failed: {validation_error}")
            return {
                "success": False,
                "error": validation_error
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

        # Validate audio format and size
        is_valid, validation_error = self._validate_audio_bytes(audio_bytes)
        if not is_valid:
            logger.warning(f"Audio validation failed: {validation_error}")
            return {
                "success": False,
                "error": validation_error
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