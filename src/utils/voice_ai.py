"""
Voice AI module for speech-to-text and text-to-speech capabilities.
Uses Whisper for STT and Dia-1.6B for TTS.
"""

import os
import torch
import numpy as np
import asyncio
import tempfile
import soundfile as sf
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from typing import Dict, Any, Optional
import logging
from config.settings import AppConfig
import io
import scipy.io.wavfile as wavfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class VoiceAI:
    """Voice AI module for speech processing"""
    
    def __init__(self, stt_model: str = "openai/whisper-large-v3-turbo", tts_model: str = "nari-labs/Dia-1.6B"):
        """
        Initialize VoiceAI with specified models
        
        Args:
            stt_model: Name of the speech-to-text model to use
            tts_model: Name of the text-to-speech model to use
        """
        self.stt_model_name = stt_model
        self.tts_model_name = tts_model
        self.stt_pipeline = None
        self.tts_pipeline = None
        
        # Voice style configurations
        self.voice_styles = {
            "default": {},
            "male": {"voice_preset": "male"},
            "female": {"voice_preset": "female"},
            "child": {"voice_preset": "child"},
            "elder": {"voice_preset": "elder"},
            "warm": {
                "voice_preset": "female", 
                "temperature": 0.7, 
                "top_k": 50, 
                "speaking_rate": 0.9
            }
        }
        
        self.current_voice = "warm"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create cache directory
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models', 'cache'), exist_ok=True)
        
        # Initialize models lazily (on first use)
    
    async def initialize_stt(self):
        """Initialize speech-to-text model (if not already initialized)"""
        if self.stt_pipeline is None:
            try:
                logger.info(f"Initializing STT model: {self.stt_model_name}")
                
                # Set cache directory
                cache_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'cache')
                
                # Initialize WhisperForConditionalGeneration with specific parameters
                self.stt_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.stt_model_name,
                    device=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    model_kwargs={"cache_dir": cache_dir}
                )
                
                logger.info(f"STT model initialized successfully on {self.device}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize STT model: {str(e)}")
                return False
        return True
    
    async def initialize_tts(self):
        """Initialize text-to-speech model (if not already initialized)"""
        if self.tts_pipeline is None:
            try:
                logger.info(f"Initializing TTS model: {self.tts_model_name}")
                
                # Set cache directory
                cache_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'cache')
                
                # Initialize TTS pipeline
                self.tts_pipeline = pipeline(
                    "text-to-speech",
                    model=self.tts_model_name,
                    device=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    model_kwargs={"cache_dir": cache_dir}
                )
                
                logger.info(f"TTS model initialized successfully on {self.device}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize TTS model: {str(e)}")
                return False
        return True

    async def speech_to_text(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Convert speech to text
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Initialize the model if not already done
            if not await self.initialize_stt():
                return {"success": False, "text": "", "error": "Failed to initialize STT model"}
            
            # Save audio data to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(audio_data)
            
            # Process audio file
            start_time = time.time()
            
            # Call the pipeline on the audio file
            transcription = self.stt_pipeline(
                temp_filename,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=False
            )
            
            # Remove the temporary file
            os.unlink(temp_filename)
            
            end_time = time.time()
            logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
            
            text = transcription.get("text", "")
            
            return {
                "success": True,
                "text": text,
                "time_taken": end_time - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {str(e)}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
    
    async def text_to_speech(self, text: str, voice_style: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice_style: Voice style to use (default/male/female/child/elder/warm)
            
        Returns:
            Dictionary with audio data
        """
        try:
            # Initialize the model if not already done
            if not await self.initialize_tts():
                return {"success": False, "audio_bytes": b"", "error": "Failed to initialize TTS model"}
            
            # Get voice style settings
            style = self.voice_styles.get(voice_style or self.current_voice, self.voice_styles["default"])
            
            # Add emotionally appropriate phrasing for mental health context
            enhanced_text = self._enhance_text_for_empathy(text)
            
            start_time = time.time()
            
            # Generate speech using the TTS pipeline
            output = self.tts_pipeline(
                enhanced_text,
                **style
            )
            
            audio_array = output["audio"]
            sampling_rate = output["sampling_rate"]
            
            # Convert audio array to bytes
            audio_bytes = self._audio_array_to_wav_bytes(audio_array, sampling_rate)
            
            end_time = time.time()
            logger.info(f"Speech generation completed in {end_time - start_time:.2f} seconds")
            
            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "time_taken": end_time - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            return {
                "success": False,
                "audio_bytes": b"",
                "error": str(e)
            }
    
    def set_voice_style(self, style: str):
        """Set the voice style to use for TTS"""
        if style in self.voice_styles:
            self.current_voice = style
            return True
        return False
    
    def _enhance_text_for_empathy(self, text: str) -> str:
        """
        Enhance text with SSML or other markers for more empathetic speech
        
        Args:
            text: Original text
            
        Returns:
            Enhanced text
        """
        # For Dia-1.6B, we can use simple techniques to enhance empathy
        # Add slight breaks for pacing
        text = text.replace(". ", ". <break time='300ms'/> ")
        text = text.replace("? ", "? <break time='300ms'/> ")
        text = text.replace("! ", "! <break time='300ms'/> ")
        
        # Emphasize key empathetic phrases
        empathy_phrases = [
            "understand", "feel", "support", "help",
            "there for you", "listen", "care", "important"
        ]
        
        for phrase in empathy_phrases:
            if phrase in text.lower():
                # Replace with emphasis, being careful about case
                pattern = phrase
                replacement = f"<emphasis level='moderate'>{phrase}</emphasis>"
                text = text.replace(pattern, replacement)
        
        return text
    
    def _audio_array_to_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert audio array to WAV bytes
        
        Args:
            audio_array: Audio as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Audio as WAV bytes
        """
        # Create a bytes buffer
        buffer = io.BytesIO()
        
        # Save the audio array to the buffer as WAV
        wavfile.write(buffer, sample_rate, audio_array)
        
        # Get the bytes from the buffer
        buffer.seek(0)
        wav_bytes = buffer.getvalue()
        
        return wav_bytes