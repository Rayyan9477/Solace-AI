"""
Dia 1.6B Text-to-Speech module for high-quality speech synthesis.
Provides an interface to the Dia 1.6B model for converting text to speech.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional
import asyncio
from transformers import AutoProcessor, AutoModel
import io
import soundfile as sf
import time
import soundfile


# Configure logger
logger = logging.getLogger(__name__)

class DiaTTS:
    """Dia 1.6B Text-to-Speech module for high-quality speech synthesis"""
    
    def __init__(self, 
                 model_name: str = "nari-labs/Dia-1.6B", 
                 cache_dir: Optional[str] = None,
                 use_gpu: bool = True):
        """
        Initialize Dia 1.6B TTS module
        
        Args:
            model_name: Name of the Dia model to use from HuggingFace
            cache_dir: Directory to store model files
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.initialized = False
        self.initialization_error = None
        
        # Create cache directory if specified
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Set device
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Configure voice styles
        self.voice_styles = {
            "default": {"speaker_id": 0, "prompt": ""},
            "warm": {"speaker_id": 0, "prompt": "In a warm and empathetic tone: "},
            "calm": {"speaker_id": 0, "prompt": "In a calm and soothing tone: "},
            "excited": {"speaker_id": 1, "prompt": "In an enthusiastic and encouraging tone: "},
            "sad": {"speaker_id": 2, "prompt": "In a gentle, compassionate tone: "},
            "professional": {"speaker_id": 0, "prompt": "In a clear and professional tone: "}
        }
        
        self.current_style = "warm"
        
        logger.info(f"DiaTTS initialized (not yet loaded). Using device: {self.device}")
        
    async def initialize(self) -> bool:
        """
        Load the Dia 1.6B model and processor
        
        Returns:
            Whether initialization was successful
        """
        if self.initialized:
            return True
            
        try:
            logger.info(f"Loading Dia 1.6B model: {self.model_name}")
            
            # Create a background task for model loading since it can be time-consuming
            loop = asyncio.get_event_loop()
            
            # Load the processor and model in a separate thread to avoid blocking
            result = await loop.run_in_executor(None, self._load_model)
            
            if result["success"]:
                self.processor = result["processor"]
                self.model = result["model"]
                self.initialized = True
                logger.info(f"Dia 1.6B model loaded successfully on {self.device}")
                return True
            else:
                self.initialization_error = result["error"]
                logger.error(f"Failed to load Dia 1.6B model: {self.initialization_error}")
                return False
                
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Error initializing Dia 1.6B model: {str(e)}", exc_info=True)
            return False
            
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the model and processor (runs in a separate thread)
        
        Returns:
            Dict with loading results
        """
        try:
            # Load processor
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_safetensors=True
            )
            
            # Configure dtype based on device
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load model
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device)
            
            # Set to inference mode
            model.eval()
            
            return {
                "success": True,
                "processor": processor,
                "model": model,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "processor": None,
                "model": None,
                "error": str(e)
            }
            
    async def generate_speech(self, text: str, style: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate speech from text
        
        Args:
            text: Text to convert to speech
            style: Voice style to use
            
        Returns:
            Dictionary containing audio data and metadata
        """
        if not self.initialized:
            if not await self.initialize():
                return {
                    "success": False,
                    "error": f"Model not initialized: {self.initialization_error}",
                    "audio_bytes": b""
                }
        
        if not text:
            return {
                "success": False,
                "error": "No text provided",
                "audio_bytes": b""
            }
            
        try:
            # Prepare text with style
            style_config = self.voice_styles.get(
                style or self.current_style, 
                self.voice_styles["default"]
            )
            
            # Apply style prompt if available
            prompt = style_config.get("prompt", "")
            input_text = f"{prompt}{text}" if prompt else text
            
            # Get speaker ID for this style
            speaker_id = style_config.get("speaker_id", 0)
            
            # Process text
            start_time = time.time()
            
            # Create a background task for generation since it can be time-consuming
            loop = asyncio.get_event_loop()
            
            # Process text in a separate thread to avoid blocking
            result = await loop.run_in_executor(
                None, 
                lambda: self._generate_audio(input_text, speaker_id)
            )
            
            end_time = time.time()
            
            if not result["success"]:
                return result
                
            logger.info(f"Speech generated in {end_time - start_time:.2f} seconds")
            
            return {
                "success": True,
                "audio_bytes": result["audio_bytes"],
                "sample_rate": result.get("sample_rate", 24000),
                "time_taken": end_time - start_time
            }
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "audio_bytes": b""
            }
            
    def _generate_audio(self, text: str, speaker_id: int = 0) -> Dict[str, Any]:
        """
        Generate audio from text (runs in a separate thread)
        
        Args:
            text: Text to convert to speech
            speaker_id: Speaker ID for voice characteristics
            
        Returns:
            Dict with generation results including audio bytes
        """
        try:
            with torch.no_grad():
                # Process text through the model
                inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    speaker_id=speaker_id
                ).to(self.device)
                
                # Generate audio
                output = self.model.generate(**inputs)
                
                # Convert to numpy array
                speech = output.cpu().numpy().squeeze()
                
                # Convert to WAV format
                audio_bytes = self._speech_to_wav(speech)
                
                return {
                    "success": True,
                    "audio_bytes": audio_bytes,
                    "sample_rate": 24000  # Dia 1.6B uses 24kHz
                }
                
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_bytes": b""
            }
            
    def _speech_to_wav(self, speech_array: np.ndarray, sample_rate: int = 24000) -> bytes:
        """
        Convert speech array to WAV format bytes
        
        Args:
            speech_array: Audio array
            sample_rate: Sample rate of the audio
            
        Returns:
            WAV audio bytes
        """
        try:
            # Create in-memory buffer
            wav_buffer = io.BytesIO()
            
            # Write WAV data to buffer
            sf.write(wav_buffer, speech_array, sample_rate, format='WAV')
            
            # Get bytes from buffer
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
            
            return wav_bytes
            
        except Exception as e:
            logger.error(f"Failed to convert speech to WAV: {str(e)}")
            return b""
            
    def set_style(self, style: str) -> bool:
        """
        Set the voice style to use
        
        Args:
            style: Voice style name
            
        Returns:
            Whether the style was successfully set
        """
        if style in self.voice_styles:
            self.current_style = style
            return True
        return False
        
    def get_available_styles(self) -> list:
        """
        Get list of available voice styles
        
        Returns:
            List of style names
        """
        return list(self.voice_styles.keys())