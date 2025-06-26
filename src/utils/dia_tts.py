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
import importlib.util
import subprocess
import sys
from .device_utils import get_device, is_cuda_available

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
        self.use_official_package = False
        self.dia_package = None
        
        # Try to import official Dia package if available
        self._check_dia_package()
        
        # Create cache directory if specified
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set device
        self.use_gpu = use_gpu and is_cuda_available()
        self.device = get_device()
        
        # Configure voice styles
        self.voice_styles = {
            "default": {"speaker_id": 0, "prompt": ""},
            "warm": {"speaker_id": 0, "prompt": "In a warm and empathetic tone: "},
            "calm": {"speaker_id": 0, "prompt": "In a calm and soothing tone: "},
            "excited": {"speaker_id": 1, "prompt": "In an enthusiastic and encouraging tone: "},
            "sad": {"speaker_id": 2, "prompt": "In a gentle, compassionate tone: "},
            "professional": {"speaker_id": 0, "prompt": "In a clear and professional tone: "},
            "gentle": {"speaker_id": 2, "prompt": "In a gentle and supportive tone: "},
            "cheerful": {"speaker_id": 1, "prompt": "In a cheerful and positive tone: "},
            "empathetic": {"speaker_id": 0, "prompt": "In an empathetic and understanding tone: "}
        }
        
        self.current_style = "warm"
        
        logger.info(f"DiaTTS initialized (not yet loaded). Using device: {self.device}")
        
    def _check_dia_package(self):
        """Check if the official Dia package is available and import it"""
        try:
            # Check if dia module is installed
            if importlib.util.find_spec("dia") is not None:
                import dia
                from dia.model import Dia
                self.dia_package = dia
                self.use_official_package = True
                logger.info("Using official Dia package for enhanced TTS capabilities")
            else:
                # Check if the repository is cloned but not installed
                repo_path = os.path.join(os.path.expanduser("~"), "dia")
                if os.path.exists(repo_path) and os.path.isdir(repo_path):
                    sys.path.append(repo_path)
                    try:
                        import dia
                        from dia.model import Dia
                        self.dia_package = dia
                        self.use_official_package = True
                        logger.info("Using cloned Dia repository for enhanced TTS capabilities")
                    except ImportError:
                        logger.warning("Dia repository exists but module cannot be imported")
                        self.use_official_package = False
                else:
                    logger.info("Official Dia package not found, using Transformers implementation")
                    self.use_official_package = False
        except Exception as e:
            logger.warning(f"Error checking for Dia package: {str(e)}")
            self.use_official_package = False
            
    @staticmethod
    def install_dia_package():
        """Install the official Dia package"""
        try:
            logger.info("Attempting to install Dia package...")
            
            # Clone the repository
            subprocess.run(
                ["git", "clone", "https://github.com/nari-labs/dia.git"],
                check=True
            )
            
            # Change to the repository directory
            repo_path = os.path.join(os.getcwd(), "dia")
            
            # Install dependencies
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=repo_path,
                check=True
            )
            
            logger.info("Dia package installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install Dia package: {str(e)}")
            return False
        
    async def initialize(self) -> bool:
        """
        Load the Dia 1.6B model and processor
        
        Returns:
            Whether initialization was successful
        """
        if self.initialized:
            return True
            
        try:
            if self.use_official_package:
                logger.info("Initializing with official Dia package...")
                
                # Use a background task for model loading
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._load_official_model)
                
                if result["success"]:
                    self.model = result["model"]
                    self.initialized = True
                    logger.info("Official Dia 1.6B model loaded successfully")
                    return True
                else:
                    self.initialization_error = result["error"]
                    logger.error(f"Failed to load official Dia model: {self.initialization_error}")
                    logger.info("Falling back to Transformers implementation...")
                    self.use_official_package = False
            
            # Fall back to Transformers implementation
            logger.info(f"Loading Dia 1.6B model using Transformers: {self.model_name}")
            
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
    
    def _load_official_model(self) -> Dict[str, Any]:
        """
        Load the model using the official Dia package
        
        Returns:
            Dict with loading results
        """
        try:
            from dia.model import Dia
            
            # Load the model from Hugging Face
            model = Dia.from_pretrained(self.model_name)
            
            return {
                "success": True,
                "model": model,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "model": None,
                "error": str(e)
            }
            
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the model and processor using Transformers (runs in a separate thread)
        
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
            # Branch based on which implementation to use
            if self.use_official_package:
                return await self._generate_speech_official(text, style)
            else:
                return await self._generate_speech_transformers(text, style)
                
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "audio_bytes": b""
            }
    
    async def _generate_speech_official(self, text: str, style: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate speech using the official Dia package
        
        Args:
            text: Text to convert to speech
            style: Voice style to use
            
        Returns:
            Dictionary containing audio data and metadata
        """
        # Prepare text with style
        style_config = self.voice_styles.get(
            style or self.current_style, 
            self.voice_styles["default"]
        )
        
        # Apply style prompt if available
        prompt = style_config.get("prompt", "")
        input_text = f"{prompt}{text}" if prompt else text
        
        # Format text for dialogue if it doesn't have speaker tags
        if "[S1]" not in input_text and "[S2]" not in input_text:
            input_text = f"[S1] {input_text}"
        
        start_time = time.time()
        
        # Run generation in a separate thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._run_official_generation(input_text)
        )
        
        end_time = time.time()
        
        if not result["success"]:
            return result
            
        logger.info(f"Speech generated with official Dia package in {end_time - start_time:.2f} seconds")
        
        return {
            "success": True,
            "audio_bytes": result["audio_bytes"],
            "sample_rate": result.get("sample_rate", 44100),
            "time_taken": end_time - start_time
        }
    
    def _run_official_generation(self, text: str) -> Dict[str, Any]:
        """
        Run the official Dia generation (in a separate thread)
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Dictionary with results and audio data
        """
        try:
            # Generate audio data using the official model
            audio_array = self.model.generate(text)
            
            # Convert to WAV format
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_array, 44100, format='WAV')
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
            
            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "sample_rate": 44100
            }
        except Exception as e:
            logger.error(f"Error in official Dia generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_bytes": b""
            }
    
    async def _generate_speech_transformers(self, text: str, style: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate speech using the Transformers implementation
        
        Args:
            text: Text to convert to speech
            style: Voice style to use
            
        Returns:
            Dictionary containing audio data and metadata
        """
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
        
    async def clone_voice(self, reference_audio_path: str, reference_text: str, target_text: str) -> Dict[str, Any]:
        """
        Clone a voice from reference audio and generate speech with the cloned voice
        
        Args:
            reference_audio_path: Path to reference audio file
            reference_text: Text content of the reference audio
            target_text: Text to convert to speech using the cloned voice
            
        Returns:
            Dictionary containing audio data and metadata
        """
        if not self.use_official_package:
            return {
                "success": False,
                "error": "Voice cloning is only available with the official Dia package",
                "audio_bytes": b""
            }
            
        if not self.initialized:
            if not await self.initialize():
                return {
                    "success": False,
                    "error": f"Model not initialized: {self.initialization_error}",
                    "audio_bytes": b""
                }
                
        if not os.path.exists(reference_audio_path):
            return {
                "success": False,
                "error": f"Reference audio file not found: {reference_audio_path}",
                "audio_bytes": b""
            }
            
        try:
            logger.info(f"Attempting voice cloning from {reference_audio_path}")
            
            # Create input text with the reference transcript as a prompt
            if "[S1]" not in target_text and "[S2]" not in target_text:
                target_text = f"[S1] {target_text}"
                
            # Create the full prompt text with reference
            full_prompt = f"{reference_text}\n{target_text}"
            
            # Run generation in a separate thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._run_voice_cloning(reference_audio_path, full_prompt)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in voice cloning: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "audio_bytes": b""
            }
    
    def _run_voice_cloning(self, reference_audio_path: str, prompt_text: str) -> Dict[str, Any]:
        """
        Run voice cloning with the official Dia package (in a separate thread)
        
        Args:
            reference_audio_path: Path to reference audio file
            prompt_text: Text prompt with reference and target text
            
        Returns:
            Dictionary with results and audio data
        """
        try:
            # Load audio file
            import librosa
            audio_array, _ = librosa.load(reference_audio_path, sr=44100)
            
            # Generate audio with voice cloning
            output_audio = self.model.voice_clone(
                audio_array,
                prompt_text
            )
            
            # Convert to WAV format
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, output_audio, 44100, format='WAV')
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
            
            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "sample_rate": 44100
            }
        except Exception as e:
            logger.error(f"Error in voice cloning: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_bytes": b""
            }