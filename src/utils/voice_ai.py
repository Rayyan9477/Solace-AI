"""
Voice AI component for the mental health chatbot.
Handles speech-to-text and text-to-speech functionality.
"""

import os
import torch
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from transformers import (AutoProcessor, SpeechT5Processor, SpeechT5ForTextToSpeech, 
                         SpeechT5HifiGan, AutoModelForSpeechSeq2Seq, 
                         AutoFeatureExtractor, AutoTokenizer)
import numpy as np
import soundfile as sf
import io
import torchaudio
from .audio_player import AudioPlayer
from .dia_tts import DiaTTS
from .device_utils import get_device, is_cuda_available
from ..config.settings import AppConfig  # Import AppConfig from settings

# Configure logger
logger = logging.getLogger(__name__)

class VoiceAI:
    """Voice AI component handling speech recognition and synthesis"""
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """
        Check if all required dependencies for voice AI are available
        
        Returns:
            Dictionary mapping dependency names to availability status
        """
        dependencies = {}
        
        try:
            import torch
            dependencies["torch"] = True
        except ImportError:
            dependencies["torch"] = False
            
        try:
            import transformers
            dependencies["transformers"] = True
        except ImportError:
            dependencies["transformers"] = False
            
        try:
            import soundfile
            dependencies["soundfile"] = True
        except ImportError:
            dependencies["soundfile"] = False
            
        try:
            import torchaudio
            dependencies["torchaudio"] = True
        except ImportError:
            dependencies["torchaudio"] = False
            
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            dependencies["numpy"] = False
            
        return dependencies
    
    def __init__(self, 
                 use_whisper: bool = True,
                 use_speecht5: bool = True,
                 use_dia: bool = True,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the Voice AI component
        
        Args:
            use_whisper: Whether to use OpenAI's Whisper for speech recognition
            use_speecht5: Whether to use Microsoft's SpeechT5 for text-to-speech
            use_dia: Whether to use Dia 1.6B for enhanced text-to-speech
            cache_dir: Directory to cache models
            device: Device to use ('cuda' or 'cpu')
        """
        # Set up device
        if device is None:
            self.device = get_device()
        else:
            self.device = device
            
        # Set cache directory
        self.cache_dir = cache_dir
        
        # Create audio player
        self.audio_player = AudioPlayer()
        
        # Set up speech recognition (Whisper)
        self.use_whisper = use_whisper
        self.whisper_model = None
        self.whisper_processor = None
        
        # Set up text-to-speech (SpeechT5)
        self.use_speecht5 = use_speecht5
        self.speecht5_processor = None
        self.speecht5_model = None
        self.speecht5_vocoder = None
        self.speaker_embeddings = None
        
        # Set up Dia 1.6B for enhanced TTS
        self.use_dia = use_dia
        self.dia_tts = None if not use_dia else DiaTTS(cache_dir=cache_dir, use_gpu=(self.device == "cuda"))
        
        # Flag to track initialization status
        self.initialized_stt = False
        self.initialized_tts = False
        self.initialized_dia = False
        
        # Set preferred TTS system
        self.preferred_tts = "dia" if use_dia else "speecht5"
        
        logger.info(f"VoiceAI initialized. Device: {self.device}, Preferred TTS: {self.preferred_tts}")
    
    async def initialize_stt(self) -> bool:
        """
        Initialize speech-to-text components
        
        Returns:
            Whether initialization was successful
        """
        if self.initialized_stt:
            return True
            
        if not self.use_whisper:
            return False
            
        try:
            logger.info("Initializing Whisper speech recognition")
            
            # Create a background task for model loading
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._load_whisper_model)
            
            if result["success"]:
                self.whisper_processor = result["processor"]
                self.whisper_model = result["model"]
                self.initialized_stt = True
                logger.info("Whisper speech recognition initialized successfully")
                return True
            else:
                logger.error(f"Failed to initialize Whisper: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {str(e)}", exc_info=True)
            return False
    
    async def initialize_tts(self) -> bool:
        """
        Initialize text-to-speech components
        
        Returns:
            Whether initialization was successful
        """
        if self.initialized_tts:
            return True
            
        if not self.use_speecht5:
            return False
            
        try:
            logger.info("Initializing SpeechT5 text-to-speech")
            
            # Create a background task for model loading
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._load_speecht5_model)
            
            if result["success"]:
                self.speecht5_processor = result["processor"]
                self.speecht5_model = result["model"]
                self.speecht5_vocoder = result["vocoder"]
                self.speaker_embeddings = result["speaker_embeddings"]
                self.initialized_tts = True
                logger.info("SpeechT5 text-to-speech initialized successfully")
                return True
            else:
                logger.error(f"Failed to initialize SpeechT5: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing text-to-speech: {str(e)}", exc_info=True)
            return False
            
    async def initialize_dia(self) -> bool:
        """
        Initialize Dia 1.6B text-to-speech
        
        Returns:
            Whether initialization was successful
        """
        if self.initialized_dia or not self.use_dia or self.dia_tts is None:
            return self.initialized_dia
            
        try:
            logger.info("Initializing Dia 1.6B text-to-speech")
            success = await self.dia_tts.initialize()
            self.initialized_dia = success
            
            if success:
                logger.info("Dia 1.6B text-to-speech initialized successfully")
            else:
                logger.error(f"Failed to initialize Dia 1.6B: {self.dia_tts.initialization_error}")
                
            return success
                
        except Exception as e:
            logger.error(f"Error initializing Dia 1.6B: {str(e)}", exc_info=True)
            return False
    
    def _load_whisper_model(self) -> Dict[str, Any]:
        """
        Load Whisper model for speech recognition
        
        Returns:
            Dictionary with model loading results
        """
        try:
            # Load processor
            processor = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                cache_dir=self.cache_dir
            )
            
            # Load model
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=self.cache_dir
            ).to(self.device)
            
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
    
    def _load_speecht5_model(self) -> Dict[str, Any]:
        """
        Load SpeechT5 model for text-to-speech
        
        Returns:
            Dictionary with model loading results
        """
        try:
            # Load processor, model and vocoder
            processor = SpeechT5Processor.from_pretrained(
                "microsoft/speecht5_tts", 
                cache_dir=self.cache_dir
            )
            
            model = SpeechT5ForTextToSpeech.from_pretrained(
                "microsoft/speecht5_tts",
                cache_dir=self.cache_dir
            ).to(self.device)
            
            vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan",
                cache_dir=self.cache_dir
            ).to(self.device)
            
            # Load speaker embeddings
            if os.path.exists("speaker_embeddings.pt"):
                speaker_embeddings = torch.load("speaker_embeddings.pt")
            else:
                # Default embeddings
                speaker_embeddings = torch.randn((1, 512)).to(self.device)
                torch.save(speaker_embeddings, "speaker_embeddings.pt")
            
            return {
                "success": True,
                "processor": processor,
                "model": model,
                "vocoder": vocoder,
                "speaker_embeddings": speaker_embeddings,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "processor": None,
                "model": None,
                "vocoder": None,
                "speaker_embeddings": None,
                "error": str(e)
            }
    
    async def transcribe_audio(self, audio_file: str) -> Dict[str, Any]:
        """
        Transcribe speech from audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        if not self.initialized_stt:
            if not await self.initialize_stt():
                return {
                    "success": False,
                    "text": "",
                    "error": "Speech recognition not initialized"
                }
        
        try:
            # Load audio
            if not os.path.exists(audio_file):
                return {
                    "success": False,
                    "text": "",
                    "error": f"Audio file not found: {audio_file}"
                }
            
            speech_array, sampling_rate = torchaudio.load(audio_file)
            
            # Resample if needed
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                speech_array = resampler(speech_array)
                sampling_rate = 16000
            
            # Convert to mono if needed
            if speech_array.shape[0] > 1:
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)
            
            # Create a background task for transcription
            loop = asyncio.get_event_loop()
            
            # Process audio in a separate thread to avoid blocking
            result = await loop.run_in_executor(
                None, 
                lambda: self._transcribe_audio_whisper(speech_array.squeeze().numpy())
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
    
    def _transcribe_audio_whisper(self, speech_array: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe speech using Whisper model
        
        Args:
            speech_array: Audio data as numpy array
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Process audio through Whisper
            input_features = self.whisper_processor(
                speech_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate token ids
            predicted_ids = self.whisper_model.generate(
                input_features, 
                max_length=256
            )
            
            # Decode token ids to text
            transcription = self.whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return {
                "success": True,
                "text": transcription
            }
            
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {str(e)}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
            
    async def text_to_speech(self, text: str, voice_style: str = "default") -> Dict[str, Any]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice_style: Voice style to use
            
        Returns:
            Dictionary with TTS results including audio bytes
        """
        if not text:
            return {
                "success": False,
                "audio_bytes": b"",
                "error": "No text provided"
            }
            
        # Choose TTS system based on preference and availability
        if self.preferred_tts == "dia" and self.use_dia:
            # Try Dia 1.6B first
            if not self.initialized_dia:
                await self.initialize_dia()
                
            if self.initialized_dia:
                result = await self.dia_tts.generate_speech(text, style=voice_style)
                
                # If successful, return the result
                if result["success"]:
                    return result
                    
                # Otherwise, log error and fall back to SpeechT5
                logger.warning(f"Dia TTS failed: {result.get('error', 'Unknown error')}. Falling back to SpeechT5.")
        
        # Fall back to SpeechT5 or if it's the preferred system
        if not self.initialized_tts:
            if not await self.initialize_tts():
                return {
                    "success": False,
                    "audio_bytes": b"",
                    "error": "Text-to-speech not initialized"
                }
                
        # Use SpeechT5
        try:
            # Create a background task for synthesis
            loop = asyncio.get_event_loop()
            
            # Process text in a separate thread to avoid blocking
            result = await loop.run_in_executor(
                None, 
                lambda: self._synthesize_speech_speecht5(text, voice_style)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}", exc_info=True)
            return {
                "success": False,
                "audio_bytes": b"",
                "error": str(e)
            }
    
    def _synthesize_speech_speecht5(self, text: str, voice_style: str = "default") -> Dict[str, Any]:
        """
        Synthesize speech using SpeechT5
        
        Args:
            text: Text to convert to speech
            voice_style: Voice style to use
            
        Returns:
            Dictionary with synthesis results
        """
        try:
            # Prepare speaker embedding based on voice style
            speaker_embedding = self.speaker_embeddings
            
            # Process text through SpeechT5
            inputs = self.speecht5_processor(text=text, return_tensors="pt").to(self.device)
            
            # Generate speech
            speech = self.speecht5_model.generate_speech(
                inputs["input_ids"], 
                speaker_embedding,
                vocoder=self.speecht5_vocoder
            ).cpu().numpy()
            
            # Convert to WAV format
            sample_rate = 16000  # SpeechT5 uses 16kHz
            audio_bytes = self._convert_to_wav(speech, sample_rate)
            
            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "sample_rate": sample_rate
            }
            
        except Exception as e:
            logger.error(f"Error in SpeechT5 synthesis: {str(e)}")
            return {
                "success": False,
                "audio_bytes": b"",
                "error": str(e)
            }
    
    def _convert_to_wav(self, speech_array: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert speech array to WAV format bytes
        
        Args:
            speech_array: Audio data as numpy array
            sample_rate: Audio sample rate
            
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
    
    async def speak_text(self, text: str, voice_style: str = "default", blocking: bool = False) -> Dict[str, Any]:
        """
        Convert text to speech and play it
        
        Args:
            text: Text to speak
            voice_style: Voice style to use
            blocking: Whether to block until audio finishes playing
            
        Returns:
            Dictionary with TTS results
        """
        try:
            # Generate speech
            result = await self.text_to_speech(text, voice_style)
            
            if not result["success"]:
                return result
                
            # Play the audio
            audio_bytes = result["audio_bytes"]
            sample_rate = result.get("sample_rate", 24000)
            
            if blocking:
                playback_result = self.audio_player.play_audio(audio_bytes, sample_rate)
            else:
                playback_result = self.audio_player.play_audio_nonblocking(audio_bytes, sample_rate)
                
            if not playback_result["success"]:
                return {
                    "success": False,
                    "error": f"Playback failed: {playback_result.get('error', 'Unknown error')}"
                }
                
            return {
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error speaking text: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def set_preferred_tts(self, tts_system: str) -> bool:
        """
        Set preferred TTS system
        
        Args:
            tts_system: TTS system to use ("dia" or "speecht5")
            
        Returns:
            Whether the system was successfully set
        """
        if tts_system in ["dia", "speecht5"]:
            self.preferred_tts = tts_system
            logger.info(f"Preferred TTS system set to {tts_system}")
            return True
        return False
    
    def set_voice_style(self, style: str) -> bool:
        """
        Set voice style for Dia TTS
        
        Args:
            style: Voice style name
            
        Returns:
            Whether the style was successfully set
        """
        if self.dia_tts is not None:
            return self.dia_tts.set_style(style)
        return False
    
    def get_available_voice_styles(self) -> List[str]:
        """
        Get list of available voice styles
        
        Returns:
            List of style names
        """
        if self.dia_tts is not None:
            return self.dia_tts.get_available_styles()
        return ["default"]

class VoiceManager:
    """Manager class for handling voice feature initialization and fallback"""
    
    def __init__(self):
        self.voice_ai = None
        self.voice_enabled = False
        self.initialization_error = None
        self.dia_tts = None
        self.use_dia = True  # Default to using Dia 1.6B for TTS if available
        
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize voice features with proper error handling
        
        Returns:
            Dict containing initialization status and any error messages
        """
        try:
            # First check if dependencies are available
            dependency_check = VoiceAI.check_dependencies()
            missing_deps = [dep for dep, status in dependency_check.items() if not status]
            
            if missing_deps:
                self.initialization_error = f"Missing dependencies: {', '.join(missing_deps)}"
                return {
                    "success": False,
                    "error": self.initialization_error,
                    "voice_ai": None,
                    "missing_dependencies": missing_deps
                }
            
            # Initialize VoiceAI for speech-to-text capabilities
            self.voice_ai = VoiceAI()
            
            # Test STT initialization
            stt_success = await self.voice_ai.initialize_stt()
            if not stt_success:
                logger.warning("Failed to initialize speech-to-text model, only text-to-speech will be available")
            
            # Initialize Dia 1.6B TTS for enhanced speech synthesis
            try:
                cache_dir = AppConfig.VOICE_CONFIG.get("cache_dir", os.path.join(os.path.dirname(__file__), '..', 'models', 'cache'))
                self.dia_tts = DiaTTS(cache_dir=cache_dir)
                dia_init_success = await self.dia_tts.initialize()
                if not dia_init_success:
                    logger.warning("Failed to initialize Dia 1.6B TTS, falling back to standard TTS")
                    self.use_dia = False
                else:
                    logger.info("Dia 1.6B TTS initialized successfully")
                    self.use_dia = True
            except Exception as dia_error:
                logger.warning(f"Failed to initialize Dia 1.6B TTS: {str(dia_error)}")
                self.use_dia = False
            
            # If Dia failed and we have no backup, try standard TTS
            if not self.use_dia:
                tts_success = await self.voice_ai.initialize_tts()
                if not tts_success:
                    logger.warning("Failed to initialize backup text-to-speech model")
                    # We can still proceed if STT works, so don't raise an exception
            
            self.voice_enabled = True
            return {
                "success": True,
                "voice_ai": self.voice_ai,
                "dia_tts": self.dia_tts if self.use_dia else None,
                "error": None
            }
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Voice initialization error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": self.initialization_error,
                "voice_ai": None,
                "dia_tts": None
            }
    
    async def text_to_speech(self, text: str, style: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert text to speech using the best available TTS system
        
        Args:
            text: Text to convert to speech
            style: Voice style to use
            
        Returns:
            Dictionary with results including audio bytes
        """
        if not text:
            return {"success": False, "error": "No text provided", "audio_bytes": b""}
        
        # First try Dia 1.6B if available and enabled
        if self.use_dia and self.dia_tts and self.dia_tts.initialized:
            try:
                logger.info(f"Using Dia 1.6B TTS for text ({len(text)} chars)")
                result = await self.dia_tts.generate_speech(text, style)
                if result["success"]:
                    return result
                
                # If Dia fails, log the error and fall back to standard TTS
                logger.warning(f"Dia TTS failed: {result.get('error', 'Unknown error')}, falling back to standard TTS")
            except Exception as e:
                logger.warning(f"Error using Dia TTS: {str(e)}, falling back to standard TTS")
        
        # Fall back to standard TTS
        if self.voice_ai:
            try:
                logger.info(f"Using standard TTS for text ({len(text)} chars)")
                return await self.voice_ai.text_to_speech(text, style)
            except Exception as e:
                logger.error(f"Error in text-to-speech conversion: {str(e)}")
                return {
                    "success": False,
                    "error": f"TTS failed: {str(e)}",
                    "audio_bytes": b""
                }
        
        return {
            "success": False,
            "error": "No TTS system available",
            "audio_bytes": b""
        }
    
    async def play_audio(self, audio_bytes: bytes) -> bool:
        """
        Play audio bytes directly using system audio
        
        Args:
            audio_bytes: WAV audio data as bytes
            
        Returns:
            Whether playback was successful
        """
        if not audio_bytes:
            logger.warning("No audio data to play")
            return False
            
        try:
            import sounddevice as sd
            from scipy.io import wavfile
            import io
            
            # Read WAV data from bytes
            buffer = io.BytesIO(audio_bytes)
            sample_rate, data = wavfile.read(buffer)
            
            # Ensure data is in float format
            if data.dtype != np.float32:
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
                
            # Play audio
            sd.play(data, sample_rate)
            sd.wait()  # Wait until audio finishes playing
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio: {str(e)}")
            return False
    
    async def speak_text(self, text: str, style: Optional[str] = None) -> bool:
        """
        Convert text to speech and play it
        
        Args:
            text: Text to speak
            style: Voice style to use
            
        Returns:
            Whether TTS and playback was successful
        """
        try:
            # Convert text to speech
            result = await self.text_to_speech(text, style)
            
            if not result["success"]:
                logger.error(f"Failed to convert text to speech: {result.get('error', 'Unknown error')}")
                return False
                
            # Play the audio
            audio_bytes = result.get("audio_bytes", b"")
            if not audio_bytes:
                logger.error("No audio data generated")
                return False
                
            return await self.play_audio(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error in speak_text: {str(e)}")
            return False
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """
        Get current initialization status
        
        Returns:
            Dict with status information
        """
        return {
            "enabled": self.voice_enabled,
            "error": self.initialization_error,
            "voice_ai": self.voice_ai,
            "dia_tts_available": self.use_dia and self.dia_tts and self.dia_tts.initialized
        }
    
    def toggle_dia_tts(self, use_dia: bool) -> bool:
        """
        Toggle between Dia 1.6B TTS and standard TTS
        
        Args:
            use_dia: Whether to use Dia 1.6B TTS
            
        Returns:
            Current status of Dia TTS usage
        """
        # Only allow enabling if Dia is actually available
        if use_dia and not (self.dia_tts and self.dia_tts.initialized):
            return False
            
        self.use_dia = use_dia
        return self.use_dia