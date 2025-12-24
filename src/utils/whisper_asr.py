"""
Standalone Whisper ASR integration for the Mental Health Chatbot.
Provides direct access to Whisper's speech-to-text capabilities.
"""

import os
import torch
import tempfile
import logging
import numpy as np
import time
import sounddevice as sd
import whisper
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import huggingface_hub
import json
import asyncio
from src.config.settings import AppConfig
from .device_utils import get_device, is_cuda_available
from .voice_emotion_analyzer import VoiceEmotionAnalyzer

logger = logging.getLogger(__name__)

# Map model names to Whisper model sizes
MODEL_MAPPINGS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small", 
    "medium": "medium",
    "large": "large",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "turbo": "large-v3", # Default to large-v3 for "turbo"
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v3-turbo": "large-v3" # HF model path maps to v3
}

class WhisperASR:
    """Direct integration with OpenAI's Whisper for ASR"""
    
    def __init__(self, model_name: str = "turbo", analyze_emotions: bool = False, use_hume_ai: bool = False, use_audeering: bool = True):
        """
        Initialize Whisper ASR
        
        Args:
            model_name: Name of the Whisper model to use ("tiny", "base", "small", "medium", "large", "turbo")
                       or HF model path like "openai/whisper-large-v3-turbo"
            analyze_emotions: Whether to analyze emotions in speech
            use_hume_ai: Whether to use Hume AI for emotion analysis (requires API key)
            use_audeering: Whether to use audeering model for emotion analysis (default: True)
        """
        self.model_name = model_name
        # Map to standard whisper model name if HF path provided
        self.whisper_model_name = MODEL_MAPPINGS.get(model_name.lower(), "large-v3")
        self.model = None
        self.device = get_device()
        
        # Define model download path
        if hasattr(AppConfig, 'VOICE_CONFIG') and 'cache_dir' in AppConfig.VOICE_CONFIG:
            self.cache_dir = AppConfig.VOICE_CONFIG['cache_dir']
        else:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cache')
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Configure huggingface_hub to use the cache directory
        huggingface_hub.config.HUGGINGFACE_HUB_CACHE = self.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir

        # Emotion analysis configuration
        self.analyze_emotions = analyze_emotions
        self.use_hume_ai = use_hume_ai
        self.use_audeering = use_audeering
        self.emotion_analyzer = None
        if analyze_emotions:
            # Get Hume AI API key from environment or AppConfig
            hume_api_key = os.environ.get("HUME_API_KEY")
            if hasattr(AppConfig, 'HUME_API_KEY'):
                hume_api_key = AppConfig.HUME_API_KEY

            # SEC-012: Validate API key before use
            if use_hume_ai:
                if not hume_api_key:
                    logger.warning("HUME_API_KEY not set - Hume AI emotion analysis disabled")
                    use_hume_ai = False
                elif len(hume_api_key.strip()) < 10:
                    logger.warning("HUME_API_KEY appears malformed (too short) - Hume AI disabled")
                    use_hume_ai = False
                    hume_api_key = None
                elif hume_api_key.strip() != hume_api_key:
                    logger.warning("HUME_API_KEY contains leading/trailing whitespace - trimming")
                    hume_api_key = hume_api_key.strip()

            # Initialize emotion analyzer with validated key
            self.emotion_analyzer = VoiceEmotionAnalyzer(
                use_hume_ai=use_hume_ai,
                hume_api_key=hume_api_key,
                device=self.device,
                use_audeering=use_audeering,
                cache_dir=self.cache_dir
            )
        
        # Configuration settings with defaults
        self.config = {
            "language": None,  # None for auto-detection
            "task": "transcribe",  # transcribe or translate
            "beam_size": 5,
            "patience": None,
            "temperature": 0.0,  # 0 = greedy decoding
            "initial_prompt": None,
            "fp16": is_cuda_available()
        }
        
        logger.info(f"Initializing WhisperASR with model '{model_name}' (using {self.whisper_model_name}) on {self.device}")
        
        # Define model download path
        if hasattr(AppConfig, 'VOICE_CONFIG') and 'cache_dir' in AppConfig.VOICE_CONFIG:
            self.cache_dir = AppConfig.VOICE_CONFIG['cache_dir']
        else:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cache')
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Configure huggingface_hub to use the cache directory
        huggingface_hub.config.HUGGINGFACE_HUB_CACHE = self.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        
        # Set audio parameters
        self.sample_rate = 16000
        self.recording_channels = 1
    
    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure the Whisper ASR parameters
        
        Args:
            config: Dictionary of configuration settings
                   Possible keys: language, translate, beam_size, patience, 
                                 temperature, initial_prompt
        
        Returns:
            Dictionary of current configuration
        """
        # Update configuration with provided values
        if config.get("language") is not None:
            self.config["language"] = config["language"]
            
        if config.get("translate") is not None:
            self.config["task"] = "translate" if config["translate"] else "transcribe"
            
        if config.get("beam_size") is not None:
            self.config["beam_size"] = config["beam_size"]
            
        if config.get("patience") is not None:
            self.config["patience"] = config["patience"]
            
        if config.get("temperature") is not None:
            self.config["temperature"] = config["temperature"]
            
        if config.get("initial_prompt") is not None:
            self.config["initial_prompt"] = config["initial_prompt"]
            
        # Log configuration changes
        logger.info(f"Updated Whisper configuration: {json.dumps(self.config, default=str)}")
        
        return self.config
    
    def load_model(self) -> bool:
        """Load the Whisper model if not already loaded"""
        if self.model is not None:
            return True
            
        try:
            start_time = time.time()
            logger.info(f"Loading Whisper model '{self.whisper_model_name}'...")
            
            # Check if we need to download from Hugging Face
            if self.model_name.startswith("openai/") or '/' in self.model_name:
                hf_model_path = self.model_name
                logger.info(f"Downloading model from Hugging Face: {hf_model_path}")
                
                # Create the cache path to make sure it exists
                model_cache_path = os.path.join(self.cache_dir, f"models--{hf_model_path.replace('/', '--')}")
                os.makedirs(model_cache_path, exist_ok=True)
                
                # Download the model info to cache
                try:
                    huggingface_hub.hf_hub_download(
                        repo_id=hf_model_path, 
                        filename="config.json",
                        cache_dir=self.cache_dir
                    )
                    logger.info(f"Successfully pre-cached model info from Hugging Face")
                except Exception as e:
                    logger.warning(f"Could not pre-cache model info: {e}")
            
            # Load the model
            self.model = whisper.load_model(
                self.whisper_model_name,
                device=self.device,
                download_root=self.cache_dir
            )
            
            end_time = time.time()
            logger.info(f"Whisper model loaded in {end_time - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            return False
    
    def record_audio(self, 
                    duration: int = 5, 
                    sample_rate: int = 16000,
                    silent_threshold: float = 0.01,
                    silence_timeout: float = 2.0) -> Tuple[bool, np.ndarray]:
        """
        Record audio from the microphone with automatic silence detection
        
        Args:
            duration: Maximum recording duration in seconds (default: 5 seconds)
            sample_rate: Audio sample rate (default: 16000 Hz)
            silent_threshold: Threshold for silence detection (default: 0.01)
            silence_timeout: Stop recording after this many seconds of silence (default: 2.0 seconds)
            
        Returns:
            Tuple of (success flag, audio data as numpy array)
        """
        try:
            print(f"🎤 Recording... (speak now, max {duration}s, silent timeout {silence_timeout}s)")
            
            # Initialize audio buffer and recording state
            audio_buffer = []
            silent_frames = 0
            frames_per_buffer = int(sample_rate * 0.1)  # 100ms chunks
            max_frames = int(duration * 10)  # Convert duration to number of frames
            silent_frames_threshold = int(silence_timeout * 10)  # Convert silence timeout to frames
            
            # Define callback for audio stream
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                audio_buffer.append(indata.copy())
                
                # Check for silence
                nonlocal silent_frames
                if np.abs(indata).mean() < silent_threshold:
                    silent_frames += 1
                else:
                    silent_frames = 0
            
            # Start recording
            with sd.InputStream(callback=audio_callback,
                               channels=1,
                               samplerate=sample_rate,
                               blocksize=frames_per_buffer):
                
                # Wait for recording to complete
                frames_recorded = 0
                recording_started = False
                
                while frames_recorded < max_frames:
                    # Sleep for 100ms (one frame duration)
                    time.sleep(0.1)
                    frames_recorded += 1
                    
                    # Check if recording has started (non-silent audio detected)
                    if not recording_started and len(audio_buffer) > 0 and np.abs(audio_buffer[-1]).mean() >= silent_threshold:
                        recording_started = True
                        print("🔊 Speech detected, recording...")
                    
                    # Check for silence timeout, but only if recording has started
                    if recording_started and silent_frames >= silent_frames_threshold:
                        print("🔇 Silence detected, stopping recording...")
                        break
                    
                    # Print progress if we're still recording
                    if frames_recorded % 10 == 0:  # Every second
                        seconds_elapsed = frames_recorded / 10
                        seconds_remaining = duration - seconds_elapsed
                        if recording_started and silent_frames > 0:
                            silence_elapsed = silent_frames / 10
                            print(f"⏱️ Recording: {seconds_elapsed:.1f}s elapsed, {silence_elapsed:.1f}s silence")
                        else:
                            print(f"⏱️ Recording: {seconds_elapsed:.1f}s elapsed, {seconds_remaining:.1f}s remaining")
            
            if len(audio_buffer) == 0:
                print("❌ No audio recorded")
                return False, np.array([])
            
            # Concatenate audio chunks
            audio_data = np.concatenate(audio_buffer, axis=0)
            audio_data = audio_data.flatten()
            
            print(f"✅ Recorded {len(audio_data) / sample_rate:.2f} seconds of audio")
            return True, audio_data
            
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            print(f"❌ Error recording audio: {str(e)}")
            return False, np.array([])
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio data using Whisper
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Check if we have a valid audio array
            if len(audio_data) == 0:
                return {"success": False, "text": "", "error": "Empty audio data"}
            
            # Load model if not already loaded
            if not self.load_model():
                return {"success": False, "text": "", "error": "Failed to load Whisper model"}
            
            start_time = time.time()
            
            # Process the audio
            logger.info("Transcribing audio with Whisper...")
            print("🔄 Transcribing audio...")
            
            # Get configuration options
            options = {
                "language": self.config["language"],
                "task": self.config["task"],
                "beam_size": self.config["beam_size"],
                "patience": self.config["patience"],
                "initial_prompt": self.config["initial_prompt"],
                "temperature": self.config["temperature"],
                "fp16": self.config["fp16"]
            }
            
            # Filter out None values
            options = {k: v for k, v in options.items() if v is not None}
            
            # Run Whisper transcription with options
            result = self.model.transcribe(
                audio_data,
                **options
            )
            
            end_time = time.time()
            transcription_time = end_time - start_time
            
            text = result.get("text", "").strip()
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds: '{text}'")
            
            if text:
                print(f"✅ Transcribed in {transcription_time:.2f}s: \"{text}\"")
                return {
                    "success": True,
                    "text": text,
                    "time_taken": transcription_time,
                    "language": result.get("language", "en"),
                    "segments": result.get("segments", [])
                }
            else:
                print("❓ No speech detected or unable to transcribe")
                return {
                    "success": False,
                    "text": "",
                    "error": "No speech detected or unable to transcribe",
                    "time_taken": transcription_time
                }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            print(f"❌ Error transcribing: {str(e)}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
    
    async def _analyze_audio_emotions(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze emotions in audio data
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary with emotion analysis results
        """
        if not self.analyze_emotions or self.emotion_analyzer is None:
            return {"success": False, "error": "Emotion analyzer not enabled"}
            
        try:
            # Save audio data to a temporary file for analysis
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write audio data to the file
                import soundfile as sf
                sf.write(temp_path, audio_data, self.sample_rate)
                
                # Analyze emotions
                emotion_results = await self.emotion_analyzer.analyze_emotions(temp_path)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
                return emotion_results
                
        except Exception as e:
            logger.error(f"Error analyzing emotions: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def transcribe_audio_with_emotion(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio data using Whisper and analyze emotions
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary with transcription and emotion analysis results
        """
        # Get transcription results
        transcription = self.transcribe_audio(audio_data)
        
        # If transcription failed or emotion analysis is disabled, return early
        if not transcription["success"] or not self.analyze_emotions:
            return transcription
            
        # Analyze emotions if transcription was successful
        emotion_results = await self._analyze_audio_emotions(audio_data)
        
        # Add emotion results to transcription results
        if emotion_results["success"]:
            transcription["emotion"] = emotion_results
            
            # Log the detected emotion
            primary_emotion = emotion_results.get("primary_emotion", "unknown")
            logger.info(f"Detected voice emotion: {primary_emotion}")
            print(f"🎭 Detected emotion: {primary_emotion}")
        else:
            logger.warning(f"Emotion analysis failed: {emotion_results.get('error', 'Unknown error')}")
            
        return transcription

    async def record_and_transcribe_with_emotion(self) -> Dict[str, Any]:
        """
        Record audio from microphone, transcribe it, and analyze emotions
        
        Returns:
            Dictionary with transcription and emotion analysis results
        """
        # Record audio
        success, audio_data = self.record_audio()
        if not success:
            return {"success": False, "text": "", "error": "Failed to record audio"}
        
        # Transcribe the audio with emotion analysis
        return await self.transcribe_audio_with_emotion(audio_data)
    
    async def process_audio_file_with_emotion(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Process an existing audio file, transcribe it, and analyze emotions
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary with transcription and emotion analysis results
        """
        try:
            # First do the standard transcription
            transcription = self.process_audio_file(audio_file_path)
            
            # If transcription failed or emotion analysis is disabled, return early
            if not transcription["success"] or not self.analyze_emotions:
                return transcription
                
            # For emotion analysis, we need to load the audio file
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_file_path)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                # Simple resampling - for production, use a proper audio library like librosa
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), int(len(audio_data) * self.sample_rate / sample_rate)),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Analyze emotions
            emotion_results = await self._analyze_audio_emotions(audio_data)
            
            # Add emotion results to transcription results
            if emotion_results["success"]:
                transcription["emotion"] = emotion_results
                
                # Log the detected emotion
                primary_emotion = emotion_results.get("primary_emotion", "unknown")
                logger.info(f"Detected voice emotion: {primary_emotion}")
                print(f"🎭 Detected emotion: {primary_emotion}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error processing audio file with emotion: {str(e)}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }

    def record_and_transcribe(self) -> Dict[str, Any]:
        """
        Record audio from microphone and transcribe it
        
        Returns:
            Dictionary with transcription results
        """
        # Record audio
        success, audio_data = self.record_audio()
        if not success:
            return {"success": False, "text": "", "error": "Failed to record audio"}
        
        # Transcribe the audio
        return self.transcribe_audio(audio_data)
    
    def process_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Process an existing audio file and transcribe it
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Load model if not already loaded
            if not self.load_model():
                return {"success": False, "text": "", "error": "Failed to load Whisper model"}
            
            # Check if file exists
            if not os.path.exists(audio_file_path):
                return {"success": False, "text": "", "error": f"Audio file not found: {audio_file_path}"}
            
            start_time = time.time()
            
            # Process the audio
            logger.info(f"Transcribing audio file: {audio_file_path}")
            print(f"🔄 Transcribing audio file: {audio_file_path}")
            
            # Get configuration options
            options = {
                "language": self.config["language"],
                "task": self.config["task"],
                "beam_size": self.config["beam_size"],
                "patience": self.config["patience"],
                "initial_prompt": self.config["initial_prompt"],
                "temperature": self.config["temperature"],
                "fp16": self.config["fp16"]
            }
            
            # Filter out None values
            options = {k: v for k, v in options.items() if v is not None}
            
            # Run Whisper transcription with options
            result = self.model.transcribe(
                audio_file_path,
                **options
            )
            
            end_time = time.time()
            transcription_time = end_time - start_time
            
            text = result.get("text", "").strip()
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds: '{text}'")
            
            if text:
                print(f"✅ Transcribed in {transcription_time:.2f}s: \"{text}\"")
                return {
                    "success": True,
                    "text": text,
                    "time_taken": transcription_time,
                    "language": result.get("language", "en"),
                    "segments": result.get("segments", [])
                }
            else:
                print("❓ No speech detected or unable to transcribe")
                return {
                    "success": False,
                    "text": "",
                    "error": "No speech detected or unable to transcribe",
                    "time_taken": transcription_time
                }
            
        except Exception as e:
            logger.error(f"Error transcribing audio file: {str(e)}")
            print(f"❌ Error transcribing: {str(e)}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }

# Simple test function for direct CLI usage
def main():
    """CLI entrypoint for testing Whisper ASR"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper ASR CLI")
    parser.add_argument(
        "--model", 
        default="turbo", 
        choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "turbo", 
                "openai/whisper-large-v3", "openai/whisper-large-v3-turbo"],
        help="Whisper model to use"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30,
        help="Maximum recording duration in seconds"
    )
    parser.add_argument(
        "--file", 
        type=str,
        help="Audio file to transcribe instead of recording"
    )
    parser.add_argument(
        "--language", 
        type=str,
        help="Language code (e.g., 'en', 'fr') or None for auto-detection"
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate non-English speech to English"
    )
    parser.add_argument(
        "--analyze-emotions",
        action="store_true",
        help="Analyze emotions in speech"
    )
    parser.add_argument(
        "--use-hume-ai",
        action="store_true",
        help="Use Hume AI for emotion analysis (requires API key)"
    )
    
    args = parser.parse_args()
    
    # Initialize Whisper ASR
    asr = WhisperASR(
        model_name=args.model,
        analyze_emotions=args.analyze_emotions,
        use_hume_ai=args.use_hume_ai
    )
    
    # Configure ASR if needed
    config = {
        "language": args.language,
        "translate": args.translate
    }
    asr.configure(config)
    
    # Use async function if emotion analysis is enabled
    if args.analyze_emotions:
        import asyncio
        
        async def run_with_emotion():
            if args.file:
                # Transcribe file with emotion analysis
                result = await asr.process_audio_file_with_emotion(args.file)
            else:
                # Record and transcribe with emotion analysis
                print(f"Using Whisper {args.model} model with emotion analysis")
                result = await asr.record_and_transcribe_with_emotion()
                
            return result
            
        result = asyncio.run(run_with_emotion())
    else:
        if args.file:
            # Transcribe file
            result = asr.process_audio_file(args.file)
        else:
            # Record and transcribe
            print(f"Using Whisper {args.model} model")
            result = asr.record_and_transcribe()
    
    if result["success"]:
        print("\nTranscription Result:")
        print(f"Text: {result['text']}")
        print(f"Language: {result.get('language', 'en')}")
        print(f"Time taken: {result.get('time_taken', 0):.2f} seconds")
        
        # Display emotion results if available
        if "emotion" in result and result["emotion"]["success"]:
            print("\nEmotion Analysis:")
            print(f"Primary emotion: {result['emotion'].get('primary_emotion', 'unknown')}")
            print(f"Confidence: {result['emotion'].get('confidence', 0):.2f}")
            
            # Display all emotions if available
            if "emotions" in result["emotion"]:
                print("\nEmotion Scores:")
                for emotion, score in result["emotion"]["emotions"].items():
                    print(f"- {emotion}: {score:.4f}")
    else:
        print(f"\nTranscription failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()