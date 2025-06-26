"""
Voice emotion analyzer for the mental health chatbot.
Provides emotion analysis from audio using Hume AI and Hugging Face models.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union
import tempfile
import soundfile as sf
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
import requests
import json
import io
from .device_utils import get_device, is_cuda_available

# Configure logger
logger = logging.getLogger(__name__)

# Define a fallback ProsodyConfig class to use when Hume AI is not available
class FallbackProsodyConfig:
    """A fallback implementation of Hume AI's ProsodyConfig for when the library is not available"""
    def __init__(self):
        self.model = "prosody"
        self.identifiers = []

    def to_dict(self):
        return {
            "model": self.model,
            "identifiers": self.identifiers
        }

# Create a dummy HumeClient to use when Hume AI is not available
class DummyHumeClient:
    """A dummy implementation of Hume AI's HumeClient for when the library is not available"""
    def __init__(self, api_key):
        self.api_key = api_key
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def submit_job(self, files, configs):
        logger.warning("DummyHumeClient: Cannot submit job, Hume AI library not installed")
        raise ImportError("Hume AI library not installed")

class VoiceEmotionAnalyzer:
    """Voice emotion analyzer using multiple models for robust analysis"""
    
    def __init__(self, 
                 use_hume_ai: bool = False,
                 hume_api_key: Optional[str] = None,
                 huggingface_model: str = "MIT/ast-finetuned-speech-commands-v2",
                 use_audeering: bool = True,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the voice emotion analyzer
        
        Args:
            use_hume_ai: Whether to use Hume AI for emotion analysis
            hume_api_key: Hume AI API key if using Hume AI
            huggingface_model: HuggingFace model to use for emotion analysis
            use_audeering: Whether to use audeering wav2vec2 model for emotion analysis
            device: Device to use (cuda or cpu)
            cache_dir: Directory to cache models
        """
        # Set up device
        if device is None:
            self.device = get_device()
        else:
            self.device = device
            
        # Configure cache directory
        self.cache_dir = cache_dir
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up Hume AI configuration
        self.use_hume_ai = use_hume_ai
        self.hume_api_key = hume_api_key
        self.hume_client = None
        self.prosody_config = None
        
        # Try to initialize Hume AI client if API key is provided
        if self.use_hume_ai and self.hume_api_key:
            self._initialize_hume_client()
                
        # Set up HuggingFace model configuration
        self.huggingface_model = huggingface_model
        self.w2v_model = None
        self.w2v_processor = None
        
        # Set up audeering wav2vec2 model configuration
        self.use_audeering = use_audeering
        self.audeering_model = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.audeering_processor = None
        self.audeering_classifier = None
        
        # Initialize audeering model if enabled
        if self.use_audeering:
            try:
                self._load_audeering_model()
            except Exception as e:
                logger.error(f"Failed to load audeering model: {str(e)}")
                self.use_audeering = False
        
        # Initialize HuggingFace model if Hume AI is not available or as backup
        if not self.use_hume_ai or huggingface_model:
            try:
                self._load_huggingface_model()
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model: {str(e)}")
        
        # Create mapping of emotions for standardization
        self.emotion_mapping = {
            "angry": "anger",
            "anger": "anger",
            "disgust": "disgust",
            "disgusted": "disgust",
            "fear": "fear",
            "fearful": "fear",
            "happy": "happiness",
            "happiness": "happiness",
            "joy": "happiness",
            "sad": "sadness",
            "sadness": "sadness",
            "surprise": "surprise",
            "surprised": "surprise",
            "neutral": "neutral",
            "calm": "neutral",
            "excited": "excitement",
            "excitement": "excitement",
            # Add audeering model emotion mappings
            "amused": "amusement",
            "amusement": "amusement",
            "anxious": "anxiety",
            "anxiety": "anxiety",
            "bored": "boredom",
            "boredom": "boredom",
            "concentrated": "concentration",
            "concentration": "concentration",
            "contempt": "contempt"
        }
        
        logger.info(f"Voice emotion analyzer initialized. Using device: {self.device}")
        logger.info(f"Hume AI enabled: {self.use_hume_ai}, HuggingFace model: {self.huggingface_model}")
        logger.info(f"Audeering model enabled: {self.use_audeering}")
    
    def _initialize_hume_client(self):
        """Initialize the Hume AI client"""
        try:
            # Only try to import Hume AI here, not at the top level
            try:
                from hume import HumeClient
                from hume.models.config import ProsodyConfig
                
                self.hume_client = HumeClient(self.hume_api_key)
                self.prosody_config = ProsodyConfig()
                logger.info("Hume AI client initialized successfully")
            except ImportError:
                logger.warning("Hume AI library not found. Will use alternative models.")
                self.hume_client = DummyHumeClient(self.hume_api_key)
                self.prosody_config = FallbackProsodyConfig()
                self.use_hume_ai = False
        except Exception as e:
            logger.error(f"Failed to initialize Hume AI client: {str(e)}")
            self.use_hume_ai = False
            self.hume_client = None
            self.prosody_config = None

    def _load_huggingface_model(self):
        """Load the HuggingFace model for emotion recognition"""
        try:
            logger.info(f"Loading HuggingFace model: {self.huggingface_model}")
            
            # Load models that work specifically with speech emotion recognition
            self.w2v_processor = AutoProcessor.from_pretrained(
                self.huggingface_model, 
                cache_dir=self.cache_dir
            )
            
            self.w2v_model = AutoModelForAudioClassification.from_pretrained(
                self.huggingface_model,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            logger.info("HuggingFace model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            raise
    
    def _load_audeering_model(self):
        """Load the audeering wav2vec2 model for emotion recognition"""
        try:
            logger.info(f"Loading audeering model: {self.audeering_model}")
            
            # Load the feature extractor and model
            self.audeering_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.audeering_model,
                cache_dir=self.cache_dir
            )
            
            self.audeering_classifier = AutoModelForSequenceClassification.from_pretrained(
                self.audeering_model,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            logger.info("Audeering model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading audeering model: {str(e)}")
            raise
    
    async def analyze_emotions(self, audio_file: str) -> Dict[str, Any]:
        """
        Analyze emotions in an audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with emotion analysis results
        """
        try:
            results = {}
            
            # Try Hume AI first if enabled
            if self.use_hume_ai and self.hume_client:
                try:
                    hume_results = await self._analyze_with_hume(audio_file)
                    if hume_results["success"]:
                        logger.info("Successfully analyzed emotions with Hume AI")
                        return hume_results
                    else:
                        logger.warning(f"Hume AI analysis failed: {hume_results.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error in Hume AI analysis: {str(e)}")
            
            # Try audeering model if enabled
            if self.use_audeering and self.audeering_classifier and self.audeering_processor:
                try:
                    audeering_results = await self._analyze_with_audeering(audio_file)
                    if audeering_results["success"]:
                        logger.info("Successfully analyzed emotions with audeering model")
                        return audeering_results
                    else:
                        logger.warning(f"Audeering analysis failed: {audeering_results.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error in audeering analysis: {str(e)}")
            
            # Fall back to HuggingFace model
            if self.w2v_model and self.w2v_processor:
                try:
                    hf_results = await self._analyze_with_huggingface(audio_file)
                    if hf_results["success"]:
                        logger.info("Successfully analyzed emotions with HuggingFace model")
                        return hf_results
                    else:
                        logger.warning(f"HuggingFace analysis failed: {hf_results.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error in HuggingFace analysis: {str(e)}")
            
            # If all methods failed, return error
            return {
                "success": False,
                "error": "All emotion analysis methods failed",
                "primary_emotion": "unknown",
                "confidence": 0.0,
                "emotions": {}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "primary_emotion": "unknown",
                "confidence": 0.0,
                "emotions": {}
            }
    
    async def _analyze_with_audeering(self, audio_file: str) -> Dict[str, Any]:
        """
        Analyze emotions using audeering wav2vec2 model
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with audeering analysis results
        """
        if not self.use_audeering or not self.audeering_classifier or not self.audeering_processor:
            return {"success": False, "error": "Audeering model not enabled or initialized"}
            
        try:
            # Load audio file
            import librosa
            import asyncio
            
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            def process_audio():
                # Load audio with librosa (handles resampling automatically)
                try:
                    speech_array, sampling_rate = librosa.load(audio_file, sr=16000)
                except:
                    # Fallback if librosa fails
                    speech_array, sampling_rate = sf.read(audio_file)
                    if sampling_rate != 16000:
                        # Simple resampling if needed
                        speech_array = np.interp(
                            np.linspace(0, len(speech_array), int(len(speech_array) * 16000 / sampling_rate)),
                            np.arange(len(speech_array)),
                            speech_array
                        )
                    sampling_rate = 16000
                
                # Process audio with model
                inputs = self.audeering_processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.audeering_classifier(**inputs)
                    
                # Get emotion predictions
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                
                # Get emotion labels from config
                emotion_labels = self.audeering_classifier.config.id2label
                
                # Format results
                emotions = {}
                for i, label in emotion_labels.items():
                    # Map to standard emotion if possible
                    std_emotion = self.emotion_mapping.get(label.lower(), label.lower())
                    emotions[std_emotion] = float(probabilities[int(i)])
                
                # Find primary emotion
                primary_emotion = max(emotions.items(), key=lambda x: x[1])
                
                return {
                    "success": True,
                    "source": "audeering",
                    "primary_emotion": primary_emotion[0],
                    "confidence": primary_emotion[1],
                    "emotions": emotions
                }
            
            # Process in a separate thread
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, process_audio)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in audeering analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source": "audeering"
            }
            
    async def _analyze_with_hume(self, audio_file: str) -> Dict[str, Any]:
        """
        Analyze emotions using Hume AI
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with Hume AI analysis results
        """
        if not self.use_hume_ai or not self.hume_client:
            return {"success": False, "error": "Hume AI not enabled or client not initialized"}
            
        try:
            # Try to import required modules within the function scope
            try:
                from hume import HumeClient
                from hume.models.config import ProsodyConfig
            except ImportError:
                logger.error("Hume AI library not installed or properly configured")
                self.use_hume_ai = False
                return {
                    "success": False,
                    "error": "Hume AI library not installed",
                    "source": "hume_ai"
                }
                
            import asyncio
            
            # Create a job to analyze the audio file
            file_object = open(audio_file, "rb")
            
            # Submit the job to Hume AI
            logger.info("Submitting audio to Hume AI for analysis")
            async with self.hume_client:
                job = await self.hume_client.submit_job(
                    files=[file_object],
                    configs=[self.prosody_config]
                )
                
                # Wait for the job to complete
                result = await job.await_complete()
                prosody_result = result.get_prosody()
                
            # Process the results
            if prosody_result and "predictions" in prosody_result:
                predictions = prosody_result["predictions"]
                if predictions and len(predictions) > 0:
                    emotions = {}
                    
                    # Get the emotions from the first prediction
                    prediction = predictions[0]
                    emotions_data = prediction.get("emotions", [])
                    
                    # Extract emotions and scores
                    for emotion_data in emotions_data:
                        name = emotion_data.get("name", "").lower()
                        score = emotion_data.get("score", 0.0)
                        
                        # Map to standard emotion name if possible
                        std_emotion = self.emotion_mapping.get(name, name)
                        emotions[std_emotion] = score
                    
                    # Find primary emotion
                    if emotions:
                        primary_emotion = max(emotions.items(), key=lambda x: x[1])
                        
                        return {
                            "success": True,
                            "source": "hume_ai",
                            "primary_emotion": primary_emotion[0],
                            "confidence": primary_emotion[1],
                            "emotions": emotions
                        }
            
            return {
                "success": False,
                "error": "No emotions detected in Hume AI analysis",
                "source": "hume_ai"
            }
            
        except ImportError as e:
            logger.error(f"Hume AI library not properly installed: {str(e)}")
            self.use_hume_ai = False
            return {
                "success": False,
                "error": "Hume AI library not properly installed",
                "source": "hume_ai"
            }
        except Exception as e:
            logger.error(f"Error in Hume AI analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source": "hume_ai"
            }
    
    async def _analyze_with_huggingface(self, audio_file: str) -> Dict[str, Any]:
        """
        Analyze emotions using HuggingFace model
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with HuggingFace analysis results
        """
        if not self.w2v_model or not self.w2v_processor:
            return {"success": False, "error": "HuggingFace model not initialized"}
            
        try:
            # Load audio file
            import librosa
            import asyncio
            
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            def process_audio():
                # Load audio with librosa (handles resampling automatically)
                try:
                    speech_array, sampling_rate = librosa.load(audio_file, sr=16000)
                except:
                    # Fallback if librosa fails
                    speech_array, sampling_rate = sf.read(audio_file)
                    if sampling_rate != 16000:
                        # Simple resampling if needed
                        speech_array = np.interp(
                            np.linspace(0, len(speech_array), int(len(speech_array) * 16000 / sampling_rate)),
                            np.arange(len(speech_array)),
                            speech_array
                        )
                    sampling_rate = 16000
                
                # Process audio with model
                inputs = self.w2v_processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.w2v_model(**inputs)
                    
                # Get emotion predictions
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                
                # Get emotion labels
                emotion_labels = self.w2v_model.config.id2label
                
                # Format results
                emotions = {}
                for i, label in emotion_labels.items():
                    # Map to standard emotion if possible
                    std_emotion = self.emotion_mapping.get(label.lower(), label.lower())
                    emotions[std_emotion] = float(probabilities[i])
                
                # Find primary emotion
                primary_emotion = max(emotions.items(), key=lambda x: x[1])
                
                return {
                    "success": True,
                    "source": "huggingface",
                    "primary_emotion": primary_emotion[0],
                    "confidence": primary_emotion[1],
                    "emotions": emotions
                }
            
            # Process in a separate thread
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, process_audio)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in HuggingFace analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source": "huggingface"
            }

# Simple test function for direct CLI usage
def main():
    """CLI entrypoint for testing voice emotion analyzer"""
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Voice Emotion Analyzer CLI")
    parser.add_argument(
        "--file", 
        type=str,
        required=True,
        help="Audio file to analyze"
    )
    parser.add_argument(
        "--use-hume",
        action="store_true",
        help="Use Hume AI for analysis (requires API key)"
    )
    parser.add_argument(
        "--hume-key",
        type=str,
        help="Hume AI API key"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MIT/ast-finetuned-speech-commands-v2",
        help="HuggingFace model to use for analysis"
    )
    parser.add_argument(
        "--use-audeering",
        action="store_true",
        help="Use audeering wav2vec2 model for emotion analysis"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = VoiceEmotionAnalyzer(
        use_hume_ai=args.use_hume,
        hume_api_key=args.hume_key,
        huggingface_model=args.model,
        use_audeering=args.use_audeering
    )
    
    async def analyze():
        # Analyze emotions
        result = await analyzer.analyze_emotions(args.file)
        
        # Display results
        print("\nEmotion Analysis Results:")
        print(f"Success: {result['success']}")
        print(f"Source: {result.get('source', 'unknown')}")
        
        if result["success"]:
            print(f"Primary emotion: {result['primary_emotion']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            print("\nAll emotions:")
            sorted_emotions = sorted(result["emotions"].items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions:
                print(f"- {emotion}: {score:.4f}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Run the analysis
    asyncio.run(analyze())

if __name__ == "__main__":
    main()