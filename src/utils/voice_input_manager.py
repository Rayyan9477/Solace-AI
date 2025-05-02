"""
Integration module for bridging standalone Whisper ASR with VoiceAI class.
Provides additional voice input methods for the Mental Health Chatbot.
"""

import sys
import os
import threading
import queue
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Callable

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the standalone Whisper ASR
from utils.whisper_asr import WhisperASR

logger = logging.getLogger(__name__)

class VoiceInputManager:
    """
    Manager for voice input that works with both UI and command-line interfaces.
    Provides a bridge between WhisperASR and the chatbot.
    """
    
    def __init__(self, model_name: str = "turbo", analyze_emotions: bool = False, use_hume_ai: bool = False, use_audeering: bool = True):
        """
        Initialize voice input manager
        
        Args:
            model_name: Name of the Whisper model to use
            analyze_emotions: Whether to analyze emotions in speech
            use_hume_ai: Whether to use Hume AI for emotion analysis (requires API key)
            use_audeering: Whether to use audeering model for emotion analysis (default: True)
        """
        self.model_name = model_name
        self.analyze_emotions = analyze_emotions
        self.use_hume_ai = use_hume_ai
        self.use_audeering = use_audeering
        
        # Initialize ASR with emotion analysis if enabled
        self.asr = WhisperASR(
            model_name=model_name,
            analyze_emotions=analyze_emotions,
            use_hume_ai=use_hume_ai,
            use_audeering=use_audeering
        )
        
        self.input_queue = queue.Queue()
        self.is_listening = False
        self.listen_thread = None
        self.callback = None
        self.emotion_callback = None
        
        # Event loop for async operations
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    
    def start_listening(self, callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Start listening for voice input in a background thread
        
        Args:
            callback: Function to call with transcribed text
            
        Returns:
            Success flag
        """
        if self.is_listening:
            logger.warning("Already listening for voice input")
            return False
        
        self.callback = callback
        self.is_listening = True
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        logger.info("Started voice input listening thread")
        return True
    
    def stop_listening(self) -> bool:
        """
        Stop listening for voice input
        
        Returns:
            Success flag
        """
        if not self.is_listening:
            return False
        
        self.is_listening = False
        
        # Wait for thread to terminate
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=1.0)
        
        logger.info("Stopped voice input listening")
        return True
    
    def _listen_loop(self):
        """Background thread for listening to voice input"""
        logger.info("Voice input listening thread started")
        
        try:
            while self.is_listening:
                # Capture and transcribe audio
                result = self.asr.record_and_transcribe()
                
                if result["success"] and result["text"]:
                    # Add to queue
                    self.input_queue.put(result["text"])
                    
                    # Call callback if provided
                    if self.callback:
                        try:
                            self.callback(result["text"])
                        except Exception as e:
                            logger.error(f"Error in voice input callback: {str(e)}")
                
                # Slight delay before next recording
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error in voice input listening thread: {str(e)}")
            self.is_listening = False
    
    def get_next_input(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Get the next voice input from the queue
        
        Args:
            timeout: How long to wait for input (None = wait forever)
            
        Returns:
            Transcribed text or None if timeout
        """
        try:
            return self.input_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def transcribe_once(self) -> Dict[str, Any]:
        """
        Record and transcribe a single utterance
        
        Returns:
            Transcription result dictionary
        """
        return self.asr.record_and_transcribe()
    
    def transcribe_file(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcription result dictionary
        """
        return self.asr.process_audio_file(file_path)
    
    def configure_whisper(self, 
                          language: Optional[str] = None, 
                          translate: bool = False,
                          beam_size: int = 5,
                          patience: float = None,
                          temperature: float = 0.0,
                          initial_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Configure advanced Whisper ASR parameters
        
        Args:
            language: Language code (e.g., 'en', 'fr', 'es'), or None for auto-detection
            translate: Whether to translate non-English speech to English
            beam_size: Beam size for decoding (higher = potentially more accurate but slower)
            patience: Beam search patience factor (higher = potentially more accurate but slower)
            temperature: Temperature for sampling (0.0 = greedy decoding, higher = more random)
            initial_prompt: Optional text to provide as context for the transcription
            
        Returns:
            Dictionary with configuration status
        """
        try:
            # Pass the configuration to the Whisper ASR
            config_result = self.asr.configure({
                "language": language,
                "translate": translate,
                "beam_size": beam_size,
                "patience": patience,
                "temperature": temperature, 
                "initial_prompt": initial_prompt
            })
            
            logger.info(f"Whisper configuration updated: {config_result}")
            return {
                "success": True,
                "config": config_result
            }
        except Exception as e:
            logger.error(f"Error configuring Whisper ASR: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def start_listening_with_emotion(self, 
                          text_callback: Optional[Callable[[str], None]] = None,
                          emotion_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> bool:
        """
        Start listening for voice input with emotion analysis in a background thread
        
        Args:
            text_callback: Function to call with transcribed text
            emotion_callback: Function to call with emotion analysis results
            
        Returns:
            Success flag
        """
        if self.is_listening:
            logger.warning("Already listening for voice input")
            return False
        
        if not self.analyze_emotions:
            logger.warning("Emotion analysis is disabled. Use analyze_emotions=True in constructor.")
            
        self.callback = text_callback
        self.emotion_callback = emotion_callback
        self.is_listening = True
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_loop_with_emotion)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        logger.info("Started voice input listening thread with emotion analysis")
        return True
    
    def _listen_loop_with_emotion(self):
        """Background thread for listening to voice input with emotion analysis"""
        logger.info("Voice input listening thread with emotion analysis started")
        
        try:
            while self.is_listening:
                # Use the async method, but run it in the event loop
                async def record_and_analyze():
                    return await self.asr.record_and_transcribe_with_emotion()
                
                # Run the async function in the event loop
                result = self.loop.run_until_complete(record_and_analyze())
                
                if result["success"] and result["text"]:
                    # Add transcribed text to queue
                    self.input_queue.put(result["text"])
                    
                    # Call text callback if provided
                    if self.callback:
                        try:
                            self.callback(result["text"])
                        except Exception as e:
                            logger.error(f"Error in voice input text callback: {str(e)}")
                    
                    # Call emotion callback if provided and emotion analysis was successful
                    if self.emotion_callback and "emotion" in result and result["emotion"]["success"]:
                        try:
                            self.emotion_callback(result["emotion"])
                        except Exception as e:
                            logger.error(f"Error in voice input emotion callback: {str(e)}")
                
                # Slight delay before next recording
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error in voice input listening thread: {str(e)}")
            self.is_listening = False
    
    async def transcribe_once_with_emotion(self) -> Dict[str, Any]:
        """
        Record and transcribe a single utterance with emotion analysis
        
        Returns:
            Dictionary with transcription and emotion analysis results
        """
        if not self.analyze_emotions:
            logger.warning("Emotion analysis is disabled. Use analyze_emotions=True in constructor.")
            
        return await self.asr.record_and_transcribe_with_emotion()
    
    async def transcribe_file_with_emotion(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file with emotion analysis
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with transcription and emotion analysis results
        """
        if not self.analyze_emotions:
            logger.warning("Emotion analysis is disabled. Use analyze_emotions=True in constructor.")
            
        return await self.asr.process_audio_file_with_emotion(file_path)

# Simple CLI test
def main():
    """Test the voice input manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Input Manager CLI")
    parser.add_argument(
        "--model", 
        default="turbo", 
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model to use"
    )
    parser.add_argument(
        "--continuous", 
        action="store_true",
        help="Continue listening until stopped"
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
    parser.add_argument(
        "--use-audeering",
        action="store_true",
        help="Use audeering wav2vec2 model for emotion analysis (default: True)"
    )
    parser.add_argument(
        "--file", 
        type=str,
        help="Audio file to transcribe instead of recording"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create voice input manager
    manager = VoiceInputManager(
        model_name=args.model,
        analyze_emotions=args.analyze_emotions,
        use_hume_ai=args.use_hume_ai,
        use_audeering=args.use_audeering
    )
    
    # Use emotion analysis if requested
    if args.analyze_emotions:
        # Import asyncio for async operations
        import asyncio
        
        if args.continuous:
            # Define callbacks for text and emotion
            def on_transcription(text):
                print(f"\n🗣️ {text}")
                if text.lower() in ["exit", "quit", "stop"]:
                    print("Stopping...")
                    manager.stop_listening()
            
            def on_emotion(emotion_data):
                primary = emotion_data.get('primary_emotion', 'unknown')
                confidence = emotion_data.get('confidence', 0.0)
                print(f"🎭 Detected emotion: {primary} ({confidence:.2f})")
                
                # Print top 3 emotions if available
                if 'emotions' in emotion_data:
                    top_emotions = sorted(
                        emotion_data['emotions'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    
                    for emotion, score in top_emotions:
                        print(f"  - {emotion}: {score:.2f}")
            
            # Start listening with emotion analysis
            print(f"Listening continuously with Whisper {args.model} + emotion analysis.")
            print("Say 'exit', 'quit', or 'stop' to end.")
            manager.start_listening_with_emotion(
                text_callback=on_transcription,
                emotion_callback=on_emotion
            )
            
            try:
                # Keep main thread alive
                while manager.is_listening:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping...")
                manager.stop_listening()
        
        elif args.file:
            # Process file with emotion analysis
            async def process_file():
                print(f"Transcribing file with Whisper {args.model} + emotion analysis...")
                result = await manager.transcribe_file_with_emotion(args.file)
                return result
                
            # Run async function
            result = asyncio.run(process_file())
            display_result(result)
        
        else:
            # Single transcription with emotion analysis
            async def transcribe_once():
                print(f"Recording once with Whisper {args.model} + emotion analysis...")
                result = await manager.transcribe_once_with_emotion()
                return result
                
            # Run async function
            result = asyncio.run(transcribe_once())
            display_result(result)
    
    else:
        # Standard processing without emotion analysis
        if args.continuous:
            # Set up callback
            def on_transcription(text):
                print(f"\n🗣️ {text}")
                if text.lower() in ["exit", "quit", "stop"]:
                    print("Stopping...")
                    manager.stop_listening()
                    
            # Start listening
            print(f"Listening continuously with Whisper {args.model}. Say 'exit', 'quit', or 'stop' to end.")
            manager.start_listening(callback=on_transcription)
            
            try:
                # Keep main thread alive
                while manager.is_listening:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping...")
                manager.stop_listening()
        
        elif args.file:
            # Process file
            print(f"Transcribing file with Whisper {args.model}...")
            result = manager.transcribe_file(args.file)
            display_result(result)
        
        else:
            # Single transcription
            print(f"Recording once with Whisper {args.model}...")
            result = manager.transcribe_once()
            display_result(result)

def display_result(result):
    """Display transcription and emotion results"""
    if result["success"]:
        print(f"\n🗣️ {result['text']}")
        
        # Display emotion results if available
        if "emotion" in result and result["emotion"]["success"]:
            emotion_data = result["emotion"]
            primary = emotion_data.get('primary_emotion', 'unknown')
            confidence = emotion_data.get('confidence', 0.0)
            print(f"\n🎭 Detected emotion: {primary} ({confidence:.2f})")
            
            # Print all emotions if available
            if 'emotions' in emotion_data:
                print("\nAll emotions:")
                sorted_emotions = sorted(
                    emotion_data['emotions'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for emotion, score in sorted_emotions:
                    print(f"  - {emotion}: {score:.4f}")
    else:
        print(f"\n❌ {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()