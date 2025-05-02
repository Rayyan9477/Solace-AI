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
    
    def __init__(self, model_name: str = "turbo"):
        """
        Initialize voice input manager
        
        Args:
            model_name: Name of the Whisper model to use
        """
        self.model_name = model_name
        self.asr = WhisperASR(model_name=model_name)
        self.input_queue = queue.Queue()
        self.is_listening = False
        self.listen_thread = None
        self.callback = None
    
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
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create voice input manager
    manager = VoiceInputManager(model_name=args.model)
    
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
    else:
        # Single transcription
        print(f"Recording once with Whisper {args.model}...")
        result = manager.transcribe_once()
        
        if result["success"]:
            print(f"\n🗣️ {result['text']}")
        else:
            print(f"\n❌ {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()