"""
Voice Cloning Integration for the Contextual-Chatbot.
Provides an interface to integrate celebrity voice cloning into the main system.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional, List

from .celebrity_voice_cloner import CelebrityVoiceCloner
from .voice_ai import VoiceManager
from .audio_player import AudioPlayer

# Configure logger
logger = logging.getLogger(__name__)

class VoiceCloneIntegration:
    """
    Voice Cloning Integration for Contextual-Chatbot.
    Provides an easy-to-use interface to integrate celebrity voice cloning
    into the main chat system.
    """
    
    def __init__(self, 
                 voice_manager: Optional[VoiceManager] = None,
                 cache_dir: Optional[str] = None,
                 sample_dir: Optional[str] = None):
        """
        Initialize the Voice Cloning Integration
        
        Args:
            voice_manager: Existing VoiceManager instance to use
            cache_dir: Directory to cache models
            sample_dir: Directory to cache voice samples
        """
        # Set up cache directory
        self.cache_dir = cache_dir
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up sample directory
        self.sample_dir = sample_dir
        if self.sample_dir is None:
            self.sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'samples', 'celebrities')
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Initialize components
        self.voice_manager = voice_manager
        self.celebrity_voice_cloner = CelebrityVoiceCloner(cache_dir=self.cache_dir, sample_dir=self.sample_dir)
        self.audio_player = AudioPlayer()
        
        # State management
        self.initialized = False
        self.initialization_error = None
        self.current_celebrity = None
        self.celebrity_mode_enabled = False
    
    async def initialize(self) -> bool:
        """
        Initialize the voice cloning integration
        
        Returns:
            Whether initialization was successful
        """
        try:
            # Initialize celebrity voice cloner
            success = await self.celebrity_voice_cloner.initialize()
            
            if not success:
                self.initialization_error = f"Failed to initialize celebrity voice cloner: {self.celebrity_voice_cloner.initialization_error}"
                logger.error(self.initialization_error)
                return False
            
            # Initialize voice manager if not provided
            if self.voice_manager is None:
                from .voice_ai import VoiceManager
                self.voice_manager = VoiceManager()
                voice_init_result = await self.voice_manager.initialize()
                
                if not voice_init_result["success"]:
                    logger.warning(f"Failed to initialize Voice Manager: {voice_init_result.get('error', 'Unknown error')}")
                    logger.info("Continuing with limited functionality")
            
            self.initialized = True
            logger.info("Voice Cloning Integration initialized successfully")
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Error initializing Voice Cloning Integration: {str(e)}")
            return False
    
    async def search_celebrity(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for a celebrity by name
        
        Args:
            query: Name of the celebrity to search for
            
        Returns:
            List of matching celebrities with metadata
        """
        if not self.initialized:
            if not await self.initialize():
                logger.error("Cannot search for celebrity: Not initialized")
                return []
        
        return await self.celebrity_voice_cloner.search_celebrity(query)
    
    async def set_celebrity_voice(self, celebrity_id: str) -> Dict[str, Any]:
        """
        Set the active celebrity voice for TTS
        
        Args:
            celebrity_id: ID of the celebrity to use
            
        Returns:
            Dictionary with result of operation
        """
        if not self.initialized:
            if not await self.initialize():
                return {
                    "success": False,
                    "error": f"Voice cloning integration not initialized: {self.initialization_error}"
                }
        
        try:
            # Get celebrity info from cache
            available = await self.celebrity_voice_cloner.list_available_celebrities()
            
            found = False
            for celeb in available:
                if celeb["id"] == celebrity_id:
                    self.current_celebrity = celeb
                    found = True
                    break
            
            if not found:
                # Try to find in search results
                search_results = await self.celebrity_voice_cloner.search_celebrity(celebrity_id)
                if search_results:
                    for celeb in search_results:
                        if celeb["id"] == celebrity_id:
                            self.current_celebrity = celeb
                            found = True
                            break
            
            if not found:
                return {
                    "success": False,
                    "error": f"Celebrity not found: {celebrity_id}"
                }
            
            # Verify voice sample
            sample_result = await self.celebrity_voice_cloner.fetch_voice_sample(
                celebrity_id, 
                self.current_celebrity.get("name", "Unknown")
            )
            
            if not sample_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to get voice sample: {sample_result.get('error', 'Unknown error')}"
                }
            
            # Enable celebrity mode
            self.celebrity_mode_enabled = True
            
            return {
                "success": True,
                "celebrity": self.current_celebrity
            }
            
        except Exception as e:
            logger.error(f"Error setting celebrity voice: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def speak_with_celebrity_voice(self, text: str) -> Dict[str, Any]:
        """
        Speak text using the current celebrity voice
        
        Args:
            text: Text to speak
            
        Returns:
            Dictionary with results
        """
        if not self.initialized or not self.celebrity_mode_enabled or self.current_celebrity is None:
            return {
                "success": False,
                "error": "Celebrity voice not set or integration not initialized"
            }
        
        try:
            # Clone voice and generate speech
            result = await self.celebrity_voice_cloner.clone_celebrity_voice(
                self.current_celebrity["id"],
                text
            )
            
            if not result["success"]:
                logger.error(f"Failed to generate speech with celebrity voice: {result.get('error', 'Unknown error')}")
                
                # Fall back to normal TTS if available
                if self.voice_manager:
                    logger.info("Falling back to normal TTS")
                    return await self.voice_manager.text_to_speech(text)
                else:
                    return result
            
            # Play audio if successful
            audio_bytes = result.get("audio_bytes", b"")
            if audio_bytes:
                self.audio_player.play_audio(audio_bytes)
            
            return result
            
        except Exception as e:
            logger.error(f"Error speaking with celebrity voice: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def disable_celebrity_mode(self) -> Dict[str, Any]:
        """
        Disable celebrity voice mode and return to normal TTS
        
        Returns:
            Dictionary with result of operation
        """
        self.celebrity_mode_enabled = False
        self.current_celebrity = None
        
        return {
            "success": True,
            "message": "Celebrity voice mode disabled"
        }
    
    async def is_celebrity_mode_active(self) -> Dict[str, Any]:
        """
        Check if celebrity voice mode is active
        
        Returns:
            Dictionary with status information
        """
        return {
            "active": self.celebrity_mode_enabled,
            "celebrity": self.current_celebrity
        }
    
    async def list_available_celebrities(self) -> List[Dict[str, Any]]:
        """
        List all celebrities with available voice samples
        
        Returns:
            List of celebrities with available samples
        """
        if not self.initialized:
            if not await self.initialize():
                logger.error("Cannot list celebrities: Not initialized")
                return []
                
        return await self.celebrity_voice_cloner.list_available_celebrities()
    
    async def speak_text(self, text: str) -> Dict[str, Any]:
        """
        Speak text with either celebrity voice or normal TTS based on current mode
        
        Args:
            text: Text to speak
            
        Returns:
            Dictionary with results
        """
        if not self.initialized:
            if not await self.initialize():
                return {
                    "success": False,
                    "error": f"Voice cloning integration not initialized: {self.initialization_error}"
                }
        
        try:
            # If celebrity mode is enabled and a celebrity is selected, use celebrity voice
            if self.celebrity_mode_enabled and self.current_celebrity is not None:
                return await self.speak_with_celebrity_voice(text)
            
            # Otherwise use normal TTS
            if self.voice_manager:
                return await self.voice_manager.text_to_speech(text)
            else:
                return {
                    "success": False,
                    "error": "No TTS system available"
                }
                
        except Exception as e:
            logger.error(f"Error in speak_text: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }