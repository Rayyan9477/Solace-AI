"""
Voice Module

Provides speech synthesis and recognition capabilities to the application.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import os
import tempfile
from pathlib import Path

from src.components.base_module import Module
from src.config.settings import AppConfig

class VoiceModule(Module):
    """
    Voice Module for the Contextual-Chatbot.
    
    Provides speech synthesis and recognition capabilities:
    - Text-to-Speech (TTS)
    - Voice cloning and customization
    - Basic speech recognition
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the module"""
        super().__init__(module_id, config)
        self.voice_engine = None
        self.voice_component = None
        self.enabled = False
        self.output_dir = None
        
        # Initialize config
        self._load_config()
    
    def _load_config(self):
        """Load configuration values"""
        if not self.config:
            return
            
        self.enabled = self.config.get("enabled", False)
        self.output_dir = self.config.get("output_dir", os.path.join(
            Path(__file__).parents[2], "data", "voice_output"
        ))
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize the module"""
        await super().initialize()
        
        if not self.enabled:
            self.logger.info("Voice module disabled in configuration")
            self.health_status = "disabled"
            return True
        
        try:
            # Try to import the voice component
            try:
                from src.components.voice_component import VoiceComponent
                
                self.voice_component = VoiceComponent(
                    output_dir=self.output_dir,
                    config=self.config
                )
                
                self.logger.info("Initialized voice component")
                self._register_services()
                return True
                
            except ImportError:
                self.logger.warning("Voice component not available, trying fallback")
                
                # Try to import the new voice component
                try:
                    from src.components.voice_component_new import VoiceComponentNew
                    
                    self.voice_component = VoiceComponentNew(
                        output_dir=self.output_dir,
                        config=self.config
                    )
                    
                    self.logger.info("Initialized new voice component")
                    self._register_services()
                    return True
                    
                except ImportError:
                    self.logger.error("No voice components available")
                    self.health_status = "degraded"
                    return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize voice module: {str(e)}")
            self.health_status = "failed"
            return False
    
    def _register_services(self):
        """Register services provided by this module"""
        self.expose_service("text_to_speech", self.text_to_speech)
        self.expose_service("get_available_voices", self.get_available_voices)
        self.expose_service("set_voice", self.set_voice)
    
    async def text_to_speech(self, text: str, voice_id: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice identifier
            
        Returns:
            Path to audio file or None if failed
        """
        if not self.initialized or not self.enabled:
            self.logger.warning("Voice module not initialized or disabled")
            return None
        
        try:
            if self.voice_component and hasattr(self.voice_component, "text_to_speech"):
                audio_path = await self.voice_component.text_to_speech(text, voice_id)
                return audio_path
            else:
                self.logger.error("Voice component does not support TTS")
                return None
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {str(e)}")
            return None
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices
        
        Returns:
            List of voice information dictionaries
        """
        if not self.initialized or not self.enabled:
            self.logger.warning("Voice module not initialized or disabled")
            return []
        
        try:
            if self.voice_component and hasattr(self.voice_component, "get_available_voices"):
                voices = await self.voice_component.get_available_voices()
                return voices
            else:
                self.logger.error("Voice component does not support voice listing")
                return []
        except Exception as e:
            self.logger.error(f"Error getting available voices: {str(e)}")
            return []
    
    async def set_voice(self, voice_id: str) -> bool:
        """
        Set the active voice
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Success status
        """
        if not self.initialized or not self.enabled:
            self.logger.warning("Voice module not initialized or disabled")
            return False
        
        try:
            if self.voice_component and hasattr(self.voice_component, "set_voice"):
                success = await self.voice_component.set_voice(voice_id)
                return success
            else:
                self.logger.error("Voice component does not support voice setting")
                return False
        except Exception as e:
            self.logger.error(f"Error setting voice: {str(e)}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the module"""
        if self.voice_component and hasattr(self.voice_component, "shutdown"):
            await self.voice_component.shutdown()
        
        return await super().shutdown()
