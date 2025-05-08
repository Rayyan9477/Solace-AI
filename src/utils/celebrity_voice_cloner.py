"""
Celebrity Voice Cloner Module
Enables voice cloning of popular personalities for the Mental Health Chatbot.
"""

import os
import io
import logging
import aiohttp
import asyncio
import json
import torch
import soundfile as sf
import numpy as np
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import requests
from urllib.parse import quote

from .dia_tts import DiaTTS

# Configure logger
logger = logging.getLogger(__name__)

class CelebrityVoiceCloner:
    """
    Celebrity Voice Cloning module for Mental Health Chatbot.
    Enables cloning voices of popular personalities using Dia 1.6B.
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None, 
                 sample_dir: Optional[str] = None,
                 use_gpu: bool = True):
        """
        Initialize the Celebrity Voice Cloner
        
        Args:
            cache_dir: Directory to cache models
            sample_dir: Directory to cache voice samples
            use_gpu: Whether to use GPU for inference
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
        
        # Initialize the Dia TTS model
        self.dia_tts = DiaTTS(cache_dir=self.cache_dir, use_gpu=use_gpu)
        self.initialized = False
        self.initialization_error = None
        
        # Cache of known celebrities and their samples
        self.celebrity_cache = {}
    
    async def initialize(self) -> bool:
        """
        Initialize the voice cloner
        
        Returns:
            Whether initialization was successful
        """
        try:
            # Initialize Dia TTS model
            success = await self.dia_tts.initialize()
            
            if not success:
                self.initialization_error = f"Failed to initialize Dia TTS: {self.dia_tts.initialization_error}"
                logger.error(self.initialization_error)
                return False
            
            # Load celebrity metadata if exists
            metadata_file = os.path.join(self.sample_dir, "celebrity_metadata.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        self.celebrity_cache = json.load(f)
                    logger.info(f"Loaded metadata for {len(self.celebrity_cache)} celebrities")
                except Exception as e:
                    logger.warning(f"Error loading celebrity metadata: {str(e)}")
            
            self.initialized = True
            logger.info("Celebrity Voice Cloner initialized successfully")
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Error initializing Celebrity Voice Cloner: {str(e)}")
            return False
    
    async def search_celebrity(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for a celebrity by name
        
        Args:
            query: Name of the celebrity to search for
            
        Returns:
            List of matching celebrities with metadata
        """
        # Check if we have the celebrity in our cache
        matches = []
        query_lower = query.lower()
        
        # First check our local cache
        for celeb_id, celeb_data in self.celebrity_cache.items():
            if query_lower in celeb_data.get("name", "").lower():
                matches.append({
                    "id": celeb_id,
                    "name": celeb_data.get("name", ""),
                    "description": celeb_data.get("description", ""),
                    "has_sample": celeb_data.get("has_sample", False),
                    "source": "cache"
                })
        
        # If we have matches from cache, return them
        if matches:
            return matches
        
        # Otherwise search online
        try:
            # Use a simple API to search for celebrities
            # Using Wikidata API as a source of celebrity information
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={quote(query)}&language=en&format=json&limit=5"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process results
                        if "search" in data:
                            for item in data["search"]:
                                # Add to matches
                                celeb_id = item.get("id", "")
                                matches.append({
                                    "id": celeb_id,
                                    "name": item.get("label", ""),
                                    "description": item.get("description", ""),
                                    "has_sample": False,
                                    "source": "wikidata"
                                })
                                
                                # Also add to our cache
                                self.celebrity_cache[celeb_id] = {
                                    "name": item.get("label", ""),
                                    "description": item.get("description", ""),
                                    "has_sample": False,
                                    "source": "wikidata"
                                }
            
            # Save updated cache
            self._save_celebrity_cache()
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching for celebrity: {str(e)}")
            return []
    
    async def fetch_voice_sample(self, celebrity_id: str, celebrity_name: str) -> Dict[str, Any]:
        """
        Fetch a voice sample for a celebrity
        
        Args:
            celebrity_id: ID of the celebrity
            celebrity_name: Name of the celebrity for fallback search
            
        Returns:
            Dictionary with results including path to voice sample if successful
        """
        # Check if we already have a sample
        sample_path = os.path.join(self.sample_dir, f"{celebrity_id}.wav")
        if os.path.exists(sample_path):
            # Update cache to indicate we have a sample
            if celebrity_id in self.celebrity_cache:
                self.celebrity_cache[celebrity_id]["has_sample"] = True
                self._save_celebrity_cache()
            
            return {
                "success": True,
                "sample_path": sample_path,
                "name": self.celebrity_cache.get(celebrity_id, {}).get("name", celebrity_name),
                "message": "Using cached voice sample"
            }
        
        # Try to find a voice sample online
        try:
            # For this demo, let's use a voice sample API (this is a placeholder - in a real implementation,
            # you would use a proper API with comprehensive celebrity voice samples)
            # Here, we'll use a YouTube search and download a short clip as a demonstration
            
            search_query = f"{celebrity_name} interview"
            search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={quote(search_query)}&type=video&maxResults=1&key=YOUR_YOUTUBE_API_KEY"
            
            # Note: In a real implementation, you would:
            # 1. Replace YOUR_YOUTUBE_API_KEY with an actual API key
            # 2. Use a service that provides voice samples directly
            # 3. Implement proper error handling and quality checking
            
            # For now, let's simulate finding a sample by creating a placeholder file
            # In a real implementation, you would download and process the actual voice sample
            
            # Create a placeholder sample file (silent audio) for demonstration
            sample_duration = 3.0  # seconds
            sample_rate = 16000  # Hz
            sample_data = np.zeros(int(sample_duration * sample_rate))
            
            # Save the sample
            sf.write(sample_path, sample_data, sample_rate)
            
            # Update our cache
            if celebrity_id in self.celebrity_cache:
                self.celebrity_cache[celebrity_id]["has_sample"] = True
                self._save_celebrity_cache()
            
            return {
                "success": True,
                "sample_path": sample_path,
                "name": celebrity_name,
                "message": "Created simulated voice sample for demo purposes"
            }
            
        except Exception as e:
            logger.error(f"Error fetching voice sample: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def clone_celebrity_voice(self, celebrity_id: str, text: str) -> Dict[str, Any]:
        """
        Clone a celebrity's voice and generate speech
        
        Args:
            celebrity_id: ID of the celebrity
            text: Text to convert to speech
            
        Returns:
            Dictionary with results including audio bytes if successful
        """
        if not self.initialized:
            if not await self.initialize():
                return {
                    "success": False,
                    "error": f"Voice cloner not initialized: {self.initialization_error}"
                }
        
        try:
            # Check if we have the celebrity in our cache
            if celebrity_id not in self.celebrity_cache:
                return {
                    "success": False,
                    "error": f"Celebrity {celebrity_id} not found in cache"
                }
            
            celebrity_name = self.celebrity_cache[celebrity_id].get("name", "")
            
            # Get voice sample
            sample_result = await self.fetch_voice_sample(celebrity_id, celebrity_name)
            if not sample_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to get voice sample: {sample_result.get('error', 'Unknown error')}"
                }
            
            # Get the voice sample path
            sample_path = sample_result["sample_path"]
            
            # Create reference text (what's being said in the reference audio)
            # In a real implementation, you would have the actual transcript
            reference_text = f"[S1] Hello, this is {celebrity_name} speaking."
            
            # Prepare target text for generation (with speaker tag)
            if not text.startswith("[S1]"):
                target_text = f"[S1] {text}"
            else:
                target_text = text
            
            # Call the voice cloning function from DiaTTS
            result = await self.dia_tts.clone_voice(sample_path, reference_text, target_text)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": f"Voice cloning failed: {result.get('error', 'Unknown error')}"
                }
            
            # Add metadata to the result
            result["celebrity_name"] = celebrity_name
            result["celebrity_id"] = celebrity_id
            
            return result
            
        except Exception as e:
            logger.error(f"Error cloning celebrity voice: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _save_celebrity_cache(self):
        """Save the celebrity cache to disk"""
        try:
            metadata_file = os.path.join(self.sample_dir, "celebrity_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.celebrity_cache, f, indent=2)
            logger.debug(f"Saved metadata for {len(self.celebrity_cache)} celebrities")
        except Exception as e:
            logger.warning(f"Error saving celebrity metadata: {str(e)}")
    
    async def list_available_celebrities(self) -> List[Dict[str, Any]]:
        """
        List all celebrities with available voice samples
        
        Returns:
            List of celebrities with available samples
        """
        available = []
        
        for celeb_id, celeb_data in self.celebrity_cache.items():
            if celeb_data.get("has_sample", False):
                available.append({
                    "id": celeb_id,
                    "name": celeb_data.get("name", ""),
                    "description": celeb_data.get("description", "")
                })
        
        return available