"""
Demo script for the Celebrity Voice Cloner feature.
Allows testing of voice cloning functionality for popular personalities.
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.celebrity_voice_cloner import CelebrityVoiceCloner
from src.utils.audio_player import AudioPlayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("celebrity_voice_clone_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def search_celebrity(voice_cloner: CelebrityVoiceCloner, query: str):
    """Search for a celebrity and display results"""
    print(f"\nSearching for celebrity: {query}")
    results = await voice_cloner.search_celebrity(query)
    
    if not results:
        print("No celebrities found matching your query.")
        return None
    
    print(f"\nFound {len(results)} results:")
    for i, celeb in enumerate(results):
        print(f"{i+1}. {celeb.get('name', 'Unknown')} - {celeb.get('description', 'No description')}")
    
    while True:
        try:
            choice = input("\nSelect a celebrity (number) or press Enter to search again: ")
            if not choice:
                return None
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(results):
                return results[choice_idx]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

async def clone_voice(voice_cloner: CelebrityVoiceCloner, celebrity_id: str, text: str):
    """Clone a celebrity's voice and play the resulting audio"""
    print(f"\nCloning voice and generating speech...")
    result = await voice_cloner.clone_celebrity_voice(celebrity_id, text)
    
    if not result["success"]:
        print(f"❌ Voice cloning failed: {result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Voice cloning successful for {result.get('celebrity_name', 'Unknown')}")
    
    # Play the audio
    player = AudioPlayer()
    audio_bytes = result.get("audio_bytes", b"")
    sample_rate = result.get("sample_rate", 44100)
    
    if audio_bytes:
        print("Playing audio...")
        player.play_audio(audio_bytes)
        return True
    else:
        print("No audio data generated")
        return False

async def list_available_celebrities(voice_cloner: CelebrityVoiceCloner):
    """List all celebrities with available voice samples"""
    print("\nListing celebrities with available voice samples:")
    available = await voice_cloner.list_available_celebrities()
    
    if not available:
        print("No celebrities with voice samples available yet.")
        return None
    
    print(f"\nFound {len(available)} celebrities with voice samples:")
    for i, celeb in enumerate(available):
        print(f"{i+1}. {celeb.get('name', 'Unknown')} - {celeb.get('description', 'No description')}")
    
    while True:
        try:
            choice = input("\nSelect a celebrity (number) or press Enter to cancel: ")
            if not choice:
                return None
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available):
                return available[choice_idx]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

async def celebrity_voice_clone_demo():
    """Main demo function for celebrity voice cloning"""
    print("\n" + "="*50)
    print("Celebrity Voice Cloning Demo")
    print("="*50)
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), 'src', 'models', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create samples directory if it doesn't exist
    sample_dir = os.path.join(os.path.dirname(__file__), 'src', 'samples', 'celebrities')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Initialize the celebrity voice cloner
    voice_cloner = CelebrityVoiceCloner(cache_dir=cache_dir, sample_dir=sample_dir)
    
    # Initialize the cloner
    print("\nInitializing celebrity voice cloner...")
    success = await voice_cloner.initialize()
    
    if not success:
        print(f"❌ Failed to initialize voice cloner: {voice_cloner.initialization_error}")
        return
    
    print("✅ Celebrity voice cloner initialized successfully!")
    
    while True:
        print("\n" + "-"*50)
        print("Options:")
        print("1. Search for a celebrity")
        print("2. Use celebrity with existing voice sample")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ")
        
        if choice == "1":
            # Search for a celebrity
            query = input("\nEnter celebrity name to search: ")
            selected_celebrity = await search_celebrity(voice_cloner, query)
            
            if selected_celebrity:
                # Get text to speak
                text = input("\nEnter text for the celebrity to speak: ")
                if text:
                    await clone_voice(voice_cloner, selected_celebrity["id"], text)
        
        elif choice == "2":
            # List available celebrities
            selected_celebrity = await list_available_celebrities(voice_cloner)
            
            if selected_celebrity:
                # Get text to speak
                text = input("\nEnter text for the celebrity to speak: ")
                if text:
                    await clone_voice(voice_cloner, selected_celebrity["id"], text)
        
        elif choice == "3":
            # Exit
            print("\nExiting Celebrity Voice Cloning Demo. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

def main():
    """Entry point for the demo script"""
    parser = argparse.ArgumentParser(description="Celebrity Voice Cloning Demo")
    parser.add_argument(
        "--celebrity", 
        help="Celebrity name to search for"
    )
    parser.add_argument(
        "--text",
        help="Text for the celebrity to speak"
    )
    
    args = parser.parse_args()
    
    if args.celebrity and args.text:
        # Run with command line arguments
        async def run_with_args():
            # Initialize the voice cloner
            voice_cloner = CelebrityVoiceCloner()
            
            # Initialize the cloner
            print("\nInitializing celebrity voice cloner...")
            success = await voice_cloner.initialize()
            
            if not success:
                print(f"❌ Failed to initialize voice cloner: {voice_cloner.initialization_error}")
                return
            
            # Search for the celebrity
            results = await voice_cloner.search_celebrity(args.celebrity)
            
            if not results:
                print(f"No celebrities found matching '{args.celebrity}'")
                return
            
            # Use the first result
            selected_celebrity = results[0]
            print(f"\nUsing celebrity: {selected_celebrity.get('name', 'Unknown')}")
            
            # Clone voice
            await clone_voice(voice_cloner, selected_celebrity["id"], args.text)
        
        asyncio.run(run_with_args())
    else:
        # Run interactive demo
        asyncio.run(celebrity_voice_clone_demo())

if __name__ == "__main__":
    main()