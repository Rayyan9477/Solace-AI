"""
Demo script for the Dia 1.6B text-to-speech functionality.
Run this to test the TTS capabilities of the chatbot.
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dia_tts import DiaTTS
from src.utils.audio_player import AudioPlayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tts_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def tts_demo(text: str = None, style: str = "warm", list_styles: bool = False):
    """
    Demonstrate Dia 1.6B text-to-speech functionality
    
    Args:
        text: Text to convert to speech (if None, use a sample text)
        style: Voice style to use
        list_styles: Whether to list available voice styles
    """
    print("Initializing Dia 1.6B text-to-speech...")
    
    # Create model directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), 'src', 'models', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize DiaTTS with Dia 1.6B
    tts = DiaTTS(cache_dir=cache_dir)
    success = await tts.initialize()
    
    if not success:
        print(f"❌ Failed to initialize Dia 1.6B: {tts.initialization_error}")
        return
    
    # Initialize audio player
    player = AudioPlayer()
    
    # If list_styles, print available styles and exit
    if list_styles:
        styles = tts.get_available_styles()
        print("\nAvailable voice styles:")
        for style_name in styles:
            print(f"- {style_name}")
        return
    
    # Default sample text
    if text is None:
        text = "Hello! I'm the Mental Health Support Bot using the Dia 1.6B text-to-speech system. " \
               "I'm here to provide empathetic and helpful responses in a natural-sounding voice. " \
               "How can I support you today?"
    
    print(f"\nConverting text to speech using style: {style}")
    print(f"Text: \"{text}\"")
    
    # Generate speech
    start_time = asyncio.get_event_loop().time()
    result = await tts.generate_speech(text, style=style)
    end_time = asyncio.get_event_loop().time()
    
    if not result["success"]:
        print(f"❌ Failed to generate speech: {result.get('error', 'Unknown error')}")
        return
    
    # Print generation stats
    duration = result.get("duration", 0)
    print(f"\nGeneration stats:")
    print(f"- Processing time: {end_time - start_time:.2f} seconds")
    print(f"- Audio duration: {duration:.2f} seconds")
    print(f"- Text length: {len(text)} characters")
    print(f"- Character rate: {len(text) / (end_time - start_time):.2f} chars/sec")
    
    # Play the audio
    print("\nPlaying audio...")
    player.play_audio(result["audio_bytes"])
    
    print("\n✅ Text-to-speech demo complete!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Dia 1.6B Text-to-Speech Demo")
    parser.add_argument(
        "--text", 
        help="Text to convert to speech"
    )
    parser.add_argument(
        "--style",
        default="warm",
        help="Voice style to use (e.g., warm, calm, excited, sad, professional)"
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available voice styles"
    )
    
    args = parser.parse_args()
    
    # Run the demo
    asyncio.run(tts_demo(args.text, args.style, args.list_styles))

if __name__ == "__main__":
    main()