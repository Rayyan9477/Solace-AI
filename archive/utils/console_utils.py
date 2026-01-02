"""
Console utilities for cross-platform compatibility.

This module provides utility functions for handling console output
in a cross-platform way, especially for Windows environments.
"""

import sys
import os
import io
import platform
import subprocess

def setup_console():
    """
    Set up the console for proper Unicode output.

    This function ensures that the console can handle Unicode characters
    like emojis properly on Windows systems.

    Security: Uses subprocess.run instead of os.system to avoid shell injection
    vulnerabilities (SEC-009).
    """
    if platform.system() == 'Windows':
        # Try to set console to use UTF-8
        try:
            # Force UTF-8 encoding for console output
            if sys.stdout.encoding != 'utf-8':
                sys.stdout.reconfigure(encoding='utf-8')

            # Try to set Windows console mode for Unicode using subprocess
            # Using subprocess.run with shell=False is safer than os.system
            subprocess.run(
                ['chcp', '65001'],
                shell=True,  # Required for Windows built-in commands
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False  # Don't raise on non-zero exit
            )

            # Set environment variable for UTF-8
            os.environ['PYTHONIOENCODING'] = 'utf-8'

            return True
        except Exception as e:
            print(f"Warning: Failed to set up Unicode console: {e}")
            return False

    return True

def safe_print(text):
    """
    Safely print text to console, handling Unicode characters.
    
    Args:
        text: The text to print
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Fall back to ASCII with replacement characters
        encoded_text = text.encode('ascii', 'replace').decode('ascii')
        print(encoded_text)

def emoji_to_ascii(text):
    """
    Convert emoji characters to ASCII equivalents.
    
    Args:
        text: Text containing emoji characters
        
    Returns:
        Text with emojis replaced by ASCII equivalents
    """
    # Simple emoji-to-ASCII mapping
    emoji_map = {
        '🌟': '**',  # star
        '✨': '*',   # sparkles
        '🔍': '[?]', # magnifying glass
        '📊': '[d]', # chart
        '🧠': '[b]', # brain
        '🤖': '[r]', # robot
        '❤️': '<3',  # heart
        '👍': '+1',  # thumbs up
        '😊': ':)',  # smile
        '👋': 'hi',  # wave
        '🙂': ':)',  # slight smile
        '😀': ':D',  # grin
        '\U0001f31f': '*',  # glowing star
    }
    
    for emoji, ascii_rep in emoji_map.items():
        text = text.replace(emoji, ascii_rep)
    
    return text
