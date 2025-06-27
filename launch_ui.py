"""
UI Launcher for Mental Health Support Chatbot
Launches the Streamlit web interface
"""

import subprocess
import sys
import os
from pathlib import Path

def check_ui_dependencies():
    """Check if UI dependencies are installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_ui_dependencies():
    """Install UI dependencies"""
    print("Installing UI dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_ui.txt"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def main():
    """Launch the Streamlit UI"""
    
    # Check if we're in the right directory
    if not os.path.exists("app_streamlit.py"):
        print("❌ app_streamlit.py not found. Please run this from the project root directory.")
        return
    
    # Check if UI dependencies are installed
    if not check_ui_dependencies():
        print("⚠️ UI dependencies not found.")
        response = input("Would you like to install them? (y/n): ").lower().strip()
        if response == 'y':
            if not install_ui_dependencies():
                print("❌ Failed to install dependencies. Exiting.")
                return
        else:
            print("❌ Cannot launch UI without dependencies. Exiting.")
            return
    
    # Launch Streamlit
    print("🚀 Launching Streamlit UI...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_streamlit.py"])
    except KeyboardInterrupt:
        print("\n👋 UI shutdown.")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")

if __name__ == "__main__":
    main()
