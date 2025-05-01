"""Script to set up voice features for the Contextual Chatbot"""
import subprocess
import sys
import os

def check_pip():
    """Check if pip is available"""
    try:
        import pip
        return True
    except ImportError:
        print("pip is not installed. Please install pip first.")
        return False

def install_requirements():
    """Install voice feature requirements"""
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements_voice.txt')
    
    if not os.path.exists(requirements_file):
        print("Error: requirements_voice.txt not found!")
        return False
    
    print("Installing voice feature dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("\nVoice dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError installing dependencies: {e}")
        return False

def main():
    """Main setup function"""
    print("Setting up voice features for Contextual Chatbot...")
    
    if not check_pip():
        return
    
    if install_requirements():
        print("\nSetup completed successfully!")
        print("You can now use voice features in the chatbot.")
    else:
        print("\nSetup failed. Please try again or install dependencies manually:")
        print("pip install -r requirements_voice.txt")

if __name__ == "__main__":
    main()