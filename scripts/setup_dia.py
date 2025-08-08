"""Setup helper to optionally install the official Dia package for enhanced TTS (voice cloning)."""
import subprocess
import sys
import os


def main():
    try:
        print("Checking for official Dia package...")
        import dia  # noqa: F401
        print("Dia package already installed.")
        return
    except Exception:
        pass

    try:
        print("Cloning and installing dia from nari-labs/dia ...")
        subprocess.check_call(["git", "clone", "https://github.com/nari-labs/dia.git"])  # nosec B603
        repo = os.path.join(os.getcwd(), "dia")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo)
        print("Dia installed successfully.")
    except Exception as e:
        print(f"Failed to install Dia: {e}")


if __name__ == "__main__":
    main()

"""
Setup script to install and configure the official Dia 1.6B package from Nari Labs.
This provides enhanced text-to-speech capabilities for the Contextual-Chatbot.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_dia_package():
    """Setup the official Dia package"""
    try:
        print("\n🎙️ Setting up Dia 1.6B TTS package from Nari Labs...\n")
        
        # Define target directory
        repo_dir = Path("dia_package")
        if not repo_dir.exists():
            repo_dir.mkdir(parents=True)
        
        # Clone the repository
        print("🔄 Cloning the Dia repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/nari-labs/dia.git", str(repo_dir)],
            check=True
        )
        
        # Install dependencies
        print("\n📦 Installing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=str(repo_dir),
            check=True
        )
        
        # Add to Python path
        repo_path = os.path.abspath(repo_dir)
        print(f"\n🔧 Adding Dia package to Python path: {repo_path}")
        
        # Create or update .env file with path
        env_file = Path(".env")
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write(f"PYTHONPATH={repo_path}\n")
        else:
            # Read existing .env file
            with open(env_file, "r") as f:
                env_content = f.read()
            
            # Check if PYTHONPATH is already set
            if "PYTHONPATH=" in env_content:
                # Update PYTHONPATH
                lines = env_content.split("\n")
                updated_lines = []
                for line in lines:
                    if line.startswith("PYTHONPATH="):
                        # Append to existing PYTHONPATH
                        if repo_path not in line:
                            updated_line = f"{line}:{repo_path}"
                            updated_lines.append(updated_line)
                        else:
                            updated_lines.append(line)
                    else:
                        updated_lines.append(line)
                
                # Write updated content
                with open(env_file, "w") as f:
                    f.write("\n".join(updated_lines))
            else:
                # Append PYTHONPATH to .env
                with open(env_file, "a") as f:
                    f.write(f"\nPYTHONPATH={repo_path}\n")
        
        # Test the installation
        print("\n🧪 Testing Dia package installation...")
        try:
            import dia
            from dia.model import Dia
            print("✅ Dia package successfully imported!")
        except ImportError as e:
            print(f"⚠️ Dia package import failed. Please restart your environment: {str(e)}")
            
        print("\n✅ Dia 1.6B setup complete!")
        print("\nTo use the Dia package, you need to restart your Python environment.")
        print("After restarting, the chatbot will automatically detect and use the Dia package for enhanced TTS.")
        print("\nFeatures now available:")
        print("- Higher quality text-to-speech synthesis")
        print("- Multi-speaker dialogue synthesis")
        print("- Voice cloning capabilities")
        print("- Non-verbal audio generation (laughs, coughs, etc.)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        print(f"❌ Error during setup: Command failed with exit code {e.returncode}")
        if e.output:
            print(f"Output: {e.output}")
        return False
        
    except Exception as e:
        logger.error(f"Error setting up Dia package: {str(e)}")
        print(f"❌ Error setting up Dia package: {str(e)}")
        return False

if __name__ == "__main__":
    setup_dia_package()