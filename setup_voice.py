"""Script to set up voice features for the Contextual Chatbot"""
import subprocess
import sys
import os
import logging
from pathlib import Path
import time

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

def download_models():
    """Download voice models from Hugging Face"""
    try:
        print("\nDownloading voice models from Hugging Face...")
        print("This may take some time depending on your internet connection.")
        
        # Import required libraries after installation
        import torch
        import huggingface_hub
        from transformers import AutoProcessor, AutoModel, pipeline, WhisperProcessor
        
        # Set cache directory
        cache_dir = Path(__file__).resolve().parent / "src" / "models" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure huggingface_hub to use the cache directory
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        huggingface_hub.config.HUGGINGFACE_HUB_CACHE = str(cache_dir)
        
        # Models to download
        models = [
            {
                "name": "Whisper V3 Turbo (STT)",
                "repo_id": "openai/whisper-large-v3-turbo",
                "filename": "pytorch_model.bin"
            },
            {
                "name": "Dia-1.6B (TTS)",
                "repo_id": "nari-labs/Dia-1.6B",
                "filename": "config.json"
            },
            {
                "name": "SpeechT5 TTS",
                "repo_id": "microsoft/speecht5_tts",
                "filename": "config.json"
            },
            {
                "name": "SpeechT5 HiFiGan",
                "repo_id": "microsoft/speecht5_hifigan",
                "filename": "config.json"
            }
        ]
        
        for i, model in enumerate(models):
            print(f"\n[{i+1}/{len(models)}] Downloading {model['name']} ({model['repo_id']})...")
            
            try:
                start_time = time.time()
                # First just get the model info/config to check access
                huggingface_hub.hf_hub_download(
                    repo_id=model["repo_id"],
                    filename=model["filename"],
                    cache_dir=cache_dir
                )
                
                # For Whisper, also initialize whisper to download the models
                if "whisper" in model["repo_id"].lower():
                    print(f"Initializing {model['name']} processor... (this will download the model)")
                    processor = WhisperProcessor.from_pretrained(
                        model["repo_id"],
                        cache_dir=cache_dir
                    )
                    
                    # Also check if we have openai-whisper package and initialize it
                    try:
                        import whisper
                        whisper_name = "large-v3" if "v3" in model["repo_id"] else "large"
                        print(f"Downloading openai/whisper model: {whisper_name}")
                        _ = whisper.load_model(whisper_name, download_root=cache_dir)
                    except ImportError:
                        print("openai-whisper package not found. Only Transformer models will be downloaded.")
                # Special handling for Dia 1.6B
                elif "dia" in model["repo_id"].lower():
                    print(f"Initializing {model['name']} processor and model... (this will download the complete model)")
                    processor = AutoProcessor.from_pretrained(
                        model["repo_id"],
                        cache_dir=cache_dir
                    )
                    
                    # Also initialize the model to ensure it's fully downloaded
                    print(f"Loading Dia 1.6B model components...")
                    use_gpu = torch.cuda.is_available()
                    dtype = torch.float16 if use_gpu else torch.float32
                    model_obj = AutoModel.from_pretrained(
                        model["repo_id"],
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        use_safetensors=True,
                        cache_dir=cache_dir
                    )
                    print(f"✅ Dia 1.6B model loaded successfully on {'GPU' if use_gpu else 'CPU'}")
                else:
                    # For other models, initialize processor and/or model
                    print(f"Initializing {model['name']} processor and model...")
                    processor = AutoProcessor.from_pretrained(
                        model["repo_id"],
                        cache_dir=cache_dir
                    )
                
                elapsed = time.time() - start_time
                print(f"✅ Downloaded {model['name']} in {elapsed:.2f} seconds")
            except Exception as e:
                print(f"❌ Error downloading {model['name']}: {str(e)}")
        
        # Check for models in cache directory
        model_dirs = [d for d in cache_dir.glob("models--*") if d.is_dir()]
        if model_dirs:
            print("\nDownloaded models:")
            for model_dir in model_dirs:
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                print(f"- {model_name}")
        
        return True
        
    except ImportError as e:
        print(f"\nError importing required libraries: {e}")
        print("Please make sure to install the requirements first.")
        return False
    except Exception as e:
        print(f"\nError downloading models: {e}")
        return False

def main():
    """Main setup function"""
    print("Setting up voice features for Contextual Chatbot...")
    
    if not check_pip():
        return
    
    if install_requirements():
        print("\nDependencies installed successfully!")
        
        # Download models
        if download_models():
            print("\nSetup completed successfully!")
            print("You can now use voice features in the chatbot.")
        else:
            print("\nModel download failed. You may need to download models manually.")
    else:
        print("\nSetup failed. Please try again or install dependencies manually:")
        print("pip install -r requirements_voice.txt")

if __name__ == "__main__":
    main()