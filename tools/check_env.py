#!/usr/bin/env python3
"""
Environment and Dependency Checker

This script verifies that the environment is correctly set up for
the Contextual-Chatbot application, checking:
1. Python version
2. Required dependencies
3. Configuration settings
4. Directory structure
5. File encodings
"""

import sys
import os
import traceback
from pathlib import Path
import platform

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

# ASCII color codes for terminal output
class Colors:
    HEADER = '\033[95m' if platform.system() != 'Windows' else ''
    BLUE = '\033[94m' if platform.system() != 'Windows' else ''
    GREEN = '\033[92m' if platform.system() != 'Windows' else ''
    YELLOW = '\033[93m' if platform.system() != 'Windows' else ''
    RED = '\033[91m' if platform.system() != 'Windows' else ''
    END = '\033[0m' if platform.system() != 'Windows' else ''
    BOLD = '\033[1m' if platform.system() != 'Windows' else ''
    UNDERLINE = '\033[4m' if platform.system() != 'Windows' else ''

def print_header(text):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}")
    print(f" {text} ".center(70))
    print(f"{'=' * 70}{Colors.END}")

def print_result(name, status, message=""):
    """Print a check result with appropriate coloring"""
    color = Colors.GREEN if status else Colors.RED
    status_text = "PASS" if status else "FAIL"
    try:
        print(f"{name:<40} {color}{status_text}{Colors.END} {message}")
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        print(f"{name:<40} {status_text} {message}")

def check_python_version():
    """Check Python version"""
    print_header("Python Environment Check")
    
    required_version = (3, 9)
    current_version = sys.version_info
    is_compatible = (current_version.major, current_version.minor) >= required_version
    
    print_result(
        "Python Version", 
        is_compatible, 
        f"(Required: {required_version[0]}.{required_version[1]}+, Current: {current_version.major}.{current_version.minor}.{current_version.micro})"
    )
    
    print_result(
        "Platform", 
        True, 
        f"({platform.system()} {platform.release()})"
    )
    
    return is_compatible

def check_dependencies():
    """Check required dependencies"""
    print_header("Dependency Check")
    
    # Core dependencies
    core_dependencies = [
        "python-dotenv",
        "pydantic",
        "torch",
        "transformers",
        "langchain",
        "google-generativeai",
        "numpy",
        "faiss-cpu",
    ]
    
    all_dependencies = {
        "Core": core_dependencies,
        "Vector Store": ["faiss-cpu", "qdrant-client", "pymilvus"],
        "API": ["fastapi", "uvicorn"],
        "Voice": [
            "openai_whisper",  # openai-whisper
            "sounddevice",
            "soundfile",
            "librosa",
            "transformers"
        ]
    }
    
    all_pass = True
    
    for category, deps in all_dependencies.items():
        print(f"\n{Colors.BOLD}{category} Dependencies:{Colors.END}")
        
        for package in deps:
            try:
                # Try to import the package
                __import__(package.replace('-', '_'))
                print_result(package, True, "(Installed)")
            except ImportError:
                print_result(package, False, "(Not installed)")
                if package in core_dependencies:
                    all_pass = False
            except Exception as e:
                print_result(package, False, f"(Error: {str(e)})")
                if package in core_dependencies:
                    all_pass = False
    
    return all_pass

def check_env_file():
    """Check .env file and API key"""
    print_header("Environment Variables Check")
    
    # Check for .env file
    env_file = Path('.env')
    env_exists = env_file.exists()
    print_result(".env file exists", env_exists)
    
    if env_exists:
        with open(env_file, 'r') as f:
            content = f.read()
            has_api_key = 'GEMINI_API_KEY' in content
            print_result("GEMINI_API_KEY in .env", has_api_key)
    
    # Check environment variable directly
    api_key = os.environ.get('GEMINI_API_KEY')
    env_var_set = bool(api_key)
    print_result("GEMINI_API_KEY environment variable", env_var_set)
    
    # Try loading with dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key_after_load = os.environ.get('GEMINI_API_KEY')
        dotenv_loaded = bool(api_key_after_load)
        print_result("dotenv loaded GEMINI_API_KEY", dotenv_loaded)
        
        return dotenv_loaded
    except ImportError:
        print_result("python-dotenv available", False)
        return env_var_set  # Return direct env var check if dotenv not available

def check_configuration():
    """Check application configuration"""
    print_header("Configuration Check")
    
    try:
        # Import config (this will validate required env vars)
        try:
            from src.config.settings import AppConfig
            print_result("AppConfig loaded", True)
            
            # Check API keys
            api_key_status = bool(AppConfig.GEMINI_API_KEY)
            print_result("GEMINI_API_KEY in AppConfig", api_key_status)
            
            # Check paths
            for path_name in ["DATA_DIR", "MODEL_DIR", "VECTOR_STORE_PATH"]:
                if hasattr(AppConfig, path_name):
                    path_value = getattr(AppConfig, path_name)
                    exists = Path(path_value).exists()
                    print_result(f"{path_name} exists", exists, f"({path_value})")
            
            # Check vector DB config
            vector_db_config = hasattr(AppConfig, "VECTOR_DB_CONFIG")
            print_result("Vector DB Configuration", vector_db_config)
            
            # Overall check
            overall = api_key_status and vector_db_config
            return overall
        except Exception as e:
            print_result("AppConfig", False, f"Error: {str(e)}")
            return False
            
    except Exception as e:
        print_result("Configuration", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def check_unicode_support():
    """Check Unicode support in the console"""
    print_header("Unicode Support Check")
    
    try:
        # Import our console utils
        try:
            from src.utils.console_utils import setup_console, safe_print, emoji_to_ascii
            print_result("Console utils imported", True)
            
            # Check if setup_console works
            setup_result = setup_console()
            print_result("Console setup", setup_result)
            
            # Test emoji conversion
            test_string = "Hello 🌟 World 🧠 !"
            ascii_string = emoji_to_ascii(test_string)
            print(f"Original: {test_string}")
            print(f"Converted: {ascii_string}")
            print_result("Emoji conversion", True, f"(Converted successfully)")
            
            return True
        except ImportError as e:
            print_result("Console utils import", False, f"Error: {str(e)}")
            return False
    except Exception as e:
        print_result("Unicode support", False, f"Error: {str(e)}")
        return False

def run_all_checks():
    """Run all environment checks"""
    results = {}
    
    # Python version check
    results["Python Version"] = check_python_version()
    
    # Dependencies check
    results["Dependencies"] = check_dependencies()
    
    # Environment variables check
    results["Environment Variables"] = check_env_file()
    
    # Configuration check
    results["Configuration"] = check_configuration()
    
    # Unicode support check
    results["Unicode Support"] = check_unicode_support()
    
    # Print summary
    print_header("Environment Check Summary")
    
    all_pass = True
    for check, result in results.items():
        if not result:
            all_pass = False
        print_result(check, result)
    
    if all_pass:
        try:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All checks passed! The environment is correctly configured.{Colors.END}")
        except UnicodeEncodeError:
            print(f"\nAll checks passed! The environment is correctly configured.")
        print("\nYou can now run the application with:")
        print(f"{Colors.BOLD}python main.py{Colors.END}")
    else:
        try:
            print(f"\n{Colors.RED}{Colors.BOLD}Some checks failed. Please fix the issues before running the application.{Colors.END}")
        except UnicodeEncodeError:
            print(f"\nSome checks failed. Please fix the issues before running the application.")
        print("\nRefer to the documentation for troubleshooting steps.")
    
    return all_pass

if __name__ == "__main__":
    print(f"{Colors.BOLD}Contextual-Chatbot Environment Checker{Colors.END}")
    print("Checking if your environment is correctly set up...\n")
    
    try:
        success = run_all_checks()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n{Colors.RED}Critical error during environment check: {str(e)}{Colors.END}")
        traceback.print_exc()
        sys.exit(1)