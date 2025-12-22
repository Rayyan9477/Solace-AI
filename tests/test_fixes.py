#!/usr/bin/env python3
"""
Test script to verify the basic functionality of the Contextual-Chatbot.

This script tests:
1. Unicode encoding handling
2. Module discovery and registration
3. Module initialization
4. Basic application startup
"""

import sys
import os
from pathlib import Path
import traceback
import importlib

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

def test_unicode_handling():
    """Test Unicode handling"""
    print_section("Testing Unicode Handling")
    
    try:
        from src.utils.console_utils import setup_console, safe_print, emoji_to_ascii
        
        # Test console setup
        setup_result = setup_console()
        print(f"Console setup result: {setup_result}")
        
        # Test emoji conversion
        test_string = "Hello 🌟 World 🧠 !"
        ascii_string = emoji_to_ascii(test_string)
        print(f"Original: {test_string}")
        print(f"Converted: {ascii_string}")
        
        # Test safe print
        print("Testing safe_print with Unicode:")
        safe_print(test_string)
        
        return True
    except Exception as e:
        print(f"Unicode handling test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_module_discovery():
    """Test module discovery and registration"""
    print_section("Testing Module Discovery")
    
    try:
        from src.components.base_module import ModuleManager
        
        # Create module manager
        manager = ModuleManager()
        
        # Test discovery of components
        print("Discovering component modules...")
        comp_count = manager.discover_module_types("src.components")
        print(f"Discovered {comp_count} component modules")
        
        # Test discovery of agents
        print("Discovering agent modules...")
        agent_count = manager.discover_module_types("src.agents")
        print(f"Discovered {agent_count} agent modules")
        
        # Print discovered module types
        print("\nDiscovered module types:")
        for type_id in manager.module_types:
            print(f"- {type_id}")
        
        # Verify critical modules exist
        critical_modules = ["LLMModule", "VectorStoreModule", "CentralVectorDBModule"]
        missing = [mod for mod in critical_modules if mod not in manager.module_types]
        
        if missing:
            print(f"Warning: Missing critical modules: {', '.join(missing)}")
            return False
        else:
            print("All critical modules discovered successfully!")
            return True
    except Exception as e:
        print(f"Module discovery test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_diagnosis_results():
    """Test DiagnosticReport class from comprehensive diagnostic report module"""
    print_section("Testing DiagnosticReport")

    try:
        # Import from the comprehensive diagnostic report module (canonical location)
        from src.diagnosis.comprehensive_diagnostic_report import DiagnosticReport
        from datetime import datetime

        # Verify the dataclass can be imported
        print(f"DiagnosticReport class imported successfully")

        # Test that CONDITION_DEFINITIONS can be imported from shared constants
        from src.diagnosis.constants import CONDITION_DEFINITIONS
        print(f"CONDITION_DEFINITIONS loaded with {len(CONDITION_DEFINITIONS)} conditions")

        # Verify all expected conditions are present
        expected_conditions = ["depression", "anxiety", "stress", "ptsd", "bipolar"]
        for condition in expected_conditions:
            if condition in CONDITION_DEFINITIONS:
                print(f"- {condition}: {len(CONDITION_DEFINITIONS[condition]['symptoms'])} symptoms")
            else:
                print(f"- {condition}: MISSING")

        return True
    except Exception as e:
        print(f"DiagnosticReport test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading"""
    print_section("Testing Configuration")
    
    try:
        from src.config.settings import AppConfig
        
        # Print key configuration values
        print(f"App name: {AppConfig.APP_NAME}")
        print(f"Log level: {AppConfig.LOG_LEVEL}")
        print(f"API key configured: {'Yes' if (AppConfig.GEMINI_API_KEY or AppConfig.OPENAI_API_KEY) else 'No'}")
        if not AppConfig.MODEL_NAME:
            print("Warning: MODEL_NAME not set in environment variables or .env file")
            return False
        if not AppConfig.EMBEDDING_CONFIG["model_name"]:
            print("Warning: EMBEDDING_MODEL_NAME not set in environment variables or .env file")
            return False
        
        return True
    except Exception as e:
        print(f"Configuration test failed: {str(e)}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and return overall success status"""
    tests = [
        ("Unicode Handling", test_unicode_handling),
        ("Module Discovery", test_module_discovery),
        ("DiagnosisResults", test_diagnosis_results),
        ("Configuration", test_config)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        success = test_func()
        results.append((name, success))
    
    # Print summary
    print_section("Test Summary")
    
    all_success = True
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        if not success:
            all_success = False
        print(f"{name}: {status}")
    
    return all_success

if __name__ == "__main__":
    print("Running Contextual-Chatbot tests...")
    
    try:
        success = run_all_tests()
        
        if success:
            print("\nAll tests passed successfully!")
            sys.exit(0)
        else:
            print("\nSome tests failed. Please check the output for details.")
            sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
