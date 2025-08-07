"""
Simple Integration Test for Adaptive Learning System

This module provides basic integration testing that works with minimal dependencies
to validate the core architecture and integration patterns.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any
import asyncio

def test_imports():
    """Test that all modules can be imported"""
    
    print("Testing module imports...")
    
    # Test core imports
    try:
        from .privacy_protection_system import PrivacyProtectionSystem, DataSensitivity, PrivacyLevel
        print("Privacy protection system imported successfully")
    except Exception as e:
        print(f"Privacy protection system import failed: {e}")
        return False
    
    try:
        from .insights_generation_system import InsightGenerationSystem, SystemInsight, InsightType, InsightPriority
        print("Insights generation system imported successfully")
    except Exception as e:
        print(f"Insights generation system import failed: {e}")
        return False
    
    try:
        from .adaptive_learning_integration import AdaptiveLearningSystemIntegrator
        print("Adaptive learning integration imported successfully")
    except Exception as e:
        print(f"Adaptive learning integration import failed: {e}")
        return False
    
    return True

def test_privacy_system():
    """Test privacy protection system basic functionality"""
    
    print("\nTesting privacy protection system...")
    
    try:
        from .privacy_protection_system import PrivacyProtectionSystem, DataSensitivity
        
        config = {
            'privacy_level': 'standard',
            'differential_privacy_epsilon': 1.0,
            'k_anonymity': 5
        }
        
        privacy_system = PrivacyProtectionSystem(config)
        
        # Test data processing
        test_data = {
            'user_feedback': 'This session was helpful',
            'session_duration': 30.5,
            'rating': 8
        }
        
        processed_data = privacy_system.process_learning_data(
            user_id='test_user_123',
            data=test_data,
            sensitivity=DataSensitivity.CONFIDENTIAL
        )
        
        # Verify anonymization
        if processed_data.anonymized_user_id == 'test_user_123':
            print("✗ User ID was not anonymized")
            return False
        
        # Test privacy compliance
        compliance = privacy_system.validate_privacy_compliance()
        
        if not compliance.get('overall_compliant'):
            print("✗ Privacy compliance validation failed")
            return False
        
        print("✓ Privacy system basic functionality working")
        return True
        
    except Exception as e:
        print(f"✗ Privacy system test failed: {e}")
        return False

def test_data_structures():
    """Test that data structures are properly defined"""
    
    print("\n📊 Testing data structures...")
    
    try:
        from .privacy_protection_system import DataPoint, PrivacyMetrics
        from .insights_generation_system import SystemInsight, InsightPriority, InsightType
        
        # Test DataPoint creation
        data_point = DataPoint(
            anonymized_user_id='anon_123',
            data_type='interaction',
            features={'test': 'data'},
            sensitivity_level=DataSensitivity.INTERNAL,
            privacy_budget_used=0.1,
            timestamp=datetime.now()
        )
        
        if not data_point.anonymized_user_id:
            print("✗ DataPoint structure invalid")
            return False
        
        # Test SystemInsight creation
        insight = SystemInsight(
            insight_id='test_insight_123',
            insight_type=InsightType.PERFORMANCE_TREND,
            priority=InsightPriority.MEDIUM,
            title='Test Insight',
            description='This is a test insight'
        )
        
        if not insight.insight_id:
            print("✗ SystemInsight structure invalid")
            return False
        
        print("✓ Data structures properly defined")
        return True
        
    except Exception as e:
        print(f"✗ Data structures test failed: {e}")
        return False

def test_configuration():
    """Test configuration handling"""
    
    print("\n⚙️  Testing configuration handling...")
    
    try:
        from .privacy_protection_system import PrivacyProtectionSystem
        
        # Test with default configuration
        default_privacy = PrivacyProtectionSystem()
        
        # Test with custom configuration
        custom_config = {
            'privacy_level': 'high',
            'differential_privacy_epsilon': 0.5,
            'data_retention_days': 60
        }
        
        custom_privacy = PrivacyProtectionSystem(config=custom_config)
        
        # Verify configuration was applied
        if custom_privacy.epsilon != 0.5:
            print("✗ Custom configuration not applied properly")
            return False
        
        if custom_privacy.data_retention_days != 60:
            print("✗ Custom retention period not set")
            return False
        
        print("✓ Configuration handling working")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_error_handling():
    """Test error handling robustness"""
    
    print("\n🛡️  Testing error handling...")
    
    try:
        from .privacy_protection_system import PrivacyProtectionSystem, DataSensitivity
        
        privacy_system = PrivacyProtectionSystem()
        
        # Test with invalid data
        try:
            result = privacy_system.process_learning_data(
                user_id=None,  # Invalid user ID
                data={'test': 'data'},
                sensitivity=DataSensitivity.CONFIDENTIAL
            )
            print("✗ Should have handled None user_id gracefully")
            return False
        except Exception:
            # Expected to fail gracefully
            pass
        
        # Test with empty data
        try:
            result = privacy_system.process_learning_data(
                user_id='test_user',
                data={},  # Empty data
                sensitivity=DataSensitivity.CONFIDENTIAL
            )
            # Should handle empty data gracefully
            if result:
                print("✓ Empty data handled gracefully")
        except Exception as e:
            print(f"✗ Empty data not handled gracefully: {e}")
            return False
        
        print("✓ Error handling working")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_file_operations():
    """Test file operations and data persistence"""
    
    print("\n💾 Testing file operations...")
    
    try:
        # Create test directory
        test_dir = 'src/data/test_adaptive_learning'
        os.makedirs(test_dir, exist_ok=True)
        
        # Test JSON serialization
        test_data = {
            'system_info': {
                'test_timestamp': datetime.now().isoformat(),
                'components_tested': ['privacy_protection', 'insights_generation'],
                'test_status': 'running'
            },
            'results': {
                'privacy_tests_passed': True,
                'integration_tests_passed': True
            }
        }
        
        # Write test file
        test_file = os.path.join(test_dir, 'integration_test.json')
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2, default=str)
        
        # Read test file
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        if loaded_data['system_info']['test_status'] != 'running':
            print("✗ File read/write integrity failed")
            return False
        
        # Cleanup
        os.remove(test_file)
        
        print("✓ File operations working")
        return True
        
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False

def generate_integration_report(test_results: Dict[str, bool]):
    """Generate integration test report"""
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    report = {
        'test_summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED'
        },
        'test_results': test_results,
        'integration_assessment': {
            'core_imports': test_results.get('imports', False),
            'privacy_protection': test_results.get('privacy_system', False),
            'data_structures': test_results.get('data_structures', False),
            'configuration': test_results.get('configuration', False),
            'error_handling': test_results.get('error_handling', False),
            'file_operations': test_results.get('file_operations', False)
        },
        'recommendations': [],
        'generated_at': datetime.now().isoformat()
    }
    
    # Add recommendations based on test results
    if not test_results.get('imports', False):
        report['recommendations'].append("Fix import dependencies before deployment")
    
    if not test_results.get('privacy_system', False):
        report['recommendations'].append("Privacy protection system needs attention")
    
    if not test_results.get('error_handling', False):
        report['recommendations'].append("Improve error handling robustness")
    
    if success_rate == 100:
        report['recommendations'].append("All basic integration tests passed - system ready for advanced testing")
    
    return report

def main():
    """Run simple integration tests"""
    
    print("Adaptive Learning System - Simple Integration Test")
    print("=" * 60)
    
    # Run tests
    test_functions = [
        ('imports', test_imports),
        ('privacy_system', test_privacy_system),
        ('data_structures', test_data_structures),
        ('configuration', test_configuration),
        ('error_handling', test_error_handling),
        ('file_operations', test_file_operations)
    ]
    
    test_results = {}
    
    for test_name, test_function in test_functions:
        try:
            result = test_function()
            test_results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            test_results[test_name] = False
    
    # Generate report
    report = generate_integration_report(test_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {report['test_summary']['total_tests']}")
    print(f"Passed: {report['test_summary']['passed_tests']}")
    print(f"Failed: {report['test_summary']['failed_tests']}")
    print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
    print(f"Overall Status: {report['test_summary']['overall_status']}")
    
    if report['test_summary']['failed_tests'] > 0:
        print("\n❌ FAILED TESTS:")
        for test_name, passed in test_results.items():
            if not passed:
                print(f"  • {test_name}")
    
    if report['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Save report
    try:
        os.makedirs('src/data/test_reports', exist_ok=True)
        report_file = f"src/data/test_reports/simple_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Report saved to: {report_file}")
        
    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")
    
    print("\n" + "=" * 60)
    
    return report['test_summary']['overall_status'] == 'PASSED'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)