"""
Basic Integration Test for Adaptive Learning System

Simple test to validate core functionality without complex dependencies.
"""

import sys
import os
from datetime import datetime

def test_privacy_system():
    """Test privacy protection system"""
    print("Testing Privacy Protection System...")
    
    try:
        # Import the privacy system
        sys.path.append(os.path.dirname(__file__))
        from privacy_protection_system import PrivacyProtectionSystem, DataSensitivity
        
        # Create privacy system
        privacy_system = PrivacyProtectionSystem()
        
        # Test data processing
        test_data = {
            'feedback': 'This session was helpful',
            'rating': 8,
            'duration': 30.5
        }
        
        processed = privacy_system.process_learning_data(
            user_id='test_user_123',
            data=test_data,
            sensitivity=DataSensitivity.CONFIDENTIAL
        )
        
        # Check anonymization
        if processed.anonymized_user_id != 'test_user_123':
            print("  - User ID anonymized: PASS")
        else:
            print("  - User ID anonymization: FAIL")
            return False
        
        # Check compliance
        compliance = privacy_system.validate_privacy_compliance()
        if compliance.get('overall_compliant', False):
            print("  - Privacy compliance: PASS")
        else:
            print("  - Privacy compliance: FAIL")
            return False
        
        print("Privacy Protection System: PASS")
        return True
        
    except Exception as e:
        print(f"Privacy Protection System: FAIL - {str(e)}")
        return False

def test_insights_system():
    """Test insights generation system"""
    print("\nTesting Insights Generation System...")
    
    try:
        from insights_generation_system import InsightGenerationSystem, SystemInsight, InsightType, InsightPriority
        
        # Create mock dependencies (simplified)
        class MockComponent:
            def __init__(self):
                self.data = {}
            
            def get_system_statistics(self):
                return {'total_users': 10, 'avg_satisfaction': 7.5}
            
            def get_feedback_statistics(self):
                return {'total_feedback_count': 50, 'avg_rating': 8.2}
            
            def get_personalization_summary(self):
                return {'active_profiles': 10, 'recommendations_made': 25}
        
        mock_outcome_tracker = MockComponent()
        mock_feedback_processor = MockComponent()
        mock_personalization_engine = MockComponent()
        mock_pattern_engine = MockComponent()
        
        # Create insights generator
        insights_generator = InsightGenerationSystem(
            outcome_tracker=mock_outcome_tracker,
            feedback_processor=mock_feedback_processor,
            personalization_engine=mock_personalization_engine,
            pattern_engine=mock_pattern_engine
        )
        
        # Test dashboard generation
        dashboard = insights_generator.get_insights_dashboard()
        if dashboard:
            print("  - Dashboard generation: PASS")
        else:
            print("  - Dashboard generation: FAIL")
            return False
        
        print("Insights Generation System: PASS")
        return True
        
    except Exception as e:
        print(f"Insights Generation System: FAIL - {str(e)}")
        return False

def test_data_structures():
    """Test core data structures"""
    print("\nTesting Data Structures...")
    
    try:
        from privacy_protection_system import DataPoint, PrivacyMetrics, DataSensitivity
        from insights_generation_system import SystemInsight, InsightType, InsightPriority
        
        # Test DataPoint
        data_point = DataPoint(
            anonymized_user_id='anon_123',
            data_type='interaction',
            features={'test': 'data'},
            sensitivity_level=DataSensitivity.INTERNAL,
            privacy_budget_used=0.1,
            timestamp=datetime.now()
        )
        
        if data_point.anonymized_user_id == 'anon_123':
            print("  - DataPoint creation: PASS")
        else:
            print("  - DataPoint creation: FAIL")
            return False
        
        # Test SystemInsight
        insight = SystemInsight(
            insight_id='test_123',
            insight_type=InsightType.PERFORMANCE_TREND,
            priority=InsightPriority.MEDIUM,
            title='Test Insight',
            description='Test description'
        )
        
        if insight.insight_id == 'test_123':
            print("  - SystemInsight creation: PASS")
        else:
            print("  - SystemInsight creation: FAIL")
            return False
        
        print("Data Structures: PASS")
        return True
        
    except Exception as e:
        print(f"Data Structures: FAIL - {str(e)}")
        return False

def test_configuration():
    """Test configuration handling"""
    print("\nTesting Configuration...")
    
    try:
        from privacy_protection_system import PrivacyProtectionSystem
        
        # Test custom configuration
        config = {
            'privacy_level': 'high',
            'differential_privacy_epsilon': 0.5,
            'data_retention_days': 60
        }
        
        privacy_system = PrivacyProtectionSystem(config=config)
        
        if privacy_system.epsilon == 0.5:
            print("  - Custom epsilon: PASS")
        else:
            print("  - Custom epsilon: FAIL")
            return False
        
        if privacy_system.data_retention_days == 60:
            print("  - Custom retention: PASS")
        else:
            print("  - Custom retention: FAIL")
            return False
        
        print("Configuration: PASS")
        return True
        
    except Exception as e:
        print(f"Configuration: FAIL - {str(e)}")
        return False

def main():
    """Run basic integration tests"""
    print("Adaptive Learning System - Basic Integration Test")
    print("=" * 55)
    
    tests = [
        ('Privacy System', test_privacy_system),
        ('Insights System', test_insights_system),
        ('Data Structures', test_data_structures),
        ('Configuration', test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"{test_name}: FAIL - {str(e)}")
    
    print("\n" + "=" * 55)
    print("TEST SUMMARY")
    print("=" * 55)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Overall: {'PASS' if passed == total else 'FAIL'}")
    
    if passed == total:
        print("\nAll basic integration tests passed!")
        print("The adaptive learning system core components are working correctly.")
    else:
        print(f"\n{total - passed} tests failed.")
        print("Please check the component implementations.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)