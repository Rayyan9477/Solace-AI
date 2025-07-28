"""
Testing Module

This module provides comprehensive testing capabilities for all diagnostic
and therapeutic systems, including unit tests, integration tests, and validation.
"""

try:
    from .comprehensive_testing import (
        ComprehensiveTester,
        TestResult,
        TestSuite,
        run_comprehensive_tests
    )
except ImportError:
    ComprehensiveTester = None
    TestResult = None
    TestSuite = None
    run_comprehensive_tests = None

__all__ = [
    'ComprehensiveTester',
    'TestResult',
    'TestSuite',
    'run_comprehensive_tests'
]