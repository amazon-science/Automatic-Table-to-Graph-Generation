#!/usr/bin/env python3
"""
Test runner for all AutoG2 Web App unit tests
"""

import sys
import os
from pathlib import Path

def run_all_tests():
    """Run all available unit tests"""
    
    print("🧪 AutoG2 Web App - Unit Test Suite")
    print("=" * 60)
    
    # Get the current directory
    test_dir = Path(__file__).parent
    
    # List of test modules to run
    test_modules = [
        'test_data_loading',
        'test_schema_generation',
        'test_caching'
    ]
    
    results = {}
    
    for test_module in test_modules:
        print(f"\n🔍 Running {test_module}...")
        print("-" * 40)
        
        try:
            # Import and run the test
            if test_module == 'test_data_loading':
                from test_data_loading import test_data_loading
                test_data_loading()
                results[test_module] = "✅ PASSED"
                
            elif test_module == 'test_schema_generation':
                from test_schema_generation import test_schema_generation
                test_schema_generation()
                results[test_module] = "✅ PASSED"
                
            elif test_module == 'test_caching':
                from test_caching import test_caching_strategies, test_cache_size_limits, test_file_type_classification
                test_caching_strategies()
                test_cache_size_limits()
                test_file_type_classification()
                results[test_module] = "✅ PASSED"
                
        except Exception as e:
            results[test_module] = f"❌ FAILED: {str(e)}"
            print(f"❌ Test {test_module} failed: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        print(f"{result:<20} {test_name}")
        if "PASSED" in result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)