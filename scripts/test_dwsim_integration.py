#!/usr/bin/env python3
"""
Test script for DWSIM Integration functionality
"""

import os
import sys
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_dwsim_integration():
    """Test DWSIM integration engine functionality"""
    
    print("🧪 Testing DWSIM Integration Engine...")
    
    try:
        from cement_ai_platform.dwsim.dwsim_connector import DWSIMIntegrationEngine, DWSIMScenario
        
        # Initialize engine
        engine = DWSIMIntegrationEngine()
        print("✅ DWSIM Integration Engine initialized")
        
        # Test scenario execution
        print("\n📊 Testing scenario execution...")
        
        # Test startup scenario
        startup_scenario = engine.standard_scenarios['startup_sequence']
        result = engine.execute_scenario(startup_scenario, "test_plant")
        
        if result['success']:
            print(f"✅ Startup scenario completed in {result['execution_duration']:.1f}s")
            
            # Check results structure
            results = result['results']
            if 'time_series' in results:
                print("✅ Time series data generated")
            if 'final_steady_state' in results:
                print("✅ Steady state results generated")
            if 'startup_metrics' in results:
                print("✅ Startup metrics generated")
        else:
            print(f"❌ Startup scenario failed: {result.get('error')}")
            return False
        
        # Test fuel switching scenario
        fuel_scenario = engine.standard_scenarios['fuel_switching']
        result = engine.execute_scenario(fuel_scenario, "test_plant")
        
        if result['success']:
            print(f"✅ Fuel switching scenario completed in {result['execution_duration']:.1f}s")
            
            results = result['results']
            if 'transition_profile' in results:
                print("✅ Transition profile generated")
            if 'environmental_impact' in results:
                print("✅ Environmental impact calculated")
        else:
            print(f"❌ Fuel switching scenario failed: {result.get('error')}")
            return False
        
        # Test optimization scenario
        opt_scenario = engine.standard_scenarios['optimization_study']
        result = engine.execute_scenario(opt_scenario, "test_plant")
        
        if result['success']:
            print(f"✅ Optimization scenario completed in {result['execution_duration']:.1f}s")
            
            results = result['results']
            if 'optimization_matrix' in results:
                print("✅ Optimization matrix generated")
            if 'optimal_solution' in results:
                print("✅ Optimal solution found")
        else:
            print(f"❌ Optimization scenario failed: {result.get('error')}")
            return False
        
        # Test emergency scenario
        emergency_scenario = engine.standard_scenarios['emergency_shutdown']
        result = engine.execute_scenario(emergency_scenario, "test_plant")
        
        if result['success']:
            print(f"✅ Emergency scenario completed in {result['execution_duration']:.1f}s")
            
            results = result['results']
            if 'shutdown_sequence' in results:
                print("✅ Shutdown sequence generated")
            if 'safety_metrics' in results:
                print("✅ Safety metrics calculated")
        else:
            print(f"❌ Emergency scenario failed: {result.get('error')}")
            return False
        
        # Test custom scenario creation
        print("\n🛠️ Testing custom scenario creation...")
        
        custom_scenario = DWSIMScenario(
            scenario_id="test_custom_001",
            scenario_name="Test Custom Scenario",
            description="Test scenario for validation",
            input_parameters={
                'raw_meal_feed': 165.0,
                'fuel_rate': 16.0,
                'kiln_speed': 3.5,
                'o2_target': 3.2
            },
            expected_outputs=['burning_zone_temp', 'free_lime_percent'],
            simulation_duration=1800,
            priority='medium'
        )
        
        result = engine.execute_scenario(custom_scenario, "test_plant")
        
        if result['success']:
            print(f"✅ Custom scenario completed in {result['execution_duration']:.1f}s")
        else:
            print(f"❌ Custom scenario failed: {result.get('error')}")
            return False
        
        # Test scenario history
        print("\n📈 Testing scenario history...")
        
        history = engine.get_scenario_history("test_plant", limit=10)
        
        if history:
            print(f"✅ Retrieved {len(history)} scenario history records")
        else:
            print("⚠️ No scenario history available (expected in demo mode)")
        
        print("\n🎉 All DWSIM integration tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dwsim_dashboard():
    """Test DWSIM dashboard functionality"""
    
    print("\n🖥️ Testing DWSIM Dashboard...")
    
    try:
        from cement_ai_platform.dwsim.dwsim_dashboard import launch_dwsim_integration_demo
        
        print("✅ DWSIM dashboard module imported successfully")
        print("ℹ️ Dashboard can be launched with: streamlit run src/cement_ai_platform/dwsim/dwsim_dashboard.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 DWSIM Integration Test Suite")
    print("=" * 50)
    
    # Test results
    test_results = {
        'dwsim_integration': False,
        'dwsim_dashboard': False
    }
    
    # Run tests
    test_results['dwsim_integration'] = test_dwsim_integration()
    test_results['dwsim_dashboard'] = test_dwsim_dashboard()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DWSIM Integration Test Results:")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All DWSIM integration tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
