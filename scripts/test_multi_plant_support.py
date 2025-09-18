#!/usr/bin/env python3
"""
Test script for Multi-Plant Support functionality
"""

import os
import sys
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_multi_plant_manager():
    """Test MultiPlantManager functionality"""
    
    print("🧪 Testing MultiPlantManager...")
    
    try:
        from cement_ai_platform.multi_plant.plant_manager import MultiPlantManager, PlantConfiguration
        
        # Initialize manager
        manager = MultiPlantManager()
        print("✅ MultiPlantManager initialized")
        
        # Test plant discovery
        all_plants = list(manager.plant_registry.values())
        print(f"✅ Discovered {len(all_plants)} plants")
        
        # Test tenant isolation
        tenants = list(set(p.tenant_id for p in all_plants))
        print(f"✅ Found {len(tenants)} tenants: {tenants}")
        
        # Test getting tenant plants
        for tenant_id in tenants[:2]:  # Test first 2 tenants
            tenant_plants = manager.get_tenant_plants(tenant_id)
            print(f"✅ Tenant {tenant_id}: {len(tenant_plants)} plants")
        
        # Test cross-plant benchmarks
        for tenant_id in tenants[:2]:
            benchmarks = manager.get_cross_plant_benchmarks(tenant_id)
            if benchmarks:
                print(f"✅ Generated benchmarks for {tenant_id}")
        
        # Test plant configuration retrieval
        if all_plants:
            test_plant = all_plants[0]
            config = manager.get_plant_config(test_plant.plant_id)
            if config:
                print(f"✅ Retrieved config for {config.plant_name}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_multi_plant_supervisor():
    """Test MultiPlantSupervisor functionality"""
    
    print("\n🧪 Testing MultiPlantSupervisor...")
    
    try:
        from cement_ai_platform.multi_plant.multi_plant_supervisor import MultiPlantSupervisor
        
        # Initialize supervisor
        supervisor = MultiPlantSupervisor()
        print("✅ MultiPlantSupervisor initialized")
        
        # Test supervisor status
        status = supervisor.get_supervisor_status()
        print(f"✅ Supervisor status retrieved: {status['supervisor_status']['initialized']}")
        
        # Test plant discovery
        active_plants = status['supervisor_status']['active_plants']
        print(f"✅ Active plants: {active_plants}")
        
        # Test orchestration start/stop
        print("🔄 Testing orchestration...")
        supervisor.start_orchestration()
        time.sleep(2)  # Let it run briefly
        
        # Check if running
        status = supervisor.get_supervisor_status()
        if status['supervisor_status']['running']:
            print("✅ Orchestration started successfully")
        
        # Stop orchestration
        supervisor.stop_orchestration()
        time.sleep(1)
        
        status = supervisor.get_supervisor_status()
        if not status['supervisor_status']['running']:
            print("✅ Orchestration stopped successfully")
        
        # Test tenant model deployment
        tenants = list(supervisor.tenant_plants.keys())
        if tenants:
            test_tenant = tenants[0]
            deployment_result = supervisor.deploy_model_to_tenant(
                test_tenant, 
                {'model_name': 'test_model', 'version': '1.0'}
            )
            print(f"✅ Model deployment test for {test_tenant}: {deployment_result['successful_deployments']}/{deployment_result['total_plants']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_mobile_dashboard():
    """Test Mobile Dashboard functionality"""
    
    print("\n🧪 Testing Mobile Dashboard...")
    
    try:
        from cement_ai_platform.mobile.mobile_dashboard import MobileCementDashboard
        
        # Initialize mobile dashboard
        mobile_app = MobileCementDashboard()
        print("✅ MobileCementDashboard initialized")
        
        print("ℹ️ Mobile dashboard can be launched with: streamlit run src/cement_ai_platform/mobile/mobile_dashboard.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Multi-Plant Support Test Suite")
    print("=" * 50)
    
    # Test results
    test_results = {
        'multi_plant_manager': False,
        'multi_plant_supervisor': False,
        'mobile_dashboard': False
    }
    
    # Run tests
    test_results['multi_plant_manager'] = test_multi_plant_manager()
    test_results['multi_plant_supervisor'] = test_multi_plant_supervisor()
    test_results['mobile_dashboard'] = test_mobile_dashboard()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Multi-Plant Support Test Results:")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All multi-plant support tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
