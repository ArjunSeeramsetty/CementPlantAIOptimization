#!/usr/bin/env python3
"""
Test script for the Unified Dashboard
Verifies that all dashboard modules can be imported and launched
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_dashboard_imports():
    """Test importing all dashboard modules"""
    print("🧪 Testing Unified Dashboard Module Imports...")
    print("-" * 50)
    
    # Test unified dashboard
    try:
        from cement_ai_platform.dashboard.unified_dashboard import main
        print("✅ Unified Dashboard: Import successful")
    except Exception as e:
        print(f"❌ Unified Dashboard: Import failed - {e}")
        return False
    
    # Test individual dashboards
    dashboards = [
        ("Streaming Dashboard", "cement_ai_platform.streaming.realtime_dashboard", "launch_streaming_demo"),
        ("Multi-Plant Dashboard", "cement_ai_platform.multi_plant.multi_plant_dashboard", "launch_multi_plant_demo"),
        ("Mobile Dashboard", "cement_ai_platform.mobile.mobile_dashboard", "launch_mobile_demo"),
        ("Maintenance Dashboard", "cement_ai_platform.maintenance.maintenance_dashboard", "launch_predictive_maintenance_demo"),
        ("Validation Dashboard", "cement_ai_platform.validation.validation_dashboard", "launch_data_validation_demo"),
        ("DWSIM Dashboard", "cement_ai_platform.dwsim.dwsim_dashboard", "launch_dwsim_integration_demo"),
    ]
    
    success_count = 0
    total_count = len(dashboards)
    
    for name, module_path, function_name in dashboards:
        try:
            module = __import__(module_path, fromlist=[function_name])
            func = getattr(module, function_name)
            print(f"✅ {name}: Import successful")
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: Import failed - {e}")
    
    print("-" * 50)
    print(f"📊 Test Results: {success_count}/{total_count} dashboards imported successfully")
    
    if success_count == total_count:
        print("🎉 All dashboard modules are ready!")
        return True
    else:
        print("⚠️  Some dashboard modules have issues")
        return False

def test_launcher_script():
    """Test the launcher script"""
    print("\n🚀 Testing Launcher Script...")
    print("-" * 30)
    
    launcher_path = Path(__file__).parent / "launch_unified_dashboard.py"
    
    if launcher_path.exists():
        print("✅ Launcher script exists")
        try:
            with open(launcher_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "streamlit" in content and "unified_dashboard" in content:
                    print("✅ Launcher script content looks correct")
                    return True
                else:
                    print("❌ Launcher script content is incorrect")
                    return False
        except Exception as e:
            print(f"❌ Error reading launcher script: {e}")
            return False
    else:
        print("❌ Launcher script not found")
        return False

def main():
    """Main test function"""
    print("🏭 JK Cement Digital Twin Platform - Unified Dashboard Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_dashboard_imports()
    
    # Test launcher
    launcher_ok = test_launcher_script()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Dashboard Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   Launcher Script: {'✅ PASS' if launcher_ok else '❌ FAIL'}")
    
    if imports_ok and launcher_ok:
        print("\n🎉 All tests passed! Unified Dashboard is ready for POC demonstration.")
        print("\n🚀 To launch the dashboard:")
        print("   streamlit run src/cement_ai_platform/dashboard/unified_dashboard.py")
        print("   OR")
        print("   python scripts/launch_unified_dashboard.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
