#!/usr/bin/env python3
"""
Test script for JK Cement Digital Twin Platform Streaming Features
Tests real-time data streaming, Pub/Sub integration, and AI agent responses
"""

import sys
import os
import time
import json

# Add src to Python path
sys.path.insert(0, 'src')

def test_streaming_integration():
    """Test streaming integration with the main platform"""
    
    print("🏭 JK Cement Digital Twin - Streaming Integration Test")
    print("=" * 60)
    
    try:
        from cement_ai_platform.agents.jk_cement_platform import JKCementDigitalTwinPlatform
        
        # Initialize platform
        print("🚀 Initializing platform with streaming capabilities...")
        platform = JKCementDigitalTwinPlatform()
        
        # Test platform status
        print("\n📊 Platform Status:")
        status = platform.get_platform_status()
        print(f"   Platform Status: {status['platform_status']}")
        print(f"   Streaming Available: {status['streaming_capabilities']['pubsub_available']}")
        print(f"   Streaming Active: {status['streaming_capabilities']['streaming_active']}")
        
        # Test streaming status
        print("\n📡 Streaming Status:")
        streaming_status = platform.get_streaming_status()
        if streaming_status['streaming_available']:
            print(f"   ✅ Streaming Available")
            print(f"   Topics: {streaming_status['topics_configured']}")
            print(f"   Project ID: {streaming_status['project_id']}")
        else:
            print(f"   ❌ Streaming Not Available: {streaming_status.get('error', 'Unknown error')}")
            return False
        
        # Test starting streaming
        print("\n🚀 Testing Streaming Start...")
        success = platform.start_real_time_streaming(interval_seconds=3)
        if success:
            print("   ✅ Streaming started successfully")
            
            # Let it run for a few seconds
            print("   ⏳ Running streaming for 10 seconds...")
            time.sleep(10)
            
            # Test stopping streaming
            print("\n⏹️ Testing Streaming Stop...")
            stop_success = platform.stop_real_time_streaming()
            if stop_success:
                print("   ✅ Streaming stopped successfully")
            else:
                print("   ❌ Failed to stop streaming")
        else:
            print("   ❌ Failed to start streaming")
            return False
        
        print("\n✅ Streaming integration test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pubsub_simulator():
    """Test the Pub/Sub simulator directly"""
    
    print("\n📡 Testing Pub/Sub Simulator Directly...")
    print("-" * 40)
    
    try:
        from cement_ai_platform.streaming.pubsub_simulator import CementPlantPubSubSimulator, RealTimeDataProcessor
        
        # Initialize simulator
        print("🚀 Initializing Pub/Sub simulator...")
        simulator = CementPlantPubSubSimulator()
        processor = RealTimeDataProcessor()
        
        # Test data generation
        print("📊 Testing sensor data generation...")
        sensor_data = simulator._generate_sensor_snapshot()
        
        print("   Process Variables:")
        for key, value in sensor_data['process'].items():
            if key != 'timestamp':
                print(f"     {key}: {value}")
        
        print("   Quality Data:")
        for key, value in sensor_data['quality'].items():
            if key != 'timestamp':
                print(f"     {key}: {value}")
        
        # Test streaming for a short period
        print("\n🚀 Testing streaming simulation...")
        simulator.start_streaming_simulation(interval_seconds=2)
        
        print("   ⏳ Running for 8 seconds...")
        time.sleep(8)
        
        simulator.stop_streaming()
        print("   ✅ Streaming test completed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Simulator test error: {e}")
        return False

def test_realtime_dashboard():
    """Test the real-time dashboard (without launching Streamlit)"""
    
    print("\n📊 Testing Real-Time Dashboard Components...")
    print("-" * 40)
    
    try:
        from cement_ai_platform.streaming.realtime_dashboard import RealTimeStreamingDashboard
        
        # Initialize dashboard
        print("🚀 Initializing dashboard...")
        dashboard = RealTimeStreamingDashboard()
        
        # Test data generation
        print("📈 Testing historical data generation...")
        dashboard._simulate_realtime_updates(None, None, None, None)
        
        print(f"   Historical data points: {len(dashboard.historical_data)}")
        
        if dashboard.historical_data:
            latest = dashboard.historical_data[-1]
            print("   Latest data point:")
            print(f"     Free Lime: {latest['free_lime_percent']:.2f}%")
            print(f"     Kiln Temp: {latest['burning_zone_temp_c']:.0f}°C")
            print(f"     Fuel Rate: {latest['fuel_rate_tph']:.2f} t/h")
        
        print("   ✅ Dashboard components test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Dashboard test error: {e}")
        return False

def main():
    """Run all streaming tests"""
    
    print("🧪 JK Cement Digital Twin - Comprehensive Streaming Test Suite")
    print("=" * 70)
    
    tests = [
        ("Platform Integration", test_streaming_integration),
        ("Pub/Sub Simulator", test_pubsub_simulator),
        ("Real-Time Dashboard", test_realtime_dashboard)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        try:
            result = test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = "💥 CRASHED"
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        print(f"{test_name:.<30} {result}")
    
    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All streaming tests passed! Platform is ready for POC demonstration.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
