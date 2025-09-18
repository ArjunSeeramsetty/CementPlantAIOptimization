#!/usr/bin/env python3
"""
Real-Time Streaming Demo Launcher for JK Cement Digital Twin Platform
Demonstrates live sensor data streaming via Google Cloud Pub/Sub
"""

import sys
import os
import subprocess
import time

def main():
    """Launch the real-time streaming demo"""
    
    print("🏭 JK Cement Digital Twin - Real-Time Streaming Demo")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/cement_ai_platform"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Add src to Python path
    sys.path.insert(0, 'src')
    
    print("🚀 Starting Real-Time Streaming Dashboard...")
    print("📡 Features:")
    print("   • Live sensor data simulation")
    print("   • Google Cloud Pub/Sub integration")
    print("   • AI agent real-time responses")
    print("   • Interactive Streamlit dashboard")
    print("   • Process variable monitoring")
    print("   • Equipment health tracking")
    print()
    
    try:
        # Launch the Streamlit dashboard
        print("🌐 Launching Streamlit dashboard...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/cement_ai_platform/streaming/realtime_dashboard.py",
            "--server.port", "8502",
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check Google Cloud credentials are set up")
        print("   3. Verify project ID in pubsub_simulator.py")

def launch_pubsub_simulator_only():
    """Launch just the Pub/Sub simulator for testing"""
    
    print("📡 Launching Pub/Sub Simulator Only...")
    
    try:
        from cement_ai_platform.streaming.pubsub_simulator import CementPlantPubSubSimulator, RealTimeDataProcessor
        
        # Initialize simulator
        simulator = CementPlantPubSubSimulator()
        processor = RealTimeDataProcessor()
        
        print("🚀 Starting streaming simulation...")
        print("Press Ctrl+C to stop")
        
        # Start streaming
        simulator.start_streaming_simulation(interval_seconds=2)
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            simulator.stop_streaming()
            print("\n⏹️ Streaming stopped")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--simulator-only":
        launch_pubsub_simulator_only()
    else:
        main()
