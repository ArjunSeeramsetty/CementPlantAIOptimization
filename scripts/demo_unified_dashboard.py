#!/usr/bin/env python3
"""
Quick Demo Script for JK Cement Digital Twin Platform POC
Shows how to launch the unified dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the unified dashboard demo"""
    
    print("🏭 JK Cement Digital Twin Platform - POC Demo")
    print("=" * 50)
    print()
    print("🚀 Available Modules:")
    print("   🔄 Real-Time Streaming - Live sensor data simulation")
    print("   🏭 Multi-Plant Support - Enterprise plant management")
    print("   📱 Mobile Dashboard - Mobile-optimized interface")
    print("   🔧 Predictive Maintenance - Failure prediction models")
    print("   🔬 Data Validation - Drift detection and quality")
    print("   ⚗️ DWSIM Integration - Physics-based simulation")
    print()
    print("📊 Features:")
    print("   ✅ Single navigation interface")
    print("   ✅ Real-time module status")
    print("   ✅ Responsive design")
    print("   ✅ Production-ready")
    print()
    
    # Check if we're in the right directory
    if not Path("src/cement_ai_platform/dashboard/unified_dashboard.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected: CementPlantAIOptimization/")
        print(f"   Current: {os.getcwd()}")
        return 1
    
    print("🌐 Launching Unified Dashboard...")
    print("📱 Dashboard will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")
    print()
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/cement_ai_platform/dashboard/unified_dashboard.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
