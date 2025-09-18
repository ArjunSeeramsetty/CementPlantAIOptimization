#!/usr/bin/env python3
"""
Launcher script for the Unified Cement Plant POC Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the unified dashboard"""
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Path to the unified dashboard
    dashboard_path = script_dir / "src" / "cement_ai_platform" / "dashboard" / "unified_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at: {dashboard_path}")
        print("Please ensure you're running this from the project root directory.")
        return 1
    
    print("🚀 Launching JK Cement Digital Twin Platform - Unified POC Dashboard...")
    print("📊 Available Modules:")
    print("   🔄 Real-Time Streaming")
    print("   🎮 HIL Interface")
    print("   🏭 Multi-Plant Support")
    print("   📱 Mobile Dashboard")
    print("   🔧 Predictive Maintenance")
    print("   🔬 Data Validation")
    print("   ⚗️ DWSIM Integration")
    print("\n🌐 Dashboard will open in your default browser...")
    print("📱 Use Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
