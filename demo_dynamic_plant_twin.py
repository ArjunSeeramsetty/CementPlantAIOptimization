#!/usr/bin/env python3
"""
Demo Script for Dynamic Plant Digital Twin
Perfect for live POC demonstrations
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the dynamic plant twin demo"""
    
    print("🏭 JK Cement Digital Twin Platform - Dynamic Plant Twin Demo")
    print("=" * 60)
    print()
    print("🚀 Dynamic Features:")
    print("   🔄 Real-time plant state simulation")
    print("   📊 Live KPIs with automatic updates")
    print("   🎯 Scenario injection for demos")
    print("   🤖 AI-powered recommendations")
    print("   📈 Interactive process trending")
    print("   🏭 Live plant schematic")
    print("   ⚙️ Equipment health monitoring")
    print()
    print("🎮 Demo Scenarios Available:")
    print("   🔥 High Temperature - Simulate kiln overheating")
    print("   ⚠️ Quality Issue - Simulate free lime elevation")
    print("   📳 Vibration Alert - Simulate equipment issues")
    print("   🌪️ Environmental Issue - Simulate high NOx")
    print("   🔄 Reset to Normal - Return to normal operation")
    print()
    print("📊 Live Metrics:")
    print("   🌡️ Kiln Temperature (Target: 1450°C)")
    print("   ✅ Free Lime (Target: <1.5%)")
    print("   📦 Production Rate (Current: ~165 t/h)")
    print("   ⚡ Energy Efficiency (Target: >90%)")
    print("   📳 Vibration (Normal: <6.0 mm/s)")
    print()
    
    # Check if we're in the right directory
    if not Path("src/cement_ai_platform/dashboard/dynamic_plant_twin.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected: CementPlantAIOptimization/")
        print(f"   Current: {os.getcwd()}")
        return 1
    
    print("🌐 Launching Dynamic Plant Twin Dashboard...")
    print("📱 Dashboard will open in your browser at: http://localhost:8501")
    print("🎛️ Use the sidebar controls to inject scenarios and adjust settings")
    print("🛑 Press Ctrl+C to stop the dashboard")
    print()
    print("-" * 60)
    
    try:
        # Launch Streamlit directly with the dynamic plant twin
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/cement_ai_platform/dashboard/dynamic_plant_twin.py",
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
