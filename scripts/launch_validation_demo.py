#!/usr/bin/env python3
"""
Data Validation Demo Launcher
Launches the Streamlit data validation and drift detection dashboard
"""

import os
import sys
import subprocess

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def launch_validation_dashboard():
    """Launches the Streamlit data validation dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'cement_ai_platform', 'validation', 'validation_dashboard.py')
    print(f"🧪 Launching Data Validation Dashboard: {dashboard_path}")
    try:
        subprocess.run(["streamlit", "run", dashboard_path], check=True)
    except FileNotFoundError:
        print("❌ Error: 'streamlit' command not found. Please ensure Streamlit is installed.")
    except Exception as e:
        print(f"❌ An error occurred while launching Streamlit: {e}")

if __name__ == "__main__":
    launch_validation_dashboard()
