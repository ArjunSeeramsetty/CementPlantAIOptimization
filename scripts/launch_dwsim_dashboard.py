#!/usr/bin/env python3
"""
Launch script for DWSIM Integration Dashboard
"""

import os
import sys
import subprocess

def launch_dwsim_dashboard():
    """Launch the DWSIM integration dashboard"""
    
    print("🚀 Launching DWSIM Integration Dashboard...")
    
    try:
        # Get the path to the dashboard
        dashboard_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'src', 
            'cement_ai_platform', 
            'dwsim', 
            'dwsim_dashboard.py'
        )
        
        # Launch with streamlit
        subprocess.run([
            "streamlit", 
            "run", 
            dashboard_path,
            "--server.port", "8504",
            "--server.headless", "false"
        ], check=True)
        
    except FileNotFoundError:
        print("❌ Error: 'streamlit' command not found. Please ensure Streamlit is installed.")
        print("Install with: pip install streamlit")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    launch_dwsim_dashboard()
