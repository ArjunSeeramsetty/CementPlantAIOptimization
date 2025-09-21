#!/usr/bin/env python3
"""
Health check script for Cement Plant AI Digital Twin
Used by Docker and Cloud Run to verify application health
"""

import requests
import sys
import os
from datetime import datetime

def check_health():
    """Check if the Streamlit application is healthy"""
    try:
        # Get the port from environment variable
        port = os.environ.get('STREAMLIT_SERVER_PORT', '8080')
        url = f"http://localhost:{port}/_stcore/health"
        
        # Make request with timeout
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"[{datetime.now()}] ✅ Health check passed - Status: {response.status_code}")
            return True
        else:
            print(f"[{datetime.now()}] ❌ Health check failed - Status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now()}] ❌ Health check failed - Connection error: {e}")
        return False
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Health check failed - Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
