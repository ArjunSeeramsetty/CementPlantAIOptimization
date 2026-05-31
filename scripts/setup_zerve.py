#!/usr/bin/env python3
"""
Zerve AI Setup and Configuration Script

This script helps you set up Zerve AI integration for the Cement Plant
AI Optimization Platform.

Usage:
    python scripts/setup_zerve.py
"""

import os
import sys
from pathlib import Path
import subprocess

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def check_python_version():
    """Check if Python version is 3.9 or higher."""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python 3.9+ required. Current version: {version.major}.{version.minor}")
        return False
    print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print_info("Checking dependencies...")
    
    required = ['requests', 'yaml', 'google.cloud.bigquery']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('.', '/'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print_warning(f"Missing dependencies: {', '.join(missing)}")
        response = input("Install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "pyyaml", "google-cloud-bigquery"])
                print_success("Dependencies installed successfully")
                return True
            except subprocess.CalledProcessError:
                print_error("Failed to install dependencies")
                return False
        return False
    
    print_success("All dependencies installed")
    return True

def check_zerve_credentials():
    """Check if Zerve credentials are configured."""
    print_info("Checking Zerve credentials...")
    
    api_key = os.getenv('ZERVE_API_KEY')
    workspace_id = os.getenv('ZERVE_WORKSPACE_ID')
    
    if not api_key or not workspace_id:
        print_warning("Zerve credentials not found in environment")
        return False
    
    # Mask API key for display
    masked_key = api_key[:8] + '*' * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else '*' * len(api_key)
    
    print_success(f"API Key: {masked_key}")
    print_success(f"Workspace ID: {workspace_id}")
    return True

def configure_zerve_credentials():
    """Interactively configure Zerve credentials."""
    print_header("Zerve AI Credential Configuration")
    
    print_info("Get your credentials from: https://app.zerve.ai/settings/api-keys")
    print()
    
    api_key = input("Enter your Zerve API Key: ").strip()
    workspace_id = input("Enter your Zerve Workspace ID: ").strip()
    environment = input("Environment (development/staging/production) [development]: ").strip() or "development"
    
    if not api_key or not workspace_id:
        print_error("API Key and Workspace ID are required")
        return False
    
    # Save to environment file
    env_file = Path('zerve_config.env')
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Update values
        content = content.replace('your_zerve_api_key_here', api_key)
        content = content.replace('your_workspace_id_here', workspace_id)
        content = content.replace('ZERVE_ENVIRONMENT=development', f'ZERVE_ENVIRONMENT={environment}')
        
        # Write back
        env_backup = env_file.with_suffix('.env.backup')
        if env_file.exists():
            import shutil
            shutil.copy(env_file, env_backup)
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print_success(f"Credentials saved to {env_file}")
        print_info(f"Backup created at {env_backup}")
        
        # Set in current environment
        os.environ['ZERVE_API_KEY'] = api_key
        os.environ['ZERVE_WORKSPACE_ID'] = workspace_id
        os.environ['ZERVE_ENVIRONMENT'] = environment
        
        return True
        
    except Exception as e:
        print_error(f"Failed to save credentials: {e}")
        return False

def test_zerve_connection():
    """Test connection to Zerve API."""
    print_info("Testing Zerve API connection...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        
        from cement_ai_platform.integrations import ZerveClient
        
        client = ZerveClient.from_env()
        
        # Try to list workflows (basic API call)
        workflows = client.list_workflows()
        
        print_success(f"Connected to Zerve API successfully")
        print_info(f"Found {len(workflows)} existing workflows")
        return True
        
    except ImportError as e:
        print_error(f"Failed to import Zerve integration: {e}")
        print_info("Make sure to install the package: pip install -e .")
        return False
    except Exception as e:
        print_error(f"Failed to connect to Zerve API: {e}")
        print_info("Check your API key and network connection")
        return False

def create_sample_workflow():
    """Create a sample workflow to test the integration."""
    print_info("Creating sample workflow...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from cement_ai_platform.integrations import ZerveClient
        
        client = ZerveClient.from_env()
        
        workflow = client.create_workflow(
            name="Test Workflow - Setup",
            description="Sample workflow created during Zerve setup",
            agent_type="data_pipeline",
            compute_profile="lightweight"
        )
        
        workflow.add_block(
            block_type="test",
            code="""
import pandas as pd
import datetime

print("✅ Zerve AI integration is working!")
print(f"Timestamp: {datetime.datetime.now()}")

# Simple test
df = pd.DataFrame({'test': [1, 2, 3]})
print(f"DataFrame created with {len(df)} rows")
            """,
            language="python"
        )
        
        print_success(f"Sample workflow created: {workflow.workflow_id}")
        print_info("Testing workflow execution...")
        
        result = workflow.execute()
        
        print_success("Sample workflow executed successfully!")
        print_info(f"Result: {result}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to create sample workflow: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete! 🎉")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print()
    print("1. 📖 Read the Quick Start Guide:")
    print("   → ZERVE_QUICK_START.md")
    print()
    print("2. 🚀 Run the examples:")
    print("   → python scripts/zerve_examples.py")
    print()
    print("3. 📚 Read the full integration guide:")
    print("   → docs/ZERVE_INTEGRATION_GUIDE.md")
    print()
    print("4. 🔧 Explore workflow templates:")
    print("   → workflows/cement_plant_workflows.json")
    print("   → workflows/agent_prompts.yml")
    print()
    print("5. 🌐 Visit Zerve Dashboard:")
    print("   → https://app.zerve.ai")
    print()
    print(f"{Colors.BOLD}Happy building with Zerve AI! 🚀{Colors.ENDC}")
    print()

def main():
    """Main setup flow."""
    print_header("Zerve AI Setup for Cement Plant AI Optimization")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print_error("Please install dependencies and run setup again")
        sys.exit(1)
    
    # Check if credentials are already configured
    if not check_zerve_credentials():
        print_info("Zerve credentials not configured")
        response = input("\nConfigure Zerve credentials now? (y/n): ")
        if response.lower() == 'y':
            if not configure_zerve_credentials():
                print_error("Credential configuration failed")
                sys.exit(1)
        else:
            print_warning("Skipping credential configuration")
            print_info("You can configure credentials later by:")
            print_info("1. Editing zerve_config.env file")
            print_info("2. Running this script again")
            sys.exit(0)
    
    # Test connection
    if not test_zerve_connection():
        print_error("Connection test failed")
        print_info("Please check your credentials and try again")
        sys.exit(1)
    
    # Create sample workflow
    response = input("\nCreate a sample workflow to test the integration? (y/n): ")
    if response.lower() == 'y':
        create_sample_workflow()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()

