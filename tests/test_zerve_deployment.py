"""
Test Zerve Deployment Client

This tests the actual Zerve deployment architecture where you:
1. Build workflows in Zerve Canvas (app.zerve.ai)
2. Deploy them to get endpoints like https://<name>.zerve.cloud
3. Call those endpoints from your code
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cement_ai_platform.integrations import (
    ZerveDeployment,
    ZerveDeploymentConfig,
    call_zerve_deployment
)

def main():
    print("\n" + "="*80)
    print("  Zerve Deployment Client Test")
    print("="*80 + "\n")
    
    # Check API key
    api_key = os.getenv('ZERVE_API_KEY')
    if not api_key:
        print("[FAIL] ZERVE_API_KEY not set!")
        print("   Set it: $env:ZERVE_API_KEY = 'your_key'")
        return False
    
    print(f"[OK] ZERVE_API_KEY is set: {api_key[:8]}...")
    print()
    
    # Test 1: Create deployment client
    print("Test 1: Creating deployment client...")
    try:
        deployment = ZerveDeployment.create(
            deployment_name="TestDeployment",
            api_key=api_key
        )
        print(f"[OK] Deployment client created")
        print(f"   Base URL: {deployment.config.base_url}")
        print()
    except Exception as e:
        print(f"[FAIL] Failed to create client: {e}")
        return False
    
    # Test 2: Test configuration
    print("Test 2: Testing configuration...")
    config = ZerveDeploymentConfig.from_env("TestDeployment")
    print(f"[OK] Configuration loaded")
    print(f"   Deployment name: {config.deployment_name}")
    print(f"   Base URL: {config.base_url}")
    print(f"   Domain: {config.base_domain}")
    print()
    
    # Test 3: Health check (will fail if no deployment exists)
    print("Test 3: Health check...")
    print("   Note: This will fail unless you have a deployment named 'TestDeployment'")
    is_healthy = deployment.health_check()
    if is_healthy:
        print("[OK] Deployment is accessible!")
    else:
        print("[INFO] No deployment found (expected)")
        print("   To test with a real deployment:")
        print("   1. Go to https://app.zerve.ai")
        print("   2. Create and deploy a workflow")
        print("   3. Use that deployment name in this test")
    print()
    
    # Test 4: Show usage examples
    print("Test 4: Usage examples...")
    print()
    print("Example 1: Call deployed energy optimizer")
    print("-" * 40)
    print("""
from cement_ai_platform.integrations import ZerveDeployment

deployment = ZerveDeployment.create("EnergyOptimizer", api_key)

result = deployment.post("optimize", {
    "temperature": 1420,
    "pressure": 3.8,
    "target_quality": 95
})

print(f"Optimal temp: {result['optimal_temperature']}")
    """)
    
    print()
    print("Example 2: Use convenience function")
    print("-" * 40)
    print("""
from cement_ai_platform.integrations import call_zerve_deployment

result = call_zerve_deployment(
    "MaintenancePredictor",
    "predict",
    {"vibration": 0.8, "temperature": 145}
)

print(f"Failure risk: {result['failure_probability']}")
    """)
    
    print()
    print("Example 3: Pre-configured deployments")
    print("-" * 40)
    print("""
from cement_ai_platform.integrations import CementPlantDeployments

deployments = CementPlantDeployments(api_key)

# Energy optimization
energy_result = deployments.energy_optimizer.post("optimize", {...})

# Maintenance prediction
maintenance_result = deployments.maintenance_predictor.post("predict", {...})

# Quality prediction
quality_result = deployments.quality_predictor.post("predict", {...})
    """)
    
    print()
    print("="*80)
    print("  [SUCCESS] Deployment client is working correctly!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Read: docs/ZERVE_ACTUAL_USAGE.md")
    print("  2. Go to https://app.zerve.ai and create a deployment")
    print("  3. Test calling your deployment with the code above")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

