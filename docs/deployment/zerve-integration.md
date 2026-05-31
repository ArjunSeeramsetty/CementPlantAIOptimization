# Zerve AI Integration Guide

This guide details the integration of Zerve AI with the Cement Plant AI Optimization Platform. It describes the integration architecture, client implementation, usage examples, configuration, and troubleshooting steps.

---

## Overview & Architecture

### Setup Status
- **API Key**: Configured in `.env` (`ZERVE_API_KEY`)
- **Client Module**: [zerve_deployment.py](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/src/cement_ai_platform/integrations/zerve_deployment.py)
- **Reference Implementation**: [zerve_integration.py](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/src/cement_ai_platform/integrations/zerve_integration.py)
- **Configuration File**: [zerve_config.yml](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/config/zerve_config.yml)

### Deployment-Based Architecture
Zerve AI operates on a deployment-based architecture rather than a centralized endpoint:
1. **Zerve Canvas**: Workflows are visually constructed and tested on the [Zerve Canvas UI](https://app.zerve.ai).
2. **Deployments**: Workflows are manually deployed through the Canvas UI to obtain dedicated, custom endpoints.
3. **Execution**: Deployed endpoints are invoked via `https://<DeploymentName>.zerve.cloud/<route>` using a Bearer token (`Authorization: Bearer <API_KEY>`).

```
┌─────────────────────────────────────────────────────────────┐
│                    Zerve Canvas (Web UI)                     │
│                     app.zerve.ai                             │
│                                                              │
│  1. Build Workflows                                          │
│  2. Test & Debug                                             │
│  3. Deploy ──────────┐                                       │
└──────────────────────┼───────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Deployment Created         │
         │                              │
         │  https://YourName.zerve.cloud│
         └─────────────┬────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Call from Your Code        │
         │                              │
         │   Python, cURL, etc.         │
         └──────────────────────────────┘
```

---

## Quick Start (5 Minutes)

### Step 1: Test Your Setup
Run the pre-configured integration verification script to check your local environment settings:
```bash
python test_zerve_deployment.py
```

### Step 2: Build and Deploy a Workflow in Zerve Canvas
1. Log in to [Zerve Canvas](https://app.zerve.ai).
2. Create a workflow block (e.g. Python code block):
   ```python
   def hello(name):
       return {
           "message": f"Hello from {name}!",
           "status": "success"
       }
   ```
3. Deploy the workflow with a deployment name, e.g., `HelloWorld`.

### Step 3: Invoke the Deployment in Python
Use the client from our integration package:
```python
import os
from cement_ai_platform.integrations import ZerveDeployment

# Initialize client for your deployment
deployment = ZerveDeployment.create(
    "HelloWorld",
    os.getenv('ZERVE_API_KEY')
)

# Call the route
result = deployment.post("greet", {"name": "Cement Plant"})
print(result)
# Expected output: {'message': 'Hello from Cement Plant!', 'status': 'success'}
```

---

## Core Components Built

### 1. Zerve Deployment Client
Located in [zerve_deployment.py](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/src/cement_ai_platform/integrations/zerve_deployment.py). This class manages requests, headers, and timeouts.
```python
from cement_ai_platform.integrations import ZerveDeployment

# Initialize deployment client
deployment = ZerveDeployment.create("EnergyOptimizer", api_key)
result = deployment.post("optimize", {
    "temperature": 1420,
    "pressure": 3.8
})
```

### 2. Convenience Functions
```python
from cement_ai_platform.integrations import call_zerve_deployment

# Perform a quick invocation
result = call_zerve_deployment(
    "MaintenancePredictor",
    "predict",
    {"vibration": 0.8}
)
```

### 3. Pre-configured Deployments Helper
Class `CementPlantDeployments` is available to access pre-configured endpoints dynamically:
```python
from cement_ai_platform.integrations import CementPlantDeployments

deployments = CementPlantDeployments(api_key)

# Access pre-configured modules
energy_res = deployments.energy_optimizer.post("optimize", {...})
maint_res = deployments.maintenance_predictor.post("predict", {...})
quality_res = deployments.quality_predictor.post("predict", {...})
```

---

## Usage Examples

### 1. Multi-Objective Energy Optimization
Construct the optimization solver in Zerve Canvas, deploy it as `EnergyOptimizer`, and invoke it:
```python
from cement_ai_platform.integrations import ZerveDeployment
import os

deployment = ZerveDeployment.create("EnergyOptimizer", os.getenv('ZERVE_API_KEY'))
current_state = {
    'temperature': 1420,
    'fan_speed': 1200,
    'production_target': 1000
}

# Run optimization on Zerve Cloud
result = deployment.post("optimize", current_state)
print(f"Optimal temperature: {result['optimal_temperature']}°C")
print(f"Optimal fan speed: {result['optimal_fan_speed']} RPM")
print(f"Expected savings: {result['expected_savings']}%")
```

### 2. Predictive Maintenance Classifier
Deploy a trained XGBoost model wrapper inside Zerve as `MaintenancePredictor` and call it:
```python
from cement_ai_platform.integrations import call_zerve_deployment

sensors = {
    'temperature': 145,
    'vibration': 0.82,
    'pressure': 4.3,
    'load': 0.95
}

prediction = call_zerve_deployment("MaintenancePredictor", "predict", sensors)
print(f"Failure Probability: {prediction['failure_probability']:.1%}")
print(f"Risk Level: {prediction['risk_level']}")
print(f"Action: {prediction['recommended_action']}")
```

---

## Configuration & Environment Setup

### Environment Variables
Configure the following in your `.env` file:
```bash
ZERVE_API_KEY=atuPNm7J...  # Your bearer auth key
ZERVE_DOMAIN=zerve.cloud  # Custom base domain (optional)
ZERVE_TIMEOUT=30          # Request timeout in seconds (optional)
```

### Dynamic Configuration
Initialize configurations dynamically using the helper classes:
```python
import os
from cement_ai_platform.integrations import ZerveDeploymentConfig

config = ZerveDeploymentConfig(
    deployment_name="EnergyOptimizer",
    api_key=os.getenv('ZERVE_API_KEY'),
    base_domain="zerve.cloud",
    timeout=45
)
print(f"Target URL: {config.base_url}")
```

---

## Best Practices

### 1. Robust Error Handling
Always trap connection and timeout exceptions:
```python
import requests
import logging
from cement_ai_platform.integrations import ZerveDeployment

logger = logging.getLogger(__name__)

def safe_call_zerve(deployment_name, route, data):
    try:
        deployment = ZerveDeployment.create(deployment_name, os.getenv('ZERVE_API_KEY'))
        return deployment.post(route, data)
    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling {deployment_name}/{route}")
        return {'error': 'timeout', 'status': 'failed'}
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to {deployment_name}")
        return {'error': 'connection', 'status': 'failed'}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {'error': str(e), 'status': 'failed'}
```

### 2. Retries with Exponential Backoff
```python
import time
import logging
from requests.exceptions import RequestException
from cement_ai_platform.integrations import ZerveDeployment

logger = logging.getLogger(__name__)

def call_with_retry(deployment_name, route, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            deployment = ZerveDeployment.create(deployment_name, os.getenv('ZERVE_API_KEY'))
            return deployment.post(route, data)
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
            time.sleep(wait_time)
```

### 3. Monitoring & Deployment Health Checks
Perform health checks against all active deployments during platform startup:
```python
from cement_ai_platform.integrations import ZerveDeployment

def run_health_checks():
    deployments = ["EnergyOptimizer", "MaintenancePredictor", "QualityPredictor"]
    status = {}
    for name in deployments:
        deployment = ZerveDeployment.create(name, os.getenv('ZERVE_API_KEY'))
        status[name] = 'UP' if deployment.health_check() else 'DOWN'
    return status
```

---

## Troubleshooting

### Connection Timeout
- **Cause**: Heavy model inference, cold starts, or poor connectivity.
- **Solution**: Set a larger timeout limit during client initialization:
  ```python
  config = ZerveDeploymentConfig.from_env("EnergyOptimizer")
  config.timeout = 60 # set to 60 seconds
  ```

### 404 Not Found
- **Cause**: Incorrect deployment name spelling, or deployment does not exist in Zerve Canvas.
- **Solution**: Confirm deployment name matches exactly what is shown in the Canvas Web UI.

### 401 Unauthorized
- **Cause**: Missing or incorrect `ZERVE_API_KEY`.
- **Solution**: Confirm `ZERVE_API_KEY` is present in your environment or `.env` file, and is loaded correctly:
  ```bash
  python -c "import os; print(os.getenv('ZERVE_API_KEY')[:8] + '...')"
  ```

---

## API Reference

### ZerveDeployment
```python
class ZerveDeployment:
    """Client for calling Zerve deployed endpoints."""
    
    def __init__(self, config: ZerveDeploymentConfig): ...
    
    @classmethod
    def from_env(cls, deployment_name: str) -> 'ZerveDeployment': ...
    
    @classmethod
    def create(cls, deployment_name: str, api_key: str) -> 'ZerveDeployment': ...
    
    def post(self, route: str, data: Dict[str, Any]) -> Dict[str, Any]: ...
    
    def get(self, route: str, params: Optional[Dict] = None) -> Dict[str, Any]: ...
    
    def health_check(self) -> bool: ...
```

### ZerveDeploymentConfig
```python
@dataclass
class ZerveDeploymentConfig:
    """Configuration for Zerve deployment."""
    deployment_name: str
    api_key: str
    base_domain: str = "zerve.cloud"
    timeout: int = 30
    
    @property
    def base_url(self) -> str: ...
```

### CementPlantDeployments
```python
class CementPlantDeployments:
    """Pre-configured deployments for cement plant."""
    
    def __init__(self, api_key: str): ...
    
    @property
    def energy_optimizer(self) -> ZerveDeployment: ...
    
    @property
    def maintenance_predictor(self) -> ZerveDeployment: ...
    
    @property
    def quality_predictor(self) -> ZerveDeployment: ...
    
    def custom(self, deployment_name: str) -> ZerveDeployment: ...
```
