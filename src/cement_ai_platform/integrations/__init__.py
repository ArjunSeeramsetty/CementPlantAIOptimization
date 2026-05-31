"""
Integrations module for external services and APIs.
"""

# Zerve deployment client (for calling deployed endpoints)
from .zerve_deployment import (
    ZerveDeployment,
    ZerveDeploymentConfig,
    CementPlantDeployments,
    call_zerve_deployment
)

# Zerve workflow/management client (reference implementation)
from .zerve_integration import (
    ZerveClient,
    ZerveWorkflow,
    ZerveAgent,
    ZerveFleet
)

__all__ = [
    # Deployment client (recommended for actual usage)
    'ZerveDeployment',
    'ZerveDeploymentConfig',
    'CementPlantDeployments',
    'call_zerve_deployment',
    # Workflow client (reference architecture)
    'ZerveClient',
    'ZerveWorkflow',
    'ZerveAgent',
    'ZerveFleet'
]

