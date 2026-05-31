"""
Zerve AI Deployment Client

This module provides utilities for calling Zerve deployed endpoints.

Zerve uses a deployment-based architecture where each workflow is deployed
to its own endpoint: https://<DeploymentName>.zerve.cloud/

Documentation: https://docs.zerve.ai
"""

import os
import requests
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ZerveDeploymentConfig:
    """Configuration for Zerve deployment."""
    deployment_name: str
    api_key: str
    base_domain: str = "zerve.cloud"
    timeout: int = 30
    
    @property
    def base_url(self) -> str:
        """Get the base URL for this deployment."""
        return f"https://{self.deployment_name}.{self.base_domain}"
    
    @classmethod
    def from_env(cls, deployment_name: str) -> 'ZerveDeploymentConfig':
        """Create config from environment variables."""
        return cls(
            deployment_name=deployment_name,
            api_key=os.getenv('ZERVE_API_KEY', ''),
            base_domain=os.getenv('ZERVE_DOMAIN', 'zerve.cloud'),
            timeout=int(os.getenv('ZERVE_TIMEOUT', '30'))
        )


class ZerveDeployment:
    """
    Client for calling Zerve deployed endpoints.
    
    Zerve deployments are accessed via:
    https://<DeploymentName>.zerve.cloud/<route>
    
    Example:
        >>> # First, deploy a workflow in Zerve Canvas (app.zerve.ai)
        >>> # Then call it from Python:
        >>> deployment = ZerveDeployment("EnergyOptimizer", api_key="your_key")
        >>> result = deployment.post("optimize", {
        ...     "temperature": 1420,
        ...     "pressure": 3.8
        ... })
    """
    
    def __init__(self, config: ZerveDeploymentConfig):
        """
        Initialize Zerve deployment client.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    @classmethod
    def from_env(cls, deployment_name: str) -> 'ZerveDeployment':
        """
        Create client from environment variables.
        
        Args:
            deployment_name: Name of your Zerve deployment
        
        Returns:
            ZerveDeployment instance
        """
        config = ZerveDeploymentConfig.from_env(deployment_name)
        return cls(config)
    
    @classmethod
    def create(cls, deployment_name: str, api_key: str) -> 'ZerveDeployment':
        """
        Create client with deployment name and API key.
        
        Args:
            deployment_name: Name of your Zerve deployment
            api_key: Your Zerve API key
        
        Returns:
            ZerveDeployment instance
        """
        config = ZerveDeploymentConfig(
            deployment_name=deployment_name,
            api_key=api_key
        )
        return cls(config)
    
    def post(self, route: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make POST request to deployment endpoint.
        
        Args:
            route: API route (e.g., "optimize", "predict")
            data: Request payload
        
        Returns:
            Response data as dictionary
        
        Example:
            >>> result = deployment.post("predict", {"temperature": 1420})
        """
        url = f"{self.config.base_url}/{route}"
        
        try:
            logger.debug(f"POST {url}")
            response = self.session.post(
                url,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed: {e}")
            raise
    
    def get(self, route: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make GET request to deployment endpoint.
        
        Args:
            route: API route (e.g., "status", "health")
            params: Query parameters
        
        Returns:
            Response data as dictionary
        
        Example:
            >>> status = deployment.get("status")
        """
        url = f"{self.config.base_url}/{route}"
        
        try:
            logger.debug(f"GET {url}")
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if deployment is healthy.
        
        Returns:
            True if deployment is accessible, False otherwise
        """
        try:
            # Try common health check routes
            for route in ['health', 'status', '']:
                try:
                    url = f"{self.config.base_url}/{route}" if route else self.config.base_url
                    response = self.session.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Deployment {self.config.deployment_name} is healthy")
                        return True
                except:
                    continue
            
            logger.warning(f"Deployment {self.config.deployment_name} health check failed")
            return False
        
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False


# Pre-configured deployments for cement plant operations

class CementPlantDeployments:
    """
    Pre-configured Zerve deployments for cement plant operations.
    
    Usage:
        >>> api_key = os.getenv('ZERVE_API_KEY')
        >>> deployments = CementPlantDeployments(api_key)
        >>> 
        >>> # Optimize energy
        >>> result = deployments.energy_optimizer.post("optimize", {
        ...     "temperature": 1420,
        ...     "fan_speed": 1200
        ... })
        >>> 
        >>> # Predict maintenance
        >>> prediction = deployments.maintenance_predictor.post("predict", {
        ...     "vibration": 0.8,
        ...     "temperature": 145
        ... })
    """
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
    
    @property
    def energy_optimizer(self) -> ZerveDeployment:
        """Energy optimization deployment."""
        return ZerveDeployment.create("EnergyOptimizer", self.api_key)
    
    @property
    def maintenance_predictor(self) -> ZerveDeployment:
        """Predictive maintenance deployment."""
        return ZerveDeployment.create("MaintenancePredictor", self.api_key)
    
    @property
    def quality_predictor(self) -> ZerveDeployment:
        """Clinker quality prediction deployment."""
        return ZerveDeployment.create("QualityPredictor", self.api_key)
    
    @property
    def process_optimizer(self) -> ZerveDeployment:
        """Process optimization deployment."""
        return ZerveDeployment.create("ProcessOptimizer", self.api_key)
    
    def custom(self, deployment_name: str) -> ZerveDeployment:
        """
        Create client for custom deployment.
        
        Args:
            deployment_name: Your deployment name
        
        Returns:
            ZerveDeployment instance
        """
        return ZerveDeployment.create(deployment_name, self.api_key)


# Convenience function

def call_zerve_deployment(
    deployment_name: str,
    route: str,
    data: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to call a Zerve deployment.
    
    Args:
        deployment_name: Name of deployment
        route: API route
        data: Request data
        api_key: API key (uses ZERVE_API_KEY env var if not provided)
    
    Returns:
        Response data
    
    Example:
        >>> result = call_zerve_deployment(
        ...     "EnergyOptimizer",
        ...     "optimize",
        ...     {"temperature": 1420}
        ... )
    """
    if api_key is None:
        api_key = os.getenv('ZERVE_API_KEY', '')
    
    deployment = ZerveDeployment.create(deployment_name, api_key)
    return deployment.post(route, data)

