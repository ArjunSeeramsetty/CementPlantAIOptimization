# src/cement_ai_platform/config/secrets.py
from google.cloud import secretmanager
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SecretManager:
    """
    Secure secret management using Google Cloud Secret Manager.
    
    This class provides methods to retrieve secrets from Google Cloud Secret Manager
    with proper error handling and caching for production use.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize Secret Manager client.
        
        Args:
            project_id (str, optional): GCP project ID. If None, uses environment variable.
        """
        self.project_id = project_id or os.getenv('CEMENT_GCP_PROJECT', 'cement-ai-opt-38517')
        
        try:
            self.client = secretmanager.SecretManagerServiceClient()
            logger.info(f"Secret Manager client initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Secret Manager client: {e}")
            self.client = None
    
    def get_secret(self, secret_id: str, version: str = "latest") -> Optional[str]:
        """
        Retrieve a secret from Google Cloud Secret Manager.
        
        Args:
            secret_id (str): The ID of the secret to retrieve
            version (str): The version of the secret (default: "latest")
            
        Returns:
            str: The secret value, or None if retrieval failed
            
        Raises:
            Exception: If secret retrieval fails
        """
        if not self.client:
            logger.error("Secret Manager client not initialized")
            return None
        
        try:
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            logger.debug(f"Successfully retrieved secret: {secret_id}")
            return secret_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            return None
    
    def get_secret_with_fallback(self, secret_id: str, fallback_env_var: str, version: str = "latest") -> str:
        """
        Retrieve a secret with fallback to environment variable.
        
        Args:
            secret_id (str): The ID of the secret to retrieve
            fallback_env_var (str): Environment variable name to use as fallback
            version (str): The version of the secret (default: "latest")
            
        Returns:
            str: The secret value from Secret Manager or environment variable
        """
        secret_value = self.get_secret(secret_id, version)
        
        if secret_value is None:
            logger.warning(f"Secret {secret_id} not found, using environment variable {fallback_env_var}")
            secret_value = os.getenv(fallback_env_var)
            
            if secret_value is None:
                logger.error(f"Neither secret {secret_id} nor environment variable {fallback_env_var} found")
                raise ValueError(f"Secret {secret_id} and environment variable {fallback_env_var} both not found")
        
        return secret_value

# Global instance for easy access
_secret_manager = None

def get_secret_manager() -> SecretManager:
    """
    Get the global Secret Manager instance.
    
    Returns:
        SecretManager: Global Secret Manager instance
    """
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager

def get_secret(secret_id: str, version: str = "latest") -> Optional[str]:
    """
    Convenience function to retrieve a secret.
    
    Args:
        secret_id (str): The ID of the secret to retrieve
        version (str): The version of the secret (default: "latest")
        
    Returns:
        str: The secret value, or None if retrieval failed
    """
    return get_secret_manager().get_secret(secret_id, version)

def get_secret_with_fallback(secret_id: str, fallback_env_var: str, version: str = "latest") -> str:
    """
    Convenience function to retrieve a secret with fallback.
    
    Args:
        secret_id (str): The ID of the secret to retrieve
        fallback_env_var (str): Environment variable name to use as fallback
        version (str): The version of the secret (default: "latest")
        
    Returns:
        str: The secret value from Secret Manager or environment variable
    """
    return get_secret_manager().get_secret_with_fallback(secret_id, fallback_env_var, version)
