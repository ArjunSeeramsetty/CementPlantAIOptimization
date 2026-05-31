"""
Zerve AI Integration for Cement Plant AI Optimization Platform

This module provides seamless integration with Zerve AI's agentic workflows,
multi-agent orchestration, and fleet execution capabilities.

Documentation: https://www.zerve.ai/workflows
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ZerveConfig:
    """Zerve AI configuration."""
    api_endpoint: str
    api_key: str
    workspace_id: Optional[str] = None
    environment: str = "development"
    timeout: int = 300
    
    @classmethod
    def from_env(cls) -> 'ZerveConfig':
        """Load configuration from environment variables."""
        return cls(
            api_endpoint=os.getenv('ZERVE_API_ENDPOINT', 'https://api.zerve.ai'),
            api_key=os.getenv('ZERVE_API_KEY', ''),
            workspace_id=os.getenv('ZERVE_WORKSPACE_ID') or None,
            environment=os.getenv('ZERVE_ENVIRONMENT', 'development'),
            timeout=int(os.getenv('ZERVE_TIMEOUT', '300'))
        )
    
    @classmethod
    def from_yaml(cls, config_path: str = 'config/zerve_config.yml') -> 'ZerveConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        api_config = config['zerve']['api']
        workspace_config = config['zerve']['workspace']
        
        return cls(
            api_endpoint=os.path.expandvars(api_config['endpoint']),
            api_key=os.path.expandvars(api_config['api_key']),
            workspace_id=os.path.expandvars(workspace_config['workspace_id']),
            environment=os.path.expandvars(workspace_config['environment']),
            timeout=api_config['timeout']
        )


class ZerveClient:
    """
    Main client for interacting with Zerve AI API.
    
    Example:
        >>> client = ZerveClient.from_env()
        >>> workflow = client.create_workflow("Data Pipeline", "Build ETL for sensor data")
        >>> result = workflow.execute()
    """
    
    def __init__(self, config: ZerveConfig):
        self.config = config
        self.session = requests.Session()
        headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        }
        # Only add workspace header if provided
        if config.workspace_id:
            headers['X-Zerve-Workspace'] = config.workspace_id
        self.session.headers.update(headers)
        
    @classmethod
    def from_env(cls) -> 'ZerveClient':
        """Create client from environment variables."""
        config = ZerveConfig.from_env()
        return cls(config)
    
    @classmethod
    def from_yaml(cls, config_path: str = 'config/zerve_config.yml') -> 'ZerveClient':
        """Create client from YAML configuration."""
        config = ZerveConfig.from_yaml(config_path)
        return cls(config)
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request to Zerve."""
        url = f"{self.config.api_endpoint}/{endpoint}"
        
        try:
            response = self.session.request(
                method, url,
                timeout=self.config.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Zerve API request failed: {e}")
            raise
    
    def create_workflow(
        self,
        name: str,
        description: str,
        agent_type: str = "data_pipeline",
        compute_profile: str = "standard"
    ) -> 'ZerveWorkflow':
        """
        Create a new workflow with agentic capabilities.
        
        Args:
            name: Workflow name
            description: Natural language description of the task
            agent_type: Type of agent (data_pipeline, ml_training, optimization, etc.)
            compute_profile: Compute resources (lightweight, standard, gpu_training, fleet)
        
        Returns:
            ZerveWorkflow instance
        """
        payload = {
            'name': name,
            'description': description,
            'agent_type': agent_type,
            'compute_profile': compute_profile,
            'environment': self.config.environment
        }
        # Only add workspace_id if provided
        if self.config.workspace_id:
            payload['workspace_id'] = self.config.workspace_id
        
        response = self._request('POST', 'v1/workflows', json=payload)
        return ZerveWorkflow(self, response['workflow_id'], name, agent_type)
    
    def get_workflow(self, workflow_id: str) -> 'ZerveWorkflow':
        """Retrieve an existing workflow."""
        response = self._request('GET', f'v1/workflows/{workflow_id}')
        return ZerveWorkflow(
            self,
            response['workflow_id'],
            response['name'],
            response['agent_type']
        )
    
    def list_workflows(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all workflows in the workspace."""
        params = {'status': status} if status else {}
        response = self._request('GET', 'v1/workflows', params=params)
        return response['workflows']
    
    def create_agent(
        self,
        agent_type: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> 'ZerveAgent':
        """
        Create a standalone agent for a specific task.
        
        Args:
            agent_type: Type of agent
            task: Natural language task description
            context: Additional context for the agent
        
        Returns:
            ZerveAgent instance
        """
        payload = {
            'agent_type': agent_type,
            'task': task,
            'context': context or {}
        }
        # Only add workspace_id if provided
        if self.config.workspace_id:
            payload['workspace_id'] = self.config.workspace_id
        
        response = self._request('POST', 'v1/agents', json=payload)
        return ZerveAgent(self, response['agent_id'], agent_type, task)
    
    def create_fleet(
        self,
        name: str,
        num_workers: int,
        task_function: Optional[str] = None
    ) -> 'ZerveFleet':
        """
        Create a fleet for massively parallel execution.
        
        Args:
            name: Fleet name
            num_workers: Number of parallel workers
            task_function: Python function to execute (as string)
        
        Returns:
            ZerveFleet instance
        """
        payload = {
            'name': name,
            'num_workers': num_workers,
            'task_function': task_function
        }
        # Only add workspace_id if provided
        if self.config.workspace_id:
            payload['workspace_id'] = self.config.workspace_id
        
        response = self._request('POST', 'v1/fleet', json=payload)
        return ZerveFleet(self, response['fleet_id'], name, num_workers)


class ZerveWorkflow:
    """
    Represents a Zerve AI workflow with agentic capabilities.
    """
    
    def __init__(
        self,
        client: ZerveClient,
        workflow_id: str,
        name: str,
        agent_type: str
    ):
        self.client = client
        self.workflow_id = workflow_id
        self.name = name
        self.agent_type = agent_type
        self._state = {}
    
    def add_block(
        self,
        block_type: str,
        code: str,
        language: str = "python",
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Add a code block to the workflow."""
        payload = {
            'block_type': block_type,
            'code': code,
            'language': language,
            'dependencies': dependencies or []
        }
        
        response = self.client._request(
            'POST',
            f'v1/workflows/{self.workflow_id}/blocks',
            json=payload
        )
        return response['block_id']
    
    def execute(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        async_execution: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the workflow with optional inputs.
        
        Args:
            inputs: Input data for the workflow
            async_execution: Whether to execute asynchronously
        
        Returns:
            Workflow execution results
        """
        payload = {
            'inputs': inputs or {},
            'async': async_execution
        }
        
        response = self.client._request(
            'POST',
            f'v1/workflows/{self.workflow_id}/execute',
            json=payload
        )
        
        if async_execution:
            return {'execution_id': response['execution_id'], 'status': 'running'}
        else:
            return response['results']
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the workflow."""
        response = self.client._request(
            'GET',
            f'v1/workflows/{self.workflow_id}/state'
        )
        self._state = response['state']
        return self._state
    
    def inspect_block(self, block_id: str) -> Dict[str, Any]:
        """Inspect inputs/outputs of a specific block."""
        response = self.client._request(
            'GET',
            f'v1/workflows/{self.workflow_id}/blocks/{block_id}/inspect'
        )
        return response
    
    def save(self, path: Optional[str] = None) -> str:
        """Save workflow definition to file."""
        if path is None:
            path = f"workflows/{self.name.replace(' ', '_').lower()}.json"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        workflow_def = {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'agent_type': self.agent_type,
            'state': self._state,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(workflow_def, f, indent=2)
        
        return path
    
    def schedule(
        self,
        cron_expression: str,
        timezone: str = "UTC"
    ) -> str:
        """Schedule workflow execution."""
        payload = {
            'cron': cron_expression,
            'timezone': timezone
        }
        
        response = self.client._request(
            'POST',
            f'v1/workflows/{self.workflow_id}/schedule',
            json=payload
        )
        return response['schedule_id']


class ZerveAgent:
    """
    Represents a Zerve AI agent for task execution.
    """
    
    def __init__(
        self,
        client: ZerveClient,
        agent_id: str,
        agent_type: str,
        task: str
    ):
        self.client = client
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.task = task
    
    def execute(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the agent task."""
        payload = {
            'context': context or {}
        }
        
        response = self.client._request(
            'POST',
            f'v1/agents/{self.agent_id}/execute',
            json=payload
        )
        return response['result']
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent execution status."""
        response = self.client._request(
            'GET',
            f'v1/agents/{self.agent_id}/status'
        )
        return response
    
    def stop(self) -> bool:
        """Stop agent execution."""
        response = self.client._request(
            'POST',
            f'v1/agents/{self.agent_id}/stop'
        )
        return response['success']


class ZerveFleet:
    """
    Represents a Zerve Fleet for massively parallel execution.
    """
    
    def __init__(
        self,
        client: ZerveClient,
        fleet_id: str,
        name: str,
        num_workers: int
    ):
        self.client = client
        self.fleet_id = fleet_id
        self.name = name
        self.num_workers = num_workers
    
    def submit_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> str:
        """Submit tasks for parallel execution."""
        payload = {
            'tasks': tasks
        }
        
        response = self.client._request(
            'POST',
            f'v1/fleet/{self.fleet_id}/tasks',
            json=payload
        )
        return response['batch_id']
    
    def get_results(
        self,
        batch_id: str,
        wait: bool = False
    ) -> List[Dict[str, Any]]:
        """Get results from fleet execution."""
        params = {'wait': str(wait).lower()}
        
        response = self.client._request(
            'GET',
            f'v1/fleet/{self.fleet_id}/results/{batch_id}',
            params=params
        )
        return response['results']
    
    def scale(self, num_workers: int) -> bool:
        """Scale the fleet to a different number of workers."""
        payload = {
            'num_workers': num_workers
        }
        
        response = self.client._request(
            'PUT',
            f'v1/fleet/{self.fleet_id}/scale',
            json=payload
        )
        self.num_workers = num_workers
        return response['success']
    
    def shutdown(self) -> bool:
        """Shutdown the fleet."""
        response = self.client._request(
            'DELETE',
            f'v1/fleet/{self.fleet_id}'
        )
        return response['success']


# Convenience functions for common cement plant tasks

def create_data_pipeline_workflow(
    client: ZerveClient,
    source: str,
    destination: str,
    transformations: Optional[List[str]] = None
) -> ZerveWorkflow:
    """
    Create a data pipeline workflow for cement plant data.
    
    Args:
        client: ZerveClient instance
        source: Source data location (e.g., BigQuery table)
        destination: Destination for processed data
        transformations: List of transformation steps
    
    Returns:
        Configured ZerveWorkflow
    """
    task_description = f"""
    Build a data pipeline to:
    1. Extract data from {source}
    2. Apply transformations: {', '.join(transformations or ['standard cleaning'])}
    3. Load processed data to {destination}
    4. Validate data quality
    5. Generate pipeline metrics
    """
    
    workflow = client.create_workflow(
        name=f"Pipeline: {source} -> {destination}",
        description=task_description,
        agent_type="data_pipeline",
        compute_profile="standard"
    )
    
    return workflow


def create_ml_training_workflow(
    client: ZerveClient,
    model_type: str,
    training_data: str,
    target_metric: str
) -> ZerveWorkflow:
    """
    Create an ML training workflow for cement plant optimization.
    
    Args:
        client: ZerveClient instance
        model_type: Type of model (e.g., "pinn", "random_forest")
        training_data: Path to training data
        target_metric: Optimization target (e.g., "energy_efficiency")
    
    Returns:
        Configured ZerveWorkflow
    """
    task_description = f"""
    Train a {model_type} model for cement plant optimization:
    1. Load and prepare training data from {training_data}
    2. Perform feature engineering and selection
    3. Train {model_type} model optimized for {target_metric}
    4. Evaluate model performance with cross-validation
    5. Save model artifacts and metrics
    6. Generate model cards and documentation
    """
    
    workflow = client.create_workflow(
        name=f"ML Training: {model_type} for {target_metric}",
        description=task_description,
        agent_type="ml_training",
        compute_profile="gpu_training"
    )
    
    return workflow


def create_optimization_workflow(
    client: ZerveClient,
    optimization_targets: List[str],
    constraints: Dict[str, Any]
) -> ZerveWorkflow:
    """
    Create an optimization workflow for cement plant operations.
    
    Args:
        client: ZerveClient instance
        optimization_targets: List of targets (e.g., ["energy", "quality", "emissions"])
        constraints: Operational constraints
    
    Returns:
        Configured ZerveWorkflow
    """
    targets_str = ', '.join(optimization_targets)
    constraints_str = ', '.join([f"{k}: {v}" for k, v in constraints.items()])
    
    task_description = f"""
    Create multi-objective optimization for cement plant:
    1. Define optimization objectives: {targets_str}
    2. Apply operational constraints: {constraints_str}
    3. Build optimization model using current plant data
    4. Run optimization solver
    5. Generate actionable recommendations
    6. Validate recommendations against safety limits
    7. Create visualization dashboard
    """
    
    workflow = client.create_workflow(
        name=f"Optimization: {targets_str}",
        description=task_description,
        agent_type="optimization",
        compute_profile="standard"
    )
    
    return workflow

