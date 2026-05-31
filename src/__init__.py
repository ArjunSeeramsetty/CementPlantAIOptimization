"""Main source module for cement plant digital twin."""

__version__ = "1.0.0"
__author__ = "Cement Plant AI Optimization Team"
__description__ = "Digital Twin POC for Cement Plant AI Optimization"

# Import main components
from .cement_ai_platform.data_sourcing import download_all_datasets
from .cement_ai_platform.simulation import generate_dcs_data, create_process_models
from .cement_ai_platform.training import generate_massive_dataset

__all__ = [
    'download_all_datasets',
    'generate_dcs_data', 
    'create_process_models',
    'generate_massive_dataset'
]
