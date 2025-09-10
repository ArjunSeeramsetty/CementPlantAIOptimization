"""Main source module for cement plant digital twin."""

__version__ = "1.0.0"
__author__ = "Cement Plant AI Optimization Team"
__description__ = "Digital Twin POC for Cement Plant AI Optimization"

# Import main components
from .data_sourcing import download_all_datasets
from .simulation import generate_dcs_data, create_process_models
from .training import generate_massive_dataset

__all__ = [
    'download_all_datasets',
    'generate_dcs_data', 
    'create_process_models',
    'generate_massive_dataset'
]
