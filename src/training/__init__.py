"""Training module for AI model development."""

from .train_gan import (
    CementPlantDataGenerator,
    generate_massive_dataset
)

__all__ = [
    'CementPlantDataGenerator',
    'generate_massive_dataset'
]
