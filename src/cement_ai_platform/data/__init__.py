"""Enhanced data pipelines for the Cement AI Platform.

Includes physics-based chemistry generation, disturbance scenarios,
TimeGAN augmentation, predictive models, and optimization data prep.
"""

from .unified_generator import UnifiedCementDataPlatform, create_unified_platform  # noqa: F401
from .data_pipeline.chemistry_data_generator import (  # noqa: F401
    EnhancedCementDataGenerator,
    CementChemistry,
    CementTimeGAN,
    OptimizationDataPrep,
    CementQualityPredictor,
    CementEnergyPredictor,
)
from .data_pipeline.synthetic_data_generator import (  # noqa: F401
    generate_synthetic_temperature_data,
    generate_synthetic_dataset,
)

__all__ = [
    "UnifiedCementDataPlatform",
    "create_unified_platform",
    "EnhancedCementDataGenerator",
    "CementChemistry",
    "CementTimeGAN",
    "OptimizationDataPrep",
    "CementQualityPredictor",
    "CementEnergyPredictor",
    # Deprecated
    "generate_synthetic_temperature_data",
    "generate_synthetic_dataset",
]

