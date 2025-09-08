from typing import Dict, Any

import numpy as np
import pandas as pd
import warnings


def generate_synthetic_temperature_data(num_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """DEPRECATED: use UnifiedCementDataPlatform.enhanced_generator instead."""
    warnings.warn(
        "synthetic_data_generator is DEPRECATED and provides only basic temperature data. "
        "Use cement_ai_platform.data.unified_generator.UnifiedCementDataPlatform for comprehensive generation.",
        DeprecationWarning,
        stacklevel=2,
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "process_temp": 85 + rng.normal(0, 3, num_rows),
        "curing_temp": 23 + rng.normal(0, 2, num_rows),
        "kiln_temp": 1450 + rng.normal(0, 50, num_rows),
    })
    outlier_indices = rng.choice(num_rows, size=max(1, num_rows // 50), replace=False)
    df.loc[outlier_indices, "process_temp"] += rng.choice([-25, 25], size=len(outlier_indices))
    return df


def generate_synthetic_dataset(num_rows: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """DEPRECATED: use UnifiedCementDataPlatform.generate_complete_poc_dataset()."""
    warnings.warn(
        "generate_synthetic_dataset is DEPRECATED. Use UnifiedCementDataPlatform for full generation.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {
        "temperatures": generate_synthetic_temperature_data(num_rows=num_rows, seed=seed),
    }



