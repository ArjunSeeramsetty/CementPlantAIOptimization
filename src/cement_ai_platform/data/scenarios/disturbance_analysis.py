from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def summarize_scenarios(scenarios: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    base = scenarios["base"]
    intensity_summary: Dict[str, float] = {}
    for name, df in scenarios.items():
        if name == "base":
            continue
        total_dev = 0.0
        param_count = 0
        for param in ["kiln_temperature", "coal_feed_rate", "kiln_speed", "LSF", "SM", "AM"]:
            if param in df.columns:
                baseline = base[param].values
                current = df[param].values
                dev = float(np.mean(np.abs(current - baseline) / np.maximum(1e-6, np.abs(baseline))))
                total_dev += dev
                param_count += 1
        intensity_summary[name] = (total_dev / max(1, param_count)) * 100.0
    return intensity_summary


