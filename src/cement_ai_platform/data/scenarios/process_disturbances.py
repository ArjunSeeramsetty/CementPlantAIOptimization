from __future__ import annotations

from typing import Dict

import numpy as np


def apply_basic_disturbances(base_params: Dict[str, float]) -> Dict[str, float]:
    out = dict(base_params)
    out["kiln_temperature"] += float(np.random.normal(0, 15))
    out["coal_feed_rate"] += float(np.random.normal(0, 150))
    out["draft_pressure"] += float(np.random.normal(0, 2))
    out["kiln_speed"] += float(np.random.normal(0, 0.2))
    return out


