from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_sequences_from_cement_data(
    cement_dataset,
    key_parameters: List[str],
    sequence_length: int = 24,
    samples_per_day: int = 50,
) -> Tuple[np.ndarray, MinMaxScaler]:
    parameter_data = cement_dataset[key_parameters].copy()
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(parameter_data)

    sequences: List[List[np.ndarray]] = []
    for _ in range(samples_per_day):
        base_idx = np.random.randint(0, len(normalized))
        base_state = normalized[base_idx].copy()
        daily: List[np.ndarray] = []
        for hour in range(sequence_length):
            if "kiln_temperature" in key_parameters:
                i = key_parameters.index("kiln_temperature")
                base_state[i] = np.clip(base_state[i] + np.random.normal(0, 0.01), 0, 1)
            if "kiln_speed" in key_parameters:
                i = key_parameters.index("kiln_speed")
                base_state[i] = np.clip(base_state[i] + np.random.normal(0, 0.02), 0, 1)
            if "coal_feed_rate" in key_parameters and "kiln_temperature" in key_parameters:
                ci = key_parameters.index("coal_feed_rate")
                ti = key_parameters.index("kiln_temperature")
                adj = -0.1 * (base_state[ti] - 0.5) + np.random.normal(0, 0.015)
                base_state[ci] = np.clip(base_state[ci] + adj, 0, 1)
            if "draft_pressure" in key_parameters:
                i = key_parameters.index("draft_pressure")
                base_state[i] = np.clip(base_state[i] + np.random.normal(0, 0.02), 0, 1)
            for param in ["raw_mill_fineness", "cement_mill_fineness"]:
                if param in key_parameters:
                    i = key_parameters.index(param)
                    base_state[i] = np.clip(base_state[i] + np.random.normal(0, 0.005), 0, 1)
            hour_factor = np.sin(2 * np.pi * hour / 24) * 0.02
            base_state = np.clip(base_state + hour_factor * np.random.random(len(key_parameters)) * 0.5, 0, 1)
            daily.append(base_state.copy())
        sequences.append(daily)
    return np.asarray(sequences), scaler



