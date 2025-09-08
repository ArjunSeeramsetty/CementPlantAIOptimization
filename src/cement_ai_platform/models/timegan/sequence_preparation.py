from __future__ import annotations

from typing import List

import numpy as np


def prepare_sequences(df, feature_cols: List[str], seq_len: int = 24) -> np.ndarray:
    """Simple rolling window sequence prep using normalized columns if available."""
    values = df[feature_cols].to_numpy()
    seqs = []
    for i in range(len(values) - seq_len + 1):
        seqs.append(values[i : i + seq_len])
    return np.asarray(seqs)


