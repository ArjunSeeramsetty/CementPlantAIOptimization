from __future__ import annotations

import numpy as np


def sample_synthetic_sequences(n_samples: int, seq_len: int, n_seq: int) -> np.ndarray:
    out = []
    for _ in range(n_samples):
        seq = np.random.normal(size=(seq_len, n_seq)) * 0.1
        for t in range(1, seq_len):
            seq[t] = 0.7 * seq[t - 1] + 0.3 * seq[t]
        out.append(seq)
    return np.asarray(out)


