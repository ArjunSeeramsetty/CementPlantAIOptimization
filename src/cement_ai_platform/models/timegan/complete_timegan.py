from __future__ import annotations

import numpy as np


class CompleteCementTimeGAN:
    """Thin wrapper over CementTimeGAN from data pipeline module for cohesion."""

    def __init__(self, seq_len: int = 24, n_seq: int = 6, use_statistical_fallback: bool = True):
        from ...data.data_pipeline.chemistry_data_generator import CementTimeGAN

        self.impl = CementTimeGAN(seq_len=seq_len, n_seq=n_seq)
        self.use_statistical_fallback = use_statistical_fallback

    def prepare_sequences(self, df, feature_cols):
        return self.impl.prepare_sequences(df, feature_cols)

    def fit(self, sequences: np.ndarray, epochs: int = 500):
        self.impl.train(sequences, epochs=epochs)

    def generate_synthetic_sequences(self, n_samples: int) -> np.ndarray:
        return self.impl.sample(n_samples)


