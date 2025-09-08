from __future__ import annotations

import numpy as np


class CompleteCementTimeGAN:
    """Complete TimeGAN with statistical fallback suitable for POC environments.

    This class learns per-timestep feature statistics and temporal correlations
    and then samples sequences that preserve those relationships. It avoids heavy
    deep-learning dependencies while keeping the interface stable.
    """

    def __init__(self, use_statistical_fallback: bool = True):
        self.use_statistical_fallback = use_statistical_fallback
        self.is_fitted = False
        self.sequence_length = 0
        self.feature_dim = 0
        self.statistical_params = {}

    def prepare_sequences(self, df, feature_cols, seq_len: int = 24):
        values = df[feature_cols].to_numpy()
        seqs = []
        for i in range(len(values) - seq_len + 1):
            seqs.append(values[i : i + seq_len])
        return np.asarray(seqs)

    def fit(self, sequences: np.ndarray, epochs: int | None = None):  # epochs kept for API compat
        self.feature_dim = int(sequences.shape[2])
        self.sequence_length = int(sequences.shape[1])
        return self._fit_statistical(sequences)

    # --------------------------- statistical fallback ---------------------------
    def _fit_statistical(self, sequences: np.ndarray):
        n_sequences, seq_len, n_features = sequences.shape
        self.statistical_params = {
            "means": np.zeros((seq_len, n_features)),
            "stds": np.zeros((seq_len, n_features)),
            "correlations": np.zeros((seq_len, n_features, n_features)),
            "temporal_correlations": np.zeros((seq_len - 1, n_features)),
            "feature_ranges": np.zeros((n_features, 2)),
        }
        for t in range(seq_len):
            step = sequences[:, t, :]
            self.statistical_params["means"][t] = np.mean(step, axis=0)
            self.statistical_params["stds"][t] = np.std(step, axis=0)
            if n_sequences > 1:
                corr = np.corrcoef(step.T)
                corr = np.nan_to_num(corr, nan=0.0)
                self.statistical_params["correlations"][t] = corr
            if t < seq_len - 1:
                nxt = sequences[:, t + 1, :]
                for f in range(n_features):
                    c = np.corrcoef(step[:, f], nxt[:, f])[0, 1]
                    self.statistical_params["temporal_correlations"][t, f] = 0.0 if np.isnan(c) else c
        all_data = sequences.reshape(-1, n_features)
        self.statistical_params["feature_ranges"][:, 0] = np.min(all_data, axis=0)
        self.statistical_params["feature_ranges"][:, 1] = np.max(all_data, axis=0)
        self.is_fitted = True
        return self

    def generate_synthetic_sequences(self, n_sequences: int = 500) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generation")
        out = []
        for _ in range(n_sequences):
            out.append(self._generate_single_sequence())
        return np.asarray(out)

    def _generate_single_sequence(self) -> np.ndarray:
        seq = []
        means_t0 = self.statistical_params["means"][0]
        stds_t0 = self.statistical_params["stds"][0]
        corr_t0 = self._ensure_positive_definite(self.statistical_params["correlations"][0])
        cov_t0 = np.outer(stds_t0, stds_t0) * corr_t0
        first = np.random.multivariate_normal(means_t0, cov_t0)
        first = self._clip_to_ranges(first)
        seq.append(first)
        for t in range(1, self.sequence_length):
            means_t = self.statistical_params["means"][t]
            stds_t = self.statistical_params["stds"][t]
            prev = seq[-1]
            temp_corr = self.statistical_params["temporal_correlations"][t - 1]
            step = np.zeros(self.feature_dim)
            for f in range(self.feature_dim):
                temporal_influence = temp_corr[f] * (prev[f] - means_t[f])
                noise = np.random.normal(0.0, stds_t[f] * (1 - abs(temp_corr[f])))
                step[f] = means_t[f] + temporal_influence + noise
            seq.append(self._clip_to_ranges(step))
        return np.asarray(seq)

    @staticmethod
    def _ensure_positive_definite(matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
        vals, vecs = np.linalg.eigh(matrix)
        vals = np.maximum(vals, min_eigenvalue)
        return vecs @ np.diag(vals) @ vecs.T

    def _clip_to_ranges(self, values: np.ndarray) -> np.ndarray:
        for f in range(len(values)):
            lo, hi = self.statistical_params["feature_ranges"][f]
            values[f] = np.clip(values[f], lo, hi)
        return values


