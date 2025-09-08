from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelMetrics:
    test_r2: float
    test_rmse: float
    cv_mean: float


class CementQualityPredictor:
    """Physics-informed multi-target cement quality predictor.

    Trains multiple algorithms per target (free_lime, c3s_content,
    compressive_strength), records performance, and exposes top features.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}

    # ----------------------------- target engineering -----------------------------
    def calculate_target_variables(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        targets: Dict[str, pd.Series] = {}

        # Free lime already present; clip to realistic range
        if "free_lime" in df:
            targets["free_lime"] = np.clip(df["free_lime"], 0.1, 5.0)

        # C3S content from Bogue; clip to realistic range
        if "C3S" in df:
            targets["c3s_content"] = np.clip(df["C3S"], 20, 80)

        # Physics-informed compressive strength (28-day MPa)
        if {"C3S", "C2S", "C3A", "cement_mill_fineness", "kiln_temperature", "free_lime", "LSF"}.issubset(
            df.columns
        ):
            c3s_contrib = df["C3S"] * 0.55
            c2s_contrib = df["C2S"] * 0.15
            c3a_contrib = df["C3A"] * 0.08
            base_strength = c3s_contrib + c2s_contrib + c3a_contrib

            fineness_normalized = (df["cement_mill_fineness"] - 280.0) / 140.0
            fineness_factor = 0.8 + 0.4 * np.clip(fineness_normalized, 0, 1)

            temp_optimal = 1450.0
            temp_factor = np.exp(-((df["kiln_temperature"] - temp_optimal) / 100.0) ** 2)
            temp_factor = 0.7 + 0.3 * temp_factor

            free_lime_penalty = 1.0 - df["free_lime"] * 0.05
            free_lime_penalty = np.clip(free_lime_penalty, 0.5, 1.0)

            lsf_optimal = 0.95
            lsf_factor = np.exp(-((df["LSF"] - lsf_optimal) / 0.1) ** 2)
            lsf_factor = 0.8 + 0.2 * lsf_factor

            comp_strength = base_strength * fineness_factor * temp_factor * free_lime_penalty * lsf_factor
            targets["compressive_strength"] = np.clip(comp_strength, 25.0, 70.0)

        return targets

    # -------------------------------- feature set ---------------------------------
    def _build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        preferred = [
            "kiln_temperature",
            "kiln_speed",
            "coal_feed_rate",
            "draft_pressure",
            "raw_mill_fineness",
            "cement_mill_fineness",
            "LSF",
            "SM",
            "AM",
            "C3S",
            "C2S",
            "C3A",
            "burnability_index",
            "heat_consumption",
        ]
        cols: List[str] = [c for c in preferred if c in df.columns]
        if not cols:
            raise ValueError("No expected feature columns found in dataframe")
        X = df[cols].copy()
        return X

    # --------------------------------- training -----------------------------------
    def train_models(self, df: pd.DataFrame, target_name: str) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame, pd.Series]:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.neural_network import MLPRegressor

        targets = self.calculate_target_variables(df)
        if target_name not in targets:
            raise ValueError(f"Target '{target_name}' cannot be computed from provided dataframe")

        X = self._build_feature_matrix(df)
        y = targets[target_name]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        candidates: Dict[str, Any] = {
            "random_forest": RandomForestRegressor(n_estimators=200, random_state=self.random_state),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=200, random_state=self.random_state),
            "neural_network": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, random_state=self.random_state),
            "linear_regression": LinearRegression(),
        }

        perf: Dict[str, Dict[str, float]] = {}
        best_name = None
        best_score = -1e9
        best_model = None

        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = float(r2_score(y_test, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                cv_scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=3)
                cv_mean = float(np.mean(cv_scores))
                perf[name] = {"test_r2": r2, "test_rmse": rmse, "cv_mean": cv_mean}
                if r2 > best_score:
                    best_score, best_name, best_model = r2, name, model
            except Exception:
                # Skip models that fail to converge in constrained environments
                continue

        if best_model is not None and best_name is not None:
            self.models[target_name] = best_model
            self.model_performance[target_name] = perf
            if hasattr(best_model, "feature_importances_"):
                importances = getattr(best_model, "feature_importances_")
                self.feature_importance[target_name] = pd.DataFrame(
                    {"feature": X.columns, "importance": importances}
                ).sort_values("importance", ascending=False)
        return perf, X_test, y_test

    def train_all_targets(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        available = self.calculate_target_variables(df)
        summary: Dict[str, Dict[str, float]] = {}
        for target_name in ["free_lime", "c3s_content", "compressive_strength"]:
            if target_name in available:
                perf, _, _ = self.train_models(df, target_name)
                # Store best score for quick summary
                if perf:
                    best = max(perf.values(), key=lambda m: m["test_r2"])  # type: ignore[index]
                    summary[target_name] = best
        return summary

    # Maintain POC compatibility
    def prepare(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        return self.train_all_targets(df)

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ...data.data_pipeline.chemistry_data_generator import CementQualityPredictor as _CQ


class CementQualityPredictor(_CQ):
    """Alias wrapper exposing the predictor under models.predictive."""

    pass


