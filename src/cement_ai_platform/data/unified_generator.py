from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np

from .data_pipeline.chemistry_data_generator import (
    CementEnergyPredictor,
    CementQualityPredictor,
    CementTimeGAN,
    EnhancedCementDataGenerator,
    OptimizationDataPrep,
)
from importlib import import_module
from types import ModuleType
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import r2_score, mean_squared_error  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore


class UnifiedCementDataPlatform:
    """Unified interface for cement plant data generation and prep."""

    def __init__(self, seed: int = 42) -> None:
        self.enhanced_generator = EnhancedCementDataGenerator(seed=seed)
        self.timegan = CementTimeGAN()
        self.quality_predictor = CementQualityPredictor()
        self.energy_predictor = CementEnergyPredictor()
        self.optimization_prep = None

    def generate_complete_poc_dataset(
        self,
        n_samples: int = 2500,
        include_timegan: bool = True,
        include_optimization: bool = True,
    ) -> Dict[str, Any]:
        base_data = self.enhanced_generator.generate_complete_dataset(n_samples)
        results: Dict[str, Any] = {"base_data": base_data}

        if include_timegan:
            try:
                features = [
                    "kiln_temperature",
                    "coal_feed_rate",
                    "kiln_speed",
                    "LSF",
                    "free_lime",
                    "heat_consumption",
                ]
                available = [f for f in features if f in base_data.columns]
                if len(available) >= 4:
                    seqs = self.timegan.prepare_sequences(base_data, available)
                    self.timegan.train(seqs, epochs=100)
                    results["timegan_synthetic"] = self.timegan.sample(500)
                else:
                    results["timegan_synthetic"] = None
            except Exception:
                results["timegan_synthetic"] = None

        try:
            results["model_performance"] = {
                "quality_models": self.quality_predictor.prepare(base_data),
                "energy_models": self.energy_predictor.train(base_data),
            }
        except Exception:
            results["model_performance"] = None

        if include_optimization:
            try:
                self.optimization_prep = OptimizationDataPrep(base_data)
                results["optimization_ready"] = self.optimization_prep.create_targets()
            except Exception:
                results["optimization_ready"] = base_data

        results["summary"] = self._summarize(results)
        return results

    def _summarize(self, results: Dict[str, Any]) -> Dict[str, Any]:
        base = results["base_data"]
        return {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(base),
            "features": len(base.columns),
            "capabilities": {
                "enhanced_chemistry": True,
                "timegan_augmentation": results.get("timegan_synthetic") is not None,
                "predictive_models": results.get("model_performance") is not None,
                "optimization_ready": results.get("optimization_ready") is not None,
            },
        }

    def quick_demo(self, n_samples: int = 500) -> Dict[str, Any]:
        return self.generate_complete_poc_dataset(
            n_samples=n_samples, include_timegan=False, include_optimization=True
        )

    # ---------------------------- extended integrations ----------------------------
    def _safe_import(self, module_path: str) -> ModuleType | None:
        try:
            return import_module(module_path)
        except Exception:
            return None

    def _safe_call(self, module: ModuleType | None, candidates: list[str], *args, **kwargs):
        if module is None:
            return None
        for name in candidates:
            func = getattr(module, name, None)
            if callable(func):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    continue
        return None

    def preprocess(
        self,
        data,
        handle_missing: bool = True,
        do_split: bool = False,
        synth: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Run preprocessing helpers if available (best-effort, non-breaking)."""
        artifacts: Dict[str, Any] = {}
        df = data.copy()

        if handle_missing:
            hm_mod = self._safe_import(
                "cement_ai_platform.data.data_pipeline.preprocessing.handle_missing"
            )
            out = self._safe_call(hm_mod, ["handle_missing_data", "handle_missing"], df)
            if out is not None:
                df = out
                artifacts["missing_handled"] = True
            else:
                artifacts["missing_handled"] = False

        if synth:
            gs_mod = self._safe_import(
                "cement_ai_platform.data.data_pipeline.preprocessing.generate_synthetic"
            )
            _ = self._safe_call(gs_mod, ["generate_synthetic", "generate"], df)
            artifacts["synthetic_generated"] = _ is not None

        split_info = None
        if do_split:
            tts_mod = self._safe_import(
                "cement_ai_platform.data.data_pipeline.preprocessing.train_test_split_"
            )
            out = self._safe_call(tts_mod, ["train_test_split_custom", "train_test_split"], df)
            if isinstance(out, dict) and {"train", "test"}.issubset(out.keys()):
                split_info = out
            else:
                X_train, X_test = train_test_split(df, test_size=test_size, random_state=random_state)
                split_info = {"train": X_train, "test": X_test}

        fs_mod = self._safe_import(
            "cement_ai_platform.data.data_pipeline.preprocessing.final_summary"
        )
        _ = self._safe_call(fs_mod, ["preprocess_final_summary", "final_summary"], df)
        artifacts["preprocess_summary"] = _

        return {"data": df, "split": split_info, "artifacts": artifacts}

    def validate(self, data) -> Dict[str, Any]:
        """Run validation checks if available (best-effort)."""
        reports: Dict[str, Any] = {}

        comp_mod = self._safe_import("cement_ai_platform.data.validation.completeness_check")
        reports["completeness"] = self._safe_call(
            comp_mod, ["run_completeness_check", "validate_completeness"], data
        )

        mec_mod = self._safe_import("cement_ai_platform.data.validation.mass_energy_constraints")
        reports["mass_energy"] = self._safe_call(
            mec_mod, ["check_mass_energy_constraints", "validate_constraints"], data
        )

        ea_mod = self._safe_import("cement_ai_platform.data.validation.energy_alignment")
        reports["energy_alignment"] = self._safe_call(
            ea_mod, ["validate_energy_alignment", "energy_alignment_report"], data
        )

        cr_mod = self._safe_import("cement_ai_platform.data.validation.comprehensive_report")
        reports["comprehensive_report"] = self._safe_call(
            cr_mod, ["generate_validation_report", "comprehensive_report"], data
        )

        tv_mod = self._safe_import("cement_ai_platform.data.validation.time_validation_summary")
        reports["time_validation"] = self._safe_call(
            tv_mod, ["time_validation_summary", "summarize_time_validation"], data
        )

        return {"reports": reports}

    def train_baselines(self, data, target: str) -> Dict[str, Any]:
        """Train simple baseline models (wrappers if present; fallback to sklearn)."""
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results: Dict[str, Any] = {}

        # Fallback baselines
        models = {
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "linear_regression": LinearRegression(),
        }
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    "r2": float(r2_score(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                }
            except Exception:
                results[name] = {"r2": None, "rmse": None}

        # Try wrapper evaluation hook
        eval_mod = self._safe_import(
            "cement_ai_platform.models.predictive.evaluation.cross_validation"
        )
        _ = self._safe_call(eval_mod, ["evaluate_models", "cross_validate"], models, X_train, y_train)
        if _ is not None:
            results["cross_validation"] = _
        return results

    def simulate_dwsim(self, data) -> Dict[str, Any]:
        """Run DWSIM simulation if wrappers are available (best-effort)."""
        out: Dict[str, Any] = {}
        fw = self._safe_import("cement_ai_platform.data.data_pipeline.dwsim.framework")
        hmb = self._safe_import("cement_ai_platform.data.data_pipeline.dwsim.heat_mass_balance")
        viz = self._safe_import("cement_ai_platform.data.data_pipeline.dwsim.visualization")
        out["framework"] = self._safe_call(fw, ["run_complete_simulation", "simulate"], data)
        out["heat_mass_balance"] = self._safe_call(hmb, ["compute_heat_mass_balance", "summary"], data)
        out["visualization"] = self._safe_call(viz, ["visualize_simulation", "plot"], data)
        return out

    def pinn_legacy_train_validate(self, data) -> Dict[str, Any]:
        """Invoke legacy PINN training/validation if present (best-effort)."""
        res: Dict[str, Any] = {}
        train_mod = self._safe_import("cement_ai_platform.models.pinn.pinn_train_legacy")
        valid_mod = self._safe_import("cement_ai_platform.models.pinn.pinn_validation_legacy")
        res["train"] = self._safe_call(train_mod, ["train_pinn", "train_model", "main"], data)
        res["validate"] = self._safe_call(valid_mod, ["validate_pinn", "validate", "main"], data)
        return res

    # ------------------------- real-data entrypoints -------------------------
    def load_real_data_global(self, *args, **kwargs):
        mod = self._safe_import("cement_ai_platform.data.data_pipeline.real_data.load_global_cement")
        return self._safe_call(mod, ["load_global_cement", "load_dataset"], *args, **kwargs)

    def load_real_data_kaggle(self, *args, **kwargs):
        mod = self._safe_import("cement_ai_platform.data.data_pipeline.real_data.load_kaggle")
        return self._safe_call(mod, ["load_kaggle_cement", "load_dataset"], *args, **kwargs)

    def integrate_real_datasets(self, *args, **kwargs):
        mod = self._safe_import("cement_ai_platform.data.data_pipeline.real_data.integrate_datasets")
        return self._safe_call(mod, ["integrate_datasets", "integrate"], *args, **kwargs)

    def analyze_real_datasets(self, *args, **kwargs):
        mod = self._safe_import("cement_ai_platform.data.data_pipeline.real_data.analyze_datasets")
        return self._safe_call(mod, ["analyze_datasets", "analyze"], *args, **kwargs)

    def final_quality_report(self, *args, **kwargs):
        mod = self._safe_import("cement_ai_platform.data.data_pipeline.real_data.final_quality_report")
        return self._safe_call(mod, ["generate_final_quality_report", "final_quality_report"], *args, **kwargs)

    # --------------------- optimization reporting outputs ---------------------
    def optimization_report(self, optimization_dataset_or_df) -> Dict[str, Any]:
        """Produce optimization reporting artifacts if wrappers exist.

        Accepts either the full dict returned by OptimizationDataPrep.create_optimization_dataset()
        or a dataframe of objectives.
        """
        final_mod = self._safe_import("cement_ai_platform.models.optimization.reporting.final_summary")
        opt_mod = self._safe_import("cement_ai_platform.models.optimization.reporting.optimization_summary")

        payload = optimization_dataset_or_df
        if isinstance(optimization_dataset_or_df, dict) and "objectives" in optimization_dataset_or_df:
            payload = optimization_dataset_or_df["objectives"]

        out: Dict[str, Any] = {}
        out["final_summary"] = self._safe_call(
            final_mod, ["generate_final_results_summary", "final_summary"], payload
        )
        out["optimization_summary"] = self._safe_call(
            opt_mod, ["generate_optimization_summary", "optimization_summary"], payload
        )
        return out


def create_unified_platform(seed: int = 42) -> UnifiedCementDataPlatform:
    return UnifiedCementDataPlatform(seed=seed)


