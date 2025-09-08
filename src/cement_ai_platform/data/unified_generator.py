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


def create_unified_platform(seed: int = 42) -> UnifiedCementDataPlatform:
    return UnifiedCementDataPlatform(seed=seed)


