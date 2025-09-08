from __future__ import annotations

from typing import Any, Dict

from ..data.data_pipeline.chemistry_data_generator import EnhancedCementDataGenerator
from ..models.optimization.multi_objective_prep import OptimizationDataPrep
from ..models.timegan.complete_timegan import CompleteCementTimeGAN
from ..models.predictive import CementEnergyPredictor, CementQualityPredictor


class CementPlantPOCPipeline:
    """End-to-end pipeline integrating enhanced data, TimeGAN, predictors, and optimization prep."""

    def __init__(self) -> None:
        self.data_generator = EnhancedCementDataGenerator()
        self.timegan = CompleteCementTimeGAN()
        self.quality_predictor = CementQualityPredictor()
        self.energy_predictor = CementEnergyPredictor()

    def run_complete_poc(self, n_samples: int = 1000) -> Dict[str, Any]:
        base_data = self.data_generator.generate_complete_dataset(n_samples)

        feature_cols = [
            "kiln_temperature",
            "raw_mill_fineness",
            "cement_mill_fineness",
            "kiln_speed",
            "coal_feed_rate",
            "draft_pressure",
        ]
        sequences = self.timegan.prepare_sequences(base_data, feature_cols)
        self.timegan.fit(sequences, epochs=100)
        synthetic_sequences = self.timegan.generate_synthetic_sequences(200)

        opt_prep = OptimizationDataPrep(base_data)
        optimization_data = opt_prep.create_optimization_dataset()

        quality_results = self.quality_predictor.prepare(base_data)
        energy_results = self.energy_predictor.train(base_data)

        return {
            "summary": {
                "rows": len(base_data),
                "seq_shape": getattr(synthetic_sequences, "shape", None),
                "quality_models": quality_results,
                "energy_models": energy_results,
            }
        }


