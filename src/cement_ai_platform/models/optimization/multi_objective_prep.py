from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class OptimizationObjectives:
    """Container for objective values (for reference/reporting)."""

    energy_efficiency: float
    quality_score: float
    sustainability_score: float


@dataclass
class DecisionVariables:
    """Decision variables for cement kiln optimization."""

    fuel_flow_rate: float  # kg/h (2800-3600)
    kiln_speed: float  # rpm (2.5-4.2)
    feed_rate: float  # tonnes/h (80-120)
    oxygen_content: float  # % (2-6)
    alt_fuel_usage: float  # % (0-30)


class OptimizationDataPrep:
    """Multi-objective prep with objectives and constraints.

    Builds an optimization-ready dataset with sampled decision variables,
    objective values, and constraint slack values.
    """

    def __init__(self, cement_data: pd.DataFrame) -> None:
        self.data = cement_data.copy()
        self.n_samples = len(cement_data)

        # Decision variable bounds
        self.decision_bounds: Dict[str, tuple[float, float]] = {
            "fuel_flow_rate": (2800, 3600),
            "kiln_speed": (2.5, 4.2),
            "feed_rate": (80, 120),
            "oxygen_content": (2, 6),
            "alt_fuel_usage": (0, 30),
        }

        # Constraint thresholds
        self.constraints: Dict[str, float] = {
            "temperature_min": 1400,
            "temperature_max": 1480,
            "quality_min": 0.75,
            "free_lime_max": 3.0,
            "burnability_min": 40.0,
            "c3s_min": 50.0,
            "lsf_min": 0.88,
            "lsf_max": 1.02,
        }

    # ------------------------------ Objective functions ------------------------------
    def calculate_energy_efficiency_objective(self, dv: DecisionVariables, s: pd.Series) -> float:
        base_heat = float(s.get("heat_consumption", 800.0))
        fuel_efficiency = 1.0 - (dv.fuel_flow_rate - 3200.0) / 3200.0 * 0.15
        speed_efficiency = 1.0 + (dv.kiln_speed - 3.2) / 3.2 * 0.1
        alt_fuel_benefit = 1.0 - dv.alt_fuel_usage / 100.0 * 0.2
        energy = base_heat * fuel_efficiency * speed_efficiency * alt_fuel_benefit
        oxygen_factor = 1.0 - (dv.oxygen_content - 2.0) / 4.0 * 0.08
        return float(energy * oxygen_factor)

    def calculate_quality_objective(self, dv: DecisionVariables, s: pd.Series) -> float:
        base_quality = (float(s.get("C3S", 60.0)) / 100.0 * 0.6 + float(s.get("burnability_index", 50.0)) / 100.0 * 0.4)
        speed_factor = 1.0 - abs(dv.kiln_speed - 3.2) / 3.2 * 0.15
        oxygen_factor = 1.0 + (dv.oxygen_content - 2.0) / 4.0 * 0.12
        optimal_feed = 100.0
        feed_factor = 1.0 - abs(dv.feed_rate - optimal_feed) / optimal_feed * 0.1
        alt_fuel_penalty = 1.0 - dv.alt_fuel_usage / 100.0 * 0.05
        return float(base_quality * speed_factor * oxygen_factor * feed_factor * alt_fuel_penalty)

    def calculate_sustainability_objective(self, dv: DecisionVariables, s: pd.Series) -> float:
        base_emissions = dv.fuel_flow_rate / 1000.0
        alt_fuel_reduction = dv.alt_fuel_usage / 100.0 * 0.3
        emissions_factor = 1.0 - alt_fuel_reduction
        oxygen_tradeoff = 1.0 + (dv.oxygen_content - 2.0) / 4.0 * 0.05
        speed_factor = 1.0 + abs(dv.kiln_speed - 3.2) / 3.2 * 0.08
        return float(base_emissions * emissions_factor * oxygen_tradeoff * speed_factor)

    # ---------------------------------- Constraints ----------------------------------
    def temperature_constraint(self, dv: DecisionVariables, s: pd.Series) -> float:
        base_temp = float(s.get("kiln_temperature", 1450.0))
        fuel_temp = (dv.fuel_flow_rate - 3200.0) / 3200.0 * 30.0
        oxygen_temp = (dv.oxygen_content - 4.0) / 4.0 * 20.0
        speed_temp = -(dv.kiln_speed - 3.2) / 3.2 * 15.0
        predicted = base_temp + fuel_temp + oxygen_temp + speed_temp
        min_violation = predicted - self.constraints["temperature_min"]
        max_violation = self.constraints["temperature_max"] - predicted
        return float(min(min_violation, max_violation))

    def quality_constraint(self, dv: DecisionVariables, s: pd.Series) -> float:
        return float(self.calculate_quality_objective(dv, s) - self.constraints["quality_min"])

    def chemistry_constraints(self, dv: DecisionVariables, s: pd.Series) -> List[float]:
        lsf = float(s.get("LSF", 0.95))
        c3s = float(s.get("C3S", 60.0))
        free_lime = float(s.get("free_lime", 1.2))
        burnability = float(s.get("burnability_index", 50.0))
        return [
            lsf - self.constraints["lsf_min"],
            self.constraints["lsf_max"] - lsf,
            c3s - self.constraints["c3s_min"],
            self.constraints["free_lime_max"] - free_lime,
            burnability - self.constraints["burnability_min"],
        ]

    def operational_constraints(self, dv: DecisionVariables) -> List[float]:
        cons: List[float] = []
        for name, (lo, hi) in self.decision_bounds.items():
            val = getattr(dv, name)
            cons.append(val - lo)
            cons.append(hi - val)
        fuel_to_feed = dv.fuel_flow_rate / max(1.0, dv.feed_rate * 1000.0)
        cons.append(0.05 - abs(fuel_to_feed - 0.032))
        return cons

    # ---------------------------------- Dataset build ---------------------------------
    def create_optimization_dataset(self) -> Dict[str, pd.DataFrame | Dict[str, object]]:
        objectives_data: List[Dict[str, float]] = []
        constraints_data: List[Dict[str, object]] = []
        decision_vars_data: List[Dict[str, float]] = []

        for idx, row in self.data.iterrows():
            dv = DecisionVariables(
                fuel_flow_rate=float(np.random.uniform(*self.decision_bounds["fuel_flow_rate"])),
                kiln_speed=float(np.random.uniform(*self.decision_bounds["kiln_speed"])),
                feed_rate=float(np.random.uniform(*self.decision_bounds["feed_rate"])),
                oxygen_content=float(np.random.uniform(*self.decision_bounds["oxygen_content"])),
                alt_fuel_usage=float(np.random.uniform(*self.decision_bounds["alt_fuel_usage"])),
            )

            energy = self.calculate_energy_efficiency_objective(dv, row)
            quality = self.calculate_quality_objective(dv, row)
            sustain = self.calculate_sustainability_objective(dv, row)
            objectives_data.append(
                {
                    "energy_efficiency": energy,
                    "quality_score": quality,
                    "sustainability_score": sustain,
                    "sample_idx": idx,
                }
            )

            temp_c = self.temperature_constraint(dv, row)
            qual_c = self.quality_constraint(dv, row)
            chem_c = self.chemistry_constraints(dv, row)
            op_c = self.operational_constraints(dv)
            constraints_data.append(
                {
                    "temperature_constraint": temp_c,
                    "quality_constraint": qual_c,
                    "chemistry_constraints": chem_c,
                    "operational_constraints": op_c,
                    "sample_idx": idx,
                }
            )

            decision_vars_data.append(
                {
                    "fuel_flow_rate": dv.fuel_flow_rate,
                    "kiln_speed": dv.kiln_speed,
                    "feed_rate": dv.feed_rate,
                    "oxygen_content": dv.oxygen_content,
                    "alt_fuel_usage": dv.alt_fuel_usage,
                    "sample_idx": idx,
                }
            )

        objectives_df = pd.DataFrame(objectives_data)
        constraints_df = pd.DataFrame(constraints_data)
        decision_vars_df = pd.DataFrame(decision_vars_data)

        summary = {
            "objectives_stats": {
                k: {
                    "mean": float(objectives_df[k].mean()),
                    "std": float(objectives_df[k].std()),
                    "min": float(objectives_df[k].min()),
                    "max": float(objectives_df[k].max()),
                }
                for k in ("energy_efficiency", "quality_score", "sustainability_score")
            },
            "constraint_violations": {
                "temperature_violations": int((constraints_df["temperature_constraint"] < 0).sum()),
                "quality_violations": int((constraints_df["quality_constraint"] < 0).sum()),
            },
            "decision_variable_ranges": self.decision_bounds,
        }

        return {
            "objectives": objectives_df,
            "constraints": constraints_df,
            "decision_variables": decision_vars_df,
            "original_data": self.data,
            "summary": summary,
        }


