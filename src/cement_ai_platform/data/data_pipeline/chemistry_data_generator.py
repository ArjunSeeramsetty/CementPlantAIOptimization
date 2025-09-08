from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


class CementChemistry:
    """Core cement chemistry calculations including Bogue equations and LSF."""

    def __init__(self) -> None:
        self.molecular_weights = {
            "CaO": 56.08,
            "SiO2": 60.08,
            "Al2O3": 101.96,
            "Fe2O3": 159.69,
            "MgO": 40.30,
            "SO3": 80.06,
            "K2O": 94.20,
            "Na2O": 61.98,
            "TiO2": 79.87,
            "P2O5": 141.94,
            "Mn2O3": 157.87,
            "Cr2O3": 151.99,
        }

        self.bogue_mw = {"C3S": 228.32, "C2S": 172.24, "C3A": 270.20, "C4AF": 485.96}

    @staticmethod
    def calculate_lsf(cao: float, sio2: float, al2o3: float, fe2o3: float) -> float:
        denom = 2.8 * float(sio2) + 1.2 * float(al2o3) + 0.65 * float(fe2o3)
        if denom == 0:
            return 0.0
        return float(cao) / denom

    @staticmethod
    def calculate_silica_modulus(sio2: float, al2o3: float, fe2o3: float) -> float:
        denom = float(al2o3) + float(fe2o3)
        if denom == 0:
            return 0.0
        return float(sio2) / denom

    @staticmethod
    def calculate_alumina_modulus(al2o3: float, fe2o3: float) -> float:
        if float(fe2o3) == 0:
            return 0.0
        return float(al2o3) / float(fe2o3)

    @staticmethod
    def calculate_bogue_compounds(cao: float, sio2: float, al2o3: float, fe2o3: float) -> Dict[str, float]:
        # Simplified Bogue equations
        c4af = 3.043 * float(fe2o3)
        c3a = 2.650 * float(al2o3) - 1.692 * float(fe2o3)
        c2s = 2.867 * float(sio2) - 0.7544 * float(cao)
        c3s = 4.071 * float(cao) - 7.600 * float(sio2) - 6.718 * float(al2o3) - 1.430 * float(fe2o3)
        return {"C3S": max(0.0, c3s), "C2S": max(0.0, c2s), "C3A": max(0.0, c3a), "C4AF": max(0.0, c4af)}

    def validate_chemistry(self, composition: Dict[str, float]) -> Dict[str, bool]:
        validation: Dict[str, bool] = {}
        validation["cao_range"] = 60.0 <= composition.get("CaO", 0.0) <= 67.0
        validation["sio2_range"] = 18.0 <= composition.get("SiO2", 0.0) <= 25.0
        validation["al2o3_range"] = 3.0 <= composition.get("Al2O3", 0.0) <= 8.0
        validation["fe2o3_range"] = 1.5 <= composition.get("Fe2O3", 0.0) <= 5.0

        cao, sio2, al2o3, fe2o3 = [composition.get(k, 0.0) for k in ("CaO", "SiO2", "Al2O3", "Fe2O3")]
        lsf = self.calculate_lsf(cao, sio2, al2o3, fe2o3)
        sm = self.calculate_silica_modulus(sio2, al2o3, fe2o3)
        am = self.calculate_alumina_modulus(al2o3, fe2o3)
        validation["lsf_range"] = 0.85 <= lsf <= 1.02
        validation["sm_range"] = 2.0 <= sm <= 3.5
        validation["am_range"] = 1.0 <= am <= 4.0
        major_total = sum(composition.get(k, 0.0) for k in ("CaO", "SiO2", "Al2O3", "Fe2O3"))
        validation["major_total"] = 85.0 <= major_total <= 95.0
        return validation


class EnhancedCementDataGenerator:
    """Enhanced data generator with chemistry relationships and process disturbances."""

    def __init__(self, seed: int = 42) -> None:
        np.random.seed(seed)
        self.chemistry = CementChemistry()
        self.process_ranges = {
            "kiln_temperature": {"min": 1400, "max": 1480, "target": 1450, "std": 15},
            "kiln_speed": {"min": 2.5, "max": 4.2, "target": 3.2, "std": 0.3},
            "coal_feed_rate": {"min": 2800, "max": 3600, "target": 3200, "std": 150},
            "draft_pressure": {"min": -15, "max": -8, "target": -12, "std": 2},
            "raw_mill_fineness": {"min": 8, "max": 15, "target": 12, "std": 1.5},
            "cement_mill_fineness": {"min": 280, "max": 420, "target": 350, "std": 25},
        }
        self.raw_material_ranges = {
            "limestone_cao": {"min": 48, "max": 54, "std": 1.5},
            "limestone_sio2": {"min": 2, "max": 8, "std": 1.2},
            "clay_sio2": {"min": 45, "max": 65, "std": 4},
            "clay_al2o3": {"min": 12, "max": 22, "std": 2.5},
            "iron_ore_fe2o3": {"min": 60, "max": 85, "std": 5},
            "sand_sio2": {"min": 85, "max": 95, "std": 2},
        }

    def generate_raw_material_composition(self, n_samples: int) -> pd.DataFrame:
        compositions: List[Dict[str, float]] = []
        for _ in range(n_samples):
            limestone_cao = float(np.random.normal(51, self.raw_material_ranges["limestone_cao"]["std"]))
            limestone_sio2 = float(np.random.normal(5, self.raw_material_ranges["limestone_sio2"]["std"]))
            clay_sio2 = float(np.random.normal(55, self.raw_material_ranges["clay_sio2"]["std"]))
            clay_al2o3_base = float(np.random.normal(17, self.raw_material_ranges["clay_al2o3"]["std"]))
            clay_al2o3 = float(clay_al2o3_base * (1 - 0.3 * (clay_sio2 - 55) / 55))
            iron_ore_fe2o3 = float(np.random.normal(72, self.raw_material_ranges["iron_ore_fe2o3"]["std"]))
            sand_sio2 = float(np.random.normal(90, self.raw_material_ranges["sand_sio2"]["std"]))
            compositions.append(
                {
                    "limestone_CaO": limestone_cao,
                    "limestone_SiO2": limestone_sio2,
                    "clay_SiO2": clay_sio2,
                    "clay_Al2O3": clay_al2o3,
                    "iron_ore_Fe2O3": iron_ore_fe2o3,
                    "sand_SiO2": sand_sio2,
                }
            )
        return pd.DataFrame(compositions)

    @staticmethod
    def calculate_raw_mix_proportions(target_lsf: float = 0.95, target_sm: float = 2.5, target_am: float = 2.0) -> Dict[str, float]:
        limestone_ratio = 0.78 + 0.05 * (target_lsf - 0.95)
        clay_ratio = 0.15 + 0.03 * (target_sm - 2.5)
        iron_ore_ratio = 0.04 + 0.02 * (target_am - 2.0)
        sand_ratio = 0.03
        total = limestone_ratio + clay_ratio + iron_ore_ratio + sand_ratio
        return {
            "limestone_ratio": limestone_ratio / total,
            "clay_ratio": clay_ratio / total,
            "iron_ore_ratio": iron_ore_ratio / total,
            "sand_ratio": sand_ratio / total,
        }

    def apply_process_disturbances(self, base_params: Dict[str, float]) -> Dict[str, float]:
        disturbed = dict(base_params)
        temp_dist = float(np.random.normal(0, self.process_ranges["kiln_temperature"]["std"]))
        disturbed["kiln_temperature"] += temp_dist
        disturbed["coal_feed_rate"] += float(-0.5 * temp_dist + np.random.normal(0, 100))
        draft_dist = float(np.random.normal(0, self.process_ranges["draft_pressure"]["std"]))
        disturbed["draft_pressure"] += draft_dist
        disturbed["kiln_speed"] += float(-0.02 * draft_dist + np.random.normal(0, 0.2))
        return disturbed

    def generate_complete_dataset(self, n_samples: int = 2500) -> pd.DataFrame:
        raw_materials = self.generate_raw_material_composition(n_samples)
        rows: List[Dict[str, Any]] = []
        for i in range(n_samples):
            rm = raw_materials.iloc[i]
            target_lsf = float(np.random.normal(0.95, 0.03))
            target_sm = float(np.random.normal(2.5, 0.2))
            target_am = float(np.random.normal(2.0, 0.15))
            mix = self.calculate_raw_mix_proportions(target_lsf, target_sm, target_am)

            raw_meal_cao = float(rm["limestone_CaO"] * mix["limestone_ratio"] + 0.5 * mix["clay_ratio"])  # clay CaO approx
            raw_meal_sio2 = float(rm["limestone_SiO2"] * mix["limestone_ratio"] + rm["clay_SiO2"] * mix["clay_ratio"] + rm["sand_SiO2"] * mix["sand_ratio"])
            raw_meal_al2o3 = float(rm["clay_Al2O3"] * mix["clay_ratio"])
            raw_meal_fe2o3 = float(rm["iron_ore_Fe2O3"] * mix["iron_ore_ratio"])
            total = raw_meal_cao + raw_meal_sio2 + raw_meal_al2o3 + raw_meal_fe2o3
            cao_pct = 100 * raw_meal_cao / total
            sio2_pct = 100 * raw_meal_sio2 / total
            al2o3_pct = 100 * raw_meal_al2o3 / total
            fe2o3_pct = 100 * raw_meal_fe2o3 / total

            lsf = self.chemistry.calculate_lsf(cao_pct, sio2_pct, al2o3_pct, fe2o3_pct)
            sm = self.chemistry.calculate_silica_modulus(sio2_pct, al2o3_pct, fe2o3_pct)
            am = self.chemistry.calculate_alumina_modulus(al2o3_pct, fe2o3_pct)
            bogue = self.chemistry.calculate_bogue_compounds(cao_pct, sio2_pct, al2o3_pct, fe2o3_pct)

            base_params = {
                "kiln_temperature": self.process_ranges["kiln_temperature"]["target"],
                "kiln_speed": self.process_ranges["kiln_speed"]["target"],
                "coal_feed_rate": self.process_ranges["coal_feed_rate"]["target"],
                "draft_pressure": self.process_ranges["draft_pressure"]["target"],
                "raw_mill_fineness": self.process_ranges["raw_mill_fineness"]["target"],
                "cement_mill_fineness": self.process_ranges["cement_mill_fineness"]["target"],
            }
            proc = self.apply_process_disturbances(base_params)

            burnability_base = 50 + 30 * (lsf - 0.9) + 2 * (proc["kiln_temperature"] - 1450)
            burnability_index = max(0.0, min(100.0, float(burnability_base + np.random.normal(0, 5))))
            heat_consumption_base = 750 + 50 * (lsf - 0.95) + 0.1 * (proc["kiln_temperature"] - 1450)
            heat_consumption = max(700.0, float(heat_consumption_base + np.random.normal(0, 20)))
            free_lime_base = 2.5 - 0.05 * (proc["kiln_temperature"] - 1450) - 0.02 * burnability_index
            free_lime = max(0.1, float(free_lime_base + np.random.normal(0, 0.3)))

            row: Dict[str, Any] = {
                "limestone_CaO": float(rm["limestone_CaO"]),
                "limestone_SiO2": float(rm["limestone_SiO2"]),
                "clay_SiO2": float(rm["clay_SiO2"]),
                "clay_Al2O3": float(rm["clay_Al2O3"]),
                "iron_ore_Fe2O3": float(rm["iron_ore_Fe2O3"]),
                "sand_SiO2": float(rm["sand_SiO2"]),
                "limestone_ratio": float(mix["limestone_ratio"]),
                "clay_ratio": float(mix["clay_ratio"]),
                "iron_ore_ratio": float(mix["iron_ore_ratio"]),
                "sand_ratio": float(mix["sand_ratio"]),
                "CaO": float(cao_pct),
                "SiO2": float(sio2_pct),
                "Al2O3": float(al2o3_pct),
                "Fe2O3": float(fe2o3_pct),
                "LSF": float(lsf),
                "SM": float(sm),
                "AM": float(am),
                "C3S": float(bogue["C3S"]),
                "C2S": float(bogue["C2S"]),
                "C3A": float(bogue["C3A"]),
                "C4AF": float(bogue["C4AF"]),
                "kiln_temperature": float(proc["kiln_temperature"]),
                "kiln_speed": float(proc["kiln_speed"]),
                "coal_feed_rate": float(proc["coal_feed_rate"]),
                "draft_pressure": float(proc["draft_pressure"]),
                "raw_mill_fineness": float(proc["raw_mill_fineness"]),
                "cement_mill_fineness": float(proc["cement_mill_fineness"]),
                "burnability_index": float(burnability_index),
                "heat_consumption": float(heat_consumption),
                "free_lime": float(free_lime),
            }
            rows.append(row)

        return pd.DataFrame(rows)


# --- Optional TimeGAN scaffolding with graceful fallback ---
try:  # pragma: no cover - optional dependency
    from ydata_synthetic.synthesizers.timeseries import TimeGAN  # type: ignore
    _timegan_available = True
except Exception:  # pragma: no cover - optional dependency
    _timegan_available = False


class CementTimeGAN:
    def __init__(self, seq_len: int = 24, n_seq: int = 6):
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.synthesizer = None
        self.scalers: Dict[str, Any] = {}

    def prepare_sequences(self, data: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler  # type: ignore

        df = data.copy()
        for col in feature_cols:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
        sequences: List[np.ndarray] = []
        for i in range(len(df) - self.seq_len + 1):
            sequences.append(df.iloc[i : i + self.seq_len][feature_cols].values)
        return np.asarray(sequences)

    def train(self, sequences: np.ndarray, epochs: int = 500) -> None:
        if _timegan_available:
            model_params = {
                "batch_size": 128,
                "lr": 5e-4,
                "noise_dim": 32,
                "layers_dim": 128,
                "gamma": 1,
                "seq_len": self.seq_len,
                "n_seq": self.n_seq,
            }
            self.synthesizer = TimeGAN(model_parameters=model_params, hidden_dim=24, seq_len=self.seq_len, n_seq=self.n_seq, gamma=1)
            training_data = sequences.reshape(len(sequences), self.seq_len * self.n_seq)
            self.synthesizer.train(training_data, train_steps=epochs)
        else:
            # Statistical fallback
            self.synthesizer = {
                "mean": sequences.mean(axis=0),
                "std": sequences.std(axis=0),
            }

    def sample(self, n_samples: int) -> np.ndarray:
        if _timegan_available and self.synthesizer is not None:
            flat = self.synthesizer.sample(n_samples)
            return flat.reshape(n_samples, self.seq_len, self.n_seq)
        # Fallback: generate AR(1)-like sequences
        out = []
        for _ in range(n_samples):
            seq = np.random.normal(size=(self.seq_len, self.n_seq)) * 0.1
            for t in range(1, self.seq_len):
                seq[t] = 0.7 * seq[t - 1] + 0.3 * seq[t]
            out.append(seq)
        return np.asarray(out)


# --- Optimization data preparation and basic predictors ---
class OptimizationDataPrep:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def create_targets(self) -> pd.DataFrame:
        df = self.data.copy()
        thermal_baseline = 3200.0
        electrical_baseline = 65.0
        thermal_eff = df.get("heat_consumption", 800.0) / thermal_baseline
        electrical_eff = df.get("mill_power", 8500.0) / (8500.0 * df.get("clinker_production", 400.0) / 400.0)
        df["energy_efficiency_score"] = (thermal_eff + electrical_eff) / 2.0
        strength_score = df.get("compressive_strength_28d", 35.0) / 50.0
        quality_score = (2.5 - df.get("free_lime", 1.2)) / 2.5
        consistency_score = 1.0 - abs(df.get("C3S", 60.0) - 60.0) / 20.0
        df["quality_score"] = (strength_score + quality_score + consistency_score) / 3.0
        alt_fuel = df.get("alternative_fuel_rate", 20.0) / 50.0
        emission_score = 1.0 - (df.get("co2_emissions_kg_t", 800.0) - 600.0) / 400.0
        df["sustainability_score"] = (alt_fuel + np.clip(emission_score, 0, 1)) / 2.0
        df["dv_fuel_flow_rate"] = df.get("coal_feed_rate", 3200.0)
        df["dv_kiln_speed"] = df.get("kiln_speed", 3.2)
        df["dv_raw_material_feed"] = df.get("raw_material_feed", 420.0)
        df["dv_oxygen_content"] = df.get("oxygen_content", 2.5)
        df["dv_alternative_fuel_rate"] = df.get("alternative_fuel_rate", 25.0)
        return df


class CementQualityPredictor:
    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}

    def prepare(self, data: pd.DataFrame):
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.metrics import r2_score  # type: ignore
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # type: ignore
        from sklearn.neural_network import MLPRegressor  # type: ignore

        process_features = [
            "kiln_temperature",
            "raw_material_feed",
            "kiln_speed",
            "heat_consumption",
            "oxygen_content",
            "alternative_fuel_rate",
        ]
        targets = ["free_lime", "compressive_strength_28d", "C3S"]
        X = data[[c for c in process_features if c in data.columns]].copy()
        y_all = data[[c for c in targets if c in data.columns]].copy()

        results: Dict[str, Any] = {}
        for target in y_all.columns:
            y = y_all[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models = {
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "neural_network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            }
            best_name = None
            best_model = None
            best_score = -1e9
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = float(r2_score(y_test, model.predict(X_test)))
                    if score > best_score:
                        best_score, best_name, best_model = score, name, model
                except Exception:
                    continue
            if best_model is not None:
                self.models[target] = best_model
                results[target] = {"model_type": best_name, "r2_score": best_score}
                if hasattr(best_model, "feature_importances_"):
                    self.feature_importance[target] = dict(zip(X.columns, best_model.feature_importances_))
        return results

    def predict(self, process_df: pd.DataFrame) -> pd.DataFrame:
        preds: Dict[str, Any] = {}
        for target, model in self.models.items():
            try:
                preds[f"predicted_{target}"] = model.predict(process_df)
            except Exception:
                preds[f"predicted_{target}"] = np.nan
        return pd.DataFrame(preds, index=process_df.index)


class CementEnergyPredictor:
    def __init__(self) -> None:
        self.thermal_model = None
        self.electrical_model = None
        self.feature_cols: List[str] = []

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["fuel_efficiency"] = df.get("alternative_fuel_rate", 20.0) / 100.0
        df["production_rate"] = df.get("clinker_production", 400.0)
        df["temperature_deviation"] = abs(df.get("kiln_temperature", 1450.0) - 1450.0)
        df["feed_consistency"] = 1.0 / (1.0 + df.get("moisture_content", 5.0) / 10.0)
        df["temp_fuel_interaction"] = df.get("kiln_temperature", 1450.0) * df.get("coal_feed_rate", 3200.0) / 1_000_000.0
        self.feature_cols = [
            "raw_material_feed",
            "kiln_speed",
            "kiln_temperature",
            "fuel_efficiency",
            "production_rate",
            "temperature_deviation",
            "feed_consistency",
            "temp_fuel_interaction",
        ]
        present = [c for c in self.feature_cols if c in df.columns]
        return df[present]

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        from sklearn.metrics import r2_score  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

        X = self._create_features(data)
        y_th = data.get("heat_consumption", pd.Series(np.zeros(len(data))))
        y_el = data.get("electrical_energy", pd.Series(np.zeros(len(data))))
        X_tr, X_te, y_th_tr, y_th_te = train_test_split(X, y_th, test_size=0.2, random_state=42)
        X_tr2, X_te2, y_el_tr, y_el_te = train_test_split(X, y_el, test_size=0.2, random_state=42)
        th = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_tr, y_th_tr)
        el = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_tr2, y_el_tr)
        self.thermal_model, self.electrical_model = th, el
        return {"thermal_r2": float(r2_score(y_th_te, th.predict(X_te))), "electrical_r2": float(r2_score(y_el_te, el.predict(X_te2)))}

    def predict(self, process_df: pd.DataFrame) -> pd.DataFrame:
        X = self._create_features(process_df)
        out = pd.DataFrame(index=process_df.index)
        if self.thermal_model is not None:
            out["predicted_thermal_energy"] = self.thermal_model.predict(X)
        if self.electrical_model is not None:
            out["predicted_electrical_energy"] = self.electrical_model.predict(X)
        if "predicted_thermal_energy" in out and "predicted_electrical_energy" in out:
            out["total_energy_score"] = (out["predicted_thermal_energy"] / 3200.0 + out["predicted_electrical_energy"] / 65.0) / 2.0
        return out


