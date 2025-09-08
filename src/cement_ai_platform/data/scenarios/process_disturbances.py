from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def apply_basic_disturbances(base_params: Dict[str, float]) -> Dict[str, float]:
    out = dict(base_params)
    out["kiln_temperature"] += float(np.random.normal(0, 15))
    out["coal_feed_rate"] += float(np.random.normal(0, 150))
    out["draft_pressure"] += float(np.random.normal(0, 2))
    out["kiln_speed"] += float(np.random.normal(0, 0.2))
    return out


def create_comprehensive_disturbance_scenarios(cement_dataset: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    base_data = cement_dataset.copy()
    n_samples = len(base_data)

    scenario_1 = base_data.copy()
    time_series = np.arange(n_samples)
    primary_cycle = np.sin(2 * np.pi * time_series / 480) * 0.08
    daily_cycle = np.sin(2 * np.pi * time_series / 1440) * 0.05
    random_walk = np.cumsum(np.random.normal(0, 0.02, n_samples))
    random_walk = (random_walk - np.mean(random_walk)) / np.std(random_walk) * 0.04
    feed_disturbance = primary_cycle + daily_cycle + random_walk + np.random.normal(0, 0.03, n_samples)
    if "kiln_temperature" in scenario_1:
        scenario_1["kiln_temperature"] *= (1 + feed_disturbance)
    if "coal_feed_rate" in scenario_1:
        scenario_1["coal_feed_rate"] *= (1 + feed_disturbance * 1.2)
    if "draft_pressure" in scenario_1:
        scenario_1["draft_pressure"] *= (1 + feed_disturbance * 0.8)
    if "burnability_index" in scenario_1 and "kiln_temperature" in scenario_1:
        temp_effect = (scenario_1["kiln_temperature"] - 1450) / 1450
        scenario_1["burnability_index"] = np.clip(scenario_1["burnability_index"] + temp_effect * 15, 0, 100)

    scenario_2 = base_data.copy()
    fuel_quality_change = np.zeros(n_samples)
    n_shipments = 5
    shipment_points = np.sort(np.random.choice(n_samples, n_shipments, replace=False))
    current_quality = 0.0
    for i, point in enumerate(shipment_points):
        quality_change = float(np.random.uniform(-0.1, 0.1))
        if i == 0:
            fuel_quality_change[:point] = current_quality
        else:
            fuel_quality_change[shipment_points[i - 1] : point] = current_quality
        current_quality += quality_change
    fuel_quality_change[shipment_points[-1] :] = current_quality
    drift = np.linspace(0, float(np.random.uniform(-0.05, 0.05)), n_samples)
    fuel_quality_change += drift
    if "coal_feed_rate" in scenario_2:
        scenario_2["coal_feed_rate"] *= (1 + fuel_quality_change)
    if "kiln_temperature" in scenario_2:
        scenario_2["kiln_temperature"] *= (1 + fuel_quality_change * 0.6)
    if "heat_consumption" in scenario_2:
        scenario_2["heat_consumption"] *= (1 - fuel_quality_change * 0.8)

    scenario_3 = base_data.copy()
    progressive_degradation = np.linspace(0, -0.06, n_samples)
    episodic_events = np.zeros(n_samples)
    n_events = 3
    for _ in range(n_events):
        event_start = int(np.random.randint(n_samples // 4, 3 * n_samples // 4))
        event_duration = int(np.random.randint(100, 300))
        event_end = min(event_start + event_duration, n_samples)
        event_intensity = float(np.random.uniform(0.03, 0.08))
        recovery_curve = -event_intensity * np.exp(-np.linspace(0, 2, event_end - event_start))
        episodic_events[event_start:event_end] += recovery_curve
    total_degradation = progressive_degradation + episodic_events
    if "kiln_speed" in scenario_3:
        scenario_3["kiln_speed"] *= (1 + total_degradation)
    if "raw_mill_fineness" in scenario_3:
        scenario_3["raw_mill_fineness"] *= (1 - total_degradation * 0.5)
    if "cement_mill_fineness" in scenario_3:
        scenario_3["cement_mill_fineness"] *= (1 - total_degradation * 0.3)

    scenario_4 = base_data.copy()
    n_quarry_changes = 4
    segment_lengths = np.random.multinomial(n_samples, np.ones(n_quarry_changes) / n_quarry_changes)
    chemistry_shifts = np.zeros(n_samples)
    current_pos = 0
    current_shift = 0.0
    for segment_length in segment_lengths:
        if segment_length == 0:
            continue
        new_shift = float(np.random.uniform(-0.03, 0.03))
        transition_length = min(50, segment_length // 2)
        if transition_length > 0:
            transition = np.linspace(current_shift, new_shift, transition_length)
            chemistry_shifts[current_pos : current_pos + transition_length] = transition
            chemistry_shifts[current_pos + transition_length : current_pos + segment_length] = new_shift
        else:
            chemistry_shifts[current_pos : current_pos + segment_length] = new_shift
        current_pos += segment_length
        current_shift = new_shift
        if current_pos >= n_samples:
            break
    if "LSF" in scenario_4:
        scenario_4["LSF"] = np.clip(scenario_4["LSF"] + chemistry_shifts * scenario_4["LSF"].std(), 0.85, 1.05)
    if "SM" in scenario_4:
        scenario_4["SM"] = np.clip(scenario_4["SM"] + chemistry_shifts * scenario_4["SM"].std(), 2.0, 3.5)
    if "AM" in scenario_4:
        scenario_4["AM"] = np.clip(scenario_4["AM"] + chemistry_shifts * scenario_4["AM"].std(), 1.0, 2.5)
    for compound in ["C3S", "C2S", "C3A", "C4AF"]:
        if compound in scenario_4:
            mult = {"C3S": 0.3, "C2S": -0.2, "C3A": 0.4, "C4AF": 0.1}[compound]
            scenario_4[compound] = np.clip(scenario_4[compound] * (1 + chemistry_shifts * mult), 0, 100)

    scenario_5 = base_data.copy()
    maintenance_effect = np.zeros(n_samples)
    n_maintenance = 2
    for _ in range(n_maintenance):
        maint_start = int(np.random.randint(n_samples // 4, 3 * n_samples // 4))
        maint_duration = int(np.random.randint(80, 150))
        maint_end = min(maint_start + maint_duration, n_samples)
        ramp_duration = maint_duration // 4
        ramp_down = np.linspace(0, -0.2, ramp_duration)
        full_maint = np.ones(maint_duration - 2 * ramp_duration) * (-0.2)
        ramp_up = np.linspace(-0.2, 0, ramp_duration)
        profile = np.concatenate([ramp_down, full_maint, ramp_up])
        actual = min(len(profile), maint_end - maint_start)
        maintenance_effect[maint_start : maint_start + actual] = profile[:actual]
    for col in ["kiln_temperature", "kiln_speed", "coal_feed_rate"]:
        if col in scenario_5:
            scenario_5[col] *= (1 + maintenance_effect)

    combined = base_data.copy()
    if "kiln_temperature" in combined:
        combined["kiln_temperature"] *= (1 + feed_disturbance * 0.6 + fuel_quality_change * 0.4 + maintenance_effect * 0.8)
    if "coal_feed_rate" in combined:
        combined["coal_feed_rate"] *= (1 + feed_disturbance * 0.8 + fuel_quality_change * 0.6 + maintenance_effect * 0.9)
    if "kiln_speed" in combined:
        combined["kiln_speed"] *= (1 + total_degradation * 0.7 + maintenance_effect * 0.6)
    for col in ["LSF", "SM", "AM"]:
        if col in combined and col in scenario_4:
            combined[col] = scenario_4[col]
    if "burnability_index" in combined and "kiln_temperature" in combined:
        temp_effect = (combined["kiln_temperature"] - 1450) / 1450
        combined["burnability_index"] = np.clip(combined["burnability_index"] + temp_effect * 12, 0, 100)
    if "free_lime" in combined and "kiln_temperature" in combined:
        temp_effect = (combined["kiln_temperature"] - 1450) / 1450
        combined["free_lime"] = np.clip(combined["free_lime"] - temp_effect * 0.3, 0.1, 5.0)

    return {
        "base": base_data,
        "feed_rate_variation": scenario_1,
        "fuel_quality_change": scenario_2,
        "equipment_degradation": scenario_3,
        "raw_material_shift": scenario_4,
        "maintenance_mode": scenario_5,
        "combined_disturbances": combined,
    }


class DisturbanceSimulator:
    """Apply single or combined process disturbances to a dataset."""

    def __init__(self, base_dataset: pd.DataFrame) -> None:
        self.base_dataset = base_dataset.copy()
        # Configuration for disturbance scenarios
        self.disturbance_scenarios: Dict[str, Dict[str, object]] = {
            "feed_rate_variation": {
                "intensity_range": (0.02, 0.12),
                "probability": 0.4,
                "affected_params": ["kiln_temperature", "coal_feed_rate", "draft_pressure"],
                "description": "Cyclical + random feed disturbances across shifts and day cycle",
            },
            "fuel_quality_change": {
                "intensity_range": (0.03, 0.10),
                "probability": 0.3,
                "affected_params": ["coal_feed_rate", "kiln_temperature", "heat_consumption"],
                "description": "Step changes per fuel shipment with drift",
            },
            "equipment_degradation": {
                "intensity_range": (0.02, 0.08),
                "probability": 0.25,
                "affected_params": ["kiln_speed", "raw_mill_fineness", "cement_mill_fineness"],
                "description": "Progressive efficiency loss and episodic events",
            },
            "raw_material_composition_shift": {
                "intensity_range": (0.01, 0.05),
                "probability": 0.2,
                "affected_params": ["LSF", "SM", "AM", "C3S", "C2S", "C3A", "C4AF"],
                "description": "Quarry face changes causing chemistry shifts",
            },
            "maintenance_mode": {
                "intensity_range": (0.10, 0.25),
                "probability": 0.2,
                "affected_params": ["kiln_temperature", "kiln_speed", "coal_feed_rate"],
                "description": "Scheduled shutdowns with ramp down/up",
            },
        }

    # ---------------------------- pattern generators ----------------------------
    def generate_feed_rate_disturbance(self, n: int, intensity: float) -> np.ndarray:
        t = np.arange(n)
        primary = np.sin(2 * np.pi * t / 480) * 0.08
        daily = np.sin(2 * np.pi * t / 1440) * 0.05
        rw = np.cumsum(np.random.normal(0, 0.02, n))
        rw = (rw - rw.mean()) / (rw.std() + 1e-9) * 0.04
        noise = np.random.normal(0, 0.03, n)
        return (primary + daily + rw + noise) * (intensity / 0.1)

    def generate_fuel_quality_disturbance(self, n: int, intensity: float) -> np.ndarray:
        changes = np.zeros(n)
        n_ship = max(3, int(5 * (intensity / 0.1)))
        points = np.sort(np.random.choice(n, n_ship, replace=False))
        q = 0.0
        last = 0
        for p in points:
            changes[last:p] = q
            q += np.random.uniform(-0.1, 0.1) * (intensity / 0.1)
            last = p
        changes[last:] = q
        drift = np.linspace(0, np.random.uniform(-0.05, 0.05) * (intensity / 0.1), n)
        return changes + drift

    def generate_equipment_degradation(self, n: int, intensity: float) -> np.ndarray:
        progressive = np.linspace(0, -0.06, n) * (intensity / 0.1)
        episodic = np.zeros(n)
        for _ in range(3):
            start = np.random.randint(n // 4, 3 * n // 4)
            duration = np.random.randint(100, 300)
            end = min(start + duration, n)
            e_int = np.random.uniform(0.03, 0.08) * (intensity / 0.1)
            episodic[start:end] += -e_int * np.exp(-np.linspace(0, 2, end - start))
        return progressive + episodic

    def generate_raw_material_shift(self, n: int, intensity: float) -> np.ndarray:
        segs = np.random.multinomial(n, np.ones(4) / 4)
        out = np.zeros(n)
        pos = 0
        current = 0.0
        for l in segs:
            if l == 0:
                continue
            new = np.random.uniform(-0.03, 0.03) * (intensity / 0.05)
            tr = min(50, l // 2)
            if tr > 0:
                out[pos : pos + tr] = np.linspace(current, new, tr)
                out[pos + tr : pos + l] = new
            else:
                out[pos : pos + l] = new
            pos += l
            current = new
        return out

    def generate_maintenance_mode(self, n: int, intensity: float) -> np.ndarray:
        effect = np.zeros(n)
        for _ in range(2):
            start = np.random.randint(n // 4, 3 * n // 4)
            duration = np.random.randint(80, 150)
            end = min(start + duration, n)
            ramp = duration // 4
            down = np.linspace(0, -0.2, ramp) * (intensity / 0.2)
            maint = np.ones(duration - 2 * ramp) * (-0.2 * (intensity / 0.2))
            up = np.linspace(-0.2, 0, ramp) * (intensity / 0.2)
            profile = np.concatenate([down, maint, up])
            effect[start : start + len(profile)] = profile[: end - start]
        return effect

    def apply_seasonal_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        season = np.sin(2 * np.pi * np.arange(n) / 8760) * 0.01
        if "kiln_temperature" in out:
            out["kiln_temperature"] *= (1 + season)
        if "draft_pressure" in out:
            out["draft_pressure"] *= (1 + season * 0.5)
        return out

    # -------------------------- disturbance application --------------------------
    def apply_single_disturbance(
        self,
        data: pd.DataFrame,
        disturbance_type: str,
        intensity: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        if disturbance_type not in self.disturbance_scenarios:
            raise ValueError(f"Unknown disturbance type: {disturbance_type}")

        scenario = self.disturbance_scenarios[disturbance_type]
        disturbed = data.copy()
        n = len(disturbed)
        if intensity is None:
            lo, hi = scenario["intensity_range"]  # type: ignore[index]
            intensity = float(np.random.uniform(lo, hi))

        if disturbance_type == "feed_rate_variation":
            pattern = self.generate_feed_rate_disturbance(n, intensity)
        elif disturbance_type == "fuel_quality_change":
            pattern = self.generate_fuel_quality_disturbance(n, intensity)
        elif disturbance_type == "equipment_degradation":
            pattern = self.generate_equipment_degradation(n, intensity)
        elif disturbance_type == "raw_material_composition_shift":
            pattern = self.generate_raw_material_shift(n, intensity)
        elif disturbance_type == "maintenance_mode":
            pattern = self.generate_maintenance_mode(n, intensity)
        else:
            pattern = np.random.normal(0, intensity, n)

        applied_changes: Dict[str, Dict[str, float]] = {}
        for param in scenario["affected_params"]:  # type: ignore[index]
            if param not in disturbed.columns:
                continue
            original = disturbed[param].to_numpy()
            if param in ["kiln_temperature", "coal_feed_rate", "kiln_speed"]:
                new_vals = original * (1 + pattern)
            elif param in ["LSF", "SM", "AM"]:
                new_vals = original + pattern * disturbed[param].std()
                if param == "LSF":
                    new_vals = np.clip(new_vals, 0.85, 1.05)
                if param == "SM":
                    new_vals = np.clip(new_vals, 2.0, 3.5)
                if param == "AM":
                    new_vals = np.clip(new_vals, 1.0, 2.5)
            elif param in ["C3S", "C2S", "C3A", "C4AF"]:
                new_vals = np.clip(original * (1 + pattern * 0.5), 0, 100)
            else:
                new_vals = original * (1 + pattern)
            disturbed[param] = new_vals
            denom = np.where(original == 0, 1e-6, original)
            change = np.mean(np.abs(new_vals - original) / np.abs(denom))
            applied_changes[param] = {
                "mean_change_pct": float(change * 100.0),
                "max_change_pct": float(np.max(np.abs(new_vals - original) / np.abs(denom)) * 100.0),
            }

        if "kiln_temperature" in disturbed and "burnability_index" in disturbed:
            temp_dev = (disturbed["kiln_temperature"] - 1450) / 1450
            disturbed["burnability_index"] = np.clip(
                disturbed["burnability_index"] + temp_dev * 10, 0, 100
            )

        info: Dict[str, object] = {
            "type": disturbance_type,
            "description": scenario["description"],  # type: ignore[index]
            "intensity": intensity,
            "applied_changes": applied_changes,
            "n_samples_affected": n,
        }
        return disturbed, info

    def apply_combined_disturbances(
        self,
        data: pd.DataFrame,
        scenario_probabilities: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
        if scenario_probabilities is None:
            scenario_probabilities = {
                name: float(cfg["probability"]) for name, cfg in self.disturbance_scenarios.items()
            }
        disturbed = data.copy()
        applied: List[Dict[str, object]] = []
        active = [
            name
            for name, p in scenario_probabilities.items()
            if np.random.random() < float(p)
        ]
        for name in active:
            disturbed, info = self.apply_single_disturbance(disturbed, name)
            applied.append(info)
        disturbed = self.apply_seasonal_effects(disturbed)
        return disturbed, applied

    def create_disturbance_scenarios(self, n_scenarios: int = 5) -> List[Tuple[pd.DataFrame, List[Dict[str, object]]]]:
        scenarios: List[Tuple[pd.DataFrame, List[Dict[str, object]]]] = []
        base = self.base_dataset.copy()
        for i in range(n_scenarios):
            if i == 0:
                probs = {k: float(cfg["probability"]) * 0.3 for k, cfg in self.disturbance_scenarios.items()}
            elif i == 1:
                probs = {k: min(0.8, float(cfg["probability"]) * 2.0) for k, cfg in self.disturbance_scenarios.items()}
            elif i == 2:
                probs = {"maintenance_mode": 0.9, "equipment_degradation": 0.6}
                for k, cfg in self.disturbance_scenarios.items():
                    probs.setdefault(k, float(cfg["probability"]) * 0.2)
            elif i == 3:
                probs = {"raw_material_composition_shift": 0.8, "fuel_quality_change": 0.6}
                for k, cfg in self.disturbance_scenarios.items():
                    probs.setdefault(k, float(cfg["probability"]) * 0.4)
            else:
                probs = {k: float(np.random.uniform(0.1, 0.7)) for k in self.disturbance_scenarios}
            scen_df, info = self.apply_combined_disturbances(base, probs)
            scenarios.append((scen_df, info))
        return scenarios



