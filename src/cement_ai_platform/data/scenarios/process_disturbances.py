from __future__ import annotations

from typing import Dict

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



