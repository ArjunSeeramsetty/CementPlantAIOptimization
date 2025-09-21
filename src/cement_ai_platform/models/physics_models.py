# FILE: src/cement_ai_platform/models/physics_models.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
from dataclasses import dataclass

@dataclass
class ProcessState:
    """Current process state"""
    kiln_temperature: float
    feed_rate: float
    fuel_rate: float
    kiln_speed: float
    o2_percentage: float
    preheater_temp: float
    cooler_temp: float
    timestamp: datetime

class PhysicsBasedProcessModel:
    """Physics-informed process models for cement manufacturing"""

    def __init__(self, plant_config):
        self.config = plant_config
        self.baseline_energy = plant_config.get_baseline_energy_consumption()
        self.process_windows = plant_config.get_optimal_process_window()

        # Physics constants
        self.BURNABILITY_CONSTANT = 0.85
        self.HEAT_TRANSFER_COEFF = 0.92
        self.RESIDENCE_TIME_FACTOR = 1.2

    def calculate_free_lime(self, kiln_temp: float, residence_time: float,
                          feed_composition: Dict[str, float]) -> float:
        """
        Physics-based free lime calculation using Lea-Parker equation
        Free Lime = f(temperature, residence time, LSF, fineness)
        """
        # Optimal temperature for complete burnability
        optimal_temp = self.config.process['kiln_temperature_c']

        # Temperature effect (Arrhenius-type relationship)
        temp_factor = np.exp(-6000 * (1/kiln_temp - 1/optimal_temp) / 8.314)

        # Residence time effect
        residence_factor = min(1.0, residence_time / 45.0)  # 45 min optimal

        # LSF effect on burnability
        lsf = self._calculate_lsf(feed_composition)
        lsf_factor = 1.0 - abs(lsf - 95) * 0.02  # Optimal LSF = 95%

        # Base free lime from config
        base_free_lime = self.config.quality['free_lime_pct']

        # Calculate actual free lime
        free_lime = base_free_lime * (2.0 - temp_factor * residence_factor * lsf_factor)

        return max(0.3, min(3.0, free_lime))  # Physical limits

    def calculate_thermal_energy(self, process_state: ProcessState,
                                fuel_mix: Dict[str, float]) -> float:
        """
        Calculate specific thermal energy consumption
        """
        base_thermal = self.baseline_energy['thermal_energy_kcal_kg']

        # Temperature efficiency effect
        optimal_temp = self.config.process['kiln_temperature_c']
        temp_efficiency = 1.0 - abs(process_state.kiln_temperature - optimal_temp) * 0.001

        # Fuel quality effect
        weighted_cv = self._calculate_weighted_calorific_value(fuel_mix)
        fuel_efficiency = weighted_cv / 6000  # Normalized to coal baseline

        # Feed rate effect (economies of scale)
        optimal_feed_rate = self.config.capacity_tpd / 24
        utilization = process_state.feed_rate / optimal_feed_rate
        scale_efficiency = 0.9 + 0.1 * min(1.0, utilization)

        # Air leakage effect
        excess_air_factor = max(1.0, (process_state.o2_percentage - 3.0) * 0.05 + 1.0)

        thermal_energy = base_thermal / (temp_efficiency * fuel_efficiency * scale_efficiency) * excess_air_factor

        return thermal_energy

    def calculate_electrical_energy(self, process_state: ProcessState) -> float:
        """Calculate specific electrical energy consumption"""
        base_electrical = self.baseline_energy['electrical_energy_kwh_t']

        # Mill loading effect
        utilization = min(1.2, process_state.feed_rate / (self.config.capacity_tpd / 24))
        mill_efficiency = 0.85 + 0.15 * utilization if utilization <= 1.0 else 1.0 - (utilization - 1.0) * 0.3

        # Technology level effect
        tech_multiplier = {
            'basic': 1.15,
            'advanced': 1.0,
            'state_of_art': 0.88
        }.get(self.config.technology_level, 1.0)

        electrical_energy = base_electrical / mill_efficiency * tech_multiplier

        return electrical_energy

    def calculate_cement_strength(self, clinker_composition: Dict[str, float],
                                fineness: float, curing_days: int = 28) -> float:
        """
        Calculate cement compressive strength using Bogue equations
        """
        c3s = clinker_composition.get('c3s_content_pct', 60)
        c2s = clinker_composition.get('c2s_content_pct', 15)
        c3a = clinker_composition.get('c3a_content_pct', 8)

        # Bogue strength contribution
        strength_28d = (c3s * 0.65 + c2s * 0.25 + c3a * 0.10) * (fineness / 3400) ** 0.3

        # Curing time effect
        if curing_days != 28:
            time_factor = math.log(curing_days) / math.log(28)
            strength_28d *= time_factor

        return strength_28d

    def predict_nox_emissions(self, process_state: ProcessState,
                            fuel_mix: Dict[str, float]) -> float:
        """Predict NOx emissions based on process conditions"""
        base_nox = self.config.environmental['nox_mg_nm3']

        # Temperature effect (exponential relationship)
        temp_factor = np.exp((process_state.kiln_temperature - 1450) / 100)

        # Excess air effect
        air_factor = 1.0 + (process_state.o2_percentage - 3.0) * 0.15

        # Alternative fuel effect (typically reduces NOx)
        alt_fuel_ratio = fuel_mix.get('alternative_fuels', 0) / sum(fuel_mix.values())
        alt_fuel_factor = 1.0 - alt_fuel_ratio * 0.2

        nox_emissions = base_nox * temp_factor * air_factor * alt_fuel_factor

        return max(200, min(800, nox_emissions))

    def _calculate_lsf(self, feed_composition: Dict[str, float]) -> float:
        """Calculate Lime Saturation Factor"""
        cao = feed_composition.get('limestone', 1200) * 0.54  # Approximate CaO from limestone
        sio2 = feed_composition.get('clay', 200) * 0.58  # Approximate SiO2 from clay
        al2o3 = feed_composition.get('clay', 200) * 0.18  # Approximate Al2O3 from clay
        fe2o3 = feed_composition.get('iron_ore', 50) * 0.85  # Approximate Fe2O3 from iron ore

        lsf = (cao / (2.8 * sio2 + 1.2 * al2o3 + 0.65 * fe2o3)) * 100
        return lsf

    def _calculate_weighted_calorific_value(self, fuel_mix: Dict[str, float]) -> float:
        """Calculate weighted average calorific value"""
        total_fuel = sum(fuel_mix.values())
        if total_fuel == 0:
            return 6000  # Default coal CV

        weighted_cv = 0
        for fuel_type, amount in fuel_mix.items():
            if fuel_type in self.config.fuel_properties:
                cv = self.config.fuel_properties[fuel_type]['cv_kcal_kg']
                weighted_cv += (amount / total_fuel) * cv

        return weighted_cv or 6000

class DynamicProcessDataGenerator:
    """Generate realistic process data using physics models"""

    def __init__(self, plant_config):
        self.config = plant_config
        self.physics_model = PhysicsBasedProcessModel(plant_config)
        self.current_state = self._initialize_process_state()

    def _initialize_process_state(self) -> ProcessState:
        """Initialize process state near optimal conditions"""
        return ProcessState(
            kiln_temperature=self.config.process['kiln_temperature_c'] + np.random.normal(0, 3),
            feed_rate=self.config.capacity_tpd / 24 + np.random.normal(0, 2),
            fuel_rate=self.config.fuel_mix['total_fuel_rate_tph'] + np.random.normal(0, 0.5),
            kiln_speed=self.config.process['kiln_speed_rpm'] + np.random.normal(0, 0.1),
            o2_percentage=3.2 + np.random.normal(0, 0.3),
            preheater_temp=850 + np.random.normal(0, 15),
            cooler_temp=120 + np.random.normal(0, 8),
            timestamp=datetime.now()
        )

    def generate_current_kpis(self) -> Dict[str, float]:
        """Generate current KPIs using physics models"""

        # Update process state with some realistic variation
        self._update_process_state()

        # Calculate residence time
        residence_time = 3600 / (self.current_state.kiln_speed * 60)  # minutes

        # Calculate KPIs using physics models
        kpis = {
            'kiln_temperature_c': self.current_state.kiln_temperature,
            'feed_rate_tph': self.current_state.feed_rate,
            'fuel_rate_tph': self.current_state.fuel_rate,
            'free_lime_pct': self.physics_model.calculate_free_lime(
                self.current_state.kiln_temperature,
                residence_time,
                self.config.raw_materials
            ),
            'thermal_energy_kcal_kg': self.physics_model.calculate_thermal_energy(
                self.current_state,
                self.config.fuel_mix
            ),
            'electrical_energy_kwh_t': self.physics_model.calculate_electrical_energy(
                self.current_state
            ),
            'cement_strength_28d': self.physics_model.calculate_cement_strength(
                self.config.quality,
                self.config.quality.get('blaine_cm2_g', 3400)
            ),
            'nox_emissions_mg_nm3': self.physics_model.predict_nox_emissions(
                self.current_state,
                self.config.fuel_mix
            ),
            'o2_percentage': self.current_state.o2_percentage,
            'production_rate_tph': self.current_state.feed_rate * 0.98,  # 98% conversion
            'oee_percentage': self._calculate_oee(),
            'energy_efficiency_pct': self._calculate_energy_efficiency()
        }

        return kpis

    def _update_process_state(self):
        """Update process state with realistic variations"""
        # Add some process dynamics
        temp_drift = np.random.normal(0, 1)
        self.current_state.kiln_temperature += temp_drift

        # Keep within operating bounds
        min_temp = self.config.process['kiln_temperature_c'] - 25
        max_temp = self.config.process['kiln_temperature_c'] + 20
        self.current_state.kiln_temperature = np.clip(
            self.current_state.kiln_temperature, min_temp, max_temp
        )

        # Update other variables with correlations
        if temp_drift > 0:
            self.current_state.o2_percentage += np.random.uniform(0, 0.1)
        else:
            self.current_state.o2_percentage -= np.random.uniform(0, 0.1)

        self.current_state.o2_percentage = np.clip(self.current_state.o2_percentage, 2.5, 4.5)

        # Update timestamp
        self.current_state.timestamp = datetime.now()

    def _calculate_oee(self) -> float:
        """Calculate Overall Equipment Effectiveness"""
        # Availability based on technology level
        availability = {
            'basic': 0.88,
            'advanced': 0.92,
            'state_of_art': 0.96
        }.get(self.config.technology_level, 0.90)

        # Performance based on current utilization
        current_rate = self.current_state.feed_rate
        design_rate = self.config.capacity_tpd / 24
        performance = min(1.0, current_rate / design_rate)

        # Quality based on free lime control
        target_free_lime = self.config.quality['free_lime_pct']
        current_free_lime = self.physics_model.calculate_free_lime(
            self.current_state.kiln_temperature,
            3600 / (self.current_state.kiln_speed * 60),
            self.config.raw_materials
        )
        quality = 1.0 - min(0.15, abs(current_free_lime - target_free_lime) * 0.1)

        oee = availability * performance * quality * 100
        return round(oee, 1)

    def _calculate_energy_efficiency(self) -> float:
        """Calculate energy efficiency percentage"""
        current_thermal = self.physics_model.calculate_thermal_energy(
            self.current_state, self.config.fuel_mix
        )
        baseline_thermal = self.config.energy['thermal']

        # Efficiency is inverse of energy consumption
        efficiency = (baseline_thermal / current_thermal) * 100
        return round(min(105, max(75, efficiency)), 1)

    def generate_historical_data(self, hours: int = 24) -> pd.DataFrame:
        """Generate historical data for the specified period"""
        data_points = []
        current_time = datetime.now()

        for i in range(hours):
            # Generate data for this hour
            kpis = self.generate_current_kpis()
            kpis['timestamp'] = current_time - timedelta(hours=i)
            data_points.append(kpis)

            # Update state every hour
            self._update_process_state()

        return pd.DataFrame(data_points).sort_values('timestamp')

    def simulate_scenario(self, scenario_type: str, duration_hours: int = 8) -> pd.DataFrame:
        """Simulate specific process scenarios"""
        scenarios = {
            'high_temperature': {'temp_offset': 20, 'fuel_offset': 0.5},
            'low_temperature': {'temp_offset': -15, 'fuel_offset': -0.3},
            'high_feed_rate': {'feed_offset': 5, 'temp_offset': 5},
            'quality_issue': {'temp_offset': -10, 'free_lime_offset': 0.5},
            'fuel_efficiency': {'fuel_offset': -1.0, 'temp_offset': 0}
        }

        if scenario_type not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_type}")

        scenario_params = scenarios[scenario_type]
        data_points = []
        current_time = datetime.now()

        # Store original state
        original_temp = self.current_state.kiln_temperature
        original_feed = self.current_state.feed_rate
        original_fuel = self.current_state.fuel_rate

        try:
            for i in range(duration_hours):
                # Apply scenario modifications
                self.current_state.kiln_temperature = original_temp + scenario_params.get('temp_offset', 0)
                self.current_state.feed_rate = original_feed + scenario_params.get('feed_offset', 0)
                self.current_state.fuel_rate = original_fuel + scenario_params.get('fuel_offset', 0)

                # Generate KPIs for this scenario
                kpis = self.generate_current_kpis()
                kpis['timestamp'] = current_time - timedelta(hours=i)
                kpis['scenario'] = scenario_type
                data_points.append(kpis)

        finally:
            # Restore original state
            self.current_state.kiln_temperature = original_temp
            self.current_state.feed_rate = original_feed
            self.current_state.fuel_rate = original_fuel

        return pd.DataFrame(data_points).sort_values('timestamp')
