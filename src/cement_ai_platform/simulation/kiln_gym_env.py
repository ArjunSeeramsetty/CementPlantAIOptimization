"""
Gym-like environment wrapper for Kiln-Cooler Sintering Zone Optimization.
Wraps AdvancedKilnModel, PreheaterTowerModel, and CoolerModel into an RL state-action loop.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any
from cement_ai_platform.models.agents.unified_kiln_cooler_controller import (
    AdvancedKilnModel,
    PreheaterTowerModel,
    CoolerModel
)

class KilnCoolerGymEnv:
    """
    OpenAI Gym-style simulation environment for the Kiln-Cooler process.
    Provides observation and action spaces, steps through process models,
    and returns rewards/penalties based on plant performance constraints.
    """

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.kiln_model = AdvancedKilnModel()
        self.preheater_model = PreheaterTowerModel()
        self.cooler_model = CoolerModel()
        
        # State boundaries (for normalization if needed)
        self.obs_low = np.array([1200.0, 5.0, 1.0, 100.0, 0.2, 300.0, 400.0], dtype=np.float32)
        self.obs_high = np.array([1600.0, 25.0, 4.5, 250.0, 3.0, 1000.0, 800.0], dtype=np.float32)
        
        # Action boundaries (adjustments delta)
        self.action_low = np.array([-0.2, -0.5, -2.0], dtype=np.float32)
        self.action_high = np.array([0.2, 0.5, 2.0], dtype=np.float32)
        
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the process variables to nominal steady-state values."""
        self.step_count = 0
        self.safety_tripped = False
        
        # Nominal physical parameters
        self.burning_zone_temp = 1450.0     # °C
        self.fuel_rate = 15.0               # TPH
        self.kiln_speed = 3.0               # RPM
        self.feed_rate = 200.0              # TPH
        self.free_lime = 1.2                # % target
        self.nox = 500.0                    # mg/Nm3
        self.preheater_outlet_temp = 550.0  # °C
        
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Construct the flat state observation array."""
        return np.array([
            self.burning_zone_temp,
            self.fuel_rate,
            self.kiln_speed,
            self.feed_rate,
            self.free_lime,
            self.nox,
            self.preheater_outlet_temp
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Run one time step of the process simulation.
        
        Args:
            action: 3D array of continuous adjustments: [speed_adj, fuel_adj, feed_adj]
            
        Returns:
            Observation, Reward, Done, Info
        """
        self.step_count += 1
        
        # 1. Clamp and Apply Action Adjustments
        speed_adj = np.clip(action[0], self.action_low[0], self.action_high[0])
        fuel_adj = np.clip(action[1], self.action_low[1], self.action_high[1])
        feed_adj = np.clip(action[2], self.action_low[2], self.action_high[2])
        
        # Apply to setpoints and clamp to physical model boundaries
        self.kiln_speed = float(np.clip(self.kiln_speed + speed_adj, 1.0, 4.5))
        self.fuel_rate = float(np.clip(self.fuel_rate + fuel_adj, 5.0, 25.0))
        self.feed_rate = float(np.clip(self.feed_rate + feed_adj, 100.0, 250.0))
        
        # 2. Simulate Process Models
        raw_meal_properties = {
            'lime_saturation_factor': 0.95,
            'silica_ratio': 2.5,
            'alumina_ratio': 1.5,
            'fineness_blaine': 3000
        }
        
        # Simulate kiln dynamics
        kiln_results = self.kiln_model.simulate_kiln_dynamics(
            feed_rate=self.feed_rate,
            fuel_rate=self.fuel_rate,
            kiln_speed=self.kiln_speed,
            raw_meal_properties=raw_meal_properties
        )
        self.burning_zone_temp = kiln_results['operating_conditions']['burning_zone_temp_c']
        
        # Simulate preheater tower balance
        gas_flow = 200000.0 + (self.fuel_rate - 15.0) * 8000.0
        preheater_results = self.preheater_model.calculate_heat_and_mass_balance(
            raw_meal_flow=self.feed_rate,
            gas_flow=gas_flow,
            raw_meal_composition={}
        )
        self.preheater_outlet_temp = preheater_results['gas_outlet_temp_c']
        
        # Simulate cooler
        self.cooler_model.calculate_cooling_performance(
            clinker_temp_in=self.burning_zone_temp,
            air_flow=150000.0,
            clinker_flow=self.feed_rate
        )
        
        # 3. Update Quality (Free Lime) & Emissions (NOx)
        # Free Lime is a function of sintering zone temp and residence time
        # Underheated kiln (<1400 C) -> high free lime (under-calcination)
        # Overheated kiln (>1500 C) -> low free lime (overburned, fuel waste)
        temp_deviation = self.burning_zone_temp - 1450.0
        self.free_lime = 1.2 - 0.008 * temp_deviation + random.normalvariate(0, 0.05)
        self.free_lime = float(np.clip(self.free_lime, 0.2, 3.0))
        
        # NOx increases with thermal temp and fuel rates
        self.nox = 500.0 + 0.9 * temp_deviation + 6.0 * (self.fuel_rate - 15.0) + random.normalvariate(0, 8.0)
        self.nox = float(np.clip(self.nox, 300.0, 1000.0))
        
        # 4. Check Safety Interlocks & Constraints
        done = False
        info = {'trip_reason': ''}
        
        if self.burning_zone_temp < 1250.0:
            self.safety_tripped = True
            info['trip_reason'] = 'Sintering zone temperature collapsed'
            done = True
        elif self.burning_zone_temp > 1580.0:
            self.safety_tripped = True
            info['trip_reason'] = 'Exceeded refractory thermal limit'
            done = True
            
        if self.step_count >= self.max_steps:
            done = True
            
        # 5. Compute Reward
        reward = self._calculate_reward()
        
        return self._get_obs(), reward, done, info

    def _calculate_reward(self) -> float:
        """Compute multivariable reward matching plant production and energy targets."""
        # Quality Penalty (Target Free Lime is 1.2%)
        quality_penalty = -40.0 * ((self.free_lime - 1.2) ** 2)
        
        # Fuel efficiency (cost penalty)
        fuel_penalty = -1.2 * self.fuel_rate
        
        # Throughput reward (production incentive)
        throughput_reward = 0.45 * (self.feed_rate - 100.0)
        
        # Emissions penalty (above target 520 mg/Nm3)
        emission_penalty = 0.0
        if self.nox > 520.0:
            emission_penalty = -0.15 * (self.nox - 520.0)
            
        # Critical trip penalty
        trip_penalty = -100.0 if self.safety_tripped else 0.0
        
        return quality_penalty + fuel_penalty + throughput_reward + emission_penalty + trip_penalty
