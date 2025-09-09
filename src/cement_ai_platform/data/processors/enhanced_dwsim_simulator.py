from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# Import the proper DWSIM integration
from .dwsim_integration import DWSIMCementSimulator


class EnhancedDWSIMSimulator:
    """
    Enhanced DWSIM process simulator for cement plant operations.
    
    Uses the official DWSIM package for chemical process simulation.
    Simulates key cement manufacturing processes:
    - Raw material grinding and blending
    - Pyroprocessing (kiln operations)
    - Clinker cooling and storage
    - Gas-solid interactions
    """
    
    def __init__(self, seed: int = 42):
        # Use the proper DWSIM integration
        self.dwsim_simulator = DWSIMCementSimulator(seed)
        self.seed = seed
    
    def simulate_raw_mill(self, feed_rate: float, moisture_content: float = 5.0) -> Dict[str, float]:
        """Simulate raw material grinding process using DWSIM."""
        return self.dwsim_simulator.simulate_raw_mill(feed_rate, moisture_content)
    
    def simulate_kiln_process(self, feed_rate: float, fuel_rate: float, 
                            kiln_speed: float, air_flow: float = 100000) -> Dict[str, float]:
        """Simulate rotary kiln pyroprocessing using DWSIM."""
        return self.dwsim_simulator.simulate_kiln_process(feed_rate, fuel_rate, kiln_speed, air_flow)
    
    def simulate_cooler(self, clinker_rate: float, inlet_temp: float = 1350) -> Dict[str, float]:
        """Simulate clinker cooling process using DWSIM."""
        return self.dwsim_simulator.simulate_cooler(clinker_rate, inlet_temp)
    
    def simulate_complete_process(self, feed_rate: float, fuel_rate: float,
                                kiln_speed: float, moisture_content: float = 5.0) -> Dict[str, Any]:
        """Simulate complete cement manufacturing process using DWSIM."""
        return self.dwsim_simulator.simulate_complete_process(feed_rate, fuel_rate, kiln_speed, moisture_content)
    
    def generate_operational_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate comprehensive operational dataset using DWSIM simulation."""
        return self.dwsim_simulator.generate_operational_dataset(n_samples)


def create_dwsim_simulator(seed: int = 42) -> EnhancedDWSIMSimulator:
    """Factory function to create a DWSIM simulator."""
    return EnhancedDWSIMSimulator(seed)