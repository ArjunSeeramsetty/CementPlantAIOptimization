import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt

class ProcessDisturbanceSimulator:
    """
    Comprehensive process disturbance simulator for realistic cement plant scenarios
    """
    
    def __init__(self, base_dataset: pd.DataFrame, seed=42):
        np.random.seed(seed)
        self.base_dataset = base_dataset.copy()
        
        # Define disturbance types with realistic parameters
        self.disturbance_scenarios = {
            'feed_rate_variation': {
                'description': 'Feed rate variations (cyclical + random)',
                'affected_params': ['kiln_temperature', 'coal_feed_rate', 'draft_pressure'],
                'probability': 0.25,
                'intensity_range': (0.05, 0.15)  # 5-15% variation
            },
            'fuel_quality_change': {
                'description': 'Fuel quality degradation/improvement',
                'affected_params': ['coal_feed_rate', 'kiln_temperature', 'heat_consumption'],
                'probability': 0.15,
                'intensity_range': (0.03, 0.12)  # 3-12% variation
            },
            'equipment_degradation': {
                'description': 'Equipment wear/degradation effects',
                'affected_params': ['kiln_speed', 'raw_mill_fineness', 'cement_mill_fineness'],
                'probability': 0.20,
                'intensity_range': (0.02, 0.08)  # 2-8% degradation
            },
            'raw_material_composition_shift': {
                'description': 'Raw material chemistry variation',
                'affected_params': ['LSF', 'SM', 'AM', 'C3S', 'C2S', 'C3A', 'C4AF'],
                'probability': 0.30,
                'intensity_range': (0.01, 0.05)  # 1-5% chemistry shift
            },
            'maintenance_mode': {
                'description': 'Reduced operation during maintenance',
                'affected_params': ['kiln_temperature', 'kiln_speed', 'coal_feed_rate'],
                'probability': 0.08,
                'intensity_range': (0.10, 0.25)  # 10-25% reduction
            }
        }
        
        # Seasonal effects
        self.seasonal_effects = {
            'temperature_seasonal': {
                'amplitude': 0.02,  # 2% seasonal variation
                'period': 365,  # yearly cycle
                'phase': 0
            },
            'humidity_seasonal': {
                'amplitude': 0.03,  # 3% seasonal variation  
                'period': 365,
                'phase': 90  # offset from temperature
            }
        }
        
        print(f"🔧 ProcessDisturbanceSimulator initialized with {len(self.base_dataset)} base samples")
        print(f"✓ {len(self.disturbance_scenarios)} disturbance scenarios configured")

    def generate_feed_rate_disturbance(self, n_samples: int, intensity: float) -> np.ndarray:
        """Generate realistic feed rate variations (cyclical + random)"""
        # Combine multiple frequency components
        time_series = np.arange(n_samples)
        
        # Primary cycle (shift pattern)
        primary_cycle = np.sin(2 * np.pi * time_series / 480) * intensity  # 8-hour cycle
        
        # Secondary cycle (daily variation)
        secondary_cycle = np.sin(2 * np.pi * time_series / 1440) * intensity * 0.6  # 24-hour cycle
        
        # Random walk component
        random_walk = np.cumsum(np.random.normal(0, intensity * 0.3, n_samples))
        random_walk = (random_walk - np.mean(random_walk)) / np.std(random_walk) * intensity * 0.4
        
        # High-frequency noise
        noise = np.random.normal(0, intensity * 0.2, n_samples)
        
        return primary_cycle + secondary_cycle + random_walk + noise

    def generate_fuel_quality_disturbance(self, n_samples: int, intensity: float) -> np.ndarray:
        """Generate fuel quality change disturbance (step changes + drift)"""
        disturbance = np.zeros(n_samples)
        
        # Number of step changes (fuel shipment changes)
        n_steps = max(1, int(n_samples / 500))  # Change every ~500 samples
        step_points = np.random.choice(n_samples, n_steps, replace=False)
        step_points = np.sort(step_points)
        
        current_level = 0
        for i, step_point in enumerate(step_points):
            # Step change in fuel quality
            step_change = np.random.uniform(-intensity, intensity)
            if i == 0:
                disturbance[:step_point] = current_level
            else:
                disturbance[step_points[i-1]:step_point] = current_level
            current_level += step_change
            
        # Fill remaining samples
        if len(step_points) > 0:
            disturbance[step_points[-1]:] = current_level
        else:
            disturbance[:] = np.random.uniform(-intensity, intensity)
            
        # Add gradual drift
        drift = np.linspace(0, np.random.uniform(-intensity/2, intensity/2), n_samples)
        
        return disturbance + drift

    def generate_equipment_degradation(self, n_samples: int, intensity: float) -> np.ndarray:
        """Generate equipment degradation pattern (progressive + episodic)"""
        # Progressive degradation (monotonic)
        progressive = np.linspace(0, -intensity, n_samples)  # Always degrading
        
        # Episodic degradation events (sudden changes)
        episodic = np.zeros(n_samples)
        n_events = max(1, int(n_samples / 800))  # Events every ~800 samples
        
        for _ in range(n_events):
            event_start = np.random.randint(0, n_samples - 100)
            event_duration = np.random.randint(50, 200)
            event_end = min(event_start + event_duration, n_samples)
            
            # Degradation event with recovery
            event_intensity = np.random.uniform(intensity * 0.5, intensity * 1.5)
            event_profile = -event_intensity * np.exp(-np.linspace(0, 3, event_end - event_start))
            episodic[event_start:event_end] += event_profile
            
        # Maintenance recovery events (partial restoration)
        n_maintenance = max(1, int(n_samples / 1200))
        for _ in range(n_maintenance):
            maint_point = np.random.randint(n_samples // 4, n_samples)
            recovery = np.random.uniform(0.3, 0.7) * intensity  # Partial recovery
            progressive[maint_point:] += recovery
            
        return progressive + episodic

    def generate_raw_material_shift(self, n_samples: int, intensity: float) -> np.ndarray:
        """Generate raw material composition shifts (quarry face changes)"""
        # Piecewise constant with transitions
        n_segments = max(2, int(n_samples / 300))  # New quarry face every ~300 samples
        segment_lengths = np.random.multinomial(n_samples, np.ones(n_segments) / n_segments)
        
        disturbance = np.zeros(n_samples)
        current_pos = 0
        current_level = 0
        
        for segment_length in segment_lengths:
            if segment_length == 0:
                continue
                
            # New quarry face composition
            new_level = np.random.uniform(-intensity, intensity)
            
            # Smooth transition
            transition_length = min(50, segment_length // 3)
            if transition_length > 0:
                transition = np.linspace(current_level, new_level, transition_length)
                disturbance[current_pos:current_pos + transition_length] = transition
                disturbance[current_pos + transition_length:current_pos + segment_length] = new_level
            else:
                disturbance[current_pos:current_pos + segment_length] = new_level
                
            current_pos += segment_length
            current_level = new_level
            
            if current_pos >= n_samples:
                break
                
        return disturbance

    def generate_maintenance_mode(self, n_samples: int, intensity: float) -> np.ndarray:
        """Generate maintenance mode disturbances (scheduled shutdowns)"""
        disturbance = np.zeros(n_samples)
        
        # Scheduled maintenance events
        n_maintenance = max(1, int(n_samples / 2000))  # Major maintenance every ~2000 samples
        
        for _ in range(n_maintenance):
            maint_start = np.random.randint(0, n_samples - 200)
            maint_duration = np.random.randint(50, 150)  # 50-150 sample maintenance
            maint_end = min(maint_start + maint_duration, n_samples)
            
            # Ramp down, maintenance, ramp up
            ramp_duration = maint_duration // 4
            
            # Ramp down
            ramp_down = np.linspace(0, -intensity, ramp_duration)
            # Full maintenance
            full_maint = np.ones(maint_duration - 2 * ramp_duration) * (-intensity)
            # Ramp up
            ramp_up = np.linspace(-intensity, 0, ramp_duration)
            
            maintenance_profile = np.concatenate([ramp_down, full_maint, ramp_up])
            actual_length = min(len(maintenance_profile), maint_end - maint_start)
            disturbance[maint_start:maint_start + actual_length] = maintenance_profile[:actual_length]
            
        return disturbance

    def apply_seasonal_effects(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply seasonal variations to the data"""
        seasonal_data = data.copy()
        n_samples = len(data)
        
        # Generate time index (assuming daily samples)
        day_of_year = np.arange(n_samples) % 365
        
        for param_name, effect in self.seasonal_effects.items():
            amplitude = effect['amplitude']
            period = effect['period']
            phase = effect['phase']
            
            # Generate seasonal variation
            seasonal_variation = amplitude * np.sin(2 * np.pi * (day_of_year + phase) / period)
            
            # Apply to temperature-sensitive parameters
            if 'temperature' in param_name.lower():
                affected_cols = ['kiln_temperature', 'heat_consumption']
            else:  # humidity effects
                affected_cols = ['raw_mill_fineness', 'cement_mill_fineness']
                
            for col in affected_cols:
                if col in seasonal_data.columns:
                    baseline = seasonal_data[col].mean()
                    seasonal_data[col] *= (1 + seasonal_variation)
                    
        return seasonal_data

# Initialize the disturbance simulator
disturbance_sim = ProcessDisturbanceSimulator(cement_dataset)

print(f"\n✅ Process Disturbance Simulator Ready!")
print(f"📊 Ready to simulate realistic plant upset conditions")