"""High-frequency DCS data simulator for cement plant digital twin."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CementPlantDCSSimulator:
    """
    Generates high-frequency DCS (Distributed Control System) data for a cement plant.
    
    This simulator creates realistic, correlated time-series data that mimics
    the output of a real cement plant's control system.
    """
    
    def __init__(self, config_path: str = 'config/plant_config.yml'):
        """Initialize the simulator with plant configuration."""
        self.config = self._load_config(config_path)
        self.tags = self._define_dcs_tags()
        self.process_correlations = self._define_process_correlations()
        
    def _load_config(self, config_path: str) -> dict:
        """Load plant configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded plant configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default plant configuration."""
        return {
            'plant': {
                'capacity_tpd': 10000,
                'kiln_type': 'Dry Process'
            },
            'dcs_tags': {
                'update_frequencies': {
                    'critical_loops': 1,
                    'process_variables': 5,
                    'quality_lab': 3600
                }
            }
        }
    
    def _define_dcs_tags(self) -> Dict[str, Tuple[float, float, str]]:
        """
        Define DCS tags with their normal operating ranges and units.
        
        Returns:
            Dict mapping tag names to (min_value, max_value, unit)
        """
        return {
            # Raw Mill Section
            'raw_mill_motor_power_kw': (1800, 2200, 'kW'),
            'raw_mill_feed_rate_tph': (190, 210, 'tph'),
            'raw_mill_outlet_temp_c': (80, 120, '°C'),
            'raw_mill_separator_speed_rpm': (800, 1200, 'rpm'),
            'raw_mill_fineness_blaine_cm2_g': (3000, 3500, 'cm²/g'),
            
            # Preheater Tower
            'preheater_stage1_temp_c': (850, 950, '°C'),
            'preheater_stage2_temp_c': (750, 850, '°C'),
            'preheater_stage3_temp_c': (650, 750, '°C'),
            'preheater_stage4_temp_c': (550, 650, '°C'),
            'preheater_stage5_temp_c': (450, 550, '°C'),
            'preheater_pressure_drop_mbar': (15, 25, 'mbar'),
            'calciner_temp_c': (900, 1000, '°C'),
            'calciner_o2_pct': (2.5, 4.0, '%'),
            
            # Rotary Kiln
            'kiln_feed_rate_tph': (190, 210, 'tph'),
            'kiln_speed_rpm': (3.0, 4.5, 'rpm'),
            'burning_zone_temp_c': (1420, 1480, '°C'),
            'kiln_torque_motor_pct': (75, 90, '%'),
            'hood_pressure_mbar': (-2.5, -1.5, 'mbar'),
            'kiln_exhaust_temp_c': (350, 450, '°C'),
            'kiln_exhaust_o2_pct': (2.5, 4.0, '%'),
            'kiln_exhaust_co_pct': (0.1, 0.5, '%'),
            
            # Fuel System
            'coal_flow_tph': (45, 55, 'tph'),
            'petcoke_flow_tph': (30, 37, 'tph'),
            'alternative_fuel_flow_tph': (7, 10, 'tph'),
            'total_fuel_flow_tph': (82, 102, 'tph'),
            
            # Clinker Cooler
            'cooler_speed_rpm': (2.0, 4.0, 'rpm'),
            'cooler_outlet_temp_c': (80, 120, '°C'),
            'cooler_fan_speed_rpm': (1000, 1500, 'rpm'),
            'cooler_pressure_mbar': (5, 15, 'mbar'),
            
            # Cement Mill
            'cement_mill_power_kw': (2000, 3000, 'kW'),
            'cement_mill_feed_rate_tph': (80, 120, 'tph'),
            'cement_mill_outlet_temp_c': (90, 110, '°C'),
            'cement_fineness_blaine_cm2_g': (3200, 3800, 'cm²/g'),
            
            # Environmental Monitoring
            'nox_mg_nm3': (300, 800, 'mg/Nm³'),
            'so2_mg_nm3': (50, 200, 'mg/Nm³'),
            'dust_mg_nm3': (10, 30, 'mg/Nm³'),
            'co2_pct': (18, 22, '%'),
            'co2_kg_t': (750, 850, 'kg/t'),
            
            # Quality Laboratory (updated less frequently)
            'free_lime_pct': (1.0, 2.0, '%'),
            'c3s_content_pct': (55, 65, '%'),
            'c2s_content_pct': (10, 20, '%'),
            'compressive_strength_3d_mpa': (15, 25, 'MPa'),
            'compressive_strength_7d_mpa': (25, 35, 'MPa'),
            'compressive_strength_28d_mpa': (40, 55, 'MPa'),
            'setting_time_initial_min': (120, 180, 'min'),
            'setting_time_final_min': (180, 240, 'min'),
            'soundness_mm': (0, 5, 'mm'),
        }
    
    def _define_process_correlations(self) -> Dict[str, str]:
        """
        Define process correlations between different DCS tags.
        
        Returns:
            Dict mapping dependent variables to their correlation expressions
        """
        return {
            # Kiln temperature affects NOx formation
            'nox_mg_nm3': 'burning_zone_temp_c * 0.4 + kiln_exhaust_o2_pct * 50',
            
            # Kiln feed rate affects burning zone temperature
            'burning_zone_temp_c': 'kiln_feed_rate_tph * 2 + calciner_temp_c * 0.8',
            
            # Free lime inversely related to burning zone temperature
            'free_lime_pct': '2.5 - (burning_zone_temp_c - 1400) / 20',
            
            # C3S content related to burning zone temperature
            'c3s_content_pct': '50 + (burning_zone_temp_c - 1400) / 2',
            
            # Compressive strength related to C3S content and fineness
            'compressive_strength_28d_mpa': 'c3s_content_pct * 0.8 + cement_fineness_blaine_cm2_g / 100',
            
            # Preheater temperatures follow decreasing pattern
            'preheater_stage2_temp_c': 'preheater_stage1_temp_c - 100',
            'preheater_stage3_temp_c': 'preheater_stage2_temp_c - 100',
            'preheater_stage4_temp_c': 'preheater_stage3_temp_c - 100',
            'preheater_stage5_temp_c': 'preheater_stage4_temp_c - 100',
            
            # Fuel system correlations
            'total_fuel_flow_tph': 'coal_flow_tph + petcoke_flow_tph + alternative_fuel_flow_tph',
            
            # Fuel flow affects burning zone temperature
            'burning_zone_temp_c': 'kiln_feed_rate_tph * 2 + calciner_temp_c * 0.8 + total_fuel_flow_tph * 0.5',
        }
    
    def generate_dcs_data(self, 
                         duration_hours: int = 24, 
                         sample_rate_seconds: int = 1,
                         include_disturbances: bool = True) -> pd.DataFrame:
        """
        Generate high-frequency DCS data for the specified duration.
        
        Args:
            duration_hours: Duration of data to generate
            sample_rate_seconds: Sampling rate in seconds
            include_disturbances: Whether to include process disturbances
            
        Returns:
            DataFrame with timestamp index and DCS tag columns
        """
        logger.info(f"Generating {duration_hours} hours of DCS data at {sample_rate_seconds}s interval...")
        
        num_samples = int(duration_hours * 3600 / sample_rate_seconds)
        timestamps = pd.date_range(
            start='2024-01-01 00:00:00',
            periods=num_samples,
            freq=f'{sample_rate_seconds}s'
        )
        
        df = pd.DataFrame(index=timestamps)
        
        # Generate base signals for each tag
        for tag, (min_val, max_val, unit) in self.tags.items():
            df[tag] = self._generate_tag_signal(
                num_samples, min_val, max_val, tag, include_disturbances
            )
        
        # Apply process correlations
        df = self._apply_process_correlations(df)
        
        # Simulate different update frequencies
        df = self._apply_update_frequencies(df, sample_rate_seconds)
        
        # Add realistic noise and ensure bounds
        df = self._add_realistic_noise(df)
        
        logger.info(f"Generated DataFrame with {len(df)} rows and {len(df.columns)} DCS tags")
        return df
    
    def _generate_tag_signal(self, 
                           num_samples: int, 
                           min_val: float, 
                           max_val: float, 
                           tag: str,
                           include_disturbances: bool) -> np.ndarray:
        """Generate a realistic signal for a specific DCS tag."""
        
        # Base signal components
        base_signal = np.zeros(num_samples)
        
        # 1. Daily cycle (24-hour pattern)
        daily_cycle = np.sin(np.linspace(0, 2 * np.pi, num_samples)) * 0.1
        
        # 2. Random walk for process variability
        random_walk = np.cumsum(np.random.normal(0, 0.02, num_samples))
        
        # 3. High-frequency noise
        noise = np.random.normal(0, 0.05, num_samples)
        
        # 4. Process disturbances (if enabled)
        if include_disturbances:
            disturbances = self._generate_disturbances(num_samples, tag)
        else:
            disturbances = np.zeros(num_samples)
        
        # Combine all components
        base_signal = daily_cycle + random_walk + noise + disturbances
        
        # Scale to the tag's operating range
        signal_range = max_val - min_val
        if base_signal.max() > base_signal.min():
            scaled_signal = min_val + (base_signal - base_signal.min()) / (base_signal.max() - base_signal.min()) * signal_range
        else:
            scaled_signal = np.full_like(base_signal, (min_val + max_val) / 2)
        
        return scaled_signal
    
    def _generate_disturbances(self, num_samples: int, tag: str) -> np.ndarray:
        """Generate realistic process disturbances."""
        disturbances = np.zeros(num_samples)
        
        # Random disturbance events (1-3 per day)
        num_disturbances = np.random.poisson(2)
        
        for _ in range(num_disturbances):
            # Random start time (ensure we have enough samples)
            max_start = max(0, num_samples - 3600)
            if max_start <= 0:
                continue
            start_idx = np.random.randint(0, max_start)
            
            # Disturbance duration (5-30 minutes)
            duration_minutes = np.random.randint(5, 30)
            duration_samples = duration_minutes * 60
            
            # Ensure we don't exceed available samples
            max_duration = num_samples - start_idx
            duration_samples = min(duration_samples, max_duration)
            
            if duration_samples <= 0:
                continue
            
            # Disturbance magnitude (5-20% of normal range)
            magnitude = np.random.uniform(0.05, 0.2)
            
            # Create disturbance profile (ramp up, hold, ramp down)
            ramp_samples = max(1, duration_samples // 4)
            hold_samples = max(1, duration_samples // 2)
            
            # Ensure we don't exceed duration_samples
            total_profile_samples = min(3 * ramp_samples + hold_samples, duration_samples)
            
            disturbance_profile = np.concatenate([
                np.linspace(0, magnitude, ramp_samples),  # Ramp up
                np.full(hold_samples, magnitude),         # Hold
                np.linspace(magnitude, 0, ramp_samples),  # Ramp down
                np.zeros(max(0, duration_samples - total_profile_samples))  # Pad
            ])
            
            # Apply disturbance
            end_idx = min(start_idx + len(disturbance_profile), num_samples)
            disturbances[start_idx:end_idx] += disturbance_profile[:end_idx-start_idx]
        
        return disturbances
    
    def _apply_process_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply process correlations between different tags."""
        for dependent_var, correlation_expr in self.process_correlations.items():
            if dependent_var in df.columns:
                try:
                    # Evaluate the correlation expression
                    df[dependent_var] = df.eval(correlation_expr)
                except Exception as e:
                    logger.warning(f"Failed to apply correlation for {dependent_var}: {e}")
        
        return df
    
    def _apply_update_frequencies(self, df: pd.DataFrame, sample_rate_seconds: int) -> pd.DataFrame:
        """Apply different update frequencies for different types of tags based on configuration."""
        
        # Get update frequencies from configuration
        update_freq = self.config.get('dcs_tags', {}).get('update_frequencies', {})
        critical_freq = update_freq.get('critical_loops', 1)
        process_freq = update_freq.get('process_variables', 5)
        quality_freq = update_freq.get('quality_lab', 3600)
        
        # Define tag categories based on configuration
        critical_tags = [
            'burning_zone_temp_c', 'kiln_speed_rpm', 'kiln_torque_motor_pct',
            'preheater_stage1_temp_c', 'calciner_temp_c', 'coal_flow_tph',
            'petcoke_flow_tph', 'alternative_fuel_flow_tph'
        ]
        
        quality_tags = [
            'free_lime_pct', 'c3s_content_pct', 'c2s_content_pct',
            'compressive_strength_3d_mpa', 'compressive_strength_7d_mpa', 
            'compressive_strength_28d_mpa', 'setting_time_initial_min',
            'setting_time_final_min', 'soundness_mm'
        ]
        
        # Apply different sampling rates based on configuration
        for tag in df.columns:
            if tag in critical_tags:
                # Critical loops: configured frequency
                df[tag] = df[tag].resample(f'{critical_freq}S').first().reindex(df.index).ffill()
            elif tag in quality_tags:
                # Quality lab: configured frequency
                df[tag] = df[tag].resample(f'{quality_freq}S').first().reindex(df.index).ffill()
            else:
                # Process variables: configured frequency
                df[tag] = df[tag].resample(f'{process_freq}S').first().reindex(df.index).ffill()
        
        return df
    
    def _add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic noise and ensure all values are within bounds."""
        for tag, (min_val, max_val, unit) in self.tags.items():
            if tag in df.columns:
                # Add small amount of noise
                noise = np.random.normal(0, (max_val - min_val) * 0.01, len(df))
                df[tag] += noise
                
                # Ensure values stay within realistic bounds
                df[tag] = np.clip(df[tag], min_val * 0.9, max_val * 1.1)
        
        return df
    
    def save_dcs_data(self, df: pd.DataFrame, output_path: str) -> bool:
        """Save DCS data to CSV file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path)
            logger.info(f"DCS data saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save DCS data: {e}")
            return False


def generate_dcs_data(duration_hours: int = 24, 
                     sample_rate_seconds: int = 1,
                     output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to generate DCS data.
    
    Args:
        duration_hours: Duration of data to generate
        sample_rate_seconds: Sampling rate in seconds
        output_path: Optional path to save the data
        
    Returns:
        DataFrame with DCS data
    """
    simulator = CementPlantDCSSimulator()
    df = simulator.generate_dcs_data(duration_hours, sample_rate_seconds)
    
    if output_path:
        simulator.save_dcs_data(df, output_path)
    
    return df


if __name__ == "__main__":
    # Generate 24 hours of DCS data and save it
    logger.info("Starting DCS data generation...")
    
    # Generate high-frequency data (1-second intervals)
    dcs_data = generate_dcs_data(
        duration_hours=24,
        sample_rate_seconds=1,
        output_path='data/processed/simulated_dcs_data.csv'
    )
    
    logger.info(f"Generated {len(dcs_data)} records with {len(dcs_data.columns)} DCS tags")
    logger.info(f"Data covers {dcs_data.index[0]} to {dcs_data.index[-1]}")
    
    # Show sample statistics
    logger.info("\nSample DCS Data Statistics:")
    logger.info(dcs_data.describe().round(2))
