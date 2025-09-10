"""TimeGAN data augmentation pipeline for generating massive datasets."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ydata_synthetic.synthesizers.timeseries.timegan.model import TimeGAN
    TIMEGAN_AVAILABLE = True
    logger.info("✅ TimeGAN imported successfully")
except ImportError:
    TIMEGAN_AVAILABLE = False
    logger.warning("TimeGAN not available. Using statistical augmentation instead.")


class CementPlantDataGenerator:
    """
    Generates massive, realistic cement plant datasets using TimeGAN augmentation.
    
    This class takes a small base dataset (from DWSIM simulation) and uses TimeGAN
    to generate a much larger, diverse dataset that includes normal operations,
    startups, shutdowns, and rare fault conditions.
    """
    
    def __init__(self, base_data_path: Optional[str] = None):
        """
        Initialize the data generator.
        
        Args:
            base_data_path: Path to base dataset (if None, will generate sample data)
        """
        self.base_data_path = base_data_path
        self.timegan_model = None
        self.scaler = None
        
    def prepare_base_data(self, dcs_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare base DCS data for TimeGAN training.
        
        Args:
            dcs_data: Base DCS data from simulation
            
        Returns:
            Prepared data ready for TimeGAN
        """
        logger.info("Preparing base data for TimeGAN training...")
        
        # Select numerical columns only
        numerical_cols = dcs_data.select_dtypes(include=[np.number]).columns
        prepared_data = dcs_data[numerical_cols].copy()
        
        # Handle missing values
        prepared_data = prepared_data.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize data for TimeGAN
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        prepared_data_scaled = pd.DataFrame(
            self.scaler.fit_transform(prepared_data),
            columns=prepared_data.columns,
            index=prepared_data.index
        )
        
        logger.info(f"Prepared {len(prepared_data)} records with {len(prepared_data.columns)} features")
        return prepared_data_scaled
    
    def train_timegan(self, prepared_data: pd.DataFrame, 
                     epochs: int = 100,
                     batch_size: int = 32) -> bool:
        """
        Train TimeGAN model on prepared data.
        
        Args:
            prepared_data: Prepared and scaled data
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            True if training successful, False otherwise
        """
        if not TIMEGAN_AVAILABLE:
            logger.warning("TimeGAN not available. Skipping training.")
            return False
        
        logger.info(f"Training TimeGAN model for {epochs} epochs...")
        
        try:
            # Initialize TimeGAN model
            from ydata_synthetic.synthesizers.base import ModelParameters, TrainParameters
            
            model_params = ModelParameters(
                hidden_dim=24,
                num_layers=3,
                seq_len=10,
                batch_size=batch_size
            )
            
            train_params = TrainParameters(
                epochs=epochs,
                batch_size=batch_size
            )
            
            self.timegan_model = TimeGAN(model_params, train_params)
            
            # Train the model
            self.timegan_model.fit(prepared_data)
            
            logger.info("TimeGAN training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"TimeGAN training failed: {e}")
            return False
    
    def generate_augmented_data(self, 
                               num_samples: int,
                               duration_hours: int = 8760) -> pd.DataFrame:
        """
        Generate augmented dataset using TimeGAN.
        
        Args:
            num_samples: Number of samples to generate
            duration_hours: Duration to simulate (default: 1 year)
            
        Returns:
            Generated dataset
        """
        if self.timegan_model is None:
            logger.warning("TimeGAN model not trained. Using statistical augmentation.")
            return self._generate_statistical_data(num_samples, duration_hours)
        
        logger.info(f"Generating {num_samples} samples using TimeGAN...")
        
        try:
            # Generate synthetic data
            synthetic_data = self.timegan_model.sample(num_samples)
            
            # Inverse transform to original scale
            if self.scaler is not None:
                synthetic_data = pd.DataFrame(
                    self.scaler.inverse_transform(synthetic_data),
                    columns=self.scaler.feature_names_in_
                )
            
            # Create realistic timestamps
            timestamps = pd.date_range(
                start='2024-01-01 00:00:00',
                periods=num_samples,
                freq='1S'
            )
            synthetic_data.index = timestamps
            
            logger.info(f"Generated {len(synthetic_data)} synthetic records")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"TimeGAN generation failed: {e}")
            return self._generate_statistical_data(num_samples, duration_hours)
    
    def _generate_statistical_data(self, 
                                  num_samples: int,
                                  duration_hours: int) -> pd.DataFrame:
        """
        Generate data using statistical methods when TimeGAN is not available.
        
        Args:
            num_samples: Number of samples to generate
            duration_hours: Duration to simulate
            
        Returns:
            Generated dataset
        """
        logger.info(f"Generating {num_samples} samples using statistical methods...")
        
        # Define realistic ranges for cement plant parameters
        parameter_ranges = {
            'raw_mill_motor_power_kw': (1800, 2200),
            'raw_mill_feed_rate_tph': (190, 210),
            'raw_mill_outlet_temp_c': (80, 120),
            'preheater_stage1_temp_c': (850, 950),
            'preheater_stage2_temp_c': (750, 850),
            'preheater_stage3_temp_c': (650, 750),
            'preheater_stage4_temp_c': (550, 650),
            'preheater_stage5_temp_c': (450, 550),
            'kiln_feed_rate_tph': (190, 210),
            'kiln_speed_rpm': (3.0, 4.5),
            'burning_zone_temp_c': (1420, 1480),
            'kiln_torque_motor_pct': (75, 90),
            'hood_pressure_mbar': (-2.5, -1.5),
            'kiln_exhaust_temp_c': (350, 450),
            'kiln_exhaust_o2_pct': (2.5, 4.0),
            'cooler_speed_rpm': (2.0, 4.0),
            'cooler_outlet_temp_c': (80, 120),
            'cement_mill_power_kw': (2000, 3000),
            'cement_mill_feed_rate_tph': (80, 120),
            'cement_fineness_blaine_cm2_g': (3200, 3800),
            'nox_mg_nm3': (300, 800),
            'so2_mg_nm3': (50, 200),
            'dust_mg_nm3': (10, 30),
            'co2_pct': (18, 22),
            'free_lime_pct': (1.0, 2.0),
            'c3s_content_pct': (55, 65),
            'c2s_content_pct': (10, 20),
            'compressive_strength_28d_mpa': (40, 55),
        }
        
        # Generate timestamps
        timestamps = pd.date_range(
            start='2024-01-01 00:00:00',
            periods=num_samples,
            freq='1S'
        )
        
        # Generate data with realistic correlations
        data = {}
        
        # Base signals with daily cycles
        t = np.linspace(0, duration_hours * 2 * np.pi / 24, num_samples)
        
        for param, (min_val, max_val) in parameter_ranges.items():
            # Create base signal with daily cycle
            base_signal = np.sin(t) * 0.1
            
            # Add random walk for process variability
            random_walk = np.cumsum(np.random.normal(0, 0.02, num_samples))
            
            # Add noise
            noise = np.random.normal(0, 0.05, num_samples)
            
            # Combine and scale
            signal = base_signal + random_walk + noise
            signal_range = max_val - min_val
            scaled_signal = min_val + (signal - signal.min()) / (signal.max() - signal.min()) * signal_range
            
            data[param] = scaled_signal
        
        # Apply process correlations
        df = pd.DataFrame(data, index=timestamps)
        
        # Kiln temperature affects NOx formation
        df['nox_mg_nm3'] = df['burning_zone_temp_c'] * 0.4 + df['kiln_exhaust_o2_pct'] * 50
        
        # Free lime inversely related to burning zone temperature
        df['free_lime_pct'] = 2.5 - (df['burning_zone_temp_c'] - 1400) / 20
        
        # C3S content related to burning zone temperature
        df['c3s_content_pct'] = 50 + (df['burning_zone_temp_c'] - 1400) / 2
        
        # Compressive strength related to C3S content and fineness
        df['compressive_strength_28d_mpa'] = (df['c3s_content_pct'] * 0.8 + 
                                             df['cement_fineness_blaine_cm2_g'] / 100)
        
        # Ensure realistic bounds
        for param, (min_val, max_val) in parameter_ranges.items():
            if param in df.columns:
                df[param] = np.clip(df[param], min_val * 0.9, max_val * 1.1)
        
        logger.info(f"Generated {len(df)} records with {len(df.columns)} parameters")
        return df
    
    def add_operational_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add realistic operational scenarios to the dataset.
        
        Args:
            df: Base dataset
            
        Returns:
            Dataset with added scenarios
        """
        logger.info("Adding operational scenarios...")
        
        # Create scenarios with non-overlapping assignment
        scenarios = ['normal'] * len(df)  # Start with all normal
        
        # Randomly assign scenarios to ensure exact length match
        indices = np.random.choice(len(df), size=int(0.2 * len(df)), replace=False)
        
        # Assign startup (5%)
        startup_count = int(0.05 * len(df))
        startup_indices = indices[:startup_count]
        for idx in startup_indices:
            scenarios[idx] = 'startup'
        
        # Assign shutdown (5%)
        shutdown_count = int(0.05 * len(df))
        shutdown_indices = indices[startup_count:startup_count + shutdown_count]
        for idx in shutdown_indices:
            scenarios[idx] = 'shutdown'
        
        # Assign disturbance (10%)
        disturbance_count = int(0.1 * len(df))
        disturbance_indices = indices[startup_count + shutdown_count:startup_count + shutdown_count + disturbance_count]
        for idx in disturbance_indices:
            scenarios[idx] = 'disturbance'
        
        # Add scenario column
        df['operational_scenario'] = scenarios
        
        # Apply scenario-specific modifications
        self._apply_scenario_modifications(df)
        
        logger.info("Operational scenarios added successfully")
        return df
    
    def _apply_scenario_modifications(self, df: pd.DataFrame):
        """Apply modifications based on operational scenarios."""
        
        # Startup scenario - lower temperatures, higher energy consumption
        startup_mask = df['operational_scenario'] == 'startup'
        df.loc[startup_mask, 'burning_zone_temp_c'] *= 0.9
        df.loc[startup_mask, 'raw_mill_motor_power_kw'] *= 1.2
        
        # Shutdown scenario - decreasing temperatures
        shutdown_mask = df['operational_scenario'] == 'shutdown'
        df.loc[shutdown_mask, 'burning_zone_temp_c'] *= 0.8
        df.loc[shutdown_mask, 'kiln_speed_rpm'] *= 0.5
        
        # Disturbance scenario - increased variability
        disturbance_mask = df['operational_scenario'] == 'disturbance'
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'operational_scenario':
                noise = np.random.normal(0, 0.1, disturbance_mask.sum())
                df.loc[disturbance_mask, col] *= (1 + noise)
    
    def save_generated_data(self, df: pd.DataFrame, output_path: str) -> bool:
        """Save generated data to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path)
            logger.info(f"Generated data saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save generated data: {e}")
            return False


def generate_massive_dataset(base_data_path: Optional[str] = None,
                           num_samples: int = 100000,
                           duration_hours: int = 8760,
                           output_path: str = 'data/processed/massive_cement_dataset.csv') -> pd.DataFrame:
    """
    Generate a massive cement plant dataset using TimeGAN augmentation.
    
    Args:
        base_data_path: Path to base dataset
        num_samples: Number of samples to generate
        duration_hours: Duration to simulate
        output_path: Output file path
        
    Returns:
        Generated dataset
    """
    logger.info("=== Starting Massive Dataset Generation ===")
    
    # Initialize data generator
    generator = CementPlantDataGenerator(base_data_path)
    
    # Load or generate base data
    if base_data_path and os.path.exists(base_data_path):
        logger.info(f"Loading base data from {base_data_path}")
        base_data = pd.read_csv(base_data_path, index_col=0, parse_dates=True)
    else:
        logger.info("Generating base data using DCS simulator")
        from ..simulation.dcs_simulator import generate_dcs_data
        base_data = generate_dcs_data(duration_hours=24, sample_rate_seconds=1)
    
    # Prepare data for TimeGAN
    prepared_data = generator.prepare_base_data(base_data)
    
    # Train TimeGAN (if available)
    generator.train_timegan(prepared_data, epochs=50)
    
    # Generate augmented data
    augmented_data = generator.generate_augmented_data(num_samples, duration_hours)
    
    # Add operational scenarios
    final_data = generator.add_operational_scenarios(augmented_data)
    
    # Save the dataset
    generator.save_generated_data(final_data, output_path)
    
    logger.info(f"=== Massive Dataset Generation Complete ===")
    logger.info(f"Generated {len(final_data)} records with {len(final_data.columns)} features")
    logger.info(f"Data covers {final_data.index[0]} to {final_data.index[-1]}")
    
    return final_data


if __name__ == "__main__":
    # Generate massive dataset
    massive_data = generate_massive_dataset(
        num_samples=100000,
        duration_hours=8760,  # 1 year
        output_path='data/processed/massive_cement_dataset.csv'
    )
    
    # Show statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total records: {len(massive_data):,}")
    logger.info(f"Total features: {len(massive_data.columns)}")
    logger.info(f"Memory usage: {massive_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Show scenario distribution
    if 'operational_scenario' in massive_data.columns:
        scenario_counts = massive_data['operational_scenario'].value_counts()
        logger.info("\nOperational Scenario Distribution:")
        for scenario, count in scenario_counts.items():
            percentage = count / len(massive_data) * 100
            logger.info(f"  {scenario}: {count:,} ({percentage:.1f}%)")
    
    logger.info("Massive dataset generation completed successfully!")
