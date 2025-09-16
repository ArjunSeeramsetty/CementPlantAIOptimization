"""
DWSIM Physics-Based Simulation for Demo Pipeline
Generates realistic process simulation data using existing DCS simulator
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.dcs_simulator import CementPlantDCSSimulator
from data_sourcing.real_world_integrator import RealWorldDataIntegrator

logger = logging.getLogger(__name__)

def simulate_process(n_hours: int = 24, sample_rate_sec: int = 60) -> pd.DataFrame:
    """
    Generate physics-based process simulation data using DCS simulator
    
    Args:
        n_hours: Duration of simulation in hours
        sample_rate_sec: Sample rate in seconds
        
    Returns:
        Simulated process data DataFrame
    """
    logger.info(f"🔄 Starting DWSIM physics simulation for {n_hours} hours...")
    
    try:
        # Initialize real-world data integrator for calibration
        integrator = RealWorldDataIntegrator()
        
        # Get calibrated parameters
        integration_results = integrator.integrate_real_world_data(use_config_plant=True)
        calibrated_params = integration_results['calibrated_params']
        
        # Initialize DCS simulator
        dcs_simulator = CementPlantDCSSimulator()
        
        # Generate calibrated simulation data
        logger.info(f"🔄 Generating {n_hours} hours of calibrated DCS data at {sample_rate_sec}s interval...")
        
        df = dcs_simulator.generate_calibrated_data(
            calibrated_params=calibrated_params,
            duration_hours=n_hours,
            sample_rate_seconds=sample_rate_sec
        )
        
        # Add simulation metadata
        df['simulation_type'] = 'physics_based'
        df['simulation_duration_hours'] = n_hours
        df['sample_rate_seconds'] = sample_rate_sec
        df['simulation_timestamp'] = datetime.now().isoformat()
        
        # Create demo data directory
        os.makedirs("demo/data/physics", exist_ok=True)
        
        # Save physics simulation data
        path = "demo/data/physics/dwsim_physics.csv"
        df.to_csv(path, index=False)
        logger.info(f"✅ Saved physics simulation data: {path} ({len(df)} rows)")
        
        # Create additional physics datasets for demo
        create_additional_physics_datasets(df, n_hours)
        
        logger.info(f"✅ DWSIM physics simulation completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"❌ DWSIM simulation failed: {e}")
        raise

def create_additional_physics_datasets(df: pd.DataFrame, n_hours: int):
    """
    Create additional physics datasets for demo pipeline
    
    Args:
        df: Main physics simulation DataFrame
        n_hours: Simulation duration
    """
    logger.info("🔄 Creating additional physics datasets...")
    
    # Add timestamp column if missing
    if 'timestamp' not in df.columns:
        logger.info("🔄 Adding timestamp column to physics data...")
        timestamps = pd.date_range(
            start="2025-09-17 00:00", 
            periods=len(df), 
            freq="T"  # Minute frequency
        )
        df['timestamp'] = timestamps
    
    # Create process variables dataset
    process_columns = [col for col in df.columns if col not in ['simulation_type', 'simulation_duration_hours', 'sample_rate_seconds', 'simulation_timestamp']]
    process_vars = df[process_columns].copy()
    process_vars_path = "demo/data/physics/process_variables.csv"
    process_vars.to_csv(process_vars_path, index=False)
    logger.info(f"✅ Saved process variables: {process_vars_path} ({len(process_vars)} rows)")
    
    # Create quality parameters dataset (simulated from process variables)
    quality_params = create_quality_from_process(process_vars)
    quality_path = "demo/data/physics/quality_parameters.csv"
    quality_params.to_csv(quality_path, index=False)
    logger.info(f"✅ Saved quality parameters: {quality_path} ({len(quality_params)} rows)")
    
    # Create energy consumption dataset
    energy_data = create_energy_dataset(process_vars)
    energy_path = "demo/data/physics/energy_consumption.csv"
    energy_data.to_csv(energy_path, index=False)
    logger.info(f"✅ Saved energy consumption: {energy_path} ({len(energy_data)} rows)")
    
    # Create emissions dataset
    emissions_data = create_emissions_dataset(process_vars)
    emissions_path = "demo/data/physics/emissions.csv"
    emissions_data.to_csv(emissions_path, index=False)
    logger.info(f"✅ Saved emissions data: {emissions_path} ({len(emissions_data)} rows)")

def create_quality_from_process(process_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create quality parameters from process variables using physics-based relationships
    
    Args:
        process_df: Process variables DataFrame
        
    Returns:
        Quality parameters DataFrame
    """
    quality_data = []
    
    for _, row in process_df.iterrows():
        # Simulate quality parameters based on process conditions
        burning_zone_temp = row.get('burning_zone_temp_c', 1450)
        kiln_speed = row.get('kiln_speed_rpm', 3.0)
        fuel_rate = row.get('fuel_rate_tph', 15.0)
        
        # Physics-based quality relationships
        # Free lime decreases with higher temperature and longer residence time
        residence_time = 60 / kiln_speed  # Simplified calculation
        free_lime = max(0.5, 3.0 - (burning_zone_temp - 1400) * 0.01 - residence_time * 0.05)
        
        # C3S content increases with higher temperature
        c3s_content = min(70, 50 + (burning_zone_temp - 1400) * 0.02)
        
        # C2S content decreases with higher temperature
        c2s_content = max(5, 25 - (burning_zone_temp - 1400) * 0.01)
        
        # Compressive strength correlates with C3S content
        compressive_strength = max(30, 35 + (c3s_content - 55) * 0.5)
        
        quality_data.append({
            'timestamp': row['timestamp'],
            'free_lime_percent': free_lime,
            'c3s_content_percent': c3s_content,
            'c2s_content_percent': c2s_content,
            'c3a_content_percent': 8.0 + (fuel_rate - 15) * 0.1,
            'water_cement_ratio': 0.4 + (kiln_speed - 3) * 0.01,
            'fineness_blaine_cm2_g': 3500 + (burning_zone_temp - 1450) * 2,
            'compressive_strength_mpa': compressive_strength,
            'free_lime_pct': free_lime  # Duplicate for compatibility
        })
    
    return pd.DataFrame(quality_data)

def create_energy_dataset(process_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create energy consumption dataset from process variables
    
    Args:
        process_df: Process variables DataFrame
        
    Returns:
        Energy consumption DataFrame
    """
    energy_data = []
    
    for _, row in process_df.iterrows():
        # Calculate energy consumption based on process conditions
        fuel_rate = row.get('fuel_rate_tph', 15.0)
        kiln_speed = row.get('kiln_speed_rpm', 3.0)
        feed_rate = row.get('feed_rate_tph', 200.0)
        
        # Thermal energy calculation
        thermal_energy = 650 + fuel_rate * 2.5  # kcal/kg clinker
        
        # Electrical energy calculation
        electrical_energy = 100 + (kiln_speed - 3) * 5 + (feed_rate - 200) * 0.1  # kWh/t
        
        # Power consumption
        total_power = electrical_energy * feed_rate / 24  # kW
        
        energy_data.append({
            'timestamp': row['timestamp'],
            'thermal_energy_kcal_kg': thermal_energy,
            'electrical_energy_kwh_t': electrical_energy,
            'total_power_kw': total_power,
            'specific_power_kwh_t': electrical_energy,
            'fuel_rate_tph': fuel_rate,
            'kiln_speed_rpm': kiln_speed,
            'feed_rate_tph': feed_rate
        })
    
    return pd.DataFrame(energy_data)

def create_emissions_dataset(process_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create emissions dataset from process variables
    
    Args:
        process_df: Process variables DataFrame
        
    Returns:
        Emissions DataFrame
    """
    emissions_data = []
    
    for _, row in process_df.iterrows():
        # Calculate emissions based on process conditions
        fuel_rate = row.get('fuel_rate_tph', 15.0)
        burning_zone_temp = row.get('burning_zone_temp_c', 1450)
        excess_air = row.get('excess_air_pct', 10.0)
        
        # CO2 emissions (kg/t clinker)
        co2_emissions = 800 + fuel_rate * 5
        
        # NOx emissions (mg/Nm³)
        nox_emissions = 400 + (burning_zone_temp - 1400) * 2 + excess_air * 5
        
        # SO2 emissions (mg/Nm³)
        so2_emissions = 100 + fuel_rate * 2
        
        # Dust emissions (mg/Nm³)
        dust_emissions = 20 + (burning_zone_temp - 1400) * 0.1
        
        emissions_data.append({
            'timestamp': row['timestamp'],
            'co2_kg_t': co2_emissions,
            'nox_mg_nm3': nox_emissions,
            'so2_mg_nm3': so2_emissions,
            'dust_mg_nm3': dust_emissions,
            'burning_zone_temp_c': burning_zone_temp,
            'fuel_rate_tph': fuel_rate,
            'excess_air_pct': excess_air
        })
    
    return pd.DataFrame(emissions_data)

def validate_simulation_data(df: pd.DataFrame) -> bool:
    """
    Validate simulation data
    
    Args:
        df: Simulation DataFrame
        
    Returns:
        True if validation passes
    """
    logger.info("🔄 Validating simulation data...")
    
    # Check required columns (using actual column names from DCS simulator)
    required_columns = ['timestamp', 'burning_zone_temp_c', 'kiln_speed_rpm', 'total_fuel_flow_tph']
    
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"❌ Missing required column: {col}")
            return False
    
    # Check data ranges
    if 'burning_zone_temp_c' in df.columns:
        if df['burning_zone_temp_c'].min() < 1200 or df['burning_zone_temp_c'].max() > 1600:
            logger.warning("⚠️ Burning zone temperature outside normal range")
    
    if 'kiln_speed_rpm' in df.columns:
        if df['kiln_speed_rpm'].min() < 1.0 or df['kiln_speed_rpm'].max() > 5.0:
            logger.warning("⚠️ Kiln speed outside normal range")
    
    logger.info("✅ Simulation data validation completed")
    return True

def get_simulation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get simulation data summary
    
    Args:
        df: Simulation DataFrame
        
    Returns:
        Simulation summary dictionary
    """
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'simulation_duration_hours': df['simulation_duration_hours'].iloc[0] if 'simulation_duration_hours' in df.columns else 'Unknown',
        'sample_rate_seconds': df['sample_rate_seconds'].iloc[0] if 'sample_rate_seconds' in df.columns else 'Unknown',
        'simulation_timestamp': df['simulation_timestamp'].iloc[0] if 'simulation_timestamp' in df.columns else 'Unknown',
        'process_variables': [col for col in df.columns if col not in ['timestamp', 'simulation_type', 'simulation_duration_hours', 'sample_rate_seconds', 'simulation_timestamp']],
        'data_ranges': {}
    }
    
    # Calculate data ranges for key variables
    key_variables = ['burning_zone_temp_c', 'kiln_speed_rpm', 'total_fuel_flow_tph', 'kiln_feed_rate_tph']
    for var in key_variables:
        if var in df.columns:
            summary['data_ranges'][var] = {
                'min': float(df[var].min()),
                'max': float(df[var].max()),
                'mean': float(df[var].mean())
            }
    
    return summary

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run DWSIM simulation
        df = simulate_process(n_hours=24, sample_rate_sec=60)
        
        # Validate simulation data
        if validate_simulation_data(df):
            # Get and print summary
            summary = get_simulation_summary(df)
            logger.info(f"📊 Simulation Summary:")
            logger.info(f"   Total Records: {summary['total_records']}")
            logger.info(f"   Total Columns: {summary['total_columns']}")
            logger.info(f"   Duration: {summary['simulation_duration_hours']} hours")
            logger.info(f"   Sample Rate: {summary['sample_rate_seconds']} seconds")
            logger.info(f"   Process Variables: {len(summary['process_variables'])}")
            
            logger.info("✅ DWSIM simulation completed successfully!")
        else:
            logger.error("❌ Simulation data validation failed")
            
    except Exception as e:
        logger.error(f"❌ DWSIM simulation failed: {e}")
        raise
