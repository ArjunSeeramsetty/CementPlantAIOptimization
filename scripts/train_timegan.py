"""
TimeGAN Training for Synthetic Data Generation
Trains TimeGAN model on combined real + physics data and generates synthetic time-series
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.train_gan import CementPlantDataGenerator

logger = logging.getLogger(__name__)

def train_and_generate(n_samples: int = 100000) -> pd.DataFrame:
    """
    Train TimeGAN model and generate synthetic data
    
    Args:
        n_samples: Number of synthetic samples to generate
        
    Returns:
        Generated synthetic DataFrame
    """
    logger.info(f"🔄 Starting TimeGAN training and generation for {n_samples} samples...")
    
    try:
        # Load real-world datasets
        logger.info("📊 Loading real-world datasets...")
        real_data = load_real_world_data()
        
        # Load physics simulation data
        logger.info("⚗️ Loading physics simulation data...")
        physics_data = load_physics_data()
        
        # Combine datasets
        logger.info("🔄 Combining real and physics data...")
        combined_data = combine_datasets(real_data, physics_data)
        
        # Initialize TimeGAN generator
        logger.info("🤖 Initializing TimeGAN generator...")
        generator = CementPlantDataGenerator()
        
        # Train TimeGAN model
        logger.info("🔄 Training TimeGAN model...")
        training_success = train_timegan_model(generator, combined_data)
        
        # Generate synthetic data
        logger.info(f"🔄 Generating {n_samples} synthetic samples...")
        synthetic_data = generate_synthetic_data(generator, n_samples, combined_data)
        
        # Save synthetic data
        os.makedirs("demo/data/synthetic", exist_ok=True)
        path = "demo/data/synthetic/timegan_synthetic.csv"
        synthetic_data.to_csv(path, index=False)
        logger.info(f"✅ Saved synthetic data: {path} ({len(synthetic_data)} rows)")
        
        # Create additional synthetic datasets
        create_additional_synthetic_datasets(synthetic_data)
        
        logger.info("✅ TimeGAN training and generation completed successfully")
        return synthetic_data
        
    except Exception as e:
        logger.error(f"❌ TimeGAN training failed: {e}")
        raise

def load_real_world_data() -> Dict[str, pd.DataFrame]:
    """
    Load real-world datasets for TimeGAN training
    
    Returns:
        Dictionary of real-world DataFrames
    """
    data = {}
    
    # Load process variables
    process_path = "demo/data/real/process_variables.csv"
    if os.path.exists(process_path):
        data['process_variables'] = pd.read_csv(process_path)
        logger.info(f"✅ Loaded process variables: {len(data['process_variables'])} rows")
    
    # Load quality parameters
    quality_path = "demo/data/real/quality_parameters.csv"
    if os.path.exists(quality_path):
        data['quality_parameters'] = pd.read_csv(quality_path)
        logger.info(f"✅ Loaded quality parameters: {len(data['quality_parameters'])} rows")
    
    return data

def load_physics_data() -> Dict[str, pd.DataFrame]:
    """
    Load physics simulation data for TimeGAN training
    
    Returns:
        Dictionary of physics DataFrames
    """
    data = {}
    
    # Load main physics simulation
    physics_path = "demo/data/physics/dwsim_physics.csv"
    if os.path.exists(physics_path):
        data['dwsim_physics'] = pd.read_csv(physics_path)
        logger.info(f"✅ Loaded physics simulation: {len(data['dwsim_physics'])} rows")
    
    # Load process variables from physics
    process_path = "demo/data/physics/process_variables.csv"
    if os.path.exists(process_path):
        data['physics_process_variables'] = pd.read_csv(process_path)
        logger.info(f"✅ Loaded physics process variables: {len(data['physics_process_variables'])} rows")
    
    # Load quality parameters from physics
    quality_path = "demo/data/physics/quality_parameters.csv"
    if os.path.exists(quality_path):
        data['physics_quality_parameters'] = pd.read_csv(quality_path)
        logger.info(f"✅ Loaded physics quality parameters: {len(data['physics_quality_parameters'])} rows")
    
    return data

def combine_datasets(real_data: Dict[str, pd.DataFrame], 
                    physics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine real and physics datasets for TimeGAN training
    
    Args:
        real_data: Real-world datasets
        physics_data: Physics simulation datasets
        
    Returns:
        Combined DataFrame
    """
    logger.info("🔄 Combining datasets for TimeGAN training...")
    
    combined_records = []
    
    # Add real-world data
    if 'process_variables' in real_data:
        for _, row in real_data['process_variables'].iterrows():
            # Handle missing timestamp column
            timestamp = row.get('timestamp', datetime.now().isoformat())
            record = {
                'timestamp': timestamp,
                'data_source': 'real_world',
                'plant_id': row.get('plant_id', 'real_plant'),
                'plant_name': row.get('plant_name', 'Real Plant'),
                'capacity_tpd': row.get('capacity_tpd', 10000),
                'kiln_type': row.get('kiln_type', 'rotary'),
                'limestone_kg_t': row.get('limestone_kg_t', 1200),
                'clay_kg_t': row.get('clay_kg_t', 200),
                'iron_ore_kg_t': row.get('iron_ore_kg_t', 50),
                'gypsum_kg_t': row.get('gypsum_kg_t', 50),
                'coal_kg_t': row.get('coal_kg_t', 150),
                'petcoke_kg_t': row.get('petcoke_kg_t', 50),
                'alternative_fuels_kg_t': row.get('alternative_fuels_kg_t', 20),
                'electrical_energy_kwh_t': row.get('electrical_energy_kwh_t', 110),
                'thermal_energy_kcal_kg_clinker': row.get('thermal_energy_kcal_kg_clinker', 690),
                'co2_kg_t': row.get('co2_kg_t', 850),
                'nox_kg_t': row.get('nox_kg_t', 500),
                'so2_kg_t': row.get('so2_kg_t', 150),
                'dust_kg_t': row.get('dust_kg_t', 25),
                'free_lime_pct': row.get('free_lime_pct', 1.8),
                'c3s_content_pct': row.get('c3s_content_pct', 58),
                'c2s_content_pct': row.get('c2s_content_pct', 15),
                'compressive_strength_28d_mpa': row.get('compressive_strength_28d_mpa', 42)
            }
            combined_records.append(record)
    
    # Add physics simulation data
    if 'dwsim_physics' in physics_data:
        for _, row in physics_data['dwsim_physics'].iterrows():
            # Handle missing timestamp column
            timestamp = row.get('timestamp', datetime.now().isoformat())
            record = {
                'timestamp': timestamp,
                'data_source': 'physics_simulation',
                'plant_id': 'simulated_plant',
                'plant_name': 'Simulated Plant',
                'capacity_tpd': 10000,
                'kiln_type': 'rotary',
                'limestone_kg_t': 1200,
                'clay_kg_t': 200,
                'iron_ore_kg_t': 50,
                'gypsum_kg_t': 50,
                'coal_kg_t': row.get('coal_flow_tph', 15) * 24,
                'petcoke_kg_t': row.get('petcoke_flow_tph', 5) * 24,
                'alternative_fuels_kg_t': row.get('alternative_fuel_flow_tph', 3) * 24,
                'electrical_energy_kwh_t': row.get('specific_power_kwh_t', 110),
                'thermal_energy_kcal_kg_clinker': row.get('thermal_energy_kcal_kg', 690),
                'co2_kg_t': row.get('co2_kg_t', 850),
                'nox_mg_nm3': row.get('nox_mg_nm3', 500),
                'so2_mg_nm3': row.get('so2_mg_nm3', 150),
                'dust_mg_nm3': row.get('dust_mg_nm3', 25),
                'free_lime_pct': row.get('free_lime_pct', 1.8),
                'c3s_content_pct': row.get('c3s_content_pct', 58),
                'c2s_content_pct': row.get('c2s_content_pct', 15),
                'compressive_strength_28d_mpa': row.get('compressive_strength_28d_mpa', 42)
            }
            combined_records.append(record)
    
    combined_df = pd.DataFrame(combined_records)
    logger.info(f"✅ Combined dataset created: {len(combined_df)} records")
    
    return combined_df

def train_timegan_model(generator: CementPlantDataGenerator, 
                       combined_data: pd.DataFrame) -> bool:
    """
    Train TimeGAN model on combined data
    
    Args:
        generator: TimeGAN generator instance
        combined_data: Combined training data
        
    Returns:
        True if training successful
    """
    logger.info("🔄 Training TimeGAN model...")
    
    try:
        # Prepare data for TimeGAN training
        training_data = prepare_timegan_data(combined_data)
        
        # Train TimeGAN model
        training_success = generator.train_timegan_model(
            prepared_data=training_data,
            epochs=50,  # Reduced for demo
            batch_size=32
        )
        
        if training_success:
            logger.info("✅ TimeGAN model training completed successfully")
        else:
            logger.warning("⚠️ TimeGAN training failed, using statistical augmentation")
        
        return training_success
        
    except Exception as e:
        logger.error(f"❌ TimeGAN training failed: {e}")
        return False

def prepare_timegan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for TimeGAN training
    
    Args:
        df: Input DataFrame
        
    Returns:
        Prepared DataFrame for TimeGAN
    """
    logger.info("🔄 Preparing data for TimeGAN training...")
    
    # Select numerical columns for training
    numerical_columns = [
        'capacity_tpd', 'limestone_kg_t', 'clay_kg_t', 'iron_ore_kg_t', 'gypsum_kg_t',
        'coal_kg_t', 'petcoke_kg_t', 'alternative_fuels_kg_t', 'electrical_energy_kwh_t',
        'thermal_energy_kcal_kg_clinker', 'co2_kg_t', 'free_lime_pct', 'c3s_content_pct',
        'c2s_content_pct', 'compressive_strength_28d_mpa'
    ]
    
    # Filter available columns
    available_columns = [col for col in numerical_columns if col in df.columns]
    
    # Create prepared data
    prepared_data = df[available_columns].copy()
    
    # Handle missing values
    prepared_data = prepared_data.fillna(method='ffill').fillna(method='bfill')
    
    # Normalize data
    for col in prepared_data.columns:
        if prepared_data[col].dtype in ['float64', 'int64']:
            prepared_data[col] = (prepared_data[col] - prepared_data[col].mean()) / prepared_data[col].std()
    
    logger.info(f"✅ Prepared data: {len(prepared_data)} records, {len(prepared_data.columns)} features")
    
    return prepared_data

def generate_synthetic_data(generator: CementPlantDataGenerator, 
                          n_samples: int, 
                          base_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic data using TimeGAN
    
    Args:
        generator: TimeGAN generator instance
        n_samples: Number of samples to generate
        base_data: Base data for generation
        
    Returns:
        Generated synthetic DataFrame
    """
    logger.info(f"🔄 Generating {n_samples} synthetic samples...")
    
    try:
        # Generate synthetic data
        synthetic_data = generator.generate_massive_dataset(
            base_data_path=None,
            base_data_df=base_data,
            num_samples=n_samples,
            duration_hours=24,
            output_path=None  # Don't save to file yet
        )
        
        # Add synthetic data metadata
        synthetic_data['data_source'] = 'timegan_synthetic'
        synthetic_data['generation_timestamp'] = datetime.now().isoformat()
        synthetic_data['generation_method'] = 'timegan'
        
        logger.info(f"✅ Generated synthetic data: {len(synthetic_data)} records")
        
        return synthetic_data
        
    except Exception as e:
        logger.error(f"❌ Synthetic data generation failed: {e}")
        # Fallback to statistical generation
        logger.info("🔄 Using statistical augmentation as fallback...")
        return generate_statistical_synthetic_data(n_samples, base_data)

def generate_statistical_synthetic_data(n_samples: int, 
                                      base_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic data using statistical methods as fallback
    
    Args:
        n_samples: Number of samples to generate
        base_data: Base data for generation
        
    Returns:
        Generated synthetic DataFrame
    """
    logger.info("🔄 Generating synthetic data using statistical methods...")
    
    # Select numerical columns
    numerical_columns = [
        'capacity_tpd', 'limestone_kg_t', 'clay_kg_t', 'iron_ore_kg_t', 'gypsum_kg_t',
        'coal_kg_t', 'petcoke_kg_t', 'alternative_fuels_kg_t', 'electrical_energy_kwh_t',
        'thermal_energy_kcal_kg_clinker', 'co2_kg_t', 'free_lime_pct', 'c3s_content_pct',
        'c2s_content_pct', 'compressive_strength_28d_mpa'
    ]
    
    available_columns = [col for col in numerical_columns if col in base_data.columns]
    
    synthetic_records = []
    
    for i in range(n_samples):
        record = {}
        
        for col in available_columns:
            if col in base_data.columns:
                # Generate synthetic value based on distribution
                mean_val = base_data[col].mean()
                std_val = base_data[col].std()
                
                # Add some randomness
                synthetic_val = np.random.normal(mean_val, std_val * 0.1)
                
                # Ensure reasonable bounds
                if col == 'free_lime_pct':
                    synthetic_val = max(0.5, min(3.0, synthetic_val))
                elif col == 'c3s_content_pct':
                    synthetic_val = max(50, min(70, synthetic_val))
                elif col == 'compressive_strength_28d_mpa':
                    synthetic_val = max(30, min(60, synthetic_val))
                
                record[col] = synthetic_val
        
        # Add metadata
        record['timestamp'] = datetime.now().isoformat()
        record['data_source'] = 'statistical_synthetic'
        record['generation_timestamp'] = datetime.now().isoformat()
        record['generation_method'] = 'statistical'
        
        synthetic_records.append(record)
    
    return pd.DataFrame(synthetic_records)

def create_additional_synthetic_datasets(synthetic_data: pd.DataFrame):
    """
    Create additional synthetic datasets for demo pipeline
    
    Args:
        synthetic_data: Main synthetic DataFrame
    """
    logger.info("🔄 Creating additional synthetic datasets...")
    
    # Create process variables dataset
    process_vars = synthetic_data[['timestamp'] + [col for col in synthetic_data.columns if col not in ['timestamp', 'data_source', 'generation_timestamp', 'generation_method']]].copy()
    process_vars_path = "demo/data/synthetic/process_variables.csv"
    process_vars.to_csv(process_vars_path, index=False)
    logger.info(f"✅ Saved synthetic process variables: {process_vars_path} ({len(process_vars)} rows)")
    
    # Create quality parameters dataset
    quality_params = create_synthetic_quality_parameters(synthetic_data)
    quality_path = "demo/data/synthetic/quality_parameters.csv"
    quality_params.to_csv(quality_path, index=False)
    logger.info(f"✅ Saved synthetic quality parameters: {quality_path} ({len(quality_params)} rows)")

def create_synthetic_quality_parameters(synthetic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create quality parameters from synthetic data
    
    Args:
        synthetic_data: Synthetic DataFrame
        
    Returns:
        Quality parameters DataFrame
    """
    quality_data = []
    
    for _, row in synthetic_data.iterrows():
        quality_data.append({
            'timestamp': row['timestamp'],
            'free_lime_percent': row.get('free_lime_pct', 1.8),
            'c3s_content_percent': row.get('c3s_content_pct', 58),
            'c2s_content_percent': row.get('c2s_content_pct', 15),
            'c3a_content_percent': 8.0,
            'water_cement_ratio': 0.4,
            'fineness_blaine_cm2_g': 3500,
            'compressive_strength_mpa': row.get('compressive_strength_28d_mpa', 42),
            'free_lime_pct': row.get('free_lime_pct', 1.8)  # Duplicate for compatibility
        })
    
    return pd.DataFrame(quality_data)

def validate_synthetic_data(df: pd.DataFrame) -> bool:
    """
    Validate synthetic data
    
    Args:
        df: Synthetic DataFrame
        
    Returns:
        True if validation passes
    """
    logger.info("🔄 Validating synthetic data...")
    
    # Check required columns
    required_columns = ['timestamp', 'data_source']
    
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"❌ Missing required column: {col}")
            return False
    
    # Check data ranges
    if 'free_lime_pct' in df.columns:
        if df['free_lime_pct'].min() < 0 or df['free_lime_pct'].max() > 5:
            logger.warning("⚠️ Free lime percentage outside normal range")
    
    logger.info("✅ Synthetic data validation completed")
    return True

def get_synthetic_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get synthetic data summary
    
    Args:
        df: Synthetic DataFrame
        
    Returns:
        Synthetic summary dictionary
    """
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'data_source': df['data_source'].iloc[0] if 'data_source' in df.columns else 'Unknown',
        'generation_method': df['generation_method'].iloc[0] if 'generation_method' in df.columns else 'Unknown',
        'generation_timestamp': df['generation_timestamp'].iloc[0] if 'generation_timestamp' in df.columns else 'Unknown',
        'synthetic_variables': [col for col in df.columns if col not in ['timestamp', 'data_source', 'generation_timestamp', 'generation_method']]
    }
    
    return summary

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Train TimeGAN and generate synthetic data
        df = train_and_generate(n_samples=100000)
        
        # Validate synthetic data
        if validate_synthetic_data(df):
            # Get and print summary
            summary = get_synthetic_summary(df)
            logger.info(f"📊 Synthetic Data Summary:")
            logger.info(f"   Total Records: {summary['total_records']}")
            logger.info(f"   Total Columns: {summary['total_columns']}")
            logger.info(f"   Data Source: {summary['data_source']}")
            logger.info(f"   Generation Method: {summary['generation_method']}")
            logger.info(f"   Synthetic Variables: {len(summary['synthetic_variables'])}")
            
            logger.info("✅ TimeGAN training and generation completed successfully!")
        else:
            logger.error("❌ Synthetic data validation failed")
            
    except Exception as e:
        logger.error(f"❌ TimeGAN training failed: {e}")
        raise
