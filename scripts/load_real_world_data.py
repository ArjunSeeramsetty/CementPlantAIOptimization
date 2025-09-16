"""
Load Real-World Data from BigQuery for Demo Pipeline
Integrates with existing BigQueryDataLoader and saves data locally for demo
"""

import os
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_sourcing.bigquery_data_loader import BigQueryDataLoader

logger = logging.getLogger(__name__)

def load_and_save_all() -> Dict[str, pd.DataFrame]:
    """
    Load all real-world datasets from BigQuery and save locally for demo pipeline
    
    Returns:
        Dictionary of loaded DataFrames
    """
    logger.info("🔄 Loading real-world data from BigQuery...")
    
    try:
        # Initialize BigQuery data loader
        loader = BigQueryDataLoader()
        
        # Load all datasets
        mendeley_df, kaggle_df, global_df = loader.load_all()
        
        # Create demo data directory
        os.makedirs("demo/data/real", exist_ok=True)
        
        # Prepare data for demo pipeline
        data = {
            'mendeley_lci': mendeley_df,
            'kaggle_concrete_strength': kaggle_df,
            'global_cement_assets': global_df
        }
        
        # Save each dataset
        for table_name, df in data.items():
            path = f"demo/data/real/{table_name}.csv"
            df.to_csv(path, index=False)
            logger.info(f"✅ Saved real data: {path} ({len(df)} rows)")
        
        # Create process variables summary for demo
        process_summary = create_process_variables_summary(mendeley_df)
        process_path = "demo/data/real/process_variables.csv"
        process_summary.to_csv(process_path, index=False)
        logger.info(f"✅ Saved process variables: {process_path} ({len(process_summary)} rows)")
        
        # Create quality parameters summary for demo
        quality_summary = create_quality_parameters_summary(kaggle_df)
        quality_path = "demo/data/real/quality_parameters.csv"
        quality_summary.to_csv(quality_path, index=False)
        logger.info(f"✅ Saved quality parameters: {quality_path} ({len(quality_summary)} rows)")
        
        # Add to data dictionary
        data['process_variables'] = process_summary
        data['quality_parameters'] = quality_summary
        
        logger.info(f"✅ Successfully loaded {len(data)} real-world datasets")
        return data
        
    except Exception as e:
        logger.error(f"❌ Failed to load real-world data: {e}")
        raise

def create_process_variables_summary(mendeley_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create process variables summary from Mendeley LCI data
    
    Args:
        mendeley_df: Mendeley LCI DataFrame
        
    Returns:
        Process variables summary DataFrame
    """
    logger.info("🔄 Creating process variables summary...")
    
    # Extract key process variables
    process_vars = []
    
    for _, row in mendeley_df.iterrows():
        # Create timestamp for demo
        timestamp = datetime.now()
        
        process_vars.append({
            'timestamp': timestamp,
            'plant_id': row['plant_id'],
            'plant_name': row['plant_name'],
            'capacity_tpd': row['capacity_tpd'],
            'kiln_type': row['kiln_type'],
            'limestone_kg_t': row['limestone_kg_t'],
            'clay_kg_t': row['clay_kg_t'],
            'iron_ore_kg_t': row['iron_ore_kg_t'],
            'gypsum_kg_t': row['gypsum_kg_t'],
            'coal_kg_t': row['coal_kg_t'],
            'petcoke_kg_t': row['petcoke_kg_t'],
            'alternative_fuels_kg_t': row['alternative_fuels_kg_t'],
            'electrical_energy_kwh_t': row['electrical_energy_kwh_t'],
            'thermal_energy_kcal_kg_clinker': row['thermal_energy_kcal_kg_clinker'],
            'co2_kg_t': row['co2_kg_t'],
            'nox_kg_t': row['nox_kg_t'],
            'so2_kg_t': row['so2_kg_t'],
            'dust_kg_t': row['dust_kg_t'],
            'free_lime_pct': row['free_lime_pct'],
            'c3s_content_pct': row['c3s_content_pct'],
            'c2s_content_pct': row['c2s_content_pct'],
            'compressive_strength_28d_mpa': row['compressive_strength_28d_mpa']
        })
    
    return pd.DataFrame(process_vars)

def create_quality_parameters_summary(kaggle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create quality parameters summary from Kaggle concrete strength data
    
    Args:
        kaggle_df: Kaggle concrete strength DataFrame
        
    Returns:
        Quality parameters summary DataFrame
    """
    logger.info("🔄 Creating quality parameters summary...")
    
    # Map Kaggle data to cement quality parameters
    quality_params = []
    
    for _, row in kaggle_df.iterrows():
        # Create timestamp for demo
        timestamp = datetime.now()
        
        quality_params.append({
            'timestamp': timestamp,
            'cement_kg_m3': row['cement_kg_m3'],
            'blast_furnace_slag_kg_m3': row['blast_furnace_slag_kg_m3'],
            'fly_ash_kg_m3': row['fly_ash_kg_m3'],
            'water_kg_m3': row['water_kg_m3'],
            'superplasticizer_kg_m3': row['superplasticizer_kg_m3'],
            'coarse_aggregate_kg_m3': row['coarse_aggregate_kg_m3'],
            'fine_aggregate_kg_m3': row['fine_aggregate_kg_m3'],
            'age_days': row['age_days'],
            'compressive_strength_mpa': row['compressive_strength_mpa'],
            # Map to cement quality parameters
            'free_lime_percent': 1.5 + (row['compressive_strength_mpa'] - 30) * 0.02,  # Simulated mapping
            'c3s_content_percent': 55 + (row['cement_kg_m3'] - 200) * 0.1,  # Simulated mapping
            'c2s_content_percent': 15 + (row['blast_furnace_slag_kg_m3'] - 100) * 0.05,  # Simulated mapping
            'c3a_content_percent': 8 + (row['fly_ash_kg_m3'] - 50) * 0.02,  # Simulated mapping
            'water_cement_ratio': row['water_kg_m3'] / row['cement_kg_m3'] if row['cement_kg_m3'] > 0 else 0.4,
            'fineness_blaine_cm2_g': 3000 + (row['superplasticizer_kg_m3'] - 5) * 50,  # Simulated mapping
            'free_lime_pct': 1.5 + (row['compressive_strength_mpa'] - 30) * 0.02  # Duplicate for compatibility
        })
    
    return pd.DataFrame(quality_params)

def validate_loaded_data(data: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate loaded data for demo pipeline
    
    Args:
        data: Dictionary of loaded DataFrames
        
    Returns:
        True if validation passes
    """
    logger.info("🔄 Validating loaded data...")
    
    required_datasets = ['mendeley_lci', 'kaggle_concrete_strength', 'global_cement_assets', 'process_variables', 'quality_parameters']
    
    for dataset_name in required_datasets:
        if dataset_name not in data:
            logger.error(f"❌ Missing required dataset: {dataset_name}")
            return False
        
        df = data[dataset_name]
        if len(df) == 0:
            logger.error(f"❌ Empty dataset: {dataset_name}")
            return False
        
        logger.info(f"✅ {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
    
    logger.info("✅ All datasets validated successfully")
    return True

def get_data_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Get summary of loaded data
    
    Args:
        data: Dictionary of loaded DataFrames
        
    Returns:
        Data summary dictionary
    """
    summary = {
        'total_datasets': len(data),
        'datasets': {},
        'total_records': 0,
        'load_timestamp': datetime.now().isoformat()
    }
    
    for name, df in data.items():
        summary['datasets'][name] = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns)
        }
        summary['total_records'] += len(df)
    
    return summary

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load and save all real-world data
        data = load_and_save_all()
        
        # Validate loaded data
        if validate_loaded_data(data):
            # Get and print summary
            summary = get_data_summary(data)
            logger.info(f"📊 Data Summary:")
            logger.info(f"   Total Datasets: {summary['total_datasets']}")
            logger.info(f"   Total Records: {summary['total_records']}")
            logger.info(f"   Load Timestamp: {summary['load_timestamp']}")
            
            logger.info("✅ Real-world data loading completed successfully!")
        else:
            logger.error("❌ Data validation failed")
            
    except Exception as e:
        logger.error(f"❌ Real-world data loading failed: {e}")
        raise
