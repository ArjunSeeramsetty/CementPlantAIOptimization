#!/usr/bin/env python3
"""
BigQuery Data Loader for Real-World Dataset Integration
Loads real-world datasets from BigQuery to integrate with synthetic data generation pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class BigQueryDataLoader:
    """
    Loads real-world datasets from BigQuery for integration with synthetic data generation.
    
    This class connects to BigQuery and loads:
    - Mendeley LCI data (mendeley_lci_data table)
    - Kaggle concrete strength data (kaggle_concrete_strength table)  
    - Global cement assets data (global_cement_assets table)
    """
    
    def __init__(self, project_id: str = 'cement-ai-opt-38517', dataset_id: str = 'cement_analytics'):
        """
        Initialize BigQuery Data Loader.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize BigQuery client."""
        try:
            self.client = bigquery.Client(project=self.project_id)
            logger.info(f"✅ Connected to BigQuery project: {self.project_id}")
            logger.info(f"📊 Dataset: {self.dataset_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BigQuery client: {e}")
            raise
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three main datasets from BigQuery.
        
        Returns:
            Tuple of (mendeley_df, kaggle_df, global_df)
        """
        logger.info("🔄 Loading all real-world datasets from BigQuery...")
        
        try:
            # Load Mendeley LCI data
            mendeley_df = self.load_mendeley_lci_data()
            logger.info(f"✅ Loaded Mendeley LCI data: {len(mendeley_df)} records")
            
            # Load Kaggle concrete strength data
            kaggle_df = self.load_kaggle_concrete_strength()
            logger.info(f"✅ Loaded Kaggle concrete strength data: {len(kaggle_df)} records")
            
            # Load Global cement assets data
            global_df = self.load_global_cement_assets()
            logger.info(f"✅ Loaded Global cement assets data: {len(global_df)} records")
            
            return mendeley_df, kaggle_df, global_df
            
        except Exception as e:
            logger.error(f"❌ Failed to load datasets: {e}")
            raise
    
    def load_mendeley_lci_data(self) -> pd.DataFrame:
        """
        Load Mendeley LCI data from BigQuery.
        
        Returns:
            DataFrame with Mendeley LCI plant data
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.mendeley_lci_data`
        ORDER BY plant_id
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"📊 Mendeley LCI data loaded: {len(df)} plants")
            logger.info(f"📋 Columns: {list(df.columns)}")
            return df
        except NotFound:
            logger.warning("⚠️ Mendeley LCI table not found, creating sample data")
            return self._create_sample_mendeley_data()
        except Exception as e:
            logger.error(f"❌ Error loading Mendeley LCI data: {e}")
            raise
    
    def load_kaggle_concrete_strength(self) -> pd.DataFrame:
        """
        Load Kaggle concrete strength data from BigQuery.
        
        Returns:
            DataFrame with concrete strength test data
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.kaggle_concrete_strength`
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"📊 Kaggle concrete strength data loaded: {len(df)} tests")
            logger.info(f"📋 Columns: {list(df.columns)}")
            return df
        except NotFound:
            logger.warning("⚠️ Kaggle concrete strength table not found, creating sample data")
            return self._create_sample_kaggle_data()
        except Exception as e:
            logger.error(f"❌ Error loading Kaggle concrete strength data: {e}")
            raise
    
    def load_global_cement_assets(self) -> pd.DataFrame:
        """
        Load Global cement assets data from BigQuery.
        
        Returns:
            DataFrame with global cement plant data
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.global_cement_assets`
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"📊 Global cement assets data loaded: {len(df)} plants")
            logger.info(f"📋 Columns: {list(df.columns)}")
            return df
        except NotFound:
            logger.warning("⚠️ Global cement assets table not found, creating sample data")
            return self._create_sample_global_data()
        except Exception as e:
            logger.error(f"❌ Error loading global cement assets data: {e}")
            raise
    
    def load_lci_plant(self, plant_name: str) -> pd.DataFrame:
        """
        Load specific plant data from Mendeley LCI dataset.
        
        Args:
            plant_name: Name of the plant to load
            
        Returns:
            DataFrame with specific plant data
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.mendeley_lci_data`
        WHERE plant_name LIKE '%{plant_name}%'
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            if len(df) == 0:
                logger.warning(f"⚠️ Plant '{plant_name}' not found in Mendeley LCI data")
                return self._get_default_plant_data()
            
            logger.info(f"✅ Loaded plant data for: {plant_name}")
            return df.iloc[0]  # Return first match as Series
        except Exception as e:
            logger.error(f"❌ Error loading plant data for {plant_name}: {e}")
            return self._get_default_plant_data()
    
    def load_process_variables(self) -> pd.DataFrame:
        """
        Load process variables data for TimeGAN training.
        
        Returns:
            DataFrame with process variables time series
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.process_variables`
        ORDER BY timestamp
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            logger.info(f"📊 Process variables loaded: {len(df)} records")
            return df
        except NotFound:
            logger.warning("⚠️ Process variables table not found")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error loading process variables: {e}")
            return pd.DataFrame()
    
    def get_plant_kpis(self, plant_name: str) -> Dict[str, float]:
        """
        Extract KPIs for a specific plant from Mendeley LCI data.
        
        Args:
            plant_name: Name of the plant
            
        Returns:
            Dictionary with plant KPIs
        """
        plant_data = self.load_lci_plant(plant_name)
        
        kpis = {
            'thermal_energy_kcal_kg_clinker': plant_data.get('thermal_energy_kcal_kg_clinker', 3200),
            'electrical_energy_kwh_t': plant_data.get('electrical_energy_kwh_t', 95),
            'co2_kg_t': plant_data.get('co2_kg_t', 800),
            'kiln_speed_rpm': plant_data.get('kiln_speed_rpm', 3.5),
            'burning_zone_temp_c': plant_data.get('burning_zone_temp_c', 1450),
            'capacity_tpd': plant_data.get('capacity_tpd', 10000),
            'kiln_type': plant_data.get('kiln_type', 'Dry Process'),
            'commissioning_year': plant_data.get('commissioning_year', 2010)
        }
        
        logger.info(f"📊 Extracted KPIs for {plant_name}: {kpis}")
        return kpis
    
    def get_quality_correlations(self) -> Dict[str, float]:
        """
        Extract quality correlations from Kaggle concrete strength data.
        
        Returns:
            Dictionary with quality correlation coefficients
        """
        kaggle_df = self.load_kaggle_concrete_strength()
        
        if len(kaggle_df) == 0:
            return self._get_default_quality_correlations()
        
        # Calculate correlations between components and strength
        correlations = {}
        
        # Map Kaggle columns to cement quality parameters
        if 'cement' in kaggle_df.columns and 'compressive_strength' in kaggle_df.columns:
            correlations['cement_strength_correlation'] = kaggle_df['cement'].corr(kaggle_df['compressive_strength'])
        
        if 'fly_ash' in kaggle_df.columns and 'compressive_strength' in kaggle_df.columns:
            correlations['fly_ash_strength_correlation'] = kaggle_df['fly_ash'].corr(kaggle_df['compressive_strength'])
        
        if 'water' in kaggle_df.columns and 'compressive_strength' in kaggle_df.columns:
            correlations['water_strength_correlation'] = kaggle_df['water'].corr(kaggle_df['compressive_strength'])
        
        # Add default correlations if not found
        correlations.update(self._get_default_quality_correlations())
        
        logger.info(f"📊 Quality correlations extracted: {correlations}")
        return correlations
    
    def _create_sample_mendeley_data(self) -> pd.DataFrame:
        """Create sample Mendeley LCI data if table not found."""
        logger.info("Creating sample Mendeley LCI data...")
        
        np.random.seed(42)
        n_plants = 6
        
        data = {
            'plant_id': [f'PLANT_{i:02d}' for i in range(1, n_plants + 1)],
            'plant_name': [
                'UltraTech Cement Plant - Gujarat',
                'ACC Limited - Maharashtra', 
                'Ambuja Cements - Rajasthan',
                'Shree Cement - Rajasthan',
                'JK Cement - Karnataka',
                'Ramco Cements - Tamil Nadu'
            ],
            'capacity_tpd': [10000, 8500, 12000, 15000, 7500, 9000],
            'kiln_type': ['Dry Process'] * n_plants,
            'commissioning_year': [2010, 2015, 2008, 2012, 2018, 2014],
            'thermal_energy_kcal_kg_clinker': np.random.normal(3200, 200, n_plants),
            'electrical_energy_kwh_t': np.random.normal(95, 8, n_plants),
            'co2_kg_t': np.random.normal(800, 50, n_plants),
            'kiln_speed_rpm': np.random.normal(3.5, 0.3, n_plants),
            'burning_zone_temp_c': np.random.normal(1450, 50, n_plants)
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_kaggle_data(self) -> pd.DataFrame:
        """Create sample Kaggle concrete strength data if table not found."""
        logger.info("Creating sample Kaggle concrete strength data...")
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'test_id': range(1, n_samples + 1),
            'cement': np.random.uniform(200, 400, n_samples),
            'blast_furnace_slag': np.random.uniform(0, 200, n_samples),
            'fly_ash': np.random.uniform(0, 150, n_samples),
            'water': np.random.uniform(150, 250, n_samples),
            'superplasticizer': np.random.uniform(0, 30, n_samples),
            'coarse_aggregate': np.random.uniform(800, 1200, n_samples),
            'fine_aggregate': np.random.uniform(600, 900, n_samples),
            'age_days': np.random.choice([1, 3, 7, 14, 28, 56, 90], n_samples),
            'compressive_strength': np.random.uniform(10, 80, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_global_data(self) -> pd.DataFrame:
        """Create sample Global cement assets data if table not found."""
        logger.info("Creating sample Global cement assets data...")
        
        np.random.seed(42)
        n_plants = 2000
        
        countries = ['USA', 'China', 'India', 'Germany', 'Brazil', 'Mexico', 'Turkey', 'Russia', 'Japan', 'Italy']
        cement_types = ['CEM I', 'CEM II', 'CEM III', 'CEM IV', 'CEM V']
        processes = ['Dry', 'Wet', 'Semi-dry']
        
        data = {
            'plant_id': [f'GLOBAL_{i:04d}' for i in range(1, n_plants + 1)],
            'country': np.random.choice(countries, n_plants),
            'cement_type': np.random.choice(cement_types, n_plants),
            'process_type': np.random.choice(processes, n_plants),
            'capacity_tpd': np.random.uniform(1000, 20000, n_plants),
            'commissioning_year': np.random.randint(1990, 2024, n_plants),
            'kiln_count': np.random.randint(1, 5, n_plants),
            'grinding_mills': np.random.randint(1, 8, n_plants)
        }
        
        return pd.DataFrame(data)
    
    def _get_default_plant_data(self) -> pd.Series:
        """Get default plant data if specific plant not found."""
        return pd.Series({
            'plant_id': 'DEFAULT_PLANT',
            'plant_name': 'UltraTech Cement Plant - Default',
            'capacity_tpd': 10000,
            'kiln_type': 'Dry Process',
            'commissioning_year': 2010,
            'thermal_energy_kcal_kg_clinker': 3200,
            'electrical_energy_kwh_t': 95,
            'co2_kg_t': 800,
            'kiln_speed_rpm': 3.5,
            'burning_zone_temp_c': 1450
        })
    
    def _get_default_quality_correlations(self) -> Dict[str, float]:
        """Get default quality correlations."""
        return {
            'cement_strength_correlation': 0.75,
            'fly_ash_strength_correlation': 0.65,
            'water_strength_correlation': -0.85,
            'c3s_strength_correlation': 0.80,
            'c2s_strength_correlation': 0.60,
            'fineness_strength_correlation': 0.70
        }


def test_bigquery_data_loader():
    """Test the BigQuery Data Loader functionality."""
    logger.info("🧪 Testing BigQuery Data Loader...")
    
    try:
        # Initialize loader
        loader = BigQueryDataLoader()
        
        # Load all datasets
        mendeley_df, kaggle_df, global_df = loader.load_all()
        
        # Test plant selection
        plant_kpis = loader.get_plant_kpis('UltraTech')
        logger.info(f"✅ Plant KPIs: {plant_kpis}")
        
        # Test quality correlations
        quality_correlations = loader.get_quality_correlations()
        logger.info(f"✅ Quality correlations: {quality_correlations}")
        
        logger.info("✅ BigQuery Data Loader test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ BigQuery Data Loader test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the loader
    test_bigquery_data_loader()
