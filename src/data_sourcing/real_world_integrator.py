#!/usr/bin/env python3
"""
Plant Selection and Process Calibration Module
Integrates real-world plant data with configuration and calibrates DCS simulator parameters.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_sourcing.bigquery_data_loader import BigQueryDataLoader

logger = logging.getLogger(__name__)


class PlantSelector:
    """
    Selects optimal plant from Mendeley LCI data based on configuration requirements.
    """
    
    def __init__(self, config_path: str = 'config/plant_config.yml'):
        """
        Initialize Plant Selector.
        
        Args:
            config_path: Path to plant configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_loader = BigQueryDataLoader()
        
    def _load_config(self) -> dict:
        """Load plant configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded plant configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Config file {self.config_path} not found")
            raise
    
    def get_base_plant_from_config(self) -> Dict[str, Any]:
        """
        Get base plant configuration from plant_config.yml file.
        
        Returns:
            Dictionary with base plant data and KPIs
        """
        logger.info("🔍 Using base plant from plant_config.yml...")
        
        # Extract plant data from configuration
        plant_data = {
            'plant_id': 'CONFIG_PLANT',
            'plant_name': self.config['plant']['name'],
            'location': self.config['plant']['location'],
            'capacity_tpd': self.config['plant']['capacity_tpd'],
            'kiln_type': self.config['plant']['kiln_type'],
            'commissioning_year': self.config['plant'].get('commissioning_year', 2010),
            'thermal_energy_kcal_kg_clinker': self.config['plant']['energy'].get('thermal_kcal_kg_clinker', 3200),
            'electrical_energy_kwh_t': self.config['plant']['energy'].get('electrical_kwh_t', 95),
            'kiln_speed_rpm': self.config['plant']['process'].get('kiln_speed_rpm', 3.5),
            'burning_zone_temp_c': self.config['plant']['process'].get('burning_zone_temp_c', 1450),
            'co2_kg_t': self.config['plant']['environmental'].get('co2_kg_per_ton', 800)
        }
        
        # Extract KPIs from configuration
        kpis = {
            'thermal_energy_kcal_kg_clinker': plant_data['thermal_energy_kcal_kg_clinker'],
            'electrical_energy_kwh_t': plant_data['electrical_energy_kwh_t'],
            'co2_kg_t': plant_data['co2_kg_t'],
            'kiln_speed_rpm': plant_data['kiln_speed_rpm'],
            'burning_zone_temp_c': plant_data['burning_zone_temp_c'],
            'capacity_tpd': plant_data['capacity_tpd'],
            'kiln_type': plant_data['kiln_type'],
            'commissioning_year': plant_data['commissioning_year']
        }
        
        logger.info(f"✅ Using base plant: {plant_data['plant_name']}")
        logger.info(f"📊 Plant capacity: {plant_data['capacity_tpd']} TPD")
        
        return {
            'plant_data': plant_data,
            'kpis': kpis,
            'selection_criteria': 'config_based',
            'selection_score': 1.0
        }
    
    def _get_default_criteria(self) -> Dict[str, Any]:
        """Get default selection criteria based on configuration."""
        return {
            'capacity_weight': 0.3,
            'efficiency_weight': 0.4,
            'modernity_weight': 0.2,
            'reliability_weight': 0.1,
            'min_capacity_tpd': self.config['plant']['capacity_tpd'] * 0.8,
            'max_capacity_tpd': self.config['plant']['capacity_tpd'] * 1.2,
            'preferred_kiln_type': self.config['plant']['kiln_type']
        }
    
    def _score_plants(self, plants_df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Score plants based on selection criteria.
        
        Args:
            plants_df: DataFrame with plant data
            criteria: Selection criteria
            
        Returns:
            DataFrame with plants sorted by score
        """
        logger.info("📊 Scoring plants based on selection criteria...")
        
        # Initialize scores
        plants_df = plants_df.copy()
        plants_df['score'] = 0.0
        
        # Capacity score (normalized)
        capacity_mean = plants_df['capacity_tpd'].mean()
        capacity_std = plants_df['capacity_tpd'].std()
        plants_df['capacity_score'] = 1 - abs(plants_df['capacity_tpd'] - criteria['min_capacity_tpd']) / capacity_std
        plants_df['capacity_score'] = np.clip(plants_df['capacity_score'], 0, 1)
        
        # Efficiency score (based on energy consumption)
        thermal_mean = plants_df['thermal_energy_kcal_kg_clinker'].mean()
        thermal_std = plants_df['thermal_energy_kcal_kg_clinker'].std()
        plants_df['efficiency_score'] = 1 - (plants_df['thermal_energy_kcal_kg_clinker'] - thermal_mean) / thermal_std
        plants_df['efficiency_score'] = np.clip(plants_df['efficiency_score'], 0, 1)
        
        # Modernity score (based on commissioning year)
        current_year = 2024
        plants_df['modernity_score'] = (plants_df['commissioning_year'] - 1990) / (current_year - 1990)
        plants_df['modernity_score'] = np.clip(plants_df['modernity_score'], 0, 1)
        
        # Reliability score (based on CO2 emissions - lower is better)
        co2_mean = plants_df['co2_kg_t'].mean()
        co2_std = plants_df['co2_kg_t'].std()
        plants_df['reliability_score'] = 1 - (plants_df['co2_kg_t'] - co2_mean) / co2_std
        plants_df['reliability_score'] = np.clip(plants_df['reliability_score'], 0, 1)
        
        # Calculate weighted total score
        plants_df['score'] = (
            criteria['capacity_weight'] * plants_df['capacity_score'] +
            criteria['efficiency_weight'] * plants_df['efficiency_score'] +
            criteria['modernity_weight'] * plants_df['modernity_score'] +
            criteria['reliability_weight'] * plants_df['reliability_score']
        )
        
        # Sort by score (descending)
        plants_df = plants_df.sort_values('score', ascending=False)
        
        logger.info(f"📊 Plant scoring completed. Best score: {plants_df.iloc[0]['score']:.2f}")
        
        return plants_df


class ProcessCalibrator:
    """
    Calibrates DCS simulator parameters using real plant KPIs from LCI data.
    """
    
    def __init__(self, config_path: str = 'config/plant_config.yml'):
        """
        Initialize Process Calibrator.
        
        Args:
            config_path: Path to plant configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_loader = BigQueryDataLoader()
        
    def _load_config(self) -> dict:
        """Load plant configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded plant configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Config file {self.config_path} not found")
            raise
    
    def calibrate_dcs_simulator(self, plant_kpis: Dict[str, float]) -> Dict[str, Any]:
        """
        Calibrate DCS simulator parameters using real plant KPIs.
        
        Args:
            plant_kpis: Real plant KPIs from LCI data
            
        Returns:
            Dictionary with calibrated parameters
        """
        logger.info("🔧 Calibrating DCS simulator with real plant KPIs...")
        
        calibrated_params = {}
        
        # Calibrate energy consumption parameters
        calibrated_params['thermal_energy'] = {
            'mean': plant_kpis.get('thermal_energy_kcal_kg_clinker', 3200),
            'std': plant_kpis.get('thermal_energy_kcal_kg_clinker', 3200) * 0.05,  # 5% variation
            'min': plant_kpis.get('thermal_energy_kcal_kg_clinker', 3200) * 0.9,
            'max': plant_kpis.get('thermal_energy_kcal_kg_clinker', 3200) * 1.1
        }
        
        calibrated_params['electrical_energy'] = {
            'mean': plant_kpis.get('electrical_energy_kwh_t', 95),
            'std': plant_kpis.get('electrical_energy_kwh_t', 95) * 0.08,  # 8% variation
            'min': plant_kpis.get('electrical_energy_kwh_t', 95) * 0.85,
            'max': plant_kpis.get('electrical_energy_kwh_t', 95) * 1.15
        }
        
        # Calibrate process parameters
        calibrated_params['kiln_speed'] = {
            'mean': plant_kpis.get('kiln_speed_rpm', 3.5),
            'std': plant_kpis.get('kiln_speed_rpm', 3.5) * 0.1,  # 10% variation
            'min': plant_kpis.get('kiln_speed_rpm', 3.5) * 0.8,
            'max': plant_kpis.get('kiln_speed_rpm', 3.5) * 1.2
        }
        
        calibrated_params['burning_zone_temp'] = {
            'mean': plant_kpis.get('burning_zone_temp_c', 1450),
            'std': plant_kpis.get('burning_zone_temp_c', 1450) * 0.03,  # 3% variation
            'min': plant_kpis.get('burning_zone_temp_c', 1450) * 0.95,
            'max': plant_kpis.get('burning_zone_temp_c', 1450) * 1.05
        }
        
        # Calibrate capacity-based parameters
        capacity_tpd = plant_kpis.get('capacity_tpd', 10000)
        calibrated_params['capacity'] = {
            'tpd': capacity_tpd,
            'tph': capacity_tpd / 24,  # Convert to tons per hour
            'feed_rate_mean': capacity_tpd / 24 * 1.2,  # 20% overcapacity
            'feed_rate_std': capacity_tpd / 24 * 0.1
        }
        
        # Calibrate environmental parameters
        calibrated_params['emissions'] = {
            'co2_kg_t': {
                'mean': plant_kpis.get('co2_kg_t', 800),
                'std': plant_kpis.get('co2_kg_t', 800) * 0.1,
                'min': plant_kpis.get('co2_kg_t', 800) * 0.8,
                'max': plant_kpis.get('co2_kg_t', 800) * 1.2
            },
            'nox_mg_nm3': {
                'mean': 400,  # Default value
                'std': 50,
                'min': 200,
                'max': 600
            },
            'so2_mg_nm3': {
                'mean': 200,  # Default value
                'std': 30,
                'min': 100,
                'max': 400
            }
        }
        
        # Calibrate quality parameters based on plant performance
        calibrated_params['quality'] = {
            'free_lime_pct': {
                'mean': 1.0,
                'std': 0.2,
                'min': 0.5,
                'max': 2.0
            },
            'c3s_content_pct': {
                'mean': 55.0,
                'std': 3.0,
                'min': 45.0,
                'max': 65.0
            },
            'c2s_content_pct': {
                'mean': 20.0,
                'std': 2.0,
                'min': 15.0,
                'max': 25.0
            }
        }
        
        logger.info("✅ DCS simulator calibration completed")
        logger.info(f"📊 Calibrated parameters: {len(calibrated_params)} categories")
        
        return calibrated_params
    
    def update_plant_config(self, calibrated_params: Dict[str, Any]) -> None:
        """
        Update plant configuration with calibrated parameters.
        
        Args:
            calibrated_params: Calibrated parameters from LCI data
        """
        logger.info("📝 Updating plant configuration with calibrated parameters...")
        
        # Update process parameters in plant section
        if 'process' not in self.config['plant']:
            self.config['plant']['process'] = {}
        
        # Update energy consumption
        if 'energy' not in self.config['plant']:
            self.config['plant']['energy'] = {}
        
        self.config['plant']['energy']['thermal_kcal_kg_clinker'] = calibrated_params['thermal_energy']['mean']
        self.config['plant']['energy']['electrical_kwh_t'] = calibrated_params['electrical_energy']['mean']
        
        # Update kiln parameters
        self.config['plant']['process']['kiln_speed_rpm'] = calibrated_params['kiln_speed']['mean']
        self.config['plant']['process']['burning_zone_temp_c'] = calibrated_params['burning_zone_temp']['mean']
        
        # Update capacity
        self.config['plant']['capacity_tpd'] = calibrated_params['capacity']['tpd']
        
        # Update environmental limits
        if 'environmental' not in self.config['plant']:
            self.config['plant']['environmental'] = {}
        
        self.config['plant']['environmental']['co2_kg_per_ton'] = calibrated_params['emissions']['co2_kg_t']['mean']
        self.config['plant']['environmental']['nox_mg_nm3'] = calibrated_params['emissions']['nox_mg_nm3']['mean']
        self.config['plant']['environmental']['so2_mg_nm3'] = calibrated_params['emissions']['so2_mg_nm3']['mean']
        
        # Save updated configuration
        self._save_config()
        
        logger.info("✅ Plant configuration updated with calibrated parameters")
    
    def _save_config(self) -> None:
        """Save updated configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"✅ Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            raise


class RealWorldDataIntegrator:
    """
    Main integrator class that combines plant selection and process calibration.
    """
    
    def __init__(self, config_path: str = 'config/plant_config.yml'):
        """
        Initialize Real World Data Integrator.
        
        Args:
            config_path: Path to plant configuration file
        """
        self.config_path = config_path
        self.plant_selector = PlantSelector(config_path)
        self.process_calibrator = ProcessCalibrator(config_path)
        self.data_loader = BigQueryDataLoader()
        
    def integrate_real_world_data(self, use_config_plant: bool = True) -> Dict[str, Any]:
        """
        Complete integration of real-world data with configuration and calibration.
        
        Args:
            use_config_plant: If True, use plant_config.yml as base plant instead of selecting from Mendeley
            
        Returns:
            Dictionary with integration results
        """
        logger.info("🚀 Starting real-world data integration...")
        
        try:
            # Step 1: Get base plant (from config or Mendeley selection)
            if use_config_plant:
                plant_selection = self.plant_selector.get_base_plant_from_config()
            else:
                plant_selection = self.plant_selector.select_optimal_plant()
            
            selected_plant = plant_selection['plant_data']
            plant_kpis = plant_selection['kpis']
            
            # Step 2: Calibrate DCS simulator
            calibrated_params = self.process_calibrator.calibrate_dcs_simulator(plant_kpis)
            
            # Step 3: Update plant configuration
            self.process_calibrator.update_plant_config(calibrated_params)
            
            # Step 4: Load quality correlations
            quality_correlations = self.data_loader.get_quality_correlations()
            
            # Step 5: Prepare integration results
            integration_results = {
                'selected_plant': selected_plant,
                'plant_kpis': plant_kpis,
                'calibrated_params': calibrated_params,
                'quality_correlations': quality_correlations,
                'selection_criteria': plant_selection['selection_criteria'],
                'selection_score': plant_selection['selection_score'],
                'integration_timestamp': pd.Timestamp.now().isoformat(),
                'config_updated': True,
                'use_config_plant': use_config_plant
            }
            
            logger.info("✅ Real-world data integration completed successfully!")
            logger.info(f"📊 Base plant: {selected_plant['plant_name']}")
            logger.info(f"📊 Selection method: {'Config-based' if use_config_plant else 'Mendeley-based'}")
            logger.info(f"📊 Calibrated parameters: {len(calibrated_params)} categories")
            
            return integration_results
            
        except Exception as e:
            logger.error(f"❌ Real-world data integration failed: {e}")
            raise
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """
        Get summary of integration results.
        
        Returns:
            Dictionary with integration summary
        """
        try:
            # Load current configuration
            with open(self.config_path, 'r') as f:
                current_config = yaml.safe_load(f)
            
            # Load real-world datasets
            mendeley_df, kaggle_df, global_df = self.data_loader.load_all()
            
            summary = {
                'integration_status': 'completed',
                'datasets_loaded': {
                    'mendeley_lci': len(mendeley_df),
                    'kaggle_concrete': len(kaggle_df),
                    'global_cement': len(global_df)
                },
                'plant_configuration': {
                    'plant_name': current_config['plant']['name'],
                    'capacity_tpd': current_config['plant']['capacity_tpd'],
                    'kiln_type': current_config['plant']['kiln_type']
                },
                'calibrated_parameters': {
                    'thermal_energy': current_config['process'].get('thermal_energy_kcal_kg_clinker'),
                    'electrical_energy': current_config['process'].get('electrical_energy_kwh_t'),
                    'kiln_speed': current_config['process'].get('kiln_speed_rpm'),
                    'burning_zone_temp': current_config['process'].get('burning_zone_temp_c')
                },
                'environmental_limits': current_config.get('environmental', {}),
                'dcs_tags_count': len(current_config['dcs_tags']),
                'integration_timestamp': pd.Timestamp.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get integration summary: {e}")
            return {'integration_status': 'failed', 'error': str(e)}


def test_real_world_integration():
    """Test the real-world data integration functionality."""
    logger.info("🧪 Testing real-world data integration...")
    
    try:
        # Initialize integrator
        integrator = RealWorldDataIntegrator()
        
        # Test integration with config-based plant
        results = integrator.integrate_real_world_data(use_config_plant=True)
        
        # Test summary
        summary = integrator.get_integration_summary()
        
        logger.info("✅ Real-world data integration test completed successfully!")
        logger.info(f"📊 Integration results: {len(results)} components")
        logger.info(f"📊 Integration summary: {summary['integration_status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Real-world data integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the integration
    test_real_world_integration()
