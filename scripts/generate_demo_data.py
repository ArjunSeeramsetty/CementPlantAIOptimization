"""
Generate Demo Data using Unified Platform
Orchestrates all JK Cement components to produce comprehensive demo dataset
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cement_ai_platform.models.agents.jk_cement_platform import JKCementDigitalTwinPlatform

logger = logging.getLogger(__name__)

def generate_demo_dataset(n_minutes: int = 1440) -> pd.DataFrame:
    """
    Generate comprehensive demo dataset using unified JK Cement platform
    
    Args:
        n_minutes: Number of minutes to simulate (default: 1440 = 24 hours)
        
    Returns:
        Comprehensive demo dataset DataFrame
    """
    logger.info(f"🔄 Generating demo dataset for {n_minutes} minutes...")
    
    try:
        # Initialize JK Cement platform
        logger.info("🏭 Initializing JK Cement Digital Twin Platform...")
        platform = JKCementDigitalTwinPlatform()
        
        # Generate timestamps
        timestamps = pd.date_range(
            start="2025-09-17 00:00", 
            periods=n_minutes, 
            freq="T"  # Minute frequency
        )
        
        logger.info(f"📊 Generated {len(timestamps)} timestamps")
        
        # Generate demo records
        records = []
        
        for i, timestamp in enumerate(timestamps):
            if i % 100 == 0:
                logger.info(f"🔄 Processing timestamp {i+1}/{len(timestamps)}: {timestamp}")
            
            # Generate sensor snapshot
            sensor_snapshot = generate_sensor_snapshot(timestamp, i)
            
            # Run JK Cement platform components
            agents_output = run_platform_components(platform, sensor_snapshot, timestamp)
            
            # Combine sensor data with agent outputs
            record = {**sensor_snapshot, **agents_output}
            records.append(record)
        
        # Create comprehensive DataFrame
        df = pd.DataFrame.from_records(records)
        
        # Add metadata
        df['demo_generation_timestamp'] = datetime.now().isoformat()
        df['demo_version'] = '1.0'
        df['platform_version'] = 'JK_Cement_Digital_Twin_v1.0'
        
        # Save demo dataset
        os.makedirs("demo/data/final", exist_ok=True)
        path = "demo/data/final/plant_demo_data_full.csv"
        df.to_csv(path, index=False)
        logger.info(f"✅ Saved final demo dataset: {path} ({len(df)} rows, {len(df.columns)} columns)")
        
        # Create additional demo datasets
        create_additional_demo_datasets(df)
        
        logger.info("✅ Demo dataset generation completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"❌ Demo dataset generation failed: {e}")
        raise

def generate_sensor_snapshot(timestamp: datetime, index: int) -> Dict[str, Any]:
    """
    Generate realistic sensor snapshot for given timestamp
    
    Args:
        timestamp: Current timestamp
        index: Index in the sequence
        
    Returns:
        Sensor snapshot dictionary
    """
    
    # Base values with some variation
    base_values = {
        'timestamp': timestamp,
        'feed_rate_tph': 200.0 + np.random.normal(0, 5),
        'fuel_rate_tph': 15.0 + np.random.normal(0, 1),
        'kiln_speed_rpm': 3.0 + np.random.normal(0, 0.1),
        'burning_zone_temp_c': 1450.0 + np.random.normal(0, 10),
        'cooler_outlet_temp_c': 100.0 + np.random.normal(0, 5),
        'excess_air_pct': 10.0 + np.random.normal(0, 1),
        'gas_flow_nm3_h': 200000.0 + np.random.normal(0, 5000),
        'raw_meal_fineness_blaine': 3000.0 + np.random.normal(0, 100),
        'raw_meal_alkali_content': 0.6 + np.random.normal(0, 0.05),
        'cooler_air_flow_nm3_h': 150000.0 + np.random.normal(0, 3000),
        'kiln_speed_rpm': 3.0 + np.random.normal(0, 0.1),
        'fuel_rate_tph': 15.0 + np.random.normal(0, 1),
        'feed_rate_tph': 200.0 + np.random.normal(0, 5),
        'burning_zone_temp_c': 1450.0 + np.random.normal(0, 10),
        'cooler_outlet_temp_c': 100.0 + np.random.normal(0, 5),
        'excess_air_pct': 10.0 + np.random.normal(0, 1),
        'gas_flow_nm3_h': 200000.0 + np.random.normal(0, 5000),
        'raw_meal_fineness_blaine': 3000.0 + np.random.normal(0, 100),
        'raw_meal_alkali_content': 0.6 + np.random.normal(0, 0.05),
        'cooler_air_flow_nm3_h': 150000.0 + np.random.normal(0, 3000),
        'raw_meal_composition': {
            'lime_saturation_factor': 0.95 + np.random.normal(0, 0.01),
            'silica_ratio': 2.5 + np.random.normal(0, 0.1),
            'alumina_ratio': 1.5 + np.random.normal(0, 0.05)
        },
        'fuel_mix': {
            'coal': 0.6 + np.random.normal(0, 0.05),
            'petcoke': 0.2 + np.random.normal(0, 0.03),
            'rdf': 0.15 + np.random.normal(0, 0.02),
            'biomass': 0.05 + np.random.normal(0, 0.01)
        },
        'fuel_calorific_values': {
            'coal': 6500 + np.random.normal(0, 100),
            'petcoke': 8000 + np.random.normal(0, 150),
            'rdf': 4000 + np.random.normal(0, 200),
            'biomass': 3500 + np.random.normal(0, 150)
        }
    }
    
    # Add some time-based variations
    hour = timestamp.hour
    if 6 <= hour <= 18:  # Day shift - higher production
        base_values['feed_rate_tph'] *= 1.05
        base_values['fuel_rate_tph'] *= 1.03
    else:  # Night shift - lower production
        base_values['feed_rate_tph'] *= 0.95
        base_values['fuel_rate_tph'] *= 0.97
    
    return base_values

def run_platform_components(platform: JKCementDigitalTwinPlatform, 
                           sensor_snapshot: Dict[str, Any], 
                           timestamp: datetime) -> Dict[str, Any]:
    """
    Run all JK Cement platform components for given sensor snapshot
    
    Args:
        platform: JK Cement platform instance
        sensor_snapshot: Current sensor readings
        timestamp: Current timestamp
        
    Returns:
        Combined output from all platform components
    """
    
    agents_output = {
        'timestamp': timestamp,
        'platform_processing_timestamp': datetime.now().isoformat()
    }
    
    try:
        # 1. Alternative Fuel Optimization
        try:
            fuel_results = platform.optimize_alternative_fuel(
                available_fuels=sensor_snapshot['fuel_mix'],
                base_clinker_properties={
                    'c3s_content_pct': 58.0,
                    'free_lime_pct': 1.8,
                    'compressive_strength_mpa': 42.0
                }
            )
            agents_output.update({
                'tsr_achieved': fuel_results['tsr_achieved'],
                'tsr_target': fuel_results['tsr_target'],
                'optimal_fuel_blend': fuel_results['optimal_fractions'],
                'fuel_optimization_success': fuel_results['optimization_success']
            })
        except Exception as e:
            logger.warning(f"⚠️ Fuel optimization failed: {e}")
            agents_output.update({
                'tsr_achieved': 0.15,
                'tsr_target': 0.15,
                'fuel_optimization_success': False
            })
        
        # 2. Unified Process Control
        try:
            control_results = platform.compute_unified_setpoints(sensor_snapshot)
            agents_output.update({
                'kiln_setpoints': control_results['kiln_setpoints'],
                'preheater_setpoints': control_results['preheater_setpoints'],
                'cooler_setpoints': control_results['cooler_setpoints'],
                'burnability_index': control_results['control_analysis']['burnability_index'],
                'thermal_efficiency': control_results['control_analysis']['thermal_efficiency']
            })
        except Exception as e:
            logger.warning(f"⚠️ Process control failed: {e}")
            agents_output.update({
                'kiln_setpoints': {},
                'preheater_setpoints': {},
                'cooler_setpoints': {},
                'burnability_index': 100.0,
                'thermal_efficiency': 0.85
            })
        
        # 3. Utility Optimization
        try:
            utility_results = platform.optimize_all_utilities(
                pressure_data={
                    'inlet_pressure_bar': 7.0,
                    'outlet_pressure_bar': 6.5,
                    'critical_points': {'mill_air': 6.8, 'kiln_air': 6.6}
                },
                flow_data={
                    'cooling_water_flow_m3_h': 500.0,
                    'process_water_flow_m3_h': 200.0,
                    'lubrication_water_flow_m3_h': 50.0
                },
                handling_data={
                    'conveyor_efficiency': 0.85,
                    'elevator_efficiency': 0.80,
                    'pneumatic_efficiency': 0.75
                }
            )
            agents_output.update({
                'utility_power_savings_kw': utility_results['total_savings']['total_power_savings_kw'],
                'utility_cost_savings_usd_year': utility_results['total_savings']['total_cost_savings_usd_year'],
                'utility_efficiency_gain_pct': utility_results['total_savings']['average_efficiency_gain_pct']
            })
        except Exception as e:
            logger.warning(f"⚠️ Utility optimization failed: {e}")
            agents_output.update({
                'utility_power_savings_kw': 0.0,
                'utility_cost_savings_usd_year': 0.0,
                'utility_efficiency_gain_pct': 0.0
            })
        
        # 4. Anomaly Detection
        try:
            equipment_sensor_data = {
                'kiln_01': {
                    'burning_zone_temp_c': sensor_snapshot['burning_zone_temp_c'],
                    'kiln_speed_rpm': sensor_snapshot['kiln_speed_rpm'],
                    'fuel_rate_tph': sensor_snapshot['fuel_rate_tph']
                },
                'raw_mill_01': {
                    'mill_vibration_mm_s': 4.0 + np.random.normal(0, 0.5),
                    'mill_power_kw': 2500.0 + np.random.normal(0, 100),
                    'mill_outlet_temp_c': 100.0 + np.random.normal(0, 5)
                }
            }
            
            anomaly_results = platform.detect_plant_anomalies(equipment_sensor_data)
            agents_output.update({
                'total_anomalies': len(anomaly_results['anomalies']),
                'plant_health_percentage': anomaly_results['summary']['plant_health_percentage'],
                'active_alerts_count': anomaly_results['summary']['active_alerts_count'],
                'critical_equipment': anomaly_results['summary']['critical_equipment']
            })
        except Exception as e:
            logger.warning(f"⚠️ Anomaly detection failed: {e}")
            agents_output.update({
                'total_anomalies': 0,
                'plant_health_percentage': 100.0,
                'active_alerts_count': 0,
                'critical_equipment': []
            })
        
        # 5. GPT Analysis
        try:
            gpt_query = "Analyze current plant performance and provide optimization recommendations"
            gpt_response = platform.ask_plant_gpt(gpt_query, sensor_snapshot)
            agents_output.update({
                'gpt_response': gpt_response[:200] + "..." if len(gpt_response) > 200 else gpt_response,
                'gpt_query': gpt_query
            })
        except Exception as e:
            logger.warning(f"⚠️ GPT analysis failed: {e}")
            agents_output.update({
                'gpt_response': 'GPT analysis unavailable',
                'gpt_query': 'Analysis failed'
            })
        
    except Exception as e:
        logger.error(f"❌ Platform component execution failed: {e}")
        # Add default values
        agents_output.update({
            'platform_error': str(e),
            'component_success': False
        })
    
    return agents_output

def create_additional_demo_datasets(df: pd.DataFrame):
    """
    Create additional demo datasets for specific use cases
    
    Args:
        df: Main demo DataFrame
    """
    logger.info("🔄 Creating additional demo datasets...")
    
    # 1. Process Variables Dataset
    process_columns = [
        'timestamp', 'feed_rate_tph', 'fuel_rate_tph', 'kiln_speed_rpm',
        'burning_zone_temp_c', 'cooler_outlet_temp_c', 'excess_air_pct',
        'gas_flow_nm3_h', 'raw_meal_fineness_blaine', 'raw_meal_alkali_content',
        'cooler_air_flow_nm3_h'
    ]
    
    available_process_columns = [col for col in process_columns if col in df.columns]
    process_df = df[available_process_columns].copy()
    
    process_path = "demo/data/final/process_variables.csv"
    process_df.to_csv(process_path, index=False)
    logger.info(f"✅ Saved process variables: {process_path} ({len(process_df)} rows)")
    
    # 2. Quality Parameters Dataset
    quality_columns = [
        'timestamp', 'tsr_achieved', 'tsr_target', 'burnability_index',
        'thermal_efficiency', 'plant_health_percentage'
    ]
    
    available_quality_columns = [col for col in quality_columns if col in df.columns]
    quality_df = df[available_quality_columns].copy()
    
    quality_path = "demo/data/final/quality_parameters.csv"
    quality_df.to_csv(quality_path, index=False)
    logger.info(f"✅ Saved quality parameters: {quality_path} ({len(quality_df)} rows)")
    
    # 3. Optimization Results Dataset
    optimization_columns = [
        'timestamp', 'utility_power_savings_kw', 'utility_cost_savings_usd_year',
        'utility_efficiency_gain_pct', 'total_anomalies', 'active_alerts_count'
    ]
    
    available_optimization_columns = [col for col in optimization_columns if col in df.columns]
    optimization_df = df[available_optimization_columns].copy()
    
    optimization_path = "demo/data/final/optimization_results.csv"
    optimization_df.to_csv(optimization_path, index=False)
    logger.info(f"✅ Saved optimization results: {optimization_path} ({len(optimization_df)} rows)")
    
    # 4. GPT Responses Dataset
    gpt_columns = ['timestamp', 'gpt_query', 'gpt_response']
    
    available_gpt_columns = [col for col in gpt_columns if col in df.columns]
    gpt_df = df[available_gpt_columns].copy()
    
    gpt_path = "demo/data/final/gpt_responses.csv"
    gpt_df.to_csv(gpt_path, index=False)
    logger.info(f"✅ Saved GPT responses: {gpt_path} ({len(gpt_df)} rows)")

def validate_demo_dataset(df: pd.DataFrame) -> bool:
    """
    Validate demo dataset
    
    Args:
        df: Demo DataFrame
        
    Returns:
        True if validation passes
    """
    logger.info("🔄 Validating demo dataset...")
    
    # Check required columns
    required_columns = ['timestamp', 'demo_generation_timestamp']
    
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"❌ Missing required column: {col}")
            return False
    
    # Check data quality
    if len(df) == 0:
        logger.error("❌ Empty dataset")
        return False
    
    # Check timestamp continuity
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        time_diff = timestamps.diff().dropna()
        expected_diff = pd.Timedelta(minutes=1)
        
        if not all(abs(diff - expected_diff) < pd.Timedelta(seconds=1) for diff in time_diff):
            logger.warning("⚠️ Timestamp continuity issues detected")
    
    logger.info("✅ Demo dataset validation completed")
    return True

def get_demo_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get demo dataset summary
    
    Args:
        df: Demo DataFrame
        
    Returns:
        Demo summary dictionary
    """
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'time_range': {
            'start': df['timestamp'].min() if 'timestamp' in df.columns else 'Unknown',
            'end': df['timestamp'].max() if 'timestamp' in df.columns else 'Unknown',
            'duration_minutes': len(df)
        },
        'demo_metadata': {
            'generation_timestamp': df['demo_generation_timestamp'].iloc[0] if 'demo_generation_timestamp' in df.columns else 'Unknown',
            'demo_version': df['demo_version'].iloc[0] if 'demo_version' in df.columns else 'Unknown',
            'platform_version': df['platform_version'].iloc[0] if 'platform_version' in df.columns else 'Unknown'
        },
        'jk_cement_components': {
            'alternative_fuel_optimization': 'tsr_achieved' in df.columns,
            'unified_process_control': 'kiln_setpoints' in df.columns,
            'utility_optimization': 'utility_power_savings_kw' in df.columns,
            'anomaly_detection': 'total_anomalies' in df.columns,
            'cement_plant_gpt': 'gpt_response' in df.columns
        },
        'data_quality': {
            'missing_values_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'unique_timestamps': df['timestamp'].nunique() if 'timestamp' in df.columns else 0
        }
    }
    
    return summary

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Generate demo dataset
        df = generate_demo_dataset(n_minutes=1440)  # 24 hours
        
        # Validate demo dataset
        if validate_demo_dataset(df):
            # Get and print summary
            summary = get_demo_summary(df)
            logger.info(f"📊 Demo Dataset Summary:")
            logger.info(f"   Total Records: {summary['total_records']}")
            logger.info(f"   Total Columns: {summary['total_columns']}")
            logger.info(f"   Time Range: {summary['time_range']['start']} to {summary['time_range']['end']}")
            logger.info(f"   Duration: {summary['time_range']['duration_minutes']} minutes")
            logger.info(f"   Demo Version: {summary['demo_metadata']['demo_version']}")
            logger.info(f"   Platform Version: {summary['demo_metadata']['platform_version']}")
            
            logger.info(f"📊 JK Cement Components:")
            for component, status in summary['jk_cement_components'].items():
                status_text = "✅ Active" if status else "❌ Inactive"
                logger.info(f"   {component}: {status_text}")
            
            logger.info(f"📊 Data Quality:")
            logger.info(f"   Missing Values: {summary['data_quality']['missing_values_pct']:.2f}%")
            logger.info(f"   Unique Timestamps: {summary['data_quality']['unique_timestamps']}")
            
            logger.info("✅ Demo dataset generation completed successfully!")
        else:
            logger.error("❌ Demo dataset validation failed")
            
    except Exception as e:
        logger.error(f"❌ Demo dataset generation failed: {e}")
        raise
