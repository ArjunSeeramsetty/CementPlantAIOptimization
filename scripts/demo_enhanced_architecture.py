"""Comprehensive demo script for the enhanced Digital Twin POC."""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_sourcing.fetch_data import download_all_datasets
from simulation.dcs_simulator import generate_dcs_data
from simulation.process_models import create_process_models, RawMealComposition, CoalProperties
from training.train_gan import generate_massive_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_data_sourcing():
    """Demonstrate data sourcing capabilities."""
    logger.info("=== DEMO: Data Sourcing ===")
    
    try:
        # Download datasets
        results = download_all_datasets()
        
        logger.info("Data Sourcing Results:")
        for dataset, status in results.items():
            status_str = "✅ SUCCESS" if status else "❌ FAILED"
            logger.info(f"  {dataset}: {status_str}")
        
        return results
        
    except Exception as e:
        logger.error(f"Data sourcing demo failed: {e}")
        return {}


def demo_dcs_simulation():
    """Demonstrate high-frequency DCS simulation."""
    logger.info("=== DEMO: DCS Simulation ===")
    
    try:
        # Generate 1 hour of high-frequency data
        dcs_data = generate_dcs_data(
            duration_hours=1,
            sample_rate_seconds=1,
            output_path='data/processed/demo_dcs_data.csv'
        )
        
        logger.info(f"Generated DCS data:")
        logger.info(f"  Records: {len(dcs_data):,}")
        logger.info(f"  Tags: {len(dcs_data.columns)}")
        logger.info(f"  Duration: {dcs_data.index[-1] - dcs_data.index[0]}")
        logger.info(f"  Memory: {dcs_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Show sample statistics
        logger.info("\nSample DCS Statistics:")
        sample_stats = dcs_data.describe().round(2)
        logger.info(sample_stats.head())
        
        return {
            'records': len(dcs_data),
            'tags': len(dcs_data.columns),
            'duration_hours': (dcs_data.index[-1] - dcs_data.index[0]).total_seconds() / 3600,
            'memory_mb': dcs_data.memory_usage(deep=True).sum() / 1024**2
        }
        
    except Exception as e:
        logger.error(f"DCS simulation demo failed: {e}")
        return {}


def demo_process_models():
    """Demonstrate expert-driven process models."""
    logger.info("=== DEMO: Process Models ===")
    
    try:
        # Create process models
        models = create_process_models()
        
        # Test raw meal composition
        raw_meal = RawMealComposition(
            sio2=22.0, cao=65.0, al2o3=5.0, fe2o3=3.0,
            mgo=2.5, k2o=0.8, na2o=0.3, so3=0.1, cl=0.05, loi=35.0
        )
        
        # Test coal properties
        coal = CoalProperties(
            volatile_matter=35.0, ash_content=15.0, moisture=8.0,
            sulfur=1.2, calorific_value=6500
        )
        
        # Test kiln performance
        kiln_performance = models['kiln_model'].calculate_kiln_performance(
            raw_meal, 3200, coal, 3.5, 200
        )
        
        logger.info("Kiln Performance Results:")
        for key, value in kiln_performance.items():
            logger.info(f"  {key}: {value:.2f}")
        
        # Test quality prediction
        clinker_composition = {'C3S': 60, 'C2S': 15, 'C3A': 8, 'C4AF': 10}
        strength_28d = models['quality_predictor'].predict_compressive_strength(
            clinker_composition, 3500, 28
        )
        
        logger.info(f"Predicted 28-day strength: {strength_28d:.1f} MPa")
        
        # Test preheater tower
        raw_meal_alkali = {'K2O': 0.8, 'Na2O': 0.3, 'SO3': 0.1, 'Cl': 0.05}
        preheater_results = models['preheater_tower'].calculate_heat_and_mass_balance(
            200, 50000, raw_meal_alkali
        )
        
        logger.info("Preheater Tower Results:")
        for key, value in preheater_results.items():
            logger.info(f"  {key}: {value:.2f}")
        
        return {
            'kiln_performance': kiln_performance,
            'strength_28d_mpa': strength_28d,
            'preheater_results': preheater_results
        }
        
    except Exception as e:
        logger.error(f"Process models demo failed: {e}")
        return {}


def demo_data_augmentation():
    """Demonstrate TimeGAN data augmentation."""
    logger.info("=== DEMO: Data Augmentation ===")
    
    try:
        # Generate massive dataset (smaller for demo)
        massive_data = generate_massive_dataset(
            base_data_path='data/processed/demo_dcs_data.csv',
            num_samples=10000,  # Smaller for demo
            duration_hours=24,
            output_path='data/processed/demo_massive_dataset.csv'
        )
        
        logger.info(f"Generated massive dataset:")
        logger.info(f"  Records: {len(massive_data):,}")
        logger.info(f"  Features: {len(massive_data.columns)}")
        logger.info(f"  Memory: {massive_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Show scenario distribution
        if 'operational_scenario' in massive_data.columns:
            scenario_counts = massive_data['operational_scenario'].value_counts()
            logger.info("\nOperational Scenario Distribution:")
            for scenario, count in scenario_counts.items():
                percentage = count / len(massive_data) * 100
                logger.info(f"  {scenario}: {count:,} ({percentage:.1f}%)")
        
        return {
            'records': len(massive_data),
            'features': len(massive_data.columns),
            'memory_mb': massive_data.memory_usage(deep=True).sum() / 1024**2,
            'scenarios': massive_data['operational_scenario'].value_counts().to_dict() if 'operational_scenario' in massive_data.columns else {}
        }
        
    except Exception as e:
        logger.error(f"Data augmentation demo failed: {e}")
        return {}


def demo_integration_with_existing():
    """Demonstrate integration with existing BigQuery data."""
    logger.info("=== DEMO: Integration with Existing Data ===")
    
    try:
        # Create sample data in the same format as existing BigQuery tables
        # (Note: Import would be from existing cement_ai_platform if available)
        
        # Create sample data in the same format as existing BigQuery tables
        sample_data = {
            'operational_parameters': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
                'plant_id': ['PLANT_001'] * 100,
                'unit_id': ['KILN_01'] * 100,
                'parameter_type': ['temperature'] * 100,
                'parameter_name': ['burning_zone_temp'] * 100,
                'value': np.random.normal(1450, 20, 100),
                'unit': ['°C'] * 100,
                'quality_flag': ['good'] * 100,
                'sensor_id': ['TEMP_001'] * 100,
                'ingestion_timestamp': pd.date_range('2024-01-01', periods=100, freq='1H')
            }),
            
            'quality_metrics': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=50, freq='2H'),
                'plant_id': ['PLANT_001'] * 50,
                'batch_id': [f'BATCH_{i:03d}' for i in range(50)],
                'product_type': ['clinker'] * 50,
                'test_type': ['compressive_strength'] * 50,
                'metric_name': ['28d_strength'] * 50,
                'measured_value': np.random.normal(45, 5, 50),
                'specification_min': [40] * 50,
                'specification_max': [55] * 50,
                'pass_fail': [True] * 50,
                'lab_technician': ['TECH_001'] * 50,
                'equipment_id': ['LAB_001'] * 50,
                'ingestion_timestamp': pd.date_range('2024-01-01', periods=50, freq='2H')
            })
        }
        
        logger.info("Created sample data compatible with existing BigQuery schema:")
        for table_name, df in sample_data.items():
            logger.info(f"  {table_name}: {len(df)} records, {len(df.columns)} columns")
        
        return {
            'tables_created': len(sample_data),
            'total_records': sum(len(df) for df in sample_data.values()),
            'schema_compatibility': True
        }
        
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        return {}


def run_comprehensive_demo():
    """Run the complete comprehensive demo."""
    logger.info("🚀 Starting Comprehensive Digital Twin POC Demo")
    logger.info("=" * 60)
    
    demo_results = {}
    
    # Run all demos
    demos = [
        ("Data Sourcing", demo_data_sourcing),
        ("DCS Simulation", demo_dcs_simulation),
        ("Process Models", demo_process_models),
        ("Data Augmentation", demo_data_augmentation),
        ("Integration", demo_integration_with_existing)
    ]
    
    for demo_name, demo_func in demos:
        try:
            logger.info(f"\n🎯 Running {demo_name} Demo...")
            results = demo_func()
            demo_results[demo_name.lower().replace(' ', '_')] = results
            logger.info(f"✅ {demo_name} Demo completed successfully")
        except Exception as e:
            logger.error(f"❌ {demo_name} Demo failed: {e}")
            demo_results[demo_name.lower().replace(' ', '_')] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'artifacts/comprehensive_demo_results_{timestamp}.json'
    
    os.makedirs('artifacts', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    # Create summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPREHENSIVE DEMO SUMMARY")
    logger.info("=" * 60)
    
    successful_demos = sum(1 for results in demo_results.values() if 'error' not in results)
    total_demos = len(demos)
    
    logger.info(f"Successful Demos: {successful_demos}/{total_demos}")
    logger.info(f"Results saved to: {results_file}")
    
    # Show key metrics
    if 'dcs_simulation' in demo_results and 'error' not in demo_results['dcs_simulation']:
        dcs_results = demo_results['dcs_simulation']
        logger.info(f"\n📈 Key Metrics:")
        logger.info(f"  DCS Records Generated: {dcs_results.get('records', 0):,}")
        logger.info(f"  DCS Tags: {dcs_results.get('tags', 0)}")
        logger.info(f"  Data Augmentation Records: {demo_results.get('data_augmentation', {}).get('records', 0):,}")
    
    logger.info("\n🎉 Comprehensive Demo Completed!")
    
    return demo_results


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the comprehensive demo
    results = run_comprehensive_demo()
    
    # Exit with appropriate code
    successful_demos = sum(1 for results in results.values() if 'error' not in results)
    if successful_demos == len(results):
        logger.info("All demos completed successfully!")
        sys.exit(0)
    else:
        logger.warning("Some demos failed. Check logs for details.")
        sys.exit(1)
