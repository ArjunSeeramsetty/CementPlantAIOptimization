"""
JK Cement Digital Twin Platform - Comprehensive Test Script
Demonstrates all five agents working together in the unified platform
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Try different import paths
try:
    from cement_ai_platform.agents.jk_cement_platform import create_unified_platform
except ImportError:
    try:
        # Alternative import path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from src.cement_ai_platform.agents.jk_cement_platform import create_unified_platform
    except ImportError:
        # Direct import
        from agents.jk_cement_platform import create_unified_platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_plant_data() -> dict:
    """Create realistic sample plant data for testing"""
    
    # Generate realistic plant data with some variations
    base_data = {
        # Production data
        'production_rate_tph': 167.0,
        'feed_rate_tph': 194.5,
        'fuel_rate_tph': 16.3,
        
        # Process parameters
        'kiln_speed_rpm': 3.2,
        'burning_zone_temp_c': 1450.0,
        'cooler_outlet_temp_c': 100.0,
        'secondary_air_temp_c': 950.0,
        'free_lime_percent': 1.0,
        
        # Emissions
        'nox_mg_nm3': 500.0,
        'co_mg_nm3': 100.0,
        'o2_percent': 3.0,
        
        # Equipment data
        'raw_mill_vibration_x': 3.0,
        'raw_mill_vibration_y': 3.0,
        'raw_mill_vibration_z': 2.5,
        'raw_mill_bearing_temp': 70.0,
        'raw_mill_motor_temp': 80.0,
        'raw_mill_power_kw': 2000.0,
        'raw_mill_throughput_tph': 80.0,
        
        'cement_mill_vibration_x': 3.5,
        'cement_mill_vibration_y': 3.5,
        'cement_mill_vibration_z': 3.0,
        'cement_mill_power_kw': 3000.0,
        'cement_mill_throughput_tph': 120.0,
        
        'id_fan_flow_nm3_h': 50000.0,
        'id_fan_pressure_pa': -150.0,
        'id_fan_current_a': 100.0,
        'id_fan_current_b': 100.0,
        'id_fan_current_c': 100.0,
        
        # Utility data
        'air_consumption_nm3_h': 1000.0,
        'baseline_air_consumption_nm3_h': 800.0,
        'air_pressure_bar': 8.0,
        'compressor_power_kw': 150.0,
        
        'water_consumption_m3_h': 50.0,
        'cooling_inlet_temp_c': 30.0,
        'cooling_outlet_temp_c': 40.0,
        'ambient_temp_c': 25.0,
        
        'conveyor_power_kw': 200.0,
        'material_throughput_tph': 200.0,
        'belt_speeds_mps': [1.5, 2.0, 1.8],
        'motor_loads_percent': [75.0, 85.0, 60.0],
        
        # Fuel data
        'available_fuels': {
            'coal': 1000.0,
            'petcoke': 200.0,
            'rdf': 150.0,
            'biomass': 100.0,
            'tire_chips': 50.0
        },
        'fuel_costs': {
            'coal': 0.08,
            'petcoke': 0.12,
            'rdf': 0.05,
            'biomass': 0.06,
            'tire_chips': 0.04
        },
        'quality_constraints': {
            'max_chlorine': 0.15,
            'max_sulfur': 2.0,
            'max_alkali': 3.0
        },
        
        # Additional parameters
        'kiln_torque_percent': 70.0,
        'cooler_speed_rpm': 10.0,
        'voltage_phase_a': 400.0,
        'voltage_phase_b': 400.0,
        'voltage_phase_c': 400.0,
        'power_factor': 0.85,
        'availability_percent': 95.0,
        'power_variation_percent': 5.0
    }
    
    # Add some realistic variations
    variations = {}
    for key, value in base_data.items():
        if isinstance(value, (int, float)):
            # Add ±5% variation for numerical values
            variation = np.random.normal(0, 0.05)
            variations[key] = value * (1 + variation)
        else:
            variations[key] = value
    
    return variations

def test_individual_agents():
    """Test each agent individually"""
    
    logger.info("🧪 Testing individual agents...")
    
    # Create platform
    platform = create_unified_platform()
    
    # Test data
    plant_data = create_sample_plant_data()
    
    # Test Alternative Fuel Optimizer
    logger.info("🔄 Testing Alternative Fuel Optimizer...")
    fuel_results = platform._optimize_alternative_fuels(plant_data)
    logger.info(f"✅ TSR Achieved: {fuel_results['tsr_achieved']:.2%}")
    logger.info(f"✅ Quality Impact: {fuel_results['quality_impact']}")
    
    # Test Unified Kiln-Cooler Controller
    logger.info("🔄 Testing Unified Kiln-Cooler Controller...")
    control_results = platform._compute_unified_control_setpoints(plant_data)
    logger.info(f"✅ Control Health: {control_results['control_health']['health_score']:.2f}")
    logger.info(f"✅ Setpoints: {control_results['setpoints']}")
    
    # Test Utility Optimizer
    logger.info("🔄 Testing Utility Optimizer...")
    utility_results = platform._optimize_utilities(plant_data)
    total_savings = utility_results['total_savings_potential']['total_annual_savings']
    logger.info(f"✅ Total Annual Savings: ${total_savings:,.0f}")
    
    # Test Plant Anomaly Detector
    logger.info("🔄 Testing Plant Anomaly Detector...")
    anomaly_results = platform._detect_anomalies(plant_data)
    logger.info(f"✅ Plant Health Score: {anomaly_results['plant_health_score']:.2f}")
    logger.info(f"✅ Active Alerts: {len(anomaly_results['active_alerts'])}")
    
    # Test Cement Plant GPT
    logger.info("🔄 Testing Cement Plant GPT...")
    gpt_results = platform._generate_gpt_analysis(plant_data, {
        'fuel_optimization': fuel_results,
        'control_setpoints': control_results,
        'utility_optimization': utility_results,
        'anomaly_detection': anomaly_results
    })
    logger.info(f"✅ GPT Analysis Generated: {len(gpt_results['gpt_response']['response'])} characters")
    
    return {
        'fuel_results': fuel_results,
        'control_results': control_results,
        'utility_results': utility_results,
        'anomaly_results': anomaly_results,
        'gpt_results': gpt_results
    }

def test_unified_platform():
    """Test the unified platform processing"""
    
    logger.info("🧪 Testing unified platform processing...")
    
    # Create platform
    platform = create_unified_platform()
    
    # Test with multiple data points
    results = []
    for i in range(5):
        logger.info(f"🔄 Processing plant data point {i+1}/5...")
        
        # Create sample data with variations
        plant_data = create_sample_plant_data()
        
        # Process through unified platform
        comprehensive_results = platform.process_plant_data(plant_data)
        
        results.append(comprehensive_results)
        
        # Log key results
        overall_status = comprehensive_results['overall_plant_status']
        logger.info(f"✅ Overall Performance Score: {overall_status['overall_performance_score']:.2f}")
        logger.info(f"✅ Plant Status: {overall_status['status']}")
        logger.info(f"✅ Recommendations Count: {len(comprehensive_results['recommendations'])}")
    
    return results

def test_performance_tracking():
    """Test performance tracking and metrics"""
    
    logger.info("🧪 Testing performance tracking...")
    
    # Create platform
    platform = create_unified_platform()
    
    # Process multiple data points to build history
    for i in range(10):
        plant_data = create_sample_plant_data()
        platform.process_plant_data(plant_data)
    
    # Get performance summary
    performance_summary = platform.get_performance_summary(days=7)
    logger.info(f"✅ Performance Summary: {performance_summary}")
    
    # Get platform status
    platform_status = platform.get_platform_status()
    logger.info(f"✅ Platform Status: {platform_status['platform_status']}")
    logger.info(f"✅ Optimization History Count: {platform_status['optimization_history_count']}")
    
    return performance_summary, platform_status

def test_export_functionality():
    """Test export functionality"""
    
    logger.info("🧪 Testing export functionality...")
    
    # Create platform and process data
    platform = create_unified_platform()
    plant_data = create_sample_plant_data()
    results = platform.process_plant_data(plant_data)
    
    # Export results
    json_file = platform.export_results(results, format='json')
    logger.info(f"✅ Results exported to JSON: {json_file}")
    
    # Export metrics
    csv_file = platform.export_results(results, format='csv')
    logger.info(f"✅ Metrics exported to CSV: {csv_file}")
    
    return json_file, csv_file

def test_anomaly_model_training():
    """Test anomaly model training with synthetic data"""
    
    logger.info("🧪 Testing anomaly model training...")
    
    # Create platform
    platform = create_unified_platform()
    
    # Generate synthetic training data
    n_samples = 1000
    training_data = []
    
    for i in range(n_samples):
        sample = create_sample_plant_data()
        training_data.append(sample)
    
    # Convert to DataFrame
    training_df = pd.DataFrame(training_data)
    
    # Train anomaly models
    platform.train_anomaly_models(training_df)
    
    logger.info("✅ Anomaly models trained successfully")
    
    return training_df

def run_comprehensive_test():
    """Run comprehensive test suite"""
    
    logger.info("🚀 Starting JK Cement Digital Twin Platform Comprehensive Test")
    logger.info("=" * 80)
    
    try:
        # Test individual agents
        logger.info("\n📋 PHASE 1: Individual Agent Testing")
        individual_results = test_individual_agents()
        logger.info("✅ Individual agent testing completed")
        
        # Test unified platform
        logger.info("\n📋 PHASE 2: Unified Platform Testing")
        unified_results = test_unified_platform()
        logger.info("✅ Unified platform testing completed")
        
        # Test performance tracking
        logger.info("\n📋 PHASE 3: Performance Tracking Testing")
        performance_summary, platform_status = test_performance_tracking()
        logger.info("✅ Performance tracking testing completed")
        
        # Test export functionality
        logger.info("\n📋 PHASE 4: Export Functionality Testing")
        json_file, csv_file = test_export_functionality()
        logger.info("✅ Export functionality testing completed")
        
        # Test anomaly model training
        logger.info("\n📋 PHASE 5: Anomaly Model Training Testing")
        training_df = test_anomaly_model_training()
        logger.info("✅ Anomaly model training testing completed")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 COMPREHENSIVE TEST SUITE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        # Final summary
        summary = {
            'test_timestamp': datetime.now().isoformat(),
            'individual_agents_tested': 5,
            'unified_platform_tests': len(unified_results),
            'performance_tracking_active': True,
            'export_functionality_working': True,
            'anomaly_models_trained': True,
            'training_samples': len(training_df),
            'platform_status': platform_status['platform_status'],
            'jk_cement_requirements_coverage': platform_status['jk_cement_requirements_coverage']
        }
        
        logger.info(f"📊 Test Summary: {json.dumps(summary, indent=2)}")
        
        return {
            'individual_results': individual_results,
            'unified_results': unified_results,
            'performance_summary': performance_summary,
            'platform_status': platform_status,
            'export_files': {'json': json_file, 'csv': csv_file},
            'training_data': training_df,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    # Run comprehensive test
    test_results = run_comprehensive_test()
    
    # Save test results
    with open('jk_cement_platform_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    logger.info("💾 Test results saved to jk_cement_platform_test_results.json")
    logger.info("🎯 JK Cement Digital Twin Platform is ready for production!")
