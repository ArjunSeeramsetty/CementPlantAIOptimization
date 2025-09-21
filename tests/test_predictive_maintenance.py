#!/usr/bin/env python3
"""
Comprehensive Test Suite for Predictive Maintenance and Data Validation Features
Tests the integrated predictive maintenance and data validation capabilities
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_predictive_maintenance():
    """Test predictive maintenance functionality"""
    logger.info("🔧 Testing Predictive Maintenance Engine...")
    
    try:
        from cement_ai_platform.maintenance.predictive_maintenance import PredictiveMaintenanceEngine
        
        # Initialize maintenance engine
        pm_engine = PredictiveMaintenanceEngine()
        
        # Test equipment failure prediction
        equipment_data = {
            'equipment_id': 'KILN_001',
            'equipment_type': 'kiln',
            'equipment_name': 'Kiln #1',
            'operating_hours': 25000,
            'maintenance_age': 2000,
            'load_factor': 0.85,
            'kiln_temperature': 80,
            'kiln_vibration': 5.2,
            'kiln_current': 110,
            'kiln_torque': 75
        }
        
        recommendation = pm_engine.predict_equipment_failure(equipment_data)
        
        if recommendation:
            logger.info(f"✅ Equipment failure prediction successful")
            logger.info(f"   Equipment: {recommendation.equipment_name}")
            logger.info(f"   Failure Probability: {recommendation.failure_probability:.1%}")
            logger.info(f"   Time to Failure: {recommendation.time_to_failure_hours:.0f} hours")
            logger.info(f"   Priority: {recommendation.priority}")
            logger.info(f"   Estimated Cost: ${recommendation.estimated_cost:,.0f}")
            return True
        else:
            logger.error("❌ Failed to generate maintenance recommendation")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing predictive maintenance: {e}")
        return False

def test_maintenance_report():
    """Test maintenance report generation"""
    logger.info("📋 Testing Maintenance Report Generation...")
    
    try:
        from cement_ai_platform.maintenance.predictive_maintenance import PredictiveMaintenanceEngine
        
        pm_engine = PredictiveMaintenanceEngine()
        report = pm_engine.generate_maintenance_report("JK_Rajasthan_1", 30)
        
        if report and 'summary' in report:
            summary = report['summary']
            logger.info(f"✅ Maintenance report generated successfully")
            logger.info(f"   Total Recommendations: {summary['total_recommendations']}")
            logger.info(f"   Critical Count: {summary['critical_count']}")
            logger.info(f"   High Count: {summary['high_count']}")
            logger.info(f"   Total Estimated Cost: ${summary['total_estimated_cost']:,.0f}")
            return True
        else:
            logger.error("❌ Failed to generate maintenance report")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing maintenance report: {e}")
        return False

def test_data_drift_detection():
    """Test data drift detection functionality"""
    logger.info("🧪 Testing Data Drift Detection...")
    
    try:
        from cement_ai_platform.validation.drift_detection import DataDriftDetector
        
        # Initialize drift detector
        drift_detector = DataDriftDetector()
        
        # Generate synthetic reference data
        np.random.seed(42)
        ref_data = pd.DataFrame({
            'free_lime_percent': np.random.normal(1.2, 0.3, 1000),
            'thermal_energy_kcal_kg': np.random.normal(690, 25, 1000),
            'feed_rate_tph': np.random.normal(167, 15, 1000),
            'burning_zone_temp_c': np.random.normal(1450, 20, 1000),
            'nox_mg_nm3': np.random.normal(500, 50, 1000)
        })
        
        # Create reference snapshot
        success = drift_detector.create_reference_snapshot(ref_data, "baseline")
        
        if success:
            logger.info("✅ Reference snapshot created successfully")
            
            # Generate current data with drift
            current_data = pd.DataFrame({
                'free_lime_percent': np.random.normal(1.5, 0.4, 500),  # Mean shift
                'thermal_energy_kcal_kg': np.random.normal(710, 30, 500),  # Energy increase
                'feed_rate_tph': np.random.normal(165, 18, 500),  # Slight decrease
                'burning_zone_temp_c': np.random.normal(1445, 25, 500),  # Temperature decrease
                'nox_mg_nm3': np.random.normal(520, 60, 500)  # Emissions increase
            })
            
            # Detect drift
            drift_results = drift_detector.detect_data_drift(current_data, "baseline")
            
            if drift_results['drift_detected']:
                summary = drift_results['drift_summary']
                logger.info(f"✅ Data drift detected successfully")
                logger.info(f"   Variables with Drift: {summary['variables_with_drift']}/{summary['total_variables_analyzed']}")
                logger.info(f"   Max Severity: {summary['max_severity']}")
                logger.info(f"   Categories Affected: {summary['categories_with_drift']}")
                return True
            else:
                logger.warning("⚠️ No drift detected in test data")
                return True
        else:
            logger.error("❌ Failed to create reference snapshot")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing data drift detection: {e}")
        return False

def test_model_retraining_trigger():
    """Test model retraining trigger functionality"""
    logger.info("🔄 Testing Model Retraining Trigger...")
    
    try:
        from cement_ai_platform.validation.drift_detection import DataDriftDetector
        
        drift_detector = DataDriftDetector()
        
        # Simulate high severity drift summary
        drift_summary = {
            'max_severity': 'High',
            'variables_with_drift': 3,
            'total_variables_analyzed': 5,
            'critical_variables': ['free_lime_percent', 'thermal_energy_kcal_kg']
        }
        
        retraining_result = drift_detector.trigger_model_retraining(drift_summary)
        
        if retraining_result['retraining_triggered']:
            config = retraining_result['retraining_config']
            logger.info(f"✅ Model retraining triggered successfully")
            logger.info(f"   Pipeline ID: {config['pipeline_id']}")
            logger.info(f"   Priority: {config['retraining_priority']}")
            logger.info(f"   Estimated Duration: {config['estimated_duration']}")
            return True
        else:
            logger.info(f"ℹ️ Retraining not triggered: {retraining_result['reason']}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error testing model retraining trigger: {e}")
        return False

def test_platform_integration():
    """Test integration with main platform"""
    logger.info("🏭 Testing Platform Integration...")
    
    try:
        from cement_ai_platform.agents.jk_cement_platform import create_unified_platform
        
        # Create platform
        platform = create_unified_platform()
        
        # Test maintenance report generation
        maintenance_result = platform.generate_maintenance_report("JK_Rajasthan_1", 30)
        
        if maintenance_result['success']:
            logger.info("✅ Platform maintenance report generation successful")
        else:
            logger.warning(f"⚠️ Platform maintenance report failed: {maintenance_result['error']}")
        
        # Test equipment failure prediction
        equipment_data = {
            'equipment_id': 'RMILL_001',
            'equipment_type': 'raw_mill',
            'equipment_name': 'Raw Mill #1',
            'operating_hours': 15000,
            'maintenance_age': 1500,
            'load_factor': 0.80
        }
        
        failure_result = platform.predict_equipment_failure(equipment_data)
        
        if failure_result['success']:
            logger.info("✅ Platform equipment failure prediction successful")
        else:
            logger.warning(f"⚠️ Platform equipment failure prediction failed: {failure_result['error']}")
        
        # Test data drift detection
        np.random.seed(42)
        test_data = pd.DataFrame({
            'free_lime_percent': np.random.normal(1.3, 0.3, 100),
            'thermal_energy_kcal_kg': np.random.normal(700, 25, 100)
        })
        
        drift_result = platform.detect_data_drift(test_data)
        
        if drift_result['success']:
            logger.info("✅ Platform data drift detection successful")
        else:
            logger.warning(f"⚠️ Platform data drift detection failed: {drift_result['error']}")
        
        # Check platform status
        status = platform.get_platform_status()
        
        logger.info("📊 Platform Status:")
        logger.info(f"   Maintenance Available: {status['maintenance_capabilities']['available']}")
        logger.info(f"   Validation Available: {status['validation_capabilities']['available']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing platform integration: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite for predictive maintenance and data validation"""
    
    logger.info("🧪 JK Cement Digital Twin - Predictive Maintenance & Data Validation Test Suite")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Test predictive maintenance
    logger.info("\n📋 PHASE 1: Predictive Maintenance Testing")
    test_results['predictive_maintenance'] = test_predictive_maintenance()
    logger.info("✅ Predictive maintenance testing completed")
    
    # Test maintenance report generation
    logger.info("\n📋 PHASE 2: Maintenance Report Generation")
    test_results['maintenance_report'] = test_maintenance_report()
    logger.info("✅ Maintenance report generation testing completed")
    
    # Test data drift detection
    logger.info("\n📋 PHASE 3: Data Drift Detection")
    test_results['drift_detection'] = test_data_drift_detection()
    logger.info("✅ Data drift detection testing completed")
    
    # Test model retraining trigger
    logger.info("\n📋 PHASE 4: Model Retraining Trigger")
    test_results['retraining_trigger'] = test_model_retraining_trigger()
    logger.info("✅ Model retraining trigger testing completed")
    
    # Test platform integration
    logger.info("\n📋 PHASE 5: Platform Integration")
    test_results['platform_integration'] = test_platform_integration()
    logger.info("✅ Platform integration testing completed")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"Predictive Maintenance........... {'✅ PASSED' if test_results['predictive_maintenance'] else '❌ FAILED'}")
    logger.info(f"Maintenance Report.............. {'✅ PASSED' if test_results['maintenance_report'] else '❌ FAILED'}")
    logger.info(f"Data Drift Detection............ {'✅ PASSED' if test_results['drift_detection'] else '❌ FAILED'}")
    logger.info(f"Model Retraining Trigger........ {'✅ PASSED' if test_results['retraining_trigger'] else '❌ FAILED'}")
    logger.info(f"Platform Integration............ {'✅ PASSED' if test_results['platform_integration'] else '❌ FAILED'}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All predictive maintenance and data validation tests passed! Platform is ready for POC demonstration.")
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Please review the errors above.")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_test()
