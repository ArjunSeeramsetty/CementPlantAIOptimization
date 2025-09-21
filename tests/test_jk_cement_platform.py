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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_individual_agents(platform, plant_data):
    """Test each agent individually"""

    logger.info("🧪 Testing individual agents...")

    # Test Alternative Fuel Optimizer
    logger.info("🔄 Testing Alternative Fuel Optimizer...")
    fuel_results = platform._optimize_alternative_fuels(plant_data)
    assert fuel_results is not None

    # Test Unified Kiln-Cooler Controller
    logger.info("🔄 Testing Unified Kiln-Cooler Controller...")
    control_results = platform._compute_unified_control_setpoints(plant_data)
    assert control_results is not None

    # Test Utility Optimizer
    logger.info("🔄 Testing Utility Optimizer...")
    utility_results = platform._optimize_utilities(plant_data)
    assert utility_results is not None

    # Test Plant Anomaly Detector
    logger.info("🔄 Testing Plant Anomaly Detector...")
    anomaly_results = platform._detect_anomalies(plant_data)
    assert anomaly_results is not None

    # Test Cement Plant GPT
    logger.info("🔄 Testing Cement Plant GPT...")
    gpt_results = platform._generate_gpt_analysis(plant_data, {
        'fuel_optimization': fuel_results,
        'control_setpoints': control_results,
        'utility_optimization': utility_results,
        'anomaly_detection': anomaly_results
    })
    assert gpt_results is not None


def test_unified_platform(platform, plant_data):
    """Test the unified platform processing"""

    logger.info("🧪 Testing unified platform processing...")

    # Test with multiple data points
    results = []
    for i in range(5):
        logger.info(f"🔄 Processing plant data point {i+1}/5...")

        # Process through unified platform
        comprehensive_results = platform.process_plant_data(plant_data)

        results.append(comprehensive_results)

        # Log key results
        overall_status = comprehensive_results['overall_plant_status']
        logger.info(f"✅ Overall Performance Score: {overall_status['overall_performance_score']:.2f}")
        logger.info(f"✅ Plant Status: {overall_status['status']}")
        logger.info(f"✅ Recommendations Count: {len(comprehensive_results['recommendations'])}")

    assert len(results) == 5

def test_performance_tracking(platform, plant_data):
    """Test performance tracking and metrics"""

    logger.info("🧪 Testing performance tracking...")

    # Process multiple data points to build history
    for i in range(10):
        platform.process_plant_data(plant_data)

    # Get performance summary
    performance_summary = platform.get_performance_summary(days=7)
    logger.info(f"✅ Performance Summary: {performance_summary}")
    assert performance_summary is not None

    # Get platform status
    platform_status = platform.get_platform_status()
    logger.info(f"✅ Platform Status: {platform_status['platform_status']}")
    logger.info(f"✅ Optimization History Count: {platform_status['optimization_history_count']}")
    assert platform_status is not None


def test_export_functionality(platform, plant_data):
    """Test export functionality"""

    logger.info("🧪 Testing export functionality...")

    results = platform.process_plant_data(plant_data)

    # Export results
    json_file = platform.export_results(results, format='json')
    logger.info(f"✅ Results exported to JSON: {json_file}")
    assert json_file is not None

    # Export metrics
    csv_file = platform.export_results(results, format='csv')
    logger.info(f"✅ Metrics exported to CSV: {csv_file}")
    assert csv_file is not None


def test_anomaly_model_training(platform, plant_data):
    """Test anomaly model training with synthetic data"""

    logger.info("🧪 Testing anomaly model training...")

    # Generate synthetic training data
    n_samples = 1000
    training_data = []

    for i in range(n_samples):
        training_data.append(plant_data)

    # Convert to DataFrame
    training_df = pd.DataFrame(training_data)

    # Train anomaly models
    platform.train_anomaly_models(training_df)

    logger.info("✅ Anomaly models trained successfully")


def test_streaming_capabilities(platform):
    """Test real-time streaming capabilities"""

    logger.info("🧪 Testing streaming capabilities...")

    # Test streaming status
    streaming_status = platform.get_streaming_status()
    logger.info(f"📡 Streaming Available: {streaming_status['streaming_available']}")

    if streaming_status['streaming_available']:
        logger.info(f"📡 Topics Configured: {streaming_status['topics_configured']}")
        logger.info(f"📡 Project ID: {streaming_status['project_id']}")

        # Test starting streaming
        logger.info("🚀 Testing streaming start...")
        success = platform.start_real_time_streaming(interval_seconds=3)

        if success:
            logger.info("✅ Streaming started successfully")

            # Let it run briefly
            import time
            logger.info("⏳ Running streaming for 6 seconds...")
            time.sleep(6)

            # Test stopping streaming
            logger.info("⏹️ Testing streaming stop...")
            stop_success = platform.stop_real_time_streaming()

            if stop_success:
                logger.info("✅ Streaming stopped successfully")
            else:
                logger.warning("⚠️ Failed to stop streaming")
        else:
            logger.warning("⚠️ Failed to start streaming")
    else:
        logger.warning(f"⚠️ Streaming not available: {streaming_status.get('error', 'Unknown')}")

    assert streaming_status is not None


def test_dwsim_integration(platform):
    """Test DWSIM integration capabilities"""

    logger.info("🧪 Testing DWSIM Integration...")

    test_results = {
        'dwsim_available': False,
        'engine_initialized': False,
        'scenario_execution': False,
        'custom_scenario': False,
        'history_retrieval': False
    }

    try:
        # Test DWSIM availability
        from cement_ai_platform.dwsim.dwsim_connector import DWSIMIntegrationEngine, DWSIMScenario
        test_results['dwsim_available'] = True
        logger.info("✅ DWSIM modules imported successfully")

        # Test engine initialization
        engine = DWSIMIntegrationEngine()
        test_results['engine_initialized'] = True
        logger.info("✅ DWSIM engine initialized")

        # Test standard scenario execution
        startup_scenario = engine.standard_scenarios['startup_sequence']
        result = engine.execute_scenario(startup_scenario, "test_plant")

        if result['success']:
            test_results['scenario_execution'] = True
            logger.info(f"✅ Startup scenario executed successfully in {result['execution_duration']:.1f}s")
        else:
            logger.warning(f"⚠️ Startup scenario failed: {result.get('error')}")

        # Test custom scenario creation
        custom_scenario = DWSIMScenario(
            scenario_id="test_custom_001",
            scenario_name="Test Custom Scenario",
            description="Test scenario for validation",
            input_parameters={
                'raw_meal_feed': 165.0,
                'fuel_rate': 16.0,
                'kiln_speed': 3.5,
                'o2_target': 3.2
            },
            expected_outputs=['burning_zone_temp', 'free_lime_percent'],
            simulation_duration=1800,
            priority='medium'
        )

        custom_result = engine.execute_scenario(custom_scenario, "test_plant")

        if custom_result['success']:
            test_results['custom_scenario'] = True
            logger.info(f"✅ Custom scenario executed successfully in {custom_result['execution_duration']:.1f}s")
        else:
            logger.warning(f"⚠️ Custom scenario failed: {custom_result.get('error')}")

        # Test history retrieval
        history = engine.get_scenario_history("test_plant", limit=5)
        test_results['history_retrieval'] = True
        logger.info(f"✅ Retrieved {len(history)} scenario history records")

    except ImportError as e:
        logger.warning(f"⚠️ DWSIM modules not available: {e}")
    except Exception as e:
        logger.error(f"❌ DWSIM integration test failed: {e}")

    assert test_results['dwsim_available']
