"""
JK Cement Requirements Demonstration Script
Comprehensive test of all JK Cement-specific requirements
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cement_ai_platform.models.agents.jk_cement_platform import JKCementDigitalTwinPlatform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_alternative_fuel_optimization(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 1: Alternative Fuel Optimization"""

    logger.info("\n" + "="*60)
    logger.info("🔥 TESTING JK CEMENT REQUIREMENT 1: ALTERNATIVE FUEL OPTIMIZATION")
    logger.info("="*60)

    # Test fuel optimization
    fuel_results = platform.optimize_alternative_fuel(
        plant_data['available_fuels'],
        plant_data['base_clinker_properties']
    )

    logger.info(f"✅ Alternative fuel optimization completed")
    assert fuel_results is not None

    # Test fuel recommendations
    current_blend = {'coal': 0.6, 'petcoke': 0.2, 'rdf': 0.15, 'biomass': 0.05}
    recommendations = platform.get_fuel_recommendations(current_blend)

    logger.info(f"📋 Fuel Recommendations Status: {recommendations['status']}")
    assert recommendations is not None


def test_cement_plant_gpt(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 2: Cement Plant GPT Interface"""

    logger.info("\n" + "="*60)
    logger.info("🤖 TESTING JK CEMENT REQUIREMENT 2: CEMENT PLANT GPT INTERFACE")
    logger.info("="*60)

    # Test various GPT queries
    queries = [
        "What's the current plant status?",
        "Analyze the clinker quality",
        "How can we optimize energy consumption?",
        "What's causing high free lime in the kiln?",
        "Recommend maintenance actions for the raw mill"
    ]

    for i, query in enumerate(queries, 1):
        logger.info(f"\n📝 Query {i}: {query}")
        response = platform.ask_plant_gpt(query, plant_data)
        logger.info(f"🤖 Response: {response[:200]}...")
        assert response is not None

    # Test specific analysis functions
    logger.info(f"\n📊 Plant Status Analysis:")
    status_response = platform.get_plant_status_gpt(plant_data['sensor_data'])
    logger.info(f"🤖 Status: {status_response[:150]}...")
    assert status_response is not None

    logger.info(f"\n📊 Quality Analysis:")
    quality_response = platform.get_quality_analysis_gpt(plant_data['quality_data'])
    logger.info(f"🤖 Quality: {quality_response[:150]}...")
    assert quality_response is not None

    logger.info(f"\n📊 Energy Optimization:")
    energy_response = platform.get_energy_optimization_gpt(plant_data['energy_data'])
    logger.info(f"🤖 Energy: {energy_response[:150]}...")
    assert energy_response is not None


def test_unified_controller(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 3: Unified Kiln-Cooler Controller"""

    logger.info("\n" + "="*60)
    logger.info("🎛️ TESTING JK CEMENT REQUIREMENT 3: UNIFIED KILN-COOLER CONTROLLER")
    logger.info("="*60)

    # Test unified setpoint computation
    setpoints = platform.compute_unified_setpoints(plant_data['sensor_data'])

    logger.info(f"✅ Unified setpoints computed successfully")
    assert setpoints is not None

    # Test control performance
    performance = platform.get_control_performance()
    logger.info(f"📊 Control Performance:")
    assert performance is not None


def test_utility_optimization(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 4: Utility Optimization"""

    logger.info("\n" + "="*60)
    logger.info("⚡ TESTING JK CEMENT REQUIREMENT 4: UTILITY OPTIMIZATION")
    logger.info("="*60)

    # Test utility optimization
    utility_results = platform.optimize_all_utilities(
        plant_data['utility_data']['pressure_data'],
        plant_data['utility_data']['flow_data'],
        plant_data['utility_data']['handling_data']
    )

    logger.info(f"✅ Utility optimization completed successfully")
    assert utility_results is not None


def test_anomaly_detection(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 5: Plant Anomaly Detection"""

    logger.info("\n" + "="*60)
    logger.info("🚨 TESTING JK CEMENT REQUIREMENT 5: PLANT ANOMALY DETECTION")
    logger.info("="*60)

    # Test anomaly detection
    anomaly_results = platform.detect_plant_anomalies(plant_data['equipment_sensor_data'])

    logger.info(f"✅ Anomaly detection completed successfully")
    assert anomaly_results is not None

    # Test equipment health report
    health_report = platform.get_equipment_health_report()
    logger.info(f"📊 Equipment Health Report:")
    assert health_report is not None


def test_comprehensive_workflow(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test comprehensive JK Cement workflow"""

    logger.info("\n" + "="*60)
    logger.info("🚀 TESTING COMPREHENSIVE JK CEMENT WORKFLOW")
    logger.info("="*60)

    # Run comprehensive workflow
    workflow_results = platform.run_jk_cement_optimization_workflow(plant_data)

    logger.info(f"✅ Comprehensive workflow completed successfully")
    assert workflow_results is not None


def test_platform_compliance(platform: JKCementDigitalTwinPlatform):
    """Test JK Cement compliance validation"""

    logger.info("\n" + "="*60)
    logger.info("✅ TESTING JK CEMENT COMPLIANCE VALIDATION")
    logger.info("="*60)

    # Test compliance validation
    compliance = platform.validate_jk_cement_compliance()

    logger.info(f"✅ Compliance validation completed")
    assert compliance is not None

    # Platform status
    platform_status = platform.get_platform_status()
    logger.info(f"📊 Platform Status:")
    assert platform_status is not None
