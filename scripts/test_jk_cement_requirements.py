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

def create_sample_plant_data() -> Dict[str, Any]:
    """Create comprehensive sample plant data for testing"""
    
    return {
        # Sensor data for unified control
        'sensor_data': {
            'feed_rate_tph': 200.0,
            'fuel_rate_tph': 15.0,
            'kiln_speed_rpm': 3.0,
            'burning_zone_temp_c': 1450.0,
            'cooler_outlet_temp_c': 100.0,
            'gas_flow_nm3_h': 200000.0,
            'raw_meal_composition': {
                'lime_saturation_factor': 0.95,
                'silica_ratio': 2.5,
                'alumina_ratio': 1.5
            },
            'raw_meal_fineness_blaine': 3000,
            'raw_meal_alkali_content': 0.6,
            'cooler_air_flow_nm3_h': 150000.0
        },
        
        # Available fuels for optimization
        'available_fuels': {
            'coal': 20.0,  # tph
            'petcoke': 5.0,
            'rdf': 8.0,
            'biomass': 3.0,
            'tire_derived_fuel': 2.0
        },
        
        # Base clinker properties
        'base_clinker_properties': {
            'c3s_content_pct': 58.0,
            'free_lime_pct': 1.8,
            'compressive_strength_mpa': 42.0,
            'alkali_content_pct': 0.5
        },
        
        # Utility system data
        'utility_data': {
            'pressure_data': {
                'inlet_pressure_bar': 7.0,
                'outlet_pressure_bar': 6.5,
                'critical_points': {
                    'mill_air': 6.8,
                    'kiln_air': 6.6,
                    'cooler_air': 6.4
                }
            },
            'flow_data': {
                'cooling_water_flow_m3_h': 500.0,
                'process_water_flow_m3_h': 200.0,
                'lubrication_water_flow_m3_h': 50.0
            },
            'handling_data': {
                'conveyor_efficiency': 0.85,
                'elevator_efficiency': 0.80,
                'pneumatic_efficiency': 0.75,
                'conveyor_power_kw': 200.0,
                'elevator_power_kw': 150.0,
                'pneumatic_power_kw': 100.0
            }
        },
        
        # Equipment sensor data for anomaly detection
        'equipment_sensor_data': {
            'kiln_01': {
                'burning_zone_temp_c': 1450.0,
                'kiln_speed_rpm': 3.0,
                'fuel_rate_tph': 15.0
            },
            'raw_mill_01': {
                'mill_vibration_mm_s': 4.5,
                'mill_power_kw': 2500.0,
                'mill_outlet_temp_c': 100.0
            },
            'cement_mill_01': {
                'mill_vibration_mm_s': 3.2,
                'mill_power_kw': 2000.0,
                'fineness_blaine_cm2_g': 3500.0
            },
            'id_fan_01': {
                'fan_vibration_mm_s': 5.0,
                'fan_power_kw': 1000.0,
                'fan_pressure_pa': -150.0
            }
        },
        
        # Quality data for GPT analysis
        'quality_data': {
            'c3s_content_pct': 58.0,
            'free_lime_pct': 1.8,
            'compressive_strength_28d_mpa': 42.0,
            'production_tph': 200.0,
            'specific_power_kwh_t': 110.0
        },
        
        # Energy data for optimization
        'energy_data': {
            'specific_power_kwh_t': 110.0,
            'thermal_energy_kcal_kg': 690.0,
            'electrical_energy_kwh_t': 110.0,
            'co2_kg_t': 850.0
        }
    }

def test_alternative_fuel_optimization(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 1: Alternative Fuel Optimization"""
    
    logger.info("\n" + "="*60)
    logger.info("🔥 TESTING JK CEMENT REQUIREMENT 1: ALTERNATIVE FUEL OPTIMIZATION")
    logger.info("="*60)
    
    try:
        # Test fuel optimization
        fuel_results = platform.optimize_alternative_fuel(
            plant_data['available_fuels'],
            plant_data['base_clinker_properties']
        )
        
        logger.info(f"✅ Alternative fuel optimization completed")
        logger.info(f"📊 TSR Achieved: {fuel_results['tsr_achieved']:.3f} ({fuel_results['tsr_achieved']*100:.1f}%)")
        logger.info(f"📊 TSR Target: {fuel_results['tsr_target']:.3f} ({fuel_results['tsr_target']*100:.1f}%)")
        logger.info(f"📊 TSR Improvement: {fuel_results['tsr_improvement_pct']:.1f}%")
        logger.info(f"📊 Quality Penalty: {fuel_results['quality_penalty']:.3f}")
        
        # Test fuel recommendations
        current_blend = {'coal': 0.6, 'petcoke': 0.2, 'rdf': 0.15, 'biomass': 0.05}
        recommendations = platform.get_fuel_recommendations(current_blend)
        
        logger.info(f"📋 Fuel Recommendations Status: {recommendations['status']}")
        if recommendations['status'] == 'optimization_needed':
            logger.info(f"📋 Current TSR: {recommendations['current_tsr']:.3f}")
            logger.info(f"📋 Target TSR: {recommendations['target_tsr']:.3f}")
            logger.info(f"📋 TSR Gap: {recommendations['tsr_gap']:.3f}")
        
        return fuel_results
        
    except Exception as e:
        logger.error(f"❌ Alternative fuel optimization test failed: {e}")
        return None

def test_cement_plant_gpt(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 2: Cement Plant GPT Interface"""
    
    logger.info("\n" + "="*60)
    logger.info("🤖 TESTING JK CEMENT REQUIREMENT 2: CEMENT PLANT GPT INTERFACE")
    logger.info("="*60)
    
    try:
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
        
        # Test specific analysis functions
        logger.info(f"\n📊 Plant Status Analysis:")
        status_response = platform.get_plant_status_gpt(plant_data['sensor_data'])
        logger.info(f"🤖 Status: {status_response[:150]}...")
        
        logger.info(f"\n📊 Quality Analysis:")
        quality_response = platform.get_quality_analysis_gpt(plant_data['quality_data'])
        logger.info(f"🤖 Quality: {quality_response[:150]}...")
        
        logger.info(f"\n📊 Energy Optimization:")
        energy_response = platform.get_energy_optimization_gpt(plant_data['energy_data'])
        logger.info(f"🤖 Energy: {energy_response[:150]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cement Plant GPT test failed: {e}")
        return False

def test_unified_controller(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 3: Unified Kiln-Cooler Controller"""
    
    logger.info("\n" + "="*60)
    logger.info("🎛️ TESTING JK CEMENT REQUIREMENT 3: UNIFIED KILN-COOLER CONTROLLER")
    logger.info("="*60)
    
    try:
        # Test unified setpoint computation
        setpoints = platform.compute_unified_setpoints(plant_data['sensor_data'])
        
        logger.info(f"✅ Unified setpoints computed successfully")
        logger.info(f"📊 Kiln Setpoints:")
        for key, value in setpoints['kiln_setpoints'].items():
            logger.info(f"   {key}: {value}")
        
        logger.info(f"📊 Preheater Setpoints:")
        logger.info(f"   Stage Temperatures: {setpoints['preheater_setpoints']['stage_temperatures']}")
        logger.info(f"   Stage Efficiencies: {setpoints['preheater_setpoints']['stage_efficiencies']}")
        
        logger.info(f"📊 Cooler Setpoints:")
        for key, value in setpoints['cooler_setpoints'].items():
            logger.info(f"   {key}: {value}")
        
        logger.info(f"📊 Control Analysis:")
        for key, value in setpoints['control_analysis'].items():
            logger.info(f"   {key}: {value}")
        
        # Test control performance
        performance = platform.get_control_performance()
        logger.info(f"📊 Control Performance:")
        logger.info(f"   Average Temperature Deviation: {performance['control_performance']['avg_temperature_deviation_c']:.2f}°C")
        logger.info(f"   Average Thermal Efficiency: {performance['control_performance']['avg_thermal_efficiency']:.3f}")
        logger.info(f"   Control Stability: {performance['control_performance']['control_stability']}")
        
        return setpoints
        
    except Exception as e:
        logger.error(f"❌ Unified controller test failed: {e}")
        return None

def test_utility_optimization(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 4: Utility Optimization"""
    
    logger.info("\n" + "="*60)
    logger.info("⚡ TESTING JK CEMENT REQUIREMENT 4: UTILITY OPTIMIZATION")
    logger.info("="*60)
    
    try:
        # Test utility optimization
        utility_results = platform.optimize_all_utilities(
            plant_data['utility_data']['pressure_data'],
            plant_data['utility_data']['flow_data'],
            plant_data['utility_data']['handling_data']
        )
        
        logger.info(f"✅ Utility optimization completed successfully")
        
        # Air optimization results
        air_results = utility_results['air_optimization']
        logger.info(f"📊 Compressed Air Optimization:")
        logger.info(f"   System Status: {air_results['pressure_analysis']['system_status']}")
        logger.info(f"   Leak Status: {air_results['leak_analysis']['leak_status']}")
        logger.info(f"   Expected Power Savings: {air_results['expected_savings']['power_reduction_kw']:.1f} kW")
        logger.info(f"   Expected Cost Savings: ${air_results['expected_savings']['cost_savings_usd_year']:,.0f}/year")
        
        # Water optimization results
        water_results = utility_results['water_optimization']
        logger.info(f"📊 Water Optimization:")
        logger.info(f"   Usage Status: {water_results['usage_analysis']['usage_status']}")
        logger.info(f"   Expected Water Savings: {water_results['expected_savings']['water_reduction_m3_h']:.1f} m³/h")
        logger.info(f"   Expected Cost Savings: ${water_results['expected_savings']['cost_savings_usd_year']:,.0f}/year")
        
        # Material handling optimization results
        material_results = utility_results['material_handling_optimization']
        logger.info(f"📊 Material Handling Optimization:")
        logger.info(f"   Efficiency Status: {material_results['efficiency_analysis']['efficiency_status']}")
        logger.info(f"   Expected Power Savings: {material_results['expected_savings']['power_reduction_kw']:.1f} kW")
        logger.info(f"   Expected Cost Savings: ${material_results['expected_savings']['cost_savings_usd_year']:,.0f}/year")
        
        # Total savings
        total_savings = utility_results['total_savings']
        logger.info(f"📊 Total Utility Savings:")
        logger.info(f"   Total Power Savings: {total_savings['total_power_savings_kw']:.1f} kW")
        logger.info(f"   Total Cost Savings: ${total_savings['total_cost_savings_usd_year']:,.0f}/year")
        logger.info(f"   Average Efficiency Gain: {total_savings['average_efficiency_gain_pct']:.1f}%")
        logger.info(f"   ROI Period: {total_savings['roi_period_years']:.1f} years")
        
        return utility_results
        
    except Exception as e:
        logger.error(f"❌ Utility optimization test failed: {e}")
        return None

def test_anomaly_detection(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test JK Cement Requirement 5: Plant Anomaly Detection"""
    
    logger.info("\n" + "="*60)
    logger.info("🚨 TESTING JK CEMENT REQUIREMENT 5: PLANT ANOMALY DETECTION")
    logger.info("="*60)
    
    try:
        # Test anomaly detection
        anomaly_results = platform.detect_plant_anomalies(plant_data['equipment_sensor_data'])
        
        logger.info(f"✅ Anomaly detection completed successfully")
        logger.info(f"📊 Total Anomalies Found: {len(anomaly_results['anomalies'])}")
        
        # Equipment status
        logger.info(f"📊 Equipment Status:")
        for equipment_id, status in anomaly_results['equipment_status'].items():
            logger.info(f"   {equipment_id}: {status['status']} ({status['severity']})")
        
        # Anomaly summary
        summary = anomaly_results['summary']
        logger.info(f"📊 Anomaly Summary:")
        logger.info(f"   Plant Health: {summary['plant_health_percentage']:.1f}%")
        logger.info(f"   Active Alerts: {summary['active_alerts_count']}")
        logger.info(f"   Critical Equipment: {summary['critical_equipment']}")
        
        # Severity distribution
        severity_dist = summary['severity_distribution']
        logger.info(f"📊 Severity Distribution:")
        for severity, count in severity_dist.items():
            logger.info(f"   {severity}: {count}")
        
        # Test equipment health report
        health_report = platform.get_equipment_health_report()
        logger.info(f"📊 Equipment Health Report:")
        logger.info(f"   Report Timestamp: {health_report['report_timestamp']}")
        logger.info(f"   Equipment Summary: {len(health_report['equipment_summary'])} equipment")
        
        return anomaly_results
        
    except Exception as e:
        logger.error(f"❌ Anomaly detection test failed: {e}")
        return None

def test_comprehensive_workflow(platform: JKCementDigitalTwinPlatform, plant_data: Dict[str, Any]):
    """Test comprehensive JK Cement workflow"""
    
    logger.info("\n" + "="*60)
    logger.info("🚀 TESTING COMPREHENSIVE JK CEMENT WORKFLOW")
    logger.info("="*60)
    
    try:
        # Run comprehensive workflow
        workflow_results = platform.run_jk_cement_optimization_workflow(plant_data)
        
        logger.info(f"✅ Comprehensive workflow completed successfully")
        logger.info(f"📊 Requirements Met: {len(workflow_results['requirements_met'])}/5")
        
        # Requirements status
        logger.info(f"📊 Requirements Status:")
        for requirement in workflow_results['requirements_met']:
            logger.info(f"   ✅ {requirement}")
        
        # Expected benefits
        benefits = workflow_results['expected_benefits']
        logger.info(f"📊 Expected Benefits:")
        logger.info(f"   Fuel Cost Savings: ${benefits['fuel_cost_savings_usd_year']:,.0f}/year")
        logger.info(f"   Utility Cost Savings: ${benefits['utility_cost_savings_usd_year']:,.0f}/year")
        logger.info(f"   Total Cost Savings: ${benefits['total_cost_savings_usd_year']:,.0f}/year")
        logger.info(f"   Energy Reduction: {benefits['energy_reduction_pct']:.1f}%")
        logger.info(f"   TSR Achievement: {benefits['tsr_achievement_pct']:.1f}%")
        logger.info(f"   ROI Period: {benefits['roi_period_years']:.1f} years")
        logger.info(f"   Carbon Footprint Reduction: {benefits['carbon_footprint_reduction_tco2_year']:.0f} tCO2/year")
        
        # Recommendations
        logger.info(f"📊 Recommendations:")
        for i, rec in enumerate(workflow_results['recommendations'], 1):
            logger.info(f"   {i}. [{rec['priority']}] {rec['action']}")
            logger.info(f"      Benefit: {rec['expected_benefit']}")
        
        return workflow_results
        
    except Exception as e:
        logger.error(f"❌ Comprehensive workflow test failed: {e}")
        return None

def test_platform_compliance(platform: JKCementDigitalTwinPlatform):
    """Test JK Cement compliance validation"""
    
    logger.info("\n" + "="*60)
    logger.info("✅ TESTING JK CEMENT COMPLIANCE VALIDATION")
    logger.info("="*60)
    
    try:
        # Test compliance validation
        compliance = platform.validate_jk_cement_compliance()
        
        logger.info(f"✅ Compliance validation completed")
        logger.info(f"📊 Overall Compliance: {compliance['overall_compliance']}")
        
        # Requirements status
        logger.info(f"📊 Requirements Status:")
        for requirement, status in compliance['requirements_status'].items():
            logger.info(f"   {status} {requirement}")
        
        # Platform status
        platform_status = platform.get_platform_status()
        logger.info(f"📊 Platform Status:")
        logger.info(f"   Initialized: {platform_status['platform_status']['initialized']}")
        logger.info(f"   Components Loaded: {platform_status['platform_status']['components_loaded']}")
        logger.info(f"   Last Updated: {platform_status['platform_status']['last_updated']}")
        
        return compliance
        
    except Exception as e:
        logger.error(f"❌ Compliance validation test failed: {e}")
        return None

def main():
    """Main test function"""
    
    logger.info("🏭 JK CEMENT DIGITAL TWIN PLATFORM - COMPREHENSIVE TESTING")
    logger.info("="*80)
    
    try:
        # Initialize platform
        logger.info("🔄 Initializing JK Cement Digital Twin Platform...")
        platform = JKCementDigitalTwinPlatform()
        
        # Create sample plant data
        logger.info("🔄 Creating sample plant data...")
        plant_data = create_sample_plant_data()
        
        # Test individual requirements
        logger.info("\n🧪 Starting individual requirement tests...")
        
        # Test 1: Alternative Fuel Optimization
        fuel_results = test_alternative_fuel_optimization(platform, plant_data)
        
        # Test 2: Cement Plant GPT
        gpt_success = test_cement_plant_gpt(platform, plant_data)
        
        # Test 3: Unified Controller
        control_results = test_unified_controller(platform, plant_data)
        
        # Test 4: Utility Optimization
        utility_results = test_utility_optimization(platform, plant_data)
        
        # Test 5: Anomaly Detection
        anomaly_results = test_anomaly_detection(platform, plant_data)
        
        # Test comprehensive workflow
        workflow_results = test_comprehensive_workflow(platform, plant_data)
        
        # Test compliance validation
        compliance_results = test_platform_compliance(platform)
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("🎉 JK CEMENT DIGITAL TWIN PLATFORM - TEST SUMMARY")
        logger.info("="*80)
        
        test_results = {
            'Alternative Fuel Optimization': fuel_results is not None,
            'Cement Plant GPT': gpt_success,
            'Unified Controller': control_results is not None,
            'Utility Optimization': utility_results is not None,
            'Anomaly Detection': anomaly_results is not None,
            'Comprehensive Workflow': workflow_results is not None,
            'Compliance Validation': compliance_results is not None
        }
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {status} {test_name}")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED! JK Cement Digital Twin Platform is fully operational!")
            logger.info("📈 Ready for JK Cement 6-month POC implementation!")
        else:
            logger.warning(f"\n⚠️ {total_tests - passed_tests} tests failed. Please review and fix issues.")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Main test execution failed: {e}")
        return None

if __name__ == "__main__":
    main()
