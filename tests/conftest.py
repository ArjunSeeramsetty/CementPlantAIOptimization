import pytest
from unittest.mock import MagicMock
from typing import Dict, Any

@pytest.fixture
def platform():
    """Mocks the JKCementDigitalTwinPlatform object."""
    mock_platform = MagicMock()
    # Add any necessary mock methods or attributes here
    mock_platform.optimize_alternative_fuel.return_value = {
        'tsr_achieved': 0.5,
        'tsr_target': 0.6,
        'tsr_improvement_pct': 10.0,
        'quality_penalty': 0.1
    }
    mock_platform.get_fuel_recommendations.return_value = {
        'status': 'ok'
    }
    mock_platform.ask_plant_gpt.return_value = "Mocked GPT response"
    mock_platform.get_plant_status_gpt.return_value = "Mocked status response"
    mock_platform.get_quality_analysis_gpt.return_value = "Mocked quality analysis"
    mock_platform.get_energy_optimization_gpt.return_value = "Mocked energy optimization"
    mock_platform.compute_unified_setpoints.return_value = {
        'kiln_setpoints': {},
        'preheater_setpoints': {'stage_temperatures': [], 'stage_efficiencies': []},
        'cooler_setpoints': {},
        'control_analysis': {}
    }
    mock_platform.get_control_performance.return_value = {
        'control_performance': {
            'avg_temperature_deviation_c': 1.0,
            'avg_thermal_efficiency': 0.8,
            'control_stability': 'stable'
        }
    }
    mock_platform.optimize_all_utilities.return_value = {
        'air_optimization': {'pressure_analysis': {'system_status': 'ok'}, 'leak_analysis': {'leak_status': 'ok'}, 'expected_savings': {'power_reduction_kw': 10, 'cost_savings_usd_year': 1000}},
        'water_optimization': {'usage_analysis': {'usage_status': 'ok'}, 'expected_savings': {'water_reduction_m3_h': 1, 'cost_savings_usd_year': 100}},
        'material_handling_optimization': {'efficiency_analysis': {'efficiency_status': 'ok'}, 'expected_savings': {'power_reduction_kw': 5, 'cost_savings_usd_year': 500}},
        'total_savings': {'total_power_savings_kw': 15, 'total_cost_savings_usd_year': 1600, 'average_efficiency_gain_pct': 5.0, 'roi_period_years': 2.0}
    }
    mock_platform.detect_plant_anomalies.return_value = {
        'anomalies': [],
        'equipment_status': {},
        'summary': {'plant_health_percentage': 100.0, 'active_alerts_count': 0, 'critical_equipment': 'None', 'severity_distribution': {}}
    }
    mock_platform.get_equipment_health_report.return_value = {
        'report_timestamp': 'now',
        'equipment_summary': []
    }
    mock_platform.run_jk_cement_optimization_workflow.return_value = {
        'requirements_met': [],
        'expected_benefits': {'fuel_cost_savings_usd_year': 10000, 'utility_cost_savings_usd_year': 1600, 'total_cost_savings_usd_year': 11600, 'energy_reduction_pct': 5.0, 'tsr_achievement_pct': 10.0, 'roi_period_years': 1.0, 'carbon_footprint_reduction_tco2_year': 1000},
        'recommendations': []
    }
    mock_platform.validate_jk_cement_compliance.return_value = {
        'overall_compliance': 'ok',
        'requirements_status': {}
    }
    mock_platform.get_platform_status.return_value = {
        'platform_status': {'initialized': True, 'components_loaded': 10, 'last_updated': 'now'},
        'optimization_history_count': 10
    }
    mock_platform.process_plant_data.return_value = {
        'overall_plant_status': {
            'overall_performance_score': 0.95,
            'status': 'ok'
        },
        'recommendations': []
    }
    return mock_platform

@pytest.fixture
def plant_data():
    """Provides sample plant data."""
    return create_sample_plant_data()

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
