"""
Unified Kiln-Cooler Controller for JK Cement Requirements
Implements integrated control of kiln, preheater, and cooler operations
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ProcessSetpoints:
    """Process setpoints for unified control"""
    kiln_speed_rpm: float
    kiln_fuel_rate_tph: float
    kiln_feed_rate_tph: float
    excess_air_pct: float
    burning_zone_temp_c: float
    cooler_speed_rpm: float
    cooler_air_flow_nm3_h: float
    preheater_stage_temps: List[float]
    calciner_temp_c: float

@dataclass
class ProcessConstraints:
    """Process constraints for safe operation"""
    max_kiln_temp_c: float = 1600.0
    min_kiln_temp_c: float = 1200.0
    max_kiln_speed_rpm: float = 4.5
    min_kiln_speed_rpm: float = 1.0
    max_fuel_rate_tph: float = 25.0
    min_fuel_rate_tph: float = 5.0
    max_cooler_temp_c: float = 200.0
    min_cooler_temp_c: float = 50.0

class AdvancedKilnModel:
    """Advanced kiln dynamics model for control optimization"""
    
    def __init__(self):
        self.thermal_time_constant = 0.5  # hours
        self.mass_time_constant = 0.3  # hours
        self.temperature_gain = 15.0  # °C per tph fuel
        self.speed_gain = 0.8  # °C per rpm
        
    def simulate_kiln_dynamics(self, 
                              feed_rate: float,
                              fuel_rate: float,
                              kiln_speed: float,
                              raw_meal_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate kiln dynamics and return operating conditions"""
        
        # Calculate burnability index
        burnability_index = self._calculate_burnability_index(
            raw_meal_properties, fuel_rate, kiln_speed
        )
        
        # Calculate thermal efficiency
        thermal_efficiency = self._calculate_thermal_efficiency(
            fuel_rate, feed_rate, kiln_speed
        )
        
        # Calculate residence time
        residence_time = self._calculate_residence_time(kiln_speed, feed_rate)
        
        # Calculate burning zone temperature
        burning_zone_temp = self._calculate_burning_zone_temp(
            fuel_rate, kiln_speed, feed_rate, raw_meal_properties
        )
        
        return {
            'burnability_analysis': {
                'burnability_index': burnability_index,
                'thermal_efficiency': thermal_efficiency,
                'residence_time_minutes': residence_time,
                'burning_zone_temp_c': burning_zone_temp
            },
            'operating_conditions': {
                'kiln_speed_rpm': kiln_speed,
                'fuel_rate_tph': fuel_rate,
                'feed_rate_tph': feed_rate,
                'burning_zone_temp_c': burning_zone_temp
            }
        }
    
    def _calculate_burnability_index(self, 
                                   raw_meal_properties: Dict[str, Any],
                                   fuel_rate: float,
                                   kiln_speed: float) -> float:
        """Calculate burnability index based on raw meal properties"""
        
        # Base burnability from raw meal composition
        lime_saturation_factor = raw_meal_properties.get('lime_saturation_factor', 0.95)
        silica_ratio = raw_meal_properties.get('silica_ratio', 2.5)
        alumina_ratio = raw_meal_properties.get('alumina_ratio', 1.5)
        
        # Calculate base burnability
        base_burnability = 100 * lime_saturation_factor * (1.0 / silica_ratio) * (1.0 / alumina_ratio)
        
        # Adjust for process conditions
        fuel_adjustment = (fuel_rate - 15.0) * 0.5  # Optimal fuel rate ~15 tph
        speed_adjustment = (kiln_speed - 3.0) * 2.0  # Optimal speed ~3 rpm
        
        burnability_index = base_burnability + fuel_adjustment + speed_adjustment
        
        return max(50.0, min(150.0, burnability_index))  # Clamp between 50-150
    
    def _calculate_thermal_efficiency(self, fuel_rate: float, feed_rate: float, kiln_speed: float) -> float:
        """Calculate thermal efficiency"""
        
        # Base efficiency
        base_efficiency = 0.85
        
        # Adjustments based on operating conditions
        fuel_efficiency = 1.0 - abs(fuel_rate - 15.0) * 0.01  # Optimal at 15 tph
        speed_efficiency = 1.0 - abs(kiln_speed - 3.0) * 0.02  # Optimal at 3 rpm
        
        thermal_efficiency = base_efficiency * fuel_efficiency * speed_efficiency
        
        return max(0.7, min(0.95, thermal_efficiency))
    
    def _calculate_residence_time(self, kiln_speed: float, feed_rate: float) -> float:
        """Calculate material residence time in kiln"""
        
        # Typical kiln length: 60m, diameter: 4.5m
        kiln_length = 60.0  # meters
        kiln_diameter = 4.5  # meters
        
        # Calculate residence time based on speed and feed rate
        base_residence_time = kiln_length / (kiln_speed * 0.1)  # Simplified calculation
        
        # Adjust for feed rate (higher feed rate = shorter residence time)
        feed_adjustment = feed_rate / 200.0  # Normalize to typical feed rate
        
        residence_time = base_residence_time * (1.0 - feed_adjustment * 0.1)
        
        return max(15.0, min(45.0, residence_time))  # Clamp between 15-45 minutes
    
    def _calculate_burning_zone_temp(self, 
                                   fuel_rate: float,
                                   kiln_speed: float,
                                   feed_rate: float,
                                   raw_meal_properties: Dict[str, Any]) -> float:
        """Calculate burning zone temperature"""
        
        # Base temperature
        base_temp = 1450.0  # °C
        
        # Fuel rate effect
        fuel_effect = (fuel_rate - 15.0) * self.temperature_gain
        
        # Kiln speed effect
        speed_effect = (kiln_speed - 3.0) * self.speed_gain
        
        # Feed rate effect (higher feed = lower temp)
        feed_effect = -(feed_rate - 200.0) * 0.5
        
        # Raw meal properties effect
        fineness_effect = (raw_meal_properties.get('fineness_blaine', 3000) - 3000) * 0.01
        
        burning_zone_temp = base_temp + fuel_effect + speed_effect + feed_effect + fineness_effect
        
        return max(1200.0, min(1600.0, burning_zone_temp))

class PreheaterTowerModel:
    """Preheater tower model for heat and mass balance calculations"""
    
    def __init__(self):
        self.stage_count = 5
        self.stage_efficiencies = [0.85, 0.80, 0.75, 0.70, 0.65]  # Typical efficiencies
        
    def calculate_heat_and_mass_balance(self,
                                      raw_meal_flow: float,
                                      gas_flow: float,
                                      raw_meal_composition: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate preheater heat and mass balance"""
        
        # Calculate stage temperatures
        stage_temps = self._calculate_stage_temperatures(gas_flow, raw_meal_flow)
        
        # Calculate stage efficiencies
        stage_efficiencies = self._calculate_stage_efficiencies(raw_meal_flow, gas_flow)
        
        # Calculate heat recovery
        heat_recovery = self._calculate_heat_recovery(stage_temps, stage_efficiencies)
        
        return {
            'stage_temperatures': stage_temps,
            'stage_efficiencies': stage_efficiencies,
            'heat_recovery_efficiency': heat_recovery,
            'gas_outlet_temp_c': stage_temps[-1],
            'raw_meal_outlet_temp_c': stage_temps[0]
        }
    
    def _calculate_stage_temperatures(self, gas_flow: float, raw_meal_flow: float) -> List[float]:
        """Calculate temperature profile through preheater stages"""
        
        # Base temperatures (°C)
        base_temps = [900, 800, 700, 600, 500]
        
        # Adjust for gas flow (higher flow = higher temps)
        gas_adjustment = (gas_flow - 200000) * 0.001  # Normalize to typical gas flow
        
        # Adjust for raw meal flow (higher flow = lower temps)
        meal_adjustment = -(raw_meal_flow - 200) * 0.5  # Normalize to typical meal flow
        
        stage_temps = []
        for i, base_temp in enumerate(base_temps):
            temp = base_temp + gas_adjustment + meal_adjustment
            stage_temps.append(max(400.0, min(1000.0, temp)))
        
        return stage_temps
    
    def _calculate_stage_efficiencies(self, raw_meal_flow: float, gas_flow: float) -> List[float]:
        """Calculate stage efficiencies based on operating conditions"""
        
        # Base efficiencies
        base_efficiencies = self.stage_efficiencies
        
        # Adjust for flow ratio
        flow_ratio = raw_meal_flow / gas_flow if gas_flow > 0 else 0.001
        optimal_ratio = 0.001  # Optimal mass flow ratio
        
        ratio_adjustment = (flow_ratio - optimal_ratio) * 100
        
        stage_efficiencies = []
        for i, base_eff in enumerate(base_efficiencies):
            # Higher stages are more sensitive to flow ratio
            sensitivity = (i + 1) * 0.1
            adjusted_eff = base_eff - ratio_adjustment * sensitivity
            stage_efficiencies.append(max(0.5, min(0.95, adjusted_eff)))
        
        return stage_efficiencies
    
    def _calculate_heat_recovery(self, stage_temps: List[float], stage_efficiencies: List[float]) -> float:
        """Calculate overall heat recovery efficiency"""
        
        # Weighted average efficiency
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Higher stages have more weight
        
        weighted_efficiency = sum(eff * weight for eff, weight in zip(stage_efficiencies, weights))
        
        return max(0.6, min(0.9, weighted_efficiency))

class CoolerModel:
    """Cooler model for clinker cooling optimization"""
    
    def __init__(self):
        self.cooling_efficiency = 0.85
        self.air_flow_coefficient = 2.5  # Nm³/kg clinker
        
    def calculate_cooling_performance(self,
                                    clinker_temp_in: float,
                                    air_flow: float,
                                    clinker_flow: float) -> Dict[str, Any]:
        """Calculate cooler cooling performance"""
        
        # Calculate cooling efficiency
        cooling_efficiency = self._calculate_cooling_efficiency(air_flow, clinker_flow)
        
        # Calculate outlet temperature
        outlet_temp = self._calculate_outlet_temp(clinker_temp_in, cooling_efficiency)
        
        # Calculate air preheating
        air_preheat_temp = self._calculate_air_preheat_temp(clinker_temp_in, outlet_temp, cooling_efficiency)
        
        return {
            'cooling_efficiency': cooling_efficiency,
            'clinker_outlet_temp_c': outlet_temp,
            'air_preheat_temp_c': air_preheat_temp,
            'heat_recovery_efficiency': cooling_efficiency * 0.8  # Typical heat recovery
        }
    
    def _calculate_cooling_efficiency(self, air_flow: float, clinker_flow: float) -> float:
        """Calculate cooler cooling efficiency"""
        
        # Optimal air flow ratio
        optimal_ratio = self.air_flow_coefficient
        actual_ratio = air_flow / clinker_flow if clinker_flow > 0 else optimal_ratio
        
        # Efficiency based on air flow ratio
        ratio_efficiency = 1.0 - abs(actual_ratio - optimal_ratio) / optimal_ratio * 0.3
        
        cooling_efficiency = self.cooling_efficiency * ratio_efficiency
        
        return max(0.6, min(0.95, cooling_efficiency))
    
    def _calculate_outlet_temp(self, inlet_temp: float, cooling_efficiency: float) -> float:
        """Calculate clinker outlet temperature"""
        
        ambient_temp = 25.0  # °C
        outlet_temp = inlet_temp - (inlet_temp - ambient_temp) * cooling_efficiency
        
        return max(ambient_temp + 10, min(inlet_temp, outlet_temp))
    
    def _calculate_air_preheat_temp(self, clinker_temp_in: float, clinker_temp_out: float, cooling_efficiency: float) -> float:
        """Calculate air preheating temperature"""
        
        # Heat transfer to air
        heat_transfer = (clinker_temp_in - clinker_temp_out) * cooling_efficiency * 0.7
        
        ambient_temp = 25.0
        air_preheat_temp = ambient_temp + heat_transfer
        
        return max(ambient_temp, min(800.0, air_preheat_temp))

class UnifiedKilnCoolerController:
    """
    Unified controller for kiln, preheater, and cooler operations.
    Implements JK Cement's requirement for integrated process control.
    """

    def __init__(self, config_path: str = "config/plant_config.yml"):
        self.kiln_model = AdvancedKilnModel()
        self.preheater_model = PreheaterTowerModel()
        self.cooler_model = CoolerModel()
        self.constraints = ProcessConstraints()
        
        # Load plant configuration
        self.config = self._load_config(config_path)
        
        # Control parameters
        self.control_gains = {
            'temperature_gain': 0.1,
            'speed_gain': 0.05,
            'fuel_gain': 0.08,
            'air_gain': 0.03
        }
        
        # Setpoint history for trend analysis
        self.setpoint_history = []
        
        logger.info("✅ Unified Kiln-Cooler Controller initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load plant configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def compute_setpoints(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute optimal setpoints for unified kiln-cooler control
        
        Args:
            sensor_data: Current sensor readings and process parameters
            
        Returns:
            Optimized setpoints for all process units
        """
        logger.info("🔄 Computing unified process setpoints...")
        
        # Extract sensor data
        feed_rate = sensor_data.get('feed_rate_tph', 200.0)
        fuel_rate = sensor_data.get('fuel_rate_tph', 15.0)
        kiln_speed = sensor_data.get('kiln_speed_rpm', 3.0)
        burning_zone_temp = sensor_data.get('burning_zone_temp_c', 1450.0)
        cooler_outlet_temp = sensor_data.get('cooler_outlet_temp_c', 100.0)
        
        # Raw meal properties
        raw_meal_properties = {
            'composition': sensor_data.get('raw_meal_composition', {}),
            'fineness': sensor_data.get('raw_meal_fineness_blaine', 3000),
            'alkali_content': sensor_data.get('raw_meal_alkali_content', 0.6),
            'lime_saturation_factor': sensor_data.get('lime_saturation_factor', 0.95),
            'silica_ratio': sensor_data.get('silica_ratio', 2.5),
            'alumina_ratio': sensor_data.get('alumina_ratio', 1.5)
        }
        
        # Gas flow data
        gas_flow = sensor_data.get('gas_flow_nm3_h', 200000.0)
        
        # 1. Optimize kiln parameters
        kiln_results = self.kiln_model.simulate_kiln_dynamics(
            feed_rate=feed_rate,
            fuel_rate=fuel_rate,
            kiln_speed=kiln_speed,
            raw_meal_properties=raw_meal_properties
        )
        
        # 2. Calculate preheater optimization
        preheater_results = self.preheater_model.calculate_heat_and_mass_balance(
            raw_meal_flow=feed_rate,
            gas_flow=gas_flow,
            raw_meal_composition=raw_meal_properties['composition']
        )
        
        # 3. Optimize cooler operation
        cooler_results = self.cooler_model.calculate_cooling_performance(
            clinker_temp_in=burning_zone_temp,
            air_flow=sensor_data.get('cooler_air_flow_nm3_h', 150000.0),
            clinker_flow=feed_rate
        )
        
        # 4. Calculate control adjustments
        control_adjustments = self._calculate_control_adjustments(
            sensor_data, kiln_results, preheater_results, cooler_results
        )
        
        # 5. Generate optimized setpoints
        optimized_setpoints = self._generate_optimized_setpoints(
            sensor_data, control_adjustments
        )
        
        # 6. Validate setpoints against constraints
        validated_setpoints = self._validate_setpoints(optimized_setpoints)
        
        # Store setpoint history
        setpoint_record = {
            'timestamp': datetime.now().isoformat(),
            'setpoints': validated_setpoints,
            'kiln_results': kiln_results,
            'preheater_results': preheater_results,
            'cooler_results': cooler_results
        }
        self.setpoint_history.append(setpoint_record)
        
        # Keep only last 100 records
        if len(self.setpoint_history) > 100:
            self.setpoint_history = self.setpoint_history[-100:]
        
        logger.info("✅ Unified setpoints computed successfully")
        
        return {
            'kiln_setpoints': {
                'kiln_speed_rpm': validated_setpoints['kiln_speed_rpm'],
                'fuel_rate_tph': validated_setpoints['fuel_rate_tph'],
                'feed_rate_tph': validated_setpoints['feed_rate_tph'],
                'excess_air_pct': validated_setpoints['excess_air_pct'],
                'burning_zone_temp_c': validated_setpoints['burning_zone_temp_c']
            },
            'preheater_setpoints': {
                'stage_temperatures': preheater_results['stage_temperatures'],
                'stage_efficiencies': preheater_results['stage_efficiencies'],
                'calciner_temp_c': validated_setpoints['calciner_temp_c']
            },
            'cooler_setpoints': {
                'cooler_speed_rpm': validated_setpoints['cooler_speed_rpm'],
                'air_flow_nm3_h': validated_setpoints['cooler_air_flow_nm3_h'],
                'target_outlet_temp_c': validated_setpoints['cooler_outlet_temp_c']
            },
            'control_analysis': {
                'burnability_index': kiln_results['burnability_analysis']['burnability_index'],
                'thermal_efficiency': kiln_results['burnability_analysis']['thermal_efficiency'],
                'cooling_efficiency': cooler_results['cooling_efficiency'],
                'heat_recovery_efficiency': preheater_results['heat_recovery_efficiency']
            }
        }

    def _calculate_control_adjustments(self, 
                                    sensor_data: Dict[str, Any],
                                    kiln_results: Dict[str, Any],
                                    preheater_results: Dict[str, Any],
                                    cooler_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate control adjustments based on process analysis"""
        
        adjustments = {}
        
        # Kiln temperature control
        target_temp = 1450.0
        current_temp = sensor_data.get('burning_zone_temp_c', 1450.0)
        temp_error = target_temp - current_temp
        
        adjustments['fuel_rate_adjustment'] = temp_error * self.control_gains['fuel_gain']
        adjustments['kiln_speed_adjustment'] = temp_error * self.control_gains['speed_gain']
        
        # Cooler temperature control
        target_cooler_temp = 80.0
        current_cooler_temp = sensor_data.get('cooler_outlet_temp_c', 100.0)
        cooler_temp_error = target_cooler_temp - current_cooler_temp
        
        adjustments['cooler_speed_adjustment'] = cooler_temp_error * self.control_gains['speed_gain']
        adjustments['air_flow_adjustment'] = cooler_temp_error * self.control_gains['air_gain']
        
        # Feed rate optimization based on burnability
        burnability_index = kiln_results['burnability_analysis']['burnability_index']
        if burnability_index < 90:
            adjustments['feed_rate_adjustment'] = -5.0  # Reduce feed rate
        elif burnability_index > 110:
            adjustments['feed_rate_adjustment'] = 5.0   # Increase feed rate
        else:
            adjustments['feed_rate_adjustment'] = 0.0
        
        return adjustments

    def _generate_optimized_setpoints(self, 
                                   sensor_data: Dict[str, Any],
                                   adjustments: Dict[str, float]) -> Dict[str, float]:
        """Generate optimized setpoints based on current values and adjustments"""
        
        # Current values
        current_fuel_rate = sensor_data.get('fuel_rate_tph', 15.0)
        current_kiln_speed = sensor_data.get('kiln_speed_rpm', 3.0)
        current_feed_rate = sensor_data.get('feed_rate_tph', 200.0)
        current_cooler_speed = sensor_data.get('cooler_speed_rpm', 2.0)
        current_air_flow = sensor_data.get('cooler_air_flow_nm3_h', 150000.0)
        
        # Calculate new setpoints
        new_fuel_rate = current_fuel_rate + adjustments['fuel_rate_adjustment']
        new_kiln_speed = current_kiln_speed + adjustments['kiln_speed_adjustment']
        new_feed_rate = current_feed_rate + adjustments['feed_rate_adjustment']
        new_cooler_speed = current_cooler_speed + adjustments['cooler_speed_adjustment']
        new_air_flow = current_air_flow + adjustments['air_flow_adjustment']
        
        # Calculate derived setpoints
        burning_zone_temp = 1450.0 + adjustments['fuel_rate_adjustment'] * 10
        calciner_temp = 900.0 + adjustments['fuel_rate_adjustment'] * 5
        cooler_outlet_temp = 80.0 + adjustments['cooler_speed_adjustment'] * 5
        
        return {
            'kiln_speed_rpm': new_kiln_speed,
            'fuel_rate_tph': new_fuel_rate,
            'feed_rate_tph': new_feed_rate,
            'excess_air_pct': 10.0,  # Optimal excess air
            'burning_zone_temp_c': burning_zone_temp,
            'cooler_speed_rpm': new_cooler_speed,
            'cooler_air_flow_nm3_h': new_air_flow,
            'calciner_temp_c': calciner_temp,
            'cooler_outlet_temp_c': cooler_outlet_temp
        }

    def _validate_setpoints(self, setpoints: Dict[str, float]) -> Dict[str, float]:
        """Validate setpoints against process constraints"""
        
        validated = setpoints.copy()
        
        # Validate kiln constraints
        validated['kiln_speed_rpm'] = max(
            self.constraints.min_kiln_speed_rpm,
            min(self.constraints.max_kiln_speed_rpm, setpoints['kiln_speed_rpm'])
        )
        
        validated['fuel_rate_tph'] = max(
            self.constraints.min_fuel_rate_tph,
            min(self.constraints.max_fuel_rate_tph, setpoints['fuel_rate_tph'])
        )
        
        validated['burning_zone_temp_c'] = max(
            self.constraints.min_kiln_temp_c,
            min(self.constraints.max_kiln_temp_c, setpoints['burning_zone_temp_c'])
        )
        
        # Validate cooler constraints
        validated['cooler_outlet_temp_c'] = max(
            self.constraints.min_cooler_temp_c,
            min(self.constraints.max_cooler_temp_c, setpoints['cooler_outlet_temp_c'])
        )
        
        return validated

    def get_control_performance(self) -> Dict[str, Any]:
        """Get control performance metrics"""
        
        if not self.setpoint_history:
            return {'status': 'No control history available'}
        
        # Calculate performance metrics
        recent_setpoints = self.setpoint_history[-10:]  # Last 10 control cycles
        
        # Temperature control performance
        temp_deviations = []
        for record in recent_setpoints:
            target_temp = record['setpoints']['burning_zone_temp_c']
            actual_temp = record.get('kiln_results', {}).get('burnability_analysis', {}).get('burning_zone_temp_c', target_temp)
            temp_deviations.append(abs(target_temp - actual_temp))
        
        avg_temp_deviation = np.mean(temp_deviations) if temp_deviations else 0
        
        # Efficiency trends
        efficiencies = []
        for record in recent_setpoints:
            efficiency = record.get('kiln_results', {}).get('burnability_analysis', {}).get('thermal_efficiency', 0.85)
            efficiencies.append(efficiency)
        
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.85
        
        return {
            'control_performance': {
                'avg_temperature_deviation_c': avg_temp_deviation,
                'avg_thermal_efficiency': avg_efficiency,
                'control_stability': 'Good' if avg_temp_deviation < 10 else 'Needs Improvement',
                'total_control_cycles': len(self.setpoint_history)
            },
            'recommendations': self._get_control_recommendations(avg_temp_deviation, avg_efficiency)
        }

    def _get_control_recommendations(self, temp_deviation: float, efficiency: float) -> List[str]:
        """Get control recommendations based on performance"""
        
        recommendations = []
        
        if temp_deviation > 15:
            recommendations.append("Increase control sensitivity for temperature control")
        
        if efficiency < 0.8:
            recommendations.append("Optimize fuel-air ratio for better thermal efficiency")
        
        if temp_deviation < 5 and efficiency > 0.85:
            recommendations.append("Control performance is excellent - maintain current settings")
        
        return recommendations

    def export_control_data(self) -> Dict[str, Any]:
        """Export control data for analysis"""
        
        return {
            'setpoint_history': self.setpoint_history[-50:],  # Last 50 records
            'control_performance': self.get_control_performance(),
            'constraints': self.constraints,
            'control_gains': self.control_gains,
            'export_timestamp': datetime.now().isoformat()
        }
