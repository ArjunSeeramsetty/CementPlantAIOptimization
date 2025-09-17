import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Conditional import for process models
try:
    from cement_ai_platform.models.process_models import AdvancedKilnModel, PreheaterTower
    PROCESS_MODELS_AVAILABLE = True
except ImportError:
    PROCESS_MODELS_AVAILABLE = False
    
    # Fallback classes with enhanced functionality
    class AdvancedKilnModel:
        def __init__(self):
            self.name = "Simplified Kiln Model"
        
        def simulate_kiln_dynamics(self, feed_rate=None, fuel_rate=None, kiln_speed=None, raw_meal_properties=None, **kwargs):
            # Simulate realistic kiln dynamics
            burnability_index = 100.0
            if fuel_rate and feed_rate:
                # Higher fuel rate relative to feed improves burnability
                fuel_feed_ratio = fuel_rate / feed_rate if feed_rate > 0 else 0.1
                burnability_index = min(120.0, 80.0 + fuel_feed_ratio * 200)
            
            if kiln_speed:
                # Kiln speed affects residence time and burnability
                burnability_index *= (1.0 + (kiln_speed - 3.0) * 0.1)
            
            return {
                'burnability_analysis': {'burnability_index': max(60.0, min(120.0, burnability_index))},
                'operating_conditions': {
                    'kiln_speed_rpm': kiln_speed or 3.2,
                    'fuel_rate_tph': fuel_rate or 16.0,
                    'feed_rate_tph': feed_rate or 167.0,
                    'excess_air_pct': 10.0,
                    'burning_zone_temp_c': 1450.0
                }
            }
    
    class PreheaterTower:
        def __init__(self):
            self.name = "Simplified Preheater Model"
        
        def calculate_heat_and_mass_balance(self, raw_meal_flow=None, gas_flow=None, raw_meal_composition=None, **kwargs):
            # Simulate realistic preheater performance
            base_efficiency = 0.85
            stage_efficiencies = []
            
            for stage in range(5):
                # Decreasing efficiency from top to bottom
                stage_eff = base_efficiency - stage * 0.05
                stage_efficiencies.append(max(0.6, stage_eff))
            
            return {
                'stage_efficiencies': stage_efficiencies,
                'stage_temperatures': [900, 800, 700, 600, 500],
                'calciner_temp_c': 895.0,
                'heat_recovery_efficiency': 0.75
            }

@dataclass
class ControllerSettings:
    """Controller configuration and limits"""
    kiln_speed_min: float = 2.8
    kiln_speed_max: float = 4.2
    fuel_rate_min: float = 14.0
    fuel_rate_max: float = 20.0
    primary_air_min: float = 40.0
    primary_air_max: float = 60.0
    burning_zone_temp_min: float = 1420.0
    burning_zone_temp_max: float = 1480.0
    cooler_speed_min: float = 6.0
    cooler_speed_max: float = 14.0

@dataclass
class ProcessTargets:
    """Process control targets"""
    free_lime_target: float = 1.0
    thermal_energy_target: float = 690.0
    nox_target: float = 500.0
    cooler_outlet_temp_target: float = 100.0
    secondary_air_temp_target: float = 950.0

class PIDController:
    """Enhanced PID controller with anti-windup and rate limiting"""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_min: float, output_max: float, 
                 rate_limit: float = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.rate_limit = rate_limit
        
        # State variables
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        self.dt = 1.0  # Default sampling time
    
    def update(self, setpoint: float, process_value: float, dt: float = 1.0) -> float:
        """Update PID controller with anti-windup and rate limiting"""
        self.dt = dt
        error = setpoint - process_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = 0
        if dt > 0:
            d_term = self.kd * (error - self.previous_error) / dt
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply limits
        output_limited = np.clip(output, self.output_min, self.output_max)
        
        # Anti-windup: adjust integral term if output is saturated
        if output != output_limited:
            self.integral -= (output - output_limited) / (self.ki + 1e-10)
        
        # Rate limiting
        if self.rate_limit and self.previous_output is not None:
            max_change = self.rate_limit * dt
            output_limited = np.clip(output_limited, 
                                   self.previous_output - max_change,
                                   self.previous_output + max_change)
        
        self.previous_error = error
        self.previous_output = output_limited
        
        return output_limited

class UnifiedKilnCoolerController:
    """
    Unified control system for kiln and cooler coordination.
    Implements advanced control strategies for optimal cement production.
    """
    
    def __init__(self):
        self.settings = ControllerSettings()
        self.targets = ProcessTargets()
        
        # Initialize process models
        self.kiln_model = AdvancedKilnModel()
        self.preheater = PreheaterTower()
        
        # Initialize PID controllers
        self._initialize_controllers()
        
        # State variables
        self.control_history = []
        self.disturbance_compensation = {}
        
    def _initialize_controllers(self):
        """Initialize PID controllers for different control loops"""
        
        # Primary control loops
        self.controllers = {
            # Kiln speed controller for residence time/quality
            'kiln_speed': PIDController(
                kp=0.5, ki=0.1, kd=0.05,
                output_min=self.settings.kiln_speed_min,
                output_max=self.settings.kiln_speed_max,
                rate_limit=0.1  # rpm/min
            ),
            
            # Fuel rate controller for temperature/energy
            'fuel_rate': PIDController(
                kp=2.0, ki=0.3, kd=0.1,
                output_min=self.settings.fuel_rate_min,
                output_max=self.settings.fuel_rate_max,
                rate_limit=0.5  # t/h/min
            ),
            
            # Primary air controller for combustion optimization
            'primary_air': PIDController(
                kp=1.5, ki=0.2, kd=0.08,
                output_min=self.settings.primary_air_min,
                output_max=self.settings.primary_air_max,
                rate_limit=2.0  # %/min
            ),
            
            # Cooler speed controller for clinker temperature
            'cooler_speed': PIDController(
                kp=1.0, ki=0.15, kd=0.05,
                output_min=self.settings.cooler_speed_min,
                output_max=self.settings.cooler_speed_max,
                rate_limit=0.2  # rpm/min
            )
        }
    
    def compute_unified_setpoints(self, sensor_data: Dict, dt: float = 1.0) -> Dict:
        """
        Compute coordinated setpoints for kiln and cooler systems
        
        Args:
            sensor_data: Current sensor readings
            dt: Time step in minutes
            
        Returns:
            Dictionary of optimized setpoints
        """
        
        # Extract key measurements
        current_free_lime = sensor_data.get('free_lime_percent', 1.0)
        burning_zone_temp = sensor_data.get('burning_zone_temp_c', 1450)
        cooler_outlet_temp = sensor_data.get('cooler_outlet_temp_c', 100)
        secondary_air_temp = sensor_data.get('secondary_air_temp_c', 950)
        nox_emissions = sensor_data.get('nox_mg_nm3', 500)
        feed_rate = sensor_data.get('feed_rate_tph', 167)
        
        # 1. Quality-based kiln speed control
        kiln_speed_setpoint = self.controllers['kiln_speed'].update(
            setpoint=self.targets.free_lime_target,
            process_value=current_free_lime,
            dt=dt
        )
        
        # 2. Energy-optimized fuel rate control
        # Calculate required thermal energy based on feed rate and targets
        required_thermal_energy = feed_rate * self.targets.thermal_energy_target / 1000
        fuel_rate_setpoint = self.controllers['fuel_rate'].update(
            setpoint=required_thermal_energy,
            process_value=sensor_data.get('fuel_rate_tph', 16.3),
            dt=dt
        )
        
        # 3. Combustion optimization - primary air control
        # Target O2 based on fuel rate and efficiency
        target_o2 = self._calculate_target_o2(fuel_rate_setpoint, nox_emissions)
        primary_air_setpoint = self.controllers['primary_air'].update(
            setpoint=target_o2,
            process_value=sensor_data.get('o2_percent', 3.0),
            dt=dt
        )
        
        # 4. Cooler coordination
        cooler_speed_setpoint = self.controllers['cooler_speed'].update(
            setpoint=self.targets.cooler_outlet_temp_target,
            process_value=cooler_outlet_temp,
            dt=dt
        )
        
        # 5. Advanced coordination adjustments
        coordination_adjustments = self._calculate_coordination_adjustments(
            sensor_data, kiln_speed_setpoint, fuel_rate_setpoint, 
            primary_air_setpoint, cooler_speed_setpoint
        )
        
        # 6. Apply coordination adjustments
        final_setpoints = {
            'kiln_speed_rpm': kiln_speed_setpoint + coordination_adjustments['kiln_speed'],
            'fuel_rate_tph': fuel_rate_setpoint + coordination_adjustments['fuel_rate'],
            'primary_air_percent': primary_air_setpoint + coordination_adjustments['primary_air'],
            'cooler_speed_rpm': cooler_speed_setpoint + coordination_adjustments['cooler_speed'],
        }
        
        # 7. Safety and operational limits check
        final_setpoints = self._apply_safety_limits(final_setpoints, sensor_data)
        
        # 8. Calculate predicted performance
        performance_prediction = self._predict_performance(final_setpoints, sensor_data)
        
        # Store control action in history
        control_action = {
            'timestamp': pd.Timestamp.now(),
            'setpoints': final_setpoints,
            'sensor_data': sensor_data,
            'performance_prediction': performance_prediction
        }
        self.control_history.append(control_action)
        
        return {
            'setpoints': final_setpoints,
            'performance_prediction': performance_prediction,
            'coordination_factors': coordination_adjustments,
            'control_health': self._assess_control_health(sensor_data, final_setpoints)
        }
    
    def _calculate_target_o2(self, fuel_rate: float, current_nox: float) -> float:
        """Calculate optimal O2 setpoint for combustion efficiency and NOx control"""
        
        # Base O2 target for efficient combustion
        base_o2 = 3.0
        
        # Adjust for NOx emissions
        if current_nox > self.targets.nox_target:
            # Reduce O2 to lower NOx (staged combustion effect)
            nox_adjustment = -0.5 * ((current_nox - self.targets.nox_target) / self.targets.nox_target)
        else:
            nox_adjustment = 0
        
        # Adjust for fuel rate (higher fuel rate may need more O2)
        fuel_adjustment = 0.1 * (fuel_rate - 16.3) / 16.3
        
        target_o2 = base_o2 + nox_adjustment + fuel_adjustment
        
        # Keep within practical limits
        return np.clip(target_o2, 2.0, 4.5)
    
    def _calculate_coordination_adjustments(self, sensor_data: Dict, 
                                          kiln_speed: float, fuel_rate: float,
                                          primary_air: float, cooler_speed: float) -> Dict:
        """Calculate coordination adjustments between kiln and cooler"""
        
        adjustments = {
            'kiln_speed': 0.0,
            'fuel_rate': 0.0,
            'primary_air': 0.0,
            'cooler_speed': 0.0
        }
        
        # 1. Thermal coupling: if kiln runs hot, increase cooler speed
        burning_zone_temp = sensor_data.get('burning_zone_temp_c', 1450)
        if burning_zone_temp > 1470:
            adjustments['cooler_speed'] += 0.5
            adjustments['fuel_rate'] -= 0.2
        elif burning_zone_temp < 1430:
            adjustments['cooler_speed'] -= 0.3
            adjustments['fuel_rate'] += 0.1
        
        # 2. Secondary air temperature coupling
        sec_air_temp = sensor_data.get('secondary_air_temp_c', 950)
        if sec_air_temp < 900:  # Low secondary air temp affects combustion
            adjustments['cooler_speed'] += 0.3
            adjustments['primary_air'] += 1.0
        
        # 3. Residence time coordination
        # If increasing kiln speed (less residence time), may need more fuel
        if kiln_speed > 3.5:
            adjustments['fuel_rate'] += 0.1 * (kiln_speed - 3.5)
        
        # 4. Load balancing
        feed_rate = sensor_data.get('feed_rate_tph', 167)
        if feed_rate > 180:  # High load
            adjustments['cooler_speed'] += 0.4
            adjustments['primary_air'] += 0.5
        
        return adjustments
    
    def _apply_safety_limits(self, setpoints: Dict, sensor_data: Dict) -> Dict:
        """Apply safety limits and operational constraints"""
        
        limited_setpoints = setpoints.copy()
        
        # Temperature safety limits
        burning_zone_temp = sensor_data.get('burning_zone_temp_c', 1450)
        if burning_zone_temp > 1475:
            # Emergency fuel reduction
            limited_setpoints['fuel_rate_tph'] = min(
                limited_setpoints['fuel_rate_tph'], 
                sensor_data.get('fuel_rate_tph', 16.3) - 0.5
            )
        
        # Kiln torque protection
        kiln_torque = sensor_data.get('kiln_torque_percent', 70)
        if kiln_torque > 85:
            # Reduce kiln speed to protect drive
            limited_setpoints['kiln_speed_rpm'] = min(
                limited_setpoints['kiln_speed_rpm'],
                sensor_data.get('kiln_speed_rpm', 3.2) - 0.1
            )
        
        # Cooler temperature protection
        cooler_outlet_temp = sensor_data.get('cooler_outlet_temp_c', 100)
        if cooler_outlet_temp > 150:
            # Increase cooler speed for better cooling
            limited_setpoints['cooler_speed_rpm'] = max(
                limited_setpoints['cooler_speed_rpm'],
                sensor_data.get('cooler_speed_rpm', 10) + 1.0
            )
        
        # Final bounds checking
        limited_setpoints['kiln_speed_rpm'] = np.clip(
            limited_setpoints['kiln_speed_rpm'],
            self.settings.kiln_speed_min,
            self.settings.kiln_speed_max
        )
        
        limited_setpoints['fuel_rate_tph'] = np.clip(
            limited_setpoints['fuel_rate_tph'],
            self.settings.fuel_rate_min,
            self.settings.fuel_rate_max
        )
        
        limited_setpoints['primary_air_percent'] = np.clip(
            limited_setpoints['primary_air_percent'],
            self.settings.primary_air_min,
            self.settings.primary_air_max
        )
        
        limited_setpoints['cooler_speed_rpm'] = np.clip(
            limited_setpoints['cooler_speed_rpm'],
            self.settings.cooler_speed_min,
            self.settings.cooler_speed_max
        )
        
        return limited_setpoints
    
    def _predict_performance(self, setpoints: Dict, current_data: Dict) -> Dict:
        """Predict performance with new setpoints"""
        
        # Use simplified models for real-time prediction
        predicted_free_lime = self._predict_free_lime(setpoints, current_data)
        predicted_energy = self._predict_thermal_energy(setpoints, current_data)
        predicted_nox = self._predict_nox(setpoints, current_data)
        
        return {
            'predicted_free_lime': predicted_free_lime,
            'predicted_thermal_energy': predicted_energy,
            'predicted_nox': predicted_nox,
            'prediction_confidence': 0.85,  # Could be improved with ML models
            'time_to_stabilize': 15  # minutes
        }
    
    def _predict_free_lime(self, setpoints: Dict, current_data: Dict) -> float:
        """Simplified free lime prediction model"""
        base_free_lime = current_data.get('free_lime_percent', 1.0)
        
        # Fuel rate effect (more fuel -> lower free lime)
        fuel_effect = -0.3 * (setpoints['fuel_rate_tph'] - 16.3)
        
        # Kiln speed effect (higher speed -> higher free lime)
        speed_effect = 0.4 * (setpoints['kiln_speed_rpm'] - 3.2)
        
        predicted = base_free_lime + fuel_effect + speed_effect
        return np.clip(predicted, 0.1, 3.0)
    
    def _predict_thermal_energy(self, setpoints: Dict, current_data: Dict) -> float:
        """Predict thermal energy consumption"""
        feed_rate = current_data.get('feed_rate_tph', 167)
        fuel_rate = setpoints['fuel_rate_tph']
        
        # Simple energy calculation
        thermal_energy = (fuel_rate * 6500) / feed_rate  # kcal/kg
        return thermal_energy
    
    def _predict_nox(self, setpoints: Dict, current_data: Dict) -> float:
        """Predict NOx emissions"""
        base_nox = current_data.get('nox_mg_nm3', 500)
        
        # Primary air effect
        air_effect = 50 * (setpoints['primary_air_percent'] - 50) / 50
        
        # Fuel rate effect
        fuel_effect = 30 * (setpoints['fuel_rate_tph'] - 16.3) / 16.3
        
        predicted = base_nox + air_effect + fuel_effect
        return np.clip(predicted, 200, 1000)
    
    def _assess_control_health(self, sensor_data: Dict, setpoints: Dict) -> Dict:
        """Assess control system health and performance"""
        
        # Check for controller saturation
        saturation_count = 0
        for controller_name, controller in self.controllers.items():
            if (abs(controller.previous_output - controller.output_max) < 0.1 or 
                abs(controller.previous_output - controller.output_min) < 0.1):
                saturation_count += 1
        
        # Check for excessive control action
        if len(self.control_history) > 1:
            recent_action = self.control_history[-1]['setpoints']
            previous_action = self.control_history[-2]['setpoints']
            
            control_changes = sum(
                abs(recent_action[key] - previous_action[key])
                for key in recent_action.keys()
            )
        else:
            control_changes = 0
        
        health_score = 1.0 - (saturation_count * 0.2) - min(control_changes * 0.1, 0.3)
        
        return {
            'health_score': max(health_score, 0.1),
            'saturated_controllers': saturation_count,
            'control_activity': control_changes,
            'status': 'Good' if health_score > 0.8 else 'Fair' if health_score > 0.6 else 'Poor'
        }
    
    def tune_controllers(self, performance_data: pd.DataFrame):
        """Auto-tune PID controllers based on historical performance"""
        # This could implement Ziegler-Nichols or other tuning methods
        # For now, implement a simple adaptive tuning approach
        
        for controller_name in self.controllers:
            if len(self.control_history) > 50:
                # Analyze recent performance
                recent_errors = []
                for action in self.control_history[-50:]:
                    # Extract relevant error metrics
                    if controller_name == 'kiln_speed':
                        error = abs(action['sensor_data'].get('free_lime_percent', 1.0) - 
                                  self.targets.free_lime_target)
                        recent_errors.append(error)
                
                # Simple adaptive adjustment
                avg_error = np.mean(recent_errors)
                if avg_error > 0.3:  # Increase aggressiveness
                    self.controllers[controller_name].kp *= 1.05
                elif avg_error < 0.1:  # Decrease aggressiveness
                    self.controllers[controller_name].kp *= 0.98
