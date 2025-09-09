"""
Plant Control System with PI Controllers and Time Delays
Implements realistic process control with feedback loops and deadtime.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class PIController:
    """
    Proportional-Integral-Derivative controller with deadtime compensation.
    
    Features:
    - Realistic deadtime modeling
    - Anti-windup protection
    - Setpoint tracking
    - Output limits
    """
    
    def __init__(self, 
                 kp: float, 
                 ki: float, 
                 kd: float, 
                 setpoint: float, 
                 deadtime_minutes: float,
                 output_min: float = -100.0,
                 output_max: float = 100.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint
        self.deadtime = deadtime_minutes * 60  # Convert to seconds
        self.output_min = output_min
        self.output_max = output_max
        
        # Controller state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        
        # History for deadtime compensation
        self.measurement_history = []
        self.output_history = []
        
        # Anti-windup
        self.integral_windup_limit = 50.0
        
        print(f"🎛️ PI Controller initialized: SP={setpoint}, Deadtime={deadtime_minutes}min")
    
    def update(self, measurement: float, current_time: float) -> float:
        """
        Update controller output based on current measurement.
        
        Args:
            measurement: Current process measurement
            current_time: Current simulation time (seconds)
            
        Returns:
            Controller output
        """
        # Store current measurement with timestamp
        self.measurement_history.append((current_time, measurement))
        
        # Find the measurement from 'deadtime' seconds ago
        delayed_measurement = measurement
        for t, val in self.measurement_history:
            if current_time - t >= self.deadtime:
                delayed_measurement = val
                break
        
        # Calculate error based on delayed measurement
        error = self.setpoint - delayed_measurement
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error
        if abs(self.integral) > self.integral_windup_limit:
            self.integral = np.sign(self.integral) * self.integral_windup_limit
        
        integral_term = self.ki * self.integral
        
        # Derivative term
        derivative = error - self.previous_error
        derivative_term = self.kd * derivative
        
        # Calculate output
        output = proportional + integral_term + derivative_term
        
        # Apply output limits
        output = max(self.output_min, min(self.output_max, output))
        
        # Store for next iteration
        self.previous_error = error
        self.previous_output = output
        
        # Clean up old history to save memory
        cutoff_time = current_time - self.deadtime * 2
        self.measurement_history = [(t, v) for t, v in self.measurement_history if t > cutoff_time]
        self.output_history = [(t, v) for t, v in self.output_history if t > cutoff_time]
        
        return output
    
    def set_setpoint(self, new_setpoint: float):
        """Update controller setpoint."""
        self.setpoint = new_setpoint
        print(f"🎯 Setpoint updated to: {new_setpoint}")
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        self.measurement_history = []
        self.output_history = []


class CementPlantController:
    """
    Comprehensive cement plant control system with multiple control loops.
    
    Features:
    - Free lime control (primary control loop)
    - Burning zone temperature control
    - Draft pressure control
    - Feed rate control
    - Multi-variable control coordination
    """
    
    def __init__(self):
        # Initialize control loops with realistic parameters
        self.control_loops = {
            'free_lime_control': PIController(
                kp=0.1, ki=0.01, kd=0.02, 
                setpoint=1.2, deadtime_minutes=25,
                output_min=-50.0, output_max=50.0
            ),
            'bzt_control': PIController(
                kp=0.5, ki=0.05, kd=0.1, 
                setpoint=1450, deadtime_minutes=12,
                output_min=-100.0, output_max=100.0
            ),
            'draft_control': PIController(
                kp=0.2, ki=0.02, kd=0.01, 
                setpoint=-2.0, deadtime_minutes=1,
                output_min=-50.0, output_max=50.0
            ),
            'feed_rate_control': PIController(
                kp=0.3, ki=0.03, kd=0.05, 
                setpoint=200.0, deadtime_minutes=5,
                output_min=-20.0, output_max=20.0
            )
        }
        
        # Control loop priorities and interactions
        self.control_priorities = {
            'free_lime_control': 1,  # Highest priority
            'bzt_control': 2,
            'draft_control': 3,
            'feed_rate_control': 4   # Lowest priority
        }
        
        # Control loop interactions (how one affects another)
        self.control_interactions = {
            'free_lime_control': {
                'fuel_rate': 1.0,      # Direct effect on fuel rate
                'kiln_speed': 0.3,     # Indirect effect on kiln speed
                'primary_air': 0.2     # Indirect effect on primary air
            },
            'bzt_control': {
                'primary_air': 1.0,    # Direct effect on primary air
                'fuel_rate': 0.5,      # Indirect effect on fuel rate
                'kiln_speed': 0.1      # Indirect effect on kiln speed
            },
            'draft_control': {
                'id_fan_speed': 1.0,   # Direct effect on ID fan
                'primary_air': 0.3,    # Indirect effect on primary air
                'fuel_rate': 0.1       # Indirect effect on fuel rate
            },
            'feed_rate_control': {
                'raw_meal_feed': 1.0,  # Direct effect on raw meal feed
                'kiln_speed': 0.2,     # Indirect effect on kiln speed
                'fuel_rate': 0.1       # Indirect effect on fuel rate
            }
        }
        
        # Current control actions
        self.current_actions = {
            'fuel_rate_change': 0.0,
            'primary_air_change': 0.0,
            'id_fan_speed_change': 0.0,
            'raw_meal_feed_change': 0.0,
            'kiln_speed_change': 0.0
        }
        
        print("🏭 Cement Plant Controller initialized")
        print("🎛️ Control loops: Free Lime, BZT, Draft, Feed Rate")
        print("⏱️ Time delays and interactions modeled")
    
    def get_control_actions(self, measurements: Dict[str, float], current_time: float) -> Dict[str, float]:
        """
        Calculate control actions based on current plant measurements.
        
        Args:
            measurements: Dict with current measurements
                {'free_lime': 1.4, 'bzt': 1460, 'draft': -2.2, 'feed_rate': 195}
            current_time: Current simulation time (seconds)
            
        Returns:
            Dict with control actions
        """
        # Calculate individual controller outputs
        controller_outputs = {}
        
        for loop_name, controller in self.control_loops.items():
            measurement_key = self._get_measurement_key(loop_name)
            if measurement_key in measurements:
                output = controller.update(measurements[measurement_key], current_time)
                controller_outputs[loop_name] = output
        
        # Calculate control actions considering interactions
        self._calculate_control_actions(controller_outputs)
        
        return self.current_actions.copy()
    
    def _get_measurement_key(self, loop_name: str) -> str:
        """Get measurement key for control loop."""
        mapping = {
            'free_lime_control': 'free_lime',
            'bzt_control': 'bzt',
            'draft_control': 'draft',
            'feed_rate_control': 'feed_rate'
        }
        return mapping.get(loop_name, 'unknown')
    
    def _calculate_control_actions(self, controller_outputs: Dict[str, float]):
        """Calculate final control actions considering loop interactions."""
        
        # Initialize actions
        actions = {
            'fuel_rate_change': 0.0,
            'primary_air_change': 0.0,
            'id_fan_speed_change': 0.0,
            'raw_meal_feed_change': 0.0,
            'kiln_speed_change': 0.0
        }
        
        # Process controllers in priority order
        sorted_loops = sorted(self.control_priorities.items(), key=lambda x: x[1])
        
        for loop_name, priority in sorted_loops:
            if loop_name in controller_outputs:
                output = controller_outputs[loop_name]
                interactions = self.control_interactions.get(loop_name, {})
                
                # Apply control actions based on interactions
                for action_key, interaction_factor in interactions.items():
                    if action_key in actions:
                        actions[action_key] += output * interaction_factor
        
        # Apply limits and smoothing
        self._apply_action_limits(actions)
        
        # Update current actions
        self.current_actions.update(actions)
    
    def _apply_action_limits(self, actions: Dict[str, float]):
        """Apply realistic limits to control actions."""
        
        # Maximum change rates (per control cycle)
        max_changes = {
            'fuel_rate_change': 5.0,      # t/h per cycle
            'primary_air_change': 10.0,   # % per cycle
            'id_fan_speed_change': 20.0,  # rpm per cycle
            'raw_meal_feed_change': 10.0, # t/h per cycle
            'kiln_speed_change': 0.5      # rpm per cycle
        }
        
        for action_key, max_change in max_changes.items():
            if action_key in actions:
                actions[action_key] = max(-max_change, min(max_change, actions[action_key]))
    
    def set_control_setpoints(self, setpoints: Dict[str, float]):
        """Update control setpoints."""
        for loop_name, setpoint in setpoints.items():
            if loop_name in self.control_loops:
                self.control_loops[loop_name].set_setpoint(setpoint)
    
    def get_control_status(self) -> Dict[str, Any]:
        """Get current control system status."""
        status = {}
        
        for loop_name, controller in self.control_loops.items():
            status[loop_name] = {
                'setpoint': controller.setpoint,
                'integral': controller.integral,
                'previous_error': controller.previous_error,
                'deadtime_minutes': controller.deadtime / 60
            }
        
        status['current_actions'] = self.current_actions.copy()
        
        return status
    
    def reset_controllers(self):
        """Reset all controllers."""
        for controller in self.control_loops.values():
            controller.reset()
        
        self.current_actions = {
            'fuel_rate_change': 0.0,
            'primary_air_change': 0.0,
            'id_fan_speed_change': 0.0,
            'raw_meal_feed_change': 0.0,
            'kiln_speed_change': 0.0
        }
        
        print("🔄 All controllers reset")


class ProcessControlSimulator:
    """
    Simulate process control with realistic dynamics and time delays.
    
    Features:
    - Process dynamics modeling
    - Time delay simulation
    - Disturbance handling
    - Control performance monitoring
    """
    
    def __init__(self, plant_controller: CementPlantController):
        self.controller = plant_controller
        self.process_state = {
            'free_lime': 1.2,
            'bzt': 1450,
            'draft': -2.0,
            'feed_rate': 200.0,
            'fuel_rate': 20.0,
            'primary_air': 50.0,
            'id_fan_speed': 1000.0,
            'kiln_speed': 3.5
        }
        
        # Process dynamics parameters
        self.process_gains = {
            'free_lime': {'fuel_rate': -0.02, 'kiln_temp': -0.001},
            'bzt': {'fuel_rate': 0.5, 'primary_air': 0.3},
            'draft': {'id_fan_speed': -0.01, 'primary_air': 0.05},
            'feed_rate': {'raw_meal_feed': 1.0}
        }
        
        # Time constants (minutes)
        self.time_constants = {
            'free_lime': 25.0,
            'bzt': 12.0,
            'draft': 1.0,
            'feed_rate': 5.0
        }
        
        self.simulation_time = 0.0
        
        print("🔄 Process Control Simulator initialized")
        print("⏱️ Process dynamics and time delays modeled")
    
    def simulate_control_cycle(self, 
                              disturbances: Optional[Dict[str, float]] = None,
                              dt_minutes: float = 1.0) -> Dict[str, Any]:
        """
        Simulate one control cycle with process dynamics.
        
        Args:
            disturbances: Optional disturbances to apply
            dt_minutes: Time step in minutes
            
        Returns:
            Dict with simulation results
        """
        dt_seconds = dt_minutes * 60
        self.simulation_time += dt_seconds
        
        # Apply disturbances if provided
        if disturbances:
            self._apply_disturbances(disturbances)
        
        # Get current measurements
        measurements = {
            'free_lime': self.process_state['free_lime'],
            'bzt': self.process_state['bzt'],
            'draft': self.process_state['draft'],
            'feed_rate': self.process_state['feed_rate']
        }
        
        # Calculate control actions
        control_actions = self.controller.get_control_actions(measurements, self.simulation_time)
        
        # Update process state based on control actions
        self._update_process_state(control_actions, dt_minutes)
        
        return {
            'time_minutes': self.simulation_time / 60,
            'measurements': measurements.copy(),
            'control_actions': control_actions.copy(),
            'process_state': self.process_state.copy(),
            'control_status': self.controller.get_control_status()
        }
    
    def _apply_disturbances(self, disturbances: Dict[str, float]):
        """Apply disturbances to process state."""
        for variable, disturbance in disturbances.items():
            if variable in self.process_state:
                self.process_state[variable] += disturbance
    
    def _update_process_state(self, control_actions: Dict[str, float], dt_minutes: float):
        """Update process state based on control actions and dynamics."""
        
        # Update fuel rate
        fuel_change = control_actions.get('fuel_rate_change', 0.0)
        self.process_state['fuel_rate'] += fuel_change
        
        # Update primary air
        air_change = control_actions.get('primary_air_change', 0.0)
        self.process_state['primary_air'] += air_change
        
        # Update ID fan speed
        fan_change = control_actions.get('id_fan_speed_change', 0.0)
        self.process_state['id_fan_speed'] += fan_change
        
        # Update raw meal feed
        feed_change = control_actions.get('raw_meal_feed_change', 0.0)
        self.process_state['feed_rate'] += feed_change
        
        # Update kiln speed
        speed_change = control_actions.get('kiln_speed_change', 0.0)
        self.process_state['kiln_speed'] += speed_change
        
        # Update process variables based on dynamics
        self._update_process_variables(dt_minutes)
    
    def _update_process_variables(self, dt_minutes: float):
        """Update process variables based on first-order dynamics."""
        
        # Free lime dynamics
        free_lime_target = 1.2 - (self.process_state['fuel_rate'] - 20) * 0.02
        tau = self.time_constants['free_lime']
        self.process_state['free_lime'] += (free_lime_target - self.process_state['free_lime']) * dt_minutes / tau
        
        # BZT dynamics
        bzt_target = 1450 + (self.process_state['fuel_rate'] - 20) * 0.5
        tau = self.time_constants['bzt']
        self.process_state['bzt'] += (bzt_target - self.process_state['bzt']) * dt_minutes / tau
        
        # Draft dynamics
        draft_target = -2.0 - (self.process_state['id_fan_speed'] - 1000) * 0.01
        tau = self.time_constants['draft']
        self.process_state['draft'] += (draft_target - self.process_state['draft']) * dt_minutes / tau


def create_plant_control_system() -> CementPlantController:
    """Factory function to create a plant control system."""
    return CementPlantController()


def create_process_control_simulator(plant_controller: CementPlantController) -> ProcessControlSimulator:
    """Factory function to create a process control simulator."""
    return ProcessControlSimulator(plant_controller)
