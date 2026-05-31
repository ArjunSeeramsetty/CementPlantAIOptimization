"""Tennessee Eastman Process (TEP) Standalone Simulator.

Simulates a multivariate chemical reactor system with dynamic state equations,
mass/energy balances, and user-injectable process drifts (anomalies).
"""

import math
import random
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta


class TennesseeEastmanSimulator:
    """
    High-fidelity dynamic simulator for the Tennessee Eastman Process.
    Models reactor pressure, temperature, separator levels, stripper dynamics,
    and includes dynamic drift (anomaly) injection.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the simulator to stable steady-state conditions."""
        # Process Measurements (Stable nominal values)
        self.state: Dict[str, float] = {
            "reactor_temp": 120.4,          # °C (Nominal 120.0)
            "reactor_pressure": 2705.0,     # kPa (Nominal 2700.0)
            "reactor_level": 75.0,          # % (Nominal 75.0)
            "reactor_cooling_water_temp": 35.2, # °C
            "separator_temp": 80.1,         # °C
            "separator_level": 50.0,        # %
            "separator_pressure": 2630.0,   # kPa
            "stripper_temp": 65.8,          # °C
            "stripper_level": 50.0,         # %
            "stripper_pressure": 2620.0,    # kPa
            "purge_rate": 8.5,              # kscmh (kilo standard cubic meters/hr)
            "product_flow_rate": 22.8,      # t/h
            "feed_a_rate": 10.2,            # t/h
            "feed_d_rate": 15.4,            # t/h
            "feed_e_rate": 21.1,            # t/h
            "recycle_flow": 102.5,          # t/h
            "compressor_work": 95.4,        # kW
            "reactor_cooling_flow": 82.3,    # m³/h
            "condenser_cooling_flow": 60.1,  # m³/h
        }
        
        # Manipulated Variables (Setpoints controlled by operator)
        self.controls: Dict[str, float] = {
            "reactor_cooling_valve": 45.0,  # % open
            "condenser_cooling_valve": 38.0,# % open
            "feed_a_valve": 32.0,           # % open
            "feed_d_valve": 48.0,           # % open
            "feed_e_valve": 55.0,           # % open
            "purge_valve": 15.0,            # % open
            "stripper_steam_valve": 22.0,   # % open
            "compressor_speed": 85.0        # %
        }
        
        # Active drifts
        self.active_drifts: Dict[str, Dict[str, Any]] = {}
        self.time_step_counter = 0
        self.safety_tripped = False
        self.trip_reason = ""

    def inject_drift(self, drift_type: str, severity: float = 1.0) -> None:
        """
        Configure an active process drift (anomaly).
        
        Supported Types:
        - 'feed_a_composition': Slow drift in Feed A input concentration (mass balance).
        - 'reactor_valve_stiction': Reactor cooling water valve gets stuck or oscillates.
        - 'feed_d_temp_step': Sudden temperature step change in incoming feed D.
        - 'catalyst_decay': Slow loss of reaction rate (reactor yield decays).
        """
        self.active_drifts[drift_type] = {
            "severity": severity,
            "start_step": self.time_step_counter
        }

    def clear_drifts(self) -> None:
        """Clear all active anomalies and reset parameters."""
        self.active_drifts.clear()
        self.safety_tripped = False
        self.trip_reason = ""

    def step(self) -> Dict[str, float]:
        """
        Run one time step (representing 30 seconds of plant operations).
        Updates state variables based on thermodynamics and controls.
        """
        # Evaluate safety limits first in case of manual override or pre-existing breach
        self._check_safety_limits()

        if self.safety_tripped:
            # Plant is in shutdown state, values drop to safe ambient values
            self._apply_shutdown_dynamics()
            return self._get_full_telemetry()

        self.time_step_counter += 1
        
        # 1. Evaluate Active Drifts
        drift_effects = self._calculate_drift_effects()
        
        # 2. Reactor Mass & Energy Balance
        # Reactor Temp increases with Reaction rate, decreases with cooling water flow
        reaction_rate = (
            self.state["feed_d_rate"] * 0.4 + 
            self.state["feed_e_rate"] * 0.6
        ) * (self.state["reactor_temp"] / 120.0) * drift_effects["reaction_mult"]
        
        heat_generated = reaction_rate * 4.8  # Exothermic heat constant
        
        # Cooling water cooling capacity
        cooling_efficiency = self.controls["reactor_cooling_valve"] / 100.0 * drift_effects["valve_efficiency"]
        heat_removed = cooling_efficiency * 5.2 * (self.state["reactor_temp"] - self.state["reactor_cooling_water_temp"])
        
        # Temp delta
        temp_delta = (heat_generated - heat_removed) * 0.1 + random.normalvariate(0, 0.05)
        self.state["reactor_temp"] = round(self.state["reactor_temp"] + temp_delta, 2)
        
        # Reactor Pressure follows ideal gas law P ~ T * moles
        moles_in = (self.state["feed_a_rate"] + self.state["feed_d_rate"] + self.state["feed_e_rate"]) * 0.8
        moles_out = (self.state["product_flow_rate"] + self.state["purge_rate"]) * 1.2
        mole_delta = (moles_in - moles_out) * 0.2
        
        pressure_delta = (mole_delta * 12.0 + (temp_delta * 8.0)) + random.normalvariate(0, 1.0)
        self.state["reactor_pressure"] = round(self.state["reactor_pressure"] + pressure_delta, 1)
        
        # 3. Separator & Stripper Dynamics
        # Level balances
        feed_inflow = self.state["feed_a_rate"] + self.state["feed_d_rate"] + self.state["feed_e_rate"]
        product_factor = (self.state["reactor_temp"] / 120.0)
        
        # Separator Level balance
        separator_inflow = feed_inflow * 0.7 * product_factor
        separator_outflow = self.state["product_flow_rate"] * 0.9
        self.state["separator_level"] = round(
            max(0.0, min(100.0, self.state["separator_level"] + (separator_inflow - separator_outflow) * 0.1)), 2
        )
        
        # Stripper dynamics (steam valve influence)
        steam_heating = (self.controls["stripper_steam_valve"] / 100.0) * 8.0
        self.state["stripper_temp"] = round(
            self.state["stripper_temp"] + (steam_heating - (self.state["stripper_temp"] - 60.0) * 0.3) * 0.1, 2
        )
        
        # Product flow is stripper outflow
        target_product_flow = (self.state["stripper_level"] / 50.0) * 22.8 * (self.state["stripper_temp"] / 65.0)
        self.state["product_flow_rate"] = round(
            target_product_flow + random.normalvariate(0, 0.1), 2
        )
        
        # 4. Check Safety Interlocks (Shutdown if limits breached)
        self._check_safety_limits()
        
        return self._get_full_telemetry()

    def _calculate_drift_effects(self) -> Dict[str, float]:
        """Calculate multipliers and offsets induced by active drifts."""
        effects = {
            "reaction_mult": 1.0,
            "valve_efficiency": 1.0,
            "feed_d_temp_offset": 0.0
        }
        
        for drift_type, info in self.active_drifts.items():
            steps_active = self.time_step_counter - info["start_step"]
            sev = info["severity"]
            
            if drift_type == "catalyst_decay":
                # Slow decay of yield
                decay = min(0.3, steps_active * 0.002 * sev)
                effects["reaction_mult"] = 1.0 - decay
                
            elif drift_type == "reactor_valve_stiction":
                # Oscillating cooling capacity because valve is stuck/slipping
                oscillation = math.sin(steps_active * 0.3) * 18.0 * sev
                # Stuck offset
                valve_offset = -12.0 * sev
                effects["valve_efficiency"] = max(0.2, min(1.5, 1.0 + (valve_offset + oscillation) / 100.0))
                
            elif drift_type == "feed_d_temp_step":
                # Instant step increase in feed temperature
                effects["feed_d_temp_offset"] = 25.0 * sev
                # Increases reactor temperature baseline
                self.state["reactor_cooling_water_temp"] = round(35.2 + effects["feed_d_temp_offset"] * 0.1, 2)
                
            elif drift_type == "feed_a_composition":
                # Slow shift in reactant ratio, reducing reactor efficiency
                shift = min(0.25, steps_active * 0.005 * sev)
                effects["reaction_mult"] = 1.0 - shift
                # Causes reactor pressure build-up due to unreacted gas
                self.state["reactor_pressure"] += 3.0 * sev
                
        return effects

    def _check_safety_limits(self) -> None:
        """Safety interlock logic to shut down the reactor if it breaches limits."""
        # 1. Reactor Temperature limit (Safety trip at 135°C)
        if self.state["reactor_temp"] >= 135.0:
            self.safety_tripped = True
            self.trip_reason = "HIGH REACTOR TEMPERATURE TRIP (>= 135°C)"
            
        # 2. Reactor Pressure limit (Safety trip at 3100 kPa)
        elif self.state["reactor_pressure"] >= 3100.0:
            self.safety_tripped = True
            self.trip_reason = "HIGH REACTOR PRESSURE TRIP (>= 3100 kPa)"

    def _apply_shutdown_dynamics(self) -> None:
        """Apply passive dynamics when reactor trips (cools down/depressurizes)."""
        # Cool down towards ambient room temperature (30°C)
        self.state["reactor_temp"] = round(self.state["reactor_temp"] - (self.state["reactor_temp"] - 30.0) * 0.08, 2)
        # Bleed pressure towards ambient atmospheric (101.3 kPa)
        self.state["reactor_pressure"] = round(self.state["reactor_pressure"] - (self.state["reactor_pressure"] - 101.3) * 0.1, 1)
        # Drop levels
        self.state["reactor_level"] = round(self.state["reactor_level"] * 0.95, 2)
        self.state["product_flow_rate"] = 0.0
        self.state["feed_a_rate"] = 0.0
        self.state["feed_d_rate"] = 0.0
        self.state["feed_e_rate"] = 0.0

    def _get_full_telemetry(self) -> Dict[str, float]:
        """Combine state and control variables into one flat dictionary."""
        telemetry = {}
        telemetry.update(self.state)
        telemetry.update(self.controls)
        telemetry["safety_tripped"] = 1.0 if self.safety_tripped else 0.0
        return telemetry

    def generate_history(self, num_steps: int = 50, drift_to_inject: str = None) -> List[Dict[str, Any]]:
        """
        Generate a consecutive list of telemetry dictionaries representing historical run.
        """
        self.reset()
        history = []
        
        # Determine when to inject drift (e.g. halfway through the run)
        inject_point = num_steps // 3
        
        for i in range(num_steps):
            if drift_to_inject and i == inject_point:
                self.inject_drift(drift_to_inject)
            
            step_data = self.step()
            
            # Add timestamp
            timestamp = datetime.now() - timedelta(seconds=(num_steps - i) * 30)
            step_data["timestamp"] = timestamp.isoformat()
            step_data["step_idx"] = float(i)
            
            history.append(step_data)
            
        return history
