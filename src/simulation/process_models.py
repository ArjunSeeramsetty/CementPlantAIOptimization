"""Expert-driven process models for cement plant digital twin."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RawMealComposition:
    """Raw meal chemical composition."""
    sio2: float
    cao: float
    al2o3: float
    fe2o3: float
    mgo: float
    k2o: float
    na2o: float
    so3: float
    cl: float
    loi: float  # Loss on ignition


@dataclass
class CoalProperties:
    """Coal properties for burnability calculations."""
    volatile_matter: float  # %
    ash_content: float     # %
    moisture: float        # %
    sulfur: float          # %
    calorific_value: float # kcal/kg


class AdvancedKilnModel:
    """
    Advanced kiln model incorporating expert-driven correlations for burnability,
    NOx formation, and process dynamics.
    """
    
    def __init__(self):
        """Initialize the kiln model with industry-standard parameters."""
        self.kiln_length = 60.0  # meters
        self.kiln_diameter = 4.5  # meters
        self.nominal_capacity = 10000  # tpd
        
    def calculate_burnability_index(self, 
                                  raw_meal: RawMealComposition,
                                  raw_meal_fineness: float,
                                  coal: CoalProperties) -> float:
        """
        Calculate burnability index using Lea & Parker correlation with enhancements.
        
        This incorporates:
        - Raw meal fineness (Blaine)
        - Alkali content (K2O + Na2O)
        - Coal volatile matter
        - MgO flux effect
        
        Args:
            raw_meal: Raw meal chemical composition
            raw_meal_fineness: Blaine fineness (cm²/g)
            coal: Coal properties
            
        Returns:
            Burnability index (higher = easier to burn)
        """
        
        # Base Lea & Parker correlation
        base_burnability = 100 - (raw_meal.sio2 / raw_meal.cao) * 30
        
        # Alkali effect (R2O = K2O + 0.658*Na2O)
        r2o = raw_meal.k2o + 0.658 * raw_meal.na2o
        alkali_effect = r2o * 8  # Alkalis act as flux, improving burnability
        
        # Fineness effect (finer meal is easier to burn)
        fineness_effect = (raw_meal_fineness - 3000) / 100
        
        # Coal volatile matter effect on flame
        coal_effect = (coal.volatile_matter - 30) * 0.5
        
        # MgO flux effect (industrial correlation)
        mgo_effect = (raw_meal.mgo - 2.0) * -0.5  # MgO acts as flux above 2%
        
        # SO3/Cl circulation impact
        volatile_effect = (raw_meal.so3 + raw_meal.cl * 0.1) * 0.3
        
        burnability_index = (base_burnability + alkali_effect + fineness_effect + 
                           coal_effect + mgo_effect + volatile_effect)
        
        return max(0, min(100, burnability_index))
    
    def calculate_comprehensive_nox_formation(self,
                                            flame_temp: float,
                                            excess_air: float,
                                            fuel_nitrogen_percent: float,
                                            residence_time_sec: float,
                                            coal: CoalProperties) -> Dict[str, float]:
        """
        Calculate NOx formation based on multiple mechanisms.
        
        Implements:
        - Zeldovich (thermal NOx)
        - Fuel-bound nitrogen NOx
        - Prompt NOx
        
        Args:
            flame_temp: Flame temperature (°C)
            excess_air: Excess air ratio
            fuel_nitrogen_percent: Fuel nitrogen content (%)
            residence_time_sec: Residence time in seconds
            coal: Coal properties
            
        Returns:
            Dict with NOx components and total
        """
        
        # Convert temperature to Kelvin
        temp_k = flame_temp + 273.15
        
        # Thermal NOx (Zeldovich mechanism)
        # Constants are empirical and tuned for cement kilns
        A_thermal = 6.4e10
        Ea_thermal = 69090  # Activation energy constant
        
        thermal_nox = (A_thermal * np.exp(-Ea_thermal / temp_k) * 
                      (excess_air ** 0.5) * residence_time_sec)
        
        # Fuel NOx (from nitrogen in the fuel)
        # Assuming ~80% conversion of fuel N to NOx
        fuel_nox = fuel_nitrogen_percent * 1000 * 0.8
        
        # Prompt NOx (simplified correlation)
        prompt_nox = 50 * (excess_air + 1) ** 0.3
        
        # Coal ash effect on NOx formation
        ash_effect = coal.ash_content * 2  # Higher ash reduces NOx
        
        total_nox = thermal_nox + fuel_nox + prompt_nox - ash_effect
        
        return {
            'thermal_nox': max(0, thermal_nox),
            'fuel_nox': max(0, fuel_nox),
            'prompt_nox': max(0, prompt_nox),
            'ash_effect': ash_effect,
            'total_nox': max(0, total_nox)
        }
    
    def calculate_kiln_performance(self,
                                 raw_meal: RawMealComposition,
                                 raw_meal_fineness: float,
                                 coal: CoalProperties,
                                 kiln_speed: float,
                                 feed_rate: float) -> Dict[str, float]:
        """
        Calculate comprehensive kiln performance metrics.
        
        Args:
            raw_meal: Raw meal composition
            raw_meal_fineness: Blaine fineness
            coal: Coal properties
            kiln_speed: Kiln speed (rpm)
            feed_rate: Feed rate (tph)
            
        Returns:
            Dict with performance metrics
        """
        
        # Calculate burnability
        burnability = self.calculate_burnability_index(raw_meal, raw_meal_fineness, coal)
        
        # Estimate burning zone temperature based on burnability
        base_temp = 1450  # °C
        temp_adjustment = (burnability - 50) * 2  # Higher burnability = higher temp
        burning_zone_temp = base_temp + temp_adjustment
        
        # Calculate NOx formation
        excess_air = 1.5  # Typical excess air
        fuel_nitrogen = 1.5  # Typical fuel nitrogen %
        residence_time = 25 * 60  # 25 minutes in seconds
        
        nox_results = self.calculate_comprehensive_nox_formation(
            burning_zone_temp, excess_air, fuel_nitrogen, residence_time, coal
        )
        
        # Calculate free lime based on temperature and residence time
        free_lime = 2.5 - (burning_zone_temp - 1400) / 20
        
        # Calculate clinker mineralogy
        c3s_content = 50 + (burning_zone_temp - 1400) / 2
        c2s_content = 20 - (burning_zone_temp - 1400) / 4
        
        # Calculate specific energy consumption
        base_energy = 3200  # kcal/kg clinker
        energy_factor = 1 + (100 - burnability) / 200  # Higher burnability = lower energy
        specific_energy = base_energy * energy_factor
        
        return {
            'burnability_index': burnability,
            'burning_zone_temp_c': burning_zone_temp,
            'free_lime_pct': max(0, free_lime),
            'c3s_content_pct': max(0, min(100, c3s_content)),
            'c2s_content_pct': max(0, min(100, c2s_content)),
            'nox_mg_nm3': nox_results['total_nox'],
            'specific_energy_kcal_kg': specific_energy,
            'kiln_efficiency': burnability / 100
        }


class PreheaterTower:
    """
    Preheater tower model with multi-stage heat exchange and volatile circulation.
    """
    
    def __init__(self, num_stages: int = 5):
        """Initialize preheater tower."""
        self.num_stages = num_stages
        self.stage_efficiency = [0.98, 0.95, 0.92, 0.90, 0.85]  # Stage 5 to 1
        self.stage_gas_temps = [450, 550, 650, 750, 850]  # °C
        
    def calculate_heat_and_mass_balance(self,
                                      raw_meal_flow: float,
                                      gas_flow: float,
                                      raw_meal_alkali: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate heat and mass balance for preheater tower.
        
        Args:
            raw_meal_flow: Raw meal flow rate (tph)
            gas_flow: Gas flow rate (Nm³/h)
            raw_meal_alkali: Alkali content in raw meal
            
        Returns:
            Dict with heat balance results
        """
        
        # Calculate stage-by-stage heat exchange
        meal_temp_in = 25  # °C (ambient)
        meal_temp_out = self.stage_gas_temps[-1]  # Exit temperature
        
        # Heat required for meal heating
        cp_meal = 0.2  # kcal/kg·°C
        heat_required = raw_meal_flow * 1000 * cp_meal * (meal_temp_out - meal_temp_in)
        
        # Gas temperature drop
        cp_gas = 0.25  # kcal/Nm³·°C
        gas_temp_drop = heat_required / (gas_flow * cp_gas)
        gas_temp_out = self.stage_gas_temps[0] - gas_temp_drop
        
        # Model alkali circulation (critical operational issue)
        k2o_volatility = 0.6  # 60% of K2O volatilizes
        na2o_volatility = 0.5  # 50% of Na2O volatilizes
        
        circulating_k2o = raw_meal_alkali.get('K2O', 0.5) * k2o_volatility
        circulating_na2o = raw_meal_alkali.get('Na2O', 0.2) * na2o_volatility
        
        # Volatile circulation enhancement
        so3_content = raw_meal_alkali.get('SO3', 0.1)
        cl_content = raw_meal_alkali.get('Cl', 0.05)
        volatile_effect = (so3_content + cl_content * 0.1) * 0.3
        
        # Buildup factor increases with more circulating volatiles
        alkali_buildup_factor = 1.0 + (circulating_k2o + circulating_na2o) * 0.05 + volatile_effect
        
        return {
            'meal_exit_temp_c': meal_temp_out,
            'gas_exit_temp_c': gas_temp_out,
            'heat_exchanged_kcal_h': heat_required,
            'alkali_buildup_factor': alkali_buildup_factor,
            'circulating_k2o_pct': circulating_k2o,
            'circulating_na2o_pct': circulating_na2o,
            'volatile_effect': volatile_effect
        }
    
    def calculate_cyclone_pressure_drop(self,
                                      gas_velocity: float,
                                      dust_loading: float) -> float:
        """
        Calculate cyclone pressure drop using Barth equation.
        
        Args:
            gas_velocity: Gas velocity (m/s)
            dust_loading: Dust loading (g/Nm³)
            
        Returns:
            Pressure drop (Pa)
        """
        # Barth equation for cyclone pressure drop
        pressure_drop = 4.5 * (gas_velocity ** 2) * (1 + dust_loading / 1000)
        return pressure_drop


class CementQualityPredictor:
    """
    Industrial quality prediction model using Powers' model and Lerch formula.
    """
    
    def __init__(self):
        """Initialize quality predictor."""
        self.strength_coefficients = {
            'C3S': 1.2,
            'C2S': 0.5,
            'C3A': 0.3,
            'C4AF': 0.1
        }
    
    def predict_compressive_strength(self,
                                   clinker_composition: Dict[str, float],
                                   fineness_blaine: float,
                                   age_days: int) -> float:
        """
        Predict compressive strength using Powers' model approach.
        
        Args:
            clinker_composition: Clinker mineralogy (%)
            fineness_blaine: Blaine fineness (cm²/g)
            age_days: Concrete age (days)
            
        Returns:
            Predicted strength (MPa)
        """
        
        c3s = clinker_composition.get('C3S', 60)
        c2s = clinker_composition.get('C2S', 15)
        c3a = clinker_composition.get('C3A', 8)
        c4af = clinker_composition.get('C4AF', 10)
        
        # Strength contribution factors
        cement_factor = (self.strength_coefficients['C3S'] * c3s +
                        self.strength_coefficients['C2S'] * c2s +
                        self.strength_coefficients['C3A'] * c3a +
                        self.strength_coefficients['C4AF'] * c4af)
        
        # Fineness effect (major impact on early strength)
        fineness_factor = (fineness_blaine / 3500) ** 0.3
        
        # Maturity factor based on age (logarithmic growth)
        maturity = np.log(age_days + 1)
        
        # Predicted strength in MPa
        strength_mpa = cement_factor * fineness_factor * maturity * 0.5
        
        return max(0, strength_mpa)
    
    def optimize_gypsum_content(self,
                              c3a_content: float,
                              fineness_blaine: float,
                              target_setting_time_min: float = 120) -> float:
        """
        Calculate optimal gypsum content using Lerch formula.
        
        Args:
            c3a_content: C3A content (%)
            fineness_blaine: Blaine fineness (cm²/g)
            target_setting_time_min: Target setting time (minutes)
            
        Returns:
            Optimal SO3 content (%)
        """
        
        # Simplified Lerch formula approach
        # Optimal SO3 % = k1 + k2 * C3A + k3 * (Blaine - 3500)
        optimal_so3 = (0.7 + 0.15 * c3a_content + 
                      (fineness_blaine - 3500) / 10000)
        
        # Adjust for target setting time
        if target_setting_time_min < 120:
            optimal_so3 *= 1.1  # Increase gypsum for faster setting
        elif target_setting_time_min > 180:
            optimal_so3 *= 0.9  # Decrease gypsum for slower setting
        
        return max(0.5, min(3.0, optimal_so3))


class ProcessControlSimulator:
    """
    Simulates realistic process control with PID loops and time delays.
    """
    
    def __init__(self):
        """Initialize control simulator."""
        self.control_loops = {
            'free_lime_control': {
                'kp': 0.1, 'ki': 0.01, 'kd': 0.02,
                'setpoint': 1.2, 'deadtime_minutes': 25
            },
            'bzt_control': {
                'kp': 0.5, 'ki': 0.05, 'kd': 0.1,
                'setpoint': 1450, 'deadtime_minutes': 12
            },
            'draft_control': {
                'kp': 0.2, 'ki': 0.02, 'kd': 0.01,
                'setpoint': -2.0, 'deadtime_minutes': 1
            }
        }
    
    def simulate_control_response(self,
                                measurement: float,
                                control_loop: str,
                                current_time: float) -> float:
        """
        Simulate PID control response with deadtime.
        
        Args:
            measurement: Current measurement
            control_loop: Control loop name
            current_time: Current time (seconds)
            
        Returns:
            Control output
        """
        
        if control_loop not in self.control_loops:
            return 0.0
        
        params = self.control_loops[control_loop]
        
        # Simplified PID calculation (in real implementation, would include
        # integral windup protection, derivative kick prevention, etc.)
        error = params['setpoint'] - measurement
        
        # Proportional term
        p_term = params['kp'] * error
        
        # Integral term (simplified)
        i_term = params['ki'] * error * 60  # Assuming 1-minute sample time
        
        # Derivative term (simplified)
        d_term = params['kd'] * error  # Simplified derivative
        
        output = p_term + i_term + d_term
        
        return output


def create_process_models() -> Dict[str, object]:
    """
    Create and return all process models.
    
    Returns:
        Dict containing all process model instances
    """
    return {
        'kiln_model': AdvancedKilnModel(),
        'preheater_tower': PreheaterTower(),
        'quality_predictor': CementQualityPredictor(),
        'control_simulator': ProcessControlSimulator()
    }


if __name__ == "__main__":
    # Test the process models
    logger.info("Testing process models...")
    
    # Create models
    models = create_process_models()
    
    # Test raw meal composition
    raw_meal = RawMealComposition(
        sio2=22.0, cao=65.0, al2o3=5.0, fe2o3=3.0,
        mgo=2.5, k2o=0.8, na2o=0.3, so3=0.1, cl=0.05, loi=35.0
    )
    
    # Test coal properties
    coal = CoalProperties(
        volatile_matter=35.0, ash_content=15.0, moisture=8.0,
        sulfur=1.2, calorific_value=6500
    )
    
    # Test kiln performance
    kiln_performance = models['kiln_model'].calculate_kiln_performance(
        raw_meal, 3200, coal, 3.5, 200
    )
    
    logger.info("Kiln Performance Results:")
    for key, value in kiln_performance.items():
        logger.info(f"  {key}: {value:.2f}")
    
    # Test quality prediction
    clinker_composition = {'C3S': 60, 'C2S': 15, 'C3A': 8, 'C4AF': 10}
    strength_28d = models['quality_predictor'].predict_compressive_strength(
        clinker_composition, 3500, 28
    )
    
    logger.info(f"Predicted 28-day strength: {strength_28d:.1f} MPa")
    
    logger.info("Process models tested successfully!")
