"""
Preheater Tower Model with Heat Exchange and Alkali Circulation
Implements the critical preheater/calciner system that was missing from the original model.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class PreheaterTower:
    """
    Advanced preheater tower model simulating heat exchange and material separation.
    
    Features:
    - Multi-stage cyclone efficiency modeling
    - Heat and mass balance calculations
    - Alkali circulation and buildup modeling
    - Pressure drop calculations
    - Volatile cycle modeling (alkali, sulfur, chlorine)
    """
    
    def __init__(self, num_stages: int = 5):
        self.num_stages = num_stages
        
        # Cyclone efficiency increases towards top (cooler) stages
        # Stage 1 (bottom, hottest) to Stage 5 (top, coolest)
        self.cyclone_efficiency = [0.98, 0.95, 0.92, 0.90, 0.85]  # Stage 1 to 5
        self.stage_gas_temps = [900, 750, 600, 450, 320]  # Exit gas temps per stage (°C)
        self.stage_meal_temps = [850, 700, 550, 400, 280]  # Exit meal temps per stage (°C)
        
        # Pressure drops per stage (Pa) - will be calculated using Barth equation
        self.pressure_drops = [800, 600, 500, 400, 300]
        
        # Alkali buildup factor (critical for operational problems)
        self.alkali_buildup_factor = 1.0
        self.sulfur_buildup_factor = 1.0
        self.chlorine_buildup_factor = 1.0
        
        # Volatility factors for different compounds
        self.volatility_factors = {
            'K2O': 0.6,    # 60% of K2O volatilizes
            'Na2O': 0.5,   # 50% of Na2O volatilizes
            'SO3': 0.8,    # 80% of SO3 volatilizes
            'Cl': 0.9      # 90% of Cl volatilizes
        }
        
        print(f"🏗️ Preheater Tower Model initialized ({num_stages} stages)")
        print("📊 Heat exchange and alkali circulation modeling active")
        print("📐 Barth equation for cyclone pressure drop calculations available")
    
    def calculate_heat_and_mass_balance(self,
                                       raw_meal_flow: float,
                                       gas_flow: float,
                                       raw_meal_composition: Dict[str, float],
                                       raw_meal_temp: float = 20.0) -> Dict[str, Any]:
        """
        Simulate comprehensive heat and mass balance in preheater tower.
        
        Args:
            raw_meal_flow: Raw meal flow rate (t/h)
            gas_flow: Gas flow rate (Nm³/h)
            raw_meal_composition: Raw meal chemical composition
            raw_meal_temp: Raw meal inlet temperature (°C)
            
        Returns:
            Dict with heat balance results and final conditions
        """
        # Stage-by-stage heat balance calculation
        stage_results = []
        current_meal_temp = raw_meal_temp
        current_gas_temp = self.stage_gas_temps[0]  # Start with hottest gas
        
        total_heat_recovered = 0
        total_pressure_drop = 0
        
        for stage in range(self.num_stages):
            stage_num = stage + 1
            
            # Heat exchange calculation
            meal_cp = 0.84  # kJ/kg·K (specific heat of raw meal)
            gas_cp = 1.05   # kJ/kg·K (specific heat of gas)
            
            # Temperature approach (minimum temperature difference)
            temp_approach = 50  # °C
            
            # Calculate heat transfer
            meal_heat_capacity = raw_meal_flow * 1000 * meal_cp  # kJ/h·K
            gas_heat_capacity = gas_flow * 1.2 * gas_cp  # kJ/h·K (assuming 1.2 kg/Nm³)
            
            # Heat transfer rate (limited by smaller heat capacity)
            heat_transfer_rate = min(meal_heat_capacity, gas_heat_capacity) * temp_approach
            
            # Update temperatures
            meal_temp_rise = heat_transfer_rate / meal_heat_capacity
            gas_temp_drop = heat_transfer_rate / gas_heat_capacity
            
            current_meal_temp += meal_temp_rise
            current_gas_temp -= gas_temp_drop
            
            # Pressure drop calculation using Barth equation
            # Typical gas velocities and dust loadings for cement preheater stages
            typical_velocities = [25, 22, 20, 18, 15]  # m/s (decreasing towards top)
            typical_dust_loadings = [200, 150, 100, 80, 60]  # g/Nm³ (decreasing towards top)
            
            gas_velocity = typical_velocities[stage] if stage < len(typical_velocities) else typical_velocities[-1]
            dust_loading = typical_dust_loadings[stage] if stage < len(typical_dust_loadings) else typical_dust_loadings[-1]
            
            stage_pressure_drop = self.calculate_cyclone_pressure_drop(gas_velocity, dust_loading)
            total_pressure_drop += stage_pressure_drop
            
            # Heat recovery
            heat_recovered = heat_transfer_rate / 1000  # MJ/h
            total_heat_recovered += heat_recovered
            
            stage_results.append({
                'stage': stage_num,
                'meal_temp_out': current_meal_temp,
                'gas_temp_out': current_gas_temp,
                'heat_transfer': heat_transfer_rate,
                'pressure_drop': stage_pressure_drop,
                'gas_velocity': gas_velocity,
                'dust_loading': dust_loading,
                'cyclone_efficiency': self.cyclone_efficiency[stage]
            })
        
        # Final meal temperature entering kiln
        final_meal_temp = current_meal_temp
        
        return {
            'stage_results': stage_results,
            'final_meal_temp': final_meal_temp,
            'final_gas_temp': current_gas_temp,
            'total_heat_recovered': total_heat_recovered,
            'total_pressure_drop': total_pressure_drop,
            'heat_recovery_efficiency': total_heat_recovered / (raw_meal_flow * 1000 * 0.84 * (final_meal_temp - raw_meal_temp))
        }
    
    def calculate_alkali_circulation(self,
                                   raw_meal_alkali: Dict[str, float],
                                   kiln_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Model alkali circulation and buildup in preheater tower.
        
        Critical for operational problems like buildups and blockages.
        
        Args:
            raw_meal_alkali: Alkali content in raw meal (K2O, Na2O %)
            kiln_conditions: Kiln operating conditions (temperature, etc.)
            
        Returns:
            Dict with alkali circulation analysis
        """
        k2o_input = raw_meal_alkali.get('K2O', 0.5)
        na2o_input = raw_meal_alkali.get('Na2O', 0.2)
        
        # Volatilization in kiln (temperature dependent)
        kiln_temp = kiln_conditions.get('temperature', 1450)
        temp_factor = (kiln_temp - 1400) / 100  # Normalized around 1400°C
        
        # Volatilization rates (increased with temperature)
        k2o_volatility = self.volatility_factors['K2O'] * (1 + temp_factor * 0.2)
        na2o_volatility = self.volatility_factors['Na2O'] * (1 + temp_factor * 0.2)
        
        # Circulating alkalis
        circulating_k2o = k2o_input * k2o_volatility
        circulating_na2o = na2o_input * na2o_volatility
        
        # Buildup calculation (simplified model)
        # Higher circulating alkalis = more buildup potential
        alkali_buildup_rate = (circulating_k2o + circulating_na2o) * 0.05
        
        # Add SO3/Cl circulation impact (volatile circulation enhancement)
        so3_content = kiln_conditions.get('so3_content', 1.0)
        cl_content = kiln_conditions.get('cl_content', 0.1)
        volatile_effect = (so3_content + cl_content * 0.1) * 0.3
        alkali_buildup_rate += volatile_effect * 0.02  # Additional buildup from volatiles
        
        # Update buildup factors
        self.alkali_buildup_factor += alkali_buildup_rate
        
        # Operational impact assessment
        buildup_risk = 'Low'
        if self.alkali_buildup_factor > 2.0:
            buildup_risk = 'High'
        elif self.alkali_buildup_factor > 1.5:
            buildup_risk = 'Medium'
        
        return {
            'k2o_input': k2o_input,
            'na2o_input': na2o_input,
            'k2o_volatility': k2o_volatility,
            'na2o_volatility': na2o_volatility,
            'circulating_k2o': circulating_k2o,
            'circulating_na2o': circulating_na2o,
            'volatile_effect': volatile_effect,
            'alkali_buildup_factor': self.alkali_buildup_factor,
            'buildup_risk': buildup_risk,
            'operational_recommendations': self._generate_alkali_recommendations(buildup_risk)
        }
    
    def calculate_volatile_cycles(self,
                                raw_meal_composition: Dict[str, float],
                                kiln_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate volatile cycles for sulfur and chlorine compounds.
        
        Args:
            raw_meal_composition: Raw meal composition including volatiles
            kiln_conditions: Kiln operating conditions
            
        Returns:
            Dict with volatile cycle analysis
        """
        so3_input = raw_meal_composition.get('SO3', 1.0)
        cl_input = raw_meal_composition.get('Cl', 0.1)
        
        kiln_temp = kiln_conditions.get('temperature', 1450)
        temp_factor = (kiln_temp - 1400) / 100
        
        # Volatilization rates
        so3_volatility = self.volatility_factors['SO3'] * (1 + temp_factor * 0.1)
        cl_volatility = self.volatility_factors['Cl'] * (1 + temp_factor * 0.1)
        
        # Circulating volatiles
        circulating_so3 = so3_input * so3_volatility
        circulating_cl = cl_input * cl_volatility
        
        # Buildup factors
        self.sulfur_buildup_factor += circulating_so3 * 0.02
        self.chlorine_buildup_factor += circulating_cl * 0.1
        
        # Environmental impact
        so2_emission_potential = circulating_so3 * 0.8  # 80% conversion to SO2
        hcl_emission_potential = circulating_cl * 0.9   # 90% conversion to HCl
        
        return {
            'so3_input': so3_input,
            'cl_input': cl_input,
            'so3_volatility': so3_volatility,
            'cl_volatility': cl_volatility,
            'circulating_so3': circulating_so3,
            'circulating_cl': circulating_cl,
            'sulfur_buildup_factor': self.sulfur_buildup_factor,
            'chlorine_buildup_factor': self.chlorine_buildup_factor,
            'so2_emission_potential': so2_emission_potential,
            'hcl_emission_potential': hcl_emission_potential
        }
    
    def simulate_preheater_performance(self,
                                     raw_meal_flow: float,
                                     gas_flow: float,
                                     raw_meal_properties: Dict[str, Any],
                                     kiln_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate complete preheater tower performance.
        
        Args:
            raw_meal_flow: Raw meal flow rate (t/h)
            gas_flow: Gas flow rate (Nm³/h)
            raw_meal_properties: Raw meal properties and composition
            kiln_conditions: Kiln operating conditions
            
        Returns:
            Comprehensive preheater performance analysis
        """
        # Heat and mass balance
        heat_balance = self.calculate_heat_and_mass_balance(
            raw_meal_flow=raw_meal_flow,
            gas_flow=gas_flow,
            raw_meal_composition=raw_meal_properties.get('composition', {}),
            raw_meal_temp=raw_meal_properties.get('temperature', 20.0)
        )
        
        # Alkali circulation
        alkali_analysis = self.calculate_alkali_circulation(
            raw_meal_alkali=raw_meal_properties.get('alkali_content', {}),
            kiln_conditions=kiln_conditions
        )
        
        # Volatile cycles
        volatile_analysis = self.calculate_volatile_cycles(
            raw_meal_composition=raw_meal_properties.get('composition', {}),
            kiln_conditions=kiln_conditions
        )
        
        # Overall performance metrics
        performance_metrics = self._calculate_performance_metrics(
            heat_balance, alkali_analysis, volatile_analysis
        )
        
        return {
            'heat_and_mass_balance': heat_balance,
            'alkali_circulation': alkali_analysis,
            'volatile_cycles': volatile_analysis,
            'performance_metrics': performance_metrics,
            'operating_conditions': {
                'raw_meal_flow': raw_meal_flow,
                'gas_flow': gas_flow,
                'num_stages': self.num_stages
            }
        }
    
    def _calculate_performance_metrics(self,
                                     heat_balance: Dict[str, Any],
                                     alkali_analysis: Dict[str, Any],
                                     volatile_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall preheater performance metrics."""
        
        # Heat recovery efficiency
        heat_recovery_efficiency = heat_balance.get('heat_recovery_efficiency', 0)
        
        # Alkali buildup risk
        alkali_risk = alkali_analysis.get('buildup_risk', 'Low')
        
        # Volatile emission potential
        so2_potential = volatile_analysis.get('so2_emission_potential', 0)
        hcl_potential = volatile_analysis.get('hcl_emission_potential', 0)
        
        # Overall performance score
        performance_score = 0.8  # Base score
        
        if heat_recovery_efficiency > 0.7:
            performance_score += 0.1
        if alkali_risk == 'Low':
            performance_score += 0.05
        if so2_potential < 2.0:
            performance_score += 0.05
        
        performance_score = min(1.0, performance_score)
        
        return {
            'heat_recovery_efficiency': heat_recovery_efficiency,
            'alkali_buildup_risk': alkali_risk,
            'volatile_emission_potential': {
                'so2': so2_potential,
                'hcl': hcl_potential
            },
            'overall_performance_score': performance_score,
            'operational_status': 'Good' if performance_score > 0.8 else 'Acceptable' if performance_score > 0.6 else 'Poor'
        }
    
    def _generate_alkali_recommendations(self, buildup_risk: str) -> List[str]:
        """Generate operational recommendations based on alkali buildup risk."""
        recommendations = []
        
        if buildup_risk == 'High':
            recommendations.append("Consider reducing alkali content in raw meal")
            recommendations.append("Increase kiln temperature to improve volatilization")
            recommendations.append("Monitor preheater buildups closely")
            recommendations.append("Consider bypass system for alkali control")
        elif buildup_risk == 'Medium':
            recommendations.append("Monitor alkali circulation trends")
            recommendations.append("Consider slight increase in kiln temperature")
            recommendations.append("Regular preheater inspection recommended")
        else:
            recommendations.append("Alkali levels are within acceptable range")
            recommendations.append("Continue current operating practices")
        
        return recommendations
    
    def calculate_cyclone_pressure_drop(self, gas_velocity: float, dust_loading: float) -> float:
        """
        Calculate cyclone pressure drop using Barth equation.
        
        Args:
            gas_velocity: Gas velocity (m/s)
            dust_loading: Dust loading (g/Nm³)
            
        Returns:
            Pressure drop (Pa)
        """
        # Barth equation for cyclone pressure drop
        # ΔP = 4.5 * v² * (1 + dust_loading/1000)
        pressure_drop = 4.5 * (gas_velocity ** 2) * (1 + dust_loading / 1000)
        return pressure_drop
    
    def calculate_stage_pressure_drops(self, 
                                     gas_velocities: List[float], 
                                     dust_loadings: List[float]) -> List[float]:
        """
        Calculate pressure drops for all stages using Barth equation.
        
        Args:
            gas_velocities: Gas velocities for each stage (m/s)
            dust_loadings: Dust loadings for each stage (g/Nm³)
            
        Returns:
            List of pressure drops (Pa)
        """
        pressure_drops = []
        for i in range(len(gas_velocities)):
            velocity = gas_velocities[i] if i < len(gas_velocities) else gas_velocities[-1]
            dust_loading = dust_loadings[i] if i < len(dust_loadings) else dust_loadings[-1]
            pressure_drop = self.calculate_cyclone_pressure_drop(velocity, dust_loading)
            pressure_drops.append(pressure_drop)
        
        return pressure_drops


def create_preheater_tower(num_stages: int = 5) -> PreheaterTower:
    """Factory function to create a preheater tower model."""
    return PreheaterTower(num_stages)
