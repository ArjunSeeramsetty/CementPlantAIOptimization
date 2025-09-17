import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.optimize import minimize, NonlinearConstraint
from dataclasses import dataclass

@dataclass
class FuelProperties:
    """Properties of different fuel types"""
    calorific_value: float  # kcal/kg
    moisture_content: float  # %
    ash_content: float  # %
    sulfur_content: float  # %
    chlorine_content: float  # ppm
    alkali_content: float  # %
    volatiles: float  # %

class AlternativeFuelOptimizer:
    """
    Advanced alternative fuel optimizer for cement plants.
    Maximizes TSR while maintaining clinker quality and operational stability.
    Based on JK Cement requirements and industry best practices.
    """
    
    def __init__(self, tsr_target: float = 0.15, max_tsr: float = 0.60):
        self.tsr_target = tsr_target
        self.max_tsr = max_tsr
        
        # Define fuel properties based on industry data
        self.fuel_database = {
            'coal': FuelProperties(6500, 8, 12, 0.8, 200, 1.2, 30),
            'petcoke': FuelProperties(8000, 2, 0.5, 3.5, 50, 0.3, 12),
            'rdf': FuelProperties(4200, 25, 15, 0.3, 8000, 2.5, 60),
            'biomass': FuelProperties(3800, 30, 8, 0.1, 300, 4.0, 70),
            'tire_chips': FuelProperties(7200, 2, 12, 1.2, 1000, 0.8, 65),
            'paper_sludge': FuelProperties(3200, 40, 20, 0.2, 500, 2.0, 65)
        }
    
    def calculate_fuel_requirements(self, 
                                  production_rate: float,
                                  thermal_energy_demand: float) -> float:
        """Calculate total fuel requirement in kg/h"""
        return (production_rate * thermal_energy_demand) / 3600  # Convert kcal/h to kg/h
    
    def calculate_tsr(self, fuel_mix: Dict[str, float]) -> float:
        """Calculate Thermal Substitution Rate"""
        total_alt_fuels = sum(fuel_mix[fuel] for fuel in fuel_mix 
                            if fuel not in ['coal', 'petcoke'])
        total_fuels = sum(fuel_mix.values())
        return total_alt_fuels / total_fuels if total_fuels > 0 else 0
    
    def calculate_quality_impact(self, fuel_mix: Dict[str, float]) -> Dict[str, float]:
        """Calculate impact on clinker quality parameters"""
        total_fuel = sum(fuel_mix.values())
        if total_fuel == 0:
            return {'chlorine_input': 0, 'sulfur_input': 0, 'alkali_input': 0}
        
        weighted_chlorine = sum(
            fuel_mix[fuel] * self.fuel_database[fuel].chlorine_content / 1000
            for fuel in fuel_mix if fuel in self.fuel_database
        ) / total_fuel
        
        weighted_sulfur = sum(
            fuel_mix[fuel] * self.fuel_database[fuel].sulfur_content
            for fuel in fuel_mix if fuel in self.fuel_database
        ) / total_fuel
        
        weighted_alkali = sum(
            fuel_mix[fuel] * self.fuel_database[fuel].alkali_content
            for fuel in fuel_mix if fuel in self.fuel_database
        ) / total_fuel
        
        return {
            'chlorine_input': weighted_chlorine,
            'sulfur_input': weighted_sulfur,
            'alkali_input': weighted_alkali,
            'quality_penalty': self._calculate_quality_penalty(
                weighted_chlorine, weighted_sulfur, weighted_alkali
            )
        }
    
    def _calculate_quality_penalty(self, cl: float, s: float, alkali: float) -> float:
        """Calculate quality penalty based on contaminant levels"""
        penalty = 0
        # Chlorine penalty (target < 0.1%)
        if cl > 0.1:
            penalty += (cl - 0.1) ** 2 * 100
        # Sulfur penalty (target < 2%)
        if s > 2.0:
            penalty += (s - 2.0) ** 2 * 10
        # Alkali penalty (target < 3%)
        if alkali > 3.0:
            penalty += (alkali - 3.0) ** 2 * 20
        return penalty
    
    def optimize_fuel_blend(self,
                          available_fuels: Dict[str, float],
                          fuel_costs: Dict[str, float],
                          quality_constraints: Dict[str, float],
                          production_rate: float = 167) -> Dict:
        """
        Optimize fuel blend for maximum TSR with quality constraints
        
        Args:
            available_fuels: Available quantities {fuel_type: kg/h}
            fuel_costs: Fuel costs {fuel_type: $/kg}
            quality_constraints: Max limits {parameter: max_value}
            production_rate: Plant production rate (t/h)
        """
        
        fuel_types = list(available_fuels.keys())
        n_fuels = len(fuel_types)
        
        # Initial guess: proportional to availability
        total_available = sum(available_fuels.values())
        x0 = np.array([available_fuels[fuel]/total_available for fuel in fuel_types])
        
        # Objective: Maximize TSR while minimizing cost and quality penalty
        def objective(x):
            fuel_mix = {fuel_types[i]: x[i] * total_available for i in range(n_fuels)}
            tsr = self.calculate_tsr(fuel_mix)
            quality_impact = self.calculate_quality_impact(fuel_mix)
            
            cost = sum(x[i] * total_available * fuel_costs.get(fuel_types[i], 0)
                      for i in range(n_fuels))
            
            # Multi-objective: maximize TSR, minimize cost and quality penalty
            return -(tsr * 100) + cost * 0.01 + quality_impact['quality_penalty']
        
        # Constraints
        constraints = []
        
        # Mass balance constraint
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        
        # TSR constraint
        def tsr_constraint(x):
            fuel_mix = {fuel_types[i]: x[i] * total_available for i in range(n_fuels)}
            return self.calculate_tsr(fuel_mix) - 0.05  # Minimum 5% TSR
        
        constraints.append({'type': 'ineq', 'fun': tsr_constraint})
        
        # Quality constraints
        def quality_constraint(x):
            fuel_mix = {fuel_types[i]: x[i] * total_available for i in range(n_fuels)}
            quality_impact = self.calculate_quality_impact(fuel_mix)
            return quality_constraints.get('max_chlorine', 0.15) - quality_impact['chlorine_input']
        
        constraints.append({'type': 'ineq', 'fun': quality_constraint})
        
        # Bounds: each fuel between 0 and available quantity
        bounds = [(0, 1) for _ in range(n_fuels)]
        
        # Solve optimization
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6, 'maxiter': 1000}
        )
        
        if result.success:
            optimized_mix = {fuel_types[i]: result.x[i] * total_available 
                           for i in range(n_fuels)}
            tsr_achieved = self.calculate_tsr(optimized_mix)
            quality_impact = self.calculate_quality_impact(optimized_mix)
            
            return {
                'success': True,
                'optimized_fuel_mix': optimized_mix,
                'tsr_achieved': tsr_achieved,
                'quality_impact': quality_impact,
                'total_cost': sum(result.x[i] * total_available * fuel_costs.get(fuel_types[i], 0)
                                for i in range(n_fuels)),
                'optimization_message': result.message
            }
        else:
            return {
                'success': False,
                'message': result.message,
                'fallback_mix': {fuel: available_fuels[fuel] * 0.8 for fuel in fuel_types}
            }

    def generate_rdf_scenarios(self, base_scenario: Dict) -> List[Dict]:
        """Generate RDF optimization scenarios for different conditions"""
        scenarios = []
        
        rdf_percentages = [0.10, 0.15, 0.20, 0.25, 0.30]  # 10% to 30% TSR
        
        for rdf_pct in rdf_percentages:
            scenario = base_scenario.copy()
            scenario['rdf_percentage'] = rdf_pct
            scenario['expected_savings'] = self._calculate_savings(rdf_pct)
            scenario['quality_risk'] = self._assess_quality_risk(rdf_pct)
            scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_savings(self, rdf_percentage: float) -> float:
        """Calculate expected cost savings from RDF usage"""
        # Typical savings: $15-25 per ton clinker at 20% TSR
        return rdf_percentage * 100 * 1.2  # Simplified calculation
    
    def _assess_quality_risk(self, rdf_percentage: float) -> str:
        """Assess quality risk level based on TSR"""
        if rdf_percentage < 0.15:
            return "Low"
        elif rdf_percentage < 0.25:
            return "Medium" 
        else:
            return "High"
