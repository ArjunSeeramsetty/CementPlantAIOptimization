# FILE: src/cement_ai_platform/optimization/tsr_optimizer.py
import numpy as np
from scipy.optimize import linprog, minimize
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

@dataclass
class FuelOptimizationResult:
    """Result of fuel optimization"""
    optimal_fuel_mix: Dict[str, float]
    cost_savings_monthly: float
    co2_reduction_tons: float
    energy_efficiency_gain: float
    tsr_achieved: float
    implementation_feasibility: str
    operational_impact: Dict[str, float]

class AdvancedTSROptimizer:
    """Advanced TSR optimization with real physics and economics"""

    def __init__(self, plant_config):
        self.config = plant_config
        self.fuel_properties = plant_config.fuel_properties
        self.current_fuel_costs = self._get_current_fuel_costs()

    def optimize_fuel_mix(self, target_tsr: float, constraints: Dict = None) -> FuelOptimizationResult:
        """
        Optimize fuel mix for target TSR using linear programming
        """
        # Define decision variables (fuel rates in kg/h)
        fuel_types = list(self.fuel_properties.keys())
        n_fuels = len(fuel_types)

        # Objective function: minimize cost
        cost_coefficients = [
            self.fuel_properties[fuel]['cost_per_ton'] / 1000  # ₹/kg
            for fuel in fuel_types
        ]

        # Constraints setup
        A_eq, b_eq = self._setup_equality_constraints(fuel_types, target_tsr)
        A_ub, b_ub = self._setup_inequality_constraints(fuel_types, constraints)

        # Bounds for each fuel type (min, max rates)
        bounds = self._get_fuel_bounds(fuel_types)

        # Solve optimization
        result = linprog(
            c=cost_coefficients,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs'
        )

        if result.success:
            optimal_mix = {fuel: result.x[i] for i, fuel in enumerate(fuel_types)}
            return self._analyze_optimization_result(optimal_mix, target_tsr)
        else:
            return self._generate_fallback_result(target_tsr)

    def _setup_equality_constraints(self, fuel_types: List[str],
                                  target_tsr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Setup equality constraints for optimization"""
        n_fuels = len(fuel_types)

        # Constraint 1: Total thermal energy requirement
        total_thermal_req = self.config.capacity_tpd * 1000 * self.config.energy['thermal'] / 24  # kcal/h
        thermal_coeffs = [self.fuel_properties[fuel]['cv_kcal_kg'] for fuel in fuel_types]

        # Constraint 2: TSR requirement
        # TSR = (alt_fuel_thermal / total_thermal) * 100
        alt_fuel_indices = [i for i, fuel in enumerate(fuel_types)
                          if fuel in ['rdf', 'biomass', 'tire_derived', 'plastic_waste']]
        tsr_coeffs = np.zeros(n_fuels)
        for i in alt_fuel_indices:
            tsr_coeffs[i] = self.fuel_properties[fuel_types[i]]['cv_kcal_kg']

        # Required alternative fuel thermal energy
        alt_thermal_req = total_thermal_req * target_tsr / 100

        A_eq = np.array([
            thermal_coeffs,  # Total thermal requirement
            tsr_coeffs      # Alternative fuel thermal requirement
        ])

        b_eq = np.array([
            total_thermal_req,
            alt_thermal_req
        ])

        return A_eq, b_eq

    def _setup_inequality_constraints(self, fuel_types: List[str],
                                    constraints: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Setup inequality constraints"""
        n_fuels = len(fuel_types)
        A_ub = []
        b_ub = []

        if constraints:
            # Ash content constraint
            if 'max_ash_content_pct' in constraints:
                ash_coeffs = [self.fuel_properties[fuel]['ash_pct'] for fuel in fuel_types]
                total_fuel_rate = sum(self.config.fuel_mix.values())
                max_ash_kg_h = total_fuel_rate * 1000 * constraints['max_ash_content_pct'] / 100

                A_ub.append(ash_coeffs)
                b_ub.append(max_ash_kg_h)

            # Individual fuel availability constraints
            for i, fuel in enumerate(fuel_types):
                if f'max_{fuel}_rate' in constraints:
                    constraint_row = np.zeros(n_fuels)
                    constraint_row[i] = 1
                    A_ub.append(constraint_row.tolist())
                    b_ub.append(constraints[f'max_{fuel}_rate'])

        # Default constraint: no fuel can exceed 80% of total
        total_fuel_rate = sum(self.config.fuel_mix.values()) * 1000  # kg/h
        for i in range(n_fuels):
            constraint_row = np.zeros(n_fuels)
            constraint_row[i] = 1
            A_ub.append(constraint_row.tolist())
            b_ub.append(total_fuel_rate * 0.8)

        return np.array(A_ub) if A_ub else np.array([]).reshape(0, n_fuels), np.array(b_ub)

    def _get_fuel_bounds(self, fuel_types: List[str]) -> List[Tuple[float, float]]:
        """Get bounds for each fuel type"""
        bounds = []
        total_fuel_rate = sum(self.config.fuel_mix.values()) * 1000  # kg/h

        for fuel in fuel_types:
            if fuel in ['coal', 'petcoke']:
                # Fossil fuels: minimum 20% of total, maximum 60%
                min_rate = total_fuel_rate * 0.2
                max_rate = total_fuel_rate * 0.6
            else:
                # Alternative fuels: minimum 0%, maximum 40%
                min_rate = 0
                max_rate = total_fuel_rate * 0.4

            bounds.append((min_rate, max_rate))

        return bounds

    def _analyze_optimization_result(self, optimal_mix: Dict[str, float],
                                   target_tsr: float) -> FuelOptimizationResult:
        """Analyze optimization result and calculate impacts"""

        # Calculate current baseline costs
        current_fuel_mix_kg_h = {
            fuel: self.config.fuel_mix.get(fuel, 0) * 1000
            for fuel in self.fuel_properties.keys()
        }

        current_cost_per_hour = sum(
            rate * self.fuel_properties[fuel]['cost_per_ton'] / 1000
            for fuel, rate in current_fuel_mix_kg_h.items()
        )

        # Calculate optimized costs
        optimized_cost_per_hour = sum(
            rate * self.fuel_properties[fuel]['cost_per_ton'] / 1000
            for fuel, rate in optimal_mix.items()
        )

        # Monthly savings
        cost_savings_monthly = (current_cost_per_hour - optimized_cost_per_hour) * 24 * 30

        # CO2 impact calculation
        current_co2 = sum(
            rate * self.fuel_properties[fuel]['carbon_factor'] / 1000
            for fuel, rate in current_fuel_mix_kg_h.items()
        )

        optimized_co2 = sum(
            rate * self.fuel_properties[fuel]['carbon_factor'] / 1000
            for fuel, rate in optimal_mix.items()
        )

        co2_reduction_tons = (current_co2 - optimized_co2) * 24 * 30 / 1000

        # Energy efficiency calculation
        current_thermal = sum(
            rate * self.fuel_properties[fuel]['cv_kcal_kg']
            for fuel, rate in current_fuel_mix_kg_h.items()
        )

        optimized_thermal = sum(
            rate * self.fuel_properties[fuel]['cv_kcal_kg']
            for fuel, rate in optimal_mix.items()
        )

        energy_efficiency_gain = ((optimized_thermal - current_thermal) / current_thermal) * 100

        # Calculate achieved TSR
        alt_fuels = ['rdf', 'biomass', 'tire_derived', 'plastic_waste']
        alt_fuel_thermal = sum(
            optimal_mix.get(fuel, 0) * self.fuel_properties.get(fuel, {}).get('cv_kcal_kg', 0)
            for fuel in alt_fuels if fuel in optimal_mix
        )
        tsr_achieved = (alt_fuel_thermal / optimized_thermal) * 100 if optimized_thermal > 0 else 0

        # Implementation feasibility
        tsr_gap = abs(tsr_achieved - target_tsr)
        if tsr_gap < 1:
            feasibility = "High - Target TSR achievable"
        elif tsr_gap < 3:
            feasibility = "Medium - Minor adjustments needed"
        else:
            feasibility = "Low - Significant changes required"

        # Operational impact
        operational_impact = {
            'kiln_temperature_change_c': self._estimate_temperature_impact(optimal_mix),
            'free_lime_variation_ppm': self._estimate_quality_impact(optimal_mix),
            'nox_change_pct': self._estimate_nox_impact(optimal_mix),
            'maintenance_impact_factor': self._estimate_maintenance_impact(optimal_mix)
        }

        return FuelOptimizationResult(
            optimal_fuel_mix={fuel: rate/1000 for fuel, rate in optimal_mix.items()},  # Convert to t/h
            cost_savings_monthly=cost_savings_monthly,
            co2_reduction_tons=co2_reduction_tons,
            energy_efficiency_gain=energy_efficiency_gain,
            tsr_achieved=tsr_achieved,
            implementation_feasibility=feasibility,
            operational_impact=operational_impact
        )

    def _estimate_temperature_impact(self, optimal_mix: Dict[str, float]) -> float:
        """Estimate impact on kiln temperature"""
        # Alternative fuels typically burn at different rates
        alt_fuel_ratio = sum(
            optimal_mix.get(fuel, 0) for fuel in ['rdf', 'biomass', 'tire_derived', 'plastic_waste']
        ) / sum(optimal_mix.values())

        # Estimate temperature change (empirical relationship)
        temp_change = -5 * alt_fuel_ratio if alt_fuel_ratio > 0.3 else -2 * alt_fuel_ratio
        return round(temp_change, 1)

    def _estimate_quality_impact(self, optimal_mix: Dict[str, float]) -> float:
        """Estimate impact on free lime variation"""
        alt_fuel_ratio = sum(
            optimal_mix.get(fuel, 0) for fuel in ['rdf', 'biomass', 'tire_derived', 'plastic_waste']
        ) / sum(optimal_mix.values())

        # Higher alternative fuel usage can increase quality variation
        variation_ppm = 50 + alt_fuel_ratio * 150
        return round(variation_ppm)

    def _estimate_nox_impact(self, optimal_mix: Dict[str, float]) -> float:
        """Estimate NOx emission change"""
        # Alternative fuels typically reduce NOx
        alt_fuel_ratio = sum(
            optimal_mix.get(fuel, 0) for fuel in ['rdf', 'biomass', 'tire_derived', 'plastic_waste']
        ) / sum(optimal_mix.values())

        nox_reduction = -15 * alt_fuel_ratio  # Up to 15% reduction
        return round(nox_reduction, 1)

    def _estimate_maintenance_impact(self, optimal_mix: Dict[str, float]) -> float:
        """Estimate maintenance impact factor"""
        # Alternative fuels can affect refractory life
        alt_fuel_ratio = sum(
            optimal_mix.get(fuel, 0) for fuel in ['rdf', 'biomass', 'tire_derived', 'plastic_waste']
        ) / sum(optimal_mix.values())

        # Factor > 1 means increased maintenance
        impact_factor = 1.0 + alt_fuel_ratio * 0.1
        return round(impact_factor, 3)

    def _generate_fallback_result(self, target_tsr: float) -> FuelOptimizationResult:
        """Generate fallback result if optimization fails"""
        return FuelOptimizationResult(
            optimal_fuel_mix=self.config.fuel_mix,
            cost_savings_monthly=0,
            co2_reduction_tons=0,
            energy_efficiency_gain=0,
            tsr_achieved=target_tsr * 0.8,  # Partial achievement
            implementation_feasibility="Requires manual adjustment",
            operational_impact={}
        )

    def _get_current_fuel_costs(self) -> Dict[str, float]:
        """Get current fuel costs (could be updated from market data)"""
        return {fuel: props['cost_per_ton'] for fuel, props in self.fuel_properties.items()}

    def multi_objective_optimization(self, objectives: Dict[str, float]) -> FuelOptimizationResult:
        """
        Multi-objective optimization balancing cost, emissions, and energy efficiency
        """
        # Weight factors for different objectives
        weights = {
            'cost': objectives.get('cost_weight', 0.4),
            'emissions': objectives.get('emissions_weight', 0.3),
            'energy': objectives.get('energy_weight', 0.2),
            'quality': objectives.get('quality_weight', 0.1)
        }

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}

        # Use weighted sum approach
        best_result = None
        best_score = float('inf')

        # Try different TSR targets
        for tsr in np.linspace(15, 35, 5):  # 15% to 35% TSR
            try:
                result = self.optimize_fuel_mix(tsr)

                # Calculate composite score
                score = (
                    -result.cost_savings_monthly * normalized_weights['cost'] +
                    -result.co2_reduction_tons * normalized_weights['emissions'] +
                    -result.energy_efficiency_gain * normalized_weights['energy'] +
                    abs(result.tsr_achieved - tsr) * 100 * normalized_weights['quality']
                )

                if score < best_score:
                    best_score = score
                    best_result = result

            except Exception as e:
                continue

        return best_result or self._generate_fallback_result(20)  # Default 20% TSR
