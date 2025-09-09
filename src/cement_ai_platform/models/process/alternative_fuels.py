"""
Alternative Fuels Processing and Co-processing System
Handles fuel characterization, blending optimization, and environmental impact assessment.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class FuelProperties:
    """Comprehensive fuel characterization parameters."""
    name: str
    calorific_value: float    # MJ/kg (lower heating value)
    moisture_content: float   # % wet basis
    ash_content: float        # % dry basis
    volatile_matter: float    # % dry ash-free basis
    fixed_carbon: float       # % dry ash-free basis
    sulfur_content: float     # % dry basis
    chlorine_content: float   # mg/kg dry basis
    heavy_metals: Dict[str, float] = field(default_factory=dict)  # mg/kg dry basis
    ultimate_analysis: Dict[str, float] = field(default_factory=dict)  # C, H, N, O %
    ash_composition: Dict[str, float] = field(default_factory=dict)  # SiO2, Al2O3, etc. %
    cost_per_ton: float = 0.0  # $/t
    availability: float = 1.0  # Availability factor (0-1)

class AlternativeFuelProcessor:
    """
    Advanced alternative fuel processing system for cement plants.
    Handles fuel preparation, blending optimization, and co-processing simulation.
    """

    def __init__(self):
        # Database of typical alternative fuels
        self.fuel_database = {
            'refuse_derived_fuel': FuelProperties(
                name='Refuse Derived Fuel (RDF)',
                calorific_value=15.0,
                moisture_content=20.0,
                ash_content=25.0,
                volatile_matter=65.0,
                fixed_carbon=10.0,
                sulfur_content=0.5,
                chlorine_content=8000,
                heavy_metals={'Hg': 0.5, 'Cd': 2.0, 'Pb': 50.0, 'Cr': 100.0},
                ultimate_analysis={'C': 45.0, 'H': 6.0, 'N': 1.0, 'O': 23.0},
                ash_composition={'SiO2': 45.0, 'Al2O3': 20.0, 'CaO': 15.0, 'Fe2O3': 8.0},
                cost_per_ton=25.0
            ),
            'biomass': FuelProperties(
                name='Biomass (Wood Waste)',
                calorific_value=12.0,
                moisture_content=30.0,
                ash_content=5.0,
                volatile_matter=75.0,
                fixed_carbon=20.0,
                sulfur_content=0.1,
                chlorine_content=2000,
                heavy_metals={'Hg': 0.1, 'Cd': 0.5, 'Pb': 5.0, 'Cr': 10.0},
                ultimate_analysis={'C': 48.0, 'H': 6.5, 'N': 0.5, 'O': 40.0},
                ash_composition={'SiO2': 55.0, 'Al2O3': 8.0, 'CaO': 25.0, 'Fe2O3': 3.0},
                cost_per_ton=40.0
            ),
            'waste_oil': FuelProperties(
                name='Waste Oil',
                calorific_value=40.0,
                moisture_content=1.0,
                ash_content=0.1,
                volatile_matter=85.0,
                fixed_carbon=14.9,
                sulfur_content=1.0,
                chlorine_content=500,
                heavy_metals={'Hg': 0.05, 'Cd': 0.1, 'Pb': 10.0, 'Cr': 5.0},
                ultimate_analysis={'C': 85.0, 'H': 12.0, 'N': 0.5, 'O': 1.5},
                ash_composition={'SiO2': 30.0, 'Al2O3': 15.0, 'CaO': 20.0, 'Fe2O3': 25.0},
                cost_per_ton=100.0
            ),
            'tire_derived_fuel': FuelProperties(
                name='Tire Derived Fuel (TDF)',
                calorific_value=30.0,
                moisture_content=2.0,
                ash_content=12.0,
                volatile_matter=65.0,
                fixed_carbon=23.0,
                sulfur_content=1.5,
                chlorine_content=1000,
                heavy_metals={'Hg': 0.2, 'Cd': 1.0, 'Pb': 20.0, 'Cr': 50.0},
                ultimate_analysis={'C': 80.0, 'H': 7.0, 'N': 0.5, 'O': 0.5},
                ash_composition={'SiO2': 25.0, 'Al2O3': 5.0, 'CaO': 15.0, 'Fe2O3': 35.0, 'ZnO': 15.0},
                cost_per_ton=0.0  # Often negative cost (paid to accept)
            ),
            'sewage_sludge': FuelProperties(
                name='Sewage Sludge',
                calorific_value=10.0,
                moisture_content=75.0,
                ash_content=35.0,
                volatile_matter=50.0,
                fixed_carbon=15.0,
                sulfur_content=1.2,
                chlorine_content=3000,
                heavy_metals={'Hg': 1.0, 'Cd': 5.0, 'Pb': 100.0, 'Cr': 200.0},
                ultimate_analysis={'C': 30.0, 'H': 4.0, 'N': 5.0, 'O': 25.0},
                ash_composition={'SiO2': 35.0, 'Al2O3': 15.0, 'CaO': 20.0, 'Fe2O3': 10.0, 'P2O5': 15.0},
                cost_per_ton=-20.0  # Negative cost - paid to accept
            )
        }

        # Environmental constraints (typical cement plant limits)
        self.environmental_limits = {
            'max_chlorine_input': 0.15,      # % of clinker production
            'max_mercury_input': 0.05,       # mg/kg clinker
            'max_cadmium_input': 0.1,        # mg/kg clinker
            'max_lead_input': 1.0,           # mg/kg clinker
            'max_alternative_fuel_rate': 0.8, # Fraction of total thermal energy
            'min_calorific_value': 8.0,      # MJ/kg minimum for stable combustion
            'max_ash_in_fuel': 30.0          # % max ash content in fuel blend
        }

        print("🔥 Alternative Fuel Processor initialized")
        print(f"📊 Fuel database: {len(self.fuel_database)} fuel types")
        print(f"🌍 Environmental limits configured")

    def optimize_fuel_blend(self, 
                          available_fuels: Dict[str, Dict],
                          target_thermal_substitution: float,
                          coal_properties: FuelProperties,
                          constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize alternative fuel blend for minimum cost while meeting environmental constraints.

        Args:
            available_fuels: Available fuels with quantities and costs
            target_thermal_substitution: Target % thermal substitution rate
            coal_properties: Reference coal properties
            constraints: Additional constraints (optional)

        Returns:
            Optimization results with optimal blend and performance metrics
        """
        from scipy.optimize import minimize

        fuel_names = list(available_fuels.keys())
        n_fuels = len(fuel_names)

        if n_fuels == 0:
            return {'error': 'No alternative fuels available'}

        # Merge default and custom constraints
        limits = {**self.environmental_limits}
        if constraints:
            limits.update(constraints)

        def objective_function(blend_fractions):
            """Minimize fuel cost per GJ of energy."""
            total_cost = 0
            total_energy = 0

            for i, fuel_name in enumerate(fuel_names):
                fuel_props = self.fuel_database[fuel_name]
                fuel_fraction = blend_fractions[i]
                fuel_cost = available_fuels[fuel_name].get('cost', fuel_props.cost_per_ton)

                # Energy contribution
                energy_contribution = fuel_fraction * fuel_props.calorific_value
                cost_contribution = fuel_fraction * fuel_cost

                total_energy += energy_contribution
                total_cost += cost_contribution

            # Add coal fraction
            coal_fraction = 1 - sum(blend_fractions)
            coal_energy = coal_fraction * coal_properties.calorific_value
            coal_cost = coal_fraction * coal_properties.cost_per_ton

            total_energy += coal_energy
            total_cost += coal_cost

            if total_energy <= 0:
                return 1e6  # Penalty for invalid solution

            return total_cost / total_energy  # Cost per GJ

        def thermal_substitution_constraint(blend_fractions):
            """Ensure target thermal substitution rate is met (relaxed)."""
            alt_fuel_energy = sum(blend_fractions[i] * self.fuel_database[fuel_names[i]].calorific_value 
                                for i in range(n_fuels))
            coal_energy = (1 - sum(blend_fractions)) * coal_properties.calorific_value
            total_energy = alt_fuel_energy + coal_energy

            if total_energy <= 0:
                return -1  # Invalid

            actual_substitution = alt_fuel_energy / total_energy
            # Allow some tolerance in substitution rate
            return actual_substitution - target_thermal_substitution * 0.8  # 80% of target is acceptable

        def chlorine_constraint(blend_fractions):
            """Chlorine input constraint."""
            weighted_chlorine = sum(blend_fractions[i] * self.fuel_database[fuel_names[i]].chlorine_content 
                                  for i in range(n_fuels))
            coal_chlorine = (1 - sum(blend_fractions)) * coal_properties.chlorine_content
            blended_chlorine = weighted_chlorine + coal_chlorine

            return limits['max_chlorine_input'] * 10000 - blended_chlorine  # Convert % to mg/kg

        def mercury_constraint(blend_fractions):
            """Mercury input constraint."""
            weighted_mercury = sum(blend_fractions[i] * self.fuel_database[fuel_names[i]].heavy_metals.get('Hg', 0) 
                                 for i in range(n_fuels))
            coal_mercury = (1 - sum(blend_fractions)) * coal_properties.heavy_metals.get('Hg', 0.1)
            blended_mercury = weighted_mercury + coal_mercury

            return limits['max_mercury_input'] - blended_mercury

        def calorific_value_constraint(blend_fractions):
            """Minimum calorific value constraint (relaxed)."""
            weighted_cv = sum(blend_fractions[i] * self.fuel_database[fuel_names[i]].calorific_value 
                            for i in range(n_fuels))
            coal_cv = (1 - sum(blend_fractions)) * coal_properties.calorific_value
            blended_cv = weighted_cv + coal_cv

            return blended_cv - limits['min_calorific_value'] * 0.9  # 90% of minimum is acceptable

        def ash_constraint(blend_fractions):
            """Maximum ash content constraint."""
            weighted_ash = sum(blend_fractions[i] * self.fuel_database[fuel_names[i]].ash_content 
                             for i in range(n_fuels))
            coal_ash = (1 - sum(blend_fractions)) * coal_properties.ash_content
            blended_ash = weighted_ash + coal_ash

            return limits['max_ash_in_fuel'] - blended_ash

        # Set up constraints
        opt_constraints = [
            {'type': 'eq', 'fun': thermal_substitution_constraint},
            {'type': 'ineq', 'fun': chlorine_constraint},
            {'type': 'ineq', 'fun': mercury_constraint},
            {'type': 'ineq', 'fun': calorific_value_constraint},
            {'type': 'ineq', 'fun': ash_constraint}
        ]

        # Bounds: each fuel fraction between 0 and max availability
        bounds = []
        for fuel_name in fuel_names:
            max_fraction = available_fuels[fuel_name].get('max_fraction', 0.5)
            bounds.append((0, max_fraction))

        # Constraint: sum of fractions <= max alternative fuel rate
        def max_alt_fuel_constraint(blend_fractions):
            return limits['max_alternative_fuel_rate'] - sum(blend_fractions)

        opt_constraints.append({'type': 'ineq', 'fun': max_alt_fuel_constraint})

        # Initial guess - more conservative
        x0 = [min(target_thermal_substitution / n_fuels, 0.1)] * n_fuels

        # Run optimization with fallback
        try:
            result = minimize(
                objective_function, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=opt_constraints,
                options={'ftol': 1e-6, 'maxiter': 100}
            )
        except Exception as e:
            # Fallback to simpler optimization
            print(f"SLSQP failed, trying COBYLA: {e}")
            result = minimize(
                objective_function, x0,
                method='COBYLA',
                bounds=bounds,
                options={'maxiter': 50}
            )

        if result.success:
            optimal_blend = {fuel_names[i]: result.x[i] for i in range(n_fuels)}
            coal_fraction = 1 - sum(result.x)
            optimal_blend['coal'] = coal_fraction

            # Calculate blended fuel properties
            blended_properties = self._calculate_blended_properties(optimal_blend, coal_properties)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(optimal_blend, coal_properties)

            return {
                'optimization_successful': True,
                'optimal_blend': optimal_blend,
                'blended_fuel_properties': blended_properties,
                'performance_metrics': performance_metrics,
                'cost_per_gj': result.fun,
                'thermal_substitution_achieved': performance_metrics['thermal_substitution_rate']
            }
        else:
            # Fallback: simple heuristic blend
            print(f"Optimization failed: {result.message}, using heuristic blend")
            heuristic_blend = self._create_heuristic_blend(fuel_names, target_thermal_substitution, available_fuels)
            
            if heuristic_blend:
                blended_properties = self._calculate_blended_properties(heuristic_blend, coal_properties)
                performance_metrics = self._calculate_performance_metrics(heuristic_blend, coal_properties)
                
                return {
                    'optimization_successful': True,
                    'optimal_blend': heuristic_blend,
                    'blended_fuel_properties': blended_properties,
                    'performance_metrics': performance_metrics,
                    'cost_per_gj': self._calculate_heuristic_cost(heuristic_blend, coal_properties),
                    'thermal_substitution_achieved': performance_metrics['thermal_substitution_rate'],
                    'method': 'heuristic_fallback'
                }
            else:
                return {
                    'optimization_successful': False,
                    'error_message': result.message,
                    'iterations': result.nit
                }

    def simulate_coprocessing_performance(self,
                                        fuel_blend: Dict[str, float],
                                        coal_properties: FuelProperties,
                                        kiln_conditions: Dict[str, float],
                                        production_rate: float) -> Dict[str, Any]:
        """
        Simulate co-processing performance including combustion efficiency and emissions.

        Args:
            fuel_blend: Fuel blend fractions (including coal)
            coal_properties: Coal reference properties
            kiln_conditions: Kiln operating conditions
            production_rate: Clinker production rate (t/h)

        Returns:
            Comprehensive co-processing performance analysis
        """
        # Calculate blended fuel properties
        blended_properties = self._calculate_blended_properties(fuel_blend, coal_properties)

        # Combustion efficiency analysis
        combustion_efficiency = self._calculate_combustion_efficiency(blended_properties, kiln_conditions)

        # Emission calculations
        emissions = self._calculate_detailed_emissions(blended_properties, kiln_conditions, production_rate)

        # Energy recovery analysis
        energy_analysis = self._calculate_energy_recovery(blended_properties, fuel_blend, combustion_efficiency)

        # Ash incorporation effects
        ash_effects = self._calculate_ash_incorporation_effects(blended_properties, production_rate)

        # Economic analysis
        economic_analysis = self._calculate_economic_impact(fuel_blend, coal_properties, energy_analysis)

        # Environmental impact
        environmental_impact = self._assess_environmental_impact(emissions, ash_effects)

        return {
            'blended_fuel_properties': blended_properties.__dict__,
            'combustion_performance': {
                'combustion_efficiency': combustion_efficiency,
                'flame_temperature': self._estimate_flame_temperature(blended_properties),
                'combustion_air_requirement': self._calculate_air_requirement(blended_properties)
            },
            'emissions': emissions,
            'energy_analysis': energy_analysis,
            'ash_incorporation': ash_effects,
            'economic_impact': economic_analysis,
            'environmental_assessment': environmental_impact,
            'operational_recommendations': self._generate_operational_recommendations(
                blended_properties, combustion_efficiency, emissions
            )
        }

    def _calculate_blended_properties(self, fuel_blend: Dict[str, float], 
                                    coal_properties: FuelProperties) -> FuelProperties:
        """Calculate weighted average properties of fuel blend."""

        # Initialize blended properties
        blended_cv = 0
        blended_moisture = 0
        blended_ash = 0
        blended_volatile = 0
        blended_sulfur = 0
        blended_chlorine = 0
        blended_metals = {'Hg': 0, 'Cd': 0, 'Pb': 0, 'Cr': 0}
        blended_ultimate = {'C': 0, 'H': 0, 'N': 0, 'O': 0}

        for fuel_name, fraction in fuel_blend.items():
            if fuel_name == 'coal':
                props = coal_properties
            else:
                props = self.fuel_database.get(fuel_name)
                if not props:
                    continue

            blended_cv += fraction * props.calorific_value
            blended_moisture += fraction * props.moisture_content
            blended_ash += fraction * props.ash_content
            blended_volatile += fraction * props.volatile_matter
            blended_sulfur += fraction * props.sulfur_content
            blended_chlorine += fraction * props.chlorine_content

            # Heavy metals
            for metal in blended_metals:
                blended_metals[metal] += fraction * props.heavy_metals.get(metal, 0)

            # Ultimate analysis
            for element in blended_ultimate:
                blended_ultimate[element] += fraction * props.ultimate_analysis.get(element, 0)

        return FuelProperties(
            name='Blended Fuel',
            calorific_value=blended_cv,
            moisture_content=blended_moisture,
            ash_content=blended_ash,
            volatile_matter=blended_volatile,
            fixed_carbon=100 - blended_volatile - blended_ash,
            sulfur_content=blended_sulfur,
            chlorine_content=blended_chlorine,
            heavy_metals=blended_metals,
            ultimate_analysis=blended_ultimate
        )

    def _calculate_performance_metrics(self, fuel_blend: Dict[str, float],
                                     coal_properties: FuelProperties) -> Dict[str, float]:
        """Calculate key performance metrics for the fuel blend."""

        # Thermal substitution rate
        alt_fuel_energy = sum(fraction * self.fuel_database[fuel_name].calorific_value 
                            for fuel_name, fraction in fuel_blend.items() 
                            if fuel_name != 'coal')

        total_energy = alt_fuel_energy + fuel_blend.get('coal', 0) * coal_properties.calorific_value
        thermal_substitution_rate = alt_fuel_energy / total_energy if total_energy > 0 else 0

        # Cost savings calculation
        coal_cost_per_gj = coal_properties.cost_per_ton / coal_properties.calorific_value

        blended_cost_per_gj = 0
        for fuel_name, fraction in fuel_blend.items():
            if fuel_name == 'coal':
                props = coal_properties
            else:
                props = self.fuel_database[fuel_name]

            fuel_cost_per_gj = props.cost_per_ton / props.calorific_value
            blended_cost_per_gj += fraction * fuel_cost_per_gj

        cost_savings_per_gj = coal_cost_per_gj - blended_cost_per_gj

        return {
            'thermal_substitution_rate': thermal_substitution_rate,
            'cost_savings_per_gj': cost_savings_per_gj,
            'cost_savings_percent': (cost_savings_per_gj / coal_cost_per_gj) * 100 if coal_cost_per_gj > 0 else 0
        }

    def _calculate_combustion_efficiency(self, fuel_props: FuelProperties,
                                       kiln_conditions: Dict[str, float]) -> float:
        """Calculate combustion efficiency based on fuel properties and kiln conditions."""

        base_efficiency = 0.95

        # Moisture penalty
        moisture_penalty = fuel_props.moisture_content * 0.008  # 0.8% per % moisture

        # Volatile matter effect (higher VM = easier ignition)
        vm_bonus = (fuel_props.volatile_matter - 30) * 0.002  # Bonus above 30% VM

        # Temperature effect
        temp_effect = (kiln_conditions.get('temperature', 1450) - 1400) / 1000 * 0.05

        # Oxygen availability effect
        oxygen_effect = (kiln_conditions.get('oxygen', 3.0) - 2.0) / 10

        efficiency = base_efficiency - moisture_penalty + vm_bonus + temp_effect + oxygen_effect

        return max(0.75, min(0.98, efficiency))

    def _calculate_detailed_emissions(self, fuel_props: FuelProperties,
                                    kiln_conditions: Dict[str, float],
                                    production_rate: float) -> Dict[str, float]:
        """Calculate detailed emissions from alternative fuel combustion."""

        # NOx emissions (mg/Nm³ @ 10% O2)
        nox_base = 600  # Base NOx from coal
        nox_fuel_nitrogen_effect = fuel_props.ultimate_analysis.get('N', 1.0) * 150
        nox_volatile_effect = (fuel_props.volatile_matter - 30) * 3
        nox_temperature_effect = (kiln_conditions.get('temperature', 1450) - 1400) * 0.5

        nox_emission = nox_base + nox_fuel_nitrogen_effect + nox_volatile_effect + nox_temperature_effect

        # SO2 emissions (mg/Nm³ @ 10% O2)
        so2_emission = fuel_props.sulfur_content * 1600  # Assuming 80% conversion to SO2

        # HCl emissions (mg/Nm³ @ 10% O2)
        hcl_emission = fuel_props.chlorine_content * 0.8  # 80% release as HCl

        # CO emissions (mg/Nm³ @ 10% O2)
        co_base = 100
        co_combustion_effect = (1.0 - self._calculate_combustion_efficiency(fuel_props, kiln_conditions)) * 500
        co_emission = co_base + co_combustion_effect

        # Dust emissions (mg/Nm³ @ 10% O2)
        dust_emission = fuel_props.ash_content * 50  # Proportional to ash content

        # Heavy metal emissions (μg/Nm³ @ 10% O2)
        hg_emission = fuel_props.heavy_metals.get('Hg', 0) * 50  # 5% mercury release
        cd_emission = fuel_props.heavy_metals.get('Cd', 0) * 10  # 1% cadmium release
        pb_emission = fuel_props.heavy_metals.get('Pb', 0) * 5   # 0.5% lead release

        # CO2 emissions (kg/t clinker)
        carbon_content = fuel_props.ultimate_analysis.get('C', 50)
        co2_emission = carbon_content * 44.0 / 12.0 * 10  # Convert C to CO2

        return {
            'NOx': nox_emission,
            'SO2': so2_emission,
            'HCl': hcl_emission,
            'CO': co_emission,
            'Dust': dust_emission,
            'Hg': hg_emission,
            'Cd': cd_emission,
            'Pb': pb_emission,
            'CO2': co2_emission
        }

    def _calculate_energy_recovery(self, fuel_props: FuelProperties,
                                 fuel_blend: Dict[str, float],
                                 combustion_efficiency: float) -> Dict[str, float]:
        """Calculate energy recovery and utilization efficiency."""

        total_energy_input = fuel_props.calorific_value  # MJ/kg fuel
        useful_energy = total_energy_input * combustion_efficiency

        # Energy losses
        moisture_loss = fuel_props.moisture_content * 0.024  # MJ/kg per % moisture
        sensible_heat_loss = total_energy_input * 0.1  # 10% sensible heat loss

        net_energy_recovery = useful_energy - moisture_loss - sensible_heat_loss
        energy_efficiency = net_energy_recovery / total_energy_input

        return {
            'total_energy_input': total_energy_input,
            'useful_energy': useful_energy,
            'moisture_loss': moisture_loss,
            'sensible_heat_loss': sensible_heat_loss,
            'net_energy_recovery': net_energy_recovery,
            'energy_efficiency': energy_efficiency
        }

    def _calculate_ash_incorporation_effects(self, fuel_props: FuelProperties,
                                           production_rate: float) -> Dict[str, Any]:
        """Calculate effects of ash incorporation into clinker."""

        ash_rate = fuel_props.ash_content / 100  # Fraction

        # Ash composition effects on clinker chemistry
        ash_composition = fuel_props.ash_composition

        # Quality impact assessment
        quality_impact = {
            'strength_effect': -ash_rate * 2.0,  # 2% strength reduction per % ash
            'setting_time_effect': ash_rate * 5.0,  # Minutes delay per % ash
            'workability_effect': -ash_rate * 1.5  # Workability index reduction
        }

        # Material balance effect
        ash_input_rate = production_rate * ash_rate  # t/h ash input

        return {
            'ash_incorporation_rate': ash_rate,
            'ash_input_rate': ash_input_rate,
            'quality_impacts': quality_impact,
            'acceptable_level': ash_rate < 0.08,  # <8% ash generally acceptable
            'ash_composition_effects': ash_composition
        }

    def _calculate_economic_impact(self, fuel_blend: Dict[str, float],
                                 coal_properties: FuelProperties,
                                 energy_analysis: Dict[str, float]) -> Dict[str, float]:
        """Calculate economic impact of alternative fuel usage."""

        # Fuel cost comparison
        coal_cost = coal_properties.cost_per_ton

        total_alt_fuel_cost = 0
        total_alt_fuel_fraction = 0

        for fuel_name, fraction in fuel_blend.items():
            if fuel_name != 'coal':
                fuel_cost = self.fuel_database[fuel_name].cost_per_ton
                total_alt_fuel_cost += fraction * fuel_cost
                total_alt_fuel_fraction += fraction

        # Weighted average alternative fuel cost
        avg_alt_fuel_cost = total_alt_fuel_cost / total_alt_fuel_fraction if total_alt_fuel_fraction > 0 else 0

        # Cost savings per ton of fuel
        cost_savings_per_ton = coal_cost - avg_alt_fuel_cost

        # Energy-based cost savings
        energy_cost_savings = cost_savings_per_ton / fuel_blend.get('coal', 1.0) if fuel_blend.get('coal', 1.0) > 0 else 0

        return {
            'coal_cost_per_ton': coal_cost,
            'alt_fuel_cost_per_ton': avg_alt_fuel_cost,
            'cost_savings_per_ton': cost_savings_per_ton,
            'energy_cost_savings': energy_cost_savings,
            'annual_savings_potential': cost_savings_per_ton * 8760 * 50  # Assuming 50 t/h fuel consumption
        }

    def _assess_environmental_impact(self, emissions: Dict[str, float],
                                   ash_effects: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall environmental impact."""

        # Emission scoring (lower is better)
        emission_scores = {
            'NOx': min(1.0, emissions['NOx'] / 800),  # Normalized to typical coal value
            'SO2': min(1.0, emissions['SO2'] / 500),
            'HCl': min(1.0, emissions['HCl'] / 50),
            'CO': min(1.0, emissions['CO'] / 200),
            'Dust': min(1.0, emissions['Dust'] / 30),
            'Hg': min(1.0, emissions['Hg'] / 50),
            'CO2': min(1.0, emissions['CO2'] / 850)
        }

        overall_emission_score = sum(emission_scores.values()) / len(emission_scores)

        # Sustainability benefits
        sustainability_benefits = {
            'waste_diversion': 1.0,  # Diverts waste from landfill
            'resource_recovery': 0.8,  # Recovers energy from waste
            'carbon_neutrality': 0.5,  # Partial carbon neutrality for biomass
            'circular_economy': 0.9  # Supports circular economy principles
        }

        return {
            'emission_scores': emission_scores,
            'overall_emission_score': overall_emission_score,
            'environmental_rating': 'Good' if overall_emission_score < 0.8 else 'Acceptable' if overall_emission_score < 1.0 else 'Poor',
            'sustainability_benefits': sustainability_benefits,
            'recommendations': self._generate_environmental_recommendations(emissions, ash_effects)
        }

    def _generate_operational_recommendations(self, fuel_props: FuelProperties,
                                            combustion_efficiency: float,
                                            emissions: Dict[str, float]) -> List[str]:
        """Generate operational recommendations for alternative fuel usage."""

        recommendations = []

        if fuel_props.moisture_content > 25:
            recommendations.append("Consider fuel drying to improve combustion efficiency")

        if combustion_efficiency < 0.9:
            recommendations.append("Optimize air/fuel ratio and temperature control")

        if emissions['NOx'] > 800:
            recommendations.append("Consider staged combustion or SNCR for NOx control")

        if emissions['HCl'] > 100:
            recommendations.append("Monitor chlorine bypass and consider lime injection")

        if fuel_props.ash_content > 20:
            recommendations.append("Monitor clinker quality and adjust raw mix composition")

        if emissions['CO'] > 200:
            recommendations.append("Improve fuel mixing and increase combustion air")

        return recommendations

    def _generate_environmental_recommendations(self, emissions: Dict[str, float],
                                              ash_effects: Dict[str, Any]) -> List[str]:
        """Generate environmental management recommendations."""

        recommendations = []

        if emissions['Hg'] > 50:
            recommendations.append("Install mercury control system (ACI or similar)")

        if emissions['SO2'] > 400:
            recommendations.append("Consider limestone injection for SO2 control")

        if not ash_effects['acceptable_level']:
            recommendations.append("Reduce ash content in fuel blend or adjust raw mix")

        if emissions['Dust'] > 50:
            recommendations.append("Optimize ESP/bagfilter performance")

        recommendations.append("Implement continuous emission monitoring")
        recommendations.append("Regular stack testing for compliance verification")

        return recommendations

    def _estimate_flame_temperature(self, fuel_props: FuelProperties) -> float:
        """Estimate adiabatic flame temperature."""
        # Simplified calculation based on calorific value and stoichiometry
        base_temp = 1800  # °C for typical coal
        cv_effect = (fuel_props.calorific_value - 25) * 20  # °C per MJ/kg deviation
        moisture_effect = -fuel_props.moisture_content * 15  # °C reduction per % moisture

        flame_temp = base_temp + cv_effect + moisture_effect
        return max(1400, min(2200, flame_temp))

    def _calculate_air_requirement(self, fuel_props: FuelProperties) -> float:
        """Calculate theoretical air requirement for combustion."""
        # Simplified calculation based on ultimate analysis
        C = fuel_props.ultimate_analysis.get('C', 50) / 100
        H = fuel_props.ultimate_analysis.get('H', 6) / 100
        O = fuel_props.ultimate_analysis.get('O', 20) / 100
        S = fuel_props.sulfur_content / 100

        # Stoichiometric air requirement (kg air/kg fuel)
        air_req = 11.5 * C + 34.5 * H - 4.3 * O + 4.3 * S

        return max(4.0, air_req)  # Minimum 4 kg air/kg fuel

    def _create_heuristic_blend(self, fuel_names: List[str], 
                              target_substitution: float,
                              available_fuels: Dict[str, Dict]) -> Optional[Dict[str, float]]:
        """Create a simple heuristic fuel blend."""
        if not fuel_names:
            return None
        
        # Simple heuristic: use cheapest available fuel up to target substitution
        cheapest_fuel = None
        cheapest_cost = float('inf')
        
        for fuel_name in fuel_names:
            fuel_cost = available_fuels[fuel_name].get('cost', self.fuel_database[fuel_name].cost_per_ton)
            if fuel_cost < cheapest_cost:
                cheapest_cost = fuel_cost
                cheapest_fuel = fuel_name
        
        if cheapest_fuel:
            # Use cheapest fuel for target substitution
            max_fraction = available_fuels[cheapest_fuel].get('max_fraction', 0.3)
            fuel_fraction = min(target_substitution, max_fraction)
            coal_fraction = 1 - fuel_fraction
            
            return {
                cheapest_fuel: fuel_fraction,
                'coal': coal_fraction
            }
        
        return None
    
    def _calculate_heuristic_cost(self, fuel_blend: Dict[str, float],
                                coal_properties: FuelProperties) -> float:
        """Calculate cost for heuristic blend."""
        total_cost = 0
        total_energy = 0
        
        for fuel_name, fraction in fuel_blend.items():
            if fuel_name == 'coal':
                props = coal_properties
            else:
                props = self.fuel_database[fuel_name]
            
            total_cost += fraction * props.cost_per_ton
            total_energy += fraction * props.calorific_value
        
        return total_cost / total_energy if total_energy > 0 else 0

# Factory function
def create_alternative_fuel_processor() -> AlternativeFuelProcessor:
    """Create an alternative fuel processor instance."""
    return AlternativeFuelProcessor()
