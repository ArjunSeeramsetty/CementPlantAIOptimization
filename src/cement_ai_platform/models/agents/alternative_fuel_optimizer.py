"""
Alternative Fuel Optimization Module for JK Cement Requirements
Maximizes TSR (Thermal Substitution Rate) by 10-15% while maintaining clinker quality
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FuelProperties:
    """Fuel properties for optimization calculations"""
    calorific_value: float  # kcal/kg
    moisture_content: float  # %
    ash_content: float  # %
    volatile_matter: float  # %
    fixed_carbon: float  # %
    sulfur_content: float  # %
    chlorine_content: float  # %
    grindability_index: float  # HGI
    combustion_efficiency: float  # 0-1

@dataclass
class QualityConstraints:
    """Quality constraints for clinker production"""
    min_c3s_content: float = 55.0  # %
    max_free_lime: float = 2.0  # %
    min_compressive_strength: float = 40.0  # MPa
    max_alkali_content: float = 0.6  # %
    max_chloride_content: float = 0.01  # %

class AlternativeFuelOptimizer:
    """
    Optimize alternative fuel blend (coal, petcoke, RDF, biomass)
    to maximize TSR while maintaining clinker quality targets.
    
    JK Cement Requirements:
    - Maximize TSR by 10-15%
    - Optimize RDF combustion with fossil fuels
    - Handle heterogeneous alternative fuel properties
    """

    def __init__(self, tsr_target: float = 0.15, quality_constraints: Optional[QualityConstraints] = None):
        self.fuel_types = ["coal", "petcoke", "rdf", "biomass", "tire_derived_fuel"]
        self.tsr_target = tsr_target
        self.quality_constraints = quality_constraints or QualityConstraints()
        
        # Fuel properties database (typical values)
        self.fuel_properties = {
            "coal": FuelProperties(
                calorific_value=6500, moisture_content=8.0, ash_content=12.0,
                volatile_matter=35.0, fixed_carbon=45.0, sulfur_content=0.8,
                chlorine_content=0.1, grindability_index=50.0, combustion_efficiency=0.95
            ),
            "petcoke": FuelProperties(
                calorific_value=8000, moisture_content=2.0, ash_content=1.0,
                volatile_matter=10.0, fixed_carbon=87.0, sulfur_content=3.0,
                chlorine_content=0.05, grindability_index=40.0, combustion_efficiency=0.98
            ),
            "rdf": FuelProperties(
                calorific_value=4000, moisture_content=15.0, ash_content=8.0,
                volatile_matter=60.0, fixed_carbon=17.0, sulfur_content=0.3,
                chlorine_content=0.5, grindability_index=30.0, combustion_efficiency=0.85
            ),
            "biomass": FuelProperties(
                calorific_value=3500, moisture_content=20.0, ash_content=3.0,
                volatile_matter=70.0, fixed_carbon=7.0, sulfur_content=0.1,
                chlorine_content=0.2, grindability_index=25.0, combustion_efficiency=0.80
            ),
            "tire_derived_fuel": FuelProperties(
                calorific_value=7500, moisture_content=1.0, ash_content=8.0,
                volatile_matter=60.0, fixed_carbon=31.0, sulfur_content=1.5,
                chlorine_content=0.1, grindability_index=35.0, combustion_efficiency=0.90
            )
        }

    def calculate_blend_properties(self, fuel_fractions: Dict[str, float]) -> FuelProperties:
        """Calculate weighted average properties of fuel blend"""
        total_fraction = sum(fuel_fractions.values())
        if total_fraction == 0:
            raise ValueError("Total fuel fraction cannot be zero")
        
        # Normalize fractions
        normalized_fractions = {k: v/total_fraction for k, v in fuel_fractions.items()}
        
        weighted_properties = {}
        properties = ['calorific_value', 'moisture_content', 'ash_content', 
                     'volatile_matter', 'fixed_carbon', 'sulfur_content', 
                     'chlorine_content', 'grindability_index', 'combustion_efficiency']
        
        for prop in properties:
            weighted_properties[prop] = sum(
                normalized_fractions[fuel] * getattr(self.fuel_properties[fuel], prop)
                for fuel in normalized_fractions.keys()
                if fuel in self.fuel_properties
            )
        
        return FuelProperties(**weighted_properties)

    def calculate_quality_impact(self, fuel_fractions: Dict[str, float], 
                                base_clinker_properties: Dict[str, float]) -> float:
        """Calculate quality penalty based on fuel blend impact on clinker"""
        blend_properties = self.calculate_blend_properties(fuel_fractions)
        
        # Quality impact factors
        quality_penalty = 0.0
        
        # Sulfur impact on C3S formation
        sulfur_impact = max(0, blend_properties.sulfur_content - 1.0) * 0.1
        
        # Chlorine impact on kiln operation
        chlorine_impact = max(0, blend_properties.chlorine_content - 0.3) * 0.2
        
        # Ash content impact on free lime
        ash_impact = max(0, blend_properties.ash_content - 10.0) * 0.05
        
        # Moisture impact on thermal efficiency
        moisture_impact = max(0, blend_properties.moisture_content - 10.0) * 0.03
        
        quality_penalty = sulfur_impact + chlorine_impact + ash_impact + moisture_impact
        
        return quality_penalty

    def objective_function(self, x: np.ndarray, available_fuels: Dict[str, float], 
                          base_clinker_properties: Dict[str, float]) -> float:
        """Objective function for optimization: maximize TSR while minimizing quality penalty"""
        
        # Convert array to fuel fractions
        fuel_fractions = {fuel: float(x[i]) for i, fuel in enumerate(self.fuel_types)}
        
        # Calculate TSR (Thermal Substitution Rate)
        alternative_fuels = ['rdf', 'biomass', 'tire_derived_fuel']
        tsr = sum(fuel_fractions[fuel] for fuel in alternative_fuels)
        
        # Calculate quality penalty
        quality_penalty = self.calculate_quality_impact(fuel_fractions, base_clinker_properties)
        
        # Objective: maximize TSR, minimize quality penalty
        # Negative because we're minimizing
        objective_value = -(tsr - self.tsr_target)**2 + quality_penalty * 10
        
        return objective_value

    def optimize_fuel_blend(self, 
                           available_fuels: Dict[str, float],
                           base_clinker_properties: Dict[str, float],
                           max_iterations: int = 1000) -> Dict[str, any]:
        """
        Optimize fuel blend to maximize TSR while maintaining quality
        
        Args:
            available_fuels: Available fuel quantities (tph)
            base_clinker_properties: Current clinker quality parameters
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization results including recommended blend and TSR
        """
        logger.info("🔄 Starting alternative fuel optimization...")
        
        # Check available fuels
        available_types = [fuel for fuel in self.fuel_types if fuel in available_fuels and available_fuels[fuel] > 0]
        if not available_types:
            raise ValueError("No alternative fuels available for optimization")
        
        # Initial guess: proportional to availability
        total_available = sum(available_fuels[fuel] for fuel in available_types)
        x0 = np.array([
            available_fuels[fuel]/total_available if fuel in available_types else 0.0
            for fuel in self.fuel_types
        ])
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: x.sum() - 1.0},  # Fractions sum to 1
        ]
        
        # Bounds: each fuel fraction between 0 and 1
        bounds = [(0.0, 1.0) for _ in self.fuel_types]
        
        # Additional constraints for fuel availability
        for i, fuel in enumerate(self.fuel_types):
            if fuel in available_fuels:
                max_fraction = min(1.0, available_fuels[fuel] / total_available)
                bounds[i] = (0.0, max_fraction)
        
        try:
            # Optimize
            result = minimize(
                self.objective_function,
                x0,
                args=(available_fuels, base_clinker_properties),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iterations}
            )
            
            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")
                # Fallback to initial guess
                optimal_fractions = {fuel: float(x0[i]) for i, fuel in enumerate(self.fuel_types)}
            else:
                optimal_fractions = {fuel: float(result.x[i]) for i, fuel in enumerate(self.fuel_types)}
            
            # Calculate results
            tsr = sum(optimal_fractions[fuel] for fuel in ['rdf', 'biomass', 'tire_derived_fuel'])
            blend_properties = self.calculate_blend_properties(optimal_fractions)
            quality_penalty = self.calculate_quality_impact(optimal_fractions, base_clinker_properties)
            
            # Calculate fuel consumption rates
            total_fuel_rate = sum(available_fuels.values())  # tph
            fuel_consumption = {
                fuel: optimal_fractions[fuel] * total_fuel_rate
                for fuel in self.fuel_types
            }
            
            results = {
                'optimal_fractions': optimal_fractions,
                'fuel_consumption_tph': fuel_consumption,
                'tsr_achieved': tsr,
                'tsr_target': self.tsr_target,
                'tsr_improvement_pct': (tsr - self.tsr_target) * 100,
                'blend_properties': blend_properties,
                'quality_penalty': quality_penalty,
                'optimization_success': result.success,
                'optimization_message': result.message
            }
            
            logger.info(f"✅ Fuel optimization completed - TSR: {tsr:.3f} ({tsr*100:.1f}%)")
            logger.info(f"📊 Quality penalty: {quality_penalty:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Fuel optimization failed: {e}")
            raise

    def get_fuel_recommendations(self, current_blend: Dict[str, float], 
                               target_tsr: float) -> Dict[str, any]:
        """Get recommendations for increasing TSR"""
        
        current_tsr = sum(current_blend.get(fuel, 0) for fuel in ['rdf', 'biomass', 'tire_derived_fuel'])
        tsr_gap = target_tsr - current_tsr
        
        if tsr_gap <= 0:
            return {
                'status': 'target_achieved',
                'current_tsr': current_tsr,
                'recommendations': []
            }
        
        recommendations = []
        
        # Recommend increasing alternative fuel fractions
        for fuel in ['rdf', 'biomass', 'tire_derived_fuel']:
            if fuel in current_blend:
                current_fraction = current_blend[fuel]
                recommended_fraction = min(1.0, current_fraction + tsr_gap * 0.5)
                increase = recommended_fraction - current_fraction
                
                if increase > 0.01:  # Only recommend if significant increase
                    recommendations.append({
                        'fuel_type': fuel,
                        'current_fraction': current_fraction,
                        'recommended_fraction': recommended_fraction,
                        'increase': increase,
                        'impact': 'positive_tsr'
                    })
        
        return {
            'status': 'optimization_needed',
            'current_tsr': current_tsr,
            'target_tsr': target_tsr,
            'tsr_gap': tsr_gap,
            'recommendations': recommendations
        }

    def validate_fuel_blend(self, fuel_fractions: Dict[str, float]) -> Dict[str, any]:
        """Validate fuel blend against operational constraints"""
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check fraction sum
        total_fraction = sum(fuel_fractions.values())
        if abs(total_fraction - 1.0) > 0.01:
            validation_results['errors'].append(f"Fuel fractions sum to {total_fraction:.3f}, should be 1.0")
            validation_results['is_valid'] = False
        
        # Check individual fuel constraints
        blend_properties = self.calculate_blend_properties(fuel_fractions)
        
        if blend_properties.sulfur_content > 2.0:
            validation_results['warnings'].append(f"High sulfur content: {blend_properties.sulfur_content:.2f}%")
        
        if blend_properties.chlorine_content > 0.5:
            validation_results['warnings'].append(f"High chlorine content: {blend_properties.chlorine_content:.2f}%")
        
        if blend_properties.moisture_content > 15.0:
            validation_results['warnings'].append(f"High moisture content: {blend_properties.moisture_content:.2f}%")
        
        return validation_results
