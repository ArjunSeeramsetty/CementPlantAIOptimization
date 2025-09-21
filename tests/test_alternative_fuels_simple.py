#!/usr/bin/env python3
"""
Simple test for alternative fuels system
"""

from cement_ai_platform.models.process.alternative_fuels import create_alternative_fuel_processor, FuelProperties

def test_simple_fuel_blend():
    """Test simple fuel blend optimization."""
    processor = create_alternative_fuel_processor()
    
    # Simple coal properties
    coal = FuelProperties(
        name='Coal',
        calorific_value=25.0,
        moisture_content=8.0,
        ash_content=12.0,
        volatile_matter=35.0,
        fixed_carbon=45.0,
        sulfur_content=1.0,
        chlorine_content=500,
        heavy_metals={'Hg': 0.1, 'Cd': 0.2, 'Pb': 5.0, 'Cr': 10.0},
        ultimate_analysis={'C': 70.0, 'H': 5.0, 'N': 1.5, 'O': 8.0},
        cost_per_ton=80.0
    )
    
    # Simple available fuels
    available_fuels = {
        'biomass': {'cost': 40.0, 'max_fraction': 0.2}
    }
    
    # Try optimization
    result = processor.optimize_fuel_blend(
        available_fuels=available_fuels,
        target_thermal_substitution=0.1,  # Low target
        coal_properties=coal
    )
    
    print(f"Optimization successful: {result.get('optimization_successful', False)}")
    if result.get('optimization_successful'):
        print(f"Optimal blend: {result.get('optimal_blend', {})}")
    else:
        print(f"Error: {result.get('error_message', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    test_simple_fuel_blend()
