#!/usr/bin/env python3
"""
Comprehensive Cement Plant Process Expansion Demo
Demonstrates advanced grinding systems, alternative fuels, and unified process optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from cement_ai_platform.models.process.unified_process_platform import create_unified_process_platform
from cement_ai_platform.models.process.grinding_systems import create_grinding_circuit_simulator, MillConfiguration, SeparatorConfiguration
from cement_ai_platform.models.process.alternative_fuels import create_alternative_fuel_processor, FuelProperties


def demo_grinding_systems() -> Dict[str, Any]:
    """Demonstrate advanced grinding circuit simulation capabilities."""
    print("🔧 Demonstrating Advanced Grinding Systems...")
    
    try:
        # Create grinding circuit simulator
        simulator = create_grinding_circuit_simulator()
        
        # Demo 1: Bond Work Index calculation
        bond_energy = simulator.calculate_bond_work_energy(
            material='limestone',
            feed_f80=10000,  # μm
            product_p80=2000,  # μm
            correction_factors={'wet_grinding': True, 'open_circuit': False}
        )
        print(f"✓ Bond Work Index calculation: {bond_energy:.2f} kWh/t")
        
        # Demo 2: Circuit optimization
        optimization_result = simulator.optimize_grinding_circuit(
            target_fineness=3800,  # Blaine cm²/g
            production_rate=200,   # t/h
            energy_cost=0.08,     # $/kWh
            material_properties={'bond_work_index': 12.0}
        )
        print(f"✓ Circuit optimization: {optimization_result['optimization_successful']}")
        
        # Demo 3: Grinding aid effects
        aid_effects = simulator.simulate_grinding_aid_effects(
            base_energy=30.0,
            grinding_aid_type='triethanolamine',
            dosage_rate=0.8
        )
        print(f"✓ Grinding aid effects: {aid_effects['energy_reduction_percent']:.1f}% energy reduction")
        
        # Demo 4: PSD simulation
        feed_psd = np.array([100, 95, 85, 70, 50, 30, 15, 8, 4, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
        mill_config = MillConfiguration(
            mill_type='ball_mill',
            diameter=4.5,
            length=12.0,
            speed=0.75,
            ball_charge=30.0,
            power_rating=3000
        )
        
        product_psd = simulator.simulate_particle_size_distribution(
            feed_psd=feed_psd,
            energy_input=25.0,
            mill_config=mill_config,
            residence_time=1.5
        )
        print(f"✓ PSD simulation: {len(product_psd)} size classes processed")
        
        return {
            "status": "success",
            "bond_energy": bond_energy,
            "optimization_result": optimization_result,
            "grinding_aid_effects": aid_effects,
            "psd_simulation": {
                "feed_psd_shape": feed_psd.shape,
                "product_psd_shape": product_psd.shape
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_alternative_fuels() -> Dict[str, Any]:
    """Demonstrate alternative fuel processing and optimization capabilities."""
    print("🔥 Demonstrating Alternative Fuel Processing...")
    
    try:
        # Create alternative fuel processor
        processor = create_alternative_fuel_processor()
        
        # Demo 1: Fuel characterization
        print(f"✓ Fuel database: {len(processor.fuel_database)} fuel types available")
        
        # Demo 2: Fuel blend optimization
        available_fuels = {
            'refuse_derived_fuel': {'cost': 25.0, 'max_fraction': 0.3},
            'biomass': {'cost': 40.0, 'max_fraction': 0.2},
            'waste_oil': {'cost': 100.0, 'max_fraction': 0.1}
        }
        
        coal_properties = FuelProperties(
            name='Reference Coal',
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
        
        blend_optimization = processor.optimize_fuel_blend(
            available_fuels=available_fuels,
            target_thermal_substitution=0.25,
            coal_properties=coal_properties
        )
        print(f"✓ Fuel blend optimization: {blend_optimization['optimization_successful']}")
        
        # Demo 3: Co-processing simulation
        if blend_optimization['optimization_successful']:
            optimal_blend = blend_optimization['optimal_blend']
            kiln_conditions = {
                'temperature': 1450,
                'oxygen': 3.0,
                'pressure': 100
            }
            
            co_processing_results = processor.simulate_coprocessing_performance(
                fuel_blend=optimal_blend,
                coal_properties=coal_properties,
                kiln_conditions=kiln_conditions,
                production_rate=200
            )
            print(f"✓ Co-processing simulation: {co_processing_results['combustion_performance']['combustion_efficiency']:.3f} efficiency")
        
        return {
            "status": "success",
            "fuel_database_size": len(processor.fuel_database),
            "blend_optimization": blend_optimization,
            "co_processing_results": co_processing_results if blend_optimization['optimization_successful'] else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_unified_process_platform() -> Dict[str, Any]:
    """Demonstrate unified process platform with comprehensive plant simulation."""
    print("🏭 Demonstrating Unified Process Platform...")
    
    try:
        # Create unified platform
        platform = create_unified_process_platform(seed=42)
        
        # Plant configuration
        plant_config = {
            'raw_mill': {
                'mill_type': 'ball_mill',
                'diameter': 4.5,
                'length': 12.0,
                'power_rating': 3000
            },
            'fuel_system': {
                'max_thermal_substitution': 0.8,
                'environmental_limits': {
                    'max_chlorine_input': 0.15,
                    'max_mercury_input': 0.05
                }
            },
            'kiln': {
                'kiln_type': 'rotary',
                'diameter': 4.8,
                'length': 60.0,
                'capacity': 3000  # t/d
            },
            'cement_mill': {
                'mill_type': 'vertical_mill',
                'power_rating': 2500
            },
            'min_production_rate': 150,
            'max_energy_consumption': 1200,
            'quality_requirements': {'min_fineness': 3000}
        }
        
        # Operating conditions
        operating_conditions = {
            'raw_materials': {
                'target_fineness': 3800,
                'production_rate': 200,
                'energy_cost': 0.08,
                'bond_work_index': 12.0,
                'feed_f80': 10000,
                'target_p80': 2000,
                'grinding_aid': {
                    'type': 'triethanolamine',
                    'dosage': 0.8
                }
            },
            'fuel_conditions': {
                'coal_cv': 25.0,
                'coal_moisture': 8.0,
                'coal_ash': 12.0,
                'coal_cost': 80.0,
                'target_thermal_substitution': 0.25,
                'available_fuels': {
                    'refuse_derived_fuel': {'cost': 25.0, 'max_fraction': 0.3},
                    'biomass': {'cost': 40.0, 'max_fraction': 0.2},
                    'waste_oil': {'cost': 100.0, 'max_fraction': 0.1}
                },
                'kiln_conditions': {
                    'temperature': 1450,
                    'oxygen': 3.0,
                    'pressure': 100
                },
                'production_rate': 200
            },
            'kiln_conditions': {
                'feed_rate': 200,
                'fuel_rate': 20,
                'kiln_speed': 3.5,
                'moisture_content': 5.0
            },
            'cement_grinding': {
                'target_fineness': 3500,
                'production_rate': 150,
                'energy_cost': 0.08,
                'clinker_bwi': 13.5,
                'clinker_composition': {
                    'C3S': 60.0, 'C2S': 20.0, 'C3A': 8.0, 'C4AF': 10.0
                }
            }
        }
        
        # Run comprehensive plant simulation
        simulation_results = platform.simulate_complete_plant(
            plant_config=plant_config,
            operating_conditions=operating_conditions
        )
        
        print(f"✓ Plant simulation: {simulation_results['simulation_status']}")
        print(f"✓ Processes simulated: {simulation_results.get('total_processes_simulated', 0)}")
        
        # Extract key results
        key_results = {}
        if 'raw_material_processing' in simulation_results:
            key_results['raw_grinding'] = simulation_results['raw_material_processing'].get('grinding_optimization', {}).get('optimization_successful', False)
        
        if 'alternative_fuels' in simulation_results:
            key_results['fuel_optimization'] = simulation_results['alternative_fuels'].get('fuel_blend_optimization', {}).get('optimization_successful', False)
        
        if 'pyroprocessing' in simulation_results:
            key_results['kiln_simulation'] = 'dwsim_simulation' in simulation_results['pyroprocessing']
        
        if 'plant_optimization' in simulation_results:
            key_results['plant_optimization'] = simulation_results['plant_optimization'].get('optimization_score', 0)
        
        return {
            "status": "success",
            "simulation_status": simulation_results['simulation_status'],
            "processes_simulated": simulation_results.get('total_processes_simulated', 0),
            "key_results": key_results,
            "full_results": simulation_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_process_comparison() -> Dict[str, Any]:
    """Demonstrate process comparison and optimization scenarios."""
    print("📊 Demonstrating Process Comparison...")
    
    try:
        platform = create_unified_process_platform(seed=42)
        
        # Scenario 1: Traditional operation (coal only)
        traditional_config = {
            'raw_mill': {'mill_type': 'ball_mill', 'power_rating': 3000},
            'fuel_system': {'max_thermal_substitution': 0.0},
            'kiln': {'kiln_type': 'rotary', 'capacity': 3000},
            'cement_mill': {'mill_type': 'ball_mill', 'power_rating': 2500}
        }
        
        traditional_conditions = {
            'raw_materials': {'target_fineness': 3500, 'production_rate': 200, 'energy_cost': 0.08},
            'fuel_conditions': {
                'coal_cv': 25.0, 'coal_cost': 80.0, 'target_thermal_substitution': 0.0,
                'available_fuels': {}, 'kiln_conditions': {'temperature': 1450}
            },
            'kiln_conditions': {'feed_rate': 200, 'fuel_rate': 20, 'kiln_speed': 3.5},
            'cement_grinding': {'target_fineness': 3500, 'production_rate': 150, 'energy_cost': 0.08}
        }
        
        # Scenario 2: Optimized operation (alternative fuels + grinding aids)
        optimized_config = {
            'raw_mill': {'mill_type': 'vertical_mill', 'power_rating': 2500},
            'fuel_system': {'max_thermal_substitution': 0.3},
            'kiln': {'kiln_type': 'rotary', 'capacity': 3000},
            'cement_mill': {'mill_type': 'vertical_mill', 'power_rating': 2000}
        }
        
        optimized_conditions = {
            'raw_materials': {
                'target_fineness': 3800, 'production_rate': 220, 'energy_cost': 0.08,
                'grinding_aid': {'type': 'triethanolamine', 'dosage': 0.8}
            },
            'fuel_conditions': {
                'coal_cv': 25.0, 'coal_cost': 80.0, 'target_thermal_substitution': 0.25,
                'available_fuels': {
                    'refuse_derived_fuel': {'cost': 25.0, 'max_fraction': 0.3},
                    'biomass': {'cost': 40.0, 'max_fraction': 0.2}
                },
                'kiln_conditions': {'temperature': 1450}
            },
            'kiln_conditions': {'feed_rate': 220, 'fuel_rate': 18, 'kiln_speed': 3.8},
            'cement_grinding': {'target_fineness': 3800, 'production_rate': 180, 'energy_cost': 0.08}
        }
        
        # Run both scenarios
        traditional_results = platform.simulate_complete_plant(traditional_config, traditional_conditions)
        optimized_results = platform.simulate_complete_plant(optimized_config, optimized_conditions)
        
        # Compare results
        comparison = {
            'traditional': {
                'status': traditional_results.get('simulation_status', 'unknown'),
                'processes': traditional_results.get('total_processes_simulated', 0)
            },
            'optimized': {
                'status': optimized_results.get('simulation_status', 'unknown'),
                'processes': optimized_results.get('total_processes_simulated', 0)
            }
        }
        
        print(f"✓ Traditional scenario: {comparison['traditional']['status']}")
        print(f"✓ Optimized scenario: {comparison['optimized']['status']}")
        
        return {
            "status": "success",
            "comparison": comparison,
            "traditional_results": traditional_results,
            "optimized_results": optimized_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Process Expansion Demo")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    parser.add_argument("--demos", nargs="+", 
                       choices=["grinding", "fuels", "unified", "comparison", "all"],
                       default=["all"], help="Demos to run")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if "grinding" in args.demos or "all" in args.demos:
        results["grinding_systems"] = demo_grinding_systems()
    
    if "fuels" in args.demos or "all" in args.demos:
        results["alternative_fuels"] = demo_alternative_fuels()
    
    if "unified" in args.demos or "all" in args.demos:
        results["unified_platform"] = demo_unified_process_platform()
    
    if "comparison" in args.demos or "all" in args.demos:
        results["process_comparison"] = demo_process_comparison()
    
    # Save results
    output_file = outdir / "comprehensive_process_expansion_demo.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 Comprehensive Process Expansion Demo completed!")
    print(f"📊 Results saved to: {output_file}")
    
    # Summary
    print(f"\n📋 Summary:")
    for demo_name, result in results.items():
        if isinstance(result, dict):
            status = result.get("status", "unknown")
            if status == "success":
                print(f"  ✅ {demo_name}: {status}")
            elif status == "error":
                print(f"  ❌ {demo_name}: {status}")
            else:
                print(f"  ⚠️  {demo_name}: {status}")
        else:
            print(f"  📊 {demo_name}: {len(result)} items")
    
    # Key achievements
    print(f"\n🚀 Key Achievements:")
    print(f"  🔧 Advanced grinding circuit simulation with PSD modeling")
    print(f"  🔥 Alternative fuel processing with blending optimization")
    print(f"  🏭 Unified process platform for comprehensive plant simulation")
    print(f"  📊 Multi-scenario comparison and optimization capabilities")
    print(f"  🌍 Environmental impact assessment and compliance checking")
    print(f"  💰 Economic analysis and cost optimization potential")


if __name__ == "__main__":
    main()
