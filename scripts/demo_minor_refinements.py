#!/usr/bin/env python3
"""
Minor Refinements Demo
Demonstrates the minor improvements: MgO effect, volatile circulation enhancement, and pressure drop calculations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from cement_ai_platform.models.process.advanced_kiln_model import create_advanced_kiln_model
from cement_ai_platform.models.process.preheater_tower_model import create_preheater_tower


def demo_mgo_effect() -> Dict[str, Any]:
    """Demonstrate MgO flux effect in burnability calculations."""
    print("🔬 Demonstrating MgO Flux Effect...")
    
    try:
        # Create advanced kiln model
        kiln_model = create_advanced_kiln_model()
        
        # Test with different MgO contents
        mgo_contents = [1.0, 2.0, 3.0, 4.0, 5.0]
        burnability_results = []
        
        for mgo_content in mgo_contents:
            raw_meal_composition = {
                'SiO2': 22.0, 'CaO': 65.0, 'Al2O3': 5.0, 'Fe2O3': 3.0, 'MgO': mgo_content
            }
            
            burnability_result = kiln_model.calculate_enhanced_burnability_index(
                raw_meal_composition=raw_meal_composition,
                raw_meal_fineness=3200,
                alkali_content={'K2O': 0.5, 'Na2O': 0.2},
                coal_vm=35.0
            )
            
            burnability_results.append({
                'mgo_content': mgo_content,
                'burnability_index': burnability_result['burnability_index'],
                'mgo_effect': burnability_result['mgo_effect']
            })
        
        print(f"✓ MgO effect tested: {len(burnability_results)} scenarios")
        
        # Show the effect
        base_burnability = burnability_results[1]['burnability_index']  # At 2% MgO
        high_mgo_burnability = burnability_results[4]['burnability_index']  # At 5% MgO
        mgo_improvement = high_mgo_burnability - base_burnability
        
        print(f"✓ MgO flux effect: {mgo_improvement:.1f} burnability improvement at 5% MgO")
        
        return {
            "status": "success",
            "mgo_effect_analysis": burnability_results,
            "mgo_improvement": mgo_improvement,
            "base_burnability": base_burnability,
            "high_mgo_burnability": high_mgo_burnability
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_volatile_circulation_enhancement() -> Dict[str, Any]:
    """Demonstrate enhanced volatile circulation impact modeling."""
    print("🌪️ Demonstrating Volatile Circulation Enhancement...")
    
    try:
        # Create preheater tower model
        preheater = create_preheater_tower(num_stages=5)
        
        # Test with different volatile contents
        volatile_scenarios = [
            {'so3_content': 0.5, 'cl_content': 0.05, 'description': 'Low volatiles'},
            {'so3_content': 1.0, 'cl_content': 0.1, 'description': 'Medium volatiles'},
            {'so3_content': 2.0, 'cl_content': 0.2, 'description': 'High volatiles'}
        ]
        
        circulation_results = []
        
        for scenario in volatile_scenarios:
            kiln_conditions = {
                'temperature': 1450,
                'so3_content': scenario['so3_content'],
                'cl_content': scenario['cl_content']
            }
            
            alkali_analysis = preheater.calculate_alkali_circulation(
                raw_meal_alkali={'K2O': 0.5, 'Na2O': 0.2},
                kiln_conditions=kiln_conditions
            )
            
            circulation_results.append({
                'scenario': scenario['description'],
                'so3_content': scenario['so3_content'],
                'cl_content': scenario['cl_content'],
                'volatile_effect': alkali_analysis['volatile_effect'],
                'alkali_buildup_factor': alkali_analysis['alkali_buildup_factor'],
                'buildup_risk': alkali_analysis['buildup_risk']
            })
        
        print(f"✓ Volatile circulation tested: {len(circulation_results)} scenarios")
        
        # Show the enhancement effect
        low_volatiles = circulation_results[0]
        high_volatiles = circulation_results[2]
        volatile_impact = high_volatiles['volatile_effect'] - low_volatiles['volatile_effect']
        
        print(f"✓ Volatile circulation enhancement: {volatile_impact:.3f} additional buildup effect")
        
        return {
            "status": "success",
            "volatile_circulation_analysis": circulation_results,
            "volatile_impact": volatile_impact,
            "low_volatiles_buildup": low_volatiles['alkali_buildup_factor'],
            "high_volatiles_buildup": high_volatiles['alkali_buildup_factor']
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_pressure_drop_calculations() -> Dict[str, Any]:
    """Demonstrate Barth equation for cyclone pressure drop calculations."""
    print("📐 Demonstrating Barth Equation Pressure Drop Calculations...")
    
    try:
        # Create preheater tower model
        preheater = create_preheater_tower(num_stages=5)
        
        # Test Barth equation with different conditions
        test_scenarios = [
            {'gas_velocity': 15, 'dust_loading': 50, 'description': 'Low velocity, low dust'},
            {'gas_velocity': 20, 'dust_loading': 100, 'description': 'Medium velocity, medium dust'},
            {'gas_velocity': 25, 'dust_loading': 200, 'description': 'High velocity, high dust'},
            {'gas_velocity': 30, 'dust_loading': 300, 'description': 'Very high velocity, very high dust'}
        ]
        
        pressure_drop_results = []
        
        for scenario in test_scenarios:
            pressure_drop = preheater.calculate_cyclone_pressure_drop(
                gas_velocity=scenario['gas_velocity'],
                dust_loading=scenario['dust_loading']
            )
            
            pressure_drop_results.append({
                'scenario': scenario['description'],
                'gas_velocity': scenario['gas_velocity'],
                'dust_loading': scenario['dust_loading'],
                'pressure_drop': pressure_drop
            })
        
        print(f"✓ Barth equation tested: {len(pressure_drop_results)} scenarios")
        
        # Test stage-wise pressure drop calculations
        gas_velocities = [25, 22, 20, 18, 15]  # m/s
        dust_loadings = [200, 150, 100, 80, 60]  # g/Nm³
        
        stage_pressure_drops = preheater.calculate_stage_pressure_drops(
            gas_velocities=gas_velocities,
            dust_loadings=dust_loadings
        )
        
        total_pressure_drop = sum(stage_pressure_drops)
        
        print(f"✓ Stage-wise pressure drops: {total_pressure_drop:.0f} Pa total")
        
        return {
            "status": "success",
            "barth_equation_tests": pressure_drop_results,
            "stage_pressure_drops": stage_pressure_drops,
            "total_pressure_drop": total_pressure_drop,
            "gas_velocities": gas_velocities,
            "dust_loadings": dust_loadings
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_integrated_refinements() -> Dict[str, Any]:
    """Demonstrate all minor refinements working together."""
    print("🔧 Demonstrating Integrated Minor Refinements...")
    
    try:
        # Create models
        kiln_model = create_advanced_kiln_model()
        preheater = create_preheater_tower()
        
        # Integrated scenario with all refinements
        print("📊 Running integrated scenario with all refinements...")
        
        # 1. Kiln model with MgO effect
        raw_meal_composition = {
            'SiO2': 22.0, 'CaO': 65.0, 'Al2O3': 5.0, 'Fe2O3': 3.0, 'MgO': 3.5
        }
        
        burnability_result = kiln_model.calculate_enhanced_burnability_index(
            raw_meal_composition=raw_meal_composition,
            raw_meal_fineness=3200,
            alkali_content={'K2O': 0.5, 'Na2O': 0.2},
            coal_vm=35.0
        )
        
        # 2. Preheater with volatile circulation enhancement
        kiln_conditions = {
            'temperature': 1450,
            'so3_content': 1.5,
            'cl_content': 0.15
        }
        
        alkali_analysis = preheater.calculate_alkali_circulation(
            raw_meal_alkali={'K2O': 0.5, 'Na2O': 0.2},
            kiln_conditions=kiln_conditions
        )
        
        # 3. Pressure drop calculations
        gas_velocities = [25, 22, 20, 18, 15]
        dust_loadings = [200, 150, 100, 80, 60]
        stage_pressure_drops = preheater.calculate_stage_pressure_drops(
            gas_velocities=gas_velocities,
            dust_loadings=dust_loadings
        )
        
        # Integrated metrics
        integrated_metrics = {
            'burnability_with_mgo': burnability_result['burnability_index'],
            'mgo_effect': burnability_result['mgo_effect'],
            'volatile_circulation_effect': alkali_analysis['volatile_effect'],
            'alkali_buildup_factor': alkali_analysis['alkali_buildup_factor'],
            'total_pressure_drop': sum(stage_pressure_drops),
            'stage_pressure_drops': stage_pressure_drops
        }
        
        print(f"✓ Integrated refinements: {len(integrated_metrics)} metrics calculated")
        
        return {
            "status": "success",
            "burnability_result": burnability_result,
            "alkali_analysis": alkali_analysis,
            "stage_pressure_drops": stage_pressure_drops,
            "integrated_metrics": integrated_metrics
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Minor Refinements Demo")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    parser.add_argument("--demos", nargs="+", 
                       choices=["mgo", "volatile", "pressure", "integrated", "all"],
                       default=["all"], help="Demos to run")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if "mgo" in args.demos or "all" in args.demos:
        results["mgo_effect"] = demo_mgo_effect()
    
    if "volatile" in args.demos or "all" in args.demos:
        results["volatile_circulation"] = demo_volatile_circulation_enhancement()
    
    if "pressure" in args.demos or "all" in args.demos:
        results["pressure_drop_calculations"] = demo_pressure_drop_calculations()
    
    if "integrated" in args.demos or "all" in args.demos:
        results["integrated_refinements"] = demo_integrated_refinements()
    
    # Save results
    output_file = outdir / "minor_refinements_demo.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 Minor Refinements Demo completed!")
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
    
    # Key refinements
    print(f"\n🔧 Minor Refinements Implemented:")
    print(f"  🔬 MgO flux effect: Industrial correlation for burnability improvement")
    print(f"  🌪️ Volatile circulation enhancement: SO3/Cl impact on alkali buildup")
    print(f"  📐 Barth equation: Cyclone pressure drop calculations")
    print(f"  🔧 Integrated refinements: All improvements working together")
    print(f"  📊 Enhanced industrial correlations: More accurate process modeling")
    print(f"  🎯 Improved process fidelity: Better representation of real plant dynamics")


if __name__ == "__main__":
    main()
