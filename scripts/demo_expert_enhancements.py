#!/usr/bin/env python3
"""
Expert-Recommended Enhancements Demo
Demonstrates industrial-grade digital twin capabilities with expert-recommended models.
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
from cement_ai_platform.models.process.plant_control_system import create_plant_control_system, create_process_control_simulator
from cement_ai_platform.models.process.industrial_quality_model import create_industrial_quality_predictor


def demo_advanced_kiln_model() -> Dict[str, Any]:
    """Demonstrate advanced kiln model with enhanced burnability and NOx calculations."""
    print("🔥 Demonstrating Advanced Kiln Model...")
    
    try:
        # Create advanced kiln model
        kiln_model = create_advanced_kiln_model()
        
        # Demo 1: Enhanced burnability index calculation
        raw_meal_composition = {
            'SiO2': 22.0, 'CaO': 65.0, 'Al2O3': 5.0, 'Fe2O3': 3.0
        }
        alkali_content = {'K2O': 0.5, 'Na2O': 0.2}
        
        burnability_result = kiln_model.calculate_enhanced_burnability_index(
            raw_meal_composition=raw_meal_composition,
            raw_meal_fineness=3200,
            alkali_content=alkali_content,
            coal_vm=35.0
        )
        print(f"✓ Enhanced burnability index: {burnability_result['burnability_index']:.1f}")
        
        # Demo 2: Comprehensive NOx formation model
        nox_result = kiln_model.calculate_comprehensive_nox_formation(
            flame_temp=1800,
            excess_air=1.5,
            fuel_nitrogen_percent=1.5,
            kiln_temperature=1450
        )
        print(f"✓ NOx formation: {nox_result['total_nox']:.0f} mg/Nm³")
        
        # Demo 3: Complete kiln dynamics simulation
        kiln_simulation = kiln_model.simulate_kiln_dynamics(
            feed_rate=200.0,
            fuel_rate=20.0,
            kiln_speed=3.5,
            raw_meal_properties={
                'composition': raw_meal_composition,
                'fineness': 3200,
                'alkali_content': alkali_content,
                'coal_vm': 35.0,
                'fuel_properties': {'calorific_value': 25.0, 'moisture_content': 8.0}
            }
        )
        print(f"✓ Kiln simulation: {len(kiln_simulation)} result categories")
        
        return {
            "status": "success",
            "burnability_analysis": burnability_result,
            "nox_formation": nox_result,
            "kiln_simulation": kiln_simulation
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_preheater_tower_model() -> Dict[str, Any]:
    """Demonstrate preheater tower model with heat exchange and alkali circulation."""
    print("🏗️ Demonstrating Preheater Tower Model...")
    
    try:
        # Create preheater tower model
        preheater = create_preheater_tower(num_stages=5)
        
        # Demo 1: Heat and mass balance
        heat_balance = preheater.calculate_heat_and_mass_balance(
            raw_meal_flow=200.0,
            gas_flow=150000,
            raw_meal_composition={'SiO2': 22.0, 'CaO': 65.0, 'Al2O3': 5.0, 'Fe2O3': 3.0},
            raw_meal_temp=20.0
        )
        print(f"✓ Heat balance: {heat_balance['final_meal_temp']:.0f}°C exit temperature")
        
        # Demo 2: Alkali circulation modeling
        alkali_analysis = preheater.calculate_alkali_circulation(
            raw_meal_alkali={'K2O': 0.5, 'Na2O': 0.2},
            kiln_conditions={'temperature': 1450}
        )
        print(f"✓ Alkali circulation: {alkali_analysis['buildup_risk']} risk")
        
        # Demo 3: Volatile cycles
        volatile_analysis = preheater.calculate_volatile_cycles(
            raw_meal_composition={'SO3': 1.0, 'Cl': 0.1},
            kiln_conditions={'temperature': 1450}
        )
        print(f"✓ Volatile cycles: SO2 potential {volatile_analysis['so2_emission_potential']:.1f}")
        
        # Demo 4: Complete preheater simulation
        preheater_simulation = preheater.simulate_preheater_performance(
            raw_meal_flow=200.0,
            gas_flow=150000,
            raw_meal_properties={
                'composition': {'SiO2': 22.0, 'CaO': 65.0, 'Al2O3': 5.0, 'Fe2O3': 3.0, 'SO3': 1.0, 'Cl': 0.1},
                'temperature': 20.0,
                'alkali_content': {'K2O': 0.5, 'Na2O': 0.2}
            },
            kiln_conditions={'temperature': 1450}
        )
        print(f"✓ Preheater simulation: {preheater_simulation['performance_metrics']['overall_performance_score']:.2f} score")
        
        return {
            "status": "success",
            "heat_balance": heat_balance,
            "alkali_analysis": alkali_analysis,
            "volatile_analysis": volatile_analysis,
            "preheater_simulation": preheater_simulation
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_plant_control_system() -> Dict[str, Any]:
    """Demonstrate plant control system with PI controllers and time delays."""
    print("🎛️ Demonstrating Plant Control System...")
    
    try:
        # Create plant control system
        controller = create_plant_control_system()
        simulator = create_process_control_simulator(controller)
        
        # Demo 1: Control loop status
        control_status = controller.get_control_status()
        print(f"✓ Control loops: {len(control_status)} active")
        
        # Demo 2: Control actions calculation
        measurements = {
            'free_lime': 1.4,
            'bzt': 1460,
            'draft': -2.2,
            'feed_rate': 195
        }
        
        control_actions = controller.get_control_actions(measurements, 0.0)
        print(f"✓ Control actions: {len(control_actions)} calculated")
        
        # Demo 3: Process control simulation
        simulation_results = []
        for cycle in range(10):
            result = simulator.simulate_control_cycle(dt_minutes=1.0)
            simulation_results.append(result)
        
        print(f"✓ Control simulation: {len(simulation_results)} cycles completed")
        
        # Demo 4: Disturbance handling
        disturbance_result = simulator.simulate_control_cycle(
            disturbances={'free_lime': 0.5, 'bzt': 50},
            dt_minutes=1.0
        )
        print(f"✓ Disturbance handling: {disturbance_result['control_actions']['fuel_rate_change']:.2f} fuel adjustment")
        
        return {
            "status": "success",
            "control_status": control_status,
            "control_actions": control_actions,
            "simulation_results": simulation_results[-3:],  # Last 3 cycles
            "disturbance_handling": disturbance_result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_industrial_quality_model() -> Dict[str, Any]:
    """Demonstrate industrial quality model with compressive strength prediction."""
    print("🏗️ Demonstrating Industrial Quality Model...")
    
    try:
        # Create quality predictor
        quality_predictor = create_industrial_quality_predictor()
        
        # Demo 1: Multi-age compressive strength prediction
        clinker_composition = {
            'C3S': 60.0, 'C2S': 20.0, 'C3A': 8.0, 'C4AF': 10.0
        }
        
        strength_predictions = {}
        for age in [1, 3, 7, 28]:
            strength_result = quality_predictor.predict_compressive_strength(
                clinker_composition=clinker_composition,
                fineness_blaine=3500,
                age_days=age,
                gypsum_content=0.7
            )
            strength_predictions[f'{age}_day'] = strength_result['predicted_strength_mpa']
        
        print(f"✓ Strength predictions: 28-day = {strength_predictions['28_day']:.1f} MPa")
        
        # Demo 2: Gypsum optimization
        gypsum_optimization = quality_predictor.optimize_gypsum_content(
            c3a_content=8.0,
            fineness_blaine=3500,
            target_setting_time_min=120
        )
        print(f"✓ Gypsum optimization: {gypsum_optimization['optimal_so3_percent']:.2f}% SO3")
        
        # Demo 3: Clinker quality analysis
        clinker_analysis = quality_predictor.analyze_clinker_quality(
            clinker_composition=clinker_composition,
            free_lime=1.2,
            kiln_conditions={'temperature': 1450}
        )
        print(f"✓ Clinker analysis: {clinker_analysis['quality_grade']} grade")
        
        # Demo 4: Comprehensive cement properties
        cement_properties = quality_predictor.predict_cement_properties(
            clinker_composition=clinker_composition,
            fineness_blaine=3500,
            gypsum_content=0.7,
            age_days=28
        )
        print(f"✓ Cement properties: {cement_properties['quality_grade']} quality")
        
        return {
            "status": "success",
            "strength_predictions": strength_predictions,
            "gypsum_optimization": gypsum_optimization,
            "clinker_analysis": clinker_analysis,
            "cement_properties": cement_properties
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def demo_integrated_expert_system() -> Dict[str, Any]:
    """Demonstrate integrated expert system with all components working together."""
    print("🏭 Demonstrating Integrated Expert System...")
    
    try:
        # Create all components
        kiln_model = create_advanced_kiln_model()
        preheater = create_preheater_tower()
        controller = create_plant_control_system()
        quality_predictor = create_industrial_quality_predictor()
        
        # Integrated simulation scenario
        print("📊 Running integrated simulation scenario...")
        
        # 1. Preheater simulation
        preheater_result = preheater.simulate_preheater_performance(
            raw_meal_flow=200.0,
            gas_flow=150000,
            raw_meal_properties={
                'composition': {'SiO2': 22.0, 'CaO': 65.0, 'Al2O3': 5.0, 'Fe2O3': 3.0},
                'temperature': 20.0,
                'alkali_content': {'K2O': 0.5, 'Na2O': 0.2}
            },
            kiln_conditions={'temperature': 1450}
        )
        
        # 2. Kiln simulation with preheater output
        kiln_result = kiln_model.simulate_kiln_dynamics(
            feed_rate=200.0,
            fuel_rate=20.0,
            kiln_speed=3.5,
            raw_meal_properties={
                'composition': {'SiO2': 22.0, 'CaO': 65.0, 'Al2O3': 5.0, 'Fe2O3': 3.0},
                'fineness': 3200,
                'alkali_content': {'K2O': 0.5, 'Na2O': 0.2},
                'coal_vm': 35.0,
                'fuel_properties': {'calorific_value': 25.0, 'moisture_content': 8.0}
            }
        )
        
        # 3. Control system response
        measurements = {
            'free_lime': kiln_result['clinker_quality']['free_lime'],
            'bzt': kiln_result['clinker_quality']['burning_zone_temp'],
            'draft': -2.0,
            'feed_rate': 200.0
        }
        
        control_actions = controller.get_control_actions(measurements, 0.0)
        
        # 4. Quality prediction
        quality_result = quality_predictor.predict_cement_properties(
            clinker_composition=kiln_result['clinker_quality'],
            fineness_blaine=3500,
            gypsum_content=0.7,
            age_days=28
        )
        
        # Integrated performance metrics
        integrated_metrics = {
            'preheater_efficiency': preheater_result['performance_metrics']['heat_recovery_efficiency'],
            'kiln_burnability': kiln_result['burnability_analysis']['burnability_index'],
            'nox_emissions': kiln_result['nox_formation']['total_nox'],
            'clinker_quality': kiln_result['clinker_quality']['free_lime'],
            'cement_strength': quality_result['strength_properties']['predicted_strength_mpa'],
            'control_response': len(control_actions)
        }
        
        print(f"✓ Integrated simulation: {len(integrated_metrics)} metrics calculated")
        
        return {
            "status": "success",
            "preheater_result": preheater_result,
            "kiln_result": kiln_result,
            "control_actions": control_actions,
            "quality_result": quality_result,
            "integrated_metrics": integrated_metrics
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Expert-Recommended Enhancements Demo")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    parser.add_argument("--demos", nargs="+", 
                       choices=["kiln", "preheater", "control", "quality", "integrated", "all"],
                       default=["all"], help="Demos to run")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if "kiln" in args.demos or "all" in args.demos:
        results["advanced_kiln_model"] = demo_advanced_kiln_model()
    
    if "preheater" in args.demos or "all" in args.demos:
        results["preheater_tower_model"] = demo_preheater_tower_model()
    
    if "control" in args.demos or "all" in args.demos:
        results["plant_control_system"] = demo_plant_control_system()
    
    if "quality" in args.demos or "all" in args.demos:
        results["industrial_quality_model"] = demo_industrial_quality_model()
    
    if "integrated" in args.demos or "all" in args.demos:
        results["integrated_expert_system"] = demo_integrated_expert_system()
    
    # Save results
    output_file = outdir / "expert_enhancements_demo.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 Expert-Recommended Enhancements Demo completed!")
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
    print(f"\n🚀 Expert-Recommended Achievements:")
    print(f"  🔥 Advanced burnability index with fineness, alkali, and coal volatile effects")
    print(f"  🏗️ Preheater tower model with heat exchange and alkali circulation")
    print(f"  🎛️ PI controller system with realistic time delays and process interactions")
    print(f"  🏗️ Industrial quality model with multi-age compressive strength prediction")
    print(f"  🏭 Integrated expert system with comprehensive process fidelity")
    print(f"  📊 Industrial-grade correlations based on decades of operational experience")
    print(f"  ⏱️ Realistic process dynamics with time delays and feedback loops")
    print(f"  🎯 Multi-objective optimization with quality, efficiency, and environmental goals")


if __name__ == "__main__":
    main()
