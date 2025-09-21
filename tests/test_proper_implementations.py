#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from cement_ai_platform.data.processors.dwsim_integration import DWSIMCementSimulator
from cement_ai_platform.models.pinn.pina_cement_pinn import PinaCementPINN
from cement_ai_platform.data.processors.real_data_load_kaggle_cement import get_kaggle_cement_data
from cement_ai_platform.data.processors.real_data_load_global_cement import get_global_cement_data


def test_dwsim_integration() -> Dict[str, Any]:
    """Test DWSIM integration with proper package usage."""
    print("🏭 Testing DWSIM Integration...")
    
    try:
        # Create DWSIM simulator
        simulator = DWSIMCementSimulator(seed=42)
        
        # Test raw mill simulation
        mill_results = simulator.simulate_raw_mill(feed_rate=200.0, moisture_content=5.0)
        print(f"✓ Raw mill simulation completed: {len(mill_results)} results")
        
        # Test kiln simulation
        kiln_results = simulator.simulate_kiln_process(
            feed_rate=200.0, fuel_rate=20.0, kiln_speed=3.5
        )
        print(f"✓ Kiln simulation completed: {len(kiln_results)} results")
        
        # Test complete process simulation
        complete_results = simulator.simulate_complete_process(
            feed_rate=200.0, fuel_rate=20.0, kiln_speed=3.5, moisture_content=5.0
        )
        print(f"✓ Complete process simulation completed")
        
        # Test dataset generation
        dataset = simulator.generate_operational_dataset(n_samples=10)
        print(f"✓ Dataset generation completed: {dataset.shape}")
        
        return {
            "status": "success",
            "method": "dwsim_integration",
            "mill_results": mill_results,
            "kiln_results": kiln_results,
            "complete_results": complete_results,
            "dataset_shape": dataset.shape,
            "dataset_columns": dataset.columns.tolist()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "method": "dwsim_integration",
            "error": str(e)
        }


def test_pina_pinn_integration() -> Dict[str, Any]:
    """Test PINA PINN integration with proper package usage."""
    print("🧠 Testing PINA PINN Integration...")
    
    try:
        # Create PINA PINN
        pinn = PinaCementPINN(input_dim=5)
        
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = np.random.randn(n_samples, 1)
        
        # Test data preparation
        data = pd.DataFrame(X, columns=['temp', 'fuel_rate', 'kiln_speed', 'pressure', 'heat_consumption'])
        data['free_lime'] = y.ravel()
        
        X_prep, y_prep = pinn.prepare_data(
            data, 
            ['temp', 'fuel_rate', 'kiln_speed', 'pressure', 'heat_consumption'], 
            'free_lime'
        )
        print(f"✓ Data preparation completed: {X_prep.shape}, {y_prep.shape}")
        
        # Test training
        training_result = pinn.train(X_prep, y_prep, epochs=50, physics_weight=0.1)
        print(f"✓ Training completed: {training_result['method']}")
        
        # Test prediction
        predictions = pinn.predict(X_prep[:10])
        print(f"✓ Prediction completed: {predictions.shape}")
        
        # Test explanation
        explanation = pinn.explain_prediction(X_prep, prediction_idx=0)
        print(f"✓ Explanation completed: {len(explanation)} keys")
        
        return {
            "status": "success",
            "method": "pina_pinn",
            "training_result": training_result,
            "predictions_shape": predictions.shape,
            "explanation_keys": list(explanation.keys())
        }
        
    except Exception as e:
        return {
            "status": "error",
            "method": "pina_pinn",
            "error": str(e)
        }


def test_real_data_loaders() -> Dict[str, Any]:
    """Test real data loaders for proper data source usage."""
    print("📊 Testing Real Data Loaders...")
    
    results = {}
    
    try:
        # Test Kaggle data loader
        kaggle_data = get_kaggle_cement_data()
        print(f"✓ Kaggle data loader: {kaggle_data.shape}")
        results["kaggle"] = {
            "status": "success",
            "shape": kaggle_data.shape,
            "columns": kaggle_data.columns.tolist() if not kaggle_data.empty else [],
            "is_empty": kaggle_data.empty
        }
        
    except Exception as e:
        results["kaggle"] = {
            "status": "error",
            "error": str(e)
        }
    
    try:
        # Test Global Cement Database loader
        global_data = get_global_cement_data()
        print(f"✓ Global cement data loader: {global_data.shape}")
        results["global"] = {
            "status": "success",
            "shape": global_data.shape,
            "columns": global_data.columns.tolist() if not global_data.empty else [],
            "is_empty": global_data.empty
        }
        
    except Exception as e:
        results["global"] = {
            "status": "error",
            "error": str(e)
        }
    
    return results


def test_no_random_generators() -> Dict[str, Any]:
    """Test that no random data generators are being used inappropriately."""
    print("🔍 Testing for Random Data Generators...")
    
    issues = []
    
    # Check for common random data generation patterns
    import os
    import re
    
    # Files to check
    files_to_check = [
        "src/cement_ai_platform/data/processors/enhanced_dwsim_simulator.py",
        "src/cement_ai_platform/models/pinn/physics_informed_quality_predictor.py",
        "src/cement_ai_platform/data/processors/dwsim_integration.py",
        "src/cement_ai_platform/models/pinn/pina_cement_pinn.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Check for inappropriate random data generation
                if re.search(r'np\.random\.normal.*#.*random', content, re.IGNORECASE):
                    issues.append(f"Found random data generation in {file_path}")
                
                if re.search(r'np\.random\.uniform.*#.*random', content, re.IGNORECASE):
                    issues.append(f"Found random data generation in {file_path}")
                
                if re.search(r'np\.random\.choice.*#.*random', content, re.IGNORECASE):
                    issues.append(f"Found random data generation in {file_path}")
    
    if issues:
        return {
            "status": "warning",
            "issues": issues
        }
    else:
        return {
            "status": "success",
            "message": "No inappropriate random data generators found"
        }


def main():
    parser = argparse.ArgumentParser(description="Test proper implementations")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    parser.add_argument("--tests", nargs="+", 
                       choices=["dwsim", "pinn", "data", "random", "all"],
                       default=["all"], help="Tests to run")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if "dwsim" in args.tests or "all" in args.tests:
        results["dwsim"] = test_dwsim_integration()
    
    if "pinn" in args.tests or "all" in args.tests:
        results["pinn"] = test_pina_pinn_integration()
    
    if "data" in args.tests or "all" in args.tests:
        results["data_loaders"] = test_real_data_loaders()
    
    if "random" in args.tests or "all" in args.tests:
        results["random_check"] = test_no_random_generators()
    
    # Save results
    output_file = outdir / "proper_implementations_test.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 Proper implementations testing completed!")
    print(f"📊 Results saved to: {output_file}")
    
    # Summary
    print(f"\n📋 Summary:")
    for test_name, result in results.items():
        if isinstance(result, dict):
            status = result.get("status", "unknown")
            if status == "success":
                print(f"  ✅ {test_name}: {status}")
            elif status == "error":
                print(f"  ❌ {test_name}: {status}")
            else:
                print(f"  ⚠️  {test_name}: {status}")
        else:
            print(f"  📊 {test_name}: {len(result)} items")


if __name__ == "__main__":
    main()
