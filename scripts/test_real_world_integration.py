#!/usr/bin/env python3
"""
Test Real-World Data Integration Workflow
Demonstrates the complete integration of real-world BigQuery datasets with synthetic data generation.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_sourcing.real_world_integrator import RealWorldDataIntegrator
from data_sourcing.bigquery_data_loader import BigQueryDataLoader
from training.quality_model_trainer import QualityModelTrainer
from simulation.dcs_simulator import CementPlantDCSSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_real_world_integration():
    """Test the complete real-world data integration workflow."""
    logger.info("🧪 Testing Real-World Data Integration Workflow")
    logger.info("=" * 60)
    
    try:
        # Step 1: Test BigQuery Data Loader
        logger.info("📊 Step 1: Testing BigQuery Data Loader...")
        data_loader = BigQueryDataLoader()
        mendeley_df, kaggle_df, global_df = data_loader.load_all()
        
        logger.info(f"✅ Mendeley LCI data: {len(mendeley_df)} records")
        logger.info(f"✅ Kaggle concrete strength: {len(kaggle_df)} records")
        logger.info(f"✅ Global cement database: {len(global_df)} records")
        
        # Step 2: Test Real-World Data Integrator
        logger.info("\n🏭 Step 2: Testing Real-World Data Integrator...")
        integrator = RealWorldDataIntegrator()
        
        # Test with config-based plant
        integration_results = integrator.integrate_real_world_data(use_config_plant=True)
        
        logger.info(f"✅ Selected plant: {integration_results['selected_plant']['plant_name']}")
        logger.info(f"✅ Calibrated parameters: {len(integration_results['calibrated_params'])} categories")
        logger.info(f"✅ Quality correlations: {len(integration_results['quality_correlations'])} correlations")
        
        # Step 3: Test Quality Model Training
        logger.info("\n🤖 Step 3: Testing Quality Model Training...")
        quality_trainer = QualityModelTrainer()
        
        # Load training data
        training_data = quality_trainer.load_training_data()
        logger.info(f"✅ Training data loaded: {len(training_data)} samples")
        
        # Train models
        training_results = quality_trainer.train_quality_models()
        logger.info(f"✅ Models trained: {len(training_results['training_results'])}")
        logger.info(f"✅ Best model: {training_results['best_model_name']}")
        
        # Step 4: Test DCS Simulator Calibration
        logger.info("\n🔧 Step 4: Testing DCS Simulator Calibration...")
        dcs_simulator = CementPlantDCSSimulator()
        
        # Generate calibrated data
        calibrated_data = dcs_simulator.generate_calibrated_data(
            integration_results['calibrated_params'],
            duration_hours=1,  # Short duration for testing
            sample_rate_seconds=60  # 1-minute intervals
        )
        
        logger.info(f"✅ Calibrated DCS data generated: {len(calibrated_data)} records")
        logger.info(f"✅ Data columns: {len(calibrated_data.columns)}")
        
        # Step 5: Test Quality Prediction
        logger.info("\n📊 Step 5: Testing Quality Prediction...")
        
        # Create sample features for prediction
        sample_features = pd.DataFrame({
            'C3S_content_pct': [55.0, 60.0, 50.0],
            'C2S_content_pct': [20.0, 18.0, 22.0],
            'C3A_content_pct': [8.0, 10.0, 6.0],
            'water_cement_ratio': [0.4, 0.35, 0.45],
            'fineness_blaine': [350.0, 380.0, 320.0],
            'free_lime_pct': [1.0, 0.8, 1.2],
            'C4AF_content_pct': [8.0, 7.5, 8.5],
            'SO3_content_pct': [2.5, 2.3, 2.7],
            'MgO_content_pct': [2.0, 1.8, 2.2],
            'curing_age_days': [28, 28, 28]
        })
        
        # Predict quality
        predictions = quality_trainer.predict_quality(sample_features)
        logger.info(f"✅ Quality predictions: {predictions}")
        
        # Step 6: Generate Comprehensive Summary
        logger.info("\n📋 Step 6: Generating Comprehensive Summary...")
        
        summary = {
            'test_timestamp': datetime.now().isoformat(),
            'bigquery_data': {
                'mendeley_records': len(mendeley_df),
                'kaggle_records': len(kaggle_df),
                'global_records': len(global_df)
            },
            'integration_results': {
                'selected_plant': integration_results['selected_plant']['plant_name'],
                'calibrated_params_count': len(integration_results['calibrated_params']),
                'quality_correlations_count': len(integration_results['quality_correlations']),
                'use_config_plant': integration_results['use_config_plant']
            },
            'quality_training': {
                'training_samples': len(training_data),
                'models_trained': len(training_results['training_results']),
                'best_model': training_results['best_model_name'],
                'best_r2_score': training_results['training_results'][training_results['best_model_name']]['test_r2']
            },
            'dcs_calibration': {
                'calibrated_data_records': len(calibrated_data),
                'calibrated_data_columns': len(calibrated_data.columns),
                'calibration_source': 'real_world_data'
            },
            'quality_predictions': {
                'sample_predictions': predictions.tolist(),
                'prediction_range': [float(np.min(predictions)), float(np.max(predictions))]
            }
        }
        
        # Save summary
        summary_file = Path('artifacts') / f"real_world_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Summary saved to: {summary_file}")
        
        # Final Results
        logger.info("\n🎉 REAL-WORLD DATA INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Total real-world records loaded: {len(mendeley_df) + len(kaggle_df) + len(global_df):,}")
        logger.info(f"🏭 Base plant: {integration_results['selected_plant']['plant_name']}")
        logger.info(f"🤖 Quality models trained: {len(training_results['training_results'])}")
        logger.info(f"🔧 DCS simulator calibrated: ✅")
        logger.info(f"📊 Calibrated data generated: {len(calibrated_data):,} records")
        logger.info(f"🎯 Quality predictions: {len(predictions)} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Real-world data integration test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def demonstrate_workflow():
    """Demonstrate the complete workflow with detailed output."""
    logger.info("🚀 DEMONSTRATING REAL-WORLD DATA INTEGRATION WORKFLOW")
    logger.info("=" * 70)
    
    # Test the workflow
    success = test_real_world_integration()
    
    if success:
        logger.info("\n✅ WORKFLOW DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info("🎯 Key Achievements:")
        logger.info("   • Real-world datasets loaded from BigQuery")
        logger.info("   • Plant configuration used as base plant")
        logger.info("   • DCS simulator calibrated with real KPIs")
        logger.info("   • Quality models trained on Kaggle data")
        logger.info("   • Synthetic data generation ready")
        logger.info("   • Complete integration pipeline established")
        
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("   1. Run comprehensive data generation workflow")
        logger.info("   2. Generate massive synthetic dataset (100K+ records)")
        logger.info("   3. Deploy to production environment")
        logger.info("   4. Set up automated retraining pipelines")
        
    else:
        logger.error("\n❌ WORKFLOW DEMONSTRATION FAILED!")
        logger.error("Please check the error messages above and fix any issues.")
    
    return success


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_workflow()
