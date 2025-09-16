"""
Demo Pipeline Orchestration Script
Top-level script to run the entire JK Cement demo pipeline in sequence
"""

import subprocess
import sys
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def run_demo_pipeline() -> Dict[str, Any]:
    """
    Run the complete JK Cement demo pipeline
    
    Returns:
        Pipeline execution results
    """
    logger.info("🚀 Starting JK Cement Demo Pipeline Execution")
    logger.info("="*60)
    
    # Define pipeline steps
    steps = [
        {
            'name': 'Load Real-World Data',
            'script': 'scripts/load_real_world_data.py',
            'description': 'Load Mendeley LCI, Kaggle, and Global Cement datasets from BigQuery'
        },
        {
            'name': 'DWSIM Physics Simulation',
            'script': 'scripts/run_dwsim_simulation.py',
            'description': 'Generate physics-based process simulation data'
        },
        {
            'name': 'TimeGAN Training',
            'script': 'scripts/train_timegan.py',
            'description': 'Train TimeGAN model and generate synthetic time-series data'
        },
        {
            'name': 'PINN Model Training',
            'script': 'scripts/train_pinn.py',
            'description': 'Train Physics-Informed Neural Network for quality prediction'
        },
        {
            'name': 'Demo Data Generation',
            'script': 'scripts/generate_demo_data.py',
            'description': 'Generate comprehensive demo dataset using unified platform'
        }
    ]
    
    # Initialize results
    results = {
        'pipeline_start_time': datetime.now().isoformat(),
        'steps_executed': [],
        'steps_failed': [],
        'total_execution_time': 0,
        'success': True
    }
    
    start_time = time.time()
    
    # Execute each step
    for i, step in enumerate(steps, 1):
        logger.info(f"\n🛠️ Step {i}/{len(steps)}: {step['name']}")
        logger.info(f"📝 Description: {step['description']}")
        logger.info(f"🔄 Running: {step['script']}")
        
        step_start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, step['script']], 
                capture_output=True, 
                text=True,
                timeout=1800  # 30 minutes timeout per step
            )
            
            step_execution_time = time.time() - step_start_time
            
            if result.returncode == 0:
                logger.info(f"✅ Step {i} completed successfully in {step_execution_time:.2f} seconds")
                logger.info(f"📊 Output: {result.stdout[-200:]}")  # Last 200 characters
                
                results['steps_executed'].append({
                    'step_number': i,
                    'name': step['name'],
                    'script': step['script'],
                    'execution_time': step_execution_time,
                    'status': 'success',
                    'output': result.stdout
                })
            else:
                logger.error(f"❌ Step {i} failed with return code {result.returncode}")
                logger.error(f"📊 Error: {result.stderr}")
                
                results['steps_failed'].append({
                    'step_number': i,
                    'name': step['name'],
                    'script': step['script'],
                    'execution_time': step_execution_time,
                    'status': 'failed',
                    'error': result.stderr,
                    'return_code': result.returncode
                })
                
                results['success'] = False
                break
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Step {i} timed out after 30 minutes")
            results['steps_failed'].append({
                'step_number': i,
                'name': step['name'],
                'script': step['script'],
                'execution_time': 1800,
                'status': 'timeout',
                'error': 'Script execution timed out'
            })
            results['success'] = False
            break
            
        except Exception as e:
            logger.error(f"❌ Step {i} failed with exception: {e}")
            results['steps_failed'].append({
                'step_number': i,
                'name': step['name'],
                'script': step['script'],
                'execution_time': time.time() - step_start_time,
                'status': 'exception',
                'error': str(e)
            })
            results['success'] = False
            break
    
    # Calculate total execution time
    results['total_execution_time'] = time.time() - start_time
    results['pipeline_end_time'] = datetime.now().isoformat()
    
    # Generate final report
    generate_pipeline_report(results)
    
    return results

def generate_pipeline_report(results: Dict[str, Any]):
    """
    Generate comprehensive pipeline execution report
    
    Args:
        results: Pipeline execution results
    """
    logger.info("\n" + "="*60)
    logger.info("📊 JK CEMENT DEMO PIPELINE EXECUTION REPORT")
    logger.info("="*60)
    
    # Overall status
    status = "✅ SUCCESS" if results['success'] else "❌ FAILED"
    logger.info(f"🎯 Overall Status: {status}")
    logger.info(f"⏱️ Total Execution Time: {results['total_execution_time']:.2f} seconds")
    logger.info(f"🕐 Start Time: {results['pipeline_start_time']}")
    logger.info(f"🕐 End Time: {results['pipeline_end_time']}")
    
    # Steps summary
    logger.info(f"\n📋 Steps Summary:")
    logger.info(f"   Total Steps: {len(results['steps_executed']) + len(results['steps_failed'])}")
    logger.info(f"   Successful: {len(results['steps_executed'])}")
    logger.info(f"   Failed: {len(results['steps_failed'])}")
    
    # Successful steps
    if results['steps_executed']:
        logger.info(f"\n✅ Successful Steps:")
        for step in results['steps_executed']:
            logger.info(f"   {step['step_number']}. {step['name']} ({step['execution_time']:.2f}s)")
    
    # Failed steps
    if results['steps_failed']:
        logger.info(f"\n❌ Failed Steps:")
        for step in results['steps_failed']:
            logger.info(f"   {step['step_number']}. {step['name']} - {step['status']}")
            logger.info(f"      Error: {step['error'][:100]}...")
    
    # Data generation summary
    if results['success']:
        logger.info(f"\n📊 Generated Demo Data:")
        demo_files = [
            "demo/data/real/process_variables.csv",
            "demo/data/real/quality_parameters.csv",
            "demo/data/physics/dwsim_physics.csv",
            "demo/data/physics/process_variables.csv",
            "demo/data/physics/quality_parameters.csv",
            "demo/data/synthetic/timegan_synthetic.csv",
            "demo/data/synthetic/process_variables.csv",
            "demo/data/synthetic/quality_parameters.csv",
            "demo/models/pinn/free_lime_pinn.pt",
            "demo/data/final/plant_demo_data_full.csv",
            "demo/data/final/process_variables.csv",
            "demo/data/final/quality_parameters.csv",
            "demo/data/final/optimization_results.csv",
            "demo/data/final/gpt_responses.csv"
        ]
        
        for file_path in demo_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"   ✅ {file_path} ({file_size:,} bytes)")
            else:
                logger.info(f"   ❌ {file_path} (not found)")
    
    # Recommendations
    logger.info(f"\n💡 Recommendations:")
    if results['success']:
        logger.info("   🎉 Demo pipeline completed successfully!")
        logger.info("   📈 Ready for JK Cement POC demonstration")
        logger.info("   🔍 Review generated data files in demo/data/ directory")
        logger.info("   🤖 Test PINN model in demo/models/pinn/ directory")
    else:
        logger.info("   🔧 Review failed steps and fix issues")
        logger.info("   🔄 Re-run pipeline after fixing problems")
        logger.info("   📞 Contact support if issues persist")

def check_prerequisites() -> bool:
    """
    Check if all prerequisites are met for pipeline execution
    
    Returns:
        True if all prerequisites are met
    """
    logger.info("🔍 Checking pipeline prerequisites...")
    
    prerequisites_met = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        prerequisites_met = False
    else:
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required directories
    required_dirs = ['src', 'config', 'scripts']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            logger.info(f"✅ Directory exists: {dir_name}")
        else:
            logger.error(f"❌ Required directory missing: {dir_name}")
            prerequisites_met = False
    
    # Check required files
    required_files = [
        'config/plant_config.yml',
        'src/data_sourcing/bigquery_data_loader.py',
        'src/simulation/dcs_simulator.py',
        'src/training/train_gan.py',
        'src/cement_ai_platform/models/agents/jk_cement_platform.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✅ Required file exists: {file_path}")
        else:
            logger.error(f"❌ Required file missing: {file_path}")
            prerequisites_met = False
    
    # Check environment variables
    env_vars = ['CEMENT_GCP_PROJECT']
    for env_var in env_vars:
        if os.getenv(env_var):
            logger.info(f"✅ Environment variable set: {env_var}")
        else:
            logger.warning(f"⚠️ Environment variable not set: {env_var}")
    
    if prerequisites_met:
        logger.info("✅ All prerequisites met!")
    else:
        logger.error("❌ Prerequisites not met. Please fix issues before running pipeline.")
    
    return prerequisites_met

def create_demo_summary(results: Dict[str, Any]):
    """
    Create a summary file with pipeline results
    
    Args:
        results: Pipeline execution results
    """
    logger.info("📝 Creating demo summary file...")
    
    summary_content = f"""
# JK Cement Demo Pipeline Summary

## Execution Results
- **Status**: {'SUCCESS' if results['success'] else 'FAILED'}
- **Total Execution Time**: {results['total_execution_time']:.2f} seconds
- **Start Time**: {results['pipeline_start_time']}
- **End Time**: {results['pipeline_end_time']}

## Steps Executed
"""
    
    for step in results['steps_executed']:
        summary_content += f"- ✅ {step['name']} ({step['execution_time']:.2f}s)\n"
    
    if results['steps_failed']:
        summary_content += "\n## Failed Steps\n"
        for step in results['steps_failed']:
            summary_content += f"- ❌ {step['name']} - {step['status']}\n"
    
    summary_content += f"""
## Generated Files
- Real-world data: demo/data/real/
- Physics simulation: demo/data/physics/
- Synthetic data: demo/data/synthetic/
- Trained models: demo/models/
- Final demo dataset: demo/data/final/

## Next Steps
1. Review generated data files
2. Test the trained PINN model
3. Prepare JK Cement POC demonstration
4. Customize parameters for specific use cases

## Support
For issues or questions, please refer to the documentation or contact the development team.
"""
    
    # Save summary file
    os.makedirs("demo", exist_ok=True)
    summary_path = "demo/pipeline_summary.md"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    logger.info(f"✅ Demo summary saved to {summary_path}")

def main():
    """Main function to run the demo pipeline"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🏭 JK CEMENT DIGITAL TWIN - DEMO PIPELINE ORCHESTRATOR")
    logger.info("="*70)
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("❌ Prerequisites not met. Exiting.")
            return False
        
        # Run demo pipeline
        results = run_demo_pipeline()
        
        # Create summary
        create_demo_summary(results)
        
        # Final status
        if results['success']:
            logger.info("\n🎉 DEMO PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("📈 Ready for JK Cement POC demonstration!")
            logger.info("📁 Check demo/ directory for generated files")
        else:
            logger.error("\n❌ DEMO PIPELINE FAILED!")
            logger.error("🔧 Please review failed steps and fix issues")
        
        return results['success']
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Pipeline execution interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
