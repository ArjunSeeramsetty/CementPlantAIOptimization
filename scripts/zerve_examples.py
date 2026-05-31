"""
Example scripts for using Zerve AI with Cement Plant AI Optimization Platform

These examples demonstrate how to use Zerve AI for various data science tasks
in the cement plant optimization context.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cement_ai_platform.integrations import ZerveClient, ZerveWorkflow, ZerveAgent, ZerveFleet


def example_1_simple_workflow():
    """
    Example 1: Create and execute a simple data pipeline workflow.
    """
    print("\n" + "="*80)
    print("Example 1: Simple Data Pipeline Workflow")
    print("="*80)
    
    # Initialize Zerve client
    client = ZerveClient.from_yaml('config/zerve_config.yml')
    
    # Create a workflow
    workflow = client.create_workflow(
        name="Sensor Data ETL",
        description="""
        Extract sensor data from BigQuery, clean and transform it,
        then load it back to a processed data table.
        """,
        agent_type="data_pipeline",
        compute_profile="standard"
    )
    
    print(f"Created workflow: {workflow.name} (ID: {workflow.workflow_id})")
    
    # Add processing blocks
    block1_id = workflow.add_block(
        block_type="extraction",
        code="""
from google.cloud import bigquery

client = bigquery.Client()
query = '''
SELECT * FROM `cement_plant_data.realtime_sensor_data`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
'''
df = client.query(query).to_dataframe()
print(f"Extracted {len(df)} rows")
        """,
        language="python"
    )
    
    block2_id = workflow.add_block(
        block_type="transformation",
        code="""
import pandas as pd
import numpy as np

# Remove outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"Cleaned data: {len(df_clean)} rows")
        """,
        language="python",
        dependencies=[block1_id]
    )
    
    # Execute workflow
    print("\nExecuting workflow...")
    results = workflow.execute()
    
    print(f"\nWorkflow execution completed!")
    print(f"Results: {results}")
    
    # Save workflow
    workflow_path = workflow.save()
    print(f"\nWorkflow saved to: {workflow_path}")
    
    return workflow


def example_2_ml_training():
    """
    Example 2: Train a machine learning model for quality prediction.
    """
    print("\n" + "="*80)
    print("Example 2: ML Training Workflow")
    print("="*80)
    
    from cement_ai_platform.integrations.zerve_integration import create_ml_training_workflow
    
    # Initialize client
    client = ZerveClient.from_yaml('config/zerve_config.yml')
    
    # Create ML training workflow
    workflow = create_ml_training_workflow(
        client=client,
        model_type="random_forest",
        training_data="cement_plant_data.quality_training_data",
        target_metric="clinker_quality"
    )
    
    print(f"Created ML training workflow: {workflow.name}")
    
    # Add custom training code
    workflow.add_block(
        block_type="model_training",
        code="""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib

# Load data
X = df[feature_columns]
y = df['clinker_quality']

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {scores}")
print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Save model
joblib.dump(model, 'models/quality_predictor_rf.pkl')
        """,
        language="python"
    )
    
    # Execute with GPU support
    print("\nStarting model training...")
    results = workflow.execute(
        inputs={'feature_columns': ['temperature', 'pressure', 'feed_rate']}
    )
    
    print(f"\nTraining completed! R² score: {results.get('r2_score', 'N/A')}")
    
    return workflow


def example_3_multi_agent_optimization():
    """
    Example 3: Use multiple agents for complex optimization task.
    """
    print("\n" + "="*80)
    print("Example 3: Multi-Agent Optimization")
    print("="*80)
    
    client = ZerveClient.from_yaml('config/zerve_config.yml')
    
    # Create multiple specialized agents
    agents = {}
    
    # Agent 1: Data analysis
    agents['analyzer'] = client.create_agent(
        agent_type="analysis",
        task="Analyze current plant operations and identify inefficiencies",
        context={'data_source': 'cement_plant_data.realtime_sensor_data'}
    )
    
    # Agent 2: Optimization
    agents['optimizer'] = client.create_agent(
        agent_type="optimization",
        task="Optimize kiln temperature and fan speed for energy efficiency",
        context={'constraints': {'min_quality': 95, 'max_temp': 1450}}
    )
    
    # Agent 3: Validation
    agents['validator'] = client.create_agent(
        agent_type="analysis",
        task="Validate optimization results against safety constraints",
        context={'safety_limits': {'temp_max': 1500, 'pressure_max': 5}}
    )
    
    print(f"Created {len(agents)} specialized agents")
    
    # Execute agents in sequence
    print("\nExecuting multi-agent workflow...")
    
    # Step 1: Analysis
    print("\n[Agent 1] Analyzing plant data...")
    analysis_result = agents['analyzer'].execute()
    print(f"Analysis complete: Found {analysis_result.get('num_issues', 0)} issues")
    
    # Step 2: Optimization
    print("\n[Agent 2] Running optimization...")
    optimization_result = agents['optimizer'].execute(
        context={'analysis': analysis_result}
    )
    print(f"Optimization complete: {optimization_result.get('improvement', 'N/A')}% improvement")
    
    # Step 3: Validation
    print("\n[Agent 3] Validating results...")
    validation_result = agents['validator'].execute(
        context={'optimization': optimization_result}
    )
    print(f"Validation: {'PASSED' if validation_result.get('valid') else 'FAILED'}")
    
    return agents, {
        'analysis': analysis_result,
        'optimization': optimization_result,
        'validation': validation_result
    }


def example_4_fleet_parallel_processing():
    """
    Example 4: Use Zerve Fleet for massively parallel batch processing.
    """
    print("\n" + "="*80)
    print("Example 4: Fleet Parallel Processing")
    print("="*80)
    
    client = ZerveClient.from_yaml('config/zerve_config.yml')
    
    # Create fleet with 10 workers
    fleet = client.create_fleet(
        name="batch_quality_analysis",
        num_workers=10
    )
    
    print(f"Created fleet: {fleet.name} with {fleet.num_workers} workers")
    
    # Prepare tasks for parallel execution
    # Each task analyzes a different batch of cement
    tasks = []
    for batch_id in range(1, 101):  # 100 batches
        task = {
            'batch_id': batch_id,
            'function': 'analyze_batch_quality',
            'params': {
                'query': f"SELECT * FROM quality_data WHERE batch_id = {batch_id}",
                'quality_threshold': 95
            }
        }
        tasks.append(task)
    
    print(f"\nSubmitting {len(tasks)} tasks to fleet...")
    
    # Submit all tasks
    batch_id = fleet.submit_tasks(tasks)
    print(f"Tasks submitted with batch ID: {batch_id}")
    
    # Wait for results
    print("Waiting for fleet execution...")
    results = fleet.get_results(batch_id, wait=True)
    
    # Process results
    successful = sum(1 for r in results if r.get('status') == 'success')
    failed = len(results) - successful
    
    print(f"\nFleet execution complete!")
    print(f"  Total tasks: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Calculate average quality
    avg_quality = sum(r.get('quality', 0) for r in results if r.get('status') == 'success') / successful
    print(f"  Average batch quality: {avg_quality:.2f}%")
    
    # Shutdown fleet
    fleet.shutdown()
    print("\nFleet shutdown complete")
    
    return results


def example_5_scheduled_workflow():
    """
    Example 5: Create a scheduled workflow for continuous monitoring.
    """
    print("\n" + "="*80)
    print("Example 5: Scheduled Workflow")
    print("="*80)
    
    from cement_ai_platform.integrations.zerve_integration import create_optimization_workflow
    
    client = ZerveClient.from_yaml('config/zerve_config.yml')
    
    # Create optimization workflow
    workflow = create_optimization_workflow(
        client=client,
        optimization_targets=['energy_efficiency', 'clinker_quality'],
        constraints={
            'min_quality': 95,
            'max_temp': 1450,
            'max_co2': 800
        }
    )
    
    print(f"Created workflow: {workflow.name}")
    
    # Schedule workflow to run every 4 hours
    schedule_id = workflow.schedule(
        cron_expression="0 */4 * * *",  # Every 4 hours
        timezone="UTC"
    )
    
    print(f"\nWorkflow scheduled successfully!")
    print(f"Schedule ID: {schedule_id}")
    print(f"Cron expression: 0 */4 * * * (every 4 hours)")
    print(f"Next execution: Check Zerve dashboard")
    
    return workflow, schedule_id


def example_6_integration_with_existing_code():
    """
    Example 6: Integrate Zerve with existing cement plant code.
    """
    print("\n" + "="*80)
    print("Example 6: Integration with Existing Code")
    print("="*80)
    
    client = ZerveClient.from_yaml('config/zerve_config.yml')
    
    # Create workflow that uses existing platform modules
    workflow = client.create_workflow(
        name="Integrated Platform Workflow",
        description="Use Zerve with existing cement platform modules",
        agent_type="data_pipeline",
        compute_profile="standard"
    )
    
    # Add block that imports and uses existing modules
    workflow.add_block(
        block_type="integration",
        code="""
# Import existing cement platform modules
from cement_ai_platform.models.pinn_model import PINNModel
from cement_ai_platform.optimization.optimizer import CementPlantOptimizer
from cement_ai_platform.data.bigquery_connector import BigQueryConnector

# Use existing BigQuery connector
connector = BigQueryConnector()
df = connector.fetch_realtime_data(hours=24)

# Use existing PINN model
pinn = PINNModel.load('models/pinn/quality_predictor.pt')
quality_predictions = pinn.predict(df)

# Use existing optimizer
optimizer = CementPlantOptimizer(
    objective='energy_efficiency',
    constraints={'min_quality': 95}
)
optimized_params = optimizer.optimize(df)

# Return results
results = {
    'data_points': len(df),
    'avg_predicted_quality': quality_predictions.mean(),
    'optimized_temperature': optimized_params['temperature'],
    'expected_savings': optimized_params['savings']
}
print(f"Integration complete: {results}")
        """,
        language="python"
    )
    
    print(f"Created integrated workflow: {workflow.name}")
    print("This workflow uses existing cement platform modules with Zerve orchestration")
    
    return workflow


def example_7_state_inspection():
    """
    Example 7: Use Zerve's state inspection for debugging.
    """
    print("\n" + "="*80)
    print("Example 7: State Inspection and Debugging")
    print("="*80)
    
    client = ZerveClient.from_yaml('config/zerve_config.yml')
    
    workflow = client.create_workflow(
        name="Debug Pipeline",
        description="Pipeline with state inspection enabled",
        agent_type="data_pipeline"
    )
    
    # Add blocks
    block1 = workflow.add_block(
        block_type="step1",
        code="import pandas as pd\ndf = pd.DataFrame({'temp': [1400, 1420, 1450]})\nprint(f'Step 1: {len(df)} rows')"
    )
    
    block2 = workflow.add_block(
        block_type="step2",
        code="df_filtered = df[df['temp'] > 1410]\nprint(f'Step 2: {len(df_filtered)} rows')",
        dependencies=[block1]
    )
    
    # Execute workflow
    workflow.execute()
    
    # Inspect state at each block
    print("\nInspecting workflow state:")
    state = workflow.get_state()
    
    for block_id in [block1, block2]:
        block_state = workflow.inspect_block(block_id)
        print(f"\nBlock {block_id}:")
        print(f"  Status: {block_state.get('status')}")
        print(f"  Execution time: {block_state.get('execution_time')}s")
        print(f"  Variables: {block_state.get('variables', {}).keys()}")
        print(f"  Output: {block_state.get('output', 'N/A')}")
    
    return workflow


def main():
    """
    Run all examples (or comment out ones you don't want to run).
    """
    print("\n" + "="*80)
    print("ZERVE AI INTEGRATION EXAMPLES")
    print("Cement Plant AI Optimization Platform")
    print("="*80)
    
    # Check if Zerve is configured
    if not os.getenv('ZERVE_API_KEY'):
        print("\n⚠️  WARNING: ZERVE_API_KEY not set!")
        print("Please configure your Zerve credentials first.")
        print("See: docs/ZERVE_INTEGRATION_GUIDE.md")
        return
    
    try:
        # Run examples
        print("\n🚀 Starting Zerve AI examples...\n")
        
        # Example 1: Simple workflow
        workflow1 = example_1_simple_workflow()
        
        # Example 2: ML training
        workflow2 = example_2_ml_training()
        
        # Example 3: Multi-agent
        agents, results = example_3_multi_agent_optimization()
        
        # Example 4: Fleet processing
        fleet_results = example_4_fleet_parallel_processing()
        
        # Example 5: Scheduled workflow
        workflow5, schedule_id = example_5_scheduled_workflow()
        
        # Example 6: Integration
        workflow6 = example_6_integration_with_existing_code()
        
        # Example 7: State inspection
        workflow7 = example_7_state_inspection()
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

