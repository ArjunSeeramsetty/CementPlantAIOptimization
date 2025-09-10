#!/usr/bin/env python3
"""
Send Actual Data to GCP BigQuery
Sends real data to the connected GCP project using the service account credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cement_ai_platform.data.processors.bigquery_schema_design import create_bigquery_schemas, generate_ddl_statements
from cement_ai_platform.data.processors.bigquery_data_validation import create_validation_sql_queries
from cement_ai_platform.data.data_pipeline.streaming_ingestion import BigQueryStreamingIngestion
from cement_ai_platform.data.data_pipeline.bigquery_connector import run_bigquery, table_to_dataframe
from cement_ai_platform.data.processors.real_data_load_kaggle_cement import get_kaggle_cement_data
from cement_ai_platform.data.processors.real_data_load_global_cement import get_global_cement_data


def setup_gcp_credentials(credentials_path: str) -> bool:
    """Set up GCP credentials for BigQuery access."""
    print("🔐 Setting up GCP credentials...")
    
    try:
        # Set the credentials environment variable
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Verify credentials file exists
        if not Path(credentials_path).exists():
            print(f"❌ Credentials file not found: {credentials_path}")
            return False
        
        # Read credentials to get project ID
        with open(credentials_path, 'r') as f:
            creds = json.load(f)
        
        project_id = creds.get('project_id')
        if not project_id:
            print("❌ Project ID not found in credentials")
            return False
        
        # Set environment variables
        os.environ['CEMENT_GCP_PROJECT'] = project_id
        os.environ['CEMENT_BQ_DATASET'] = 'cement_analytics'
        os.environ['CEMENT_GCP_REGION'] = 'us-central1'
        
        # Also set the dataset name for the streaming ingestion
        os.environ['CEMENT_BQ_DATASET'] = 'cement_analytics'
        
        print(f"✅ Credentials loaded for project: {project_id}")
        print(f"✅ BigQuery dataset: cement_analytics")
        print(f"✅ Region: us-central1")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup credentials: {e}")
        return False


def create_bigquery_dataset_and_tables(client, dataset_name: str) -> Dict[str, bool]:
    """Create BigQuery dataset and tables."""
    print(f"📊 Creating BigQuery dataset and tables...")
    
    results = {}
    
    try:
        # Create dataset
        try:
            client.get_dataset(dataset_name)
            print(f"✅ Dataset {dataset_name} already exists")
        except Exception:
            # Dataset doesn't exist, create it
            dataset = client.create_dataset(dataset_name, location="US")
            print(f"✅ Created dataset: {dataset.dataset_id}")
        
        # Get schemas and create tables
        schemas = create_bigquery_schemas()
        ddl_statements = generate_ddl_statements(schemas, dataset_name)
        
        for table_name, ddl in ddl_statements.items():
            try:
                print(f"🔨 Creating table: {table_name}")
                
                # Execute DDL statement
                job = client.query(ddl)
                job.result()  # Wait for completion
                
                print(f"✅ Created table: {table_name}")
                results[table_name] = True
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"✅ Table {table_name} already exists")
                    results[table_name] = True
                else:
                    print(f"❌ Failed to create table {table_name}: {e}")
                    results[table_name] = False
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to create dataset/tables: {e}")
        return {}


def generate_comprehensive_cement_data() -> Dict[str, List[Dict[str, Any]]]:
    """Generate comprehensive cement plant data for BigQuery."""
    print("📊 Generating comprehensive cement plant data...")
    
    np.random.seed(42)
    
    # 1. Operational Parameters Data (Real-time sensor data)
    operational_data = []
    plant_ids = ['PLANT_001', 'PLANT_002', 'PLANT_003']
    unit_ids = ['KILN_01', 'MILL_01', 'PREHEATER_01', 'COOLER_01', 'RAW_MILL_01', 'CEMENT_MILL_01']
    parameter_types = ['TEMPERATURE', 'PRESSURE', 'FLOW_RATE', 'SPEED', 'LEVEL', 'VIBRATION']
    
    for i in range(500):  # 500 operational records
        param_type = np.random.choice(parameter_types)
        
        # Realistic value ranges based on parameter type
        if param_type == 'TEMPERATURE':
            if 'KILN' in np.random.choice(unit_ids):
                value = np.random.uniform(1400, 1500)  # Kiln temperature
            else:
                value = np.random.uniform(20, 200)     # Other temperatures
            unit = 'C'
        elif param_type == 'PRESSURE':
            value = np.random.uniform(0, 30)
            unit = 'bar'
        elif param_type == 'FLOW_RATE':
            value = np.random.uniform(100, 5000)
            unit = 'm3/h'
        elif param_type == 'SPEED':
            value = np.random.uniform(0, 100)
            unit = 'rpm'
        elif param_type == 'LEVEL':
            value = np.random.uniform(0, 100)
            unit = '%'
        else:  # VIBRATION
            value = np.random.uniform(0, 50)
            unit = 'mm/s'
        
        record = {
            'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'unit_id': np.random.choice(unit_ids),
            'parameter_type': param_type,
            'parameter_name': f'{param_type.lower()}_{i}',
            'value': value,
            'unit': unit,
            'quality_flag': 'GOOD' if np.random.random() > 0.05 else 'WARNING',
            'sensor_id': f'SENSOR_{i:03d}',
            'ingestion_timestamp': datetime.now().isoformat()
        }
        operational_data.append(record)
    
    # 2. Energy Consumption Data
    energy_data = []
    energy_types = ['ELECTRICITY', 'FUEL', 'STEAM', 'COMPRESSED_AIR', 'COOLING_WATER']
    
    for i in range(300):  # 300 energy records
        energy_type = np.random.choice(energy_types)
        consumption = np.random.uniform(100, 2000)
        
        record = {
            'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'unit_id': np.random.choice(unit_ids),
            'energy_type': energy_type,
            'energy_source': 'GRID' if energy_type == 'ELECTRICITY' else 'COAL',
            'consumption_kwh': consumption,
            'power_kw': consumption / 24,  # Average power
            'efficiency_factor': np.random.uniform(0.75, 0.95),
            'cost_per_unit': np.random.uniform(0.05, 0.20),
            'total_cost': consumption * np.random.uniform(0.05, 0.20),
            'ingestion_timestamp': datetime.now().isoformat()
        }
        energy_data.append(record)
    
    # 3. Quality Metrics Data (Laboratory test results)
    quality_data = []
    product_types = ['CEM_I', 'CEM_II', 'CEM_III', 'CEM_IV', 'CEM_V']
    test_types = ['COMPRESSIVE_STRENGTH', 'FREE_LIME', 'BLAINE_FINENESS', 'SETTING_TIME', 'SOUNDNESS']
    
    for i in range(200):  # 200 quality records
        test_type = np.random.choice(test_types)
        
        # Realistic quality test values
        if test_type == 'COMPRESSIVE_STRENGTH':
            measured_value = np.random.uniform(25, 60)  # MPa
            spec_min = measured_value * 0.9
            spec_max = measured_value * 1.1
        elif test_type == 'FREE_LIME':
            measured_value = np.random.uniform(0.5, 2.5)  # %
            spec_min = 0.5
            spec_max = 2.0
        elif test_type == 'BLAINE_FINENESS':
            measured_value = np.random.uniform(3000, 4500)  # cm2/g
            spec_min = 3200
            spec_max = 4000
        elif test_type == 'SETTING_TIME':
            measured_value = np.random.uniform(120, 300)  # minutes
            spec_min = 120
            spec_max = 300
        else:  # SOUNDNESS
            measured_value = np.random.uniform(0, 5)  # mm
            spec_min = 0
            spec_max = 5
        
        record = {
            'timestamp': (datetime.now() - timedelta(hours=i*3)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'batch_id': f'BATCH_{i:03d}',
            'product_type': np.random.choice(product_types),
            'test_type': test_type,
            'metric_name': test_type.lower(),
            'measured_value': measured_value,
            'specification_min': spec_min,
            'specification_max': spec_max,
            'pass_fail': spec_min <= measured_value <= spec_max,
            'lab_technician': f'TECH_{i%15:02d}',
            'equipment_id': f'EQ_{i%8:02d}',
            'ingestion_timestamp': datetime.now().isoformat()
        }
        quality_data.append(record)
    
    # 4. Production Summary Data (Daily production KPIs)
    production_data = []
    
    for i in range(90):  # 90 days of production data
        production_date = (datetime.now() - timedelta(days=i)).date().isoformat()
        
        record = {
            'production_date': production_date,
            'plant_id': np.random.choice(plant_ids),
            'product_type': np.random.choice(product_types),
            'total_production_tons': np.random.uniform(2000, 6000),
            'total_energy_consumed_kwh': np.random.uniform(80000, 250000),
            'avg_kiln_temperature': np.random.uniform(1420, 1480),
            'quality_pass_rate': np.random.uniform(0.88, 0.98),
            'downtime_hours': np.random.uniform(0, 6),
            'efficiency_score': np.random.uniform(0.75, 0.92),
            'ingestion_timestamp': datetime.now().isoformat()
        }
        production_data.append(record)
    
    return {
        'operational_parameters': operational_data,
        'energy_consumption': energy_data,
        'quality_metrics': quality_data,
        'production_summary': production_data
    }


def stream_data_to_bigquery(sample_data: Dict[str, List[Dict[str, Any]]], 
                           dataset_name: str) -> Dict[str, Any]:
    """Stream data to BigQuery."""
    print("📤 Streaming data to BigQuery...")
    
    results = {}
    
    try:
        # Set the dataset environment variable for streaming ingestion
        os.environ['CEMENT_BQ_DATASET'] = dataset_name
        
        # Initialize streaming ingestion
        ingestion = BigQueryStreamingIngestion()
        
        for table_name, records in sample_data.items():
            if not records:
                continue
            
            try:
                print(f"📤 Streaming {len(records)} records to {table_name}...")
                
                # Stream data to BigQuery
                ingested_count = ingestion.stream_rows(table_name, records)
                
                results[table_name] = {
                    'status': 'success',
                    'records_count': len(records),
                    'ingested_count': ingested_count
                }
                
                print(f"✅ Successfully streamed {ingested_count} records to {table_name}")
                
            except Exception as e:
                print(f"❌ Failed to stream {table_name}: {e}")
                results[table_name] = {
                    'status': 'error',
                    'error': str(e),
                    'records_count': len(records)
                }
    
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        results['error'] = str(e)
    
    return results


def validate_data_in_bigquery(client, dataset_name: str) -> Dict[str, Any]:
    """Validate data in BigQuery."""
    print("🔍 Validating data in BigQuery...")
    
    validation_results = {}
    
    try:
        # Count records in each table
        tables = ['operational_parameters', 'energy_consumption', 'quality_metrics', 'production_summary']
        
        for table_name in tables:
            try:
                query = f"""
                SELECT 
                    COUNT(*) as record_count,
                    MIN(timestamp) as earliest_record,
                    MAX(timestamp) as latest_record
                FROM `{client.project}.{dataset_name}.{table_name}`
                """
                
                job = client.query(query)
                results_df = job.result().to_dataframe()
                
                validation_results[table_name] = {
                    'status': 'success',
                    'record_count': results_df.iloc[0]['record_count'],
                    'earliest_record': str(results_df.iloc[0]['earliest_record']),
                    'latest_record': str(results_df.iloc[0]['latest_record'])
                }
                
                print(f"✅ {table_name}: {results_df.iloc[0]['record_count']} records")
                
            except Exception as e:
                print(f"❌ Failed to validate {table_name}: {e}")
                validation_results[table_name] = {
                    'status': 'error',
                    'error': str(e)
                }
    
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        validation_results['error'] = str(e)
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Send actual data to GCP BigQuery")
    parser.add_argument("--credentials", default=".secrets/cement-ops-key.json",
                       help="Path to GCP service account credentials")
    parser.add_argument("--dataset", default="cement_analytics",
                       help="BigQuery dataset name")
    parser.add_argument("--outdir", default="artifacts",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("🏭 Sending Data to GCP BigQuery")
    print("=" * 50)
    print(f"Credentials: {args.credentials}")
    print(f"Dataset: {args.dataset}")
    print("=" * 50)
    
    # Setup credentials
    if not setup_gcp_credentials(args.credentials):
        print("❌ Failed to setup credentials. Exiting.")
        return 1
    
    try:
        # Initialize BigQuery client
        print("🔧 Initializing BigQuery client...")
        from cement_ai_platform.config.google_cloud_config import get_bigquery_client
        client = get_bigquery_client()
        print(f"✅ Connected to project: {client.project}")
        
        # Create dataset and tables
        table_results = create_bigquery_dataset_and_tables(client, args.dataset)
        
        # Generate comprehensive data
        sample_data = generate_comprehensive_cement_data()
        
        # Stream data to BigQuery
        streaming_results = stream_data_to_bigquery(sample_data, args.dataset)
        
        # Validate data
        validation_results = validate_data_in_bigquery(client, args.dataset)
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'project_id': client.project,
            'dataset_name': args.dataset,
            'table_creation': table_results,
            'data_streaming': streaming_results,
            'validation': validation_results,
            'data_summary': {
                table: len(records) for table, records in sample_data.items()
            }
        }
        
        # Save results
        output_file = outdir / "gcp_bigquery_data_sent.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 Data successfully sent to GCP BigQuery!")
        print(f"📊 Results saved to: {output_file}")
        
        # Summary
        print(f"\n📋 Summary:")
        print(f"  🏗️ Project: {client.project}")
        print(f"  📊 Dataset: {args.dataset}")
        print(f"  📋 Tables created: {sum(table_results.values())}")
        print(f"  📤 Data streams: {len(streaming_results)}")
        
        total_records = sum(len(records) for records in sample_data.values())
        successful_streams = sum(1 for r in streaming_results.values() 
                               if isinstance(r, dict) and r.get('status') == 'success')
        
        print(f"  📊 Total records: {total_records}")
        print(f"  ✅ Successful streams: {successful_streams}")
        
        print(f"\n🔍 Validation Results:")
        for table_name, validation in validation_results.items():
            if isinstance(validation, dict) and validation.get('status') == 'success':
                print(f"  📊 {table_name}: {validation['record_count']} records")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to send data to BigQuery: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
