#!/usr/bin/env python3
"""
BigQuery Environment Setup Script
Sets up BigQuery environment using existing code blocks from the project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cement_ai_platform.data.processors.bigquery_schema_design import create_bigquery_schemas, generate_ddl_statements
from cement_ai_platform.data.processors.bigquery_data_validation import create_validation_sql_queries
from cement_ai_platform.data.data_pipeline.streaming_ingestion import BigQueryStreamingIngestion
from cement_ai_platform.data.data_pipeline.bigquery_connector import run_bigquery, table_to_dataframe
from cement_ai_platform.config.google_cloud_config import get_bigquery_client
from cement_ai_platform.config.settings import get_settings


def setup_environment_variables() -> Dict[str, str]:
    """Set up environment variables for BigQuery."""
    print("🔧 Setting up environment variables...")
    
    # Default environment variables
    env_vars = {
        'CEMENT_BQ_DATASET': 'cement_analytics_dev',
        'CEMENT_GCP_PROJECT': 'cement-ai-platform-dev',
        'CEMENT_GCP_REGION': 'us-central1',
        'CEMENT_LOG_LEVEL': 'INFO'
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")
    
    return env_vars


def create_bigquery_dataset(client, dataset_name: str, location: str = "US") -> bool:
    """Create BigQuery dataset if it doesn't exist."""
    print(f"📊 Creating BigQuery dataset: {dataset_name}")
    
    try:
        # Check if dataset exists
        try:
            client.get_dataset(dataset_name)
            print(f"✅ Dataset {dataset_name} already exists")
            return True
        except Exception:
            # Dataset doesn't exist, create it
            pass
        
        # Create dataset
        dataset = client.create_dataset(dataset_name, location=location)
        print(f"✅ Created dataset: {dataset.dataset_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False


def create_bigquery_tables(client, dataset_name: str) -> Dict[str, bool]:
    """Create BigQuery tables using existing schema design."""
    print("📋 Creating BigQuery tables using existing schema design...")
    
    # Get schemas from existing code
    schemas = create_bigquery_schemas()
    ddl_statements = generate_ddl_statements(schemas, dataset_name)
    
    results = {}
    
    for table_name, ddl in ddl_statements.items():
        try:
            print(f"🔨 Creating table: {table_name}")
            
            # Execute DDL statement
            job = client.query(ddl)
            job.result()  # Wait for completion
            
            print(f"✅ Created table: {table_name}")
            results[table_name] = True
            
        except Exception as e:
            print(f"❌ Failed to create table {table_name}: {e}")
            results[table_name] = False
    
    return results


def generate_sample_data() -> Dict[str, List[Dict[str, Any]]]:
    """Generate sample data for testing BigQuery streaming."""
    print("📊 Generating sample data for BigQuery streaming...")
    
    np.random.seed(42)
    
    # Sample operational parameters data
    operational_data = []
    plant_ids = ['PLANT_001', 'PLANT_002', 'PLANT_003']
    unit_ids = ['KILN_01', 'MILL_01', 'PREHEATER_01', 'COOLER_01']
    parameter_types = ['TEMPERATURE', 'PRESSURE', 'FLOW_RATE', 'SPEED']
    
    for i in range(100):
        record = {
            'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'unit_id': np.random.choice(unit_ids),
            'parameter_type': np.random.choice(parameter_types),
            'parameter_name': f'param_{i}',
            'value': np.random.uniform(0, 1000),
            'unit': 'C' if np.random.choice(parameter_types) == 'TEMPERATURE' else 'bar',
            'quality_flag': 'GOOD' if np.random.random() > 0.1 else 'WARNING',
            'sensor_id': f'SENSOR_{i:03d}',
            'ingestion_timestamp': datetime.now().isoformat()
        }
        operational_data.append(record)
    
    # Sample energy consumption data
    energy_data = []
    for i in range(50):
        record = {
            'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'unit_id': np.random.choice(unit_ids),
            'consumption_kwh': np.random.uniform(100, 1000),
            'demand_kw': np.random.uniform(50, 500),
            'power_factor': np.random.uniform(0.8, 1.0),
            'voltage_v': np.random.uniform(400, 600),
            'current_a': np.random.uniform(100, 1000),
            'frequency_hz': np.random.uniform(49.5, 50.5),
            'quality_flag': 'GOOD' if np.random.random() > 0.05 else 'WARNING',
            'meter_id': f'METER_{i:03d}',
            'ingestion_timestamp': datetime.now().isoformat()
        }
        energy_data.append(record)
    
    # Sample quality metrics data
    quality_data = []
    quality_types = ['COMPRESSIVE_STRENGTH', 'FREE_LIME', 'BLAINE_FINENESS', 'SETTING_TIME']
    
    for i in range(75):
        record = {
            'timestamp': (datetime.now() - timedelta(hours=i*2)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'unit_id': 'QUALITY_LAB',
            'quality_type': np.random.choice(quality_types),
            'measured_value': np.random.uniform(10, 100),
            'target_value': np.random.uniform(15, 90),
            'tolerance_min': np.random.uniform(5, 20),
            'tolerance_max': np.random.uniform(80, 95),
            'unit': 'MPa' if 'STRENGTH' in np.random.choice(quality_types) else 'min',
            'test_method': 'ASTM_C39' if 'STRENGTH' in np.random.choice(quality_types) else 'ASTM_C191',
            'sample_id': f'SAMPLE_{i:03d}',
            'technician_id': f'TECH_{i%10:02d}',
            'quality_flag': 'GOOD' if np.random.random() > 0.08 else 'WARNING',
            'ingestion_timestamp': datetime.now().isoformat()
        }
        quality_data.append(record)
    
    return {
        'operational_parameters': operational_data,
        'energy_consumption': energy_data,
        'quality_metrics': quality_data
    }


def stream_data_to_bigquery(sample_data: Dict[str, List[Dict[str, Any]]], 
                           dataset_name: str, dry_run: bool = False) -> Dict[str, Any]:
    """Stream sample data to BigQuery using existing streaming ingestion."""
    print("📤 Streaming data to BigQuery...")
    
    results = {}
    
    try:
        # Initialize streaming ingestion
        ingestion = BigQueryStreamingIngestion()
        
        for table_name, records in sample_data.items():
            if not records:
                continue
            
            if dry_run:
                print(f"🔍 DRY RUN: Would stream {len(records)} records to {table_name}")
                results[table_name] = {
                    'status': 'dry_run',
                    'records_count': len(records)
                }
            else:
                try:
                    # Stream data to BigQuery
                    ingested_count = ingestion.stream_rows(table_name, records)
                    
                    results[table_name] = {
                        'status': 'success',
                        'records_count': len(records),
                        'ingested_count': ingested_count
                    }
                    
                    print(f"✅ Streamed {ingested_count} records to {table_name}")
                    
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


def validate_bigquery_data(client, dataset_name: str) -> Dict[str, Any]:
    """Validate BigQuery data using existing validation queries."""
    print("🔍 Validating BigQuery data...")
    
    try:
        # Get validation queries from existing code
        validation_queries = create_validation_sql_queries()
        
        validation_results = {}
        
        for query_name, query_template in validation_queries.items():
            try:
                # Format query with project and dataset
                query = query_template.format(
                    project_id=client.project,
                    dataset_name=dataset_name
                )
                
                # Execute validation query
                job = client.query(query)
                results_df = job.result().to_dataframe()
                
                validation_results[query_name] = {
                    'status': 'success',
                    'results': results_df.to_dict('records')
                }
                
                print(f"✅ Validation {query_name} completed")
                
            except Exception as e:
                print(f"❌ Validation {query_name} failed: {e}")
                validation_results[query_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return {'error': str(e)}


def test_bigquery_queries(client, dataset_name: str) -> Dict[str, Any]:
    """Test BigQuery queries using existing connector."""
    print("🔍 Testing BigQuery queries...")
    
    test_queries = {
        'count_records': f"""
            SELECT 
                'operational_parameters' as table_name,
                COUNT(*) as record_count
            FROM `{client.project}.{dataset_name}.operational_parameters`
            
            UNION ALL
            
            SELECT 
                'energy_consumption' as table_name,
                COUNT(*) as record_count
            FROM `{client.project}.{dataset_name}.energy_consumption`
            
            UNION ALL
            
            SELECT 
                'quality_metrics' as table_name,
                COUNT(*) as record_count
            FROM `{client.project}.{dataset_name}.quality_metrics`
        """,
        
        'recent_data': f"""
            SELECT 
                table_name,
                MAX(timestamp) as latest_timestamp,
                COUNT(*) as recent_records
            FROM (
                SELECT 'operational_parameters' as table_name, timestamp
                FROM `{client.project}.{dataset_name}.operational_parameters`
                WHERE DATE(timestamp) = CURRENT_DATE()
                
                UNION ALL
                
                SELECT 'energy_consumption' as table_name, timestamp
                FROM `{client.project}.{dataset_name}.energy_consumption`
                WHERE DATE(timestamp) = CURRENT_DATE()
                
                UNION ALL
                
                SELECT 'quality_metrics' as table_name, timestamp
                FROM `{client.project}.{dataset_name}.quality_metrics`
                WHERE DATE(timestamp) = CURRENT_DATE()
            )
            GROUP BY table_name
        """
    }
    
    query_results = {}
    
    for query_name, query in test_queries.items():
        try:
            # Use existing connector
            results_df = run_bigquery(query, client.project)
            
            query_results[query_name] = {
                'status': 'success',
                'results': results_df.to_dict('records')
            }
            
            print(f"✅ Query {query_name} completed")
            
        except Exception as e:
            print(f"❌ Query {query_name} failed: {e}")
            query_results[query_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    return query_results


def main():
    parser = argparse.ArgumentParser(description="Setup BigQuery environment using existing code blocks")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Perform a dry run without actually creating resources")
    parser.add_argument("--dataset", default="cement_analytics_dev", 
                       help="BigQuery dataset name")
    parser.add_argument("--location", default="US", 
                       help="BigQuery dataset location")
    parser.add_argument("--outdir", default="artifacts", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("🏭 BigQuery Environment Setup")
    print("=" * 50)
    print("Using existing code blocks from the project")
    print("=" * 50)
    
    # Setup environment variables
    env_vars = setup_environment_variables()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual resources will be created")
        return 0
    
    try:
        # Initialize BigQuery client
        print("🔧 Initializing BigQuery client...")
        client = get_bigquery_client()
        print(f"✅ Connected to project: {client.project}")
        
        # Create dataset
        dataset_created = create_bigquery_dataset(client, args.dataset, args.location)
        if not dataset_created:
            print("❌ Failed to create dataset. Exiting.")
            return 1
        
        # Create tables
        table_results = create_bigquery_tables(client, args.dataset)
        
        # Generate sample data
        sample_data = generate_sample_data()
        
        # Stream data to BigQuery
        streaming_results = stream_data_to_bigquery(sample_data, args.dataset, dry_run=args.dry_run)
        
        # Validate data
        validation_results = validate_bigquery_data(client, args.dataset)
        
        # Test queries
        query_results = test_bigquery_queries(client, args.dataset)
        
        # Compile results
        setup_results = {
            'timestamp': datetime.now().isoformat(),
            'environment_variables': env_vars,
            'dataset_name': args.dataset,
            'location': args.location,
            'table_creation': table_results,
            'data_streaming': streaming_results,
            'validation': validation_results,
            'query_tests': query_results,
            'sample_data_summary': {
                table: len(records) for table, records in sample_data.items()
            }
        }
        
        # Save results
        output_file = outdir / "bigquery_environment_setup.json"
        with open(output_file, 'w') as f:
            json.dump(setup_results, f, indent=2, default=str)
        
        print(f"\n🎉 BigQuery environment setup completed!")
        print(f"📊 Results saved to: {output_file}")
        
        # Summary
        print(f"\n📋 Summary:")
        print(f"  📊 Dataset: {args.dataset}")
        print(f"  📍 Location: {args.location}")
        print(f"  📋 Tables created: {sum(table_results.values())}")
        print(f"  📤 Data streams: {len(streaming_results)}")
        print(f"  🔍 Validations: {len(validation_results)}")
        print(f"  🔍 Query tests: {len(query_results)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
