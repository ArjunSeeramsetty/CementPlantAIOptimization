#!/usr/bin/env python3
"""
Complete BigQuery Workflow Demo
Demonstrates the complete BigQuery streaming workflow using all existing code blocks.
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


def setup_bigquery_environment() -> Dict[str, str]:
    """Set up BigQuery environment variables."""
    print("🔧 Setting up BigQuery environment...")
    
    env_vars = {
        'CEMENT_BQ_DATASET': 'cement_analytics_dev',
        'CEMENT_GCP_PROJECT': 'cement-ai-platform-dev',
        'CEMENT_GCP_REGION': 'us-central1',
        'CEMENT_LOG_LEVEL': 'INFO'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")
    
    return env_vars


def demonstrate_schema_design() -> Dict[str, Any]:
    """Demonstrate BigQuery schema design using existing code."""
    print("📋 Demonstrating BigQuery schema design...")
    
    # Use existing schema design code
    schemas = create_bigquery_schemas()
    ddl_statements = generate_ddl_statements(schemas, "cement_analytics_dev")
    
    schema_demo = {
        'total_tables': len(schemas),
        'tables': {},
        'ddl_statements': ddl_statements
    }
    
    for table_key, schema_info in schemas.items():
        table_name = schema_info['table_name']
        schema_demo['tables'][table_name] = {
            'partition_field': schema_info['partition_field'],
            'partition_type': schema_info['partition_type'],
            'clustering_fields': schema_info['clustering_fields'],
            'column_count': len(schema_info['schema']),
            'columns': [col['name'] for col in schema_info['schema']]
        }
        print(f"✅ Table {table_name}: {len(schema_info['schema'])} columns")
    
    return schema_demo


def generate_comprehensive_sample_data() -> Dict[str, List[Dict[str, Any]]]:
    """Generate comprehensive sample data for all BigQuery tables."""
    print("📊 Generating comprehensive sample data...")
    
    np.random.seed(42)
    
    # 1. Operational Parameters Data
    operational_data = []
    plant_ids = ['PLANT_001', 'PLANT_002', 'PLANT_003']
    unit_ids = ['KILN_01', 'MILL_01', 'PREHEATER_01', 'COOLER_01', 'RAW_MILL_01']
    parameter_types = ['TEMPERATURE', 'PRESSURE', 'FLOW_RATE', 'SPEED', 'LEVEL']
    
    for i in range(200):
        param_type = np.random.choice(parameter_types)
        record = {
            'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'unit_id': np.random.choice(unit_ids),
            'parameter_type': param_type,
            'parameter_name': f'{param_type.lower()}_{i}',
            'value': np.random.uniform(0, 1000),
            'unit': 'C' if param_type == 'TEMPERATURE' else 'bar' if param_type == 'PRESSURE' else 'm3/h',
            'quality_flag': 'GOOD' if np.random.random() > 0.1 else 'WARNING',
            'sensor_id': f'SENSOR_{i:03d}',
            'ingestion_timestamp': datetime.now().isoformat()
        }
        operational_data.append(record)
    
    # 2. Energy Consumption Data
    energy_data = []
    energy_types = ['ELECTRICITY', 'FUEL', 'STEAM', 'COMPRESSED_AIR']
    
    for i in range(150):
        energy_type = np.random.choice(energy_types)
        consumption = np.random.uniform(100, 1000)
        record = {
            'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'unit_id': np.random.choice(unit_ids),
            'energy_type': energy_type,
            'energy_source': 'GRID' if energy_type == 'ELECTRICITY' else 'COAL',
            'consumption_kwh': consumption,
            'power_kw': consumption / 24,  # Average power
            'efficiency_factor': np.random.uniform(0.8, 1.0),
            'cost_per_unit': np.random.uniform(0.05, 0.15),
            'total_cost': consumption * np.random.uniform(0.05, 0.15),
            'ingestion_timestamp': datetime.now().isoformat()
        }
        energy_data.append(record)
    
    # 3. Quality Metrics Data
    quality_data = []
    product_types = ['CEM_I', 'CEM_II', 'CEM_III', 'CEM_IV', 'CEM_V']
    test_types = ['COMPRESSIVE_STRENGTH', 'FREE_LIME', 'BLAINE_FINENESS', 'SETTING_TIME']
    
    for i in range(100):
        test_type = np.random.choice(test_types)
        measured_value = np.random.uniform(10, 100)
        record = {
            'timestamp': (datetime.now() - timedelta(hours=i*2)).isoformat(),
            'plant_id': np.random.choice(plant_ids),
            'batch_id': f'BATCH_{i:03d}',
            'product_type': np.random.choice(product_types),
            'test_type': test_type,
            'metric_name': test_type.lower(),
            'measured_value': measured_value,
            'specification_min': measured_value * 0.9,
            'specification_max': measured_value * 1.1,
            'pass_fail': measured_value >= measured_value * 0.9,
            'lab_technician': f'TECH_{i%10:02d}',
            'equipment_id': f'EQ_{i%5:02d}',
            'ingestion_timestamp': datetime.now().isoformat()
        }
        quality_data.append(record)
    
    # 4. Production Summary Data
    production_data = []
    
    for i in range(30):  # 30 days of production data
        production_date = (datetime.now() - timedelta(days=i)).date().isoformat()
        record = {
            'production_date': production_date,
            'plant_id': np.random.choice(plant_ids),
            'product_type': np.random.choice(product_types),
            'total_production_tons': np.random.uniform(1000, 5000),
            'total_energy_consumed_kwh': np.random.uniform(50000, 200000),
            'avg_kiln_temperature': np.random.uniform(1400, 1500),
            'quality_pass_rate': np.random.uniform(0.85, 0.98),
            'downtime_hours': np.random.uniform(0, 8),
            'efficiency_score': np.random.uniform(0.7, 0.95),
            'ingestion_timestamp': datetime.now().isoformat()
        }
        production_data.append(record)
    
    return {
        'operational_parameters': operational_data,
        'energy_consumption': energy_data,
        'quality_metrics': quality_data,
        'production_summary': production_data
    }


def demonstrate_data_validation() -> Dict[str, Any]:
    """Demonstrate data validation using existing validation code."""
    print("🔍 Demonstrating data validation...")
    
    # Use existing validation queries
    validation_queries = create_validation_sql_queries()
    
    validation_demo = {
        'validation_queries': list(validation_queries.keys()),
        'query_count': len(validation_queries),
        'sample_queries': {}
    }
    
    # Show sample validation queries
    for query_name, query_template in validation_queries.items():
        # Truncate query for display
        sample_query = query_template[:200] + "..." if len(query_template) > 200 else query_template
        validation_demo['sample_queries'][query_name] = sample_query
        print(f"✅ Validation query: {query_name}")
    
    return validation_demo


def demonstrate_streaming_ingestion(sample_data: Dict[str, List[Dict[str, Any]]], 
                                  dry_run: bool = True) -> Dict[str, Any]:
    """Demonstrate BigQuery streaming ingestion."""
    print("📤 Demonstrating BigQuery streaming ingestion...")
    
    streaming_demo = {
        'tables_to_stream': list(sample_data.keys()),
        'total_records': sum(len(records) for records in sample_data.values()),
        'streaming_results': {}
    }
    
    try:
        # Initialize streaming ingestion
        ingestion = BigQueryStreamingIngestion()
        
        for table_name, records in sample_data.items():
            if dry_run:
                print(f"🔍 DRY RUN: Would stream {len(records)} records to {table_name}")
                streaming_demo['streaming_results'][table_name] = {
                    'status': 'dry_run',
                    'records_count': len(records),
                    'sample_record': records[0] if records else None
                }
            else:
                try:
                    # Stream data to BigQuery
                    ingested_count = ingestion.stream_rows(table_name, records)
                    
                    streaming_demo['streaming_results'][table_name] = {
                        'status': 'success',
                        'records_count': len(records),
                        'ingested_count': ingested_count
                    }
                    
                    print(f"✅ Streamed {ingested_count} records to {table_name}")
                    
                except Exception as e:
                    print(f"❌ Failed to stream {table_name}: {e}")
                    streaming_demo['streaming_results'][table_name] = {
                        'status': 'error',
                        'error': str(e),
                        'records_count': len(records)
                    }
    
    except Exception as e:
        print(f"❌ Streaming demo failed: {e}")
        streaming_demo['error'] = str(e)
    
    return streaming_demo


def demonstrate_real_data_integration() -> Dict[str, Any]:
    """Demonstrate integration with real data sources."""
    print("🔗 Demonstrating real data integration...")
    
    # Load real data sources
    kaggle_data = get_kaggle_cement_data()
    global_data = get_global_cement_data()
    
    integration_demo = {
        'kaggle_data': {
            'available': not kaggle_data.empty,
            'records': len(kaggle_data),
            'columns': list(kaggle_data.columns) if not kaggle_data.empty else []
        },
        'global_data': {
            'available': not global_data.empty,
            'records': len(global_data),
            'columns': list(global_data.columns) if not global_data.empty else []
        },
        'integration_status': 'sample_data_used' if kaggle_data.empty and global_data.empty else 'real_data_available'
    }
    
    if kaggle_data.empty and global_data.empty:
        print("⚠️ No real data sources available, using sample data")
    else:
        print("✅ Real data sources available")
    
    return integration_demo


def demonstrate_query_capabilities() -> Dict[str, Any]:
    """Demonstrate BigQuery query capabilities."""
    print("🔍 Demonstrating BigQuery query capabilities...")
    
    query_demo = {
        'sample_queries': {
            'operational_summary': """
                SELECT 
                    plant_id,
                    unit_id,
                    parameter_type,
                    COUNT(*) as record_count,
                    AVG(value) as avg_value,
                    MIN(value) as min_value,
                    MAX(value) as max_value
                FROM `cement_analytics_dev.operational_parameters`
                WHERE DATE(timestamp) = CURRENT_DATE()
                GROUP BY plant_id, unit_id, parameter_type
                ORDER BY record_count DESC
            """,
            
            'energy_consumption_analysis': """
                SELECT 
                    plant_id,
                    energy_type,
                    SUM(consumption_kwh) as total_consumption,
                    AVG(efficiency_factor) as avg_efficiency,
                    SUM(total_cost) as total_cost
                FROM `cement_analytics_dev.energy_consumption`
                WHERE DATE(timestamp) = CURRENT_DATE()
                GROUP BY plant_id, energy_type
                ORDER BY total_consumption DESC
            """,
            
            'quality_trends': """
                SELECT 
                    plant_id,
                    test_type,
                    DATE(timestamp) as test_date,
                    AVG(measured_value) as avg_value,
                    COUNT(*) as test_count,
                    SUM(CASE WHEN pass_fail THEN 1 ELSE 0 END) as pass_count
                FROM `cement_analytics_dev.quality_metrics`
                WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
                GROUP BY plant_id, test_type, test_date
                ORDER BY test_date DESC
            """,
            
            'production_efficiency': """
                SELECT 
                    plant_id,
                    product_type,
                    AVG(total_production_tons) as avg_production,
                    AVG(efficiency_score) as avg_efficiency,
                    AVG(quality_pass_rate) as avg_quality_rate,
                    SUM(downtime_hours) as total_downtime
                FROM `cement_analytics_dev.production_summary`
                WHERE production_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
                GROUP BY plant_id, product_type
                ORDER BY avg_efficiency DESC
            """
        },
        'query_count': 4,
        'capabilities': [
            'Time-series analysis',
            'Aggregation and grouping',
            'Date filtering and partitioning',
            'Multi-table joins',
            'Real-time analytics',
            'Performance optimization'
        ]
    }
    
    print(f"✅ Generated {query_demo['query_count']} sample queries")
    print(f"✅ Query capabilities: {len(query_demo['capabilities'])} features")
    
    return query_demo


def main():
    parser = argparse.ArgumentParser(description="Complete BigQuery workflow demonstration")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Perform a dry run without actually streaming data")
    parser.add_argument("--outdir", default="artifacts", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("🏭 Complete BigQuery Workflow Demonstration")
    print("=" * 60)
    print("Using all existing BigQuery code blocks from the project")
    print("=" * 60)
    
    # Setup environment
    env_vars = setup_bigquery_environment()
    
    # Demonstrate schema design
    schema_demo = demonstrate_schema_design()
    
    # Generate sample data
    sample_data = generate_comprehensive_sample_data()
    
    # Demonstrate data validation
    validation_demo = demonstrate_data_validation()
    
    # Demonstrate streaming ingestion
    streaming_demo = demonstrate_streaming_ingestion(sample_data, dry_run=args.dry_run)
    
    # Demonstrate real data integration
    integration_demo = demonstrate_real_data_integration()
    
    # Demonstrate query capabilities
    query_demo = demonstrate_query_capabilities()
    
    # Compile comprehensive results
    workflow_results = {
        'timestamp': datetime.now().isoformat(),
        'environment_setup': env_vars,
        'schema_design': schema_demo,
        'sample_data': {
            table: len(records) for table, records in sample_data.items()
        },
        'data_validation': validation_demo,
        'streaming_ingestion': streaming_demo,
        'real_data_integration': integration_demo,
        'query_capabilities': query_demo,
        'workflow_summary': {
            'total_tables': schema_demo['total_tables'],
            'total_records': streaming_demo['total_records'],
            'validation_queries': validation_demo['query_count'],
            'sample_queries': query_demo['query_count'],
            'dry_run': args.dry_run
        }
    }
    
    # Save results
    output_file = outdir / "complete_bigquery_workflow_demo.json"
    with open(output_file, 'w') as f:
        json.dump(workflow_results, f, indent=2, default=str)
    
    print(f"\n🎉 Complete BigQuery workflow demonstration completed!")
    print(f"📊 Results saved to: {output_file}")
    
    # Summary
    print(f"\n📋 Workflow Summary:")
    print(f"  📊 Tables designed: {schema_demo['total_tables']}")
    print(f"  📤 Records generated: {streaming_demo['total_records']}")
    print(f"  🔍 Validation queries: {validation_demo['query_count']}")
    print(f"  🔍 Sample queries: {query_demo['query_count']}")
    print(f"  🔗 Real data integration: {integration_demo['integration_status']}")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Set up GCP project and BigQuery")
    print(f"  2. Configure authentication credentials")
    print(f"  3. Run: python scripts/setup_bigquery_environment.py")
    print(f"  4. Data will be streamed to BigQuery tables")
    print(f"  5. Use queries for real-time analytics")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
