#!/usr/bin/env python3
"""
BigQuery Integration Demo
Demonstrates how real data would be sent to BigQuery with proper setup.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cement_ai_platform.data.processors.real_data_load_kaggle_cement import get_kaggle_cement_data
from cement_ai_platform.data.processors.real_data_load_global_cement import get_global_cement_data


def create_realistic_sample_data() -> Dict[str, pd.DataFrame]:
    """Create realistic sample data that mimics real cement plant data."""
    print("📊 Creating realistic sample cement plant data...")
    
    # Sample Kaggle-style cement data (concrete composition and strength)
    np.random.seed(42)  # For reproducible results
    
    kaggle_data = pd.DataFrame({
        'cement': np.random.uniform(200, 400, 200),  # kg/m³
        'blast_furnace_slag': np.random.uniform(0, 200, 200),  # kg/m³
        'fly_ash': np.random.uniform(0, 150, 200),  # kg/m³
        'water': np.random.uniform(150, 250, 200),  # kg/m³
        'superplasticizer': np.random.uniform(0, 30, 200),  # kg/m³
        'coarse_aggregate': np.random.uniform(800, 1200, 200),  # kg/m³
        'fine_aggregate': np.random.uniform(600, 900, 200),  # kg/m³
        'age_days': np.random.choice([1, 3, 7, 14, 28, 56, 90], 200),
        'compressive_strength': np.random.uniform(10, 80, 200),  # MPa
        'w_c_ratio': np.random.uniform(0.3, 0.6, 200),
        'slump': np.random.uniform(50, 200, 200),  # mm
        'curing_temp': np.random.uniform(20, 30, 200),  # °C
        'curing_humidity': np.random.uniform(60, 90, 200),  # %
        'aggregate_size': np.random.choice([10, 20, 40], 200),  # mm
        'admixture_type': np.random.choice(['None', 'Superplasticizer', 'Retarder'], 200),
        'mix_design': np.random.choice(['Standard', 'High-strength', 'Low-heat'], 200),
        'test_method': np.random.choice(['ASTM C39', 'BS EN 12390'], 200)
    })
    
    # Sample Global Cement Database data (facility information)
    countries = ['USA', 'China', 'India', 'Germany', 'Brazil', 'Mexico', 'Turkey', 'Russia', 'Japan', 'Italy']
    cement_types = ['CEM I', 'CEM II', 'CEM III', 'CEM IV', 'CEM V']
    processes = ['Dry', 'Wet', 'Semi-dry']
    kiln_types = ['Rotary', 'Vertical']
    
    global_data = pd.DataFrame({
        'country': np.random.choice(countries, 100),
        'facility_name': [f'Cement Plant {i+1}' for i in range(100)],
        'latitude': np.random.uniform(-60, 60, 100),
        'longitude': np.random.uniform(-180, 180, 100),
        'cement_type': np.random.choice(cement_types, 100),
        'capacity_mt_year': np.random.uniform(0.5, 5.0, 100),  # Million tonnes per year
        'year_established': np.random.randint(1950, 2020, 100),
        'production_process': np.random.choice(processes, 100),
        'kiln_type': np.random.choice(kiln_types, 100),
        'limestone_pct': np.random.uniform(70, 85, 100),  # %
        'clay_pct': np.random.uniform(10, 20, 100),  # %
        'iron_ore_pct': np.random.uniform(2, 8, 100),  # %
        'silica_sand_pct': np.random.uniform(0, 5, 100),  # %
        'gypsum_pct': np.random.uniform(3, 8, 100),  # %
        'energy_efficiency': np.random.uniform(2.8, 4.2, 100),  # GJ/t clinker
        'co2_emissions': np.random.uniform(800, 1200, 100),  # kg CO2/t clinker
        'water_usage': np.random.uniform(100, 300, 100),  # L/t clinker
        'electricity_consumption': np.random.uniform(80, 120, 100),  # kWh/t clinker
        'dust_emissions': np.random.uniform(10, 50, 100),  # mg/Nm³
        'nox_emissions': np.random.uniform(200, 800, 100)  # mg/Nm³
    })
    
    return {
        'kaggle': kaggle_data,
        'global': global_data
    }


def demonstrate_data_loading() -> Dict[str, pd.DataFrame]:
    """Demonstrate the data loading process."""
    print("🔍 Demonstrating real data loading process...")
    
    # Try to load real data first
    print("📊 Attempting to load Kaggle cement dataset...")
    kaggle_data = get_kaggle_cement_data()
    
    print("🌍 Attempting to load Global cement database...")
    global_data = get_global_cement_data()
    
    # If no real data available, create realistic sample data
    if kaggle_data.empty and global_data.empty:
        print("⚠️ No real data sources available.")
        print("💡 Creating realistic sample data for demonstration...")
        return create_realistic_sample_data()
    
    return {
        'kaggle': kaggle_data,
        'global': global_data
    }


def prepare_bigquery_schema(data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, str]]]:
    """Prepare BigQuery schema for the data."""
    print("📋 Preparing BigQuery schema...")
    
    schemas = {}
    
    for source_name, df in data.items():
        if df.empty:
            continue
        
        schema = []
        for column in df.columns:
            dtype = df[column].dtype
            
            # Map pandas dtypes to BigQuery types
            if dtype == 'object':
                bq_type = 'STRING'
            elif dtype in ['int64', 'int32']:
                bq_type = 'INTEGER'
            elif dtype in ['float64', 'float32']:
                bq_type = 'FLOAT'
            elif dtype == 'bool':
                bq_type = 'BOOLEAN'
            elif dtype == 'datetime64[ns]':
                bq_type = 'TIMESTAMP'
            else:
                bq_type = 'STRING'  # Default fallback
            
            schema.append({
                'name': column,
                'type': bq_type,
                'mode': 'NULLABLE'
            })
        
        schemas[source_name] = schema
        print(f"✅ Prepared schema for {source_name}: {len(schema)} fields")
    
    return schemas


def demonstrate_bigquery_ingestion(data: Dict[str, pd.DataFrame], 
                                  schemas: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """Demonstrate how data would be ingested into BigQuery."""
    print("📤 Demonstrating BigQuery ingestion process...")
    
    ingestion_demo = {}
    
    for source_name, df in data.items():
        if df.empty:
            continue
        
        table_name = f"cement_{source_name}_data"
        
        # Prepare sample records for demonstration
        sample_records = df.head(5).to_dict('records')
        
        # Add metadata
        for record in sample_records:
            record['data_source'] = source_name
            record['ingestion_timestamp'] = datetime.now().isoformat()
            record['record_id'] = f"{source_name.upper()}_{len(sample_records)}"
        
        ingestion_demo[source_name] = {
            'table_name': table_name,
            'total_records': len(df),
            'sample_records': sample_records,
            'schema': schemas[source_name],
            'bigquery_sql': f"""
-- BigQuery table creation SQL for {table_name}
CREATE TABLE `cement_analytics_dev.{table_name}` (
{chr(10).join([f"  {field['name']} {field['type']}," for field in schemas[source_name][:-1]])}
  {schemas[source_name][-1]['name']} {schemas[source_name][-1]['type']}
)
PARTITION BY DATE(ingestion_timestamp)
CLUSTER BY data_source;
            """.strip()
        }
        
        print(f"✅ Prepared ingestion demo for {source_name}: {len(df)} records")
    
    return ingestion_demo


def main():
    print("🏭 BigQuery Integration Demo")
    print("=" * 50)
    
    # Demonstrate data loading
    data = demonstrate_data_loading()
    
    # Prepare BigQuery schemas
    schemas = prepare_bigquery_schema(data)
    
    # Demonstrate BigQuery ingestion
    ingestion_demo = demonstrate_bigquery_ingestion(data, schemas)
    
    # Create comprehensive demo results
    demo_results = {
        'timestamp': datetime.now().isoformat(),
        'data_sources': list(data.keys()),
        'data_summary': {
            source: {
                'records_count': len(df),
                'columns': list(df.columns),
                'sample_data': df.head(3).to_dict('records') if not df.empty else []
            }
            for source, df in data.items()
        },
        'bigquery_schemas': schemas,
        'ingestion_demo': ingestion_demo,
        'setup_instructions': {
            'environment_variables': [
                'CEMENT_BQ_DATASET=cement_analytics_dev',
                'CEMENT_GCP_PROJECT=your-gcp-project-id',
                'GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json'
            ],
            'bigquery_setup': [
                '1. Create a GCP project',
                '2. Enable BigQuery API',
                '3. Create a service account with BigQuery permissions',
                '4. Download service account key file',
                '5. Set environment variables',
                '6. Run: python scripts/run_real_data_to_bigquery.py'
            ]
        }
    }
    
    # Save results
    output_file = Path("artifacts/bigquery_integration_demo.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n🎉 BigQuery integration demo completed!")
    print(f"📊 Results saved to: {output_file}")
    
    # Summary
    print(f"\n📋 Summary:")
    for source, df in data.items():
        print(f"  📊 {source}: {len(df)} records, {len(df.columns)} columns")
    
    print(f"\n🔍 BigQuery Setup:")
    print(f"  📋 Tables to create: {len(ingestion_demo)}")
    for source, demo in ingestion_demo.items():
        print(f"    • {demo['table_name']}: {demo['total_records']} records")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Set up GCP project and BigQuery")
    print(f"  2. Configure environment variables")
    print(f"  3. Run: python scripts/run_real_data_to_bigquery.py")
    print(f"  4. Data will be streamed to BigQuery tables")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
