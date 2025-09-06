"""
BigQuery Data Warehouse Schema Design for Cement Plant Sensor Data

This block defines the schema for time-series partitioned tables to store
operational parameters, energy consumption, and quality metrics.
"""

import json
from datetime import datetime, timedelta

# Define the BigQuery table schemas for cement plant sensor data
def create_bigquery_schemas():
    """
    Creates comprehensive BigQuery table schemas for cement plant data warehouse
    """
    
    # 1. Operational Parameters Table - Time partitioned by timestamp
    operational_params_schema = {
        "table_name": "operational_parameters",
        "partition_field": "timestamp",
        "partition_type": "DAY",  # Daily partitioning for efficient queries
        "clustering_fields": ["plant_id", "unit_id", "parameter_type"],
        "schema": [
            {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Measurement timestamp"},
            {"name": "plant_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique plant identifier"},
            {"name": "unit_id", "type": "STRING", "mode": "REQUIRED", "description": "Process unit identifier (kiln, mill, etc.)"},
            {"name": "parameter_type", "type": "STRING", "mode": "REQUIRED", "description": "Type of operational parameter"},
            {"name": "parameter_name", "type": "STRING", "mode": "REQUIRED", "description": "Specific parameter name"},
            {"name": "value", "type": "FLOAT64", "mode": "REQUIRED", "description": "Measured value"},
            {"name": "unit", "type": "STRING", "mode": "NULLABLE", "description": "Unit of measurement"},
            {"name": "quality_flag", "type": "STRING", "mode": "NULLABLE", "description": "Data quality indicator"},
            {"name": "sensor_id", "type": "STRING", "mode": "NULLABLE", "description": "Source sensor identifier"},
            {"name": "ingestion_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Data ingestion timestamp"}
        ]
    }
    
    # 2. Energy Consumption Table - Time partitioned by timestamp
    energy_consumption_schema = {
        "table_name": "energy_consumption",
        "partition_field": "timestamp", 
        "partition_type": "DAY",
        "clustering_fields": ["plant_id", "energy_type", "unit_id"],
        "schema": [
            {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Measurement timestamp"},
            {"name": "plant_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique plant identifier"},
            {"name": "unit_id", "type": "STRING", "mode": "REQUIRED", "description": "Process unit identifier"},
            {"name": "energy_type", "type": "STRING", "mode": "REQUIRED", "description": "Type of energy (electrical, thermal, fuel)"},
            {"name": "energy_source", "type": "STRING", "mode": "NULLABLE", "description": "Energy source (grid, coal, gas, etc.)"},
            {"name": "consumption_kwh", "type": "FLOAT64", "mode": "REQUIRED", "description": "Energy consumption in kWh"},
            {"name": "power_kw", "type": "FLOAT64", "mode": "NULLABLE", "description": "Power consumption in kW"},
            {"name": "efficiency_factor", "type": "FLOAT64", "mode": "NULLABLE", "description": "Energy efficiency factor"},
            {"name": "cost_per_unit", "type": "FLOAT64", "mode": "NULLABLE", "description": "Cost per energy unit"},
            {"name": "total_cost", "type": "FLOAT64", "mode": "NULLABLE", "description": "Total energy cost"},
            {"name": "ingestion_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Data ingestion timestamp"}
        ]
    }
    
    # 3. Quality Metrics Table - Time partitioned by timestamp
    quality_metrics_schema = {
        "table_name": "quality_metrics", 
        "partition_field": "timestamp",
        "partition_type": "DAY",
        "clustering_fields": ["plant_id", "product_type", "test_type"],
        "schema": [
            {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Test/measurement timestamp"},
            {"name": "plant_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique plant identifier"},
            {"name": "batch_id", "type": "STRING", "mode": "REQUIRED", "description": "Production batch identifier"},
            {"name": "product_type", "type": "STRING", "mode": "REQUIRED", "description": "Cement product type"},
            {"name": "test_type", "type": "STRING", "mode": "REQUIRED", "description": "Quality test type"},
            {"name": "metric_name", "type": "STRING", "mode": "REQUIRED", "description": "Quality metric name"},
            {"name": "measured_value", "type": "FLOAT64", "mode": "REQUIRED", "description": "Measured quality value"},
            {"name": "specification_min", "type": "FLOAT64", "mode": "NULLABLE", "description": "Minimum specification limit"},
            {"name": "specification_max", "type": "FLOAT64", "mode": "NULLABLE", "description": "Maximum specification limit"},
            {"name": "pass_fail", "type": "BOOLEAN", "mode": "NULLABLE", "description": "Quality test pass/fail status"},
            {"name": "lab_technician", "type": "STRING", "mode": "NULLABLE", "description": "Lab technician identifier"},
            {"name": "equipment_id", "type": "STRING", "mode": "NULLABLE", "description": "Test equipment identifier"},
            {"name": "ingestion_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Data ingestion timestamp"}
        ]
    }
    
    # 4. Production Summary Table - Time partitioned by production_date
    production_summary_schema = {
        "table_name": "production_summary",
        "partition_field": "production_date",
        "partition_type": "DAY", 
        "clustering_fields": ["plant_id", "product_type"],
        "schema": [
            {"name": "production_date", "type": "DATE", "mode": "REQUIRED", "description": "Production date"},
            {"name": "plant_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique plant identifier"},
            {"name": "product_type", "type": "STRING", "mode": "REQUIRED", "description": "Cement product type"},
            {"name": "total_production_tons", "type": "FLOAT64", "mode": "REQUIRED", "description": "Total production in tons"},
            {"name": "total_energy_consumed_kwh", "type": "FLOAT64", "mode": "NULLABLE", "description": "Total energy consumed"},
            {"name": "avg_kiln_temperature", "type": "FLOAT64", "mode": "NULLABLE", "description": "Average kiln temperature"},
            {"name": "quality_pass_rate", "type": "FLOAT64", "mode": "NULLABLE", "description": "Quality test pass rate %"},
            {"name": "downtime_hours", "type": "FLOAT64", "mode": "NULLABLE", "description": "Total downtime hours"},
            {"name": "efficiency_score", "type": "FLOAT64", "mode": "NULLABLE", "description": "Overall efficiency score"},
            {"name": "ingestion_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED", "description": "Data ingestion timestamp"}
        ]
    }
    
    return {
        "operational_parameters": operational_params_schema,
        "energy_consumption": energy_consumption_schema,
        "quality_metrics": quality_metrics_schema,
        "production_summary": production_summary_schema
    }

# Create the schemas
bigquery_schemas = create_bigquery_schemas()

# Display schema information
print("BigQuery Data Warehouse Schema Design")
print("=" * 50)
print(f"Number of tables designed: {len(bigquery_schemas)}")
print()

for table_name, schema_info in bigquery_schemas.items():
    print(f"Table: {schema_info['table_name']}")
    print(f"  Partition: {schema_info['partition_field']} ({schema_info['partition_type']})")
    print(f"  Clustering: {', '.join(schema_info['clustering_fields'])}")
    print(f"  Columns: {len(schema_info['schema'])}")
    print()

# Create DDL statements for BigQuery
def generate_ddl_statements(schemas, dataset_name="cement_plant_data"):
    """Generate BigQuery DDL statements"""
    
    ddl_statements = {}
    
    for table_key, schema_info in schemas.items():
        table_name = schema_info['table_name']
        
        # Build column definitions
        _column_defs = []
        for _col in schema_info['schema']:
            _col_def = f"  {_col['name']} {_col['type']}"
            if _col['mode'] == 'REQUIRED':
                _col_def += " NOT NULL"
            _column_defs.append(_col_def)
        
        # Create DDL
        ddl = f"""CREATE TABLE `{dataset_name}.{table_name}` (
{chr(10).join(_column_defs)}
)
PARTITION BY DATE({schema_info['partition_field']})
CLUSTER BY {', '.join(schema_info['clustering_fields'])}
OPTIONS(
  description="Time-series data for {table_name.replace('_', ' ')}",
  partition_expiration_days=2555  -- 7 years retention
);"""
        
        ddl_statements[table_name] = ddl
    
    return ddl_statements

# Generate DDL statements
ddl_statements = generate_ddl_statements(bigquery_schemas)

print("Generated DDL Statements:")
print("=" * 30)
for _table_name, _ddl in ddl_statements.items():
    print(f"\n-- {_table_name.upper()} TABLE")
    print(_ddl)

# Export schemas for downstream blocks
schema_design = {
    "schemas": bigquery_schemas,
    "ddl_statements": ddl_statements,
    "dataset_name": "cement_plant_data",
    "project_id": "your-project-id",  # To be configured
    "design_summary": {
        "total_tables": len(bigquery_schemas),
        "partition_strategy": "Daily partitioning for efficient time-series queries",
        "clustering_strategy": "By plant_id and relevant categorical fields",
        "retention_policy": "7 years (2555 days)"
    }
}

print(f"\n✅ Schema design complete - {len(schema_design['schemas'])} tables defined")