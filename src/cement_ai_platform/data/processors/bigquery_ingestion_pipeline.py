"""
BigQuery Data Ingestion Pipeline Configuration

This block defines the data ingestion pipeline for streaming sensor data
into the time-partitioned BigQuery tables with automated validation.
"""

import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import schema design from previous block
print("Configuring BigQuery Data Ingestion Pipeline")
print("=" * 50)

# Pipeline configuration
pipeline_config = {
    "project_id": schema_design["project_id"],
    "dataset_name": schema_design["dataset_name"],
    "streaming_config": {
        "batch_size": 1000,  # Records per batch
        "flush_interval_seconds": 30,  # Flush every 30 seconds
        "max_latency_seconds": 60,  # Maximum acceptable latency
        "retry_attempts": 3,
        "dead_letter_queue": True
    },
    "validation_rules": {
        "required_fields": ["timestamp", "plant_id"],
        "timestamp_format": "ISO 8601",
        "value_ranges": {
            "temperature": {"min": -50, "max": 2000},  # Celsius
            "pressure": {"min": 0, "max": 50},         # Bar
            "flow_rate": {"min": 0, "max": 10000},     # m3/h
            "power": {"min": 0, "max": 50000}          # kW
        },
        "quality_flags": ["GOOD", "BAD", "UNCERTAIN", "MAINTENANCE"]
    }
}

# Data source configurations
data_sources = {
    "operational_sensors": {
        "source_type": "streaming",
        "frequency": "1_minute",  # 1-minute intervals
        "parameters": [
            {"name": "kiln_temperature", "unit": "°C", "type": "temperature"},
            {"name": "kiln_pressure", "unit": "bar", "type": "pressure"},
            {"name": "raw_material_flow", "unit": "t/h", "type": "flow_rate"},
            {"name": "fuel_flow", "unit": "m3/h", "type": "flow_rate"},
            {"name": "fan_speed", "unit": "rpm", "type": "speed"},
            {"name": "cooler_temperature", "unit": "°C", "type": "temperature"}
        ]
    },
    "energy_meters": {
        "source_type": "streaming",
        "frequency": "15_minutes",  # 15-minute intervals
        "parameters": [
            {"name": "electrical_power", "unit": "kW", "type": "power"},
            {"name": "thermal_energy", "unit": "kWh", "type": "energy"},
            {"name": "fuel_consumption", "unit": "m3/h", "type": "flow_rate"},
            {"name": "grid_frequency", "unit": "Hz", "type": "frequency"}
        ]
    },
    "quality_tests": {
        "source_type": "batch",
        "frequency": "2_hours",  # Every 2 hours
        "parameters": [
            {"name": "compressive_strength", "unit": "MPa", "type": "strength"},
            {"name": "fineness", "unit": "m2/kg", "type": "fineness"},
            {"name": "setting_time", "unit": "minutes", "type": "time"},
            {"name": "chemical_composition", "unit": "%", "type": "percentage"}
        ]
    }
}

# Create ingestion functions
def create_streaming_insert_template():
    """Template for streaming inserts"""
    template = """
-- Streaming Insert Template for {table_name}
INSERT INTO `{project_id}.{dataset_name}.{table_name}` (
    {column_list}
)
SELECT
    {select_columns}
FROM EXTERNAL_QUERY(
    "projects/{project_id}/locations/{location}/connections/{connection_id}",
    '''
    SELECT * FROM sensor_data
    WHERE timestamp >= CURRENT_TIMESTAMP() - INTERVAL 1 HOUR
    '''
);
"""
    return template

def create_validation_queries():
    """Create data validation queries"""
    validation_queries = {}

    for table_name in schema_design["schemas"].keys():
        validation_query = f"""
-- Data Quality Validation for {table_name}
WITH data_quality_check AS (
  SELECT
    DATE(timestamp) as check_date,
    plant_id,
    COUNT(*) as total_records,
    COUNT(CASE WHEN timestamp IS NULL THEN 1 END) as null_timestamps,
    COUNT(CASE WHEN plant_id IS NULL THEN 1 END) as null_plant_ids,
    MIN(timestamp) as min_timestamp,
    MAX(timestamp) as max_timestamp,
    COUNTIF(timestamp > CURRENT_TIMESTAMP()) as future_timestamps,
    COUNTIF(timestamp < CURRENT_TIMESTAMP() - INTERVAL 30 DAY) as old_timestamps
  FROM `{pipeline_config['project_id']}.{pipeline_config['dataset_name']}.{table_name}`
  WHERE DATE(timestamp) = CURRENT_DATE()
  GROUP BY DATE(timestamp), plant_id
),
quality_summary AS (
  SELECT
    check_date,
    SUM(total_records) as daily_total,
    SUM(null_timestamps) as total_null_timestamps,
    SUM(null_plant_ids) as total_null_plant_ids,
    SUM(future_timestamps) as total_future_timestamps,
    SUM(old_timestamps) as total_old_timestamps,
    ROUND(100.0 * (1 - (SUM(null_timestamps) + SUM(null_plant_ids)) / SUM(total_records)), 2) as data_quality_score
  FROM data_quality_check
  GROUP BY check_date
)
SELECT
  check_date,
  daily_total,
  data_quality_score,
  CASE
    WHEN data_quality_score >= 95 THEN 'EXCELLENT'
    WHEN data_quality_score >= 90 THEN 'GOOD'
    WHEN data_quality_score >= 80 THEN 'FAIR'
    ELSE 'POOR'
  END as quality_rating,
  total_null_timestamps,
  total_null_plant_ids,
  total_future_timestamps,
  total_old_timestamps
FROM quality_summary;
"""
        validation_queries[table_name] = validation_query

    return validation_queries

# Generate sample data for testing
def generate_sample_sensor_data(records=1000):
    """Generate sample sensor data for pipeline testing"""

    np.random.seed(42)
    base_time = datetime.now() - timedelta(hours=24)

    # Generate operational parameters sample data
    operational_sample = []
    plants = ["PLANT_001", "PLANT_002", "PLANT_003"]
    units = ["KILN_01", "MILL_01", "COOLER_01", "PREHEATER_01"]
    param_types = ["TEMPERATURE", "PRESSURE", "FLOW", "SPEED"]

    for i in range(records):
        timestamp = base_time + timedelta(minutes=i)
        plant_id = np.random.choice(plants)
        unit_id = np.random.choice(units)
        param_type = np.random.choice(param_types)

        # Generate realistic values based on parameter type
        if param_type == "TEMPERATURE":
            value = np.random.normal(850, 50)  # Kiln temperature around 850°C
            unit = "°C"
            param_name = "kiln_temperature"
        elif param_type == "PRESSURE":
            value = np.random.normal(15, 2)    # Pressure around 15 bar
            unit = "bar"
            param_name = "system_pressure"
        elif param_type == "FLOW":
            value = np.random.normal(1000, 100)  # Flow around 1000 t/h
            unit = "t/h"
            param_name = "material_flow"
        else:  # SPEED
            value = np.random.normal(1500, 100)  # Speed around 1500 rpm
            unit = "rpm"
            param_name = "fan_speed"

        operational_sample.append({
            "timestamp": timestamp.isoformat(),
            "plant_id": plant_id,
            "unit_id": unit_id,
            "parameter_type": param_type,
            "parameter_name": param_name,
            "value": round(value, 2),
            "unit": unit,
            "quality_flag": np.random.choice(["GOOD", "GOOD", "GOOD", "UNCERTAIN"]),  # 75% good quality
            "sensor_id": f"SENSOR_{np.random.randint(1000, 9999)}",
            "ingestion_timestamp": datetime.now().isoformat()
        })

    return pd.DataFrame(operational_sample)

# Generate sample data
sample_data = generate_sample_sensor_data(100)
print(f"Generated {len(sample_data)} sample sensor records")
print("\nSample data preview:")
print(sample_data.head())

# Create ingestion templates
streaming_template = create_streaming_insert_template()
validation_queries = create_validation_queries()

print(f"\nCreated validation queries for {len(validation_queries)} tables")

# Performance optimization settings
performance_config = {
    "table_settings": {
        "require_partition_filter": True,  # Require partition filter in queries
        "partition_expiration_days": 2555,  # 7 years retention
        "enable_table_clustering": True
    },
    "query_optimization": {
        "use_legacy_sql": False,
        "use_query_cache": True,
        "maximum_bytes_billed": "100GB",  # Cost control
        "dry_run_validation": True
    },
    "monitoring": {
        "enable_audit_logs": True,
        "alert_thresholds": {
            "ingestion_lag_minutes": 30,
            "data_quality_score": 90,
            "daily_volume_variance": 0.2  # 20% variance alert
        }
    }
}

# Export pipeline configuration
ingestion_pipeline = {
    "config": pipeline_config,
    "data_sources": data_sources,
    "validation_queries": validation_queries,
    "sample_data": sample_data,
    "performance_config": performance_config,
    "streaming_template": streaming_template,
    "pipeline_summary": {
        "total_data_sources": len(data_sources),
        "streaming_sources": sum(1 for s in data_sources.values() if s["source_type"] == "streaming"),
        "batch_sources": sum(1 for s in data_sources.values() if s["source_type"] == "batch"),
        "total_parameters": sum(len(s["parameters"]) for s in data_sources.values()),
        "validation_rules": len(validation_queries)
    }
}

print(f"\n✅ Ingestion pipeline configured:")
print(f"  - {ingestion_pipeline['pipeline_summary']['total_data_sources']} data sources")
print(f"  - {ingestion_pipeline['pipeline_summary']['total_parameters']} parameters")
print(f"  - {ingestion_pipeline['pipeline_summary']['validation_rules']} validation rules")